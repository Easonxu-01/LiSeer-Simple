'''
Author: EASON XU
Date: 2025-05-29 03:33:39
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-12-04 15:52:05
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/cylinder3d/cylinder3d.py
'''
from mmdet.models import DETECTORS
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.models import builder
import yaml
import numpy as np
import copy
import random
import open3d as o3d
import torch
from typing import Any, Dict, List, Optional, Union
from mmengine.config import ConfigDict
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
from itertools import repeat
import collections
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_min
from typing import Tuple


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_pair = _ntuple(2, "_pair")


class VoxelizationByGridShape(nn.Module):
    """Voxelization that allows inferring voxel size automatically based on
    grid shape.

    Please refer to `Point-Voxel CNN for Efficient 3D Deep Learning
    <https://arxiv.org/abs/1907.03739>`_ for more details.

    Args:
        point_cloud_range (list):
            [x_min, y_min, z_min, x_max, y_max, z_max]
        max_num_points (int): max number of points per voxel
        voxel_size (list): list [x, y, z] or [rho, phi, z]
            size of single voxel.
        grid_shape (list): [L, W, H], grid shape of voxelization.
        max_voxels (tuple or int): max number of voxels in
            (training, testing) time
        deterministic: bool. whether to invoke the non-deterministic
            version of hard-voxelization implementations. non-deterministic
            version is considerablly fast but is not deterministic. only
            affects hard voxelization. default True. for more information
            of this argument and the implementation insights, please refer
            to the following links:
            https://github.com/open-mmlab/mmdetection3d/issues/894
            https://github.com/open-mmlab/mmdetection3d/pull/904
            it is an experimental feature and we will appreciate it if
            you could share with us the failing cases.
    """

    def __init__(self,
                 point_cloud_range: List,
                 max_num_points: int,
                 voxel_size: List = [],
                 grid_shape: List[int] = [],
                 max_voxels: Union[tuple, int] = 20000,
                 deterministic: bool = True):
        super().__init__()
        if voxel_size and grid_shape:
            raise ValueError('voxel_size is mutually exclusive grid_shape')
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        if voxel_size:
            self.voxel_size = voxel_size
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
            grid_shape = (point_cloud_range[3:] -
                          point_cloud_range[:3]) / voxel_size
            grid_shape = torch.round(grid_shape).long().tolist()
            self.grid_shape = grid_shape
        elif grid_shape:
            grid_shape = torch.tensor(grid_shape, dtype=torch.float32)
            voxel_size = (point_cloud_range[3:] - point_cloud_range[:3]) / (
                grid_shape - 1)
            voxel_size = voxel_size.tolist()
            self.voxel_size = voxel_size
            # Ensure grid_shape is available regardless of initialization path
            self.grid_shape = grid_shape.long().tolist()
        else:
            raise ValueError('must assign a value to voxel_size or grid_shape')

    def forward(self, input_1: Union[torch.Tensor, List[torch.Tensor]], input_2: Union[torch.Tensor, List[torch.Tensor]]) -> tuple:
        """Forward function.

        Args:
            input (torch.Tensor or List[torch.Tensor]): Input point cloud data.
                If input is a tensor, shape should be [N, 3].
                If input is a list of tensors, each tensor shape should be [N_i, 3],
                where N_i is the number of points in each frame.

        Returns:
            tuple: (voxels, coors) where:
                - voxels: shape [N, C] or List[Tensor] with shape [N_i, C]
                - coors: shape [N, 4] or List[Tensor] with shape [N_i, 4]
        """
        if isinstance(input_1, list):
            # Handle list of point clouds
            voxels_list = []
            coors_list = []
            voxel_semantic_mask_list = []
            point2voxel_maps_list = []
            for points, pts_semantic_mask in zip(input_1, input_2):
                voxels, coors, voxel_semantic_mask, point2voxel_maps = self.voxelization(points, pts_semantic_mask)
                voxels_list.append(voxels)
                coors_list.append(coors)
                voxel_semantic_mask_list.append(voxel_semantic_mask)
                point2voxel_maps_list.append(point2voxel_maps)
            return voxels_list, coors_list, voxel_semantic_mask_list, point2voxel_maps_list
        else:
            # Handle single point cloud
            return self.voxelization(input_1, input_2)
        
    def dynamic_point_to_voxel_forward_mean_python(self,        feats: torch.Tensor,   # [N, C]
        coors: torch.Tensor,   # [N, ndim] (int)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mean-reduction version of dynamic_point_to_voxel_forward.

        Returns:
        voxel_feats        [M, C]  float
        voxel_coors        [M, ndim]  int
        point2voxel_map    [N]  long, maps each point -> voxel idx in [0, M)
        voxel_points_count [M]  long, number of points per voxel
        """
        if feats.dim() == 3:
            feats = feats.reshape(-1, feats.shape[-1])
        assert feats.dim() == 2 and coors.dim() == 2 and feats.size(0) == coors.size(0)
        N, C = feats.shape

        voxel_coors, inverse_map, counts = torch.unique(
            coors, dim=0, return_inverse=True, return_counts=True
        )  # voxel_coors [M, ndim], inverse_map [N], counts [M]

        voxel_feats = feats.new_zeros((voxel_coors.size(0), C))
        voxel_feats.index_add_(0, inverse_map, feats)
        voxel_feats = voxel_feats / counts.clamp_min(1).to(voxel_feats.dtype).unsqueeze(1)
        point2voxel_map = inverse_map
        return voxel_feats, voxel_coors, point2voxel_map
    
    def voxelization(self, input: torch.Tensor, pts_semantic_mask: torch.Tensor):
        voxels, coors, voxel_semantic_mask, point2voxel_maps = [], [], [], []
        for i, res in enumerate(input):
            rho = torch.sqrt(res[:, 0]**2 + res[:, 1]**2)
            phi = torch.atan2(res[:, 1], res[:, 0])
            polar_res = torch.stack((rho, phi, res[:, 2]), dim=-1)

            min_bound = polar_res.new_tensor(self.point_cloud_range[:3])
            max_bound = polar_res.new_tensor(self.point_cloud_range[3:])

            try:
                polar_res_clamp = torch.clamp(polar_res, min_bound, max_bound)
            except TypeError:
                polar_res_clamp = polar_res.clone()
                for coor_idx in range(3):
                    polar_res_clamp[:, coor_idx][polar_res[:, coor_idx] > max_bound[coor_idx]] = max_bound[coor_idx]
                    polar_res_clamp[:, coor_idx][polar_res[:, coor_idx] < min_bound[coor_idx]] = min_bound[coor_idx]

            res_coors = torch.floor((polar_res_clamp - min_bound) / polar_res_clamp.new_tensor(self.voxel_size)).int()
            # Clamp discrete voxel indices to valid grid range [0, grid_shape[d)-1]
            for coor_idx in range(3):
                res_coors[:, coor_idx].clamp_(min=0, max=int(self.grid_shape[coor_idx] - 1))
            voxel_semantic_mask_i, _, point2voxel_map_i = self.dynamic_point_to_voxel_forward_mean_python(F.one_hot(pts_semantic_mask.long()).float(), res_coors)
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)  # prepend batch index
            voxels.append(torch.cat((polar_res, res[:, :2], res[:, 3:]), dim=-1))
            coors.append(res_coors)
            voxel_semantic_mask.append(torch.argmax(voxel_semantic_mask_i, dim=-1))
            point2voxel_maps.append(point2voxel_map_i)

        voxels = torch.cat(voxels, dim=0)           # [N_points_total, C]
        coors = torch.cat(coors, dim=0)             # [N_voxels_total, 4]
        voxel_semantic_mask = torch.cat(voxel_semantic_mask, dim=0)
        point2voxel_maps = torch.cat(point2voxel_maps, dim=0)
        
        return voxels, coors, voxel_semantic_mask, point2voxel_maps


    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'voxel_size=' + str(self.voxel_size)
        s += ', grid_shape=' + str(self.grid_shape)
        s += ', point_cloud_range=' + str(self.point_cloud_range)
        s += ', max_num_points=' + str(self.max_num_points)
        s += ', max_voxels=' + str(self.max_voxels)
        s += ', deterministic=' + str(self.deterministic)
        s += ')'
        return s

@DETECTORS.register_module()
class Cylinder3D(Base3DDetector):
    """Cylinder3D detector.
    
    Args:
        voxel_encoder (dict): Config of voxel encoder.
        backbone (dict): Config of backbone.
        decode_head (dict): Config of decode head.
        empty_idx (int): Index of empty voxel.
        random (bool): Whether to use random sampling.
        sensor (dict): Config of sensor.
        voxel (bool): Whether to use voxel.
        voxel_type (str): Type of voxel.
        max_voxels (int): Maximum number of voxels.
        voxel_layer (dict): Config of voxel layer.
    """

    def __init__(self,
            voxel_encoder=None,
            backbone=None,
            decode_head=None,
            empty_idx = None,
            random = False,
            sensor=None,
            voxel=True, 
            voxel_type='cylindrical', 
            max_voxels = None, 
            voxel_layer = None,
            **kwargs):
        super().__init__()

        # Initialize all attributes first
        self.voxel_encoder = None
        self.backbone = None
        self.decode_head = None
        self.fp16_enabled = False
        self.empty_idx = empty_idx
        self.random = random
        self.sensor_config = sensor
        self.voxel = voxel
        self.voxel_type = voxel_type
        self.max_voxels = max_voxels
        self.voxel_layer = None

        # Then set the values
        if voxel_encoder:
            self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        if backbone:
            self.backbone = builder.build_backbone(backbone)
        if decode_head:
            self.decode_head = builder.build_neck(decode_head)
        if voxel:
            self.voxel_layer = VoxelizationByGridShape(**voxel_layer)
            
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def extract_feat(self, voxels, coors):
        """Extract features from points."""
        encoded_feats = self.voxel_encoder(voxels, coors)
        x = self.backbone(encoded_feats[0], encoded_feats[1],batch_size=len(voxels))
        # Also return voxel coordinates after scatter for consistent mapping
        return x, encoded_feats[1]

    def forward_train(self,
                points=None,
                train_pts_label=None,
                return_loss=True,
                **kwargs
        ):
        # """
        # Forward training function.
        # """
        
        if self.random:
            points, train_pts_label = self._sampling(points, train_pts_label)
        if isinstance(points, list):
            if all(pl.shape == points[0].shape for pl in points):
                points = torch.cat([pl[None, ...] for pl in points], dim=0)
            else:
                points = [pl[None, ...] for pl in points]
        if isinstance(train_pts_label, list):
            if all(pl.shape == train_pts_label[0].shape for pl in train_pts_label):
                train_pts_label = torch.cat([pl[None, ...] for pl in train_pts_label], dim=0)
            else:
                train_pts_label = [pl[None, ...] for pl in train_pts_label]
        
        voxels, coors, voxel_semantic_mask, point2voxel_maps = self.voxel_layer(points, train_pts_label)
        
        # Handle list of voxels and coors
        if isinstance(voxels, list):
            features_list = []
            voxel_coors_list = []
            for v, c in zip(voxels, coors):
                features, voxel_coors_after_scatter = self.extract_feat(v, c)
                features_list.append(features)
                voxel_coors_list.append(voxel_coors_after_scatter)
            features = features_list
            voxel_coors_after_scatter = voxel_coors_list
        else:
            features, voxel_coors_after_scatter = self.extract_feat(voxels, coors)
        

        if isinstance(features, list):
            outs_list = []
            for f, p, l, v, p2v, pc, vc in zip(features, points, train_pts_label, voxel_semantic_mask, point2voxel_maps, coors, voxel_coors_after_scatter):
                outs = self.decode_head(p, f, point_labels=l, voxel_labels=v, point2voxel_maps=p2v, point_coors=pc, voxel_coors=vc, return_loss=return_loss)
                outs_list.append(outs)
            # Sum up values with the same keys
            outs = {}
            for out_dict in outs_list:
                for key, value in out_dict.items():
                    if key not in outs:
                        outs[key] = value
                    else:
                        outs[key] += value
        else:
            outs = self.decode_head(points, features, point_labels=train_pts_label, voxel_labels=voxel_semantic_mask, point2voxel_maps=point2voxel_maps, point_coors=coors, voxel_coors=voxel_coors_after_scatter, return_loss=return_loss)
        return outs
    
    def simple_test(self,
                points=None,
                train_pts_label=None,
                return_loss=False,
                dataset_flag= None,
        ):
        """Forward testing function for semantic segmentation.
        """
        if isinstance(points, list):
            if all(pl.shape == points[0].shape for pl in points):
                points = torch.cat([pl[None, ...] for pl in points], dim=0)
            else:
                points = [pl[None, ...] for pl in points]
        if isinstance(train_pts_label, list):
            if all(pl.shape == train_pts_label[0].shape for pl in train_pts_label):
                train_pts_label = torch.cat([pl[None, ...] for pl in train_pts_label], dim=0)
            else:
                train_pts_label = [pl[None, ...] for pl in train_pts_label]
                
        voxels, coors, voxel_semantic_mask, point2voxel_maps = self.voxel_layer(points, train_pts_label)
        
        # Handle list of voxels and coors
        if isinstance(voxels, list):
            features_list = []
            voxel_coors_list = []
            for v, c in zip(voxels, coors):
                features, voxel_coors_after_scatter = self.extract_feat(v, c)
                features_list.append(features)
                voxel_coors_list.append(voxel_coors_after_scatter)
            features = features_list
            voxel_coors_after_scatter = voxel_coors_list
        else:
            features, voxel_coors_after_scatter = self.extract_feat(voxels, coors)
        

        if isinstance(features, list):
            predict_labels_list = []
            val_pt_labs_list = []
            for f, p, l, pc, vc in zip(features, points, train_pts_label, coors, voxel_coors_after_scatter):
                predict_labels_pts = self.decode_head(p, f, point_labels=l, point_coors=pc, voxel_coors=vc, return_loss=return_loss)
                predict_labels_pts = predict_labels_pts.squeeze(-1).squeeze(-1)
                predict_labels_pts = torch.argmax(predict_labels_pts, dim=1)  # bs, n
                predict_labels_pts = predict_labels_pts.detach().cpu()
                val_pt_labs = l.squeeze(-1).squeeze(-1).detach().cpu()  # Apply same transformations as predict_labels_pts
                
                predict_labels_list.append(predict_labels_pts)
                val_pt_labs_list.append(val_pt_labs)
            
            # Evaluate each sample separately and combine metrics
            all_hist = None
            for pred, gt in zip(predict_labels_list, val_pt_labs_list):
                hist = self.evaluation_semantic(pred, gt, dataset_flag)
                if all_hist is None:
                    all_hist = hist
                else:
                    all_hist += hist
            
            test_output = {
                'SSC_metric': all_hist,
                'pred': predict_labels_list,  # Keep predictions as a list
            }
        else:
            predict_labels_pts = self.decode_head(points, features, point_labels=train_pts_label, point_coors=coors, voxel_coors=voxel_coors_after_scatter, return_loss=return_loss)
            predict_labels_pts = predict_labels_pts.squeeze(-1).squeeze(-1)
            predict_labels_pts = torch.argmax(predict_labels_pts, dim=1)  # bs, n
            predict_labels_pts = predict_labels_pts.detach().cpu()
            val_pt_labs = train_pts_label.squeeze(-1).reshape(predict_labels_pts.shape[0]).detach().cpu()  # Apply same transformations as predict_labels_pts
            
            IoU_metric = self.evaluation_semantic(predict_labels_pts, val_pt_labs, dataset_flag)
            
            test_output = {
                'SSC_metric': IoU_metric,
                'pred': predict_labels_pts,
            }

        return test_output
    
    def forward_test(self,
                points=None,
                train_pts_label=None,
                return_loss=False,
                dataset_flag= None,
                **kwargs
        ):
        """
        Forward testing function.
        """
        return self.simple_test(points, train_pts_label, return_loss, dataset_flag, )
    
    def evaluation_semantic(self, pred, gt, dataset_flag):
        """Evaluate semantic segmentation results.
        
        Args:
            pred (torch.Tensor): Predicted point cloud labels, shape [B, N, num_classes]
            gt (torch.Tensor): Ground truth point cloud labels, shape [B, N]
            dataset_flag (torch.Tensor): Dataset type flag
            visible_mask (torch.Tensor, optional): Mask for visible points
            
        Returns:
            tuple: (IoU histogram matrix, visible IoU histogram matrix)
        """
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()
        
        # Determine number of classes based on dataset
        if isinstance(dataset_flag, list):
            if len(dataset_flag) == 1:
                dataset_flag = dataset_flag[0].item()
            elif all(flag.item() == dataset_flag[0].item() for flag in dataset_flag):
                dataset_flag = dataset_flag[0].item()
            else:
                raise NotImplementedError("Different dataset_flag when eval")
        else:
            assert isinstance(dataset_flag, torch.Tensor)
            dataset_flag = dataset_flag.item()
            
        if (np.sort(np.unique(gt))[-2] if len(np.unique(gt)) >= 2 else None)  <= 7:
            max_label = 8
        elif dataset_flag == 1:
            max_label = 17
        elif dataset_flag == 2:
            max_label = 20
        elif dataset_flag == 3:
            max_label = 23
            
        # Create mask for valid points (excluding noise/unknown)
        valid_mask = gt != 0
        
        # Compute IoU histogram for all valid points
        hist = fast_hist(pred[valid_mask], gt[valid_mask], max_label=max_label)
            
        return hist
        
    def _sampling(self, points, labels):
        
        LiDAR_height=self.sensor_config.get('LiDAR_height', [1, 2])
        num_of_beams=self.sensor_config.get('num_of_beams', [16, 128]) #[16, 128]
        horizontal_angular_resolution=self.sensor_config.get('horizontal_angular_resolution', [900, 3600]) #[900, 3600]
        lower_vertical_field_of_view_bound=self.sensor_config.get('lower_vertical_field_of_view_bound', [-40, -5])
        upper_vertical_field_of_view_bound=self.sensor_config.get('upper_vertical_field_of_view_bound', [0, 25])
        self.lidar_height = round(random.uniform(LiDAR_height[0], LiDAR_height[1]), 1)
        self.beams = sample_beams(num_of_beams[0], num_of_beams[1])
        self.horizontal_resolution = random.choice(range(horizontal_angular_resolution[0], horizontal_angular_resolution[1]+1, 100))
        self.vertical_lower_angle = round(random.uniform(lower_vertical_field_of_view_bound[0], lower_vertical_field_of_view_bound[1]), 1)
        self.vertical_upper_angle = round(random.uniform(upper_vertical_field_of_view_bound[0], upper_vertical_field_of_view_bound[1]), 1)
        
        vertical_angles_rad = torch.deg2rad(
        torch.linspace(self.vertical_lower_angle, self.vertical_upper_angle, self.beams, dtype=torch.float32)
    )
        horizontal_angles_rad = torch.deg2rad(
        torch.linspace(0, 360, self.horizontal_resolution, dtype=torch.float32)
    )
        ray_dir_grid = torch.stack(torch.meshgrid(vertical_angles_rad, horizontal_angles_rad, indexing='ij'), dim=-1)
        ray_directions = torch.stack([
        torch.cos(ray_dir_grid[..., 0]) * torch.cos(ray_dir_grid[..., 1]),
        torch.cos(ray_dir_grid[..., 0]) * torch.sin(ray_dir_grid[..., 1]),
        torch.sin(ray_dir_grid[..., 0])
    ], dim=-1).reshape(-1, 3)
        lidar_origin = torch.tensor([0, 0, self.lidar_height], dtype=torch.float32)
        
        results_points = []
        results_labels = []
        
        for idx, pt in enumerate(points):
            point_cloud_tensor = pt.to(torch.float32)
            device = point_cloud_tensor.device
            ray_directions_device = ray_directions.to(device)
            lidar_origin_device = lidar_origin.to(device)
            horizontal_angles_rad_device = horizontal_angles_rad.to(device)
            vertical_angles_rad_device = vertical_angles_rad.to(device)
            

            vector_to_points = point_cloud_tensor - lidar_origin_device
            ranges = torch.norm(vector_to_points, dim=1)
            max_range_mask = ranges <= 100.0
            filtered_points = point_cloud_tensor[max_range_mask]
            filtered_labels = labels[idx][max_range_mask]
            vector_to_points = vector_to_points[max_range_mask]

            point_azimuths = torch.atan2(vector_to_points[:, 1], vector_to_points[:, 0]) % (2 * torch.pi)
            point_elevations = torch.asin(vector_to_points[:, 2] / (torch.norm(vector_to_points, dim=1) + 1e-6))

            azimuth_bins = torch.bucketize(point_azimuths, horizontal_angles_rad_device)
            elevation_bins = torch.bucketize(point_elevations, vertical_angles_rad_device)
            azimuth_bins = torch.clamp(azimuth_bins, max=self.horizontal_resolution - 1)
            elevation_bins = torch.clamp(elevation_bins, max=self.beams - 1)
            bin_indices = elevation_bins * self.horizontal_resolution + azimuth_bins

            # 这里不会发生 400K×200K 的广播，因为对于每个点只取了其对应的光线
            ray_dir_points = ray_directions_device[bin_indices]  # shape=(num_filtered_points, 3)
            distances = torch.norm(torch.cross(vector_to_points, ray_dir_points, dim=1), dim=1)
            
            min_distances, min_arg_indices = scatter_min(distances, bin_indices, dim=0, dim_size=self.beams * self.horizontal_resolution)
            valid_mask = (min_distances <= 0.05).nonzero(as_tuple=False).squeeze(1)
            valid_indices = torch.unique(min_arg_indices[valid_mask])  
            valid_indices = valid_indices[(valid_indices > 0) & (valid_indices < filtered_points.shape[0])]
            if valid_indices.numel() > 0:
                sampled_points = filtered_points[valid_indices]
                sampled_labels = filtered_labels[valid_indices]
            else:
                sampled_points = torch.empty((0, 3), device=device)
                sampled_labels = torch.empty((0,), device=device, dtype=labels[idx].dtype)
                        
            results_points.append(sampled_points.to(torch.float32))
            results_labels.append(sampled_labels)
            # save_tensor_as_ply(points[idx], '/home/eason/workspace_perception/UniLiDAR/temp/sampled_points_check.ply')
            
        return results_points, results_labels
        # beam_choices = [16, 32, 64, 128]
        # all_results_points = []
        # all_results_labels = []

        # LiDAR_height = self.sensor_config.get('LiDAR_height', [1, 2])
        # horizontal_angular_resolution = self.sensor_config.get('horizontal_angular_resolution', [900, 3600])
        # lower_vertical_field_of_view_bound = self.sensor_config.get('lower_vertical_field_of_view_bound', [-40, -5])
        # upper_vertical_field_of_view_bound = self.sensor_config.get('upper_vertical_field_of_view_bound', [0, 25])

        # # 只保存一份原始点云和标签（假设points和labels为list，每个元素为一个frame）
        # save_tensor_as_ply(points[0], f'/home/eason/workspace_perception/UniLiDAR/temp/raw_points.ply')
        # labels[0].cpu().numpy().astype(np.uint8).tofile(f'/home/eason/workspace_perception/UniLiDAR/temp/raw_labels.bin')

        # for beam in beam_choices:
        #     self.beams = beam
        #     self.lidar_height = round(random.uniform(LiDAR_height[0], LiDAR_height[1]), 1)
        #     self.horizontal_resolution = random.choice(range(horizontal_angular_resolution[0], horizontal_angular_resolution[1]+1, 100))
        #     self.vertical_lower_angle = round(random.uniform(lower_vertical_field_of_view_bound[0], lower_vertical_field_of_view_bound[1]), 1)
        #     self.vertical_upper_angle = round(random.uniform(upper_vertical_field_of_view_bound[0], upper_vertical_field_of_view_bound[1]), 1)

        #     vertical_angles_rad = torch.deg2rad(
        #         torch.linspace(self.vertical_lower_angle, self.vertical_upper_angle, self.beams, dtype=torch.float32)
        #     )
        #     horizontal_angles_rad = torch.deg2rad(
        #         torch.linspace(0, 360, self.horizontal_resolution, dtype=torch.float32)
        #     )
        #     ray_dir_grid = torch.stack(torch.meshgrid(vertical_angles_rad, horizontal_angles_rad, indexing='ij'), dim=-1)
        #     ray_directions = torch.stack([
        #         torch.cos(ray_dir_grid[..., 0]) * torch.cos(ray_dir_grid[..., 1]),
        #         torch.cos(ray_dir_grid[..., 0]) * torch.sin(ray_dir_grid[..., 1]),
        #         torch.sin(ray_dir_grid[..., 0])
        #     ], dim=-1).reshape(-1, 3)
        #     lidar_origin = torch.tensor([0, 0, self.lidar_height], dtype=torch.float32)

        #     results_points = []
        #     results_labels = []

        #     # 只对第一个点云采样并保存
        #     idx = 0
        #     pt = points[idx]
        #     point_cloud_tensor = pt.to(torch.float32)
        #     device = point_cloud_tensor.device
        #     ray_directions_device = ray_directions.to(device)
        #     lidar_origin_device = lidar_origin.to(device)
        #     horizontal_angles_rad_device = horizontal_angles_rad.to(device)
        #     vertical_angles_rad_device = vertical_angles_rad.to(device)

        #     vector_to_points = point_cloud_tensor - lidar_origin_device
        #     ranges = torch.norm(vector_to_points, dim=1)
        #     max_range_mask = ranges <= 100.0
        #     filtered_points = point_cloud_tensor[max_range_mask]
        #     filtered_labels = labels[idx][max_range_mask]
        #     vector_to_points = vector_to_points[max_range_mask]

        #     point_azimuths = torch.atan2(vector_to_points[:, 1], vector_to_points[:, 0]) % (2 * torch.pi)
        #     point_elevations = torch.asin(vector_to_points[:, 2] / (torch.norm(vector_to_points, dim=1) + 1e-6))

        #     azimuth_bins = torch.bucketize(point_azimuths, horizontal_angles_rad_device)
        #     elevation_bins = torch.bucketize(point_elevations, vertical_angles_rad_device)
        #     azimuth_bins = torch.clamp(azimuth_bins, max=self.horizontal_resolution - 1)
        #     elevation_bins = torch.clamp(elevation_bins, max=self.beams - 1)
        #     bin_indices = elevation_bins * self.horizontal_resolution + azimuth_bins

        #     ray_dir_points = ray_directions_device[bin_indices]
        #     distances = torch.norm(torch.cross(vector_to_points, ray_dir_points, dim=1), dim=1)

        #     min_distances, min_arg_indices = scatter_min(distances, bin_indices, dim=0, dim_size=self.beams * self.horizontal_resolution)
        #     valid_mask = (min_distances <= 0.05).nonzero(as_tuple=False).squeeze(1)
        #     valid_indices = torch.unique(min_arg_indices[valid_mask])
        #     valid_indices = valid_indices[(valid_indices > 0) & (valid_indices < filtered_points.shape[0])]
        #     if valid_indices.numel() > 0:
        #         sampled_points = filtered_points[valid_indices]
        #         sampled_labels = filtered_labels[valid_indices]
        #     else:
        #         sampled_points = torch.empty((0, 3), device=device)
        #         sampled_labels = torch.empty((0,), device=device, dtype=labels[idx].dtype)

        #     results_points.append(sampled_points.to(torch.float32))
        #     results_labels.append(sampled_labels)
        #     # 保存采样点云和标签，文件名加上beam
        #     save_tensor_as_ply(results_points[0], f'/home/eason/workspace_perception/UniLiDAR/temp/sampled_points_check_{beam}.ply')
        #     sampled_labels.cpu().numpy().astype(np.uint8).tofile(f'/home/eason/workspace_perception/UniLiDAR/temp/sampled_labels_{beam}.bin')

        #     all_results_points.append(results_points)
        #     all_results_labels.append(results_labels)

        # return all_results_points, all_results_labels
    
def save_tensor_as_ply(tensor, file_path):
    # 将 tensor 移动到 CPU 上并转换为 numpy 数组
    tensor_cpu = tensor.cpu().numpy()

    # 创建 open3d 点云对象
    point_cloud = o3d.geometry.PointCloud()

    # 设置点云的点
    point_cloud.points = o3d.utility.Vector3dVector(tensor_cpu[:, :3])

    # 如果 tensor 包含颜色信息（假设颜色在第4到第6列）
    if tensor_cpu.shape[1] >= 6:
        point_cloud.colors = o3d.utility.Vector3dVector(tensor_cpu[:, 3:6] / 255.0)

    # 保存为 .ply 文件
    o3d.io.write_point_cloud(file_path, point_cloud)

def sample_beams(lower_beams_bound, upper_beams_bound):
    # Define the fixed variance
    variance = 16
    std_dev = np.sqrt(variance)
    
    candidate_means = [16, 32, 64, 80, 128]
    # Filter means within the given range
    valid_means = [m for m in candidate_means if lower_beams_bound <= m <= upper_beams_bound]
    

    beam_range = np.arange(lower_beams_bound, upper_beams_bound + 1)
    # Calculate the probability density
    probability_density = np.zeros_like(beam_range, dtype=float)
    for mean in valid_means:
        # Add the normal distribution for each mean
        probability_density += np.exp(-0.5 * ((beam_range - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))
    
    # Normalize the probability density
    probability_density /= probability_density.sum()
    
    # Sample from the range based on the computed probability density
    # sampled_beam =int(np.random.choice(beam_range, p=probability_density))
    sampled_beam = int(np.random.choice(valid_means))
    return sampled_beam

    
def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]

    return nuScenes_label_name

def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)
