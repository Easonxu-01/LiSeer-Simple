'''
Author: EASON XU
Date: 2023-10-03 01:57:21
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-05-31 09:00:33
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/datasets/pipelines/cylinder_voxelize.py
'''
import numba as nb
import numpy as np
from mmcv.runner import auto_fp16, force_fp32
from mmdet.datasets.builder import PIPELINES
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from mmengine.config import ConfigDict
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
from mmcv.utils import ext_loader
from itertools import repeat
import numpy as np
import collections
import torch
from torch.autograd import Function
from torch import nn
from torch.nn import functional as F

ext_module = ext_loader.load_ext('_ext', [
    'dynamic_voxelize_forward', 'hard_voxelize_forward',
])

class _Voxelization(Function):
    
    @staticmethod
    def forward(
            ctx: Any,
            points: torch.Tensor,
            voxel_size: Union[tuple, float],
            coors_range: Union[tuple, float],
            max_points: int = 35,
            max_voxels: int = 20000,
            deterministic: bool = True) -> Union[Tuple[torch.Tensor], Tuple]:
        
        # """Convert kitti points(N, >=3) to voxels.

        # Args:
        #     points (torch.Tensor): [N, ndim]. Points[:, :3] contain xyz points
        #         and points[:, 3:] contain other information like reflectivity.
        #     voxel_size (tuple or float): The size of voxel with the shape of
        #         [3].
        #     coors_range (tuple or float): The coordinate range of voxel with
        #         the shape of [6].
        #     max_points (int, optional): maximum points contained in a voxel. if
        #         max_points=-1, it means using dynamic_voxelize. Default: 35.
        #     max_voxels (int, optional): maximum voxels this function create.
        #         for second, 20000 is a good choice. Users should shuffle points
        #         before call this function because max_voxels may drop points.
        #         Default: 20000.
        #     deterministic: bool. whether to invoke the non-deterministic
        #         version of hard-voxelization implementations. non-deterministic
        #         version is considerablly fast but is not deterministic. only
        #         affects hard voxelization. default True. for more information
        #         of this argument and the implementation insights, please refer
        #         to the following links:
        #         https://github.com/open-mmlab/mmdetection3d/issues/894
        #         https://github.com/open-mmlab/mmdetection3d/pull/904
        #         it is an experimental feature and we will appreciate it if
        #         you could share with us the failing cases.

        # Returns:
        #     tuple[torch.Tensor]: tuple[torch.Tensor]: A tuple contains three
        #     elements. The first one is the output voxels with the shape of
        #     [M, max_points, n_dim], which only contain points and returned
        #     when max_points != -1. The second is the voxel coordinates with
        #     shape of [M, 3]. The last is number of point per voxel with the
        #     shape of [M], which only returned when max_points != -1.
        # """
        
        if max_points == -1 or max_voxels == -1:
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
            ext_module.dynamic_voxelize_forward(
                points,
                torch.tensor(voxel_size, dtype=torch.float),
                torch.tensor(coors_range, dtype=torch.float),
                coors,
                NDim=3)
            return coors
        else:
            voxels = points.new_zeros(
                size=(max_voxels, max_points, points.size(1)))
            coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
            num_points_per_voxel = points.new_zeros(
                size=(max_voxels, ), dtype=torch.int)
            voxel_num = torch.zeros(size=(), dtype=torch.long)
            ext_module.hard_voxelize_forward(
                points,
                torch.tensor(voxel_size, dtype=torch.float),
                torch.tensor(coors_range, dtype=torch.float),
                voxels,
                coors,
                num_points_per_voxel,
                voxel_num,
                max_points=max_points,
                max_voxels=max_voxels,
                NDim=3,
                deterministic=deterministic)
            # select the valid voxels
            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
            return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply


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
        else:
            raise ValueError('must assign a value to voxel_size or grid_shape')

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return voxelization(input, self.voxel_size, self.point_cloud_range,
                            self.max_num_points, max_voxels,
                            self.deterministic)

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

@force_fp32()
@PIPELINES.register_module()
class cylinder_voxelize(object):
    def __init__(self, voxel=True, voxel_type='cylindrical', max_voxels: Optional[int] = None, voxel_layer: OptConfigType = None):
        self.voxel = voxel
        self.voxel_type = voxel_type
        self.max_voxels = max_voxels
        if voxel:
            self.voxel_layer = VoxelizationByGridShape(**voxel_layer)
    
    def __call__(self, results):
        # lidar branch
        if 'points' in results.keys():
            points = results['points']
            if self.voxel:
                if self.voxel_type == 'hard':
                    voxels, coors, num_points, voxel_centers = self.voxelize(points)
                    results['voxels'] = voxels
                    results['coors'] = coors
                    results['num_points'] = num_points
                    results['voxel_centers'] = voxel_centers
                else:
                    voxels, coors = self.voxelize(points)
                    results['voxels'] = voxels
                    results['coors'] = coors

        return results
    
    @torch.no_grad()
    def voxelize(self, points):
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """
        if self.voxel_type == 'hard':
            voxels, coors, num_points, voxel_centers = [], [], [], []
            for i, res in enumerate(points):
                res_voxels, res_coors, res_num_points = self.voxel_layer(res)
                res_voxel_centers = (
                    res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                        self.voxel_layer.voxel_size) + res_voxels.new_tensor(
                            self.voxel_layer.point_cloud_range[0:3])
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num_points)
                voxel_centers.append(res_voxel_centers)

            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)
            num_points = torch.cat(num_points, dim=0)
            voxel_centers = torch.cat(voxel_centers, dim=0)
            
            return voxels, coors, num_points, voxel_centers
        
        elif self.voxel_type == 'dynamic':
            coors = []
            # dynamic voxelization only provide a coors mapping
            for i, res in enumerate(points):
                res_coors = self.voxel_layer(res)
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                coors.append(res_coors)
            voxels = torch.cat(points, dim=0)
            coors = torch.cat(coors, dim=0)
            
            return voxels, coors
        elif self.voxel_type == 'cylindrical':
            voxels, coors = [], []
            for res in enumerate(points):
                rho = torch.sqrt(res[:, 0]**2 + res[:, 1]**2)
                phi = torch.atan2(res[:, 1], res[:, 0])
                polar_res = torch.stack((rho, phi, res[:, 2]), dim=-1)
                min_bound = polar_res.new_tensor(
                    self.voxel_layer.point_cloud_range[:3])
                max_bound = polar_res.new_tensor(
                    self.voxel_layer.point_cloud_range[3:])
                try:  # only support PyTorch >= 1.9.0
                    polar_res_clamp = torch.clamp(polar_res, min_bound,
                                                  max_bound)
                except TypeError:
                    polar_res_clamp = polar_res.clone()
                    for coor_idx in range(3):
                        polar_res_clamp[:, coor_idx][
                            polar_res[:, coor_idx] >
                            max_bound[coor_idx]] = max_bound[coor_idx]
                        polar_res_clamp[:, coor_idx][
                            polar_res[:, coor_idx] <
                            min_bound[coor_idx]] = min_bound[coor_idx]
                res_coors = torch.floor(
                    (polar_res_clamp - min_bound) / polar_res_clamp.new_tensor(
                        self.voxel_layer.voxel_size)).int()
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                res_voxels = torch.cat((polar_res, res[:, :2], res[:, 3:]),
                                       dim=-1)
                voxels.append(res_voxels)
                coors.append(res_coors)
            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)

        else:
            raise ValueError(f'Invalid voxelization type {self.voxel_type}')

        return voxels, coors
        
@nb.jit('u1[:,:,:](u1[:,:,:],i4[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

        
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)
