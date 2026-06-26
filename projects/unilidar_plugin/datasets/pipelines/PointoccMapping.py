import numpy as np
import numba as nb
import torch
from mmdet.datasets.builder import PIPELINES
import yaml
from mmcv.parallel import DataContainer
import os
import mmcv
from mmdet3d.core.points import LiDARPoints
from itertools import product

@PIPELINES.register_module()
class PointoccMapping(object):
    """Map original semantic class to valid category ids.

    Required Keys:

    - seg_label_mapping (np.ndarray)
    - pts_semantic_mask (np.ndarray)

    Added Keys:

    - points (np.float32)

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    """
    def __init__(self, occ_path, grid_size, grid_size_vox=None, grid_size_occ=None, coarse_ratio=4, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], fill_label=0, unique_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 3], min_volume_space=[0, -np.pi, -5], cal_visible=False, RPR = False, restrict_pc_range = [-25.6, -25.6, -3.4, 25.6, 25.6, 3.0],
                 unilidar=False,):
        'Initialization'
        self.occ_path = occ_path
        self.grid_size = np.asarray(grid_size).astype(np.int32)
        self.grid_size_vox = np.asarray(grid_size_vox).astype(np.int32)
        self.grid_size_occ = np.asarray(grid_size_occ).astype(np.int32)
        self.grid_size_occ_coarse = (np.asarray(grid_size_occ) / coarse_ratio).astype(np.int32)
        self.pc_range = np.asarray(pc_range).astype(np.float32)
        self.voxel_size = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size_occ
        self.voxel_size_coarse = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size_occ_coarse
        self.fill_label = fill_label
        self.unique_label = unique_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.cal_visible = cal_visible
        self.unilidar = unilidar
        self.cascade_ratio = torch.tensor(coarse_ratio)
        if isinstance(RPR, tuple):
            RPR = RPR[0]
        self.RPR = RPR,
        # if self.unilidar:
        #     self.grid_size_occ_coarse = (np.asarray([256, 256, 32]) / coarse_ratio).astype(np.int32)
        #     self.voxel_size_coarse = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size_occ_coarse
        self.restrict_pc_range = np.array(restrict_pc_range)
        # if self.RPR:
        #     self.voxel_size = (self.restrict_pc_range[3:] - self.restrict_pc_range[:3]) / self.grid_size_occ
        #     self.voxel_size_coarse = (self.restrict_pc_range[3:] - self.restrict_pc_range[:3]) / self.grid_size_occ_coarse

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
            Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        if results.get('voxel_semantic_mask') is not None:
            if isinstance(results['voxel_semantic_mask'], np.ndarray):
                results['processed_label'] = torch.from_numpy(results['voxel_semantic_mask'])
            else:
                results['processed_label'] = results['voxel_semantic_mask']
            
        else:
            rel_path = 'scene_{0}/occupancy/{1}.npy'.format(results['scene_token'], results['lidar_token'])
            #  [z y x cls] or [z y x vx vy vz cls]
            pcd = np.load(os.path.join(self.occ_path, rel_path))
            results['pcd'] = pcd
            # 确保必要的键存在
            assert 'pcd' in results
            pcd = results['pcd'].data if isinstance(results['pcd'], DataContainer) else results['pcd']
            occ_label = pcd[..., -1:]
            occ_label[occ_label==0] = 255
            occ_xyz_grid = pcd[..., [2,1,0]]  # x y z
            # process labels
            label_voxel_pair = np.concatenate([occ_xyz_grid, occ_label], axis=-1)
            label_voxel_pair = label_voxel_pair[np.lexsort((occ_xyz_grid[:, 0], occ_xyz_grid[:, 1], occ_xyz_grid[:, 2])), :].astype(np.int32)
            processed_label = np.ones(self.grid_size_occ, dtype=np.uint8) * self.fill_label
            processed_label = nb_process_label(processed_label, label_voxel_pair)
            results['processed_label'] = torch.from_numpy(processed_label)
            
        assert 'points' in results
        points_ori = results['points'].data if isinstance(results['points'], DataContainer) else results['points']
        if isinstance(points_ori, LiDARPoints):
            results['points'] = points_ori.tensor
            points = points_ori.tensor
            xyz, feat = points[:, :3], points[:, 3:]
            xyz_pol = cart2polar(xyz)
        else:
        # random points augmentation by rotation
            xyz, feat = points_ori[:, :3], points_ori[:, 3:]
            xyz_pol = cart2polar(xyz)

        assert self.fixed_volume_space
        max_bound = np.asarray(self.max_volume_space)
        min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        intervals = crop_range / (self.grid_size)
        intervals_vox = crop_range / (self.grid_size_vox)
        if (intervals == 0).any(): 
            print("Zero interval!")
        xyz_pol_grid = np.clip(xyz_pol, min_bound, max_bound - 1e-3)
        grid_ind = (np.floor((xyz_pol_grid - min_bound) / intervals)).astype(np.int32)
        # get voxel_position_grid_coarse
        if self.RPR and results.get('voxel_semantic_mask') is None:
            dim_array = np.ones(len(self.grid_size_occ_coarse) + 1, np.int32)
            dim_array[0] = -1
            voxel_position_coarse = ((np.indices(self.grid_size_occ_coarse) + 0.5) * self.voxel_size_coarse.reshape(dim_array) + self.restrict_pc_range[:3].reshape(dim_array)).reshape(3, -1).transpose(1,0)
            voxel_position_grid_coarse = (np.clip(cart2polar(voxel_position_coarse), min_bound, max_bound - 1e-3) - min_bound) / intervals_vox
        elif self.RPR and results.get('voxel_semantic_mask') is not None:
            dim_array = np.ones(len(self.grid_size_occ_coarse) + 1, np.int32)
            dim_array[0] = -1
            voxel_position_coarse = ((np.indices(self.grid_size_occ_coarse) + 0.5) * self.voxel_size_coarse.reshape(dim_array) + self.restrict_pc_range[:3][:3].reshape(dim_array)).reshape(3, -1).transpose(1,0)
            voxel_position_grid_coarse = (np.clip(cart2polar(voxel_position_coarse), min_bound, max_bound - 1e-3) - min_bound) / intervals_vox
        else:
            dim_array = np.ones(len(self.grid_size_occ_coarse) + 1, np.int32)
            dim_array[0] = -1
            voxel_position_coarse = ((np.indices(self.grid_size_occ_coarse) + 0.5) * self.voxel_size_coarse.reshape(dim_array) + self.pc_range[:3].reshape(dim_array)).reshape(3, -1).transpose(1,0)
            voxel_position_grid_coarse = (np.clip(cart2polar(voxel_position_coarse), min_bound, max_bound - 1e-3) - min_bound) / intervals_vox
        
        results['train_grid_vox_coarse'] = torch.from_numpy(voxel_position_grid_coarse).to(torch.float32)
        

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_feat = np.concatenate((return_xyz, xyz_pol, xyz[:, :2], feat), axis=1)
        
        results['voxel_label'] = torch.from_numpy(grid_ind).to(torch.float32)
        results['train_grid'] = torch.from_numpy(return_feat).to(torch.float32)
        
        unique_label = np.asarray(self.unique_label)
        results['unique_label'] = torch.from_numpy(unique_label)
        
        if results.get('valid_mask') is not None:
            results['visible_mask'] = results['valid_mask']
        elif results.get('visible_mask') is None and self.cal_visible:
            visible_mask = np.zeros(torch.tensor(results['processed_label'].shape).numpy(), dtype=np.uint8)
            
            # lidar branch
            if 'points' in results.keys():
                pts = results['points'].cpu().numpy()[:, :3]
                if not self.RPR:
                    pts_in_range = ((pts>=self.pc_range[:3]) & (pts<self.pc_range[3:])).sum(1)==3
                    pts = pts[pts_in_range]
                    pts = (pts - self.pc_range[:3])/self.voxel_size
                else:
                    pts_in_range = ((pts>=self.restrict_pc_range[:3]) & (pts<self.restrict_pc_range[3:])).sum(1)==3
                    pts = pts[pts_in_range]
                    pts = (pts - self.restrict_pc_range[:3])/self.voxel_size
                pts = np.concatenate([pts, np.ones((pts.shape[0], 1)).astype(pts.dtype)], axis=1) 
                pts = pts[np.lexsort((pts[:, 0], pts[:, 1], pts[:, 2])), :].astype(np.int64)
                pts_occ_label = np.zeros(torch.tensor(results['processed_label'].shape).numpy(), dtype=np.uint8)
                voxel_pts = nb_process_label(pts_occ_label, pts)  # W H D 1:occupied 0:free
                visible_mask = visible_mask | voxel_pts
                results['lidar_visible_mask'] = voxel_pts
            
            # if isinstance(self.RPR, tuple):
            #     self.RPR = self.RPR[0]
            # if self.RPR == True:
            #     restricted_visible_mask = visible_mask[start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2]]
            #     results['visible_mask'] = torch.from_numpy(restricted_visible_mask)
            # else:
            #     results['visible_mask'] = torch.from_numpy(visible_mask)
            results['visible_mask'] = torch.from_numpy(visible_mask)
            

        return results
    
    
@PIPELINES.register_module()
class PointsegMapping(object):
    """Map original semantic class to valid category ids.

    Required Keys:

    - seg_label_mapping (np.ndarray)
    - pts_semantic_mask (np.ndarray)

    Added Keys:

    - points (np.float32)

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    """
    def __init__(self, grid_size, grid_size_vox, coarse_ratio=4, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], fill_label=0, unique_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 3], min_volume_space=[0, -np.pi, -5], cal_visible=False, RPR = False, restrict_pc_range = [-25.6, -25.6, -3.4, 25.6, 25.6, 3.0],
                 unilidar=False,file_client_args=dict(backend='disk')):
        'Initialization'
        self.grid_size = np.asarray(grid_size).astype(np.int32)
        self.grid_size_vox = np.asarray(grid_size_vox).astype(np.int32)
        self.pc_range = np.asarray(pc_range).astype(np.float32)
        self.fill_label = fill_label
        self.unique_label = unique_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.cal_visible = cal_visible
        self.unilidar = unilidar
        self.cascade_ratio = torch.tensor(coarse_ratio)
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        if isinstance(RPR, tuple):
            RPR = RPR[0]
        self.RPR = RPR,
        self.restrict_pc_range = np.array(restrict_pc_range)

        # 读取yaml文件
        label_mapping = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                   'configs', 'pointocc', 'label_mapping', 'nuscenes.yaml')
        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
            Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        if results.get('pts_semantic_mask') is not None:
            labels = results['pts_semantic_mask'].reshape(-1, 1)
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
        else:
            assert 'lidarseg_labels_filename' in results
            if self.file_client is None:
                self.file_client = mmcv.FileClient(**self.file_client_args)

            def _load_lidarseg_labels(lidarseg_labels_filename, rpr_indices=None):
                try:
                    mmcv.check_file_exist(lidarseg_labels_filename)
                    if lidarseg_labels_filename.endswith('.bin'):
                        points_label = np.fromfile(
                            lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
                    else:
                        points_label = np.fromfile(
                            lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
                except ConnectionError:
                    labels_bytes = self.file_client.get(lidarseg_labels_filename)
                    points_label = np.frombuffer(labels_bytes, dtype=np.uint8).reshape([-1, 1])
                points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
                mapped = points_label.astype(np.uint8)
                rpr = self.RPR[0] if isinstance(self.RPR, tuple) else self.RPR
                if rpr and rpr_indices is not None:
                    mapped = mapped[rpr_indices]
                return mapped

            if results.get('dense7sparse'):
                sparse_labels = _load_lidarseg_labels(
                    results['lidarseg_labels_filename'],
                    results.get('RPR_indices_sparse'))
                dense_labels = _load_lidarseg_labels(
                    results['dense_lidarseg_labels_filename'],
                    results.get('RPR_indices_dense'))
                labels = np.concatenate([sparse_labels, dense_labels], axis=0)
            else:
                labels = _load_lidarseg_labels(
                    results['lidarseg_labels_filename'],
                    results.get('RPR_indices'))
            results['pts_semantic_mask'] = torch.from_numpy(labels)
        
        assert 'points' in results
        points_ori = results['points'].data if isinstance(results['points'], DataContainer) else results['points']
        if isinstance(points_ori, LiDARPoints):
            results['points'] = points_ori.tensor
            points = points_ori.tensor
            xyz, feat = points[:, :3], points[:, 3:]
            xyz_pol = cart2polar(xyz)
        else:
        # random points augmentation by rotation
            xyz, feat = points_ori[:, :3], points_ori[:, 3:]
            xyz_pol = cart2polar(xyz)

        assert self.fixed_volume_space
        max_bound = np.asarray(self.max_volume_space)
        min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        intervals = crop_range / (self.grid_size)
        intervals_vox = crop_range / (self.grid_size_vox)
        if (intervals == 0).any(): 
            print("Zero interval!")
        xyz_pol_grid = np.clip(xyz_pol, min_bound, max_bound - 1e-3)
        grid_ind = (np.floor((xyz_pol_grid - min_bound) / intervals)).astype(np.int32)
        grid_ind_vox = (np.floor((xyz_pol_grid - min_bound) / intervals_vox)).astype(np.int32)
        grid_ind_vox_float = ((xyz_pol_grid - min_bound) / intervals_vox).astype(np.float32)
        results['grid_ind'] = torch.from_numpy(grid_ind).to(torch.float32)
        results['grid_ind_vox'] = torch.from_numpy(grid_ind_vox_float).to(torch.float32)

        # process labels
        processed_label = np.ones(self.grid_size_vox, dtype=np.uint8) * self.fill_label
        label_voxel_pair = np.concatenate([grid_ind_vox, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind_vox[:, 0], grid_ind_vox[:, 1], grid_ind_vox[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        results['train_voxel_label'] = torch.from_numpy(processed_label).type(torch.LongTensor)
        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_feat = np.concatenate((return_xyz, xyz_pol, xyz[:, :2], feat), axis=1)
        
        results['train_pts_label'] = torch.from_numpy(labels).squeeze(-1)
        results['train_grid'] = torch.from_numpy(return_feat).to(torch.float32)
        
        unique_label = np.asarray(self.unique_label)
        results['unique_label'] = torch.from_numpy(unique_label)
        
        # Save parameters for data augmentation to recompute train_grid
        results['_pointseg_grid_size'] = self.grid_size.tolist() if isinstance(self.grid_size, np.ndarray) else list(self.grid_size)
        results['_pointseg_grid_size_vox'] = self.grid_size_vox.tolist() if isinstance(self.grid_size_vox, np.ndarray) else list(self.grid_size_vox)
        results['_pointseg_max_volume_space'] = list(self.max_volume_space)
        results['_pointseg_min_volume_space'] = list(self.min_volume_space)
        results['_pointseg_fill_label'] = self.fill_label
            
        return results


    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str
    
    
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)
    
@nb.jit([
    'u1[:,:,:](u1[:,:,:], i4[:,:])',
    'u1[:,:,:](u1[:,:,:], i8[:,:])'
], nopython=True, cache=True)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    """
    处理标签数据，支持任意整数类型的 `sorted_label_voxel_pair`。
    """
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
