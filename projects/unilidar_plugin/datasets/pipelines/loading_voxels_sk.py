'''
Author: EASON XU
Date: 2023-12-21 13:07:43
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2026-05-12 20:43:29
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/datasets/pipelines/loading_voxels_sk.py
'''
import mmcv
from mmcv.parallel import DataContainer
import numpy as np
import numba as nb
import open3d as o3d
from mmdet.datasets.builder import PIPELINES
import torch
from scipy import stats
from scipy.ndimage import zoom
from skimage import transform
import torch.nn.functional as F
from mmdet3d.core.points import get_points_type, LiDARPoints
import random
import os
import shutil

@PIPELINES.register_module()
class LoadPointsFromFile_RPR(object):
    """Load Points From File and restrict the range of point clouds.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 RPR=False,
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 dataset_flag = 1,
                 shift_coors=[0, 0, 0],
                 Random = False,
                 dense7sparse=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.point_cloud_range = np.array(point_cloud_range)
        self.shift_coors = np.array(shift_coors)
        if isinstance(RPR, tuple):
            RPR = RPR[0]
        self.RPR = RPR
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.dataset_flag = [dataset_flag]
        self.random = Random
        self.dense7sparse = dense7sparse

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:s
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            elif pts_filename.endswith('.ply'):
                pcd = o3d.io.read_point_cloud(pts_filename)
                points = np.asarray(pcd.points)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        except ConnectionError:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
            
        return torch.from_numpy(points)

    def _process_points(self, pts_filename, load_dim=None, use_dim=None,
                        reset_random=False):
        """Load one point cloud, apply RPR and optional shift_height."""
        if load_dim is None:
            load_dim = self.load_dim
        if use_dim is None:
            use_dim = list(self.use_dim)

        points = self._load_points(pts_filename)
        use_dim = list(use_dim)

        if reset_random and self.random and points.dim() == 1:
            raw_layout_by_flag = {
                1: 5,      # nuScenes raw
                2: 4,      # SemanticKITTI raw
                3: 6,      # Waymo raw
            }
            dataset_flag = int(self.dataset_flag[0]) if len(self.dataset_flag) > 0 else -1
            raw_load_dim = raw_layout_by_flag.get(dataset_flag, load_dim)
            if raw_load_dim > 0 and (points.numel() % raw_load_dim == 0):
                load_dim = raw_load_dim

        if points.dim() == 1:
            if load_dim <= 0 or (points.numel() % load_dim != 0):
                candidate_dims = [load_dim, 6, 5, 4, 3]
                for cand in candidate_dims:
                    if cand > 0 and (points.numel() % cand == 0):
                        load_dim = cand
                        break
            points = points.reshape(-1, load_dim)
        elif points.dim() == 2:
            load_dim = points.shape[1]
        else:
            raise ValueError(
                f'Unsupported points shape: {tuple(points.shape)} for file {pts_filename}')

        use_dim = [d for d in use_dim if d < load_dim]
        if len(use_dim) == 0:
            use_dim = list(range(min(3, load_dim)))
        points = points[:, use_dim]

        indices = np.where(
            (points[:, 0] >= self.point_cloud_range[0])
            & (points[:, 0] < self.point_cloud_range[3])
            & (points[:, 1] >= self.point_cloud_range[1])
            & (points[:, 1] < self.point_cloud_range[4])
            & (points[:, 2] >= self.point_cloud_range[2])
            & (points[:, 2] < self.point_cloud_range[5]))[0]

        rpr = self.RPR[0] if isinstance(self.RPR, tuple) else self.RPR
        if rpr:
            points = points[indices]

        if self.shift_height:
            shifted_points = points[:, :3] - self.shift_coors
            points[:, :3] = shifted_points

        return points, indices

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        reset_random = bool(results.get('reset_random', False))
        use_dense7sparse = (
            self.random and self.dense7sparse
            and results.get('dense_pts_filename') is not None)

        if use_dense7sparse:
            raw_layout_by_flag = {1: 5, 2: 4, 3: 6}
            sparse_load_dim = raw_layout_by_flag.get(
                int(self.dataset_flag[0]), self.load_dim)
            sparse_points, sparse_indices = self._process_points(
                results['pts_filename'],
                load_dim=sparse_load_dim,
                use_dim=self.use_dim,
                reset_random=reset_random)
            dense_points, dense_indices = self._process_points(
                results['dense_pts_filename'],
                load_dim=3,
                use_dim=self.use_dim,
                reset_random=False)
            points = np.concatenate([sparse_points, dense_points], axis=0)
            results['dense7sparse'] = True
            results['RPR_indices_sparse'] = sparse_indices
            results['RPR_indices_dense'] = dense_indices
        else:
            pts_filename = results['pts_filename']
            if not self.random:
                if 'LIDAR_TOP/refine' in pts_filename:
                    pts_filename = pts_filename.replace('LIDAR_TOP/refine', 'LIDAR_TOP', 1)
                    pts_filename = pts_filename.replace('.ply', '.bin')
                if 'refine' in pts_filename:
                    pts_filename = pts_filename.replace('refine', 'velodyne', 1)
                    pts_filename = pts_filename.replace('.ply', '.bin')
            points, indices = self._process_points(
                pts_filename, reset_random=reset_random)
            results['RPR_indices'] = indices

        attribute_dims = None
        if self.use_color:
            assert len(self.use_dim) >= 6
            attribute_dims = dict(color=[
                points.shape[1] - 3,
                points.shape[1] - 2,
                points.shape[1] - 1,
            ])

        if self.dataset_flag[0] == 1:
            points_class = get_points_type(self.coord_type)
            points = points_class(
                points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        results['dataset_flag'] = self.dataset_flag
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile_Sampling(object):
    """Load Points From File and restrict the range of point clouds.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 use_color=False,
                 shift_coors=[0, 0, 0],
                 file_client_args=dict(backend='disk'),
                 LiDAR_height=[1, 2],
                 num_of_beams=[16, 128],
                 horizontal_angular_resolution=[900, 3600],
                 lower_vertical_field_of_view_bound=[-40, -5],
                 upper_vertical_field_of_view_bound=[0, 25],):

        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.shift_coors = np.array(shift_coors)
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.lidar_height = round(random.uniform(LiDAR_height[0], LiDAR_height[1]), 1)
        self.beams = random.randint(num_of_beams[0], num_of_beams[1])
        self.horizontal_resolution = random.choice(range(horizontal_angular_resolution[0], horizontal_angular_resolution[1]+1, 100))
        self.vertical_lower_angle = round(random.uniform(lower_vertical_field_of_view_bound[0], lower_vertical_field_of_view_bound[1]), 1)
        self.vertical_upper_angle = round(random.uniform(upper_vertical_field_of_view_bound[0], upper_vertical_field_of_view_bound[1]), 1)

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            elif pts_filename.endswith('.ply'):
                pcd = o3d.io.read_point_cloud(pts_filename)
                points = np.asarray(pcd.points)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        except ConnectionError:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)

        # points = self._sampling_numpy(points)
        return torch.from_numpy(points)

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clou;ds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        # points_2 = self._sampling(points.cpu().numpy())
        # points_1 = self._sampling_numpy(points.cpu().numpy())
        attribute_dims = None
            

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))
        
        # if len(points.shape)==2:
        #     points = points.reshape(1, points.shape[0], points.shape[1])
        # if points.shape[1]==5:
        # points_class = get_points_type(self.coord_type)
        # points = points_class(
        #     points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        # repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadVoxels(object):
    def __init__(
            self,
            to_float32=True,
            use_semantic=False,
            cylinder=False,
            occ_path=None,
            grid_size=[512, 512, 40],
            unoccupied=0,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            RPR=False,
            restrict_pc_range=[-25.6, -25.6, -3.4, 25.6, 25.6, 3.0],
            gt_resize_ratio=1,
            cal_visible=False,
            use_vel=False,
            random=False,
            dummy_pts_semantic=False,
            file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.use_semantic = use_semantic
        self.cylinder = cylinder
        self.occ_path = occ_path
        self.cal_visible = cal_visible
        self.random = random
        self.dummy_pts_semantic = dummy_pts_semantic
        self.grid_size = np.array(grid_size)
        self.unoccupied = unoccupied
        self.pc_range = pc_range
        self.RPR = RPR
        self.restrict_pc_range = restrict_pc_range
        self.voxel_size = (np.array(self.pc_range[3:]) - np.array(self.pc_range[:3])) / self.grid_size
        self.gt_resize_ratio = gt_resize_ratio
        self.use_vel = use_vel
        self.file_client_args = file_client_args
        self.file_client = None

    def _load_pts_semantic_mask(self, pts_semantic_mask_path, is_dense=False):
        """Load point-wise semantic labels for SemanticKITTI / Waymo."""
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            if 'waymo' in pts_semantic_mask_path:
                mmcv.check_file_exist(pts_semantic_mask_path)
                if is_dense:
                    pts_semantic_mask = np.fromfile(
                        pts_semantic_mask_path, dtype=np.int32).reshape(-1, 1)
                else:
                    pts_semantic_mask = np.fromfile(
                        pts_semantic_mask_path, dtype=np.int32).reshape(-1, 2)[:, 1]
                    pts_semantic_mask = pts_semantic_mask.reshape(-1, 1)
            else:
                mmcv.check_file_exist(pts_semantic_mask_path)
                pts_semantic_mask = np.fromfile(
                    pts_semantic_mask_path, dtype=np.uint32).astype(np.int64)
                pts_semantic_mask = (pts_semantic_mask % (2 ** 16)).reshape(-1, 1)
        except ConnectionError:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            pts_semantic_mask = np.frombuffer(mask_bytes, dtype='int').copy()
            if pts_semantic_mask.ndim == 1:
                pts_semantic_mask = pts_semantic_mask.reshape(-1, 1)
        return pts_semantic_mask

    @staticmethod
    def _num_points_from_results(results):
        """Point count N aligned with LoadPoints / PointsegMapping ``results['points']``."""
        pts = results['points']
        while isinstance(pts, DataContainer):
            pts = pts.data
        if hasattr(pts, 'tensor'):
            pts = pts.tensor
        if isinstance(pts, torch.Tensor):
            t = pts
            if t.dim() == 3:
                return int(t.shape[1])
            return int(t.shape[0])
        pts = np.asarray(pts)
        if pts.ndim == 3:
            return int(pts.shape[1])
        return int(pts.shape[0])

    def __call__(self, results):
        
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        results.setdefault('pts_seg_fields', [])
        results.setdefault('seg_fields', [])

        reset_random = bool(results.get('reset_random', False))
        effective_random = self.random and (not reset_random)

        if self.dummy_pts_semantic:
            n = self._num_points_from_results(results)
            # Match non-Waymo branch: 1d int64 labels (unused at inference).
            pts_semantic_mask = np.zeros((n,), dtype=np.int64)
            pts_semantic_mask_path = results.get('pts_semantic_mask_path', '')
        elif results.get('dense7sparse'):
            pts_semantic_mask_path = results['pts_semantic_mask_path']
            sparse_mask = self._load_pts_semantic_mask(
                results['pts_semantic_mask_path'], is_dense=False)
            dense_mask = self._load_pts_semantic_mask(
                results['dense_pts_semantic_mask_path'], is_dense=True)
            rpr = self.RPR[0] if isinstance(self.RPR, tuple) else self.RPR
            if rpr:
                sparse_mask = sparse_mask[results['RPR_indices_sparse']]
                dense_mask = dense_mask[results['RPR_indices_dense']]
            pts_semantic_mask = np.concatenate([sparse_mask, dense_mask], axis=0)
        else:
            pts_semantic_mask_path = results['pts_semantic_mask_path']
            if 'waymo' in pts_semantic_mask_path:
                pts_semantic_mask = self._load_pts_semantic_mask(
                    pts_semantic_mask_path, is_dense=effective_random)
            else:
                pts_semantic_mask = self._load_pts_semantic_mask(
                    pts_semantic_mask_path, is_dense=False)
        if results.get('voxel_path', None) is not None and self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        if results.get('voxel_path', None) is not None:
            voxel_path = results['voxel_path'] 
            voxel_semantic_mask_path = results['voxel_semantic_mask_path'] 
            voxel_occ_mask_path = results['voxel_occ_mask_path'] 
            voxel_invalidation = results['voxel_invalidation'] 
            try:
                mmcv.check_file_exist(voxel_path)
                voxel = _read_occupancy_SemKITTI(voxel_path)
                # destination_path = "/home/eason/workspace_perception/UniLiDAR/temp"
                # shutil.copy(voxel_path, destination_path)
                # print(f"Ori file copied from {voxel_path} to {destination_path}")
            except ConnectionError:
                mask_bytes = self.file_client.get(voxel_path)
                # add .copy() to fix read-only bug
                voxel = np.frombuffer(
                    mask_bytes, dtype='int').copy()
            try:
                mmcv.check_file_exist(voxel_semantic_mask_path)
                voxel_semantic_mask = _read_voxellabel_SemKITTI(voxel_semantic_mask_path)
            except ConnectionError:
                mask_bytes = self.file_client.get(voxel_semantic_mask_path)
                # add .copy() to fix read-only bug
                voxel_semantic_mask = np.frombuffer(
                    mask_bytes, dtype='int').copy()
            try:
                mmcv.check_file_exist(voxel_occ_mask_path)
                voxel_occ_mask = _read_occluded_SemKITTI(voxel_occ_mask_path)
            except ConnectionError:
                mask_bytes = self.file_client.get(voxel_occ_mask_path)
                # add .copy() to fix read-only bug
                voxel_occ_mask = np.frombuffer(
                    mask_bytes, dtype='int').copy()
            try:
                mmcv.check_file_exist(voxel_invalidation)
                voxel_invalid = _read_invalid_SemKITTI(voxel_invalidation)
            except ConnectionError:
                mask_bytes = self.file_client.get(voxel_invalidation)
                # add .copy() to fix read-only bug
                voxel_invalid = np.frombuffer(
                    mask_bytes, dtype='int').copy()
                
            voxel_semantic_mask[
                np.isclose(voxel_invalid, 1)
            ] = 260  # Setting to unknown all voxels marked on invalid mask...
            
        points = results['points']

        if not results.get('dense7sparse'):
            if len(points.shape)==2:
                if points.shape[0] != pts_semantic_mask.shape[0]:
                    RPR_indices = results['RPR_indices']
                    pts_semantic_mask = pts_semantic_mask[RPR_indices]
                    results.pop('RPR_indices')
            elif len(points.shape)==3:
                if points.shape[1] != pts_semantic_mask.shape[0]:
                    RPR_indices = results['RPR_indices']
                    pts_semantic_mask = pts_semantic_mask[RPR_indices]
                    results.pop('RPR_indices')
        else:
            results.pop('RPR_indices_sparse', None)
            results.pop('RPR_indices_dense', None)

                #这里增加一个功能：当判断是Waymo的数据时，将pts_semantic_mask中标签为0的点全部滤除，points根据idx执行对应操作。
        if 'waymo' in pts_semantic_mask_path:
            waymo_idx = (pts_semantic_mask == 0).squeeze()
            points = points[~waymo_idx]
            results['points'] = points
            pts_semantic_mask = pts_semantic_mask[~waymo_idx]
        
        # results['pts_semantic_mask'] = torch.from_numpy(pts_semantic_mask.astype(np.float32))
        results['pts_semantic_mask'] = pts_semantic_mask
        
        if results.get('voxel_path', None) is not None:
            results['voxel'] = torch.from_numpy(voxel)
            # results['voxel_semantic_mask'] = torch.from_numpy(voxel_semantic_mask.astype(np.float32))
            results['voxel_semantic_mask'] = voxel_semantic_mask.reshape(self.grid_size)
            results['voxel_occ_mask'] = torch.from_numpy(voxel_occ_mask)
            results['voxel_invalid'] = torch.from_numpy(voxel_invalid)
            
        # 计算限定范围在processed_label中的索引范围
        pc_range_min = np.array(self.pc_range[:3])
        pc_range_max = np.array(self.pc_range[3:])
        restrict_pc_range_min = np.array(self.restrict_pc_range[:3])
        restrict_pc_range_max = np.array(self.restrict_pc_range[3:])

        if results.get('voxel_path', None) is not None:
            shape = results['voxel_semantic_mask'].shape
            
            start_index = []
            end_index = []

            for i in range(3):  # 对X, Y, Z分别判断
                if restrict_pc_range_min[i] >= pc_range_min[i] and restrict_pc_range_max[i] <= pc_range_max[i]:
                    start = int((restrict_pc_range_min[i] - pc_range_min[i]) / 0.2)
                    end = int((restrict_pc_range_max[i] - pc_range_min[i]) / 0.2)
                else:
                    start = 0
                    end = shape[i]
                start_index.append(start)
                end_index.append(end)

            start_index = np.array(start_index)
            end_index = np.array(end_index)
            if isinstance(self.RPR, tuple):
                self.RPR = self.RPR[0]
            if self.RPR == True:
                # 添加边界检查：确保end_index不超过voxel_semantic_mask的最大索引上限
                # 获取voxel_semantic_mask的shape
                voxel_shape = results['voxel_semantic_mask'].shape
                # 限制end_index不超过各维度的最大值
                end_index = np.minimum(end_index, voxel_shape)
                
                # 提取对应的体素
                restricted_voxels = results['voxel_semantic_mask'][start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2]]
                results['voxel_semantic_mask'] = restricted_voxels
                
        results['pts_seg_fields'].append('pts_semantic_mask')
        if results.get('voxel_path', None) is not None:
            results['seg_fields'].append('voxel')
            results['seg_fields'].append('voxel_semantic_mask')
            results['seg_fields'].append('voxel_occ_mask')
            results['seg_fields'].append('voxel_invalid')
        
        if self.cal_visible:
            visible_mask = np.zeros(self.grid_size, dtype=np.uint8)
            # # camera branch
            # if 'img_inputs' in results.keys():
            #     _, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:6]
            #     occ_uvds = self.project_points(torch.Tensor(untransformed_occ), 
            #                                     rots, trans, intrins, post_rots, post_trans)  # N 6 3
            #     N, n_cam, _ = occ_uvds.shape
            #     img_visible_mask = np.zeros((N, n_cam))
            #     img_h, img_w = results['img_inputs'][0].shape[-2:]
            #     for cam_idx in range(n_cam):
            #         basic_mask = (occ_uvds[:, cam_idx, 0] >= 0) & (occ_uvds[:, cam_idx, 0] < img_w) & \
            #                     (occ_uvds[:, cam_idx, 1] >= 0) & (occ_uvds[:, cam_idx, 1] < img_h) & \
            #                     (occ_uvds[:, cam_idx, 2] >= 0)

            #         basic_valid_occ = occ_uvds[basic_mask, cam_idx]  # M 3
            #         M = basic_valid_occ.shape[0]  # TODO M~=?
            #         basic_valid_occ[:, 2] = basic_valid_occ[:, 2] * 10
            #         basic_valid_occ = basic_valid_occ.cpu().numpy()
            #         basic_valid_occ = basic_valid_occ.astype(np.int16)  # TODO first round then int?
            #         depth_canva = np.ones((img_h, img_w), dtype=np.uint16) * 2048
            #         nb_valid_mask = np.zeros((M), dtype=np.bool)
            #         nb_valid_mask = nb_process_img_points(basic_valid_occ, depth_canva, nb_valid_mask)  # M
            #         img_visible_mask[basic_mask, cam_idx] = nb_valid_mask

            #     img_visible_mask = img_visible_mask.sum(1) > 0  # N  1:occupied  0: free
            #     img_visible_mask = img_visible_mask.reshape(-1, 1).astype(pcd_label.dtype) 

            #     img_pcd_np = np.concatenate([transformed_occ, img_visible_mask], axis=-1)
            #     img_pcd_np = img_pcd_np[np.lexsort((transformed_occ[:, 0], transformed_occ[:, 1], transformed_occ[:, 2])), :]
            #     img_pcd_np = img_pcd_np.astype(np.int64)
            #     img_occ_label = np.zeros(self.grid_size, dtype=np.uint8)
            #     voxel_img = nb_process_label(img_occ_label, img_pcd_np) 
            #     visible_mask = visible_mask | voxel_img
            #     results['img_visible_mask'] = voxel_img


            # lidar branch
            if 'points' in results.keys():
                if isinstance(points, LiDARPoints):
                    pts = results['points'].tensor.cpu().numpy()[:, :3]
                else:
                    pts = results['points'].cpu().numpy()[:, :3]
                pts_in_range = ((pts>=self.pc_range[:3]) & (pts<self.pc_range[3:])).sum(1)==3
                pts = pts[pts_in_range]
                pts = (pts - self.pc_range[:3])/self.voxel_size
                pts = np.concatenate([pts, np.ones((pts.shape[0], 1)).astype(pts.dtype)], axis=1) 
                pts = pts[np.lexsort((pts[:, 0], pts[:, 1], pts[:, 2])), :].astype(np.int64)
                pts_occ_label = np.zeros(self.grid_size, dtype=np.uint8)
                voxel_pts = nb_process_label(pts_occ_label, pts)  # W H D 1:occupied 0:free
                visible_mask = visible_mask | voxel_pts
                results['lidar_visible_mask'] = voxel_pts

            visible_mask = torch.from_numpy(visible_mask)
            results['visible_mask'] = visible_mask

        return results

    def voxel2world(self, voxel):
        """
        voxel: [N, 3]
        """
        return voxel * self.voxel_size[None, :] + self.pc_range[:3][None, :]


    def world2voxel(self, wolrd):
        """
        wolrd: [N, 3]
        """
        return (wolrd - self.pc_range[:3][None, :]) / self.voxel_size[None, :]


    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}'
        return repr_str

    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        
        # from lidar to camera
        points = points.reshape(-1, 1, 3)
        points = points - trans.reshape(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # from camera to raw pixel
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd
    
# b1:boolean, u1: uint8, i2: int16, u2: uint16
@nb.jit('b1[:](i2[:,:],u2[:,:],b1[:])', nopython=True, cache=True, parallel=False)
def nb_process_img_points(basic_valid_occ, depth_canva, nb_valid_mask):
    # basic_valid_occ M 3
    # depth_canva H W
    # label_size = M   # for original occ, small: 2w mid: ~8w base: ~30w
    canva_idx = -1 * np.ones_like(depth_canva, dtype=np.int16)
    for i in range(basic_valid_occ.shape[0]):
        occ = basic_valid_occ[i]
        if occ[2] < depth_canva[occ[1], occ[0]]:
            if canva_idx[occ[1], occ[0]] != -1:
                nb_valid_mask[canva_idx[occ[1], occ[0]]] = False

            canva_idx[occ[1], occ[0]] = i
            depth_canva[occ[1], occ[0]] = occ[2]
            nb_valid_mask[i] = True
    return nb_valid_mask

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label_withvel(processed_label, sorted_label_voxel_pair):
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


# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
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


def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed

def pack(array):
    """ convert a boolean array into a bitwise array. """
    array = array.reshape((-1))
    #compressing bit flags.
    # yapf: disable
    compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
    # yapf: enable

    return np.array(compressed, dtype=np.uint8)

def _read_SemKITTI(path, dtype, do_unpack):
    bin = np.fromfile(path, dtype=dtype)  # Flattened array
    if do_unpack:
        bin = unpack(bin)
    return bin


def _read_ptslabel_SemKITTI(path):
    label = _read_SemKITTI(path, dtype=np.uint32, do_unpack=False)
    return label

def _read_voxellabel_SemKITTI(path):
    label = _read_SemKITTI(path, dtype=np.uint16, do_unpack=False)
    return label


def _read_invalid_SemKITTI(path):
    invalid = _read_SemKITTI(path, dtype=np.uint8, do_unpack=True)
    return invalid


def _read_occluded_SemKITTI(path):
    occluded = _read_SemKITTI(path, dtype=np.uint8, do_unpack=True)
    return occluded


def _read_occupancy_SemKITTI(path):
    occupancy = _read_SemKITTI(path, dtype=np.uint8, do_unpack=True).astype(np.float32)
    return occupancy


def _read_pointcloud_SemKITTI(path):
    'Return pointcloud semantic kitti with remissions (x, y, z, intensity)'
    pointcloud = _read_SemKITTI(path, dtype=np.float32, do_unpack=False)
    pointcloud = pointcloud.reshape((-1, 4))
    return pointcloud


def _read_calib_SemKITTI(calib_path):
    """
    :param calib_path: Path to a calibration text file.
    :return: dict with calibration matrices.
    """
    calib_all = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if line == '\n':
                break
        key, value = line.split(':', 1)
        calib_all[key] = np.array([float(x) for x in value.split()])

    # reshape matrices
    calib_out = {}
    calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
    calib_out['Tr'] = np.identity(4)  # 4x4 matrix
    calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)
    return calib_out