#import open3d as o3d
import mmcv
import numpy as np
import numba as nb

from mmdet.datasets.builder import PIPELINES
import os
import open3d as o3d
import torch
import copy
from mmdet3d.core.points import BasePoints, get_points_type
import open3d as o3d

import random

@PIPELINES.register_module()
class LoadPointsFromMultiSweeps_RPR(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 RPR = False,
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.RPR = RPR
        self.point_cloud_range = np.array(point_cloud_range)
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

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
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        
        indices = np.where((points.tensor[:, 0] >= self.point_cloud_range[0]) & \
                            (points.tensor[:, 0] < self.point_cloud_range[3]) & \
                            (points.tensor[:, 1] >= self.point_cloud_range[1]) & \
                            (points.tensor[:, 1] < self.point_cloud_range[4]) & \
                            (points.tensor[:, 2] >= self.point_cloud_range[2]) & \
                            (points.tensor[:, 2] < self.point_cloud_range[5]))[0]
        
        # 使用索引提取符合条件的点
        if isinstance(self.RPR, tuple):
            self.RPR = self.RPR[0]
        if self.RPR == True:
            points.tensor = points.tensor[indices]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'

@PIPELINES.register_module()
class LoadPointsFromDenseMultiSweeps_Sampling(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False,
                 LiDAR_height=[1, 2],
                 num_of_beams=[16, 128],
                 horizontal_angular_resolution=[900, 3600],
                 lower_vertical_field_of_view_bound=[-40, -5],
                 upper_vertical_field_of_view_bound=[0, 25],):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.lidar_height = round(random.uniform(LiDAR_height[0], LiDAR_height[1]), 1)
        self.beams = random.randint(num_of_beams[0], num_of_beams[1])
        self.horizontal_resolution = random.choice(range(horizontal_angular_resolution[0], horizontal_angular_resolution[1]+1, 100))
        self.vertical_lower_angle = round(random.uniform(lower_vertical_field_of_view_bound[0], lower_vertical_field_of_view_bound[1]), 1)
        self.vertical_upper_angle = round(random.uniform(upper_vertical_field_of_view_bound[0], upper_vertical_field_of_view_bound[1]), 1)
        
    def _sampling(self, points):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # point_cloud_tensor = torch.from_numpy(points).to(device)
        point_cloud_tensor = points
        if point_cloud_tensor.ndim == 1:
            point_cloud_tensor = point_cloud_tensor.reshape(-1, self.load_dim)
        
        vertical_angles_rad = torch.deg2rad(
            torch.linspace(self.vertical_lower_angle, self.vertical_upper_angle, self.beams, device=device)
        )

        lidar_origin = torch.tensor([0, 0, self.lidar_height], device=device, dtype=torch.float32)

        # Compute vector from lidar_origin to each point
        vector_to_points = point_cloud_tensor - lidar_origin

        # Compute ranges
        ranges = torch.norm(vector_to_points, dim=1)

        # Compute azimuths and elevations
        azimuths = torch.atan2(vector_to_points[:, 1], vector_to_points[:, 0])
        elevations = torch.asin(vector_to_points[:, 2] / ranges)

        # Create bins for azimuth and elevation
        horizontal_bin_edges = torch.linspace(
            -torch.pi, torch.pi, self.horizontal_resolution + 1, device=device, dtype=torch.float32
        )
        
        vertical_diffs = torch.diff(vertical_angles_rad)
        vertical_bin_edges_mid = (vertical_angles_rad[:-1] + vertical_angles_rad[1:]) / 2
        first_bin_edge = vertical_angles_rad[0] - vertical_diffs[0] / 2
        last_bin_edge = vertical_angles_rad[-1] + vertical_diffs[-1] / 2
        vertical_bin_edges = torch.cat((
            torch.tensor([first_bin_edge], device=device),
            vertical_bin_edges_mid,
            torch.tensor([last_bin_edge], device=device)
        ))

        # Compute bin indices
        azimuth_bin_indices = torch.bucketize(azimuths, horizontal_bin_edges) - 1
        elevation_bin_indices = torch.bucketize(elevations, vertical_bin_edges) - 1

        # Clip bin indices to valid range
        azimuth_bin_indices = torch.clamp(
            azimuth_bin_indices, 0, self.horizontal_resolution - 1
        )
        elevation_bin_indices = torch.clamp(
            elevation_bin_indices, 0, len(vertical_angles_rad) - 1
        )
        
        # Compute bin centers
        azimuth_bin_centers = (horizontal_bin_edges[:-1] + horizontal_bin_edges[1:]) / 2
        elevation_bin_centers = (vertical_bin_edges[:-1] + vertical_bin_edges[1:]) / 2

        # For each point, get the bin center
        azimuth_bin_center = azimuth_bin_centers[azimuth_bin_indices]
        elevation_bin_center = elevation_bin_centers[elevation_bin_indices]

        # Compute angular distance to bin center
        delta_azimuth = azimuths - azimuth_bin_center
        delta_elevation = elevations - elevation_bin_center

        # Handle wrapping of azimuth angle
        delta_azimuth = torch.remainder(delta_azimuth + torch.pi, 2 * torch.pi) - torch.pi

        angular_distance = torch.sqrt(delta_azimuth**2 + delta_elevation**2)

        # Compute a 2D grid of bins
        bin_indices = elevation_bin_indices * self.horizontal_resolution + azimuth_bin_indices

        # Lexsort by bin_indices and ranges
        keys = torch.stack([angular_distance, bin_indices])
        sorted_indices = torch.argsort(keys, dim=1)[0]

        sorted_bin_indices = bin_indices[sorted_indices]
        sorted_point_indices = sorted_indices

        
        # Find unique bin_indices and their first occurrence indices
        unique_bin_indices, inverse_indices = torch.unique(
            sorted_bin_indices, return_inverse=True
        )

        # Create a mask to identify the first occurrence of each unique element
        first_occurrence_mask = torch.zeros_like(inverse_indices, dtype=torch.bool)
        first_occurrence_mask[inverse_indices] = ~first_occurrence_mask[inverse_indices]

        # Get the indices of the points with minimum range in each bin
        min_point_indices = sorted_point_indices[first_occurrence_mask]

        # Get the sampled points
        # sampled_points = point_cloud_tensor[min_point_indices]

        return min_point_indices

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
                
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        if points.shape[1] == 5:
            points.tensor[:, 4] = 0
            # min_point_indices = self._sampling(points.tensor[:, 0:3])
            # points = points[min_point_indices, :]
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points_sweep[:, 0:3]
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        # points = points[:, self.use_dim]
        
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'
    
    
@PIPELINES.register_module()
class LoadOccGTFromFileWaymo(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
            crop_x=False,
            use_infov_mask=True, 
            use_lidar_mask=False, 
            use_camera_mask=True,
            FREE_LABEL=23, 
            num_classes=None,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], RPR = False, restrict_pc_range = [-25.6, -25.6, -3.4, 25.6, 25.6, 3.0],
        ):
        self.data_root = data_root # this is occ_gt_data_root in config file
        self.crop_x = crop_x
        self.use_infov_mask = use_infov_mask
        self.use_lidar_mask = use_lidar_mask
        self.use_camera_mask = use_camera_mask
        self.FREE_LABEL = FREE_LABEL
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.RPR = RPR
        self.restrict_pc_range = restrict_pc_range

    def __call__(self, results):
        # Step 1: get the occupancy ground truth file path
        pts_filename = results['pts_filename']
        basename = os.path.basename(pts_filename)
        seq_name = basename[1:4]
        frame_name = basename[4:7]

        file_path = os.path.join(self.data_root, seq_name, '{}.npz'.format(frame_name))
        
        # Step 2: load the file
        occ_labels = np.load(file_path)
        semantics = occ_labels['voxel_label']
        mask_infov = occ_labels['infov'].astype(bool)
        mask_lidar = occ_labels['origin_voxel_state'].astype(bool)
        mask_camera = occ_labels['final_voxel_state'].astype(bool)

        # Step 3: crop the x axis
        if self.crop_x: # default is False
            w, h, d = semantics.shape
            semantics = semantics[w//2:, :, :]
            mask_infov = mask_infov[w//2:, :, :]
            mask_lidar = mask_lidar[w//2:, :, :]
            mask_camera = mask_camera[w//2:, :, :]

        # Step 4: unify the mask
        mask = np.ones_like(semantics).astype(bool) # 200, 200, 16
        if self.use_infov_mask:
            mask = mask & mask_infov
        if self.use_lidar_mask:
            mask = mask & mask_lidar
        if self.use_camera_mask:
            mask = mask & mask_camera
        mask = mask.astype(bool)
        # results['visible_mask'] = torch.from_numpy(mask)

        # Step 5: change the FREE_LABEL to num_classes-1
        if self.FREE_LABEL is not None:
            semantics[semantics == self.FREE_LABEL] = self.num_classes - 1
            # 变换到 (D, H, W) -> (1, D, H, W) -> (1, 1, D, H, W)
        semantics = torch.from_numpy(semantics).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
        mask = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
        downsampled = torch.nn.functional.max_pool3d(semantics.squeeze(1), kernel_size=2, stride=2)  # [1, D, H, W, C]
        downsampled_mask = torch.nn.functional.max_pool3d(mask.squeeze(1), kernel_size=2, stride=2)  # [1, D, H, W, C]
        # 变回 (H', W', D') 形状
        semantics = downsampled.squeeze(0).squeeze(0).permute(1, 2, 0).long().numpy()  # [H//2, W//2, D//2]
        mask = downsampled_mask.squeeze(0).squeeze(0).permute(1, 2, 0).long().numpy()  # [H//2, W//2, D//2]
        results['voxel_semantic_mask'] = semantics
        results['visible_mask'] = mask

        # 计算限定范围在processed_label中的索引范围
        pc_range_min = np.array(self.pc_range[:3])
        pc_range_max = np.array(self.pc_range[3:])
        restrict_pc_range_min = np.array(self.restrict_pc_range[:3])
        restrict_pc_range_max = np.array(self.restrict_pc_range[3:])

        shape = semantics.shape
        
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
            # 提取对应的体素
            restricted_voxels = semantics[start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2]]
            results['voxel_semantic_mask'] = torch.from_numpy(restricted_voxels).type(torch.LongTensor)
            resticted_mask = mask[start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2]]
            results['visible_mask'] = torch.from_numpy(resticted_mask)
            # self.grid_size_occ = grid_size_occ_RPR.numpy().astype(np.int32)
        else:
            results['voxel_semantic_mask'] = torch.from_numpy(semantics).type(torch.LongTensor)
            results['visible_mask'] = torch.from_numpy(mask)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)


@PIPELINES.register_module()
class LoadOccupancy(object):

    def __init__(self, to_float32=True, use_semantic=False, cylinder=False, occ_path=None, grid_size=[512, 512, 40], unoccupied=0,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], RPR = False, restrict_pc_range = [-25.6, -25.6, -3.4, 25.6, 25.6, 3.0], gt_resize_ratio=1, cal_visible=False, use_vel=False):
        self.to_float32 = to_float32
        self.use_semantic = use_semantic
        self.cylinder = cylinder
        self.occ_path = occ_path
        self.cal_visible = cal_visible
        self.grid_size = np.array(grid_size)
        self.unoccupied = unoccupied
        self.pc_range = np.array(pc_range)
        if isinstance(RPR, tuple):
            RPR = RPR[0]
        self.RPR = RPR,
        self.restrict_pc_range = np.array(restrict_pc_range)
        self.voxel_size = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        self.gt_resize_ratio = gt_resize_ratio
        self.use_vel = use_vel
    
    def __call__(self, results):
        
        rel_path = 'scene_{0}/occupancy/{1}.npy'.format(results['scene_token'], results['lidar_token'])
        #  [z y x cls] or [z y x vx vy vz cls]
        pcd = np.load(os.path.join(self.occ_path, rel_path))
        # results['pcd'] = pcd
        pcd_label = pcd[..., -1:]
        pcd_label[pcd_label==0] = 255
        pcd = pcd[..., [2,1,0]]
        pcd[:,:3] = cart2polar(pcd[:,:3])if self.cylinder else pcd[:,:3]
        pcd_np_cor = self.voxel2world(pcd[:,:3] + 0.5)  # x y z
        untransformed_occ = copy.deepcopy(pcd_np_cor)  # N 4
        # bevdet augmentation
        pcd_np_cor = (results['bda_mat'] @ torch.from_numpy(pcd_np_cor).unsqueeze(-1).float()).squeeze(-1).numpy()
        pcd_np_cor = self.world2voxel(pcd_np_cor)

        # make sure the point is in the grid
        pcd_np_cor = np.clip(pcd_np_cor, np.array([0,0,0]), self.grid_size - 1)
        transformed_occ = copy.deepcopy(pcd_np_cor)
        pcd_np = np.concatenate([pcd_np_cor, pcd_label], axis=-1)

        # velocity
        if self.use_vel:
            pcd_vel = pcd[..., [3,4,5]]  # x y z
            pcd_vel = (results['bda_mat'] @ torch.from_numpy(pcd_vel).unsqueeze(-1).float()).squeeze(-1).numpy()
            pcd_vel = np.concatenate([pcd_np, pcd_vel], axis=-1)  # [x y z cls vx vy vz]
            results['gt_vel'] = pcd_vel

        # 255: noise, 1-16 normal classes, 0 unoccupied
        pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
        pcd_np = pcd_np.astype(np.int64)
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied
        processed_label = nb_process_label(processed_label, pcd_np)
        # 计算限定范围在processed_label中的索引范围
        pc_range_min = np.array(self.pc_range[:3])
        pc_range_max = np.array(self.pc_range[3:])
        restrict_pc_range_min = np.array(self.restrict_pc_range[:3])
        restrict_pc_range_max = np.array(self.restrict_pc_range[3:])

        shape = processed_label.shape
        
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
            # 提取对应的体素
            restricted_voxels = processed_label[start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2]]
            results['gt_occ'] = restricted_voxels
            # restricted_voxels = processed_label[start_index[1]:end_index[1], start_index[0]:end_index[0], start_index[2]:end_index[2]]
            # results['gt_occ'] = restricted_voxels
        else:
            results['gt_occ'] = processed_label


        if self.cal_visible:
            visible_mask = np.zeros(self.grid_size, dtype=np.uint8)
            # camera branch
            if 'img_inputs' in results.keys():
                _, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:6]
                occ_uvds = self.project_points(torch.Tensor(untransformed_occ), 
                                                rots, trans, intrins, post_rots, post_trans)  # N 6 3
                N, n_cam, _ = occ_uvds.shape
                img_visible_mask = np.zeros((N, n_cam))
                img_h, img_w = results['img_inputs'][0].shape[-2:]
                for cam_idx in range(n_cam):
                    basic_mask = (occ_uvds[:, cam_idx, 0] >= 0) & (occ_uvds[:, cam_idx, 0] < img_w) & \
                                (occ_uvds[:, cam_idx, 1] >= 0) & (occ_uvds[:, cam_idx, 1] < img_h) & \
                                (occ_uvds[:, cam_idx, 2] >= 0)

                    basic_valid_occ = occ_uvds[basic_mask, cam_idx]  # M 3
                    M = basic_valid_occ.shape[0]  # TODO M~=?
                    basic_valid_occ[:, 2] = basic_valid_occ[:, 2] * 10
                    basic_valid_occ = basic_valid_occ.cpu().numpy()
                    basic_valid_occ = basic_valid_occ.astype(np.int16)  # TODO first round then int?
                    depth_canva = np.ones((img_h, img_w), dtype=np.uint16) * 2048
                    nb_valid_mask = np.zeros((M), dtype=np.bool)
                    nb_valid_mask = nb_process_img_points(basic_valid_occ, depth_canva, nb_valid_mask)  # M
                    img_visible_mask[basic_mask, cam_idx] = nb_valid_mask

                img_visible_mask = img_visible_mask.sum(1) > 0  # N  1:occupied  0: free
                img_visible_mask = img_visible_mask.reshape(-1, 1).astype(pcd_label.dtype) 

                img_pcd_np = np.concatenate([transformed_occ, img_visible_mask], axis=-1)
                img_pcd_np = img_pcd_np[np.lexsort((transformed_occ[:, 0], transformed_occ[:, 1], transformed_occ[:, 2])), :]
                img_pcd_np = img_pcd_np.astype(np.int64)
                img_occ_label = np.zeros(self.grid_size, dtype=np.uint8)
                voxel_img = nb_process_label(img_occ_label, img_pcd_np) 
                visible_mask = visible_mask | voxel_img
                results['img_visible_mask'] = voxel_img


            # lidar branch
            if 'points' in results.keys():
                pts = results['points'].tensor.cpu().numpy()[:, :3]
                pts_in_range = ((pts>=self.pc_range[:3]) & (pts<self.pc_range[3:])).sum(1)==3
                pts = pts[pts_in_range]
                pts = (pts - self.pc_range[:3])/self.voxel_size
                pts = np.concatenate([pts, np.ones((pts.shape[0], 1)).astype(pts.dtype)], axis=1) 
                pts = pts[np.lexsort((pts[:, 0], pts[:, 1], pts[:, 2])), :].astype(np.int64)
                pts_occ_label = np.zeros(self.grid_size, dtype=np.uint8)
                voxel_pts = nb_process_label(pts_occ_label, pts)  # W H D 1:occupied 0:free
                visible_mask = visible_mask | voxel_pts
                results['lidar_visible_mask'] = voxel_pts

            if isinstance(self.RPR, tuple):
                self.RPR = self.RPR[0]
            if self.RPR == True:
                restricted_visible_mask = visible_mask[start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2]]
                results['visible_mask'] = torch.from_numpy(restricted_visible_mask)
            else:
                results['visible_mask'] = torch.from_numpy(visible_mask)

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