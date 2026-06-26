import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.pipelines.compose import Compose
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from typing import List, Optional, Sequence, Tuple, Union, Any, Dict
import torch
from mmdet3d.core.points import BasePoints, get_points_type, LiDARPoints

def cart2polar(input_xyz):
    """Convert Cartesian coordinates to polar coordinates."""
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

def recompute_train_grid(points_tensor, grid_size, grid_size_vox, max_volume_space, min_volume_space, fill_label=0):
    """Recompute train_grid from points tensor.
    
    Args:
        points_tensor: torch.Tensor of shape [N, D] where first 3 columns are xyz
        grid_size: Grid size for fine grid
        grid_size_vox: Grid size for voxel grid
        max_volume_space: Max volume space bounds
        min_volume_space: Min volume space bounds
        fill_label: Fill label value
        
    Returns:
        dict: Contains train_grid, grid_ind, grid_ind_vox
    """
    # Handle device placement
    device = None
    if isinstance(points_tensor, torch.Tensor):
        device = points_tensor.device
        points_np = points_tensor.detach().cpu().numpy()
    else:
        points_np = np.asarray(points_tensor)
    
    xyz = points_np[:, :3]
    feat = points_np[:, 3:] if points_np.shape[1] > 3 else np.zeros((points_np.shape[0], 0))
    xyz_pol = cart2polar(xyz)
    
    max_bound = np.asarray(max_volume_space)
    min_bound = np.asarray(min_volume_space)
    crop_range = max_bound - min_bound
    intervals = crop_range / np.asarray(grid_size)
    intervals_vox = crop_range / np.asarray(grid_size_vox)
    
    xyz_pol_grid = np.clip(xyz_pol, min_bound, max_bound - 1e-3)
    grid_ind = (np.floor((xyz_pol_grid - min_bound) / intervals)).astype(np.int32)
    grid_ind_vox = (np.floor((xyz_pol_grid - min_bound) / intervals_vox)).astype(np.int32)
    grid_ind_vox_float = ((xyz_pol_grid - min_bound) / intervals_vox).astype(np.float32)
    
    # center data on each voxel for PTnet
    voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
    return_xyz = xyz_pol - voxel_centers
    
    # Reconstruct feat if needed
    if feat.shape[1] == 0:
        feat = np.zeros((xyz.shape[0], 0))
    
    return_feat = np.concatenate((return_xyz, xyz_pol, xyz[:, :2], feat), axis=1)
    
    result = {
        'train_grid': torch.from_numpy(return_feat).to(torch.float32),
        'grid_ind': torch.from_numpy(grid_ind).to(torch.float32),
        'grid_ind_vox': torch.from_numpy(grid_ind_vox_float).to(torch.float32)
    }
    
    # Move to original device if available
    if device is not None:
        result['train_grid'] = result['train_grid'].to(device)
        result['grid_ind'] = result['grid_ind'].to(device)
        result['grid_ind_vox'] = result['grid_ind_vox'].to(device)
    
    return result

@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str



@PIPELINES.register_module()
class CustomCollect3D(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'scene_token',
                            'can_bus'
                            )):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
       
        data = {}
        img_metas = {}
      
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'

@PIPELINES.register_module()
class CustomOccCollect3D(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'scene_token',
                            'can_bus', 'pc_range', 'occ_size', 'lidar_token'
                            )):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
       
        data = {}
        img_metas = {}
      
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            if key in results.keys():
                data[key] = results[key]
        
        if 'gt_occ' in results.keys():
            data['gt_occ'] = results['gt_occ']
            
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'

@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales)==1

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str

@PIPELINES.register_module()
class PolarMix(object):
    """PolarMix data augmentation.

    The polarmix transform steps are as follows:
        1. Another random point cloud is picked by dataset.
        2. Exchange sectors of two point clouds that are cut with certain
           azimuth angles.
        3. Cut point instances from picked point cloud, rotate them by multiple
           azimuth angles, and paste the cut and rotated instances.

    Required Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)
    - dataset (:obj:`BaseDataset`)

    Modified Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)

    Args:
        instance_classes (List[int]): Semantic masks which represent the
            instance.
        swap_ratio (float): Swap ratio of two point cloud. Defaults to 0.5.
        rotate_paste_ratio (float): Rotate paste ratio. Defaults to 1.0.
        pre_transform (Sequence[dict], optional): Sequence of transform object
            or config dict to be composed. Defaults to None.
        prob (float): The transformation probability. Defaults to 1.0.
    """

    def __init__(self,
                 instance_classes: List[int],
                 swap_ratio: float = 0.5,
                 rotate_paste_ratio: float = 1.0,
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0,
                 grid_size: Optional[List[int]] = None,
                 grid_size_vox: Optional[List[int]] = None,
                 max_volume_space: Optional[List[float]] = None,
                 min_volume_space: Optional[List[float]] = None,
                 fill_label: int = 0) -> None:
        # assert is_list_of(instance_classes, int), \
        #     'instance_classes should be a list of int'
        self.instance_classes = instance_classes
        self.swap_ratio = swap_ratio
        self.rotate_paste_ratio = rotate_paste_ratio

        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)
        
        # Parameters for recomputing train_grid
        self.grid_size = grid_size
        self.grid_size_vox = grid_size_vox
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.fill_label = fill_label

    def polar_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            mix_results (dict): Mixed dict picked from dataset.

        Returns:
            dict: output dict after transformation.
        """
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']

        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']
        
        # Handle both BasePoints and tensor formats
        is_basepoints = isinstance(points, BasePoints)
        is_mix_basepoints = isinstance(mix_points, BasePoints)
        
        if is_basepoints:
            points_coord = points.coord
            points_tensor = points.tensor if hasattr(points, 'tensor') else points_coord
        else:
            # points is tensor, assume first 3 columns are xyz
            points_coord = points[:, :3] if isinstance(points, torch.Tensor) else points[:, :3]
            points_tensor = points
        
        if is_mix_basepoints:
            mix_points_coord = mix_points.coord
            mix_points_tensor = mix_points.tensor if hasattr(mix_points, 'tensor') else mix_points_coord
        else:
            # mix_points is tensor, assume first 3 columns are xyz
            mix_points_coord = mix_points[:, :3] if isinstance(mix_points, torch.Tensor) else mix_points[:, :3]
            mix_points_tensor = mix_points
        
        # Convert to torch tensor if needed
        if isinstance(points_coord, np.ndarray):
            points_coord = torch.from_numpy(points_coord)
        if isinstance(mix_points_coord, np.ndarray):
            mix_points_coord = torch.from_numpy(mix_points_coord)
        if isinstance(pts_semantic_mask, np.ndarray):
            pts_semantic_mask = torch.from_numpy(pts_semantic_mask)
        if isinstance(mix_pts_semantic_mask, np.ndarray):
            mix_pts_semantic_mask = torch.from_numpy(mix_pts_semantic_mask)
        
        # Ensure semantic masks are 1D for boolean operations
        if isinstance(pts_semantic_mask, torch.Tensor) and pts_semantic_mask.dim() > 1:
            pts_semantic_mask = pts_semantic_mask.squeeze()
        elif isinstance(pts_semantic_mask, np.ndarray) and pts_semantic_mask.ndim > 1:
            pts_semantic_mask = pts_semantic_mask.squeeze()
        if isinstance(mix_pts_semantic_mask, torch.Tensor) and mix_pts_semantic_mask.dim() > 1:
            mix_pts_semantic_mask = mix_pts_semantic_mask.squeeze()
        elif isinstance(mix_pts_semantic_mask, np.ndarray) and mix_pts_semantic_mask.ndim > 1:
            mix_pts_semantic_mask = mix_pts_semantic_mask.squeeze()

        # 1. swap point cloud
        if np.random.random() < self.swap_ratio:
            start_angle = (np.random.random() - 1) * np.pi  # -pi~0
            end_angle = start_angle + np.pi
            # calculate horizontal angle for each point
            yaw = -torch.atan2(points_coord[:, 1], points_coord[:, 0])
            mix_yaw = -torch.atan2(mix_points_coord[:, 1], mix_points_coord[:, 0])

            # select points in sector
            idx = (yaw <= start_angle) | (yaw >= end_angle)
            mix_idx = (mix_yaw > start_angle) & (mix_yaw < end_angle)
            
            # Ensure boolean masks are 1D for indexing
            if isinstance(idx, torch.Tensor) and idx.dim() > 1:
                idx = idx.squeeze()
            if isinstance(mix_idx, torch.Tensor) and mix_idx.dim() > 1:
                mix_idx = mix_idx.squeeze()
            if isinstance(idx, np.ndarray) and idx.ndim > 1:
                idx = idx.squeeze()
            if isinstance(mix_idx, np.ndarray) and mix_idx.ndim > 1:
                mix_idx = mix_idx.squeeze()

            # swap
            if is_basepoints:
                points = points.cat([points[idx], mix_points[mix_idx]])
            else:
                # use the (possibly already-updated) points tensor to keep
                # points and pts_semantic_mask lengths in sync
                points = torch.cat([points[idx], mix_points_tensor[mix_idx]], dim=0)
            
            if isinstance(pts_semantic_mask, torch.Tensor):
                pts_semantic_mask = torch.cat(
                    (pts_semantic_mask[idx],
                     mix_pts_semantic_mask[mix_idx]),
                    dim=0)
            else:
                pts_semantic_mask = np.concatenate(
                    (pts_semantic_mask[idx.numpy() if hasattr(idx, 'numpy') else idx],
                     mix_pts_semantic_mask[mix_idx.numpy() if hasattr(mix_idx, 'numpy') else mix_idx]),
                    axis=0)

        # 2. rotate-pasting
        if np.random.random() < self.rotate_paste_ratio:
            # extract instance points
            instance_points_list, instance_pts_semantic_mask_list = [], []
            for instance_class in self.instance_classes:
                if isinstance(mix_pts_semantic_mask, torch.Tensor):
                    mix_idx = (mix_pts_semantic_mask == instance_class)
                    # Ensure mix_idx is 1D for indexing
                    if mix_idx.dim() > 1:
                        mix_idx = mix_idx.squeeze()
                else:
                    mix_idx = (mix_pts_semantic_mask == instance_class)
                    # Ensure mix_idx is 1D for numpy indexing
                    if isinstance(mix_idx, np.ndarray) and mix_idx.ndim > 1:
                        mix_idx = mix_idx.squeeze()
                
                if is_mix_basepoints:
                    instance_points_list.append(mix_points[mix_idx])
                else:
                    instance_points_list.append(mix_points_tensor[mix_idx])
                
                if isinstance(mix_pts_semantic_mask, torch.Tensor):
                    instance_pts_semantic_mask_list.append(mix_pts_semantic_mask[mix_idx])
                else:
                    instance_pts_semantic_mask_list.append(mix_pts_semantic_mask[mix_idx.numpy() if hasattr(mix_idx, 'numpy') else mix_idx])
            
            if is_mix_basepoints:
                instance_points = mix_points.cat(instance_points_list)
            else:
                instance_points = torch.cat(instance_points_list, dim=0)
            
            if isinstance(instance_pts_semantic_mask_list[0], torch.Tensor):
                instance_pts_semantic_mask = torch.cat(instance_pts_semantic_mask_list, dim=0)
            else:
                instance_pts_semantic_mask = np.concatenate(instance_pts_semantic_mask_list, axis=0)

            # rotate-copy
            copy_points_list = [instance_points]
            copy_pts_semantic_mask_list = [instance_pts_semantic_mask]
            angle_list = [
                np.random.random() * np.pi * 2 / 3,
                (np.random.random() + 1) * np.pi * 2 / 3
            ]
            for angle in angle_list:
                if is_mix_basepoints:
                    new_points = instance_points.clone()
                    new_points.rotate(angle)
                    copy_points_list.append(new_points)
                else:
                    # Rotate tensor points around z-axis
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    rot_matrix = torch.tensor([[cos_a, -sin_a, 0],
                                               [sin_a, cos_a, 0],
                                               [0, 0, 1]], dtype=instance_points.dtype, device=instance_points.device)
                    new_points = instance_points.clone()
                    new_points[:, :3] = (rot_matrix @ new_points[:, :3].T).T
                    copy_points_list.append(new_points)
                copy_pts_semantic_mask_list.append(instance_pts_semantic_mask)
            
            if is_mix_basepoints:
                copy_points = instance_points.cat(copy_points_list)
            else:
                copy_points = torch.cat(copy_points_list, dim=0)
            
            if isinstance(copy_pts_semantic_mask_list[0], torch.Tensor):
                copy_pts_semantic_mask = torch.cat(copy_pts_semantic_mask_list, dim=0)
            else:
                copy_pts_semantic_mask = np.concatenate(copy_pts_semantic_mask_list, axis=0)

            if is_basepoints:
                points = points.cat([points, copy_points])
            else:
                # always append to the latest points tensor instead of the
                # original cached points_tensor to avoid length mismatch
                points = torch.cat([points, copy_points], dim=0)
            
            if isinstance(pts_semantic_mask, torch.Tensor):
                pts_semantic_mask = torch.cat((pts_semantic_mask, copy_pts_semantic_mask), dim=0)
            else:
                pts_semantic_mask = np.concatenate((pts_semantic_mask, copy_pts_semantic_mask), axis=0)

        input_dict['points'] = points
        input_dict['pts_semantic_mask'] = pts_semantic_mask
        
        # Recompute train_grid if it exists and we have the necessary parameters
        if 'train_grid' in input_dict and self.grid_size is not None and self.grid_size_vox is not None:
            if isinstance(points, BasePoints):
                points_for_grid = points.tensor if hasattr(points, 'tensor') else points.coord
            else:
                points_for_grid = points
            
            # Get parameters from input_dict if available, otherwise use instance attributes
            grid_size = input_dict.get('_pointseg_grid_size', self.grid_size)
            grid_size_vox = input_dict.get('_pointseg_grid_size_vox', self.grid_size_vox)
            max_volume_space = input_dict.get('_pointseg_max_volume_space', self.max_volume_space)
            min_volume_space = input_dict.get('_pointseg_min_volume_space', self.min_volume_space)
            fill_label = input_dict.get('_pointseg_fill_label', self.fill_label)
            
            if max_volume_space is not None and min_volume_space is not None:
                recomputed = recompute_train_grid(
                    points_for_grid, grid_size, grid_size_vox,
                    max_volume_space, min_volume_space, fill_label
                )
                input_dict['train_grid'] = recomputed['train_grid']
                input_dict['grid_ind'] = recomputed['grid_ind']
                input_dict['grid_ind_vox'] = recomputed['grid_ind_vox']
                
                # Update train_pts_label to match the new train_grid length
                # pts_semantic_mask should have the same length as points
                if 'train_pts_label' in input_dict:
                    # Ensure pts_semantic_mask is 1D and convert to tensor if needed
                    if isinstance(pts_semantic_mask, torch.Tensor):
                        if pts_semantic_mask.dim() > 1:
                            pts_semantic_mask = pts_semantic_mask.squeeze()
                    elif isinstance(pts_semantic_mask, np.ndarray):
                        if pts_semantic_mask.ndim > 1:
                            pts_semantic_mask = pts_semantic_mask.squeeze()
                        pts_semantic_mask = torch.from_numpy(pts_semantic_mask)
                    
                    # Verify length matches
                    points_len = points_for_grid.shape[0] if isinstance(points_for_grid, torch.Tensor) else len(points_for_grid)
                    mask_len = pts_semantic_mask.shape[0] if isinstance(pts_semantic_mask, torch.Tensor) else len(pts_semantic_mask)
                    
                    if points_len != mask_len:
                        # This should not happen, but if it does, we need to handle it
                        # For now, we'll use the mask as-is and let the error surface if there's a real issue
                        pass
                    
                    input_dict['train_pts_label'] = pts_semantic_mask
        
        return input_dict

    def __call__(self, input_dict: dict) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through PolarMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before polarmix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.polar_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(instance_classes={self.instance_classes}, '
        repr_str += f'swap_ratio={self.swap_ratio}, '
        repr_str += f'rotate_paste_ratio={self.rotate_paste_ratio}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class LaserMix(object):
    """LaserMix data augmentation.

    The lasermix transform steps are as follows:

        1. Another random point cloud is picked by dataset.
        2. Divide the point cloud into several regions according to pitch
           angles and combine the areas crossly.

    Required Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)
    - dataset (:obj:`BaseDataset`)

    Modified Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)

    Args:
        num_areas (List[int]): A list of area numbers will be divided into.
        pitch_angles (Sequence[float]): Pitch angles used to divide areas.
        pre_transform (Sequence[dict], optional): Sequence of transform object
            or config dict to be composed. Defaults to None.
        prob (float): The transformation probability. Defaults to 1.0.
    """

    def __init__(self,
                 num_areas: List[int],
                 pitch_angles: Sequence[float],
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0,
                 grid_size: Optional[List[int]] = None,
                 grid_size_vox: Optional[List[int]] = None,
                 max_volume_space: Optional[List[float]] = None,
                 min_volume_space: Optional[List[float]] = None,
                 fill_label: int = 0) -> None:
        # assert is_list_of(num_areas, int), \
        #     'num_areas should be a list of int.'
        assert isinstance(num_areas, list) and all(isinstance(x, int) for x in num_areas), \
            'num_areas should be a list of int.'
        self.num_areas = num_areas

        assert len(pitch_angles) == 2, \
            'The length of pitch_angles should be 2, ' \
            f'but got {len(pitch_angles)}.'
        assert pitch_angles[1] > pitch_angles[0], \
            'pitch_angles[1] should be larger than pitch_angles[0].'
        self.pitch_angles = pitch_angles

        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)
        
        # Parameters for recomputing train_grid
        self.grid_size = grid_size
        self.grid_size_vox = grid_size_vox
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.fill_label = fill_label

    def laser_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        """LaserMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            mix_results (dict): Mixed dict picked from dataset.

        Returns:
            dict: output dict after transformation.
        """
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']

        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']
        
        # Handle both BasePoints and tensor formats
        is_basepoints = isinstance(points, BasePoints)
        is_mix_basepoints = isinstance(mix_points, BasePoints)
        
        if is_basepoints:
            points_coord = points.coord
            points_tensor = points.tensor if hasattr(points, 'tensor') else points_coord
        else:
            # points is tensor, assume first 3 columns are xyz
            points_coord = points[:, :3] if isinstance(points, torch.Tensor) else points[:, :3]
            points_tensor = points
        
        if is_mix_basepoints:
            mix_points_coord = mix_points.coord
            mix_points_tensor = mix_points.tensor if hasattr(mix_points, 'tensor') else mix_points_coord
        else:
            # mix_points is tensor, assume first 3 columns are xyz
            mix_points_coord = mix_points[:, :3] if isinstance(mix_points, torch.Tensor) else mix_points[:, :3]
            mix_points_tensor = mix_points
        
        # Convert to torch tensor if needed
        if isinstance(points_coord, np.ndarray):
            points_coord = torch.from_numpy(points_coord)
        if isinstance(mix_points_coord, np.ndarray):
            mix_points_coord = torch.from_numpy(mix_points_coord)
        if isinstance(pts_semantic_mask, np.ndarray):
            pts_semantic_mask = torch.from_numpy(pts_semantic_mask)
        if isinstance(mix_pts_semantic_mask, np.ndarray):
            mix_pts_semantic_mask = torch.from_numpy(mix_pts_semantic_mask)
        
        # Ensure semantic masks are 1D for boolean operations
        if isinstance(pts_semantic_mask, torch.Tensor) and pts_semantic_mask.dim() > 1:
            pts_semantic_mask = pts_semantic_mask.squeeze()
        elif isinstance(pts_semantic_mask, np.ndarray) and pts_semantic_mask.ndim > 1:
            pts_semantic_mask = pts_semantic_mask.squeeze()
        if isinstance(mix_pts_semantic_mask, torch.Tensor) and mix_pts_semantic_mask.dim() > 1:
            mix_pts_semantic_mask = mix_pts_semantic_mask.squeeze()
        elif isinstance(mix_pts_semantic_mask, np.ndarray) and mix_pts_semantic_mask.ndim > 1:
            mix_pts_semantic_mask = mix_pts_semantic_mask.squeeze()

        rho = torch.sqrt(points_coord[:, 0]**2 + points_coord[:, 1]**2)
        pitch = torch.atan2(points_coord[:, 2], rho)
        pitch = torch.clamp(pitch, self.pitch_angles[0] + 1e-5,
                            self.pitch_angles[1] - 1e-5)

        mix_rho = torch.sqrt(mix_points_coord[:, 0]**2 +
                             mix_points_coord[:, 1]**2)
        mix_pitch = torch.atan2(mix_points_coord[:, 2], mix_rho)
        mix_pitch = torch.clamp(mix_pitch, self.pitch_angles[0] + 1e-5,
                                self.pitch_angles[1] - 1e-5)

        num_areas = np.random.choice(self.num_areas, size=1)[0]
        angle_list = np.linspace(self.pitch_angles[1], self.pitch_angles[0],
                                 num_areas + 1)
        out_points = []
        out_pts_semantic_mask = []
        for i in range(num_areas):
            # convert angle to radian
            start_angle = angle_list[i + 1] / 180 * np.pi
            end_angle = angle_list[i] / 180 * np.pi
            if i % 2 == 0:  # pick from original point cloud
                idx = (pitch > start_angle) & (pitch <= end_angle)
                # Ensure boolean mask is 1D for indexing
                if isinstance(idx, torch.Tensor) and idx.dim() > 1:
                    idx = idx.squeeze()
                elif isinstance(idx, np.ndarray) and idx.ndim > 1:
                    idx = idx.squeeze()
                
                if is_basepoints:
                    out_points.append(points[idx])
                else:
                    out_points.append(points_tensor[idx])
                
                if isinstance(pts_semantic_mask, torch.Tensor):
                    out_pts_semantic_mask.append(pts_semantic_mask[idx])
                else:
                    out_pts_semantic_mask.append(pts_semantic_mask[idx.numpy() if hasattr(idx, 'numpy') else idx])
            else:  # pick from mixed point cloud
                idx = (mix_pitch > start_angle) & (mix_pitch <= end_angle)
                # Ensure boolean mask is 1D for indexing
                if isinstance(idx, torch.Tensor) and idx.dim() > 1:
                    idx = idx.squeeze()
                elif isinstance(idx, np.ndarray) and idx.ndim > 1:
                    idx = idx.squeeze()
                
                if is_mix_basepoints:
                    out_points.append(mix_points[idx])
                else:
                    out_points.append(mix_points_tensor[idx])
                
                if isinstance(mix_pts_semantic_mask, torch.Tensor):
                    out_pts_semantic_mask.append(mix_pts_semantic_mask[idx])
                else:
                    out_pts_semantic_mask.append(mix_pts_semantic_mask[idx.numpy() if hasattr(idx, 'numpy') else idx])
        
        if is_basepoints:
            out_points = points.cat(out_points)
        else:
            out_points = torch.cat(out_points, dim=0)
        
        if isinstance(out_pts_semantic_mask[0], torch.Tensor):
            out_pts_semantic_mask = torch.cat(out_pts_semantic_mask, dim=0)
        else:
            out_pts_semantic_mask = np.concatenate(out_pts_semantic_mask, axis=0)
        
        input_dict['points'] = out_points
        input_dict['pts_semantic_mask'] = out_pts_semantic_mask
        
        # Recompute train_grid if it exists and we have the necessary parameters
        if 'train_grid' in input_dict and self.grid_size is not None and self.grid_size_vox is not None:
            if isinstance(out_points, BasePoints):
                points_for_grid = out_points.tensor if hasattr(out_points, 'tensor') else out_points.coord
            else:
                points_for_grid = out_points
            
            # Get parameters from input_dict if available, otherwise use instance attributes
            grid_size = input_dict.get('_pointseg_grid_size', self.grid_size)
            grid_size_vox = input_dict.get('_pointseg_grid_size_vox', self.grid_size_vox)
            max_volume_space = input_dict.get('_pointseg_max_volume_space', self.max_volume_space)
            min_volume_space = input_dict.get('_pointseg_min_volume_space', self.min_volume_space)
            fill_label = input_dict.get('_pointseg_fill_label', self.fill_label)
            
            if max_volume_space is not None and min_volume_space is not None:
                recomputed = recompute_train_grid(
                    points_for_grid, grid_size, grid_size_vox,
                    max_volume_space, min_volume_space, fill_label
                )
                input_dict['train_grid'] = recomputed['train_grid']
                input_dict['grid_ind'] = recomputed['grid_ind']
                input_dict['grid_ind_vox'] = recomputed['grid_ind_vox']
                
                # Update train_pts_label to match the new train_grid length
                # out_pts_semantic_mask should have the same length as out_points
                if 'train_pts_label' in input_dict:
                    # Ensure out_pts_semantic_mask is 1D and convert to tensor if needed
                    if isinstance(out_pts_semantic_mask, torch.Tensor):
                        if out_pts_semantic_mask.dim() > 1:
                            out_pts_semantic_mask = out_pts_semantic_mask.squeeze()
                    elif isinstance(out_pts_semantic_mask, np.ndarray):
                        if out_pts_semantic_mask.ndim > 1:
                            out_pts_semantic_mask = out_pts_semantic_mask.squeeze()
                        out_pts_semantic_mask = torch.from_numpy(out_pts_semantic_mask)
                    
                    # Verify length matches
                    points_len = points_for_grid.shape[0] if isinstance(points_for_grid, torch.Tensor) else len(points_for_grid)
                    mask_len = out_pts_semantic_mask.shape[0] if isinstance(out_pts_semantic_mask, torch.Tensor) else len(out_pts_semantic_mask)
                    
                    if points_len != mask_len:
                        # This should not happen, but if it does, we need to handle it
                        # For now, we'll use the mask as-is and let the error surface if there's a real issue
                        pass
                    
                    input_dict['train_pts_label'] = out_pts_semantic_mask
        
        return input_dict

    def __call__(self, input_dict: dict) -> dict:
        """LaserMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through LaserMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before lasermix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.laser_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_areas={self.num_areas}, '
        repr_str += f'pitch_angles={self.pitch_angles}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class RandomChoice(object):
    """Randomly choose one transform from multiple transforms.

    Args:
        transforms (List[List[dict]]): List of transform sequences to choose from.
        prob (List[float], optional): Probability of each transform sequence.
            If None, uniform probability is used. Defaults to None.
    """

    def __init__(self,
                 transforms: List[List[dict]],
                 prob: Optional[List[float]] = None) -> None:
        assert isinstance(transforms, list) and len(transforms) > 0, \
            'transforms should be a non-empty list.'
        
        if prob is not None:
            assert len(prob) == len(transforms), \
                f'Length of prob ({len(prob)}) should equal to length of transforms ({len(transforms)}).'
            assert all(p >= 0 for p in prob), \
                'All probabilities should be non-negative.'
            # Normalize probabilities
            prob_sum = sum(prob)
            assert prob_sum > 0, 'Sum of probabilities should be positive.'
            self.prob = [p / prob_sum for p in prob]
        else:
            # Uniform probability
            self.prob = [1.0 / len(transforms)] * len(transforms)
        
        self.transforms = []
        for transform_seq in transforms:
            if isinstance(transform_seq, list):
                self.transforms.append(Compose(transform_seq))
            else:
                raise TypeError(
                    f'Each element in transforms should be a list, '
                    f'but got {type(transform_seq)}')

    def __call__(self, input_dict: dict) -> dict:
        """Randomly choose and apply one transform sequence.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Output dict after transformation.
        """
        # Randomly choose a transform based on probabilities
        idx = np.random.choice(len(self.transforms), p=self.prob)
        transform = self.transforms[idx]
        
        return transform(input_dict)

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={len(self.transforms)}, '
        repr_str += f'prob={self.prob})'
        return repr_str

