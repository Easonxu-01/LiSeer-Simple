'''
Author: EASON XU
Date: 2024-01-18 14:56:53
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-09-18 11:19:14
Description: 头部注释

FilePath: /UniLiDAR/projects/unilidar_plugin/datasets/pipelines/VoxelClassMapping.py
'''
import trimesh
import mmcv
import numpy as np
import numba as nb
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
import yaml, os
import torch
from scipy import stats
from scipy.ndimage import zoom
from skimage import transform
import pdb
import torch.nn.functional as F
import copy


@PIPELINES.register_module()
class VoxelClassMapping(object):
    """Map original semantic class to valid category ids.

    Required Keys:

    - seg_label_mapping (np.ndarray)
    - pts_semantic_mask (np.ndarray)

    Added Keys:

    - points (np.float32)

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    """

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
            Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        # 初始化变量
        pts_semantic_mask = None
        pts_semantic_mask_train = None
        voxel_semantic_mask = None

        # 检查并获取 pts_semantic_mask
        if results.get('pts_semantic_mask', None) is not None:
            pts_semantic_mask = results['pts_semantic_mask']
        if results.get('train_pts_label', None) is not None:
            pts_semantic_mask_train = results['train_pts_label']
        
        # 检查并获取 voxel_semantic_mask
        if results.get('train_voxel_label', None) is not None:
            voxel_semantic_mask = results['train_voxel_label']
        elif results.get('gt_occ', None) is not None:
            voxel_semantic_mask = results['gt_occ'].data
        elif results.get('processed_label', None) is not None:
            voxel_semantic_mask = results['processed_label']
        elif results.get('voxel_semantic_mask', None) is not None:
            voxel_semantic_mask = results['voxel_semantic_mask']
        
        if 'seg_label_mapping' not in results:
            if 'labels_map' in results:
                results['seg_label_mapping'] = {}
                results['seg_label_mapping'] = results['labels_map']   
        assert 'seg_label_mapping' in results
        seg_label_mapping = results['seg_label_mapping']

        if isinstance(seg_label_mapping, dict):
            # 获取可能的最大标签值
            max_label_mapping = max(seg_label_mapping.keys())
            max_voxel_label = 0  # 初始化默认值
            
            # 安全地获取最大标签值
            if voxel_semantic_mask is not None:
                max_voxel_label = voxel_semantic_mask.max()
            if pts_semantic_mask is not None:
                if pts_semantic_mask_train is not None:
                    max_pts_label = pts_semantic_mask_train.max()
                else:
                    max_pts_label = pts_semantic_mask.max()
                
                max_label_pts = max(max_label_mapping, max_pts_label)
                if isinstance(max_label_pts, torch.Tensor):
                    max_label_pts = max_label_pts.item()
                mapping_array_pts = np.zeros(max_label_pts + 1, dtype=int)
                for original_label, mapped_label in seg_label_mapping.items():
                    mapping_array_pts[original_label] = mapped_label
                
            # in completion we have to distinguish empty and invalid voxels.
            # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
            if pts_semantic_mask is not None:
                converted_pts_sem_mask = mapping_array_pts[pts_semantic_mask]
                results['pts_semantic_mask'] = torch.from_numpy(converted_pts_sem_mask)
            if pts_semantic_mask_train is not None:
                converted_pts_sem_mask_train = torch.from_numpy(mapping_array_pts[pts_semantic_mask_train])
                results['train_pts_label'] = converted_pts_sem_mask_train
            if results.get('train_voxel_label', None) is not None:
                converted_train_voxel_label = torch.from_numpy(mapping_array_pts[voxel_semantic_mask]) 
                results['train_voxel_label'] = converted_train_voxel_label
            
            max_label_voxel = max(max_label_mapping, max_voxel_label)
            if isinstance(max_label_voxel, torch.Tensor):
                max_label_voxel = max_label_voxel.item()
            # 创建一个足够大的映射数组
            mapping_array_voxel = np.zeros(max_label_voxel + 1, dtype=int)
            # 填充映射数组
            for original_label, mapped_label in seg_label_mapping.items():
                mapping_array_voxel[original_label] = mapped_label
            if max_label_voxel == 260:
                mapping_array_voxel[mapping_array_voxel == 0] = 260
                mapping_array_voxel[0] = 0
            elif max_label_voxel != 15:
                mapping_array_voxel[mapping_array_voxel == 0] = 255  # map 0 to 'invalid'
                mapping_array_voxel[0] = 0  # only 'empty' stays 'empty'
            
            if voxel_semantic_mask is not None and results.get('train_voxel_label', None) is None:
                converted_voxel_semantic_mask = mapping_array_voxel[voxel_semantic_mask]
                converted_voxel_semantic_mask[converted_voxel_semantic_mask == 260] = 255

            # results['pts_semantic_mask'] = converted_pts_sem_mask
            if results.get('voxel_semantic_mask', None) is not None:
                results['voxel_semantic_mask'] = converted_voxel_semantic_mask
            elif results.get('gt_occ', None) is not None:
                results['gt_occ'] = DC(converted_voxel_semantic_mask.to(torch.uint8))
            elif results.get('processed_label', None) is not None:
                results['processed_label'] = converted_voxel_semantic_mask
            if results.get('pts_semantic_mask', None) is not None:
                results['pts_semantic_mask'] = converted_pts_sem_mask

        # 'eval_ann_info' will be passed to evaluator
        # if 'eval_ann_info' in results:
        #     assert 'pts_semantic_mask' in results['eval_ann_info']
        #     results['eval_ann_info']['pts_semantic_mask'] = \
        #         converted_pts_sem_mask
        if results.get('gt_occ', None) is None and results.get('voxel_semantic_mask', None) is not None:
            processed_label = results['voxel_semantic_mask']
            results['gt_occ'] = processed_label
        if results.get('train_pts_label', None) is None and results.get('pts_semantic_mask', None) is not None:
            if isinstance(converted_pts_sem_mask, torch.Tensor):
                results['train_pts_label'] = converted_pts_sem_mask
            else:
                results['train_pts_label'] = torch.from_numpy(converted_pts_sem_mask)

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str
    