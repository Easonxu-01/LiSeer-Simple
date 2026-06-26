'''
Author: EASON XU
Date: 2025-05-29 02:23:07
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-09-17 16:32:17
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/cylinder3d/cylinder3dhead.py
'''

import torch
from spconv.pytorch.conv import SubMConv3d
from spconv.pytorch.modules import SparseModule
from mmengine.config import ConfigDict
from typing import Optional, Union, List
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]
from mmdet.models import HEADS
from mmengine.model import BaseModule
from projects.unilidar_plugin.occupancy.dense_heads.lovasz_softmax import lovasz_softmax
import torch
import torch.nn.functional as F

@HEADS.register_module()
class Cylinder3DHead(BaseModule):
    """Cylinder3D decoder head.

    Decoder head used in `Cylinder3D <https://arxiv.org/abs/2011.10033>`_.
    Refer to the
    `official code <https://https://github.com/xinge008/Cylinder3D>`_.

    Args:
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Defaults to 0.
        conv_cfg (dict or :obj:`ConfigDict`): Config of conv layers.
            Defaults to dict(type='Conv1d').
        norm_cfg (dict or :obj:`ConfigDict`): Config of norm layers.
            Defaults to dict(type='BN1d').
        act_cfg (dict or :obj:`ConfigDict`): Config of activation layers.
            Defaults to dict(type='ReLU').
        loss_ce (dict or :obj:`ConfigDict`): Config of CrossEntropy loss.
            Defaults to dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0).
        loss_lovasz (dict or :obj:`ConfigDict`): Config of Lovasz loss.
            Defaults to dict(type='LovaszLoss', loss_weight=1.0).
        conv_seg_kernel_size (int): The kernel size used in conv_seg.
            Defaults to 3.
        ignore_index (int): The label index to be ignored. When using masked
            BCE loss, ignore_index should be set to None. Defaults to 19.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 channels: int,
                 num_classes: int,
                 dropout_ratio: float = 0,
                 conv_cfg: ConfigType = dict(type='Conv1d'),
                 norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 conv_seg_kernel_size: int = 3,
                 ignore_index: int = 0,
                 loss_weight: List[float] = [1,1],
                 dual: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:
        super(Cylinder3DHead, self).__init__(init_cfg=init_cfg)

        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv_seg_kernel_size = conv_seg_kernel_size
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.loss_lovasz = lovasz_softmax
        self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.conv_seg = self.build_conv_seg(
            channels=channels,
            num_classes=num_classes,
            kernel_size=conv_seg_kernel_size)
        self.dual = dual

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> SparseModule:
        return SubMConv3d(
            channels,
            num_classes,
            indice_key='logit',
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=True)
        
    def cls_seg(self, feat):
        """Classify each points."""
        output = self.conv_seg(feat)
        return output

    def forward(self, points, sparse_voxels, point_labels=None, voxel_labels=None, point2voxel_maps=None, point_coors=None, voxel_coors=None, return_loss=True):
        """Forward function."""
        sparse_logits = self.cls_seg(sparse_voxels)
        seg_logit_feat = sparse_logits.features
        seg_indices = sparse_logits.indices  # [M', 4] order: [batch, z, y, x]
        loss_dict = {}
        # Build mapping from provided voxel_coors -> seg_logit indices to resolve pruning/reordering
        voxel_to_seg_idx = None
        if voxel_coors is not None:
            # Hash function for coords
            with torch.no_grad():
                device = seg_indices.device
                vc = voxel_coors.to(device).int()
                si = seg_indices.to(device).int()
                vmax = torch.tensor([480, 360, 64], device=device, dtype=torch.int32)  # generous upper bounds
                def make_hash(idx):
                    return idx[:, 0] * (vmax[0] * vmax[1] * vmax[2]) + idx[:, 1] * (vmax[1] * vmax[2]) + idx[:, 2] * vmax[2] + idx[:, 3]
                hash_voxel = make_hash(vc)
                hash_seg = make_hash(si)
                # map seg hash -> seg row
                seg_hash2row = {int(hash_seg[i].item()): i for i in range(hash_seg.shape[0])}
                voxel_to_seg_idx = torch.full((vc.shape[0],), -1, device=device, dtype=torch.long)
                for i in range(vc.shape[0]):
                    h = int(hash_voxel[i].item())
                    if h in seg_hash2row:
                        voxel_to_seg_idx[i] = seg_hash2row[h]
        # Resolve point mapping safely
        if point2voxel_maps is not None:
            p2v = point2voxel_maps.long()
            if voxel_to_seg_idx is not None:
                # Translate original voxel idx -> seg idx; invalid becomes -1
                safe_indices = torch.where((p2v >= 0) & (p2v < voxel_to_seg_idx.numel()), p2v, torch.zeros_like(p2v))
                seg_rows = voxel_to_seg_idx.clone()
                point_seg_idx = torch.where((safe_indices >= 0) & (safe_indices < seg_rows.numel()), seg_rows[safe_indices], torch.full_like(safe_indices, -1))
            else:
                # Fallback: if sizes already match and within bounds, use directly
                max_allowed = seg_logit_feat.shape[0]
                point_seg_idx = torch.where((p2v >= 0) & (p2v < max_allowed), p2v, torch.full_like(p2v, -1))
            valid_pts = point_seg_idx >= 0
            point_voxel_preds = torch.zeros((p2v.shape[0], seg_logit_feat.shape[1]), device=seg_logit_feat.device, dtype=seg_logit_feat.dtype)
            if valid_pts.any():
                point_voxel_preds[valid_pts] = seg_logit_feat[point_seg_idx[valid_pts]]
        elif (point_coors is not None) and (voxel_coors is not None):
            # Recompute mapping from coordinates if no map provided
            with torch.no_grad():
                device = seg_indices.device
                pc = point_coors.to(device).int()
                vc = voxel_coors.to(device).int()
                vmax = torch.tensor([480, 360, 64], device=device, dtype=torch.int32)
                def make_hash(idx):
                    return idx[:, 0] * (vmax[0] * vmax[1] * vmax[2]) + idx[:, 1] * (vmax[1] * vmax[2]) + idx[:, 2] * vmax[2] + idx[:, 3]
                hash_voxel = make_hash(vc)
                voxel_hash2row = {int(hash_voxel[i].item()): i for i in range(hash_voxel.shape[0])}
                point_hash = make_hash(pc)
                v_rows = torch.full((pc.shape[0],), -1, device=device, dtype=torch.long)
                for i in range(pc.shape[0]):
                    h = int(point_hash[i].item())
                    if h in voxel_hash2row:
                        v_rows[i] = voxel_hash2row[h]
            if voxel_to_seg_idx is None:
                point_seg_idx = v_rows
            else:
                point_seg_idx = torch.full_like(v_rows, -1)
                valid_v = v_rows >= 0
                point_seg_idx[valid_v] = voxel_to_seg_idx[v_rows[valid_v]]
            valid_pts = point_seg_idx >= 0
            point_voxel_preds = torch.zeros((point_seg_idx.shape[0], seg_logit_feat.shape[1]), device=seg_logit_feat.device, dtype=seg_logit_feat.dtype)
            if valid_pts.any():
                point_voxel_preds[valid_pts] = seg_logit_feat[point_seg_idx[valid_pts]]
        else:
            point_voxel_preds = None
        if return_loss:
            # 保证在DDP下即使空batch也返回固定的键与Tensor值，避免不同rank间collective不对齐
            zero_voxel = seg_logit_feat.sum() * 0
            zero_point = (point_voxel_preds.sum() * 0) if (point_voxel_preds is not None) else zero_voxel

            # 固定输出键（始终存在），并以零标量作为初始值（与当前设备/精度对齐）
            ce_total = zero_voxel
            lovasz_total = zero_point

            # Voxel-level loss（基于稀疏体素logits）
            if voxel_labels is not None:
                # 对齐voxel_labels到seg_logit_feat的索引顺序（若存在裁剪/重排）
                target_voxel_labels = voxel_labels
                if voxel_to_seg_idx is not None:
                    Mprime = seg_logit_feat.shape[0]
                    target_voxel_labels = torch.full(
                        (Mprime,), self.ignore_index,
                        device=seg_logit_feat.device,
                        dtype=voxel_labels.dtype
                    )
                    valid_map = (voxel_to_seg_idx >= 0) & (voxel_to_seg_idx < Mprime)
                    src_idx = torch.nonzero(valid_map, as_tuple=False).squeeze(1)
                    dst_idx = voxel_to_seg_idx[valid_map]
                    if src_idx.numel() > 0:
                        target_voxel_labels[dst_idx] = voxel_labels[src_idx]

                # 仅当存在有效（非ignore）标签时才计算，以避免空集合导致的不确定行为
                if (target_voxel_labels != self.ignore_index).any():
                    ce_total = ce_total + self.loss_ce(seg_logit_feat, target_voxel_labels)
                    lovasz_total = lovasz_total + self.loss_lovasz(
                        seg_logit_feat, target_voxel_labels, ignore=self.ignore_index
                    )

            # Point-level loss（将体素logits映射至点后计算）
            if (point_labels is not None) and (point_voxel_preds is not None):
                point_labels = point_labels.reshape(-1)
                # 过滤掉无效的标签值（越界或ignore_index）
                valid_mask = (
                    (point_labels >= 0)
                    & (point_labels < self.num_classes)
                    & (point_labels != self.ignore_index)
                )
                if valid_mask.any():
                    valid_preds = point_voxel_preds[valid_mask]
                    valid_labels = point_labels[valid_mask]
                    ce_total = ce_total + self.loss_ce(valid_preds, valid_labels)
                    lovasz_total = lovasz_total + self.loss_lovasz(
                        valid_preds, valid_labels, ignore=self.ignore_index
                    )

            # 始终返回固定键，值为同设备/同精度Tensor，空batch时即为零
            loss_dict['loss_ce'] = self.loss_weight[0] * ce_total
            loss_dict['loss_lovasz'] = self.loss_weight[1] * lovasz_total
            return loss_dict
        else:
            return point_voxel_preds

    def map_voxel_preds_to_points(self, points, sparse_output, point_cloud_range, voxel_size, batch_idx=0):
        """
        Map voxel predictions from SparseConvTensor to point-level predictions based on cylinder voxelization.

        Args:
            points (torch.Tensor): [batch_size, N_points, 3] or [N_points, 3] point coordinates (x, y, z).
            sparse_output (SparseConvTensor): Output of spconv SparseConv network, with attributes `features` and `indices`.
            point_cloud_range (list or tuple): [min_rho, min_phi, min_z, max_rho, max_phi, max_z].
            voxel_size (list or tuple): voxel size in (rho, phi, z).
            batch_idx (int): Batch index for current batch (default is 0 for single batch).

        Returns:
            torch.Tensor: Point-level voxel predictions, shape [batch_size, N_points, num_classes] or [N_points, num_classes].
        """
        device = points.device
        is_batched = len(points.shape) == 3
        if is_batched:
            batch_size = points.shape[0]
            points = points.reshape(-1, points.shape[2])  # [batch_size * N_points, 3]

        # Step 1: Construct hash table from SparseConvTensor
        voxel_predictions = sparse_output.features  # (N_voxels, num_classes)
        voxel_coords = sparse_output.indices  # (N_voxels, 4), [batch_idx, z, y, x]

        voxel_hashes = (
            voxel_coords[:, 0] * (480 * 360 * 20)
            + voxel_coords[:, 1] * (360 * 20)
            + voxel_coords[:, 2] * 20
            + voxel_coords[:, 3]
        )

        voxel_hash2idx = {int(voxel_hashes[i].item()): i for i in range(voxel_hashes.shape[0])}

        # Step 2: Convert points to cylindrical coordinates and hash
        rho = torch.sqrt(points[:, 0]**2 + points[:, 1]**2)
        phi = torch.atan2(points[:, 1], points[:, 0])
        polar_points = torch.stack((rho, phi, points[:, 2]), dim=-1)

        min_bound = torch.tensor(point_cloud_range[:3], device=device)
        max_bound = torch.tensor(point_cloud_range[3:], device=device)
        voxel_size = torch.tensor(voxel_size, device=device)

        polar_res_clamp = torch.clamp(polar_points, min_bound, max_bound)
        res_coors = torch.floor((polar_res_clamp - min_bound) / voxel_size).int()
        
        if is_batched:
            # Create batch indices for each point
            batch_indices = torch.arange(batch_size, device=device).repeat_interleave(points.shape[0] // batch_size)
            batch_indices = batch_indices.unsqueeze(1)
        else:
            batch_indices = torch.full((points.shape[0], 1), batch_idx, dtype=torch.int32, device=device)
            
        res_coors_batch = torch.cat((batch_indices, res_coors), dim=1)

        points_hashes = (
            res_coors_batch[:, 0] * (480 * 360 * 20)
            + res_coors_batch[:, 1] * (360 * 20)
            + res_coors_batch[:, 2] * 20
            + res_coors_batch[:, 3]
        )

        # Step 3: Efficient mapping using hash table
        points_hashes_np = points_hashes.cpu().numpy()
        voxel_idx_list = [voxel_hash2idx.get(int(h), -1) for h in points_hashes_np]

        voxel_idx_tensor = torch.tensor(voxel_idx_list, device=device)

        valid_mask = voxel_idx_tensor >= 0

        # Allocate prediction tensor
        point_voxel_preds = torch.zeros((points.shape[0], voxel_predictions.shape[1]), device=device)

        # Assign voxel predictions to valid points
        point_voxel_preds[valid_mask] = voxel_predictions[voxel_idx_tensor[valid_mask]]

        if is_batched:
            # Reshape back to [batch_size, N_points, num_classes]
            point_voxel_preds = point_voxel_preds.reshape(batch_size, -1, voxel_predictions.shape[1])

        return point_voxel_preds

