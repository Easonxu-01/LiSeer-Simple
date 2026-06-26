'''
Author: EASON XU
Date: 2024-12-17 13:30:22
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-09-10 21:20:51
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/pointocc_model/pointtpv_occ.py
'''
import torch
from mmdet3d.models import builder
from mmdet.models import DETECTORS
from mmdet3d.models.detectors import CenterPoint
import yaml
import numpy as np
import torch.nn.functional as F
import copy
import random
import open3d as o3d

@DETECTORS.register_module()
class PointTPV_Occ(CenterPoint):
    
    def __init__(self,
                 lidar_tokenizer=None,
                 lidar_backbone=None,
                 lidar_neck=None,
                 tpv_aggregator=None,
                 empty_idx = None,
                 random = False,
                 sensor=None,
                 **kwargs,
                 ):

        super().__init__()

        if lidar_tokenizer:
            self.lidar_tokenizer = builder.build_middle_encoder(lidar_tokenizer)
        if lidar_backbone:
            self.lidar_backbone = builder.build_backbone(lidar_backbone)
        if lidar_neck:
            self.lidar_neck = builder.build_neck(lidar_neck)
        if tpv_aggregator:
            self.tpv_aggregator = builder.build_voxel_encoder(tpv_aggregator)

        self.fp16_enabled = False
        self.empty_idx = empty_idx
        self.random = random
        self.sensor_config = sensor

    def extract_feat(self, train_grid, voxel_label):
        """Extract features of points."""
        x_3view = self.lidar_tokenizer(train_grid, voxel_label)
        tpv_list = []
        x_tpv = self.lidar_backbone(x_3view)
        for x in x_tpv:
            x = self.lidar_neck(x)
            if not isinstance(x, torch.Tensor):
                x = x[0]
            tpv_list.append(x)
        return tpv_list

    def forward_train(self,
                train_grid=None,
                voxel_label=None,
                grid_ind_vox=None,
                train_grid_vox_coarse=None,
                processed_label=None,
                return_loss=True,
                dataset_flag= None,
                **kwargs
        ):
        """Forward training function.
        """
        if self.random:
            train_grid, voxel_label = self._sampling(train_grid,voxel_label)
        if isinstance(processed_label, list):
            if all(pl.shape == processed_label[0].shape for pl in processed_label):
                processed_label = torch.cat([pl[None, ...] for pl in processed_label], dim=0)
            else:
                processed_label = [pl[None, ...] for pl in processed_label]
        if isinstance(train_grid_vox_coarse, list):
            if all(tgvc.shape == train_grid_vox_coarse[0].shape for tgvc in train_grid_vox_coarse):
                train_grid_vox_coarse = torch.cat([tgvc[None, ...] for tgvc in train_grid_vox_coarse], dim=0)
            else:
                train_grid_vox_coarse = [tgvc[None, ...] for tgvc in train_grid_vox_coarse]
            
            
        x_lidar_tpv = self.extract_feat(train_grid=train_grid, voxel_label=voxel_label)
        outs = self.tpv_aggregator(x_lidar_tpv, voxels=grid_ind_vox, voxels_coarse=train_grid_vox_coarse, voxel_label=processed_label, dataset_flag= dataset_flag, return_loss=return_loss)
        return outs
    
    def simple_test(self,
                train_grid=None,
                voxel_label=None,
                grid_ind_vox=None,
                train_grid_vox_coarse=None,
                processed_label=None,
                dataset_flag= None,
                visible_mask = None,
                return_loss=False,
        ):
        
        x_lidar_tpv = self.extract_feat(train_grid=train_grid, voxel_label=voxel_label)
        output = self.tpv_aggregator(x_lidar_tpv, voxels=grid_ind_vox, voxels_coarse=train_grid_vox_coarse, voxel_label=processed_label, dataset_flag= dataset_flag, return_loss=return_loss)
        gt_occ = processed_label
        
        if not hasattr(self, 'pts_bbox_head'):
            assert self.tpv_aggregator is not None, "Unknown Model Type"
            if not self.tpv_aggregator.dual:
                pred_c = output['output_voxels']
                SC_metric, _ = self.evaluation_semantic(pred_c, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
                SSC_metric, _ = self.evaluation_semantic(pred_c, gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
            elif len(dataset_flag)==1 or all(dataset_flag[0]==dataset_flag[i] for i in range(len(dataset_flag))):
                if self.training == False:
                    flag = dataset_flag[0]
                    if flag.item() == 1:
                        pred_c_1 = output['output_voxels_1']
                        SC_metric_1, _ = self.evaluation_semantic(pred_c_1, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
                        SSC_metric_1, _ = self.evaluation_semantic(pred_c_1, gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
                        SC_metric = SC_metric_1 
                    elif flag.item() == 2:
                        pred_c_2 = output['output_voxels_2']
                        SC_metric_2, _ = self.evaluation_semantic(pred_c_2, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
                        SSC_metric_2, _ = self.evaluation_semantic(pred_c_2, gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
                        SC_metric = SC_metric_2
            else:
                pred_c_1, pred_c_2 = output['output_voxels_1'], output['output_voxels_2']
                SC_metric_1, _ = self.evaluation_semantic(pred_c_1, gt_occ[::2], dataset_flag[::2], eval_type='SC', visible_mask=visible_mask)
                SSC_metric_1, _ = self.evaluation_semantic(pred_c_1, gt_occ[::2], dataset_flag[::2], eval_type='SSC', visible_mask=visible_mask)
                SC_metric_2, _ = self.evaluation_semantic(pred_c_2, gt_occ[1::2], dataset_flag[1::2], eval_type='SC', visible_mask=visible_mask)
                SSC_metric_2, _ = self.evaluation_semantic(pred_c_2, gt_occ[1::2], dataset_flag[1::2], eval_type='SSC', visible_mask=visible_mask)
                SC_metric = SC_metric_1 + SC_metric_2
        elif not self.pts_bbox_head.dual:
            pred_c = output['output_voxels'][0]
            SC_metric, _ = self.evaluation_semantic(pred_c, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
            SSC_metric, _ = self.evaluation_semantic(pred_c, gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
        else:
            if len(dataset_flag)==1 or all(dataset_flag[0]==dataset_flag[i] for i in range(len(dataset_flag))):
                if self.training == False:
                    flag = dataset_flag[0]
                    if flag.item() == 1:
                        pred_c_1 = output['output_voxels_1'][0]
                        SC_metric_1, _ = self.evaluation_semantic(pred_c_1, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
                        SSC_metric_1, _ = self.evaluation_semantic(pred_c_1, gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
                        SC_metric = SC_metric_1 
                    elif flag.item() == 2:
                        pred_c_2 = output['output_voxels_2'][0]
                        SC_metric_2, _ = self.evaluation_semantic(pred_c_2, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
                        SSC_metric_2, _ = self.evaluation_semantic(pred_c_2, gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
                        SC_metric = SC_metric_2
            else:
                pred_c_1, pred_c_2 = output['output_voxels_1'][0], output['output_voxels_2'][0]
                SC_metric_1, _ = self.evaluation_semantic(pred_c_1, gt_occ[::2], dataset_flag[::2], eval_type='SC', visible_mask=visible_mask)
                SSC_metric_1, _ = self.evaluation_semantic(pred_c_1, gt_occ[::2], dataset_flag[::2], eval_type='SSC', visible_mask=visible_mask)
                SC_metric_2, _ = self.evaluation_semantic(pred_c_2, gt_occ[1::2], dataset_flag[1::2], eval_type='SC', visible_mask=visible_mask)
                SSC_metric_2, _ = self.evaluation_semantic(pred_c_2, gt_occ[1::2], dataset_flag[1::2], eval_type='SSC', visible_mask=visible_mask)
                SC_metric = SC_metric_1 + SC_metric_2
            # SSC_metric = SSC_metric_1 + SSC_metric_2
        
        pred_f = None
        SSC_metric_fine = None
        if not hasattr(self, 'pts_bbox_head'):
            output_voxels_fine = output.get('output_voxels_fine', None)
            if output_voxels_fine is not None:
                if output['output_coords_fine'] is not None:
                    fine_pred = output['output_voxels_fine'][0]  # N ncls
                    fine_coord = output['output_coords_fine'][0]  # 3 N
                    pred_f = self.empty_idx * torch.ones_like(gt_occ)[:, None].repeat(1, fine_pred.shape[1], 1, 1, 1).float()
                    pred_f[:, :, fine_coord[0], fine_coord[1], fine_coord[2]] = fine_pred.permute(1, 0)[None]
                else:
                    pred_f = output['output_voxels_fine'][0]
                    
                SC_metric_fine, _ = self.evaluation_semantic(pred_f, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
                SSC_metric_fine, SSC_occ_metric_fine = self.evaluation_semantic(pred_f,  gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
        elif not self.pts_bbox_head.dual:
            output_voxels_fine = output.get('output_voxels_fine', None)
            if output_voxels_fine is not None:
                if output['output_coords_fine'] is not None:
                    fine_pred = output['output_voxels_fine'][0]  # N ncls
                    fine_coord = output['output_coords_fine'][0]  # 3 N
                    pred_f = self.empty_idx * torch.ones_like(gt_occ)[:, None].repeat(1, fine_pred.shape[1], 1, 1, 1).float()
                    pred_f[:, :, fine_coord[0], fine_coord[1], fine_coord[2]] = fine_pred.permute(1, 0)[None]
                else:
                    pred_f = output['output_voxels_fine'][0]
                    
                SC_metric_fine, _ = self.evaluation_semantic(pred_f, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
                SSC_metric_fine, SSC_occ_metric_fine = self.evaluation_semantic(pred_f,  gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
        else:
            if len(dataset_flag)==1 or all(dataset_flag[0]==dataset_flag[i] for i in range(len(dataset_flag))):
                if self.training == False:
                    flag = dataset_flag[0]
                    if flag.item() == 1:
                        output_voxels_fine_1 = output.get('output_voxels_fine_1', None)
                        if output_voxels_fine_1 is not None:
                            if output['output_coords_fine_1'] is not None:
                                fine_pred_1 = output['output_voxels_fine_1'][0]  # N ncls
                                fine_coord_1 = output['output_coords_fine_1'][0]  # 3 N
                                pred_f_1 = self.empty_idx * torch.ones_like(gt_occ)[:, None].repeat(1, fine_pred_1.shape[1], 1, 1, 1).float()
                                pred_f_1[:, :, fine_coord_1[0], fine_coord_1[1], fine_coord_1[2]] = fine_pred_1.permute(1, 0)[None]
                                SC_metric_fine_1, _ = self.evaluation_semantic(pred_f_1, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
                                SSC_metric_fine_1, _ = self.evaluation_semantic(pred_f_1, gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
                                SC_metric_fine = SC_metric_fine_1
                    elif flag.item() == 2:
                        output_voxels_fine_2 = output.get('output_voxels_fine_2', None)
                        if output_voxels_fine_2 is not None:
                            if output['output_coords_fine_2'] is not None:
                                fine_pred_2 = output['output_voxels_fine_2'][0]
                                fine_coord_2 = output['output_coords_fine_2'][0]
                                pred_f_2 = self.empty_idx * torch.ones_like(gt_occ)[:, None].repeat(1, fine_pred_2.shape[1], 1, 1, 1).float()
                                pred_f_2[:, :, fine_coord_2[0], fine_coord_2[1], fine_coord_2[2]] = fine_pred_2.permute(1, 0)[None]
                                SC_metric_fine_2, _ = self.evaluation_semantic(pred_f_2, gt_occ, dataset_flag, eval_type='SC', visible_mask=visible_mask)
                                SSC_metric_fine_2, _ = self.evaluation_semantic(pred_f_2, gt_occ, dataset_flag, eval_type='SSC', visible_mask=visible_mask)
                                SC_metric_fine = SC_metric_fine_2
            else:
                output_voxels_fine_1 = output.get('output_voxels_fine_1', None)
                output_voxels_fine_2 = output.get('output_voxels_fine_2', None)
                if output['output_voxels_fine_1'] is not None:
                    if output_voxels_fine_1 is not None:
                        fine_pred_1 = output['output_voxels_fine_1'][0]  # N ncls
                        fine_coord_1 = output['output_coords_fine_1'][0]  # 3 N
                        pred_f_1 = self.empty_idx * torch.ones_like(gt_occ[::2])[:, None].repeat(1, fine_pred_1.shape[1], 1, 1, 1).float()
                        pred_f_1[:, :, fine_coord_1[0], fine_coord_1[1], fine_coord_1[2]] = fine_pred_1.permute(1, 0)[None]
                if output_voxels_fine_2 is not None:
                    if output['output_coords_fine_2'] is not None:
                        fine_pred_2 = output['output_voxels_fine_2'][0]  # N ncls
                        fine_coord_2 = output['output_coords_fine_2'][0]  # 3 N
                        pred_f_2 = self.empty_idx * torch.ones_like(gt_occ[1::2])[:, None].repeat(1, fine_pred_2.shape[1], 1, 1, 1).float()
                        pred_f_2[:, :, fine_coord_2[0], fine_coord_2[1], fine_coord_2[2]] = fine_pred_2.permute(1, 0)[None]

                    SC_metric_fine_1, _ = self.evaluation_semantic(pred_f_1, gt_occ[::2], dataset_flag[::2], eval_type='SC', visible_mask=visible_mask)
                    SSC_metric_fine_1, _ = self.evaluation_semantic(pred_f_1, gt_occ[::2], dataset_flag[::2], eval_type='SSC', visible_mask=visible_mask)
                    SC_metric_fine_2, _ = self.evaluation_semantic(pred_f_2, gt_occ[1::2], dataset_flag[1::2], eval_type='SC', visible_mask=visible_mask)
                    SSC_metric_fine_2, _ = self.evaluation_semantic(pred_f_2, gt_occ[1::2], dataset_flag[1::2], eval_type='SSC', visible_mask=visible_mask)
                    SC_metric_fine = SC_metric_fine_1 + SC_metric_fine_2
                    # SSC_metric_fine = SSC_metric_fine_1 + SSC_metric_fine_2
        
        if not hasattr(self, 'pts_bbox_head'):
            assert self.tpv_aggregator is not None, "Unknown Model Type"
            if not self.tpv_aggregator.dual:
                test_output = {
                'SC_metric': SC_metric,
                'SSC_metric': SSC_metric,
                'pred_c': pred_c,
            }
                if SSC_metric_fine is not None:
                    test_output['SC_metric_fine'] = SC_metric_fine
                    test_output['SSC_metric_fine'] = SSC_metric_fine
                    test_output['pred_f'] = pred_f
            elif len(dataset_flag)==1 or all(dataset_flag[0]==dataset_flag[i] for i in range(len(dataset_flag))):
                if self.training == False:
                    flag = dataset_flag[0]
                    if flag.item() == 1:
                        test_output = {
                            'SC_metric': SC_metric,
                            'SC_metric_1': SC_metric_1,
                            'SSC_metric_1': SSC_metric_1,
                            'pred_c_1': pred_c_1,
                        }
                        # SSC_metric_fine_1=test_output.get('SSC_metric_fine_1',None)
                        try:
                            # 尝试访问变量，如果它们已定义
                            test_output['SC_metric_fine'] = SC_metric_fine
                            test_output['SC_metric_fine_1'] = SC_metric_fine_1
                            test_output['SSC_metric_fine_1'] = SSC_metric_fine_1
                            test_output['pred_f_1'] = pred_f_1
                        except NameError:
                            pass
                        
                    if flag.item() == 2:
                        test_output = {
                            'SC_metric': SC_metric,
                            'SC_metric_2': SC_metric_2,
                            'SSC_metric_2': SSC_metric_2,
                            'pred_c_2': pred_c_2,
                        }
                        # SSC_metric_fine_2=test_output.get('SSC_metric_fine_2',None)
                        try:
                            test_output['SC_metric_fine'] = SC_metric_fine
                            test_output['SC_metric_fine_2'] = SC_metric_fine_2
                            test_output['SSC_metric_fine_2'] = SSC_metric_fine_2
                            test_output['pred_f_2'] = pred_f_2
                        except NameError:
                            pass
            else:
                test_output = {
                    'SC_metric': SC_metric,
                    'SC_metric_1': SC_metric_1,
                    'SC_metric_2': SC_metric_2,
                    'SSC_metric_1': SSC_metric_1,
                    'SSC_metric_2': SSC_metric_2,
                    'pred_c_1': pred_c_1,
                    'pred_c_2': pred_c_2,
                }

                try:
                    # 尝试访问变量，如果它们已定义
                    test_output['SC_metric_fine'] = SC_metric_fine
                    test_output['SC_metric_fine_1'] = SC_metric_fine_1
                    test_output['SC_metric_fine_2'] = SC_metric_fine_2
                    test_output['SSC_metric_fine_1'] = SSC_metric_fine_1
                    test_output['SSC_metric_fine_2'] = SSC_metric_fine_2
                    test_output['pred_f_1'] = pred_f_1
                    test_output['pred_f_2'] = pred_f_2
                except NameError:
                    # 如果变量未定义，就会捕获到NameError异常
                    # 这里可以不执行任何操作，或者打印一条消息等
                    pass

        elif not self.pts_bbox_head.dual:
            test_output = {
                'SC_metric': SC_metric,
                'SSC_metric': SSC_metric,
                'pred_c': pred_c,
            }

            if SSC_metric_fine is not None:
                test_output['SC_metric_fine'] = SC_metric_fine
                test_output['SSC_metric_fine'] = SSC_metric_fine
                test_output['pred_f'] = pred_f
        else:
            if len(dataset_flag)==1 or all(dataset_flag[0]==dataset_flag[i] for i in range(len(dataset_flag))):
                if self.training == False:
                    flag = dataset_flag[0]
                    if flag.item() == 1:
                        test_output = {
                            'SC_metric': SC_metric,
                            'SC_metric_1': SC_metric_1,
                            'SSC_metric_1': SSC_metric_1,
                            'pred_c_1': pred_c_1,
                        }
                        # SSC_metric_fine_1=test_output.get('SSC_metric_fine_1',None)
                        try:
                            # 尝试访问变量，如果它们已定义
                            test_output['SC_metric_fine'] = SC_metric_fine
                            test_output['SC_metric_fine_1'] = SC_metric_fine_1
                            test_output['SSC_metric_fine_1'] = SSC_metric_fine_1
                            test_output['pred_f_1'] = pred_f_1
                        except NameError:
                            # 如果变量未定义，就会捕获到NameError异常
                            # 这里可以不执行任何操作，或者打印一条消息等
                            pass
                    if flag.item() == 2:
                        test_output = {
                            'SC_metric': SC_metric,
                            'SC_metric_2': SC_metric_2,
                            'SSC_metric_2': SSC_metric_2,
                            'pred_c_2': pred_c_2,
                        }
                        # SSC_metric_fine_2=test_output.get('SSC_metric_fine_2',None)
                        try:
                            # 尝试访问变量，如果它们已定义
                            test_output['SC_metric_fine'] = SC_metric_fine
                            test_output['SC_metric_fine_2'] = SC_metric_fine_2
                            test_output['SSC_metric_fine_2'] = SSC_metric_fine_2
                            test_output['pred_f_2'] = pred_f_2
                        except NameError:
                            # 如果变量未定义，就会捕获到NameError异常
                            # 这里可以不执行任何操作，或者打印一条消息等
                            pass
                        
            else:
                test_output = {
                    'SC_metric': SC_metric,
                    'SC_metric_1': SC_metric_1,
                    'SC_metric_2': SC_metric_2,
                    'SSC_metric_1': SSC_metric_1,
                    'SSC_metric_2': SSC_metric_2,
                    'pred_c_1': pred_c_1,
                    'pred_c_2': pred_c_2,
                }

                try:
                    # 尝试访问变量，如果它们已定义
                    test_output['SC_metric_fine'] = SC_metric_fine
                    test_output['SC_metric_fine_1'] = SC_metric_fine_1
                    test_output['SC_metric_fine_2'] = SC_metric_fine_2
                    test_output['SSC_metric_fine_1'] = SSC_metric_fine_1
                    test_output['SSC_metric_fine_2'] = SSC_metric_fine_2
                    test_output['pred_f_1'] = pred_f_1
                    test_output['pred_f_2'] = pred_f_2
                except NameError:
                    # 如果变量未定义，就会捕获到NameError异常
                    # 这里可以不执行任何操作，或者打印一条消息等
                    pass
                

        return test_output
    
    def forward_test(self,
                train_grid=None,
                voxel_label=None,
                grid_ind_vox=None,
                train_grid_vox_coarse=None,
                processed_label=None,
                dataset_flag= None,
                visible_mask = None,
                return_loss=False,
                **kwargs
        ):
        """
        Forward testing function.
        """
        if isinstance(processed_label, list):
            if all(pl.shape == processed_label[0].shape for pl in processed_label):
                processed_label = torch.cat([pl[None, ...] for pl in processed_label], dim=0)
            else:
                processed_label = [pl[None, ...] for pl in processed_label]
        if isinstance(train_grid_vox_coarse, list):
            if all(tgvc.shape == train_grid_vox_coarse[0].shape for tgvc in train_grid_vox_coarse):
                train_grid_vox_coarse = torch.cat([tgvc[None, ...] for tgvc in train_grid_vox_coarse], dim=0)
            else:
                train_grid_vox_coarse = [tgvc[None, ...] for tgvc in train_grid_vox_coarse]
            
        return self.simple_test(train_grid, voxel_label, grid_ind_vox, train_grid_vox_coarse, processed_label, dataset_flag, visible_mask, return_loss)
    
    def evaluation_semantic(self, pred, gt, dataset_flag, eval_type, visible_mask=None):
        if isinstance(gt, list):        
            if all(isinstance(x, torch.Tensor) for x in gt):
                for i in range(len(gt)):
                    gt[i] = gt[i].reshape(256, 256, 32)
                gt = torch.stack(gt, dim=0)
        if len(gt.shape) != 4:
            gt = gt.reshape(-1, 256, 256, 32)
        _, H, W, D = gt.shape
        # if not isinstance(pred, list):
        #     pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        #     pred = torch.argmax(pred[0], dim=0).cpu().numpy()
        # else:
        #     pred_1 = pred[0]
        #     pred_2 = pred[1]
        #     pred_1 = F.interpolate(pred_1, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        #     pred_1 = torch.argmax(pred_1[0], dim=0).cpu().numpy()
        #     pred_2 = F.interpolate(pred_2, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        #     pred_2 = torch.argmax(pred_2[0], dim=0).cpu().numpy()
        #     pred = np.concatenate((pred_1,pred_2), axis=0)
        pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        pred = torch.argmax(pred[0], dim=0).cpu().numpy()
            
        gt = gt[0].cpu().numpy()
        gt = gt.astype(np.int)
        #TODO:
        # ignore noise
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
        
        if (np.sort(np.unique(gt))[-2] if len(np.unique(gt)) >= 2 else None)  <= 8:
            noise_mask = gt != 255
            max_label = 9
        elif dataset_flag == 1:
            noise_mask = gt != 255
            max_label = 17
        elif dataset_flag == 2:
            noise_mask = gt != 255
            max_label = 20
        elif dataset_flag == 3:
            noise_mask = gt != 255
            max_label = 16

        if eval_type == 'SC':
            # 0 1 split
            gt[gt != self.empty_idx] = 1
            pred[pred != self.empty_idx] = 1
            return fast_hist(pred[noise_mask], gt[noise_mask], max_label=2), None


        if eval_type == 'SSC':
            hist_occ = None
            if visible_mask is not None:
                visible_mask = visible_mask[0].cpu().numpy()
                mask = noise_mask & (visible_mask!=0)
                hist_occ = fast_hist(pred[mask], gt[mask], max_label=max_label)

            hist = fast_hist(pred[noise_mask], gt[noise_mask], max_label=max_label)
            return hist, hist_occ
        
    def _sampling(self, points, labels):
        
        LiDAR_height=self.sensor_config.get('LiDAR_height', [1, 2])
        num_of_beams=self.sensor_config.get('num_of_beams', [16, 128])
        horizontal_angular_resolution=self.sensor_config.get('horizontal_angular_resolution', [900, 3600])
        lower_vertical_field_of_view_bound=self.sensor_config.get('lower_vertical_field_of_view_bound', [-40, -5])
        upper_vertical_field_of_view_bound=self.sensor_config.get('upper_vertical_field_of_view_bound', [0, 25])
        self.lidar_height = round(random.uniform(LiDAR_height[0], LiDAR_height[1]), 1)
        self.beams = sample_beams(num_of_beams[0], num_of_beams[1])
        self.horizontal_resolution = random.choice(range(horizontal_angular_resolution[0], horizontal_angular_resolution[1]+1, 100))
        self.vertical_lower_angle = round(random.uniform(lower_vertical_field_of_view_bound[0], lower_vertical_field_of_view_bound[1]), 1)
        self.vertical_upper_angle = round(random.uniform(upper_vertical_field_of_view_bound[0], upper_vertical_field_of_view_bound[1]), 1)
        
        results_points = []
        results_labels = []
        
        for idx, pt in enumerate(points):
            point_cloud_tensor = torch.concat((points[idx][:,6:8],points[idx][:,5].reshape(-1,1) ),dim=1)
            device = point_cloud_tensor.device

            vertical_angles_rad = torch.deg2rad(
        torch.linspace(self.vertical_lower_angle, self.vertical_upper_angle, self.beams, dtype=torch.float32)
    ).to(device)
            horizontal_angles_rad = torch.deg2rad(
            torch.linspace(0, 360, self.horizontal_resolution, dtype=torch.float32)
        ).to(device)
            ray_dir_grid = torch.stack(torch.meshgrid(vertical_angles_rad, horizontal_angles_rad, indexing='ij'), dim=-1)
            ray_directions = torch.stack([
            torch.cos(ray_dir_grid[..., 0]) * torch.cos(ray_dir_grid[..., 1]),
            torch.cos(ray_dir_grid[..., 0]) * torch.sin(ray_dir_grid[..., 1]),
            torch.sin(ray_dir_grid[..., 0])
        ], dim=-1).reshape(-1, 3).to(device)


            lidar_origin = torch.tensor([0, 0, self.lidar_height], device=device, dtype=point_cloud_tensor.dtype)

            vector_to_points = point_cloud_tensor - lidar_origin
            ranges = torch.norm(vector_to_points, dim=1)
            max_range_mask = ranges <= 100.0
            filtered_points = point_cloud_tensor[max_range_mask]
            filtered_labels = labels[idx][max_range_mask]
            vector_to_points = vector_to_points[max_range_mask]

            point_azimuths = torch.atan2(vector_to_points[:, 1], vector_to_points[:, 0]) % (2 * torch.pi)
            point_elevations = torch.asin(vector_to_points[:, 2] / torch.norm(vector_to_points, dim=1))

            azimuth_bins = torch.bucketize(point_azimuths, horizontal_angles_rad)
            elevation_bins = torch.bucketize(point_elevations, vertical_angles_rad)
            azimuth_bins = torch.clamp(azimuth_bins, max=self.horizontal_resolution - 1)
            elevation_bins = torch.clamp(elevation_bins, max=self.beams - 1)
            bin_indices = elevation_bins * self.horizontal_resolution + azimuth_bins

            # 这里不会发生 400K×200K 的广播，因为对于每个点只取了其对应的光线
            ray_dir_points = ray_directions[bin_indices]  # shape=(num_filtered_points, 3)
            cross_prod = torch.cross(vector_to_points, ray_dir_points, dim=1)
            distances = torch.norm(cross_prod, dim=1)
            
            # --- 9. 对 bin 进行排序归约，消除对每个 bin 的显式 for 循环 ---
            sorted_bins, sort_indices = torch.sort(bin_indices)
            sorted_distances = distances[sort_indices]
            sorted_filtered_points = filtered_points[sort_indices]
            sorted_filtered_labels = filtered_labels[sort_indices]
            
            # 由 sorted_bins 计算每个 segment（即连续相同 bin）的边界
            unique_bins, bin_counts = torch.unique_consecutive(sorted_bins, return_counts=True)
            bin_offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(bin_counts, dim=0)])
            
            # 利用 torch.split 对 sorted_distances 按每个 segment 分割（每个 segment内点数较少）
            segments = torch.split(sorted_distances, bin_counts.tolist())
            segment_min_list = []
            segment_argmin_list = []
            for seg in segments:
                # 每个 seg 为一个 1D tensor
                min_val, argmin_val = torch.min(seg, dim=0)
                segment_min_list.append(min_val)
                segment_argmin_list.append(argmin_val)
            segment_min = torch.stack(segment_min_list)  # 每个 segment 的最小值
            
            # 计算每个 segment 中最小值对应在 sorted_distances 全局位置的索引
            sampled_sorted_indices = []
            for i, local_argmin in enumerate(segment_argmin_list):
                global_idx = bin_offsets[i].item() + local_argmin.item()
                sampled_sorted_indices.append(global_idx)
            
            # --- 10. 仅保留每个 segment 中最小距离 <= 0.05 的点作为采样 ---
            valid_indices = []
            for i, global_idx in enumerate(sampled_sorted_indices):
                if segment_min[i] <= 0.05:
                    valid_indices.append(global_idx)
            if valid_indices:
                valid_indices_tensor = torch.tensor(valid_indices, device=device)
                sampled_points = points[idx][valid_indices_tensor]
                sampled_labels = sorted_filtered_labels[valid_indices_tensor]
            else:
                sampled_points = torch.empty((0, 3), device=device)
                sampled_labels = torch.empty((0,), device=device, dtype=labels[idx].dtype)
            
            results_points.append(sampled_points)
            results_labels.append(sampled_labels)
            
            # save_tensor_as_ply(points[idx], '/home/eason/workspace_perception/UniLiDAR/temp/sampled_points_check.ply')
            
        return results_points, results_labels
    
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