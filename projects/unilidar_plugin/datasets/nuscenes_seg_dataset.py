'''
Author: EASON XU
Date: 2025-04-27 04:09:56
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-08-20 08:28:49
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/datasets/nuscenes_seg_dataset.py
'''
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import os
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from projects.unilidar_plugin.utils.formating import cm_to_ious, format_SC_results, format_SSC_results, format_SSC_results_dg

@DATASETS.register_module()
class NuscSegDataset(NuScenesDataset):
    def __init__(
            self,
            pc_range,
            seg_label_mapping,
            classes,
            random,
            reset_random_prob=0.1,
            dense7sparse=False,
            repeat=1,
            **kwargs):
        super().__init__(**kwargs)
        self.data_infos = list(sorted(self.data_infos, key=lambda e: e['timestamp']))
        self.data_infos = self.data_infos[::self.load_interval]
        self.pc_range = pc_range
        self.random = random
        self.reset_random_prob = float(reset_random_prob)
        self.dense7sparse = bool(dense7sparse) and bool(random)
        self.seg_label_mapping = seg_label_mapping
        self.classes = classes
        self._set_group_flag()

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
            
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            
            return data

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        # Pass the dataset to the pipeline during training to support mixed
        # data augmentation, such as polarmix and lasermix.
        input_dict['dataset'] = self
        
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)

        if input_dict is None:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_data_info(self, index):

        info = self.data_infos[index]
        
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            # frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            pc_range = np.array(self.pc_range),
            lidar_token=info['lidar_token'],
            lidarseg=info['lidarseg'],
            seg_label_mapping=self.seg_label_mapping,
            classes=self.classes,
            curr=info,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            
            lidar2cam_dic = {}
            
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
                
                lidar2cam_dic[cam_type] = lidar2cam_rt.T

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    lidar2cam_dic=lidar2cam_dic,
                ))
        if self.modality['use_lidar']:
            # FIXME alter lidar path
            input_dict['pts_filename'] = input_dict['pts_filename'].replace('./data/nuscenes/', self.data_root)
            input_dict['lidarseg_labels_filename'] = os.path.join(self.data_root, 'lidarseg/v1.0-trainval',  input_dict['lidarseg'])
            input_dict['reset_random'] = False
            if self.random:
                input_dict['reset_random'] = bool(np.random.rand() < self.reset_random_prob)
                sparse_pts = input_dict['pts_filename']
                sparse_labels = input_dict['lidarseg_labels_filename']
                dense_pts = (
                    sparse_pts.replace('LIDAR_TOP', 'LIDAR_TOP/refine', 1)
                    .replace('.bin', '.ply'))
                dense_labels = sparse_labels.replace('lidarseg', 'lidarseg_refine', 1)
                if self.dense7sparse:
                    input_dict['dense_pts_filename'] = dense_pts
                    input_dict['dense_lidarseg_labels_filename'] = dense_labels
                    input_dict['dense7sparse'] = True
                elif not input_dict['reset_random']:
                    input_dict['pts_filename'] = dense_pts
                    input_dict['lidarseg_labels_filename'] = dense_labels
            for sw in input_dict['sweeps']:
                sw['data_path'] = sw['data_path'].replace('./data/nuscenes/', self.data_root)

        return input_dict

    def evaluate(self, results, logger=None, **kawrgs):
        eval_results = {}
        
        if 'SC_metric_1' in results.keys():
            ''' evaluate SC '''
            evaluation_semantic = sum(results['SC_metric_1'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
            for key, val in res_dic.items():
                eval_results['SC_{}'.format(key)] = val
            if logger is not None:
                logger.info('SC Evaluation: SC_metric_1')
                logger.info(res_table)
        
        if 'SC_metric' in results.keys():
            ''' evaluate SC '''
            evaluation_semantic = sum(results['SC_metric'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
            for key, val in res_dic.items():
                eval_results['SC_{}'.format(key)] = val
            if logger is not None:
                logger.info('SC Evaluation: SC_metric')
                logger.info(res_table)
        
        if 'SSC_metric_1' in results.keys():
            ''' evaluate SSC '''
            evaluation_semantic = sum(results['SSC_metric_1'])
            ious = cm_to_ious(evaluation_semantic)
            if len(ious) == 17:
                res_table, res_dic = format_SSC_results(ious, return_dic=True)
            else:
                res_table, res_dic = format_SSC_results_dg(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC Evaluation: SSC_metric_1')
                logger.info(res_table)
                
        if 'SSC_metric' in results.keys():
            ''' evaluate SSC '''
            evaluation_semantic = sum(results['SSC_metric'])
            ious = cm_to_ious(evaluation_semantic)
            if len(ious) == 17:
                res_table, res_dic = format_SSC_results(ious, return_dic=True)
            else:
                res_table, res_dic = format_SSC_results_dg(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC Evaluation: SSC_metric')
                logger.info(res_table)

        ''' evaluate SC '''
        if 'SC_metric_fine_1' in results.keys():
            evaluation_semantic = sum(results['SC_metric_fine_1'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SC_results(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SC_fine Evaluation: SC_metric_fine_1')
                logger.info(res_table)
        
        ''' evaluate SSC_fine '''
        if 'SSC_metric_fine_1' in results.keys():
            evaluation_semantic = sum(results['SSC_metric_fine_1'])
            ious = cm_to_ious(evaluation_semantic)
            if len(ious) == 17:
                res_table, res_dic = format_SSC_results(ious, return_dic=True)
            else:
                res_table, res_dic = format_SSC_results_dg(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC_fine Evaluation: SSC_metric_fine_1')
                logger.info(res_table)
                
                ''' evaluate SC '''
        if 'SC_metric_fine' in results.keys():
            evaluation_semantic = sum(results['SC_metric_fine'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SC_results(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SC_fine Evaluation: SC_metric_fine')
                logger.info(res_table)
        
        ''' evaluate SSC_fine '''
        if 'SSC_metric_fine' in results.keys():
            evaluation_semantic = sum(results['SSC_metric_fine'])
            ious = cm_to_ious(evaluation_semantic)
            if len(ious) == 17:
                res_table, res_dic = format_SSC_results(ious, return_dic=True)
            else:
                res_table, res_dic = format_SSC_results_dg(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC_fine fine Evaluation: SSC_metric_fine')
                logger.info(res_table)
            
        return eval_results

