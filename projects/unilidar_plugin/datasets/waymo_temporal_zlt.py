'''
Author: EASON XU
Date: 2025-06-09 09:10:29
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-08-19 02:19:37
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/datasets/waymo_temporal_zlt.py
'''
import re
import random
import pickle
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets import DATASETS
from projects.unilidar_plugin.utils.formating import cm_to_ious, format_SC_results, format_SSCOcc_results_waymo, format_SSCSeg_results_waymo, format_SSC_results_dg
import torch.utils.data
from projects.unilidar_plugin.datasets.zltwaymo import CustomWaymoDataset

@DATASETS.register_module()
class CustomWaymoDataset_T(CustomWaymoDataset):

    CLASSES = ('Car', 'Pedestrian', 'Sign', 'Cyclist')

    def __init__(self,
                 *args,
                 load_interval=1,
                 history_len=1, 
                 input_sample_policy=None,
                 skip_len=0,
                 withimage=False,
                 pose_file=None,
                 offset=0,
                 use_streaming=False,
                 seg_label_mapping=None,
                 random=False,
                 reset_random_prob=0.1,
                 dense7sparse=False,
                 **kwargs):
        with open(pose_file, 'rb') as f:
            pose_all = pickle.load(f)
            self.pose_all = pose_all
        self.length_waymo = sum([len(scene) for k, scene in pose_all.items()])
        self.history_len = history_len
        self.input_sample_policy = input_sample_policy
        self.skip_len = skip_len
        self.withimage = withimage
        self.seg_label_mapping = seg_label_mapping
        self.load_interval_waymo = load_interval
        self.length = self.length_waymo
        self.offset = offset
        self.evaluation_kwargs = kwargs
        self.use_streaming = use_streaming
        self.random = random
        self.reset_random_prob = float(reset_random_prob)
        self.dense7sparse = bool(dense7sparse) and bool(random)
        super().__init__(*args, **kwargs)
        if len(self.data_infos) != self.length_waymo:
            self.length_waymo = len(self.data_infos)
            self.length = self.length_waymo
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            print(f"[Main process] Dataset length: {len(self)}")
        else:
            print(f"[Worker {worker_info.id}] Dataset length: {len(self)}")


    def __len__(self):
        # if not hasattr(self, 'data_infos') or self.data_infos is None:
        #     return 0
        # return len(self.data_infos) // self.load_interval_waymo
        if len(self.data_infos) // self.load_interval_waymo >= 23691:
            return 23691
        else:
            return len(self.data_infos) // self.load_interval_waymo

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        if self.use_streaming:
            return self.prepare_streaming_train_data(idx)
        
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_data(self, index):
        '''
        prepare data for training
        Args:
            index (Int): the index of the data
        Returns:
            data (Dict): the data dict for training
        '''

        # Step 1: get the index list of the history data
        index *= self.load_interval_waymo
        if self.history_len == 1:
            idx_list = [index]
        else:
            queue_start_index = index - self.history_len
            idx_list = list(range(queue_start_index, index))
            random.shuffle(idx_list)
            idx_list = sorted(idx_list[1:]) # drop one frame to add some randomness
            idx_list.append(index)
            
        # Step 2: sample the index list
        i_list = self.get_input_idx(idx_list)

        # Step 3: get the data info according to the index list
        data_queue = []
        for i in i_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None: 
                return None

            # Step 4: prepare the data by dataloader pipeline
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            data_queue.append(example)

        # Step 5: union the data_queue into one single sample
        if self.filter_empty_gt and (data_queue[0] is None):
            return None
        if self.withimage:
            return self.union2one(data_queue)
        else:
            return data_queue[-1]

    def prepare_test_data(self, index):
        index += self.offset
        input_dict = self.get_data_info(index)

        if input_dict is None:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
    
    def get_input_idx(self, idx_list):
        '''
        sample the input index list
        Args:
            idx_list (List[int]): the index list from `index - self.history_len` to `index`. 
                                  It contains current frame index, but it dropped another random frame index to add randomness. 
                                  So the length is `self.history_len`.
        Returns:
            sampled_idx_list (List[int]): the index list after sampling
        '''

        if self.input_sample_policy['type'] == 'normal':
            return idx_list
        
        elif self.input_sample_policy['type'] == 'large interval':
            sampled_idx_list = []
            for i in range(0, self.input_sample_policy['number']):
                sampled_idx = max(0, self.history_len - 1 - i * self.input_sample_policy['interval'])
                sampled_idx_list.append(idx_list[sampled_idx])
            return sorted(sampled_idx_list)
        
        elif self.input_sample_policy['type'] == 'random interval':
            fix_interval = self.input_sample_policy['fix interval']
            slow_interval = random.randint(0, fix_interval-1)
            random_interval = random.choice([fix_interval, slow_interval])

            sampled_idx_list = []
            for i in range(0, self.input_sample_policy['number']):
                sampled_idx = max(self.history_len - 1 - i * random_interval, 0)
                sampled_idx_list.append(idx_list[sampled_idx])
                
            return sorted(sampled_idx_list)
        
        else:
            raise NotImplementedError('not implemented input_sample_policy type')

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        Args: 
            queue (List[Dict]): the sample queue
        Returns:
            queue (Dict): the single sample
        """
        
        # Step 1: 1. union the `img` tensor into a single tensor. 
        # 2. union the `img_metas` dict into a dict[dict]
        # 3. add prev_bev_exists and scene_token
        prev_scene_token=None
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['sample_idx']//1000 != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['sample_idx'] // 1000
                metas_map[i]['scene_token']= prev_scene_token

            else:
                metas_map[i]['scene_token'] = prev_scene_token
                metas_map[i]['prev_bev_exists'] = True

        # Step 2: pack them together
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        return queue


    def get_data_info(self, index):
        '''
        get the data info according to the index. Most of them are image meta data. 
        Args: 
            index (Int): the index of the data.
        Returns:
            input dict (Dict): the data info dict.
        '''

        # Step 1: get the data info
        info = self.data_infos_full[index]
        
        # Step 2: get the image file name and idx
        if isinstance(info, dict) and info.get('sample_idx') is not None:
            sample_idx = info['sample_idx']
            scene_idx = sample_idx % 1000000 // 1000
            frame_idx = sample_idx % 1000000 % 1000
            # img_filename = os.path.join(self.data_root, info['image']['image_path'])
        else:
            match = re.search(r'(\d+)\.bin$', info['point_cloud']['velodyne_path'])
            sample_idx = int(match.group(1))
            scene_idx = sample_idx % 1000000 // 1000
            frame_idx = sample_idx % 1000000 % 1000
            # img_filename = os.path.join(self.data_root, info['image']['image_path'])

        # # Step 3: get the `lidar2img` (why here it get the lidar2img and in the following code it get another lidar2img)
        # rect = info['calib']['R0_rect'].astype(np.float32)
        # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        # P0 = info['calib']['P0'].astype(np.float32)
        # lidar2img = P0 @ rect @ Trv2c

        # the Tr_velo_to_cam is computed for all images but not saved in .info for img1-4
        # the size of img0-2: 1280x1920; img3-4: 886x1920. Attention

        # Step 4: get the image paths, lidar2img, intrinsics, sensor2ego for each image
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics_rts = []
            sensor2ego_rts = []

            for idx_img in range(self.num_views):
                pose = self.pose_all[scene_idx][frame_idx][idx_img]

                intrinsics = pose['intrinsics'] # sensor2img
                sensor2ego = pose['sensor2ego']
                lidar2img = intrinsics @ np.linalg.inv(sensor2ego)
                ego2global = pose['ego2global']
                
                # Attention! (this code means the pose info dismatch the image data file)
                if idx_img == 2: 
                    image_paths.append(img_filename.replace('image_0', f'image_3'))
                elif idx_img == 3: 
                    image_paths.append(img_filename.replace('image_0', f'image_2'))
                else:
                    image_paths.append(img_filename.replace('image_0', f'image_{idx_img}'))

                lidar2img_rts.append(lidar2img)
                intrinsics_rts.append(intrinsics)
                sensor2ego_rts.append(sensor2ego)

        # Step 5: get the pts filename by function `_get_pts_filename` in class `CustomWaymoDataset`
        pts_filename = self._get_pts_filename(sample_idx)
        pts_labelname = pts_filename.replace('.bin', '.label')

        # Step 6: pack the data info into a dict
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            pts_semantic_mask_path=pts_labelname,
            seg_label_mapping=self.seg_label_mapping,
            reset_random=False,
            img_prefix=None,
        )
        
        if self.random:
            input_dict['reset_random'] = bool(np.random.rand() < self.reset_random_prob)
            sparse_pts = input_dict['pts_filename']
            sparse_labels = input_dict['pts_semantic_mask_path']
            dense_pts = (
                sparse_pts.replace('velodyne', 'refine', 1)
                .replace('.bin', '.ply'))
            dense_labels = sparse_labels.replace('velodyne', 'dense_labels', 1)
            if self.dense7sparse:
                input_dict['dense_pts_filename'] = dense_pts
                input_dict['dense_pts_semantic_mask_path'] = dense_labels
                input_dict['dense7sparse'] = True
            elif not input_dict['reset_random']:
                input_dict['pts_filename'] = dense_pts
                input_dict['pts_semantic_mask_path'] = dense_labels

        if self.modality['use_camera']:
            input_dict['img_filename'] = image_paths
            input_dict['lidar2img'] = lidar2img_rts
            input_dict['cam_intrinsic'] = intrinsics_rts
            input_dict['sensor2ego'] = sensor2ego_rts
            ego2global = self.pose_all[scene_idx][frame_idx][0]['ego2global']
            input_dict['ego2global'] = ego2global
            input_dict['global_to_curr_lidar_rt'] = np.linalg.inv(pose['ego2global'])

        # Step 7: get the annos info
        # annos = self.get_ann_info(index)
        # input_dict['ann_info'] = annos

        # Step 8: get the can_bus info (In `waymo` dataset, we do not have can_bus info)
        can_bus = np.zeros(9)
        input_dict['can_bus'] = can_bus

        return input_dict

    def get_ann_info(self, index):
        '''
        get the annotation info according to the index.
        Args:
            index (Int): the index of the data.
        Returns:
            annos (Dict): the annotation info dict.
        '''

        if self.test_mode == True:
            info = self.data_infos[index]
        else: info = self.data_infos_full[index]
        
        # rect = info['calib']['R0_rect'].astype(np.float32)
        # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        # annos = info['annos']
        # # we need other objects to avoid collision when sample
        # annos = self.remove_dontcare(annos)

        # loc = annos['location']
        # dims = annos['dimensions']
        # rots = annos['rotation_y']
        # gt_names = annos['name']
        # gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
        #                               axis=1).astype(np.float32)

        # gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
        #     self.box_mode_3d, np.linalg.inv(rect @ Trv2c))


        # gt_bboxes = annos['bbox']

        # selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        # gt_bboxes = gt_bboxes[selected].astype('float32')
        # gt_names = gt_names[selected]
        # gt_labels = []
        # for cat in gt_names:
        #     if cat in self.CLASSES:
        #         gt_labels.append(self.CLASSES.index(cat))
        #     else:
        #         gt_labels.append(-1)
        # gt_labels = np.array(gt_labels).astype(np.int64)
        # gt_labels_3d = copy.deepcopy(gt_labels)

        # anns_results = dict(
        #     gt_bboxes_3d=gt_bboxes_3d,
        #     gt_labels_3d=gt_labels_3d,
        #     bboxes=gt_bboxes,
        #     labels=gt_labels,
        #     gt_names=gt_names)
        
        return anns_results

    def evaluate(self, results, logger=None, **kawrgs):
        eval_results = {}
        
        if 'SC_metric_2' in results.keys():
            ''' evaluate SC '''
            evaluation_semantic = sum(results['SC_metric_2'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
            for key, val in res_dic.items():
                eval_results['SC_{}'.format(key)] = val
            if logger is not None:
                logger.info('SC Evaluation')
                logger.info(res_table)
        
        if 'SC_metric' in results.keys():
            ''' evaluate SC '''
            evaluation_semantic = sum(results['SC_metric'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
            for key, val in res_dic.items():
                eval_results['SC_{}'.format(key)] = val
            if logger is not None:
                logger.info('SC Evaluation')
                logger.info(res_table)
        
        if 'SSC_metric_2' in results.keys():
            ''' evaluate SSC '''
            evaluation_semantic = sum(results['SSC_metric_2'])
            ious = cm_to_ious(evaluation_semantic)
            if len(ious) == 23:
                res_table, res_dic = format_SSCSeg_results_waymo(ious, return_dic=True)
            elif len(ious) == 16:
                res_table, res_dic = format_SSCOcc_results_waymo(ious, return_dic=True)
            else:
                res_table, res_dic = format_SSC_results_dg(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC Evaluation')
                logger.info(res_table)
                
        if 'SSC_metric' in results.keys():
            ''' evaluate SSC '''
            evaluation_semantic = sum(results['SSC_metric'])
            ious = cm_to_ious(evaluation_semantic)
            if len(ious) == 23:
                res_table, res_dic = format_SSCSeg_results_waymo(ious, return_dic=True)
            elif len(ious) == 16:
                res_table, res_dic = format_SSCOcc_results_waymo(ious, return_dic=True)
            else:
                res_table, res_dic = format_SSC_results_dg(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC Evaluation')
                logger.info(res_table)

        ''' evaluate SC '''
        if 'SC_metric_fine_2' in results.keys():
            evaluation_semantic = sum(results['SC_metric_fine_2'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
            for key, val in res_dic.items():
                eval_results['SC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SC_fine Evaluation')
                logger.info(res_table)
        
        ''' evaluate SSC_fine '''
        if 'SSC_metric_fine_2' in results.keys():
            evaluation_semantic = sum(results['SSC_metric_fine_2'])
            ious = cm_to_ious(evaluation_semantic)
            if len(ious) == 23:
                res_table, res_dic = format_SSCSeg_results_waymo(ious, return_dic=True)
            elif len(ious) == 16:
                res_table, res_dic = format_SSCOcc_results_waymo(ious, return_dic=True)
            else:
                res_table, res_dic = format_SSC_results_dg(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC_fine Evaluation')
                logger.info(res_table)
                
                ''' evaluate SC '''
        if 'SC_metric_fine' in results.keys():
            evaluation_semantic = sum(results['SC_metric_fine'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
            for key, val in res_dic.items():
                eval_results['SC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SC_fine Evaluation')
                logger.info(res_table)
        
        ''' evaluate SSC_fine '''
        if 'SSC_metric_fine' in results.keys():
            evaluation_semantic = sum(results['SSC_metric_fine'])
            ious = cm_to_ious(evaluation_semantic)
            if len(ious) == 23:
                res_table, res_dic = format_SSCSeg_results_waymo(ious, return_dic=True)
            elif len(ious) == 16:
                res_table, res_dic = format_SSCOcc_results_waymo(ious, return_dic=True)
            else:
                res_table, res_dic = format_SSC_results_dg(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC_fine fine Evaluation')
                logger.info(res_table)
            
        return eval_results
