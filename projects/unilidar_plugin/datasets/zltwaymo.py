'''
Author: EASON XU
Date: 2025-06-09 09:11:06
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-07-22 08:09:16
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/datasets/zltwaymo.py
'''
import mmcv
import numpy as np
import os
import tempfile
import torch
from mmcv.utils import print_log
from os import path as osp
# ERROR ROOT at LINE 331, AT line 236 in format_result, we adjust the worker to be really small
from mmdet3d.datasets import DATASETS #really fucked up for not adding '3d'
from mmdet3d.core.bbox import Box3DMode, points_cam2img
from mmdet3d.datasets.kitti_dataset import KittiDataset
# from .waymo_let_metric import compute_waymo_let_metric

from waymo_open_dataset import dataset_pb2 as open_dataset
import mmcv
import numpy as np
import tensorflow as tf
from glob import glob
from os.path import join
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2


class KITTI2Waymo(object):
    """KITTI predictions to Waymo converter.
    This class serves as the converter to change predictions from KITTI to
    Waymo format.
    Args:
        kitti_result_files (list[dict]): Predictions in KITTI format.
        waymo_tfrecords_dir (str): Directory to load waymo raw data.
        waymo_results_save_dir (str): Directory to save converted predictions
            in waymo format (.bin files).
        waymo_results_final_path (str): Path to save combined
            predictions in waymo format (.bin file), like 'a/b/c.bin'.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (str): Number of parallel processes.
    """

    def __init__(self,
                 kitti_result_files,
                 waymo_tfrecords_dir,
                 waymo_results_save_dir,
                 waymo_results_final_path,
                 prefix,
                 workers=64):

        self.kitti_result_files = kitti_result_files
        self.waymo_tfrecords_dir = waymo_tfrecords_dir
        self.waymo_results_save_dir = waymo_results_save_dir
        self.waymo_results_final_path = waymo_results_final_path
        self.prefix = prefix
        self.workers = int(workers)
        self.name2idx = {}
        for idx, result in enumerate(kitti_result_files):
            if len(result['sample_idx']) > 0:
                self.name2idx[str(result['sample_idx'][0])] = idx

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }

        self.T_ref_to_front_cam = np.array([[0.0, 0.0, 1.0, 0.0],
                                            [-1.0, 0.0, 0.0, 0.0],
                                            [0.0, -1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]])

        self.get_file_names()
        self.create_folder()

    def get_file_names(self):
        """Get file names of waymo raw data."""
        self.waymo_tfrecord_pathnames = sorted(
            glob(join(self.waymo_tfrecords_dir, '*.tfrecord')))
        print(len(self.waymo_tfrecord_pathnames), 'tfrecords found.')

    def create_folder(self):
        """Create folder for data conversion."""
        mmcv.mkdir_or_exist(self.waymo_results_save_dir)

    def parse_objects(self, kitti_result, T_k2w, context_name,
                      frame_timestamp_micros):
        """Parse one prediction with several instances in kitti format and
        convert them to `Object` proto.
        Args:
            kitti_result (dict): Predictions in kitti format.
                - name (np.ndarray): Class labels of predictions.
                - dimensions (np.ndarray): Height, width, length of boxes.
                - location (np.ndarray): Bottom center of boxes (x, y, z).
                - rotation_y (np.ndarray): Orientation of boxes.
                - score (np.ndarray): Scores of predictions.
            T_k2w (np.ndarray): Transformation matrix from kitti to waymo.
            context_name (str): Context name of the frame.
            frame_timestamp_micros (int): Frame timestamp.
        Returns:
            :obj:`Object`: Predictions in waymo dataset Object proto.
        """

        def parse_one_object(instance_idx):
            """Parse one instance in kitti format and convert them to `Object`
            proto.
            Args:
                instance_idx (int): Index of the instance to be converted.
            Returns:
                :obj:`Object`: Predicted instance in waymo dataset \
                    Object proto.
            """
            cls = kitti_result['name'][instance_idx]
            length = round(kitti_result['dimensions'][instance_idx, 0], 4)
            height = round(kitti_result['dimensions'][instance_idx, 1], 4)
            width = round(kitti_result['dimensions'][instance_idx, 2], 4)
            x = round(kitti_result['location'][instance_idx, 0], 4)
            y = round(kitti_result['location'][instance_idx, 1], 4)
            z = round(kitti_result['location'][instance_idx, 2], 4)
            rotation_y = round(kitti_result['rotation_y'][instance_idx], 4)
            score = round(kitti_result['score'][instance_idx], 4)

            # y: downwards; move box origin from bottom center (kitti) to
            # true center (waymo)
            y -= height / 2
            # frame transformation: kitti -> waymo
            x, y, z = self.transform(T_k2w, x, y, z)

            # different conventions
            heading = -(rotation_y + np.pi / 2)
            while heading < -np.pi:
                heading += 2 * np.pi
            while heading > np.pi:
                heading -= 2 * np.pi

            box = label_pb2.Label.Box()
            box.center_x = x
            box.center_y = y
            box.center_z = z
            box.length = length
            box.width = width
            box.height = height
            box.heading = heading

            o = metrics_pb2.Object()
            o.object.box.CopyFrom(box)
            o.object.type = self.k2w_cls_map[cls]
            o.score = score

            o.context_name = context_name
            o.frame_timestamp_micros = frame_timestamp_micros

            return o

        objects = metrics_pb2.Objects()

        for instance_idx in range(len(kitti_result['name'])):
            o = parse_one_object(instance_idx)
            objects.objects.append(o)

        return objects

    def convert_one(self, file_idx):
        """Convert action for single file.
        Args:
            file_idx (int): Index of the file to be converted.
        """
        file_pathname = self.waymo_tfrecord_pathnames[file_idx]
        file_data = tf.data.TFRecordDataset(file_pathname, compression_type='')

        for frame_num, frame_data in enumerate(file_data):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(frame_data.numpy()))
            filename = f'{self.prefix}{file_idx:03d}{frame_num:03d}'

            for camera in frame.context.camera_calibrations:
                # FRONT = 1, see dataset.proto for details
                if camera.name == 1:
                    T_front_cam_to_vehicle = np.array(
                        camera.extrinsic.transform).reshape(4, 4)

            T_k2w = T_front_cam_to_vehicle @ self.T_ref_to_front_cam

            context_name = frame.context.name
            frame_timestamp_micros = frame.timestamp_micros

            if filename in self.name2idx:
                kitti_result = \
                    self.kitti_result_files[self.name2idx[filename]]
                objects = self.parse_objects(kitti_result, T_k2w, context_name,
                                             frame_timestamp_micros)
            else:
                print(filename, 'not found.(bevformer)')
                objects = metrics_pb2.Objects()

            with open(
                    join(self.waymo_results_save_dir, f'{filename}.bin'),
                    'wb') as f:
                f.write(objects.SerializeToString())

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        mmcv.track_parallel_progress(self.convert_one, range(len(self)),
                                     self.workers)
        print('\nFinished ...')

        # combine all files into one .bin
        pathnames = sorted(glob(join(self.waymo_results_save_dir, '*.bin')))
        combined = self.combine(pathnames)

        with open(self.waymo_results_final_path, 'wb') as f:
            f.write(combined.SerializeToString())

    def __len__(self):
        """Length of the filename list."""
        return len(self.waymo_tfrecord_pathnames)

    def transform(self, T, x, y, z):
        """Transform the coordinates with matrix T.
        Args:
            T (np.ndarray): Transformation matrix.
            x(float): Coordinate in x axis.
            y(float): Coordinate in y axis.
            z(float): Coordinate in z axis.
        Returns:
            list: Coordinates after transformation.
        """
        pt_bef = np.array([x, y, z, 1.0]).reshape(4, 1)
        pt_aft = np.matmul(T, pt_bef)
        return pt_aft[:3].flatten().tolist()

    def combine(self, pathnames):
        """Combine predictions in waymo format for each sample together.
        Args:
            pathnames (str): Paths to save predictions.
        Returns:
            :obj:`Objects`: Combined predictions in Objects proto.
        """
        combined = metrics_pb2.Objects()

        for pathname in pathnames:
            objects = metrics_pb2.Objects()
            with open(pathname, 'rb') as f:
                objects.ParseFromString(f.read())
            for o in objects.objects:
                combined.objects.append(o)

        return combined



@DATASETS.register_module()
class CustomWaymoDataset(KittiDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Waymo Dataset.

    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': box in LiDAR coordinates
            - 'Depth': box in depth coordinates, usually for indoor dataset
            - 'Camera': box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [-85, -85, -5, 85, 85, 5].
    """

    CLASSES = ('Car', 'Pedestrian', 'Sign', 'Cyclist')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 num_views=5,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 gt_bin = None,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range,
        )

        self.num_views = num_views
        assert self.num_views <= 5
        # to load a subset, just set the load_interval in the dataset config
        self.load_interval = load_interval
        if isinstance(self.data_infos, dict) and self.data_infos.get('data_list') is not None:
            self.meta_infos = self.data_infos['metainfo']
            # self.data_infos = self.data_infos['data_list'][::load_interval]
            data_infos = self.data_infos['data_list'][::load_interval]
            self.data_infos = []
            # seg_annotated_indices = []
            for idx, info in enumerate(data_infos):
                if info["lidar_points"]["seg_annotated"] == True:
                    self.data_infos.append(info)
        else:
            data_infos = []
            if self.data_infos[0].get("point_cloud", {}).get("velodyne_path", None) is not None:
                annotated_indices_path = 'seg_annotated_indices.txt'
                with open(annotated_indices_path, 'r') as f:
                    annotated_files = set(line.strip() for line in f.readlines())
                for idx, info in enumerate(self.data_infos):
                    filename = os.path.basename(info["point_cloud"]["velodyne_path"])
                    if filename in annotated_files:
                        data_infos.append(info)
                    else:
                        continue
                self.data_infos = data_infos
                print(f"Loaded {len(self.data_infos)} annotated files from seg_annotated_indices.txt")
            else:
                self.data_infos = self.data_infos[::load_interval]
            # seg_annotated_indices.append(info["lidar_points"]["lidar_path"])
        print(f"Loaded {len(self.data_infos)} files from {self.ann_file}")
        # # Save seg_annotated indices to txt file
        # if seg_annotated_indices:
        #     save_path = '/home/eason/workspace_perception/UniLiDAR/seg_annotated_indices.txt'
        #     with open(save_path, 'w') as f:
        #         for idx in seg_annotated_indices:
        #             f.write(f'{idx}\n')
        #     print(f'Saved {len(seg_annotated_indices)} seg_annotated indices to {save_path}')
        self.data_infos_full = self.data_infos
        # Ensure sampler flag length matches current dataset length after any filtering/subsampling
        # This avoids a mismatch that would make the sampler see a tiny dataset and yield very few steps per epoch
        try:
            import numpy as _np
            self.flag = _np.zeros(len(self.data_infos), dtype=_np.uint8)
        except Exception:
            pass
        if test_mode == True:

            if gt_bin != None:
                self.gt_bin = gt_bin
            # elif load_interval==1 and 'val' in ann_file:
            #     self.gt_bin = 'gt.bin'
            # elif load_interval==5 and 'val' in ann_file:
            #     self.gt_bin = 'gt_subset.bin'
            # elif load_interval==20 and 'train' in ann_file:
            #     self.gt_bin = 'gt_train_subset.bin'
            # else:
            #     assert gt_bin == 'wrong'
    def _get_pts_filename(self, idx):
        pts_filename = osp.join(self.root_split, self.pts_prefix, f'{idx:07d}.bin')
        return pts_filename

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Standard input_dict consists of the
                data information.

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str | None): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        # index=475  # in infos_train.pkl is index 485
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        lidar2img = P0 @ rect @ Trv2c

        # the Tr_velo_to_cam is computed for all images but not saved in .info for img1-4
        # the size of img0-2: 1280x1920; img3-4: 886x1920
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []

            # load calibration for all 5 images.
            calib_path = img_filename.replace('image_0', 'calib').replace('.png', '.txt')
            Tr_velo_to_cam_list = []
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            for line_num in range(6, 6 + self.num_views):
                trans = np.array([float(info) for info in lines[line_num].split(' ')[1:13]]).reshape(3, 4)
                trans = np.concatenate([trans, np.array([[0., 0., 0., 1.]])], axis=0).astype(np.float32)
                Tr_velo_to_cam_list.append(trans)
            assert np.allclose(Tr_velo_to_cam_list[0], info['calib']['Tr_velo_to_cam'].astype(np.float32))

            for idx_img in range(self.num_views):
                rect = info['calib']['R0_rect'].astype(np.float32)
                # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
                Trv2c = Tr_velo_to_cam_list[idx_img]
                P0 = info['calib'][f'P{idx_img}'].astype(np.float32)
                lidar2img = P0 @ rect @ Trv2c

                image_paths.append(img_filename.replace('image_0', f'image_{idx_img}'))
                lidar2img_rts.append(lidar2img)

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
        )
        if self.modality['use_camera']:
            input_dict['img_filename'] = image_paths
            input_dict['lidar2img'] = lidar2img_rts
        input_dict['pose'] = info['pose']
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None,
                       data_format='waymo'):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            data_format (str | None): Output data format. Default: 'waymo'.
                Another supported choice is 'kitti'.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        assert ('waymo' in data_format or 'kitti' in data_format), \
            f'invalid data_format {data_format}'
        print("still work before format_results ---  if not isinstance")
        # np.save('debug_eval/zltwaymo_eval_result_before_format_results',outputs)
        # print('saved!')
        # exit(0)
        if (not isinstance(outputs[0], dict)) or 'img_bbox' in outputs[00]:
            raise TypeError('Not supported type for reformat results.')
        elif 'pts_bbox' in outputs[0]:#we go this way
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name #saving path
                if submission_prefix is not None:
                    submission_prefix_ = f'{submission_prefix}_{name}'
                else:
                    submission_prefix_ = None
                result_files_ = self.bbox2result_kitti(results_, self.CLASSES,
                                                       pklfile_prefix_,
                                                       submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES,
                                                  pklfile_prefix,
                                                  submission_prefix)
        # print(result_files)
        # np.save('debug_eval/zltwaymo_eval_result_kitti_format',result_files)## turn into cam-coord, it sucks
        # exit(0)
        # open('zlt_output_kitti_format_debug.txt','w').write(str(result_files))  #we got absolutely right data
        # exit(0)  
        if 'waymo' in data_format:
            waymo_root = osp.join(
                self.data_root.split('kitti_format')[0], 'waymo_format')
            if 'train' in self.ann_file:
                waymo_tfrecords_dir = osp.join(waymo_root, 'training')
                prefix = '0'
            elif self.split == 'training':
                waymo_tfrecords_dir = osp.join(waymo_root, 'validation')
                prefix = '1'
            elif self.split == 'testing':
                waymo_tfrecords_dir = osp.join(waymo_root, 'testing')
                prefix = '2'
            else:
                raise ValueError('Not supported split value.')
            save_tmp_dir = tempfile.TemporaryDirectory()
            waymo_results_save_dir = save_tmp_dir.name
            waymo_results_final_path = f'{pklfile_prefix}.bin'
            print("still work before converter init!!!")
            if 'pts_bbox' in result_files:#result_files deprecated
                converter = KITTI2Waymo(result_files['pts_bbox'],
                                        waymo_tfrecords_dir,
                                        waymo_results_save_dir,
                                        waymo_results_final_path, prefix)
            else:
                converter = KITTI2Waymo(result_files, waymo_tfrecords_dir,
                                        waymo_results_save_dir,
                                        waymo_results_final_path, prefix)
            print("still work before converter convert!!!")
            print(waymo_tfrecords_dir, waymo_results_save_dir, waymo_results_final_path)
            # exit(0)
            converter.convert()         
            print("still work after converter convert!!!")
            save_tmp_dir.cleanup()

        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='waymo',
                 logger=None,

                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 jsonfile_prefix=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default: 'waymo'. Another supported metric is 'kitti'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.

        Returns:
            dict[str: float]: results of each evaluation metric
        """
        print("metric here is-----------{}".format(metric))
        # np.save('debug_eval/zltwaymo_eval_result',results)
        # print('saved!')## result still correct here!
        # exit(0)
        assert ('waymo' in metric or 'kitti' in metric), \
            f'invalid metric {metric}'

        if 'waymo' in metric:
            waymo_root = osp.join(
                self.data_root.split('kitti_format')[0], 'waymo_format')
            if pklfile_prefix is None:
                eval_tmp_dir = tempfile.TemporaryDirectory()
                pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
            else:
                eval_tmp_dir = None
            result_files, tmp_dir = self.format_results(
                results,
                pklfile_prefix,
                submission_prefix,
                data_format='waymo')# xxxxxxx not found inside, maybe it's OK

            import shutil
            shutil.copy(f'{pklfile_prefix}.bin', 'work_dirs/result.bin')
            from time import time
            _ = time()
            ap_dict = None
            print('time usage of compute_let_metric: {} s'.format(time()-_))        
            
            if eval_tmp_dir is not None:
                eval_tmp_dir.cleanup()

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir,pipeline=pipeline)
        return ap_dict

    def just_evaluate(self,
                 metric='waymo',
                 logger=None,

                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 jsonfile_prefix=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default: 'waymo'. Another supported metric is 'kitti'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.

        Returns:
            dict[str: float]: results of each evaluation metric
        """
        print("metric here is-----------{}".format(metric))
        # np.save('debug_eval/zltwaymo_eval_result',results)
        # print('saved!')## result still correct here!
        # exit(0)
        assert ('waymo' in metric or 'kitti' in metric), \
            f'invalid metric {metric}'

        if 'waymo' in metric:

            from time import time
            _ = time()
            ap_dict = None# compute_waymo_let_metric(f'data/waymo/waymo_format/gt.bin', 'work_dirs/result.bin')
            print('time usage of compute_let_metric: {} s'.format(time() - _))

        return ap_dict


    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert results to kitti format for evaluation and test submission.

        Args:
            net_outputs (List[np.ndarray]): list of array storing the
                bbox and score
            class_nanes (List[String]): A list of class names
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            List[dict]: A list of dict have the kitti 3d format
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)
        # np.save('debug_eval/zltwaymo_eval_net_outputs',net_outputs)   # data_size * [output one frame]
        # exit(0)
        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            image_shape = info['image']['image_shape'][:2]
            # if you are going to replace final result.bin with gt boxes, do it here
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            # np.save('debug_eval/zltwaymo_box_dict',box_dict)
            # print(box_dict)
            # exit(0)
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                anno = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(
                        -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

                if submission_prefix is not None:
                    curr_file = f'{submission_prefix}/{sample_idx:07d}.txt'
                    with open(curr_file, 'w') as f:
                        bbox = anno['bbox']
                        loc = anno['location']
                        dims = anno['dimensions']  # lhw -> hwl

                        for idx in range(len(bbox)):
                            print(
                                '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.
                                format(anno['name'][idx], anno['alpha'][idx],
                                       bbox[idx][0], bbox[idx][1],
                                       bbox[idx][2], bbox[idx][3],
                                       dims[idx][1], dims[idx][2],
                                       dims[idx][0], loc[idx][0], loc[idx][1],
                                       loc[idx][2], anno['rotation_y'][idx],
                                       anno['score'][idx]),
                                file=f)
            else:
                annos.append({
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                })
            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the boxes into valid format.

        Args:
            box_dict (dict): Bounding boxes to be converted.

                - boxes_3d (:obj:``LiDARInstance3DBoxes``): 3D bounding boxes.
                - scores_3d (np.ndarray): Scores of predicted boxes.
                - labels_3d (np.ndarray): Class labels of predicted boxes.
            info (dict): Dataset information dictionary.

        Returns:
            dict: Valid boxes after conversion.

                - bbox (np.ndarray): 2D bounding boxes (in camera 0).
                - box3d_camera (np.ndarray): 3D boxes in camera coordinates.
                - box3d_lidar (np.ndarray): 3D boxes in lidar coordinates.
                - scores (np.ndarray): Scores of predicted boxes.
                - label_preds (np.ndarray): Class labels of predicted boxes.
                - sample_idx (np.ndarray): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        # TODO: remove the hack of yaw
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        P0 = box_preds.tensor.new_tensor(P0)    # that is to say, box_2d_pred only projected to cam0 image！

        box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c) #box3d in camera coord

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P0)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        valid_inds = valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )