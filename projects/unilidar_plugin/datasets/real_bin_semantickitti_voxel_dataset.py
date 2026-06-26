# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import pickle
import tempfile

from mmdet.datasets import DATASETS

from .semantickitti_voxel_dataset import SemantickittiVoxelDataset


@DATASETS.register_module()
class RealBinSemantickittiVoxelDataset(SemantickittiVoxelDataset):
    """SemanticKITTI-style bins listed in a txt file (no semantickitti_infos pkl).

    Expects ``data_root``/``list_txt`` with one filename per line:
    - ``with_ext``: lines like ``000000.bin`` (under ``bin_subdir``).
    - ``stem_only``: lines like ``000000`` -> ``bin/000000.bin``.
    """

    def __init__(
            self,
            occ_size,
            seg_label_mapping,
            classes,
            pc_range,
            occ_root,
            random,
            repeat=1,
            list_txt='bin/list.txt',
            list_format='with_ext',
            bin_subdir='bin',
            **kwargs):
        data_root = kwargs.get('data_root', None)
        if data_root is None:
            raise ValueError('RealBinSemantickittiVoxelDataset requires data_root')

        list_path = osp.join(data_root, list_txt)
        if not osp.isfile(list_path):
            raise FileNotFoundError(f'List file not found: {list_path}')

        data_list = []
        with open(list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if list_format == 'stem_only':
                    fname = line + '.bin' if not line.endswith('.bin') else line
                else:
                    fname = line if line.endswith('.bin') else (line + '.bin')
                rel = osp.join(bin_subdir, fname)
                stem = osp.splitext(osp.basename(fname))[0]
                data_list.append({
                    'sample_id': stem,
                    'lidar_points': {
                        'lidar_path': rel,
                        'num_pts_feats': 4,
                    },
                })

        if not data_list:
            raise ValueError(f'No entries loaded from {list_path}')

        fd, ann_path = tempfile.mkstemp(suffix='_realdata_infos.pkl')
        os.close(fd)
        with open(ann_path, 'wb') as pf:
            pickle.dump({'data_list': data_list}, pf)
        kwargs['ann_file'] = ann_path
        self._temp_ann_file = ann_path

        super().__init__(
            occ_size,
            seg_label_mapping,
            classes,
            pc_range,
            occ_root,
            random,
            repeat=repeat,
            **kwargs)

    def __del__(self):
        p = getattr(self, '_temp_ann_file', None)
        if p and osp.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['sample_id'],
            lidar_points=info['lidar_points'],
            pts_filename=info['lidar_points']['lidar_path'],
            pts_semantic_mask_path='__DUMMY__',
            num_pts_feats=info['lidar_points']['num_pts_feats'],
            seg_label_mapping=self.seg_label_mapping,
            classes=self.classes,
            curr=info,
        )
        if self.modality['use_lidar']:
            input_dict['pts_filename'] = osp.join(self.data_root, input_dict['pts_filename'])
            input_dict['filename'] = input_dict['pts_filename']
            input_dict['voxel_path'] = None
            input_dict['voxel_semantic_mask_path'] = None
            input_dict['voxel_occ_mask_path'] = None
            input_dict['voxel_invalidation'] = None
        input_dict['occ_size'] = self.occ_size
        input_dict['pc_range'] = self.pc_range
        input_dict['scene_token'] = 'realdata'
        input_dict['lidar_token'] = str(info['sample_id'])
        return input_dict
