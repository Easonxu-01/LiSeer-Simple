'''
Author: EASON XU
Date: 2026-05-14 16:12:27
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2026-05-14 16:12:29
Description: 头部注释
FilePath: /UniLiDAR/projects/configs/pointocc/pointtpv_realdata_infer.py
'''
'''
Author: EASON XU
Date: 2026-05-11 15:17:57
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2026-05-11 16:02:07
Description: 头部注释
FilePath: /UniLiDAR/projects/configs/pointocc/pointtpv_realdata_infer.py
'''
# RealData: bin point clouds listed by txt, inference-only pipeline (dummy point labels).
#
# mmcv 加载本文件时会在独立命名空间中执行：`_base_` 只做字典合并，不会把 base 里的
# Python 变量（如 Random、RPR）注入本文件。因此下面从 pointtpv_semantickitti_seg_random.py
# 拷贝了 infer_pipeline / train_config 所依赖的变量与 train_pipeline，与 base 保持一致。
#
# Usage:
#   python tools/test.py projects/configs/pointocc/pointtpv_realdata_infer.py
# With optional cfg-options: load_from=/path/to.pth REAL_DATA_NAME=Wildcat test_cfg.save_inference_preds=True

_base_ = ['./pointtpv_semantickitti_seg_random.py']

# Checkpoint for inference. Used when the 2nd CLI positional checkpoint is omitted.
load_from = "/home/eason/workspace_perception/UniLiDAR/ckpts/UniLiDAR_sk.pth"

# ----- RealData-only paths (test/val dataset) -----
REAL_DATA_NAME = 'CS55'
REAL_DATA_ROOTS = dict(
    YHS='data/Real_data/YHS',
    Wildcat='data/Real_data/Wildcat',
    CS55='data/Real_data/CS55',
)
real_data_root = REAL_DATA_ROOTS[REAL_DATA_NAME]
list_txt = 'bin/list.txt'
list_format = 'with_ext'
bin_subdir = 'bin'

# ----- Copied from pointtpv_semantickitti_seg_random.py (must exist before infer_pipeline / train_config) -----
dataset_type = 'SemantickittiVoxelDataset'
data_root = 'data/semantickitti/'
file_client_args = dict(backend='disk')

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
    'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
    'trunck', 'terrian', 'pole', 'traffic-sign'
]
labels_map = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7, 32: 8,
    40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17,
    80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}

CLASS_NAMES_DG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature', 'Living Being', 'Movable objects', 'Other Ground']

SEMANTICKITTI_TO_CL = {
    0: 0, 1: 0, 10: 3, 11: 6, 13: 3, 15: 3, 16: 3, 18: 3, 20: 3, 30: 5, 31: 5, 32: 5,
    40: 1, 44: 1, 48: 7, 49: 7, 50: 2, 51: 2, 52: 2, 60: 1, 70: 4, 71: 4, 72: 4, 80: 6, 81: 6,
    99: 6, 252: 3, 253: 5, 254: 5, 255: 5, 256: 3, 257: 3, 258: 3, 259: 3,
}

DG = True

if DG:
    meta_info = dict(
        classes=CLASS_NAMES_DG, seg_label_mapping=SEMANTICKITTI_TO_CL, max_label=8,
        voxel_label_mapping=SEMANTICKITTI_TO_CL)
else:
    meta_info = dict(
        classes=class_names, seg_label_mapping=labels_map, max_label=19, voxel_label_mapping=labels_map)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

Random = True
consistency = True
d2skd = False
dense_train = False
sensor_film = True
plugin = True
plugin_dir = "projects/unilidar_plugin/"
img_norm_cfg = None
occ_path = "./data/semantickitti"
train_ann_file = "./data/semantickitti/semantickitti_infos_train.pkl"
val_ann_file = "./data/semantickitti/semantickitti_infos_val.pkl"

unilidar = False
cylinder = False
RPR = True
coor_alignment = False
ori_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [256, 256, 32]
final_occ_size = [480, 360, 32]
model_empty_idx = 0
empty_idx = 0
num_cls_nu = 8 if DG else 17
num_cls_sk = 8 if DG else 20
visible_mask = True

sample_from_voxel = False
sample_from_img = False

sensor = dict(
    LiDAR_height=[1.5, 1.85],
    num_of_beams=[32, 64],
    horizontal_angular_resolution=[600, 1800],
    lower_vertical_field_of_view_bound=[-28, -22],
    upper_vertical_field_of_view_bound=[2, 15],
)

schedule = [(0, 8, {
    'num_of_beams': (32, 64),
    'horizontal_angular_resolution': (900, 2400),
    'lower_vertical_field_of_view_bound': (-30, -20),
    'upper_vertical_field_of_view_bound': (0, 10),
    'LiDAR_height': (1.7, 1.9)
}),
    (8, 16, {
        'num_of_beams': (24, 72),
        'horizontal_angular_resolution': (600, 3000),
        'lower_vertical_field_of_view_bound': (-35, -15),
        'upper_vertical_field_of_view_bound': (0, 15),
        'LiDAR_height': (1.5, 2)
    }),
    (16, 24, {
        'num_of_beams': (16, 80),
        'horizontal_angular_resolution': (400, 4000),
        'lower_vertical_field_of_view_bound': (-35, -10),
        'upper_vertical_field_of_view_bound': (0, 20),
        'LiDAR_height': (1.2, 2)
    })]

cascade_ratio = 2
dataset_flag = 1 if dataset_type == 'NuscOCCDataset' else 2

cumulative_iters = 1
find_unused_parameters = False
unique_label = [0, 1, 2, 3, 4, 5, 6, 7] if DG else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
track_running_stats = False

_dim_ = 128

tpv_w_ = 240
tpv_h_ = 180
tpv_z_ = 16
scale_w = 2
scale_h = 2
scale_z = 2

grid_size = final_occ_size
grid_size_occ = occ_size
coarse_ratio = cascade_ratio
sweeps_num = 10
nbr_class = num_cls_sk

model = dict(
    type='PointTPV_Seg',
    tpv_aggregator=dict(
        type='TPVAggregator_Seg',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        loss_weight=[1,0.4],
        nbr_classes=nbr_class,
        in_dims=_dim_,
        hidden_dims=2*_dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z,
        consistency = consistency,
        sensor_film = sensor_film,
    ),
    lidar_tokenizer=dict(
        type='CylinderEncoder_Seg',
        grid_size=grid_size,
        in_channels=8 if Random else 10,
        out_channels=256,
        fea_compre=None,
        base_channels=256,
        split=[16,16,16],
        track_running_stats=track_running_stats,
    ),
    lidar_backbone=dict(
        type='Swin',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        in_channels=256,
        patch_size=4,
        strides=[1,2,2,2],
        frozen_stages=-1,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1,2,3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
          type='Pretrained',
          checkpoint='pretrain/swin_tiny_patch4_window7_224.pth'),
    ),
    lidar_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=_dim_,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(
          type='BN2d',
          requires_grad=True,
          track_running_stats=track_running_stats),
        act_cfg=dict(
          type='ReLU',
          inplace=True),
        upsample_cfg=dict(
          mode='bilinear',
          align_corners=False),
    ),
    sensor = sensor,
    consistency = consistency,
)

train_pipeline = [
    dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=3 if Random else 4,
        use_dim=3 if Random else 4,
        shift_height=coor_alignment,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag,
        shift_coors=[0, 0, -0.2],
        Random=Random),
    dict(
        type='LoadVoxels',
        to_float32=True,
        use_semantic=True,
        cylinder=cylinder,
        occ_path=occ_path,
        grid_size=occ_size,
        use_vel=False,
        unoccupied=empty_idx,
        pc_range=point_cloud_range,
        cal_visible=visible_mask,
        file_client_args=dict(backend='disk')),
    dict(type='VoxelClassMapping'),
    dict(
        type='PointsegMapping',
        grid_size=grid_size,
        grid_size_vox=[tpv_w_ * scale_w, tpv_h_ * scale_h, tpv_z_ * scale_z],
        coarse_ratio=coarse_ratio,
        pc_range=ori_point_cloud_range,
        fill_label=0,
        unique_label=unique_label,
        fixed_volume_space=True,
        max_volume_space=[51.2, 3.1415926, 3],
        min_volume_space=[0, -3.1415926, -3.4],
        cal_visible=False,
        RPR=RPR,
        restrict_pc_range=point_cloud_range),
    dict(type='Collect3Dinput', keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label', 'train_pts_label']),
]

train_config = dict(
    type=dataset_type,
    data_root=data_root,
    occ_root=occ_path,
    ann_file=train_ann_file,
    pipeline=train_pipeline,
    modality=input_modality,
    classes=CLASS_NAMES_DG if DG else class_names,
    test_mode=False,
    occ_size=occ_size,
    seg_label_mapping=SEMANTICKITTI_TO_CL if DG else labels_map,
    pc_range=point_cloud_range,
    filter_empty_gt=True,
    random=Random,
    repeat=2 if Random else 1,
)

# ----- RealData infer pipeline (dummy point labels) -----
infer_pipeline = [
    dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=3 if Random else 4,
        shift_height=coor_alignment,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag,
        shift_coors=[0, 0, -0.2],
    ),
    dict(
        type='LoadVoxels',
        to_float32=True,
        use_semantic=True,
        cylinder=cylinder,
        occ_path=occ_path,
        grid_size=occ_size,
        use_vel=False,
        unoccupied=empty_idx,
        pc_range=point_cloud_range,
        cal_visible=visible_mask,
        dummy_pts_semantic=True,
        file_client_args=dict(backend='disk'),
    ),
    dict(type='VoxelClassMapping'),
    dict(
        type='PointsegMapping',
        grid_size=grid_size,
        grid_size_vox=[tpv_w_ * scale_w, tpv_h_ * scale_h, tpv_z_ * scale_z],
        coarse_ratio=coarse_ratio,
        pc_range=ori_point_cloud_range,
        fill_label=0,
        unique_label=unique_label,
        fixed_volume_space=True,
        max_volume_space=[50, 3.1415926, 3],
        min_volume_space=[0, -3.1415926, -5],
        cal_visible=visible_mask,
        RPR=RPR,
        restrict_pc_range=point_cloud_range,
    ),
    dict(
        type='Collect3Dinput',
        keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label', 'train_pts_label', 'dataset_flag'],
        meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token', 'pts_filename'],
    ),
]

test_config = dict(
    type='RealBinSemantickittiVoxelDataset',
    occ_root=real_data_root,
    data_root=real_data_root,
    list_txt=list_txt,
    list_format=list_format,
    bin_subdir=bin_subdir,
    ann_file='./data/semantickitti/semantickitti_infos_val.pkl',
    pipeline=infer_pipeline,
    modality=input_modality,
    classes=CLASS_NAMES_DG if DG else class_names,
    occ_size=occ_size,
    seg_label_mapping=SEMANTICKITTI_TO_CL if DG else labels_map,
    pc_range=point_cloud_range,
    filter_empty_gt=False,
    random=False,
    test_mode=True,
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

test_cfg = dict(
    save_inference_preds=True,
    save_inference_ply=True,
    inference_pred_dir='/data3/Boreas/boreas-objects/pred_sk',
)

work_dir = './work_dirs/realdata_infer'

