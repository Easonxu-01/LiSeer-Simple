'''
LiSeer unified LiDAR semantic-segmentation config.

This is the friendly entry point for training / evaluation. It selects one of the
three datasets and exposes only the four switches that correspond to the paper
components. All four are ON by default. The detailed per-dataset reference
configs (pointtpv_{nusc,semantickitti,waymo}_seg_random.py) are kept alongside
this file for advanced tuning.
'''

# ============================================================
# User-facing switches (the only knobs you usually need)
# ============================================================
DATASET = 'nuscenes'        # 'nuscenes' | 'semantickitti' | 'waymo'

Random_Samplling = True     # Random Sampling : random LiDAR-pattern re-sampling from dense clouds
COLA             = True     # COLA           : coarse cross-dataset label taxonomy (8 shared classes)
CBA_CL           = True     # CBA-CL         : class-balanced adaptive consistency learning (two views)
S_FilM           = True     # S-FiLM         : sensor-conditioned FiLM modulation

# ============================================================
# Internal aliases (do not edit)
# ============================================================
Random          = Random_Samplling
DG              = COLA
consistency     = CBA_CL
sensor_film     = S_FilM
ResetRandomProb = 0.2       # fixed internally
dense_train     = False
d2skd           = False     # cross-density distillation stays in the SemanticKITTI reference config
plugin          = True
plugin_dir      = "projects/unilidar_plugin/"

assert DATASET in ('nuscenes', 'semantickitti', 'waymo'), f"unknown DATASET={DATASET}"

_base_ = {
    'nuscenes':      ['./_base_/default_runtime.py', './_base_/custom_nus-3d.py'],
    'semantickitti': ['./_base_/default_runtime.py', './_base_/semantickitti.py'],
    'waymo':         ['./_base_/default_runtime.py'],
}[DATASET]

# ============================================================
# Shared constants
# ============================================================
file_client_args = dict(backend='disk')
input_modality = dict(
    use_lidar=True, use_camera=False, use_radar=False, use_map=False, use_external=False)

CLASS_NAMES_DG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature',
                  'Living Being', 'Movable objects', 'Other Ground']

unilidar = False
cylinder = False
RPR = True
coor_alignment = False
ori_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
model_empty_idx = 0
empty_idx = 0
visible_mask = True
track_running_stats = False
cumulative_iters = 1
find_unused_parameters = False

_dim_ = 128
tpv_w_ = 240
tpv_h_ = 180
tpv_z_ = 16
scale_w = 2
scale_h = 2
scale_z = 2
cascade_ratio = 2
coarse_ratio = cascade_ratio
sweeps_num = 10

# ============================================================
# Per-dataset constants
# ============================================================
if DATASET == 'nuscenes':
    dataset_type = 'NuscSegDataset'
    data_root = 'data/nuscenes/'
    train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
    val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
    occ_path = None
    use_voxels = False
    class_names = ['noise', 'flat.driveable_surface', 'flat.sidewalk', 'flat.terrain',
                   'flat.other', 'static.manmade', 'static.vegetation', 'static.other',
                   'vehicle.ego']
    labels_map = None
    SEG_LABEL_MAPPING = {
        0: 0, 1: 6, 2: 6, 3: 3, 4: 3, 5: 3, 6: 6, 7: 5, 8: 6, 9: 3, 10: 3,
        11: 1, 12: 7, 13: 7, 14: 4, 15: 2, 16: 4,
    }
    num_cls = 8 if DG else 17
    in_channels = 8 if Random else 10
    loss_weight = [1, 0.2]
    dataset_flag = 1
    load_dim_train = 3 if Random else 5
    load_dim_test = 5
    use_dim = 3 if Random else 5
    repeat = 3 if Random else 1
    occ_size = None
    grid_size = [480, 360, 32]
    samples_per_gpu = 4
    workers_per_gpu = 8
    lr_min = 2e-4
    max_epochs = 24
    sensor = dict(
        LiDAR_height=[1.5, 1.85],
        num_of_beams=[16, 80],
        horizontal_angular_resolution=[600, 1800],
        lower_vertical_field_of_view_bound=[-30, -15],
        upper_vertical_field_of_view_bound=[2, 15],
    )
    schedule = [(0, 8, {'num_of_beams': (32, 64), 'horizontal_angular_resolution': (900, 2400),
                        'lower_vertical_field_of_view_bound': (-35, -20),
                        'upper_vertical_field_of_view_bound': (0, 10), 'LiDAR_height': (1.7, 1.9)}),
                (8, 16, {'num_of_beams': (24, 80), 'horizontal_angular_resolution': (600, 3000),
                         'lower_vertical_field_of_view_bound': (-40, -15),
                         'upper_vertical_field_of_view_bound': (0, 15), 'LiDAR_height': (1.5, 2)}),
                (16, 24, {'num_of_beams': (16, 96), 'horizontal_angular_resolution': (400, 4000),
                          'lower_vertical_field_of_view_bound': (-45, -10),
                          'upper_vertical_field_of_view_bound': (0, 20), 'LiDAR_height': (1.2, 2)})]
    dynamic_loss_balance = dict(
        enabled=True, alpha=0.1, step=0.001, momentum=0.2, min_weight=0.1, max_weight=2.0, ema_beta=0.8,
        base_weights=dict(point=1.0, voxel=0.9, sensor=1.2, consistency=0.4),
        task_min_weights=dict(point=0.9, voxel=0.65, sensor=0.8, consistency=0.15),
        task_max_weights=dict(point=2.5, voxel=1.6, sensor=3.0, consistency=0.8),
        voxel_guard_enabled=True, voxel_guard_tol=0.02, voxel_guard_gain=3.0)

elif DATASET == 'semantickitti':
    dataset_type = 'SemantickittiVoxelDataset'
    data_root = 'data/semantickitti/'
    occ_path = "./data/semantickitti"
    train_ann_file = "./data/semantickitti/semantickitti_infos_train.pkl"
    val_ann_file = "./data/semantickitti/semantickitti_infos_val.pkl"
    use_voxels = True
    class_names = ['unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
                   'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground',
                   'building', 'fence', 'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign']
    labels_map = {
        0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7, 32: 8,
        40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17,
        80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
    }
    SEG_LABEL_MAPPING = {
        0: 0, 1: 0, 10: 3, 11: 6, 13: 3, 15: 3, 16: 3, 18: 3, 20: 3, 30: 5, 31: 5, 32: 5,
        40: 1, 44: 1, 48: 7, 49: 7, 50: 2, 51: 2, 52: 2, 60: 1, 70: 4, 71: 4, 72: 4, 80: 6, 81: 6,
        99: 6, 252: 3, 253: 5, 254: 5, 255: 5, 256: 3, 257: 3, 258: 3, 259: 3,
    }
    num_cls = 8 if DG else 20
    in_channels = 8 if Random else 9
    loss_weight = [1, 0.2]
    dataset_flag = 2
    load_dim_train = 3 if Random else 4
    load_dim_test = 4
    use_dim = 3 if Random else 4
    repeat = 3 if Random else 1
    occ_size = [256, 256, 32]
    grid_size = [480, 360, 32]
    samples_per_gpu = 1
    workers_per_gpu = 1
    lr_min = 2e-4
    max_epochs = 24
    sensor = dict(
        LiDAR_height=[1.5, 1.85],
        num_of_beams=[32, 64],
        horizontal_angular_resolution=[600, 1800],
        lower_vertical_field_of_view_bound=[-28, -22],
        upper_vertical_field_of_view_bound=[2, 15],
    )
    schedule = [(0, 8, {'num_of_beams': (32, 64), 'horizontal_angular_resolution': (900, 2400),
                        'lower_vertical_field_of_view_bound': (-35, -20),
                        'upper_vertical_field_of_view_bound': (0, 10), 'LiDAR_height': (1.7, 1.9)}),
                (8, 16, {'num_of_beams': (24, 80), 'horizontal_angular_resolution': (600, 3000),
                         'lower_vertical_field_of_view_bound': (-40, -15),
                         'upper_vertical_field_of_view_bound': (0, 15), 'LiDAR_height': (1.5, 2)}),
                (16, 24, {'num_of_beams': (16, 96), 'horizontal_angular_resolution': (400, 4000),
                          'lower_vertical_field_of_view_bound': (-45, -10),
                          'upper_vertical_field_of_view_bound': (0, 20), 'LiDAR_height': (1.2, 2)})]
    dynamic_loss_balance = dict(
        enabled=True, alpha=0.1, step=0.003, momentum=0.3, min_weight=0.1, max_weight=2.0, ema_beta=0.8,
        base_weights=dict(point=1.5, voxel=0.9, sensor=1.5, consistency=0.4),
        task_min_weights=dict(point=1.2, voxel=0.65, sensor=1.2, consistency=0.15),
        task_max_weights=dict(point=2.5, voxel=1.6, sensor=2.5, consistency=0.8),
        voxel_guard_enabled=True, voxel_guard_tol=0.02, voxel_guard_gain=3.0)

else:  # waymo
    dataset_type = 'CustomWaymoDataset_T'
    data_root = 'data/waymo/kitti_format/'
    occ_path = "data/Waymo-Occ"
    train_ann_file = data_root + 'waymo_infos_train.pkl'
    val_ann_file = data_root + 'waymo_infos_val.pkl'
    pose_file = occ_path + '/cam_infos.pkl'
    val_pose_file = occ_path + '/cam_infos_vali.pkl'
    input_sample_policy = {"type": "random interval", "fix interval": 5, "number": 7}
    use_voxels = True
    class_names = [
        'TYPE_UNDEFINED', 'TYPE_CAR', 'TYPE_TRUCK', 'TYPE_BUS', 'TYPE_OTHER_VEHICLE',
        'TYPE_MOTORCYCLIST', 'TYPE_BICYCLIST', 'TYPE_PEDESTRIAN', 'TYPE_SIGN', 'TYPE_TRAFFIC_LIGHT',
        'TYPE_POLE', 'TYPE_CONSTRUCTION_CONE', 'TYPE_BICYCLE', 'TYPE_MOTORCYCLE', 'TYPE_BUILDING',
        'TYPE_VEGETATION', 'TYPE_TREE_TRUNK', 'TYPE_CURB', 'TYPE_ROAD', 'TYPE_LANE_MARKER',
        'TYPE_OTHER_GROUND', 'TYPE_WALKABLE', 'TYPE_SIDEWALK']
    labels_map = {i: i for i in range(23)}
    SEG_LABEL_MAPPING = {
        0: 0, 1: 3, 2: 3, 3: 3, 4: 3, 5: 5, 6: 5, 7: 5, 8: 6, 9: 6, 10: 6, 11: 6, 12: 6, 13: 6,
        14: 2, 15: 4, 16: 4, 17: 1, 18: 1, 19: 1, 20: 7, 21: 7, 22: 7,
    }
    num_cls = 8 if DG else 23
    in_channels = 8 if Random else 10
    loss_weight = [1, 0.2]
    dataset_flag = 3
    load_dim_train = 3 if Random else 6
    load_dim_test = 6
    use_dim = 3 if Random else 5
    repeat = 2 if Random else 1
    occ_size = [256, 256, 32]
    grid_size = [480, 360, 32]
    samples_per_gpu = 4
    workers_per_gpu = 8
    lr_min = 3e-4
    max_epochs = 24
    sensor = dict(
        LiDAR_height=[1.5, 1.85],
        num_of_beams=[32, 64],
        horizontal_angular_resolution=[600, 1800],
        lower_vertical_field_of_view_bound=[-28, -22],
        upper_vertical_field_of_view_bound=[2, 15],
    )
    schedule = [(0, 8, {'num_of_beams': (32, 64), 'horizontal_angular_resolution': (900, 2400),
                        'lower_vertical_field_of_view_bound': (-35, -20),
                        'upper_vertical_field_of_view_bound': (0, 15), 'LiDAR_height': (1.5, 1.9)}),
                (8, 16, {'num_of_beams': (24, 80), 'horizontal_angular_resolution': (600, 3000),
                         'lower_vertical_field_of_view_bound': (-40, -15),
                         'upper_vertical_field_of_view_bound': (0, 20), 'LiDAR_height': (1.2, 2)}),
                (16, 24, {'num_of_beams': (16, 96), 'horizontal_angular_resolution': (400, 4000),
                          'lower_vertical_field_of_view_bound': (-45, -10),
                          'upper_vertical_field_of_view_bound': (0, 25), 'LiDAR_height': (0.8, 2.2)})]
    dynamic_loss_balance = dict(
        enabled=True, alpha=0.1, step=0.0002, momentum=0.3, min_weight=0.1, max_weight=2.0, ema_beta=0.8,
        base_weights=dict(point=1.2, voxel=0.9, sensor=1.2, consistency=0.4),
        task_min_weights=dict(point=0.9, voxel=0.65, sensor=0.8, consistency=0.15),
        task_max_weights=dict(point=1.8, voxel=1.6, sensor=3.0, consistency=0.8),
        voxel_guard_enabled=True, voxel_guard_tol=0.02, voxel_guard_gain=3.0)

nbr_class = num_cls
unique_label = [0, 1, 2, 3, 4, 5, 6, 7] if DG else list(range(1, num_cls + 1))
seg_label_mapping = SEG_LABEL_MAPPING if DG else labels_map
classes = CLASS_NAMES_DG if DG else class_names

# ============================================================
# Model
# ============================================================
model = dict(
    type='PointTPV_Seg',
    tpv_aggregator=dict(
        type='TPVAggregator_Seg',
        tpv_h=tpv_h_, tpv_w=tpv_w_, tpv_z=tpv_z_,
        loss_weight=loss_weight,
        nbr_classes=nbr_class,
        in_dims=_dim_, hidden_dims=2 * _dim_, out_dims=_dim_,
        scale_h=scale_h, scale_w=scale_w, scale_z=scale_z,
        consistency=consistency,
        dense_train=dense_train,
        d2skd=d2skd,
        outside_d2skd=d2skd,
        sensor_film=sensor_film,
    ),
    lidar_tokenizer=dict(
        type='CylinderEncoder_Seg',
        grid_size=grid_size,
        in_channels=in_channels,
        out_channels=256, fea_compre=None, base_channels=256,
        split=[16, 16, 16], track_running_stats=track_running_stats,
    ),
    lidar_backbone=dict(
        type='Swin', embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4, in_channels=256, patch_size=4, strides=[1, 2, 2, 2],
        frozen_stages=-1, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.2, patch_norm=True, out_indices=[1, 2, 3], with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/swin_tiny_patch4_window7_224.pth'),
    ),
    lidar_neck=dict(
        type='GeneralizedLSSFPN', in_channels=[192, 384, 768], out_channels=_dim_,
        start_level=0, num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True, track_running_stats=track_running_stats),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False),
    ),
    empty_idx=empty_idx,
    random=Random,
    reset_random_prob=ResetRandomProb,
    sensor=sensor,
    consistency=consistency,
    d2skd=d2skd,
    dense_train=dense_train,
    dynamic_loss_balance=dynamic_loss_balance,
)

# ============================================================
# Pipelines
# ============================================================
_load_train = dict(
    type='LoadPointsFromFile_RPR', coord_type='LIDAR',
    load_dim=load_dim_train, use_dim=use_dim,
    shift_height=coor_alignment, RPR=RPR,
    point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
    dataset_flag=dataset_flag, shift_coors=[0, 0, -0.2],
    Random=Random, dense7sparse=Random)
_load_test = dict(
    type='LoadPointsFromFile_RPR', coord_type='LIDAR',
    load_dim=load_dim_test, use_dim=use_dim,
    shift_height=coor_alignment, RPR=RPR,
    point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
    dataset_flag=dataset_flag, shift_coors=[0, 0, -0.2], Random=False)

if use_voxels:
    if DATASET == 'waymo':
        _voxels_train = dict(
            type='LoadVoxels', to_float32=True, use_semantic=True, cylinder=cylinder,
            occ_path=occ_path, grid_size=occ_size, use_vel=False, unoccupied=empty_idx,
            pc_range=ori_point_cloud_range, RPR=RPR, restrict_pc_range=point_cloud_range,
            cal_visible=visible_mask, random=Random, file_client_args=dict(backend='disk'))
        _voxels_test = dict(
            type='LoadVoxels', to_float32=True, use_semantic=True, cylinder=cylinder,
            occ_path=occ_path, grid_size=occ_size, use_vel=False, unoccupied=empty_idx,
            pc_range=ori_point_cloud_range, RPR=RPR, restrict_pc_range=point_cloud_range,
            cal_visible=visible_mask, file_client_args=dict(backend='disk'))
    else:
        _voxels_train = dict(
            type='LoadVoxels', to_float32=True, use_semantic=True, cylinder=cylinder,
            occ_path=occ_path, grid_size=occ_size, use_vel=False, unoccupied=empty_idx,
            pc_range=point_cloud_range, cal_visible=visible_mask, file_client_args=dict(backend='disk'))
        _voxels_test = _voxels_train

_segmap_train = dict(
    type='PointsegMapping', grid_size=grid_size,
    grid_size_vox=[tpv_w_ * scale_w, tpv_h_ * scale_h, tpv_z_ * scale_z],
    coarse_ratio=coarse_ratio, pc_range=ori_point_cloud_range,
    fill_label=0, unique_label=unique_label, fixed_volume_space=True,
    max_volume_space=[51.2, 3.1415926, 3], min_volume_space=[0, -3.1415926, -3.4],
    cal_visible=False, RPR=RPR, restrict_pc_range=point_cloud_range)
_segmap_test = dict(
    type='PointsegMapping', grid_size=grid_size,
    grid_size_vox=[tpv_w_ * scale_w, tpv_h_ * scale_h, tpv_z_ * scale_z],
    coarse_ratio=coarse_ratio, pc_range=ori_point_cloud_range,
    fill_label=0, unique_label=unique_label, fixed_volume_space=True,
    max_volume_space=[50, 3.1415926, 3], min_volume_space=[0, -3.1415926, -5],
    cal_visible=visible_mask, RPR=RPR, restrict_pc_range=point_cloud_range)

if use_voxels:
    train_pipeline = [_load_train, _voxels_train, dict(type='VoxelClassMapping'), _segmap_train,
                      dict(type='Collect3Dinput',
                           keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label',
                                 'train_pts_label', 'reset_random'])]
    test_pipeline = [_load_test, _voxels_test, dict(type='VoxelClassMapping'), _segmap_test,
                     dict(type='Collect3Dinput',
                          keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label',
                                'train_pts_label', 'dataset_flag'],
                          meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token'])]
else:
    train_pipeline = [_load_train, _segmap_train, dict(type='VoxelClassMapping'),
                      dict(type='Collect3Dinput',
                           keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label',
                                 'train_pts_label', 'reset_random'])]
    test_pipeline = [_load_test, _segmap_test, dict(type='VoxelClassMapping'),
                     dict(type='Collect3Dinput',
                          keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label',
                                'train_pts_label', 'dataset_flag'],
                          meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token'])]

# ============================================================
# Data
# ============================================================
train_config = dict(
    type=dataset_type, data_root=data_root, ann_file=train_ann_file,
    pipeline=train_pipeline, classes=classes, modality=input_modality,
    test_mode=False, pc_range=point_cloud_range,
    seg_label_mapping=seg_label_mapping,
    random=Random, reset_random_prob=ResetRandomProb, repeat=repeat,
    dense7sparse=Random)
test_config = dict(
    type=dataset_type, data_root=data_root, ann_file=val_ann_file,
    pipeline=test_pipeline, classes=classes, modality=input_modality,
    pc_range=point_cloud_range, seg_label_mapping=seg_label_mapping, random=False)

if DATASET == 'nuscenes':
    train_config.update(dict(use_valid_flag=True, box_type_3d='LiDAR'))
elif DATASET == 'semantickitti':
    train_config.update(dict(occ_root=occ_path, occ_size=occ_size, filter_empty_gt=True))
    test_config.update(dict(occ_root=occ_path, occ_size=occ_size, filter_empty_gt=True))
else:  # waymo
    train_config.update(dict(occ_root=occ_path, occ_size=occ_size, filter_empty_gt=True,
                             pose_file=pose_file, input_sample_policy=input_sample_policy,
                             split='training'))
    test_config.update(dict(occ_root=occ_path, occ_size=occ_size, filter_empty_gt=True,
                            pose_file=val_pose_file, split='training'))

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# ============================================================
# Optimizer / schedule / runtime
# ============================================================
optimizer = dict(
    type='AdamW', lr=5e-4,
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=30, norm_type=2))
lr_config = dict(policy='CosineAnnealing', warmup='linear',
                 warmup_iters=500, warmup_ratio=1.0 / 3, min_lr=lr_min)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
evaluation = dict(interval=1, pipeline=test_pipeline, save_best='SSC_mean', rule='greater')
checkpoint_config = dict(interval=2)

work_dir = './work_dirs/liseer_seg_' + DATASET

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='LiDARCurriculumHook', schedule=schedule),
    ])
