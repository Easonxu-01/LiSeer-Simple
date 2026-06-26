'''
Author: EASON XU
Date: 2024-12-23 01:24:22
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2026-06-11 22:10:29
Description: 头部注释
FilePath: /UniLiDAR/projects/configs/pointocc/pointtpv_waymo_seg_random.py
'''

_base_ = [
    './_base_/default_runtime.py',
]

dataset_type = 'CustomWaymoDataset_T'
data_root = 'data/waymo/kitti_format/'
file_client_args = dict(backend='disk')

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

class_names = [
    'TYPE_UNDEFINED',
    'TYPE_CAR',
    'TYPE_TRUCK',
    'TYPE_BUS',
    'TYPE_OTHER_VEHICLE', # Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction vehicles, RV, limo, tram).
    'TYPE_MOTORCYCLIST',
    'TYPE_BICYCLIST',
    'TYPE_PEDESTRIAN',
    'TYPE_SIGN',
    'TYPE_TRAFFIC_LIGHT',
    'TYPE_POLE', # Lamp post, traffic sign pole etc.
    'TYPE_CONSTRUCTION_CONE', # Construction cone/pole.
    'TYPE_BICYCLE',
    'TYPE_MOTORCYCLE',
    'TYPE_BUILDING',
    'TYPE_VEGETATION', # Bushes, tree branches, tall grasses, flowers etc.
    'TYPE_TREE_TRUNK',
    'TYPE_CURB', # Curb on the edge of roads. This does not include road boundaries if there's no curb.
    'TYPE_ROAD', # Surface a vehicle could drive on. This include the driveway connecting parking lot and road over a section of sidewalk.
    'TYPE_LANE_MARKER', # Marking on the road that's specifically for defining lanes such as single/double white/yellow lines.
    'TYPE_OTHER_GROUND', # Marking on the road other than lane markers, bumps, cateyes, railtracks etc.
    'TYPE_WALKABLE', # Most horizontal surface that's not drivable, e.g. grassy hill, pedestrian walkway stairs etc.
    'TYPE_SIDEWALK', # Nicely paved walkable surface when pedestrians most likely to walk on.
]


labels_map = {
    0: 0,  # TYPE_UNDEFINED
    1: 1,  # TYPE_CAR
    2: 2,  # TYPE_TRUCK
    3: 3,  # TYPE_BUS
    4: 4,  # TYPE_OTHER_VEHICLE, Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction vehicles, RV, limo, tram).
    5: 5,  # TYPE_MOTORCYCLIST
    6: 6,  # TYPE_BICYCLIST
    7: 7,  # TYPE_PEDESTRIAN
    8: 8,  # TYPE_SIGN
    9: 9,  # TYPE_TRAFFIC_LIGHT
    10: 10, # TYPE_POLE, Lamp post, traffic sign pole etc.
    11: 11, # TYPE_CONSTRUCTION_CONE, Construction cone/pole.
    12: 12, # TYPE_BICYCLE
    13: 13, # TYPE_MOTORCYCLE
    14: 14, # TYPE_BUILDING
    15: 15, # TYPE_VEGETATION, Bushes, tree branches, tall grasses, flowers etc.
    16: 16, # TYPE_TREE_TRUNK
    17: 17, # TYPE_CURB, Curb on the edge of roads. This does not include road boundaries if there's no curb.
    18: 18, # TYPE_ROAD, Surface a vehicle could drive on. This include the driveway connecting parking lot and road over a section of sidewalk.
    19: 19, # TYPE_LANE_MARKER, Marking on the road that's specifically for defining lanes such as single/double white/yellow lines.
    20: 20, # TYPE_OTHER_GROUND, Marking on the road other than lane markers, bumps, cateyes, railtracks etc.
    21: 21, # TYPE_WALKABLE, Most horizontal surface that's not drivable, e.g. grassy hill, pedestrian walkway stairs etc.
    22: 22, # TYPE_SIDEWALK, Nicely paved walkable surface when pedestrians most likely to walk on.
}

color_map = { #rgb value
    0: [32, 119, 181],
    1: [228, 119, 194],
    2: [43, 160, 43],
    3: [220, 220, 141],
    4: [197, 176, 213],
    5: [209, 255, 6],
    6: [248, 182, 210],
    7: [152,224, 137],
    8: [29, 190, 208],
    9: [21, 255, 92],
    10: [174, 199, 232],
    11: [172,127,127],
    12: [215, 39, 40],
    13: [12, 116, 255],
    14: [140, 86, 74],
    15: [255, 127, 25],
    16: [200, 200, 200],
    17: [255, 152, 149],
    18: [158, 218, 229],
    19: [196, 156, 148],
    20: [255, 187, 120],
    21: [188, 190, 33],
    22: [148, 103, 189],
}

CLASS_NAMES_DG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature', 'Living Being', 'Movable objects', 'Other Ground']

WAYMO_TO_CL = {
    0: 0,   # TYPE_UNDEFINED -> Other
    1: 3,   # TYPE_CAR -> Vehicle
    2: 3,   # TYPE_TRUCK -> Vehicle
    3: 3,   # TYPE_BUS -> Vehicle
    4: 3,   # TYPE_OTHER_VEHICLE -> Vehicle
    5: 5,   # TYPE_MOTORCYCLIST -> Living Being
    6: 5,   # TYPE_BICYCLIST -> Living Being
    7: 5,   # TYPE_PEDESTRIAN -> Living Being
    8: 6,   # TYPE_SIGN -> Movable objects
    9: 6,   # TYPE_TRAFFIC_LIGHT -> Movable objects
    10: 6,  # TYPE_POLE -> Movable objects
    11: 6,  # TYPE_CONSTRUCTION_CONE -> Movable objects
    12: 6,  # TYPE_BICYCLE -> Movable objects
    13: 6,  # TYPE_MOTORCYCLE -> Movable objects
    14: 2,  # TYPE_BUILDING -> Structure
    15: 4,  # TYPE_VEGETATION -> Nature
    16: 4,  # TYPE_TREE_TRUNK -> Nature
    17: 1,  # TYPE_CURB -> Driveable Ground
    18: 1,  # TYPE_ROAD -> Driveable Ground
    19: 1,  # TYPE_LANE_MARKER -> Driveable Ground
    20: 7,  # TYPE_OTHER_GROUND -> Other Ground
    21: 7,  # TYPE_WALKABLE -> Other Ground
    22: 7,  # TYPE_SIDEWALK -> Other Ground
}

DG = True


#TODO: metainfo
if DG:
    meta_info = dict(
        classes=CLASS_NAMES_DG, seg_label_mapping=WAYMO_TO_CL, max_label=8, voxel_label_mapping=WAYMO_TO_CL)
else:
    meta_info = dict(
        classes=class_names, seg_label_mapping=labels_map, max_label=19, voxel_label_mapping=labels_map)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

Random = True
dense7sparse = True
ResetRandomProb = 0.2
consistency = True
dense_train = False
d2skd = False
sensor_film = True
plugin = True
plugin_dir = "projects/unilidar_plugin/"
img_norm_cfg = None
occ_path = "data/Waymo-Occ"
train_ann_file = data_root + 'waymo_infos_train.pkl'
val_ann_file = data_root + 'waymo_infos_val.pkl'

unilidar = False
cylinder=False
RPR = True
coor_alignment = False
ori_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# point_cloud_range = [0, -25.6, -3.4, 51.2, 25.6, 3.0] if unilidar else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [256, 256, 32] # ground truth
final_occ_size =  [480, 360, 32]# model output
model_empty_idx = 0  # noise 0-->255
empty_idx = 0  # noise 0-->255
num_cls = 8 if DG else 23
visible_mask = True


sensor = dict(
    LiDAR_height=[1.5, 1.85],
    num_of_beams=[32, 64],
    horizontal_angular_resolution=[600, 1800],
    lower_vertical_field_of_view_bound=[-28, -22],
    upper_vertical_field_of_view_bound=[2, 15],
    # LiDAR_height=[1, 2],
    # num_of_beams=[16, 128],
    # horizontal_angular_resolution=[900, 3600],
    # lower_vertical_field_of_view_bound=[-40, -5],
    # upper_vertical_field_of_view_bound=[0, 25],
)

schedule=[(0, 8, {
                'num_of_beams': (32, 64),
                'horizontal_angular_resolution': (900, 2400),
                'lower_vertical_field_of_view_bound': (-35, -20),
                'upper_vertical_field_of_view_bound': (0, 15),
                'LiDAR_height': (1.5, 1.9)
            }),
                      (8, 16, {
                          'num_of_beams': (24, 80),
                          'horizontal_angular_resolution': (600, 3000),
                          'lower_vertical_field_of_view_bound': (-40, -15),
                          'upper_vertical_field_of_view_bound': (0, 20),
                          'LiDAR_height': (1.2, 2)
                      }),
                      (16, 24, {
                          'num_of_beams': (16, 96),
                          'horizontal_angular_resolution': (400, 4000),
                          'lower_vertical_field_of_view_bound': (-45, -10),
                          'upper_vertical_field_of_view_bound': (0, 25),
                          'LiDAR_height': (0.8, 2.2)
                      })]

cascade_ratio = 2 
dataset_flag = 3


cumulative_iters = 1
find_unused_parameters = False
unique_label = [0, 1, 2, 3, 4, 5, 6, 7] if DG else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20,21,22]
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
nbr_class = num_cls

input_sample_policy = {
    "type": "random interval",
    "fix interval": 5,
    "number": 7,
} # only for training
pose_file = occ_path + '/cam_infos.pkl'
val_pose_file = occ_path + '/cam_infos_vali.pkl'

model = dict(
    type='PointTPV_Seg',
    tpv_aggregator=dict(
        type='TPVAggregator_Seg',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        loss_weight=[1,0.2],
        nbr_classes=nbr_class,
        in_dims=_dim_,
        hidden_dims=2*_dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z,
        consistency = consistency,
        dense_train = dense_train,
        d2skd = d2skd,
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
    empty_idx = 0,
    random = Random,
    reset_random_prob = ResetRandomProb,
    sensor = sensor,
    consistency = consistency,
    dense_train = dense_train,
    d2skd = d2skd,
    dynamic_loss_balance=dict(
        enabled=True,
        alpha=0.1,          # GradNorm target strength
        step=0.0002,          # per-step update size
        momentum=0.3,       # smoothing for weight updates
        min_weight=0.1,
        max_weight=2.0,
        ema_beta=0.8,       # EMA for loss trend
        # Task grouping follows fixed order in model:
        # point / voxel / sensor / consistency
        base_weights=dict(
            point=1.2,
            voxel=0.9,
            sensor=1.2,
            consistency=0.4,
        ),

        task_min_weights=dict(
            point=0.9,
            voxel=0.65,
            sensor=0.8,
            consistency=0.15,
        ),
        task_max_weights=dict(
            point=1.8,
            voxel=1.6,
            sensor=3.0,
            consistency=0.8,
        ),
        voxel_guard_enabled=True,
        voxel_guard_tol=0.02,
        voxel_guard_gain=3.0,
    ),
)


bda_aug_conf = dict(
            # rot_lim=(-22.5, 22.5),
            rot_lim=(-0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5)

train_pipeline = [
dict(
    type='LoadPointsFromFile_RPR',
    coord_type='LIDAR',
    load_dim=3 if Random else 6,
    use_dim=3 if Random else 5,
    shift_height=coor_alignment,
    RPR=RPR,
    point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
    dataset_flag=dataset_flag,
    shift_coors=[0, 0, -0.2],
    Random = Random,
    dense7sparse = dense7sparse if Random else False),
dict(
    type='LoadVoxels',
    to_float32=True, 
    use_semantic=True, 
    cylinder=cylinder, 
    occ_path=occ_path, 
    grid_size=occ_size, 
    use_vel=False,
    unoccupied=empty_idx, 
    pc_range = ori_point_cloud_range,
    RPR = RPR,
    restrict_pc_range = point_cloud_range,
    cal_visible=visible_mask,
    random=Random,
    file_client_args=dict(backend='disk')),
dict(type='VoxelClassMapping'),
dict(type='PointsegMapping', 
            grid_size = grid_size,
            grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
            coarse_ratio = coarse_ratio,
            pc_range = ori_point_cloud_range,
            RPR = RPR,
            restrict_pc_range = point_cloud_range,
            fill_label = 0,
            unique_label = unique_label,
            fixed_volume_space = True,
            max_volume_space = [51.2, 3.1415926, 3],
            min_volume_space = [0, -3.1415926, -3.4],
            cal_visible=False,
            ),
dict(type='Collect3Dinput', keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label', 'train_pts_label', 'reset_random']),
]

test_pipeline = [
    dict(
    type='LoadPointsFromFile_RPR',
    coord_type='LIDAR',
    load_dim=6,
    use_dim=3 if Random else 5,
    shift_height=coor_alignment,
    RPR=RPR,
    point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
    dataset_flag=dataset_flag,
    shift_coors=[0, 0, -0.2],),
    dict(
    type='LoadVoxels',
    to_float32=True, 
    use_semantic=True, 
    cylinder=cylinder, 
    occ_path=occ_path, 
    grid_size=occ_size, 
    use_vel=False,
    unoccupied=empty_idx, 
    pc_range = ori_point_cloud_range,
    RPR = RPR,
    restrict_pc_range = point_cloud_range,
    cal_visible=visible_mask,
    file_client_args=dict(backend='disk')),
    dict(type='VoxelClassMapping'),
    dict(type='PointsegMapping', 
            grid_size = grid_size,
            grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
            coarse_ratio = coarse_ratio,
            pc_range = ori_point_cloud_range,
            fill_label = 0,
            unique_label = unique_label,
            fixed_volume_space = True,
            max_volume_space = [50, 3.1415926, 3],
            min_volume_space = [0, -3.1415926, -5],
            cal_visible=visible_mask,
            RPR = RPR,
            restrict_pc_range = point_cloud_range),
    dict(type='Collect3Dinput', keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label', 'train_pts_label', 'dataset_flag'],
            meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
]
    

test_config=dict(
    type=dataset_type,
    occ_root=occ_path,
    data_root=data_root,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    modality=input_modality,
    pose_file=val_pose_file,
    split='training',
    classes=CLASS_NAMES_DG if DG else class_names,
    occ_size=occ_size,
    seg_label_mapping=WAYMO_TO_CL if DG else labels_map,
    pc_range=point_cloud_range,
    filter_empty_gt=True,
    random=False,
)

train_config=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_path,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        modality=input_modality,
        pose_file=pose_file,
        input_sample_policy=input_sample_policy,
        split='training',
        classes=CLASS_NAMES_DG if DG else class_names,
        test_mode=False,
        occ_size=occ_size,
        seg_label_mapping=WAYMO_TO_CL if DG else labels_map,
        pc_range=point_cloud_range,
        filter_empty_gt=True,
        random=Random,
        reset_random_prob=ResetRandomProb,
        repeat=2 if Random else 1,
        dense7sparse = dense7sparse if Random else False,
),

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8, #12
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=5e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=30, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr=3e-4)

runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)

load_from = None
#work_dir = './work_dirs/unilidar_seg_pointseg_waymotrain_randomD_DG_RPR' 
work_dir = './work_dirs/unilidar_seg_pointseggrad_real2waymotrain_sixhardrandomLCur_DG_film_seesaw_reconsistency_RPR' 
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='LiDARCurriculumHook', schedule=schedule),
        dict(
            type='SwanlabLoggerHook',
            init_kwargs=dict(
                project='UniLiDAR_seg',
                name='unilidar_seg_pointseggrad_real2waymotrain_sixhardrandomLCur_DG_film_seesaw_reconsistency_RPR_4*A800',
                job_type='training',
                notes='the-first-test',
                config = {"lr": optimizer.get('lr', 5e-4), "total_batch_size": data.get('samples_per_gpu', 1) * 4, "total_epoch": runner.get('max_epochs', 24), "weight_decay": optimizer.get('weight_decay', 0.01)}
            ))
    ])

checkpoint_config = dict(interval=2)




