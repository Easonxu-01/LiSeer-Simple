'''
Author: EASON XU
Date: 2025-06-09 06:35:34
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-11-13 19:49:48
Description: 头部注释
FilePath: /UniLiDAR/projects/configs/cylinder3d/cylinder3d_waymo_seg_random.py
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

CLASS_NAMES_DG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature', 'Living Being', 'Dynamic Object', 'Static Object', 'Other Ground']

WAYMO_TO_CL = {
    0: 0,   # TYPE_UNDEFINED -> Other
    1: 3,   # TYPE_CAR -> Vehicle
    2: 3,   # TYPE_TRUCK -> Vehicle
    3: 3,   # TYPE_BUS -> Vehicle
    4: 3,   # TYPE_OTHER_VEHICLE -> Vehicle
    5: 5,   # TYPE_MOTORCYCLIST -> Living Being
    6: 5,   # TYPE_BICYCLIST -> Living Being
    7: 5,   # TYPE_PEDESTRIAN -> Living Being
    8: 7,   # TYPE_SIGN -> Static Object
    9: 7,   # TYPE_TRAFFIC_LIGHT -> Static Object
    10: 7,  # TYPE_POLE -> Static Object
    11: 6,  # TYPE_CONSTRUCTION_CONE -> Dynamic Object
    12: 6,  # TYPE_BICYCLE -> Vehicle
    13: 6,  # TYPE_MOTORCYCLE -> Vehicle
    14: 2,  # TYPE_BUILDING -> Structure
    15: 4,  # TYPE_VEGETATION -> Nature
    16: 4,  # TYPE_TREE_TRUNK -> Nature
    17: 1,  # TYPE_CURB -> Driveable Ground
    18: 1,  # TYPE_ROAD -> Driveable Ground
    19: 1,  # TYPE_LANE_MARKER -> Driveable Ground
    20: 8,  # TYPE_OTHER_GROUND -> Other Ground
    21: 8,  # TYPE_WALKABLE -> Other Ground
    22: 8,  # TYPE_SIDEWALK -> Other Ground
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
plugin = True
plugin_dir = "projects/unilidar_plugin/"
img_norm_cfg = None
occ_path = "data/Waymo-Occ"
train_ann_file = data_root + '/waymo_infos_train.pkl'
val_ann_file = data_root + '/waymo_infos_val.pkl'

unilidar = False
cylinder=False
RPR = True
coor_alignment = False
ori_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# point_cloud_range = [0, -25.6, -3.4, 51.2, 25.6, 3.0] if unilidar else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [256, 256, 32] # ground truth
final_occ_size =  [480, 360, 32]# model output
model_empty_idx = 0
empty_idx = 0  

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

cascade_ratio = 2 
dataset_flag = 3
num_cls = 9 if DG else 23

cumulative_iters = 1
find_unused_parameters = False
unique_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20,21,22]
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
    type='Cylinder3D',
    voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels= 5 if Random else 7,
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    backbone=dict(
        type='Asymm3DSpconv',
        grid_size=grid_size,
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1)),
    decode_head=dict(
        type='Cylinder3DHead',
        channels=128,
        num_classes=nbr_class,
        ignore_index=empty_idx,
        loss_weight=[1,1],
        dual=unilidar,
    ),
    random = Random,
    sensor = sensor,
    voxel=True,
    voxel_type='cylindrical',
    voxel_layer=dict(
        grid_shape=grid_size,
        point_cloud_range=[0, -3.14159265359, -5, 51.2, 3.14159265359, 3],
        max_num_points=-1,
        max_voxels=-1,
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
    Random = Random),
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
            fill_label = 0,
            unique_label = unique_label,
            fixed_volume_space = True,
            max_volume_space = [51.2, 3.1415926, 3],
            min_volume_space = [0, -3.1415926, -3.4],
            cal_visible=False,
            RPR = RPR,
            restrict_pc_range = point_cloud_range),
dict(type='Collect3Dinput', keys=['points','train_pts_label']),
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
            max_volume_space = [51.2, 3.1415926, 3],
            min_volume_space = [0, -3.1415926, -3.4],
            cal_visible=False,
            RPR = RPR,
            restrict_pc_range = point_cloud_range),
    dict(type='Collect3Dinput', keys=['points','train_pts_label', 'dataset_flag'],
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
        repeat=2 if Random else 1,
),

data = dict(
    samples_per_gpu=1, #8
    workers_per_gpu=4, #8
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
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)

load_from = None
work_dir = './work_dirs/unilidar_seg_cylinder3d_waymotrain_DG_RPR'    
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='SwanlabLoggerHook',
            init_kwargs=dict(
                project='UniLiDAR_seg',
                name='unilidar_seg_cylinder3d_waymotrain_DG_RPR_4*A800',
                job_type='training',
                notes='the-first-test',
                config = {"lr": optimizer.get('lr', 5e-4), "total_batch_size": data.get('samples_per_gpu', 1) * 8, "total_epoch": runner.get('max_epochs', 24), "weight_decay": optimizer.get('weight_decay', 0.01)},
            )
        )
    ])

checkpoint_config = dict(interval=2)




