'''
Author: EASON XU
Date: 2024-12-14 23:59:33
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-08-24 07:03:24
Description: 头部注释
FilePath: /UniLiDAR/projects/configs/baselines/LiDAR_128x128x10_waymo_random.py
'''
_base_ = [
    '../_base_/default_runtime.py'
]

dataset_type = 'CustomWaymoDataset_T'
data_root = 'data/waymo/kitti_format/'
file_client_args = dict(backend='disk')


class_names = [
    'TYPE_GENERALOBJECT',
    'TYPE_VEHICLE',
    'TYPE_PEDESTRIAN',
    'TYPE_SIGN',
    'TYPE_BICYCLIST',
    'TYPE_TRAFFIC_LIGHT',
    'TYPE_POLE',
    'TYPE_CONSTRUCTION_CONE',
    'TYPE_BICYCLE',
    'TYPE_MOTORCYCLE',
    'TYPE_BUILDING',
    'TYPE_VEGETATION',
    'TYPE_TREE_TRUNK',
    'TYPE_ROAD',
    'TYPE_WALKABLE',
    'TYPE_FREE',
]


labels_map = {
    0: 15,  # TYPE_GENERALOBJECT
    1: 1,  # TYPE_VEHICLE
    2: 2,  # TYPE_PEDESTRIAN
    3: 3,  # TYPE_SIGN
    4: 4,  # TYPE_BICYCLIST
    5: 5,  # TYPE_TRAFFIC_LIGHT
    6: 6,  # TYPE_POLE
    7: 7,  # TYPE_CONSTRUCTION_CONE
    8: 8,  # TYPE_BICYCLE
    9: 9,  # TYPE_MOTORCYCLE
    10: 10, # TYPE_BUILDING
    11: 11, # TYPE_VEGETATION
    12: 12, # TYPE_TREE_TRUNK
    13: 13, # TYPE_ROAD
    14: 14, # TYPE_WALKABLE
    15: 0, # TYPE_FREE
}

CLASS_NAMES_DG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature', 'Living Being', 'Dynamic Object', 'Static Object', 'Other Ground']

WAYMO_TO_CL ={
    0: 7,  # GO
    1: 3,  # TYPE_VEHICLE
    2: 5,  # TYPE_PEDESTRIAN
    3: 7,  # TYPE_SIGN
    4: 6,  # TYPE_BICYCLIST
    5: 7,  # TYPE_TRAFFIC_LIGHT
    6: 7,  # TYPE_POLE
    7: 6,  # TYPE_CONSTRUCTION_CONE
    8: 3,  # TYPE_BICYCLE
    9: 3,  # TYPE_MOTORCYCLE
    10: 2, # TYPE_BUILDING
    11: 4, # TYPE_VEGETATION
    12: 4, # TYPE_TREE_TRUNK
    13: 1, # TYPE_ROAD
    14: 8, # TYPE_WALKABLE
    15: 0, # TYPE_FREE
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

plugin = True
plugin_dir = "projects/unilidar_plugin/"
img_norm_cfg = None
occ_path = "data/Waymo-Occ"
train_ann_file = occ_path + '/waymo_infos_train.pkl'
val_ann_file = occ_path + '/waymo_infos_val.pkl'
occ_gt_data_root = occ_path + '/training/'
occ_val_gt_data_root = occ_path + '/validation/'

Random = True
unilidar = False
RPR=True
cylinder=False
coor_alignment = False
# [-72.0,-72.0,-3.4,72.0,72.0,3.0]
ori_point_cloud_range = [-80.0, -80.0, -1.0, 80.0, 80.0, 5.4]
point_cloud_range = [-51.2, -51.2, -1.0, 51.2, 51.2, 3.0]
occ_size = [512, 512, 20] # ground truth
final_occ_size = occ_size
voxel_channels = [80, 160, 320, 640]
empty_idx = 0  
num_cls = 9 if DG else 16 
nbr_class = num_cls
visible_mask = True

sample_from_voxel = False
sample_from_img = False

numC_Trans = 80
init_size = 64
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)

find_unused_parameters = False
#model 
spatial_shape = [256, 256, 32] if cylinder else [1024,1024,40]
cascade_ratio = final_occ_size[0] * 8 // spatial_shape[0] 
voxel_size=[0.05,0.05,0.05]

dataset_flag = 3
# norm_cfg=dict(type='UniNorm3d', dataset_from_flag=dataset_flag, eps=1e-3, momentum=0.01, voxel_coord=True),
sensor = dict(
    LiDAR_height=[1, 2],
    num_of_beams=[32, 64],
    horizontal_angular_resolution=[600, 3600],
    lower_vertical_field_of_view_bound=[-40, -10],
    upper_vertical_field_of_view_bound=[0, 25],
    # LiDAR_height=[1, 1.01],
    # num_of_beams=[31, 33],
    # horizontal_angular_resolution=[900, 3600],
    # lower_vertical_field_of_view_bound=[-15, -14],
    # upper_vertical_field_of_view_bound=[5, 6],
)

input_sample_policy = {
    "type": "random interval",
    "fix interval": 5,
    "number": 7,
} # only for training
pose_file = occ_path + '/cam_infos.pkl'
val_pose_file = occ_path + '/cam_infos_vali.pkl'

model = dict(
    type='OccNet',
    loss_norm=True,
    sensor=sensor,
    pts_voxel_layer=dict(
        max_num_points=10, 
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        voxel_size= voxel_size,  # xy size follow centerpoint
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x',
        input_channel=3 if Random else 5,
        base_channel=init_size // 4,
        out_channel=init_size //4 * 5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[spatial_shape[0], spatial_shape[1], spatial_shape[2]]
        ),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=init_size //4 * 5,
        block_inplanes=voxel_channels,
        out_indices=voxel_out_indices,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    occ_encoder_neck=dict(
        type='FPN3D',
        with_cp=True,
        in_channels=voxel_channels,
        out_channels=voxel_out_channel,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    pts_bbox_head=dict(
        type='OccHead',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=False,
        cascade_ratio=cascade_ratio,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        final_occ_size=final_occ_size,
        fine_topk=15000,
        dual = unilidar,
        empty_idx= empty_idx,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
    ),
    loss_bbox=dict(
        type='OccLoss',
        balance_cls_weight=True,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=0.3,
            loss_voxel_sem_scal_weight=1.5,
            loss_voxel_geo_scal_weight=0.5,
            loss_voxel_lovasz_weight=1.7,
        ),
        cascade_ratio=cascade_ratio,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        dual=unilidar,
        num_cls=num_cls,),
    empty_idx= empty_idx,
    spatial_shape=spatial_shape,
    random=Random,
)

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
        type='LoadOccGTFromFileWaymo',
        data_root=occ_gt_data_root,
        crop_x=False,
        use_infov_mask=False,
        use_camera_mask=False,
        use_lidar_mask=True,
        num_classes=16,
        pc_range = ori_point_cloud_range,
        RPR = RPR,
        restrict_pc_range = point_cloud_range,
    ),
    dict(type='VoxelClassMapping'),
    dict(type='Collect3Dinput', keys=['points', 'dataset_flag', 'gt_occ']) ]
test_pipeline = [
    dict(
    type='LoadPointsFromFile_RPR',
    coord_type='LIDAR',
    load_dim=6,
    use_dim=3 if Random else 5,
    shift_height=False,
    RPR=RPR,
    point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
    dataset_flag=dataset_flag,
    shift_coors=[0, 0, -0.2],
    Random = False),
    dict(
        type='LoadOccGTFromFileWaymo',
        data_root=occ_val_gt_data_root,
        crop_x=False,
        use_infov_mask=False,
        use_camera_mask=False,
        use_lidar_mask=True,
        num_classes=16,
        pc_range = ori_point_cloud_range,
        RPR = RPR,
        restrict_pc_range = point_cloud_range,
    ),
    dict(type='VoxelClassMapping'),
    dict(type='Collect3Dinput', keys=['points', 'dataset_flag', 'visible_mask', 'gt_occ']) ]

test_config=dict(
    type=dataset_type,
    occ_root=occ_path,
    data_root=data_root,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    modality=input_modality,
    pose_file=val_pose_file,
    split='training',
    input_sample_policy=input_sample_policy,
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
        split='training',
        input_sample_policy=input_sample_policy,
        classes=CLASS_NAMES_DG if DG else class_names,
        test_mode=False,
        occ_size=occ_size,
        seg_label_mapping=WAYMO_TO_CL if DG else labels_map,
        pc_range=point_cloud_range,
        filter_empty_gt=True,
        random=Random,
        repeat=2,
),

data = dict(
    samples_per_gpu=1, #6
    workers_per_gpu=1, #32
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

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)
# load_from = 'ckpts/unilidar_nu_SA_RPR_bbuninorm_25625632_40.9_25.6.pth'
# work_dir = './work_dirs/unilidar_sk_SA_RPRT_bbuninorm_12812816'   
work_dir = './work_dirs/unilidar_L_sktrain_random_RPR_DG'    
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='WandbLoggerHook',
        #     init_kwargs=dict(
        #         project='uniLiDAR_Random',
        #         name='unilidar_L_sktrain_random_RPR_DG_8*4090',
        #         job_type='training',
        #         notes='the-first-test',
        #         config = {"lr": optimizer.get('lr', 5e-4), "total_batch_size": data.get('samples_per_gpu', 1) * 4, "total_epoch": runner.get('max_epochs', 24), "weight_decay": optimizer.get('weight_decay', 0.01)}
        #     )
        # )
    ])


checkpoint_config = dict(interval=2)

# custom_hooks = [
#     dict(type='OccEfficiencyHook'),
# ]

