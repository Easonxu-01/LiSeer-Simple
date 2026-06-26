'''
Author: EASON XU
Date: 2023-11-05 02:12:53
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-01-06 06:20:32
Description: 头部注释
FilePath: /UniLiDAR/projects/configs/baselines/LiDAR_128x128x10_semantickitti.py
'''
_base_ = [
    '../_base_/datasets/semantickitti.py',
    '../_base_/default_runtime.py'
]

dataset_type = 'SemantickittiVoxelDataset'
data_root = 'data/semantickitti/'
file_client_args = dict(backend='disk')


class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
               'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
               'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
               'trunck', 'terrian', 'pole', 'traffic-sign'
]
labels_map = {
    0 : 0,     # "unlabeled"
    1 : 0,    # "outlier" mapped to "unlabeled" --------------------------mapped
    10 : 1,     # "car"
    11 : 2,     # "bicycle"
    13 : 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15 : 3,     # "motorcycle"
    16 : 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18 : 4,    # "truck"
    20 : 5,   # "other-vehicle"
    30 : 6,     # "person"
    31 : 7,     # "bicyclist"
    32: 8,     # "motorcyclist"
    40: 9,     # "road"
    44: 10,    # "parking"
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,     # "lane-marking" to "road" ---------------------------------mapped
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,    # "moving-car" to "car" ------------------------------------mapped
    253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,    # "moving-person" to "person" ------------------------------mapped
    255: 8,   # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,    # "moving-truck" to "truck" --------------------------------mapped
    259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

#TODO: metainfo
metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=19, voxel_label_mapping=labels_map)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

plugin = True
plugin_dir = "projects/unilidar_plugin/"
img_norm_cfg = None
occ_path = "./data/semantickitti"
train_ann_file = "./data/semantickitti/semantickitti_infos_train.pkl"
val_ann_file = "./data/semantickitti/semantickitti_infos_val.pkl"

RPR=False
cylinder=False
coor_alignment = False
# [-72.0,-72.0,-3.4,72.0,72.0,3.0]
ori_point_cloud_range = [-51.2, -51.2, -3.4, 51.2, 51.2, 3.0]
point_cloud_range = [0.0, -3.1415926, -3.4, 25.6, 3.1415926, 3.0] if cylinder else [0, -25.6, -3.4, 51.2, 25.6, 3.0]
# [0, -25.6, -3.4, 51.2, 25.6, 3.0]
occ_size = [256, 256, 32]
final_occ_size = occ_size
voxel_channels = [80, 160, 320, 640]
empty_idx = 0  
num_cls = 20  
visible_mask = True

sample_from_voxel = False
sample_from_img = False

numC_Trans = 80
init_size = 64
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)

find_unused_parameters = False

#model 
spatial_shape = [256, 256, 32] if cylinder else [1024,1024,128]
cascade_ratio = final_occ_size[0] * 8 // spatial_shape[0] 
voxel_size=[0.05,0.05,0.05]

dataset_flag = 1 if dataset_type == 'NuscOCCDataset' else 2
# norm_cfg=dict(type='UniNorm3d', dataset_from_flag=dataset_flag, eps=1e-3, momentum=0.01, voxel_coord=True),
model = dict(
    type='OccNet',
    loss_norm=True,
    pts_voxel_layer=dict(
        max_num_points=10, 
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        voxel_size= voxel_size,  # xy size follow centerpoint
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x',
        input_channel=4,
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
        soft_weights=True,
        cascade_ratio=cascade_ratio,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        final_occ_size=occ_size,
        fine_topk=15000,
        empty_idx=empty_idx,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
    ),
    loss_bbox=dict(
        type='OccLoss',
        balance_cls_weight=True,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
        cascade_ratio=cascade_ratio,
        dataset_flag = dataset_flag,
        num_cls=num_cls,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,),
    empty_idx=empty_idx,
    spatial_shape=spatial_shape,
)

train_pipeline = [
    dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
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
        pc_range=point_cloud_range, 
        cal_visible=visible_mask,
        file_client_args=dict(backend='disk')),
    dict(type='VoxelClassMapping'),
    dict(type='Collect3Dinput', keys=['points', 'voxel', 'pts_semantic_mask', 'dataset_flag', 'voxel_semantic_mask', 'voxel_occ_mask','voxel_invalid'])
]
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=False,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.78539816, 0.78539816],
    #     scale_ratio_range=[0.95, 1.05],
    #     translation_std=[0.1, 0.1, 0.1],
    # ),

test_pipeline = [
    dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        shift_height=False,
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
        pc_range=point_cloud_range, 
        cal_visible=visible_mask,
        file_client_args=dict(backend='disk')),
    dict(type='VoxelClassMapping'),
    dict(type='Collect3Dinput', keys=['points', 'voxel', 'dataset_flag', 'visible_mask','pts_semantic_mask', 'voxel_semantic_mask', 'voxel_occ_mask','voxel_invalid'])
]

# eval_pipeline = [
#     dict(
#         type='LoadPointsFromFile_RPR',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         shift_height=True,
#         RPR=restrict_pointcloud_range,
#         point_cloud_range=point_cloud_range,
#         dataset_flag=dataset_flag,
#         shift_coors=[0, 0, -0.2],),
#     dict(type='Collect3Dinput', keys=['points', 'dataset_flag',])
# ]

test_config=dict(
    type=dataset_type,
    occ_root=occ_path,
    data_root=data_root,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    modality=input_modality,
    classes=class_names,
    occ_size=occ_size,
    seg_label_mapping=labels_map,
    pc_range=point_cloud_range,
    filter_empty_gt=True,
)

train_config=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_path,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        occ_size=occ_size,
        seg_label_mapping=labels_map,
        pc_range=point_cloud_range,
        filter_empty_gt=True,
),

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=15,
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=3e-4,
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
# load_from = 'ckpts/unilidar_nu_SA_RPR_bbuninorm_25625632_40.9_25.6.pth'
# work_dir = './work_dirs/unilidar_sk_SA_RPRT_bbuninorm_12812816'   
work_dir = './work_dirs/unilidar_sk_RPRmin_25625632' 
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='mm_UniLiDAR',
                name='unilidar_sk_RPRmin_25625632',
                job_type='training',
                notes='the-first-test',
            )
        )
    ])

checkpoint_config = dict(interval=2)

custom_hooks = [
    dict(type='OccEfficiencyHook'),
]

