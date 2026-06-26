'''
Author: EASON XU
Date: 2025-05-29 02:13:11
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-11-13 20:35:48
Description: 头部注释
FilePath: /UniLiDAR/projects/configs/cylinder3d/cylinder3d_nusc_seg_random.py
'''
_base_ = [
    './_base_/default_runtime.py',
    './_base_/custom_nus-3d.py',
]

dataset_type = 'NuscSegDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
DG = True
Random = True
plugin = True
plugin_dir = "projects/unilidar_plugin/"

# occ_path = "./data/nuScenes-Occupancy"
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
# For nuScenes we usually do 10-class detection
class_names = ['noise', 'flat.driveable_surface', 'flat.sidewalk', 'flat.terrain', 
             'flat.other', 'static.manmade', 'static.vegetation', 'static.other', 
             'vehicle.ego']

CLASS_NAMES_DG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature', 'Living Being', 'Dynamic Object', 'Static Object', 'Other Ground']

NUSCENES_TO_CL =  {
        0: 0,
        1: 6,
        2: 7,
        3: 3,
        4: 3,
        5: 3,
        6: 3,
        7: 5,
        8: 6,
        9: 3,
        10: 3,
        11: 1,
        12: 8,
        13: 8,
        14: 4,
        15: 2,
        16: 4, 
}

unilidar=False
cylinder=False
RPR = False
coor_alignment = False
ori_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [0, -25.6, -3.4, 51.2, 25.6, 3.0]
# point_cloud_range = [0, -25.6, -3.4, 51.2, 25.6, 3.0] if unilidar else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
model_empty_idx = 0  # noise 0-->255
empty_idx = 0  # noise 0-->255
num_cls_nu = 9 if DG else 17
# len(class_names)  # 0 free, 1-16 obj
visible_mask = True

sensor = dict(
    LiDAR_height=[1.75, 1.85],
    num_of_beams=[32, 64],
    horizontal_angular_resolution=[600, 1800],
    lower_vertical_field_of_view_bound=[-35, -25],
    upper_vertical_field_of_view_bound=[5, 15],
    # LiDAR_height=[1, 2],
    # num_of_beams=[16, 128],
    # horizontal_angular_resolution=[900, 3600],
    # lower_vertical_field_of_view_bound=[-40, -5],
    # upper_vertical_field_of_view_bound=[0, 25],
)

cascade_ratio = 2 
dataset_flag_nu = 1 if dataset_type == 'NuscSegDataset' else 2


cumulative_iters = 1
find_unused_parameters = False
unique_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
track_running_stats = False

_dim_ = 128 

tpv_w_ = 240
tpv_h_ = 180
tpv_z_ = 16
scale_w = 2
scale_h = 2
scale_z = 2


coarse_ratio = cascade_ratio
sweeps_num = 10
grid_size = [480, 360, 32]
nbr_class = num_cls_nu


model = dict(
    type='Cylinder3D',
    voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels=5 if Random else 7,
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
        point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2],
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


if DG:
    train_pipeline = [
        dict(type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=3 if Random else 5,
        use_dim=3 if Random else 5,
        shift_height=coor_alignment,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=[0, 0, -0.2],
        Random = Random),
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
        dict(type='VoxelClassMapping'),
        dict(type='Collect3Dinput', keys=['points','train_pts_label']),
    ]
    test_pipeline = [
        dict(type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        shift_height=coor_alignment,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=[0, 0, -0.2],
        Random = False),
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
        dict(type='VoxelClassMapping'),
        dict(type='Collect3Dinput', keys=['points','train_pts_label', 'dataset_flag'],
                meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
    ]
else:
    train_pipeline = [
        dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=3 if Random else 5,
        use_dim=3 if Random else 5,
        shift_height=coor_alignment,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=[0, 0, -0.2],
        Random = Random),
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
        load_dim=5,
        use_dim=5,
        shift_height=coor_alignment,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=[0, 0, -0.2],
        Random = False),
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
        dict(type='Collect3Dinput', keys=['points','train_pts_label', 'dataset_flag'],
                meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
    ]
    

test_config=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    classes=CLASS_NAMES_DG if DG else class_names,
    modality=input_modality,
    pc_range=point_cloud_range,
    seg_label_mapping= NUSCENES_TO_CL if DG else None,
    random=False,
)

train_config=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=CLASS_NAMES_DG if DG else class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        pc_range=point_cloud_range,
        box_type_3d='LiDAR',
        seg_label_mapping= NUSCENES_TO_CL if DG else None,
        random=Random,
        repeat=2 if Random else 1,),

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# optimizer = dict(
#     type='AdamW',
#     lr=3e-3,
#     paramwise_cfg=dict(
#         custom_keys={
#             'img_backbone': dict(lr_mult=0.1),
#         }),
#     weight_decay=0.01)

# optimizer_config = dict(grad_clip=dict(max_norm=30, norm_type=2))
# # learning policy
# lr_config = dict(
#     policy='CosineRestart',  # Hook type
#     periods=[12, 8, 4],  # Restart periods (3 cycles, 24 total epochs)
#     restart_weights=[1, 0.5, 0.3],  # Restart weights (decaying weight per cycle)
#     min_lr_ratio=1e-2,  # Minimum learning rate as a ratio of base_lr
#     by_epoch=True,  # Adjust LR based on epoch count
#     warmup='linear',  # Warmup strategy
#     warmup_iters=9999,
#     warmup_ratio=1.0 / 3,
# )

optimizer = dict(
    type='AdamW',
    lr=6e-4,
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
    warmup_iters=2500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)

# load_from = '/home/eason/workspace_perception/UniLiDAR/work_dirs/unilidar_PoinOcc_nu_sk_RPR(re)_lw_cw_ckpt/best_SSC_mean_epoch_13.pth'
work_dir = './work_dirs/unilidar_P_nutrain_DG_RPR'    
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='SwanlabLoggerHook',
        #     init_kwargs=dict(
        #         project='UniLiDAR_Random',
        #         name='unilidar_P_nutrain_random_DG_RPR_8*4090',
        #         job_type='training',
        #         notes='the-first-test',
        #         config = {"lr": optimizer.get('lr', 5e-4), "total_batch_size": data.get('samples_per_gpu', 1) * 8, "total_epoch": runner.get('max_epochs', 24), "weight_decay": optimizer.get('weight_decay', 0.01)}
        #     ))
    ])

checkpoint_config = dict(interval=2)




