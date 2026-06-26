'''
Author: EASON XU
Date: 2024-12-23 01:24:22
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-08-24 07:05:27
Description: 头部注释
FilePath: /UniLiDAR/projects/configs/pointocc/pointtpv_waymo_occ_random.py
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

DG = False


#TODO: metainfo
if DG:
    meta_info = dict(
        classes=CLASS_NAMES_DG, seg_label_mapping=WAYMO_TO_CL, max_label=8, voxel_label_mapping=WAYMO_TO_CL)
else:
    meta_info = dict(
        classes=class_names, seg_label_mapping=labels_map, max_label=19, voxel_label_mapping=labels_map)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

Random = False 
plugin = True
plugin_dir = "projects/unilidar_plugin/"
img_norm_cfg = None
occ_path = "data/Waymo-Occ"
train_ann_file = occ_path + '/waymo_infos_train.pkl'
val_ann_file = occ_path + '/waymo_infos_val.pkl'
occ_gt_data_root = occ_path + '/training/'
occ_val_gt_data_root = occ_path + '/validation/'

unilidar = False
cylinder=False
RPR = True
coor_alignment = False
# ori_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# # point_cloud_range = [0, -25.6, -3.4, 51.2, 25.6, 3.0] if unilidar else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# point_cloud_range = [-80.0, -80.0, -1.0, 80.0, 80.0, 5.4]
ori_point_cloud_range = [-80.0, -80.0, -1.0, 80.0, 80.0, 5.4]
point_cloud_range = [-51.2, -51.2, -1.0, 51.2, 51.2, 3.0]
occ_size = [512, 512, 20] # ground truth
final_occ_size =  [480, 360, 32]# model output
model_empty_idx = 0  # noise 0-->255
empty_idx = 0  # noise 0-->255
num_cls = 9 if DG else 16
visible_mask = True

sample_from_voxel = False
sample_from_img = False

sensor = dict(
    LiDAR_height=[1, 2],
    num_of_beams=[32, 64],
    horizontal_angular_resolution=[900, 3600],
    lower_vertical_field_of_view_bound=[-40, -5],
    upper_vertical_field_of_view_bound=[0, 25],
    # LiDAR_height=[1, 1.01],
    # num_of_beams=[31, 33],
    # horizontal_angular_resolution=[2400, 2500],
    # lower_vertical_field_of_view_bound=[-15, -14],
    # upper_vertical_field_of_view_bound=[5, 6],
)

cascade_ratio = 2 
dataset_flag = 3


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
    type='PointTPV_Occ',
    tpv_aggregator=dict(
        type='TPVAggregator_Occ',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        grid_size_occ=grid_size_occ,
        coarse_ratio=coarse_ratio,
        loss_weight=[1,1,1,1],
        nbr_classes=nbr_class,
        in_dims=_dim_,
        hidden_dims=2*_dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z
    ),
    lidar_tokenizer=dict(
        type='CylinderEncoder_Occ',
        grid_size=grid_size,
        in_channels=8 if Random else 10,
        out_channels=128,
        fea_compre=None,
        base_channels=128,
        split=[8,8,8],
        track_running_stats=track_running_stats,
    ),
    lidar_backbone=dict(
        type='Swin',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        in_channels=128,
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
    sensor = sensor,
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
        type='LoadOccGTFromFileWaymo',
        data_root=occ_gt_data_root,
        crop_x=False,
        use_infov_mask=False,
        use_camera_mask=False,
        use_lidar_mask=True,
        num_classes=nbr_class,
        pc_range = ori_point_cloud_range,
        RPR = RPR,
        restrict_pc_range = point_cloud_range,
    ),
    dict(type='VoxelClassMapping'),
    dict(type='PointoccMapping', 
                occ_path=occ_path,
                grid_size = grid_size,
                grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
                grid_size_occ = grid_size_occ,
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
    dict(type='Collect3Dinput', keys=['processed_label', 'train_grid', 'train_grid_vox_coarse', 'voxel_label']),
]

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
        num_classes=nbr_class,
        pc_range = ori_point_cloud_range,
        RPR = RPR,
        restrict_pc_range = point_cloud_range,
    ),
    dict(type='VoxelClassMapping'),
    dict(type='PointoccMapping', 
            occ_path=occ_path,
            grid_size = grid_size,
            grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
            grid_size_occ = grid_size_occ,
            coarse_ratio = coarse_ratio,
            pc_range = ori_point_cloud_range,
            RPR = RPR,
            restrict_pc_range = point_cloud_range,
            fill_label = 0,
            unique_label = unique_label,
            fixed_volume_space = True,
            max_volume_space = [50, 3.1415926, 3],
            min_volume_space = [0, -3.1415926, -5],
            cal_visible=visible_mask),
    dict(type='Collect3Dinput', keys=['processed_label', 'train_grid', 'train_grid_vox_coarse', 'voxel_label',
                                        'visible_mask', 'dataset_flag'],
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
),

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1, #12
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
work_dir = './work_dirs/unilidar_P_sktrain_random_DG_RPR'    
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='SwanlabLoggerHook',
        #     init_kwargs=dict(
        #         project='mm_UniLiDAR',
        #         name='unilidar_P_nutrain_random_DG_RPR_8*4090',
        #         job_type='training',
        #         notes='the-first-test',
        #         config = {"lr": optimizer.get('lr', 5e-4), "total_batch_size": data.get('samples_per_gpu', 1) * 8, "total_epoch": runner.get('max_epochs', 24), "weight_decay": optimizer.get('weight_decay', 0.01)}
        #     ))
    ])

checkpoint_config = dict(interval=2)




