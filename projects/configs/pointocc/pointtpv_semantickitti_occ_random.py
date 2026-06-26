'''
Author: EASON XU
Date: 2025-01-06 06:13:06
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-11-04 11:03:10
Description: 头部注释
FilePath: /UniLiDAR/projects/configs/pointocc/pointtpv_semantickitti_occ_random.py
'''

_base_ = [
    './_base_/default_runtime.py',
    './_base_/semantickitti.py',
]

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

CLASS_NAMES_DG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature', 'Living Being', 'Dynamic Object', 'Static Object', 'Other Ground']

SEMANTICKITTI_TO_CL ={
    0: 0,
    1: 0,
    10: 3,
    11: 3,
    13: 3,
    15: 3,
    16: 3,
    18: 3,
    20: 3,
    30: 5,
    31: 5,
    32: 5,
    40: 1,
    44: 1,
    48: 8,
    49: 8,
    50: 2,
    51: 2,
    52: 2,
    60: 1,
    70: 4,
    71: 4,
    72: 4,
    80: 7,
    81: 7,
    99: 0,
    252: 6,
    253: 3,
    254: 5,
    255: 5,
    256: 6,
    257: 6,
    258: 6,
    259: 6
}

DG = True


#TODO: metainfo
if DG:
    meta_info = dict(
        classes=CLASS_NAMES_DG, seg_label_mapping=SEMANTICKITTI_TO_CL, max_label=8, voxel_label_mapping=SEMANTICKITTI_TO_CL)
else:
    meta_info = dict(
        classes=class_names, seg_label_mapping=labels_map, max_label=19, voxel_label_mapping=labels_map)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

Random = True 
plugin = True
plugin_dir = "projects/unilidar_plugin/"
img_norm_cfg = None
occ_path = "./data/semantickitti"
train_ann_file = "./data/semantickitti/semantickitti_infos_train.pkl"
val_ann_file = "./data/semantickitti/semantickitti_infos_val.pkl"

unilidar = False
cylinder=False
RPR = True
coor_alignment = False
ori_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# point_cloud_range = [0, -25.6, -3.4, 51.2, 25.6, 3.0] if unilidar else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [0, -25.6, -3.4, 51.2, 25.6, 3.0]
occ_size = [256, 256, 32] # ground truth
final_occ_size =  [480, 360, 32]# model output
model_empty_idx = 0  # noise 0-->255
empty_idx = 0  # noise 0-->255
num_cls_nu = 9 if DG else 17  # 0 free, 1-16 obj
num_cls_sk = 9 if DG else 20  # 0 free, 1-16 obj
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
dataset_flag = 2


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
nbr_class = num_cls_sk

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
        in_channels=8 if Random else 9,
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

if Random:
    train_pipeline = [
        dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=3,
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
        random=Random,
        unoccupied=empty_idx, 
        pc_range = ori_point_cloud_range,
        RPR = RPR,
        restrict_pc_range = point_cloud_range,
        cal_visible=visible_mask,
        file_client_args=dict(backend='disk')),
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
        load_dim=4,
        use_dim=3,
        shift_height=False,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag,
        shift_coors=[0, 0, -0.2],
        Random = False),
        dict(
        type='LoadVoxels',
        to_float32=True, 
        use_semantic=True, 
        cylinder=cylinder, 
        occ_path=occ_path, 
        grid_size=occ_size, 
        use_vel=False,
        random=False,
        unoccupied=empty_idx, 
        pc_range=point_cloud_range, 
        cal_visible=visible_mask,
        file_client_args=dict(backend='disk')),
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
        dict(type='Collect3Dinput', keys=['processed_label', 'train_grid', 'train_grid_vox_coarse', 'voxel_label',
                                          'visible_mask', 'dataset_flag'],
             meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
    ]

else:
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
    classes=CLASS_NAMES_DG if DG else class_names,
    occ_size=occ_size,
    seg_label_mapping=SEMANTICKITTI_TO_CL if DG else labels_map,
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
        classes=CLASS_NAMES_DG if DG else class_names,
        test_mode=False,
        occ_size=occ_size,
        seg_label_mapping=SEMANTICKITTI_TO_CL if DG else labels_map,
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




