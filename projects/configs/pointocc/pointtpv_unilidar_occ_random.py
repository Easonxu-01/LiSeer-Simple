'''
Author: EASON XU
Date: 2025-01-08 05:21:25
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-01-27 10:00:17
Description: 头部注释
FilePath: /UniLiDAR/projects/configs/pointocc/pointtpv_unilidar_occ_random.py
'''

_base_ = [
    '../_base_/default_runtime.py',
    '../datasets/custom_nus-3d.py',
]
dataset_type_sk = 'SemantickittiVoxelDataset'
data_root_sk = 'data/semantickitti/'
dataset_type_nu = 'NuscOCCDataset'
data_root_nu = 'data/nuscenes/'
file_client_args = dict(backend='disk')
#test
gpu_ids = [0,1,2,3,4,5,6,7]
seed=42

class_names_sk = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
               'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
               'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
               'trunck', 'terrian', 'pole', 'traffic-sign'
]
# For nuScenes we usually do 10-class detection
class_names_nu = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
labels_map_sk = {
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

metainfo_sk = dict(
    classes=class_names_sk, seg_label_mapping=labels_map_sk, max_label=259, voxel_label_mapping=labels_map_sk)
input_modality_sk = dict(use_lidar=True, use_camera=False)
input_modality_nu = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
backend_args = None

Random = False 
plugin = True
plugin_dir = "projects/unilidar_plugin/"
img_norm_cfg = None
occ_path_sk = "./data/semantickitti"
occ_path_nu = "./data/nuScenes-Occupancy"
train_ann_file_sk = "./data/semantickitti/semantickitti_infos_train.pkl"
val_ann_file_sk = "./data/semantickitti/semantickitti_infos_val.pkl"
train_ann_file_nu = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file_nu = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
 
fine_tune = False
unilidar = True
test_dual = False
cylinder=False
RPR = True
coor_alignment_nu = False
coor_alignment_sk = False
ori_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] if unilidar else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range_nu = [0, -25.6, -3.4, 51.2, 25.6, 3.0]
point_cloud_range_sk = [0, -25.6, -3.4, 51.2, 25.6, 3.0]
RPR_scale = [(point_cloud_range_nu[3] - point_cloud_range_nu[0])/(ori_point_cloud_range[3]-ori_point_cloud_range[0]), 
             (point_cloud_range_nu[4] - point_cloud_range_nu[1])/(ori_point_cloud_range[4]-ori_point_cloud_range[1]),
             (point_cloud_range_nu[5] - point_cloud_range_nu[2])/(ori_point_cloud_range[5]-ori_point_cloud_range[2])]
# [0, -25.6, -3.4, 51.2, 25.6, 3.0]
# point_cloud_range = [0.0, -3.1415926, -5.0, 51.2, 3.1415926, 3.0] if cylinder else [0, -25.6, -3.4, 51.2, 25.6, 3.0]
occ_size_nu = [512, 512, 40]
occ_size_nu_RPR = [round(512*RPR_scale[0]), round(512*RPR_scale[1]), round(40*RPR_scale[2])]
occ_size_sk = [256, 256, 32]
final_occ_size =  [480, 360, 32]# model output
model_empty_idx = 0  # noise 0-->255
empty_idx_sk = 0  
empty_idx_nu = 0  # noise 0-->255
num_cls_nu = 17  # 0 free, 1-16 obj
num_cls_sk = 20  # 0 free, 1-16 obj
visible_mask = True

sample_from_voxel = False
sample_from_img = False

sensor = dict(
    LiDAR_height=[1, 2],
    num_of_beams=[16, 128],
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
dataset_flag_nu = 1 
dataset_flag_sk = 2


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
grid_size_occ_nu = occ_size_nu
grid_size_occ_sk = occ_size_sk
coarse_ratio = cascade_ratio
sweeps_num = 10
nbr_class_nu = num_cls_nu
nbr_class_sk = num_cls_sk

model = dict(
    type='PointTPV_Occ',
    tpv_aggregator=dict(
        type='TPVAggregator_Occ',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        grid_size_occ=[occ_size_nu_RPR if RPR else grid_size_occ_nu, grid_size_occ_sk] if unilidar else grid_size_occ_nu,
        coarse_ratio=coarse_ratio,
        loss_weight=[0.5,1.5,1.8,0.2],
        nbr_classes=[nbr_class_nu, nbr_class_sk] if unilidar else nbr_class_nu,
        in_dims=_dim_,
        hidden_dims=2*_dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z,
        dual=unilidar,
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
    empty_idx = 0
)


bda_aug_conf_nu = dict(
            rot_lim=(-22.5, 22.5),
            # rot_lim=(-0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.0,
            flip_dy_ratio=0.5)

if Random:
    train_pipeline_nu = [
        dict(
            type='LoadPointsFromFile_Sampling',
            coord_type='LIDAR',
            load_dim=3,
            use_dim=3,
            LiDAR_height=sensor.get('LiDAR_height', [1, 2]),
            num_of_beams=sensor.get('num_of_beams', [16, 128]),
            horizontal_angular_resolution=sensor.get('horizontal_angular_resolution', [900, 3600]),
            lower_vertical_field_of_view_bound=sensor.get('lower_vertical_field_of_view_bound', [-40, -5]),
            upper_vertical_field_of_view_bound=sensor.get('upper_vertical_field_of_view_bound', [0, 25])),
        dict(type='PointoccMapping', 
                occ_path=occ_path_nu,
                grid_size = grid_size,
                grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
                grid_size_occ = grid_size_occ_nu,
                coarse_ratio = coarse_ratio,
                pc_range = ori_point_cloud_range,
                fill_label = 0,
                unique_label = unique_label,
                fixed_volume_space = True,
                max_volume_space = [50, 3.1415926, 3],
                min_volume_space = [0, -3.1415926, -5],
                cal_visible=visible_mask,
                RPR = RPR,
                restrict_pc_range = point_cloud_range_nu,
                unilidar = unilidar,),
        dict(type='Collect3Dinput', keys=['processed_label', 'train_grid', 'train_grid_vox_coarse','voxel_label']),
    ]

    test_pipeline_nu = [
        dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=3,
        shift_height=coor_alignment_nu,
        RPR=RPR,
        point_cloud_range=point_cloud_range_nu if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=[0, 0, -0.2],),
        dict(type='PointoccMapping', 
                occ_path=occ_path_nu,
                grid_size = grid_size,
                grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
                grid_size_occ = grid_size_occ_nu,
                coarse_ratio = coarse_ratio,
                pc_range = ori_point_cloud_range,
                fill_label = 0,
                unique_label = unique_label,
                fixed_volume_space = True,
                max_volume_space = [50, 3.1415926, 3],
                min_volume_space = [0, -3.1415926, -5],
                cal_visible=visible_mask,
                RPR = RPR,
                restrict_pc_range = point_cloud_range_nu,
                unilidar = unilidar,),
        dict(type='Collect3Dinput', keys=['processed_label', 'train_grid', 'train_grid_vox_coarse', 'voxel_label', 'visible_mask', 'dataset_flag'],
                meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
    ]
    train_pipeline_sk = [
        dict(
            type='LoadPointsFromFile_Sampling',
            coord_type='LIDAR',
            load_dim=3,
            use_dim=3,
            LiDAR_height=sensor.get('LiDAR_height', [1, 2]),
            num_of_beams=sensor.get('num_of_beams', [16, 128]),
            horizontal_angular_resolution=sensor.get('horizontal_angular_resolution', [900, 3600]),
            lower_vertical_field_of_view_bound=sensor.get('lower_vertical_field_of_view_bound', [-40, -5]),
            upper_vertical_field_of_view_bound=sensor.get('upper_vertical_field_of_view_bound', [0, 25])),
        dict(
        type='LoadVoxels',
        to_float32=True, 
        use_semantic=True, 
        cylinder=cylinder, 
        occ_path=occ_path_sk, 
        grid_size=occ_size_sk, 
        use_vel=False,
        unoccupied=empty_idx_sk, 
        pc_range=point_cloud_range_sk, 
        cal_visible=visible_mask,
        file_client_args=dict(backend='disk')),
    dict(type='VoxelClassMapping'),
    dict(type='PointoccMapping', 
                occ_path=occ_path_sk,
                grid_size = grid_size,
                grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
                grid_size_occ = grid_size_occ_sk,
                coarse_ratio = coarse_ratio,
                pc_range = ori_point_cloud_range,
                fill_label = 0,
                unique_label = unique_label,
                fixed_volume_space = True,
                max_volume_space = [50, 3.1415926, 3],
                min_volume_space = [0, -3.1415926, -5],
                cal_visible=visible_mask,
                RPR = RPR,
                restrict_pc_range = point_cloud_range_sk,
                unilidar = unilidar,),
    dict(type='Collect3Dinput', keys=['processed_label', 'train_grid', 'train_grid_vox_coarse', 'voxel_label']),
    ]

    test_pipeline_sk = [
        dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=3,
        shift_height=coor_alignment_sk,
        RPR=RPR,
        point_cloud_range=point_cloud_range_sk if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_sk,
        shift_coors=[0, 0, -0.2],),
        dict(
        type='LoadVoxels',
        to_float32=True, 
        use_semantic=True, 
        cylinder=cylinder, 
        occ_path=occ_path_sk, 
        grid_size=occ_size_sk, 
        use_vel=False,
        unoccupied=empty_idx_sk, 
        pc_range=point_cloud_range_sk, 
        cal_visible=visible_mask,
        file_client_args=dict(backend='disk')),
        dict(type='VoxelClassMapping'),
        dict(type='PointoccMapping', 
                occ_path=occ_path_sk,
                grid_size = grid_size,
                grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
                grid_size_occ = grid_size_occ_sk,
                coarse_ratio = coarse_ratio,
                pc_range = ori_point_cloud_range,
                fill_label = 0,
                unique_label = unique_label,
                fixed_volume_space = True,
                max_volume_space = [50, 3.1415926, 3],
                min_volume_space = [0, -3.1415926, -5],
                cal_visible=visible_mask,
                RPR = RPR,
                restrict_pc_range = point_cloud_range_sk,
                unilidar = unilidar,),
        dict(type='Collect3Dinput', keys=['processed_label', 'train_grid', 'train_grid_vox_coarse', 'voxel_label',
                                          'visible_mask', 'dataset_flag'],
             meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
    ]
    
else:
    train_pipeline_nu = [
    dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        shift_height=coor_alignment_nu,
        RPR=RPR,
        point_cloud_range=point_cloud_range_nu if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=[0, 0, -0.2],),
    dict(type='LoadPointsFromMultiSweeps_RPR',
        sweeps_num=10,
        RPR=RPR,
        point_cloud_range=point_cloud_range_nu if RPR else ori_point_cloud_range,),
    dict(type='PointoccMapping', 
                occ_path=occ_path_nu,
                grid_size = grid_size,
                grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
                grid_size_occ = grid_size_occ_nu,
                coarse_ratio = coarse_ratio,
                pc_range = ori_point_cloud_range,
                fill_label = 0,
                unique_label = unique_label,
                fixed_volume_space = True,
                max_volume_space = [50, 3.1415926, 3],
                min_volume_space = [0, -3.1415926, -5],
                cal_visible=visible_mask,
                RPR = RPR,
                restrict_pc_range = point_cloud_range_nu,
                unilidar = unilidar,),
    dict(type='Collect3Dinput', keys=['processed_label', 'train_grid', 'train_grid_vox_coarse', 'voxel_label', 'dataset_flag']),
]

    test_pipeline_nu = [
        dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        shift_height=coor_alignment_nu,
        RPR=RPR,
        point_cloud_range=point_cloud_range_nu if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=[0, 0, -0.2],),
        dict(type='LoadPointsFromMultiSweeps_RPR',
        sweeps_num=10,
        RPR=False,
        point_cloud_range=point_cloud_range_nu if RPR else ori_point_cloud_range,),
        dict(type='PointoccMapping', 
                occ_path=occ_path_nu,
                grid_size = grid_size,
                grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
                grid_size_occ = grid_size_occ_nu,
                coarse_ratio = coarse_ratio,
                pc_range = ori_point_cloud_range,
                fill_label = 0,
                unique_label = unique_label,
                fixed_volume_space = True,
                max_volume_space = [51.2, 3.1415926, 3],
                min_volume_space = [0, -3.1415926, -3.4],
                cal_visible=visible_mask,
                RPR = RPR,
                restrict_pc_range = point_cloud_range_nu,
                unilidar = unilidar,),
        dict(type='Collect3Dinput', keys=['processed_label', 'train_grid', 'train_grid_vox_coarse', 'voxel_label', 'visible_mask', 'dataset_flag'],
                meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
    ]
    train_pipeline_sk = [
    dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        shift_height=coor_alignment_sk,
        RPR=RPR,
        point_cloud_range=point_cloud_range_sk if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_sk,
        shift_coors=[0, 0, -0.2],),
    dict(
        type='LoadVoxels',
        to_float32=True, 
        use_semantic=True, 
        cylinder=cylinder, 
        occ_path=occ_path_sk, 
        grid_size=occ_size_sk, 
        use_vel=False,
        unoccupied=empty_idx_sk, 
        pc_range=point_cloud_range_sk, 
        cal_visible=visible_mask,
        file_client_args=dict(backend='disk')),
    dict(type='VoxelClassMapping'),
    dict(type='PointoccMapping', 
                occ_path=occ_path_sk,
                grid_size = grid_size,
                grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
                grid_size_occ = grid_size_occ_sk,
                coarse_ratio = coarse_ratio,
                pc_range = ori_point_cloud_range,
                fill_label = 0,
                unique_label = unique_label,
                fixed_volume_space = True,
                max_volume_space = [51.2, 3.1415926, 3],
                min_volume_space = [0, -3.1415926, -3.4],
                cal_visible=visible_mask,
                RPR = RPR,
                restrict_pc_range = point_cloud_range_sk,
                unilidar = unilidar,),
    dict(type='Collect3Dinput', keys=['processed_label', 'train_grid', 'train_grid_vox_coarse', 'voxel_label', 'dataset_flag']),
]

    test_pipeline_sk = [
        dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        shift_height=coor_alignment_sk,
        RPR=RPR,
        point_cloud_range=point_cloud_range_sk if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_sk,
        shift_coors=[0, 0, -0.2],),
        dict(
        type='LoadVoxels',
        to_float32=True, 
        use_semantic=True, 
        cylinder=cylinder, 
        occ_path=occ_path_sk, 
        grid_size=occ_size_sk, 
        use_vel=False,
        unoccupied=empty_idx_sk, 
        pc_range=point_cloud_range_sk, 
        cal_visible=visible_mask,
        file_client_args=dict(backend='disk')),
        dict(type='VoxelClassMapping'),
        dict(type='PointoccMapping', 
                occ_path=occ_path_sk,
                grid_size = grid_size,
                grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
                grid_size_occ = grid_size_occ_sk,
                coarse_ratio = coarse_ratio,
                pc_range = ori_point_cloud_range,
                fill_label = 0,
                unique_label = unique_label,
                fixed_volume_space = True,
                max_volume_space = [51.2, 3.1415926, 3],
                min_volume_space = [0, -3.1415926, -3.4],
                cal_visible=visible_mask,
                RPR = RPR,
                restrict_pc_range = point_cloud_range_sk,
                unilidar = unilidar,),
        dict(type='Collect3Dinput', keys=['processed_label', 'train_grid', 'train_grid_vox_coarse', 'voxel_label',
                                          'visible_mask', 'dataset_flag'],
             meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
    ]
    
    

test_config_sk=dict(
    type=dataset_type_sk,
    occ_root=occ_path_sk,
    data_root=data_root_sk,
    ann_file=val_ann_file_sk,
    pipeline=test_pipeline_sk,
    modality=input_modality_sk,
    classes=class_names_sk,
    occ_size=occ_size_sk,
    seg_label_mapping=labels_map_sk,
    pc_range=point_cloud_range_sk,
    filter_empty_gt=True,
    random=False,
)

train_config_sk=dict(
        type=dataset_type_sk,
        data_root=data_root_sk,
        occ_root=occ_path_sk,
        ann_file=train_ann_file_sk,
        pipeline=train_pipeline_sk,
        modality=input_modality_sk,
        classes=class_names_sk,
        test_mode=False,
        occ_size=occ_size_sk,
        seg_label_mapping=labels_map_sk,
        pc_range=point_cloud_range_sk,
        filter_empty_gt=True,
        random=Random,
),

data_sk = dict(
    train=train_config_sk,
    val=test_config_sk,
    test=test_config_sk,
)

test_config_nu=dict(
    type=dataset_type_nu,
    occ_root=occ_path_nu,
    data_root=data_root_nu,
    ann_file=val_ann_file_nu,
    pipeline=test_pipeline_nu,
    classes=class_names_nu,
    modality=input_modality_nu,
    occ_size=occ_size_nu,
    pc_range=point_cloud_range_nu,
    random=False,
)

train_config_nu=dict(
        type=dataset_type_nu,
        data_root=data_root_nu,
        occ_root=occ_path_nu,
        ann_file=train_ann_file_nu,
        pipeline=train_pipeline_nu,
        classes=class_names_nu,
        modality=input_modality_nu,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size_nu,
        pc_range=point_cloud_range_nu,
        box_type_3d='LiDAR',
        random=Random),

data_nu = dict(
    train=train_config_nu,
    val=test_config_nu,
    test=test_config_nu,
)
data = data_nu

data_merge = dict(
    samples_per_gpu=6,
    workers_per_gpu=24,
    shuffler_sampler=dict(type='BalancedDistributedGroupSampler'),
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
    policy='CosineRestart',  # Hook type
    periods=[12, 8, 4],  # Restart periods (3 cycles, 24 total epochs)
    restart_weights=[1, 0.5, 0.3],  # Restart weights (decaying weight per cycle)
    min_lr_ratio=1e-2,  # Minimum learning rate as a ratio of base_lr
    by_epoch=True,  # Adjust LR based on epoch count
    warmup='linear',  # Warmup strategy
    warmup_iters=9999,
    warmup_ratio=1.0 / 5,
)

runner = dict(type='EpochBasedRunner', max_epochs=24)

evaluation_sk = dict(
    interval=1,
    pipeline=test_pipeline_sk,
    save_best='SSC_mean',
    rule='greater',
)
evaluation_nu = dict(
    interval=1,
    pipeline=test_pipeline_nu,
    save_best='SSC_mean',
    rule='greater',
)
evaluation = evaluation_nu

# load_from = '/home/eason/workspace_perception/UniLiDAR/work_dirs/unilidar_PoinOcc_nu_sk_RPR(re)_lw_cw_ckpt/best_SSC_mean_epoch_13.pth'
work_dir = './work_dirs/unilidar_PoinOcc_nu_sk_RPR_lw_cw'    
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='mm_UniLiDAR',
                name='unilidar_PoinOcc_nu_sk_RPR_lw_cw_4A800_1',
                job_type='training',
                notes='the-first-test',
                config = {"lr": optimizer.get('lr', 5e-4), "total_batch_size": data.get('samples_per_gpu', 1) * 8, "total_epoch": runner.get('max_epochs', 24), "weight_decay": optimizer.get('weight_decay', 0.01)}
            ))
    ])

checkpoint_config = dict(interval=2)




