'''
Author: EASON XU
Date: 2024-12-23 01:24:22
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2026-06-11 22:07:42
Description: 头部注释
FilePath: /UniLiDAR/projects/configs/pointocc/pointtpv_nusc_seg_random.py
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
dense7sparse = True
ResetRandomProb = 0.0
plugin = True
consistency = True
d2skd = False
dense_train = False
sensor_film = True
plugin_dir = "projects/unilidar_plugin/"

# occ_path = "./data/nuScenes-Occupancy"
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
# For nuScenes we usually do 10-class detection
class_names = ['noise', 'flat.driveable_surface', 'flat.sidewalk', 'flat.terrain', 
             'flat.other', 'static.manmade', 'static.vegetation', 'static.other', 
             'vehicle.ego']

# CLASS_NAMES_DG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature', 'Living Being', 'Dynamic Object', 'Static Object', 'Other Ground']

# NUSCENES_TO_CL =  {
#     0: 0,   # noise -> Other
#     24: 1,  # flat.driveable_surface -> Driveable Ground
#     25: 8,  # flat.sidewalk -> Structure
#     26: 3,  # flat.terrain -> Nature
#     27: 4,  # flat.other -> Other Ground
#     28: 2,  # static.manmade -> Structure
#     29: 4,  # static.vegetation -> Nature
#     30: 5,  # static.other -> Static Object
#     31: 3,  # vehicle.ego -> Vehicle
# }

CLASS_NAMES_DG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature', 'Living Being', 'Movable objects', 'Other Ground']

# Mapping from the original nuScenes semantic ids to the coarse DG taxonomy.
# Each entry documents: fine label -> CLASS_NAMES_DG target.
NUSCENES_TO_CL =  {
        0: 0,   # void / ignore -> Other
        1: 6,   # barrier -> Movable objects
        2: 6,   # bicycle -> Movable objects
        3: 3,   # bus -> Vehicle
        4: 3,   # car -> Vehicle
        5: 3,   # construction_vehicle -> Vehicle
        6: 6,   # motorcycle -> Movable objects
        7: 5,   # pedestrian -> Living Being
        8: 6,   # traffic_cone -> Movable objects
        9: 3,   # trailer -> Vehicle
        10: 3,  # truck -> Vehicle
        11: 1,  # driveable_surface -> Driveable Ground
        12: 7,  # other_flat -> Other Ground
        13: 7,  # sidewalk -> Other Ground
        14: 4,  # terrain -> Nature
        15: 2,  # manmade -> Structure
        16: 4,  # vegetation -> Nature
}

unilidar=False
cylinder=False
RPR = True
coor_alignment = False
ori_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# point_cloud_range = [0, -25.6, -3.4, 51.2, 25.6, 3.0]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# point_cloud_range = [0, -25.6, -3.4, 51.2, 25.6, 3.0] if unilidar else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
model_empty_idx = 0  # noise 0-->255
empty_idx = 0  # noise 0-->255
num_cls_nu = 8 if DG else 17
# len(class_names)  # 0 free, 1-16 obj
visible_mask = True

sensor = dict(
    LiDAR_height=[1.5, 1.85],
    num_of_beams=[16, 80],
    horizontal_angular_resolution=[600, 1800],
    lower_vertical_field_of_view_bound=[-30, -15],
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
                'upper_vertical_field_of_view_bound': (0, 10),
                'LiDAR_height': (1.7, 1.9)
            }),
                      (8, 16, {
                          'num_of_beams': (24, 80),
                          'horizontal_angular_resolution': (600, 3000),
                          'lower_vertical_field_of_view_bound': (-40, -15),
                          'upper_vertical_field_of_view_bound': (0, 15),
                          'LiDAR_height': (1.5, 2)
                      }),
                      (16, 24, {
                          'num_of_beams': (16, 96),
                          'horizontal_angular_resolution': (400, 4000),
                          'lower_vertical_field_of_view_bound': (-45, -10),
                          'upper_vertical_field_of_view_bound': (0, 20),
                          'LiDAR_height': (1.2, 2)
                      })]

cascade_ratio = 2 
dataset_flag_nu = 1 if dataset_type == 'NuscSegDataset' else 2


cumulative_iters = 1
find_unused_parameters = False
unique_label = [0, 1, 2, 3, 4, 5, 6, 7] if DG else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
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
    dynamic_loss_balance=dict(
        enabled=True,
        alpha=0.1,          # GradNorm target strength
        step=0.001,          # per-step update size
        momentum=0.2,       # smoothing for weight updates
        min_weight=0.1,
        max_weight=2.0,
        ema_beta=0.8,       # EMA for loss trend
        # Task grouping follows fixed order in model:
        # point / voxel / sensor / consistency
        base_weights=dict(
            point=1.0,
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
            point=2.5,
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
        Random = Random,
        dense7sparse = dense7sparse if Random else False),
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
        # dict(
        #     type='RandomChoice',
        #     transforms=[
        #         [
        #             dict(
        #                 type='LaserMix',
        #                 num_areas=[3, 4, 5, 6],
        #                 pitch_angles=[-25, 3],
        #                 grid_size=grid_size,
        #                 grid_size_vox=[tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
        #                 max_volume_space=[51.2, 3.1415926, 3],
        #                 min_volume_space=[0, -3.1415926, -3.4],
        #                 fill_label=0,
        #                 pre_transform=[
        #                     dict(
        #                         type='LoadPointsFromFile_RPR',
        #                         coord_type='LIDAR',
        #                         load_dim=3 if Random else 5,
        #                         use_dim=3 if Random else 5,
        #                         shift_height=coor_alignment,
        #                         RPR=RPR,
        #                         point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        #                         dataset_flag=dataset_flag_nu,
        #                         shift_coors=[0, 0, -0.2],
        #                         Random=Random),
        #                     dict(type='PointsegMapping',
        #                         grid_size=grid_size,
        #                         grid_size_vox=[tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
        #                         coarse_ratio=coarse_ratio,
        #                         pc_range=ori_point_cloud_range,
        #                         fill_label=0,
        #                         unique_label=unique_label,
        #                         fixed_volume_space=True,
        #                         max_volume_space=[51.2, 3.1415926, 3],
        #                         min_volume_space=[0, -3.1415926, -3.4],
        #                         cal_visible=False,
        #                         RPR=RPR,
        #                         restrict_pc_range=point_cloud_range),
        #                     dict(type='VoxelClassMapping'),
        #                 ],
        #                 prob=1)
        #         ],
        #         [
        #             dict(
        #                 type='PolarMix',
        #                 instance_classes=[0, 1, 2, 3, 4, 5, 6, 7],
        #                 swap_ratio=0.5,
        #                 rotate_paste_ratio=1.0,
        #                 grid_size=grid_size,
        #                 grid_size_vox=[tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
        #                 max_volume_space=[51.2, 3.1415926, 3],
        #                 min_volume_space=[0, -3.1415926, -3.4],
        #                 fill_label=0,
        #                 pre_transform=[
        #                     dict(
        #                         type='LoadPointsFromFile_RPR',
        #                         coord_type='LIDAR',
        #                         load_dim=3 if Random else 5,
        #                         use_dim=3 if Random else 5,
        #                         shift_height=coor_alignment,
        #                         RPR=RPR,
        #                         point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        #                         dataset_flag=dataset_flag_nu,
        #                         shift_coors=[0, 0, -0.2],
        #                         Random=Random),
        #                     dict(type='PointsegMapping',
        #                         grid_size=grid_size,
        #                         grid_size_vox=[tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
        #                         coarse_ratio=coarse_ratio,
        #                         pc_range=ori_point_cloud_range,
        #                         fill_label=0,
        #                         unique_label=unique_label,
        #                         fixed_volume_space=True,
        #                         max_volume_space=[51.2, 3.1415926, 3],
        #                         min_volume_space=[0, -3.1415926, -3.4],
        #                         cal_visible=False,
        #                         RPR=RPR,
        #                         restrict_pc_range=point_cloud_range),
        #                     dict(type='VoxelClassMapping'),
        #                 ],
        #                 prob=1)
        #         ],
        #     ],
        #     prob=[0.5, 0.5]),
        dict(type='Collect3Dinput', keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label', 'train_pts_label', 'reset_random']),
    ]
    test_pipeline = [
        dict(type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=3 if Random else 5,
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
        dict(type='Collect3Dinput', keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label', 'train_pts_label', 'dataset_flag'],
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
        Random = Random,
        dense7sparse = dense7sparse if Random else False),
    # dict(type='LoadPointsFromMultiSweeps_RPR',
    #     sweeps_num=10,
    #     RPR=False,
    #     point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,),
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
    # dict(
    #         type='RandomChoice',
    #         transforms=[
    #             [
    #                 dict(
    #                     type='LaserMix',
    #                     num_areas=[3, 4, 5, 6],
    #                     pitch_angles=[-25, 3],
    #                     grid_size=grid_size,
    #                     grid_size_vox=[tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
    #                     max_volume_space=[51.2, 3.1415926, 3],
    #                     min_volume_space=[0, -3.1415926, -3.4],
    #                     fill_label=0,
    #                     pre_transform=[
    #                         dict(
    #                             type='LoadPointsFromFile_RPR',
    #                             coord_type='LIDAR',
    #                             load_dim=3 if Random else 5,
    #                             use_dim=3 if Random else 5,
    #                             shift_height=coor_alignment,
    #                             RPR=RPR,
    #                             point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
    #                             dataset_flag=dataset_flag_nu,
    #                             shift_coors=[0, 0, -0.2],
    #                             Random=Random),
    #                         dict(type='PointsegMapping',
    #                             grid_size=grid_size,
    #                             grid_size_vox=[tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
    #                             coarse_ratio=coarse_ratio,
    #                             pc_range=ori_point_cloud_range,
    #                             fill_label=0,
    #                             unique_label=unique_label,
    #                             fixed_volume_space=True,
    #                             max_volume_space=[51.2, 3.1415926, 3],
    #                             min_volume_space=[0, -3.1415926, -3.4],
    #                             cal_visible=False,
    #                             RPR=RPR,
    #                             restrict_pc_range=point_cloud_range),
    #                     ],
    #                     prob=1)
    #             ],
    #             [
    #                 dict(
    #                     type='PolarMix',
    #                     instance_classes=[0, 1, 2, 3, 4, 5, 6, 7],
    #                     swap_ratio=0.5,
    #                     rotate_paste_ratio=1.0,
    #                     grid_size=grid_size,
    #                     grid_size_vox=[tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
    #                     max_volume_space=[51.2, 3.1415926, 3],
    #                     min_volume_space=[0, -3.1415926, -3.4],
    #                     fill_label=0,
    #                     pre_transform=[
    #                         dict(
    #                             type='LoadPointsFromFile_RPR',
    #                             coord_type='LIDAR',
    #                             load_dim=3 if Random else 5,
    #                             use_dim=3 if Random else 5,
    #                             shift_height=coor_alignment,
    #                             RPR=RPR,
    #                             point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
    #                             dataset_flag=dataset_flag_nu,
    #                             shift_coors=[0, 0, -0.2],
    #                             Random=Random),
    #                         dict(type='PointsegMapping',
    #                             grid_size=grid_size,
    #                             grid_size_vox=[tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
    #                             coarse_ratio=coarse_ratio,
    #                             pc_range=ori_point_cloud_range,
    #                             fill_label=0,
    #                             unique_label=unique_label,
    #                             fixed_volume_space=True,
    #                             max_volume_space=[51.2, 3.1415926, 3],
    #                             min_volume_space=[0, -3.1415926, -3.4],
    #                             cal_visible=False,
    #                             RPR=RPR,
    #                             restrict_pc_range=point_cloud_range),
    #                     ],
    #                     prob=1)
    #             ],
    #         ],
    #         prob=[0.5, 0.5]),
    dict(type='Collect3Dinput', keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label', 'train_pts_label', 'reset_random']),
]

    test_pipeline = [
        dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=3 if Random else 5,
        shift_height=coor_alignment,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=[0, 0, -0.2],
        Random = False),
        # dict(type='LoadPointsFromMultiSweeps_RPR',
        # sweeps_num=10,
        # RPR=False,
        # point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,),
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
        reset_random_prob=ResetRandomProb,
        repeat=3 if Random else 1,
        dense7sparse = dense7sparse if Random else False)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
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

# optimizer = dict(
#     type='AdamW',
#     lr=6e-4,
#     paramwise_cfg=dict(
#         custom_keys={
#             'img_backbone': dict(lr_mult=0.1),
#         }),
#     weight_decay=0.01)
# # 与 LiDAR 课程阶段对齐：每阶段独立 Cosine 退火，进入新阶段时 LR 从 base 重启
# lr_curriculum_periods = [end - start for start, end, _ in schedule]
# lr_curriculum_restart_weights = [1.0, 0.4, 0.2]
# optimizer_config = dict(grad_clip=dict(max_norm=30, norm_type=2))
# # 三阶段 CosineRestart：与 schedule 等长 periods，每阶段末降到 min_lr 后在下一阶段初回到 base_lr
# lr_config = dict(
#     policy='CosineRestart',
#     periods=lr_curriculum_periods,
#     restart_weights=lr_curriculum_restart_weights,
#     by_epoch=True,
#     warmup='linear',
#     warmup_iters=2500,
#     warmup_ratio=1.0 / 3,
#     min_lr=2e-4)

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
    min_lr=2e-4)

runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)

# load_from = 'ckpts/best_SSC_mean_epoch_13.pth'
work_dir = './work_dirs/unilidar_seg_pointseggrad_repeatnutrain_sixhardrandomLCur_DG_film_seesaw_rereconsistency_RPR'    
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='LiDARCurriculumHook', schedule=schedule),
        dict(
            type='SwanlabLoggerHook',
            init_kwargs=dict(
                project='UniLiDAR_seg',
                name='unilidar_seg_pointseggrad_repeatnutrain_sixhardrandomLCur_DG_film_seesaw_rereconsistency_RPR_4*A800',
                job_type='training',
                notes='ori_lr, re_consistency, pointseggrad',
                config = {"lr": optimizer.get('lr', 5e-4), "total_batch_size": data.get('samples_per_gpu', 1) * 4, "total_epoch": runner.get('max_epochs', 24), "weight_decay": optimizer.get('weight_decay', 0.01)}
            ))
    ])

checkpoint_config = dict(interval=2)




