_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

dataset_type = 'NuscOCCDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
plugin = True
plugin_dir = "projects/unilidar_plugin/"
img_norm_cfg = None
occ_path = "./data/nuScenes-Occupancy"
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
unilidar=False
cylinder=False
RPR = True
coor_alignment = False
ori_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] if cylinder else [0, -25.6, -3.4, 51.2, 25.6, 3.0]
# restrict_point_cloud_range = [-25.6, -25.6, -3.4, 25.6, 25.6, 3.0]
occ_size = [512, 512, 40] # ground truth
final_occ_size =  [256, 256, 32]# model output
voxel_channels = [80, 160, 320, 640]
model_empty_idx = 0  # noise 0-->255
empty_idx = 0  # noise 0-->255
num_cls_nu = 17  # 0 free, 1-16 obj
num_cls_sk = 20  # 0 free, 1-16 obj
visible_mask = True

sample_from_voxel = True
sample_from_img = False

numC_Trans = 80
init_size = 64
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)

find_unused_parameters = False

num_beams = 32
start_vertical_angle=-30.67
end_vertical_angle=10.67    
num_beam_points=1080

spatial_shape = [256, 256, 32] if cylinder else [1024, 1024, 128] #model init voxel spatial shape
# voxel_size = [(point_cloud_range[3]-point_cloud_range[0])/spatial_shape[0], \
#                 (point_cloud_range[4]-point_cloud_range[1])/spatial_shape[1], \
#                 (point_cloud_range[5]-point_cloud_range[2])/spatial_shape[2]] #model init voxel size
voxel_size=[0.05,0.05,0.05]
cascade_ratio = final_occ_size[0] * 8 // spatial_shape[0] 
dataset_flag_nu = 1 if dataset_type == 'NuscOCCDataset' else 2

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
        norm_cfg=dict(type='UniNorm3d', dataset_from_flag=dataset_flag_nu, eps=1e-3, momentum=0.01, voxel_coord=True),
    ),
    occ_encoder_neck=dict(
        type='FPN3D',
        with_cp=True,
        in_channels=voxel_channels,
        out_channels=voxel_out_channel,
        norm_cfg=dict(type='UniNorm3d', dataset_from_flag=dataset_flag_nu, eps=1e-3, momentum=0.01, voxel_coord=True),
    ),
    pts_bbox_head=dict(
        type='OccHead',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        cascade_ratio=cascade_ratio,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        final_occ_size=final_occ_size,
        fine_topk=15000,
        dual = unilidar,
        empty_idx= model_empty_idx,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=[num_cls_nu,num_cls_sk] if unilidar else num_cls_nu,
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
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        dual=True,
        num_cls=[num_cls_nu, num_cls_sk] if unilidar else False,),
    empty_idx=model_empty_idx,
    spatial_shape=spatial_shape,
)
    # pts_middle_encoder=dict(
    #     type='PointVoxelEnc',
    #     input_channel=4, 
    #     init_size=init_size,
    #     spatial_shape=spatial_shape,
    #     ),
    # occ_fuser=dict(
    #     type='PVcrossattention',
    #     num_beams=num_beams,
    #     start_vertical_angle=start_vertical_angle,
    #     end_vertical_angle=end_vertical_angle,
    #     num_beam_points=num_beam_points,  
    #     feat_dim=init_size * 16,
    #     dim=init_size * 4,
    #     spatial_shape=[spatial_shape[0]//16, spatial_shape[1]//16, spatial_shape[2]//4] if cylinder else [spatial_shape[0]//8, spatial_shape[1]//8, spatial_shape[2]//8],
    #     qkv_bias=True,
    #     heads=4,
    #     dim_head=32,
    #     skip=False,
    # ),


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
        load_dim=5,
        use_dim=5,
        shift_height=coor_alignment,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=[0, 0, -0.2],),
    dict(type='LoadPointsFromMultiSweeps_RPR',
        sweeps_num=10,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        input_modality=input_modality),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, cylinder=cylinder, occ_path=occ_path, grid_size=occ_size, use_vel=False,
            unoccupied=empty_idx, pc_range=ori_point_cloud_range, RPR= RPR, restrict_pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3Dinput', keys=['gt_occ', 'points', 'dataset_flag']),
]
    # dict(type='cylinder_voxelize',rotate_aug=True, flip_aug=True, scale_aug=True, transform_aug=True,
    #     fixed_volume_space = True, max_volume_space = [50, 3.1415926, 3], min_volume_space = [0, -3.1415926, -5], grid_size = [512, 360, 32],
    #     pc_range=point_cloud_range),
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
        shift_coors=[0, 0, -0.2],),
    dict(type='LoadPointsFromMultiSweeps_RPR',
        sweeps_num=10,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        input_modality=input_modality,
        is_train=False),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, cylinder=cylinder, occ_path=occ_path, grid_size=occ_size, use_vel=False,
            unoccupied=empty_idx, pc_range=ori_point_cloud_range, RPR= RPR, restrict_pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3Dinput', keys=['gt_occ', 'points', 'dataset_flag', 'visible_mask'],meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
]
    # dict(type='cylinder_voxelize',rotate_aug=True, flip_aug=True, scale_aug=True, transform_aug=True,
    #     fixed_volume_space = True, max_volume_space = [50, 3.1415926, 3], min_volume_space = [0, -3.1415926, -5], grid_size = [512, 360, 32],
    #     pc_range=point_cloud_range),

test_config=dict(
    type=dataset_type,
    occ_root=occ_path,
    data_root=data_root,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    random=False,
)

train_config=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_path,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        box_type_3d='LiDAR',
        random=False,),

data = dict(
    samples_per_gpu=2,
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

optimizer_config = dict(grad_clip=dict(max_norm=30, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
# lr_config = dict(
#     policy='CosineRestart',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     periods=[5, 4, 3, 2, 1],
#     restart_weights=[5, 4, 3, 2, 1],
#     min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)
load_from = 'ckpts/unilidar_nu_RPRmin_12812816_nu_38.5_23.7.pth'
work_dir = './work_dirs/unilidar_nu_RPRmin_bbnorm_12812816'    
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='mm_UniLiDAR',
                name='unilidar_nu_RPRmin_bbnorm_12812816',
                job_type='training',
                notes='the-first-test',
                config = {"lr": 5e-4, "total_batch_size": 32, "total_epoch": 24, "weight_decay": 0.01}
            )
        )
    ])

checkpoint_config = dict(interval=2)

custom_hooks = [
    dict(type='OccEfficiencyHook'),
]

