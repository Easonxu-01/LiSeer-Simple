# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Xiaofeng Wang
# ---------------------------------------------
import random
import warnings
import sys
import copy
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info, load_checkpoint)
from mmcv.utils import build_from_cfg

from mmdet.core import EvalHook

from mmdet.datasets import (build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
import time
import os.path as osp
from projects.unilidar_plugin.datasets.builder import build_dataloader
from projects.unilidar_plugin.core.evaluation.eval_hooks import OccDistEvalHook, OccEvalHook
from projects.unilidar_plugin.datasets import custom_build_dataset, ConcatenatedDataset
from mmdet3d.datasets import build_dataset
sys.setrecursionlimit(1000000)

import os

os.environ['JOBLIB_TEMP_FOLDER'] = '/data1' 

# import os, pandarallel

#clean up the /tmp folder	
# if os.path.isdir("/tmp") : 
#     os.system('rm -R /tmp/*')

# os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'	
# pandarallel.initialize(nb_workers = int(os.cpu_count())-1, use_memory_fs = False , progress_bar=True,verbose=2 ) 

def custom_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    
    logger = get_root_logger(cfg.log_level)
    
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    
    
    
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        ) for ds in dataset
    ]

    # 构建 student / teacher 模型（用于 d2skd 蒸馏）
    use_d2skd = getattr(cfg, 'd2skd', False)
    model_s = model  # student
    model_t = None   # teacher

    if use_d2skd:
        # 复制一份结构完全相同的 teacher 模型
        model_t = copy.deepcopy(model_s)

        # teacher 的权重：优先使用 cfg.d2skd_load_from，没有则回退到 cfg.load_from
        teacher_ckpt = getattr(cfg, 'd2skd_load_from', None)
        if teacher_ckpt is None:
            teacher_ckpt = getattr(cfg, 'load_from', None)

        if teacher_ckpt is not None:
            load_checkpoint(model_t, teacher_ckpt, map_location='cpu')
        else:
            logger.warning('d2skd = True 但未提供 teacher checkpoint (d2skd_load_from / load_from)，'
                           'teacher 将使用随机初始化参数。')

        # 冻结 teacher 参数，仅做 inference
        for p in model_t.parameters():
            p.requires_grad = False
        model_t.eval()
        
        # 设置 dense_train：student 为 False，teacher 为 True
        # 在模型包装前设置，确保设置生效
        if hasattr(model_s, 'tpv_aggregator'):
            model_s.tpv_aggregator.dense_train = False
            model_s.tpv_aggregator.d2skd = False
        if hasattr(model_t, 'tpv_aggregator'):
            model_t.tpv_aggregator.dense_train = True
            model_t.tpv_aggregator.d2skd = True
            model_t.dense_train = True
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model_s = MMDistributedDataParallel(
            model_s.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if use_d2skd and model_t is not None:
            # Teacher 模型所有参数都被冻结，不需要 DDP，直接放到 GPU 上即可
            # 注意：teacher 只用于 inference，不需要梯度同步
            model_t = model_t.cuda()
    else:
        model_s = MMDataParallel(
            model_s.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if use_d2skd and model_t is not None:
            # Teacher 模型所有参数都被冻结，不需要 DataParallel，直接放到 GPU 上即可
            model_t = model_t.cuda(cfg.gpu_ids[0])

    # 之后逻辑仍然以 model 指向 student 以保持兼容
    model = model_s
    
    # 分布式下，同步各进程的模型参数（包括 teacher）
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            # Rank 0 的进程负责初始化参数，其他进程将从这里同步参数
            pass
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        if use_d2skd and model_t is not None:
            for param in model_t.parameters():
                dist.broadcast(param.data, src=0)
    
    # build runner
    # 只对 student 构建优化器，teacher 始终冻结
    optimizer = build_optimizer(model, cfg.optimizer)

    assert 'runner' in cfg
    if use_d2skd:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                model_t=model_t,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
    else:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting TODO
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            assert NotImplementedError()
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        )
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_cfg['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        eval_hook = OccDistEvalHook if distributed else OccEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            # hook = build_from_cfg(hook_cfg, HOOKS) 
            # FIXME hardcode specifying dataloader as parameter 
            # val_dataloader 只在 validate=True 时定义
            hook_kwargs = {}
            if validate and 'val_dataloader' in locals():
                hook_kwargs['dataloader'] = val_dataloader
            hook = build_from_cfg(hook_cfg, HOOKS, hook_kwargs) 
            runner.register_hook(hook, priority=priority)

    # 将 teacher 模型挂到 runner 上以供后续使用
    if use_d2skd and model_t is not None:
        # runner.model 仍然是 student；额外挂一个 model_t
        runner.model_t = model_t

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def custom_train_multidb_detector(model,
                   dataset_1,
                   dataset_2,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    
    logger = get_root_logger(cfg.log_level)
    
    dataset_1 = dataset_1 if isinstance(dataset_1, (list, tuple)) else [dataset_1]
    dataset_2 = dataset_2 if isinstance(dataset_2, (list, tuple)) else [dataset_2]
    
    merged_dataset = ConcatenatedDataset(dataset_1, dataset_2)
    # merged_dataset = [dataset_1[0], dataset_2[0]]
    
    data_loaders = [
        build_dataloader(
            merged_dataset,
            cfg.data_merge.samples_per_gpu,
            cfg.data_merge.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data_merge.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data_merge.nonshuffler_sampler,  # dict(type='DistributedSampler'),
            drop_last=True,
        )
    ]
    

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        # model._set_static_graph()
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)


    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    assert 'runner' in cfg
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting TODO
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data_nu.val.pop('samples_per_gpu', 1)
        
        # if val_samples_per_gpu > 1:
        #     assert NotImplementedError()
        #     # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        #     cfg.data_nu.val.pipeline = replace_ImageToTensor(
        #         cfg.data_nu.val.pipeline)
        #     cfg.data_sk.val.pipeline = replace_ImageToTensor(
        #         cfg.data_sk.val.pipeline)
        # val_dataset_1 = [build_dataset(cfg.data_nu.val, dict(test_mode=True))]
        # val_dataset_2 = [build_dataset(cfg.data_sk.val, dict(test_mode=True))]
        # val_dataset_1 = val_dataset_1 if isinstance(val_dataset_1, (list, tuple)) else [val_dataset_1]
        # val_dataset_2 = val_dataset_2 if isinstance(val_dataset_2, (list, tuple)) else [val_dataset_2]

        # val_dataset_merge = ConcatenatedDataset(val_dataset_1, val_dataset_2)

        # val_dataloader = build_dataloader(
        #     val_dataset,
        #     samples_per_gpu=val_samples_per_gpu,
        #     workers_per_gpu=cfg.data_merge.workers_per_gpu,
        #     dist=distributed,
        #     shuffle=False,
        #     shuffler_sampler=cfg.data_merge.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
        #     nonshuffler_sampler=cfg.data_merge.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        #     drop_last=True,
        # )

        
        # eval_cfg_nu = cfg.get('evaluation_nu', {})
        # eval_cfg_nu['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        # eval_cfg_nu['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        # eval_cfg_sk = cfg.get('evaluation_sk', {})
        # eval_cfg_sk['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        # eval_cfg_sk['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        # eval_hook = OccDistEvalHook if distributed else OccEvalHook
        # runner.register_hook(eval_hook(val_dataloader, **eval_cfg_nu))
        # runner.register_hook(eval_hook(val_dataloader, **eval_cfg_sk))
        if val_samples_per_gpu > 1:
            assert NotImplementedError()
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data_merge.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            shuffler_sampler=cfg.data_merge.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data_merge.nonshuffler_sampler,  # dict(type='DistributedSampler'),
            drop_last=True,
        )
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_cfg['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        eval_hook = OccDistEvalHook if distributed else OccEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            # hook = build_from_cfg(hook_cfg, HOOKS) 
            # FIXME hardcode specifying dataloader as parameter 
            # val_dataloader 只在 validate=True 时定义
            hook_kwargs = {}
            if validate and 'val_dataloader' in locals():
                hook_kwargs['dataloader'] = val_dataloader
            hook = build_from_cfg(hook_cfg, HOOKS, hook_kwargs) 
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
