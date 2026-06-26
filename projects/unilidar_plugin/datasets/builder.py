
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader

from mmdet.datasets.samplers import GroupSampler
from projects.unilidar_plugin.datasets.samplers.group_sampler import DistributedGroupSampler
from projects.unilidar_plugin.datasets.samplers.distributed_sampler import DistributedSampler
from projects.unilidar_plugin.datasets.samplers.sampler import build_sampler

def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     shuffler_sampler=None,
                     nonshuffler_sampler=None,
                     drop_last=True,
                     **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    
    # 添加调试信息
    # print(f"[DEBUG] Dataset length: {len(dataset)}")
    # if hasattr(dataset, 'flag'):
    #     print(f"[DEBUG] Dataset flag size: {dataset.flag.size}")
    #     print(f"[DEBUG] Group sizes: {np.bincount(dataset.flag)}")
    # print(f"[DEBUG] World size: {world_size}, Rank: {rank}")
    # print(f"[DEBUG] Samples per GPU: {samples_per_gpu}")
    
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = build_sampler(shuffler_sampler if shuffler_sampler is not None else dict(type='DistributedGroupSampler'),
                                     dict(
                                         dataset=dataset,
                                         samples_per_gpu=samples_per_gpu,
                                         num_replicas=world_size,
                                         rank=rank,
                                         seed=seed)
                                     )

        else:
            sampler = build_sampler(nonshuffler_sampler if nonshuffler_sampler is not None else dict(type='DistributedSampler'),
                                     dict(
                                         dataset=dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=shuffle,
                                         seed=seed)
                                     )

        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        print('WARNING!!!!, Only can be used for obtain inference speed!!!!')
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=True,
        drop_last=drop_last,
        prefetch_factor=4,
        persistent_workers=True,
        worker_init_fn=init_fn,
        **kwargs)

    # # 添加dataloader长度调试信息
    # print(f"[DEBUG] DataLoader length: {len(data_loader)}")
    # if sampler is not None:
    #     print(f"[DEBUG] Sampler length: {len(sampler)}")

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Copyright (c) OpenMMLab. All rights reserved.
import platform
from mmcv.utils import Registry, build_from_cfg

from mmdet.datasets import DATASETS
from mmdet.datasets.builder import _concat_dataset

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry('Object sampler')


def custom_build_dataset(cfg, default_args=None):
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                 ConcatDataset, RepeatDataset)
    if isinstance(cfg, (list, tuple)):
        # list/tuple 形式的 cfg：把多个子 cfg 各自构建后用 ConcatDataset 拼接。
        # 若第一个子 cfg 提供了 `repeat` 字段（>1），则把列表整体复制 repeat 次，
        # 实现 "把这组数据集重复 N 倍" 的语义；不指定时退化为简单 concat。
        sub_datasets = [custom_build_dataset(c, default_args) for c in cfg]
        first = cfg[0] if len(cfg) > 0 else None
        repeat_n = None
        if first is not None and hasattr(first, 'get'):
            repeat_n = first.get('repeat', None)
        if repeat_n is not None and int(repeat_n) > 1:
            sub_datasets = sub_datasets * int(repeat_n)
        dataset = ConcatDataset(sub_datasets)
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            custom_build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            custom_build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset':
        dataset = CBGSDataset(custom_build_dataset(cfg['dataset'], default_args))
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        # 通用 `repeat` 支持：单 dict 配置中若顶层带有 `repeat` 字段（>1），
        # 则在构建底层 dataset 后用 RepeatDataset 包一层，使其 __len__ 变为
        # 原长度 * repeat（flag 也会被自动 tile，与 DistributedGroupSampler 兼容）。
        # 同时无论 repeat 是否生效，都会从 cfg 中剥离该字段，避免传给那些
        # __init__ 不接受 `repeat` 的数据集类（如 ZLTWaymoDataset）时报错。
        repeat = cfg.get('repeat', None)
        if repeat is not None:
            cfg = copy.deepcopy(cfg)
            cfg.pop('repeat')
        base_dataset = build_from_cfg(cfg, DATASETS, default_args)
        if repeat is not None and int(repeat) > 1:
            dataset = RepeatDataset(base_dataset, int(repeat))
        else:
            dataset = base_dataset

    return dataset
