# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from .sampler import SAMPLER
import random
from IPython import embed


@SAMPLER.register_module()
class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[list(
                    torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

@SAMPLER.register_module()
class BalancedDistributedGroupSampler(Sampler):
    """
    Ensures every distributed batch draws from both source datasets. When one dataset
    is shorter than the other it is over-sampled with replacement to fill the batch.
    """

    def __init__(self, dataset, samples_per_gpu=2, num_replicas=None, rank=None, seed=0):
        self.rank, self.num_replicas = get_dist_info()
        if num_replicas is not None:
            self.num_replicas = num_replicas
        if rank is not None:
            self.rank = rank

        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.seed = seed
        self.epoch = 0

        assert hasattr(self.dataset, 'dataflag'), "Dataset must have a 'dataflag' attribute"
        self.flag = self.dataset.dataflag
        self.dataset1_indices = np.where(self.flag == 0)[0]
        self.dataset2_indices = np.where(self.flag == 1)[0]

        self.total_samples = len(self.dataset1_indices) + len(self.dataset2_indices)
        self.total_size = self.total_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed + self.rank)

        # Round the per-dataset index counts up so they divide evenly across GPUs,
        # repeating indices where necessary.
        total_samples_per_dataset = max(len(self.dataset1_indices), len(self.dataset2_indices))
        total_samples_needed = total_samples_per_dataset * 2
        total_samples_per_gpu = self.samples_per_gpu * self.num_replicas

        if total_samples_needed % total_samples_per_gpu != 0:
            total_samples_needed += total_samples_per_gpu - (total_samples_needed % total_samples_per_gpu)

        # Over-sample so both datasets contribute an equal number of indices.
        dataset1_indices = np.random.choice(self.dataset1_indices, total_samples_needed // 2, replace=True)
        dataset2_indices = np.random.choice(self.dataset2_indices, total_samples_needed // 2, replace=True)

        np.random.shuffle(dataset1_indices)
        np.random.shuffle(dataset2_indices)

        # Interleave the two datasets so each batch mixes both.
        indices = np.empty(total_samples_needed, dtype=int)
        indices[0::2] = dataset1_indices[:total_samples_needed // 2]
        indices[1::2] = dataset2_indices[:total_samples_needed // 2]

        # Slice out this replica's share.
        indices_per_replica = len(indices) // self.num_replicas
        offset = indices_per_replica * self.rank
        indices = indices[offset:offset + indices_per_replica]

        return iter(indices)

    def __len__(self):
        # Length is per replica, matching the iterations actually run each epoch.
        return 2 * max(len(self.dataset1_indices),len(self.dataset2_indices)) // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch
