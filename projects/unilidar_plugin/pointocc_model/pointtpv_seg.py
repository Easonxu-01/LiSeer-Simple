import torch
import torch.distributed as dist
from mmdet3d.models import builder
from mmdet.models import DETECTORS
from mmdet3d.models.detectors import CenterPoint
import yaml
import numpy as np
import copy
import random
import math
import open3d as o3d
import os
import os.path as osp
from mmcv.parallel import DataContainer
from torch_scatter import scatter_min
from .Spectraldistiller import TPVLowFreqSpectralDistiller
from .tpv_aggregator import _cb_adc_consistency_loss
import torch.nn.functional as F

@DETECTORS.register_module()
class PointTPV_Seg(CenterPoint):

    def __init__(self,
                 lidar_tokenizer=None,
                 lidar_backbone=None,
                 lidar_neck=None,
                 tpv_aggregator=None,
                 empty_idx = None,
                 random = False,
                 reset_random_prob = 0.0,
                 sensor=None,
                 consistency = False,
                 dense_train = False,
                 d2skd = False,
                 dynamic_loss_balance=None,
                 **kwargs,
                 ):

        super().__init__()

        if lidar_tokenizer:
            self.lidar_tokenizer = builder.build_middle_encoder(lidar_tokenizer)
        if lidar_backbone:
            self.lidar_backbone = builder.build_backbone(lidar_backbone)
        if lidar_neck:
            self.lidar_neck = builder.build_neck(lidar_neck)
        if tpv_aggregator:
            self.tpv_aggregator = builder.build_voxel_encoder(tpv_aggregator)

        self.fp16_enabled = False
        self.empty_idx = empty_idx
        self.random = random
        self.reset_random_prob = float(reset_random_prob)
        self.sensor_config = sensor
        self.consistency = consistency
        self.dense_train = dense_train
        self.d2skd = d2skd
        self.dynamic_loss_balance = dynamic_loss_balance if dynamic_loss_balance is not None else {}
        self.sample_repeat_count = 2
        # GradNorm-style dynamic multi-loss balancing (group-level)
        # task order is fixed to keep state deterministic.
        self._balance_task_names = ['point', 'voxel', 'sensor', 'consistency']
        self._balance_eps = 1e-6
        self._balance_enabled = bool(self.dynamic_loss_balance.get('enabled', True))
        self._balance_alpha = float(self.dynamic_loss_balance.get('alpha', 0.5))
        self._balance_step = float(self.dynamic_loss_balance.get('step', 0.05))
        self._balance_momentum = float(self.dynamic_loss_balance.get('momentum', 0.2))
        self._balance_min_weight = float(self.dynamic_loss_balance.get('min_weight', 0.2))
        self._balance_max_weight = float(self.dynamic_loss_balance.get('max_weight', 5.0))
        self._balance_ema_beta = float(self.dynamic_loss_balance.get('ema_beta', 0.9))
        self._voxel_guard_enabled = bool(self.dynamic_loss_balance.get('voxel_guard_enabled', True))
        self._voxel_guard_tol = float(self.dynamic_loss_balance.get('voxel_guard_tol', 0.02))
        self._voxel_guard_gain = float(self.dynamic_loss_balance.get('voxel_guard_gain', 0.6))
        base_weights_cfg = self.dynamic_loss_balance.get('base_weights', {})
        task_min_cfg = self.dynamic_loss_balance.get('task_min_weights', {})
        task_max_cfg = self.dynamic_loss_balance.get('task_max_weights', {})
        base_w = []
        task_min_w = []
        task_max_w = []
        for name in self._balance_task_names:
            base_w.append(float(base_weights_cfg.get(name, 1.0)))
            task_min_w.append(float(task_min_cfg.get(name, self._balance_min_weight)))
            task_max_w.append(float(task_max_cfg.get(name, self._balance_max_weight)))
        self.register_buffer(
            "_balance_weights",
            torch.ones(len(self._balance_task_names), dtype=torch.float32),
        )
        self.register_buffer(
            "_balance_initial_losses",
            torch.zeros(len(self._balance_task_names), dtype=torch.float32),
        )
        self.register_buffer(
            "_balance_base_weights",
            torch.tensor(base_w, dtype=torch.float32),
        )
        self.register_buffer(
            "_balance_task_min_weights",
            torch.tensor(task_min_w, dtype=torch.float32),
        )
        self.register_buffer(
            "_balance_task_max_weights",
            torch.tensor(task_max_w, dtype=torch.float32),
        )
        self.register_buffer(
            "_balance_loss_ema",
            torch.zeros(len(self._balance_task_names), dtype=torch.float32),
        )
        self.register_buffer(
            "_reset_random_sync",
            torch.zeros(1, dtype=torch.int32),
            persistent=False,
        )
        if self.d2skd:
            self.spectral_distiller = TPVLowFreqSpectralDistiller()
        # Registered as buffers so the tensors are built once and follow the module device.
        min_bound = torch.tensor([0, -3.1415926, -3.4], dtype=torch.float32)
        max_bound = torch.tensor([51.2, 3.1415926, 3], dtype=torch.float32)
        grid_size = torch.tensor([480, 360, 32], dtype=torch.float32)
        grid_size_vox = torch.tensor([480, 360, 32], dtype=torch.float32)
        grid_size_vox = torch.tensor([480, 360, 32], dtype=torch.float32)
        self.register_buffer("min_bound", min_bound)
        self.register_buffer("max_bound", max_bound)
        self.register_buffer("grid_size", grid_size)
        self.register_buffer("grid_size_vox", grid_size_vox)
        # build_model(..., test_cfg=...) passes via kwargs; super().__init__() does not
        # receive them, so keep detector-level cfg here for inference (e.g. save preds).
        self.train_cfg = kwargs.get('train_cfg', None)
        self.test_cfg = kwargs.get('test_cfg', None)

    def _group_loss_keys(self, loss_dict):
        group_keys = {name: [] for name in self._balance_task_names}
        for key in loss_dict.keys():
            if not key.startswith('loss_'):
                continue
            if key.startswith('loss_point_'):
                group_keys['point'].append(key)
            elif key.startswith('loss_voxel_'):
                group_keys['voxel'].append(key)
            elif key == 'loss_sensor':
                group_keys['sensor'].append(key)
            elif key == 'loss_consistency':
                group_keys['consistency'].append(key)
        return group_keys

    def _get_reference_param(self):
        # Prefer a TPV aggregator parameter as shared representation proxy.
        for p in self.tpv_aggregator.parameters():
            if p.requires_grad:
                return p
        return None

    def _apply_dynamic_loss_balance(self, loss_dict):
        if (not self._balance_enabled) or (not isinstance(loss_dict, dict)):
            return loss_dict
        if not any(k.startswith('loss_') for k in loss_dict.keys()):
            return loss_dict

        group_keys = self._group_loss_keys(loss_dict)
        grouped_losses = {}
        active_indices = []
        for idx, name in enumerate(self._balance_task_names):
            keys = group_keys.get(name, [])
            if len(keys) == 0:
                continue
            loss_sum = None
            for k in keys:
                v = loss_dict[k]
                if torch.is_tensor(v):
                    loss_sum = v if loss_sum is None else (loss_sum + v)
            if loss_sum is None or (not loss_sum.requires_grad):
                continue
            grouped_losses[name] = loss_sum
            active_indices.append(idx)

        if len(active_indices) <= 1:
            return loss_dict

        ref_param = self._get_reference_param()
        if ref_param is None:
            return loss_dict

        grad_norms = {}
        for name, loss_val in grouped_losses.items():
            grad = torch.autograd.grad(
                loss_val,
                ref_param,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                grad_norms[name] = loss_val.detach().new_tensor(0.0)
            else:
                grad_norms[name] = grad.detach().norm(p=2)

        with torch.no_grad():
            # Only update weights for tasks with sufficiently large current loss.
            # Tasks with tiny loss keep their previous weights unchanged.
            update_indices = []
            no_update_loss_thresh = 1e-4
            for idx in active_indices:
                n = self._balance_task_names[idx]
                l_cur = grouped_losses[n].detach().abs().to(self._balance_weights.device)
                if float(l_cur.item()) >= no_update_loss_thresh:
                    update_indices.append(idx)

            if len(update_indices) > 1:
                update_idx_tensor = torch.tensor(update_indices, device=self._balance_weights.device, dtype=torch.long)

                # Initialize L0 for tasks when they first appear.
                for idx in update_indices:
                    if self._balance_initial_losses[idx] <= 0:
                        n = self._balance_task_names[idx]
                        self._balance_initial_losses[idx] = grouped_losses[n].detach().abs().clamp_min(self._balance_eps)
                    curr_loss = grouped_losses[self._balance_task_names[idx]].detach().abs().to(self._balance_weights.device)
                    if self._balance_loss_ema[idx] <= 0:
                        self._balance_loss_ema[idx] = curr_loss
                    else:
                        self._balance_loss_ema[idx] = (
                            self._balance_ema_beta * self._balance_loss_ema[idx]
                            + (1.0 - self._balance_ema_beta) * curr_loss
                        )

                grad_stack = []
                rel_stack = []
                for idx in update_indices:
                    n = self._balance_task_names[idx]
                    g = grad_norms[n].to(self._balance_weights.device)
                    grad_stack.append(g)
                    l_cur = grouped_losses[n].detach().abs().to(self._balance_weights.device)
                    l0 = self._balance_initial_losses[idx].clamp_min(self._balance_eps)
                    rel_stack.append(l_cur / l0)

                grad_stack = torch.stack(grad_stack)
                rel_stack = torch.stack(rel_stack)
                grad_avg = grad_stack.mean().clamp_min(self._balance_eps)
                rel_rate = rel_stack / rel_stack.mean().clamp_min(self._balance_eps)
                target_grad = grad_avg * torch.pow(rel_rate, self._balance_alpha)

                curr_w = self._balance_weights.index_select(0, update_idx_tensor)
                update_ratio = (target_grad / grad_stack.clamp_min(self._balance_eps)).clamp(0.2, 5.0)
                new_w = curr_w * torch.pow(update_ratio, self._balance_step)
                new_w = torch.clamp(new_w, min=self._balance_min_weight, max=self._balance_max_weight)
                new_w = new_w / new_w.mean().clamp_min(self._balance_eps)
                blended_w = (1.0 - self._balance_momentum) * curr_w + self._balance_momentum * new_w
                # Apply task priors and per-task bounds.
                base_w_active = self._balance_base_weights.index_select(0, update_idx_tensor)
                min_w_active = self._balance_task_min_weights.index_select(0, update_idx_tensor)
                max_w_active = self._balance_task_max_weights.index_select(0, update_idx_tensor)
                blended_w = blended_w * base_w_active
                blended_w = blended_w / blended_w.mean().clamp_min(self._balance_eps)
                blended_w = torch.max(torch.min(blended_w, max_w_active), min_w_active)

                # Voxel-rise guard: if voxel loss is rising above EMA, boost voxel weight.
                if self._voxel_guard_enabled and ('voxel' in grouped_losses):
                    voxel_idx = self._balance_task_names.index('voxel')
                    pos = (update_idx_tensor == voxel_idx).nonzero(as_tuple=False)
                    if pos.numel() > 0:
                        v_pos = int(pos[0].item())
                        voxel_cur = grouped_losses['voxel'].detach().abs().to(self._balance_weights.device)
                        voxel_ema = self._balance_loss_ema[voxel_idx].clamp_min(self._balance_eps)
                        if voxel_cur > voxel_ema * (1.0 + self._voxel_guard_tol):
                            rise_ratio = voxel_cur / voxel_ema
                            boost = 1.0 + self._voxel_guard_gain * (rise_ratio - 1.0)
                            boost = torch.clamp(boost, min=1.0, max=2.5)
                            boosted = blended_w[v_pos] * boost
                            blended_w[v_pos] = torch.clamp(boosted, min=min_w_active[v_pos], max=max_w_active[v_pos])
                            blended_w = blended_w / blended_w.mean().clamp_min(self._balance_eps)
                            blended_w = torch.max(torch.min(blended_w, max_w_active), min_w_active)

                self._balance_weights.index_copy_(0, update_idx_tensor, blended_w)

        # Apply group weights to individual losses and log current weights.
        for idx, name in enumerate(self._balance_task_names):
            keys = group_keys.get(name, [])
            if len(keys) == 0:
                continue
            w = self._balance_weights[idx]
            for k in keys:
                if torch.is_tensor(loss_dict[k]):
                    loss_dict[k] = loss_dict[k] * w
            loss_dict[f'balancer_w_{name}'] = w.detach()

        return loss_dict

    def extract_feat(self, train_grid, voxel_label):
        """Extract features of points."""
        x_3view = self.lidar_tokenizer(train_grid, voxel_label)
        tpv_list = []
        x_tpv = self.lidar_backbone(x_3view)
        for x in x_tpv:
            x = self.lidar_neck(x)
            if not isinstance(x, torch.Tensor):
                x = x[0]
            tpv_list.append(x)
        return tpv_list

    def _should_reset_random(self, reset_random):
        """Return True only if the whole local batch was loaded as raw data."""
        if reset_random is None:
            return False
        if isinstance(reset_random, torch.Tensor):
            if reset_random.numel() == 0:
                return False
            return bool((reset_random != 0).all().item())
        if isinstance(reset_random, (list, tuple)):
            if len(reset_random) == 0:
                return False
            return all(self._should_reset_random(v) for v in reset_random)
        return bool(reset_random)

    def _all_ranks_reset_random_flag(self, local_reset_random):
        """Only skip model-side sampling when every DDP rank loaded a fully raw batch."""
        if not (dist.is_available() and dist.is_initialized()):
            return local_reset_random
        flag = self._reset_random_sync
        flag.fill_(1 if local_reset_random else 0)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def forward_train(self,
                train_grid=None,
                grid_ind=None,
                grid_ind_vox=None,
                train_voxel_label=None,
                train_pts_label=None,
                return_loss=True,
                dataset_flag= None,
                reset_random=None,
                model_t=None,
                **kwargs
        ):
        """Forward training function.
        """
        sensor_params = None
        # Un-resampled inputs, kept for the distillation teacher.
        grid_ind_orig = grid_ind
        grid_ind_vox_orig = grid_ind_vox
        x_lidar_tpv_t = None
        loss_distill = None

        train_grid_orig = train_grid
        local_reset_random = self._should_reset_random(reset_random)
        reset_random_flag = self._all_ranks_reset_random_flag(local_reset_random) if self.random else False
        enable_model_random = self.random and (not reset_random_flag)
        if enable_model_random:
            train_grid, grid_ind, grid_ind_vox, train_pts_label, sensor_params = self._sampling(train_grid, grid_ind, grid_ind_vox, train_pts_label)

        # CBA-CL runs inside the aggregator's own consistency branch, where both views
        # receive gradients. Here we only extract features for the whole batch (both
        # views at once); pairing and the consistency loss stay in the aggregator.
        # Reached only when random sampling produced 2x samples.
        if self.consistency and enable_model_random and isinstance(grid_ind, list) and len(grid_ind) >= 2:
            repeat = max(1, getattr(self, "sample_repeat_count", 2))
            total_samples = len(grid_ind)
            if total_samples % repeat != 0:
                repeat = total_samples
            base_bs = total_samples // repeat

            # One backbone pass over the full batch, paired views included.
            x_lidar_tpv = self.extract_feat(train_grid, grid_ind)

            # The aggregator indexes per view: points / point_labels / sensor are per-view.
            points_for_agg = grid_ind_vox if grid_ind_vox else grid_ind

            # voxel_label and dataset_flag are per-scene and are not duplicated by the
            # sampler, so expand them per view; both views share the scene's label/flag.
            per_view_voxel = []
            per_view_flag = []
            for b in range(base_bs):
                if isinstance(train_voxel_label, list):
                    vl = train_voxel_label[b]
                else:
                    vl = train_voxel_label[b:b+1] if train_voxel_label is not None else None
                if isinstance(vl, torch.Tensor) and vl.dim() == 3:
                    vl = vl.unsqueeze(0)
                if isinstance(dataset_flag, list):
                    df = dataset_flag[b]
                elif isinstance(dataset_flag, torch.Tensor):
                    df = dataset_flag[b:b+1] if b < dataset_flag.shape[0] else dataset_flag
                else:
                    df = dataset_flag
                for _ in range(repeat):
                    per_view_voxel.append(vl)
                    per_view_flag.append(df)

            agg_loss = self.tpv_aggregator(
                x_lidar_tpv,
                points=points_for_agg,
                voxel_label=per_view_voxel,
                point_labels=train_pts_label,
                dataset_flag=per_view_flag,
                return_loss=True,
                sensor_vec=sensor_params,
                skip_consistency=False,
            )
            agg_loss = self._apply_dynamic_loss_balance(agg_loss)
            return agg_loss

        if model_t is not None and self.d2skd:
            # Teacher runs entirely under no_grad to avoid holding a graph in memory.
            with torch.no_grad():
                x_lidar_tpv_t = model_t.extract_feat(
                    train_grid=train_grid_orig,
                    voxel_label=grid_ind_orig
                )
            x_lidar_tpv = self.extract_feat(
                train_grid=train_grid,
                voxel_label=grid_ind
            )
            loss_distill = self.spectral_distiller(x_lidar_tpv, x_lidar_tpv_t)
        else:
            x_lidar_tpv = self.extract_feat(
                train_grid=train_grid,
                voxel_label=grid_ind
            )
        grid_ind_ = grid_ind_vox if grid_ind_vox is not None else grid_ind
        if isinstance(grid_ind_, list):
            if all(pl.shape == grid_ind_[0].shape for pl in grid_ind_):
                grid_ind_ = torch.cat([pl[None, ...] for pl in grid_ind_], dim=0)
            else:
                grid_ind_ = [pl[None, ...] for pl in grid_ind_]
        if isinstance(train_pts_label, list):
            if all(pl.shape == train_pts_label[0].shape for pl in train_pts_label):
                train_pts_label = torch.cat([pl[None, ...] for pl in train_pts_label], dim=0)
            else:
                train_pts_label = [pl[None, ...] for pl in train_pts_label]
        if isinstance(train_voxel_label, list):
            if all(pl.shape == train_voxel_label[0].shape for pl in train_voxel_label):
                train_voxel_label = torch.cat([pl[None, ...] for pl in train_voxel_label], dim=0)
            else:
                train_voxel_label = [pl[None, ...] for pl in train_voxel_label]
        if isinstance(grid_ind_vox_orig, list):
            if all(pl.shape == grid_ind_vox_orig[0].shape for pl in grid_ind_vox_orig):
                grid_ind_vox_orig = torch.cat([pl[None, ...] for pl in grid_ind_vox_orig], dim=0)
            else:
                grid_ind_vox_orig = [pl[None, ...] for pl in grid_ind_vox_orig]
        if isinstance(grid_ind_orig, list):
            if all(pl.shape == grid_ind_orig[0].shape for pl in grid_ind_orig):
                grid_ind_orig = torch.cat([pl[None, ...] for pl in grid_ind_orig], dim=0)
            else:
                grid_ind_orig = [pl[None, ...] for pl in grid_ind_orig]
        # When batch contains frames with different point counts, run per-sample and aggregate losses
        grid_is_list = isinstance(grid_ind_, list)
        voxel_is_list = isinstance(train_voxel_label, list)
        pts_is_list = isinstance(train_pts_label, list)
        flag_is_list = isinstance(dataset_flag, list) if dataset_flag is not None else False

        if grid_is_list or pts_is_list or voxel_is_list or flag_is_list:
            bs = x_lidar_tpv[0].shape[0]
            agg_loss = None
            for i in range(bs):
                x_i = [feat.index_select(0, torch.tensor([i], device=feat.device)) for feat in x_lidar_tpv]
                if grid_is_list:
                    points_i = grid_ind_[i]
                else:
                    points_i = grid_ind_[i:i+1]
                if voxel_is_list:
                    voxel_label_i = train_voxel_label[i]
                else:
                    voxel_label_i = train_voxel_label[i:i+1] if train_voxel_label is not None else None
                if pts_is_list:
                    pl_i = train_pts_label[i]
                    point_labels_i = pl_i
                else:
                    point_labels_i = train_pts_label[i:i+1] if train_pts_label is not None else None
                if flag_is_list:
                    dataset_flag_i = dataset_flag[i]
                elif isinstance(dataset_flag, torch.Tensor):
                    dataset_flag_i = dataset_flag[i:i+1]
                else:
                    dataset_flag_i = dataset_flag
                sensor_params_i = None
                if sensor_params is not None:
                    if isinstance(sensor_params, torch.Tensor):
                        if sensor_params.dim() > 1 and sensor_params.size(0) > 1:
                            sensor_params_i = sensor_params[min(i, sensor_params.size(0) - 1)]
                        else:
                            sensor_params_i = sensor_params
                    else:
                        sensor_params_i = sensor_params
                # Teacher predictions, only needed when distillation is enabled.
                logits_vox_t = None
                logits_pts_t = None
                if model_t is not None and self.d2skd:
                    x_i_t = [feat.index_select(0, torch.tensor([i], device=feat.device)) for feat in x_lidar_tpv_t]
                    # The teacher consumes the original, un-resampled grid indices.
                    if isinstance(grid_ind_vox_orig, list):
                        points_i_t = grid_ind_vox_orig[i] if i < len(grid_ind_vox_orig) else grid_ind_orig[i]
                    elif grid_ind_vox_orig is not None:
                        if isinstance(grid_ind_vox_orig, torch.Tensor):
                            points_i_t = grid_ind_vox_orig[i:i+1] if i < grid_ind_vox_orig.shape[0] else grid_ind_orig[i:i+1]
                        else:
                            points_i_t = points_i  # fallback
                    else:
                        if isinstance(grid_ind_orig, list):
                            points_i_t = grid_ind_orig[i]
                        else:
                            points_i_t = grid_ind_orig[i:i+1]

                    with torch.no_grad():
                        teacher_outs = model_t.tpv_aggregator(
                            x_i_t,
                            points=points_i_t,
                            voxel_label=None,
                            point_labels=None,
                            dataset_flag=dataset_flag_i,
                            return_loss=False,
                            sensor_vec=None,
                        )
                        logits_vox_t, logits_pts_t, _ = teacher_outs

                loss_i = self.tpv_aggregator(
                    x_i,
                    points=points_i,
                    voxel_label=voxel_label_i,
                    point_labels=point_labels_i,
                    dataset_flag=dataset_flag_i,
                    return_loss=return_loss,
                    sensor_vec=sensor_params_i,
                    logits_vox_t=logits_vox_t,
                    logits_pts_t=logits_pts_t,
                )
                # Fold the spectral distillation term into the loss dict.
                if model_t is not None and self.d2skd and return_loss and loss_distill is not None:
                    if isinstance(loss_i, dict):
                        loss_i['loss_inter_distill'] = 250.0 * loss_distill
                if agg_loss is None:
                    agg_loss = {k: loss_i[k] for k in loss_i}
                else:
                    for k in loss_i:
                        agg_loss[k] = agg_loss.get(k, 0) + loss_i[k]
            for k in agg_loss:
                agg_loss[k] = agg_loss[k] / bs
            agg_loss = self._apply_dynamic_loss_balance(agg_loss)
            return agg_loss
        else:
            # Collapse the remaining list inputs into batched tensors. Equal shapes can be
            # concatenated; otherwise fall back to the first element so the batch stays valid.
            if isinstance(grid_ind_, list):
                if all(pl.shape == grid_ind_[0].shape for pl in grid_ind_):
                    grid_ind_ = torch.cat([pl[None, ...] for pl in grid_ind_], dim=0)
                else:
                    grid_ind_ = grid_ind_[0].unsqueeze(0) if len(grid_ind_) > 0 else grid_ind_

            if isinstance(train_voxel_label, list):
                if all(pl.shape == train_voxel_label[0].shape for pl in train_voxel_label):
                    train_voxel_label = torch.cat([pl[None, ...] for pl in train_voxel_label], dim=0)
                else:
                    train_voxel_label = train_voxel_label[0].unsqueeze(0) if len(train_voxel_label) > 0 else train_voxel_label

            if isinstance(train_pts_label, list):
                if all(pl.shape == train_pts_label[0].shape for pl in train_pts_label):
                    train_pts_label = torch.cat([pl[None, ...] for pl in train_pts_label], dim=0)
                else:
                    train_pts_label = train_pts_label[0].unsqueeze(0) if len(train_pts_label) > 0 else train_pts_label

            # Teacher predictions, only needed when distillation is enabled.
            logits_vox_t = None
            logits_pts_t = None
            if model_t is not None and self.d2skd:
                # The teacher consumes the original, un-resampled grid indices.
                grid_ind_t = grid_ind_vox_orig if grid_ind_vox_orig is not None else grid_ind_orig
                if isinstance(grid_ind_t, list):
                    if all(pl.shape == grid_ind_t[0].shape for pl in grid_ind_t):
                        grid_ind_t = torch.cat([pl[None, ...] for pl in grid_ind_t], dim=0)
                    else:
                        grid_ind_t = grid_ind_t[0].unsqueeze(0) if len(grid_ind_t) > 0 else grid_ind_t

                with torch.no_grad():
                    teacher_outs = model_t.tpv_aggregator(
                        x_lidar_tpv_t,
                        points=grid_ind_t,
                        voxel_label=None,
                        point_labels=None,
                        dataset_flag=dataset_flag,
                        return_loss=False,
                        sensor_vec=None,
                    )
                    logits_vox_t, logits_pts_t, _ = teacher_outs
            outs = self.tpv_aggregator(
                x_lidar_tpv,
                points=grid_ind_,
                voxel_label=train_voxel_label,
                point_labels=train_pts_label,
                dataset_flag=dataset_flag,
                return_loss=return_loss,
                sensor_vec=sensor_params if sensor_params is not None else None,
                logits_vox_t=logits_vox_t,
                logits_pts_t=logits_pts_t,
            )

            # Fold the spectral distillation term into the loss dict.
            if model_t is not None and self.d2skd and return_loss and loss_distill is not None:
                if isinstance(outs, dict):
                    outs['loss_inter_distill'] = 666 * loss_distill

            if isinstance(outs, dict):
                outs = self._apply_dynamic_loss_balance(outs)
            return outs

    def train_step(self, data, optimizer, model_t=None, **kwargs):
        """Override train_step so a distillation teacher (``model_t``) can be passed through."""
        losses = self.forward_train(model_t=model_t, **data)
        loss, log_vars = self._parse_losses(losses)

        # Infer the batch size from the train_grid batch dimension.
        num_samples = 1
        if 'train_grid' in data:
            tg = data['train_grid']
            if isinstance(tg, torch.Tensor):
                num_samples = tg.shape[0]
            elif isinstance(tg, list) and len(tg) > 0 and isinstance(tg[0], torch.Tensor):
                num_samples = len(tg)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=num_samples)
        return outputs

    def simple_test(self,
                train_grid=None,
                grid_ind=None,
                grid_ind_vox=None,
                train_voxel_label=None,
                train_pts_label=None,
                dataset_flag= None,
                return_loss=False,
                **kwargs):
        """Forward testing function for semantic segmentation.

        Args:
            train_grid (torch.Tensor): Input point cloud data
            voxel_label (torch.Tensor): Ground truth labels
            grid_ind_vox (torch.Tensor): Grid indices for voxels
            dataset_flag (torch.Tensor): Dataset type flag
            visible_mask (torch.Tensor): Mask for visible points
            return_loss (bool): Whether to return loss

        Returns:
            dict: Test results containing predictions and metrics
        """
        x_lidar_tpv = self.extract_feat(train_grid=train_grid, voxel_label=grid_ind)
        grid_ind_ = grid_ind_vox if grid_ind_vox is not None else grid_ind
        if isinstance(grid_ind_, list):
            if all(pl.shape == grid_ind_[0].shape for pl in grid_ind_):
                grid_ind_ = torch.cat([pl[None, ...] for pl in grid_ind_], dim=0)
            else:
                grid_ind_ = [pl[None, ...] for pl in grid_ind_]
        if isinstance(train_pts_label, list):
            if all(pl.shape == train_pts_label[0].shape for pl in train_pts_label):
                train_pts_label = torch.cat([pl[None, ...] for pl in train_pts_label], dim=0)
            else:
                train_pts_label = [pl[None, ...] for pl in train_pts_label]
        if isinstance(train_voxel_label, list):
            if all(pl.shape == train_voxel_label[0].shape for pl in train_voxel_label):
                train_voxel_label = torch.cat([pl[None, ...] for pl in train_voxel_label], dim=0)
            else:
                train_voxel_label = [pl[None, ...] for pl in train_voxel_label]
        pred_list = None
        # If variable points per frame, run per-sample and pad results to evaluate
        if (
            isinstance(grid_ind_, list)
            or isinstance(train_pts_label, list)
            or isinstance(train_voxel_label, list)
            or (isinstance(dataset_flag, list) if dataset_flag is not None else False)
        ):
            bs = x_lidar_tpv[0].shape[0]
            pred_list = []
            gt_list = []
            pred_sensor_output_list = []
            for i in range(bs):
                # Per-frame TPV features; keep a batch dimension of 1
                x_i = [feat.index_select(0, torch.tensor([i], device=feat.device)) for feat in x_lidar_tpv]
                # Ensure points have shape [1, N_i, 3]
                if isinstance(grid_ind_, list):
                    points_i = grid_ind_[i].unsqueeze(0)
                else:
                    points_i = grid_ind_[i:i+1]
                # Ensure voxel labels have shape [1, W, H, D] if provided
                if isinstance(train_voxel_label, list):
                    voxel_label_i = train_voxel_label[i].unsqueeze(0)
                else:
                    voxel_label_i = train_voxel_label[i:i+1] if train_voxel_label is not None else None
                # Ensure point labels have shape [1, N_i, 1, 1] if provided
                if isinstance(train_pts_label, list):
                    pl_i = train_pts_label[i]
                    point_labels_i = pl_i.unsqueeze(0) if pl_i.dim() == 1 else pl_i.unsqueeze(0) if pl_i.dim() == 2 else pl_i.unsqueeze(0) if pl_i.dim() == 3 else pl_i
                else:
                    point_labels_i = train_pts_label[i:i+1] if train_pts_label is not None else None
                if isinstance(dataset_flag, list):
                    dataset_flag_i = dataset_flag[i]
                elif isinstance(dataset_flag, torch.Tensor):
                    dataset_flag_i = dataset_flag[i:i+1]
                else:
                    dataset_flag_i = dataset_flag
                _, logits_pts_i, pred_sensor_output = self.tpv_aggregator(
                    x_i,
                    points=points_i,
                    voxel_label=voxel_label_i,
                    point_labels=point_labels_i,
                    dataset_flag=dataset_flag_i,
                    return_loss=return_loss,
                )
                logits_pts_i = logits_pts_i.squeeze(-1).squeeze(-1).squeeze(0)  # [classes, n_i]
                pred_i = torch.argmax(logits_pts_i, dim=0).detach().cpu()
                gt_i = point_labels_i.squeeze().detach().cpu()
                pred_list.append(pred_i)
                gt_list.append(gt_i)
                pred_sensor_output_list.append(pred_sensor_output)
            max_n = max(p.shape[0] for p in pred_list)
            predict_labels_pts = torch.zeros((bs, max_n), dtype=pred_list[0].dtype)
            val_pt_labs = torch.zeros((bs, max_n), dtype=gt_list[0].dtype)
            for i in range(bs):
                n_i = pred_list[i].shape[0]
                predict_labels_pts[i, :n_i] = pred_list[i]
                val_pt_labs[i, :n_i] = gt_list[i]
        else:
            _, predict_labels_pts, pred_sensor_output = self.tpv_aggregator(x_lidar_tpv, points=grid_ind_, voxel_label=train_voxel_label, point_labels=train_pts_label, dataset_flag= dataset_flag, return_loss=return_loss)
            predict_labels_pts = predict_labels_pts.squeeze(-1).squeeze(-1)
            predict_labels_pts = torch.argmax(predict_labels_pts, dim=1) # bs, n
            predict_labels_pts = predict_labels_pts.detach().cpu()
            val_pt_labs = train_pts_label.squeeze(-1).detach().cpu()

        tc = getattr(self, 'test_cfg', None)
        save_infer = False
        if tc is not None:
            if hasattr(tc, 'get'):
                save_infer = bool(tc.get('save_inference_preds', False))
            else:
                save_infer = bool(getattr(tc, 'save_inference_preds', False))
        if save_infer:
            test_output = {
                'pred': predict_labels_pts,
                'pred_sensor_output': pred_sensor_output,
            }
        else:
            IoU_metric = self.evaluation_semantic(
                predict_labels_pts, val_pt_labs, dataset_flag)
            test_output = {
                'SSC_metric': IoU_metric,
                'pred': predict_labels_pts,
                'pred_sensor_output': pred_sensor_output,
            }

        self._maybe_save_inference_preds(
            predict_labels_pts, pred_list, train_grid, kwargs)

        return test_output

    def forward_test(self,
                train_grid=None,
                grid_ind=None,
                grid_ind_vox=None,
                train_voxel_label=None,
                train_pts_label=None,
                dataset_flag= None,
                return_loss=False,
                **kwargs
        ):
        """
        Forward testing function.
        """
        return self.simple_test(
            train_grid, grid_ind, grid_ind_vox, train_voxel_label, train_pts_label,
            dataset_flag, return_loss, **kwargs)

    def _unwrap_train_grid_for_save(self, train_grid):
        """Strip DataContainer / single-element list wrappers from collate."""
        if train_grid is None:
            return None
        x = train_grid
        while isinstance(x, DataContainer):
            x = x.data
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        return x

    def _maybe_save_inference_preds(
            self, predict_labels_pts, pred_list, train_grid, kwargs):
        """Write .label (uint32) and optional .ply (train_grid xyz) when enabled."""
        tc = getattr(self, 'test_cfg', None)
        if tc is None:
            return
        if hasattr(tc, 'get'):
            save = tc.get('save_inference_preds', False)
            out_root = tc.get('inference_pred_dir', './work_dirs/realdata_preds')
            save_ply = tc.get('save_inference_ply', True)
        else:
            save = getattr(tc, 'save_inference_preds', False)
            out_root = getattr(tc, 'inference_pred_dir', './work_dirs/realdata_preds')
            save_ply = getattr(tc, 'save_inference_ply', True)
        if not save:
            return
        img_metas = kwargs.get('img_metas', None)
        if img_metas is None:
            return
        metas = img_metas.data[0] if hasattr(img_metas, 'data') else img_metas
        if not isinstance(metas, (list, tuple)):
            metas = [metas]
        os.makedirs(out_root, exist_ok=True)
        pred_np = predict_labels_pts.detach().cpu().numpy()
        bs = pred_np.shape[0]
        tg_in = train_grid if train_grid is not None else kwargs.get('train_grid')
        tg = self._unwrap_train_grid_for_save(tg_in)
        for b in range(bs):
            meta = metas[b] if b < len(metas) else metas[0]
            stem = None
            if isinstance(meta, dict):
                tid = meta.get('lidar_token')
                if tid is not None:
                    stem = str(tid)
                if stem is None:
                    for key in ('pts_filename', 'filename'):
                        p = meta.get(key)
                        if p:
                            stem = osp.splitext(osp.basename(str(p)))[0]
                            break
                if stem is None and meta.get('sample_idx') is not None:
                    stem = str(meta['sample_idx'])
            if stem is None:
                stem = f'frame_{b:06d}'
            if pred_list is not None:
                n_i = pred_list[b].shape[0]
                labels = pred_np[b, :n_i].astype(np.uint32)
            else:
                labels = pred_np[b].astype(np.uint32)
                n_i = int(labels.shape[0])
            out_path = osp.join(out_root, f'{stem}.label')
            labels.tofile(out_path)
            if save_ply and tg is not None:
                rows = None
                if isinstance(tg, torch.Tensor):
                    if tg.dim() == 3:
                        rows = tg[b, :n_i]
                    elif tg.dim() == 2 and bs == 1 and b == 0:
                        rows = tg[:n_i]
                elif isinstance(tg, (list, tuple)) and b < len(tg):
                    tgb = tg[b]
                    if isinstance(tgb, torch.Tensor) and tgb.dim() == 2:
                        rows = tgb[:n_i]
                    elif isinstance(tgb, torch.Tensor) and tgb.dim() == 3:
                        rows = tgb[0, :n_i] if tgb.shape[0] == 1 else tgb[b, :n_i]
                if rows is not None and rows.shape[0] >= n_i:
                    rows = rows[:n_i]
                    ply_path = osp.join(out_root, f'{stem}.ply')
                    save_tensor_as_ply(rows, ply_path)

    def evaluation_semantic(self, pred, gt, dataset_flag):
        """Evaluate semantic segmentation results.

        Args:
            pred (torch.Tensor): Predicted point cloud labels, shape [B, N, num_classes]
            gt (torch.Tensor): Ground truth point cloud labels, shape [B, N]
            dataset_flag (torch.Tensor): Dataset type flag
            visible_mask (torch.Tensor, optional): Mask for visible points

        Returns:
            tuple: (IoU histogram matrix, visible IoU histogram matrix)
        """
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()

        # Determine number of classes based on dataset
        if isinstance(dataset_flag, list):
            if len(dataset_flag) == 1:
                dataset_flag = dataset_flag[0].item()
            elif all(flag.item() == dataset_flag[0].item() for flag in dataset_flag):
                dataset_flag = dataset_flag[0].item()
            else:
                raise NotImplementedError("Different dataset_flag when eval")
        else:
            assert isinstance(dataset_flag, torch.Tensor)
            dataset_flag = dataset_flag.item()

        uniq_gt = np.unique(gt)
        if len(uniq_gt) >= 2 and np.sort(uniq_gt)[-2] <= 7:
            max_label = 8
        elif dataset_flag == 1:
            max_label = 17
        elif dataset_flag == 2:
            max_label = 20
        elif dataset_flag == 3:
            max_label = 23
        else:
            max_label = 8

        # Create mask for valid points (excluding noise/unknown)
        valid_mask = gt != 0

        # Compute IoU histogram for all valid points
        hist = fast_hist(pred[valid_mask], gt[valid_mask], max_label=max_label)

        return hist

    @torch.no_grad()
    def _mixup_minority(self, points, grid_ind, grid_ind_vox, labels, num_add=2):
        """
        Instance-aware minority class mixup with physical placement validation.

        Args:
            points:        [N, C]   layout: [:, 0:3]=rel_xyz, [:, 3:6]=xyz_pol(rho,phi,z),
                                            [:, 6:8]=xy(cart),  [:, 8:]=other features
            grid_ind:      [N, 3]   voxel idx in polar grid (float)
            grid_ind_vox:  [N, 3]   voxel idx in cubic grid  (float)
            labels:        [N]
            num_add:       how many copies per instance to *attempt* to place
                        (each copy is only kept if placement is valid)

        Returns:
            augmented points / grid_ind / grid_ind_vox / labels
        """
        device = points.device
        dtype = points.dtype

        MINORITY_CLASSES = (5, 6)

        # ---- Tunables ------------------------------------------------------
        # Cluster: BFS in polar voxel space; max neighbour distance in voxel idx
        CLUSTER_MAX_VOX_DIST = 1
        CLUSTER_MIN_PTS = 3                # ignore stray minority points
        # Per-attempt search:
        PLACEMENT_TRIES = 8                # how many random poses tried per copy
        ROT_RANGE_RAD = 0.25 * math.pi     # ±45° azimuthal rotation
        RHO_JITTER_REL = 0.20              # ±20% radial scale jitter
        Z_JITTER = 0.05                    # ±5 cm vertical jitter on top of ground snap
        # Placement constraints (in metres)
        NEAR_FRONT_TOL = 0.50              # if any existing point in target cell is
                                        # closer than (instance_min_depth - TOL),
                                        # placement is rejected (occlusion violation)
        GROUND_SEARCH_RADIUS = 0.6         # m, radial+azimuthal radius in metric space
        GROUND_SEARCH_DZ = 1.0             # max distance below instance bottom to look for ground
        MAX_GROUND_LIFT = 0.5              # max +z shift the snap may impose
        MAX_GROUND_DROP = 0.5              # max -z shift the snap may impose
        # ROI from class attributes
        min_bound = self.min_bound
        max_bound = self.max_bound
        grid_size = self.grid_size
        grid_size_vox = self.grid_size_vox
        crop_range = max_bound - min_bound
        intervals = crop_range / grid_size
        intervals_vox = crop_range / grid_size_vox

        # 0. Select minority points
        minority_mask = torch.zeros_like(labels, dtype=torch.bool)
        for c in MINORITY_CLASSES:
            minority_mask = minority_mask | (labels == c)
        if not minority_mask.any():
            return points, grid_ind, grid_ind_vox, labels

        minor_pts = points[minority_mask]
        minor_lbs = labels[minority_mask]
        minor_grid = grid_ind[minority_mask]      # polar voxel idx
        # ground reference: existing non-minority points
        bg_mask = ~minority_mask
        bg_xy = points[bg_mask][:, 6:8]            # [B, 2] cartesian xy
        bg_z = points[bg_mask][:, 5]               # [B] z
        bg_rho = points[bg_mask][:, 3]             # [B] rho
        bg_phi = points[bg_mask][:, 4]             # [B] phi
        bg_labels = labels[bg_mask]
        # rough "ground" candidates: classes 1 (Driveable Ground), 7 (Other Ground)
        # If you don't trust this, fall back to "lowest-z in column" later.
        ground_class_mask = (bg_labels == 1) | (bg_labels == 7)
        ground_xy = bg_xy[ground_class_mask]
        ground_z = bg_z[ground_class_mask]

        # 1. Cluster minority points -> instances
        #    Use a hash on the polar voxel grid + 6-neighbour BFS.
        #    For typical KITTI frames this is < 50 minority points -> trivial cost.
        N_minor = minor_pts.shape[0]
        if N_minor < CLUSTER_MIN_PTS:
            return points, grid_ind, grid_ind_vox, labels

        # Do the clustering on CPU (small N, host-side is faster and simpler)
        minor_grid_cpu = minor_grid.detach().to(torch.int64).cpu().numpy()
        # Build a hash table voxel -> point index
        vox_to_idx = {}
        for i in range(N_minor):
            key = (int(minor_grid_cpu[i, 0]),
                int(minor_grid_cpu[i, 1]),
                int(minor_grid_cpu[i, 2]))
            vox_to_idx.setdefault(key, []).append(i)

        visited = [False] * N_minor
        instances = []   # list of list-of-point-indices
        for seed in range(N_minor):
            if visited[seed]:
                continue
            stack = [seed]
            comp = []
            while stack:
                cur = stack.pop()
                if visited[cur]:
                    continue
                visited[cur] = True
                comp.append(cur)
                ck = minor_grid_cpu[cur]
                # 6-neighbour search (could widen with CLUSTER_MAX_VOX_DIST)
                r = CLUSTER_MAX_VOX_DIST
                for d0 in range(-r, r + 1):
                    for d1 in range(-r, r + 1):
                        for d2 in range(-r, r + 1):
                            if d0 == 0 and d1 == 0 and d2 == 0:
                                continue
                            nbr = (int(ck[0]) + d0,
                                int(ck[1]) + d1,
                                int(ck[2]) + d2)
                            if nbr in vox_to_idx:
                                for j in vox_to_idx[nbr]:
                                    if not visited[j]:
                                        stack.append(j)
            if len(comp) >= CLUSTER_MIN_PTS:
                instances.append(comp)

        if not instances:
            return points, grid_ind, grid_ind_vox, labels

        # 2. For each instance, attempt num_add valid placements
        add_points_list = []
        add_labels_list = []
        add_grid_list = []
        add_grid_vox_list = []

        # Pre-compute existing-cloud sorted-by-phi index for fast cell lookup
        all_xy = points[:, 6:8]                          # [N, 2]
        # Pre-compute sorted-by-phi index over the WHOLE existing cloud,
        # used for occlusion-violation check. Sorting once per frame is O(N log N);
        # inside the placement loop each query is O(log N + window).
        all_rho_full = points[:, 3].contiguous()         # rho
        all_phi_full = points[:, 4].contiguous()         # phi in [-pi, pi]

        # Sort by phi once. We will binary-search phi-windows inside the loop.
        phi_sorted, phi_sort_idx = torch.sort(all_phi_full)
        rho_sorted = all_rho_full[phi_sort_idx]

        # To handle the wrap-around at ±pi without writing two-segment queries,
        # build a duplicated "ring" array: [phi - 2pi  |  phi  |  phi + 2pi].
        # This way any window centred near ±pi still maps to a single contiguous slice.
        TWO_PI = 2.0 * math.pi
        phi_ring = torch.cat([phi_sorted - TWO_PI, phi_sorted, phi_sorted + TWO_PI], dim=0)
        rho_ring = torch.cat([rho_sorted,           rho_sorted, rho_sorted          ], dim=0)

        # Same idea for ground points, also keyed by phi for quick local lookup.
        if ground_xy.shape[0] > 0:
            ground_phi = torch.atan2(ground_xy[:, 1], ground_xy[:, 0])
            g_sorted_phi, g_sort_idx = torch.sort(ground_phi)
            g_sorted_z = ground_z[g_sort_idx]
            g_sorted_xy = ground_xy[g_sort_idx]
            g_phi_ring = torch.cat([g_sorted_phi - TWO_PI, g_sorted_phi, g_sorted_phi + TWO_PI], dim=0)
            g_z_ring = torch.cat([g_sorted_z, g_sorted_z, g_sorted_z], dim=0)
            g_xy_ring = torch.cat([g_sorted_xy, g_sorted_xy, g_sorted_xy], dim=0)
        else:
            g_phi_ring = None

        phi_tol = 1.5 * TWO_PI / 1800.0   # ~1.5 horizontal bins
        GROUND_PHI_TOL = max(phi_tol * 4, 0.02)  # wider window for ground lookup

        for inst in instances:
            inst_idx = torch.tensor(inst, device=device, dtype=torch.long)
            ip = minor_pts[inst_idx]                     # [K, C]
            il = minor_lbs[inst_idx]                     # [K]

            ip_xy = ip[:, 6:8]                           # cartesian xy
            ip_z = ip[:, 5]                              # z
            # Instance centroid (in cartesian, on xy plane)
            cx = ip_xy[:, 0].mean()
            cy = ip_xy[:, 1].mean()
            c_rho = torch.sqrt(cx * cx + cy * cy)
            c_phi = torch.atan2(cy, cx)
            # Per-instance points relative to centroid (so we can rotate as a rigid body)
            rel_xy = ip_xy - torch.stack([cx, cy])
            z_min_inst = ip_z.min()
            z_max_inst = ip_z.max()

            for _copy in range(num_add):
                placed = False
                for _try in range(PLACEMENT_TRIES):
                    # ---- propose a new pose ---------------------------------
                    d_phi = (torch.rand((), device=device) * 2 - 1) * ROT_RANGE_RAD
                    rho_scale = 1.0 + (torch.rand((), device=device) * 2 - 1) * RHO_JITTER_REL
                    new_rho = (c_rho * rho_scale).clamp(min=1.0)
                    new_phi = c_phi + d_phi
                    # wrap to [-pi, pi]
                    new_phi = (new_phi + math.pi) % (2 * math.pi) - math.pi
                    new_cx = new_rho * torch.cos(new_phi)
                    new_cy = new_rho * torch.sin(new_phi)

                    # rotate instance rigidly so it still "faces" the new bearing
                    cos_dphi = torch.cos(d_phi)
                    sin_dphi = torch.sin(d_phi)
                    R = torch.stack([
                        torch.stack([cos_dphi, -sin_dphi]),
                        torch.stack([sin_dphi,  cos_dphi]),
                    ])
                    # scale radial component to honour rho_scale (keeps wider object wider
                    # at new range — optional; we set scale=1 along radial too for safety)
                    new_rel_xy = rel_xy @ R.T
                    new_xy = new_rel_xy + torch.stack([new_cx, new_cy])

                    # ---- find ground at target xy (phi-window restricted) ---
                    if g_phi_ring is None:
                        target_ground_z = torch.tensor(0.0, device=device, dtype=dtype)
                    else:
                        # find the phi-window centred at new_phi
                        lo = torch.searchsorted(g_phi_ring, (new_phi - GROUND_PHI_TOL).reshape(1)).item()
                        hi = torch.searchsorted(g_phi_ring, (new_phi + GROUND_PHI_TOL).reshape(1)).item()
                        if hi <= lo:
                            continue
                        cand_xy = g_xy_ring[lo:hi]
                        cand_z = g_z_ring[lo:hi]
                        dxy = cand_xy - torch.stack([new_cx, new_cy])
                        d2 = (dxy * dxy).sum(dim=1)
                        nearest = d2.argmin()
                        if torch.sqrt(d2[nearest]) > GROUND_SEARCH_RADIUS:
                            continue
                        target_ground_z = cand_z[nearest]

                    z_shift = target_ground_z - z_min_inst
                    # clip shift to avoid pathological large jumps
                    z_shift = torch.clamp(z_shift, min=-MAX_GROUND_DROP, max=MAX_GROUND_LIFT)
                    # tiny vertical jitter on top of ground snap
                    z_shift = z_shift + (torch.rand((), device=device) * 2 - 1) * Z_JITTER
                    new_z = ip_z + z_shift

                    # ---- occlusion-violation check (phi-window binary search) ---
                    # Replaces the previous O(K * B_full) dense pairwise check, which was
                    # the cause of the 10 GB+ memory spikes per try.
                    new_rho_pts = torch.sqrt(new_xy[:, 0] ** 2 + new_xy[:, 1] ** 2)
                    new_phi_pts = torch.atan2(new_xy[:, 1], new_xy[:, 0])

                    # Per instance-point, find the slice of background rho values whose phi
                    # is within phi_tol. Then check if any of those is closer than
                    # (this point's rho - NEAR_FRONT_TOL).
                    lo_q = new_phi_pts - phi_tol
                    hi_q = new_phi_pts + phi_tol
                    # searchsorted is vectorised over K; output shape [K]
                    lo_idx = torch.searchsorted(phi_ring, lo_q.contiguous())
                    hi_idx = torch.searchsorted(phi_ring, hi_q.contiguous())

                    conflict_any = False
                    for k in range(new_rho_pts.shape[0]):
                        lk = int(lo_idx[k].item())
                        hk = int(hi_idx[k].item())
                        if hk <= lk:
                            continue
                        # this is a small slice (typically a few thousand points), safe
                        cand_rho = rho_ring[lk:hk]
                        if (cand_rho < (new_rho_pts[k] - NEAR_FRONT_TOL)).any():
                            conflict_any = True
                            break
                    if conflict_any:
                        continue

                    # ---- bounds check (polar ROI) ---------------------------
                    new_xyz_pol = torch.stack([new_rho_pts, new_phi_pts, new_z], dim=1)
                    in_bound = ((new_xyz_pol >= min_bound) &
                                (new_xyz_pol < (max_bound - 1e-3))).all(dim=1)
                    if not in_bound.all():
                        continue

                    # ---- valid placement: build output rows ------------------
                    # grid indices
                    new_grid_ind = torch.floor((new_xyz_pol - min_bound) / intervals).to(torch.int32)
                    new_grid_ind_vox_float = (new_xyz_pol - min_bound) / intervals_vox
                    # voxel centres for rel_xyz
                    voxel_centers = (new_grid_ind.float() + 0.5) * intervals + min_bound
                    new_rel_xyz = new_xyz_pol - voxel_centers

                    # carry-over features unchanged (intensity etc.)
                    feat = ip[:, 8:]
                    # assemble: [rel_xyz (3) | xyz_pol (3) | xy (2) | feat]
                    chunk_aug = torch.cat([new_rel_xyz, new_xyz_pol, new_xy, feat], dim=1)

                    add_points_list.append(chunk_aug)
                    add_labels_list.append(il)
                    add_grid_list.append(new_grid_ind.float())
                    add_grid_vox_list.append(new_grid_ind_vox_float)
                    placed = True
                    break
                # if not placed after PLACEMENT_TRIES, skip this copy
                if not placed:
                    continue

        if not add_points_list:
            return points, grid_ind, grid_ind_vox, labels

        final_points = torch.cat([points] + add_points_list, dim=0)
        final_labels = torch.cat([labels] + add_labels_list, dim=0)
        final_grid = torch.cat([grid_ind] + add_grid_list, dim=0)
        final_grid_vox = torch.cat([grid_ind_vox] + add_grid_vox_list, dim=0)

        return final_points, final_grid, final_grid_vox, final_labels

    @torch.no_grad()
    def _sampling(self, points, grid_ind, grid_ind_vox, labels):
        # Dense training mode: uniform sub-sampling.
        if self.dense_train:
            was_single_tensor = not isinstance(points, list)

            # Normalise every input to a detached list.
            if isinstance(points, list):
                points = [p.detach() for p in points]
            else:
                points = [points.detach()]

            if isinstance(grid_ind, list):
                grid_ind = [g.detach() if isinstance(g, torch.Tensor) else g for g in grid_ind]
            else:
                grid_ind = [grid_ind.detach() if isinstance(grid_ind, torch.Tensor) else grid_ind]

            if isinstance(grid_ind_vox, list):
                grid_ind_vox = [g.detach() if isinstance(g, torch.Tensor) else g for g in grid_ind_vox]
            else:
                grid_ind_vox = [grid_ind_vox.detach() if isinstance(grid_ind_vox, torch.Tensor) else grid_ind_vox]

            if isinstance(labels, list):
                labels = [l.detach() if isinstance(l, torch.Tensor) else l for l in labels]
            else:
                labels = [labels.detach() if isinstance(labels, torch.Tensor) else labels]

            results_points = []
            results_grid_ind = []
            results_grid_ind_vox = []
            results_labels = []

            max_points = 500000

            for idx in range(len(points)):
                curr_points = points[idx]
                curr_labels = labels[idx]
                curr_grid_ind = grid_ind[idx]
                curr_grid_ind_vox = grid_ind_vox[idx]

                num_points = curr_points.shape[0]

                if num_points <= max_points:
                    results_points.append(curr_points)
                    results_grid_ind.append(curr_grid_ind)
                    results_grid_ind_vox.append(curr_grid_ind_vox)
                    results_labels.append(curr_labels)
                else:
                    # Uniformly sub-sample down to max_points.
                    device = curr_points.device
                    indices = torch.randperm(num_points, device=device)[:max_points]

                    sampled_points = curr_points[indices]
                    sampled_labels = curr_labels[indices]
                    sampled_grid_ind = curr_grid_ind[indices]
                    sampled_grid_ind_vox = curr_grid_ind_vox[indices]

                    results_points.append(sampled_points)
                    results_grid_ind.append(sampled_grid_ind)
                    results_grid_ind_vox.append(sampled_grid_ind_vox)
                    results_labels.append(sampled_labels)

            sensor_params_tensor = None

            # Mirror the input container type in the return value.
            if was_single_tensor and len(results_points) == 1:
                return results_points[0], results_grid_ind[0], results_grid_ind_vox[0], results_labels[0], sensor_params_tensor
            else:
                return results_points, results_grid_ind, results_grid_ind_vox, results_labels, sensor_params_tensor

        # Original sampling logic (ray-casting based)
        points = [p.detach() for p in points] if isinstance(points, list) else points.detach()

        LiDAR_height = self.sensor_config.get('LiDAR_height', [1, 2])
        num_of_beams = self.sensor_config.get('num_of_beams', [16, 128])
        horizontal_angular_resolution = self.sensor_config.get('horizontal_angular_resolution', [900, 3600])
        lower_vertical_field_of_view_bound = self.sensor_config.get('lower_vertical_field_of_view_bound', [-40, -5])
        upper_vertical_field_of_view_bound = self.sensor_config.get('upper_vertical_field_of_view_bound', [0, 25])

        mixup_prob = 0.15

        def _sample_sensor_params_tuple():
            lidar_height = round(random.uniform(LiDAR_height[0], LiDAR_height[1]), 1)
            beams = sample_beams(num_of_beams[0], num_of_beams[1])
            horizontal_resolution = random.choice(
                range(horizontal_angular_resolution[0], horizontal_angular_resolution[1] + 1, 100)
            )
            vertical_lower_angle = round(
                random.uniform(lower_vertical_field_of_view_bound[0],
                               lower_vertical_field_of_view_bound[1]), 1
            )
            vertical_upper_angle = round(
                random.uniform(upper_vertical_field_of_view_bound[0],
                               upper_vertical_field_of_view_bound[1]), 1
            )
            return (beams, lidar_height, horizontal_resolution,
                    vertical_lower_angle, vertical_upper_angle)

        def _normalize_sensor_params_tuple(params):
            beams, lidar_height, hres, vlow, vhigh = params
            beam_span = max(num_of_beams[1] - num_of_beams[0], 1)
            height_span = max(LiDAR_height[1] - LiDAR_height[0], 1e-6)
            hres_lo, hres_hi = horizontal_angular_resolution
            hres_span = max(hres_hi - hres_lo, 1)
            vlow_lo, vlow_hi = lower_vertical_field_of_view_bound
            vlow_span = max(vlow_hi - vlow_lo, 1e-6)
            vup_lo, vup_hi = upper_vertical_field_of_view_bound
            vup_span = max(vup_hi - vup_lo, 1e-6)
            return np.array([
                (beams - num_of_beams[0]) / beam_span,
                (lidar_height - LiDAR_height[0]) / height_span,
                (hres - hres_lo) / hres_span,
                (vlow - vlow_lo) / vlow_span,
                (vhigh - vup_lo) / vup_span,
            ], dtype=np.float64)

        def _sensor_params_distance(params_a, params_b):
            diff = _normalize_sensor_params_tuple(params_a) - _normalize_sensor_params_tuple(params_b)
            return float(np.linalg.norm(diff))

        def _sample_diverse_sensor_params(repeat_count, num_candidates=12):
            """Greedy maximin: each new sample maximizes min distance to prior ones."""
            if repeat_count <= 1:
                return [_sample_sensor_params_tuple()]
            selected = [_sample_sensor_params_tuple()]
            for _ in range(repeat_count - 1):
                best_candidate = None
                best_min_dist = -1.0
                for _ in range(num_candidates):
                    candidate = _sample_sensor_params_tuple()
                    min_dist = min(_sensor_params_distance(candidate, prev) for prev in selected)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_candidate = candidate
                selected.append(best_candidate)
            return selected

        def _run_single_sampling(sensor_params_tuple=None):
            # 0. Draw the LiDAR configuration.
            if sensor_params_tuple is None:
                (beams, lidar_height, horizontal_resolution,
                 vertical_lower_angle, vertical_upper_angle) = _sample_sensor_params_tuple()
            else:
                (beams, lidar_height, horizontal_resolution,
                 vertical_lower_angle, vertical_upper_angle) = sensor_params_tuple

            results_points = []
            results_grid_ind = []
            results_grid_ind_vox = []
            results_labels = []

            device = points[0].device
            dtype = points[0].dtype
            sensor_params = torch.tensor(
                [beams, lidar_height, horizontal_resolution,
                vertical_lower_angle, vertical_upper_angle],
                device=device
            )

            # Sampling hyper-parameters
            MAX_RANGE = 80.0
            EPS = 1e-8
            LARGE_VAL = 1e10

            # Angular candidate gating. Each bin uses its full footprint: under
            # nearest-bin (round) assignment the bins tile the angular space exactly,
            # so a scale of 1.0 is the physically correct choice.
            ANGULAR_MARGIN_SCALE = 1.0

            # Occlusion-envelope suppression
            OCC_AZ_WIN = 11          # azimuth window width when testing occlusion (odd)
            OCC_EL_WIN = 10          # how many beams above the point to inspect
            OCC_SUPPORT_MIN = 3      # supporting neighbours near the closest foreground depth
            OCC_DEPTH_MARGIN = 0.5   # metres a point must lie behind the reference to count as occluded;
                                     # kept generous so thin objects against walls (poles, pedestrians,
                                     # cyclists) are not dropped
            OCC_NEAR_BAND = 0.25
            OCC_REL_MARGIN = 0.02
            RING_KEEP_BAND = 0.20    # depth bandwidth for "similar" same-ring neighbours
            RING_KEEP_MIN = 6        # similar neighbours required to trigger same-ring protection
            RING_KEEP_WIN = 7        # half-width of the same-ring window (centre excluded)

            # Ring continuity (local median shrinkage)
            SMOOTH_WIN = 15
            SMOOTH_SUPPORT_MIN = 4
            SMOOTH_SPREAD_MAX = 1.6

            # Range-adaptive shrinkage strength.
            SMOOTH_REL_THRESH = 0.002
            SMOOTH_ABS_FLOOR = 0.02
            SMOOTH_BLEND_NEAR = 0.88
            SMOOTH_BLEND_FAR = 0.8   # maximum far-range shrinkage; a value near 1.0 would replace
                                     # distant outliers wholesale with the median and erase far-away
                                     # minority classes, so it is paired with the class bypass below
            SMOOTH_BLEND_REF = 8.0

            # Extra gate: agreement with the local depth trend.
            SMOOTH_TREND_REL_BAND = 0.010
            SMOOTH_TREND_ABS_BAND = 0.03

            # Class-aware minority protection, on the mapped COLA taxonomy:
            # 5 = Living Being, 6 = Movable objects.
            MINORITY_CLASSES = (5, 6)

            # Helpers
            def wrap_to_pi(angle):
                return (angle + torch.pi) % (2 * torch.pi) - torch.pi

            def make_empty_outputs(curr_points, curr_labels, curr_grid_ind, curr_grid_ind_vox):
                device = curr_points.device
                dtype = curr_points.dtype
                return (
                    torch.empty((0, curr_points.shape[1]), device=device, dtype=dtype),
                    torch.empty((0,), device=device, dtype=curr_labels.dtype),
                    torch.empty((0, 3), device=device, dtype=curr_grid_ind.dtype),
                    torch.empty((0, 3), device=device, dtype=curr_grid_ind_vox.dtype),
                )

            def circular_median_filter_2d(depth_img, hit_img, win):
                """Batch version: process all beams at once. depth_img/hit_img: [B, H]."""
                B, H = depth_img.shape
                pad = win // 2
                depth_pad = torch.cat(
                    [depth_img[:, -pad:], depth_img, depth_img[:, :pad]], dim=1
                )  # [B, H+2*pad]
                valid_pad = torch.cat(
                    [hit_img[:, -pad:], hit_img, hit_img[:, :pad]], dim=1
                )
                depth_unf = depth_pad.unfold(1, win, 1)  # [B, H, win]
                valid_unf = valid_pad.unfold(1, win, 1)
                masked = torch.where(
                    valid_unf, depth_unf, torch.full_like(depth_unf, LARGE_VAL)
                )
                sorted_vals, _ = torch.sort(masked, dim=2)
                support = valid_unf.sum(dim=2)
                med_idx = torch.clamp(
                    torch.div(support - 1, 2, rounding_mode='floor'), min=0
                ).long().clamp(max=win - 1)
                max_idx = torch.clamp(support - 1, min=0).long().clamp(max=win - 1)
                med_row = sorted_vals.gather(2, med_idx.unsqueeze(2)).squeeze(2)
                min_vals = sorted_vals[:, :, 0]
                max_vals = sorted_vals.gather(2, max_idx.unsqueeze(2)).squeeze(2)
                spread_row = max_vals - min_vals
                return med_row, spread_row, support

            # Pre-compute the ray grid shared by every frame in this batch.
            vertical_angles_rad = torch.deg2rad(
                torch.linspace(vertical_lower_angle, vertical_upper_angle, beams, device=device, dtype=dtype)
            )
            horizontal_step = 2.0 * math.pi / horizontal_resolution
            horizontal_angles_rad = torch.arange(horizontal_resolution, device=device, dtype=dtype) * horizontal_step
            vertical_step = (vertical_angles_rad[-1] - vertical_angles_rad[0]).item() / max(1, beams - 1)

            vv, hh = torch.meshgrid(vertical_angles_rad, horizontal_angles_rad, indexing='ij')
            ray_directions = torch.stack([
                torch.cos(vv) * torch.cos(hh),
                torch.cos(vv) * torch.sin(hh),
                torch.sin(vv)
            ], dim=-1).reshape(-1, 3)
            lidar_origin = torch.tensor([0.0, 0.0, float(lidar_height)], device=device, dtype=dtype)

            az_half = 0.5 * horizontal_step
            el_half = 0.5 * vertical_step if beams > 1 else 1e6

            # Hoist to Python scalars: much faster inside the per-frame loop.
            az_margin_val = az_half * ANGULAR_MARGIN_SCALE
            el_margin_val = el_half * ANGULAR_MARGIN_SCALE
            footprint_angle_val = math.sqrt(az_half ** 2 + el_half ** 2) * ANGULAR_MARGIN_SCALE
            cos_footprint = math.cos(footprint_angle_val)
            num_rays = beams * horizontal_resolution

            # 1. Per-frame sampling loop
            for idx in range(len(points)):
                curr_points = points[idx]
                curr_labels = labels[idx]
                curr_grid_ind = grid_ind[idx]
                curr_grid_ind_vox = grid_ind_vox[idx]

                # 1. Minority-class copy / mixup
                if random.random() < mixup_prob:
                    curr_points, curr_grid_ind, curr_grid_ind_vox, curr_labels = \
                        self._mixup_minority(
                            curr_points, curr_grid_ind, curr_grid_ind_vox, curr_labels, num_add=2
                        )

                # 2. Gather xyz in the stored column layout
                point_cloud_tensor = torch.cat(
                    (curr_points[:, 6:8], curr_points[:, 5].unsqueeze(1)),
                    dim=1
                )  # [N, 3], x y z

                vec_all = point_cloud_tensor - lidar_origin
                range_all = torch.norm(vec_all, dim=1)

                # 3. Range filter
                keep_mask = (range_all > EPS) & (range_all <= MAX_RANGE)
                if not keep_mask.any():
                    sampled_points, sampled_labels, sampled_grid_ind, sampled_grid_ind_vox = \
                        make_empty_outputs(curr_points, curr_labels, curr_grid_ind, curr_grid_ind_vox)
                    results_points.append(sampled_points)
                    results_labels.append(sampled_labels)
                    results_grid_ind.append(sampled_grid_ind)
                    results_grid_ind_vox.append(sampled_grid_ind_vox)
                    continue
                kept_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
                kept_points_raw = curr_points[kept_idx]
                kept_labels = curr_labels[kept_idx]
                kept_grid_ind = curr_grid_ind[kept_idx]
                kept_grid_ind_vox = curr_grid_ind_vox[kept_idx]

                kept_xyz = point_cloud_tensor[kept_idx]   # [M, 3]
                kept_vec = vec_all[kept_idx]              # [M, 3]
                kept_range = range_all[kept_idx]          # [M]
                kept_range_safe = torch.clamp(kept_range, min=EPS)

                # 4. Map each point to its nearest ray centre
                point_az = torch.atan2(kept_vec[:, 1], kept_vec[:, 0]) % (2.0 * torch.pi)
                point_el = torch.asin(
                    torch.clamp(kept_vec[:, 2] / kept_range_safe, -1.0, 1.0)
                )
                az_bin = torch.round(point_az / horizontal_step).long() % horizontal_resolution
                if beams > 1:
                    el_bin = torch.round((point_el - vertical_angles_rad[0]) / max(vertical_step, EPS)).long()
                else:
                    el_bin = torch.zeros_like(az_bin)
                el_bin.clamp_(0, beams - 1)
                flat_bin = el_bin * horizontal_resolution + az_bin
                ray_dir_for_points = ray_directions[flat_bin]

                # 6. Angular consistency gate, avoiding a costly arccos.
                #    az_margin/el_margin are the full half-widths, so every point falls
                #    inside the footprint of its own nearest bin.
                unit_vec = kept_vec / kept_range.unsqueeze(1)
                cos_sim = torch.sum(unit_vec * ray_dir_for_points, dim=1).clamp_(-1.0, 1.0)

                assigned_az = horizontal_angles_rad[az_bin]
                assigned_el = vertical_angles_rad[el_bin]

                diff_az = torch.abs(point_az - assigned_az)
                delta_az = torch.min(diff_az, 2.0 * math.pi - diff_az)
                delta_el = torch.abs(point_el - assigned_el)
                depths = torch.sum(kept_vec * ray_dir_for_points, dim=1)

                valid_mask = (
                    (depths > 0.0) &
                    (delta_az <= az_margin_val) &
                    (delta_el <= el_margin_val) &
                    (cos_sim >= cos_footprint)
                )
                if not valid_mask.any():
                    sp, sl, sgi, sgiv = make_empty_outputs(curr_points, curr_labels, curr_grid_ind, curr_grid_ind_vox)
                    results_points.append(sp); results_labels.append(sl)
                    results_grid_ind.append(sgi); results_grid_ind_vox.append(sgiv)
                    continue
                valid_bins = flat_bin[valid_mask]
                valid_depths = depths[valid_mask]

                # 7. First pass: keep only the closest point per bin (plain z-buffer)
                num_rays = beams * horizontal_resolution
                min_depth_flat, min_idx_in_valid = scatter_min(
                    valid_depths,
                    valid_bins,
                    dim=0,
                    dim_size=num_rays
                )
                hit_mask_flat = torch.isfinite(min_depth_flat)
                K = valid_depths.shape[0]
                idx_ok = (min_idx_in_valid >= 0) & (min_idx_in_valid < K)
                hit_mask_flat = hit_mask_flat & idx_ok

                # Map indices from the valid subset back into the kept subset.
                valid_pos = valid_mask.nonzero(as_tuple=False).squeeze(1)
                chosen_kept_idx_flat = torch.full(
                    (num_rays,), -1, device=device, dtype=torch.long
                )
                chosen_kept_idx_flat[hit_mask_flat] = valid_pos[min_idx_in_valid[hit_mask_flat]]

                depth_img = min_depth_flat.view(beams, horizontal_resolution)          # [B, H]
                hit_img = hit_mask_flat.view(beams, horizontal_resolution)             # [B, H]
                idx_img = chosen_kept_idx_flat.view(beams, horizontal_resolution)      # [B, H]

                MINORITY_DEPTH_TOL = 0.4   # metres

                # Minority-class subset of the valid candidates.
                valid_labels_all = kept_labels[valid_pos]            # [V]，V = valid_mask.sum()
                is_minor_valid = torch.zeros_like(valid_labels_all, dtype=torch.bool)
                for _cls in MINORITY_CLASSES:
                    is_minor_valid = is_minor_valid | (valid_labels_all == _cls)

                if is_minor_valid.any():
                    minor_bins = valid_bins[is_minor_valid]
                    minor_depths = valid_depths[is_minor_valid]
                    minor_local_pos = is_minor_valid.nonzero(as_tuple=False).squeeze(1)   # index into valid_*

                    # Second scatter_min over minority candidates: closest minority point per bin.
                    minor_min_depth, minor_min_idx = scatter_min(
                        minor_depths, minor_bins, dim=0, dim_size=num_rays
                    )
                    minor_hit = torch.isfinite(minor_min_depth)
                    K_minor = minor_depths.shape[0]
                    minor_ok = (minor_min_idx >= 0) & (minor_min_idx < K_minor)
                    minor_hit = minor_hit & minor_ok

                    # Promote only if the minority point is within TOL of the bin's closest depth.
                    promote_mask = (
                        minor_hit &
                        hit_mask_flat &
                        (minor_min_depth <= min_depth_flat + MINORITY_DEPTH_TOL)
                    )

                    # Also cover bins with no prior hit but a minority candidate.
                    promote_only_minor = minor_hit & (~hit_mask_flat)
                    promote_mask = promote_mask | promote_only_minor

                    if promote_mask.any():
                        new_kept_idx = valid_pos[minor_local_pos[minor_min_idx[promote_mask]]]
                        chosen_kept_idx_flat[promote_mask] = new_kept_idx
                        min_depth_flat = torch.where(promote_mask, minor_min_depth, min_depth_flat)
                        hit_mask_flat = hit_mask_flat | promote_only_minor

                # Refresh the per-bin images after promotion.
                depth_img = min_depth_flat.view(beams, horizontal_resolution)
                hit_img   = hit_mask_flat.view(beams, horizontal_resolution)
                idx_img   = chosen_kept_idx_flat.view(beams, horizontal_resolution)

                # 7.5 Per-bin semantic class and minority mask, used by steps 8 and 9.
                safe_idx_img = torch.clamp(idx_img, min=0)        # -1 -> 0 so the gather is in range
                labels_lookup = kept_labels[safe_idx_img]         # [B, H]; entries at -1 are garbage
                minority_bin_mask = torch.zeros_like(hit_img)
                for _cls in MINORITY_CLASSES:
                    minority_bin_mask = minority_bin_mask | (labels_lookup == _cls)
                minority_bin_mask = minority_bin_mask & hit_img   # valid only where a point was hit

                # 8. Second pass: occlusion suppression. Preserves continuous distant ground
                #    and drops only background points genuinely hidden by foreground.
                #    Minority-class bins bypass leak_mask entirely.
                depth_for_occ = torch.where(hit_img, depth_img, depth_img.new_tensor(LARGE_VAL))

                # 8.1 Collect occluders from beams above only, not symmetrically.
                az_pad = OCC_AZ_WIN // 2
                upper_stack_list = []
                for db in range(1, OCC_EL_WIN + 1):
                    src_b = torch.clamp(torch.arange(beams, device=device) + db, max=beams - 1)
                    upper_depth = depth_for_occ[src_b]
                    upper_pad = F.pad(
                        upper_depth.unsqueeze(0).unsqueeze(0),
                        (az_pad, az_pad, 0, 0), mode='circular'
                    ).squeeze(0).squeeze(0)
                    upper_unf = upper_pad.unfold(1, OCC_AZ_WIN, 1)
                    upper_stack_list.append(upper_unf)

                if upper_stack_list:
                    upper_stack = torch.cat(upper_stack_list, dim=2)
                    occ_local_min = upper_stack.amin(dim=2)
                    near_support = (upper_stack <= (occ_local_min.unsqueeze(2) + OCC_NEAR_BAND)).sum(dim=2)
                else:
                    occ_local_min = torch.full_like(depth_img, LARGE_VAL)
                    near_support = torch.zeros_like(depth_img, dtype=torch.long)

                # 8.2 Same-ring continuity protection, vectorised with unfold.
                ring_pad = RING_KEEP_WIN
                depth_pad = F.pad(
                    depth_img.unsqueeze(0).unsqueeze(0),
                    (ring_pad, ring_pad, 0, 0), mode='circular'
                ).squeeze(0).squeeze(0)
                hit_pad = F.pad(
                    hit_img.unsqueeze(0).unsqueeze(0).float(),
                    (ring_pad, ring_pad, 0, 0), mode='circular'
                ).squeeze(0).squeeze(0).bool()
                win_size = 2 * RING_KEEP_WIN + 1
                depth_unf = depth_pad.unfold(1, win_size, 1)
                hit_unf = hit_pad.unfold(1, win_size, 1)
                neighbor_idx = (0, 1, 2, 4, 5, 6)  # centre excluded
                depth_neighbors = depth_unf[:, :, neighbor_idx]
                hit_neighbors = hit_unf[:, :, neighbor_idx]
                depth_diff_ok = torch.abs(depth_neighbors - depth_img.unsqueeze(2)) <= RING_KEEP_BAND
                same_ring_support = (hit_neighbors & depth_diff_ok).sum(dim=2)
                ring_keep_mask = hit_img & (same_ring_support >= RING_KEEP_MIN)

                # 8.3 Range-adaptive depth threshold.
                adaptive_margin = OCC_DEPTH_MARGIN + OCC_REL_MARGIN * occ_local_min

                # 8.4 Drop only points that genuinely look occluded; ~minority_bin_mask
                #     makes minority classes immune to occlusion suppression.
                leak_mask = (
                    hit_img &
                    torch.isfinite(occ_local_min) &
                    (near_support >= OCC_SUPPORT_MIN) &
                    ((depth_img - occ_local_min) > adaptive_margin) &
                    (~ring_keep_mask) &
                    (~minority_bin_mask)   # minority-class protection
                )
                hit_img = hit_img & (~leak_mask)
                idx_img = torch.where(hit_img, idx_img, torch.full_like(idx_img, -1))
                depth_img = torch.where(hit_img, depth_img, depth_img.new_tensor(LARGE_VAL))
                # minority_bin_mask needs no refresh: leak_mask never removes minority bins,
                # so their hit/idx/labels are unchanged.

                # 9. Third pass: range-adaptive ring continuity (vectorised).
                #    Minority-class bins are forced to blend_factor 0.
                med_row, spread_row, support_row = circular_median_filter_2d(
                    depth_img, hit_img, SMOOTH_WIN
                )
                abs_diff = torch.abs(depth_img - med_row)
                trend_band = torch.clamp(depth_img * SMOOTH_TREND_REL_BAND, min=SMOOTH_TREND_ABS_BAND)
                trend_ok = abs_diff <= (3.0 * trend_band)
                smoothable = (
                    hit_img &
                    (support_row >= SMOOTH_SUPPORT_MIN) &
                    ((spread_row <= SMOOTH_SPREAD_MAX) | trend_ok)
                )
                adaptive_thresh = torch.maximum(
                    torch.full_like(depth_img, SMOOTH_ABS_FLOOR),
                    depth_img * SMOOTH_REL_THRESH
                )
                outlier_mask = smoothable & (abs_diff > adaptive_thresh)
                normal_mask = smoothable & (~outlier_mask)
                blend = SMOOTH_BLEND_NEAR + (
                    SMOOTH_BLEND_FAR - SMOOTH_BLEND_NEAR
                ) * torch.clamp(depth_img / SMOOTH_BLEND_REF, 0.0, 1.0)
                blend_factor = torch.where(
                    outlier_mask, blend,
                    torch.where(normal_mask, 0.5 * blend, torch.zeros_like(blend))
                )
                # Never smooth minority bins: the median would drag them to background depth.
                blend_factor = torch.where(
                    minority_bin_mask, torch.zeros_like(blend_factor), blend_factor
                )
                new_row = torch.lerp(depth_img, med_row, blend_factor)
                final_depth_img = torch.where(hit_img, new_row, depth_img)

                # 10. Emit real points: keep the selected point but move its xyz onto the
                #     ray at the smoothed depth from final_depth_img.
                final_hit_flat = hit_img.view(-1)
                final_idx_flat = idx_img.view(-1)
                final_depth_flat = final_depth_img.view(-1)
                valid_out_mask = final_hit_flat & (final_idx_flat >= 0) & torch.isfinite(final_depth_flat)
                if not valid_out_mask.any():
                    sampled_points, sampled_labels, sampled_grid_ind, sampled_grid_ind_vox = \
                        make_empty_outputs(curr_points, curr_labels, curr_grid_ind, curr_grid_ind_vox)
                    results_points.append(sampled_points)
                    results_labels.append(sampled_labels)
                    results_grid_ind.append(sampled_grid_ind)
                    results_grid_ind_vox.append(sampled_grid_ind_vox)
                    continue
                chosen_kept_idx = final_idx_flat[valid_out_mask]                  # index into kept_*
                chosen_bins = valid_out_mask.nonzero(as_tuple=False).squeeze(1)   # flat bin ids
                chosen_depths = final_depth_flat[valid_out_mask].unsqueeze(1)     # [S, 1]
                sampled_points = kept_points_raw[chosen_kept_idx].clone()
                sampled_labels = kept_labels[chosen_kept_idx]
                sampled_grid_ind = kept_grid_ind[chosen_kept_idx]
                sampled_grid_ind_vox = kept_grid_ind_vox[chosen_kept_idx]
                chosen_ray_dirs = ray_directions[chosen_bins]                     # [S, 3]
                projected_xyz = lidar_origin.unsqueeze(0) + chosen_depths * chosen_ray_dirs
                # Write xyz back into the stored column layout.
                sampled_points[:, 6] = projected_xyz[:, 0]  # x
                sampled_points[:, 7] = projected_xyz[:, 1]  # y
                sampled_points[:, 5] = projected_xyz[:, 2]  # z
                results_points.append(sampled_points)
                results_labels.append(sampled_labels)
                results_grid_ind.append(sampled_grid_ind)
                results_grid_ind_vox.append(sampled_grid_ind_vox)

            return results_points, results_grid_ind, results_grid_ind_vox, results_labels, sensor_params


        # CBA-CL: draw several diverse LiDAR configurations for the same scene.
        if self.consistency:
            all_points, all_grid_ind, all_grid_ind_vox, all_labels = [], [], [], []
            sensor_params_entries = []
            diverse_sensor_params = _sample_diverse_sensor_params(self.sample_repeat_count)

            for i in range(self.sample_repeat_count):
                res_points, res_grid, res_grid_vox, res_labels, sensor_params = _run_single_sampling(
                    sensor_params_tuple=diverse_sensor_params[i])
                all_points.extend(res_points)
                all_grid_ind.extend(res_grid)
                all_grid_ind_vox.extend(res_grid_vox)
                all_labels.extend(res_labels)
                sensor_params_entries.extend([sensor_params for _ in res_points])

                # Release cached blocks between rounds; each round holds a full extra view.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            #save results
            save_tensor_as_ply(all_points[0], "Consistency_1.ply")
            all_labels[0].cpu().numpy().tofile("Consistency_1.bin")
            save_tensor_as_ply(all_points[1], "Consistency_2.ply")
            all_labels[1].cpu().numpy().tofile("Consistency_2.bin")

            sensor_params_tensor = torch.stack(sensor_params_entries, dim=0) if sensor_params_entries else None
            return all_points, all_grid_ind, all_grid_ind_vox, all_labels, sensor_params_tensor

        res_points, res_grid, res_grid_vox, res_labels, sensor_params = _run_single_sampling()
        sensor_params_tensor = torch.stack([sensor_params for _ in res_points], dim=0) if res_points else None
        return res_points, res_grid, res_grid_vox, res_labels, sensor_params_tensor

def train_grid_features_to_xyz_numpy(feature_2d):
    """XYZ from PointsegMapping ``train_grid`` layout (cols xy + z slot, same as legacy ply)."""
    if isinstance(feature_2d, torch.Tensor):
        arr = feature_2d.detach().float().cpu().numpy()
    else:
        arr = np.asarray(feature_2d, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.shape[1] < 8:
        if arr.shape[1] >= 3:
            return arr[:, :3].astype(np.float64)
        raise ValueError(
            f'train_grid needs >=8 cols for polar layout, got {arr.shape[1]}')
    xyz = np.concatenate((arr[:, 6:8], arr[:, 5:6]), axis=1).astype(np.float64)
    return xyz


def save_tensor_as_ply(tensor, file_path):
    """Save ``train_grid``-style point features as PLY (xyz only)."""
    xyz = train_grid_features_to_xyz_numpy(tensor)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(file_path, point_cloud)

# cache for sample_beams: (lower, upper) -> (beam_range, probability_density)
_sample_beams_cache = {}


def sample_beams(lower_beams_bound, upper_beams_bound):
    key = (lower_beams_bound, upper_beams_bound)
    if key not in _sample_beams_cache:
        variance = 16
        std_dev = np.sqrt(variance)
        candidate_means = [16, 32, 64, 80, 128]
        valid_means = [m for m in candidate_means if lower_beams_bound <= m <= upper_beams_bound]
        beam_range = np.arange(lower_beams_bound, upper_beams_bound + 1, dtype=np.int64)
        probability_density = np.zeros(len(beam_range), dtype=np.float64)
        for mean in valid_means:
            probability_density += np.exp(-0.5 * ((beam_range - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))
        probability_density /= probability_density.sum()
        _sample_beams_cache[key] = (beam_range, probability_density)
    beam_range, probability_density = _sample_beams_cache[key]
    return int(np.random.choice(beam_range, p=probability_density))


def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]

    return nuScenes_label_name

def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)
