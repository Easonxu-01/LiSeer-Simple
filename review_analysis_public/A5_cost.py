#!/usr/bin/env python3
"""A5: measured compute cost of the LiSeer components (D31-D35).

Protocol: run only on an idle machine -- any other load makes the numbers
meaningless. Every component switch used here is a CONFIG flag (Random /
sensor_film / consistency), so no model code is modified. The one piece of
instrumentation, timing Phi_samp, is a runtime monkey-patch of
PointTPV_Seg._sampling that is never written to disk.

  D31  Phi_samp ms/iteration and its share of one iteration
  D32  S-Film ms/iteration (A/B) and its exact added parameter count
  D33  CBA-CL ms/iteration (A/B)
  D34  P-baseline inference FPS
  D35  LiSeer-P inference FPS

Usage:
    python review_analysis/A5_cost.py --mode train --config <cfg> --tag full
    python review_analysis/A5_cost.py --mode fps   --config <cfg> --tag liseer
    python review_analysis/A5_cost.py --mode params --config <cfg>
"""
import argparse
import json
import os
import os.path as osp
import time

import numpy as np
import torch


def _npoints(data):
    """Point count of the frame in this batch, whatever container holds it."""
    import torch as _t
    for key in ('points', 'points_occ', 'img_inputs'):
        v = data.get(key)
        if v is None:
            continue
        d = getattr(v, 'data', v)
        while isinstance(d, (list, tuple)) and d:
            d = d[0]
        if isinstance(d, _t.Tensor) and d.dim() >= 2:
            return int(d.shape[0])
    return float('nan')


def build(cfg_path, for_train=True):
    from mmcv import Config
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataset, build_dataloader
    import importlib
    cfg = Config.fromfile(cfg_path)
    if cfg.get('plugin'):
        importlib.import_module(
            cfg.get('plugin_dir', 'projects/unilidar_plugin/')
            .rstrip('/').replace('/', '.'))
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    ds_cfg = cfg.data.train if for_train else cfg.data.val
    if ds_cfg.get('type') == 'CBGSDataset':
        ds_cfg = ds_cfg['dataset']
    ds_cfg = dict(ds_cfg)
    ds_cfg.pop('test_mode', None)
    if not for_train:
        ds_cfg['test_mode'] = True
    dataset = build_dataset(ds_cfg)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=4,
                              num_gpus=1, dist=False, shuffle=for_train,
                              seed=0)
    return cfg, model, dataset, loader


# ---------------------------------------------------------------- D31/D32/D33
def run_train(cfg_path, tag, n_warm=50, n_meas=200, time_sampler=False):
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import build_optimizer

    cfg, model, dataset, loader = build(cfg_path, for_train=True)
    model.init_weights()
    model = MMDataParallel(model.cuda(), device_ids=[0])
    optimizer = build_optimizer(model, cfg.optimizer)

    samp = {'t': 0.0, 'n': 0}
    if time_sampler:
        inner = model.module
        orig = type(inner)._sampling

        def timed(self, *a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig(self, *a, **kw)
            torch.cuda.synchronize()
            samp['t'] += time.perf_counter() - t0
            samp['n'] += 1
            return out
        type(inner)._sampling = timed

    per_iter, samp_iter, npts = [], [], []
    it = iter(loader)
    total = n_warm + n_meas
    for k in range(total):
        try:
            data = next(it)
        except StopIteration:
            it = iter(loader)
            data = next(it)
        if k == n_warm:
            samp['t'], samp['n'] = 0.0, 0
        before_t, before_n = samp['t'], samp['n']
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses = model.train_step(data, optimizer)
        loss = losses['loss'] if isinstance(losses, dict) else losses
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if k >= n_warm:
            per_iter.append(dt * 1000.0)
            samp_iter.append((samp['t'] - before_t) * 1000.0)
            npts.append(_npoints(data))
        if (k + 1) % 50 == 0:
            print(f'  {tag}: {k+1}/{total}  iter={np.median(per_iter or [dt*1000]):.1f} ms',
                  flush=True)

    if time_sampler:
        type(model.module)._sampling = orig
    out = {'tag': tag, 'config': cfg_path, 'n_measured': len(per_iter),
           'per_iter_ms': [round(v, 4) for v in per_iter],
           'sampler_per_iter_ms': ([round(v, 4) for v in samp_iter]
                                   if time_sampler else None),
           'iter_ms_mean': float(np.mean(per_iter)),
           'iter_ms_median': float(np.median(per_iter)),
           'iter_ms_p95': float(np.percentile(per_iter, 95)),
           'points_mean': (float(np.nanmean(npts)) if npts and np.isfinite(npts).any() else None)}
    if time_sampler:
        out.update({'sampler_ms_mean': float(np.mean(samp_iter)),
                    'sampler_ms_median': float(np.median(samp_iter)),
                    'sampler_ms_p95': float(np.percentile(samp_iter, 95)),
                    'sampler_calls_per_iter': samp['n'] / max(len(per_iter), 1),
                    'sampler_pct_of_iter': 100.0 * float(np.median(samp_iter)) /
                                           float(np.median(per_iter))})
    return out


# ------------------------------------------------------------------- D34/D35
def run_fps(cfg_path, tag, n_warm=50, n_meas=200):
    from mmcv.parallel import MMDataParallel
    cfg, model, dataset, loader = build(cfg_path, for_train=False)
    model.init_weights()
    model = MMDataParallel(model.cuda().eval(), device_ids=[0])
    lat, npts = [], []
    it = iter(loader)
    with torch.no_grad():
        for k in range(n_warm + n_meas):
            try:
                data = next(it)
            except StopIteration:
                it = iter(loader)
                data = next(it)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if k >= n_warm:
                lat.append(dt * 1000.0)
                npts.append(_npoints(data))
            if (k + 1) % 50 == 0:
                print(f'  {tag}: {k+1}/{n_warm+n_meas}  '
                      f'{np.median(lat or [dt*1000]):.1f} ms', flush=True)
    lat = np.asarray(lat)
    return {'tag': tag, 'config': cfg_path, 'n_measured': len(lat),
            'latency_ms': [round(float(v), 4) for v in lat],
            'latency_ms_mean': float(lat.mean()),
            'latency_ms_median': float(np.median(lat)),
            'latency_ms_p95': float(np.percentile(lat, 95)),
            'fps_mean': float(1000.0 / lat.mean()),
            'fps_median': float(1000.0 / np.median(lat)),
            'fps_p95': float(1000.0 / np.percentile(lat, 95)),
            'points_mean': (float(np.nanmean(npts)) if npts and np.isfinite(npts).any() else None)}


# ----------------------------------------------------------------- D32 params
def count_params(cfg_path):
    cfg, model, _, _ = build(cfg_path, for_train=True)
    total = sum(p.numel() for p in model.parameters())
    named = {}
    for name, mod in model.named_modules():
        low = name.lower()
        if ('sensor' in low or 'film' in low) and '.' not in name.split('sensor')[-1][:1]:
            n = sum(p.numel() for p in mod.parameters(recurse=True))
            if n:
                named[name] = n
    # keep only top-level S-Film blocks, not their children
    keep = {}
    for k in sorted(named, key=len):
        if not any(k.startswith(p + '.') for p in keep):
            keep[k] = named[k]
    return {'total_params': total, 'sfilm_modules': keep,
            'sfilm_params': int(sum(keep.values())),
            'sfilm_params_M': round(sum(keep.values()) / 1e6, 4),
            'total_params_M': round(total / 1e6, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['train', 'fps', 'params'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--warm', type=int, default=50)
    ap.add_argument('--meas', type=int, default=200)
    ap.add_argument('--time-sampler', action='store_true')
    ap.add_argument('--out', default='review_analysis/_source_data/A5_raw')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or osp.basename(a.config).replace('.py', '')

    if a.mode == 'train':
        r = run_train(a.config, tag, a.warm, a.meas, a.time_sampler)
    elif a.mode == 'fps':
        r = run_fps(a.config, tag, a.warm, a.meas)
    else:
        r = count_params(a.config)
        r['tag'] = tag
    r['gpu'] = torch.cuda.get_device_name(0)
    r['torch'] = torch.__version__
    r['cuda'] = torch.version.cuda
    p = osp.join(a.out, f'{a.mode}_{tag}.json')
    json.dump(r, open(p, 'w'), indent=2)
    print(json.dumps(r, indent=2))
    print(f'-> {p}')


if __name__ == '__main__':
    main()
