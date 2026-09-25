#!/usr/bin/env python3
"""A3 v2 Step 0/1/3: per-frame histograms for one (dataset, set).

Two departures from the original run_a3_f1f4.py, both required by the revised
protocol:

  * histograms are kept PER FRAME, never pooled. Pooling 200 frames averages
    the frame-to-frame variation away, which is the very quantity the
    real-to-real baseline is supposed to measure.
  * per-point features are computed on the WHOLE frame (a neighbourhood must
    see every point), then the point set is subsampled to a fixed N* before
    histogramming, so the JS estimation bias is identical on both sides of
    every comparison regardless of how many points a scan has.

N* is 10,000 rather than the 20,000 of the written protocol: the sparsest draw
the training curriculum can make (16 beams x 900 rays) cannot produce 20,000
points at all, so that threshold would have silently deleted the sparse tail of
the curriculum -- exactly the part the coverage question is about.

Usage:
    python review_analysis/perframe.py --dataset semantickitti --set sim_cond
"""
import argparse
import os
import os.path as osp
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy.spatial as spatial

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from A3_run_f1f4 import (F1_EDGES, F2_EDGES, F3_EDGES, F4_EDGES,
                         F3_RADIUS, F3_MIN_NBR, GROUND_IDS)
from A3_sets import select, load_frames, make_sim
from A3_simulator import estimate_ground_z

N_STAR = 10000
SEED = 0
OUT_DIR = 'review_analysis/_source_data/A3_perframe'
KEYS = ['f1', 'f2', 'f3_lin', 'f3_plan', 'f3_scat', 'f4']
NBINS = {'f1': len(F1_EDGES) - 1, 'f2': len(F2_EDGES) - 1, 'f3_lin': len(F3_EDGES) - 1,
         'f3_plan': len(F3_EDGES) - 1, 'f3_scat': len(F3_EDGES) - 1,
         'f4': len(F4_EDGES) - 1}


def point_values(xyz, lab, gids):
    """Every per-point quantity, computed on the full frame."""
    n = len(xyz)
    r = np.sqrt((xyz ** 2).sum(1))

    tree = spatial.cKDTree(xyz)
    nn = tree.query(xyz, k=2, workers=-1)[0][:, 1] if n >= 2 else np.zeros(n)

    lin = plan = scat = np.zeros(0)
    f3_idx = np.zeros(0, np.int64)
    if n >= F3_MIN_NBR:
        nbrs = tree.query_ball_point(xyz, r=F3_RADIUS, workers=-1)
        cnt = np.fromiter((len(x) for x in nbrs), np.int64, count=n)
        keep = np.flatnonzero(cnt >= F3_MIN_NBR)
        if len(keep):
            flat = np.concatenate([np.asarray(nbrs[i], np.int64) for i in keep])
            owner = np.repeat(np.arange(len(keep)), cnt[keep])
            patch = xyz[flat] - xyz[keep][owner]
            cov = np.empty((len(keep), 3, 3))
            for a in range(3):
                for b in range(a, 3):
                    s = np.bincount(owner, weights=patch[:, a] * patch[:, b],
                                    minlength=len(keep))
                    cov[:, a, b] = cov[:, b, a] = s
            cov /= cnt[keep][:, None, None]
            eig = np.maximum(np.linalg.eigvalsh(cov), 0.0)     # ascending
            l3, l2, l1 = eig[:, 0], eig[:, 1], eig[:, 2]
            good = eig.sum(1) >= 1e-12
            d = np.maximum(l1, 1e-12)
            lin, plan, scat = ((l1 - l2) / d)[good], ((l2 - l3) / d)[good], (l3 / d)[good]
            f3_idx = keep[good]

    gz, hit = estimate_ground_z(xyz, lab, gids)
    return {'r': r, 'nn': nn, 'lin': lin, 'plan': plan, 'scat': scat,
            'f3_idx': f3_idx, 'z_rel': xyz[:, 2] - gz,
            'ground_z': gz, 'ground_hit': hit, 'n_full': n}


def histograms(pv, rng):
    """Subsample the point set to N*, then bin. Returns (hists, meta)."""
    n = pv['n_full']
    if n < N_STAR:
        return None, {'skipped': True, 'n_full': n}
    sub = rng.choice(n, N_STAR, replace=False)
    keep_mask = np.zeros(n, bool)
    keep_mask[sub] = True

    h = {}
    for key, vals, edges in [('f1', pv['r'][sub], F1_EDGES),
                             ('f2', pv['nn'][sub], F2_EDGES),
                             ('f4', pv['z_rel'][sub], F4_EDGES)]:
        c, _ = np.histogram(vals, bins=edges)
        h[key] = c / max(c.sum(), 1)

    sel3 = keep_mask[pv['f3_idx']] if len(pv['f3_idx']) else np.zeros(0, bool)
    n_f3 = int(sel3.sum())
    for key, vals in [('f3_lin', pv['lin']), ('f3_plan', pv['plan']),
                      ('f3_scat', pv['scat'])]:
        if n_f3 >= 100:
            c, _ = np.histogram(vals[sel3], bins=F3_EDGES)
            h[key] = c / max(c.sum(), 1)
        else:
            h[key] = np.zeros(NBINS[key])
    return h, {'skipped': False, 'n_full': n, 'n_f3_eff': n_f3,
               'f3_skip_frac': 1.0 - n_f3 / N_STAR,
               'ground_z': pv['ground_z'], 'ground_hit': pv['ground_hit']}


_CTX = {}


def _init(dataset, which):
    _CTX['dataset'] = dataset
    _CTX['which'] = which


def _work(job):
    i, spec = job
    ds, which = _CTX['dataset'], _CTX['which']
    gids = GROUND_IDS[ds]
    fr = load_frames(ds, [spec])[0]
    if which in ('realA', 'realB'):
        xyz, lab = fr['real_xyz'], fr['real_lab']
        info = {}
    else:
        xyz, lab, info = make_sim(ds, fr, which, frame_seed=SEED * 100000 + i)
    if len(xyz) < 50:
        return i, None, {'skipped': True, 'n_full': int(len(xyz))}
    pv = point_values(np.ascontiguousarray(xyz[:, :3], np.float64), lab, gids)
    h, meta = histograms(pv, np.random.RandomState(SEED * 100000 + i))
    meta['sim_ground_z'] = info.get('scene_ground_z', np.nan)
    meta['lidar_height'] = info.get('lidar_height', np.nan)
    return i, h, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    choices=['semantickitti', 'nuscenes', 'waymo'])
    ap.add_argument('--set', required=True, dest='which',
                    choices=['realA', 'realB', 'sim_cur', 'sim_cond'])
    ap.add_argument('--nproc', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    A, B = select(a.dataset)
    specs = B if a.which == 'realB' else A          # every sim comes from A
    if a.limit:
        specs = specs[:a.limit]
    jobs = list(enumerate(specs))

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print(f'[{a.dataset}/{a.which}] {len(jobs)} frames, {a.nproc} workers', flush=True)

    res = {}
    with Pool(a.nproc, initializer=_init, initargs=(a.dataset, a.which)) as p:
        for k, (i, h, meta) in enumerate(p.imap_unordered(_work, jobs, chunksize=1), 1):
            res[i] = (h, meta)
            if k % 25 == 0:
                print(f'  {k}/{len(jobs)}  {time.time()-t0:.0f}s', flush=True)

    order = sorted(res)
    ok = [i for i in order if res[i][0] is not None]
    out = {k: np.stack([res[i][0][k] for i in ok]) for k in KEYS}
    out['frame_index'] = np.asarray(ok, np.int32)
    for f in ['n_full', 'n_f3_eff', 'f3_skip_frac', 'ground_z', 'ground_hit',
              'sim_ground_z', 'lidar_height']:
        out[f] = np.asarray([res[i][1].get(f, np.nan) for i in ok], np.float64)
    out['n_skipped'] = np.asarray([len(order) - len(ok)])
    out['N_STAR'] = np.asarray([N_STAR])

    path = osp.join(OUT_DIR, f'{a.dataset}_{a.which}.npz')
    np.savez_compressed(path, **out)
    gh = out['ground_hit']
    print(f'[{a.dataset}/{a.which}] kept {len(ok)}/{len(order)} '
          f'(skipped {len(order)-len(ok)} below N*={N_STAR}), '
          f'{time.time()-t0:.0f}s -> {path}', flush=True)
    print(f'  pts/frame mean={out["n_full"].mean():.0f}  '
          f'F3 eff mean={np.nanmean(out["n_f3_eff"]):.0f} '
          f'(skip {np.nanmean(out["f3_skip_frac"]):.3f})  '
          f'ground hit mean={np.nanmean(gh):.3f}', flush=True)
    if np.nanmean(gh) < 0.10:
        print(f'  WARNING ground-class hit rate {np.nanmean(gh):.3f} < 0.10 -- '
              f'F4 is falling back to the percentile estimator', flush=True)


if __name__ == '__main__':
    main()
