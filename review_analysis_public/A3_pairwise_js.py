#!/usr/bin/env python3
"""A3 v2 Step 1/3: frame-to-frame distances, the revised protocol's core.

The pooled protocol averaged 200 histograms per side and took one distance.
Averaging removes the frame-to-frame variation, so its real-to-real baseline
measured an estimation residual that shrinks with n rather than the natural
scan-to-scan variability that the protocol is meant to capture.

Here every pair is kept:

    D_r2r  = { JS(realA_i,    realB_j) }        the natural variability
    D_cond = { JS(sim_cond_i, realB_j) }        fidelity, sensor conditioned
    D_cur  = { JS(sim_cur_i,  realB_j) }        the curriculum mixture

    ratio   = median(D_x) / median(D_r2r)
    overlap = P( d in D_x <= Q95(D_r2r) )

95% CIs come from resampling FRAMES with replacement (500x, seed=0) and
recomputing the medians and the ratio inside each resample -- the pair matrix
is computed once, so a resample is just an index into it.

Writes review_analysis/_source_data/A3_frame_level_{dataset}.csv and the pair matrices.
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from A3_run_f1f4 import F1_EDGES, F2_EDGES, F3_EDGES, F4_EDGES

IN_DIR = 'review_analysis/_source_data/A3_perframe'
KEYS = ['f1', 'f2', 'f3_lin', 'f3_plan', 'f3_scat', 'f4']
SCALAR = {'f1': F1_EDGES, 'f2': F2_EDGES, 'f4': F4_EDGES}
N_BOOT = 500
SEED = 0


def centres(edges):
    e = np.asarray(edges, float)
    if not np.isfinite(e[-1]):                 # F2's overflow bin
        e = e.copy()
        e[-1] = e[-2] + (e[-2] - e[-3])
    return 0.5 * (e[1:] + e[:-1])


def js_matrix(P, Q, chunk=32):
    """Pairwise Jensen-Shannon divergence, log base 2, in [0,1]."""
    P = np.asarray(P, np.float64)
    Q = np.asarray(Q, np.float64)
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-15)
    Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-15)
    out = np.empty((len(P), len(Q)))
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk][:, None, :]
        m = 0.5 * (p + Q[None, :, :])
        with np.errstate(divide='ignore', invalid='ignore'):
            lp = np.where(p > 0, p * np.log2(np.where(m > 0, p / m, 1.0)), 0.0)
            lq = np.where(Q[None] > 0,
                          Q[None] * np.log2(np.where(m > 0, Q[None] / m, 1.0)), 0.0)
        out[s:s + chunk] = 0.5 * (lp.sum(-1) + lq.sum(-1))
    return np.clip(out, 0.0, 1.0)


def w1_matrix(P, Q, edges, chunk=32):
    """1-D Wasserstein-1 between histograms on a common grid, in metres."""
    c = centres(edges)
    w = np.diff(np.concatenate([[c[0] - (c[1] - c[0]) / 2],
                                (c[1:] + c[:-1]) / 2,
                                [c[-1] + (c[-1] - c[-2]) / 2]]))
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-15)
    Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-15)
    cp, cq = np.cumsum(P, 1), np.cumsum(Q, 1)
    out = np.empty((len(P), len(Q)))
    for s in range(0, len(P), chunk):
        d = np.abs(cp[s:s + chunk][:, None, :] - cq[None, :, :])
        out[s:s + chunk] = (d * w[None, None, :]).sum(-1)
    return out


def boot(M_num, M_den, n_boot=N_BOOT, seed=SEED):
    """Frame-level bootstrap of median(num)/median(den) and the two medians."""
    rng = np.random.RandomState(seed)
    nS, nB = M_num.shape
    nA = M_den.shape[0]
    r, mn, md = [], [], []
    for _ in range(n_boot):
        iB = rng.randint(0, nB, nB)
        a = np.median(M_den[np.ix_(rng.randint(0, nA, nA), iB)])
        s = np.median(M_num[np.ix_(rng.randint(0, nS, nS), iB)])
        mn.append(s)
        md.append(a)
        r.append(s / max(a, 1e-15))
    q = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)),
                   float(np.percentile(v, 97.5)))
    return q(r), q(mn), q(md)


def load(ds, which):
    p = osp.join(IN_DIR, f'{ds}_{which}.npz')
    return np.load(p) if osp.exists(p) else None


def run(ds):
    sets = {w: load(ds, w) for w in ['realA', 'realB', 'sim_cur', 'sim_cond']}
    missing = [w for w, v in sets.items() if v is None]
    if missing:
        print(f'{ds}: missing {missing}, skipping')
        return []
    rows, mats = [], {}
    nA, nB = len(sets['realA']['f1']), len(sets['realB']['f1'])
    for key in KEYS:
        A, B = sets['realA'][key], sets['realB'][key]
        M_r2r = js_matrix(A, B)
        mats[f'{key}_r2r'] = M_r2r
        q95 = float(np.percentile(M_r2r, 95))
        for level, src in [('cond', 'sim_cond'), ('cur', 'sim_cur')]:
            S = sets[src][key]
            M = js_matrix(S, B)
            mats[f'{key}_{level}'] = M
            (rm, rlo, rhi), (sm, slo, shi), (dm, dlo, dhi) = boot(M, M_r2r)
            rows.append(dict(
                dataset=ds, family=key, metric='JS', level=level,
                n_pairs=M.size, median_r2r=float(np.median(M_r2r)),
                median_s2r=float(np.median(M)), ratio=rm, ratio_ci_lo=rlo,
                ratio_ci_hi=rhi, overlap95=float((M <= q95).mean()),
                median_r2r_ci=f'[{dlo:.5f},{dhi:.5f}]',
                median_s2r_ci=f'[{slo:.5f},{shi:.5f}]',
                n_frames_A=nA, n_frames_B=nB, n_frames_sim=len(S),
                n_skipped=int(sets[src]['n_skipped'][0]),
                notes=''))
        if key in SCALAR:
            e = SCALAR[key]
            W_r2r = w1_matrix(A, B, e)
            for level, src in [('cond', 'sim_cond'), ('cur', 'sim_cur')]:
                W = w1_matrix(sets[src][key], B, e)
                (rm, rlo, rhi), _, _ = boot(W, W_r2r)
                rows.append(dict(
                    dataset=ds, family=key, metric='W1', level=level,
                    n_pairs=W.size, median_r2r=float(np.median(W_r2r)),
                    median_s2r=float(np.median(W)), ratio=rm, ratio_ci_lo=rlo,
                    ratio_ci_hi=rhi,
                    overlap95=float((W <= np.percentile(W_r2r, 95)).mean()),
                    median_r2r_ci='', median_s2r_ci='',
                    n_frames_A=nA, n_frames_B=nB, n_frames_sim=len(sets[src][key]),
                    n_skipped=int(sets[src]['n_skipped'][0]),
                    notes='metres' if key != 'f2' else 'metres; F2 overflow bin '
                          'placed one width past the last edge'))
    np.savez_compressed(f'review_analysis/_source_data/A3_pairmats_{ds}.npz', **mats)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*',
                    default=['semantickitti', 'nuscenes', 'waymo'])
    a = ap.parse_args()
    import csv
    allrows = []
    for ds in a.datasets:
        r = run(ds)
        allrows += r
        for x in r:
            if x['metric'] == 'JS':
                print(f'{ds:14s} {x["family"]:8s} {x["level"]:5s} '
                      f'r2r={x["median_r2r"]:.5f} s2r={x["median_s2r"]:.5f} '
                      f'ratio={x["ratio"]:7.2f} [{x["ratio_ci_lo"]:.2f},{x["ratio_ci_hi"]:.2f}] '
                      f'overlap={x["overlap95"]:.3f}')
    if allrows:
        os.makedirs('review_analysis/_source_data', exist_ok=True)
        with open('review_analysis/_source_data/A3_frame_level.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
            w.writeheader()
            w.writerows(allrows)
        print(f'\nwrote review_analysis/_source_data/A3_frame_level.csv ({len(allrows)} rows)')


if __name__ == '__main__':
    main()
