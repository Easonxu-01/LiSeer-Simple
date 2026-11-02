#!/usr/bin/env python3
"""A3 v2 Step 4: where does a family's remaining gap come from?

JS is a sum over bins, so it decomposes exactly:

    JS(p,q) = sum_b 0.5*( p_b log2(p_b/m_b) + q_b log2(q_b/m_b) ),  m=(p+q)/2

Averaging that per-bin term over all frame pairs says which part of the
distribution the divergence actually lives in, and the bin edges say what that
part is physically (a nearest-neighbour spacing, a range band, a height). A
mechanism is only reported when the decomposition supports it.

Writes review_analysis/_source_data/A3_diagnostics.md.
"""
import json
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from A3_run_f1f4 import F1_EDGES, F2_EDGES, F3_EDGES, F4_EDGES

EDGES = {'f1': F1_EDGES, 'f2': F2_EDGES, 'f3_lin': F3_EDGES, 'f3_plan': F3_EDGES,
         'f3_scat': F3_EDGES, 'f4': F4_EDGES}
UNITS = {'f1': 'm (range)', 'f2': 'm (nearest-neighbour spacing)',
         'f3_lin': 'linearity', 'f3_plan': 'planarity', 'f3_scat': 'scattering',
         'f4': 'm (height above ground)'}
DATASETS = ['semantickitti', 'nuscenes', 'waymo']
KEYS = list(EDGES)
TRIGGER = 3.0            # only diagnose families whose ratio CI lower bound > 3


def per_bin(P, Q, max_pairs=4000, seed=0):
    """Mean per-bin JS contribution over a sample of frame pairs."""
    rng = np.random.RandomState(seed)
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-15)
    Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-15)
    i = rng.randint(0, len(P), min(max_pairs, len(P) * len(Q)))
    j = rng.randint(0, len(Q), len(i))
    p, q = P[i], Q[j]
    m = 0.5 * (p + q)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(p > 0, p * np.log2(np.where(m > 0, p / m, 1.0)), 0.0)
        b = np.where(q > 0, q * np.log2(np.where(m > 0, q / m, 1.0)), 0.0)
    return (0.5 * (a + b)).mean(0)


def main():
    import csv
    rows = list(csv.DictReader(open('review_analysis/_source_data/A3_frame_level.csv')))
    nom = json.load(open('review_analysis/_source_data/A3_nominal_configs.json'))
    L = []
    w = L.append
    w('# A3 diagnosis: which bins the residual gap comes from (Step 4)\n')
    w('> The JS divergence is a sum over bins and decomposes exactly. The table below gives the largest-contributing bins and their physical location.')
    w('> Only families whose `ratio_cond` has a 95% CI lower bound above 3 are decomposed.\n')

    flagged = []
    for r in rows:
        if r['metric'] != 'JS' or r['level'] != 'cond':
            continue
        if float(r['ratio_ci_lo']) > TRIGGER:
            flagged.append(r)

    w(f'\n## Families that triggered the diagnosis: {len(flagged)}\n')
    if not flagged:
        w('None. Every family has a `ratio_cond` 95% CI lower bound of 3 or below.\n')

    w('| Dataset | Family | ratio_cond [95% CI] | overlap | Top-5 contributing bins | Share of JS |')
    w('|---|---|---|---|---|---|')
    detail = []
    for r in flagged:
        ds, key = r['dataset'], r['family']
        m = np.load(f'review_analysis/_source_data/A3_pairmats_{ds}.npz')
        S = np.load(f'review_analysis/_source_data/A3_perframe/{ds}_sim_cond.npz')[key]
        B = np.load(f'review_analysis/_source_data/A3_perframe/{ds}_realB.npz')[key]
        A = np.load(f'review_analysis/_source_data/A3_perframe/{ds}_realA.npz')[key]
        c_s = per_bin(S, B)
        c_r = per_bin(A, B)
        e = np.asarray(EDGES[key], float)
        if not np.isfinite(e[-1]):
            e = e.copy()
            e[-1] = e[-2] + (e[-2] - e[-3])
        top = np.argsort(c_s)[::-1][:5]
        frac = c_s[top].sum() / max(c_s.sum(), 1e-15)
        loc = '; '.join(f'[{e[b]:.3g},{e[b+1]:.3g})' for b in sorted(top))
        w(f'| {ds} | {key} | {float(r["ratio"]):.2f} '
          f'[{float(r["ratio_ci_lo"]):.2f},{float(r["ratio_ci_hi"]):.2f}] '
          f'| {float(r["overlap95"]):.3f} | {loc} | {frac:.1%} |')
        detail.append((ds, key, e, c_s, c_r, top))

    w('\n## Per-family detail\n')
    for ds, key, e, c_s, c_r, top in detail:
        n = nom[ds]
        w(f'\n### {ds} / {key}  ({UNITS[key]})\n')
        w('| Bin | Range | sim-real contribution | real-real contribution | Ratio |')
        w('|---|---|---|---|---|')
        for b in sorted(top):
            ratio = c_s[b] / max(c_r[b], 1e-12)
            w(f'| {b} | [{e[b]:.4g}, {e[b+1]:.4g}) | {c_s[b]:.6f} | {c_r[b]:.6f} | {ratio:.1f}× |')
        if key == 'f2':
            sp = n['spacing_med_deg']
            w(f'\nThe measured median beam spacing of this dataset is {sp} deg, its non-uniformity {n["nonuniformity"]}x,')
            w(f'and its ego-motion drift {n["ego_motion_drift_deg"]} deg.')
            w('A contribution concentrated in the small-spacing bins points to a difference between the beam spacing of the real scan and that of the simulated sampling;')
            w('a contribution concentrated in the large-spacing bins points to a difference in point density. The bin positions above separate the two.\n')
    open('review_analysis/_source_data/A3_diagnostics.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L[:40]))
    print(f'\nwrote review_analysis/_source_data/A3_diagnostics.md')


if __name__ == '__main__':
    main()
