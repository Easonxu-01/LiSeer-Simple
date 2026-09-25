#!/usr/bin/env python3
"""Fit the Waymo TOP LiDAR origin in the vehicle frame.

Waymo clouds are stored in the vehicle frame, whose origin sits at the rear
axle, while the TOP LiDAR is on the roof well forward of it. Measuring
elevation from (0,0,z) therefore makes a single beam's apparent elevation swing
sinusoidally with azimuth (up to 3 deg, far more than the 0.36 deg beam pitch).
Fitting the true origin removes that swing.

The cost is the mean elevation spread inside each (range-image row, azimuth
sector) cell: at the true origin every such cell is one beam seen over a narrow
time window, so its spread collapses.
"""
import json
import os
import os.path as osp

import numpy as np

D = 'data/waymo/kitti_format/training/velodyne'
NCOL, NROW = 2650, 64
SEED = 0


def load_top(stem, max_pts=120000):
    a = np.fromfile(osp.join(D, stem + '.bin'), np.float32).reshape(-1, 6)
    idx = a[:, 5]
    m = (idx >= 0) & (idx < NROW * NCOL)
    row = (idx[m] // NCOL).astype(np.int32)
    col = (idx[m] % NCOL).astype(np.int32)
    p = a[m, :3].astype(np.float64)
    if len(p) > max_pts:
        s = np.random.RandomState(SEED).choice(len(p), max_pts, replace=False)
        p, row, col = p[s], row[s], col[s]
    return p, row, col


def cost(p, row, col, o, nsec=36, minp=20):
    rho = np.hypot(p[:, 0] - o[0], p[:, 1] - o[1])
    k = rho > 3.0
    el = np.degrees(np.arctan2(p[k, 2] - o[2], rho[k]))
    key = row[k] * nsec + (col[k] * nsec) // NCOL
    order = np.argsort(key, kind='stable')
    key, el = key[order], el[order]
    bnd = np.flatnonzero(np.diff(key)) + 1
    g = np.split(el, bnd)
    s = [x.std() for x in g if len(x) > minp]
    return float(np.mean(s)) if s else np.inf


def fit(p, row, col):
    best = (np.inf, (0.0, 0.0, 2.18))
    for ox in np.arange(-0.5, 2.51, 0.25):
        for oy in np.arange(-0.6, 0.61, 0.3):
            for oz in np.arange(1.9, 2.51, 0.1):
                c = cost(p, row, col, (ox, oy, oz))
                if c < best[0]:
                    best = (c, (ox, oy, oz))
    for step, span in [(0.08, 0.25), (0.02, 0.08)]:
        b = best[1]
        for ox in np.arange(b[0] - span, b[0] + span + 1e-9, step):
            for oy in np.arange(b[1] - span, b[1] + span + 1e-9, step):
                for oz in np.arange(b[2] - span, b[2] + span + 1e-9, step):
                    c = cost(p, row, col, (ox, oy, oz))
                    if c < best[0]:
                        best = (c, (round(float(ox), 3), round(float(oy), 3), round(float(oz), 3)))
    return best


def table(stems, o, nsec=36):
    tabs, drifts = [], []
    for s_ in stems:
        p, row, col = load_top(s_, max_pts=200000)
        rho = np.hypot(p[:, 0] - o[0], p[:, 1] - o[1])
        k = rho > 3.0
        el = np.degrees(np.arctan2(p[k, 2] - o[2], rho[k]))
        r, sec = row[k], (col[k] * nsec) // NCOL
        T = np.full((NROW, nsec), np.nan)
        for rr in range(NROW):
            mr = r == rr
            if mr.sum() < 50:
                continue
            e_, s_sec = el[mr], sec[mr]
            for q in range(nsec):
                mm = s_sec == q
                if mm.sum() > 20:
                    T[rr, q] = np.median(e_[mm])
        tabs.append(np.nanmedian(T, axis=1))
        drifts.append(np.nanstd(T, axis=1))
    tbl = np.nanmedian(np.asarray(tabs), axis=0)
    drift = float(np.nanmedian(np.asarray(drifts)))
    return tbl, drift


if __name__ == '__main__':
    stems = [s.strip().replace('.bin', '') for s in
             open('seg_annotated_indices.txt').read().split()]
    use = [s for s in stems if osp.exists(osp.join(D, s + '.bin'))][:5]

    p, row, col = load_top(use[0])
    c0 = cost(p, row, col, (0.0, 0.0, 2.18))
    sc, o = fit(p, row, col)
    print(f'origin fit: {o}  cost={sc:.4f} deg   (cost at (0,0,2.18) = {c0:.4f})')

    tbl, drift = table(use, o)
    fin = np.isfinite(tbl)
    t = np.sort(tbl[fin])
    d = np.diff(t)
    print(f'beams={fin.sum()}  FOV=[{t[0]:.3f}, {t[-1]:.3f}]')
    print(f'spacing med={np.median(d):.4f} min={d.min():.4f} max={d.max():.4f} '
          f'NONUNIFORMITY={d.max()/d.min():.2f}x')
    print(f'residual per-beam azimuth drift = {drift:.4f} deg')
    print('table:', [round(float(v), 4) for v in t])

    json.dump({'origin_xyz': list(o), 'fit_cost_deg': sc, 'cost_at_axis_origin': c0,
               'n_beams': int(fin.sum()), 'table_deg': [float(v) for v in t],
               'residual_drift_deg': drift,
               'nonuniformity': float(d.max() / d.min()),
               'src': 'MEASURED (range-image row index col5//2650; origin by per-(row,sector) '
                      'elevation-spread minimisation)'},
              open('review_analysis/_source_data/A3_waymo_beam_table.json', 'w'), indent=2)
