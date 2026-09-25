#!/usr/bin/env python3
"""A3 Step 2 (final): each dataset's nominal sensor configuration s*.

Every beam table here is read out EXACTLY -- no clustering, no fitting of the
table itself -- because all three formats turn out to carry the beam identity:

  SemanticKITTI  points are stored ring by ring, azimuth increasing inside a
                 ring, so a ring boundary is an azimuth wrap
                 (corr(index, elevation) = -0.99).
  nuScenes       5th channel is the ring index.
  Waymo          6th channel is the flattened range-image index; row = idx//2650
                 selects the TOP unit's 64 beams and separates it from the four
                 short-range side units that share the file.

Two things still have to be fitted, and both are reported with their evidence:
the Waymo sensor origin (its clouds are in the vehicle frame) and the mounting
height above ground (no format records it).

Because SemanticKITTI and Waymo clouds are ego-motion compensated, a beam's
apparent elevation drifts around the sweep; tables are therefore taken as the
median over azimuth sectors and the drift is reported separately.
"""
import glob
import json
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from A3_run_f1f4 import GROUND_IDS, discover_nusc_frames, load_nusc_frames

N_FRAMES = 10
RHO_MIN = 3.0
NSEC = 36
CURRICULUM = {'beams': [16, 32, 64, 128], 'height_m': [1.0, 2.0],
              'hres': [900, 3600], 'v_lo_deg': [-40, -5], 'v_hi_deg': [0, 25]}


def sector_table(el, az, ring, n_rings, nsec=NSEC):
    """Median elevation per (ring, azimuth sector), then median over sectors."""
    sec = ((az + 180.0) * nsec / 360.0).astype(int) % nsec
    T = np.full((n_rings, nsec), np.nan)
    for r in range(n_rings):
        mr = ring == r
        if mr.sum() < 50:
            continue
        e, s = el[mr], sec[mr]
        for q in range(nsec):
            mm = s == q
            if mm.sum() > 20:
                T[r, q] = np.median(e[mm])
    tbl = np.nanmedian(T, axis=1)
    drift = float(np.nanmedian(np.nanstd(T, axis=1)))
    return tbl, drift


def _modal(tabs):
    """Median across frames, keeping only frames that yielded the modal beam
    count (a sweep occasionally loses its topmost ring entirely)."""
    from collections import Counter
    n = Counter(len(x) for x in tabs).most_common(1)[0][0]
    keep = [x for x in tabs if len(x) == n]
    return np.nanmedian(np.asarray(keep), 0)


def summarise(tbl, drift, extra):
    t = np.sort(tbl[np.isfinite(tbl)])
    d = np.diff(t)
    out = {'n_beams': int(len(t)),
           'table_deg': [round(float(v), 4) for v in t],
           'fov_lo_deg': round(float(t[0]), 3), 'fov_hi_deg': round(float(t[-1]), 3),
           'spacing_med_deg': round(float(np.median(d)), 4),
           'spacing_min_deg': round(float(d.min()), 4),
           'spacing_max_deg': round(float(d.max()), 4),
           'nonuniformity': round(float(d.max() / d.min()), 2),
           'ego_motion_drift_deg': round(drift, 4)}
    out.update(extra)
    return out


def az_gap_to_hres(az, ring, n_rings):
    g = []
    for r in range(n_rings):
        a = np.sort(az[ring == r])
        if len(a) < 100:
            continue
        d = np.diff(a)
        d = d[(d > 1e-6) & (d < 5.0)]
        if len(d):
            g.append(float(np.median(d)))
    if not g:
        return None, None
    m = float(np.median(g))
    return int(round(360.0 / m)), round(m, 5)


def mount_height(xyzs, labs, ids, z_s):
    hs, hits = [], []
    for xyz, lab in zip(xyzs, labs):
        if lab is None:
            continue
        rho = np.hypot(xyz[:, 0], xyz[:, 1])
        gm = np.isin(lab, list(ids))
        hits.append(float(gm.mean()))
        s = gm & (rho < 20.0)
        if s.sum() >= 100:
            hs.append(z_s - float(np.median(xyz[s, 2])))
    if not hs:
        return None
    return {'height_m': round(float(np.mean(hs)), 3), 'std_m': round(float(np.std(hs)), 4),
            'ground_hit_rate': round(float(np.mean(hits)), 4), 'n_frames': len(hs)}


# --------------------------------------------------------------------------
def sk_rings(xyz, n_target=64, merge_deg=0.15):
    """Ring id from the storage order: azimuth wraps at each ring boundary.

    A ring occasionally shows a spurious extra wrap, so runs whose median
    elevations are nearly equal are merged back together (a true ring boundary
    is a step of at least ~0.33 deg on this sensor) until n_target remain.
    """
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    cut = np.flatnonzero(np.diff(az) < -180.0) + 1
    bounds = np.concatenate([[0], cut, [len(xyz)]])
    rho = np.hypot(xyz[:, 0], xyz[:, 1])
    el = np.degrees(np.arctan2(xyz[:, 2], np.maximum(rho, 1e-6)))
    runs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] > bounds[i]]
    med = [float(np.median(el[a:b])) for a, b in runs]
    n_raw = len(runs)
    # always land on exactly n_target rings so tables are comparable across
    # frames; the closest-elevation pair is always the spurious split
    while len(runs) > n_target:
        d = [abs(med[i + 1] - med[i]) for i in range(len(runs) - 1)]
        j = int(np.argmin(d))
        runs[j] = (runs[j][0], runs[j + 1][1])
        med[j] = float(np.median(el[runs[j][0]:runs[j][1]]))
        del runs[j + 1], med[j + 1]
    if len(runs) != n_target:
        raise ValueError(f'sk_rings: got {len(runs)} runs, need {n_target}')
    ring = np.zeros(len(xyz), np.int32)
    for k, (a, b) in enumerate(runs):
        ring[a:b] = k
    return ring, n_raw, len(runs)


def main():
    res = {'curriculum_TableB1': CURRICULUM}

    # ---------------- SemanticKITTI -------------------------------------
    ps = sorted(glob.glob('/data/Datasets/semantickitti/sequences/00/velodyne/*.bin'))[:N_FRAMES]
    xyzs = [np.fromfile(p, np.float32).reshape(-1, 4)[:, :3] for p in ps]
    labs = [(np.fromfile(p.replace('/velodyne/', '/labels/').replace('.bin', '.label'),
                         np.uint32) & 0xFFFF) for p in ps]
    tabs, drifts, hres_ = [], [], []
    n_raw = None
    for xyz in xyzs:
        ring, n_raw, n_ring = sk_rings(xyz)
        rho = np.hypot(xyz[:, 0], xyz[:, 1])
        m = rho > RHO_MIN
        el = np.degrees(np.arctan2(xyz[m, 2], rho[m]))
        az = np.degrees(np.arctan2(xyz[m, 1], xyz[m, 0]))
        t, d = sector_table(el, az, ring[m], n_ring)
        tabs.append(np.sort(t[np.isfinite(t)]))
        drifts.append(d)
        h, g = az_gap_to_hres(az, ring[m], n_ring)
        if h:
            hres_.append(h)
    res['semantickitti'] = summarise(
        _modal(tabs), float(np.median(drifts)),
        {'sensor_model': 'Velodyne HDL-64E',
         'frame': 'sensor frame', 'origin_xyz': [0.0, 0.0, 0.0],
         'origin_src': 'sensor-frame data, origin is the sensor',
         'beam_id_src': f'MEASURED EXACTLY (storage order is ring by ring; '
                        f'{n_raw} azimuth-wrap runs merged by elevation to 64)',
         'n_horiz': int(np.median(hres_)),
         'n_horiz_src': 'MEASURED (360 deg / median azimuth gap inside a ring)',
         'mount_height': mount_height(xyzs, labs, GROUND_IDS['semantickitti'], 0.0),
         'mount_height_src': 'MEASURED (sensor z minus median z of ground-class '
                             'points within rho<20 m)',
         'n_points_mean': int(np.mean([len(x) for x in xyzs]))})

    # ---------------- nuScenes ------------------------------------------
    ps = sorted(glob.glob('/data/Datasets/nuscenes/samples/LIDAR_TOP/*.pcd.bin'))[:N_FRAMES]
    arrs = [np.fromfile(p, np.float32).reshape(-1, 5) for p in ps]
    tabs, drifts, hres_ = [], [], []
    for a in arrs:
        rho = np.hypot(a[:, 0], a[:, 1])
        m = rho > RHO_MIN
        el = np.degrees(np.arctan2(a[m, 2], rho[m]))
        az = np.degrees(np.arctan2(a[m, 1], a[m, 0]))
        ring = a[m, 4].astype(np.int32)
        t, d = sector_table(el, az, ring, 32)
        tabs.append(np.sort(t[np.isfinite(t)]))
        drifts.append(d)
        h, g = az_gap_to_hres(az, ring, 32)
        if h:
            hres_.append(h)
    fr = discover_nusc_frames('/data/Datasets/nuscenes')[:N_FRAMES]
    nfr = load_nusc_frames('/data/Datasets/nuscenes', fr)
    res['nuscenes'] = summarise(
        _modal(tabs), float(np.median(drifts)),
        {'sensor_model': 'Velodyne HDL-32E',
         'frame': 'sensor frame', 'origin_xyz': [0.0, 0.0, 0.0],
         'origin_src': 'sensor-frame data, origin is the sensor',
         'beam_id_src': 'MEASURED EXACTLY (ring index, 5th channel)',
         'n_horiz': int(np.median(hres_)),
         'n_horiz_src': 'MEASURED (360 deg / median azimuth gap inside a ring)',
         'mount_height': mount_height([f['real_xyz'] for f in nfr],
                                      [f['real_lab'] for f in nfr],
                                      GROUND_IDS['nuscenes'], 0.0),
         'mount_height_src': 'MEASURED (sensor z minus median z of ground-class '
                             'points within rho<20 m)',
         'n_points_mean': int(np.mean([len(a) for a in arrs]))})

    # ---------------- Waymo ---------------------------------------------
    wj = json.load(open('review_analysis/_source_data/A3_waymo_beam_table.json'))
    O = wj['origin_xyz']
    NCOL, NROW = 2650, 64
    D = 'data/waymo/kitti_format/training/velodyne'
    stems = [s.strip().replace('.bin', '') for s in
             open('seg_annotated_indices.txt').read().split()]
    use = [s for s in stems if osp.exists(osp.join(D, s + '.bin'))][:N_FRAMES]
    xyzs, labs, tabs, drifts, hres_ = [], [], [], [], []
    for s_ in use:
        a = np.fromfile(osp.join(D, s_ + '.bin'), np.float32).reshape(-1, 6)
        idx = a[:, 5]
        sel = (idx >= 0) & (idx < NROW * NCOL)
        row = (idx[sel] // NCOL).astype(np.int32)
        p = a[sel, :3].astype(np.float64)
        rho = np.hypot(p[:, 0] - O[0], p[:, 1] - O[1])
        m = rho > RHO_MIN
        el = np.degrees(np.arctan2(p[m, 2] - O[2], rho[m]))
        az = np.degrees(np.arctan2(p[m, 1] - O[1], p[m, 0] - O[0]))
        t, d = sector_table(el, az, row[m], NROW)
        tabs.append(np.sort(t[np.isfinite(t)]))
        drifts.append(d)
        h, g = az_gap_to_hres(az, row[m], NROW)
        if h:
            hres_.append(h)
        xyzs.append(a[:, :3])
        labs.append(np.fromfile(osp.join(D, s_ + '.label'), np.int32).reshape(-1, 2)[:, 1])
    res['waymo'] = summarise(
        _modal(tabs), float(np.median(drifts)),
        {'sensor_model': 'Waymo TOP LiDAR (file also holds 4 short-range side units)',
         'frame': 'vehicle frame, z=0 at ground', 'origin_xyz': O,
         'origin_src': f'MEASURED (fit; per-(row,sector) elevation spread '
                       f'{wj["fit_cost_deg"]:.4f} deg at the optimum vs '
                       f'{wj["cost_at_axis_origin"]:.4f} deg on the vertical axis)',
         'beam_id_src': 'MEASURED EXACTLY (range-image index, 6th channel; '
                        'row = idx // 2650 keeps the TOP unit only)',
         'n_horiz': int(np.median(hres_)),
         'n_horiz_src': 'MEASURED (360 deg / median azimuth gap inside a row)',
         'mount_height': mount_height(xyzs, labs, GROUND_IDS['waymo'], O[2]),
         'mount_height_src': 'MEASURED (fitted sensor z minus median z of '
                             'ground-class points within rho<20 m)',
         'n_points_mean': int(np.mean([len(x) for x in xyzs]))})

    json.dump(res, open('review_analysis/_source_data/A3_nominal_configs.json', 'w'), indent=2)

    for ds in ['semantickitti', 'nuscenes', 'waymo']:
        v = res[ds]
        mh = v['mount_height']
        print(f'\n=== {ds}  [{v["sensor_model"]}] ===')
        print(f'  origin        : {v["origin_xyz"]}  ({v["origin_src"][:55]})')
        print(f'  beams         : {v["n_beams"]}   ({v["beam_id_src"][:60]})')
        print(f'  FOV           : [{v["fov_lo_deg"]}, {v["fov_hi_deg"]}] deg')
        print(f'  spacing       : med={v["spacing_med_deg"]} min={v["spacing_min_deg"]} '
              f'max={v["spacing_max_deg"]}  NONUNIFORMITY={v["nonuniformity"]}x')
        print(f'  ego-motion    : {v["ego_motion_drift_deg"]} deg drift over a sweep')
        print(f'  n_horiz       : {v["n_horiz"]}')
        print(f'  mount height  : {mh["height_m"]} m (std {mh["std_m"]}, '
              f'ground hit {mh["ground_hit_rate"]})')
        print(f'  pts/frame     : {v["n_points_mean"]}')
        c = CURRICULUM
        inside = (v['n_beams'] in c['beams'],
                  c['height_m'][0] <= mh['height_m'] <= c['height_m'][1],
                  c['hres'][0] <= v['n_horiz'] <= c['hres'][1],
                  c['v_lo_deg'][0] <= v['fov_lo_deg'] <= c['v_lo_deg'][1],
                  c['v_hi_deg'][0] <= v['fov_hi_deg'] <= c['v_hi_deg'][1])
        print(f'  inside curriculum range?  beams={inside[0]} height={inside[1]} '
              f'hres={inside[2]} fov_lo={inside[3]} fov_hi={inside[4]}')


if __name__ == '__main__':
    main()
