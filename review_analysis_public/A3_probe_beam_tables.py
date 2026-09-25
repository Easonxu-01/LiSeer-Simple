#!/usr/bin/env python3
"""A3 Step 2b: recover real beam elevation tables; quantify motion-compensation
smear with the sample-size confound controlled.

Key controls vs the first probe:
  * occupancy is always measured on a FIXED number of points, so a sector and
    the whole sweep are compared at equal sample size;
  * the Waymo sensor-height fit excludes the near-zero elevation band, which
    otherwise collapses every ground return into one bin and produces a
    degenerate optimum at z_s = 0.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from A3_run_f1f4 import GROUND_IDS

RHO_MIN = 3.0
EL_LO, EL_HI, EL_NB = -32.0, 18.0, 10000       # 0.005 deg bins
N_FIX = 3000                                   # fixed sample size for occupancy
SEED = 0


def elev(xyz, z_s=0.0):
    rho = np.hypot(xyz[:, 0], xyz[:, 1])
    m = rho > RHO_MIN
    return np.degrees(np.arctan2(xyz[m, 2] - z_s, rho[m]))


def occ_fixed(el, n=N_FIX, rng=None):
    """Occupied fine bins after subsampling to exactly n points."""
    el = el[(el > EL_LO) & (el < EL_HI)]
    if len(el) < n:
        return None
    rng = rng or np.random.RandomState(SEED)
    el = el[rng.choice(len(el), n, replace=False)]
    h, _ = np.histogram(el, bins=EL_NB, range=(EL_LO, EL_HI))
    return int((h > 0).sum())


def smear_controlled(xyz, z_s=0.0, n_sectors=36):
    """Global vs per-sector occupancy at EQUAL sample size."""
    rng = np.random.RandomState(SEED)
    g = occ_fixed(elev(xyz, z_s), N_FIX, rng)
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    per, w = [], 360.0 / n_sectors
    for s in range(n_sectors):
        lo = -180 + s * w
        m = (az >= lo) & (az < lo + w)
        o = occ_fixed(elev(xyz[m], z_s), N_FIX, rng) if m.sum() > N_FIX * 2 else None
        if o:
            per.append(o)
    if not per or not g:
        return None
    med = float(np.median(per))
    return {'global_occ_at_n%d' % N_FIX: g, 'sector_occ_median': med,
            'smear_ratio': round(g / max(med, 1), 2), 'n_sectors': len(per)}


def kmeans_1d(x, k, iters=80):
    x = np.sort(np.asarray(x, dtype=np.float64))
    c = np.quantile(x, np.linspace(0, 1, k + 2)[1:-1])
    for _ in range(iters):
        e = 0.5 * (c[1:] + c[:-1])
        a = np.searchsorted(e, x)
        s = np.bincount(a, weights=x, minlength=k)
        n = np.bincount(a, minlength=k)
        new = np.sort(np.where(n > 0, s / np.maximum(n, 1), c))
        if np.allclose(new, c, atol=1e-10):
            c = new
            break
        c = new
    e = 0.5 * (c[1:] + c[:-1])
    a = np.searchsorted(e, x)
    n = np.bincount(a, minlength=k)
    w = np.array([x[a == j].std() if n[j] > 1 else 0.0 for j in range(k)])
    return c, n, w


def recover_table_per_sector(xyz, k, z_s=0.0, n_sectors=36, min_pts=1500):
    """Cluster elevation inside each azimuth sector, then align sectors.

    Under ego-motion compensation the whole table shifts per sector, so the
    SHAPE (spacings) is recoverable even when the absolute angles drift. We
    report the median table after removing each sector's own median offset.
    """
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    w = 360.0 / n_sectors
    tables, tight = [], []
    for s in range(n_sectors):
        lo = -180 + s * w
        m = (az >= lo) & (az < lo + w)
        if m.sum() < min_pts:
            continue
        el = elev(xyz[m], z_s)
        el = el[(el > EL_LO) & (el < EL_HI)]
        if len(el) < min_pts:
            continue
        c, n, wd = kmeans_1d(el, k)
        live = n > 0
        if live.sum() < k * 0.8:
            continue
        d = np.diff(c[live])
        tables.append(c)
        tight.append(float(np.mean(wd[live]) / max(np.median(d), 1e-9)))
    if len(tables) < 5:
        return None
    T = np.array(tables)
    centred = T - np.median(T, axis=1, keepdims=True)
    tbl = np.median(centred, axis=0)
    d = np.diff(tbl)
    return {'k': k, 'n_sectors_used': len(tables),
            'tightness': round(float(np.median(tight)), 4),
            'table_centred_deg': [round(float(v), 3) for v in tbl],
            'spacing_med': round(float(np.median(d)), 4),
            'spacing_min': round(float(d.min()), 4),
            'spacing_max': round(float(d.max()), 4),
            'nonuniformity': round(float(d.max() / max(d.min(), 1e-9)), 2)}


def fit_sensor_z_excl_ground(xyz, grid, excl=(-4.0, 4.0)):
    """Sensor height maximising per-sector ring sharpness, ground band excluded.

    Points near zero elevation are dominated by the ground plane, which is
    razor-sharp at z_s = 0 for a body-frame cloud and creates a spurious
    optimum there; excluding that band removes the degeneracy.
    """
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    sub = []
    for s in range(36):
        lo = -180 + s * 10.0
        m = (az >= lo) & (az < lo + 10.0)
        if m.sum() >= 4000:
            sub.append(xyz[m])
        if len(sub) >= 8:
            break
    if not sub:
        return None
    rng = np.random.RandomState(SEED)
    curve = []
    for z in grid:
        os_ = []
        for x in sub:
            el = elev(x, z)
            el = el[(el < excl[0]) | (el > excl[1])]
            o = occ_fixed(el, 2000, rng)
            if o:
                os_.append(o)
        curve.append(float(np.mean(os_)) if os_ else np.nan)
    curve = np.asarray(curve)
    ok = np.isfinite(curve)
    return {'z_best': round(float(grid[ok][int(np.argmin(curve[ok]))]), 3),
            'occ_min': round(float(np.nanmin(curve)), 1),
            'occ_max': round(float(np.nanmax(curve)), 1),
            'curve': [[round(float(z), 2), (round(float(c), 1) if np.isfinite(c) else None)]
                      for z, c in zip(grid, curve)]}


# --------------------------------------------------------------------------
def main():
    out = {}
    rs = np.random.RandomState(SEED)

    # ---- SemanticKITTI: HDL-64E, 64 beams ------------------------------
    p = sorted(glob.glob('/data/Datasets/semantickitti/sequences/00/velodyne/*.bin'))[0]
    sk = np.fromfile(p, np.float32).reshape(-1, 4)[:, :3]
    out['semantickitti'] = {
        'smear_controlled': smear_controlled(sk),
        'table_k64': recover_table_per_sector(sk, 64),
    }

    # ---- nuScenes: ring channel is ground truth -------------------------
    p = sorted(glob.glob('/data/Datasets/nuscenes/samples/LIDAR_TOP/*.pcd.bin'))[0]
    a = np.fromfile(p, np.float32).reshape(-1, 5)
    out['nuscenes'] = {
        'smear_controlled': smear_controlled(a[:, :3]),
        'table_k32': recover_table_per_sector(a[:, :3], 32),
    }
    # nuScenes ground height needs lidarseg
    import os.path as osp
    seg_dir = '/data/Datasets/nuscenes/lidarseg/v1.0-trainval'
    hs, hits = [], []
    try:
        import pickle
        # match bin -> token via the lidarseg filename embedded in the sample
        # data path is not available here, so use the A3 discovery helper
        from A3_run_f1f4 import discover_nusc_frames, load_nusc_frames
        fr = discover_nusc_frames('/data/Datasets/nuscenes')[:10]
        frames = load_nusc_frames('/data/Datasets/nuscenes', fr)
        for f in frames:
            xyz, lab = f['real_xyz'], f['real_lab']
            rho = np.hypot(xyz[:, 0], xyz[:, 1])
            gm = np.isin(lab, list(GROUND_IDS['nuscenes']))
            hits.append(float(gm.mean()))
            m = gm & (rho < 20.0)
            if m.sum() >= 100:
                hs.append(-float(np.median(xyz[m, 2])))
        out['nuscenes']['ground_height'] = {
            'height_m': round(float(np.mean(hs)), 3), 'std': round(float(np.std(hs)), 4),
            'ground_hit_rate': round(float(np.mean(hits)), 4), 'n': len(hs)}
    except Exception as e:
        out['nuscenes']['ground_height'] = {'error': f'{type(e).__name__}: {e}'}

    # ---- Waymo: body frame, sensor height unknown -----------------------
    stems = [s.strip().replace('.bin', '') for s in
             open('seg_annotated_indices.txt').read().split()][:10]
    d = 'data/waymo/kitti_format/training/velodyne'
    wf = [s for s in stems if osp.exists(osp.join(d, s + '.bin'))]
    wa = np.fromfile(osp.join(d, wf[0] + '.bin'), np.float32).reshape(-1, 6)[:, :3]
    fit = fit_sensor_z_excl_ground(wa, np.arange(0.0, 3.51, 0.05))
    out['waymo'] = {'fit_sensor_z': fit}
    zb = fit['z_best'] if fit else 0.0
    out['waymo']['smear_controlled_at_zbest'] = smear_controlled(wa, zb)
    out['waymo']['table_k64_at_zbest'] = recover_table_per_sector(wa, 64, zb)
    hs, hits = [], []
    for s in wf:
        xyz = np.fromfile(osp.join(d, s + '.bin'), np.float32).reshape(-1, 6)[:, :3]
        lab = np.fromfile(osp.join(d, s + '.label'), np.int32).reshape(-1, 2)[:, 1]
        rho = np.hypot(xyz[:, 0], xyz[:, 1])
        gm = np.isin(lab, list(GROUND_IDS['waymo']))
        hits.append(float(gm.mean()))
        m = gm & (rho < 20.0)
        if m.sum() >= 100:
            hs.append(zb - float(np.median(xyz[m, 2])))
    out['waymo']['ground_height_above_ground'] = {
        'height_m': round(float(np.mean(hs)), 3), 'std': round(float(np.std(hs)), 4),
        'ground_hit_rate': round(float(np.mean(hits)), 4), 'n': len(hs),
        'note': f'sensor z taken as fitted z_best={zb}'}

    json.dump(out, open('review_analysis/_source_data/A3_beam_tables.json', 'w'), indent=2)
    for ds, v in out.items():
        print(f'\n=== {ds} ===')
        for k, vv in v.items():
            if isinstance(vv, dict) and 'table_centred_deg' in vv:
                print(f'  {k}: k={vv["k"]} sectors={vv["n_sectors_used"]} tightness={vv["tightness"]} '
                      f'spacing med/min/max={vv["spacing_med"]}/{vv["spacing_min"]}/{vv["spacing_max"]} '
                      f'nonuniformity={vv["nonuniformity"]}')
                print(f'    table: {vv["table_centred_deg"]}')
            elif isinstance(vv, dict) and 'curve' in vv:
                print(f'  {k}: z_best={vv["z_best"]} occ_min={vv["occ_min"]} occ_max={vv["occ_max"]}')
                print(f'    curve: {[c for c in vv["curve"][::4]]}')
            else:
                print(f'  {k}: {vv}')


if __name__ == '__main__':
    main()
