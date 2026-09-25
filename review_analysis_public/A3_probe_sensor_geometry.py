#!/usr/bin/env python3
"""A3 Step 2 probe: recover each dataset's real sensor geometry from the data.

Answers five questions, all with falsifiable measurements:
  Q1  How badly is beam structure smeared, and is the smearing azimuth-driven
      (the signature of ego-motion compensation)?
  Q2  Can the beam elevation table be recovered per azimuth sector?
  Q3  Where is the Waymo sensor origin (its clouds are in the vehicle frame)?
  Q4  nuScenes' exact elevation table, from the ring index channel.
  Q5  Mounting height above ground, from ground-class points.

Writes review_analysis/_source_data/A3_sensor_probe.json.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from A3_run_f1f4 import GROUND_IDS

RHO_MIN = 3.0
N_FRAMES = 10
EL_LO, EL_HI, EL_NB = -32.0, 18.0, 10000     # 0.005 deg bins


def elev(xyz, z_s=0.0):
    rho = np.hypot(xyz[:, 0], xyz[:, 1])
    m = rho > RHO_MIN
    return np.degrees(np.arctan2(xyz[m, 2] - z_s, rho[m])), rho[m], xyz[m]


def occupancy(el):
    """How many fine elevation bins are occupied. Crisp rings -> few bins."""
    el = el[(el > EL_LO) & (el < EL_HI)]
    h, _ = np.histogram(el, bins=EL_NB, range=(EL_LO, EL_HI))
    return int((h > 0).sum()), len(el)


def sector_scan(xyz, z_s=0.0, n_sectors=36):
    """Occupied-bin count inside each azimuth sector vs across all azimuths.

    If ego-motion compensation is smearing the rings, per-sector occupancy is
    far lower than global occupancy: within a sector every point shares almost
    the same timestamp, so the correction is almost the same for all of them.
    """
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    el_all, _, _ = elev(xyz, z_s)
    g_occ, g_n = occupancy(el_all)
    per = []
    w = 360.0 / n_sectors
    for s in range(n_sectors):
        lo = -180 + s * w
        m = (az >= lo) & (az < lo + w)
        if m.sum() < 300:
            continue
        e, _, _ = elev(xyz[m], z_s)
        o, n = occupancy(e)
        if n > 300:
            per.append(o)
    return {'global_occupied': g_occ, 'global_n': g_n,
            'sector_occupied_median': int(np.median(per)) if per else None,
            'n_sectors_used': len(per),
            'smear_factor': round(g_occ / max(np.median(per), 1), 2) if per else None}


def drift_vs_azimuth(xyz, z_s=0.0, n_sectors=36, k=64):
    """Track one beam's apparent elevation around the sweep.

    Take the strongest elevation mode inside each sector and follow it; a real
    (uncompensated) sensor keeps it constant, ego-motion correction makes it
    drift.
    """
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    w = 360.0 / n_sectors
    peaks = []
    for s in range(n_sectors):
        lo = -180 + s * w
        m = (az >= lo) & (az < lo + w)
        if m.sum() < 300:
            continue
        e, _, _ = elev(xyz[m], z_s)
        e = e[(e > -20) & (e < 0)]        # a densely populated band
        if len(e) < 200:
            continue
        h, ed = np.histogram(e, bins=2000, range=(-20, 0))
        peaks.append(float(0.5 * (ed[1:] + ed[:-1])[np.argmax(h)]))
    if len(peaks) < 5:
        return None
    return {'n': len(peaks), 'std_deg': round(float(np.std(peaks)), 4),
            'ptp_deg': round(float(np.ptp(peaks)), 4)}


def fit_waymo_z(xyz, grid=np.arange(0.0, 3.51, 0.02)):
    """Sensor height that makes the per-sector ring structure sharpest.

    Elevation is only ring-discrete when measured from the true origin, so the
    correct height minimises per-sector occupied-bin count.
    """
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    sectors = []
    for s in range(36):
        lo = -180 + s * 10.0
        m = (az >= lo) & (az < lo + 10.0)
        if m.sum() >= 800:
            sectors.append(xyz[m])
        if len(sectors) >= 8:
            break
    if not sectors:
        return None
    curve = []
    for z in grid:
        occ = [occupancy(elev(sub, z)[0])[0] for sub in sectors]
        curve.append(float(np.mean(occ)))
    curve = np.asarray(curve)
    return {'z_best': round(float(grid[int(np.argmin(curve))]), 3),
            'occ_at_best': round(float(curve.min()), 1),
            'occ_at_z0': round(float(curve[0]), 1),
            'curve': [[round(float(z), 2), round(float(c), 1)]
                      for z, c in zip(grid[::10], curve[::10])]}


def ground_height(frames, ground_ids, z_s=0.0):
    hs, hits = [], []
    for f in frames:
        if f.get('lab') is None:
            continue
        xyz = f['xyz']
        rho = np.hypot(xyz[:, 0], xyz[:, 1])
        gm = np.isin(f['lab'], list(ground_ids))
        hits.append(float(gm.mean()))
        m = gm & (rho < 20.0)
        if m.sum() >= 100:
            hs.append(z_s - float(np.median(xyz[m, 2])))
    if not hs:
        return None
    return {'height_m': round(float(np.mean(hs)), 3),
            'std': round(float(np.std(hs)), 4),
            'ground_hit_rate': round(float(np.mean(hits)), 4),
            'n_frames': len(hs)}


# --------------------------------------------------------------------------
def load_sk(n=N_FRAMES):
    out = []
    for p in sorted(glob.glob('/data/Datasets/semantickitti/sequences/00/velodyne/*.bin'))[:n]:
        lp = p.replace('/velodyne/', '/labels/').replace('.bin', '.label')
        out.append({'xyz': np.fromfile(p, np.float32).reshape(-1, 4)[:, :3],
                    'lab': (np.fromfile(lp, np.uint32) & 0xFFFF) if os.path.exists(lp) else None})
    return out


def load_nu(n=N_FRAMES):
    out = []
    for p in sorted(glob.glob('/data/Datasets/nuscenes/samples/LIDAR_TOP/*.pcd.bin'))[:n]:
        a = np.fromfile(p, np.float32).reshape(-1, 5)
        out.append({'xyz': a[:, :3], 'ring': a[:, 4].astype(np.int32), 'lab': None})
    return out


def load_wa(n=N_FRAMES):
    """Only frames that actually carry a .label file (seg_annotated subset)."""
    stems = [s.strip().replace('.bin', '') for s in
             open('seg_annotated_indices.txt').read().split()][:n]
    out = []
    d = 'data/waymo/kitti_format/training/velodyne'
    for s in stems:
        bp, lp = os.path.join(d, s + '.bin'), os.path.join(d, s + '.label')
        if not (os.path.exists(bp) and os.path.exists(lp)):
            continue
        out.append({'xyz': np.fromfile(bp, np.float32).reshape(-1, 6)[:, :3],
                    'lab': np.fromfile(lp, np.int32).reshape(-1, 2)[:, 1]})
    return out


if __name__ == '__main__':
    res = {}

    # ---- SemanticKITTI ---------------------------------------------------
    sk = load_sk()
    res['semantickitti'] = {
        'n_points_mean': int(np.mean([len(f['xyz']) for f in sk])),
        'Q1_sector_scan': sector_scan(sk[0]['xyz']),
        'Q2_drift_vs_azimuth': drift_vs_azimuth(sk[0]['xyz']),
        'Q5_ground_height': ground_height(sk, GROUND_IDS['semantickitti'], 0.0),
    }

    # ---- nuScenes --------------------------------------------------------
    nu = load_nu()
    a = nu[0]
    rho = np.hypot(a['xyz'][:, 0], a['xyz'][:, 1])
    m = rho > RHO_MIN
    el = np.degrees(np.arctan2(a['xyz'][m, 2], rho[m]))
    rg = a['ring'][m]
    tbl = []
    for r in range(int(rg.max()) + 1):
        s = rg == r
        if s.sum() > 100:
            tbl.append([int(r), round(float(np.median(el[s])), 3),
                        round(float(np.std(el[s])), 4)])
    res['nuscenes'] = {
        'n_points_mean': int(np.mean([len(f['xyz']) for f in nu])),
        'Q1_sector_scan': sector_scan(a['xyz']),
        'Q2_drift_vs_azimuth': drift_vs_azimuth(a['xyz']),
        'Q4_elev_table_from_ring': sorted(tbl, key=lambda t: t[1]),
        'Q4_mean_within_ring_std_deg': round(float(np.mean([t[2] for t in tbl])), 4),
    }

    # ---- Waymo -----------------------------------------------------------
    wa = load_wa()
    if wa:
        res['waymo'] = {
            'n_frames_with_labels': len(wa),
            'n_points_mean': int(np.mean([len(f['xyz']) for f in wa])),
            'Q3_fit_sensor_z': fit_waymo_z(wa[0]['xyz']),
        }
        zb = res['waymo']['Q3_fit_sensor_z']['z_best'] if res['waymo']['Q3_fit_sensor_z'] else 0.0
        res['waymo']['Q1_sector_scan_at_zbest'] = sector_scan(wa[0]['xyz'], zb)
        res['waymo']['Q1_sector_scan_at_z0'] = sector_scan(wa[0]['xyz'], 0.0)
        res['waymo']['Q2_drift_vs_azimuth'] = drift_vs_azimuth(wa[0]['xyz'], zb)
        res['waymo']['Q5_ground_height'] = ground_height(wa, GROUND_IDS['waymo'], zb)
    else:
        res['waymo'] = {'error': 'no frames with .label found'}

    os.makedirs('review_analysis/_source_data', exist_ok=True)
    json.dump(res, open('review_analysis/_source_data/A3_sensor_probe.json', 'w'), indent=2)
    print(json.dumps(res, indent=2)[:6000])
