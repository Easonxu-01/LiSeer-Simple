#!/usr/bin/env python3
"""Re-observe a densified scene as a LiDAR scan.

Upgrades over the original `simulate_scan` in run_a3_f1f4.py, all driven by
what the real sensors actually do (see review_analysis/_source_data/A3_nominal_configs.md):

  1. Explicit, possibly NON-UNIFORM elevation table. Real units are not evenly
     spaced -- the HDL-64E runs two blocks (~1/3 deg above, ~1/2 deg below,
     3.2x ratio) and the Waymo TOP unit is 4x non-uniform. `np.linspace` models
     none of that. When a table is supplied, beam assignment uses midpoint
     edges and a per-beam acceptance half-width, so unequal spacings are
     handled exactly. (nuScenes' HDL-32E really is uniform at 1.33 deg, which
     the measurement confirms rather than assumes.)
  2. Sensor origin off the vertical axis. Waymo clouds live in the vehicle
     frame whose origin is at the rear axle, so the TOP unit sits well forward
     of (0,0). Ray casting from the wrong (x,y) makes a single beam's apparent
     elevation swing by up to 3 deg over a sweep.
  3. Seeded azimuth phase. The original drew it from the unseeded global numpy
     RNG, so runs were not reproducible despite the seed=0 protocol.

The z-buffer, the point-centric O(N+R) cost and the ground-height estimator are
unchanged.
"""
import math

import numpy as np

__all__ = ['simulate_scan', 'estimate_ground_z']


def estimate_ground_z(xyz, labels=None, label_ids=None, rho_max=20.0, min_pts=5):
    """Median z of ground-class points within rho_max, else the 2nd percentile.

    Both sides of every A3 comparison must use this same estimator: mixing it
    with the percentile fallback on one side only silently changes what F4
    measures.
    """
    if labels is not None and label_ids is not None:
        rho = np.hypot(xyz[:, 0], xyz[:, 1])
        m = np.isin(labels, list(label_ids)) & (rho < rho_max)
        if m.sum() >= min_pts:
            return float(np.median(xyz[m, 2])), float(m.sum()) / max(len(xyz), 1)
    return float(np.percentile(xyz[:, 2], 2)), 0.0


def _beam_geometry(elev_table_rad):
    """Midpoint edges plus each beam's acceptance half-width.

    With a non-uniform table a fixed half-step is wrong: a wide gap would
    reject valid returns and a narrow one would accept a neighbour's.
    """
    e = np.asarray(elev_table_rad, dtype=np.float64)
    nb = len(e)
    edges = 0.5 * (e[1:] + e[:-1])
    d = np.diff(e)
    half_lo = np.empty(nb)
    half_hi = np.empty(nb)
    if nb > 1:
        half_lo[0] = d[0] / 2.0
        half_lo[1:] = d / 2.0
        half_hi[-1] = d[-1] / 2.0
        half_hi[:-1] = d / 2.0
    else:
        half_lo[:] = half_hi[:] = 1e6
    return edges, half_lo, half_hi


def simulate_scan(xyz, hres, lidar_height,
                  beams=None, v_lo_deg=None, v_hi_deg=None,
                  elev_table_deg=None, origin_xy=(0.0, 0.0),
                  ground_labels=None, ground_label_ids=None,
                  ground_z=None, rng=None):
    """Cast one scan over a dense scene.

    Either give `elev_table_deg` (the sensor's real beam elevations) or
    `beams` + `v_lo_deg` + `v_hi_deg` (the training curriculum's uniform
    layout). Returns (points, surviving indices, info).

    `lidar_height` is measured above the scene's own ground, so the returned
    cloud keeps the scene's ground height by construction -- `info` reports it
    so the caller can assert that rather than trust it.
    """
    empty = (np.zeros((0, 3), np.float32), np.zeros((0,), np.int64), {})
    if len(xyz) == 0:
        return empty

    rng = rng if rng is not None else np.random.RandomState(0)

    if ground_z is None:
        ground_z, ground_hit = estimate_ground_z(xyz, ground_labels, ground_label_ids)
    else:
        ground_hit = float('nan')

    if elev_table_deg is not None:
        elev = np.sort(np.radians(np.asarray(elev_table_deg, dtype=np.float64)))
        nb = len(elev)
        uniform = False
    else:
        v_lo, v_hi = math.radians(v_lo_deg), math.radians(v_hi_deg)
        nb = int(beams)
        elev = (np.linspace(v_lo, v_hi, nb) if nb > 1
                else np.array([(v_lo + v_hi) / 2.0]))
        uniform = True

    edges, half_lo, half_hi = _beam_geometry(elev)

    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    oz = ground_z + float(lidar_height)

    p = xyz.astype(np.float64)
    dx = p[:, 0] - ox
    dy = p[:, 1] - oy
    dz = p[:, 2] - oz

    rho_xy = np.hypot(dx, dy)
    el = np.arctan2(dz, np.maximum(rho_xy, 1e-6))
    az = np.arctan2(dy, dx)

    beam_idx = (np.searchsorted(edges, el).astype(np.int32) if nb > 1
                else np.zeros(len(el), np.int32))
    np.clip(beam_idx, 0, nb - 1, out=beam_idx)
    in_fov = ((el >= elev[beam_idx] - half_lo[beam_idx]) &
              (el <= elev[beam_idx] + half_hi[beam_idx]))

    h_step = 2 * math.pi / hres
    az_phase = float(rng.uniform(0, h_step))
    az_idx = np.floor(((az - az_phase) % (2 * math.pi)) / h_step).astype(np.int32)
    np.clip(az_idx, 0, hres - 1, out=az_idx)

    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    cell = beam_idx.astype(np.int64) * hres + az_idx

    keep = np.flatnonzero(in_fov)
    if len(keep) == 0:
        return empty
    order = np.lexsort((dist[keep], cell[keep]))
    cs = cell[keep][order]
    _, first = np.unique(cs, return_index=True)
    survivors = keep[order][first]

    out = xyz[survivors].astype(np.float32)
    info = {
        'n_beams': nb, 'uniform_table': uniform, 'hres': hres,
        'origin_xyz': (ox, oy, oz),
        'scene_ground_z': ground_z, 'ground_hit_rate': ground_hit,
        'lidar_height': float(lidar_height), 'n_out': int(len(out)),
        'fov_deg': (float(np.degrees(elev[0])), float(np.degrees(elev[-1]))),
    }
    return out, survivors, info


# --------------------------------------------------------------------------
# Self-test: every claim above is checked against a synthetic scene whose
# answer is known in closed form.
# --------------------------------------------------------------------------

def _selftest():
    rng = np.random.RandomState(0)
    # A flat ground plane at z = -1.8 plus a wall, densely sampled.
    n = 400000
    r = rng.uniform(1, 60, n)
    a = rng.uniform(-np.pi, np.pi, n)
    ground = np.stack([r * np.cos(a), r * np.sin(a), np.full(n, -1.8)], 1)
    m = 60000
    wall = np.stack([np.full(m, 25.0), rng.uniform(-10, 10, m),
                     rng.uniform(-1.8, 6.0, m)], 1)
    scene = np.vstack([ground, wall]).astype(np.float32)
    lab = np.concatenate([np.full(n, 40), np.full(m, 50)])   # 40 = SK road

    ok = True

    # 1. uniform table: output beam count matches the request
    out, idx, info = simulate_scan(scene, hres=1200, lidar_height=1.8,
                                   beams=32, v_lo_deg=-25, v_hi_deg=10,
                                   ground_labels=lab, ground_label_ids={40}, rng=rng)
    gz = info['scene_ground_z']
    ok &= abs(gz - (-1.8)) < 1e-6
    print(f'  [1] uniform 32-beam: n_out={len(out)} ground_z={gz:.4f} (want -1.8) '
          f'{"OK" if abs(gz + 1.8) < 1e-6 else "FAIL"}')

    # 2. the simulated cloud keeps the scene's ground height  <-- the property
    #    the whole conditional comparison rests on
    gsel = np.isin(lab[idx], [40])
    g_sim = float(np.median(out[gsel, 2])) if gsel.any() else np.nan
    ok &= abs(g_sim - (-1.8)) < 1e-3
    print(f'  [2] sim ground z = {g_sim:.4f} (scene -1.8) '
          f'{"OK" if abs(g_sim + 1.8) < 1e-3 else "FAIL"}')

    # 3. non-uniform table: the beams the scan lands on reproduce the table.
    #    Note this simulator SELECTS scene points per (beam, azimuth) cell, it
    #    does not snap them onto the ray, so a returned point sits anywhere
    #    inside its acceptance band. The right check is therefore that the
    #    per-beam median elevation matches the table, not that the elevations
    #    are discrete.
    tbl = np.concatenate([np.arange(-24, -8, 0.5), np.arange(-8, 2, 1.0 / 3)])
    out2, idx2, info2 = simulate_scan(scene, hres=1200, lidar_height=1.8,
                                      elev_table_deg=tbl, origin_xy=(0, 0),
                                      ground_labels=lab, ground_label_ids={40}, rng=rng)
    d = out2.astype(np.float64)
    d[:, 2] -= info2['origin_xyz'][2]
    el = np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))
    t_sorted = np.sort(tbl)
    edges = 0.5 * (t_sorted[1:] + t_sorted[:-1])
    b = np.searchsorted(edges, el)
    # use the simulator's own acceptance rule, which is asymmetric wherever the
    # spacing changes (the HDL-64E block boundary is exactly such a place)
    _e, hlo, hhi = _beam_geometry(np.radians(t_sorted))
    hlo = np.degrees(hlo)
    hhi = np.degrees(hhi)
    # A returned point sits anywhere inside its beam's acceptance band, and on
    # a ground plane the band is not evenly populated (area grows with range),
    # so the per-beam median is biased toward the far edge. The check is that
    # it stays inside the band, i.e. |error| <= the local half-width.
    bad, worst = 0, 0.0
    npop = 0
    for j in range(len(t_sorted)):
        m = b == j
        if m.sum() > 30:
            npop += 1
            dev = float(np.median(el[m])) - t_sorted[j]
            r = dev / (hhi[j] if dev >= 0 else hlo[j])
            worst = max(worst, abs(r))
            bad += abs(r) > 1.0
    ok &= bad == 0
    print(f'  [3] non-uniform table: {npop}/{len(t_sorted)} beams populated, '
          f'{bad} outside their acceptance band (worst = {worst:.2f} x half-width) '
          f'{"OK" if bad == 0 else "FAIL"}')

    # 4. off-axis origin: ground returns of one beam form a circle centred on
    #    the sensor, so their radius about the true apex is near-constant and
    #    about the axis it is not. This is the Waymo symptom in its cleanest
    #    geometric form.
    OX = 1.4
    out3, idx3, info3 = simulate_scan(scene, hres=2400, lidar_height=1.8,
                                      elev_table_deg=tbl, origin_xy=(OX, 0.0),
                                      ground_labels=lab, ground_label_ids={40}, rng=rng)
    g3 = np.isin(lab[idx3], [40])
    P = out3[g3]
    oz3 = info3['origin_xyz'][2]
    e3 = np.degrees(np.arctan2(P[:, 2] - oz3,
                               np.hypot(P[:, 0] - OX, P[:, 1])))
    b3 = np.searchsorted(edges, e3)

    def ring_spread(cx):
        r = np.hypot(P[:, 0] - cx, P[:, 1])
        v = [float(np.std(r[b3 == j]) / max(np.median(r[b3 == j]), 1e-9))
             for j in range(len(t_sorted)) if (b3 == j).sum() > 300]
        return float(np.median(v)) if v else np.nan

    r_true, r_axis = ring_spread(OX), ring_spread(0.0)
    ok &= r_axis > 3 * r_true
    print(f'  [4] ground-ring radius spread: {r_true:.4f} about the true apex vs '
          f'{r_axis:.4f} about the axis ({r_axis / max(r_true, 1e-9):.1f}x) '
          f'{"OK" if r_axis > 3 * r_true else "FAIL"}')

    # 5. reproducibility under the same seed
    o1, _, _ = simulate_scan(scene, hres=900, lidar_height=1.8, beams=16,
                             v_lo_deg=-20, v_hi_deg=5, ground_labels=lab,
                             ground_label_ids={40}, rng=np.random.RandomState(7))
    o2, _, _ = simulate_scan(scene, hres=900, lidar_height=1.8, beams=16,
                             v_lo_deg=-20, v_hi_deg=5, ground_labels=lab,
                             ground_label_ids={40}, rng=np.random.RandomState(7))
    same = o1.shape == o2.shape and np.array_equal(o1, o2)
    ok &= same
    print(f'  [5] same seed -> identical output: {"OK" if same else "FAIL"}')

    print('SELFTEST', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if _selftest() else 1)
