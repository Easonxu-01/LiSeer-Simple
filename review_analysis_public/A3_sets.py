#!/usr/bin/env python3
"""Frame selection and the four comparison sets shared by every A3 v2 script.

real_A / real_B  : the 200+200 real scans (protocol split, seed=0)
sim_cur          : real_A's dense scenes re-observed with the TRAINING
                   CURRICULUM's random sensor draw (Table B1) -- the mixture
                   the coverage question is about
sim_cond         : the same scenes re-observed with the dataset's own measured
                   sensor (review_analysis/_source_data/A3_nominal_configs.json) -- the fidelity
                   question. Real beam table, real horizontal resolution, real
                   origin, mounting height drawn in a +-0.15 m band around the
                   measured value so it stays a distribution rather than a
                   single deterministic scan.
"""
import json
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from A3_run_f1f4 import (GROUND_IDS, discover_sk_frames, discover_nusc_frames,
                         discover_waymo_frames, load_sk_frames, load_nusc_frames,
                         load_waymo_frames, sample_sensor_params)
from A3_simulator import simulate_scan, estimate_ground_z

SEED = 0
N_FRAMES = 400
N_SIDE = 200
HEIGHT_BAND = 0.15

BASE = {'semantickitti': '/data/Datasets/semantickitti',
        'nuscenes': '/data/Datasets/nuscenes',
        'waymo': 'data/waymo/kitti_format'}
DISCOVER = {'semantickitti': discover_sk_frames, 'nuscenes': discover_nusc_frames,
            'waymo': discover_waymo_frames}
LOAD = {'semantickitti': load_sk_frames, 'nuscenes': load_nusc_frames,
        'waymo': load_waymo_frames}

_NOM = None


def nominal(dataset):
    global _NOM
    if _NOM is None:
        _NOM = json.load(open('review_analysis/_source_data/A3_nominal_configs.json'))
    return _NOM[dataset]


def select(dataset):
    """The protocol's 400-frame draw, split into real_A / real_B."""
    allf = DISCOVER[dataset](BASE[dataset])
    if len(allf) < N_FRAMES:
        raise RuntimeError(f'{dataset}: only {len(allf)} frames with dense data')
    rng = np.random.RandomState(SEED)
    idx = sorted(rng.choice(len(allf), size=N_FRAMES, replace=False))
    sel = [allf[i] for i in idx]
    return sel[:N_SIDE], sel[N_SIDE:]


def load_frames(dataset, specs):
    return LOAD[dataset](BASE[dataset], specs)


def make_sim(dataset, frame, mode, frame_seed):
    """Re-observe one dense scene. Returns (xyz, labels, info)."""
    rng = np.random.RandomState(frame_seed)
    gids = GROUND_IDS[dataset]
    dense_xyz, dense_lab = frame['dense_xyz'], frame['dense_lab']
    if mode == 'sim_cur':
        beams, height, hres, v_lo, v_hi = sample_sensor_params(rng)
        xyz, idx, info = simulate_scan(
            dense_xyz, hres=hres, lidar_height=height, beams=beams,
            v_lo_deg=v_lo, v_hi_deg=v_hi, origin_xy=(0.0, 0.0),
            ground_labels=dense_lab, ground_label_ids=gids, rng=rng)
    elif mode == 'sim_cond':
        nom = nominal(dataset)
        h0 = nom['mount_height']['height_m']
        height = float(rng.uniform(h0 - HEIGHT_BAND, h0 + HEIGHT_BAND))
        xyz, idx, info = simulate_scan(
            dense_xyz, hres=int(nom['n_horiz']), lidar_height=height,
            elev_table_deg=nom['table_deg'],
            origin_xy=(nom['origin_xyz'][0], nom['origin_xyz'][1]),
            ground_labels=dense_lab, ground_label_ids=gids, rng=rng)
    else:
        raise ValueError(mode)
    if len(idx) == 0:
        return xyz, np.zeros((0,), dense_lab.dtype), info
    return xyz, dense_lab[idx], info
