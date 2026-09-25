#!/usr/bin/env python3
"""A3 F1–F4: sim-real distribution metrics for SemanticKITTI, nuScenes, Waymo.

Usage:
    python review_analysis/run_a3_f1f4.py --dataset semantickitti
    python review_analysis/run_a3_f1f4.py --dataset nuscenes
    python review_analysis/run_a3_f1f4.py --dataset waymo
    python review_analysis/run_a3_f1f4.py --all    # all three in sequence

Output:
    review_analysis/_source_data/A3_simreal_metrics.csv   (appended, one row per dataset×family×metric)
    review_analysis/_source_data/A3_histograms.npz        (all histograms for plotting and audit)

Protocol (per spec §1.2):
    - 400 frames drawn with seed=0 from frames that have a dense counterpart
    - Split: real_A = first 200, real_B = second 200
    - sim: real_A scenes re-observed with one random sensor draw each (seed=0 per frame)
    - d_sim2real = dist(sim_from_A, real_B)
    - d_real2real = dist(real_A,    real_B)
    - Bootstrap 100× at the frame level (seed=0), 95% CI on ratio inside each resample

Sensor parameter ranges (from pointtpv_seg.py sensor_config defaults = Table B1):
    num_of_beams:                   [16, 128]
    LiDAR_height (m):               [1.0, 2.0]
    horizontal_angular_resolution:  [900, 3600]   (integer steps of 100)
    lower_vertical_field_of_view:   [-40, -5]     (degrees)
    upper_vertical_field_of_view:   [0,   25]     (degrees)
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import scipy.spatial as spatial
import csv

# ---------------------------------------------------------------------------
# Histogram bin definitions (fixed across all three sets per spec §1.3)
# ---------------------------------------------------------------------------

F1_EDGES = np.linspace(0, 80, 41)          # 40 bins, 0-80 m, 2 m each
F2_EDGES = np.concatenate([np.linspace(0, 1.0, 101), [np.inf]])  # 100 bins + overflow
F3_EDGES = np.linspace(0, 1.0, 51)         # 50 bins for linearity/planarity/scattering
F4_EDGES = np.linspace(-3.0, 10.0, 53)     # 52 bins, 0.25 m each

BOOTSTRAP_N = 100
BOOTSTRAP_SEED = 0
SAMPLE_SEED = 0
N_FRAMES = 400
N_SIDE = 200        # real_A = frames[0:200], real_B = frames[200:400]
F3_RADIUS = 0.5     # fixed-radius neighbourhood for PCA (spec §1.3 F3)
F3_MIN_NBR = 10     # skip points with fewer neighbours

# Ground label IDs per dataset (label_id = raw_class & 0xFFFF for SK/Waymo)
# nuScenes: driveable_surface=1, other_flat=9 (fine-17 mapping: 1=driveable)
# SemanticKITTI: 40=road, 44=parking, 48=walking, 49=other-ground, 60=lane-marking, 72=terrain
# Waymo: check actual labels at runtime
PROJECT_ROOT = os.environ.get('UNILIDAR_ROOT', os.getcwd())  # repo root

# A3 reads RAW label files, so these must be raw dataset ids, not the
# remapped ids used in the training configs.
#   SemanticKITTI raw: 40=road 44=parking 48=sidewalk 49=other-ground
#                      60=lane-marking 72=terrain
#   nuScenes raw lidarseg: 24=driveable_surface 25=other_flat
#                          26=sidewalk 27=terrain
#     (the training config's 1/9 are REMAPPED ids - using them here matched
#      nothing at all, so F4 silently fell back to a percentile estimate)
#   Waymo raw WOD: 18=road 19=lane_marker 20=other_ground
#                  21=walkable 22=sidewalk
GROUND_IDS = {
    'semantickitti': {40, 44, 48, 49, 60, 72},
    'nuscenes':      {24, 25, 26, 27},
    'waymo':         {18, 19, 20, 21, 22},
}


# ---------------------------------------------------------------------------
# Dataset frame loaders
# Each returns a list of dicts: {'real_xyz': (N,3), 'real_lab': (N,), 'dense_xyz': (M,3), 'dense_lab': (M,)}
# ---------------------------------------------------------------------------

def load_sk_frames(base_dir, frame_indices):
    """Load SemanticKITTI frames.

    base_dir: /data/Datasets/semantickitti
    frame_indices: list of (seq_str, frame_str) tuples
    """
    frames = []
    for seq, fr in frame_indices:
        seq_dir = os.path.join(base_dir, 'sequences', seq)
        # Real scan
        bin_path = os.path.join(seq_dir, 'velodyne', fr + '.bin')
        lbl_path = os.path.join(seq_dir, 'labels', fr + '.label')
        real_pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        real_lbl = np.fromfile(lbl_path, dtype=np.uint32) & 0xFFFF
        # Dense: refine/{frame}.npy (N,3 float32) + dense_labels/{frame}.label.
        # NOT dense_clean_undo/ - that is an undo journal of points DELETED by
        # the label-cleaning pass (see tools/label_fidelity/fix_torn_rollback.py);
        # it holds a few thousand removed points, not the scene.
        npy_path = os.path.join(seq_dir, 'refine', fr + '.npy')
        dlbl_path = os.path.join(seq_dir, 'dense_labels', fr + '.label')
        dense_xyz = np.load(npy_path).astype(np.float32)
        dense_lbl = np.fromfile(dlbl_path, dtype=np.uint32) & 0xFFFF
        if dense_lbl.shape[0] != dense_xyz.shape[0]:
            raise ValueError(
                f'SK {seq}/{fr}: refine has {dense_xyz.shape[0]} points but '
                f'dense_labels has {dense_lbl.shape[0]} - counts must match.')
        frames.append({'real_xyz': real_pts, 'real_lab': real_lbl,
                       'dense_xyz': dense_xyz, 'dense_lab': dense_lbl})
    return frames


def load_nusc_frames(base_dir, frame_indices, nusc_api=None):
    """Load nuScenes frames.

    base_dir: /data/Datasets/nuscenes
    frame_indices: list of (sample_token, lidar_filename) tuples
    """
    frames = []
    for token, lidar_fn in frame_indices:
        # Real scan: samples/LIDAR_TOP/<filename>.pcd.bin  (x,y,z,intensity,ring)
        bin_path = os.path.join(base_dir, 'samples', 'LIDAR_TOP', lidar_fn)
        raw = np.fromfile(bin_path, dtype=np.float32)
        ncols = 5 if len(raw) % 5 == 0 else 4
        real_pts = raw.reshape(-1, ncols)[:, :3]
        # Real labels: lidarseg/<token>_lidarseg.bin  (uint8 class IDs)
        seg_path = os.path.join(base_dir, 'lidarseg', 'v1.0-trainval',
                                token + '_lidarseg.bin')
        real_lbl = np.fromfile(seg_path, dtype=np.uint8).astype(np.uint32)
        # Dense: samples/LIDAR_TOP/refine/<file>.pcd.npy (N,3 float32) plus
        # lidarseg_refine/v1.0-trainval/<token>_lidarseg.bin (uint8), matching
        # nuscenes_seg_dataset.py. NOT lidarseg_clean_undo/ - that is an undo
        # journal of points DELETED by the label-cleaning pass, not the scene.
        npy_path = os.path.join(base_dir, 'samples', 'LIDAR_TOP', 'refine',
                                os.path.splitext(lidar_fn)[0] + '.npy')
        dlbl_path = os.path.join(base_dir, 'lidarseg_refine', 'v1.0-trainval',
                                 token + '_lidarseg.bin')
        dense_xyz = np.load(npy_path).astype(np.float32)
        dense_lbl = np.fromfile(dlbl_path, dtype=np.uint8).astype(np.uint32)
        if dense_lbl.shape[0] != dense_xyz.shape[0]:
            raise ValueError(
                f'nuScenes {token}: refine has {dense_xyz.shape[0]} points but '
                f'lidarseg_refine has {dense_lbl.shape[0]} - counts must match.')
        frames.append({'real_xyz': real_pts, 'real_lab': real_lbl,
                       'dense_xyz': dense_xyz, 'dense_lab': dense_lbl})
    return frames


def load_waymo_frames(base_dir, frame_indices):
    """Load Waymo frames.

    base_dir: project data/waymo/kitti_format
    frame_indices: list of frame_stem strings (e.g. '0000024')
    """
    frames = []
    train_dir = os.path.join(base_dir, 'training')
    for stem in frame_indices:
        # Real scan
        # Waymo point clouds are 6-column (load_dim=6 in the project configs).
        bin_path = os.path.join(train_dir, 'velodyne', stem + '.bin')
        real_pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 6)[:, :3]
        # Real labels live beside the .bin, stored int32 as (N,2); column 1 is
        # the semantic id. Matches _load_pts_semantic_mask() in
        # projects/unilidar_plugin/datasets/pipelines/loading_voxels_sk.py.
        lbl_path = os.path.join(train_dir, 'velodyne', stem + '.label')
        real_lbl = np.fromfile(lbl_path, dtype=np.int32).reshape(-1, 2)[:, 1]
        # Dense: this tree stores dense xyz as refine/{stem}.npy (N,3 float32),
        # with per-point labels in dense_labels/{stem}.label (uint32, same N).
        # dense_clean_undo/ exists here but is empty.
        npz_path = os.path.join(train_dir, 'dense_clean_undo', stem + '.npz')
        npy_path = os.path.join(train_dir, 'refine', stem + '.npy')
        if os.path.exists(npz_path):
            d = np.load(npz_path)
            dense_xyz = d['xyz'].astype(np.float32)
            dense_lbl = (d['lab'].astype(np.uint32) & 0xFFFF)
        elif os.path.exists(npy_path):
            dense_xyz = np.load(npy_path).astype(np.float32)
            # Dense labels are int32 (N,1) - the is_dense branch of
            # _load_pts_semantic_mask(), unlike the (N,2) sparse layout.
            dlbl_path = os.path.join(train_dir, 'dense_labels', stem + '.label')
            dense_lbl = np.fromfile(dlbl_path, dtype=np.int32).reshape(-1, 1)[:, 0]
            if dense_lbl.shape[0] != dense_xyz.shape[0]:
                raise ValueError(
                    f'Waymo {stem}: refine npy has {dense_xyz.shape[0]} points but '
                    f'dense_labels has {dense_lbl.shape[0]} - counts must match.')
        else:
            raise FileNotFoundError(
                f'Waymo dense not found for {stem}: tried {npz_path} and {npy_path}')
        frames.append({'real_xyz': real_pts, 'real_lab': real_lbl,
                       'dense_xyz': dense_xyz, 'dense_lab': dense_lbl})
    return frames


# ---------------------------------------------------------------------------
# Frame index discovery
# ---------------------------------------------------------------------------

def discover_sk_frames(base_dir, train_sequences=None):
    """Return sorted list of (seq, frame) for all SK frames that have dense data."""
    if train_sequences is None:
        # Standard training split: 00-07, 09-10
        train_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    result = []
    for seq in train_sequences:
        dense_dir = os.path.join(base_dir, 'sequences', seq, 'refine')
        vel_dir   = os.path.join(base_dir, 'sequences', seq, 'velodyne')
        if not os.path.isdir(dense_dir) or not os.path.isdir(vel_dir):
            continue
        dense_stems = {os.path.splitext(f)[0] for f in os.listdir(dense_dir) if f.endswith('.npy')}
        lbl_dir     = os.path.join(base_dir, 'sequences', seq, 'dense_labels')
        if os.path.isdir(lbl_dir):
            dense_stems &= {os.path.splitext(f)[0] for f in os.listdir(lbl_dir)
                            if f.endswith('.label')}
        vel_stems   = {os.path.splitext(f)[0] for f in os.listdir(vel_dir) if f.endswith('.bin')}
        common = sorted(dense_stems & vel_stems)
        for stem in common:
            result.append((seq, stem))
    return result


def discover_nusc_frames(base_dir):
    """Return sorted list of (token, lidar_filename) for nuScenes frames with dense data."""
    import pickle
    pkl_path = os.path.join(base_dir, '..', '..', 'workspace_perception', 'UniLiDAR',
                            'data', 'metadata', 'nuscenes_infos_train.pkl')
    # Try multiple candidate paths
    candidates = [
        os.path.join(base_dir, 'nuscenes_infos_train.pkl'),
        os.path.join(base_dir, '..', 'nuscenes_infos_train.pkl'),
        os.path.join(PROJECT_ROOT, 'data/metadata/nuscenes_infos_train.pkl'),
        os.path.join(os.path.dirname(base_dir), 'nuscenes_infos_train.pkl'),
    ]
    infos = None
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.exists(c):
            with open(c, 'rb') as f:
                d = pickle.load(f)
            if isinstance(d, dict):
                # older schema: {'infos': [...]}; mmdet3d>=1.x: {'data_list': [...]}
                infos = d.get('infos', d.get('data_list'))
                if infos is None:
                    raise KeyError(
                        f'{c}: expected "infos" or "data_list", got {sorted(d)[:8]}')
            else:
                infos = d
            break
    if infos is None:
        raise FileNotFoundError(
            'nuscenes_infos_train.pkl not found. '
            'Tried: ' + str(candidates))

    dense_dir = os.path.join(base_dir, 'lidarseg_refine', 'v1.0-trainval')
    dense_tokens = {f.split('_lidarseg')[0] for f in os.listdir(dense_dir) if f.endswith('.bin')}

    result = []
    for info in infos:
        token = info.get('token', '')
        # older schema: info['lidar_path']; newer: info['lidar_points']['lidar_path']
        lidar_path = info.get('lidar_path', '')
        if not lidar_path:
            lidar_path = (info.get('lidar_points') or {}).get('lidar_path', '')
        lidar_fn = os.path.basename(lidar_path)
        # Dense files are named by the lidarseg token from pts_semantic_mask_path,
        # which differs from the sample token.
        seg_path = info.get('pts_semantic_mask_path', '') or ''
        seg_token = os.path.basename(seg_path).split('_lidarseg')[0]
        key = seg_token if seg_token in dense_tokens else token
        if key in dense_tokens and lidar_fn:
            result.append((key, lidar_fn))
    return sorted(result)


def discover_waymo_frames(base_dir):
    """Return sorted list of frame stems for Waymo training frames with dense data."""
    train_dir = os.path.join(base_dir, 'training')
    vel_dir   = os.path.join(train_dir, 'velodyne')
    dense_dir = os.path.join(train_dir, 'dense_clean_undo')
    lbl_dir   = os.path.join(train_dir, 'dense_labels')

    refine_dir = os.path.join(train_dir, 'refine')
    if not os.path.isdir(dense_dir) and not os.path.isdir(refine_dir):
        raise FileNotFoundError(
            f'Waymo dense not found: neither {dense_dir} nor {refine_dir}')

    dense_stems = set()
    if os.path.isdir(dense_dir):
        dense_stems = {os.path.splitext(f)[0] for f in os.listdir(dense_dir) if f.endswith('.npz')}
    if not dense_stems and os.path.isdir(refine_dir):
        # This tree keeps dense xyz in refine/{stem}.npy instead.
        dense_stems = {os.path.splitext(f)[0] for f in os.listdir(refine_dir) if f.endswith('.npy')}
    vel_stems   = {os.path.splitext(f)[0] for f in os.listdir(vel_dir) if f.endswith('.bin')}
    lbl_stems   = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.label')}
    # Only part of the tree is segmentation-annotated; seg_annotated_indices.txt
    # lists those frames (same file the Waymo dataset class uses).
    # Two copies exist and they differ: the kitti_format/ one is truncated
    # (max index 158052) while the project-root one covers the whole tree, so
    # prefer the root copy and fall back to the local one.
    idx_candidates = [
        os.path.join(PROJECT_ROOT, 'seg_annotated_indices.txt'),
        os.path.join(base_dir, 'seg_annotated_indices.txt'),
    ]
    idx_file = next((c for c in idx_candidates if os.path.isfile(c)), '')
    if idx_file:
        # Two formats exist in the tree: bare integers ("24") under
        # kitti_format/, and padded filenames ("0000024.bin") at the project
        # root. Normalise both to the 7-digit stem used on disk.
        with open(idx_file) as f:
            seg_stems = set()
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                stem = os.path.splitext(os.path.basename(ln))[0]
                seg_stems.add(stem.zfill(7) if stem.isdigit() else stem)
        if seg_stems:
            vel_stems &= seg_stems
    # Real-scan labels sit beside the .bin in velodyne/, not in dense_labels/.
    real_lbl_stems = {os.path.splitext(f)[0] for f in os.listdir(vel_dir)
                      if f.endswith('.label')}
    return sorted(dense_stems & vel_stems & lbl_stems & real_lbl_stems)


# ---------------------------------------------------------------------------
# Per-frame feature extraction
# ---------------------------------------------------------------------------

def extract_frame_features(xyz, ground_ids=None, ground_lab=None):
    """
    Compute all F1–F4 per-frame histograms.

    Returns dict with keys: f1, f2, f3_lin, f3_plan, f3_scat, f4
    Each is a normalized histogram (sums to 1, or 0 if no valid points).
    Also returns 'skipped_f3_frac': fraction of F3 points skipped.
    """
    N = len(xyz)
    result = {}

    # ---- F1: radial density -------------------------------------------------
    r = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    h, _ = np.histogram(r, bins=F1_EDGES)
    total = h.sum()
    result['f1'] = h / max(total, 1)

    # ---- F2: nearest-neighbour distance -------------------------------------
    if N >= 2:
        tree = spatial.cKDTree(xyz)
        # Query 2 neighbours (self + nearest)
        dists, _ = tree.query(xyz, k=2)
        nn_dist = dists[:, 1]
        h2, _ = np.histogram(nn_dist, bins=F2_EDGES)
        result['f2'] = h2 / max(h2.sum(), 1)
    else:
        result['f2'] = np.zeros(len(F2_EDGES) - 1)

    # ---- F3: local geometry signature (fixed-radius PCA) --------------------
    if N >= F3_MIN_NBR:
        tree3 = spatial.cKDTree(xyz)
        idxs_list = tree3.query_ball_point(xyz, r=F3_RADIUS, workers=-1)
        counts = np.fromiter((len(n) for n in idxs_list), dtype=np.int64,
                             count=len(idxs_list))
        keep = counts >= F3_MIN_NBR
        skipped = int((~keep).sum())
        if keep.any():
            # Batch the PCA: build every neighbourhood's covariance with one
            # segmented reduction, then eigendecompose all 3x3 matrices at
            # once. Same math as the per-point loop, ~2 orders faster.
            keep_idx = np.flatnonzero(keep)
            nbr_counts = counts[keep_idx]
            flat_nbrs = np.concatenate([np.asarray(idxs_list[i], dtype=np.int64)
                                        for i in keep_idx])
            owner = np.repeat(np.arange(len(keep_idx)), nbr_counts)
            patch = xyz[flat_nbrs] - xyz[keep_idx][owner]
            # outer products per point, summed per neighbourhood
            n_keep = len(keep_idx)
            cov = np.empty((n_keep, 3, 3), dtype=np.float64)
            for a in range(3):
                for b in range(a, 3):
                    sums = np.bincount(owner, weights=patch[:, a] * patch[:, b],
                                       minlength=n_keep)
                    cov[:, a, b] = sums
                    cov[:, b, a] = sums
            cov /= nbr_counts[:, None, None]
            eig = np.linalg.eigvalsh(cov)                      # ascending
            eig = np.maximum(eig, 0.0)
            lam3, lam2, lam1 = eig[:, 0], eig[:, 1], eig[:, 2]
            denom = np.maximum(lam1, 1e-12)
            degenerate = eig.sum(axis=1) < 1e-12
            lin_vals  = ((lam1 - lam2) / denom)[~degenerate]
            plan_vals = ((lam2 - lam3) / denom)[~degenerate]
            scat_vals = (lam3 / denom)[~degenerate]
            skipped += int(degenerate.sum())
        else:
            lin_vals = plan_vals = scat_vals = np.array([])
        if len(lin_vals):
            h_lin,  _ = np.histogram(lin_vals,  bins=F3_EDGES)
            h_plan, _ = np.histogram(plan_vals, bins=F3_EDGES)
            h_scat, _ = np.histogram(scat_vals, bins=F3_EDGES)
            result['f3_lin']  = h_lin  / max(h_lin.sum(),  1)
            result['f3_plan'] = h_plan / max(h_plan.sum(), 1)
            result['f3_scat'] = h_scat / max(h_scat.sum(), 1)
            result['skipped_f3_frac'] = skipped / max(N, 1)
        else:
            result['f3_lin']  = np.zeros(len(F3_EDGES) - 1)
            result['f3_plan'] = np.zeros(len(F3_EDGES) - 1)
            result['f3_scat'] = np.zeros(len(F3_EDGES) - 1)
            result['skipped_f3_frac'] = 1.0
    else:
        result['f3_lin']  = np.zeros(len(F3_EDGES) - 1)
        result['f3_plan'] = np.zeros(len(F3_EDGES) - 1)
        result['f3_scat'] = np.zeros(len(F3_EDGES) - 1)
        result['skipped_f3_frac'] = 1.0

    # ---- F4: height distribution (relative to ground) -----------------------
    z = xyz[:, 2].copy()
    if ground_lab is not None and ground_ids is not None:
        rho2d = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
        mask_g = np.isin(ground_lab, list(ground_ids)) & (rho2d < 20.0)
        gpts = z[mask_g]
        if len(gpts) >= 5:
            ground_z = float(np.median(gpts))
        else:
            ground_z = float(np.percentile(z, 2))
        z = z - ground_z
    h4, _ = np.histogram(z, bins=F4_EDGES)
    result['f4'] = h4 / max(h4.sum(), 1)

    return result


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def pool_histograms(frame_feats, key):
    """Average per-frame normalized histograms to get a pooled distribution."""
    arrs = [f[key] for f in frame_feats if key in f]
    if not arrs:
        return None
    return np.mean(arrs, axis=0)


def js_divergence(p, q):
    """Jensen-Shannon divergence, log base 2, value in [0, 1]."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Normalize (guard against zero sum from empty histograms)
    sp, sq = p.sum(), q.sum()
    if sp < 1e-15 or sq < 1e-15:
        return np.nan
    p = p / sp
    q = q / sq
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def wasserstein1d(p_hist, q_hist, edges):
    """1D Wasserstein distance between two histograms over the given bin edges."""
    p = np.asarray(p_hist, dtype=np.float64)
    q = np.asarray(q_hist, dtype=np.float64)
    sp, sq = p.sum(), q.sum()
    if sp < 1e-15 or sq < 1e-15:
        return np.nan
    p /= sp
    q /= sq
    # Use midpoints for finite bins, last finite edge for overflow
    mids = 0.5 * (edges[:-1] + edges[1:])
    if not np.isfinite(mids[-1]):
        mids[-1] = edges[-2] + (edges[-2] - edges[-3])
    # Wasserstein = integral of |CDF_p - CDF_q|
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    bin_widths = np.diff(mids, prepend=mids[0] - (mids[1] - mids[0]))
    return float(np.sum(np.abs(cdf_p - cdf_q) * bin_widths))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_distances(feats_sim, feats_realB, key, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """
    Bootstrap at the frame level.

    feats_sim   : list of per-frame feature dicts (sim_from_A, length N_SIDE)
    feats_realB : list of per-frame feature dicts (real_B,     length N_SIDE)

    Returns (js_vals, was_vals, ratio_vals) each of length n_boot.
    was_vals is None if the key has no corresponding Wasserstein edges.
    """
    rng = np.random.RandomState(seed)
    n = len(feats_sim)
    assert len(feats_realB) == n

    # Wasserstein edges lookup
    was_edges = {'f1': F1_EDGES, 'f2': F2_EDGES, 'f4': F4_EDGES,
                 'f3_lin': None, 'f3_plan': None, 'f3_scat': None}

    js_sim2real_vals, js_real2real_vals, ratio_vals = [], [], []

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        sim_boot   = [feats_sim[i]   for i in idx]
        realB_boot = [feats_realB[i] for i in idx]
        # For real2real we resample real_A independently (passed via same object as feats_sim here
        # — actually we need real_A feats too; see call site)
        # (will be patched at call site — see compute_metrics)
        h_sim   = pool_histograms(sim_boot,   key)
        h_realB = pool_histograms(realB_boot, key)
        if h_sim is None or h_realB is None:
            continue
        js_sim2real_vals.append(js_divergence(h_sim, h_realB))

    return js_sim2real_vals


def bootstrap_all_distances(feats_sim, feats_realA, feats_realB,
                            key, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """
    Returns arrays: js_sim2real[n_boot], js_real2real[n_boot], ratio[n_boot].
    Each bootstrap resamples frames independently for the two comparisons.
    """
    rng = np.random.RandomState(seed)
    n = len(feats_sim)
    assert len(feats_realA) == n and len(feats_realB) == n

    js_s2r_vals, js_r2r_vals, ratio_vals = [], [], []

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        sim_b   = [feats_sim[i]   for i in idx]
        realA_b = [feats_realA[i] for i in idx]
        realB_b = [feats_realB[i] for i in idx]

        h_sim   = pool_histograms(sim_b,   key)
        h_realA = pool_histograms(realA_b, key)
        h_realB = pool_histograms(realB_b, key)
        if h_sim is None or h_realA is None or h_realB is None:
            continue

        js_s2r = js_divergence(h_sim,   h_realB)
        js_r2r = js_divergence(h_realA, h_realB)
        js_s2r_vals.append(js_s2r)
        js_r2r_vals.append(js_r2r)
        if js_r2r > 1e-15:
            ratio_vals.append(js_s2r / js_r2r)

    return (np.array(js_s2r_vals), np.array(js_r2r_vals), np.array(ratio_vals))


def ci95(arr):
    """Return (mean, lo, hi) 95% CI from bootstrap array."""
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ---------------------------------------------------------------------------
# Main metrics computation
# ---------------------------------------------------------------------------

FAMILIES = [
    ('F1_radial_density',    'f1',     '-',         True),   # has Wasserstein
    ('F2_nn_distance',       'f2',     '-',         True),
    ('F3_geom_signature',    'f3_lin', 'linearity', False),  # JS only
    ('F3_geom_signature',    'f3_plan','planarity',  False),
    ('F3_geom_signature',    'f3_scat','scattering', False),
    ('F4_height_dist',       'f4',     '-',         True),
]

WAS_EDGES_MAP = {
    'f1': F1_EDGES,
    'f2': F2_EDGES,
    'f4': F4_EDGES,
}


def compute_metrics(dataset_name, frames_realA, frames_realB, frames_sim,
                    ground_ids, n_points_stats):
    """
    Compute all F1–F4 metrics for one dataset.
    Returns list of CSV row dicts.
    """
    print(f'  Extracting features for {N_SIDE} sim frames...')
    t0 = time.time()
    feats_sim   = [extract_frame_features(f['dense_xyz'],
                                          ground_ids=ground_ids,
                                          ground_lab=f['dense_lab']) for f in frames_sim]
    print(f'  Extracting features for {N_SIDE} real_A frames...')
    feats_realA = [extract_frame_features(f['real_xyz'],
                                          ground_ids=ground_ids,
                                          ground_lab=f['real_lab']) for f in frames_realA]
    print(f'  Extracting features for {N_SIDE} real_B frames...')
    feats_realB = [extract_frame_features(f['real_xyz'],
                                          ground_ids=ground_ids,
                                          ground_lab=f['real_lab']) for f in frames_realB]
    print(f'  Feature extraction done in {time.time()-t0:.1f}s')

    # Skipped fraction (F3)
    skipped_sim   = np.mean([f.get('skipped_f3_frac', 0) for f in feats_sim])
    skipped_realA = np.mean([f.get('skipped_f3_frac', 0) for f in feats_realA])
    skipped_realB = np.mean([f.get('skipped_f3_frac', 0) for f in feats_realB])
    print(f'  F3 skipped frac: sim={skipped_sim:.3f}, realA={skipped_realA:.3f}, realB={skipped_realB:.3f}')

    rows = []
    for fam_name, feat_key, subq, has_was in FAMILIES:
        print(f'  Bootstrap {fam_name}/{subq}...')
        js_s2r, js_r2r, ratio = bootstrap_all_distances(
            feats_sim, feats_realA, feats_realB, feat_key)

        js_s2r_m, js_s2r_lo, js_s2r_hi = ci95(js_s2r)
        js_r2r_m, js_r2r_lo, js_r2r_hi = ci95(js_r2r)
        rat_m,    rat_lo,    rat_hi     = ci95(ratio)

        # skipped_frac: only meaningful for F3
        sf = skipped_sim if 'f3' in feat_key else 0.0

        row_base = dict(
            dataset=dataset_name,
            family=fam_name,
            subquantity=subq,
            n_scans_per_side=N_SIDE,
            n_points_sim=n_points_stats['sim'],
            n_points_realA=n_points_stats['realA'],
            n_points_realB=n_points_stats['realB'],
            skipped_frac=f'{sf:.4f}',
        )

        rows.append({**row_base,
                     'metric': 'JS',
                     'd_sim2real': js_s2r_m,  'd_sim2real_lo': js_s2r_lo,  'd_sim2real_hi': js_s2r_hi,
                     'd_real2real': js_r2r_m, 'd_real2real_lo': js_r2r_lo, 'd_real2real_hi': js_r2r_hi,
                     'ratio': rat_m,          'ratio_lo': rat_lo,           'ratio_hi': rat_hi})

        # Wasserstein (scalar families only)
        if has_was and feat_key in WAS_EDGES_MAP:
            edges = WAS_EDGES_MAP[feat_key]
            h_sim   = pool_histograms(feats_sim,   feat_key)
            h_realA = pool_histograms(feats_realA, feat_key)
            h_realB = pool_histograms(feats_realB, feat_key)
            w_s2r = wasserstein1d(h_sim,   h_realB, edges)
            w_r2r = wasserstein1d(h_realA, h_realB, edges)
            w_ratio = w_s2r / w_r2r if (w_r2r and w_r2r > 1e-15) else np.nan
            rows.append({**row_base,
                         'metric': 'Wasserstein_m',
                         'd_sim2real': w_s2r,   'd_sim2real_lo': np.nan, 'd_sim2real_hi': np.nan,
                         'd_real2real': w_r2r,  'd_real2real_lo': np.nan, 'd_real2real_hi': np.nan,
                         'ratio': w_ratio,       'ratio_lo': np.nan,      'ratio_hi': np.nan})

    return rows


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset_name):
    print(f'\n{"="*60}')
    print(f'A3 F1–F4: {dataset_name}')
    print(f'{"="*60}')

    # --- Discover frames ---
    if dataset_name == 'semantickitti':
        base_dir = '/data/Datasets/semantickitti'
        all_frames = discover_sk_frames(base_dir)
        ground_ids = GROUND_IDS['semantickitti']
    elif dataset_name == 'nuscenes':
        base_dir = '/data/Datasets/nuscenes'
        all_frames = discover_nusc_frames(base_dir)
        ground_ids = GROUND_IDS['nuscenes']
    elif dataset_name == 'waymo':
        base_dir = os.path.join(PROJECT_ROOT, 'data/waymo/kitti_format')
        all_frames = discover_waymo_frames(base_dir)
        ground_ids = GROUND_IDS['waymo']
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    print(f'  Found {len(all_frames)} frames with dense data.')
    if len(all_frames) < N_FRAMES:
        raise RuntimeError(
            f'{dataset_name}: only {len(all_frames)} frames available, need {N_FRAMES}.')

    # --- Sample 400 frames with seed=0 ---
    rng = np.random.RandomState(SAMPLE_SEED)
    idx400 = rng.choice(len(all_frames), size=N_FRAMES, replace=False)
    idx400 = sorted(idx400)
    selected = [all_frames[i] for i in idx400]
    sel_A = selected[:N_SIDE]
    sel_B = selected[N_SIDE:]

    # --- Load real scans ---
    print(f'  Loading {N_SIDE} real_A frames...')
    t0 = time.time()
    if dataset_name == 'semantickitti':
        frames_realA = load_sk_frames(base_dir, sel_A)
        frames_realB = load_sk_frames(base_dir, sel_B)
    elif dataset_name == 'nuscenes':
        frames_realA = load_nusc_frames(base_dir, sel_A)
        frames_realB = load_nusc_frames(base_dir, sel_B)
    elif dataset_name == 'waymo':
        frames_realA = load_waymo_frames(base_dir, sel_A)
        frames_realB = load_waymo_frames(base_dir, sel_B)
    print(f'  Real frames loaded in {time.time()-t0:.1f}s')

    # --- Simulate scans from real_A dense scenes ---
    print(f'  Simulating {N_SIDE} scans from real_A dense scenes...')
    t0 = time.time()
    sim_rng = np.random.RandomState(SAMPLE_SEED)
    frames_sim = []
    for i, fa in enumerate(frames_realA):
        beams, height, hres, v_lo, v_hi = sample_sensor_params(sim_rng)
        sim_xyz, sim_idx = simulate_scan(
            fa['dense_xyz'], beams, height, hres, v_lo, v_hi,
            ground_labels=fa['dense_lab'],
            ground_label_ids=ground_ids)
        frames_sim.append({'dense_xyz': sim_xyz,
                           'dense_lab': fa['dense_lab'][sim_idx]})
        if (i + 1) % 50 == 0:
            print(f'    simulated {i+1}/{N_SIDE}')
    print(f'  Simulation done in {time.time()-t0:.1f}s')

    # Point count stats (mean across frames)
    n_points_stats = {
        'sim':   int(np.mean([len(f['dense_xyz']) for f in frames_sim])),
        'realA': int(np.mean([len(f['real_xyz'])  for f in frames_realA])),
        'realB': int(np.mean([len(f['real_xyz'])  for f in frames_realB])),
    }
    print(f'  Mean pts/frame: sim={n_points_stats["sim"]}, '
          f'realA={n_points_stats["realA"]}, realB={n_points_stats["realB"]}')

    # --- Compute metrics ---
    rows = compute_metrics(dataset_name, frames_realA, frames_realB, frames_sim,
                           ground_ids, n_points_stats)

    # --- Save histograms for plotting ---
    histo_data = {}
    for tag, feats in [('sim', [extract_frame_features(f['dense_xyz']) for f in frames_sim]),
                       ('realA', [extract_frame_features(f['real_xyz']) for f in frames_realA]),
                       ('realB', [extract_frame_features(f['real_xyz']) for f in frames_realB])]:
        for key in ['f1', 'f2', 'f3_lin', 'f3_plan', 'f3_scat', 'f4']:
            h = pool_histograms(feats, key)
            if h is not None:
                histo_data[f'{dataset_name}_{tag}_{key}'] = h

    return rows, histo_data


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    'dataset', 'family', 'subquantity', 'metric',
    'd_sim2real', 'd_sim2real_lo', 'd_sim2real_hi',
    'd_real2real', 'd_real2real_lo', 'd_real2real_hi',
    'ratio', 'ratio_lo', 'ratio_hi',
    'n_scans_per_side', 'n_points_sim', 'n_points_realA', 'n_points_realB',
    'skipped_frac',
]


def write_csv(rows, path, append=False):
    mode = 'a' if append else 'w'
    write_header = not (append and os.path.exists(path))
    with open(path, mode, newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow({k: (f'{row[k]:.6f}' if isinstance(row[k], float) else row[k])
                        for k in CSV_FIELDS})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='A3 F1-F4 sim-real distribution metrics')
    parser.add_argument('--dataset', choices=['semantickitti', 'nuscenes', 'waymo'])
    parser.add_argument('--all', action='store_true', help='Run all three datasets')
    args = parser.parse_args()

    if not args.all and args.dataset is None:
        parser.error('Specify --dataset <name> or --all')

    datasets = ['semantickitti', 'nuscenes', 'waymo'] if args.all else [args.dataset]

    os.makedirs('review_analysis/_source_data', exist_ok=True)
    os.makedirs('figs', exist_ok=True)

    csv_path = 'review_analysis/_source_data/A3_simreal_metrics.csv'
    npz_path = 'review_analysis/_source_data/A3_histograms.npz'

    all_rows = []
    all_histos = {}

    for ds in datasets:
        try:
            rows, histos = run_dataset(ds)
            all_rows.extend(rows)
            all_histos.update(histos)
            # Write partial results immediately
            write_csv(rows, csv_path, append=os.path.exists(csv_path))
            print(f'\n  Wrote {len(rows)} rows to {csv_path}')
        except Exception as e:
            print(f'\nERROR processing {ds}: {e}')
            import traceback; traceback.print_exc()
            print('Continuing with next dataset...')

    # Save all histograms
    if all_histos:
        # Load existing histograms if appending
        if os.path.exists(npz_path):
            existing = dict(np.load(npz_path))
            existing.update(all_histos)
            all_histos = existing
        np.savez(npz_path, **all_histos)
        print(f'\nHistograms saved to {npz_path}')

    print(f'\nDone. Total rows written: {len(all_rows)}')
    print(f'CSV: {csv_path}')
    print(f'Histograms: {npz_path}')
    print('\nNext: run review_analysis/plot_a3.py to generate figs/R3.pdf')




# ---------------------------------------------------------------------------
# Sensor sampling (mirrors pointtpv_seg.py logic exactly)
# ---------------------------------------------------------------------------

def _sample_beams_int(lo, hi):
    """Log-uniform over powers-of-2 in [lo, hi], then nearest power of 2."""
    powers = [2**k for k in range(int(math.log2(lo)), int(math.log2(hi)) + 1)
              if lo <= 2**k <= hi]
    if not powers:
        return lo
    return random.choice(powers)


def sample_sensor_params(rng=None):
    """Draw one sensor configuration using the training curriculum ranges."""
    if rng is not None:
        # Use numpy rng for reproducibility per frame
        beams = int(rng.choice([2**k for k in range(4, 8)
                                if 16 <= 2**k <= 128]))
        height = float(rng.uniform(1.0, 2.0))
        hres = int(rng.choice(np.arange(900, 3601, 100)))
        v_lo = float(round(rng.uniform(-40.0, -5.0), 1))
        v_hi = float(round(rng.uniform(0.0, 25.0), 1))
    else:
        beams = _sample_beams_int(16, 128)
        height = random.uniform(1.0, 2.0)
        hres = random.choice(range(900, 3601, 100))
        v_lo = round(random.uniform(-40.0, -5.0), 1)
        v_hi = round(random.uniform(0.0, 25.0), 1)
    return beams, height, hres, v_lo, v_hi


# ---------------------------------------------------------------------------
# Synthetic scan via raycasting (point-centric, O(N+R))
# ---------------------------------------------------------------------------

def simulate_scan(xyz, beams, lidar_height, hres, v_lo_deg, v_hi_deg,
                  ground_labels=None, ground_label_ids=(1, 7)):
    """
    Re-observe a dense scene as a simulated LiDAR scan.

    Args:
        xyz: (N,3) float32 point positions in sensor-centric frame
        beams: number of elevation channels
        lidar_height: mounting height above ground (m)
        hres: number of horizontal rays (azimuth steps)
        v_lo_deg / v_hi_deg: elevation range (degrees)
        ground_labels: (N,) int labels, used to estimate ground z
        ground_label_ids: label IDs considered ground

    Returns:
        (M,3) float32 points that survive the z-buffer selection
    """
    if len(xyz) == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))

    # Ground z estimation: median z of ground points within 20 m (spec §1.3 F4)
    ground_z = 0.0
    if ground_labels is not None:
        mask_ground = np.isin(ground_labels, list(ground_label_ids))
        rho = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
        mask_near = rho < 20.0
        gpts = xyz[mask_ground & mask_near, 2]
        if len(gpts) >= 5:
            ground_z = float(np.median(gpts))
        else:
            ground_z = float(np.percentile(xyz[:, 2], 2))

    origin_z = ground_z + lidar_height

    v_lo = math.radians(v_lo_deg)
    v_hi = math.radians(v_hi_deg)
    if beams > 1:
        elev_angles = np.linspace(v_lo, v_hi, beams, dtype=np.float64)
    else:
        elev_angles = np.array([(v_lo + v_hi) / 2.0], dtype=np.float64)

    h_step = 2 * math.pi / hres
    az_phase = float(np.random.uniform(0, h_step))

    pts = xyz.astype(np.float64)
    ox, oy, oz = 0.0, 0.0, origin_z

    dx = pts[:, 0] - ox
    dy = pts[:, 1] - oy
    dz = pts[:, 2] - oz

    # Azimuth and elevation of each point
    az = np.arctan2(dy, dx)                        # (-π, π)
    rho_xy = np.sqrt(dx**2 + dy**2)
    el = np.arctan2(dz, np.maximum(rho_xy, 1e-6))

    # Assign each point to the nearest beam
    el_step = (v_hi - v_lo) / max(beams - 1, 1)
    el_half = 0.5 * el_step if beams > 1 else 1e6
    beam_idx = np.round((el - v_lo) / max(el_step, 1e-9)).astype(np.int32)
    beam_idx = np.clip(beam_idx, 0, beams - 1)
    in_fov = np.abs(el - elev_angles[beam_idx]) <= el_half

    # Assign each point to the nearest azimuth bin
    az_norm = (az - az_phase) % (2 * math.pi)
    az_idx = np.floor(az_norm / h_step).astype(np.int32)
    az_idx = np.clip(az_idx, 0, hres - 1)

    # Z-buffer: for each (beam, az) cell, keep the nearest point
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    cell = beam_idx * hres + az_idx

    # Only points in FOV
    valid = in_fov
    cell_v = cell[valid]
    dist_v = dist[valid]
    idx_v = np.where(valid)[0]

    if len(idx_v) == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))

    # Sort by cell, then by distance; keep first occurrence per cell
    order = np.lexsort((dist_v, cell_v))
    cell_sorted = cell_v[order]
    idx_sorted = idx_v[order]
    _, first = np.unique(cell_sorted, return_index=True)
    survivors = idx_sorted[first]

    # Return the surviving indices as well so the caller can carry the dense
    # labels across. F4 needs a real ground estimate on the sim side too;
    # without labels it silently falls back to a different estimator than the
    # real side, which inflates the ratio.
    return xyz[survivors].astype(np.float32), survivors


if __name__ == '__main__':
    main()
