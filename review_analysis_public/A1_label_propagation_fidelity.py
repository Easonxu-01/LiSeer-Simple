#!/usr/bin/env python3
"""A1 semantic fidelity of label propagation (-> Fig R1, D1-D7).

Method (reproduces the propagation algorithm of tools/generate_dense_points_labels_sk.py exactly):
  1. take the annotated points of N real sparse scans; withhold 20% at random and propagate from the remaining 80%;
  2. score the propagated labels against the ground truth on the withheld 20% -> semantic fidelity (accuracy, per-class IoU);
  3. boundary-voxel dilation ratio: boundary voxels of the dense labels divided by the same count for the sparse labels.

A withheld point must be scored with the label propagated into its own voxel, which is the same path a dense point takes,
so this measures the error of the propagation algorithm itself and nothing else.
"""
import argparse, json, os, sys
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree

LEARNING_MAP = None  # see --map; uses labels_map from the config
VOXEL = 0.2


def voxelize(p, v=VOXEL):
    return np.floor(p / v).astype(np.int32)


def build_voxel_labels(pts, lab, v=VOXEL):
    """Sparse points -> per-voxel majority-vote label (as in generate_dense_labels, vectorized).

    The original loops over points with a dict and Counter; here a joint unique over (voxel, label) plus an argmax over counts
    gives the same result two orders of magnitude faster (a per-point Python loop is not feasible at 200 frames x 120k points).
    Ties are broken by the smaller label value; Counter.most_common breaks ties by insertion order,
    which makes no difference to the accuracy and IoU statistics computed here.
    """
    vi = voxelize(pts, v)
    keys, inv = np.unique(vi, axis=0, return_inverse=True)
    pair = np.stack([inv, lab.astype(np.int64)], 1)
    upair, cnt = np.unique(pair, axis=0, return_counts=True)
    # sort by (voxel, count) and take the last entry of each voxel = the largest count
    order = np.lexsort((upair[:, 1], cnt, upair[:, 0]))
    up, cn = upair[order], cnt[order]
    last = np.ones(up.shape[0], bool)
    last[:-1] = up[1:, 0] != up[:-1, 0]
    win = up[last]
    out_lab = np.zeros(keys.shape[0], np.int64)
    out_lab[win[:, 0]] = win[:, 1]
    return keys, out_lab


def propagate(vox_keys, vox_lab, query_pts, v=VOXEL):
    """Query point -> propagated label: the voxel's own label if it has one, otherwise the nearest filled voxel (as in the original)."""
    qi = voxelize(query_pts, v)
    tree = cKDTree(vox_keys)
    # exact hit: distance zero
    dist, idx = tree.query(qi, k=1)
    return vox_lab[idx], dist > 0


def boundary_voxels(pts, lab, v=VOXEL):
    """Number of boundary voxels: voxels with a differently labelled voxel in their 6-neighbourhood."""
    keys, vlab = build_voxel_labels(pts, lab, v)
    lut = {tuple(k): l for k, l in zip(keys, vlab)}
    nb = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    b = 0
    for k, l in lut.items():
        for d in nb:
            n = (k[0]+d[0], k[1]+d[1], k[2]+d[2])
            m = lut.get(n)
            if m is not None and m != l:
                b += 1
                break
    return b, len(lut)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/semantickitti/sequences')
    ap.add_argument('--seq', default='08')
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--holdout', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='review_analysis/_source_data/A1_label_propagation.json')
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    vdir = os.path.join(a.root, a.seq, 'velodyne')
    ldir = os.path.join(a.root, a.seq, 'labels')
    ddir = os.path.join(a.root, a.seq, 'refine')
    dldir = os.path.join(a.root, a.seq, 'dense_labels')
    files = sorted(f for f in os.listdir(vdir) if f.endswith('.bin'))[:a.frames]

    tot = cor = 0
    miss_tot = 0
    per_cls = {}                    # cls -> [inter, union_pred, union_gt]
    dil_num = dil_den = 0
    n_sparse = n_dense = 0
    for fi, f in enumerate(files):
        stem = f[:-4]
        pts = np.fromfile(os.path.join(vdir, f), np.float32).reshape(-1, 4)[:, :3]
        lab = np.fromfile(os.path.join(ldir, stem + '.label'), np.uint32) & 0xFFFF
        if pts.shape[0] != lab.shape[0]:
            continue
        # annotated points only
        m = lab > 0
        pts, lab = pts[m], lab[m]
        if pts.shape[0] < 100:
            continue

        # --- 80/20 hold-out ---
        idx = rng.permutation(pts.shape[0])
        nho = int(round(a.holdout * pts.shape[0]))
        ho, keep = idx[:nho], idx[nho:]
        vk, vlb = build_voxel_labels(pts[keep], lab[keep])
        pred, miss = propagate(vk, vlb, pts[ho])
        gt = lab[ho]
        tot += gt.size; cor += int((pred == gt).sum()); miss_tot += int(miss.sum())
        for c in np.unique(np.concatenate([gt, pred])):
            e = per_cls.setdefault(int(c), [0, 0, 0])
            e[0] += int(((pred == c) & (gt == c)).sum())
            e[1] += int((pred == c).sum())
            e[2] += int((gt == c).sum())

        # --- boundary-voxel dilation ratio: dense vs sparse ---
        dp = os.path.join(ddir, stem + '.npy')
        dl = os.path.join(dldir, stem + '.label')
        if os.path.isfile(dp) and os.path.isfile(dl):
            dpts = np.load(dp)
            dlab = np.fromfile(dl, np.uint32) & 0xFFFF
            if dpts.shape[0] == dlab.shape[0]:
                md = dlab > 0
                bd, nd = boundary_voxels(dpts[md], dlab[md])
                bs, ns = boundary_voxels(pts, lab)
                dil_num += bd; dil_den += bs
                n_dense += int(md.sum()); n_sparse += pts.shape[0]
        if (fi + 1) % 25 == 0:
            print(f'  {fi+1}/{len(files)} frames  running accuracy {cor/max(tot,1):.4f}', flush=True)

    acc = cor / max(tot, 1)
    ious = {c: (v[0] / max(v[1] + v[2] - v[0], 1)) for c, v in per_cls.items() if v[2] > 0}
    miou = float(np.mean(list(ious.values()))) if ious else 0.0
    res = dict(
        seq=a.seq, frames=len(files), holdout=a.holdout, seed=a.seed, voxel=VOXEL,
        D1_holdout_points=tot,
        D2_propagation_accuracy=acc,
        D3_propagation_mIoU=miou,
        D4_per_class_IoU={str(k): round(v, 4) for k, v in sorted(ious.items())},
        D5_nearest_voxel_fallback_ratio=miss_tot / max(tot, 1),
        D6_boundary_voxel_dilation=(dil_num / dil_den) if dil_den else None,
        D7_mean_points=dict(sparse=n_sparse / max(len(files), 1),
                            dense=n_dense / max(len(files), 1)),
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != 'D4_per_class_IoU'},
                     indent=2, ensure_ascii=False))
    print(f'\nwritten to {a.out}')


if __name__ == '__main__':
    main()
