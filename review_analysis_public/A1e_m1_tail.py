#!/usr/bin/env python3
"""A1d/M1 addendum: mean and tail statistics of the boundary displacement distance.

Uses exactly the definition of M1 in A1d, over the same 200 frames:
  d = Euclidean distance from each sparse boundary voxel to the nearest dense one (voxel units x 0.2 m).
Only M1 is recomputed (M3 and M4 are skipped), which makes this much faster than A1d.
The distance array is also written out, and the tail is traced to its origin (range from the sensor, and whether the class is dynamic).
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from A1_label_propagation_fidelity import build_voxel_labels

VOXEL = 0.2
NB6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
# SemanticKITTI dynamic classes (moving-*)
MOVING = {252, 253, 254, 255, 256, 257, 258, 259}


def boundary_voxels_on(keys, vlab):
    idx = {tuple(k): i for i, k in enumerate(keys)}
    out = []
    for i, k in enumerate(keys):
        kt = (int(k[0]), int(k[1]), int(k[2]))
        li = vlab[i]
        for d in NB6:
            j = idx.get((kt[0]+d[0], kt[1]+d[1], kt[2]+d[2]))
            if j is not None and vlab[j] != li:
                out.append((kt, li))
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/semantickitti/sequences')
    ap.add_argument('--seq', default='08')
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--out', default='review_analysis/_source_data/A1e_m1_tail.json')
    a = ap.parse_args()

    vdir = os.path.join(a.root, a.seq, 'velodyne')
    ldir = os.path.join(a.root, a.seq, 'labels')
    ddir = os.path.join(a.root, a.seq, 'refine')
    dldir = os.path.join(a.root, a.seq, 'dense_labels')
    files = sorted(f for f in os.listdir(vdir) if f.endswith('.bin'))[:a.frames]

    D, RNG, MOV = [], [], []      # distance / range of the voxel from the sensor / dynamic-class flag
    nf = 0
    for fi, f in enumerate(files):
        stem = f[:-4]
        dp, dl = os.path.join(ddir, stem + '.npy'), os.path.join(dldir, stem + '.label')
        if not (os.path.isfile(dp) and os.path.isfile(dl)):
            continue
        pts = np.fromfile(os.path.join(vdir, f), np.float32).reshape(-1, 4)[:, :3]
        lab = np.fromfile(os.path.join(ldir, stem + '.label'), np.uint32) & 0xFFFF
        dpts = np.load(dp)
        dlab = np.fromfile(dl, np.uint32) & 0xFFFF
        if pts.shape[0] != lab.shape[0] or dpts.shape[0] != dlab.shape[0]:
            continue
        m, md = lab > 0, dlab > 0
        pts, lab, dpts, dlab = pts[m].astype(np.float64), lab[m], dpts[md], dlab[md]
        if pts.shape[0] < 100 or dpts.shape[0] < 100:
            continue

        ks, ls = build_voxel_labels(pts, lab)
        kd, ld = build_voxel_labels(dpts, dlab)
        Bs = boundary_voxels_on(ks, ls)
        Bd = boundary_voxels_on(kd, ld)
        if not Bs or not Bd:
            continue
        arr_s = np.array([k for k, _ in Bs], np.float64)
        lab_s = np.array([l for _, l in Bs], np.int64)
        arr_d = np.array([k for k, _ in Bd], np.float64)
        dd, _ = cKDTree(arr_d).query(arr_s, k=1)
        D.append(dd * VOXEL)
        # provenance: horizontal distance from the voxel centre to the sensor origin (0,0,0)
        ctr = (arr_s + 0.5) * VOXEL
        RNG.append(np.sqrt(ctr[:, 0] ** 2 + ctr[:, 1] ** 2))
        MOV.append(np.isin(lab_s, list(MOVING)))
        nf += 1
        if (fi + 1) % 40 == 0:
            print(f'  {fi+1}/{len(files)}', flush=True)

    d = np.concatenate(D); rg = np.concatenate(RNG); mv = np.concatenate(MOV)
    np.savez_compressed('review_analysis/_source_data/A1e_m1_distances.npz', d=d, rng=rg, moving=mv)

    tail = d > 0.4                      # the 1.69% not covered by the quantiles already reported
    far = d > 1.0
    def rstat(mask):
        if mask.sum() == 0:
            return None
        return dict(n=int(mask.sum()),
                    range_median_m=round(float(np.median(rg[mask])), 2),
                    range_p90_m=round(float(np.percentile(rg[mask], 90)), 2),
                    moving_frac=round(float(mv[mask].mean()), 4))

    res = dict(
        task='A1d-supplement (boundary displacement, tail statistics)', status='DONE', script='review_analysis/A1e_m1_tail.py',
        frames=nf, seq=a.seq, voxel_size_m=VOXEL,
        n_sparse_boundary_voxels=int(d.size),
        M1_mean_d_m=round(float(d.mean()), 4),
        M1_p99_d_m=round(float(np.percentile(d, 99)), 4),
        M1_max_d_m=round(float(d.max()), 2),
        M1_frac_d_gt_1m=round(float((d > 1.0).mean()), 6),
        M1_frac_d_gt_2m=round(float((d > 2.0).mean()), 6),
        # recompute the published quantiles to check they agree with A1d
        recheck=dict(
            frac_d_eq_0=round(float((d == 0).mean()), 4),
            frac_d_le_1vox=round(float((d <= VOXEL + 1e-9).mean()), 4),
            frac_d_le_2vox=round(float((d <= 2 * VOXEL + 1e-9).mean()), 4),
            median_d_m=round(float(np.median(d)), 4),
            p95_d_m=round(float(np.percentile(d, 95)), 4)),
        tail_origin=dict(
            all_boundary=rstat(np.ones_like(d, bool)),
            d_gt_0p4m=rstat(tail),
            d_gt_1m=rstat(far)),
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
