#!/usr/bin/env python3
"""A2 geometric fidelity of densification (-> Fig R2, D8).

Two measurements:
  1. per-class nearest-neighbour distance from a real point to a dense point: whether densification places points off the real surface.
     For each real sparse point the nearest dense point is found and the distance distribution is collected per class.
  2. diffusion-response profile of the surface variation: original versus dense.
     Surface variation = lambda_0 / (lambda_0+lambda_1+lambda_2), the share of the smallest eigenvalue of a local PCA
     (about 0 on a plane, about 1/3 at an edge or corner). The diffusion response applies the operator of the CODE-4 family,
     D_alpha = I - alpha*L, to that scalar field and reads the retained energy e_b after b steps.
     If densification is faithful the two profiles should have the same shape: a denser cloud should not change the geometric spectrum of the surface.
"""
import argparse, json, os
import numpy as np
from scipy.spatial import cKDTree

ALPHA = 0.25
B = 8
K = 16          # neighbours used for the local PCA and for the graph neighbourhood


def surface_variation(pts, k=K):
    """Per-point surface variation = lambda_min / sum(lambda)."""
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k)
    nb = pts[idx]                                   # [N,k,3]
    c = nb - nb.mean(1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', c, c) / k
    w = np.linalg.eigvalsh(cov)                     # ascending
    s = w.sum(1)
    return np.where(s > 1e-12, w[:, 0] / np.maximum(s, 1e-12), 0.0), idx


def diffusion_profile(x, idx, alpha=ALPHA, b=B):
    """e_b = ||D^b x||^2 / ||x||^2 with D = I - alpha*L, where L is the row-normalized kNN graph Laplacian.

    Same family as CODE-4: L = I - W with W the row-normalized adjacency, spectrum in [0,2] and L*1 = 0.
    """
    x = x - x.mean()
    n0 = float((x * x).sum())
    if n0 <= 0:
        return [0.0] * b
    out, y = [], x.copy()
    for _ in range(b):
        # L y = y - mean_neighbors(y)  ->  D y = y - alpha*(y - mean_nb(y))
        y = y - alpha * (y - y[idx].mean(1))
        out.append(float((y * y).sum()) / n0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/semantickitti/sequences')
    ap.add_argument('--seq', default='08')
    ap.add_argument('--frames', type=int, default=50)
    ap.add_argument('--sub', type=int, default=20000, help='points subsampled for the profile')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='review_analysis/_source_data/A2_densification_geometry.json')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    vdir = os.path.join(a.root, a.seq, 'velodyne')
    ldir = os.path.join(a.root, a.seq, 'labels')
    ddir = os.path.join(a.root, a.seq, 'refine')
    files = sorted(f for f in os.listdir(vdir) if f.endswith('.bin'))[:a.frames]

    per_cls = {}                       # cls -> list of nn distances
    prof_s, prof_d = [], []
    nf = 0
    for fi, f in enumerate(files):
        stem = f[:-4]
        dp = os.path.join(ddir, stem + '.npy')
        if not os.path.isfile(dp):
            continue
        pts = np.fromfile(os.path.join(vdir, f), np.float32).reshape(-1, 4)[:, :3]
        lab = np.fromfile(os.path.join(ldir, stem + '.label'), np.uint32) & 0xFFFF
        dpts = np.load(dp).astype(np.float64)
        if pts.shape[0] != lab.shape[0] or dpts.shape[0] < 100:
            continue
        m = lab > 0
        pts, lab = pts[m].astype(np.float64), lab[m]

        # --- 1) per-class nearest-neighbour distance, real point -> dense point ---
        d, _ = cKDTree(dpts).query(pts, k=1)
        for c in np.unique(lab):
            per_cls.setdefault(int(c), []).append(d[lab == c])

        # --- 2) diffusion response of the surface variation ---
        for src, acc in ((pts, prof_s), (dpts, prof_d)):
            n = src.shape[0]
            sel = rng.choice(n, min(a.sub, n), replace=False)
            sv, idx = surface_variation(src[sel])
            acc.append(diffusion_profile(sv, idx))
        nf += 1
        if (fi + 1) % 10 == 0:
            print(f'  {fi+1}/{len(files)} frames', flush=True)

    stats = {}
    for c, arrs in sorted(per_cls.items()):
        v = np.concatenate(arrs)
        stats[str(c)] = dict(n=int(v.size), mean_cm=round(float(v.mean()) * 100, 3),
                             median_cm=round(float(np.median(v)) * 100, 3),
                             p95_cm=round(float(np.percentile(v, 95)) * 100, 3))
    allv = np.concatenate([np.concatenate(x) for x in per_cls.values()]) if per_cls else np.zeros(1)
    ps = np.array(prof_s).mean(0).tolist() if prof_s else []
    pd = np.array(prof_d).mean(0).tolist() if prof_d else []
    res = dict(
        seq=a.seq, frames=nf, alpha=ALPHA, B=B, knn=K, subsample=a.sub,
        D8_nn_distance_overall=dict(
            mean_cm=round(float(allv.mean()) * 100, 3),
            median_cm=round(float(np.median(allv)) * 100, 3),
            p95_cm=round(float(np.percentile(allv, 95)) * 100, 3)),
        D8_nn_distance_per_class=stats,
        diffusion_profile_sparse=[round(v, 6) for v in ps],
        diffusion_profile_dense=[round(v, 6) for v in pd],
        profile_max_abs_diff=round(float(np.max(np.abs(np.array(ps) - np.array(pd)))), 6)
        if ps and pd else None,
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items()
                      if k != 'D8_nn_distance_per_class'}, indent=2, ensure_ascii=False))
    print(f'\nwritten to {a.out}')


if __name__ == '__main__':
    main()
