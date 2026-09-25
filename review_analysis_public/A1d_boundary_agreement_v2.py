#!/usr/bin/env python3
"""A1 补充 v2：边界一致性的修正测法（M1-M5）。

背景：A1b 的 E1_containment 受三条机制影响
（稀疏体素标签不稳、占据邻居不对称、体素级精确匹配过严），这里给出修正后的测法。
本脚本的四个指标在**看到任何结果之前**定死，无论高低照实回报。

M4 最重要：有真值、不依赖体素化、不依赖任何边界集合定义。
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from A1_label_propagation_fidelity import build_voxel_labels, propagate

VOXEL = 0.2
NB6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
RADII = (0.2, 0.3, 0.5)
BANDS = [(0.0, 0.2, 'band_0_20cm'), (0.2, 0.4, 'band_20_40cm'),
         (0.4, 1.0, 'band_40_100cm'), (1.0, np.inf, 'band_gt_100cm')]


def voxelize(p, v=VOXEL):
    return np.floor(p / v).astype(np.int64)


def dist_to_nearest_other_class(pts, lab):
    """每点到最近的**异类**原始标注点的欧氏距离。

    只用原始稀疏标注（那是真值），不涉及致密云。逐类建树：对类 c 的点，
    查询"所有非 c 点"构成的树。
    """
    d = np.full(pts.shape[0], np.inf)
    classes = np.unique(lab)
    if classes.size < 2:
        return d
    for c in classes:
        m = lab == c
        other = pts[~m]
        if other.shape[0] == 0:
            continue
        dd, _ = cKDTree(other).query(pts[m], k=1)
        d[m] = dd
    return d


def boundary_mask_pointwise(pts, lab, r):
    """点级边界：半径 r 内存在异类点。两侧用同一函数 -> 完全对称。"""
    tree = cKDTree(pts)
    pairs = tree.query_ball_point(pts, r)
    out = np.zeros(pts.shape[0], bool)
    for i, nb in enumerate(pairs):
        li = lab[i]
        for j in nb:
            if lab[j] != li:
                out[i] = True
                break
    return out


def boundary_voxels_on(keys, vlab, restrict=None):
    """体素级边界（6-连通，邻居须已占据）。restrict 给定时只在该集合内判定。"""
    idx = {tuple(k): i for i, k in enumerate(keys)}
    allow = None if restrict is None else restrict
    out = []
    for i, k in enumerate(keys):
        kt = (int(k[0]), int(k[1]), int(k[2]))
        if allow is not None and kt not in allow:
            continue
        li = vlab[i]
        for d in NB6:
            n = (kt[0]+d[0], kt[1]+d[1], kt[2]+d[2])
            if allow is not None and n not in allow:
                continue
            j = idx.get(n)
            if j is not None and vlab[j] != li:
                out.append(kt)
                break
    return set(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/semantickitti/sequences')
    ap.add_argument('--seq', default='08')
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--holdout', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='review_analysis/_source_data/A1d_boundary_agreement_v2.json')
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    vdir = os.path.join(a.root, a.seq, 'velodyne')
    ldir = os.path.join(a.root, a.seq, 'labels')
    ddir = os.path.join(a.root, a.seq, 'refine')
    dldir = os.path.join(a.root, a.seq, 'dense_labels')
    files = sorted(f for f in os.listdir(vdir) if f.endswith('.bin'))[:a.frames]

    # M4 累加器
    m4 = {n: [0, 0] for _, _, n in BANDS}          # [correct, total]
    # M3 累加器
    m3 = {r: [0, 0, 0] for r in RADII}             # [inter, n_sparse, n_dense]
    # M2 / M1 / M5
    m2_i = m2_s = m2_v = 0
    sp_vox_tot = 0
    m1_d = []
    lost_pc, kept_pc = [], []
    nf = 0

    for fi, f in enumerate(files):
        stem = f[:-4]
        dp, dl = os.path.join(ddir, stem + '.npy'), os.path.join(dldir, stem + '.label')
        pts = np.fromfile(os.path.join(vdir, f), np.float32).reshape(-1, 4)[:, :3]
        lab = np.fromfile(os.path.join(ldir, stem + '.label'), np.uint32) & 0xFFFF
        if pts.shape[0] != lab.shape[0]:
            continue
        m = lab > 0
        pts, lab = pts[m].astype(np.float64), lab[m]
        if pts.shape[0] < 100:
            continue

        # ---------- M4：留出实验 + 到最近异类真值点的距离分层 ----------
        idx = rng.permutation(pts.shape[0])
        nho = int(round(a.holdout * pts.shape[0]))
        ho, keep = idx[:nho], idx[nho:]
        vk, vlb = build_voxel_labels(pts[keep], lab[keep])
        pred, _ = propagate(vk, vlb, pts[ho])
        gt = lab[ho]
        dist_all = dist_to_nearest_other_class(pts, lab)     # 全帧真值距离
        dho = dist_all[ho]
        ok = (pred == gt)
        for lo, hi, name in BANDS:
            sel = (dho >= lo) & (dho < hi)
            m4[name][0] += int(ok[sel].sum())
            m4[name][1] += int(sel.sum())

        if not (os.path.isfile(dp) and os.path.isfile(dl)):
            nf += 1
            continue
        dpts = np.load(dp).astype(np.float64)
        dlab = np.fromfile(dl, np.uint32) & 0xFFFF
        if dpts.shape[0] != dlab.shape[0]:
            nf += 1
            continue
        md = dlab > 0
        dpts, dlab = dpts[md], dlab[md]
        if dpts.shape[0] < 100:
            nf += 1
            continue

        # ---------- M3：点级、无体素化、两侧同函数 ----------
        # 真实点在致密云中的标签：取最近致密点的标签（真实点是致密云的子集，
        # 最近邻距离应≈0；这样两侧比较的是同一批物理点）
        _, nn = cKDTree(dpts).query(pts, k=1)
        for r in RADII:
            bs = boundary_mask_pointwise(pts, lab, r)
            # 致密侧：以真实点为中心，在致密云里找异类邻居
            tree_d = cKDTree(dpts)
            nbs = tree_d.query_ball_point(pts, r)
            lab_d_self = dlab[nn]
            bd = np.zeros(pts.shape[0], bool)
            for i, nb in enumerate(nbs):
                li = lab_d_self[i]
                for j in nb:
                    if dlab[j] != li:
                        bd[i] = True
                        break
            m3[r][0] += int((bs & bd).sum())
            m3[r][1] += int(bs.sum())
            m3[r][2] += int(bd.sum())

        # ---------- 体素侧：M1 / M2 / M5 ----------
        ks, ls = build_voxel_labels(pts, lab)
        kd, ld = build_voxel_labels(dpts, dlab)
        Bs = boundary_voxels_on(ks, ls)
        Bd = boundary_voxels_on(kd, ld)
        sp_vox_tot += ks.shape[0]
        if Bs and Bd:
            # M1：稀疏边界体素 -> 最近致密边界体素的距离（米）
            arr_d = np.array(sorted(Bd), np.float64)
            arr_s = np.array(sorted(Bs), np.float64)
            dd, _ = cKDTree(arr_d).query(arr_s, k=1)
            m1_d.append(dd * VOXEL)
            # M5：丢失 vs 保留的边界体素，每体素点数
            vi = voxelize(pts)
            uk, cnt = np.unique(vi, axis=0, return_counts=True)
            pc = {tuple(k): int(c) for k, c in zip(uk, cnt)}
            for k in Bs:
                (kept_pc if k in Bd else lost_pc).append(pc.get(k, 0))
        # M2：共同支撑集（两侧都占据的体素）上重算
        set_s = set(map(tuple, ks))
        set_d = set(map(tuple, kd))
        both = set_s & set_d
        if both:
            Bs2 = boundary_voxels_on(ks, ls, restrict=both)
            Bd2 = boundary_voxels_on(kd, ld, restrict=both)
            m2_i += len(Bs2 & Bd2); m2_s += len(Bs2); m2_v += len(both)
        nf += 1
        if (fi + 1) % 20 == 0:
            b0 = m4['band_0_20cm']; bi = m4['band_gt_100cm']
            print(f'  {fi+1}/{len(files)}  M4 band0={b0[0]/max(b0[1],1):.4f} '
                  f'interior={bi[0]/max(bi[1],1):.4f}', flush=True)

    d1 = np.concatenate(m1_d) if m1_d else np.zeros(1)
    acc = lambda t: round(t[0] / max(t[1], 1), 4)
    res = dict(
        task='A1-supplement-v2 (boundary agreement, corrected metrics)', status='DONE', script='review_analysis/A1d_boundary_agreement_v2.py',
        frames=nf, seq=a.seq, holdout=a.holdout, seed=a.seed, voxel_size_m=VOXEL,
        M4_accuracy_by_boundary_distance={
            n: dict(acc=acc(m4[n]), n=m4[n][1]) for _, _, n in BANDS},
        M4_overall_boundary_band_accuracy=acc(m4['band_0_20cm']),
        M4_interior_accuracy=acc(m4['band_gt_100cm']),
        M4_gap=round(acc(m4['band_gt_100cm']) - acc(m4['band_0_20cm']), 4),
        M3={f'r{r}': dict(
            containment=round(m3[r][0] / max(m3[r][1], 1), 4),
            reverse=round(m3[r][0] / max(m3[r][2], 1), 4),
            n_bnd_sparse=m3[r][1], n_bnd_dense=m3[r][2]) for r in RADII},
        M2_containment_common_support=round(m2_i / max(m2_s, 1), 4),
        M2_n_voxels_common=m2_v,
        M2_common_frac_of_sparse_occupied=round(m2_v / max(sp_vox_tot, 1), 4),
        M1_frac_d_eq_0=round(float((d1 == 0).mean()), 4),
        M1_frac_d_le_1vox=round(float((d1 <= VOXEL + 1e-9).mean()), 4),
        M1_frac_d_le_2vox=round(float((d1 <= 2 * VOXEL + 1e-9).mean()), 4),
        M1_median_d_m=round(float(np.median(d1)), 4),
        M1_p95_d_m=round(float(np.percentile(d1, 95)), 4),
        M5_lost_boundary_point_count=dict(
            lost_median=round(float(np.median(lost_pc)), 2) if lost_pc else None,
            kept_median=round(float(np.median(kept_pc)), 2) if kept_pc else None,
            n_lost=len(lost_pc), n_kept=len(kept_pc)),
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
