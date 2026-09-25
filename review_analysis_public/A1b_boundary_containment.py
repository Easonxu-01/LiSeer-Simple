#!/usr/bin/env python3
"""A1 追加项：边界包含率（E1-E6）。

回答的问题不是"边界体素变多了吗"（那是致密化的必然结果），而是
**稀疏标注认定的边界位置，在致密标注里是否仍然是边界** —— 即边界有没有被
移动或加宽（bleeding），还是只是沿轮廓补得更完整。

边界体素定义（稀疏侧与致密侧完全同一规则）：
  · 体素级：先按 0.2m 体素对点做多数投票得到体素标签，再比较邻居体素标签
    （不是点级"体素内含 >=2 种标签的点"）；
  · 6-连通邻域；
  · 邻居必须**已占据**（存在于该侧的体素集合内）才参与比较。

⚠ 这条"邻居必须已占据"在两侧造成结构性不对称，见脚本输出的 asymmetry 段：
  稀疏侧点少、体素稀疏，一个真实边界体素的邻居常常是空的，于是**不被判为边界**。
  因此 B_sparse 系统性偏少，且偏向"两侧都有足够点"的强边界。
  这会**高估** containment（因为留下的都是最容易在致密侧仍然成立的边界）。
  脚本一并报出两侧的占据率与"因邻居为空而未能判定"的体素比例，供如实说明。
"""
import argparse, json, os
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree

VOXEL = 0.2
NB6 = np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)], np.int64)


def voxelize(p, v=VOXEL):
    return np.floor(p / v).astype(np.int64)


def voxel_labels(pts, lab, v=VOXEL):
    """体素多数投票（向量化；并列取标签值较小者）。"""
    vi = voxelize(pts, v)
    keys, inv = np.unique(vi, axis=0, return_inverse=True)
    pair = np.stack([inv, lab.astype(np.int64)], 1)
    up, cnt = np.unique(pair, axis=0, return_counts=True)
    order = np.lexsort((up[:, 1], cnt, up[:, 0]))
    up, cnt = up[order], cnt[order]
    last = np.ones(up.shape[0], bool)
    last[:-1] = up[1:, 0] != up[:-1, 0]
    win = up[last]
    out = np.zeros(keys.shape[0], np.int64)
    out[win[:, 0]] = win[:, 1]
    return keys, out


def boundary_set(keys, vlab):
    """返回 (边界体素坐标 [B,3], 有空邻居的体素数, 体素总数)。

    判定：6-邻域内存在**已占据**且标签不同的邻居。
    同时统计"至少有一个邻居为空"的体素数 —— 这是两侧不对称的度量。
    """
    n = keys.shape[0]
    idx = {tuple(k): i for i, k in enumerate(keys)}
    is_b = np.zeros(n, bool)
    has_empty_nb = np.zeros(n, bool)
    for i, k in enumerate(keys):
        kt = (int(k[0]), int(k[1]), int(k[2]))
        li = vlab[i]
        for d in NB6:
            j = idx.get((kt[0] + int(d[0]), kt[1] + int(d[1]), kt[2] + int(d[2])))
            if j is None:
                has_empty_nb[i] = True
            elif vlab[j] != li:
                is_b[i] = True
    return keys[is_b], int(has_empty_nb.sum()), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/semantickitti/sequences')
    ap.add_argument('--seq', default='08')
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--out', default='review_analysis/_source_data/A1b_boundary_containment.json')
    a = ap.parse_args()

    vdir = os.path.join(a.root, a.seq, 'velodyne')
    ldir = os.path.join(a.root, a.seq, 'labels')
    ddir = os.path.join(a.root, a.seq, 'refine')
    dldir = os.path.join(a.root, a.seq, 'dense_labels')
    files = sorted(f for f in os.listdir(vdir) if f.endswith('.bin'))[:a.frames]

    inter_tot = bs_tot = bd_tot = 0
    per_frame = []
    new_d = []                     # 新增边界体素到最近稀疏边界体素的距离(米)
    bs_counts, bd_counts = [], []
    sp_empty_nb = sp_vox = dn_empty_nb = dn_vox = 0
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
        pts, lab, dpts, dlab = pts[m], lab[m], dpts[md], dlab[md]
        if pts.shape[0] < 100 or dpts.shape[0] < 100:
            continue

        ks, ls = voxel_labels(pts, lab)
        kd, ld = voxel_labels(dpts, dlab)
        Bs, se, sv = boundary_set(ks, ls)
        Bd, de, dv = boundary_set(kd, ld)
        sp_empty_nb += se; sp_vox += sv; dn_empty_nb += de; dn_vox += dv
        bs_counts.append(Bs.shape[0]); bd_counts.append(Bd.shape[0])
        if Bs.shape[0] == 0 or Bd.shape[0] == 0:
            continue

        setd = set(map(tuple, Bd))
        inter = sum(1 for k in map(tuple, Bs) if k in setd)
        inter_tot += inter; bs_tot += Bs.shape[0]; bd_tot += Bd.shape[0]
        per_frame.append(inter / Bs.shape[0])

        # E4：致密新增的边界体素 -> 最近的稀疏边界体素的距离
        sets = set(map(tuple, Bs))
        new = np.array([k for k in map(tuple, Bd) if k not in sets], np.int64)
        if new.size:
            d, _ = cKDTree(Bs.astype(np.float64)).query(new.astype(np.float64), k=1)
            new_d.append(d * VOXEL)          # 体素单位 -> 米
        nf += 1
        if (fi + 1) % 25 == 0:
            print(f'  {fi+1}/{len(files)}  containment(pooled)={inter_tot/max(bs_tot,1):.4f}', flush=True)

    pf = np.array(per_frame)
    nd = np.concatenate(new_d) if new_d else np.zeros(1)
    res = dict(
        task='A1-supplement (boundary containment)', status='DONE',
        script='review_analysis/A1b_boundary_containment.py',
        frames=nf, seq=a.seq, voxel_size_m=VOXEL,
        boundary_definition=('6-connected; voxel-level majority vote then compare '
                             'neighbour voxel labels; neighbour must be OCCUPIED; '
                             'identical rule on sparse and dense sides'),
        E1_containment_pooled=round(inter_tot / max(bs_tot, 1), 4),
        E2_containment_perframe_mean=round(float(pf.mean()), 4),
        E2_containment_perframe_median=round(float(np.median(pf)), 4),
        E2_containment_perframe_p05=round(float(np.percentile(pf, 5)), 4),
        E3_new_boundary_frac=round((bd_tot - inter_tot) / max(bd_tot, 1), 4),
        E4_dist_new_to_sparse_median_m=round(float(np.median(nd)), 4),
        E4_dist_new_to_sparse_p95_m=round(float(np.percentile(nd, 95)), 4),
        E4_frac_within_one_voxel=round(float((nd <= VOXEL + 1e-9).mean()), 4),
        E5_sparse_boundary_voxels_mean=round(float(np.mean(bs_counts)), 1),
        E6_dense_boundary_voxels_mean=round(float(np.mean(bd_counts)), 1),
        D6_is_pooled_ratio=True,
        asymmetry=dict(
            sparse_voxels_with_empty_neighbour_frac=round(sp_empty_nb / max(sp_vox, 1), 4),
            dense_voxels_with_empty_neighbour_frac=round(dn_empty_nb / max(dn_vox, 1), 4),
            sparse_occupied_voxels_mean=round(sp_vox / max(nf, 1), 1),
            dense_occupied_voxels_mean=round(dn_vox / max(nf, 1), 1)),
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
