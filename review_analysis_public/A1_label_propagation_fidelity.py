#!/usr/bin/env python3
"""A1 标签传播语义保真（→ Fig R1, D1-D7）。

方法（严格复刻 tools/generate_dense_points_labels_sk.py 的传播算法）：
  1. 取 N 帧真实稀疏扫描的标注点；随机留出 20%，只用剩余 80% 做传播；
  2. 在留出的 20% 点上评传播标签 vs 真值 → 语义保真度（准确率、逐类 IoU）；
  3. 边界体素膨胀比：稠密标签在类别边界处的体素数 / 稀疏标签的对应量。

留出点必须用"传播到它所在体素的标签"来评——与稠密点走的是同一条路径，
所以这测的正是传播算法本身的误差，不掺别的因素。
"""
import argparse, json, os, sys
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree

LEARNING_MAP = None  # 见 --map，用配置里的 labels_map
VOXEL = 0.2


def voxelize(p, v=VOXEL):
    return np.floor(p / v).astype(np.int32)


def build_voxel_labels(pts, lab, v=VOXEL):
    """稀疏点 -> 体素多数投票标签（与 generate_dense_labels 一致，向量化实现）。

    源码用 dict + Counter 逐点循环；这里用 (voxel,label) 联合唯一化 + 计数取最大，
    结果等价但快两个数量级（200 帧 × 12 万点的规模下逐点 Python 循环不可行）。
    并列时取标签值较小者——Counter.most_common 在并列时依插入序，
    对本分析（准确率/IoU 统计）无影响。
    """
    vi = voxelize(pts, v)
    keys, inv = np.unique(vi, axis=0, return_inverse=True)
    pair = np.stack([inv, lab.astype(np.int64)], 1)
    upair, cnt = np.unique(pair, axis=0, return_counts=True)
    # 按 (voxel, count) 排序后取每个 voxel 的最后一条 = 计数最大者
    order = np.lexsort((upair[:, 1], cnt, upair[:, 0]))
    up, cn = upair[order], cnt[order]
    last = np.ones(up.shape[0], bool)
    last[:-1] = up[1:, 0] != up[:-1, 0]
    win = up[last]
    out_lab = np.zeros(keys.shape[0], np.int64)
    out_lab[win[:, 0]] = win[:, 1]
    return keys, out_lab


def propagate(vox_keys, vox_lab, query_pts, v=VOXEL):
    """查询点 -> 传播标签：本体素有标签就用，否则最近已填体素（与源码一致）。"""
    qi = voxelize(query_pts, v)
    tree = cKDTree(vox_keys)
    # 精确命中：距离为 0
    dist, idx = tree.query(qi, k=1)
    return vox_lab[idx], dist > 0


def boundary_voxels(pts, lab, v=VOXEL):
    """边界体素数：6-邻域内存在不同标签的体素。"""
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
        # 只用有标注的点
        m = lab > 0
        pts, lab = pts[m], lab[m]
        if pts.shape[0] < 100:
            continue

        # --- 80/20 留出 ---
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

        # --- 边界体素膨胀比：稠密 vs 稀疏 ---
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
            print(f'  {fi+1}/{len(files)} 帧  当前准确率 {cor/max(tot,1):.4f}', flush=True)

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
    print(f'\n写入 {a.out}')


if __name__ == '__main__':
    main()
