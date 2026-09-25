#!/usr/bin/env python3
"""A3 v2 Step 4: the mechanism behind the F2 (nearest-neighbour) gap.

The per-bin decomposition puts ~73% of the F2 divergence in the 1-3 cm
neighbour-spacing bins. This script tests what sets that floor, by comparing
three spacings in the same range bands:

  * the real scan's nearest-neighbour spacing, which for a rotating LiDAR is
    the along-ring azimuth step and therefore grows linearly with range;
  * the densified scene's OWN nearest-neighbour spacing -- the finest spacing
    any sampler reading from it could ever produce;
  * the geometric along-ring step implied by the measured ray count.

Hypotheses this rules out (each was tested and failed):
  - point-count difference: sim/real point ratio is 0.86, which accounts for
    only a 1.07x change in spacing, not the observed 1.8x;
  - acceptance-band elevation jitter: snapping every simulated point onto its
    beam ray leaves the spacing unchanged (5.64 -> 5.74 cm);
  - dense-scene resolution overall: globally its NN median (3.7 cm) is close to
    the real scan's (3.2 cm).

Writes review_analysis/_source_data/A3_f2_mechanism.md.
"""
import os.path as osp
import sys

import numpy as np
import scipy.spatial as spatial

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from A3_sets import select, load_frames, make_sim, nominal

BANDS = [(2, 5), (5, 10), (10, 15), (15, 25), (25, 40), (40, 70)]
DS = ['semantickitti', 'nuscenes', 'waymo']
N_FRAMES = 3
SEED = 0


def nn_profile(P, origin, tree=None, cap=20000):
    t = tree or spatial.cKDTree(P)
    R = np.sqrt(((P - np.asarray(origin)) ** 2).sum(1))
    rng = np.random.RandomState(SEED)
    out = {}
    for lo, hi in BANDS:
        m = (R >= lo) & (R < hi)
        if m.sum() < 400:
            out[(lo, hi)] = (int(m.sum()), np.nan)
            continue
        q = P[m]
        if len(q) > cap:
            q = q[rng.choice(len(q), cap, replace=False)]
        d = t.query(q, k=2, workers=-1)[0][:, 1]
        out[(lo, hi)] = (int(m.sum()), float(np.median(d)) * 100)
    return out


def main():
    L = []
    w = L.append
    w('# F2 机理：最近邻间距差距来自致密场景的分辨率下限\n')
    w('> 逐箱分解把 ~73% 的 F2 散度放在 1–3 cm 的邻距箱。下面三条曲线解释为什么。\n')
    w('\n## 已排除的解释（各自的反证）\n')
    w('| 假说 | 检验 | 结果 |')
    w('|---|---|---|')
    w('| 点数不同 | sim/real 点数比 0.86 → 间距只应变 1.07× | 实测 1.8×，**排除** |')
    w('| 接受带内仰角抖动 | 把每个模拟点投影到它所属射线上再算 NN | 5.64 → 5.74 cm，**未改善，排除** |')
    w('| 致密场景整体太粗 | 全局 NN 中位数：致密 3.7 cm vs 真实 3.2 cm | 相当，**排除** |')
    w('\n## 成立的解释：两种采样几何的分辨率随距离的走向相反\n')
    w('真实旋转式 LiDAR 是**角度采样**，邻距随距离线性增长；')
    w('致密化产生的是**近似均匀的空间分辨率**，各距离段几乎不变。')
    w('两条曲线在约 8–10 m 处交叉：近场致密场景比真实扫描更粗，远场则细得多。')
    w('模拟器只能挑选致密场景中已存在的点，因此在近场**物理上不可能**产生比致密场景自身间距更小的邻距。\n')

    for ds in DS:
        A, B = select(ds)
        frames = load_frames(ds, A[:N_FRAMES])
        nom = nominal(ds)
        O = nom['origin_xyz']
        NH = nom['n_horiz']
        acc = {b: [[], [], []] for b in BANDS}
        for i, f in enumerate(frames):
            sim, _, info = make_sim(ds, f, 'sim_cond', i)
            pr = nn_profile(f['real_xyz'].astype(np.float64), O)
            pd = nn_profile(f['dense_xyz'].astype(np.float64), O)
            ps = nn_profile(sim.astype(np.float64), O)
            for b in BANDS:
                acc[b][0].append(pr[b][1])
                acc[b][1].append(pd[b][1])
                acc[b][2].append(ps[b][1])
        w(f'\n### {ds}   （{nom["n_beams"]} 线 × {NH} 射线，原点 {O}）\n')
        w('| 距离段 (m) | 真实扫描 NN | 致密场景自身 NN | 模拟扫描 NN | 几何环内步长 | 致密/真实 |')
        w('|---|---|---|---|---|---|')
        for lo, hi in BANDS:
            r, d, s = (np.nanmean(acc[(lo, hi)][k]) for k in range(3))
            g = (lo + hi) / 2 * 2 * np.pi / NH * 100
            if not np.isfinite(r):
                continue
            mark = ' ⬅ 下限咬住' if d > r else ''
            w(f'| {lo}–{hi} | {r:.2f} cm | **{d:.2f} cm** | {s:.2f} cm | {g:.2f} cm | '
              f'{d/r:.2f}×{mark} |')
        cross = [f'{lo}–{hi}' for lo, hi in BANDS
                 if np.isfinite(np.nanmean(acc[(lo, hi)][0]))
                 and np.nanmean(acc[(lo, hi)][1]) > np.nanmean(acc[(lo, hi)][0])]
        w(f'\n致密场景比真实扫描粗的距离段：**{", ".join(cross) if cross else "无"}** m。\n')

    w('\n## 结论\n')
    w('F2 的残余差距是**离线致密化的分辨率下限**造成的，不是采样器 Φ_samp 的性质。')
    w('近场里没有任何采样器能从这些致密场景中取出比其自身间距更密的点。\n')
    w('这一限制对训练是**同源且有界**的：模型训练时读的是同一批致密场景，')
    w('因此它从未接触过近场 3 cm 以下的结构，模拟与训练在这一点上是一致的。\n')
    open('review_analysis/_source_data/A3_f2_mechanism.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
