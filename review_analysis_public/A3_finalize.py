#!/usr/bin/env python3
"""A3 v2 Step 8: final tables and the D18 summary.

Archives the pooled-scale results (kept for audit, not quoted) and writes the
frame-level ones as review_analysis/_source_data/A3_simreal_metrics.csv.
"""
import csv
import json
import os
import os.path as osp
import shutil

FAM = ['f1', 'f2', 'f3_lin', 'f3_plan', 'f3_scat', 'f4']
LABEL = {'f1': 'F1 radial density', 'f2': 'F2 nearest-neighbour spacing',
         'f3_lin': 'F3 linearity', 'f3_plan': 'F3 planarity',
         'f3_scat': 'F3 scattering', 'f4': 'F4 height'}
DS = ['semantickitti', 'nuscenes', 'waymo']
SHORT = {'semantickitti': 'SemanticKITTI', 'nuscenes': 'nuScenes', 'waymo': 'Waymo'}


def main():
    old = 'review_analysis/_source_data/A3_simreal_metrics.csv'
    if osp.exists(old) and not osp.exists('review_analysis/_source_data/A3_simreal_metrics.POOLED_v3.csv'):
        shutil.copy(old, 'review_analysis/_source_data/A3_simreal_metrics.POOLED_v3.csv')
        print('archived pooled results -> review_analysis/_source_data/A3_simreal_metrics.POOLED_v3.csv')
    shutil.copy('review_analysis/_source_data/A3_frame_level.csv', old)
    print(f'wrote {old}')

    rows = [r for r in csv.DictReader(open(old))]
    js = {(r['dataset'], r['family'], r['level']): r for r in rows if r['metric'] == 'JS'}
    w1 = {(r['dataset'], r['family'], r['level']): r for r in rows if r['metric'] == 'W1'}
    cov = {}
    if osp.exists('review_analysis/_source_data/A3_coverage.csv'):
        for r in csv.DictReader(open('review_analysis/_source_data/A3_coverage.csv')):
            cov[(r['dataset'], r['family'], r['level'])] = r
    nom = json.load(open('review_analysis/_source_data/A3_nominal_configs.json'))

    L = []
    w = L.append
    w('# A3 结果\n')
    w('> 尺度：**逐帧配对**。分母是同一传感器两次真实扫描之间的 JS 分布的中位数，')
    w('> 分子是模拟扫描与真实扫描之间的同一量。`overlap` = 模拟↔真实帧对中落在')
    w('> 真实↔真实帧对 95% 范围内的比例。95% CI 为帧层面 bootstrap 500 次。\n')
    w('> `cond` = 采样器条件到该数据集实测的传感器配置（保真层）。')
    w('> `cur` = 训练课程的全范围随机抽取（覆盖层参照，不作为保真证据）。\n')

    for lvl, name in [('cond', '保真层 Level 1a：条件到实测传感器配置'),
                      ('cur', '覆盖层参照：课程全范围随机')]:
        w(f'\n## {name}\n')
        w('| 统计量 | ' + ' | '.join(SHORT[d] for d in DS) + ' |')
        w('|---|---|---|---|')
        for f in FAM:
            cells = []
            for d in DS:
                r = js.get((d, f, lvl))
                if not r:
                    cells.append('—')
                    continue
                rt = float(r['ratio'])
                mark = '**' if rt < 1.5 else ''
                cells.append(f'{mark}{rt:.2f}{mark} [{float(r["ratio_ci_lo"]):.2f}, '
                             f'{float(r["ratio_ci_hi"]):.2f}] · {float(r["overlap95"])*100:.0f}%')
            w(f'| {LABEL[f]} | ' + ' | '.join(cells) + ' |')
        w('\n每格：`ratio [95% CI] · overlap`。粗体 = ratio < 1.5。\n')

    w('\n## 辅指标：Wasserstein-1（米）\n')
    w('| 统计量 | ' + ' | '.join(SHORT[d] for d in DS) + ' |')
    w('|---|---|---|---|')
    for f in ['f1', 'f2', 'f4']:
        cells = []
        for d in DS:
            r = w1.get((d, f, 'cond'))
            cells.append('—' if not r else
                         f'r2r {float(r["median_r2r"]):.3f} / s2r {float(r["median_s2r"]):.3f} '
                         f'→ {float(r["ratio"]):.2f}×')
        w(f'| {LABEL[f]} | ' + ' | '.join(cells) + ' |')

    if cov:
        w('\n## 覆盖层 Level 2：课程对真实扫描的支持度覆盖\n')
        w('> 判据为 Kynkaanniemi et al. (NeurIPS 2019) 的 k-NN 支持度 recall（k=3）：')
        w('> 真实扫描被覆盖 ⟺ 它落在某个模拟扫描以其自身第 k 近邻距离为半径的球内。')
        w('> **必须对照控制基线读**：同一传感器的两半真实数据（real_A→real_B）在这个判据下')
        w('> 只得约 0.85 而非 1.0（k=3 的有限样本半径），所以 0.85 才是天花板。\n')
        w('| 统计量 | ' + ' | '.join(SHORT[d] for d in DS) + ' |')
        w('|---|---|---|---|')
        for f in FAM:
            cells = []
            for d in DS:
                c = cov.get((d, f, 'cur'))
                if not c:
                    cells.append('—')
                    continue
                rel = float(c.get('coverage_vs_control', 0))
                mark = '**' if rel >= 0.9 else ''
                cells.append(f'{mark}{float(c["coverage"])*100:.0f}%{mark} '
                             f'(控制 {float(c["control_realA_realB"])*100:.0f}%, '
                             f'相对 {rel:.2f})')
            w(f'| {LABEL[f]} | ' + ' | '.join(cells) + ' |')
        w('\n每格：`recall (控制基线, 相对值)`。粗体 = 相对控制 ≥ 0.90，即达到真实数据自身的覆盖水平。\n')
        w('对照：`sim_cond` 的 recall 普遍**低于** `sim_cur`（例如 SK 的 F4 0.745 vs 0.910）。')
        w('宽混合覆盖得更好、条件化保真得更好——两层结构在数据上自洽。\n')

    w('\n## 配置层面的覆盖\n')
    w('| 参数 | ' + ' | '.join(SHORT[d] for d in DS) + ' | 课程范围 |')
    w('|---|---|---|---|---|')
    c = nom['curriculum_TableB1']
    w(f'| 线数 | ' + ' | '.join(str(nom[d]['n_beams']) for d in DS) + f' | {c["beams"]} |')
    w(f'| 安装高度 (m) | ' + ' | '.join(f'{nom[d]["mount_height"]["height_m"]}' for d in DS)
      + f' | {c["height_m"]} |')
    w(f'| 水平射线数 | ' + ' | '.join(str(nom[d]['n_horiz']) for d in DS) + f' | {c["hres"]} |')
    w(f'| 垂直下界角 (°) | ' + ' | '.join(f'{nom[d]["fov_lo_deg"]}' for d in DS)
      + f' | {c["v_lo_deg"]} |')
    w(f'| 垂直上界角 (°) | ' + ' | '.join(f'{nom[d]["fov_hi_deg"]}' for d in DS)
      + f' | {c["v_hi_deg"]} |')

    open('review_analysis/_source_data/A3_D18.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('wrote review_analysis/_source_data/A3_D18.md')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
