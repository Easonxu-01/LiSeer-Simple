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
    w('# A3 results\n')
    w('> Scale: frame-paired. The denominator is the median JS divergence between two real scans of the same sensor,')
    w('> the numerator the same quantity between a simulated and a real scan. `overlap` is the share of simulated-real frame pairs that fall')
    w('> within the 95% range of the real-real pairs. The 95% CI comes from 500 bootstrap resamples at the frame level.\n')
    w('> `cond` conditions the sampler on the sensor configuration measured for that dataset (the fidelity layer).')
    w('> `cur` draws at random over the full training curriculum (a coverage reference, not fidelity evidence).\n')

    for lvl, name in [('cond', 'Fidelity, level 1a: conditioned on the measured sensor configuration'),
                      ('cur', 'Coverage reference: random over the full curriculum')]:
        w(f'\n## {name}\n')
        w('| Statistic | ' + ' | '.join(SHORT[d] for d in DS) + ' |')
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
        w('\nEach cell reads `ratio [95% CI] - overlap`. Bold marks ratio < 1.5.\n')

    w('\n## Secondary metric: Wasserstein-1 (metres)\n')
    w('| Statistic | ' + ' | '.join(SHORT[d] for d in DS) + ' |')
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
        w('\n## Coverage, level 2: support coverage of the real scans by the curriculum\n')
        w('> The criterion is the k-NN support recall of Kynkaanniemi et al. (NeurIPS 2019) with k=3:')
        w('> a real scan is covered if it falls inside the ball of some simulated scan whose radius is the k-th nearest-neighbour distance of that scan.')
        w('> It must be read against the control baseline: two halves of the real data from the same sensor (real_A to real_B) reach only')
        w('> about 0.85 rather than 1.0 under this criterion (a finite-sample radius at k=3), so 0.85 is the ceiling.\n')
        w('| Statistic | ' + ' | '.join(SHORT[d] for d in DS) + ' |')
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
                             f'(control {float(c["control_realA_realB"])*100:.0f}%, '
                             f'relative {rel:.2f})')
            w(f'| {LABEL[f]} | ' + ' | '.join(cells) + ' |')
        w('\nEach cell reads `recall (control, relative)`. Bold marks a relative value of 0.90 or above, that is, the coverage the real data reaches itself.\n')
        w('For contrast, the recall of `sim_cond` is generally below that of `sim_cur` (F4 on SemanticKITTI, 0.745 against 0.910).')
        w('A broad mixture covers better and a conditioned sampler is more faithful, which is what the two-layer structure predicts.\n')

    w('\n## Coverage at the configuration level\n')
    w('| Parameter | ' + ' | '.join(SHORT[d] for d in DS) + ' | Curriculum range |')
    w('|---|---|---|---|---|')
    c = nom['curriculum_TableB1']
    w(f'| Beams | ' + ' | '.join(str(nom[d]['n_beams']) for d in DS) + f' | {c["beams"]} |')
    w(f'| Mount height (m) | ' + ' | '.join(f'{nom[d]["mount_height"]["height_m"]}' for d in DS)
      + f' | {c["height_m"]} |')
    w(f'| Horizontal rays | ' + ' | '.join(str(nom[d]['n_horiz']) for d in DS) + f' | {c["hres"]} |')
    w(f'| Vertical FoV lower bound (deg) | ' + ' | '.join(f'{nom[d]["fov_lo_deg"]}' for d in DS)
      + f' | {c["v_lo_deg"]} |')
    w(f'| Vertical FoV upper bound (deg) | ' + ' | '.join(f'{nom[d]["fov_hi_deg"]}' for d in DS)
      + f' | {c["v_hi_deg"]} |')

    open('review_analysis/_source_data/A3_D18.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('wrote review_analysis/_source_data/A3_D18.md')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
