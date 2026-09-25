#!/usr/bin/env python3
"""A3 v2 Step 8: Response Fig. R3.

Top block, one panel per statistic: the fidelity test. Each dataset gets the
distribution of real-to-real frame distances next to the distribution of
conditioned-simulation-to-real distances. If the sampler reproduces the sensor,
the two overlap.

Bottom block: the coverage test. (e) every real sensor's measured configuration
against the curriculum's range, (f) the per-frame cloud the curriculum spans
with the real datasets inside it, (g) support coverage per statistic.
"""
import csv
import json
import os.path as osp
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

DS = ['semantickitti', 'nuscenes', 'waymo']
SHORT = {'semantickitti': 'SemanticKITTI', 'nuscenes': 'nuScenes', 'waymo': 'Waymo'}
KEYS = ['f1', 'f2', 'f3_lin', 'f3_plan', 'f3_scat', 'f4']
TITLE = {'f1': 'F1  radial density', 'f2': 'F2  nearest-neighbour spacing',
         'f3_lin': 'F3  linearity', 'f3_plan': 'F3  planarity',
         'f3_scat': 'F3  scattering', 'f4': 'F4  height above ground'}
C_R2R, C_COND, C_CUR = '#4C72B0', '#DD8452', '#937860'


def load_rows():
    return list(csv.DictReader(open('review_analysis/_source_data/A3_frame_level.csv')))


def main():
    rows = load_rows()
    nom = json.load(open('review_analysis/_source_data/A3_nominal_configs.json'))
    cur = nom['curriculum_TableB1']
    cov = {}
    if osp.exists('review_analysis/_source_data/A3_coverage.csv'):
        for r in csv.DictReader(open('review_analysis/_source_data/A3_coverage.csv')):
            cov[(r['dataset'], r['family'], r['level'])] = r
    mats = {d: np.load(f'review_analysis/_source_data/A3_pairmats_{d}.npz') for d in DS
            if osp.exists(f'review_analysis/_source_data/A3_pairmats_{d}.npz')}
    get = {(r['dataset'], r['family'], r['level']): r for r in rows
           if r['metric'] == 'JS'}

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.05], hspace=0.42, wspace=0.30)

    # ---- top block: fidelity -------------------------------------------
    for k, key in enumerate(KEYS):
        ax = fig.add_subplot(gs[k // 3, (k % 3) * 2:(k % 3) * 2 + 2])
        data, pos, cols, labels = [], [], [], []
        for i, d in enumerate(DS):
            if d not in mats:
                continue
            data += [mats[d][f'{key}_r2r'].ravel(), mats[d][f'{key}_cond'].ravel()]
            pos += [i * 3 + 0.6, i * 3 + 1.4]
            cols += [C_R2R, C_COND]
            labels.append(SHORT[d])
        if not data:
            continue
        bp = ax.boxplot(data, positions=pos, widths=0.62, showfliers=False,
                        patch_artist=True, medianprops=dict(color='k', lw=1.4))
        for b, c in zip(bp['boxes'], cols):
            b.set_facecolor(c)
            b.set_alpha(0.75)
        ax.set_yscale('log')
        lo = min(np.percentile(x, 1) for x in data)
        hi = max(np.percentile(x, 99) for x in data)
        ax.set_ylim(lo * 0.55, hi * 4.0)          # headroom for the annotations
        import matplotlib.transforms as mtrans
        blend = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
        for i, d in enumerate(DS):
            r = get.get((d, key, 'cond'))
            if not r:
                continue
            ax.text(i * 3 + 1.0, 0.90,
                    f'{float(r["ratio"]):.1f}×  ·  {float(r["overlap95"]) * 100:.0f}%',
                    ha='center', va='center', fontsize=9, transform=blend,
                    bbox=dict(fc='white', ec='none', alpha=.75, pad=1.2))
        ax.set_xticks([i * 3 + 1.0 for i in range(len(labels))])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('JS divergence, frame pairs', fontsize=9)
        ax.set_title(TITLE[key], fontsize=10.5, weight='bold', pad=7)
        ax.grid(axis='y', alpha=0.25)
    fig.legend(handles=[Patch(fc=C_R2R, alpha=.75, label='real ↔ real  (natural scan-to-scan variability)'),
                        Patch(fc=C_COND, alpha=.75, label='simulated ↔ real  (sampler conditioned on the measured sensor)')],
               loc='upper center', ncol=2, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.995))

    # ---- (e) configuration coverage -------------------------------------
    ax = fig.add_subplot(gs[2, 0:2])
    # The training curriculum is not a fixed range: LiDARCurriculumHook widens
    # it over three stages, and the nuScenes and Waymo runs use different
    # schedules. Each sensor is therefore placed inside ITS OWN run's union,
    # normalised, so one bar can carry all three honestly.
    NU_UNION = {'beams': (16, 96), 'height': (1.2, 2.0), 'hres': (400, 4000),
                'lo': (-45, -10), 'hi': (0, 20)}
    WA_UNION = {'beams': (16, 96), 'height': (0.8, 2.2), 'hres': (400, 4000),
                'lo': (-45, -10), 'hi': (0, 25)}
    UNION = {'semantickitti': NU_UNION, 'nuscenes': NU_UNION, 'waymo': WA_UNION}
    params = [('beams', 'n_beams', 'beams'),
              ('height (m)', 'mount', 'height'),
              ('rays / rev', 'n_horiz', 'hres'),
              ('FOV low (°)', 'fov_lo_deg', 'lo'),
              ('FOV high (°)', 'fov_hi_deg', 'hi')]
    mk = {'semantickitti': 'o', 'nuscenes': 's', 'waymo': '^'}
    for y, (name, field, key) in enumerate(params):
        ax.barh(y, 1.0, left=0.0, height=0.42, color='#BBBBBB', alpha=.55,
                edgecolor='k', lw=.6)
        for d in DS:
            lo, hi = UNION[d][key]
            v = (nom[d]['mount_height']['height_m'] if field == 'mount'
                 else nom[d][field])
            ax.plot((v - lo) / (hi - lo), y, mk[d], ms=8, mec='k', mew=.8,
                    color=C_COND, zorder=3)
        rngs = {UNION[d][key] for d in DS}
        lab = ' / '.join(f'{a:g}' for a, _ in sorted(rngs))
        lab2 = ' / '.join(f'{b:g}' for _, b in sorted(rngs))
        ax.text(-0.02, y, lab, ha='right', va='center', fontsize=8)
        ax.text(1.02, y, lab2, ha='left', va='center', fontsize=8)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels([p[0] for p in params], fontsize=9)
    ax.set_xlim(-0.22, 1.22)
    ax.set_xticks([])
    ax.set_title('(e) every measured sensor lies inside its run\'s curriculum',
                 fontsize=10, weight='bold')
    ax.legend(handles=[plt.Line2D([], [], ls='', marker=mk[d], color=C_COND,
                                  mec='k', label=SHORT[d]) for d in DS],
              fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.04),
              ncol=3, frameon=False)
    ax.set_ylim(-0.75, len(params) - 0.4)

    # ---- (f) per-frame cloud --------------------------------------------
    ax = fig.add_subplot(gs[2, 2:4])
    sc = (np.load('review_analysis/_source_data/A3_coverage_scatter.npz')
          if osp.exists('review_analysis/_source_data/A3_coverage_scatter.npz') else None)
    if sc is not None:
        for d in DS:
            kx, ky = f'{d}_sim_cur_npts', f'{d}_sim_cur_nnmed'
            if kx in sc:
                ax.scatter(sc[kx], sc[ky] * 100, s=9, c=C_CUR, alpha=.30,
                           lw=0, label='curriculum draws' if d == DS[0] else None)
        for d in DS:
            kx, ky = f'{d}_realB_npts', f'{d}_realB_nnmed'
            if kx in sc:
                ax.scatter(sc[kx], sc[ky] * 100, s=16, marker=mk[d], alpha=.85,
                           edgecolor='k', lw=.3, label=SHORT[d])
        ax.set_xscale('log')
        ax.set_xlabel('points per scan', fontsize=9)
        ax.set_ylabel('median nearest-neighbour spacing (cm)', fontsize=9)
        ax.legend(fontsize=8, frameon=False)
    ax.set_title('(f) real scans sit inside the curriculum cloud',
                 fontsize=10, weight='bold')
    ax.grid(alpha=.25)

    # ---- (g) support coverage -------------------------------------------
    ax = fig.add_subplot(gs[2, 4:6])
    if cov:
        wdt = 0.26
        x = np.arange(len(KEYS))
        for i, d in enumerate(DS):
            v = [float(cov[(d, k, 'cur')]['coverage']) * 100 if (d, k, 'cur') in cov
                 else np.nan for k in KEYS]
            e = [[(float(cov[(d, k, 'cur')]['coverage']) -
                   float(cov[(d, k, 'cur')]['ci_lo'])) * 100 for k in KEYS],
                 [(float(cov[(d, k, 'cur')]['ci_hi']) -
                   float(cov[(d, k, 'cur')]['coverage'])) * 100 for k in KEYS]]
            ax.bar(x + (i - 1) * wdt, v, wdt, yerr=e, capsize=2,
                   label=SHORT[d], alpha=.85, edgecolor='k', lw=.5)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace('f3_', 'F3 ').replace('f', 'F')
                            for k in KEYS], fontsize=8, rotation=20)
        ax.set_ylabel('real scans covered (%)', fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, frameon=False)
    ax.set_title('(g) support coverage of the curriculum', fontsize=10, weight='bold')
    ax.grid(axis='y', alpha=.25)

    fig.savefig('figs/R3.pdf', bbox_inches='tight')
    fig.savefig('figs/R3.png', dpi=300, bbox_inches='tight')
    print('wrote figs/R3.pdf and figs/R3.png')


if __name__ == '__main__':
    main()
