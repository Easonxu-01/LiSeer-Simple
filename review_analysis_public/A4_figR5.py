"""A4: Response Fig. R5 - output-level sensor invariance (Table R9's figure).

Reads review_analysis/_source_data/A4_raw.json (written by review_analysis/A4_consistency.py) and renders
the two LiSeer checkpoints side by side.  No numbers are computed here; this
file only draws what a4_consistency.py measured.

Visual system matches review_analysis/A6_figR4_multi.py: surface #fcfcfb, the
sequential blue ramp RAMP, ink #0b0b0b / #52514e, accent #eb6834.  Two series =
two categorical slots (RAMP[5] blue, ACCENT orange); the pair was checked for
CVD separation (worst adjacent dE 25.5 protan, 37.4 normal - passes).
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from review_analysis.A6_figR4 import RAMP, INK, INK2, INK3, SURFACE, ACCENT  # noqa: E402

SERIES = [RAMP[5], ACCENT]          # fixed order, never cycled
GRID = '#ececE8'
SPINE = '#dcdbd6'


def style(ax, xlabel='', ylabel='', title=''):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5, colors=INK2, length=3)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)
        sp.set_linewidth(0.8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=INK2)
    if title:
        ax.set_title(title, fontsize=10.5, color=INK, pad=8, fontweight='medium')


def panel_invariance(ax, names, ms, raw):
    """(a) The headline: cross-sensor vs same-sensor agreement, per checkpoint.

    Both bars are scored on the SHARED support (the points that qualify in both
    sweeps), so the only difference between them is whether the sensor varies.
    """
    groups = ['Unanimous\nacross views', 'Mean modal\nvote share']
    x = np.arange(len(groups))
    w = 0.19
    for j, (nm, m) in enumerate(zip(names, ms)):
        c = m.get('control_same_config')
        vary = ([c['varying_on_matched_support_agreement_pct'],
                 c['varying_on_matched_support_modal_vote_share_pct']] if c
                else [m['agreement_pct'], m['mean_modal_vote_share_pct']])
        ctrl = ([c['matched_support_agreement_pct'],
                 c['matched_support_modal_vote_share_pct']] if c else None)
        off = (j - 0.5) * 2 * w
        ax.bar(x + off - w * 0.52, vary, w, color=SERIES[j], zorder=3,
               edgecolor=SURFACE, linewidth=1.6)
        if ctrl:
            ax.bar(x + off + w * 0.52, ctrl, w, color=SERIES[j], zorder=3,
                   alpha=0.34, edgecolor=SERIES[j], linewidth=1.0, hatch='///')
        for k, v in enumerate(vary):
            ax.text(x[k] + off - w * 0.52, v + 1.1, f'{v:.1f}', ha='center',
                    va='bottom', fontsize=7.6, color=INK2)
        if ctrl:
            for k, v in enumerate(ctrl):
                ax.text(x[k] + off + w * 0.52, v + 1.1, f'{v:.1f}', ha='center',
                        va='bottom', fontsize=7.6, color=INK3)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8.8, color=INK2)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    style(ax, ylabel='% of shared-support points',
          title='(a)  Prediction is stable when the sensor changes')
    c0 = ms[0].get('control_same_config')
    n = (f'{c0["matched_support_qualifying_points"]:,} shared points  ·  '
         if c0 else '')
    ax.text(0.5, -0.185,
            f'{n}solid = 50 different sensors   ·   hatched = 1 sensor, 50 repeats',
            transform=ax.transAxes, ha='center', va='top', fontsize=8,
            color=INK3)


def panel_threshold(ax, names, ms, raw):
    """(b) Is the >=10 threshold cherry-picked?  Sweep it."""
    kmark = raw['min_obs']
    sw0 = [s for s in ms[0]['threshold_sweep'] if s['n'] > 0]
    # Where the surviving point count collapses, the curve is a small-sample
    # artefact (only points every configuration happened to hit) - shade it so
    # nobody reads the upturn as a real effect.
    thin = [s['k'] for s in sw0 if s['n'] < 0.02 * raw['_qual']]
    if thin:
        ax.axvspan(min(thin) - 0.5, raw['n_configs'] + 1, color='#f0eeea',
                   zorder=1, lw=0)
        ax.text(min(thin) + 0.7, 0.955, '< 2% of the\npoints remain',
                fontsize=7.4, color=INK3, ha='left', va='top',
                transform=ax.get_xaxis_transform(), zorder=7)
    for j, (nm, m) in enumerate(zip(names, ms)):
        sw = [s for s in m['threshold_sweep'] if s['n'] > 0]
        k = [s['k'] for s in sw]
        a = [s['agreement_pct'] for s in sw]
        ax.plot(k, a, '-', lw=2.0, color=SERIES[j], zorder=4, label=nm)
        at = next(s for s in sw if s['k'] == kmark)
        ax.plot([kmark], [at['agreement_pct']], 'o', ms=7, mfc=SERIES[j],
                mec=SURFACE, mew=1.6, zorder=6)
    ax.axvline(kmark, color=INK3, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
    # The dashed rule is the visual link, so the label needs no leader line.
    ax.text(kmark + 0.9, 0.945,
            f'reported threshold\n$k$ = {kmark}  ({raw["_qual"]:,} points)',
            transform=ax.get_xaxis_transform(), fontsize=8, color=INK2,
            ha='left', va='top', zorder=7)
    style(ax, xlabel='minimum number of observing configurations, $k$',
          ylabel='unanimous predictions (%)',
          title='(b)  Agreement is not an artefact of the threshold')
    ax.set_xlim(0, raw['n_configs'] + 1)


def panel_perconfig(ax, names, ms, raw):
    """(c) Every single configuration, not just the aggregate.

    Grouped by measure, coloured by checkpoint (identity comes from the figure
    legend), so no extra axis labels are needed under the ticks.
    """
    rng = np.random.default_rng(0)
    for j, (nm, m) in enumerate(zip(names, ms)):
        series = [np.asarray(m['per_config_consensus_agreement']),
                  np.asarray(m['per_config_acc'])]
        for g, vals in enumerate(series):
            xc = g * 1.5 + (j - 0.5) * 0.5
            jit = rng.uniform(-0.14, 0.14, len(vals))
            ax.plot(xc + jit, vals, 'o', ms=3.6, mfc=SERIES[j], mec='none',
                    alpha=0.7, zorder=3)
            ax.plot([xc - 0.26, xc + 0.26], [vals.mean()] * 2, '-', lw=2.4,
                    color=SERIES[j], zorder=5, solid_capstyle='round')
            ax.text(xc, 102.0, f'{vals.mean():.1f}', ha='center', va='bottom',
                    fontsize=8, color=SERIES[j])
    ax.set_xticks([0.0, 1.5])
    ax.set_xticklabels(['agreement with the\ncross-sensor consensus',
                        'accuracy vs\ndense-reconstruction GT'],
                       fontsize=8.6, color=INK2)
    ax.set_ylim(45, 110)
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.set_xlim(-0.62, 2.12)
    style(ax, ylabel='% (one dot = one configuration)',
          title=f'(c)  All {raw["n_configs"]} configurations, individually')


def panel_vs_beams(ax, names, ms, raw):
    """(d) Does invariance depend on where you are in the sensor space?

    A mild positive trend exists and is reported as measured; what matters is
    its size - a couple of points across the entire 32-64 beam range.
    """
    beams = np.asarray([c['beams'] for c in raw['configs']], dtype=float)
    lo = raw['sensor_ranges']['beams']
    for j, (nm, m) in enumerate(zip(names, ms)):
        con = np.asarray(m['per_config_consensus_agreement'])
        ax.plot(beams, con, 'o', ms=4.8, mfc=SERIES[j], mec=SURFACE, mew=0.7,
                alpha=0.9, zorder=4, label=nm)
        b = np.polyfit(beams, con, 1)
        xs = np.linspace(lo[0], lo[1], 50)
        ax.plot(xs, np.polyval(b, xs), '-', lw=1.6, color=SERIES[j],
                alpha=0.5, zorder=3)
        r = float(np.corrcoef(beams, con)[0, 1])
        span = b[0] * (lo[1] - lo[0])
        ax.text(0.975, 0.125 - 0.082 * j,
                f'{nm}:  {span:+.1f} pp across {lo[0]}-{lo[1]} beams  '
                f'($r$ = {r:+.2f})', transform=ax.transAxes, ha='right',
                fontsize=8, color=SERIES[j], zorder=7,
                bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.8,
                          alpha=0.88))
    style(ax, xlabel='number of beams in the configuration',
          ylabel='agreement with the cross-sensor consensus (%)',
          title='(d)  Agreement barely moves across the sensor envelope')
    ax.set_xlim(lo[0] - 1.5, lo[1] + 1.5)


def render_main(raw, names, ms, args):
    fig = plt.figure(figsize=(13.2, 9.9), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.235,
                          left=0.062, right=0.985, top=0.850, bottom=0.098)
    raw['_qual'] = ms[0]['qualifying_points']
    panel_invariance(fig.add_subplot(gs[0, 0]), names, ms, raw)
    panel_threshold(fig.add_subplot(gs[0, 1]), names, ms, raw)
    panel_perconfig(fig.add_subplot(gs[1, 0]), names, ms, raw)
    panel_vs_beams(fig.add_subplot(gs[1, 1]), names, ms, raw)

    handles = [Patch(facecolor=SERIES[j], edgecolor='none', label=nm)
               for j, nm in enumerate(names)]
    handles.append(Patch(facecolor='#e8e6e2', edgecolor=INK3, hatch='///',
                         linewidth=0.8, label='same-sensor control  (panel a)'))
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=9.5,
               frameon=False, labelcolor=INK2, handletextpad=0.7,
               columnspacing=2.6, bbox_to_anchor=(0.5, 0.912))

    fig.suptitle('One scene, 50 random LiDAR configurations: the semantic '
                 'prediction does not follow the sensor',
                 fontsize=15, color=INK, y=0.975, fontweight='medium')
    fig.text(0.5, 0.934,
             f'SemanticKITTI {raw["frame_id"].split("/", 1)[1]}  ·  '
             f'{raw["dense_points"]:,} dense points, '
             f'{ms[0]["qualifying_points"]:,} observed by '
             f'$\\geq$ {raw["min_obs"]} of the {raw["n_configs"]} configurations  ·  '
             f'beams {min(c["beams"] for c in raw["configs"])}'
             f'–{max(c["beams"] for c in raw["configs"])}  ·  '
             + ('inference only, identical frame and identical configurations '
                'for every checkpoint' if len(ms) > 1 else
                'inference only'),
             ha='center', fontsize=9.2, color=INK3)

    fig.text(0.5, 0.020,
             'Accuracy is scored against the dense reconstruction\'s propagated '
             'labels, which are noisier than SemanticKITTI\'s per-scan '
             'annotations - these values are NOT comparable to the paper\'s mIoU. '
             'The agreement statistics are label-free and unaffected.',
             ha='center', fontsize=7.8, color=INK3, style='italic')
    save(fig, args, 'FigR5_consistency')


def render_perclass(raw, names, ms, args):
    fig, ax = plt.subplots(figsize=(8.6, 5.4), facecolor=SURFACE)
    classes = [c for c in raw['class_names'] if c != 'Other']
    y = np.arange(len(classes))
    h = 0.36
    for j, (nm, m) in enumerate(zip(names, ms)):
        vals = [m['per_class'][c]['agreement_pct'] or 0.0 for c in classes]
        ax.barh(y + (j - 0.5) * h, vals, h * 0.9, color=SERIES[j], zorder=3,
                edgecolor=SURFACE, linewidth=1.4, label=nm)
        for k, v in enumerate(vals):
            ax.text(v + 0.9, y[k] + (j - 0.5) * h, f'{v:.1f}', va='center',
                    ha='left', fontsize=7.8, color=INK2)
    ns = [m['per_class'][c]['n'] for c in classes]
    ax.set_yticks(y)
    ax.set_yticklabels([f'{c}\n({n:,} pts)' for c, n in zip(classes, ns)],
                       fontsize=8.4, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    style(ax, xlabel='unanimous across all observing configurations (%)',
          title='Fig. R5b  Per-class output invariance')
    ax.legend(fontsize=9, frameon=False, labelcolor=INK2, loc='lower right')
    fig.tight_layout()
    save(fig, args, 'FigR5_perclass')


def save(fig, args, stem):
    os.makedirs(args.outdir, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = f'{args.outdir}/{stem}.{ext}'
        fig.savefig(p, dpi=args.dpi, facecolor=SURFACE, bbox_inches='tight')
        print(f'[fig] saved {p}')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='review_analysis/_source_data/A4_raw.json')
    ap.add_argument('--outdir', default='figs/a4')
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--label', default='LiSeer',
                    help='name shown in the figures. Defaults to the method '
                         'name; never use the checkpoint filename, which '
                         'identifies an unreleased model')
    args = ap.parse_args()

    raw = json.load(open(args.raw))
    ckpts = list(raw['checkpoints'])
    ms = [raw['checkpoints'][n] for n in ckpts]
    # Figures are published artefacts: they must not carry internal checkpoint
    # filenames, which name unreleased models. Show the METHOD instead. With a
    # single checkpoint that is simply "LiSeer"; were several ever plotted they
    # would need distinguishing labels that are still not the file names.
    names = ([args.label] if len(ckpts) == 1
             else [f'{args.label} ({i + 1})' for i in range(len(ckpts))])
    print(f'[fig] {len(ckpts)} checkpoints: {ckpts} -> labelled {names}')
    render_main(raw, names, ms, args)
    render_perclass(raw, names, ms, args)


if __name__ == '__main__':
    main()
