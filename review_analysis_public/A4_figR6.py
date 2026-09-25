"""Response Fig. R6 -- 50 sensor configurations observing one scene.

A figure-only redesign of the point-cloud sheets. It reads the SAME cached
render as the published version (`beam_concentration=2.6`) and re-draws it:
no inference, no resampling, no change to configuration order, pairing,
field of view, or class definitions.

The design brief (2026-09-11) is: strip every explanatory sentence off the
canvas -- those move to the LaTeX caption -- and make the one claim the
figure exists to support visible at a glance: *the 50 scans differ enormously,
the predicted semantics barely move*.

Three changes do most of that work:

  * draw order by class frequency, rarest last, so Living Being / Vehicle /
    Objects are never buried under the 44%-of-points Nature class;
  * a palette that pushes the four background classes down in saturation and
    pulls the three safety-critical foreground classes up;
  * marker size that adapts to scan density, so a 7.5k-point 16-beam scan and
    a 98k-point 127-beam scan carry comparable ink.

Usage (no GPU needed -- pure re-render from cache):

    python review_analysis/A4_figR6.py \
        --cache review_analysis/_source_data/a4_vis_bc2p6_LiSeer_SK.npz \
        --outdir figs/a4/R6
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Circle, Rectangle                  # noqa: E402

NUM_CLASSES = 8
IGNORE_LABEL = 0

# Visual system: identical to the first R6 version / the published sheets.
# The 2026-09-11 redesign moved these; that change was withdrawn on
# 2026-09-12 and everything below is the original again.
INK, INK2, INK3 = '#0b0b0b', '#52514e', '#8a8984'
SURFACE = '#ffffff'
PANEL_BG = '#ffffff'         # v1: panel interior is the page white
PANEL_EDGE = '#dcdbd6'       # v1 SPINE
RING = '#e3e2de'             # v1 RING

CLASS_NAMES_FIG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle',
                   'Nature', 'Living Being', 'Objects', 'Other Ground']

# ---------------------------------------------------------------------------
# COLA-8 class colours -- the project's own map, byte-for-byte what the
# manuscript, the appendix and the on-vehicle results use, including the
# `[::-1]` channel reversal the repository's plotting code applies before
# handing values to matplotlib. That reversal is the house convention; it is
# kept exactly as written and NOT "corrected" to plain RGB.
# ---------------------------------------------------------------------------
COLOR_MAP_DG_ORI = {
    0: [0, 0, 0],
    1: [250, 217, 213],
    2: [117, 166, 240],
    3: [161, 233, 157],
    4: [250, 206, 160],
    5: [255, 242, 204],
    6: [186, 200, 211],
    7: [225, 213, 231],
}
ACTIVE_MAP = {k: np.array(v[::-1], dtype=np.float32) / 255.0
              for k, v in COLOR_MAP_DG_ORI.items()}

LIM = 51.5
RINGS_R6 = (10, 20, 30, 40, 50)      # v1 ring set


def label_colors(labels):
    lab = np.asarray(labels)
    out = np.empty((lab.shape[0], 3), dtype=np.float32)
    for u in np.unique(lab):
        out[lab == u] = ACTIVE_MAP.get(int(u), np.float32([0.5, 0.5, 0.5]))
    return out


def rare_class_order(per_config):
    """Class indices big-first, so a frequency-ordered scatter leaves the
    rare classes on top. Only used when --rare-on-top is passed; the default
    path draws by height exactly as v1 does."""
    tot = np.zeros(NUM_CLASSES, dtype=np.int64)
    for rec in per_config:
        tot += np.bincount(np.asarray(rec['pred']).astype(int),
                           minlength=NUM_CLASSES)
    return list(np.argsort(-tot)), tot


def bev_panel(ax, xyz, labels, s, origin_ms=4.4, rings=True, draw_order=None):
    """One BEV panel, rendered exactly as the first R6 version renders it.

    Default painter's order is by height, so structure and vehicles are not
    buried under the road surface drawn after them -- a draw-order choice
    only, no point is dropped. `draw_order` (from --rare-on-top) swaps that
    for a by-frequency order; it changes which class wins an overlap, so it
    is off unless asked for.
    """
    ax.set_facecolor(PANEL_BG)
    if rings:
        for r in RINGS_R6:
            ax.add_patch(Circle((0, 0), r, fill=False, ec=RING, lw=0.6,
                                zorder=0))
    if len(xyz):
        lab = np.asarray(labels).astype(int)
        if draw_order is None:
            o = np.argsort(xyz[:, 2], kind='stable')          # v1: by height
            ax.scatter(xyz[o, 0], xyz[o, 1], s=s, c=label_colors(lab[o]),
                       linewidths=0, zorder=2, rasterized=True)
        else:
            for rank, c in enumerate(draw_order):
                m = lab == c
                if not m.any():
                    continue
                ax.scatter(xyz[m, 0], xyz[m, 1], s=s,
                           c=ACTIVE_MAP.get(int(c),
                                            np.float32([0.5, 0.5, 0.5]))[None, :],
                           linewidths=0, zorder=2 + rank, rasterized=True)
    # v1 origin marker: ink, never a class colour, so it cannot be misread
    # as a semantic label
    ax.plot(0, 0, marker='P', ms=origin_ms, mfc=INK, mec='white',
            mew=max(0.35, origin_ms * 0.16), zorder=20, ls='none')
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(PANEL_EDGE)
        sp.set_linewidth(0.8)


def legend_band(fig, fx, fy, x0, y_in, width_in, H, band_h, fontsize=7.0):
    """Horizontal legend strip, identical on all three sheets.

    Two rows of four: eight class names on a single row collide at any font
    size that still satisfies the 7 pt floor. The "left = prediction" note
    lives in the per-column headers instead of here, which frees the whole
    band width for the class names.
    """
    ax = fig.add_axes([fx(x0), fy(y_in + band_h), fx(width_in), band_h / H])
    ax.set_facecolor(SURFACE)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Legend order follows the class index, not the draw order: the reader is
    # looking things up by name/number here, not by prominence.
    per_row = 4
    xs = np.linspace(0.0, 0.76, per_row + 1)[:per_row]
    for c in range(NUM_CLASSES):
        row, col = divmod(c, per_row)
        yc = 0.72 - row * 0.44
        ax.add_patch(Rectangle((xs[col], yc - 0.15), 0.012, 0.30,
                               fc=ACTIVE_MAP.get(c, [0.5, 0.5, 0.5]),
                               ec=PANEL_EDGE, lw=0.5,
                               transform=ax.transAxes, clip_on=False))
        lbl = CLASS_NAMES_FIG[c] + (' (ignore)' if c == IGNORE_LABEL else '')
        ax.text(xs[col] + 0.019, yc, lbl, fontsize=fontsize, color=INK2,
                va='center', ha='left')
    return ax


def render_sheet(per_config, sel_idx, draw_order, args, meta, out_stem,
                 lead_dense=None):
    """One sheet. Layout is identical across sheets except for sheet 1's
    first cell, which holds the dense reference instead of a pair."""
    npair = 3
    W = args.width
    ml = mr = 0.22

    content_w = W - ml - mr
    g_in = 0.02                     # inside a pair: panels nearly touch
    g_pair = 0.18                   # between pairs: clear separation
    panel = (content_w - (npair - 1) * g_pair - npair * g_in) / (2 * npair)

    # Two caption lines: the single-line form specified in the brief does not
    # fit a 2.3-inch pair cell at the 5.5 pt floor -- adjacent captions
    # overlapped. Split per the brief's fallback: "#id + beams" on line 1.
    cap_h = 0.30
    row_gap = 0.06
    legend_h = 0.40
    colhdr_h = 0.14                 # "prediction | inherited" over row 1

    n_cells = len(sel_idx) + (1 if lead_dense is not None else 0)
    nrows = int(np.ceil(n_cells / npair))

    top_pad = 0.08
    bottom_pad = 0.10
    fixed = top_pad + legend_h + colhdr_h + bottom_pad
    grid_h = nrows * (panel + cap_h) + (nrows - 1) * row_gap
    H = fixed + grid_h
    if H > args.max_height:
        # Shrink the panels (not the type) until the sheet fits the page box.
        avail = args.max_height - fixed - nrows * cap_h - (nrows - 1) * row_gap
        panel = avail / nrows
        grid_h = nrows * (panel + cap_h) + (nrows - 1) * row_gap
        H = fixed + grid_h
        content_w = npair * (2 * panel + g_in) + (npair - 1) * g_pair
        ml = mr = (W - content_w) / 2

    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
    fx = lambda v: v / W
    fy = lambda v: 1.0 - v / H

    y = top_pad
    legend_band(fig, fx, fy, ml, y, content_w, H, legend_h - 0.06,
                fontsize=args.legend_fs)
    y += legend_h

    # Column headers, once per sheet, instead of a tag on all 100 panels.
    for c in range(npair):
        x0 = ml + c * (2 * panel + g_in + g_pair)
        fig.text(fx(x0 + panel / 2), fy(y + 0.02), 'prediction',
                 fontsize=args.cap_fs + 0.6, color=INK, ha='center', va='top',
                 fontweight='medium')
        fig.text(fx(x0 + panel + g_in + panel / 2), fy(y + 0.02), 'inherited',
                 fontsize=args.cap_fs + 0.6, color=INK3, ha='center', va='top')
    y += colhdr_h

    cell = 0
    if lead_dense is not None:
        dxyz, dlab = lead_dense
        x0 = ml
        # The reference occupies one pair-width cell so the grid stays regular.
        ax = fig.add_axes([fx(x0), fy(y + panel), fx(panel), panel / H])
        # v1 scales the dense panel's point size by (panel/8.4)**2
        bev_panel(ax, dxyz, dlab, s=args.s_dense * (panel / 8.4) ** 2,
                  draw_order=draw_order, origin_ms=3.6)
        for sp in ax.spines.values():
            sp.set_edgecolor(INK3)
            sp.set_linewidth(0.9)
        ax2 = fig.add_axes([fx(x0 + panel + g_in), fy(y + panel), fx(panel),
                            panel / H])
        ax2.set_facecolor(SURFACE)
        ax2.set_xticks([]); ax2.set_yticks([])
        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.text(0.04, 0.62, 'DENSE\nREFERENCE', fontsize=args.cap_fs + 1.6,
                 color=INK, va='center', ha='left', fontweight='medium',
                 linespacing=1.35, transform=ax2.transAxes)
        ax2.text(0.04, 0.34, f'{len(dxyz):,} points\nground-truth labels',
                 fontsize=args.cap_fs, color=INK2, va='center', ha='left',
                 linespacing=1.4, transform=ax2.transAxes)
        fig.text(fx(x0 + panel + g_in / 2), fy(y + panel + 0.065),
                 'dense reconstruction · every scan is a subset of it',
                 fontsize=args.cap_fs, color=INK3, ha='center', va='top')
        cell = 1

    for k, idx in enumerate(sel_idx):
        rec = per_config[idx]
        pos = cell + k
        r, c = divmod(pos, npair)
        x0 = ml + c * (2 * panel + g_in + g_pair)
        y0 = y + r * (panel + cap_h + row_gap)
        # FIXED point size, as in v1 -- deliberately NOT adapted to the scan's
        # point count: the 13x density spread between a 16-beam and a 127-beam
        # scan is the thing this figure is evidence for, and equalising ink
        # would erase it.
        for j, key in enumerate(('pred', 'inherited')):
            ax = fig.add_axes([fx(x0 + j * (panel + g_in)), fy(y0 + panel),
                               fx(panel), panel / H])
            bev_panel(ax, rec['xyz'], rec[key], s=args.s_cfg,
                      draw_order=draw_order, origin_ms=3.2)
        cfg = rec['cfg']
        xc = x0 + panel + g_in / 2
        l1 = (f'#{rec["index"] + 1:02d} · {cfg["beams"]} beams · '
              f'{cfg["hres"]} az · {rec["n"]:,} pts')
        l2 = (f'FoV {cfg["vlow"]:+.1f}…{cfg["vup"]:+.1f}° · '
              f'h {cfg["height"]:.2f} m')
        fig.text(fx(xc), fy(y0 + panel + 0.065), l1, fontsize=args.cap_fs,
                 color=INK, ha='center', va='top')
        fig.text(fx(xc), fy(y0 + panel + 0.175), l2, fontsize=args.cap_fs - 0.3,
                 color=INK3, ha='center', va='top')

    save(fig, out_stem, args.dpi, args.png)


def save(fig, stem, dpi, also_png):
    exts = ('pdf', 'png') if also_png else ('pdf',)
    for ext in exts:
        p = f'{stem}.{ext}'
        fig.savefig(p, dpi=dpi, facecolor=SURFACE)
        print(f'[fig] wrote {p}  ({os.path.getsize(p) / 1e6:.2f} MB)  '
              f'{fig.get_figwidth():.2f}x{fig.get_figheight():.2f} in')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', default='review_analysis/_source_data/a4_vis_bc2p6_LiSeer_SK.npz')
    ap.add_argument('--raw', default='review_analysis/_source_data/A4_raw.json')
    ap.add_argument('--outdir', default='figs/a4/R6')
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--width', type=float, default=7.5)
    ap.add_argument('--max-height', type=float, default=8.5)
    ap.add_argument('--cap-fs', type=float, default=5.6)
    ap.add_argument('--legend-fs', type=float, default=7.0)
    ap.add_argument('--s-cfg', type=float, default=0.30,
                    help='scatter point area (pt^2) in a configuration panel, '
                         'FIXED for every scan (v1 behaviour). v1 used 0.75 '
                         'at a 1.95-inch panel; this sheet\'s panel is ~1.16 '
                         'inch, so 0.75*(1.16/1.95)**2 ~= 0.27 keeps the same '
                         'apparent density. Rounded to 0.30.')
    ap.add_argument('--s-dense', type=float, default=0.30,
                    help='v1 value for the 8.4-inch dense panel; scaled by '
                         '(panel/8.4)**2 like v1 does')
    ap.add_argument('--rare-on-top', action='store_true', default=False,
                    help='draw classes by descending frequency so Vehicle / '
                         'Living Being / Objects win overlaps instead of '
                         'being covered by Nature. Changes which class is '
                         'visible where, so it differs from v1: OFF by '
                         'default, opt in only.')
    ap.add_argument('--png', action='store_true', default=False,
                    help='also write PNG (proofing only; the letter uses PDF)')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    z = np.load(args.cache, allow_pickle=True)
    raw = json.load(open(args.raw))
    configs = raw['configs']
    bc = float(z['beam_concentration'])
    assert abs(bc - 2.6) < 1e-9, (
        f'cache has beam_concentration={bc}, expected 2.6 -- this figure must '
        f'use the same render batch as the published version')

    accs = z['accs']
    per_config = [dict(index=i, cfg=configs[i], xyz=z[f'xyz_{i}'],
                       pred=z[f'pred_{i}'].astype(np.int64),
                       inherited=z[f'inh_{i}'].astype(np.int64),
                       n=len(z[f'xyz_{i}']), acc=float(accs[i]))
                  for i in range(len(configs))]

    # Configuration order: ascending beam count. Unchanged from the published
    # figure -- same sort key, same tie-breaks.
    order = sorted(range(len(per_config)),
                   key=lambda i: (per_config[i]['cfg']['beams'],
                                  per_config[i]['cfg']['hres'],
                                  per_config[i]['index']))

    if args.rare_on_top:
        draw_order, _ = rare_class_order(per_config)
        print('[draw order] --rare-on-top: by descending class frequency, '
              + ' > '.join(CLASS_NAMES_FIG[c] for c in draw_order))
    else:
        draw_order = None
        print('[draw order] v1 default: by height (rare classes may be '
              'covered by Nature; pass --rare-on-top to change this)')

    dense_xyz, dense_lab = z['dense_xyz'], z['dense_lab'].astype(np.int64)

    # 50 pairs + 1 reference cell = 51 cells over 3 sheets at 3 per row.
    # 18 / 18 / 15 cells => 6 / 6 / 5 rows.
    splits = [(0, 17), (17, 35), (35, 50)]
    meta = dict(frame_id=str(z['frame_id']))
    for si, (a, b) in enumerate(splits, start=1):
        sel = order[a:b]
        render_sheet(per_config, sel, draw_order, args, meta,
                     os.path.join(args.outdir, f'R6_pointclouds_p{si}'),
                     lead_dense=(dense_xyz, dense_lab) if si == 1 else None)

    print(f'\n[R6] {len(order)} configurations over {len(splits)} sheets; '
          f'point counts {min(r["n"] for r in per_config):,}'
          f'-{max(r["n"] for r in per_config):,}')


if __name__ == '__main__':
    main()
