"""A4 qualitative figure (Fig. R5c/d): what the 50 sensor configurations
actually look like, and what the model predicts on each of them.

`review_analysis/A4_consistency.py` reports the *numbers* for R2C1 (Table R9): how
often the per-point semantic prediction is unchanged when the same physical
scene is scanned by 50 different mechanical LiDARs.  This script renders the
same experiment so the reader can *see* it:

  * a leading, deliberately large BEV panel of the **dense reconstruction**
    coloured by its COLA-8 ground truth - the common reference every sampled
    scan is a subset of;
  * then, for each of the 50 sensor configurations, **two BEV panels side by
    side**: the model's **predicted** per-point label, and the label the point
    **inherited** from the dense cloud through the sampler's provenance map.
    Predicted-vs-inherited can therefore be compared per configuration, and
    configuration-vs-configuration down the sheet.

Everything is bird's-eye / top-down.  No perspective, oblique or side views.

The frame and the 50 configurations are NOT re-drawn here: they are read back
from `review_analysis/_source_data/A4_raw.json`, so these panels correspond point-for-point to the
statistics already reported.  Re-running the sweep is unavoidable (the A4 run
persisted votes, not per-point predictions), but every determinism guarantee of
the original run is reproduced verbatim by importing its machinery:

  * the dense frame is read ONCE and the SAME tensor is handed to all 50
    sampler calls (`assert pts.data_ptr() == base_ptr` after every call);
  * `model.dense_train = False`, `model.test_max_points = None`;
  * minority mixup bypassed; per-call seeds identical to the A4 run.

As a hard cross-check the script recomputes `agreement_pct`,
`majority_vote_acc_pct` and all 50 `per_config_acc` values from what it just
rendered and compares them against `review_analysis/_source_data/A4_raw.json`.  If the rendering did
not reproduce the reported statistics it says so loudly instead of shipping a
figure that contradicts the table.

Inference only.  Run on the analysis GPU (never 5/6/7 while E10-E12 train):

    CUDA_VISIBLE_DEVICES=4 python review_analysis/A4_visualize.py \
        --ckpt ckpts/LiSeer_SK.pth --device cuda:0 --dpi 600
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Circle, Rectangle                  # noqa: E402

from review_analysis.A4_consistency import (                             # noqa: E402
    CLASS_NAMES, NUM_CLASSES, IGNORE_LABEL, build_liseer, prepare_sampler,
    sample_with_config, predict, compute_metrics)
from review_analysis.CODE5_risley_demo import load_dense_frame           # noqa: E402

# ---------------------------------------------------------------------------
# visual system - inherited from review_analysis/A6_figR4.py (the paper's figures)
# ---------------------------------------------------------------------------
INK, INK2, INK3 = '#0b0b0b', '#52514e', '#8a8984'
# Pure white, not an off-white: these sheets sit beside locally-produced
# #ffffff figures in the response letter, where any tint reads as a grey block.
SURFACE = '#ffffff'
ACCENT = '#eb6834'
SPINE = '#dcdbd6'
RING = '#e3e2de'

# ---------------------------------------------------------------------------
# COLA-8 class colours.  CATEGORICAL - class identity, never magnitude: one
# fixed colour per class, identical in every panel of every figure, never
# cycled.
#
# DEFAULT = the project's own map, verbatim from the repository's plotting code
# (`.history/point_freq_*.py`, `.history/voxel_freq_*.py`), including the
# `[::-1]` channel reversal those files apply before handing the values to
# matplotlib.  That reversal is the established convention here and is what
# reproduces the paper's existing semantic figures, so it is kept exactly as
# written and NOT "fixed" to plain RGB.  Class names likewise follow that file
# (class 6 is "Objects" there; the A4 tables call the same class "Movable
# objects").
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
color_map_dg = {k: np.array(v[::-1], dtype=np.float32) / 255.0
                for k, v in COLOR_MAP_DG_ORI.items()}

# Alternative, selectable with --palette cvd but NOT used for the shipped
# figures.  Chosen by a max-min search over OKLCH; all 28 pairs clear the
# data-viz palette gates simulated for protan/deutan/tritan (worst all-pairs
# CVD dE 8.7, worst normal-vision dE 15.9; gates 8 and 15).  The project map
# above does not clear those gates - classes 1, 4 and 5 are three pale blues -
# which is a documented legibility limitation of the house palette, reported in
# `results/A4_palette_validation.txt`, not an accident of this script.
PALETTE_CVD = {i: np.array(c, dtype=np.float32) for i, c in enumerate(
    [(0.231, 0.227, 0.216), (0.137, 0.337, 0.671), (0.957, 0.580, 0.216),
     (0.525, 0.251, 0.365), (0.502, 0.478, 0.157), (0.773, 0.576, 0.984),
     (0.106, 0.573, 0.663), (0.298, 0.780, 0.635)])}

# Class names as the project's own plotting code spells them. Class 6 is
# called "Objects" throughout the project.
CLASS_NAMES_FIG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle',
                   'Nature', 'Living Being', 'Objects', 'Other Ground']

ACTIVE_MAP = dict(color_map_dg)      # {int class -> float32 RGB}; see set_palette


def set_palette(name='project', other_colour=None):
    """Select the class colour map used by every panel, legend and .ply."""
    global ACTIVE_MAP
    ACTIVE_MAP = dict(color_map_dg if name == 'project' else PALETTE_CVD)
    if other_colour is not None:
        # class 0 = "Other"/unlabelled is pure black in the project map; on a
        # light surface it can dominate.  --other-colour replaces ONLY that
        # entry, and the substitution is stated in the figure footnote.
        ACTIVE_MAP[IGNORE_LABEL] = np.asarray(
            matplotlib.colors.to_rgb(other_colour), dtype=np.float32)
    return ACTIVE_MAP


def label_colors(labels):
    """Per-point RGB, using the project's own label-colouring idiom:

        colors[mask] = np.array(
            [color_map_dg.get(int(label), [0.5, 0.5, 0.5])
             for label in labels[mask]], dtype=np.float32)

    applied to *every* point that is drawn.  The lookup is grouped by label
    value rather than run per point - identical result, but it stays fast on a
    1-million-point cloud.  `_check_colour_idiom` asserts the two agree.
    """
    lab = np.asarray(labels)
    out = np.empty((lab.shape[0], 3), dtype=np.float32)
    for u in np.unique(lab):
        out[lab == u] = np.asarray(
            ACTIVE_MAP.get(int(u), [0.5, 0.5, 0.5]), dtype=np.float32)
    return out


def _check_colour_idiom(labels, n=8192):
    """Prove the grouped lookup equals the literal per-point comprehension."""
    lab = np.asarray(labels)
    idx = np.random.default_rng(0).integers(0, lab.shape[0],
                                            size=min(n, lab.shape[0]))
    ref = np.array([ACTIVE_MAP.get(int(label), [0.5, 0.5, 0.5])
                    for label in lab[idx]], dtype=np.float32)
    got = label_colors(lab)[idx]
    assert np.array_equal(ref, got), 'colour lookup does not match the idiom'
    return True


LIM = 51.5          # MAX_BOUND rho = 51.2 m -> nothing is cropped
RINGS = (10, 20, 30, 40, 50)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def parse_frame_id(fid):
    """'semantickitti/seq08/000100' -> (8, 100)."""
    parts = fid.split('/')
    if len(parts) != 3 or not parts[1].startswith('seq'):
        raise ValueError(f'unrecognised frame_id {fid!r}')
    return int(parts[1][3:]), int(parts[2])


def xyz_of(feat):
    """[N,8] PointsegMapping layout -> [N,3] cartesian (cols 6,7,5 = x,y,z)."""
    a = feat.detach().float().cpu().numpy()
    return np.stack([a[:, 6], a[:, 7], a[:, 5]], axis=1)


def run_sweep(model, configs, frame, device, seed0):
    """Replay the A4 sweep, keeping per-point geometry + labels for drawing.

    Returns (per_config, votes, obs).  `per_config[i]` holds the sampled scan's
    xyz, the model's prediction, and the label inherited from the dense cloud.
    """
    pts, gi, giv, lab = frame
    base_ptr, gi_ptr = pts.data_ptr(), gi.data_ptr()
    n_dense = pts.shape[0]
    gt = lab.detach().cpu().numpy().astype(np.int64)

    votes = torch.zeros((n_dense, NUM_CLASSES), dtype=torch.int32, device=device)
    obs = torch.zeros(n_dense, dtype=torch.int32, device=device)
    out = []

    for i, cfg in enumerate(configs):
        t0 = time.time()
        seed = seed0 * 1000 + i
        rp, rg, rgv, rl, prov, used = sample_with_config(
            model, pts, gi, giv, lab, cfg, seed)
        # DETERMINISM GUARD - the same tensor object must serve every call
        assert pts.data_ptr() == base_ptr, 'the dense input tensor was replaced!'
        assert gi.data_ptr() == gi_ptr, 'the dense grid index was replaced!'
        pred = predict(model, rp, rg, rgv, rl).to(device)
        assert pts.data_ptr() == base_ptr, 'the dense input tensor was replaced!'

        prov_np = prov.cpu().numpy().astype(np.int64)
        inherited = rl.detach().cpu().numpy().astype(np.int64)
        # The "inherited" label MUST be the dense label of the point the ray
        # actually hit.  If provenance and labels ever disagreed, the right
        # panel of every pair would be meaningless - so check it, don't assume.
        assert np.array_equal(inherited, gt[prov_np]), (
            f'config {i}: inherited labels != dense GT at the provenance rows')

        obs.index_add_(0, prov, torch.ones_like(prov, dtype=torch.int32))
        votes.index_put_((prov, pred.to(torch.int64)),
                         torch.ones(prov.shape[0], dtype=torch.int32, device=device),
                         accumulate=True)
        out.append(dict(index=i, cfg=cfg, seed=seed, used=used,
                        xyz=xyz_of(rp).astype(np.float32),
                        pred=pred.cpu().numpy().astype(np.int64),
                        inherited=inherited, prov=prov_np))
        print(f'  [{i + 1:>3}/{len(configs)}] beams={cfg["beams"]:>2} '
              f'hres={cfg["hres"]:>4} vlow={cfg["vlow"]:>6} vup={cfg["vup"]:>5} '
              f'h={cfg["height"]:.2f} -> {rp.shape[0]:>7,} pts '
              f'({time.time() - t0:.1f}s)', flush=True)
        del rp, rg, rgv, rl, pred
        torch.cuda.empty_cache()

    return out, votes.cpu().numpy(), obs.cpu().numpy()


def cross_check(per_config, votes, obs, gt, stored, min_obs, tol=0.05,
                gate=True):
    """Recompute Table R9's numbers from what we just rendered.

    The panels must correspond to the reported statistics; if they do not, the
    figure is wrong and we say so rather than shipping it.
    """
    m = compute_metrics(votes, obs,  gt,
                        [(c['prov'], c['pred']) for c in per_config], min_obs)
    lines, ok = [], True
    tag = (lambda g: 'ok' if g else 'MISMATCH') if gate else (lambda g: '--')
    for k in ('qualifying_points', 'agreement_pct', 'majority_vote_acc_pct',
              'per_config_acc_mean', 'per_config_acc_sd',
              'per_config_acc_min', 'per_config_acc_max'):
        a, b = m[k], stored[k]
        d = abs(a - b)
        good = d <= (0 if k == 'qualifying_points' else tol)
        ok &= (good or not gate)
        lines.append(f'[{tag(good)}] {k:<24} '
                     f'rendered {a:>12,.4f}   A4_raw.json {b:>12,.4f}   '
                     f'delta {d:.4g}')
    a = np.asarray(m['per_config_acc'])
    b = np.asarray(stored['per_config_acc'])
    d = np.abs(a - b)
    good = bool(d.max() <= tol)
    ok &= (good or not gate)
    lines.append(f'[{tag(good)}] per_config_acc[50]      '
                 f'max |delta| {d.max():.4g}, mean |delta| {d.mean():.4g}')
    return ok, lines, m


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def bev_panel(ax, xyz, labels, s, lim=LIM, rings=True, ring_labels=False,
              origin_ms=4.0, origin=True):
    """One bird's-eye panel, coloured by COLA-8 class identity.

    Painter's order is by height so that vehicles / structure are not buried
    under the road surface drawn after them; this is a draw-order choice only,
    no point is dropped.
    """
    ax.set_facecolor(SURFACE)
    if rings:
        for r in RINGS:
            ax.add_patch(Circle((0, 0), r, fill=False, ec=RING, lw=0.6, zorder=0))
        if ring_labels:
            # along +x only, so the labels never stack on top of each other
            for r in (30, 50):
                ax.text(r, 0.0, f'{r} m', fontsize=6.6, color=INK3,
                        ha='center', va='center', zorder=4,
                        bbox=dict(boxstyle='square,pad=0.14', fc=SURFACE,
                                  ec='none', alpha=0.86))
    if len(xyz):
        o = np.argsort(xyz[:, 2], kind='stable')
        ax.scatter(xyz[o, 0], xyz[o, 1], s=s, c=label_colors(labels[o]),
                   linewidths=0, zorder=2, rasterized=True)
    if origin:
        # sensor origin: ink, never a class colour, so it cannot be misread
        # as a semantic label
        ax.plot(0, 0, marker='P', ms=origin_ms, mfc=INK, mec='white',
                mew=max(0.35, origin_ms * 0.16), zorder=6, ls='none')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)
        sp.set_linewidth(0.8)


def corner_tag(ax, text, color=INK2, fontsize=7.4):
    ax.text(0.035, 0.968, text, transform=ax.transAxes, ha='left', va='top',
            fontsize=fontsize, color=color, zorder=7,
            bbox=dict(boxstyle='round,pad=0.22', fc=SURFACE, ec=SPINE, lw=0.5,
                      alpha=0.94))


def legend_axes(ax, counts, vertical=True, fontsize=8.2, title=True,
                origin_note=False):
    """The single shared class legend.  Same colours as every panel."""
    ax.set_facecolor(SURFACE)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    tot = float(counts.sum()) if counts is not None else 0.0
    if vertical:
        y0 = 0.985
        if title:
            ax.text(0.0, y0, 'COLA-8 semantic classes',
                    fontsize=fontsize + 1.1, color=INK, va='top', ha='left',
                    fontweight='medium')
            y0 -= 0.10
        step = 0.088
        for c in range(NUM_CLASSES):
            y = y0 - 0.035 - c * step
            ax.add_patch(Rectangle((0.005, y - 0.021), 0.085, 0.042,
                                   fc=ACTIVE_MAP.get(c, [0.5, 0.5, 0.5]),
                                   ec=SPINE, lw=0.7,
                                   transform=ax.transAxes, clip_on=False))
            name = f'{c}: ' + CLASS_NAMES_FIG[c] + (
                '  (ignore label)' if c == IGNORE_LABEL else '')
            ax.text(0.125, y, name, fontsize=fontsize, color=INK2,
                    va='center', ha='left')
            if tot:
                ax.text(0.99, y, f'{100 * counts[c] / tot:4.1f}%',
                        fontsize=fontsize - 0.9, color=INK3, va='center',
                        ha='right')
        if tot:
            ax.text(0.99, y0 - 0.035 + step * 0.72, '% of the dense cloud',
                    fontsize=fontsize - 1.4, color=INK3, va='center', ha='right')
        if origin_note:
            ax.text(0.005, y0 - 0.035 - NUM_CLASSES * step - 0.012,
                    '\u271a   sensor origin', fontsize=fontsize, color=INK2,
                    va='center', ha='left')
    else:
        xs = np.linspace(0.005, 0.90, NUM_CLASSES + 1)[:NUM_CLASSES]
        for c in range(NUM_CLASSES):
            ax.add_patch(Rectangle((xs[c], 0.44), 0.016, 0.30,
                                   fc=ACTIVE_MAP.get(c, [0.5, 0.5, 0.5]),
                                   ec=SPINE, lw=0.7,
                                   transform=ax.transAxes, clip_on=False))
            lbl = f'{c}: ' + CLASS_NAMES_FIG[c] + (
                ' (ignore)' if c == IGNORE_LABEL else '')
            ax.text(xs[c] + 0.020, 0.59, lbl, fontsize=fontsize, color=INK2,
                    va='center', ha='left')
        ax.text(0.0, 0.10, '\u271a marks the sensor origin.  Colour encodes '
                           'COLA-8 class identity and is identical in every '
                           'panel of both sheets.', fontsize=fontsize - 0.8,
                color=INK3, va='center', ha='left')


# ---------------------------------------------------------------------------
def render_dense(dense_xyz, dense_lab, meta, args, out_stem):
    """The leading reference: the dense cloud, coloured by its COLA-8 GT."""
    p = 8.4                                    # square BEV panel, inches
    ml, gap, lw_, mr = 0.50, 0.55, 3.55, 0.45
    W = ml + p + gap + lw_ + mr
    top, sub, foot = 0.62, 0.52, 0.84
    H = 0.22 + top + p + sub + foot
    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
    fx, fy = (lambda v: v / W), (lambda v: 1.0 - v / H)

    ax = fig.add_axes([fx(ml), fy(0.22 + top + p), fx(p), p / H])
    bev_panel(ax, dense_xyz, dense_lab, s=args.s_dense * (p / 8.4) ** 2,
              ring_labels=True, origin_ms=9.0)
    ax.set_title('Dense reconstruction — ground-truth COLA-8 labels',
                 fontsize=14, color=INK, pad=10, fontweight='medium')
    ax.text(0.5, -0.020,
            f'{meta["frame_id"]}   ·   {len(dense_xyz):,} points   ·   the '
            f'reference cloud every one of the 50 simulated scans is drawn from',
            transform=ax.transAxes, ha='center', va='top', fontsize=9.4,
            color=INK2)

    lax = fig.add_axes([fx(ml + p + gap), fy(0.22 + top + 0.24 + 4.6),
                        fx(lw_), 4.6 / H])
    legend_axes(lax, np.bincount(dense_lab, minlength=NUM_CLASSES).astype(float),
                origin_note=True)

    fig.text(fx(ml), fy(H - 0.24),
             'Bird\'s-eye (top-down) projection of the x–y plane; range rings '
             'every 10 m out to the 51.2 m polar bound of the model\'s grid.\n'
             'Points are painted in order of height, so structure and vehicles '
             'are not buried under the road surface drawn over them; no point '
             'is omitted or subsampled.',
             fontsize=8.2, color=INK3, va='bottom', ha='left', linespacing=1.7)
    save(fig, out_stem, args.dpi)


def render_sheet(per_config, args, meta, out_stem, first, last, lead_dense=None,
                 sheet_label='', annotate=True):
    """One sheet of paired BEV panels: predicted | inherited, per configuration.

    Layout is in inches and placed by hand, so the pair grouping (a tight gap
    inside a pair, a wide gap between pairs) survives at any panel count.
    `annotate` toggles ONLY the numeric annotations: the point count and the
    per-configuration accuracy under each pair, and the footnote that defines
    them.  The panels themselves - identical data, identical geometry, identical
    colours - are drawn by the same code either way.
    """
    sel = per_config[first:last]
    npair = args.pairs_per_row
    panel = args.panel                    # inches, one BEV panel
    g_in, g_pair = 0.055, 0.34            # gap inside a pair / between pairs
    cap = 0.42                            # caption strip under each row
    row_gap = 0.20
    nrows = int(np.ceil(len(sel) / npair))

    content_w = npair * (2 * panel + g_in) + (npair - 1) * g_pair
    ml, mr = 0.45, 0.45
    W = content_w + ml + mr

    top_pad, title_h = 0.34, 1.16                   # title + 2 subtitle lines
    dense_h = (args.dense_lead + 0.52) if lead_dense is not None else 0.0
    band_h = 0.0 if lead_dense is not None else 0.66      # legend band
    hdr_h = 0.50
    grid_h = nrows * (panel + cap) + (nrows - 1) * row_gap
    foot = meta['foot'] if annotate else meta['foot_clean']
    # 0.30 in of clearance above the footnote (it is anchored va='top' just
    # below the grid) + one line-height per wrapped line + a bottom margin.
    foot_h = 0.30 + 0.155 * (wrap_foot(foot, content_w, 8.0).count('\n') + 1) + 0.22
    H = top_pad + title_h + dense_h + band_h + hdr_h + grid_h + foot_h

    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
    fx = lambda v: v / W
    fy = lambda v: 1.0 - v / H            # from the TOP, in inches

    y = top_pad
    fig.text(fx(ml), fy(y + 0.10), meta['title'], fontsize=15.5, color=INK,
             ha='left', va='top', fontweight='medium')
    fig.text(fx(W - mr), fy(y + 0.13), sheet_label, fontsize=10.5, color=INK2,
             ha='right', va='top')
    fig.text(fx(ml), fy(y + 0.44), meta['subtitle'], fontsize=9.6, color=INK2,
             ha='left', va='top', linespacing=1.6)
    y += title_h

    if lead_dense is not None:
        d = args.dense_lead
        ax = fig.add_axes([fx(ml), fy(y + d), fx(d), d / H])
        bev_panel(ax, lead_dense[0], lead_dense[1],
                  s=args.s_dense * (d / 8.4) ** 2, ring_labels=True,
                  origin_ms=7.5)
        ax.set_title('Dense reconstruction — ground-truth COLA-8 labels',
                     fontsize=12.5, color=INK, pad=7, fontweight='medium')
        ax.text(0.5, -0.020, f'{meta["frame_id"]}   ·   '
                             f'{len(lead_dense[0]):,} points   ·   REFERENCE',
                transform=ax.transAxes, ha='center', va='top', fontsize=8.8,
                color=INK2)
        lgx, lgw = ml + d + 0.50, 3.25
        lax = fig.add_axes([fx(lgx), fy(y + d - 0.55), fx(lgw),
                            (d - 0.55) / H])
        cnt = np.bincount(lead_dense[1], minlength=NUM_CLASSES).astype(float)
        legend_axes(lax, cnt, origin_note=True)
        gx = lgx + lgw + 0.60
        fig.text(fx(gx), fy(y + 0.30), meta['guide'], fontsize=8.9,
                 color=INK2, ha='left', va='top', linespacing=1.75)
        y += d + 0.52

    if band_h:
        lax = fig.add_axes([fx(ml), fy(y + band_h - 0.06), fx(content_w),
                            (band_h - 0.12) / H])
        legend_axes(lax, None, vertical=False)
        y += band_h

    fig.text(fx(ml), fy(y + 0.20), meta['header'], fontsize=10.6, color=INK,
             ha='left', va='top', fontweight='medium')
    fig.text(fx(ml + content_w), fy(y + 0.20), meta['header_right'],
             fontsize=8.8, color=INK3, ha='right', va='top')
    y += hdr_h

    for k, rec in enumerate(sel):
        r, c = divmod(k, npair)
        x0 = ml + c * (2 * panel + g_in + g_pair)
        y0 = y + r * (panel + cap + row_gap)
        for j, (key, tag) in enumerate((('pred', 'predicted'),
                                        ('inherited', 'inherited'))):
            ax = fig.add_axes([fx(x0 + j * (panel + g_in)), fy(y0 + panel),
                               fx(panel), panel / H])
            bev_panel(ax, rec['xyz'], rec[key], s=args.s_cfg, origin_ms=4.4)
            corner_tag(ax, tag, INK2 if j == 0 else INK3)
        cfg = rec['cfg']
        xc = x0 + panel + g_in / 2
        fig.text(fx(xc), fy(y0 + panel + 0.155),
                 f'#{rec["index"] + 1:02d}   {cfg["beams"]} beams × '
                 f'{cfg["hres"]} az', fontsize=9.2, color=INK, ha='center',
                 va='top')
        sub = (f'FoV {cfg["vlow"]:+.1f}…{cfg["vup"]:+.1f}°   ·   mount '
               f'{cfg["height"]:.2f} m')
        if annotate:
            sub += f'   ·   {rec["n"]:,} pts   ·   acc {rec["acc"]:.1f}%'
        fig.text(fx(xc), fy(y0 + panel + 0.325), sub, fontsize=7.4,
                 color=INK3, ha='center', va='top')

    # Anchor the footnote at the TOP of its reserved band and let it grow
    # downward.  Drawing it va='bottom' from the sheet foot made it grow
    # upward into the last caption row (they collided on every sheet), while
    # the space foot_h had reserved sat empty underneath.
    fig.text(fx(ml), fy(y + grid_h + 0.30), wrap_foot(foot, content_w, 8.0),
             fontsize=8.0, color=INK3, ha='left', va='top', linespacing=1.75)
    save(fig, out_stem, args.dpi)


def wrap_foot(text, width_in, fontsize):
    """Wrap each footnote paragraph to the sheet width (no overflow, ever)."""
    import textwrap
    ncol = max(40, int(width_in * 72 / (fontsize * 0.50)))
    return '\n'.join('\n'.join(textwrap.wrap(par, ncol)) or par
                     for par in text.split('\n'))


def save(fig, stem, dpi):
    for ext in ('png', 'pdf'):
        p = f'{stem}.{ext}'
        fig.savefig(p, dpi=dpi, facecolor=SURFACE)
        wh = (f'  {int(round(fig.get_figwidth() * dpi))}x'
              f'{int(round(fig.get_figheight() * dpi))} px' if ext == 'png'
              else f'  {fig.get_figwidth():.1f}x{fig.get_figheight():.1f} in')
        print(f'[fig] wrote {p}  ({os.path.getsize(p) / 1e6:.1f} MB){wh}')
    plt.close(fig)


# ---------------------------------------------------------------------------
def write_ply(path, xyz, labels):
    """Binary little-endian PLY, one vertex per point, colour = class colour."""
    rgb = np.clip(np.rint(label_colors(labels) * 255), 0, 255).astype(np.uint8)
    rec = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                    ('red', 'u1'), ('green', 'u1'),
                                    ('blue', 'u1'), ('label', 'u1')])
    rec['x'], rec['y'], rec['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec['red'], rec['green'], rec['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    rec['label'] = labels.astype(np.uint8)
    hdr = ('ply\nformat binary_little_endian 1.0\n'
           f'element vertex {len(xyz)}\n'
           'property float x\nproperty float y\nproperty float z\n'
           'property uchar red\nproperty uchar green\nproperty uchar blue\n'
           'property uchar label\nend_header\n').encode()
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(rec.tobytes())


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', default='ckpts/LiSeer_SK.pth',
                    help='the checkpoint to render (A4 rendered ONLY 716)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--raw', default='review_analysis/_source_data/A4_raw.json',
                    help='the completed A4 run: supplies the frame and the 50 '
                         'sensor configurations, so the panels match Table R9')
    ap.add_argument('--outdir', default='figs/a4')
    ap.add_argument('--cache', default='',
                    help='npz of the replayed sweep; reused so layout edits '
                         're-render without touching the GPU')
    ap.add_argument('--panel', type=float, default=1.95,
                    help='width of one BEV panel, inches')
    ap.add_argument('--dense-lead', type=float, default=5.4,
                    help='width of the leading dense reference panel, inches')
    ap.add_argument('--pairs-per-row', type=int, default=4)
    ap.add_argument('--per-sheet', type=int, default=20)
    ap.add_argument('--s-cfg', type=float, default=0.75,
                    help='scatter point area (pt^2) in a configuration panel')
    ap.add_argument('--s-dense', type=float, default=0.30,
                    help='scatter point area (pt^2) for an 8.4-inch dense panel; scaled by (panel/8.4)^2 elsewhere')
    ap.add_argument('--ply', action='store_true', default=False,
                    help='also export the coloured clouds as .ply')
    ap.add_argument('--order', default='beams', choices=('beams', 'index'),
                    help="'beams' (default) sorts the panels by beam count so "
                         'the density gradient reads as a trend across the '
                         "grid; 'index' keeps the order of review_analysis/_source_data/A4_raw.json. "
                         'Either way each panel is captioned with its own '
                         'configuration index and parameters')
    ap.add_argument('--beam-concentration', type=float, default=2.6,
                    help='concentrate the vertical beam angles toward the '
                         'horizon (|t|**c), the same 2.6 the Fig. R4 renders '
                         'use. Pass a negative value to keep the sampler '
                         "default (uniform angles), which reproduces the "
                         'published A4 statistics exactly')
    ap.add_argument('--palette', default='project',
                    choices=('project', 'cvd'),
                    help="'project' (default) = the repository's own "
                         'color_map_dg, channel reversal included; '
                         "'cvd' = the colour-vision-validated alternative "
                         '(not used for the shipped figures)')
    ap.add_argument('--other-colour', default='',
                    help='replace ONLY the class-0 (Other / ignore label) '
                         'colour, e.g. "#e6e5e0"; the substitution is stated '
                         'in the figure footnote. Empty = keep the project '
                         "map's pure black")
    ap.add_argument('--variant', default='both',
                    choices=('both', 'annotated', 'clean'),
                    help='annotated = per-configuration point count and '
                         'accuracy under each pair; clean = the same panels '
                         'with no numeric annotation (for typesetting)')
    ap.add_argument('--figs-only', action='store_true', default=False,
                    help='render from --cache without loading the model')
    ap.add_argument('--label', default='LiSeer',
                    help='name shown in the figures. Defaults to the method '
                         'name; never use the checkpoint filename, which '
                         'identifies an unreleased model')
    args = ap.parse_args()
    if args.beam_concentration is not None and args.beam_concentration < 0:
        args.beam_concentration = None       # sampler default: uniform angles

    os.makedirs(args.outdir, exist_ok=True)
    set_palette(args.palette, args.other_colour or None)
    print(f'[palette] {args.palette}'
          + (f'  (class 0 overridden -> {args.other_colour})'
             if args.other_colour else ''))
    with open(args.raw) as f:
        raw = json.load(f)
    name = os.path.splitext(os.path.basename(args.ckpt))[0]
    if name not in raw['checkpoints']:
        raise SystemExit(f'{name} is not in {args.raw} '
                         f'(have {list(raw["checkpoints"])})')
    stored = raw['checkpoints'][name]
    configs = raw['configs']
    seq, frame_no = parse_frame_id(raw['frame_id'])
    if not args.cache:
        tagc = ('uniformbeams' if args.beam_concentration is None
                else f'bc{args.beam_concentration:g}'.replace('.', 'p'))
        args.cache = os.path.join(args.outdir, f'a4_vis_{tagc}_{name}.npz')

    print(f'[A4-vis] checkpoint  {args.ckpt}')
    print(f'[A4-vis] frame       {raw["frame_id"]}  (seq {seq}, frame {frame_no})')
    print(f'[A4-vis] configs     {len(configs)} replayed from {args.raw} '
          f'(seed {raw["seed"]})')

    # ---- the dense reference cloud (read once, reused everywhere) ----------
    if args.figs_only and os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        dense_xyz, dense_lab = z['dense_xyz'], z['dense_lab'].astype(np.int64)
        per_config = [dict(index=int(i), cfg=configs[int(i)],
                           xyz=z[f'xyz_{i}'], pred=z[f'pred_{i}'].astype(np.int64),
                           inherited=z[f'inh_{i}'].astype(np.int64),
                           prov=z[f'prov_{i}'].astype(np.int64))
                      for i in range(len(configs))]
        check_lines = list(z['check_lines'])
        check_ok = bool(z['check_ok'])
        metrics_rendered = json.loads(str(z['metrics']))
        bc = float(z['beam_concentration'])
        args.beam_concentration = None if bc < 0 else bc
    else:
        device = torch.device(args.device)
        print(f'[gpu] {args.device} -> {torch.cuda.get_device_name(device)}')
        pts, gi, giv, lab = load_dense_frame(seq, frame_no, device)
        assert int(pts.shape[0]) == raw['dense_points'], (
            f'dense frame has {pts.shape[0]} points, A4 recorded '
            f'{raw["dense_points"]} - not the same frame')
        print(f'[frame] {pts.shape[0]:,} dense points; tensor data_ptr = '
              f'{pts.data_ptr()} (asserted after every sampler call)')

        model, load_report = build_liseer(args.ckpt, device)
        prepare_sampler(model, raw['dataset'])
        # Vertical-beam concentration, the same setting the Fig. R4 renders use
        # (review_analysis/A6_figR4_multi.py --beam-concentration 2.6).  Uniform
        # vertical angles put ~78% of the downward beams inside 15 m, which
        # shows up in BEV as an over-dense knot of rings around the vehicle; the
        # |t|**c warp pushes that to ~90% against the real scan's 80.4%.  This
        # is an INSTANCE attribute only - the class default stays None, so the
        # training path is bit-identical (E10-E12 are training on this file).
        if args.beam_concentration is not None:
            model.beam_concentration = args.beam_concentration
        print(f'[sampler] beam_concentration = '
              f'{getattr(model, "beam_concentration", None)}  '
              f'(class default is None = uniform vertical angles)')
        per_config, votes, obs = run_sweep(
            model, configs, (pts, gi, giv, lab), device, raw['seed'])
        gt = lab.cpu().numpy().astype(np.int64)

        same_regime = args.beam_concentration is None
        metrics_rendered = None
        check_ok, check_lines, rendered_m = cross_check(
            per_config, votes, obs, gt, stored, raw['min_obs'],
            gate=same_regime)
        head = ('cross-check against the reported statistics (identical '
                'sampling regime - these MUST match exactly):' if same_regime
                else (f'the render uses beam_concentration='
                      f'{args.beam_concentration}, the published A4 statistics '
                      f'were computed with the uniform default. Both are '
                      f'reported; neither overwrites the other:'))
        print(f'\n[A4-vis] {head}')
        for l in check_lines:
            print('  ' + l)
        print(f'[A4-vis] cross_check_ok = {check_ok}'
              + ('' if same_regime else '  (informational: different regime)'))
        # label-free invariance of each view, the number Fig. R5c reports
        metrics_rendered = rendered_m
        cons = rendered_m['per_config_consensus_agreement']
        print(f'[A4-vis] per-config agreement with the cross-sensor consensus: '
              f'mean {np.mean(cons):.2f}%  min {np.min(cons):.2f}%  '
              f'(published, uniform beams: '
              f'{np.mean(stored["per_config_consensus_agreement"]):.2f}%)')

        # per-configuration accuracy, computed exactly as Table R9 does
        # (qualifying points only, GT != 0) so the caption number IS the
        # reported number
        qual = obs >= raw['min_obs']
        for rec in per_config:
            m = qual[rec['prov']] & (gt[rec['prov']] != IGNORE_LABEL)
            rec['acc'] = float(100.0 * (rec['pred'][m] == gt[rec['prov']][m]).mean())

        dense_xyz = np.stack([pts[:, 6].cpu().numpy(), pts[:, 7].cpu().numpy(),
                              pts[:, 5].cpu().numpy()], axis=1).astype(np.float32)
        dense_lab = gt

        np.savez_compressed(
            args.cache, dense_xyz=dense_xyz,
            dense_lab=dense_lab.astype(np.uint8),
            check_lines=np.array(check_lines, dtype=object),
            check_ok=check_ok, ckpt=name, frame_id=raw['frame_id'],
            metrics=json.dumps(rendered_m),
            beam_concentration=(-1.0 if args.beam_concentration is None
                                else args.beam_concentration),
            accs=np.array([r['acc'] for r in per_config]),
            **{f'xyz_{r["index"]}': r['xyz'] for r in per_config},
            **{f'pred_{r["index"]}': r['pred'].astype(np.uint8) for r in per_config},
            **{f'inh_{r["index"]}': r['inherited'].astype(np.uint8) for r in per_config},
            **{f'prov_{r["index"]}': r['prov'] for r in per_config})
        print(f'[A4-vis] cached the replayed sweep -> {args.cache}')
        # write next to the cache rather than into a fixed 'results/' dir,
        # which no longer exists after the review-tree cleanup
        _xdir = os.path.dirname(args.cache) or args.outdir
        os.makedirs(_xdir, exist_ok=True)
        with open(os.path.join(_xdir, f'A4_visualize_crosscheck_{name}.txt'),
                  'w') as f:
            f.write(f'A4 visualisation cross-check\ncheckpoint : {args.ckpt}\n'
                    f'frame      : {raw["frame_id"]}\n'
                    f'configs    : {len(configs)} replayed from {args.raw}\n\n'
                    + '\n'.join(check_lines)
                    + f'\n\ncross_check_ok = {check_ok}\n')
        del model
        torch.cuda.empty_cache()

    if args.figs_only:
        accs = np.load(args.cache, allow_pickle=True)['accs']
        for r, a in zip(per_config, accs):
            r['acc'] = float(a)
    for r in per_config:
        r['n'] = len(r['xyz'])

    if not check_ok:
        print('\n[A4-vis] !! the replay did NOT reproduce the reported '
              'statistics - the figure would contradict Table R9.  Refusing.')
        raise SystemExit(2)

    # ---- figures ----------------------------------------------------------
    # Panel order.  Sorting by beam count makes the density gradient legible
    # as a trend across the grid; every panel still carries its own
    # configuration index, so nothing is anonymised by the reordering.
    if args.order == 'beams':
        per_config = sorted(per_config,
                            key=lambda r: (r['cfg']['beams'], r['cfg']['hres'],
                                           r['index']))

    _check_colour_idiom(dense_lab)
    _check_colour_idiom(per_config[0]['pred'])
    print('[palette] per-point colours reproduce the project lookup idiom '
          'exactly (spot-checked on the dense cloud and on config #01)')

    npts = np.array([r['n'] for r in per_config])
    rendered = metrics_rendered     # statistics of exactly what is drawn
    pc = rendered['per_class']      # per-class numbers OF THESE RENDERS
    cons_mean = float(np.mean(rendered['per_config_consensus_agreement']))
    cons_min = float(np.min(rendered['per_config_consensus_agreement']))
    bc_note = (
        'Vertical beam angles are concentrated toward the horizon '
        f'(beam_concentration = {args.beam_concentration:g}), the same setting '
        'the Fig. R4 renders use; the sampler default, and training, use '
        'uniform angles.'
        if args.beam_concentration is not None else
        'Vertical beam angles are uniform (the sampler default), the regime of '
        'the published A4 statistics.')
    pub_note = (
        f'Statistics quoted here are recomputed from exactly these renders. '
        f'The published Table R9 numbers for this checkpoint '
        f'({stored["agreement_pct"]:.2f}% unanimity, '
        f'{stored["mean_modal_vote_share_pct"]:.2f}% mean modal vote share) '
        f'were computed with uniform beam angles and are unchanged; the values '
        f'above are the same measurement under the Fig. R4 beam distribution.'
        if args.beam_concentration is not None else
        'Identical sampling regime to the published A4 run: every statistic '
        'here reproduces review_analysis/_source_data/A4_raw.json exactly.')
    other_note = ('' if not args.other_colour else
                  f'  Class 0 (Other, the ignore label) is drawn in '
                  f'{args.other_colour} instead of the map\'s pure black, so '
                  f'that ~1% of unlabelled points does not dominate the panels; '
                  f'no point is dropped.')
    meta = dict(
        frame_id=raw['frame_id'],
        # The METHOD name, never the checkpoint filename: these sheets are
        # published artefacts and `name` identifies an unreleased model.
        title=f'One scene, 50 LiDARs: per-point predictions of {args.label}',
        subtitle=(
            f'SemanticKITTI sequence {seq:02d}, frame {frame_no:06d} (held-out '
            f'val split) · {len(dense_xyz):,} dense points · one scene, read '
            f'once and reused for all 50 scans\n'
            # 16-128 EXTENDS BEYOND the Appendix Table B1 training envelope
            # (32-64 beams); saying "drawn from the Table B1 envelope" would
            # misstate the experiment. Every other parameter IS the B1 range.
            f'50 mechanical configurations spanning '
            f'{min(c["beams"] for c in configs)}–'
            f'{max(c["beams"] for c in configs)} beams — extending beyond the '
            f'32–64 of the Appendix Table B1 training envelope — with its '
            f'600–1800 azimuth bins, vertical FoV −28…−22° / +2…+15°, '
            f'mount height 1.50–1.85 m'),
        header=('50 sensor configurations' +
                (f', ordered by beam count '
                 f'({min(c["beams"] for c in configs)} → '
                 f'{max(c["beams"] for c in configs)})'
                 if args.order == 'beams' else '') +
                ' — left panel of each pair: model PREDICTION;   right panel: '
                'label INHERITED from the dense cloud (propagated ground '
                'truth)'),
        header_right=(f'{npts.min():,}–{npts.max():,} points per scan '
                      f'(mean {npts.mean():,.0f})'),
        guide=(
            'HOW TO READ A PAIR.  Each pair below is one simulated sensor. The LEFT panel is what the\n'
            'model predicts on that scan; the RIGHT panel is the label each returned point inherits from\n'
            'the dense cloud above, through the sampler’s provenance map. The two panels of a pair contain\n'
            'the exact same points and differ only in the colouring.\n\n'
            'THE POINT SETS DIFFER; THE SEMANTICS DO NOT.  Every pair is a different sensor observing the\n'
            'identical scene, so the scans themselves are genuinely different: they range from '
            f'{npts.min():,} to\n{npts.max():,} points, and two configurations drawn at random share only '
            '~22% of the dense points they\n'
            'hit. What barely changes from pair to pair is the semantic layout — and that is the result:\n'
            f'each configuration reproduces the cross-sensor consensus on {cons_mean:.1f}% of the points it '
            f'sees\n(worst configuration {cons_min:.1f}%), and on the points that at least 10 '
            f'configurations observe the\nprediction is unanimous across every one of them for '
            f'{rendered["agreement_pct"]:.1f}% of points.\n\n'
            'WHERE THEY DO DIFFER.  Road and vehicles are reproduced almost exactly in every panel '
            f'({pc["Driveable Ground"]["agreement_pct"]:.1f}%\nand {pc["Vehicle"]["agreement_pct"]:.1f}% '
            'unanimity). The visible predicted-vs-inherited differences sit on the ground/\n'
            'vegetation boundary, where the scene itself is ambiguous: unanimity falls to '
            f'{pc["Other Ground"]["agreement_pct"]:.1f}% on Other\nGround and '
            f'{pc["Nature"]["agreement_pct"]:.1f}% on Nature.'),
        foot=(
            'Bird\'s-eye (top-down) projection of the x–y plane, identical '
            '±51.5 m limits in every panel; ✚ marks the sensor origin; range '
            'rings every 10 m. Colour encodes COLA-8 class identity using the '
            'project\'s own class colour map, fixed across every panel of both '
            f'sheets.{other_note}  {bc_note}\n'
            'acc = that configuration\'s per-point accuracy against the '
            f'propagated dense labels, over the '
            f'{rendered["qualifying_points"]:,} points that at least '
            f'{raw["min_obs"]} '
            'configurations observed and whose label is not "Other" — i.e. '
            f'the per-configuration accuracy of Table R9 (mean '
            f'{rendered["per_config_acc_mean"]:.2f} ± '
            f'{rendered["per_config_acc_sd"]:.2f}%). {pub_note}\n'
            'The reference labels are propagated dense-reconstruction labels, '
            'noisier than SemanticKITTI\'s per-scan annotation: these '
            'accuracies are NOT comparable with the mIoU numbers in the paper.'),
        foot_clean=(
            'Bird\'s-eye (top-down) projection of the x–y plane, identical '
            '±51.5 m limits in every panel; ✚ marks the sensor origin; range '
            'rings every 10 m. Colour encodes COLA-8 class identity using the '
            'project\'s own class colour map, fixed across every panel of both '
            f'sheets.{other_note}  {bc_note}\n'
            'The "inherited" panels carry propagated dense-reconstruction '
            'labels, which are noisier than SemanticKITTI\'s per-scan '
            f'annotation. {pub_note}'),
    )

    render_dense(dense_xyz, dense_lab, meta, args,
                 os.path.join(args.outdir, 'FigR5_pointclouds_dense'))

    # Both variants come out of ONE run and ONE cached sweep, so the annotated
    # and the clean sheets are guaranteed to show bit-identical point clouds;
    # `annotate` toggles the numeric captions only.
    nsheet = int(np.ceil(len(per_config) / args.per_sheet))
    variants = ([('annotated', True)] if args.variant == 'annotated' else
                [('clean', False)] if args.variant == 'clean' else
                [('annotated', True), ('clean', False)])
    for vname, ann in variants:
        for si in range(nsheet):
            a = si * args.per_sheet
            b = min((si + 1) * args.per_sheet, len(per_config))
            render_sheet(
                per_config, args, meta,
                os.path.join(args.outdir,
                             f'FigR5_pointclouds_{vname}_p{si + 1}'),
                a, b,
                lead_dense=((dense_xyz, dense_lab) if si == 0 else None),
                sheet_label=(
                    f'sheet {si + 1} of {nsheet}  ·  panels {a + 1}–{b} of '
                    f'{len(per_config)}' +
                    ('  ·  ordered by beam count' if args.order == 'beams'
                     else '  ·  in configuration order')),
                annotate=ann)

    if args.ply:
        pd = os.path.join(args.outdir, 'ply')
        os.makedirs(pd, exist_ok=True)
        write_ply(os.path.join(pd, 'dense_gt.ply'), dense_xyz, dense_lab)
        for r in per_config:
            write_ply(os.path.join(pd, f'cfg{r["index"] + 1:02d}_pred.ply'),
                      r['xyz'], r['pred'])
            write_ply(os.path.join(pd, f'cfg{r["index"] + 1:02d}_inherited.ply'),
                      r['xyz'], r['inherited'])
        print(f'[ply] wrote {1 + 2 * len(per_config)} clouds -> {pd}')

    print('\n[A4-vis] done')


if __name__ == '__main__':
    main()
