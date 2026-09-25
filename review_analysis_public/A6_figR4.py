"""A6: Response Fig. R4 — solid-state (Risley) non-repetitive scan demonstration.

Panels: dense reconstruction | mechanical 64-line | Risley T=0.1/0.2/0.5 s,
all in BEV, plus an inset row with the angular pattern and the coverage curve.

Reads the dense frame ONCE and reuses the same tensor for every sampler call
(see NEEDS_DECISION.md 2026-08-31: _dense_sampling's randperm would otherwise
resample a different 500k subset per call).

Inference only. Run on the analysis GPU.
"""
import argparse, math, os, sys, time
import numpy as np
import torch
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from review_analysis.CODE5_risley_demo import (
    load_dense_frame, build_model, to_xyz, save_ply, MIN_BOUND, MAX_BOUND, GRID_SIZE)

# Sequential blue ramp (validated palette, light mode), light -> dark
RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
INK, INK2, INK3 = '#0b0b0b', '#52514e', '#8a8984'
# Pure white, not an off-white: these figures sit beside locally-produced
# #ffffff ones in the response letter, where any tint reads as a grey block.
SURFACE = '#ffffff'
ACCENT = '#eb6834'   # categorical slot 2, for the sensor origin / FoV edge

CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('seq_blue', RAMP)


def mechanical_reference(pts, lab, device, beams=64, hres=1800,
                         vlow=-24.9, vup=2.0, lidar_height=1.73, seed=0):
    """64-line spinning reference with a random azimuth phase (roll)."""
    from torch_scatter import scatter_min
    g = torch.Generator(device='cpu').manual_seed(seed)
    va = torch.deg2rad(torch.linspace(vlow, vup, beams, device=device))
    hstep = 2 * math.pi / hres
    phase = float(torch.rand((), generator=g)) * hstep
    ha = torch.arange(hres, device=device) * hstep + phase
    vv, hh = torch.meshgrid(va, ha, indexing='ij')
    rays = torch.stack([torch.cos(vv) * torch.cos(hh),
                        torch.cos(vv) * torch.sin(hh),
                        torch.sin(vv)], dim=-1).reshape(-1, 3)
    xyz = torch.stack([pts[:, 6], pts[:, 7], pts[:, 5]], dim=1)
    gm = ((lab == 1) | (lab == 7)) & (pts[:, 3] < 20.0)
    gz = (pts[gm, 5].median() if gm.any()
          else pts[:, 5].kthvalue(max(1, int(0.02 * pts.shape[0]))).values)
    orig = torch.stack([torch.zeros((), device=device), torch.zeros((), device=device),
                        gz + lidar_height])
    vec = xyz - orig
    rng = torch.norm(vec, dim=1)
    keep = (rng > 1e-8) & (rng <= 80.0)
    vec, rng = vec[keep], rng[keep]
    az = torch.atan2(vec[:, 1], vec[:, 0]) % (2 * math.pi)
    el = torch.asin(torch.clamp(vec[:, 2] / rng, -1, 1))
    ab = torch.round((az - phase) / hstep).long() % hres
    vstep = (va[-1] - va[0]).item() / (beams - 1)
    eb = torch.round((el - va[0]) / vstep).long().clamp_(0, beams - 1)
    fb = eb * hres + ab
    d = torch.sum(vec * rays[fb], dim=1)
    ok = d > 0
    md, mi = scatter_min(d[ok], fb[ok], dim=0, dim_size=beams * hres)
    hit = torch.isfinite(md) & (mi >= 0) & (mi < int(ok.sum()))
    sel = hit.nonzero(as_tuple=False).squeeze(1)
    return (orig.unsqueeze(0) + md[hit].unsqueeze(1) * rays[sel]).cpu().numpy()


def coverage_curve(model, pts, gi, giv, lab, n_frames=10, T=0.1,
                   fov_half_deg=35.2, nb=240, seed=0):
    """Cumulative angular coverage, denominator = circular FoV (not its bbox).

    `seed` fixes the Risley starting phase of every accumulated frame. With
    seed=None the phase is redrawn per call and the curve is NOT reproducible:
    three consecutive unseeded runs gave 97.1 / 95.0 / 92.1 % at 0.5 s. A
    published figure has to be reproducible, so this defaults to a fixed seed;
    pass seed=None only to inspect the spread.
    """
    lim = math.radians(fov_half_deg + 5.0)
    r = math.radians(fov_half_deg)
    ii, jj = np.meshgrid(np.arange(nb), np.arange(nb), indexing='ij')
    cu = (ii + 0.5) / nb * 2 * lim - lim
    cv = (jj + 0.5) / nb * 2 * lim - lim
    in_fov = (cu * cu + cv * cv) <= r * r
    n_reach = int(in_fov.sum())
    grid = np.zeros((nb, nb), dtype=bool)
    covs, first_rays = [], None
    for f in range(n_frames):
        # Distinct phase per accumulated frame (otherwise every frame would
        # retrace the same rays and coverage would never grow), but a
        # deterministic sequence of them.
        _, _, _, _, inf = model.sample_solid_state(
            [pts], [gi], [giv], [lab], integration_time=T,
            seed=None if seed is None else seed + f)
        rs = inf[0]['ray_directions']
        if first_rays is None:
            first_rays = rs
        th = np.arccos(np.clip(rs[:, 0], -1, 1))
        ph = np.arctan2(rs[:, 2], rs[:, 1])
        u, v = th * np.cos(ph), th * np.sin(ph)
        bi = np.clip(((u + lim) / (2 * lim) * nb).astype(int), 0, nb - 1)
        bj = np.clip(((v + lim) / (2 * lim) * nb).astype(int), 0, nb - 1)
        grid[bi, bj] = True
        covs.append((grid & in_fov).sum() / n_reach)
    return np.array(covs), first_rays


def bev_panel(ax, xyz, title, subtitle, lim=48, s=0.09, show_fov=None,
              origin_xy=(0.0, 0.0)):
    ax.set_facecolor(SURFACE)
    # recessive range rings
    for r in (10, 20, 30, 40):
        ax.add_patch(Circle(origin_xy, r, fill=False, ec='#e3e2de',
                            lw=0.6, zorder=0))
        ax.text(origin_xy[0] + r * 0.7071, origin_xy[1] + r * 0.7071, f'{r}m',
                fontsize=6, color=INK3, ha='left', va='bottom', zorder=1)
    if show_fov is not None:
        half = show_fov
        ax.add_patch(Wedge(origin_xy, lim * 0.97, -half, half, width=lim * 0.97,
                           facecolor='none', ec=ACCENT,
                           lw=0.9, ls=(0, (5, 3)), zorder=1))
    z = xyz[:, 2]
    ax.scatter(xyz[:, 0], xyz[:, 1], s=s, c=np.clip(z, -3, 2.5),
               cmap=CMAP, vmin=-3, vmax=2.5, linewidths=0, zorder=2, rasterized=True)
    ax.plot(*origin_xy, marker='o', ms=4.5, mfc=ACCENT, mec='white',
            mew=1.0, zorder=5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#dcdbd6'); sp.set_linewidth(0.8)
    ax.set_title(title, fontsize=10.5, color=INK, pad=7, fontweight='medium')
    ax.text(0.5, -0.045, subtitle, transform=ax.transAxes, ha='center',
            va='top', fontsize=8.5, color=INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--outdir', default='figs/code5')
    ap.add_argument('--T', type=float, nargs='+', default=[0.1, 0.2, 0.5])
    ap.add_argument('--cache', default='',
                    help='npz path; reuse sampled results so pure layout '
                         'edits re-render without re-running the sampler')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(0); np.random.seed(0)

    cache_ok = args.cache and os.path.exists(args.cache)
    if cache_ok:
        print(f'[A6] reusing cached sampling from {args.cache}')
        z = np.load(args.cache, allow_pickle=True)
        dense_xyz = z['dense_xyz']
        mech = z['mech']
        covs = z['covs']
        rays0 = z['rays0']
        solid_xyz = {float(T): z[f'solid_{T}'] for T in args.T}
        solid_meta = {float(T): tuple(z[f'meta_{T}']) for T in args.T}
        n_dense = int(z['n_dense'])
    else:
        print(f'[A6] loading seq {args.seq:02d} frame {args.frame:06d}')
        pts, gi, giv, lab = load_dense_frame(args.seq, args.frame, device)
        base_ptr = pts.data_ptr()
        print(f'[A6] dense points: {pts.shape[0]:,} '
              f'(tensor resident, ptr={base_ptr:#x})')
        model = build_model(device)

        solid_xyz, solid_meta = {}, {}
        for T in args.T:
            t0 = time.time()
            rp, _, _, _, inf = model.sample_solid_state(
                [pts], [gi], [giv], [lab], integration_time=T,
                footprint_scale=1.0, occlusion_mode='knn', seed=0)
            # A4 determinism guard: every call must see the SAME tensor.
            assert pts.data_ptr() == base_ptr, 'input tensor was replaced!'
            solid_xyz[T] = to_xyz(rp[0])
            solid_meta[T] = (rp[0].shape[0], inf[0]['num_rays'],
                             inf[0]['footprint_angle'])
            print(f'  T={T}s  R={inf[0]["num_rays"]:,}  hit={rp[0].shape[0]:,}  '
                  f'({time.time()-t0:.0f}s)')
            save_ply(rp[0], f'{args.outdir}/figR4_solid_T{T}.ply')

        mech = mechanical_reference(pts, lab, device)
        print(f'  mechanical 64x1800: {len(mech):,}')
        covs, rays0 = coverage_curve(model, pts, gi, giv, lab)
        print(f'  coverage {covs[0]*100:.1f}% -> {covs[-1]*100:.2f}%')
        dense_xyz = to_xyz(pts)
        n_dense = pts.shape[0]
        if args.cache:
            np.savez_compressed(
                args.cache, dense_xyz=dense_xyz, mech=mech, covs=covs,
                rays0=rays0, n_dense=n_dense,
                **{f'solid_{T}': solid_xyz[T] for T in args.T},
                **{f'meta_{T}': np.array(solid_meta[T]) for T in args.T})
            print(f'[A6] cached sampling -> {args.cache}')

    # ---------------- figure ----------------
    n_solid = len(args.T)
    fig = plt.figure(figsize=(4.05 * (2 + n_solid), 6.9), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2 + n_solid, height_ratios=[3.05, 1.0],
                          hspace=0.30, wspace=0.09,
                          left=0.014, right=0.986, top=0.915, bottom=0.075)

    ax = fig.add_subplot(gs[0, 0])
    bev_panel(ax, dense_xyz, 'Dense reconstruction',
              f'{n_dense:,} points  ·  input scene', s=0.02)
    ax.text(-0.02, 1.045, 'a', transform=ax.transAxes, fontsize=13,
            fontweight='bold', color=INK, va='bottom')

    ax = fig.add_subplot(gs[0, 1])
    bev_panel(ax, mech, 'Mechanical spinning, 64-line',
              f'{len(mech):,} points  ·  360° rings')
    ax.text(-0.02, 1.045, 'b', transform=ax.transAxes, fontsize=13,
            fontweight='bold', color=INK, va='bottom')

    for k, T in enumerate(args.T):
        ax = fig.add_subplot(gs[0, 2 + k])
        n_hit, n_ray, _ = solid_meta[T]
        bev_panel(ax, solid_xyz[T], f'Solid-state (Risley), $T$ = {T} s',
                  f'{int(n_hit):,} points  ·  {int(n_ray):,} rays',
                  show_fov=35.2, s=0.16)
        if k == 0:
            ax.text(-0.02, 1.045, 'c', transform=ax.transAxes, fontsize=13,
                    fontweight='bold', color=INK, va='bottom')

    # --- inset: angular pattern ---
    axp = fig.add_subplot(gs[1, 0])
    axp.set_facecolor(SURFACE)
    th = np.arccos(np.clip(rays0[:, 0], -1, 1))
    ph = np.arctan2(rays0[:, 2], rays0[:, 1])
    axp.scatter(np.degrees(th * np.cos(ph)), np.degrees(th * np.sin(ph)),
                s=0.012, c=RAMP[4], linewidths=0, rasterized=True)
    axp.add_patch(Circle((0, 0), 35.2, fill=False, ec=ACCENT, lw=1.0,
                         ls=(0, (5, 3))))
    axp.set_aspect('equal')
    axp.set_xlim(-42, 42); axp.set_ylim(-42, 42)
    axp.set_xlabel('$u$ (deg)', fontsize=8.5, color=INK2)
    axp.set_ylabel('$v$ (deg)', fontsize=8.5, color=INK2)
    axp.tick_params(labelsize=7.5, colors=INK2, length=3)
    for sp in axp.spines.values():
        sp.set_edgecolor('#dcdbd6'); sp.set_linewidth(0.8)
    axp.set_title('Angular pattern ($T$ = 0.1 s)', fontsize=9.5, color=INK, pad=5)
    axp.text(-0.16, 1.10, 'd', transform=axp.transAxes, fontsize=13,
             fontweight='bold', color=INK, va='bottom')

    # --- inset: coverage curve ---
    axc = fig.add_subplot(gs[1, 1:3])
    axc.set_facecolor(SURFACE)
    tt = np.arange(1, len(covs) + 1) * 0.1
    axc.plot(tt, covs * 100, '-', lw=2.0, color=RAMP[4], zorder=3)
    axc.plot(tt, covs * 100, 'o', ms=4.5, mfc=RAMP[4], mec='white', mew=0.9,
             zorder=4)
    i08 = int(np.argmin(np.abs(tt - 0.8)))
    axc.annotate(f'{covs[i08]*100:.1f}% at 0.8 s',
                 xy=(tt[i08], covs[i08] * 100), xytext=(0.60, 0.24),
                 textcoords='axes fraction', fontsize=8.5, color=INK2,
                 ha='left',
                 arrowprops=dict(arrowstyle='-', color=INK3, lw=0.8,
                                 shrinkA=2, shrinkB=4,
                                 connectionstyle='angle,angleA=0,angleB=90,rad=0'))
    axc.set_xlabel('cumulative integration time (s)', fontsize=8.5, color=INK2)
    axc.set_ylabel('FoV coverage (%)', fontsize=8.5, color=INK2)
    axc.set_ylim(max(0, covs.min() * 100 - 8), 103)
    axc.grid(True, color='#ececE8', lw=0.7, zorder=0)
    axc.set_axisbelow(True)
    axc.tick_params(labelsize=7.5, colors=INK2, length=3)
    for sp in axc.spines.values():
        sp.set_edgecolor('#dcdbd6'); sp.set_linewidth(0.8)
    axc.set_title('Coverage grows with integration time', fontsize=9.5,
                  color=INK, pad=5)
    axc.text(-0.075, 1.10, 'e', transform=axc.transAxes, fontsize=13,
             fontweight='bold', color=INK, va='bottom')

    # --- caption block ---
    axt = fig.add_subplot(gs[1, 3:])
    axt.axis('off')
    axt.text(0.0, 1.02,
             'Same scene, three sensor geometries',
             fontsize=10, color=INK, va='top', fontweight='medium')
    axt.text(0.0, 0.855,
             'The point-centric formulation consumes only a set of ray\n'
             'directions. Swapping the mechanical angular grid (b) for a\n'
             'Risley-prism trajectory (c) leaves the depth reduction,\n'
             'occlusion and output stages unchanged.',
             fontsize=8.4, color=INK2, va='top', linespacing=1.5)
    # Legend below the caption block. Keep every artist INSIDE the axes:
    # clip_on=False artists outside the axes make bbox_inches='tight'
    # expand the whole canvas and leave a large empty band.
    handles = [
        Line2D([], [], marker='o', ls='none', mfc=ACCENT, mec='white',
               ms=6, label='sensor origin'),
        Line2D([], [], color=ACCENT, lw=1.1, ls=(0, (4, 2.5)),
               label='solid-state FoV (70.4°)'),
    ]
    axt.legend(handles=handles, loc='lower left', fontsize=8,
               frameon=False, labelcolor=INK2, handletextpad=0.7,
               borderaxespad=0.0, handlelength=1.8)

    sm = matplotlib.cm.ScalarMappable(
        cmap=CMAP, norm=matplotlib.colors.Normalize(-3, 2.5))
    cax = fig.add_axes([0.905, 0.115, 0.008, 0.115])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('height $z$ (m)', fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2, length=2)
    cb.outline.set_edgecolor('#dcdbd6'); cb.outline.set_linewidth(0.7)

    fig.suptitle('Simulating a non-repetitive solid-state scan from the same '
                 'dense reconstruction', fontsize=12.5, color=INK,
                 y=0.975, fontweight='medium')

    for ext, dpi in (('png', 220), ('pdf', 220)):
        p = f'{args.outdir}/FigR4_solid_state.{ext}'
        fig.savefig(p, dpi=dpi, facecolor=SURFACE, bbox_inches='tight')
        print(f'[A6] saved {p}')
    plt.close(fig)

    with open(f'{args.outdir}/FigR4_stats.txt', 'w') as f:
        f.write(f'frame: seq{args.seq:02d}/{args.frame:06d}\n')
        f.write(f'dense points: {n_dense}\n')
        f.write(f'mechanical 64x1800: {len(mech)}\n')
        for T in args.T:
            nh, nr, fa = solid_meta[T]
            f.write(f'solid T={T}: rays={int(nr)} hits={int(nh)} '
                    f'footprint_deg={math.degrees(fa):.4f}\n')
        for i, c in enumerate(covs):
            f.write(f'coverage @ {0.1*(i+1):.1f}s: {c*100:.2f}%\n')
    print('[A6] done')


if __name__ == '__main__':
    main()
