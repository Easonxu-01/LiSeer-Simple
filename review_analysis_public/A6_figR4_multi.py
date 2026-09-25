"""A6: Response Fig. R4 — per-dataset solid-state scan demonstration.

One figure per dataset (SemanticKITTI / nuScenes / Waymo), 3 randomly chosen
frames per dataset, each row:

  raw real scan | dense reconstruction | mechanical (simulated) | Risley T=0.1/0.2/0.5 s

Every simulated sensor config is printed under its panel, so "which panels are
real data and which are generated" is unambiguous. Mechanical configs are rolled
randomly per row from the dataset's own training range.

The coverage-vs-integration-time curve is a property of the Risley
parameterisation, not of any dataset, so it is rendered ONCE as its own figure.

Determinism (see NEEDS_DECISION.md 2026-08-31): each dense frame is loaded once
and the SAME tensor is reused for every sampler call; _dense_sampling (whose
randperm would resample a different 500k subset per call) is never touched.

Inference only.
"""
import argparse, math, os, random, sys, time
import numpy as np
import torch
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from review_analysis.CODE5_risley_demo import build_model, to_xyz, MIN_BOUND, MAX_BOUND, GRID_SIZE
from review_analysis.A6_figR4 import RAMP, INK, INK2, INK3, SURFACE, ACCENT, CMAP, bev_panel

# ---- per-dataset sensor sampling ranges (from each dataset's train config) ----
SENSOR_RANGES = {
    'semantickitti': dict(beams=(32, 64), hres=(600, 1800), vlow=(-28, -22),
                          vup=(2, 15), height=(1.5, 1.85)),
    'nuscenes':      dict(beams=(16, 80), hres=(600, 1800), vlow=(-30, -15),
                          vup=(2, 15), height=(1.5, 1.85)),
    'waymo':         dict(beams=(32, 64), hres=(600, 1800), vlow=(-28, -22),
                          vup=(2, 15), height=(1.5, 1.85)),
}

SK_TO_CL = {0:0,1:0,10:3,11:6,13:3,15:3,16:3,18:3,20:3,30:5,31:5,32:5,40:1,
            44:1,48:7,49:7,50:2,51:2,52:2,60:1,70:4,71:4,72:4,80:6,81:6,99:6,
            252:3,253:5,254:5,255:5,256:3,257:3,258:3,259:3}


def _polar_feats(xyz, device):
    """Build the [N,8] PointsegMapping layout the sampler expects."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rho = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    pol = np.stack([rho, phi, z], axis=1)
    mn, mx, gs = MIN_BOUND.numpy(), MAX_BOUND.numpy(), GRID_SIZE.numpy()
    iv = (mx - mn) / gs
    pol_c = np.maximum(np.minimum(pol, mx - 1e-3), mn)
    gi = np.floor((pol_c - mn) / iv)
    rel = pol - ((gi + 0.5) * iv + mn)
    f = np.zeros((len(xyz), 8), dtype=np.float32)
    f[:, 0:3] = rel; f[:, 3] = rho; f[:, 4] = phi
    f[:, 5] = z; f[:, 6] = x; f[:, 7] = y
    t = lambda a, d=torch.float32: torch.from_numpy(
        np.ascontiguousarray(a)).to(device=device, dtype=d)
    return t(f), t(gi), t((pol_c - mn) / iv)


def _read_ply_or_npy(path):
    npy = path[:-4] + '.npy'
    if os.path.exists(npy):
        return np.load(npy).astype(np.float64)
    return np.asarray(o3d.io.read_point_cloud(path).points)


def list_frames(ds):
    """Return a list of (raw_path, dense_path, label_path, tag) candidates."""
    out = []
    if ds == 'semantickitti':
        root = 'data/semantickitti/sequences'
        for seq in ['00', '02', '05', '07', '09']:
            d = f'{root}/{seq}/refine'
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.endswith('.ply'):
                    stem = f[:-4]
                    out.append((f'{root}/{seq}/velodyne/{stem}.bin',
                                f'{d}/{f}',
                                f'{root}/{seq}/dense_labels/{stem}.label',
                                f'seq {seq} / {stem}'))
    elif ds == 'nuscenes':
        d = '/data1/nuscenes/samples/LIDAR_TOP/refine'
        raw = '/data1/nuscenes/samples/LIDAR_TOP'
        for f in sorted(os.listdir(d))[:4000]:
            if f.endswith('.ply'):
                # dense is "<token>.pcd.ply"; the raw sweep is "<token>.pcd.bin"
                out.append((f'{raw}/{f[:-4]}.bin', f'{d}/{f}', None, f[:-8][-22:]))
    elif ds == 'waymo':
        base = 'data/waymo/kitti_format/training'
        for f in sorted(os.listdir(f'{base}/refine'))[:4000]:
            if f.endswith('.ply'):
                stem = f[:-4]
                out.append((f'{base}/velodyne/{stem}.bin',
                            f'{base}/refine/{f}',
                            f'{base}/dense_labels/{stem}.label',
                            f'frame {stem}'))
    return out


def load_raw(path, ds):
    dims = {'semantickitti': 4, 'nuscenes': 5, 'waymo': 6}[ds]
    a = np.fromfile(path, dtype=np.float32)
    n = a.size // dims
    return a[:n * dims].reshape(n, dims)[:, :3].astype(np.float64)


def load_labels(path, n):
    if path is None or not os.path.exists(path):
        return np.zeros(n, dtype=np.int64)
    raw = np.fromfile(path, dtype=np.uint32) & 0xFFFF
    lut = np.zeros(max(SK_TO_CL) + 1, dtype=np.int64)
    for k, v in SK_TO_CL.items():
        lut[k] = v
    lab = lut[np.clip(raw, 0, len(lut) - 1)]
    if len(lab) < n:
        lab = np.pad(lab, (0, n - len(lab)))
    return lab[:n]


def build_full_model(device, sensor_cfg, beam_concentration):
    """Shell exposing the REAL _sampling (not a re-implementation).

    _sampling only needs these attributes; we bind the genuine unbound methods
    so the figure is produced by the same code path the model uses.
    """
    from projects.unilidar_plugin.pointocc_model.pointtpv_seg import PointTPV_Seg
    m = build_model(device)
    m.consistency = False
    m.dense_train = False
    m.sample_repeat_count = 1
    m.sensor_config = sensor_cfg
    m.beam_concentration = beam_concentration
    m._mixup_minority = PointTPV_Seg._mixup_minority.__get__(m)
    m._sampling = PointTPV_Seg._sampling.__get__(m)
    return m


def mech_sample(model, pts, gi, giv, lab, cfg):
    """Mechanical scan via the model's OWN _sampling (forced sensor config).

    We monkeypatch the module's random draws so the sampler uses exactly the
    config we report under the panel, then restore them. Everything else --
    ray construction, binning, z-buffer, occlusion, ring smoothing, range
    noise, dropout -- is the model's real code.
    """
    import projects.unilidar_plugin.pointocc_model.pointtpv_seg as M
    real_uniform, real_choice, real_sample_beams = (
        M.random.uniform, M.random.choice, M.sample_beams)
    seq = {'n': 0}

    def fake_uniform(a, b):
        seq['n'] += 1
        return {1: cfg['height'], 2: cfg['vlow'], 3: cfg['vup']}.get(seq['n'],
                                                                     real_uniform(a, b))

    M.random.uniform = fake_uniform
    M.random.choice = lambda seq_: cfg['hres']
    M.sample_beams = lambda lo, hi: cfg['beams']
    try:
        rp, rg, rgv, rl, _ = model._sampling([pts], [gi], [giv], [lab])
    finally:
        M.random.uniform, M.random.choice, M.sample_beams = (
            real_uniform, real_choice, real_sample_beams)
    return to_xyz(rp[0])


def roll_config(ds):
    """Roll every sensor parameter uniformly inside the dataset's range.

    Deliberately NOT the canonical 16/32/64/80 line counts that training's
    sample_beams draws from: this figure demonstrates that the formulation
    accepts arbitrary configurations, so beams is uniform over the range too.
    """
    r = SENSOR_RANGES[ds]
    return dict(
        beams=random.randint(r['beams'][0], r['beams'][1]),
        hres=random.randint(r['hres'][0], r['hres'][1]),
        vlow=round(random.uniform(*r['vlow']), 1),
        vup=round(random.uniform(*r['vup']), 1),
        height=round(random.uniform(*r['height']), 2))


def render_dataset(ds, args, device, model):
    frames = list_frames(ds)
    if not frames:
        print(f'[{ds}] no frames found — skipped')
        return
    rng = random.Random(args.seed)
    picks = rng.sample(frames, min(args.n_frames, len(frames)))
    T_LIST = args.T
    ncol = 3 + len(T_LIST)
    nrow = len(picks)

    fig = plt.figure(figsize=(3.62 * ncol, 4.30 * nrow), facecolor=SURFACE)
    gs = fig.add_gridspec(nrow, ncol, hspace=0.34, wspace=0.07,
                          left=0.012, right=0.895, top=0.895, bottom=0.028)

    for r, (raw_p, dense_p, lab_p, tag) in enumerate(picks):
        print(f'[{ds}] row {r+1}/{nrow}: {tag}')
        # Fail loudly: a missing raw sweep would otherwise render as a silently
        # empty "Real scan" panel (this bit nuScenes once — wrong suffix).
        if not os.path.exists(raw_p):
            raise FileNotFoundError(f'[{ds}] raw scan not found: {raw_p}')
        raw_xyz = load_raw(raw_p, ds)
        if len(raw_xyz) == 0:
            raise ValueError(f'[{ds}] raw scan is empty: {raw_p}')
        dense_xyz = _read_ply_or_npy(dense_p)
        pts, gi, giv = _polar_feats(dense_xyz, device)
        lab = torch.from_numpy(load_labels(lab_p, len(dense_xyz))).to(device)
        base_ptr = pts.data_ptr()

        cfg = roll_config(ds)
        full = build_full_model(device, dict(
            LiDAR_height=list(SENSOR_RANGES[ds]['height']),
            num_of_beams=list(SENSOR_RANGES[ds]['beams']),
            horizontal_angular_resolution=list(SENSOR_RANGES[ds]['hres']),
            lower_vertical_field_of_view_bound=list(SENSOR_RANGES[ds]['vlow']),
            upper_vertical_field_of_view_bound=list(SENSOR_RANGES[ds]['vup']),
        ), args.beam_concentration)
        mech = mech_sample(full, pts, gi, giv, lab, cfg)

        solid = {}
        for T in T_LIST:
            sp, _, _, _, inf = model.sample_solid_state(
                [pts], [gi], [giv], [lab], integration_time=T,
                footprint_scale=1.0, occlusion_mode='knn', seed=args.seed + r)
            assert pts.data_ptr() == base_ptr, 'input tensor was replaced!'
            solid[T] = (to_xyz(sp[0]), inf[0]['num_rays'])

        cols = [
            (raw_xyz, 'Real scan (dataset)',
             f'{len(raw_xyz):,} points  ·  measured', 0.16, None),
            (dense_xyz, 'Dense reconstruction',
             f'{len(dense_xyz):,} points  ·  reconstructed', 0.02, None),
            (mech, 'Mechanical spinning (simulated)',
             f'{len(mech):,} pts  ·  {cfg["beams"]}×{cfg["hres"]}, '
             f'FoV {cfg["vlow"]}°/{cfg["vup"]}°, h={cfg["height"]}m', 0.09, None),
        ]
        for T in T_LIST:
            sx, nray = solid[T]
            cols.append((sx, f'Solid-state Risley (simulated), $T$={T}s',
                         f'{len(sx):,} pts  ·  {nray:,} rays  ·  FoV 70.4°, '
                         f'h=1.73m', 0.16, 35.2))

        for c, (xyz, title, sub, s, fov) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            bev_panel(ax, xyz, title if r == 0 else '', sub,
                      s=s, show_fov=fov)
            if r != 0:
                ax.set_title('')
            if c == 0:
                ax.text(-0.035, 0.5, tag, transform=ax.transAxes, rotation=90,
                        va='center', ha='center', fontsize=8, color=INK2)

    sm = matplotlib.cm.ScalarMappable(
        cmap=CMAP, norm=matplotlib.colors.Normalize(-3, 2.5))
    cax = fig.add_axes([0.905, 0.028, 0.010, 0.867])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('height $z$ (m)', fontsize=9.5, color=INK2)
    cb.ax.tick_params(labelsize=8, colors=INK2, length=2)
    cb.outline.set_edgecolor('#dcdbd6'); cb.outline.set_linewidth(0.7)

    handles = [
        Line2D([], [], marker='o', ls='none', mfc=ACCENT, mec='white',
               ms=6, label='sensor origin'),
        Line2D([], [], color=ACCENT, lw=1.1, ls=(0, (4, 2.5)),
               label='solid-state FoV (70.4°)'),
        Line2D([], [], color='#e3e2de', lw=1.0, label='range rings (10–40 m)'),
    ]
    # Legend sits on its own band BELOW the title, never sharing its line.
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=9,
               frameon=False, labelcolor=INK2, handletextpad=0.7,
               columnspacing=2.4, bbox_to_anchor=(0.45, 0.947))

    pretty = {'semantickitti': 'SemanticKITTI', 'nuscenes': 'nuScenes',
              'waymo': 'Waymo'}[ds]
    fig.suptitle(f'{pretty}: simulating mechanical and non-repetitive '
                 f'solid-state scans from the same dense reconstruction',
                 fontsize=14, color=INK, y=0.982, fontweight='medium', x=0.45)

    for ext in ('png', 'pdf'):
        p = f'{args.outdir}/FigR4_{ds}.{ext}'
        fig.savefig(p, dpi=args.dpi, facecolor=SURFACE, bbox_inches='tight')
        print(f'[{ds}] saved {p}')
    plt.close(fig)


def render_coverage(args, device, model):
    """Dataset-independent: a property of the Risley parameterisation."""
    from review_analysis.A6_figR4 import coverage_curve
    ds_root = 'data/semantickitti/sequences/00'
    xyz = _read_ply_or_npy(f'{ds_root}/refine/000000.ply')
    pts, gi, giv = _polar_feats(xyz, device)
    lab = torch.from_numpy(
        load_labels(f'{ds_root}/dense_labels/000000.label', len(xyz))).to(device)
    covs, _ = coverage_curve(model, pts, gi, giv, lab)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    tt = np.arange(1, len(covs) + 1) * 0.1
    ax.plot(tt, covs * 100, '-', lw=2.2, color=RAMP[4], zorder=3)
    ax.plot(tt, covs * 100, 'o', ms=5, mfc=RAMP[4], mec='white', mew=1.0,
            zorder=4)
    i8 = int(np.argmin(np.abs(tt - 0.8)))
    ax.annotate(f'{covs[i8]*100:.1f}% at 0.8 s',
                xy=(tt[i8], covs[i8] * 100), xytext=(0.55, 0.26),
                textcoords='axes fraction', fontsize=10, color=INK2, ha='left',
                arrowprops=dict(arrowstyle='-', color=INK3, lw=0.9,
                                shrinkA=2, shrinkB=5))
    ax.set_xlabel('cumulative integration time (s)', fontsize=10.5, color=INK2)
    ax.set_ylabel('FoV coverage (%)', fontsize=10.5, color=INK2)
    ax.set_ylim(max(0, covs.min() * 100 - 8), 103)
    ax.grid(True, color='#ececE8', lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, colors=INK2, length=3)
    for sp in ax.spines.values():
        sp.set_edgecolor('#dcdbd6'); sp.set_linewidth(0.8)
    ax.set_title('Non-repetitive scan: FoV coverage grows with integration time',
                 fontsize=12, color=INK, pad=10, fontweight='medium')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        p = f'{args.outdir}/FigR4_coverage.{ext}'
        fig.savefig(p, dpi=args.dpi, facecolor=SURFACE, bbox_inches='tight')
        print(f'[coverage] saved {p}')
    plt.close(fig)
    with open(f'{args.outdir}/FigR4_coverage.txt', 'w') as f:
        for i, c in enumerate(covs):
            f.write(f'{0.1*(i+1):.1f}s: {c*100:.2f}%\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['semantickitti', 'nuscenes', 'waymo'])
    ap.add_argument('--n-frames', type=int, default=3)
    ap.add_argument('--T', type=float, nargs='+', default=[0.1, 0.2, 0.5])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--outdir', default='figs/code5')
    ap.add_argument('--dpi', type=int, default=360)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--beam-concentration', type=float, default=2.6,
                    help='warp vertical beams toward the horizon (1.0 = the '
                         'original uniform spacing). Applied inside the real '
                         'sampler via its beam_concentration attribute.')
    ap.add_argument('--skip-coverage', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    model = build_model(device)

    for ds in args.datasets:
        t0 = time.time()
        try:
            render_dataset(ds, args, device, model)
            print(f'[{ds}] done in {time.time()-t0:.0f}s')
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'[{ds}] FAILED: {e}')

    if not args.skip_coverage:
        render_coverage(args, device, model)
    print('[A6] all done')


if __name__ == '__main__':
    main()
