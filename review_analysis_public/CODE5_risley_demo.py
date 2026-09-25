"""CODE-5 test harness + Fig R4 renderer (Risley non-repetitive scan prototype).

Runs regression tests B/C/D from the G1 doc and renders the BEV comparison:
dense cloud | mechanical 64-line | three solid-state configs.

Inference only. GPU 0 (analysis card) by default.
"""
import argparse, math, os, sys, time
import numpy as np
import torch
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIN_BOUND = torch.tensor([0, -3.1415926, -3.4], dtype=torch.float32)
MAX_BOUND = torch.tensor([51.2, 3.1415926, 3], dtype=torch.float32)
GRID_SIZE = torch.tensor([480, 360, 32], dtype=torch.float32)

SK_TO_CL = {0:0,1:0,10:3,11:6,13:3,15:3,16:3,18:3,20:3,30:5,31:5,32:5,40:1,
            44:1,48:7,49:7,50:2,51:2,52:2,60:1,70:4,71:4,72:4,80:6,81:6,99:6,
            252:3,253:5,254:5,255:5,256:3,257:3,258:3,259:3}


def load_dense_frame(seq, frame, device):
    """Build a [N,8] PointsegMapping-layout tensor from a dense refine frame."""
    root = f'data/semantickitti/sequences/{seq:02d}'
    ply = f'{root}/refine/{frame:06d}.ply'
    npy = ply[:-4] + '.npy'
    if os.path.exists(npy):
        xyz = np.load(npy).astype(np.float64)
    else:
        xyz = np.asarray(o3d.io.read_point_cloud(ply).points)
    lab_path = f'{root}/dense_labels/{frame:06d}.label'
    raw = np.fromfile(lab_path, dtype=np.uint32) & 0xFFFF
    lut = np.zeros(max(SK_TO_CL) + 1, dtype=np.int64)
    for k, v in SK_TO_CL.items():
        lut[k] = v
    labels = lut[np.clip(raw, 0, len(lut) - 1)]
    n = min(len(xyz), len(labels))
    xyz, labels = xyz[:n], labels[:n]

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rho = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    pol = np.stack([rho, phi, z], axis=1)
    mn, mx = MIN_BOUND.numpy(), MAX_BOUND.numpy()
    gs = GRID_SIZE.numpy()
    intervals = (mx - mn) / gs
    pol_c = np.maximum(np.minimum(pol, mx - 1e-3), mn)
    gi = np.floor((pol_c - mn) / intervals)
    centers = (gi + 0.5) * intervals + mn
    rel = pol - centers

    feat = np.zeros((n, 8), dtype=np.float32)
    feat[:, 0:3] = rel
    feat[:, 3] = rho
    feat[:, 4] = phi
    feat[:, 5] = z
    feat[:, 6] = x
    feat[:, 7] = y
    t = lambda a, d=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=d)
    return (t(feat), t(gi), t((pol_c - mn) / intervals),
            t(labels, torch.long))


def build_model(device):
    """Minimal shell exposing the sampler methods without loading weights."""
    from projects.unilidar_plugin.pointocc_model.pointtpv_seg import PointTPV_Seg

    class Shell(torch.nn.Module):
        pass
    m = Shell()
    m.register_buffer('min_bound', MIN_BOUND.clone())
    m.register_buffer('max_bound', MAX_BOUND.clone())
    m.register_buffer('grid_size', GRID_SIZE.clone())
    m.register_buffer('grid_size_vox', GRID_SIZE.clone())
    m.sample_solid_state = PointTPV_Seg.sample_solid_state.__get__(m)
    return m.to(device)


def to_xyz(feat):
    a = feat.detach().float().cpu().numpy()
    return np.concatenate((a[:, 6:8], a[:, 5:6]), axis=1)


def save_ply(feat, path):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(to_xyz(feat).astype(np.float64))
    o3d.io.write_point_cloud(path, pc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--outdir', default='figs/code5')
    ap.add_argument('--scan-scale', type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(0); np.random.seed(0)

    print(f'[load] seq {args.seq:02d} frame {args.frame:06d}')
    pts, gi, giv, lab = load_dense_frame(args.seq, args.frame, device)
    print(f'[load] dense points: {pts.shape[0]:,}')
    model = build_model(device)

    T_LIST = [0.1, 0.2, 0.5]
    results = {}

    # ---------------- TEST C: point-count magnitude, per T ----------------
    print('\n=== solid-state sampling ===')
    for T in T_LIST:
        t0 = time.time()
        rp, rg, rgv, rl, info = model.sample_solid_state(
            [pts], [gi], [giv], [lab],
            integration_time=T, footprint_scale=args.scan_scale,
            occlusion_mode='knn', seed=0)
        dt = time.time() - t0
        R = info[0]['num_rays']
        n = rp[0].shape[0]
        results[T] = (rp[0], rl[0], info[0])
        print(f'  T={T}s  R={R:>7,}  hit={n:>7,}  hit/R={n/R:.3f}  '
              f'footprint={math.degrees(info[0]["footprint_angle"]):.4f}deg  {dt:.1f}s')
        save_ply(rp[0], f'{args.outdir}/solid_T{T}.ply')

    # ---------------- TEST B: non-repetitiveness across frames ----------------
    print('\n=== TEST B: non-repetitive across frames ===')
    ray_sets = []
    for f in range(10):
        _, _, _, _, inf = model.sample_solid_state(
            [pts], [gi], [giv], [lab], integration_time=0.1, seed=None)
        ray_sets.append(inf[0]['ray_directions'])
    ident = sum(1 for i in range(len(ray_sets)) for j in range(i + 1, len(ray_sets))
                if np.array_equal(ray_sets[i], ray_sets[j]))
    print(f'  identical ray-set pairs among 10 frames: {ident} (expect 0)')

    # cumulative angular coverage must grow monotonically
    # Coverage denominator must be the CIRCULAR FoV, not the square bounding
    # box: the cone only reaches |theta| <= fov_half, which is ~61% of the
    # square. Using the square makes a fully-saturated pattern read as ~61%.
    NB = 200
    lim = math.radians(40.0)
    fov_r = math.radians(35.2)
    ii, jj = np.meshgrid(np.arange(NB), np.arange(NB), indexing='ij')
    cu = (ii + 0.5) / NB * 2 * lim - lim
    cv = (jj + 0.5) / NB * 2 * lim - lim
    in_fov = (cu * cu + cv * cv) <= fov_r * fov_r
    n_reachable = int(in_fov.sum())
    grid = np.zeros((NB, NB), dtype=bool)
    covs = []
    for rs in ray_sets:
        th = np.arccos(np.clip(rs[:, 0], -1, 1))
        ph = np.arctan2(rs[:, 2], rs[:, 1])
        u, v = th * np.cos(ph), th * np.sin(ph)
        bi = np.clip(((u + lim) / (2 * lim) * NB).astype(int), 0, NB - 1)
        bj = np.clip(((v + lim) / (2 * lim) * NB).astype(int), 0, NB - 1)
        grid[bi, bj] = True
        covs.append((grid & in_fov).sum() / n_reachable)
    mono = all(covs[i] <= covs[i + 1] + 1e-12 for i in range(len(covs) - 1))
    print(f'  cumulative coverage: {covs[0]:.4f} -> {covs[-1]:.4f}, monotone={mono}')

    # ---------------- TEST D: output contract ----------------
    print('\n=== TEST D: output contract ===')
    feat, info = results[0.5][0], results[0.5][2]
    x, y, z = feat[:, 6], feat[:, 7], feat[:, 5]
    rho_c = torch.sqrt(x * x + y * y)
    phi_c = torch.atan2(y, x)
    ok_rho = torch.allclose(feat[:, 3], rho_c, atol=1e-4)
    ok_phi = torch.allclose(feat[:, 4], phi_c, atol=1e-4)
    mn = MIN_BOUND.to(device); mx = MAX_BOUND.to(device); gs = GRID_SIZE.to(device)
    iv = (mx - mn) / gs
    pol = torch.stack([rho_c, phi_c, z], dim=1)
    pol_c = torch.maximum(torch.minimum(pol, mx - 1e-3), mn)
    gi_c = torch.floor((pol_c - mn) / iv)
    rel_c = pol - ((gi_c + 0.5) * iv + mn)
    ok_rel = torch.allclose(feat[:, 0:3], rel_c, atol=1e-4)
    print(f'  col3==rho: {ok_rho} | col4==phi: {ok_phi} | col0:3==rel_polar: {ok_rel}')

    # ---------------- mechanical reference (random params) ----------------
    print('\n=== mechanical reference (random roll) ===')
    import random as _r
    beams = 64; hres = 1800
    vlow, vup = -24.9, 2.0
    va = torch.deg2rad(torch.linspace(vlow, vup, beams, device=device))
    hstep = 2 * math.pi / hres
    ha = torch.arange(hres, device=device) * hstep + _r.random() * hstep
    vv, hh = torch.meshgrid(va, ha, indexing='ij')
    rays = torch.stack([torch.cos(vv) * torch.cos(hh),
                        torch.cos(vv) * torch.sin(hh),
                        torch.sin(vv)], dim=-1).reshape(-1, 3)
    xyz_all = torch.stack([pts[:, 6], pts[:, 7], pts[:, 5]], dim=1)
    gm = ((lab == 1) | (lab == 7)) & (pts[:, 3] < 20.0)
    gz = pts[gm, 5].median() if gm.any() else pts[:, 5].kthvalue(
        max(1, int(0.02 * pts.shape[0]))).values
    orig = torch.tensor([0., 0., 0.], device=device); orig[2] = gz + 1.73
    vec = xyz_all - orig; rng = torch.norm(vec, dim=1)
    keep = (rng > 1e-8) & (rng <= 80.0)
    vec, rng = vec[keep], rng[keep]
    az = torch.atan2(vec[:, 1], vec[:, 0]) % (2 * math.pi)
    el = torch.asin(torch.clamp(vec[:, 2] / rng, -1, 1))
    ab = torch.round((az - ha[0]) / hstep).long() % hres
    vstep = (va[-1] - va[0]).item() / (beams - 1)
    eb = torch.round((el - va[0]) / vstep).long().clamp_(0, beams - 1)
    fb = eb * hres + ab
    d = torch.sum(vec * rays[fb], dim=1)
    from torch_scatter import scatter_min
    ok = d > 0
    md, mi = scatter_min(d[ok], fb[ok], dim=0, dim_size=beams * hres)
    hitm = torch.isfinite(md) & (mi >= 0) & (mi < ok.sum())
    mech_xyz = (orig.unsqueeze(0) + md[hitm].unsqueeze(1) * rays[hitm.nonzero().squeeze(1)]
                ).cpu().numpy()
    print(f'  mechanical 64x1800: {len(mech_xyz):,} points')

    # ---------------- Fig R4: BEV panels ----------------
    print('\n=== rendering BEV ===')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(26, 5.6))
    LIM = 50

    def bev(ax, xyz, title, s=0.06):
        ax.scatter(xyz[:, 0], xyz[:, 1], s=s, c=xyz[:, 2],
                   cmap='viridis', vmin=-3, vmax=3, linewidths=0)
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        ax.set_aspect('equal'); ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    bev(axes[0], to_xyz(pts), f'Dense reconstruction\n{pts.shape[0]:,} pts', s=0.02)
    bev(axes[1], mech_xyz, f'Mechanical 64-line\n{len(mech_xyz):,} pts')
    for k, T in enumerate(T_LIST):
        f_ = results[T][0]
        bev(axes[2 + k], to_xyz(f_),
            f'Solid-state (Risley) T={T}s\n{f_.shape[0]:,} pts')
    plt.tight_layout()
    p = f'{args.outdir}/figR4_bev.png'
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  saved {p}')

    # angular rose pattern + coverage curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    rs = results[0.1][2]['ray_directions']
    th = np.arccos(np.clip(rs[:, 0], -1, 1)); ph = np.arctan2(rs[:, 2], rs[:, 1])
    axes[0].scatter(np.degrees(th * np.cos(ph)), np.degrees(th * np.sin(ph)),
                    s=0.02, c='k', linewidths=0)
    axes[0].set_aspect('equal'); axes[0].set_xlabel('u (deg)'); axes[0].set_ylabel('v (deg)')
    axes[0].set_title('Risley angular pattern, T=0.1 s')
    axes[1].plot([0.1 * (i + 1) for i in range(len(covs))],
                 [c * 100 for c in covs], 'o-')
    axes[1].set_xlabel('cumulative integration time (s)')
    axes[1].set_ylabel('FoV coverage (%)')
    axes[1].set_title('Coverage vs integration time'); axes[1].grid(alpha=.3)
    plt.tight_layout()
    p2 = f'{args.outdir}/figR4_pattern_coverage.png'
    plt.savefig(p2, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  saved {p2}')
    save_ply(pts, f'{args.outdir}/dense.ply')
    print('\n[done]')


if __name__ == '__main__':
    main()
