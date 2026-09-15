"""Prototype sampling path for non-repetitive solid-state (Risley-prism) scanning.

The point-centric sampler consumes only a set of ray directions, so moving from a
spinning LiDAR to a Livox-type prism scanner replaces exactly three grid-dependent
operations and leaves the rest of the sampler alone:

================================  ===========================================
replaced                          reused unchanged
================================  ===========================================
ray construction                  nearest-hit reduction
                                  (``sampling_common.nearest_hit_reduce``)
point-to-ray assignment           minority-class promotion and exemption
(kd-tree NN, not analytic bins)   the noise / dropout models
visibility neighbourhood          the output and reprojection stage
(ray kNN graph, not a range
image)
================================  ===========================================

The scanline regularisation of the mechanical path is *disabled*: it smooths along a
scan line and a non-repetitive pattern has none.

This is a prototype.  LiSeer has never been trained or evaluated on solid-state
data; the path exists to show that the formulation is not tied to a ring-structured
grid.  Ported from the S3 ``CODE5`` patch, restructured into module-level functions
so it can be exercised without constructing a detector.
"""
import math

import numpy as np
import torch

from . import sampling_common

GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0

# Mirrors PointTPV_Seg.__init__; used when no grid is supplied.
DEFAULT_GRID = dict(
    min_bound=(0.0, -3.1415926, -3.4),
    max_bound=(51.2, 3.1415926, 3.0),
    grid_size=(480.0, 360.0, 32.0),
    grid_size_vox=(480.0, 360.0, 32.0),
)

DEFAULT_RISLEY = dict(
    # --- from the manufacturer's specification (Livox Avia) ---
    fov_deg=70.4,               # circular field of view, full angle
    point_rate=240000.0,        # points / s, first-return spec
    # --- design choices of this prototype ---
    t_int=0.5,                  # integration time T in seconds; R = point_rate * T
    f1_hz=7294.0,               # first prism rotation frequency
    freq_ratio=None,            # f2/f1; None -> -1/golden, the counter-rotating ratio
                                # that is hardest to approximate rationally, i.e. the
                                # longest-closing pattern
    f2_hz=None,                 # set this to pin f2 directly instead of freq_ratio
    rho_deg=None,               # per-prism deflection; None -> fov_deg/4 so that
                                # rho1 == rho2 and theta reaches 0 (no centre hole)
    beam_divergence_deg=0.28,   # manufacturer spec; lower bound of the footprint
    footprint_scale=1.0,        # multiplies the local mean inter-ray spacing
    height_m=1.73,              # mounting height above the estimated ground
    occlusion_mode='knn',       # 'knn' (ray-neighbourhood) or 'none'
    occ_knn_k=48,               # neighbourhood size of the visibility graph
    minority_depth_tol=0.4,     # same value the mechanical path uses
    seed=None,                  # seeds the prism phases; None -> redrawn per frame
    recompute_voxel_index=False,
)

# Constants taken from the mechanical path.
MAX_RANGE = 80.0
EPS = 1e-8
LARGE_VAL = 1e10
OCC_SUPPORT_MIN = 3
OCC_DEPTH_MARGIN = 0.5
OCC_NEAR_BAND = 0.25
OCC_REL_MARGIN = 0.02
RING_KEEP_BAND = 0.20
RING_KEEP_MIN = 6

_CFG_ALIASES = dict(
    integration_time='t_int',
    lidar_height='height_m',
    fov_half_deg=None,          # handled separately: half angle -> full angle
)


def resolve_risley_cfg(cfg=None):
    """Normalise a sensor config into :data:`DEFAULT_RISLEY`' keys.

    Accepts either a flat mapping of prism parameters or a full sensor config with
    the parameters nested under a ``risley`` key (the config-file layout).
    """
    out = dict(DEFAULT_RISLEY)
    if not cfg:
        return out
    flat = dict(cfg.get('risley', cfg) if hasattr(cfg, 'get') else cfg)
    if 'fov_half_deg' in flat:
        flat['fov_deg'] = 2.0 * float(flat.pop('fov_half_deg'))
    for alias, canonical in _CFG_ALIASES.items():
        if canonical and alias in flat:
            flat[canonical] = flat.pop(alias)
    # keys of the surrounding sensor config are not prism parameters
    unknown = [k for k in flat if k not in out]
    flat = {k: v for k, v in flat.items() if k in out}
    out.update(flat)
    out['_ignored'] = sorted(unknown)
    return out


def _prism_frequencies(cfg):
    f1 = float(cfg['f1_hz'])
    if cfg.get('f2_hz') is not None:
        return f1, float(cfg['f2_hz'])
    ratio = cfg.get('freq_ratio')
    if ratio is None:
        ratio = -1.0 / GOLDEN_RATIO
    return f1, f1 * float(ratio)


# ---------------------------------------------------------------------------
# 1. ray construction (replaces the regular ring grid)
# ---------------------------------------------------------------------------


def build_risley_rays(cfg=None, rng=None):
    """Time-ordered ray directions of a two-prism Risley scanner.

    The deflection of the two counter-rotating prisms adds in the transverse plane::

        u(t) = rho1 cos(2 pi f1 t + psi1) + rho2 cos(2 pi f2 t + psi2)
        v(t) = rho1 sin(2 pi f1 t + psi1) + rho2 sin(2 pi f2 t + psi2)

    with ``theta = hypot(u, v)`` the off-axis angle and ``phi = atan2(v, u)`` the roll
    angle about the boresight (+x).  The initial phases ``psi1``/``psi2`` are drawn
    per call, which is what makes the pattern non-repetitive from frame to frame.

    Returns a ``(R, 3)`` float64 array of unit vectors.
    """
    cfg = resolve_risley_cfg(cfg)
    if rng is None:
        seed = cfg.get('seed')
        rng = np.random.default_rng(seed)

    num_rays = int(round(float(cfg['point_rate']) * float(cfg['t_int'])))
    if num_rays <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    fov_half = 0.5 * float(cfg['fov_deg'])
    rho_deg = cfg.get('rho_deg')
    if rho_deg is None:
        rho_deg = fov_half / 2.0        # rho1 == rho2 => theta spans [0, fov_half]
    rho_rad = math.radians(float(rho_deg))
    f1, f2 = _prism_frequencies(cfg)

    psi1 = rng.uniform(0.0, 2.0 * np.pi)
    psi2 = rng.uniform(0.0, 2.0 * np.pi)
    t = np.linspace(0.0, float(cfg['t_int']), num_rays, endpoint=False)
    a1 = 2.0 * np.pi * f1 * t + psi1
    a2 = 2.0 * np.pi * f2 * t + psi2
    u = rho_rad * np.cos(a1) + rho_rad * np.cos(a2)
    v = rho_rad * np.sin(a1) + rho_rad * np.sin(a2)

    theta = np.sqrt(u * u + v * v)
    phi = np.arctan2(v, u)
    rays = np.stack([
        np.cos(theta),
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
    ], axis=-1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays


# ---------------------------------------------------------------------------
# 2. point -> ray assignment (replaces analytic angular binning)
# ---------------------------------------------------------------------------


def assign_points_to_rays(points, rays, tree=None):
    """Nearest ray for every point, by exact kd-tree search.

    ``points`` may be raw vectors from the sensor origin or already-normalised
    directions; they are normalised here either way.  Euclidean distance between
    unit vectors is monotone in the angle between them, so the Euclidean nearest ray
    *is* the angular nearest ray -- which is why an exact tree query can stand in for
    the analytic bin lookup of the mechanical path.

    Returns ``(index_into_rays, tree)``.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    unit = pts / np.maximum(norms, EPS)
    if tree is None:
        tree = cKDTree(np.asarray(rays, dtype=np.float64))
    _, nn = tree.query(unit, k=1)
    return np.asarray(nn).astype(np.int64), tree


# ---------------------------------------------------------------------------
# 3. visibility neighbourhood (replaces the range-image window)
# ---------------------------------------------------------------------------


def visibility_knn(rays, k, tree=None):
    """Angular k-nearest-neighbour graph over a ray set, self excluded.

    The mechanical path reads occluders out of a ``[beams, H]`` range image with an
    ``unfold``.  A non-repetitive pattern has no such shape, and forcing one would
    invent adjacency that the sensor does not have, so the neighbourhood is built
    geometrically instead.  The occlusion criterion and every threshold that consumes
    this neighbourhood are unchanged.

    Returns an ``(R, k')`` index array with ``k' = min(k, R - 1)``.
    """
    from scipy.spatial import cKDTree

    rays = np.asarray(rays, dtype=np.float64)
    if rays.shape[0] <= 1:
        return np.zeros((rays.shape[0], 0), dtype=np.int64)
    if tree is None:
        tree = cKDTree(rays)
    kq = min(int(k) + 1, rays.shape[0])
    _, nbr = tree.query(rays, k=kq)
    return np.asarray(nbr)[:, 1:].astype(np.int64)


def local_ray_spacing(rays, tree=None, probe=4096, rng=None):
    """Median angular distance to the 6 nearest rays, over a random probe subset."""
    from scipy.spatial import cKDTree

    rays = np.asarray(rays, dtype=np.float64)
    n = rays.shape[0]
    if n < 8:
        return float('inf')
    if tree is None:
        tree = cKDTree(rays)
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(n, size=min(probe, n), replace=False)
    nn_d, _ = tree.query(rays[idx], k=7)
    # chord length between unit vectors -> angle
    return float(np.median(2.0 * np.arcsin(np.clip(nn_d[:, 1:], 0.0, 2.0) / 2.0)))


# ---------------------------------------------------------------------------
# pattern diagnostics
# ---------------------------------------------------------------------------


def fov_coverage(rays, fov_deg=70.4, n_grid=240):
    """Fraction of the circular field of view the ray set lands in, in percent.

    The field of view is discretised on an ``n_grid x n_grid`` grid in the angular
    plane ``(theta cos phi, theta sin phi)``; cells whose centre falls outside the
    cone are excluded, and a cell counts as covered when at least one ray lands in
    it.  This is a plain occupancy measure of the discrete ray set -- no beam
    footprint is assumed -- so it is bounded by ``R / cells``.

    With the prism parameters fixed before any coverage was computed, the prototype
    reaches 50.9% coverage at 0.1 s and 98.8% at 0.8 s (Appendix B.7), consistent
    with the manufacturer's published behaviour.
    """
    rays = np.asarray(rays, dtype=np.float64)
    R = math.radians(0.5 * float(fov_deg))
    if rays.shape[0] == 0:
        return 0.0
    theta = np.arccos(np.clip(rays[:, 0], -1.0, 1.0))
    phi = np.arctan2(rays[:, 2], rays[:, 1])
    gu, gv = theta * np.cos(phi), theta * np.sin(phi)
    ii = np.clip(((gu + R) / (2.0 * R) * n_grid).astype(np.int64), 0, n_grid - 1)
    jj = np.clip(((gv + R) / (2.0 * R) * n_grid).astype(np.int64), 0, n_grid - 1)
    hit = np.zeros((n_grid, n_grid), dtype=bool)
    hit[ii, jj] = True
    centres = (np.arange(n_grid) + 0.5) / n_grid * 2.0 * R - R
    xx, yy = np.meshgrid(centres, centres, indexing='ij')
    in_fov = (xx * xx + yy * yy) <= R * R
    return float(hit[in_fov].mean() * 100.0)


# ---------------------------------------------------------------------------
# 4. the sampler
# ---------------------------------------------------------------------------


def _as_tensor(value, device, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def sample_solidstate(points, grid_ind, grid_ind_vox, labels,
                      cfg=None, grid=None, perturb_cfg=None, training=False,
                      return_info=False):
    """Re-observe dense points with a Risley-prism pattern.

    Signature and return contract match ``PointTPV_Seg._sampling``: lists of frames
    in, ``(points, grid_ind, grid_ind_vox, labels, sensor_params)`` out.  The last
    element is always ``None`` -- solid-state parameters are deliberately not fed to
    S-Film, which was trained on spinning-LiDAR parameters only.

    With ``return_info=True`` a sixth element carries per-frame diagnostics.
    """
    cfg = resolve_risley_cfg(cfg)
    if return_info:
        info_dicts = []

    was_single = isinstance(points, torch.Tensor)
    if was_single:
        points, grid_ind = [points], [grid_ind]
        grid_ind_vox, labels = [grid_ind_vox], [labels]

    points = [p.detach() for p in points]
    device = points[0].device
    dtype = points[0].dtype

    grid = dict(grid or DEFAULT_GRID)
    min_bound_s = _as_tensor(grid['min_bound'], device, dtype)
    max_bound_s = _as_tensor(grid['max_bound'], device, dtype)
    grid_size_s = _as_tensor(grid['grid_size'], device, dtype)
    grid_size_vox_s = _as_tensor(grid['grid_size_vox'], device, dtype)
    intervals_s = (max_bound_s - min_bound_s) / grid_size_s
    intervals_vox_s = (max_bound_s - min_bound_s) / grid_size_vox_s

    pcfg = sampling_common.resolve_perturb_cfg(perturb_cfg)
    _pt_on = bool(pcfg['enabled']) and bool(training)

    minority_depth_tol = float(cfg['minority_depth_tol'])
    height_m = float(cfg['height_m'])
    recompute_idx = bool(cfg['recompute_voxel_index'])

    results_points, results_grid_ind = [], []
    results_grid_ind_vox, results_labels = [], []

    base_seed = cfg.get('seed')

    for idx in range(len(points)):
        curr_points = points[idx]
        curr_labels = labels[idx]
        curr_grid_ind = grid_ind[idx]
        curr_grid_ind_vox = grid_ind_vox[idx]

        def _empty():
            return (
                torch.empty((0, curr_points.shape[1]), device=device, dtype=dtype),
                torch.empty((0,), device=device, dtype=curr_labels.dtype),
                torch.empty((0, 3), device=device, dtype=curr_grid_ind.dtype),
                torch.empty((0, 3), device=device, dtype=curr_grid_ind_vox.dtype),
            )

        def _bail(**extra):
            sp, sl, sgi, sgiv = _empty()
            results_points.append(sp); results_labels.append(sl)
            results_grid_ind.append(sgi); results_grid_ind_vox.append(sgiv)
            if return_info:
                info_dicts.append(dict(num_hit=0, **extra))

        # ---- REPLACED: ray construction ----------------------------------
        rng = np.random.default_rng(None if base_seed is None else base_seed + idx)
        ray_np = build_risley_rays(cfg, rng=rng)
        num_rays = ray_np.shape[0]
        if num_rays == 0:
            _bail(num_rays=0)
            continue
        ray_directions = torch.from_numpy(ray_np).to(device=device, dtype=dtype)

        from scipy.spatial import cKDTree
        ray_tree = cKDTree(ray_np)

        # Footprint threshold: the beam cannot resolve finer than its divergence,
        # and it should not be asked to resolve finer than the ray spacing either.
        spacing = local_ray_spacing(ray_np, tree=ray_tree, rng=rng)
        footprint_angle = max(math.radians(float(cfg['beam_divergence_deg'])),
                              float(cfg['footprint_scale']) * spacing)
        cos_footprint = math.cos(footprint_angle)

        # ---- REUSED: ground / origin estimate and range filter ------------
        point_cloud_tensor = torch.cat(
            (curr_points[:, 6:8], curr_points[:, 5].unsqueeze(1)), dim=1)
        ground_mask = ((curr_labels == 1) | (curr_labels == 7)) & (curr_points[:, 3] < 20.0)
        if ground_mask.any():
            ground_z = curr_points[ground_mask, 5].median()
        else:
            z_all = curr_points[:, 5]
            k = max(1, int(0.02 * z_all.numel()))
            ground_z = z_all.kthvalue(k).values
        lidar_origin = torch.stack([
            torch.zeros((), device=device, dtype=dtype),
            torch.zeros((), device=device, dtype=dtype),
            (ground_z + height_m).to(dtype),
        ])

        vec_all = point_cloud_tensor - lidar_origin
        range_all = torch.norm(vec_all, dim=1)
        keep_mask = (range_all > EPS) & (range_all <= MAX_RANGE)
        if not keep_mask.any():
            _bail(num_rays=num_rays, footprint_angle=footprint_angle)
            continue
        kept_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
        kept_points_raw = curr_points[kept_idx]
        kept_labels = curr_labels[kept_idx]
        kept_grid_ind = curr_grid_ind[kept_idx]
        kept_grid_ind_vox = curr_grid_ind_vox[kept_idx]
        kept_vec = vec_all[kept_idx]
        kept_range = range_all[kept_idx]

        # ---- REPLACED: point -> ray assignment ---------------------------
        unit_vec = kept_vec / kept_range.unsqueeze(1)
        nn_ray, _ = assign_points_to_rays(
            unit_vec.detach().cpu().numpy(), ray_np, tree=ray_tree)
        flat_bin = torch.from_numpy(nn_ray).to(device=device).long()
        ray_dir_for_points = ray_directions[flat_bin]

        # ---- REUSED: footprint consistency gate (threshold redefined) -----
        cos_sim = torch.sum(unit_vec * ray_dir_for_points, dim=1).clamp_(-1.0, 1.0)
        depths = torch.sum(kept_vec * ray_dir_for_points, dim=1)
        valid_mask = (depths > 0.0) & (cos_sim >= cos_footprint)
        if not valid_mask.any():
            _bail(num_rays=num_rays, footprint_angle=footprint_angle)
            continue
        valid_bins = flat_bin[valid_mask]
        valid_depths = depths[valid_mask]

        # ---- REUSED: nearest-hit reduction -------------------------------
        # The very same function the mechanical path calls.
        min_depth_flat, min_idx_in_valid, hit_mask_flat = \
            sampling_common.nearest_hit_reduce(valid_depths, valid_bins, num_rays)

        valid_pos = valid_mask.nonzero(as_tuple=False).squeeze(1)
        K = valid_depths.shape[0]
        safe_min_idx = torch.clamp(min_idx_in_valid, 0, max(K - 1, 0))
        chosen_kept_idx_flat = torch.where(
            hit_mask_flat, valid_pos[safe_min_idx],
            torch.full_like(safe_min_idx, -1))
        depth_flat = torch.where(hit_mask_flat, min_depth_flat,
                                 min_depth_flat.new_tensor(LARGE_VAL))

        # ---- REUSED: minority-class promotion ----------------------------
        valid_labels_all = kept_labels[valid_pos]
        is_minor_valid = torch.zeros_like(valid_labels_all, dtype=torch.bool)
        for _cls in sampling_common.MINORITY_CLASSES:
            is_minor_valid = is_minor_valid | (valid_labels_all == _cls)
        if is_minor_valid.any():
            minor_depths = valid_depths[is_minor_valid]
            minor_bins = valid_bins[is_minor_valid]
            minor_local_pos = is_minor_valid.nonzero(as_tuple=False).squeeze(1)
            minor_min_depth, minor_min_idx, minor_hit = \
                sampling_common.nearest_hit_reduce(minor_depths, minor_bins, num_rays)
            promote = minor_hit & hit_mask_flat & (
                minor_min_depth <= depth_flat + minority_depth_tol)
            promote_only_minor = minor_hit & (~hit_mask_flat)
            promote = promote | promote_only_minor
            Km = minor_depths.shape[0]
            safe_mi = torch.clamp(minor_min_idx, 0, max(Km - 1, 0))
            promoted_idx = valid_pos[minor_local_pos[safe_mi]]
            chosen_kept_idx_flat = torch.where(promote, promoted_idx,
                                               chosen_kept_idx_flat)
            depth_flat = torch.where(promote, minor_min_depth, depth_flat)
            hit_mask_flat = hit_mask_flat | promote_only_minor

        safe_idx = torch.clamp(chosen_kept_idx_flat, min=0)
        labels_lookup = kept_labels[safe_idx]
        minority_ray_mask = torch.zeros_like(hit_mask_flat)
        for _cls in sampling_common.MINORITY_CLASSES:
            minority_ray_mask = minority_ray_mask | (labels_lookup == _cls)
        minority_ray_mask = minority_ray_mask & hit_mask_flat

        # ---- REPLACED: visibility neighbourhood --------------------------
        # Same asymmetry as the mechanical path (only rays *above* the current one
        # can occlude it) and the same thresholds; only the neighbourhood differs.
        occ_k = int(cfg['occ_knn_k'])
        if cfg['occlusion_mode'] == 'knn' and int(hit_mask_flat.sum()) > occ_k:
            hit_ray_idx = hit_mask_flat.nonzero(as_tuple=False).squeeze(1)
            hit_np = ray_np[hit_ray_idx.cpu().numpy()]
            nbr = visibility_knn(hit_np, occ_k)
            nbr_t = torch.from_numpy(nbr).to(device=device).long()

            hit_depths = depth_flat[hit_ray_idx]
            hit_elev = ray_directions[hit_ray_idx][:, 2]   # z of the ray = elevation
            nbr_depths = hit_depths[nbr_t]
            nbr_elev = hit_elev[nbr_t]
            upper = nbr_elev > hit_elev.unsqueeze(1)
            upper_depths = torch.where(upper, nbr_depths,
                                       nbr_depths.new_tensor(LARGE_VAL))
            occ_local_min = upper_depths.amin(dim=1)
            near_support = (upper_depths <=
                            (occ_local_min.unsqueeze(1) + OCC_NEAR_BAND)).sum(dim=1)
            # the mechanical same-ring protection becomes "depth-similar neighbours"
            same_support = (torch.abs(nbr_depths - hit_depths.unsqueeze(1))
                            <= RING_KEEP_BAND).sum(dim=1)
            ring_keep = same_support >= RING_KEEP_MIN
            adaptive_margin = OCC_DEPTH_MARGIN + OCC_REL_MARGIN * occ_local_min
            leak = (torch.isfinite(occ_local_min) &
                    (occ_local_min < LARGE_VAL) &
                    (near_support >= OCC_SUPPORT_MIN) &
                    ((hit_depths - occ_local_min) > adaptive_margin) &
                    (~ring_keep) &
                    (~minority_ray_mask[hit_ray_idx]))
            if leak.any():
                drop_rays = hit_ray_idx[leak]
                hit_mask_flat = hit_mask_flat.clone()
                hit_mask_flat[drop_rays] = False
                chosen_kept_idx_flat = chosen_kept_idx_flat.clone()
                chosen_kept_idx_flat[drop_rays] = -1

        # DISABLED: the ring-continuity median filter of the mechanical path. It
        # shrinks a depth towards the median along a scan line, and there are none.

        # ---- REUSED: output and reprojection -----------------------------
        valid_out = hit_mask_flat & (chosen_kept_idx_flat >= 0) & torch.isfinite(depth_flat)
        if not valid_out.any():
            _bail(num_rays=num_rays, footprint_angle=footprint_angle)
            continue
        chosen_kept_idx = chosen_kept_idx_flat[valid_out]
        chosen_bins = valid_out.nonzero(as_tuple=False).squeeze(1)
        chosen_depths = depth_flat[valid_out].unsqueeze(1)

        # Optional perturbation suite, exactly as on the mechanical path.
        if _pt_on and pcfg['ray_dropout'] > 0 and chosen_kept_idx.numel() > 16:
            keep_pt = sampling_common.ray_dropout_mask(
                chosen_kept_idx.shape[0], pcfg['ray_dropout'], device,
                labels=kept_labels[chosen_kept_idx],
                exempt_minority=pcfg['dropout_exempt_minority'])
            chosen_kept_idx = chosen_kept_idx[keep_pt]
            chosen_bins = chosen_bins[keep_pt]
            chosen_depths = chosen_depths[keep_pt]
            if chosen_kept_idx.numel() == 0:
                _bail(num_rays=num_rays, footprint_angle=footprint_angle)
                continue
        if _pt_on and pcfg['range_noise_cm'] > 0:
            chosen_depths = sampling_common.apply_range_noise(
                chosen_depths, pcfg['range_noise_cm'])

        sampled_points = kept_points_raw[chosen_kept_idx].clone()
        sampled_labels = kept_labels[chosen_kept_idx]
        sampled_grid_ind = kept_grid_ind[chosen_kept_idx]
        sampled_grid_ind_vox = kept_grid_ind_vox[chosen_kept_idx]
        chosen_ray_dirs = ray_directions[chosen_bins]
        if _pt_on and pcfg['angular_jitter_mrad'] > 0:
            chosen_ray_dirs = sampling_common.apply_angular_jitter(
                chosen_ray_dirs, pcfg['angular_jitter_mrad'])
        projected_xyz = lidar_origin.unsqueeze(0) + chosen_depths * chosen_ray_dirs

        proj_x, proj_y, proj_z = projected_xyz[:, 0], projected_xyz[:, 1], projected_xyz[:, 2]
        sampled_points[:, 6] = proj_x
        sampled_points[:, 7] = proj_y
        sampled_points[:, 5] = proj_z

        if recompute_idx:
            # Off by default so the output stage matches the mechanical path exactly.
            # On, the polar features and voxel indices are rebuilt from the moved
            # point, which is self-consistent but no longer a verbatim reuse.
            proj_rho = torch.sqrt(proj_x * proj_x + proj_y * proj_y)
            proj_phi = torch.atan2(proj_y, proj_x)
            xyz_pol = torch.stack([proj_rho, proj_phi, proj_z], dim=1)
            xyz_pol_grid = torch.maximum(
                torch.minimum(xyz_pol, max_bound_s - 1e-3), min_bound_s)
            new_grid_ind = torch.floor((xyz_pol_grid - min_bound_s) / intervals_s)
            new_grid_ind_vox = (xyz_pol_grid - min_bound_s) / intervals_vox_s
            voxel_centers = (new_grid_ind + 0.5) * intervals_s + min_bound_s
            sampled_points[:, 0:3] = xyz_pol - voxel_centers
            sampled_points[:, 3] = proj_rho
            sampled_points[:, 4] = proj_phi
            sampled_grid_ind = new_grid_ind.to(curr_grid_ind.dtype)
            sampled_grid_ind_vox = new_grid_ind_vox.to(curr_grid_ind_vox.dtype)

        results_points.append(sampled_points)
        results_labels.append(sampled_labels)
        results_grid_ind.append(sampled_grid_ind)
        results_grid_ind_vox.append(sampled_grid_ind_vox)
        if return_info:
            info_dicts.append(dict(
                num_rays=num_rays,
                num_hit=int(sampled_points.shape[0]),
                footprint_angle=footprint_angle,
                local_spacing=spacing,
                fov_coverage=fov_coverage(ray_np, cfg['fov_deg']),
                ray_directions=ray_np,
                # ray id and depth behind every returned point, so the occlusion
                # criterion can be re-applied to the output by a test.
                chosen_rays=chosen_bins.detach().cpu().numpy(),
                chosen_depths=chosen_depths.detach().cpu().numpy().reshape(-1),
            ))

    out = (results_points, results_grid_ind, results_grid_ind_vox, results_labels, None)
    if return_info:
        return out + (info_dicts,)
    return out
