"""Deterministic synthetic dense frames used as sampler test fixtures.

No point cloud ships with the repository, so the regression baseline is built from
synthetic scenes instead.  They are generated from a fixed seed and use the same
column layout as ``PointsegMapping`` (``[:, 0:3]`` rel_xyz, ``[:, 3:6]`` polar
rho/phi/z, ``[:, 6:8]`` cartesian xy), so they exercise exactly the code path a real
dense frame would.
"""
import numpy as np
import torch

# Mirrors PointTPV_Seg.__init__
MIN_BOUND = np.array([0.0, -3.1415926, -3.4], dtype=np.float64)
MAX_BOUND = np.array([51.2, 3.1415926, 3.0], dtype=np.float64)
GRID_SIZE = np.array([480.0, 360.0, 32.0], dtype=np.float64)
GRID_SIZE_VOX = np.array([480.0, 360.0, 32.0], dtype=np.float64)

GROUND_Z = -1.73

# A fixed, fully determined sensor range: every draw has exactly one outcome, so the
# sampled configuration is reproducible while the RNG is still consumed as usual.
FIXTURE_SENSOR = dict(
    LiDAR_height=[1.73, 1.73],
    num_of_beams=[32, 32],
    horizontal_angular_resolution=[900, 900],
    lower_vertical_field_of_view_bound=[-25.0, -25.0],
    upper_vertical_field_of_view_bound=[3.0, 3.0],
)


def _cart_to_columns(xyz):
    """[N,3] cartesian -> [N,8] stored layout."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rho = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    xyz_pol = np.stack([rho, phi, z], axis=1)
    intervals = (MAX_BOUND - MIN_BOUND) / GRID_SIZE
    intervals_vox = (MAX_BOUND - MIN_BOUND) / GRID_SIZE_VOX
    clipped = np.minimum(np.maximum(xyz_pol, MIN_BOUND), MAX_BOUND - 1e-3)
    grid_ind = np.floor((clipped - MIN_BOUND) / intervals)
    grid_ind_vox = (clipped - MIN_BOUND) / intervals_vox
    centers = (grid_ind + 0.5) * intervals + MIN_BOUND
    rel = xyz_pol - centers
    feats = np.concatenate([rel, xyz_pol, np.stack([x, y], axis=1)], axis=1)
    return feats, grid_ind, grid_ind_vox


def _plane_disk(rng, n, r_lo, r_hi):
    r = np.sqrt(rng.uniform(r_lo ** 2, r_hi ** 2, n))
    a = rng.uniform(-np.pi, np.pi, n)
    z = GROUND_Z + 0.03 * np.sin(r * 0.4) + rng.normal(0.0, 0.01, n)
    return np.stack([r * np.cos(a), r * np.sin(a), z], axis=1)


def _box(rng, n, center, size):
    c = np.asarray(center, dtype=np.float64)
    s = np.asarray(size, dtype=np.float64)
    # points on the box surface: pick a face, then a position on it
    face = rng.integers(0, 6, n)
    uv = rng.uniform(-0.5, 0.5, (n, 3))
    for axis in range(3):
        lo = face == 2 * axis
        hi = face == 2 * axis + 1
        uv[lo, axis] = -0.5
        uv[hi, axis] = 0.5
    return c + uv * s


def make_frame(seed):
    """Return ``(points[N,8], grid_ind[N,3], grid_ind_vox[N,3], labels[N])`` tensors."""
    rng = np.random.default_rng(seed)
    chunks = []

    def add(xyz, label):
        chunks.append((xyz, np.full(xyz.shape[0], label, dtype=np.int64)))

    add(_plane_disk(rng, 14000, 1.5, 19.0), 1)     # driveable ground
    add(_plane_disk(rng, 9000, 19.0, 46.0), 7)     # other ground
    # building facades: two long slabs
    for sx in (-1.0, 1.0):
        n = 5000
        y = rng.uniform(-30.0, 30.0, n)
        z = rng.uniform(GROUND_Z, GROUND_Z + 7.0, n)
        x = np.full(n, sx * 22.0) + rng.normal(0.0, 0.03, n)
        add(np.stack([x, y, z], axis=1), 2)        # structure
    # vehicles
    for k, c in enumerate([(9.0, 2.2), (16.0, -3.1), (-11.0, 3.4), (26.0, 1.0)]):
        add(_box(rng, 2200, (c[0], c[1], GROUND_Z + 0.78), (4.4, 1.85, 1.55)), 3)
    # vegetation blobs
    for c in [(13.0, 8.0), (-8.0, -9.5), (31.0, -6.0)]:
        n = 1800
        d = rng.normal(0.0, 0.9, (n, 3))
        add(np.stack([c[0] + d[:, 0], c[1] + d[:, 1],
                      GROUND_Z + 2.4 + d[:, 2]], axis=1), 4)
    # living beings (minority class 5): pedestrian-sized cylinders
    for c in [(7.5, -1.6), (12.0, 4.6), (20.0, -2.0), (5.0, 5.5)]:
        n = 520
        a = rng.uniform(-np.pi, np.pi, n)
        rr = 0.24 * np.sqrt(rng.uniform(0.6, 1.0, n))
        z = GROUND_Z + rng.uniform(0.0, 1.75, n)
        add(np.stack([c[0] + rr * np.cos(a), c[1] + rr * np.sin(a), z], axis=1), 5)
    # movable objects (minority class 6)
    for c in [(10.5, -5.2), (18.5, 5.8), (24.0, -4.4)]:
        add(_box(rng, 360, (c[0], c[1], GROUND_Z + 0.55), (1.7, 0.6, 1.1)), 6)

    xyz = np.concatenate([c[0] for c in chunks], axis=0)
    labels = np.concatenate([c[1] for c in chunks], axis=0)
    order = rng.permutation(xyz.shape[0])
    xyz, labels = xyz[order], labels[order]

    feats, grid_ind, grid_ind_vox = _cart_to_columns(xyz)
    return (
        torch.from_numpy(feats).to(torch.float32),
        torch.from_numpy(grid_ind).to(torch.int32),
        torch.from_numpy(grid_ind_vox).to(torch.float32),
        torch.from_numpy(labels),
    )


FRAME_SEEDS = (1001, 1002, 1003)


def make_frames(seeds=FRAME_SEEDS):
    return [make_frame(s) for s in seeds]
