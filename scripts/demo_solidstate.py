#!/usr/bin/env python3
"""Re-observe one dense frame four ways: spinning, then the prism prototype.

    python scripts/demo_solidstate.py --out out/solidstate

Writes a bird's-eye-view panel per sampling and an ``.npz`` with the sampled point
clouds.  With no ``--frame`` it uses the synthetic fixture from ``tests/``, so the
script runs without any dataset.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

INTEGRATION_TIMES = (0.1, 0.2, 0.5)

# A single fixed spinning configuration, so the reference panel is reproducible.
DEMO_SENSOR = dict(
    LiDAR_height=[1.73, 1.73],
    num_of_beams=[32, 32],
    horizontal_angular_resolution=[900, 900],
    lower_vertical_field_of_view_bound=[-25.0, -25.0],
    upper_vertical_field_of_view_bound=[3.0, 3.0],
)


def load_modules():
    """Import the sampler, falling back to the test stubs off-server."""
    try:
        from projects.unilidar_plugin.pointocc_model import pointtpv_seg, solidstate_sampler
        return pointtpv_seg, solidstate_sampler
    except ImportError:
        from tests._stub_env import load_sampler_modules
        seg, _, solid = load_sampler_modules()
        print('[note] mmdet3d/torch_scatter unavailable; using the test stubs')
        return seg, solid


def load_frame(path):
    if path is None:
        from tests.synth_frames import make_frame
        return make_frame(1001)
    blob = np.load(path)
    return (torch.from_numpy(blob['points']).float(),
            torch.from_numpy(blob['grid_ind']).to(torch.int32),
            torch.from_numpy(blob['grid_ind_vox']).float(),
            torch.from_numpy(blob['labels']).long())


def xy(points):
    return points[:, 6].numpy(), points[:, 7].numpy()


def save_panels(panels, out_dir, extent):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[note] matplotlib unavailable; point clouds written, figures skipped')
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
    for ax, (title, pts, labels) in zip(np.atleast_1d(axes), panels):
        x, y = xy(pts)
        ax.scatter(x, y, s=0.12, c=labels.numpy(), cmap='tab10', vmin=0, vmax=9,
                   linewidths=0)
        ax.set_aspect('equal')
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_title(f'{title}\n{pts.shape[0]:,} pts', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    path = os.path.join(out_dir, 'demo_solidstate_bev.png')
    fig.savefig(path, dpi=200)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', default=None,
                    help='npz with points/grid_ind/grid_ind_vox/labels')
    ap.add_argument('--out', default='out/solidstate')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--extent', type=float, default=50.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seg, solid = load_modules()
    frame = load_frame(args.frame)
    points, grid_ind, grid_ind_vox, labels = frame
    print(f'dense frame: {points.shape[0]:,} points')

    panels = [('Dense reconstruction', points, labels)]
    blob = dict(dense_points=points.numpy(), dense_labels=labels.numpy())

    # --- spinning reference (the default path, untouched) ---
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = seg.PointTPV_Seg(sensor=dict(DEMO_SENSOR))
    model.train()
    with torch.no_grad():
        out_p, _, _, out_l, _ = model._sampling(
            [points.clone()], [grid_ind.clone()],
            [grid_ind_vox.clone()], [labels.clone()])
    panels.append(('Mechanical spinning', out_p[0], out_l[0]))
    blob['spinning_points'] = out_p[0].numpy()
    blob['spinning_labels'] = out_l[0].numpy()
    print(f'spinning          : {out_p[0].shape[0]:,} pts')

    # --- prism prototype at three integration times ---
    for t in INTEGRATION_TIMES:
        cfg = dict(pattern='risley', risley=dict(t_int=t, seed=args.seed))
        p, _, _, l, _, info = solid.sample_solidstate(
            [points.clone()], [grid_ind.clone()], [grid_ind_vox.clone()],
            [labels.clone()], cfg=cfg, return_info=True)
        rays = info[0]['num_rays']
        panels.append((f'Risley prototype, T={t}s', p[0], l[0]))
        blob[f'risley_{t}_points'] = p[0].numpy()
        blob[f'risley_{t}_labels'] = l[0].numpy()
        print(f'risley T={t:<4}     : {p[0].shape[0]:,} pts from {rays:,} rays, '
              f'FoV coverage {info[0]["fov_coverage"]:.1f}%')

    npz = os.path.join(args.out, 'demo_solidstate.npz')
    np.savez_compressed(npz, **blob)
    print('wrote', npz)
    save_panels(panels, args.out, args.extent)


if __name__ == '__main__':
    main()
