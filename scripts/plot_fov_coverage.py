#!/usr/bin/env python3
"""Field-of-view coverage of the prism prototype against integration time.

    python scripts/plot_fov_coverage.py --out out/solidstate

Coverage is the occupancy of a 240x240 angular grid over the circular field of view
(see ``solidstate_sampler.fov_coverage``).  Appendix B.7 reports 50.9% coverage at
0.1 s and 98.8% at 0.8 s, consistent with the manufacturer's published behaviour.
"""
import argparse
import csv
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

TIMES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def load_module():
    try:
        from projects.unilidar_plugin.pointocc_model import solidstate_sampler
        return solidstate_sampler
    except ImportError:
        from tests._stub_env import load_sampler_modules
        return load_sampler_modules()[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out/solidstate')
    ap.add_argument('--repeats', type=int, default=3,
                    help='phase draws averaged per integration time')
    ap.add_argument('--grid', type=int, default=240)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    solid = load_module()
    fov_deg = solid.DEFAULT_RISLEY['fov_deg']
    rows = []
    for t in TIMES:
        vals = [solid.fov_coverage(
            solid.build_risley_rays(dict(t_int=t, seed=100 + i)),
            fov_deg, n_grid=args.grid) for i in range(args.repeats)]
        rays = int(round(solid.DEFAULT_RISLEY['point_rate'] * t))
        rows.append((t, rays, float(np.mean(vals)), float(np.std(vals))))
        print(f'T={t:<5} rays={rays:>7,}  coverage={np.mean(vals):6.2f}% '
              f'(+-{np.std(vals):.2f})')

    csv_path = os.path.join(args.out, 'fov_coverage.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['integration_time_s', 'rays', 'coverage_percent', 'std'])
        w.writerows(rows)
    print('wrote', csv_path)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[note] matplotlib unavailable; csv written, figure skipped')
        return
    t = [r[0] for r in rows]
    c = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(t, c, '-o', lw=2.2, ms=4.5, color='#2f6fb0')
    ax.set_xlabel('cumulative integration time (s)')
    ax.set_ylabel('FoV coverage (%)')
    ax.set_title('Non-repetitive scan: FoV coverage grows with integration time')
    ax.grid(alpha=.3)
    fig.tight_layout()
    path = os.path.join(args.out, 'fov_coverage.png')
    fig.savefig(path, dpi=200)
    print('wrote', path)


if __name__ == '__main__':
    main()
