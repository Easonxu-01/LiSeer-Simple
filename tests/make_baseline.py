"""Regenerate ``tests/baseline_outputs.npz``.

Run this on the *pre-change* sampler to capture the reference, then again never:
the file is committed so the regression test can prove the default path is
untouched.  Usage: ``python tests/make_baseline.py``
"""
import numpy as np

from tests.sampler_harness import BASELINE_NPZ, build_model, run_all_frames


def main():
    model = build_model()
    runs = run_all_frames(model)
    blob = {}
    for i, r in enumerate(runs):
        for k, v in r.items():
            blob[f'frame{i}_{k}'] = v
        print(f'frame{i}: {r["points"].shape[0]} points, '
              f'labels {sorted(set(r["labels"].tolist()))}')
    np.savez_compressed(BASELINE_NPZ, **blob)
    print('wrote', BASELINE_NPZ)


if __name__ == '__main__':
    main()
