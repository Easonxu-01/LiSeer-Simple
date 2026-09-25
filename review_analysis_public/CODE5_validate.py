"""CODE-5 patch self-check. Run from the repo root
after applying the patch:

    python review_analysis/CODE5_validate.py

All three checks must print OK. This does not require a GPU with much
headroom -- it loads one real SemanticKITTI dense frame and runs the sampler
a handful of times.
"""
import sys
import os
import random

sys.path.insert(0, os.getcwd())
import torch
import numpy as np


def _seed_all(s):
    """torch.manual_seed alone is NOT enough -- _sampling also draws from
    Python's random and numpy's RNG (dropout probabilities, phase, etc.).
    Missing either of the other two makes two 'identical' calls diverge in
    point count, which looks exactly like a real bug but isn't one."""
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


def check_method_exists():
    from projects.unilidar_plugin.pointocc_model.pointtpv_seg import PointTPV_Seg
    ok = hasattr(PointTPV_Seg, 'sample_solid_state') and callable(
        getattr(PointTPV_Seg, 'sample_solid_state'))
    print(f'[1/3] sample_solid_state method present and callable: '
          f'{"OK" if ok else "FAIL"}')
    if not ok:
        raise SystemExit('sample_solid_state missing -- patch not applied '
                          'or applied incorrectly')


def check_beam_concentration_default_noop():
    """beam_concentration must default to None and be a byte-identical
    no-op versus a model instance that never had the attribute set at all."""
    from review_analysis.CODE5_risley_demo import build_model, load_dense_frame

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pts, gi, giv, lab = load_dense_frame(8, 100, device)  # SK seq08 frame100

    sensor_cfg = dict(
        LiDAR_height=[1.5, 1.85], num_of_beams=[32, 64],
        horizontal_angular_resolution=[600, 1800],
        lower_vertical_field_of_view_bound=[-28, -22],
        upper_vertical_field_of_view_bound=[2, 15])

    from projects.unilidar_plugin.pointocc_model.pointtpv_seg import PointTPV_Seg

    def wire(m):
        m.consistency = False
        m.dense_train = False
        m.sample_repeat_count = 1
        m.sensor_config = sensor_cfg
        m._mixup_minority = PointTPV_Seg._mixup_minority.__get__(m)
        m._sampling = PointTPV_Seg._sampling.__get__(m)
        return m

    m_no_attr = wire(build_model(device))
    if hasattr(m_no_attr, 'beam_concentration'):
        delattr(m_no_attr, 'beam_concentration')

    m_explicit_none = wire(build_model(device))
    m_explicit_none.beam_concentration = None

    # Both calls must see the same random draws: fix the config-selection
    # randomness the same way review_analysis/A6_figR4_multi.py does, so the only
    # thing that can differ between the two runs is beam_concentration itself.
    import projects.unilidar_plugin.pointocc_model.pointtpv_seg as M
    real_uniform, real_choice, real_sample_beams = (
        M.random.uniform, M.random.choice, M.sample_beams)
    M.random.uniform = lambda a, b: (a + b) / 2
    M.random.choice = lambda seq_: 1200
    M.sample_beams = lambda lo, hi: 48
    try:
        _seed_all(0)
        r1 = m_no_attr._sampling([pts], [gi], [giv], [lab])
        _seed_all(0)
        r2 = m_explicit_none._sampling([pts], [gi], [giv], [lab])
    finally:
        M.random.uniform, M.random.choice, M.sample_beams = (
            real_uniform, real_choice, real_sample_beams)

    def _unwrap(x):
        return x[0] if isinstance(x, (list, tuple)) else x

    ok = all(torch.equal(_unwrap(a), _unwrap(b)) for a, b in zip(r1[:4], r2[:4]))
    print(f'[2/3] beam_concentration default (no attr) == explicit None: '
          f'{"OK" if ok else "FAIL"}')
    if not ok:
        raise SystemExit('beam_concentration default is NOT a no-op -- '
                          'the mechanical sampling path has changed behaviour')


def check_solid_state_runs():
    from review_analysis.CODE5_risley_demo import build_model, load_dense_frame

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pts, gi, giv, lab = load_dense_frame(8, 100, device)
    m = build_model(device)

    results = {}
    for T in (0.1, 0.5):
        sp, sg, sgv, sl, info = m.sample_solid_state(
            [pts], [gi], [giv], [lab], integration_time=T,
            footprint_scale=1.0, occlusion_mode='knn', seed=0)
        n_pts = len(sp[0])
        n_rays = info[0]['num_rays']
        results[T] = (n_pts, n_rays)
        print(f'      T={T}s: {n_pts:,} points, {n_rays:,} rays')

    ok = (results[0.1][1] < results[0.5][1]) and all(
        v[0] > 0 for v in results.values())
    print(f'[3/3] sample_solid_state runs and ray count grows with T: '
          f'{"OK" if ok else "FAIL"}')
    if not ok:
        raise SystemExit('sample_solid_state did not behave as expected')


if __name__ == '__main__':
    check_method_exists()
    check_beam_concentration_default_noop()
    check_solid_state_runs()
    print('\nALL CHECKS PASSED.')
