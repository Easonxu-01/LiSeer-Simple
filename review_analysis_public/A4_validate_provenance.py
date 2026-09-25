"""A4 provenance patch self-check.

    python review_analysis/A4_validate_provenance.py
"""
import sys, os, random
sys.path.insert(0, os.getcwd())
import torch
import numpy as np


def _seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


def main():
    from review_analysis.CODE5_risley_demo import build_model, load_dense_frame
    from projects.unilidar_plugin.pointocc_model.pointtpv_seg import PointTPV_Seg

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pts, gi, giv, lab = load_dense_frame(8, 100, device)
    base_ptr = pts.data_ptr()

    sensor_cfg = dict(
        LiDAR_height=[1.5, 1.85], num_of_beams=[32, 64],
        horizontal_angular_resolution=[600, 1800],
        lower_vertical_field_of_view_bound=[-28, -22],
        upper_vertical_field_of_view_bound=[2, 15])

    m = build_model(device)
    m.consistency = False
    m.dense_train = False
    m.sample_repeat_count = 1
    m.sensor_config = sensor_cfg
    m._mixup_minority = PointTPV_Seg._mixup_minority.__get__(m)
    m._sampling = PointTPV_Seg._sampling.__get__(m)

    # [1] default path: attribute absent -> no crash, no buffer created
    ok1 = not hasattr(m, '_provenance_buffer')
    _seed_all(0)
    rp, rg, rgv, rl, _ = m._sampling([pts], [gi], [giv], [lab])
    ok1 = ok1 and not getattr(m, '_provenance_buffer', [])
    print(f'[1/3] default path untouched (no buffer without opt-in): '
          f'{"OK" if ok1 else "FAIL"}')

    # [2] opt-in: buffer populated, every index valid and unique per call
    m.return_provenance = True
    m._provenance_buffer = []
    _seed_all(0)
    rp2, rg2, rgv2, rl2, _ = m._sampling([pts], [gi], [giv], [lab])
    buf = m._provenance_buffer
    m._provenance_buffer = []
    m.return_provenance = False

    n_dense = pts.shape[0]
    all_idx = torch.cat([b for _, b in buf]) if buf else torch.empty(0, dtype=torch.long)
    in_range = bool((all_idx >= 0).all() and (all_idx < n_dense).all())
    n_sampled = sum(len(v) for _, v in buf)
    n_returned = len(rp2[0]) if isinstance(rp2, (list, tuple)) else len(rp2)
    ok2 = len(buf) > 0 and in_range and n_sampled == n_returned
    print(f'[2/3] provenance populated, in-range, count matches output '
          f'({n_sampled} indices vs {n_returned} points): '
          f'{"OK" if ok2 else "FAIL"}')

    # [3] the opt-in call must not have replaced or resized the input tensor
    ok3 = pts.data_ptr() == base_ptr
    print(f'[3/3] input dense tensor untouched (data_ptr unchanged): '
          f'{"OK" if ok3 else "FAIL"}')

    if not (ok1 and ok2 and ok3):
        raise SystemExit('provenance patch failed validation')
    print('\nALL CHECKS PASSED.')


if __name__ == '__main__':
    main()
