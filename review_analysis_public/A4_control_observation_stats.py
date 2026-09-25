"""Control-group observation statistics for the A4 / R2C1 experiment.

The main experiment observes one dense scene with 50 DIFFERENT sensor
configurations. The control observes it 50 times with ONE fixed configuration,
so the only thing varying is the sampler's own stochasticity (azimuth phase,
drop-out, range noise, occlusion). `review_analysis/_source_data/A4_raw.json` keeps the control's
aggregate agreement numbers but not its per-point observation counts, so the
threshold-sweep curve that exists for the main run has no control counterpart.

This script produces that counterpart, plus a decomposition of where the
per-repeat differences come from:

  A. n_points_k for k = 1..50  -- how many dense points are hit by at least
     k of the 50 repeats (directly comparable to the main run's curve)
  B. the fraction seen by all 50 repeats, and the mean observation count
     under BOTH denominators the main run reports
  C. for the points seen by all 50: how far apart the 50 returned positions
     are. NOT "exactly identical coordinates" -- the sampler adds range noise
     sigma(r) = 1cm + 0.1%*r, a continuous Gaussian, so exact float equality
     has probability ~0 and would report 0.0% regardless of behaviour. The
     spread itself is the informative quantity, so this reports the
     distribution of per-point spread and the fraction within several
     tolerances.

No model inference: observation counts and returned coordinates depend only
on the sampler, so this runs the sampling path alone.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from review_analysis.A4_consistency import build_liseer, prepare_sampler   # noqa: E402
from review_analysis.CODE5_risley_demo import load_dense_frame             # noqa: E402


def sample_once(model, cfg, frame, seed):
    """One scan with a forced sensor config; returns (dense_idx, xyz).

    Mirrors review_analysis/A4_consistency.sample_with_config: the same monkeypatch
    of the three config-selection draws, the same provenance opt-in.
    """
    import projects.unilidar_plugin.pointocc_model.pointtpv_seg as M
    pts, gi, giv, lab = frame

    real_uniform, real_choice, real_beams = (
        M.random.uniform, M.random.choice, M.sample_beams)
    seq = {'n': 0}

    def fake_uniform(a, b):
        seq['n'] += 1
        return {1: cfg['height'], 2: cfg['vlow'], 3: cfg['vup']}.get(
            seq['n'], real_uniform(a, b))

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    M.random.uniform = fake_uniform
    M.random.choice = lambda seq_: cfg['hres']
    M.sample_beams = lambda lo, hi: cfg['beams']
    model.return_provenance = True
    model._provenance_buffer = []
    try:
        rp, _, _, _, _ = model._sampling([pts], [gi], [giv], [lab])
    finally:
        M.random.uniform, M.random.choice, M.sample_beams = (
            real_uniform, real_choice, real_beams)
        model.return_provenance = False
        buf = model._provenance_buffer
        model._provenance_buffer = []

    prov = torch.cat([b for _, b in buf]).cpu().numpy().astype(np.int64)
    feat = rp[0] if isinstance(rp, (list, tuple)) else rp
    a = feat.detach().float().cpu().numpy()
    xyz = np.stack([a[:, 6], a[:, 7], a[:, 5]], axis=1)   # cols 6,7,5 = x,y,z
    assert len(prov) == len(xyz), (len(prov), len(xyz))
    return prov, xyz


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', default='ckpts/LiSeer_SK.pth')
    ap.add_argument('--raw', default='review_analysis/_source_data/A4_raw.json')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='review_analysis/_source_data/A4_control_observation_stats.json')
    args = ap.parse_args()

    raw = json.load(open(args.raw))
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    stored = raw['checkpoints'][ckpt_name]
    ctrl_cfg = stored['control_same_config']['config']
    n_reps = len(raw['configs'])
    seq, frame_no = (lambda p: (int(p[1][3:]), int(p[2])))(
        raw['frame_id'].split('/'))

    device = torch.device(args.device)
    pts, gi, giv, lab = load_dense_frame(seq, frame_no, device)
    base_ptr = pts.data_ptr()
    n_dense = int(pts.shape[0])
    print(f'[frame] {raw["frame_id"]}  {n_dense:,} dense points')
    print(f'[control] fixed config {ctrl_cfg}, {n_reps} repeats, '
          f'seeds 0..{n_reps - 1} (same as the A4 control run)')

    model, _ = build_liseer(args.ckpt, device, verbose=False)
    prepare_sampler(model, raw['dataset'])
    assert getattr(model, 'beam_concentration', None) is None, \
        'beam_concentration must stay at the sampler default here'

    obs = np.zeros(n_dense, dtype=np.int32)
    # Running sum / sum-of-squares per dense point -> per-point spread without
    # holding 50 x N coordinates in memory.
    sx = np.zeros((n_dense, 3), dtype=np.float64)
    sxx = np.zeros((n_dense, 3), dtype=np.float64)

    for i in range(n_reps):
        prov, xyz = sample_once(model, ctrl_cfg, (pts, gi, giv, lab), seed=i)
        assert pts.data_ptr() == base_ptr, 'input tensor was replaced'
        np.add.at(obs, prov, 1)
        np.add.at(sx, prov, xyz.astype(np.float64))
        np.add.at(sxx, prov, xyz.astype(np.float64) ** 2)
        print(f'  [ctrl {i + 1:>2}/{n_reps}] {len(prov):>7,} pts')

    # ---- A. threshold sweep -------------------------------------------------
    sweep = [dict(k=int(k), n_points_k=int((obs >= k).sum()))
             for k in range(1, n_reps + 1)]

    # ---- B. headline counts, both denominators ------------------------------
    seen = obs > 0
    n_seen = int(seen.sum())
    n_all = int((obs >= n_reps).sum())
    out = dict(
        frame_id=raw['frame_id'], dataset=raw['dataset'],
        checkpoint=f'{ckpt_name}.pth',
        control_config=ctrl_cfg, n_repeats=n_reps, seeds='0..%d' % (n_reps - 1),
        dense_points=n_dense,
        observed_at_least_once=n_seen,
        # The main run's headline 9.77 is obs[obs>0].mean(), i.e. averaged over
        # observed points, NOT over all dense points. Both are given so the
        # comparison cannot be made across mismatched denominators.
        mean_observations_over_observed=float(obs[seen].mean()),
        mean_observations_over_all_dense=float(obs.mean()),
        observed_by_all_repeats=n_all,
        observed_by_all_repeats_pct_of_observed=float(100.0 * n_all / max(n_seen, 1)),
        observed_by_all_repeats_pct_of_dense=float(100.0 * n_all / n_dense),
        threshold_sweep=sweep,
        main_run_for_comparison=dict(
            observed_at_least_once=stored['observed_at_least_once'],
            mean_observations_over_observed=stored['mean_observations'],
            observed_by_all_configs=int(
                [x for x in stored['threshold_sweep'] if x['k'] == n_reps][0]['n']),
        ),
    )

    # ---- C. coordinate spread on the all-repeats points ---------------------
    for label, mask in (('all_repeats', obs >= n_reps),
                        ('at_least_45', obs >= 45)):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            out[f'coord_spread_{label}'] = dict(n_points=0)
            continue
        c = obs[idx][:, None].astype(np.float64)
        var = np.maximum(sxx[idx] / c - (sx[idx] / c) ** 2, 0.0)
        # per-point 3-D RMS displacement about its own mean position
        rms = np.sqrt(var.sum(axis=1))
        out[f'coord_spread_{label}'] = dict(
            n_points=int(idx.size),
            rms_m=dict(
                mean=float(rms.mean()), median=float(np.median(rms)),
                p90=float(np.percentile(rms, 90)),
                p99=float(np.percentile(rms, 99)), max=float(rms.max())),
            frac_within=dict(
                (f'{t * 100:g}cm', float((rms <= t).mean()))
                for t in (0.001, 0.01, 0.03, 0.10)),
            exactly_identical_frac=float((rms == 0.0).mean()),
            note='rms_m = per-point RMS displacement about that point\'s own '
                 'mean returned position, over the repeats that observed it. '
                 'Range noise is sigma(r)=1cm+0.1%*r, so a nonzero spread is '
                 'expected and "exactly identical" is ~0 by construction.',
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)

    print(f'\n[control] observed>=1: {n_seen:,}  '
          f'(main run {stored["observed_at_least_once"]:,})')
    print(f'[control] mean obs/point over observed: '
          f'{out["mean_observations_over_observed"]:.2f}  '
          f'(main run {stored["mean_observations"]:.2f})')
    print(f'[control] seen by all {n_reps}: {n_all:,} '
          f'({out["observed_by_all_repeats_pct_of_observed"]:.2f}% of observed; '
          f'main run {out["main_run_for_comparison"]["observed_by_all_configs"]:,})')
    for label in ('all_repeats', 'at_least_45'):
        d = out[f'coord_spread_{label}']
        if d['n_points']:
            print(f'[control] spread on {label} ({d["n_points"]:,} pts): '
                  f'median {d["rms_m"]["median"] * 100:.2f} cm, '
                  f'p90 {d["rms_m"]["p90"] * 100:.2f} cm, '
                  f'<=1cm {d["frac_within"]["1cm"] * 100:.1f}%')
    print(f'[control] wrote {args.out}')


if __name__ == '__main__':
    main()
