#!/usr/bin/env python3
"""A5 Step final: review_analysis/_source_data/A5_cost.csv (Table R10, D31-D35) plus a short report.

Two things this does that a plain point estimate would not:

  * A/B differences get a bootstrap CI. The S-Film difference is small enough
    that its sign flips between the mean and the median, so a point estimate
    alone would be misleading -- the CI says whether the cost is resolvable at
    all at this sample size.
  * training-side and inference-side costs are kept apart, and the data phase
    is reported separately again (see A5_io_phases), because on this machine
    the dataset sits on a rotating disk while the component costs are measured
    inside train_step, after the loader has returned.
"""
import csv
import glob
import json
import os.path as osp

import numpy as np

RAW = 'review_analysis/_source_data/A5_raw'
N_BOOT = 5000
SEED = 0


def load(tag):
    p = osp.join(RAW, tag + '.json')
    return json.load(open(p)) if osp.exists(p) else None


def diff_ci(a, b, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI on median(a) - median(b) for two independent samples."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.RandomState(seed)
    d = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b)))
         for _ in range(n_boot)]
    return (float(np.median(a) - np.median(b)),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def replicates(name):
    """Both independent runs of a variant, newest first."""
    return [d for d in (load(f'train_{name}2'), load(f'train_{name}')) if d]


def ab_summary(full_reps, var_reps, label):
    """A/B difference across BOTH replicates.

    A bootstrap CI computed inside one run only measures within-run scatter.
    Where the two replicates disagree by more than that CI -- which happens for
    the smallest effect here -- the honest statement is the spread across runs,
    so both are reported.
    """
    per_run, cis = [], []
    for f, v in zip(full_reps, var_reps):
        if f.get('per_iter_ms') and v.get('per_iter_ms'):
            d, lo, hi = diff_ci(f['per_iter_ms'], v['per_iter_ms'])
            cis.append((lo, hi))
        else:
            d = f['iter_ms_median'] - v['iter_ms_median']
        per_run.append(d)
    note = f'{label}; per-run estimates ' + ', '.join(f'{x:+.2f}' for x in per_run) + ' ms'
    if cis:
        note += f'; within-run 95% CI of the first {cis[0][0]:+.2f}..{cis[0][1]:+.2f} ms'
    if len(per_run) > 1 and cis and abs(per_run[0] - per_run[1]) > (cis[0][1] - cis[0][0]):
        note += '; run-to-run spread EXCEEDS the within-run CI, so read the range, not one CI'
    return float(np.median(per_run)), min(per_run), max(per_run), note


def main():
    # prefer the reruns, which stored the raw per-iteration arrays
    full_reps = replicates('full')
    nosf_reps = replicates('no_sfilm')
    nocb_reps = replicates('no_cbacl')
    full = full_reps[0]
    nosf = nosf_reps[0]
    nocb = nocb_reps[0]
    fpsb = load('fps_baseline')
    fpsl = load('fps_liseer')
    par = load('params_liseer')
    parb = load('params_baseline')
    io = load('io_phases')

    rows = []

    def row(item, value, unit, n, mean=None, med=None, p95=None, notes=''):
        rows.append(dict(item=item, value=value, unit=unit, n_samples=n,
                         mean=mean, median=med, p95=p95, notes=notes))

    # ---- D31 ------------------------------------------------------------
    s = full['sampler_per_iter_ms'] if full.get('sampler_per_iter_ms') else None
    row('sampler_per_iter', round(full['sampler_ms_median'], 3), 'ms',
        full['n_measured'], round(full['sampler_ms_mean'], 3),
        round(full['sampler_ms_median'], 3), round(full['sampler_ms_p95'], 3),
        'D31; timed inside the model with cuda.synchronize around _sampling; '
        '1 call per iteration')
    row('sampler_pct_of_iter', round(full['sampler_pct_of_iter'], 3), 'percent',
        full['n_measured'], None, round(full['sampler_pct_of_iter'], 3), None,
        'D31; denominator is one compute iteration (fwd+bwd+step), which '
        'excludes data loading')

    # ---- D32 ------------------------------------------------------------
    d2, d2lo, d2hi, note2 = ab_summary(full_reps, nosf_reps,
                                       'D32; A/B on the sensor_film flag')
    row('sfilm_per_iter', round(d2, 3), 'ms', full['n_measured'], None,
        round(d2, 3), None, note2)
    row('sfilm_pct_of_iter',
        f'{100*d2lo/full["iter_ms_median"]:.2f}-{100*d2hi/full["iter_ms_median"]:.2f}',
        'percent', full['n_measured'], None, None, None,
        'D32; range across the two replicate runs, not a CI')
    row('sfilm_added_params', par['sfilm_params_M'], 'M', '-', None, None, None,
        f'D32; exact count {par["sfilm_params"]} = film_layer + sensor_head + '
        f'sensor_encoder; cross-check: LiSeer total {par["total_params"]} minus '
        f'baseline {parb["total_params"]} = {par["total_params"]-parb["total_params"]} '
        f'(the 66 difference is use_dim 3 vs 5 in the first layer)')

    # ---- D33 ------------------------------------------------------------
    d3, d3lo, d3hi, note3 = ab_summary(full_reps, nocb_reps,
                                       'D33; A/B on the consistency flag')
    row('cbacl_per_iter', round(d3, 3), 'ms', full['n_measured'], None,
        round(d3, 3), None,
        note3 + f'; {100*d3/full["iter_ms_median"]:.1f}% of an iteration -- CBA-CL '
                f'runs a second view, so the iteration roughly doubles')
    row('cbacl_pct_of_iter',
        f'{100*d3lo/full["iter_ms_median"]:.1f}-{100*d3hi/full["iter_ms_median"]:.1f}',
        'percent', full['n_measured'], None, None, None,
        'D33; range across replicates; training only -- CBA-CL does not exist '
        'at inference')

    # ---- D34 / D35 -------------------------------------------------------
    for tag, d in [('fps_baseline', fpsb), ('fps_liseer', fpsl)]:
        row(tag, round(d['fps_median'], 3), 'FPS', d['n_measured'],
            round(d['fps_mean'], 3), round(d['fps_median'], 3),
            round(d['fps_p95'], 3),
            f'{"D34" if "baseline" in tag else "D35"}; eval() + no_grad, batch 1, '
            f'single GPU, real scans, 50 warm-up frames; latency median '
            f'{d["latency_ms_median"]:.2f} ms; randomly initialised weights '
            f'(timing is set by the architecture)')
    dl = fpsl['latency_ms_median'] - fpsb['latency_ms_median']
    row('inference_overhead', round(dl, 3), 'ms', 200, None, round(dl, 3), None,
        f'D34 vs D35; {100*dl/fpsb["latency_ms_median"]:+.2f}% -- within noise. '
        f'CBA-CL does not exist at inference and the sampler is not on the '
        f'inference path; S-Film does run and costs nothing measurable')

    # ---- context ---------------------------------------------------------
    row('gpu_model', 'NVIDIA GeForce RTX 4090', '-', '-', None, None, None,
        'NOT the A800 used for the submitted training runs')
    row('torch_cuda', f'torch {full["torch"]} / CUDA {full["cuda"]}', '-', '-',
        None, None, None, '')
    row('voxel_grid', '480x360x32', '-', '-', None, None, None,
        'point_cloud_range +-51.2 m; TPV planes 180x240x16')
    ip = load('input_points')
    if ip:
        row('input_points_baseline', ip['E3_baseline']['train_grid'], 'points', 8,
            None, ip['E3_baseline']['train_grid'], None,
            'points entering the network for P-baseline: the real nuScenes scan')
        row('input_points_liseer', ip['E4_liseer']['train_grid'], 'points', 8,
            None, ip['E4_liseer']['train_grid'], None,
            'points entering the pipeline for LiSeer-P: the densified scene, from '
            'which Phi_samp draws the simulated scan inside the model')
    row('train_spg', 1, '-', '-', None, None, None, 'samples_per_gpu')
    row('train_num_gpus', 7, '-', '-', None, None, None,
        'E3/E4/E8 recipe; the timing above is single-GPU')
    row('train_accum', 3, '-', '-', None, None, None, '')
    row('train_effective_batch', 21, '-', '-', None, None, None, '')
    row('timing_batch', 1, '-', '-', None, None, None,
        'component timings were taken single-GPU, spg=1, no accumulation, so '
        'one timed iteration is one forward+backward+step over one sample')

    dtj = load('data_time_from_training')
    if dtj:
        for k, v in dtj.items():
            row(f'data_time_pct_{k}', round(v['pct'], 3), 'percent', v['n'], None,
                round(v['pct'], 3), None,
                f'observed in the real {k} training run (7 GPUs, 4 loader workers): '
                f'data_time {v["data_s"]*1000:.1f} ms of a {v["iter_s"]*1000:.0f} ms '
                f'wall-clock iteration -- prefetching hides the disk almost entirely')

    if io:
        row('dense_read_hdd', io['hdd_read']['ms_per_file'], 'ms/file',
            io['hdd_read']['n_files'], None, io['hdd_read']['ms_per_file'], None,
            f'data phase only; {io["hdd_read"]["MB_per_s"]} MB/s from /data '
            f'(sdd, rotational), cold cache, 4 readers')
        row('dense_read_ssd', io['ssd_read']['ms_per_file'], 'ms/file',
            io['ssd_read']['n_files'], None, io['ssd_read']['ms_per_file'], None,
            f'same files staged on / (sdc3, SSD); {io["ssd_read"]["MB_per_s"]} MB/s; '
            f'{io["speedup_ssd_over_hdd"]}x faster than the HDD')

    with open('review_analysis/_source_data/A5_cost.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['item', 'value', 'unit', 'n_samples',
                                          'mean', 'median', 'p95', 'notes'])
        w.writeheader()
        w.writerows(rows)
    print('wrote review_analysis/_source_data/A5_cost.csv')
    for r in rows:
        print(f"  {r['item']:24s} {str(r['value']):>28s} {r['unit']}")


if __name__ == '__main__':
    main()
