#!/usr/bin/env python3
"""A3 v2 Step 6: does the training curriculum COVER the real sensors?

This is the half of R2C3 that the fidelity test cannot answer. The curriculum
is a mixture over a wide configuration range; a real dataset is one fixed
sensor. Asking whether the two distributions are equal is the wrong question --
a wide mixture is never equal to a point mass, however good the sampler is. The
right question is whether the real sensor lies INSIDE the mixture's support.

The measure is the support-recall criterion used for generative models
(Kynkaanniemi et al., NeurIPS 2019), read in the per-frame histogram space with
JS as the metric:

    a real scan j is covered  <=>  its nearest simulated neighbour is no
    further than the 95th percentile of how far real scans lie from each other.

Precision (are all simulated scans like some real sensor?) is deliberately NOT
reported: the curriculum includes configurations, 16 beams for one, that match
no real dataset here on purpose, so low precision is the design working, and
quoting it would only muddle the claim.

Writes review_analysis/_source_data/A3_coverage.csv and review_analysis/_source_data/A3_coverage_scatter.npz.
"""
import csv
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

KEYS = ['f1', 'f2', 'f3_lin', 'f3_plan', 'f3_scat', 'f4']
DATASETS = ['semantickitti', 'nuscenes', 'waymo']
N_BOOT = 500
SEED = 0


K_NN = 3


def knn_radii(M_self, k=K_NN):
    """Distance from each sample to its k-th nearest neighbour within its own
    set (self-distance excluded)."""
    D = M_self.copy()
    np.fill_diagonal(D, np.inf)
    return np.partition(D, k - 1, axis=1)[:, k - 1]


def recall_knn(M_sim_real, M_sim_sim, k=K_NN):
    """Kynkaanniemi et al. (NeurIPS 2019) recall: a real scan is covered when
    it falls inside the ball some simulated scan spans out to its own k-th
    nearest simulated neighbour. The radii come from the SIMULATED set, which
    is the point -- a wide mixture covers a narrow one even though its typical
    member is not especially close to any single real scan."""
    r = knn_radii(M_sim_sim, k)
    return (M_sim_real <= r[:, None]).any(axis=0)


def strict_cover(M_sim_real, M_r2r):
    """Stricter variant: a real scan counts only if a simulated scan comes as
    close to it as real scans typically come to each other. This asks a
    fidelity question wearing a coverage label, and is reported alongside so
    the difference between the two is visible rather than hidden in a choice
    of threshold."""
    q95 = float(np.percentile(M_r2r.min(axis=0), 95))
    return M_sim_real.min(axis=0) <= q95, q95


def control_recall(M_r2r, M_aa, k=K_NN):
    """Recall of real_B by real_A: the CEILING this criterion can reach.

    With 200 samples a k=3 radius is finite, so two halves of the same real
    data score about 0.85, not 1.0. Reading a simulated recall against 1.0
    instead of against this control would overstate every shortfall.
    """
    return float(recall_knn(M_r2r, M_aa, k).mean())


def coverage_ci(M_sim_real, M_sim_sim, M_r2r, n_boot=N_BOOT, seed=SEED):
    cov = float(recall_knn(M_sim_real, M_sim_sim).mean())
    strict, q95 = strict_cover(M_sim_real, M_r2r)
    rng = np.random.RandomState(seed)
    nS, nB = M_sim_real.shape
    b = []
    for _ in range(n_boot):
        iS = rng.randint(0, nS, nS)
        jB = rng.randint(0, nB, nB)
        b.append(float(recall_knn(M_sim_real[np.ix_(iS, jB)],
                                  M_sim_sim[np.ix_(iS, iS)]).mean()))
    return (cov, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)),
            float(strict.mean()), q95)


def main():
    rows, scatter = [], {}
    for ds in DATASETS:
        p = f'review_analysis/_source_data/A3_pairmats_{ds}.npz'
        if not osp.exists(p):
            print(f'{ds}: no pair matrices, run pairwise_js.py first')
            continue
        m = np.load(p)
        for key in KEYS:
            M_r2r = m[f'{key}_r2r']
            for level in ['cur', 'cond']:
                k = f'{key}_{level}'
                if k not in m:
                    continue
                src = 'sim_cur' if level == 'cur' else 'sim_cond'
                S = np.load(f'review_analysis/_source_data/A3_perframe/{ds}_{src}.npz')[key]
                A = np.load(f'review_analysis/_source_data/A3_perframe/{ds}_realA.npz')[key]
                from A3_pairwise_js import js_matrix
                M_ss = js_matrix(S, S)
                ctrl = control_recall(M_r2r, js_matrix(A, A))
                cov, lo, hi, strict, q95 = coverage_ci(m[k], M_ss, M_r2r)
                rows.append(dict(dataset=ds, family=key, level=level,
                                 coverage=round(cov, 4), ci_lo=round(lo, 4),
                                 ci_hi=round(hi, 4),
                                 control_realA_realB=round(ctrl, 4),
                                 coverage_vs_control=round(cov / max(ctrl, 1e-9), 4),
                                 strict_coverage=round(strict, 4),
                                 q95_nn_real=round(q95, 6)))
                if level == 'cur':
                    print(f'{ds:14s} {key:8s} recall={cov:.3f} [{lo:.3f},{hi:.3f}]  '
                          f'control(realA->realB)={ctrl:.3f}  '
                          f'rel={cov/max(ctrl,1e-9):.2f}')
        # panel (f): per-frame scalars, curriculum cloud vs real clusters
        for w in ['realA', 'realB', 'sim_cur', 'sim_cond']:
            f = f'review_analysis/_source_data/A3_perframe/{ds}_{w}.npz'
            if osp.exists(f):
                d = np.load(f)
                # median nearest-neighbour distance from the F2 histogram
                from A3_run_f1f4 import F2_EDGES
                c = np.asarray(F2_EDGES, float)
                c[-1] = c[-2] + (c[-2] - c[-3])
                c = 0.5 * (c[1:] + c[:-1])
                cdf = np.cumsum(d['f2'], 1)
                med = c[np.argmax(cdf >= 0.5, axis=1)]
                scatter[f'{ds}_{w}_npts'] = d['n_full']
                scatter[f'{ds}_{w}_nnmed'] = med
    if rows:
        with open('review_analysis/_source_data/A3_coverage.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        np.savez_compressed('review_analysis/_source_data/A3_coverage_scatter.npz', **scatter)
        print(f'\nwrote review_analysis/_source_data/A3_coverage.csv ({len(rows)} rows) '
              f'and review_analysis/_source_data/A3_coverage_scatter.npz')


if __name__ == '__main__':
    main()
