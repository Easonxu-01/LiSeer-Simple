"""A4 (R2C1): output-level sensor invariance of LiSeer on one fixed scene.

Reviewer 2, comment 1 asks for *direct* evidence of scene/sensor
disentanglement rather than a performance table.  The measurement here is
**output-level invariance**: one dense scene is observed by N (default 50)
randomly drawn mechanical-LiDAR configurations, and we ask whether the
per-point semantic prediction for the *same physical point* is the same
regardless of which sensor observed it.

Why output-level and not feature probing: intermediate features are *supposed*
to carry sensor identity (S-FiLM is explicitly conditioned on the sensor
vector), so "a probe can read the sensor off the features" is the designed
behaviour, not a defect.  Disentanglement is a claim about what reaches the
*semantic head*, so it must be measured there.

Determinism (the premise of the whole experiment - NEEDS_DECISION.md
2026-08-31):
  * the dense frame is read ONCE and the SAME tensor object is handed to all N
    sampler calls (`assert pts.data_ptr() == base_ptr` after every call);
  * `model.dense_train = False`   -> `_dense_sampling`'s `randperm(N)[:500k]`
    is never reached, so the underlying point set never changes;
  * `model.test_max_points = None` -> `simple_test` never truncates;
  * minority mixup (a *training* augmentation that synthesises new points) is
    bypassed, so all N views observe the identical point set.
Together these guarantee the N views differ only in the sensor configuration.

Provenance: the sampler re-projects each returned point onto its ray centre and
adds range noise, so the output coordinates cannot be matched back to the dense
cloud.  `PointTPV_Seg._sampling` therefore grew a default-OFF provenance hook
(`self.return_provenance`) that records `kept_idx[chosen_kept_idx]`, i.e. the
row index of each sampled point in the input dense cloud.  Default behaviour is
bit-identical - see `review_analysis/_source_data/A4_provenance_proof.txt`.

Inference only.  Run on the analysis GPU (never 5/6/7 while E10-E12 train).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from review_analysis.CODE5_risley_demo import (  # noqa: E402
    load_dense_frame, MIN_BOUND, MAX_BOUND, GRID_SIZE)
# SK coarse-class mapping is imported from the project's coarse_mapping.py (single source of truth).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'projects', 'unilidar_plugin', 'datasets'))
from coarse_mapping import SEMANTICKITTI_TO_CL as SK_TO_CL  # noqa: E402

# COLA-8 label space actually used by the SemanticKITTI LiSeer models
# (nbr_class=8; class 0 = "Other"/unlabeled and is excluded from accuracy,
#  exactly as PointTPV_Seg.evaluation_semantic does with `gt != 0`).
CLASS_NAMES = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature',
               'Living Being', 'Objects', 'Other Ground']
NUM_CLASSES = 8
IGNORE_LABEL = 0

# Per-dataset sensor sampling ranges = each dataset's own training config,
# i.e. the envelope reported in Appendix Table B1.
SENSOR_RANGES = {
    'semantickitti': dict(beams=(16, 128), hres=(600, 1800), vlow=(-28, -22),
                          vup=(2, 15), height=(1.5, 1.85)),
}


# ----------------------------------------------------------------------------
# model construction
# ----------------------------------------------------------------------------
def build_liseer(ckpt_path, device, verbose=True):
    """Build the model from the checkpoint's OWN embedded training config.

    Every checkpoint stores `meta['config']`, so the architecture is
    reconstructed exactly as trained.  Loading is attempted with strict=True
    first; a fallback to strict=False is only accepted when *every* offending
    key is provably inference-irrelevant (see `_classify_keys`).
    """
    from mmcv import Config
    from mmcv.runner import load_state_dict
    from mmdet3d.models import build_model
    import projects.unilidar_plugin  # noqa: F401  (registers PointTPV_Seg etc.)

    ck = torch.load(ckpt_path, map_location='cpu')
    cfg = Config.fromstring(ck['meta']['config'], '.py')
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    # never fetch the ImageNet Swin init: every weight comes from the ckpt
    if cfg.model.get('lidar_backbone', None) is not None:
        cfg.model.lidar_backbone.init_cfg = None

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    sd = ck['state_dict']
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}

    own = model.state_dict()
    missing = sorted(set(own) - set(sd))
    unexpected = sorted(set(sd) - set(own))
    shape_mismatch = sorted(k for k in set(own) & set(sd)
                            if own[k].shape != sd[k].shape)
    strict_ok = not (missing or unexpected or shape_mismatch)
    load_state_dict(model, sd, strict=False)

    report = dict(
        checkpoint=os.path.abspath(ckpt_path),
        sha256_head=_file_sha256(ckpt_path),
        epoch=ck['meta'].get('epoch'), iter=ck['meta'].get('iter'),
        n_tensors_in_ckpt=len(sd), n_tensors_in_model=len(own),
        strict_load_ok=strict_ok,
        missing_keys=missing, unexpected_keys=unexpected,
        shape_mismatch_keys=shape_mismatch,
        unexpected_by_family={k: dict(n=len(v), examples=v[:3])
                              for k, v in key_families(unexpected).items()},
        weights_loaded=len(set(own) & set(sd)),
    )
    verdict, reason = _classify_keys(missing, unexpected, shape_mismatch)
    report['load_verdict'] = verdict
    report['load_reason'] = reason
    if verbose:
        print(f'[load] {ckpt_path}')
        print(f'[load]   epoch={report["epoch"]}  ckpt tensors={len(sd)}  '
              f'model tensors={len(own)}  strict={strict_ok}')
        print(f'[load]   missing={len(missing)} unexpected={len(unexpected)} '
              f'shape_mismatch={len(shape_mismatch)}  '
              f'loaded={report["weights_loaded"]}')
        for k in missing[:8]:
            print(f'[load]     MISSING    {k}')
        for fam, info in report['unexpected_by_family'].items():
            print(f'[load]     UNEXPECTED {info["n"]:>4}x  {fam:<28} '
                  f'e.g. {info["examples"][0]}')
        print(f'[load]   verdict: {verdict} - {reason}')
    if verdict != 'OK':
        raise RuntimeError(
            f'refusing to run inference with {ckpt_path}: {reason}. '
            'Loading past a real structural mismatch would silently evaluate a '
            'partially random network.')

    model.to(device).eval()
    return model, report


# Key families that a *training* checkpoint carries but inference never reads.
# Anything outside these is treated as a real structural mismatch and aborts.
TRAINING_ONLY_FAMILIES = [
    ('ema_ shadow weights',
     lambda k: k.startswith('ema_'),
     'EMA shadow copy maintained by the EMA hook; tools/test.py likewise loads '
     'only the primary weights, so this reproduces the published protocol'),
    ('_balance_* buffers',
     lambda k: k.startswith('_balance'),
     'GradNorm-style adaptive loss-balancer state; declared persistent=False in '
     'the current code, hence absent from the rebuilt model'),
    ('_distill_* buffers',
     lambda k: k.startswith('_distill'),
     'distillation gradient-ratio controller state (persistent=False)'),
    ('_consistency_flip buffer',
     lambda k: k.startswith('_consistency'),
     'student/teacher role-alternation counter (persistent=False)'),
    ('distillation loss modules',
     lambda k: 'spectral_distiller' in k or 'vanilla_kd' in k,
     'kernels/weights of the spectral (or vanilla-KD) distillation loss; only '
     'reachable from forward_train, never from simple_test'),
]


def key_families(keys):
    """Group keys into the known training-only families; 'OTHER' is fatal."""
    out, rest = {}, list(keys)
    for name, pred, _why in TRAINING_ONLY_FAMILIES:
        hit = [k for k in rest if pred(k)]
        if hit:
            out[name] = hit
        rest = [k for k in rest if not pred(k)]
    if rest:
        out['OTHER'] = rest
    return out


def _classify_keys(missing, unexpected, shape_mismatch):
    """Decide whether the load is safe.

    Fatal on any missing weight or shape mismatch; unexpected keys are tolerated
    only when every one of them belongs to a known training-only family.
    """
    if shape_mismatch:
        return 'FATAL', f'shape mismatch on {len(shape_mismatch)} tensors'
    if missing:
        return 'FATAL', (f'{len(missing)} weights the model needs are absent '
                         f'from the checkpoint, e.g. {missing[:3]}')
    if not unexpected:
        return 'OK', 'exact match, strict=True would have succeeded'
    fam = key_families(unexpected)
    if 'OTHER' in fam:
        return 'FATAL', ('unexpected keys outside every known training-only '
                         f'family: {fam["OTHER"][:5]}')
    summary = ', '.join(f'{n}x {k}' for k, n in
                        ((k, len(v)) for k, v in fam.items()))
    return 'OK', (f'0 missing weights; {len(unexpected)} unexpected keys are '
                  f'all training-only state ({summary}), none reachable from '
                  f'simple_test')


def _file_sha256(path, nbytes=64 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------
# sensor configurations + sampling
# ----------------------------------------------------------------------------
def roll_configs(dataset, n, seed):
    """N configurations drawn uniformly over the dataset's declared envelope.

    Uniform (rather than training's canonical-line PMF for `beams` and its
    100-step grid for `hres`) deliberately covers the *whole* Table B1 range,
    including non-canonical line counts - a strictly harder test.
    """
    r = SENSOR_RANGES[dataset]
    rng = random.Random(seed)
    return [dict(beams=rng.randint(*r['beams']),
                 hres=rng.randint(*r['hres']),
                 vlow=round(rng.uniform(*r['vlow']), 1),
                 vup=round(rng.uniform(*r['vup']), 1),
                 height=round(rng.uniform(*r['height']), 2))
            for _ in range(n)]


def prepare_sampler(model, dataset):
    """Put the real sampler into the deterministic inference regime."""
    from projects.unilidar_plugin.pointocc_model.pointtpv_seg import PointTPV_Seg
    if not hasattr(PointTPV_Seg, '_sampling'):
        raise RuntimeError('sampler missing')
    r = SENSOR_RANGES[dataset]
    # `consistency` on the DETECTOR only selects the paired-view branch inside
    # _sampling; the aggregator's own `consistency` flag is left untouched and
    # is inert at test time anyway (it needs sensor_vec, which is None here).
    model.consistency = False
    model.sample_repeat_count = 1
    model.dense_train = False          # never enter _dense_sampling's randperm
    model.test_max_points = None       # never truncate in simple_test
    model.sensor_config = dict(
        LiDAR_height=list(r['height']), num_of_beams=list(r['beams']),
        horizontal_angular_resolution=list(r['hres']),
        lower_vertical_field_of_view_bound=list(r['vlow']),
        upper_vertical_field_of_view_bound=list(r['vup']))
    # Minority mixup synthesises *new* points, which would break "the same
    # underlying scene observed N times".  Bypass it (this is a training-time
    # augmentation and has no role at inference).  The `random.random() <
    # mixup_prob` draw still happens, so the RNG stream is unchanged.
    model._mixup_minority = (
        lambda p, gi, giv, l, num_add=2: (p, gi, giv, l))
    assert model.dense_train is False
    assert getattr(model, 'test_max_points', None) is None
    assert getattr(model, 'beam_concentration', None) is None, \
        'beam_concentration must stay at the training default (uniform beams)'


def sample_with_config(model, pts, gi, giv, lab, cfg, seed):
    """One scan through the model's OWN `_sampling`, with the sensor
    configuration forced to `cfg` and provenance recorded.

    The three module-level draws that pick the sensor parameters are
    temporarily replaced (same technique as review_analysis/A6_figR4_multi.py); every
    other stochastic element of the scan - azimuth phase, occlusion, ring
    smoothing, range noise, dropout - runs untouched, seeded by `seed`.
    """
    import projects.unilidar_plugin.pointocc_model.pointtpv_seg as M

    random.seed(seed)
    np.random.seed(seed % (2 ** 31))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    real_uniform, real_choice, real_beams = (
        M.random.uniform, M.random.choice, M.sample_beams)
    order = {'n': 0}

    def fake_uniform(a, b):
        # _sample_sensor_params_tuple draws, in order: height, vlow, vup.
        order['n'] += 1
        return {1: cfg['height'], 2: cfg['vlow'], 3: cfg['vup']}.get(
            order['n'], real_uniform(a, b))

    M.random.uniform = fake_uniform
    M.random.choice = lambda _seq: cfg['hres']
    M.sample_beams = lambda lo, hi: cfg['beams']
    model.return_provenance = True
    model._provenance_buffer = []
    try:
        out = model._sampling([pts], [gi], [giv], [lab])
    finally:
        M.random.uniform, M.random.choice, M.sample_beams = (
            real_uniform, real_choice, real_beams)
        model.return_provenance = False
    buf = model._provenance_buffer
    model._provenance_buffer = []

    assert order['n'] >= 3, (
        f'sensor draw was not intercepted ({order["n"]} uniform calls)')
    rp, rg, rgv, rl, sp = out
    assert len(buf) == 1 and buf[0][0] == 0, (
        f'expected provenance for exactly one frame, got {len(buf)} entries')
    prov = buf[0][1]
    assert prov.shape[0] == rp[0].shape[0], (
        f'provenance/points length mismatch: {prov.shape[0]} vs {rp[0].shape[0]}')
    assert int(prov.max()) < pts.shape[0], (
        'provenance index outside the dense cloud - mixup was not bypassed')
    # sensor params actually used, read back out of the sampler's own tensor
    used = sp[0].tolist()
    assert int(used[0]) == cfg['beams'] and int(used[2]) == cfg['hres'], (
        f'forced config not honoured: sampler used {used}, asked {cfg}')
    return rp[0], rg[0], rgv[0], rl[0], prov, used


@torch.no_grad()
def predict(model, feats, grid_ind, grid_ind_vox, labels, dataset_flag=2):
    """Per-point semantic prediction through the model's real inference entry."""
    out = model.simple_test(
        train_grid=[feats], grid_ind=[grid_ind], grid_ind_vox=[grid_ind_vox],
        train_voxel_label=None, train_pts_label=[labels],
        dataset_flag=torch.tensor([dataset_flag], device=feats.device),
        return_loss=False)
    pred = out['pred']
    if pred.dim() == 2:
        pred = pred[0]
    assert pred.shape[0] == feats.shape[0], (
        f'prediction length {pred.shape[0]} != input {feats.shape[0]}')
    return pred


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def compute_metrics(votes, obs, gt, per_config, min_obs, restrict=None):
    """All statistics restricted to points seen by >= `min_obs` configurations.

    votes : [N, C] int32, votes[i, c] = #configs that predicted c for dense point i
    obs   : [N]    int32, #configs that observed dense point i
    gt    : [N]    int64, COLA-8 ground truth of the dense cloud
    per_config : list of (prov_idx np.int64[n_c], pred np.int64[n_c])
    restrict : optional [N] bool, an extra support restriction (used to score the
               same-sensor control on the varying-sensor run's qualifying set)
    """
    qual = obs >= min_obs
    if restrict is not None:
        qual = qual & restrict
    n_qual = int(qual.sum())
    if n_qual == 0:
        raise RuntimeError(f'no dense point was observed by >= {min_obs} configs')

    top = votes[qual].max(axis=1)
    obs_q = obs[qual]
    unanimous = top == obs_q                        # every observer agreed
    agreement_pct = 100.0 * unanimous.mean()
    # How concentrated are the votes?  Strict unanimity is a demanding
    # criterion when a point has ~15 observers; the modal vote share says how
    # close to unanimous the typical point actually is.
    modal_share = 100.0 * (top / np.maximum(obs_q, 1))

    majority = votes[qual].argmax(axis=1)
    gt_q = gt[qual]
    lab = gt_q != IGNORE_LABEL                      # same rule as evaluation_semantic
    correct = majority == gt_q
    majority_acc = 100.0 * (majority[lab] == gt_q[lab]).mean()
    agreement_pct_labeled = 100.0 * unanimous[lab].mean()
    # Is the disagreement structured?  If the views only disagree where the
    # consensus itself is wrong, the disagreement is scene ambiguity, not
    # sensor sensitivity.
    m_ok, m_bad = lab & correct, lab & ~correct
    agree_when_right = float(100.0 * unanimous[m_ok].mean()) if m_ok.any() else None
    agree_when_wrong = float(100.0 * unanimous[m_bad].mean()) if m_bad.any() else None

    maj_full = np.zeros(gt.shape[0], dtype=np.int64)
    maj_full[qual] = majority
    accs, ns, matched, consensus = [], [], [], []
    for prov, pred in per_config:
        m = qual[prov] & (gt[prov] != IGNORE_LABEL)
        if m.sum() == 0:
            continue
        accs.append(100.0 * (pred[m] == gt[prov][m]).mean())
        ns.append(int(m.sum()))
        # majority-vote accuracy on exactly this configuration's own support,
        # so "ensemble vs single view" is an apples-to-apples comparison
        matched.append(100.0 * (maj_full[prov][m] == gt[prov][m]).mean())
        # label-free invariance of THIS view: how often does it reproduce the
        # cross-sensor consensus?  Plotted against each sensor parameter, this
        # is the direct "does the prediction depend on the sensor?" curve.
        mq = qual[prov]
        consensus.append(100.0 * (pred[mq] == maj_full[prov][mq]).mean())
    accs = np.asarray(accs, dtype=np.float64)
    matched = np.asarray(matched, dtype=np.float64)
    consensus = np.asarray(consensus, dtype=np.float64)

    # per-class agreement over the qualifying set (label-conditioned)
    per_class = {}
    for c in range(NUM_CLASSES):
        m = gt_q == c
        per_class[CLASS_NAMES[c]] = dict(
            n=int(m.sum()),
            agreement_pct=float(100.0 * unanimous[m].mean()) if m.any() else None,
            majority_acc_pct=(float(100.0 * (majority[m] == c).mean())
                              if m.any() else None))

    # agreement as a function of the >=k threshold (is k=10 cherry-picked?)
    n_cfg = int(obs.max())
    sweep = []
    for k in range(1, n_cfg + 1):
        qk = obs >= k
        if not qk.any():
            sweep.append(dict(k=k, n=0, agreement_pct=None,
                              majority_acc_pct=None))
            continue
        uk = votes[qk].max(axis=1) == obs[qk]
        mk = votes[qk].argmax(axis=1)
        gk = gt[qk]
        lk = gk != IGNORE_LABEL
        sweep.append(dict(
            k=k, n=int(qk.sum()),
            agreement_pct=float(100.0 * uk.mean()),
            majority_acc_pct=(float(100.0 * (mk[lk] == gk[lk]).mean())
                              if lk.any() else None)))

    # Degeneracy control: perfect agreement is trivial for a model that always
    # predicts one class.  Report the majority-vote class distribution and its
    # normalised entropy so the reader can see the predictions are structured.
    counts = np.bincount(majority, minlength=NUM_CLASSES).astype(np.float64)
    p = counts / counts.sum()
    nz = p[p > 0]
    pred_entropy = float(-(nz * np.log(nz)).sum() / np.log(NUM_CLASSES))
    gtc = np.bincount(gt_q, minlength=NUM_CLASSES).astype(np.float64)
    gtp = gtc / gtc.sum()
    gtnz = gtp[gtp > 0]

    return dict(
        qualifying_points=n_qual,
        agreement_pct=float(agreement_pct),
        majority_vote_acc_pct=float(majority_acc),
        per_config_acc_mean=float(accs.mean()),
        per_config_acc_sd=float(accs.std(ddof=1)),
        per_config_acc_min=float(accs.min()),
        per_config_acc_max=float(accs.max()),
        # ---- diagnostics (not part of Table R9's required fields) ----
        qualifying_points_labeled=int(lab.sum()),
        agreement_pct_labeled=float(agreement_pct_labeled),
        mean_modal_vote_share_pct=float(modal_share.mean()),
        median_modal_vote_share_pct=float(np.median(modal_share)),
        agreement_pct_where_majority_correct=agree_when_right,
        agreement_pct_where_majority_wrong=agree_when_wrong,
        majority_acc_matched_mean=float(matched.mean()),
        ensemble_gain_pp=float(matched.mean() - accs.mean()),
        mean_observations_qualifying=float(obs_q.mean()),
        dense_points=int(obs.shape[0]),
        observed_at_least_once=int((obs > 0).sum()),
        mean_observations=float(obs[obs > 0].mean()),
        per_config_acc=accs.tolist(),
        per_config_consensus_agreement=consensus.tolist(),
        per_config_consensus_agreement_mean=float(consensus.mean()),
        per_config_consensus_agreement_min=float(consensus.min()),
        per_config_eval_points=ns,
        per_class=per_class,
        threshold_sweep=sweep,
        # degeneracy control
        majority_class_distribution={CLASS_NAMES[c]: float(100 * p[c])
                                     for c in range(NUM_CLASSES)},
        majority_pred_entropy_norm=pred_entropy,
        gt_entropy_norm=float(-(gtnz * np.log(gtnz)).sum() / np.log(NUM_CLASSES)),
        n_classes_predicted=int((counts > 0).sum()),
    )


def sanity_check(m, n_configs, min_obs):
    """Catch incoherent numbers before they are reported."""
    ok, notes = True, []
    a = np.asarray(m['per_config_acc'])

    c1 = m['per_config_acc_min'] <= m['per_config_acc_mean'] <= m['per_config_acc_max']
    notes.append(f'[{"ok" if c1 else "FAIL"}] mean in [min, max]: '
                 f'{m["per_config_acc_min"]:.3f} <= {m["per_config_acc_mean"]:.3f} '
                 f'<= {m["per_config_acc_max"]:.3f}')
    ok &= c1

    c2 = abs(a.mean() - m['per_config_acc_mean']) < 1e-9 and \
        abs(a.std(ddof=1) - m['per_config_acc_sd']) < 1e-9
    notes.append(f'[{"ok" if c2 else "FAIL"}] mean/sd recomputed from the '
                 f'{len(a)} per-config accuracies match')
    ok &= c2

    # On a unanimous point every configuration predicts what the majority
    # predicts, so a single configuration can differ from the majority vote
    # only on non-unanimous points.  Hence the gap is bounded by the
    # disagreement rate.
    gap = abs(m['majority_vote_acc_pct'] - m['per_config_acc_mean'])
    bound = 100.0 - m['agreement_pct_labeled']
    c3 = gap <= bound + 1e-6
    notes.append(f'[{"ok" if c3 else "FAIL"}] |majority - per-config mean| = '
                 f'{gap:.3f} <= disagreement rate {bound:.3f}')
    ok &= c3

    c4 = m['majority_vote_acc_pct'] >= m['per_config_acc_mean'] - 1e-6
    notes.append(f'[{"ok" if c4 else "note"}] majority vote '
                 f'({m["majority_vote_acc_pct"]:.3f}) >= per-config mean '
                 f'({m["per_config_acc_mean"]:.3f})')

    sw = [s for s in m['threshold_sweep'] if s['n'] > 0]
    ns = [s['n'] for s in sw]
    c5 = all(ns[i] >= ns[i + 1] for i in range(len(ns) - 1))
    at10 = next((s for s in sw if s['k'] == min_obs), None)
    notes.append(f'[{"ok" if c5 else "FAIL"}] qualifying_points decreases '
                 f'monotonically with k: k=1 -> {ns[0]:,}, '
                 f'k={min_obs} -> {at10["n"]:,}, k={n_configs} -> {ns[-1]:,}')
    ok &= c5

    c6 = at10 is not None and at10['n'] == m['qualifying_points']
    notes.append(f'[{"ok" if c6 else "FAIL"}] sweep at k={min_obs} reproduces '
                 f'qualifying_points ({m["qualifying_points"]:,})')
    ok &= c6

    c7 = 0.0 <= m['agreement_pct'] <= 100.0 and 0.0 <= m['majority_vote_acc_pct'] <= 100.0
    notes.append(f'[{"ok" if c7 else "FAIL"}] percentages in [0, 100]')
    ok &= c7

    c8 = len(a) == n_configs
    notes.append(f'[{"ok" if c8 else "FAIL"}] {len(a)} per-config accuracies '
                 f'for {n_configs} configurations')
    ok &= c8

    # Agreement is only evidence of invariance if the predictions are not
    # degenerate (a constant-output model would agree with itself perfectly).
    c9 = m['n_classes_predicted'] >= 4 and m['majority_pred_entropy_norm'] > 0.3
    top = max(m['majority_class_distribution'].values())
    notes.append(f'[{"ok" if c9 else "FAIL"}] predictions non-degenerate: '
                 f'{m["n_classes_predicted"]}/{NUM_CLASSES} classes predicted, '
                 f'normalised entropy {m["majority_pred_entropy_norm"]:.3f} '
                 f'(GT {m["gt_entropy_norm"]:.3f}), largest class {top:.1f}%')
    ok &= c9
    return ok, notes


# ----------------------------------------------------------------------------
def run_sweep(model, args, configs, frame, device, label):
    """Sample + predict once per configuration, accumulating per-dense-point votes."""
    pts, gi, giv, lab = frame
    base_ptr, gi_ptr = pts.data_ptr(), gi.data_ptr()
    n_dense = pts.shape[0]

    votes = torch.zeros((n_dense, NUM_CLASSES), dtype=torch.int32, device=device)
    obs = torch.zeros(n_dense, dtype=torch.int32, device=device)
    per_config, cfg_log = [], []

    for i, cfg in enumerate(configs):
        t0 = time.time()
        seed = args.seed * 1000 + i
        rp, rg, rgv, rl, prov, used = sample_with_config(
            model, pts, gi, giv, lab, cfg, seed)
        # DETERMINISM GUARD: the same tensor object must serve every call.
        assert pts.data_ptr() == base_ptr, 'the dense input tensor was replaced!'
        assert gi.data_ptr() == gi_ptr, 'the dense grid index was replaced!'
        pred = predict(model, rp, rg, rgv, rl).to(device)
        assert pts.data_ptr() == base_ptr, 'the dense input tensor was replaced!'

        obs.index_add_(0, prov, torch.ones_like(prov, dtype=torch.int32))
        votes.index_put_((prov, pred.to(torch.int64)),
                         torch.ones(prov.shape[0], dtype=torch.int32, device=device),
                         accumulate=True)
        per_config.append((prov.cpu().numpy().astype(np.int64),
                           pred.cpu().numpy().astype(np.int64)))
        cfg_log.append(dict(index=i, seed=seed, **cfg, n_points=int(rp.shape[0]),
                            sensor_params_used=used))
        print(f'  [{label} {i + 1:>3}/{len(configs)}] beams={cfg["beams"]:>2} '
              f'hres={cfg["hres"]:>4} vlow={cfg["vlow"]:>6} vup={cfg["vup"]:>5} '
              f'h={cfg["height"]:.2f} -> {rp.shape[0]:>7,} pts '
              f'({time.time() - t0:.1f}s)', flush=True)
        del rp, rg, rgv, rl, pred
        torch.cuda.empty_cache()

    return votes.cpu().numpy(), obs.cpu().numpy(), per_config, cfg_log


def run_checkpoint(ckpt, args, configs, frame, device):
    model, load_report = build_liseer(ckpt, device)
    prepare_sampler(model, args.dataset)
    gt = frame[3].cpu().numpy().astype(np.int64)

    votes, obs, per_config, cfg_log = run_sweep(
        model, args, configs, frame, device, 'vary')
    m = compute_metrics(votes, obs, gt, per_config, args.min_obs)
    m['configs'] = cfg_log
    m['load_report'] = load_report

    # ---- same-sensor control -------------------------------------------
    # Repeat the sweep with ONE fixed configuration.  Everything else - frame,
    # per-call seeds, number of views, the whole stochastic scan pipeline
    # (azimuth phase, dropout, range noise) - is identical.  The only
    # difference is whether the sensor parameters vary.  The gap between the
    # two agreement numbers is therefore the part of the disagreement that is
    # attributable to the sensor configuration itself, as opposed to ordinary
    # scan-sampling noise.  This is a within-model control, not a baseline
    # model comparison (see D-A4-2).
    if args.control:
        fixed = configs[args.control_index]
        print(f'  [control] all {len(configs)} views use one fixed config: {fixed}')
        c_votes, c_obs, c_per, c_log = run_sweep(
            model, args, [fixed] * len(configs), frame, device, 'ctrl')
        ctrl = compute_metrics(c_votes, c_obs, gt, c_per, args.min_obs)
        # A fixed configuration only ever illuminates the points its own rays
        # reach, so the two sweeps do not qualify the same points.  Score BOTH
        # runs on the intersection of the two qualifying sets - that is the
        # only apples-to-apples comparison.
        both = (obs >= args.min_obs) & (c_obs >= args.min_obs)
        ctrl_matched = compute_metrics(c_votes, c_obs, gt, c_per, args.min_obs,
                                       restrict=both)
        vary_matched = compute_metrics(votes, obs, gt, per_config, args.min_obs,
                                       restrict=both)
        m['control_same_config'] = dict(
            config=fixed, config_index=args.control_index,
            qualifying_points=ctrl['qualifying_points'],
            agreement_pct=ctrl['agreement_pct'],
            mean_modal_vote_share_pct=ctrl['mean_modal_vote_share_pct'],
            majority_vote_acc_pct=ctrl['majority_vote_acc_pct'],
            per_config_acc_mean=ctrl['per_config_acc_mean'],
            per_config_acc_sd=ctrl['per_config_acc_sd'],
            mean_observations_qualifying=ctrl['mean_observations_qualifying'],
            # both sweeps scored on the INTERSECTION of the two supports:
            matched_support_qualifying_points=ctrl_matched['qualifying_points'],
            matched_support_agreement_pct=ctrl_matched['agreement_pct'],
            matched_support_modal_vote_share_pct=(
                ctrl_matched['mean_modal_vote_share_pct']),
            matched_support_majority_acc_pct=ctrl_matched['majority_vote_acc_pct'],
            varying_on_matched_support_agreement_pct=vary_matched['agreement_pct'],
            varying_on_matched_support_modal_vote_share_pct=(
                vary_matched['mean_modal_vote_share_pct']),
            varying_on_matched_support_majority_acc_pct=(
                vary_matched['majority_vote_acc_pct']),
            configs=c_log,
        )
        # the gap on an identical point set = the part of the disagreement that
        # the sensor configuration is responsible for
        m['sensor_attributable_disagreement_pp'] = float(
            ctrl_matched['agreement_pct'] - vary_matched['agreement_pct'])
        m['sensor_attributable_vote_share_pp'] = float(
            ctrl_matched['mean_modal_vote_share_pct']
            - vary_matched['mean_modal_vote_share_pct'])

    del model
    torch.cuda.empty_cache()
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', nargs='+', required=True,
                    help='one or more LiSeer checkpoints; all of them see the '
                         'SAME frame and the SAME sensor configurations')
    ap.add_argument('--dataset', default='semantickitti',
                    choices=sorted(SENSOR_RANGES))
    ap.add_argument('--n-configs', type=int, default=50)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seq', type=int, default=8,
                    help='SemanticKITTI sequence (8 = the held-out val split)')
    ap.add_argument('--frame', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-obs', type=int, default=10,
                    help='a dense point counts only if at least this many '
                         'configurations observed it')
    ap.add_argument('--outdir', default='results')
    ap.add_argument('--tag', default='A4')
    ap.add_argument('--beams', type=int, nargs=2, metavar=('LO', 'HI'),
                    default=None,
                    help='override the beam-count envelope, e.g. --beams 32 64 '
                         'for the training range or --beams 16 128 to extrapolate '
                         'a factor of two beyond it at both ends. Every other '
                         'sensor parameter keeps its declared range. Both '
                         'envelopes are reported in A4_IMPLEMENTATION_NOTE.md; '
                         'the numbers are NOT interchangeable (different '
                         'qualifying-point supports)')
    ap.add_argument('--control', action='store_true', default=True,
                    help='also run the same-sensor control sweep (default on)')
    ap.add_argument('--no-control', dest='control', action='store_false')
    ap.add_argument('--control-index', type=int, default=0,
                    help='which of the rolled configurations the control repeats')
    args = ap.parse_args()

    if args.dataset != 'semantickitti':
        raise SystemExit(
            'Only SemanticKITTI is supported: this machine holds no LiSeer '
            'weights for nuScenes/Waymo (the dense teachers are NOT LiSeer). '
            'See A4_IMPLEMENTATION_NOTE.md.')

    if args.beams is not None:
        lo, hi = args.beams
        if lo < 1 or hi < lo:
            raise SystemExit(f'--beams LO HI must satisfy 1 <= LO <= HI '
                             f'(got {lo} {hi})')
        SENSOR_RANGES[args.dataset]['beams'] = (lo, hi)
    print(f'[envelope] beams = {SENSOR_RANGES[args.dataset]["beams"]}')

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    print(f'[gpu] {args.device} -> {torch.cuda.get_device_name(device)}')

    # ---- the frame is read ONCE; every sampler call reuses this tensor ----
    pts, gi, giv, lab = load_dense_frame(args.seq, args.frame, device)
    frame_id = f'semantickitti/seq{args.seq:02d}/{args.frame:06d}'
    print(f'[frame] {frame_id}: {pts.shape[0]:,} dense points, '
          f'{int((lab != IGNORE_LABEL).sum()):,} labelled')
    print(f'[frame] tensor data_ptr = {pts.data_ptr()} (asserted after every call)')

    configs = roll_configs(args.dataset, args.n_configs, args.seed)
    print(f'[cfg] {len(configs)} configurations rolled from seed {args.seed}; '
          f'identical for every checkpoint')

    raw = dict(
        frame_id=frame_id, dataset=args.dataset, n_configs=args.n_configs,
        min_obs=args.min_obs, seed=args.seed, device=args.device,
        dense_points=int(pts.shape[0]),
        sensor_ranges=SENSOR_RANGES[args.dataset],
        class_names=CLASS_NAMES, ignore_label=IGNORE_LABEL,
        configs=configs, checkpoints={}, generated=time.strftime('%Y-%m-%d %H:%M:%S'),
    )
    raw_path = os.path.join(args.outdir, f'{args.tag}_raw.json')

    for ckpt in args.ckpt:
        name = os.path.splitext(os.path.basename(ckpt))[0]
        print(f'\n===== {name} =====', flush=True)
        t0 = time.time()
        m = run_checkpoint(ckpt, args, configs, (pts, gi, giv, lab), device)
        m['wall_seconds'] = round(time.time() - t0, 1)
        ok, notes = sanity_check(m, args.n_configs, args.min_obs)
        m['sanity_ok'] = bool(ok)
        m['sanity_notes'] = notes
        raw['checkpoints'][name] = m
        print(f'\n[{name}] qualifying_points      = {m["qualifying_points"]:,} '
              f'(mean {m["mean_observations_qualifying"]:.1f} observers/point)')
        print(f'[{name}] agreement_pct          = {m["agreement_pct"]:.2f}')
        print(f'[{name}] modal vote share       = {m["mean_modal_vote_share_pct"]:.2f} '
              f'(median {m["median_modal_vote_share_pct"]:.2f})')
        print(f'[{name}] majority_vote_acc_pct  = {m["majority_vote_acc_pct"]:.2f}')
        print(f'[{name}] per_config_acc         = {m["per_config_acc_mean"]:.2f} '
              f'+/- {m["per_config_acc_sd"]:.2f} '
              f'[{m["per_config_acc_min"]:.2f}, {m["per_config_acc_max"]:.2f}]')
        print(f'[{name}] unanimity | majority correct = '
              f'{m["agreement_pct_where_majority_correct"]:.2f}  '
              f'| majority wrong = {m["agreement_pct_where_majority_wrong"]:.2f}')
        if 'control_same_config' in m:
            c = m['control_same_config']
            print(f'[{name}] CONTROL (one fixed sensor, {args.n_configs} repeats):')
            print(f'[{name}]   control on its own support    = '
                  f'{c["agreement_pct"]:.2f} ({c["qualifying_points"]:,} pts, '
                  f'{c["mean_observations_qualifying"]:.1f} obs/pt)')
            print(f'[{name}]   --- on the SHARED support '
                  f'({c["matched_support_qualifying_points"]:,} pts) ---')
            print(f'[{name}]   50 different sensors : unanimity '
                  f'{c["varying_on_matched_support_agreement_pct"]:.2f}, '
                  f'vote share '
                  f'{c["varying_on_matched_support_modal_vote_share_pct"]:.2f}')
            print(f'[{name}]   1 sensor x 50 repeats: unanimity '
                  f'{c["matched_support_agreement_pct"]:.2f}, vote share '
                  f'{c["matched_support_modal_vote_share_pct"]:.2f}')
            print(f'[{name}]   => disagreement attributable to the SENSOR = '
                  f'{m["sensor_attributable_disagreement_pp"]:.2f} pp '
                  f'(unanimity), {m["sensor_attributable_vote_share_pp"]:.2f} pp '
                  f'(modal vote share)')
        print(f'[{name}] --- sanity ---')
        for n in notes:
            print(f'[{name}]   {n}')
        print(f'[{name}] sanity_ok = {ok}   ({m["wall_seconds"]}s)')
        # persist after every checkpoint so a disconnect never loses work
        with open(raw_path, 'w') as f:
            json.dump(raw, f, indent=1)
        print(f'[{name}] wrote {raw_path}')

    write_table(raw, os.path.join(args.outdir, f'{args.tag}_table_R9.csv'))
    print('\n[A4] done')


def write_table(raw, path):
    import csv
    cols = ['checkpoint', 'dataset', 'frame_id', 'n_configs', 'min_observations',
            'qualifying_points', 'agreement_pct', 'majority_vote_acc_pct',
            'per_config_acc_mean', 'per_config_acc_sd', 'per_config_acc_min',
            'per_config_acc_max', 'mean_modal_vote_share_pct',
            'agreement_pct_ctrl_same_sensor', 'sensor_attributable_disagreement_pp',
            'dense_points', 'observed_at_least_once',
            'mean_observations', 'mean_observations_qualifying',
            'qualifying_points_labeled',
            'agreement_pct_labeled', 'ckpt_epoch', 'load_verdict',
            'n_unexpected_keys', 'n_missing_keys', 'sanity_ok']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, m in raw['checkpoints'].items():
            lr = m['load_report']
            ctrl = m.get('control_same_config')
            w.writerow([
                name, raw['dataset'], raw['frame_id'], raw['n_configs'],
                raw['min_obs'], m['qualifying_points'],
                f'{m["agreement_pct"]:.2f}', f'{m["majority_vote_acc_pct"]:.2f}',
                f'{m["per_config_acc_mean"]:.2f}', f'{m["per_config_acc_sd"]:.2f}',
                f'{m["per_config_acc_min"]:.2f}', f'{m["per_config_acc_max"]:.2f}',
                f'{m["mean_modal_vote_share_pct"]:.2f}',
                (f'{ctrl["matched_support_agreement_pct"]:.2f}' if ctrl else ''),
                (f'{m["sensor_attributable_disagreement_pp"]:.2f}' if ctrl else ''),
                m['dense_points'], m['observed_at_least_once'],
                f'{m["mean_observations"]:.2f}',
                f'{m["mean_observations_qualifying"]:.2f}',
                m['qualifying_points_labeled'],
                f'{m["agreement_pct_labeled"]:.2f}', lr['epoch'],
                lr['load_verdict'], len(lr['unexpected_keys']),
                len(lr['missing_keys']), m['sanity_ok']])
    print(f'[table] wrote {path}')


if __name__ == '__main__':
    main()
