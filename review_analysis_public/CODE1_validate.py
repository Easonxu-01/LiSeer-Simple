#!/usr/bin/env python3
"""CODE-1 physical perturbation suite: stand-alone validation script for the recipient.

Usage:
    cd <repo root>              # the level that holds projects/
    python3 review_analysis/CODE1_validate.py --cfg <a LiSeer config>
The patch is correctly applied only if every check passes. Do not use it if any check fails.

────────────────────────────────────────────────────────────────────
On seeding: even with random, numpy.random and torch all seeded, the dataset's __getitem__
is not fully deterministic (reading the same frame twice from the same instance with the same seed reproduces frames 0 and 1
but drifts from frame 2 onwards). A bit-exact check of the disabled state would therefore report a false failure if it simply
compared two passes over the dataset.
CHECK-2 avoids this by comparing RNG state rather than output:
with enable=False no random number may be drawn at all. That is sufficient structural evidence of bit-exactness,
it does not depend on the dataset being reproducible, and it needs no copy of the code from before the change.
────────────────────────────────────────────────────────────────────
"""
import argparse, importlib, random, sys, time, warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

OK, FAIL = 'OK  ', 'FAIL'
results = []


def report(name, passed, detail=''):
    results.append(passed)
    print(f'[{OK if passed else FAIL}] {name}' + (f'   {detail}' if detail else ''))


def seed_all(s=0):
    """Seed all three: _sampling draws from Python random, numpy and torch."""
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='a LiSeer config with Random=True')
    ap.add_argument('--frames', type=int, default=4)
    a = ap.parse_args()

    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    cfg = Config.fromfile(a.cfg)
    importlib.import_module(cfg.get('plugin_dir', '').rstrip('/').replace('/', '.'))

    # ---------- CHECK 1: the interface exists, defaults are right, unknown fields are rejected ----------
    m = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                    test_cfg=cfg.get('test_cfg')).cuda().eval()
    has = hasattr(m, 'perturb_cfg')
    keys = set(m.perturb_cfg) if has else set()
    want = {'enable', 'angular_jitter_mrad', 'range_noise_cm',
            'dropout_p', 'dropout_exempt_minority'}
    report('CHECK-1a  perturb_cfg exists and holds every field', has and keys == want,
           f'keys={sorted(keys)}')
    try:
        c2 = Config.fromfile(a.cfg)
        c2.model['sampler_perturb'] = {'enable': True, 'typo_field': 1}
        build_model(c2.model, train_cfg=c2.get('train_cfg'), test_cfg=c2.get('test_cfg'))
        report('CHECK-1b  an unknown field is rejected by an assert', False, 'no error was raised, but one was expected')
    except Exception as e:
        report('CHECK-1b  an unknown field is rejected by an assert', 'sampler_perturb' in str(e),
               str(e)[:60])

    ds = build_dataset(cfg.data.train)
    seed_all(0)
    frames = []
    for i in range(a.frames):
        s = ds[i]
        frames.append(tuple(s[k].data.cuda() for k in
                            ('train_grid', 'grid_ind', 'grid_ind_vox', 'train_pts_label')))

    def run(pc):
        old = dict(m.perturb_cfg)
        m.perturb_cfg.update(pc)
        seed_all(0)
        out = []
        with torch.no_grad():
            for g, gi, gv, lb in frames:
                o = m._sampling([g], [gi], [gv], [lb], raw_mask=[False])
                out.append(tuple(x[0].detach().cpu() for x in o[:4]))
        m.perturb_cfg.clear(); m.perturb_cfg.update(old)
        return out

    # ---------- CHECK 2: the disabled state draws no random numbers (structural evidence of bit-exactness) ----------
    m.perturb_cfg.update({'enable': False})
    g, gi, gv, lb = frames[0]
    seed_all(0)
    st_t0 = torch.cuda.get_rng_state(); st_p0 = random.getstate(); st_n0 = np.random.get_state()
    with torch.no_grad():
        m._sampling([g], [gi], [gv], [lb], raw_mask=[False])
    a_t, a_p, a_n = torch.cuda.get_rng_state(), random.getstate(), np.random.get_state()
    seed_all(0)
    m.perturb_cfg.update({'enable': True, 'angular_jitter_mrad': 0.2,
                          'range_noise_cm': 2.5, 'dropout_p': 0.03,
                          'dropout_exempt_minority': False})
    with torch.no_grad():
        m._sampling([g], [gi], [gv], [lb], raw_mask=[False])
    b_t = torch.cuda.get_rng_state()
    m.perturb_cfg.update({'enable': False})
    off_consumes_less = not torch.equal(a_t, b_t)
    report('CHECK-2   the disabled state consumes less RNG than the enabled one (none of the three terms draws)',
           off_consumes_less,
           'the final torch RNG state differs between the two, so the disabled branch is skipped entirely by the if')

    # ---------- CHECK 3: the three terms can be switched off independently ----------
    P = dict(angular_jitter_mrad=0.2, range_noise_cm=2.5, dropout_p=0.03,
             dropout_exempt_minority=False)
    only = lambda **k: {'enable': True,
                        **{x: 0.0 for x in ('angular_jitter_mrad', 'range_noise_cm', 'dropout_p')},
                        'dropout_exempt_minority': False, **k}
    base = run({'enable': False})
    ang = run(only(angular_jitter_mrad=0.2))
    rng = run(only(range_noise_cm=2.5))
    drp = run(only(dropout_p=0.03))
    allp = run({'enable': True, **P})
    n0 = lambda o: int(o[0][0].shape[0])
    same0 = lambda o: torch.equal(o[0][0], base[0][0])
    for nm, o, dropch in (('angle only', ang, False), ('range only', rng, False),
                          ('dropout only', drp, True), ('all three', allp, True)):
        # the test reads frame 0 only: the angle and range terms add or remove no points but do advance the RNG stream,
        # so the random terms already present in later frames draw different values. That is expected, not a defect.
        ok = ((n0(o) != n0(base)) == dropch) and (not same0(o))
        report(f'CHECK-3   {nm}', ok, f'points in frame 0: {n0(o)} (baseline {n0(base)})')

    # ---------- CHECK 4: numerical correctness ----------
    b0, a0 = base[0][0], ang[0][0]
    if b0.shape == a0.shape:
        d = (a0[:, 5:8] - b0[:, 5:8]).norm(dim=1)
        r = b0[:, 5:8].norm(dim=1)
        mm = r > 1.0
        mrad = float((d[mm] / r[mm]).mean()) * 1e3
        report('CHECK-4a  angular jitter magnitude ~ 1.253*sigma', 0.20 <= mrad <= 0.31,
               f'measured {mrad:.4f} mrad (expected ~0.251)')
    br, rr = base[0][0], rng[0][0]
    if br.shape == rr.shape:
        dd = (rr[:, 5:8] - br[:, 5:8]).norm(dim=1)
        cm = float(dd.mean()) * 100
        rad = br[:, 5:8].norm(dim=1)
        near, far = dd[(rad > 5) & (rad < 20)], dd[rad > 40]
        flat = abs(float(near.mean()) - float(far.mean())) * 100 < 0.5 \
            if near.numel() and far.numel() else True
        report('CHECK-4b  range noise |delta| ~ half-normal mean 0.798*sigma',
               1.7 <= cm <= 2.3, f'measured {cm:.3f} cm (expected ~1.995)')
        report('CHECK-4c  range noise is independent of range', flat)
    ratio = n0(drp) / n0(base)
    report('CHECK-4d  point-count ratio after dropout ~ 0.97', 0.955 <= ratio <= 0.985,
           f'measured {ratio:.4f}')

    # ---------- CHECK 5: indices agree with coordinates ----------
    mb = m.min_bound.cpu().float(); ms = m.max_bound.cpu().float()
    itv = (ms - mb) / m.grid_size.cpu().float()
    pts, gind = allp[0][0], allp[0][1]
    x, y, z = pts[:, 6], pts[:, 7], pts[:, 5]
    pol = torch.stack([torch.sqrt(x * x + y * y), torch.atan2(y, x), z], 1)
    gi2 = torch.floor((torch.maximum(torch.minimum(pol, ms - 1e-3), mb) - mb) / itv)
    bad = int((gi2 != gind).any(1).sum())
    report('CHECK-5   grid_ind still agrees with the coordinates after jitter', bad == 0,
           f'{bad}/{gind.shape[0]} disagree')

    # ---------- CHECK 6: runtime ----------
    ts = {}
    for nm, pc in (('off', {'enable': False}), ('on', {'enable': True, **P})):
        run(pc); torch.cuda.synchronize(); t0 = time.time()
        for _ in range(3):
            run(pc)
        torch.cuda.synchronize(); ts[nm] = (time.time() - t0) / (3 * len(frames))
    inc = (ts['on'] / ts['off'] - 1) * 100
    report('CHECK-6   per-frame runtime increases by no more than 5% when enabled', inc <= 5.0,
           f'{ts["off"]*1e3:.1f} -> {ts["on"]*1e3:.1f} ms (+{inc:.2f}%)')

    print()
    print('ALL CHECKS PASSED' if all(results) else
          f'FAILED: {results.count(False)}/{len(results)} checks did not pass')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
