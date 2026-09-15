"""Optional physical-perturbation suite: off by default, statistics when on."""
import math
import random

import numpy as np
import pytest
import torch

from tests._stub_env import load_sampler_modules
from tests.sampler_harness import (BASELINE_NPZ, build_model, run_all_frames,
                                   run_frame, seed_all)
from tests.synth_frames import make_frame

SUITE_ON = dict(enabled=True, angular_jitter_mrad=0.2,
                range_noise_cm=2.5, ray_dropout=0.03)
SUITE_OFF = dict(SUITE_ON, enabled=False)


@pytest.fixture(scope='module')
def baseline():
    return np.load(BASELINE_NPZ)


def _assert_matches(runs, baseline):
    for i, run in enumerate(runs):
        for key, value in run.items():
            ref = baseline[f'frame{i}_{key}']
            assert ref.shape == value.shape, f'frame{i} {key}: shape changed'
            assert np.array_equal(ref, value), f'frame{i} {key}: values changed'


# ---------------------------------------------------------------- default path


def test_default_path_is_bit_exact(baseline):
    """No perturbation argument at all: output identical to the committed baseline."""
    _assert_matches(run_all_frames(build_model()), baseline)


def test_disabled_suite_is_bit_exact(baseline):
    """Parameters present but ``enabled: False``: still identical."""
    _assert_matches(run_all_frames(build_model(perturb_suite=SUITE_OFF)), baseline)


def test_codeone_aliases_are_accepted(baseline):
    """The CODE-1 spelling maps onto the same switch."""
    model = build_model(sampler_perturb=dict(enable=False, dropout_p=0.03,
                                             angular_jitter_mrad=0.2,
                                             range_noise_cm=2.5))
    assert model.perturb_cfg['enabled'] is False
    assert model.perturb_cfg['ray_dropout'] == pytest.approx(0.03)
    _assert_matches(run_all_frames(model), baseline)


def test_unknown_field_is_rejected():
    with pytest.raises(AssertionError):
        build_model(perturb_suite=dict(enabled=True, angular_jitter_mrd=0.2))


def test_disabled_suite_consumes_no_random_numbers():
    """A disabled suite must not advance the RNG, or every later draw shifts."""
    frame = make_frame(1001)

    def rng_state_after(**kwargs):
        model = build_model(**kwargs)
        run_frame(model, frame, seed=4242)
        return torch.get_rng_state().clone()

    off = rng_state_after()
    off_explicit = rng_state_after(perturb_suite=SUITE_OFF)
    on = rng_state_after(perturb_suite=SUITE_ON)
    assert torch.equal(off, off_explicit), 'disabled suite touched the RNG'
    assert not torch.equal(off, on), 'enabled suite drew no random numbers'


def test_eval_mode_disables_the_suite(baseline):
    """The suite is training-time only: evaluation must be unperturbed."""
    model = build_model(perturb_suite=SUITE_ON)
    model.eval()
    _assert_matches(run_all_frames(model), baseline)


def test_enabled_suite_changes_the_output():
    frame = make_frame(1001)
    plain = run_frame(build_model(), frame, seed=99)
    perturbed = run_frame(build_model(perturb_suite=SUITE_ON), frame, seed=99)
    assert perturbed['points'].shape[0] < plain['points'].shape[0]
    kept = perturbed['points'].shape[0] / plain['points'].shape[0]
    assert 0.955 < kept < 0.985, f'kept fraction {kept:.4f} inconsistent with p=3%'


# ------------------------------------------------------- operator statistics
# The operators are exercised directly: 200 repetitions of a fixed ray set give
# tight enough statistics without re-running the whole sampler 200 times.

REPEATS = 200
NUM_RAYS = 4096


def _fixed_rays(n=NUM_RAYS):
    g = torch.Generator().manual_seed(7)
    el = torch.rand(n, generator=g) * math.radians(28.0) - math.radians(25.0)
    az = torch.rand(n, generator=g) * 2 * math.pi
    return torch.stack([torch.cos(el) * torch.cos(az),
                        torch.cos(el) * torch.sin(az),
                        torch.sin(el)], dim=1)


def test_angular_jitter_statistics():
    _, common, _ = load_sampler_modules()
    sigma_mrad = 0.2
    rays = _fixed_rays()
    # tangent basis, same construction the operator uses
    aux = torch.where((rays[:, 2].abs() < 0.9).unsqueeze(1),
                      rays.new_tensor([0., 0., 1.]).expand_as(rays),
                      rays.new_tensor([1., 0., 0.]).expand_as(rays))
    e1 = torch.cross(rays, aux, dim=1)
    e1 = e1 / e1.norm(dim=1, keepdim=True)
    e2 = torch.cross(rays, e1, dim=1)

    seed_all(31337)
    comp1, comp2 = [], []
    for _ in range(REPEATS):
        out = common.apply_angular_jitter(rays, sigma_mrad)
        delta = out - rays
        comp1.append(torch.sum(delta * e1, dim=1))
        comp2.append(torch.sum(delta * e2, dim=1))
    c1 = torch.cat(comp1) * 1e3      # rad -> mrad
    c2 = torch.cat(comp2) * 1e3

    for axis, comp in (('e1', c1), ('e2', c2)):
        mean = float(comp.mean())
        std = float(comp.std())
        assert abs(mean) < 0.1 * sigma_mrad, f'{axis}: mean {mean:.4f} mrad not ~0'
        assert abs(std - sigma_mrad) <= 0.1 * sigma_mrad, \
            f'{axis}: std {std:.4f} mrad outside 0.2 mrad +-10%'
    # directions stay unit length
    seed_all(1)
    out = common.apply_angular_jitter(rays, sigma_mrad)
    assert torch.allclose(out.norm(dim=1), torch.ones(rays.shape[0]), atol=1e-5)


def test_range_noise_statistics():
    _, common, _ = load_sampler_modules()
    sigma_cm = 2.5
    depths = torch.linspace(2.0, 79.0, NUM_RAYS).unsqueeze(1)
    seed_all(4242)
    deltas = [common.apply_range_noise(depths, sigma_cm) - depths
              for _ in range(REPEATS)]
    d = torch.cat(deltas).reshape(-1) * 100.0      # m -> cm
    assert abs(float(d.mean())) < 0.1 * sigma_cm
    assert abs(float(d.std()) - sigma_cm) <= 0.1 * sigma_cm, \
        f'std {float(d.std()):.4f} cm outside 2.5 cm +-10%'
    # range-independent: the far half has the same spread as the near half
    near = torch.cat([x[:NUM_RAYS // 2] for x in deltas]).reshape(-1) * 100.0
    far = torch.cat([x[NUM_RAYS // 2:] for x in deltas]).reshape(-1) * 100.0
    assert abs(float(near.std()) - float(far.std())) < 0.1 * sigma_cm


def test_ray_dropout_rate():
    _, common, _ = load_sampler_modules()
    p = 0.03
    seed_all(24680)
    kept = sum(int(common.ray_dropout_mask(NUM_RAYS, p, 'cpu').sum())
               for _ in range(REPEATS))
    dropped = 1.0 - kept / (REPEATS * NUM_RAYS)
    assert abs(dropped - p) <= 0.005, \
        f'dropout rate {dropped:.4f} outside 3% +-0.5 pp'


def test_ray_dropout_can_exempt_minority():
    _, common, _ = load_sampler_modules()
    labels = torch.tensor([5, 6] * (NUM_RAYS // 2))
    seed_all(1)
    keep = common.ray_dropout_mask(NUM_RAYS, 0.5, 'cpu', labels=labels,
                                   exempt_minority=True)
    assert bool(keep.all()), 'minority points were dropped despite the exemption'
