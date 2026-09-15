"""Prototype Risley-prism sampling path."""
import math

import numpy as np
import pytest
import torch

from tests._stub_env import load_sampler_modules
from tests.sampler_harness import build_model, run_frame
from tests.synth_frames import FIXTURE_SENSOR, make_frame

pytest.importorskip('scipy', reason='the solid-state path needs scipy.spatial')

FOV_DEG = 70.4


@pytest.fixture(scope='module')
def mods():
    return load_sampler_modules()


@pytest.fixture(scope='module')
def frame():
    return make_frame(1001)


def _cfg(**kw):
    base = dict(t_int=0.1, seed=11)
    base.update(kw)
    return dict(pattern='risley', risley=base)


# ------------------------------------------------------------------ pattern


def test_rays_stay_inside_the_circular_fov(mods):
    _, _, solid = mods
    rays = solid.build_risley_rays(dict(t_int=0.2, seed=3))
    theta = np.degrees(np.arccos(np.clip(rays[:, 0], -1.0, 1.0)))
    assert rays.shape == (48000, 3)
    assert theta.max() <= 0.5 * FOV_DEG + 1e-6
    assert theta.min() < 1.0, 'the pattern should reach the boresight, not ring it'
    assert np.allclose(np.linalg.norm(rays, axis=1), 1.0, atol=1e-12)


def test_pattern_is_non_repetitive_across_frames(mods):
    _, _, solid = mods
    a = solid.build_risley_rays(dict(t_int=0.05))
    b = solid.build_risley_rays(dict(t_int=0.05))
    assert not np.allclose(a, b), 'phases were not redrawn; pattern repeats'
    c = solid.build_risley_rays(dict(t_int=0.05, seed=5))
    d = solid.build_risley_rays(dict(t_int=0.05, seed=5))
    assert np.allclose(c, d), 'a fixed seed must reproduce the pattern'


def test_fov_coverage_grows_with_integration_time(mods):
    """Coverage of the non-repetitive pattern densifies with integration time.

    Appendix B.7 reports 50.9% coverage at 0.1 s and 98.8% at 0.8 s on the
    prototype's own 240x240 angular grid.
    """
    _, _, solid = mods
    times = (0.1, 0.2, 0.5, 0.8)
    cov = [solid.fov_coverage(
        solid.build_risley_rays(dict(t_int=t, seed=101)), FOV_DEG) for t in times]
    assert cov == sorted(cov), f'coverage not monotone in T: {cov}'
    assert cov[0] > 0.0
    assert cov[-1] > 2.0 * cov[0]


# --------------------------------------------------------------- assignment


def test_assignment_picks_the_angularly_nearest_ray(mods):
    _, _, solid = mods
    rays = solid.build_risley_rays(dict(t_int=0.02, seed=1))
    rng = np.random.default_rng(0)
    probe = rays[rng.choice(rays.shape[0], 256, replace=False)]
    probe = probe + rng.normal(0.0, 1e-3, probe.shape)      # nudge off the rays
    nn, _ = solid.assign_points_to_rays(probe * 7.5, rays)  # scale: must normalise
    unit = probe / np.linalg.norm(probe, axis=1, keepdims=True)
    brute = np.argmax(unit @ rays.T, axis=1)                # largest cosine = nearest
    assert np.array_equal(nn, brute)


def test_visibility_knn_shape_and_self_exclusion(mods):
    _, _, solid = mods
    rays = solid.build_risley_rays(dict(t_int=0.01, seed=2))
    k = 16
    nbr = solid.visibility_knn(rays, k)
    assert nbr.shape == (rays.shape[0], k)
    assert not (nbr == np.arange(rays.shape[0])[:, None]).any(), 'self is a neighbour'


# ------------------------------------------------------------------ sampler


def test_interface_matches_the_spinning_path(mods, frame):
    _, _, solid = mods
    points, grid_ind, grid_ind_vox, labels = frame
    spin = run_frame(build_model(), frame, seed=5)
    out_p, out_g, out_gv, out_l, sensor_params = solid.sample_solidstate(
        [points.clone()], [grid_ind.clone()], [grid_ind_vox.clone()],
        [labels.clone()], cfg=_cfg())

    assert sensor_params is None, 'solid-state parameters must not reach S-Film'
    assert out_p[0].shape[1] == spin['points'].shape[1]
    assert out_p[0].dtype == torch.from_numpy(spin['points']).dtype
    assert out_g[0].dtype == torch.from_numpy(spin['grid_ind']).dtype
    assert out_gv[0].dtype == torch.from_numpy(spin['grid_ind_vox']).dtype
    assert out_l[0].dtype == torch.from_numpy(spin['labels']).dtype
    n = out_p[0].shape[0]
    assert n > 0
    assert out_g[0].shape == (n, 3) and out_gv[0].shape == (n, 3)
    assert out_l[0].shape == (n,)
    assert set(out_l[0].tolist()) <= set(spin['labels'].tolist())


def test_dispatch_through_the_model(mods, frame):
    seg, _, _ = mods
    points, grid_ind, grid_ind_vox, labels = frame
    model = seg.PointTPV_Seg(sensor=dict(FIXTURE_SENSOR, **_cfg()))
    model.train()
    with torch.no_grad():
        out_p, _, _, _, sensor_params = model._sampling(
            [points.clone()], [grid_ind.clone()],
            [grid_ind_vox.clone()], [labels.clone()])
    assert out_p[0].shape[0] > 0 and sensor_params is None


def test_unknown_pattern_raises(mods, frame):
    seg, _, _ = mods
    points, grid_ind, grid_ind_vox, labels = frame
    model = seg.PointTPV_Seg(sensor=dict(FIXTURE_SENSOR, pattern='mems'))
    with pytest.raises(ValueError):
        model._sampling([points], [grid_ind], [grid_ind_vox], [labels])


def test_nearest_hit_reduction_is_the_same_function(mods, frame):
    """Both paths must go through ``sampling_common.nearest_hit_reduce``."""
    _, common, solid = mods
    points, grid_ind, grid_ind_vox, labels = frame
    original = common.nearest_hit_reduce
    calls = {'n': 0}

    def counting(*args, **kwargs):
        calls['n'] += 1
        return original(*args, **kwargs)

    common.nearest_hit_reduce = counting
    try:
        run_frame(build_model(), frame, seed=5)
        spinning_calls = calls['n']
        assert spinning_calls > 0, 'the spinning path bypassed the shared reduction'
        solid.sample_solidstate([points.clone()], [grid_ind.clone()],
                                [grid_ind_vox.clone()], [labels.clone()],
                                cfg=_cfg())
        assert calls['n'] > spinning_calls, \
            'the solid-state path bypassed the shared reduction'
    finally:
        common.nearest_hit_reduce = original


def test_no_leaky_ghosts(mods, frame):
    """The occlusion stage removes exactly the rays the criterion flags.

    The criterion is the mechanical path's, unchanged: a ray leaks if rays above it
    in its neighbourhood see something much closer, with enough support and no
    depth-similar neighbours to protect it.  Minority classes are exempt by design.

    It is evaluated on the *pre-suppression* hit set -- the neighbourhood the sampler
    itself used.  Re-deriving a neighbourhood from the surviving rays would invent
    adjacency the sampler never saw and flag points spuriously; the mechanical path
    is likewise a single pass and makes no fixed-point claim.
    """
    _, _, solid = mods
    points, grid_ind, grid_ind_vox, labels = frame

    def run(mode):
        return solid.sample_solidstate(
            [points.clone()], [grid_ind.clone()], [grid_ind_vox.clone()],
            [labels.clone()], cfg=_cfg(occ_knn_k=48, occlusion_mode=mode),
            return_info=True)

    _, _, _, open_labels, _, open_info = run('none')
    _, _, _, _, _, occ_info = run('knn')

    diag = open_info[0]
    ray_ids = diag['chosen_rays']
    rays = diag['ray_directions'][ray_ids]
    depths = torch.from_numpy(diag['chosen_depths']).double()

    nbr = torch.from_numpy(solid.visibility_knn(rays, 48)).long()
    elev = torch.from_numpy(rays[:, 2].copy()).double()
    nbr_depths = depths[nbr]
    upper = elev[nbr] > elev.unsqueeze(1)
    upper_depths = torch.where(upper, nbr_depths,
                               torch.full_like(nbr_depths, solid.LARGE_VAL))
    occ_min = upper_depths.amin(dim=1)
    support = (upper_depths <= (occ_min.unsqueeze(1) + solid.OCC_NEAR_BAND)).sum(dim=1)
    same = (torch.abs(nbr_depths - depths.unsqueeze(1))
            <= solid.RING_KEEP_BAND).sum(dim=1)
    margin = solid.OCC_DEPTH_MARGIN + solid.OCC_REL_MARGIN * occ_min
    leak = ((occ_min < solid.LARGE_VAL) &
            (support >= solid.OCC_SUPPORT_MIN) &
            ((depths - occ_min) > margin) &
            (same < solid.RING_KEEP_MIN))

    minority = torch.zeros_like(open_labels[0], dtype=torch.bool)
    for cls in solid.sampling_common.MINORITY_CLASSES:
        minority = minority | (open_labels[0] == cls)

    flagged = set(ray_ids[(leak & ~minority).numpy()].tolist())
    kept = set(occ_info[0]['chosen_rays'].tolist())
    removed = set(ray_ids.tolist()) - kept

    assert flagged, 'the criterion flagged nothing; the test would be vacuous'
    assert removed == flagged, (
        f'{len(removed - flagged)} ray(s) removed without being flagged, '
        f'{len(flagged - removed)} flagged ghost(s) left in the output')


def test_occlusion_can_be_disabled(mods, frame):
    """Sanity check that the occlusion stage is what removes those points."""
    _, _, solid = mods
    points, grid_ind, grid_ind_vox, labels = frame
    args = ([points.clone()], [grid_ind.clone()], [grid_ind_vox.clone()],
            [labels.clone()])
    with_occ = solid.sample_solidstate(*args, cfg=_cfg())[0][0].shape[0]
    without = solid.sample_solidstate(*args, cfg=_cfg(occlusion_mode='none'))[0][0].shape[0]
    assert without > with_occ, 'occlusion suppression removed nothing'


def test_perturbation_suite_applies_on_this_path_too(mods, frame):
    _, _, solid = mods
    points, grid_ind, grid_ind_vox, labels = frame
    args = ([points.clone()], [grid_ind.clone()], [grid_ind_vox.clone()],
            [labels.clone()])
    plain = solid.sample_solidstate(*args, cfg=_cfg())[0][0]
    torch.manual_seed(0)
    perturbed = solid.sample_solidstate(
        *args, cfg=_cfg(),
        perturb_cfg=dict(enabled=True, angular_jitter_mrad=0.2,
                         range_noise_cm=2.5, ray_dropout=0.03),
        training=True)[0][0]
    assert perturbed.shape[0] < plain.shape[0]
    # and nothing happens outside training
    unperturbed = solid.sample_solidstate(
        *args, cfg=_cfg(),
        perturb_cfg=dict(enabled=True, ray_dropout=0.03), training=False)[0][0]
    assert unperturbed.shape[0] == plain.shape[0]
