"""Shared helpers for the sampler tests: build the detector, run one frame."""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._stub_env import load_sampler_modules  # noqa: E402
from tests.synth_frames import FIXTURE_SENSOR, make_frames  # noqa: E402

BASELINE_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'baseline_outputs.npz')


def modules():
    return load_sampler_modules()


def build_model(sensor=None, **kwargs):
    seg, _, _ = load_sampler_modules()
    model = seg.PointTPV_Seg(sensor=dict(sensor or FIXTURE_SENSOR), **kwargs)
    model.train()
    return model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_frame(model, frame, seed):
    """Run ``_sampling`` on one frame under a fixed seed; return numpy outputs."""
    points, grid_ind, grid_ind_vox, labels = frame
    seed_all(seed)
    with torch.no_grad():
        out_p, out_g, out_gv, out_l, _ = model._sampling(
            [points.clone()], [grid_ind.clone()],
            [grid_ind_vox.clone()], [labels.clone()])
    return dict(
        points=out_p[0].detach().cpu().numpy(),
        grid_ind=out_g[0].detach().cpu().numpy(),
        grid_ind_vox=out_gv[0].detach().cpu().numpy(),
        labels=out_l[0].detach().cpu().numpy(),
    )


FRAME_RUN_SEEDS = (20260915, 20260916, 20260917)


def run_all_frames(model):
    return [run_frame(model, f, s)
            for f, s in zip(make_frames(), FRAME_RUN_SEEDS)]
