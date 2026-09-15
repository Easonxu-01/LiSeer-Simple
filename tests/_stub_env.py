"""Import ``pointtpv_seg`` on a machine without the full mmdet3d/mmcv stack.

The sampler itself only needs ``torch``, ``numpy`` and ``scatter_min``; the rest of
``pointtpv_seg.py``'s imports exist because the file also holds the detector.  This
module loads the file under a synthetic package name with those heavy dependencies
stubbed, so the sampling code can be exercised by the unit tests.

``torch_scatter`` is stubbed too when it is absent.  The stub reproduces the two
properties the sampler relies on: the per-bin minimum, and an argmin that is set to
``src.shape[0]`` (out of range) for empty bins.  Ties resolve to the first
occurrence, which is deterministic -- enough for a self-consistent regression
baseline.  On a real install the genuine ``torch_scatter`` is used unchanged.
"""
import importlib.util
import os
import sys
import types

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, 'projects', 'unilidar_plugin', 'pointocc_model')
PKG = '_liseer_sampler_pkg'


def _fallback_scatter_min(src, index, dim=0, out=None, dim_size=None):
    assert dim == 0, 'stub only supports dim=0'
    K = src.shape[0]
    N = int(dim_size)
    inf = torch.full((N,), float('inf'), device=src.device, dtype=src.dtype)
    if K == 0:
        return torch.zeros(N, device=src.device, dtype=src.dtype), \
               torch.full((N,), K, device=src.device, dtype=torch.long)
    vals = inf.scatter_reduce(0, index, src, reduce='amin', include_self=True)
    pos = torch.arange(K, device=src.device, dtype=torch.long)
    is_min = src == vals[index]
    cand = torch.where(is_min, pos, torch.full_like(pos, K))
    arg = torch.full((N,), K, device=src.device, dtype=torch.long)
    arg = arg.scatter_reduce(0, index, cand, reduce='amin', include_self=True)
    vals = torch.where(torch.isinf(vals), torch.zeros_like(vals), vals)
    return vals, arg


class _Registry:
    def register_module(self, *a, **kw):
        def deco(cls):
            return cls
        if a and callable(a[0]):
            return a[0]
        return deco


def _install_stubs():
    if 'torch_scatter' not in sys.modules:
        try:
            import torch_scatter  # noqa: F401
        except ImportError:
            mod = types.ModuleType('torch_scatter')
            mod.scatter_min = _fallback_scatter_min
            mod.STUBBED = True
            sys.modules['torch_scatter'] = mod

    if 'mmdet3d' not in sys.modules:
        try:
            import mmdet3d  # noqa: F401
        except ImportError:
            mmdet3d = types.ModuleType('mmdet3d')
            models = types.ModuleType('mmdet3d.models')
            builder = types.ModuleType('mmdet3d.models.builder')
            for name in ('build_middle_encoder', 'build_backbone', 'build_neck',
                         'build_voxel_encoder', 'build_head', 'build_loss'):
                setattr(builder, name, lambda cfg, *a, **kw: None)
            builder.VOXEL_ENCODERS = _Registry()
            models.builder = builder
            detectors = types.ModuleType('mmdet3d.models.detectors')

            class CenterPoint(torch.nn.Module):
                def __init__(self, *a, **kw):
                    super().__init__()

            detectors.CenterPoint = CenterPoint
            models.detectors = detectors
            mmdet3d.models = models
            sys.modules.update({'mmdet3d': mmdet3d, 'mmdet3d.models': models,
                                'mmdet3d.models.builder': builder,
                                'mmdet3d.models.detectors': detectors})

    if 'mmdet' not in sys.modules:
        try:
            import mmdet  # noqa: F401
        except ImportError:
            mmdet = types.ModuleType('mmdet')
            mdmodels = types.ModuleType('mmdet.models')
            mdmodels.DETECTORS = _Registry()
            mmdet.models = mdmodels
            sys.modules.update({'mmdet': mmdet, 'mmdet.models': mdmodels})

    if 'mmcv' not in sys.modules:
        try:
            import mmcv  # noqa: F401
        except ImportError:
            mmcv = types.ModuleType('mmcv')
            parallel = types.ModuleType('mmcv.parallel')

            class DataContainer:
                def __init__(self, data, *a, **kw):
                    self.data = data

            parallel.DataContainer = DataContainer
            mmcv.parallel = parallel
            sys.modules.update({'mmcv': mmcv, 'mmcv.parallel': parallel})


def _load_file(mod_name, path):
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_sampler_modules():
    """Return ``(pointtpv_seg, sampling_common, solidstate_sampler_or_None)``."""
    _install_stubs()

    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [MODEL_DIR]
        sys.modules[PKG] = pkg

    # Spectraldistiller is pure torch: load the real file.
    _load_file(PKG + '.Spectraldistiller',
               os.path.join(MODEL_DIR, 'Spectraldistiller.py'))
    # tpv_aggregator drags in mmengine and the projects.* tree; only one symbol is
    # imported by pointtpv_seg and it is never touched by the sampler.
    if PKG + '.tpv_aggregator' not in sys.modules:
        agg = types.ModuleType(PKG + '.tpv_aggregator')
        agg._cb_adc_consistency_loss = lambda *a, **kw: None
        sys.modules[PKG + '.tpv_aggregator'] = agg

    common = None
    common_path = os.path.join(MODEL_DIR, 'sampling_common.py')
    if os.path.exists(common_path):
        common = _load_file(PKG + '.sampling_common', common_path)
    solid = None
    solid_path = os.path.join(MODEL_DIR, 'solidstate_sampler.py')
    if os.path.exists(solid_path):
        solid = _load_file(PKG + '.solidstate_sampler', solid_path)

    seg = _load_file(PKG + '.pointtpv_seg',
                     os.path.join(MODEL_DIR, 'pointtpv_seg.py'))
    return seg, common, solid


def scatter_min_is_stubbed():
    import torch_scatter
    return bool(getattr(torch_scatter, 'STUBBED', False))
