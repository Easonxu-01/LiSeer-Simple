"""Primitives shared by the spinning and the solid-state sampling paths.

Two groups live here:

* ``nearest_hit_reduce`` -- the plain z-buffer reduction (closest point per ray).
  Both sampling paths call it through this module, so there is exactly one
  implementation of the nearest-hit rule.
* the optional physical-perturbation suite (``resolve_perturb_cfg`` plus the three
  operators).  The operators never gate themselves: the caller keeps the whole
  ``if`` around the call so that a disabled suite touches the random number
  generator zero times and the sampling stream stays bit-identical.
"""
import torch
from torch_scatter import scatter_min

# Mapped COLA taxonomy: 5 = Living Being, 6 = Movable objects.
MINORITY_CLASSES = (5, 6)

EPS = 1e-8

# ---------------------------------------------------------------------------
# nearest-hit (z-buffer) reduction -- shared by both sampling paths
# ---------------------------------------------------------------------------


def nearest_hit_reduce(valid_depths, valid_bins, num_rays):
    """Keep the closest candidate per ray.

    Returns ``(min_depth, min_idx, hit_mask)`` where ``min_idx`` indexes into
    ``valid_depths`` and ``hit_mask`` is False for rays with no usable candidate.
    """
    min_depth_flat, min_idx_in_valid = scatter_min(
        valid_depths,
        valid_bins,
        dim=0,
        dim_size=num_rays
    )
    hit_mask_flat = torch.isfinite(min_depth_flat)
    K = valid_depths.shape[0]
    idx_ok = (min_idx_in_valid >= 0) & (min_idx_in_valid < K)
    hit_mask_flat = hit_mask_flat & idx_ok
    return min_depth_flat, min_idx_in_valid, hit_mask_flat


# ---------------------------------------------------------------------------
# optional physical-perturbation suite
# ---------------------------------------------------------------------------

PERTURB_DEFAULTS = dict(
    enabled=False,               # master switch; training-time only
    angular_jitter_mrad=0.0,     # per-ray pointing jitter, sigma per lateral axis
    range_noise_cm=0.0,          # extra Gaussian range noise, sigma
    ray_dropout=0.0,             # per-ray dropout probability
    dropout_exempt_minority=False,
)

# CODE-1 (S1) shipped the switch under different names; both spellings are accepted
# so an existing server-side config keeps working.
_PERTURB_ALIASES = dict(
    enable='enabled',
    dropout_p='ray_dropout',
)


def resolve_perturb_cfg(user_cfg):
    """Normalise a user config into :data:`PERTURB_DEFAULTS`' keys.

    ``None`` or an empty mapping means "everything off".  An unrecognised key is a
    hard error rather than a silently ignored typo.
    """
    cfg = dict(PERTURB_DEFAULTS)
    if not user_cfg:
        return cfg
    unknown = [k for k in user_cfg
               if k not in cfg and k not in _PERTURB_ALIASES]
    assert not unknown, (
        f'unknown perturbation-suite field(s): {sorted(unknown)}; '
        f'expected a subset of {sorted(cfg)}')
    for key, value in user_cfg.items():
        cfg[_PERTURB_ALIASES.get(key, key)] = value
    cfg['enabled'] = bool(cfg['enabled'])
    cfg['dropout_exempt_minority'] = bool(cfg['dropout_exempt_minority'])
    for key in ('angular_jitter_mrad', 'range_noise_cm', 'ray_dropout'):
        cfg[key] = float(cfg[key])
    return cfg


def ray_dropout_mask(num, p, device, labels=None, exempt_minority=False):
    """Per-ray keep mask; each ray is dropped independently with probability ``p``.

    Class-agnostic by default: a real detector does not know the class of what it
    failed to detect.
    """
    keep = torch.rand(num, device=device) >= p
    if exempt_minority and labels is not None:
        is_minor = torch.zeros_like(labels, dtype=torch.bool)
        for _cls in MINORITY_CLASSES:
            is_minor = is_minor | (labels == _cls)
        keep = keep | is_minor
    return keep


def apply_range_noise(depths, sigma_cm):
    """Add Gaussian range noise (3-sigma clipped) to the winning z-buffer depths.

    Injected before the coordinates and voxel indices are recomputed, so the noise
    flows into every derived feature and the output stays self-consistent.
    """
    sigma_r = sigma_cm * 0.01
    noise = torch.randn_like(depths).clamp_(-3.0, 3.0) * sigma_r
    return (depths + noise).clamp_min(0.1)


def apply_angular_jitter(ray_dirs, sigma_mrad):
    """Jitter ray directions in their own tangent plane.

    Applied on the output side, after point selection, so it only moves the returned
    points.  Perturbing the global ray grid instead would make the footprint gate
    (``cos_sim >= cos_footprint``) reject extra boundary points, which would show up
    as dropout rather than as pointing error, and would break the property that the
    bins tile the angular space exactly at ``ANGULAR_MARGIN_SCALE = 1.0``.  The
    physical counterpart is pointing/calibration error: beam divergence is already
    covered by the snap-to-ray mechanism.

    ``sigma_mrad`` is the standard deviation per lateral axis, so the total angular
    offset is Rayleigh distributed with mean ``1.253 * sigma``.  Depth is unchanged,
    so the lateral displacement ``r * sigma`` grows linearly with range (1.6 cm at
    80 m).
    """
    sigma_t = sigma_mrad * 1e-3          # mrad -> rad
    d = ray_dirs
    # Tangent-plane basis: pick the auxiliary axis that cannot be parallel to d.
    aux = torch.where(
        (d[:, 2].abs() < 0.9).unsqueeze(1),
        d.new_tensor([0., 0., 1.]).expand_as(d),
        d.new_tensor([1., 0., 0.]).expand_as(d))
    e1 = torch.cross(d, aux, dim=1)
    e1 = e1 / e1.norm(dim=1, keepdim=True).clamp_min(EPS)
    e2 = torch.cross(d, e1, dim=1)       # d, e1 orthonormal -> e2 is already unit
    g = torch.randn(d.shape[0], 2, device=d.device, dtype=d.dtype).clamp_(-3.0, 3.0)
    d = d + sigma_t * (g[:, 0:1] * e1 + g[:, 1:2] * e2)
    return d / d.norm(dim=1, keepdim=True).clamp_min(EPS)
