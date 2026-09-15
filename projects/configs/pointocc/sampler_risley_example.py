"""Example: the two optional sampler features, in this repository's config format.

Not a runnable training config -- it carries no dataset, schedule or backbone. Copy
the two blocks below into a real config in this directory.

    # 1. optional physical-perturbation suite (training-time only, default off)
    model = dict(
        type='PointTPV_Seg',
        ...,
        perturb_suite=perturb_suite,
        sensor=sensor,
    )

    # 2. prototype solid-state path: select it through the sensor config
    sensor = dict(..., pattern='risley', risley=risley)

LiSeer has not been trained or evaluated on solid-state data; the prism path is a
prototype that demonstrates architectural compatibility.
"""

# ---------------------------------------------------------------- perturbation
# Layered on top of the sampler's own stochastic terms, never replacing them.
# `enabled=False` is bit-identical to omitting the block entirely: while off the
# suite draws no random numbers at all.
perturb_suite = dict(
    enabled=False,
    angular_jitter_mrad=0.2,      # per-ray pointing jitter, sigma per lateral axis
    range_noise_cm=2.5,           # extra Gaussian range noise, sigma
    ray_dropout=0.03,             # per-ray dropout probability, class-agnostic
    dropout_exempt_minority=False,
)

# ------------------------------------------------------------------ solid state
# Provenance: fov_deg, point_rate and beam_divergence_deg come from the Livox Avia
# specification; every other value is a design choice of this prototype.
risley = dict(
    fov_deg=70.4,
    point_rate=240000,
    beam_divergence_deg=0.28,
    t_int=0.1,                    # integration time T in seconds
    f1_hz=7294.0,
    freq_ratio=-0.6180339887498949,   # f2/f1 = -1/golden  =>  f2 = -4507.94 Hz
    rho_deg=None,                 # None -> fov_deg/4, so theta spans [0, 35.2 deg]
    height_m=1.73,
    footprint_scale=1.0,
    occlusion_mode='knn',
    occ_knn_k=48,
    minority_depth_tol=0.4,
    seed=None,                    # None -> phases redrawn per frame (non-repetitive)
    recompute_voxel_index=False,
)

# The spinning ranges are unchanged; `pattern` is the only added key, and a config
# without it behaves exactly as before.
sensor = dict(
    LiDAR_height=[1, 2],
    num_of_beams=[16, 80],
    horizontal_angular_resolution=[900, 3600],
    lower_vertical_field_of_view_bound=[-40, -5],
    upper_vertical_field_of_view_bound=[0, 25],
    pattern='risley',
    risley=risley,
)
