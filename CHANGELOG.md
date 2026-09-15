# Changelog

## 2026-09

Two optional, default-off extensions to the physics-based controllable sampler. The
default training and inference path is unchanged: with no new option enabled the
sampler's output is bit-identical to the previous release, verified against a
committed baseline in `tests/baseline_outputs.npz`.

### Added

- **Optional physical-perturbation suite** (`perturb_suite`, disabled by default):
  per-ray angular jitter (sigma = 0.2 mrad), Gaussian range noise (sigma = 2.5 cm)
  and per-ray dropout (p = 3%), layered on top of the sampler's existing
  randomization. Training-time only. While disabled the suite draws no random
  numbers, so it cannot shift the sampling stream. The CODE-1 spellings
  (`sampler_perturb`, `enable`, `dropout_p`) are accepted as aliases.
- **Prototype Risley-prism (solid-state) sampler**
  (`projects/unilidar_plugin/pointocc_model/solidstate_sampler.py`), selected with
  `pattern: risley` in the sensor config. Ray construction, point-to-ray assignment
  and the visibility neighbourhood are replaced; the nearest-hit reduction,
  minority-class protection, noise models and output stage are reused unchanged. The
  scanline regularisation is disabled. LiSeer has not been trained or evaluated on
  solid-state data.
- `projects/unilidar_plugin/pointocc_model/sampling_common.py`: the nearest-hit
  (z-buffer) reduction both sampling paths now share, plus the three perturbation
  operators.
- Example configs: `projects/configs/sampler_risley_example.yaml` and
  `projects/configs/pointocc/sampler_risley_example.py` (examples only; not part of
  any training config).
- Demo scripts: `scripts/demo_solidstate.py` (spinning plus the prototype at
  T = 0.1 / 0.2 / 0.5 s) and `scripts/plot_fov_coverage.py` (field-of-view coverage
  against integration time). Both run without a dataset.
- Tests: `tests/test_perturb_suite.py`, `tests/test_solidstate_sampler.py`, and the
  bit-exactness baseline they check against.
- Project page: a `#solid-state` section explaining which three operations the
  prototype replaces, with the field-of-view coverage curve drawn natively in SVG;
  and a `#pointcloud` section with twelve interactive views -- three scenes, four
  observations each -- where the four cameras of a row are locked together. Point
  clouds are packed by `scripts/pack_pointcloud.py` into an interleaved int16
  format, eight bytes per point, and each row renders on demand so a still page
  costs nothing.

### Changed

- The nearest-hit reduction in `_run_single_sampling` now calls
  `sampling_common.nearest_hit_reduce` instead of inlining `scatter_min`. Same
  operations in the same order; the output is unchanged.
