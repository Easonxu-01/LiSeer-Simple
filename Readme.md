# LiSeer

**Learning Generalizability from Randomness: Zero-Shot LiDAR Semantic Segmentation via Scene–Sensor Disentanglement**

LiSeer is a **sensor-agnostic** LiDAR semantic-segmentation framework. Instead of retraining for every new or upgraded LiDAR, it learns representations that transfer **zero-shot** across sensors and datasets by disentangling *scene-intrinsic geometry* from *sensor-specific sampling*.

LiDAR perception is typically locked into a single-dataset, single-sensor training paradigm. In practice, models trained on one configuration often fail to generalize because of **beam-density differences**, **inconsistent label taxonomies across datasets**, and **viewpoint / mounting variability**. LiSeer addresses these domain shifts with a controllable data engine and two lightweight modules, and is validated on **nuScenes**, **SemanticKITTI**, **Waymo**, and **real autonomous-vehicle LiDAR sequences** (32 / 80 / 128-beam).

## Key components

| Paper name | Config switch | Description |
| --- | --- | --- |
| Random Sampling | `Random_Samplling` | Densifies sparse scans offline (via [LiDiff](https://github.com/PRBonn/LiDiff)), then re-samples them online under randomized, physically grounded LiDAR configurations — beams, resolution, field of view, and mounting height — exposing the model to a broad family of cross-sensor observations. |
| COLA | `COLA` | Coarse cross-dataset label alignment taxonomy (8 shared classes) that bridges inconsistent semantic definitions across datasets and enables cross-dataset transfer. |
| CBA-CL | `CBA_CL` | Class-Balanced Adaptive Consistency Learning across two augmented views of the same scene; enhances boundaries and minority classes and steers optimization toward scene-consistent semantics rather than sensor-specific artifacts. |
| S-FiLM | `S_FilM` | Sensor-aware feature modulation (FiLM) that gives sensor variation an explicit modeling pathway, so the shared decoder can focus on scene-intrinsic structure. |

All four switches live at the top of `projects/configs/pointocc/pointtpv_seg.py` and default to `True`.

## Optional sampler extensions

Two optional, default-off features of the physics-based controllable sampler. Neither
changes the default training or inference path: with no new option enabled the
sampler's output is bit-identical to the release without them
(`tests/test_perturb_suite.py` asserts this against a committed baseline).

**Optional physical-perturbation suite.** In addition to the randomization the
sampler already performs (a LiDAR configuration redrawn every iteration -- beams,
angular resolution, field of view, mounting height -- plus minority-class mixup), an
explicit perturbation suite can be enabled for training: per-ray angular jitter
(sigma = 0.2 mrad), Gaussian range noise (sigma = 2.5 cm) and random ray dropout
(p = 3%). The three terms are layered on top of the existing randomization, not
substituted for it. It applies during training only -- evaluation and inference are
never perturbed -- and it is disabled by default; all reported results are produced
without it. While disabled the suite draws no random numbers, so it cannot shift the
sampling stream. See `projects/configs/pointocc/sampler_risley_example.py` for the
switch, or pass `perturb_suite=dict(enabled=True, ...)` to the model.

**Solid-state scanning (prototype).** The point-centric sampler requires only a set
of ray directions. A prototype path for non-repetitive Risley-prism scanning
(Livox-type rosette patterns) is provided in
`projects/unilidar_plugin/pointocc_model/solidstate_sampler.py`: ray construction,
point-to-ray assignment and the visibility neighbourhood are replaced, while the
nearest-hit reduction, minority-class protection, noise models and output stage are
reused unchanged -- both paths call the same
`sampling_common.nearest_hit_reduce`. The scanline regularisation is disabled, as a
non-repetitive pattern has no scan lines. Select the path with `pattern: risley` in
the sensor config (see `projects/configs/sampler_risley_example.yaml`); a config
without a `pattern` key is `spinning` and unaffected. **LiSeer has not been trained
or evaluated on solid-state data; the prototype demonstrates architectural
compatibility.**

Prism parameter provenance:

| Parameter | Value | Source |
| --- | --- | --- |
| `fov_deg` | 70.4 (circular, full angle) | Livox Avia specification |
| `point_rate` | 240 000 points/s | Livox Avia first-return specification |
| `beam_divergence_deg` | 0.28 | Livox Avia specification |
| `t_int` | 0.1 / 0.2 / 0.5 s | design choice (figure panels) |
| `f1_hz` | 7294.0 | design choice |
| `freq_ratio` (`f2`/`f1`) | -1/golden = -0.618034, i.e. f2 = -4507.94 Hz | design choice: counter-rotating at the ratio hardest to approximate rationally, so the pattern closes as late as possible |
| `rho_deg` | `None` -> `fov_deg`/4 = 17.6 | design choice: rho1 == rho2 so the off-axis angle reaches 0 and the pattern has no centre hole |
| prism initial phases | drawn per frame | design choice: this is what makes the scan non-repetitive across frames |
| `height_m` | 1.73 | design choice, matching the mechanical path's mounting range |

Two demo scripts reproduce the appendix figures' behaviour without any dataset
(they fall back to a synthetic dense frame):

```shell
python scripts/demo_solidstate.py      --out out/solidstate   # spinning + T = 0.1/0.2/0.5 s
python scripts/plot_fov_coverage.py    --out out/solidstate   # coverage vs integration time
```

`scripts/plot_fov_coverage.py` measures field-of-view coverage on the prototype's
own 240x240 angular grid -- the denominator is the circular field of view itself,
not its square bounding box. With the prism parameters fixed before any coverage was
computed, the prototype reaches **50.9% coverage at 0.1 s and 98.8% at 0.8 s**,
consistent with the manufacturer's published behaviour.

The [project page](https://easonxu-01.github.io/LiSeer-Simple/) carries the same
comparison as interactive point clouds: one scene per dataset, re-observed four ways
(real scan, dense reconstruction, spinning LiDAR, prism prototype at T = 0.8 s), with
the four cameras in a row locked together. Points are coloured by the project's
COLA-8 colour map, with a legend, and the two simulated views carry the sensor
configuration they were sampled with. `scripts/pack_pointcloud.py` converts the
exported clouds into the binary the viewer reads:

```shell
python scripts/pack_pointcloud.py --src <exported clouds>   # -> docs/assets/pointcloud/
python scripts/pack_pointcloud.py --demo                    # from the synthetic fixture
```

Tests for both features:

```shell
python -m pytest tests -q
```

## Datasets & evaluation

LiSeer is trained and evaluated on three public benchmarks plus custom on-vehicle data:

| Dataset | Role |
| --- | --- |
| [nuScenes](https://www.nuscenes.org/) | Primary benchmark; strong zero-shot cross-LiDAR generalization |
| [SemanticKITTI](http://www.semantic-kitti.org/) | Outdoor driving; cross-dataset transfer |
| [Waymo Open Dataset](https://waymo.com/open/) | Large-scale autonomous driving |
| Real-vehicle sequences | 32 / 80 / 128-beam LiDAR collected in Beijing Yizhuang (~100 sequences) |

Dense point clouds and dense labels required for Random Sampling are generated with [LiDiff](https://github.com/PRBonn/LiDiff) and the scripts in `tools/` — see [`docs/prepare_data.md`](docs/prepare_data.md) for the full pipeline and expected folder layout.

## Available resources

| Resource | Description | Link |
| --- | --- | --- |
| Pretrained checkpoints | LiSeer-V / LiSeer-P (4 checkpoints per dataset, ~0.25 GiB each) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/6d0d7ba5e36449fa96cf/) |
| Real-vehicle clips | 32 / 80 / 128-beam on-vehicle sequences | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/26f0485231e5455eae19/) |
| Densified data | Pre-generated dense point clouds & labels (optional; can also be reproduced locally) | See [`docs/prepare_data.md`](docs/prepare_data.md) |

This release ships **training / evaluation configs**, the **pointseg model code**, and instructions to reproduce densification and training end-to-end.

## Repository layout

```
LiSeer-Simple
├── docs/                     # documentation (install, data prep, train/eval)
├── projects/
│   ├── configs/pointocc/     # unified + per-dataset reference configs
│   └── unilidar_plugin/      # model, datasets, pipelines, hooks
├── tools/                    # LiDiff completion, dense-label generation, train/test
├── run.sh / run_eval.sh      # train / eval wrappers
└── setup.py
```

## Getting started

1. **Install** the environment — [`docs/install.md`](docs/install.md)
2. **Prepare data** (raw datasets + dense point clouds + dense labels) — [`docs/prepare_data.md`](docs/prepare_data.md)
3. **Train & evaluate** — [`docs/trainval.md`](docs/trainval.md)

Quick start (8 GPUs, nuScenes via the unified config):

```shell
export PYTHONPATH="."
bash run.sh ./projects/configs/pointocc/pointtpv_seg.py 8
```

Per-dataset reference configs with full hyper-parameters are kept alongside the unified entry point:

- `projects/configs/pointocc/pointtpv_nusc_seg_random.py`
- `projects/configs/pointocc/pointtpv_semantickitti_seg_random.py`
- `projects/configs/pointocc/pointtpv_waymo_seg_random.py`

For inference on custom `.bin` scans, use `projects/configs/pointocc/pointtpv_realdata_infer.py`.

## Citation

If you find this work useful, please cite:

```bibtex
@article{xu2026liseer,
  title   = {Learning Generalizability from Randomness: Zero-Shot
             LiDAR Semantic Segmentation via Scene--Sensor Disentanglement},
  author  = {Xu, Zikun and others},
  journal = {Preprint},
  year    = {2026}
}
```
