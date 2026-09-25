# LiSeer — revision analysis code and source data

Analysis scripts and the raw numbers behind the additional figures and tables
prepared during revision. File names follow the analysis IDs used in the
response to reviewers (A1–A6, CODE-1/2/5). Every figure and table can be
regenerated from `_source_data/` without re-running GPU inference.

All scripts are run from the root of the main LiSeer repository
(the directory that contains `projects/`), e.g.

```bash
python review_analysis/A4_figR6.py --outdir <out>
```

## Analysis scripts

| File | Analysis | Purpose |
|---|---|---|
| `A1_label_propagation_fidelity.py` | A1 | Semantic fidelity of label propagation (hold-out accuracy, per-class IoU, boundary-voxel ratio) |
| `A1b_boundary_containment.py` | A1 | Boundary containment between sparse and densified labels (E1–E6) |
| `A1c_coarse_and_confusion.py` | A1 | Coarse-8 propagation accuracy and confusion matrix (E7–E11) |
| `A1d_boundary_agreement_v2.py` | A1 | Boundary agreement with distance-banded accuracy (M1–M5) |
| `A1e_m1_tail.py` | A1 | Tail statistics of boundary displacement (M1) |
| `A2_densification_geometry.py` | A2 | Geometric fidelity of densification (nearest-neighbour distances, diffusion-response profile) |
| `A3_nominal_final.py`, `A3_probe_*.py`, `A3_fit_waymo_origin.py` | A3 | Measured sensor configurations of SemanticKITTI / nuScenes / Waymo |
| `A3_sets.py`, `A3_simulator.py`, `A3_run_f1f4.py` | A3 | Real/simulated scan sets and the F1–F4 descriptors |
| `A3_pairwise_js.py`, `A3_perframe.py`, `A3_coverage.py` | A3 | Frame-pair Jensen–Shannon distances, per-frame scalars, curriculum coverage |
| `A3_diagnose*.py` | A3 | Per-bin decomposition of the residual F1/F2 gaps and their mechanism |
| `A3_finalize.py`, `A3_plot_R3.py` | A3 | Summary tables and Fig. R3 |
| `A4_consistency.py` | A4 | Output-level sensor invariance over 50 sensor configurations (Table R9) |
| `A4_visualize.py`, `A4_figR5.py`, `A4_figR6.py` | A4 | Fig. R5 statistics and Fig. R6 point-cloud panels |
| `A4_control_observation_stats.py` | A4 | Observation-count and coordinate-spread statistics of the same-sensor control group |
| `A5_cost.py`, `A5_io_phases.py`, `A5_finalize.py` | A5 | Measured compute cost of each component (sampler, S-Film, CBA-CL, inference FPS) |
| `A6_figR4.py`, `A6_figR4_multi.py` | A6 | Fig. R4 renderings and coverage curves |
| `CODE5_risley_demo.py` | CODE-5 | Risley-prism sampler scaffold reused by several scripts |
| `WEB_export_pointclouds.py` | — | Point-cloud export for the project-page 3D viewer |

## Validation scripts

Self-checks for the code added to the main repository during revision.

| File | Checks |
|---|---|
| `CODE1_validate.py` | Physical perturbation suite (13 checks; CHECK-6 is a timing check and can fluctuate) |
| `CODE2_validate.py` | Densification degradation (4 checks) |
| `CODE5_validate.py` | Risley-prism sampler (3 checks) |
| `A4_validate_provenance.py` | Point-provenance tracking used by A4 (3 checks) |

```bash
python review_analysis/CODE2_validate.py
CUDA_VISIBLE_DEVICES=0 python review_analysis/CODE5_validate.py
CUDA_VISIBLE_DEVICES=0 python review_analysis/A4_validate_provenance.py
CUDA_VISIBLE_DEVICES=0 python review_analysis/CODE1_validate.py --cfg <a LiSeer config>
```

## `_source_data/` — raw numbers behind every figure and table

| File | Content |
|---|---|
| `A1_label_propagation.json`, `A1b_*.json`, `A1c_*.json`, `A1d_*.json`, `A1e_*.json`, `A1e_m1_distances.npz` | A1 results |
| `A2_densification_geometry.json` | A2 results |
| `A3_nominal_configs.json`, `A3_sensor_probe.json`, `A3_beam_tables.json`, `A3_waymo_beam_table.json` | Measured sensor configurations |
| `A3_pairwise_js.npz`, `A3_pairmats_*.npz`, `A3_perframe_scalars.npz`, `A3_perframe/` | Frame-pair JS distances and per-frame scalars (each `.npz` carries a `_README` key) |
| `A3_histograms.npz`, `A3_simreal_metrics.csv`, `A3_frame_level.csv`, `A3_coverage.csv`, `A3_coverage_scatter.npz` | A3 summary tables and Fig. R3 inputs |
| `A4_raw.json`, `A4_table_R9.csv` | A4 main experiment (16–128 beams envelope), Table R9 |
| `A4_beams32_64_raw.json`, `A4_beams32_64_table_R9.csv` | A4 training-envelope variant (32–64 beams) |
| `A4_control_observation_stats.json` | Same-sensor control group statistics |
| `a4_vis_bc2p6_LiSeer_SK.npz`, `a4_vis_uniformbeams_LiSeer_SK.npz` | Render caches for Fig. R5/R6 (beam concentration 2.6 and uniform beams) |
| `A5_cost.csv`, `A5_raw/` | A5 cost measurements |

Reproduce the A4 figures from the caches (no GPU needed):

```bash
python review_analysis/A4_figR6.py --outdir <out>
python review_analysis/A4_figR5.py --raw review_analysis/_source_data/A4_raw.json --outdir <out>
```

## Experiment configurations

The degradation-robustness runs (Table R3) use `configs/review_config/E10_degraded_sk_sigma003.py`,
`E11_…006.py` and `E12_…010.py` in the main repository (σ = 3 / 6 / 10 cm).
