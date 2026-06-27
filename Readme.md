# LiSeer

**Learning Generalizability from Randomness: Zero-Shot LiDAR Semantic Segmentation via Scene–Sensor Disentanglement**

LiSeer is a LiDAR semantic-segmentation framework that learns sensor-agnostic representations so a single model can generalize **zero-shot** across LiDAR sensors and datasets (nuScenes, SemanticKITTI, Waymo). It disentangles *scene* content from *sensor* characteristics through random LiDAR-pattern sampling, a coarse cross-dataset label taxonomy, consistency learning, and sensor-conditioned feature modulation.

## Key components

| Paper name | Config switch | Description |
| --- | --- | --- |
| Random Sampling | `Random_Samplling` | Randomly re-samples dense point clouds into diverse sparse LiDAR patterns during training, exposing the model to a wide range of sensor geometries. |
| COLA | `COLA` | Coarse cross-dataset label taxonomy (8 shared classes) enabling cross-dataset / zero-shot transfer. |
| CBA-CL | `CBA_CL` | Class-Balanced Adaptive Consistency Learning across two augmented views of the same scene. |
| S-FiLM | `S_FilM` | Sensor-conditioned FiLM modulation that injects sensor parameters into the network. |

## Repository layout

```
LiSeer-Simple
├── docs/                     # documentation (this README links here)
├── projects/
│   ├── configs/pointocc/     # training / eval configs
│   └── unilidar_plugin/      # model, datasets, pipelines, hooks
├── tools/                    # data generation, training & test entry points
├── run.sh / run_eval.sh      # train / eval wrappers
└── setup.py
```

## Getting started

1. **Install** the environment — see [`docs/install.md`](docs/install.md).
2. **Prepare data** (raw datasets + dense point clouds + dense labels) — see [`docs/prepare_data.md`](docs/prepare_data.md).
3. **Train & evaluate** — see [`docs/trainval.md`](docs/trainval.md).

Quick start (8 GPUs, nuScenes via the unified config):

```shell
export PYTHONPATH="."
bash run.sh ./projects/configs/pointocc/pointtpv_seg.py 8
```

The four components above are toggled directly at the top of `projects/configs/pointocc/pointtpv_seg.py` (all enabled by default).

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
