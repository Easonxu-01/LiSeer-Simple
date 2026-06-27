# Training & Evaluation

All commands assume the repo root as the working directory and `export PYTHONPATH="."` (the `run.sh` / `run_eval.sh` wrappers set this for you). Edit the `conda activate <env>` line inside those wrappers to match the env name from [`install.md`](install.md).

## Configs

The recommended entry point is the **unified config** `projects/configs/pointocc/pointtpv_seg.py`. It exposes four switches that map to the paper components, plus a dataset selector:

```python
DATASET          = 'nuscenes'   # 'nuscenes' | 'semantickitti' | 'waymo'
Random_Samplling = True         # random LiDAR-pattern sampling
COLA             = True         # coarse cross-dataset label taxonomy (8 classes)
CBA_CL           = True         # class-balanced adaptive consistency learning
S_FilM           = True         # sensor-conditioned FiLM modulation
```

The per-dataset detailed reference configs (full hyper-parameters) are also kept:

- `projects/configs/pointocc/pointtpv_nusc_seg_random.py`
- `projects/configs/pointocc/pointtpv_semantickitti_seg_random.py`
- `projects/configs/pointocc/pointtpv_waymo_seg_random.py`

## Training

Train with N GPUs:

```shell
bash run.sh ./projects/configs/pointocc/pointtpv_seg.py 8
```

Or use a reference config directly, e.g. SemanticKITTI:

```shell
bash run.sh ./projects/configs/pointocc/pointtpv_semantickitti_seg_random.py 8
```

`run.sh` forwards extra args to `tools/dist_train.sh`, so you can append e.g. `--cfg-options ...`.

## Evaluation

```shell
bash run_eval.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM
```

## Inference on real / custom LiDAR

To run inference on raw `.bin` scans (no labels) listed by a `list.txt`, use the real-data config and point `load_from` at your checkpoint:

```shell
bash run_eval.sh ./projects/configs/pointocc/pointtpv_realdata_infer.py $PATH_TO_CKPT $GPU_NUM
```

Set `REAL_DATA_NAME` / `REAL_DATA_ROOTS` and the prediction output dir inside `pointtpv_realdata_infer.py`. Predictions (and optional `.ply`) are written to `test_cfg.inference_pred_dir`.

## Visualization

Generate predictions to a directory and render them:

```shell
bash run_eval.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM --show --show-dir $PATH
```
