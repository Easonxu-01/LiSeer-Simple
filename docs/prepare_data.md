# Dataset preparation

LiSeer is trained and evaluated on three LiDAR semantic-segmentation datasets: **nuScenes**, **SemanticKITTI** and **Waymo Open Dataset**. For every dataset you need three ingredients:

1. The **raw dataset** (point clouds + per-point semantic labels) and the dataset info `.pkl` files.
2. **Dense point clouds**, generated with the diffusion-based scene completion model [LiDiff](https://github.com/PRBonn/LiDiff).
3. **Dense labels**, propagated from the sparse ground-truth labels onto the dense point clouds with the scripts in `tools/`.

The dense point clouds / dense labels are what the random-sampling (`Random_Samplling`) branch consumes during training (`dense7sparse=True`). The first two steps are dataset specific; the dense-label step is fully scripted.

---

## Step 1 — Raw datasets

### nuScenes

Download the nuScenes V1.0 full dataset and lidarseg annotations from the [official site](https://www.nuscenes.org/download), then download the train/val info pickles ([train](https://github.com/JeffWang987/OpenOccupancy/releases/tag/train_pkl) / [val](https://github.com/JeffWang987/OpenOccupancy/releases/tag/val_pkl)) and place them under `data/nuscenes/`.

```
LiSeer-Simple
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   │   └── LIDAR_TOP/                 # raw lidar .bin
│   │   ├── sweeps/
│   │   ├── lidarseg/
│   │   │   └── v1.0-trainval/             # per-point labels (*.bin)
│   │   ├── v1.0-trainval/
│   │   ├── nuscenes_occ_infos_train.pkl
│   │   ├── nuscenes_occ_infos_val.pkl
```

### SemanticKITTI

Download SemanticKITTI from the [official site](http://www.semantic-kitti.org/dataset.html) and the train/val/test info pickles ([train](https://drive.google.com/file/d/1AlbseAbUkBrVjEZTbDbYcsA3l6LqqTEO/view?usp=drive_link) / [val](https://drive.google.com/file/d/1gF7rHdZqzcu2mwjzflKF5jcE-wOue0CI/view?usp=drive_link) / [test](https://drive.google.com/file/d/1InCnqx2oIKxIB9Kjb89RPcLood-a__2q/view?usp=drive_link)).

```
LiSeer-Simple
├── data/
│   ├── semantickitti/
│   │   ├── sequences/
│   │   │   ├── 00/
│   │   │   │   ├── velodyne/              # raw lidar .bin
│   │   │   │   └── labels/                # per-point labels (*.label)
│   │   │   ├── 08/                        # validation
│   │   │   └── 11/ .. 21/                 # test
│   │   ├── semantickitti_infos_train.pkl
│   │   ├── semantickitti_infos_val.pkl
│   │   ├── semantickitti_infos_test.pkl
```

### Waymo

Convert the Waymo Open Dataset to KITTI format (mmdet3d converter) and prepare the info pickles.

```
LiSeer-Simple
├── data/
│   ├── waymo/
│   │   ├── kitti_format/
│   │   │   ├── training/
│   │   │   │   └── velodyne/              # raw lidar .bin (xyz + features)
│   │   │   ├── waymo_infos_train.pkl
│   │   │   ├── waymo_infos_val.pkl
```

---

## Step 2 — Dense point clouds (LiDiff)

We use [LiDiff](https://github.com/PRBonn/LiDiff) for diffusion-based scene completion. First install LiDiff following its own README (it requires `MinkowskiEngine`, `diffusers` and `pytorch_lightning`, so a **separate conda env** is recommended), and download the LiDiff diffusion + refinement checkpoints.

Then run the completion pipeline shipped in this repo for each dataset. These scripts wrap LiDiff inference and write one completed (dense) point cloud per scan as a `.ply` file into a `refine/` folder next to the raw point clouds:

```shell
# nuScenes
python tools/diff_completion_pipeline_ddp_nu.py    --diff <lidiff_diff.ckpt> --refine <lidiff_refine.ckpt>
# SemanticKITTI
python tools/diff_completion_pipeline_ddp_sk.py    --diff <lidiff_diff.ckpt> --refine <lidiff_refine.ckpt>
# Waymo
python tools/diff_completion_pipeline_ddp_waymo.py --diff <lidiff_diff.ckpt> --refine <lidiff_refine.ckpt>
```

Resulting dense clouds:

- nuScenes:  `data/nuscenes/samples/LIDAR_TOP/refine/<token>.ply`
- SemanticKITTI: `data/semantickitti/sequences/<seq>/refine/<frame>.ply`
- Waymo: `data/waymo/kitti_format/training/refine/<frame>.ply`

---

## Step 3 — Dense labels

The dense labels are produced by voxelizing the sparse ground truth (voxel size 0.2 m), majority-voting a label per occupied voxel, and propagating it to the dense points (nearest occupied voxel for empty ones). Edit the hard-coded `root_dir` / output directory at the bottom of each script, then run:

```shell
python tools/generate_dense_points_labels_nu.py      # nuScenes
python tools/generate_dense_points_labels_sk.py      # SemanticKITTI
python tools/generate_dense_points_labels_waymo.py   # Waymo
```

Resulting dense labels:

- nuScenes:  `data/nuscenes/lidarseg_refine/v1.0-trainval/<token>_lidarseg.bin`
- SemanticKITTI: `data/semantickitti/sequences/<seq>/dense_labels/<frame>.label`
- Waymo: `data/waymo/kitti_format/training/dense_labels/<frame>.label`

---

## Final folder structure

```
LiSeer-Simple
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   │   └── LIDAR_TOP/
│   │   │       ├── *.bin                       # raw points
│   │   │       └── refine/*.ply                # dense points  (Step 2)
│   │   ├── sweeps/
│   │   ├── lidarseg/v1.0-trainval/*.bin        # raw labels
│   │   ├── lidarseg_refine/v1.0-trainval/*.bin # dense labels  (Step 3)
│   │   ├── v1.0-trainval/
│   │   ├── nuscenes_occ_infos_train.pkl
│   │   └── nuscenes_occ_infos_val.pkl
│   ├── semantickitti/
│   │   ├── sequences/
│   │   │   └── XX/
│   │   │       ├── velodyne/*.bin              # raw points
│   │   │       ├── labels/*.label              # raw labels
│   │   │       ├── refine/*.ply                # dense points  (Step 2)
│   │   │       └── dense_labels/*.label        # dense labels  (Step 3)
│   │   ├── semantickitti_infos_train.pkl
│   │   ├── semantickitti_infos_val.pkl
│   │   └── semantickitti_infos_test.pkl
│   └── waymo/
│       └── kitti_format/
│           ├── training/
│           │   ├── velodyne/*.bin              # raw points
│           │   ├── refine/*.ply                # dense points  (Step 2)
│           │   └── dense_labels/*.label        # dense labels  (Step 3)
│           ├── waymo_infos_train.pkl
│           └── waymo_infos_val.pkl
```
