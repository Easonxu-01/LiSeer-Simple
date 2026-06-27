# Step-by-step installation instructions

The codebase has been tested with **Python 3.8**, **PyTorch 1.10.1** and **CUDA 11.3**.

**1. Create a conda virtual environment and activate it.**

```shell
conda create -n liseer python=3.8 -y
conda activate liseer
```

**2. Install PyTorch and torchvision (tested on torch==1.10.1 & cuda==11.3).**

```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

**3. Install gcc>=5 in the conda env.**

```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**4. Install MMCV / MMDet / MMSeg / MMEngine.**

Follow the [official MMCV instructions](https://github.com/open-mmlab/mmcv). The versions below are the ones we tested against:

```shell
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
pip install mmengine==0.10.3
```

**5. Install mmdet3d from source.**

```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
cd ..
```

**6. Install the remaining dependencies.**

```shell
# core
pip install numpy==1.22.4
pip install numba==0.48.0
pip install scipy==1.10.1
pip install scikit-learn==1.3.2

# sparse conv & scatter (CUDA 11.3 builds)
pip install spconv-cu113==2.3.6
pip install torch-scatter==2.0.9

# backbones / misc
pip install timm==1.0.11
pip install einops==0.8.0
pip install transforms3d==0.4.2
pip install open3d==0.17.0
pip install PyMCubes==0.1.4
pip install fvcore==0.1.5.post20221221
pip install Ipython
pip install yapf==0.40.1
pip install setuptools==59.5.0

# dataset devkits
pip install nuscenes-devkit==1.1.11
# Waymo only:
pip install waymo-open-dataset-tf-2-6-0==1.4.8
```

**7. Install the LiSeer plugin in develop mode.**

This installs the `projects.unilidar_plugin` package so that the config `plugin_dir` can be resolved at train/eval time. The codebase no longer ships any CUDA extension, so this step is a pure Python install.

```shell
cd LiSeer-Simple
export PYTHONPATH="."
python setup.py develop
```

> Note: data preparation (raw datasets, dense point clouds and dense labels) is described separately in [`prepare_data.md`](prepare_data.md).
