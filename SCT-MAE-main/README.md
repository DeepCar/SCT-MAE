# SCT-MAE

<div align="center">

<img width="1376" height="768" alt="it7Q78fH7MJTQsEU5I6HMrpRpsY-jScWgxuIRI2FnwatJTPjoQ" src="https://github.com/user-attachments/assets/a1084647-40bb-4564-9bd6-afb9c3bc5bb1" />


**Abdollah Amirkhani** · **Milad Mandalzadeh**

*Iran University of Science and Technology*

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![3D Detection](https://img.shields.io/badge/Task-3D_Object_Detection-4C8CBF?style=flat-square)
![Self-Supervised Learning](https://img.shields.io/badge/Paradigm-Self--Supervised_Learning-6BA77C?style=flat-square)

</div>

---

## Overview

**SCT-MAE** is a self-supervised pre-training framework for LiDAR-based 3D object detection that combines range-aware volumetric masked autoencoding with a Shape Completion Transformer that autoregressively reconstructs voxel sequences, yielding robust geometric representations that transfer to downstream detection.

---

## Key Ideas

- **Range-aware / spatial-range masking** — masking is driven by the geometric structure of the point cloud (range and spatial context) rather than uniform random voxel drops, so pre-training tokens are informative and prediction is non-trivial.
- **Volumetric masked autoencoding** — the scene is voxelized into a volumetric grid; a large portion of voxels is masked and predicted from visible context, forcing the network to learn 3D shape priors.
- **Vector quantization** — a discrete latent codebook regularizes the learned representation and provides compact, reusable geometric tokens.
- **Shape Completion Transformer (SCT)** — an autoregressive decoder that reconstructs the masked voxel sequence, effectively performing shape completion and encouraging long-range, object-level geometric reasoning.

## Architecture Pipeline

```
LiDAR Point Cloud
        │
        ▼
 Voxelization (volumetric grid)
        │
        ▼
 Range-aware / spatial-range Masking
        │
        ├── visible tokens ──► 3D Backbone Encoder (OpenPCDet-style)
        │                              │
        │                              ▼
        │                     Vector Quantization (discrete codes)
        │                              │
        │                              ▼
        └── masked targets ──► Shape Completion Transformer
                               (autoregressive voxel-sequence reconstruction)
                                       │
                                       ▼
                        Reconstruction loss (pre-training: train_voxel_mae.py)
                                       │
                                       ▼
              Pre-trained weights ──► Downstream 3D Detection (train.py)
```

Pre-training is performed with the voxel-MAE modules (`voxel_mae.py`, `voxel_mae_res.py`); the learned encoder weights are then used to initialize standard detectors (e.g., SECOND, PV-RCNN, CenterPoint variants) for supervised 3D detection.

---

## Repository Structure

```
SCT-MAE/
├── pcdet/                      # Core of the OpenPCDet-based 3D detection framework
├── tools/
│   ├── cfgs/
│   │   ├── dataset_configs/    # kitti_dataset.yaml, nuscenes_dataset.yaml,
│   │   │                       # waymo_dataset.yaml, lyft_dataset.yaml, pandaset_dataset.yaml
│   │   ├── voxel_mae_models/   # voxel_mae_kitti.yaml, voxel_mae_nuscenes.yaml,
│   │   │                       # voxel_mae_waymo.yaml (pre-training configs)
│   │   └── <detector models>/  # SECOND, PV-RCNN, CenterPoint, PV-RCNN++, ... configs
│   ├── scripts/
│   │   ├── dist_train.sh
│   │   ├── dist_test.sh
│   │   └── dist_train_voxel_mae.sh
│   ├── train.py
│   ├── test.py
│   └── demo.py
├── train_voxel_mae.py          # Voxel-MAE pre-training entry point
├── voxel_mae.py                # Volumetric masked autoencoder module
├── voxel_mae_res.py            # Range-aware variant with residual / range encodings
├── docs/                       # Documentation and images
├── setup.py
└── requirements.txt
```



---

## Requirements

- Linux (recommended), CUDA-capable GPU
- Python 3.x with PyTorch
- Dependencies listed in `requirements.txt`
- The underlying 3D detection framework is **OpenPCDet** (bundled here as `pcdet/`); follow its upstream documentation for CUDA/ops specifics (e.g., spconv, KITTI devkit evaluation).

## Installation

```bash
# 1. Create and activate a virtual environment / conda env (recommended)

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install the pcdet package in development mode
python setup.py develop
```

---

## Dataset Setup

Configuration files are provided for **KITTI**, **nuScenes**, **Waymo**, **Lyft**, and **PandaSet** under `tools/cfgs/dataset_configs/`. The main workflow of this project is documented for **KITTI**, **nuScenes**, and **Waymo**.

1. Download the datasets from their official providers.
2. Generate the info/annotation files required by OpenPCDet for the chosen dataset (see OpenPCDet dataset preparation instructions).
3. Edit the `DATA_PATH` field inside the corresponding YAML file (e.g., `tools/cfgs/dataset_configs/kitti_dataset.yaml`) to point to your local dataset directory.
4. Keep the same directory layout expected by OpenPCDet for each dataset.



---

## Pre-training (Voxel-MAE)



```bash
bash tools/scripts/dist_train_voxel_mae.sh \
    --cfg_file tools/cfgs/voxel_mae_models/voxel_mae_kitti.yaml \
    --batch_size 4 \
    --workers 8 \
    --extra_tag voxel_mae_pretrain
```

Equivalent (single-process) form:

```bash
python train_voxel_mae.py \
    --cfg_file tools/cfgs/voxel_mae_models/voxel_mae_kitti.yaml \
    --extra_tag voxel_mae_pretrain
```

## Downstream Training



```bash
bash tools/scripts/dist_train.sh \
    --cfg_file tools/cfgs/kitti_models/second.yaml \
    --pretrained_model <path/to/voxel_mae_pretrained.pth> \
    --batch_size 4 \
    --workers 8
```

## Evaluation


```bash
bash tools/scripts/dist_test.sh \
    --cfg_file tools/cfgs/kitti_models/second.yaml \
    --ckpt <path/to/trained_checkpoint.pth>
```

Single-inference demo on a sample point cloud:

```bash
python demo.py \
    --cfg_file tools/cfgs/kitti_models/second.yaml \
    --ckpt <path/to/trained_checkpoint.pth> \
    --data_path <path/to/sample_point_cloud.bin>
```

---

## Limitations & Responsible Reproducibility

- **No data, checkpoints, logs, or outputs are shipped** with this repository; you must obtain and prepare datasets yourself.
- Absolute numbers may differ across hardware, driver, CUDA/spconv versions, and random seeds. Results reported in the associated paper were produced under the authors' specific environment.
- **We do not claim independent retraining of paper baselines or guaranteed bit-wise reproducibility.** The provided configs and scripts are templates intended to make the method reproducible *in spirit*; please report your own environment details when comparing results.
- Large-scale dataset preparation (especially Waymo and nuScenes) requires significant disk space and download time.

---

## Citation

If you use this code, please cite the associated work:

```bibtex
}
}
```

## Acknowledgements

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) — the underlying 3D object detection framework on which this project is built.
- The PyTorch and open-source 3D perception communities whose tools made this work possible.

---

<div align="center">

*Thanks for visiting — feedback and issues are welcome.* ✨

</div>
