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

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of [OpenPCDet(v0.5)](https://github.com/open-mmlab/OpenPCDet).

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) .

## Usage

### First Pre-training Voxel-MAE

KITTI:

```
Train with multiple GPUs:
bash ./scripts/dist_train_voxel_mae.sh ${NUM_GPUS}  --cfg_file cfgs/kitti_models/voxel_mae_kitti.yaml --batch_size ${BATCH_SIZE}

Train with a single GPU:
python3 train_voxel_mae.py  --cfg_file cfgs/kitti_models/voxel_mae_kitti.yaml --batch_size ${BATCH_SIZE}
```

Waymo:

```
python3 train_voxel_mae.py  --cfg_file cfgs/kitti_models/voxel_mae_waymo.yaml --batch_size ${BATCH_SIZE}
```

nuScenes:

```
python3 train_voxel_mae.py  --cfg_file cfgs/kitti_models/voxel_mae_nuscenes.yaml --batch_size ${BATCH_SIZE}
```

### Then traing OpenPCDet

Same as [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) with pre-trained model from our Voxel-MAE.
```
bash ./scripts/dist_train.sh ${NUM_GPUS}  --cfg_file cfgs/kitti_models/second.yaml --batch_size ${BATCH_SIZE} --pretrained_model ../output/kitti/voxel_mae/ckpt/check_point_10.pth
```
