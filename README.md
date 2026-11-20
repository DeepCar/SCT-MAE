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



---

# SCT-MAE: Installation, Dataset Preparation, and Usage

## Installation

Please refer to [INSTALL.md](https://github.com/chaytonmin/Occupancy-MAE/blob/main/docs/INSTALL.md) for the installation of [OpenPCDet (v0.5)](https://github.com/open-mmlab/OpenPCDet).

**Note:** SCT-MAE includes its own `pcdet` implementation. Install the dependencies and run `python setup.py develop` from the SCT-MAE project root rather than replacing its code with a separate OpenPCDet checkout. Ensure that the installed `spconv` supports the `spconv.pytorch` imports used by the model.

## Getting Started

Please refer to [GETTING_STARTED.md](https://github.com/chaytonmin/Occupancy-MAE/blob/main/docs/GETTING_STARTED.md).

For ONCE-specific dataset preparation, see [ONCE Dataset Preparation and Configuration](#once-dataset-preparation-and-configuration) below.

## Usage

### First Pre-training Voxel-MAE

Run the following commands from the `tools` directory.

**KITTI:**

Train with multiple GPUs:

```bash
bash ./scripts/dist_train_voxel_mae.sh ${NUM_GPUS} --cfg_file cfgs/kitti_models/voxel_mae_kitti.yaml --batch_size ${BATCH_SIZE}
```

Train with a single GPU:

```bash
python3 train_voxel_mae.py --cfg_file cfgs/kitti_models/voxel_mae_kitti.yaml --batch_size ${BATCH_SIZE}
```

**Waymo:**

```bash
python3 train_voxel_mae.py --cfg_file cfgs/kitti_models/voxel_mae_waymo.yaml --batch_size ${BATCH_SIZE}
```

**nuScenes:**

```bash
python3 train_voxel_mae.py --cfg_file cfgs/kitti_models/voxel_mae_nuscenes.yaml --batch_size ${BATCH_SIZE}
```

**ONCE:**

Train with multiple GPUs (requires the distributed script to support the ONCE configuration):

```bash
bash ./scripts/dist_train_voxel_mae.sh ${NUM_GPUS} --cfg_file cfgs/once_models/voxel_mae_once.yaml --batch_size ${BATCH_SIZE}
```

Train with a single GPU:

```bash
python3 train_voxel_mae.py --cfg_file cfgs/once_models/voxel_mae_once.yaml --batch_size ${BATCH_SIZE}
```

### ONCE Dataset Preparation and Configuration

The ONCE extension introduces dataset loading and configuration paths for self-supervised SCT-MAE pre-training on unlabeled LiDAR point clouds. This section documents the intended integration and does not claim completed ONCE experiments.

#### 1. Dataset directory structure

Download the required ONCE split from the [official ONCE dataset website](https://once-for-auto-driving.github.io/download.html#downloads). With the provided `once_dataset.yaml` configuration, organize the point clouds as follows:

```text
SCT-MAE-main/
├── data/
│   └── once/
│       └── raw_small/
│           └── data/
│               ├── 000000/
│               │   └── lidar_roof/
│               │       ├── <frame_id>.bin
│               │       └── ...
│               ├── 000001/
│               │   └── lidar_roof/
│               │       └── ...
│               └── ...
├── pcdet/
└── tools/
```

The ONCE LiDAR `.bin` files contain four point features: `(x, y, z, intensity)`. The proposed pre-training loader reads these point clouds without requiring 3D bounding-box labels. Depending on the loader implementation, a layout with sequence folders directly under `raw_small/` may also be supported.

#### 2. ONCE dataset loader

```text
pcdet/datasets/once/once_dataset.py
```

This module reads LiDAR frames and applies point-feature encoding and voxelization. The ONCE dataset class is registered in:

```text
pcdet/datasets/__init__.py
```

The self-supervised loader described here reads point clouds directly; it does **not** implement the original OpenPCDet `create_once_infos` command. That command must not be assumed to work unless the corresponding function has been added.

#### 3. Dataset configuration

```text
tools/cfgs/dataset_configs/once_dataset.yaml
```

This configuration specifies `DATA_PATH`, `POINT_CLOUD_RANGE`, point features, voxel size, and the voxelization settings. Adjust `DATA_PATH` to match the local ONCE dataset location.

#### 4. Pre-training model configuration

```text
tools/cfgs/once_models/voxel_mae_once.yaml
```

This file references the ONCE dataset configuration and defines the model and optimization parameters. The SCT-MAE backbone implementation is located at:

```text
pcdet/models/backbones_3d/voxel_mae.py
```

The training entry point is:

```text
tools/train_voxel_mae.py
```

#### 5. Running ONCE pre-training

From the project root:

```bash
cd tools
```

Single GPU (example batch size; adjust for available GPU memory):

```bash
python3 train_voxel_mae.py \
    --cfg_file cfgs/once_models/voxel_mae_once.yaml \
    --batch_size 4
```

Multiple GPUs (example with four GPUs; requires distributed launcher compatibility):

```bash
bash ./scripts/dist_train_voxel_mae.sh 4 \
    --cfg_file cfgs/once_models/voxel_mae_once.yaml \
    --batch_size 4
```

**Memory consideration:** ONCE's large point-cloud range can lead to high GPU memory use in dense occupancy reconstruction. The example batch size is not a verified hardware recommendation; voxelization resolution and reconstruction memory requirements should be checked before full-scale training.

#### 6. Implementation and evaluation status

The ONCE integration is being extended through dataset loading and model configuration while retaining SCT-MAE's range-aware voxel masking, vector-quantized shape representation, and autoregressive shape completion. Successful end-to-end ONCE training, checkpoint validation, and downstream 3D detection evaluation have not yet been established. No ONCE benchmark scores are reported here.

### Then Training OpenPCDet

Same as [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), using the pre-trained model from Voxel-MAE.

```bash
bash ./scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/kitti_models/second.yaml --batch_size ${BATCH_SIZE} --pretrained_model ../output/kitti/voxel_mae/ckpt/check_point_10.pth
```

The downstream example above is for **KITTI**, not ONCE. ONCE downstream detection requires an appropriate detection model configuration, labeled data, and a compatible evaluation pipeline. The checkpoint path must match the actual pre-training output.
