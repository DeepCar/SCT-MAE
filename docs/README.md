# Self-Supervised Range-Aware Volumetric Masked Autoencoder with Shape Completion Transformer for LiDAR 3D Object Detection



## Introduction
SCT-MAE (Shape Completion Transformer Masked Autoencoder) is an advanced self-supervised framework designed to address the challenges of LiDAR-based 3D object detection, specifically focusing on the sparsity, heterogeneity, and occlusion of LiDAR data. By integrating range-aware masking, vector quantization, and an autoregressive transformer decoder, SCT-MAE moves beyond traditional local patch reconstruction, enabling the model to capture and complete coherent global structures of 3D objects. The framework excels at learning robust, high-quality 3D representations from sparse and incomplete data without the need for manual annotations, making it a scalable and efficient solution for autonomous systems. Extensive testing on prominent 3D detection benchmarks such as KITTI, nuScenes, and ONCE demonstrates that SCT-MAE significantly improves both the accuracy and robustness of object detection, especially in challenging scenarios involving sparse point clouds and heavy occlusions. This repository provides the implementation of SCT-MAE, offering a comprehensive solution for leveraging self-supervised learning to enhance 3D object detection from LiDAR data in autonomous driving and robotics applications.

<p align="center">
<img src="docs/ُSCT_MAE" width="100%"/>Flowchart of Voxel-MAE
</p>

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

##  License
Our codes are released under the Apache 2.0 license.

## Acknowledgement

This repository is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

