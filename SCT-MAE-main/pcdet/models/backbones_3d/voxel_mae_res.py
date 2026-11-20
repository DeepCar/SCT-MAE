from functools import partial

import torch
import torch.nn as nn
import spconv.pytorch as spconv

from ...utils.spconv_utils import replace_feature
from .spconv_backbone import post_act_block

from .voxel_mae_article_aligned import Voxel_MAE, SCT


class SparseBasicBlock(spconv.SparseModule):
    """
    Sparse residual block.

    SubMConv3d -> BN -> ReLU
                -> SubMConv3d -> BN
                -> Residual Addition -> ReLU
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        indice_key=None,
        norm_fn=None
    ):
        super().__init__()

        if norm_fn is None:
            norm_fn = partial(
                nn.BatchNorm1d,
                eps=1e-3,
                momentum=0.01
            )

        self.conv1 = spconv.SubMConv3d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            indice_key=indice_key
        )

        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()

        self.conv2 = spconv.SubMConv3d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            indice_key=indice_key
        )

        self.bn2 = norm_fn(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)

        out = replace_feature(
            out,
            self.relu(self.bn1(out.features))
        )

        out = self.conv2(out)

        out = replace_feature(
            out,
            self.bn2(out.features)
        )

        if self.downsample is not None:
            identity = self.downsample(x)

            if hasattr(identity, "features"):
                identity = identity.features

        if out.features.shape != identity.shape:
            raise RuntimeError(
                "Residual feature dimensions do not match"
            )

        out = replace_feature(
            out,
            self.relu(out.features + identity)
        )

        return out


class Voxel_MAE_res(Voxel_MAE):
    """
    Residual SCT-MAE with explicit masked-token autoregressive decoding.

    Uses the same teacher-forcing and inference contract as the corrected
    base model; only the sparse encoder is replaced here.

    Inherits:
        - Range-aware masking
        - Vector quantization
        - Hybrid positional encoding
        - Autoregressive transformer
        - Teacher forcing
        - Occupancy reconstruction
        - Composite loss

    Replaces the sparse encoder with
    residual sparse convolutional blocks.
    """

    def __init__(
        self,
        model_cfg,
        input_channels,
        grid_size,
        voxel_size=None,
        point_cloud_range=None,
        **kwargs
    ):
        if voxel_size is None:
            voxel_size = model_cfg.get(
                "VOXEL_SIZE",
                None
            )

        if point_cloud_range is None:
            point_cloud_range = model_cfg.get(
                "POINT_CLOUD_RANGE",
                None
            )

        if voxel_size is None:
            raise ValueError(
                "voxel_size is required"
            )

        if point_cloud_range is None:
            raise ValueError(
                "point_cloud_range is required"
            )

        super().__init__(
            model_cfg=model_cfg,
            input_channels=input_channels,
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            **kwargs
        )

        norm_fn = partial(
            nn.BatchNorm1d,
            eps=1e-3,
            momentum=0.01
        )

        block = post_act_block

        # ----------------------------------------------
        # Input sparse convolution
        # ----------------------------------------------

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channels,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="subm1"
            ),
            norm_fn(16),
            nn.ReLU()
        )

        # ----------------------------------------------
        # Residual stage 1
        # ----------------------------------------------

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(
                16,
                16,
                norm_fn=norm_fn,
                indice_key="res1"
            ),
            SparseBasicBlock(
                16,
                16,
                norm_fn=norm_fn,
                indice_key="res1"
            )
        )

        # ----------------------------------------------
        # Residual stage 2
        # ----------------------------------------------

        self.conv2 = spconv.SparseSequential(
            block(
                16,
                32,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=1,
                indice_key="spconv2",
                conv_type="spconv"
            ),
            SparseBasicBlock(
                32,
                32,
                norm_fn=norm_fn,
                indice_key="res2"
            ),
            SparseBasicBlock(
                32,
                32,
                norm_fn=norm_fn,
                indice_key="res2"
            )
        )

        # ----------------------------------------------
        # Residual stage 3
        # ----------------------------------------------

        self.conv3 = spconv.SparseSequential(
            block(
                32,
                64,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=1,
                indice_key="spconv3",
                conv_type="spconv"
            ),
            SparseBasicBlock(
                64,
                64,
                norm_fn=norm_fn,
                indice_key="res3"
            ),
            SparseBasicBlock(
                64,
                64,
                norm_fn=norm_fn,
                indice_key="res3"
            )
        )

        # ----------------------------------------------
        # Residual stage 4
        # ----------------------------------------------

        self.conv4 = spconv.SparseSequential(
            block(
                64,
                128,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=(0, 1, 1),
                indice_key="spconv4",
                conv_type="spconv"
            ),
            SparseBasicBlock(
                128,
                128,
                norm_fn=norm_fn,
                indice_key="res4"
            ),
            SparseBasicBlock(
                128,
                128,
                norm_fn=norm_fn,
                indice_key="res4"
            )
        )

        # ----------------------------------------------
        # Final sparse downsampling
        # ----------------------------------------------

        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(
                128,
                128,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                padding=model_cfg.get("last_pad", 0),
                bias=False,
                indice_key="spconv_down2"
            ),
            norm_fn(128),
            nn.ReLU()
        )

        # ----------------------------------------------
        # SCT for 128-channel residual encoder
        # ----------------------------------------------

        self.sct = SCT(
            in_channels=128,
            grid_size=self.token_grid,
            down_channels=self.embed_dim,
            dict_size=self.dict_size,
            beta=float(
                model_cfg.get("VQ_BETA", 0.25)
            )
        )

        # All remaining modules are inherited
        # from Voxel_MAE:
        #
        # self.pos_encoder
        # self.decoder
        # self.to_volume
        # self.deconv1
        # self.deconv2
        # self.deconv3
        # self.criterion

    def _encode(self, sparse_tensor):
        x = self.conv_input(sparse_tensor)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv_out(x)

        return x