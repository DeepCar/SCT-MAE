from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

from .spconv_backbone import post_act_block


class HybridPosEncoding3D(nn.Module):
    """
    Hybrid 3D positional encoding.

    PE = PE_sinusoidal + PE_learnable

    Input/output shape: (B, N, C)
    """

    def __init__(self, grid_size, embed_dim):
        super().__init__()

        self.grid_size = tuple(grid_size)
        self.embed_dim = embed_dim

        D, H, W = self.grid_size
        N = D * H * W

        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, D),
                torch.linspace(0, 1, H),
                torch.linspace(0, 1, W),
                indexing="ij"
            ),
            dim=-1
        ).reshape(N, 3)

        pe = torch.zeros(N, embed_dim)

        for axis in range(3):
            dims = list(range(axis, embed_dim, 3))
            pairs = len(dims) // 2

            if pairs == 0:
                continue

            frequencies = torch.exp(
                -math.log(10000.0)
                * torch.arange(pairs).float()
                / max(pairs - 1, 1)
            )

            angles = (
                coords[:, axis:axis + 1]
                * frequencies.unsqueeze(0)
                * (2.0 * math.pi)
            )

            sin_indices = [
                dims[2 * j] for j in range(pairs)
            ]

            cos_indices = [
                dims[2 * j + 1] for j in range(pairs)
            ]

            pe[:, sin_indices] = torch.sin(angles)
            pe[:, cos_indices] = torch.cos(angles)

        self.register_buffer(
            "fixed",
            pe.unsqueeze(0),
            persistent=False
        )

        self.learnable = nn.Parameter(
            torch.zeros(1, N, embed_dim)
        )

        nn.init.normal_(self.learnable, std=0.02)

    def get_position(self):
        return self.fixed + self.learnable

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected tensor (B,N,C)")

        if x.shape[1] != self.fixed.shape[1]:
            raise ValueError("Invalid sequence length")

        if x.shape[2] != self.embed_dim:
            raise ValueError("Invalid embedding dimension")

        return x + self.get_position().to(dtype=x.dtype)


class SCT(nn.Module):
    """
    Vector-quantized shape completion bottleneck.

    Uses:
      - 3D feature projection
      - spatial compression
      - learnable codebook
      - nearest-neighbor quantization
      - straight-through estimator
      - codebook and commitment losses
    """

    def __init__(
        self,
        in_channels,
        grid_size=(2, 8, 8),
        down_channels=256,
        dict_size=512,
        beta=0.25
    ):
        super().__init__()

        self.grid_size = tuple(grid_size)
        self.beta = beta

        self.proj = nn.Conv3d(
            in_channels,
            down_channels,
            kernel_size=1
        )

        self.dictionary = nn.Embedding(
            dict_size,
            down_channels
        )

        nn.init.uniform_(
            self.dictionary.weight,
            -1.0 / dict_size,
            1.0 / dict_size
        )

    def forward(self, volume):
        # volume: (B,C,D,H,W)

        projected = self.proj(volume)

        compressed = F.adaptive_avg_pool3d(
            projected,
            self.grid_size
        )

        flat = compressed.flatten(2)
        flat = flat.transpose(1, 2).contiguous()

        codebook = self.dictionary.weight

        distances = (
            flat.square().sum(dim=-1, keepdim=True)
            - 2.0 * torch.matmul(flat, codebook.T)
            + codebook.square().sum(dim=-1)
        )

        indices = torch.argmin(distances, dim=-1)

        quantized = self.dictionary(indices)

        codebook_loss = F.mse_loss(
            quantized,
            flat.detach()
        )

        commitment_loss = F.mse_loss(
            flat,
            quantized.detach()
        )

        vq_loss = (
            codebook_loss
            + self.beta * commitment_loss
        )

        quantized_st = (
            flat + (quantized - flat).detach()
        )

        return quantized_st, indices, vq_loss


class EnhancedAutoregressiveDecoder(nn.Module):
    """
    Autoregressive codebook-token prediction.

    Training:
        teacher forcing + causal attention

    Inference:
        sequential token generation
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=16,
        num_layers=12,
        dict_size=512
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )

        self.sos = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(
            embed_dim,
            dict_size
        )

    @staticmethod
    def causal_mask(length, device):
        return torch.triu(
            torch.ones(
                length,
                length,
                device=device,
                dtype=torch.bool
            ),
            diagonal=1
        )

    def decode(self, tokens, memory):
        mask = self.causal_mask(
            tokens.shape[1],
            tokens.device
        )

        hidden = self.transformer(
            tgt=tokens,
            memory=memory,
            tgt_mask=mask
        )

        return self.classifier(hidden)

    def forward_train(
        self,
        memory,
        target_ids,
        observed_ids,
        token_mask,
        embedding,
        position
    ):
        """Teacher forcing with an explicit known/unknown token split.

        token_mask[b,t] = True means position t must be predicted.
        At decoder step t, the input consists of SOS and positions < t:
        observed ids for known positions, GT ids for masked positions.
        Thus, future target ids never reach decoder self-attention.
        All positions may be queried, but CE is applied only to masked ones.
        Encoder memory contains features from the partially observed scan.
        """
        if target_ids.shape != observed_ids.shape or target_ids.shape != token_mask.shape:
            raise ValueError("Target, observed and token mask shapes must match")
        B, N = target_ids.shape
        previous_ids = torch.where(token_mask, target_ids, observed_ids)[:, :-1]
        previous_tokens = embedding(previous_ids).detach()
        start_token = self.sos.expand(B, 1, -1)
        decoder_input = torch.cat([start_token, previous_tokens], dim=1)
        decoder_input = decoder_input + position
        return self.decode(decoder_input, memory)

    @torch.no_grad()
    def generate(
        self,
        memory,
        observed_ids,
        token_mask,
        embedding,
        position
    ):
        """Causal inference: copy observed ids, predict only unknown ids.

        Unlike teacher forcing, masked positions in the prefix contain model
        predictions. Consequently, exposure bias / accumulated errors are
        possible during inference and must be discussed in the manuscript.
        """
        B, N, _ = memory.shape
        if observed_ids.shape != (B, N) or token_mask.shape != (B, N):
            raise ValueError("Observed IDs and token mask must have shape (B,N)")

        if not token_mask.any().item():
            return observed_ids.clone()

        tokens = self.sos.expand(B, 1, -1)
        generated_ids = []
        for step in range(N):
            decoder_input = tokens + position[:, :step + 1]
            logits = self.decode(decoder_input, memory)
            predicted_id = logits[:, -1].argmax(dim=-1)
            next_id = torch.where(
                token_mask[:, step], predicted_id, observed_ids[:, step]
            )
            generated_ids.append(next_id)
            if step + 1 < N:
                tokens = torch.cat(
                    [tokens, embedding(next_id).unsqueeze(1)], dim=1
                )
        return torch.stack(generated_ids, dim=1)


class Voxel_MAE(nn.Module):

    def __init__(
        self,
        model_cfg,
        input_channels,
        grid_size,
        voxel_size,
        point_cloud_range,
        **kwargs
    ):
        super().__init__()

        self.model_cfg = model_cfg
        self.sparse_shape = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(
            nn.BatchNorm1d,
            eps=1e-3,
            momentum=0.01
        )

        block = post_act_block

        # --------------------------------------------------
        # Sparse 3D Encoder
        # --------------------------------------------------

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channels,
                16,
                3,
                padding=1,
                bias=False,
                indice_key="subm1"
            ),
            norm_fn(16),
            nn.ReLU()
        )

        self.conv1 = spconv.SparseSequential(
            block(
                16, 16, 3,
                norm_fn=norm_fn,
                padding=1,
                indice_key="subm1"
            )
        )

        self.conv2 = spconv.SparseSequential(
            block(
                16, 32, 3,
                norm_fn=norm_fn,
                stride=(1, 2, 2),
                padding=1,
                indice_key="spconv2",
                conv_type="spconv"
            ),
            block(
                32, 32, 3,
                norm_fn=norm_fn,
                padding=1,
                indice_key="subm2"
            ),
            block(
                32, 32, 3,
                norm_fn=norm_fn,
                padding=1,
                indice_key="subm2"
            )
        )

        self.conv3 = spconv.SparseSequential(
            block(
                32, 64, 3,
                norm_fn=norm_fn,
                stride=(1, 2, 2),
                padding=1,
                indice_key="spconv3",
                conv_type="spconv"
            ),
            block(
                64, 64, 3,
                norm_fn=norm_fn,
                padding=1,
                indice_key="subm3"
            ),
            block(
                64, 64, 3,
                norm_fn=norm_fn,
                padding=1,
                indice_key="subm3"
            )
        )

        self.conv4 = spconv.SparseSequential(
            block(
                64, 64, 3,
                norm_fn=norm_fn,
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                indice_key="spconv4",
                conv_type="spconv"
            ),
            block(
                64, 64, 3,
                norm_fn=norm_fn,
                padding=1,
                indice_key="subm4"
            ),
            block(
                64, 64, 3,
                norm_fn=norm_fn,
                padding=1,
                indice_key="subm4"
            )
        )

        use_conv_out = model_cfg.get(
            "RETURN_ENCODED_TENSOR",
            True
        )

        if use_conv_out:
            last_pad = model_cfg.get("last_pad", 0)

            self.conv_out = spconv.SparseSequential(
                spconv.SparseConv3d(
                    64,
                    128,
                    (3, 1, 1),
                    stride=(2, 1, 1),
                    padding=last_pad,
                    bias=False,
                    indice_key="spconv_down2"
                ),
                norm_fn(128),
                nn.ReLU()
            )

            encoder_channels = 128
        else:
            self.conv_out = None
            encoder_channels = 64

        # --------------------------------------------------
        # SCT-MAE Configuration
        # --------------------------------------------------

        self.token_grid = tuple(
            model_cfg.get(
                "SCT_TOKEN_GRID",
                (2, 8, 8)
            )
        )

        self.dict_size = int(
            model_cfg.get("SCT_DICT_SIZE", 512)
        )

        self.embed_dim = int(
            model_cfg.get("SCT_EMBED_DIM", 256)
        )

        num_heads = int(
            model_cfg.get("SCT_HEADS", 16)
        )

        num_layers = int(
            model_cfg.get("SCT_LAYERS", 12)
        )

        self.sct = SCT(
            in_channels=encoder_channels,
            grid_size=self.token_grid,
            down_channels=self.embed_dim,
            dict_size=self.dict_size,
            beta=float(
                model_cfg.get("VQ_BETA", 0.25)
            )
        )

        self.pos_encoder = HybridPosEncoding3D(
            self.token_grid,
            self.embed_dim
        )

        self.decoder = EnhancedAutoregressiveDecoder(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dict_size=self.dict_size
        )

        # --------------------------------------------------
        # Volumetric Reconstruction Decoder
        # --------------------------------------------------

        self.to_volume = nn.Linear(
            self.embed_dim,
            64
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(
                32, 8,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )

        self.deconv3 = nn.Conv3d(
            8, 1, kernel_size=1
        )

        # --------------------------------------------------
        # Losses
        # --------------------------------------------------

        self.criterion = nn.BCEWithLogitsLoss()

        self.occ_weight = float(
            model_cfg.get("OCC_WEIGHT", 0.7)
        )

        self.vq_weight = float(
            model_cfg.get("VQ_WEIGHT", 0.3)
        )

        self.ar_weight = float(
            model_cfg.get("AR_WEIGHT", 1.0)
        )

        self.mask_ratios = tuple(
            model_cfg.get(
                "RANGE_MASK_RATIOS",
                (0.9, 0.7, 0.5)
            )
        )

        self.token_order = model_cfg.get(
            "SCT_TOKEN_ORDER",
            "raster"
        )

        if self.token_order not in ("raster", "reverse"):
            raise ValueError(
                "Token order must be raster or reverse"
            )

        # If true, evaluation also hides occupied input voxels for a genuine
        # completion benchmark. Otherwise eval simply preserves all input.
        self.mask_on_eval = bool(model_cfg.get("MASK_ON_EVAL", False))
        # A coarse token is unknown if more than this fraction of its occupied
        # input voxels were masked. Empty cells are also unknown.

        self.forward_re_dict = {}
        self.num_point_features = 16

    def _encode(self, sparse_tensor):
        x = self.conv_input(sparse_tensor)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if self.conv_out is not None:
            x = self.conv_out(x)

        return x

    def _mask(self, coords):
        """
        coords: [batch, z, y, x]

        Compute horizontal range in meters.
        """

        x = (
            (coords[:, 3].float() + 0.5)
            * float(self.voxel_size[0])
            + float(self.point_cloud_range[0])
        )

        y = (
            (coords[:, 2].float() + 0.5)
            * float(self.voxel_size[1])
            + float(self.point_cloud_range[1])
        )

        distance = torch.sqrt(
            x.square() + y.square()
        )

        region = torch.where(
            distance <= 30.0,
            0,
            torch.where(
                distance <= 50.0,
                1,
                2
            )
        )

        keep = torch.zeros(
            coords.shape[0],
            device=coords.device,
            dtype=torch.bool
        )

        for batch_id in coords[:, 0].unique():
            for region_id, ratio in enumerate(
                self.mask_ratios
            ):
                indices = torch.nonzero(
                    (coords[:, 0] == batch_id)
                    & (region == region_id),
                    as_tuple=True
                )[0]

                if indices.numel() == 0:
                    continue

                num_keep = max(
                    1,
                    round(
                        indices.numel() * (1.0 - ratio)
                    )
                )

                permutation = torch.randperm(
                    indices.numel(),
                    device=coords.device
                )

                keep[
                    indices[permutation[:num_keep]]
                ] = True

        return keep

    def _sparse(self, features, coords, batch_size):
        return spconv.SparseConvTensor(
            features=features,
            indices=coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

    def _order(self, tokens):
        if self.token_order == "reverse":
            return tokens.flip(dim=1)
        return tokens

    def _unorder(self, tokens):
        if self.token_order == "reverse":
            return tokens.flip(dim=1)
        return tokens

    def _token_mask_from_voxels(self, voxel_coords, keep_mask, batch_size,
                                dense_occupancy):
        """Lift the range-masked occupied voxel positions to SCT token cells.

        A coarse token is a reconstruction target iff at least one occupied
        input voxel contributing to that cell was removed by range masking.
        This is a deterministic rule (no extra masking threshold or ground-
        truth-dependent occupancy-fraction hyperparameter). The voxel
        coordinates and keep_mask are known when constructing the self-
        supervised pretext task. This is NOT a mask estimator for arbitrary
        externally occluded LiDAR scenes at deployment time.
        """
        masked_coords = voxel_coords[~keep_mask]
        masked_dense = self._sparse(
            torch.ones((masked_coords.shape[0], 1),
                       device=dense_occupancy.device,
                       dtype=dense_occupancy.dtype),
            masked_coords, batch_size
        ).dense()
        masked_coarse = F.adaptive_max_pool3d(masked_dense, self.token_grid)
        return (masked_coarse > 0).flatten(2).squeeze(1)

    def get_loss(self, tb_dict=None):
        if tb_dict is None:
            tb_dict = {}

        data = self.forward_re_dict

        loss_occ = self.criterion(
            data["pred"],
            data["target"]
        )

        loss_vq = data["vq_loss"]
        loss_ar = data["ar_loss"]

        total_loss = (
            self.occ_weight * loss_occ
            + self.vq_weight * loss_vq
            + self.ar_weight * loss_ar
        )

        tb_dict.update({
            "loss_rpn": total_loss.item(),
            "loss_bce": loss_occ.item(),
            "loss_sct": loss_vq.item(),
            "loss_ar": loss_ar.item()
        })

        return total_loss, tb_dict

    def forward(self, batch_dict):
        voxel_features = batch_dict["voxel_features"]
        voxel_coords = batch_dict["voxel_coords"]
        batch_size = batch_dict["batch_size"]

        # 1. Range-aware masking
        if self.training or self.mask_on_eval:
            keep_mask = self._mask(voxel_coords)
        else:
            keep_mask = torch.ones(
                voxel_coords.shape[0],
                device=voxel_coords.device,
                dtype=torch.bool
            )

        partial_sparse = self._sparse(
            voxel_features[keep_mask],
            voxel_coords[keep_mask],
            batch_size
        )

        # 2. Full occupancy target
        occupancy_sparse = self._sparse(
            torch.ones(
                voxel_coords.shape[0],
                1,
                device=voxel_features.device
            ),
            voxel_coords,
            batch_size
        )

        target_occupancy = occupancy_sparse.dense()

        # 3. Sparse encoding
        encoded = self._encode(partial_sparse).dense()

        # 4. Vector quantization: observed tokens come ONLY from partial input.
        memory, observed_ids, vq_loss = self.sct(encoded)

        # Coarse token mask follows the range-masking operation itself.
        # Masked coordinates are available in self-supervised pre-training.
        token_mask = self._token_mask_from_voxels(
            voxel_coords, keep_mask, batch_size, target_occupancy
        )
        memory = self._order(memory)
        observed_ids = self._order(observed_ids)
        token_mask = self._order(token_mask)

        # 5. Hybrid positional encoding
        position = self.pos_encoder.get_position()

        position = position.to(
            device=memory.device,
            dtype=memory.dtype
        )

        position = self._order(position)

        memory = memory + position

        # 6. Autoregressive reconstruction
        if self.training:
            # Full input is used only to build target indices.
            with torch.no_grad():
                full_sparse = self._sparse(
                    voxel_features,
                    voxel_coords,
                    batch_size
                )

                full_encoded = self._encode(
                    full_sparse
                ).dense()

                _, target_ids, _ = self.sct(
                    full_encoded
                )

            target_ids = self._order(target_ids)

            logits = self.decoder.forward_train(
                memory=memory,
                target_ids=target_ids,
                observed_ids=observed_ids,
                token_mask=token_mask,
                embedding=self.sct.dictionary,
                position=position
            )

            # Supervise ONLY masked coarse tokens, never the visible tokens.
            token_ce = F.cross_entropy(
                logits.reshape(-1, self.dict_size),
                target_ids.reshape(-1), reduction="none"
            ).view_as(target_ids)
            mask_float = token_mask.to(token_ce.dtype)
            ar_loss = (token_ce * mask_float).sum() / mask_float.sum().clamp(min=1)

            # Differentiable predictions for unknown positions; preserve
            # partial-encoder features for visible positions.
            probabilities = F.softmax(logits, dim=-1)
            predicted_features = torch.matmul(
                probabilities, self.sct.dictionary.weight
            )
            observed_features = self.sct.dictionary(observed_ids)
            token_features = torch.where(
                token_mask.unsqueeze(-1), predicted_features, observed_features
            )

        else:
            predicted_ids = self.decoder.generate(
                memory=memory,
                observed_ids=observed_ids,
                token_mask=token_mask,
                embedding=self.sct.dictionary,
                position=position
            )

            token_features = self.sct.dictionary(
                predicted_ids
            )

            ar_loss = memory.new_zeros(())

        # 7. Restore spatial ordering
        token_features = self._unorder(
            token_features
        )

        D, H, W = self.token_grid

        volume = self.to_volume(token_features)

        volume = volume.transpose(
            1, 2
        ).contiguous()

        volume = volume.reshape(
            batch_size,
            64,
            D,
            H,
            W
        )

        # 8. Volumetric reconstruction
        prediction = self.deconv1(volume)
        prediction = self.deconv2(prediction)
        prediction = self.deconv3(prediction)

        prediction = F.interpolate(
            prediction,
            size=target_occupancy.shape[2:],
            mode="trilinear",
            align_corners=False
        )

        self.forward_re_dict = {
            "pred": prediction,
            "target": target_occupancy,
            "vq_loss": vq_loss,
            "ar_loss": ar_loss,
            "token_mask": token_mask,
            "masked_token_count": token_mask.sum().detach()
        }

        return batch_dict