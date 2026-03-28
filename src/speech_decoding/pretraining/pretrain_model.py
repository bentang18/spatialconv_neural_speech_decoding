"""Unified pretrain model with configurable spatial_mode.

Phase 1: collapse mode (reuses existing SpatialConvReadIn + SharedBackbone).
Phase 3: preserve and attend modes (Tasks 10-12).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from speech_decoding.models.spatial_conv import SpatialConvReadIn
from speech_decoding.models.backbone import SharedBackbone
from speech_decoding.pretraining.masking import SpanMasker, generate_span_mask
from speech_decoding.pretraining.decoder import ReconstructionDecoder


class PretrainModel(nn.Module):
    """Unified pretraining model for masked span prediction.

    Supports spatial_mode='collapse' (Phase 1), with 'preserve' and 'attend'
    added in Phase 3.

    Architecture:
        Input (B, H, W, T) → Conv2d read-in → spatial encoder → temporal model
        → masked span prediction → MSE on masked frames
    """

    def __init__(self, config: dict, grid_shape: tuple[int, int]):
        super().__init__()
        self.config = config
        self.spatial_mode = config.get("spatial_mode", "collapse")
        self.grid_shape = grid_shape

        sc = config.get("spatial_conv", {})
        d = config["d"]
        gru_hidden = config["gru_hidden"]
        temporal_stride = config["temporal_stride"]

        if self.spatial_mode == "collapse":
            self.readin = SpatialConvReadIn(
                grid_h=grid_shape[0], grid_w=grid_shape[1],
                C=sc.get("channels", 8),
                pool_h=sc.get("pool_h", 4), pool_w=sc.get("pool_w", 8),
            )
            backbone_input_dim = self.readin.out_dim

            self.backbone = SharedBackbone(
                D=backbone_input_dim, H=gru_hidden,
                temporal_stride=temporal_stride,
                gru_layers=config.get("gru_layers", 2),
                gru_dropout=0.0, feat_drop_max=0.0,
                gru_input_dim=d,
            )

            self.decoder = ReconstructionDecoder(
                input_dim=gru_hidden * 2, output_dim=backbone_input_dim, mode="collapse",
            )

            # Masker operates in the GRU input space (d-dim after Conv1d)
            self.masker = SpanMasker(d=d)
            self._target_dim = backbone_input_dim

        else:
            raise NotImplementedError(f"spatial_mode={self.spatial_mode} not yet implemented")

        self.mask_ratio = tuple(config.get("mask_ratio", [0.4, 0.6]))
        self.mask_spans = tuple(config.get("mask_spans", [3, 6]))

    def _spatial_encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, T) → (B, D, T) spatial features (collapse mode)."""
        if self.spatial_mode == "collapse":
            return self.readin(x)
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass without masking (for downstream fine-tuning).

        Args:
            x: (B, H, W, T) raw grid input at 200Hz.
        Returns:
            (B, T', 2H) temporal features.
        """
        spatial = self._spatial_encode(x)
        features = self.backbone(spatial)
        return features

    def forward(self, x: torch.Tensor, compute_loss: bool = True) -> dict[str, torch.Tensor]:
        """Forward pass with masked span prediction.

        Masking is inserted BETWEEN Conv1d and GRU by decomposing the backbone
        into: layernorm → temporal_conv → MASK → gru. This forces the model
        to predict masked frames from bidirectional context.

        backbone._feature_dropout and _time_mask are skipped (feat_drop_max=0.0
        in pretraining config).
        """
        spatial = self._spatial_encode(x)  # (B, d_flat, T)

        if self.spatial_mode == "collapse":
            stride = self.config["temporal_stride"]
            T_raw = spatial.shape[2]
            T_strided = T_raw // stride

            # Compute reconstruction targets: avg-pool spatial features to strided resolution
            targets = F.avg_pool1d(spatial, kernel_size=stride, stride=stride)  # (B, d_flat, T')
            targets = targets.permute(0, 2, 1)  # (B, T', d_flat)

            # Generate temporal mask at strided resolution
            rng = np.random.RandomState()
            mask = generate_span_mask(
                T=T_strided, mask_ratio=self.mask_ratio,
                n_spans=self.mask_spans, rng=rng,
            )
            mask_tensor = torch.from_numpy(mask).to(x.device)

            # Decompose backbone: layernorm → temporal_conv → MASK → gru
            # backbone._feature_dropout is skipped (feat_drop_max=0.0, and no-op at eval)
            # backbone._time_mask is skipped (no-op at eval; pretraining uses its own masking)
            h = self.backbone.layernorm(spatial.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.backbone.temporal_conv(h)   # (B, d, T')
            h = h.permute(0, 2, 1)              # (B, T', d)

            # Apply span mask (replaces masked frames with learnable [MASK] token)
            h_masked, mask_tensor = self.masker(h, mask_tensor)

            # Run GRU on masked sequence
            gru_out, _ = self.backbone.gru(h_masked)  # (B, T', 2H)

            # Decode all positions (loss computed only on masked ones)
            predictions = self.decoder(gru_out)  # (B, T', d_flat)

        result = {
            "predictions": predictions,
            "targets": targets,
            "mask": mask_tensor,
        }

        if compute_loss:
            pred_masked = predictions[:, mask_tensor]
            tgt_masked = targets[:, mask_tensor]
            loss = F.mse_loss(pred_masked, tgt_masked)
            result["loss"] = loss

        return result
