"""LeWM-style pretraining model for uECOG data.

Adapts the LeWorldModel framework (Maes et al., 2026) to neural time series:
- Encoder: SpatialConv -> LayerNorm -> Conv1d -> BiGRU (matches PretrainModel)
- Projector: Linear -> BatchNorm (maps to SIGReg-compatible space)
- Predictor: MLP that predicts next temporal embedding from current
- Loss: MSE(z_hat_{t+1}, z_{t+1}) + lambda * SIGReg(Z)

Key adaptations from LeWM (vision/RL) to uECOG:
- No action conditioning (speech is passive observation)
- Conv2d spatial encoder instead of ViT
- BiGRU between Conv1d and projector — critical for temporal modeling,
  and ensures GRU weights transfer to PretrainModel for Stage 3
- Smaller predictor (MLP, not transformer) — matches our ~120K param regime
- SIGReg pooled across batch AND time (batch size 8 too small for per-step)

After pretraining, encoder weights (including BiGRU) transfer to PretrainModel
for Stage 3 fine-tuning (predictor and projector are discarded).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from speech_decoding.models.spatial_conv import SpatialConvReadIn
from speech_decoding.pretraining.sigreg import sigreg


class LeWMModel(nn.Module):
    """LeWM-style pretraining for uECOG temporal prediction.

    Architecture:
        Encoder: Conv2d(spatial) -> LayerNorm -> Conv1d(temporal, stride) -> BiGRU -> z_t
        Projector: Linear -> BatchNorm (SIGReg-compatible projection)
        Predictor: Linear -> GELU -> Linear (next-embedding prediction)

    Training objective:
        L = MSE(z_hat_{t+1}, z_{t+1}) + lambda * SIGReg(Z)

    No stop-gradient, no EMA, fully end-to-end.
    """

    def __init__(self, config: dict, grid_shape: tuple[int, int]):
        super().__init__()
        self.config = config
        self.grid_shape = grid_shape
        H, W = grid_shape

        # --- Encoder: matches PretrainModel's readin + backbone encoder ---
        sc = config.get("spatial_conv", {})
        channels = sc.get("channels", 8)
        pool_h = sc.get("pool_h", 4)
        pool_w = sc.get("pool_w", 8)
        num_layers = sc.get("num_layers", 1)

        self.readin = SpatialConvReadIn(
            grid_h=H, grid_w=W,
            C=channels, pool_h=pool_h, pool_w=pool_w,
            num_layers=num_layers,
        )
        d_flat = self.readin.out_dim  # C * pool_h * pool_w

        d = config.get("d", 64)
        stride = config.get("temporal_stride", 10)
        self.d = d
        self.stride = stride

        # LayerNorm + Conv1d + BiGRU: matches SharedBackbone exactly
        gru_hidden = config.get("gru_hidden", 32)
        gru_layers = config.get("gru_layers", 2)
        self.gru_hidden = gru_hidden

        self.ln = nn.LayerNorm(d_flat)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_flat, d, kernel_size=stride, stride=stride),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            d, gru_hidden, num_layers=gru_layers, batch_first=True,
            bidirectional=True, dropout=0.0,
        )

        # BiGRU output dim = 2 * gru_hidden
        gru_out_dim = gru_hidden * 2

        # --- Projector: maps to SIGReg-compatible space ---
        # Needed because LayerNorm before Conv1d can interfere with
        # SIGReg's Gaussian enforcement (same issue as ViT in LeWM).
        self.projector = nn.Sequential(
            nn.Linear(gru_out_dim, d),
            nn.BatchNorm1d(d),
        )

        # --- Predictor: predicts next embedding from current ---
        predictor_hidden = config.get("predictor_hidden", d * 4)
        self.predictor = nn.Sequential(
            nn.Linear(d, predictor_hidden),
            nn.GELU(),
            nn.Linear(predictor_hidden, d),
        )

        # --- SIGReg config ---
        self.sigreg_lambda = config.get("sigreg_lambda", 0.1)
        self.sigreg_M = config.get("sigreg_M", 1024)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, H, W, T) -> (B, T', 2*gru_hidden).

        Returns BiGRU hidden states (before projection).
        Used for downstream feature extraction.
        """
        # Spatial read-in: (B, H, W, T) -> (B, d_flat, T)
        spatial = self.readin(x)

        # LayerNorm over feature dim
        spatial = self.ln(spatial.permute(0, 2, 1)).permute(0, 2, 1)

        # Temporal conv with stride: (B, d_flat, T) -> (B, d, T')
        temporal = self.temporal_conv(spatial)

        # BiGRU: (B, T', d) -> (B, T', 2*gru_hidden)
        gru_out, _ = self.gru(temporal.permute(0, 2, 1))

        return gru_out  # (B, T', 2*gru_hidden)

    def forward(
        self, x: torch.Tensor, compute_loss: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with LeWM training objective.

        Args:
            x: (B, H, W, T) input grid data.
            compute_loss: whether to compute prediction + SIGReg loss.

        Returns:
            dict with keys:
                loss: total loss (pred + lambda*sigreg), if compute_loss
                pred_loss: MSE prediction loss (scalar), if compute_loss
                sigreg_loss: SIGReg value (scalar), if compute_loss
                embeddings: (B, T', d) projected embeddings
        """
        z = self.encode(x)  # (B, T', 2*gru_hidden)
        B, T_prime, gru_out_dim = z.shape

        # Project for SIGReg-compatible space
        z_proj = self.projector(z.reshape(B * T_prime, gru_out_dim))
        z_proj = z_proj.reshape(B, T_prime, self.d)

        d = self.d
        if not compute_loss:
            return {"embeddings": z_proj}

        # --- Prediction loss (no stop-gradient, end-to-end) ---
        z_input = z_proj[:, :-1].reshape(-1, d)  # (B*(T'-1), d)
        z_pred = self.predictor(z_input).reshape(B, T_prime - 1, d)
        z_target = z_proj[:, 1:]  # gradients flow through encoder

        pred_loss = F.mse_loss(z_pred, z_target)

        # --- SIGReg loss (pooled across batch AND time) ---
        Z_flat = z_proj.reshape(B * T_prime, d)
        sigreg_loss = sigreg(Z_flat, M=self.sigreg_M)

        loss = pred_loss + self.sigreg_lambda * sigreg_loss

        return {
            "loss": loss,
            "pred_loss": pred_loss.item(),
            "sigreg_loss": sigreg_loss.item(),
            "embeddings": z_proj,
        }

    def transfer_encoder_weights(self, pretrain_model: nn.Module) -> None:
        """Transfer encoder weights to a PretrainModel for fine-tuning.

        Copies readin, LayerNorm, temporal_conv, and BiGRU weights.
        Projector and predictor are discarded.

        Args:
            pretrain_model: PretrainModel instance to receive weights.
        """
        pretrain_model.readin.load_state_dict(self.readin.state_dict())
        pretrain_model.backbone.layernorm.load_state_dict(self.ln.state_dict())
        pretrain_model.backbone.temporal_conv.load_state_dict(
            self.temporal_conv.state_dict()
        )
        pretrain_model.backbone.gru.load_state_dict(self.gru.state_dict())
