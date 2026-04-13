"""VICReg contrastive SSL pretraining for uECOG data.

Two augmented views of the same trial -> same encoder -> projector -> VICReg loss.
Augmentations preserve phoneme content (time shift, amp scale, channel dropout,
noise, temporal stretch) -- the model must learn features invariant to these.

VICReg loss = lambda_inv * invariance + lambda_var * variance + lambda_cov * covariance
- Invariance: MSE between projected embeddings of two views
- Variance: hinge loss on per-dim std (prevent collapse to a point)
- Covariance: decorrelate dimensions (prevent collapse to a line/plane)

After pretraining: transfer encoder weights to PretrainModel. Projector discarded.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from speech_decoding.data.augmentation import augment_from_config
from speech_decoding.models.spatial_conv import SpatialConvReadIn


DEFAULT_SSL_AUGMENTATION = {
    "time_shift_frames": 30,
    "amp_scale_std": 0.3,
    "channel_dropout_max": 0.2,
    "noise_frac": 0.05,
    "temporal_stretch": True,
    "temporal_stretch_max_rate": 0.15,
}


def off_diagonal(M: torch.Tensor) -> torch.Tensor:
    """Return flattened off-diagonal elements of a square matrix."""
    n = M.shape[0]
    return M.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambda_inv: float = 25.0,
    lambda_var: float = 25.0,
    lambda_cov: float = 1.0,
) -> dict[str, torch.Tensor | float]:
    """VICReg loss on two sets of projected embeddings.

    Args:
        z1, z2: (N, D) projected embeddings from two views.
        lambda_inv/var/cov: loss weights.
    Returns:
        dict with total loss, inv_loss, var_loss, cov_loss (all scalars).
    """
    N, D = z1.shape

    # Invariance: MSE between views
    inv_loss = F.mse_loss(z1, z2)

    # Variance: std of each dimension should be >= 1
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = (torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))) / 2

    # Covariance: off-diagonal elements of covariance matrix should be 0
    z1_c = z1 - z1.mean(dim=0)
    z2_c = z2 - z2.mean(dim=0)
    cov1 = (z1_c.T @ z1_c) / max(N - 1, 1)
    cov2 = (z2_c.T @ z2_c) / max(N - 1, 1)
    # Zero out diagonal, sum squares of off-diagonal
    cov_loss = (
        off_diagonal(cov1).pow(2).sum() / D
        + off_diagonal(cov2).pow(2).sum() / D
    ) / 2

    total = lambda_inv * inv_loss + lambda_var * var_loss + lambda_cov * cov_loss
    return {
        "loss": total,
        "inv_loss": inv_loss.item(),
        "var_loss": var_loss.item(),
        "cov_loss": cov_loss.item(),
    }


class VICRegModel(nn.Module):
    """VICReg contrastive SSL pretraining for uECOG temporal features.

    Architecture:
        Encoder: Conv2d(spatial) -> LayerNorm -> Conv1d(temporal, stride) -> BiGRU -> z_t
        Projector: Linear -> BN1d -> ReLU -> Linear -> BN1d -> ReLU -> Linear (3-layer MLP)

    Training objective:
        Two augmented views of same trial -> encoder -> mean-pool -> projector
        L = lambda_inv * MSE(z1, z2) + lambda_var * var_hinge + lambda_cov * cov_off_diag

    No stop-gradient, no EMA, fully end-to-end.
    Augmentation creates the learning signal; variance + covariance terms prevent collapse.
    """

    def __init__(self, config: dict, grid_shape: tuple[int, int]):
        super().__init__()
        self.config = config
        self.grid_shape = grid_shape
        H, W = grid_shape

        # --- Encoder: matches PretrainModel/LeWMModel/JEPAModel structure ---
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

        # --- Projector: 3-layer MLP for VICReg ---
        proj_dim = config.get("vicreg_proj_dim", 256)
        self.projector = nn.Sequential(
            nn.Linear(gru_out_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

        # --- VICReg loss weights ---
        self.lambda_inv = config.get("vicreg_lambda_inv", 25.0)
        self.lambda_var = config.get("vicreg_lambda_var", 25.0)
        self.lambda_cov = config.get("vicreg_lambda_cov", 1.0)

        # --- Augmentation config ---
        self.aug_config = dict(DEFAULT_SSL_AUGMENTATION)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, H, W, T) -> (B, T', 2*gru_hidden).

        Returns BiGRU hidden states (before projection).
        Used for downstream feature extraction. No augmentation applied.
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

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        """Project encoder output through 3-layer MLP.

        Args:
            z: (B, T', 2*gru_hidden) encoder output.
        Returns:
            (B, proj_dim) mean-pooled projected embeddings.
        """
        # Mean-pool over temporal dimension
        z_pooled = z.mean(dim=1)  # (B, 2*gru_hidden)
        # Project through MLP (BN1d expects (B, D))
        return self.projector(z_pooled)  # (B, proj_dim)

    def forward(
        self, x: torch.Tensor, compute_loss: bool = True,
    ) -> dict[str, torch.Tensor | float]:
        """Forward pass with VICReg training objective.

        If compute_loss: create two augmented views, encode both, mean-pool,
        project, compute VICReg loss.
        If not compute_loss: encode raw input, mean-pool, return embeddings.

        Args:
            x: (B, H, W, T) input grid data.
            compute_loss: whether to compute VICReg loss.

        Returns:
            dict with keys:
                loss: total VICReg loss (scalar), if compute_loss
                inv_loss: invariance loss (float), if compute_loss
                var_loss: variance loss (float), if compute_loss
                cov_loss: covariance loss (float), if compute_loss
                embeddings: (B, 2*gru_hidden) mean-pooled encoded x, if not compute_loss
        """
        if not compute_loss:
            z = self.encode(x)  # (B, T', 2*gru_hidden)
            return {"embeddings": z.mean(dim=1)}

        # Create two augmented views
        x1 = augment_from_config(x, self.aug_config, training=True)
        x2 = augment_from_config(x, self.aug_config, training=True)

        # Encode both views
        z1 = self.encode(x1)  # (B, T', 2*gru_hidden)
        z2 = self.encode(x2)  # (B, T', 2*gru_hidden)

        # Mean-pool + project
        p1 = self._project(z1)  # (B, proj_dim)
        p2 = self._project(z2)  # (B, proj_dim)

        # Compute VICReg loss
        result = vicreg_loss(
            p1, p2,
            lambda_inv=self.lambda_inv,
            lambda_var=self.lambda_var,
            lambda_cov=self.lambda_cov,
        )

        return result

    def transfer_encoder_weights(self, pretrain_model: nn.Module) -> None:
        """Transfer encoder weights to a PretrainModel for fine-tuning.

        Copies readin, LayerNorm, temporal_conv, and BiGRU weights.
        Projector is discarded.

        Args:
            pretrain_model: PretrainModel instance to receive weights.
        """
        pretrain_model.readin.load_state_dict(self.readin.state_dict())
        pretrain_model.backbone.layernorm.load_state_dict(self.ln.state_dict())
        pretrain_model.backbone.temporal_conv.load_state_dict(
            self.temporal_conv.state_dict()
        )
        pretrain_model.backbone.gru.load_state_dict(self.gru.state_dict())
