"""BYOL pretraining model for uECOG data.

Bootstrap Your Own Latent: predicts augmented view representations
using an EMA target encoder. No negative pairs, no covariance terms.

- Online encoder + projector + predictor: gets gradients
- Target encoder + projector (EMA copy): no gradients, stop-grad targets
- Two augmented views of same trial → symmetrized L2-normalized MSE loss
- Collapse prevented by predictor asymmetry + EMA (architectural, not statistical)

Key advantage over VICReg: works at any batch size (no covariance estimation).
Key advantage over JEPA: augmentation-based invariance directly teaches
phoneme-preserving features (not temporal smoothness).

After pretraining: transfer online encoder weights (readin, LN, Conv1d, BiGRU)
to PretrainModel. Projector + predictor + target encoder discarded.
"""
from __future__ import annotations

import math
from copy import deepcopy

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


def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """BYOL regression loss: MSE of L2-normalized representations.

    Equivalent to 2 - 2 * cosine_similarity(p, z).

    Args:
        p: (N, D) predictor output (online path).
        z: (N, D) projection output (target path, detached).
    Returns:
        Scalar loss.
    """
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()


class BYOLModel(nn.Module):
    """BYOL pretraining for uECOG temporal representation learning.

    Architecture:
        Online: Conv2d(spatial) -> LN -> Conv1d(temporal) -> BiGRU -> projector -> predictor
        Target: EMA copy of online encoder + projector (no predictor, no gradients)

    Training objective:
        view1 = augment(x), view2 = augment(x)
        p1 = predictor(projector(encode_online(view1)))
        z2 = projector_target(encode_target(view2))  [stop_grad]
        L = byol_loss(p1, z2) + byol_loss(p2, z1)  [symmetrized]

    Collapse prevention:
        The predictor asymmetry (online has predictor, target doesn't) combined
        with EMA makes collapse an unstable fixed point. Proven in Grill et al. 2020.
    """

    def __init__(self, config: dict, grid_shape: tuple[int, int]):
        super().__init__()
        self.config = config
        self.grid_shape = grid_shape
        H, W = grid_shape

        # --- Online encoder ---
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
        d_flat = self.readin.out_dim

        d = config.get("d", 64)
        stride = config.get("temporal_stride", 10)
        self.d = d
        self.stride = stride

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

        gru_out_dim = gru_hidden * 2

        # --- Online projector: 2-layer MLP (BYOL convention) ---
        proj_dim = config.get("byol_proj_dim", 256)
        self.proj_dim = proj_dim
        self.projector = nn.Sequential(
            nn.Linear(gru_out_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

        # --- Online predictor: 2-layer MLP (online only — asymmetry prevents collapse) ---
        pred_hidden = config.get("byol_pred_hidden", 256)
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim),
        )

        # --- Target encoder + projector: deep copy, no gradients ---
        self.target_readin = deepcopy(self.readin)
        self.target_ln = deepcopy(self.ln)
        self.target_temporal_conv = deepcopy(self.temporal_conv)
        self.target_gru = deepcopy(self.gru)
        self.target_projector = deepcopy(self.projector)
        for p in self._target_params():
            p.requires_grad = False

        # --- EMA state (same schedule as JEPA) ---
        self.ema_step = 0
        self.ema_total_steps = config.get("ema_total_steps", 5000)
        self.ema_momentum = config.get("ema_momentum", 0.996)
        self.ema_momentum_end = config.get("ema_momentum_end", 1.0)

        # --- Augmentation config ---
        self.aug_config = dict(DEFAULT_SSL_AUGMENTATION)

    def _target_params(self):
        """Iterator over all target parameters (encoder + projector)."""
        for module in (self.target_readin, self.target_ln,
                       self.target_temporal_conv, self.target_gru,
                       self.target_projector):
            yield from module.parameters()

    def _online_target_pairs(self):
        """Yield (online_param, target_param) pairs for EMA update."""
        for online_mod, target_mod in (
            (self.readin, self.target_readin),
            (self.ln, self.target_ln),
            (self.temporal_conv, self.target_temporal_conv),
            (self.gru, self.target_gru),
            (self.projector, self.target_projector),
        ):
            for op, tp in zip(online_mod.parameters(), target_mod.parameters()):
                yield op, tp

    def _get_ema_momentum(self) -> float:
        """Cosine-scheduled EMA momentum: ramps from start to end."""
        start = self.ema_momentum
        end = self.ema_momentum_end
        if self.ema_total_steps <= 0:
            return end
        progress = min(self.ema_step / self.ema_total_steps, 1.0)
        tau = end - (end - start) * (math.cos(math.pi * progress) + 1) / 2
        return tau

    @torch.no_grad()
    def ema_update(self):
        """Update target encoder + projector via exponential moving average."""
        tau = self._get_ema_momentum()
        for online_p, target_p in self._online_target_pairs():
            target_p.data.mul_(tau).add_(online_p.data, alpha=1 - tau)
        self.ema_step += 1

    def _encode_online(self, x: torch.Tensor) -> torch.Tensor:
        """Online encoder: (B, H, W, T) -> (B, T', 2*gru_hidden)."""
        spatial = self.readin(x)
        h = self.ln(spatial.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.temporal_conv(h)
        h = h.permute(0, 2, 1)
        gru_out, _ = self.gru(h)
        return gru_out

    @torch.no_grad()
    def _encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """Target encoder: (B, H, W, T) -> (B, T', 2*gru_hidden). No gradients."""
        spatial = self.target_readin(x)
        h = self.target_ln(spatial.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.target_temporal_conv(h)
        h = h.permute(0, 2, 1)
        gru_out, _ = self.target_gru(h)
        return gru_out

    def _project_online(self, z: torch.Tensor) -> torch.Tensor:
        """Online projector: mean-pool temporal then MLP."""
        z_pooled = z.mean(dim=1)  # (B, 2H)
        return self.projector(z_pooled)  # (B, proj_dim)

    @torch.no_grad()
    def _project_target(self, z: torch.Tensor) -> torch.Tensor:
        """Target projector: mean-pool temporal then MLP (EMA, no grad)."""
        z_pooled = z.mean(dim=1)  # (B, 2H)
        return self.target_projector(z_pooled)  # (B, proj_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full online encoder (no augmentation, no projector) for downstream use.

        Args:
            x: (B, H, W, T) raw grid input.
        Returns:
            (B, T', 2*gru_hidden) features.
        """
        return self._encode_online(x)

    def forward(
        self, x: torch.Tensor, compute_loss: bool = True,
    ) -> dict[str, torch.Tensor | float]:
        """Forward pass with BYOL training objective.

        Two augmented views → online encoder+projector+predictor and
        target encoder+projector (stop_grad). Symmetrized cosine loss.

        Args:
            x: (B, H, W, T) input grid data.
            compute_loss: whether to compute BYOL loss.

        Returns:
            dict with keys:
                loss: symmetrized BYOL loss (scalar), if compute_loss
                loss_12: loss(view1→view2) component (float)
                loss_21: loss(view2→view1) component (float)
                embeddings: (B, 2*gru_hidden) mean-pooled features, if not compute_loss
        """
        if not compute_loss:
            z = self.encode(x)
            return {"embeddings": z.mean(dim=1)}

        # Create two augmented views
        x1 = augment_from_config(x, self.aug_config, training=True)
        x2 = augment_from_config(x, self.aug_config, training=True)

        # Online path: encode → project → predict
        z1_online = self._encode_online(x1)   # (B, T', 2H)
        z2_online = self._encode_online(x2)   # (B, T', 2H)
        p1 = self.predictor(self._project_online(z1_online))  # (B, proj_dim)
        p2 = self.predictor(self._project_online(z2_online))  # (B, proj_dim)

        # Target path: encode → project (stop_grad via @torch.no_grad)
        t1 = self._project_target(self._encode_target(x1))  # (B, proj_dim)
        t2 = self._project_target(self._encode_target(x2))  # (B, proj_dim)

        # Symmetrized BYOL loss
        loss_12 = byol_loss(p1, t2)
        loss_21 = byol_loss(p2, t1)
        loss = (loss_12 + loss_21) / 2

        return {
            "loss": loss,
            "loss_12": loss_12.item(),
            "loss_21": loss_21.item(),
        }

    def transfer_encoder_weights(self, pretrain_model: nn.Module) -> None:
        """Transfer online encoder weights to a PretrainModel for fine-tuning.

        Uses the online encoder (not target) because BYOL's online encoder
        is the one optimized by gradients. The target encoder is a smoothed
        copy used only for generating regression targets.

        Projector, predictor, and target encoder/projector are discarded.

        Args:
            pretrain_model: PretrainModel instance to receive weights.
        """
        pretrain_model.readin.load_state_dict(self.readin.state_dict())
        pretrain_model.backbone.layernorm.load_state_dict(self.ln.state_dict())
        pretrain_model.backbone.temporal_conv.load_state_dict(
            self.temporal_conv.state_dict()
        )
        pretrain_model.backbone.gru.load_state_dict(self.gru.state_dict())
