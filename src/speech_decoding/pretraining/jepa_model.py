"""JEPA pretraining model for uECOG data.

Joint Embedding Predictive Architecture: predicts in representation space
using an EMA target encoder, not in observation space.

- Online encoder: sees MASKED input, gets gradients
- Target encoder: EMA copy, sees FULL input, no gradients
- Predictor MLP: maps online representations to target space
- Loss: MSE on masked positions only (predictions vs stop_grad targets)

The EMA + masking asymmetry prevents representation collapse without
extra loss terms (proven in I-JEPA, V-JEPA).

After pretraining: transfer online encoder weights (readin, LN, Conv1d, BiGRU)
to PretrainModel. Predictor + target encoder are discarded.
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from speech_decoding.models.spatial_conv import SpatialConvReadIn
from speech_decoding.pretraining.masking import SpanMasker, generate_span_mask


class JEPAModel(nn.Module):
    """JEPA pretraining for uECOG temporal representation learning.

    Architecture:
        Online encoder: Conv2d(spatial) -> LayerNorm -> Conv1d(temporal) -> [MASK] -> BiGRU
        Target encoder: EMA copy of online (no mask, no gradients)
        Predictor: Linear -> GELU -> Linear (maps online -> target space at masked positions)

    Training objective:
        L = MSE(predictor(online[masked]), stop_grad(target[masked]))

    EMA schedule:
        tau = end - (end - start) * (cos(pi * step / total) + 1) / 2
        Ramps from ema_momentum (0.996) to ema_momentum_end (1.0) over training.
    """

    def __init__(self, config: dict, grid_shape: tuple[int, int]):
        super().__init__()
        self.config = config
        self.grid_shape = grid_shape
        H, W = grid_shape

        # --- Online encoder: matches PretrainModel/LeWMModel structure ---
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

        # --- Target encoder: deep copy, no gradients ---
        self.target_readin = deepcopy(self.readin)
        self.target_ln = deepcopy(self.ln)
        self.target_temporal_conv = deepcopy(self.temporal_conv)
        self.target_gru = deepcopy(self.gru)
        for p in self._target_params():
            p.requires_grad = False

        # --- Predictor MLP: maps online -> target representation space ---
        predictor_hidden = config.get("predictor_hidden", 256)
        self.predictor = nn.Sequential(
            nn.Linear(gru_out_dim, predictor_hidden),
            nn.GELU(),
            nn.Linear(predictor_hidden, gru_out_dim),
        )

        # --- Masker: span masking with learnable [MASK] token ---
        self.masker = SpanMasker(d=d)

        # --- Masking config ---
        self.mask_ratio = tuple(config.get("mask_ratio", [0.4, 0.6]))
        self.mask_spans = tuple(config.get("mask_spans", [3, 6]))

        # --- EMA state ---
        self.ema_step = 0
        self.ema_total_steps = config.get("ema_total_steps", 5000)
        self.ema_momentum = config.get("ema_momentum", 0.996)
        self.ema_momentum_end = config.get("ema_momentum_end", 1.0)

    def _target_params(self):
        """Iterator over all target encoder parameters."""
        for module in (self.target_readin, self.target_ln,
                       self.target_temporal_conv, self.target_gru):
            yield from module.parameters()

    def _online_target_pairs(self):
        """Yield (online_param, target_param) pairs for EMA update."""
        for online_mod, target_mod in (
            (self.readin, self.target_readin),
            (self.ln, self.target_ln),
            (self.temporal_conv, self.target_temporal_conv),
            (self.gru, self.target_gru),
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
        """Update target encoder via exponential moving average of online encoder."""
        tau = self._get_ema_momentum()
        for online_p, target_p in self._online_target_pairs():
            target_p.data.mul_(tau).add_(online_p.data, alpha=1 - tau)
        self.ema_step += 1

    def _encode_online(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """Run online encoder. If mask given, apply masking between Conv1d and GRU.

        Args:
            x: (B, H, W, T) input.
            mask: (T',) boolean mask or None.

        Returns:
            (B, T', 2*gru_hidden) GRU output.
        """
        spatial = self.readin(x)  # (B, d_flat, T)
        h = self.ln(spatial.permute(0, 2, 1)).permute(0, 2, 1)  # (B, d_flat, T)
        h = self.temporal_conv(h)  # (B, d, T')
        h = h.permute(0, 2, 1)    # (B, T', d)

        if mask is not None:
            h, _ = self.masker(h, mask)

        gru_out, _ = self.gru(h)  # (B, T', 2H)
        return gru_out

    @torch.no_grad()
    def _encode_target(self, x: torch.Tensor):
        """Run target encoder (full input, no mask, no gradients).

        Args:
            x: (B, H, W, T) input.

        Returns:
            (B, T', 2*gru_hidden) GRU output.
        """
        spatial = self.target_readin(x)
        h = self.target_ln(spatial.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.target_temporal_conv(h)
        h = h.permute(0, 2, 1)
        gru_out, _ = self.target_gru(h)
        return gru_out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full online encoder (no mask, no predictor) for downstream use.

        Args:
            x: (B, H, W, T) raw grid input.
        Returns:
            (B, T', 2*gru_hidden) features.
        """
        return self._encode_online(x, mask=None)

    def forward(
        self, x: torch.Tensor, compute_loss: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with JEPA training objective.

        Online encoder sees masked input, target encoder sees full input.
        Loss = MSE on masked positions between predictor(online) and stop_grad(target).

        Args:
            x: (B, H, W, T) input grid data.
            compute_loss: whether to compute JEPA loss.

        Returns:
            dict with keys:
                loss: MSE on masked positions (scalar), if compute_loss
                predictions: (B, T', 2H) predictor output
                targets: (B, T', 2H) target encoder output (detached)
                mask: (T',) boolean mask
        """
        T_raw = x.shape[-1]
        T_prime = T_raw // self.stride

        # Generate mask
        rng = np.random.RandomState()
        mask = generate_span_mask(
            T=T_prime, mask_ratio=self.mask_ratio,
            n_spans=self.mask_spans, rng=rng,
        )
        mask_tensor = torch.from_numpy(mask).to(x.device)

        # Online encoder with masking
        online_out = self._encode_online(x, mask=mask_tensor)  # (B, T', 2H)

        # Predictor on online representations
        B, T_p, D = online_out.shape
        predictions = self.predictor(online_out.reshape(B * T_p, D))
        predictions = predictions.reshape(B, T_p, D)

        # Target encoder (full input, no gradients)
        targets = self._encode_target(x)  # (B, T', 2H)

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

    def transfer_encoder_weights(self, pretrain_model: nn.Module) -> None:
        """Transfer TARGET encoder weights to a PretrainModel for fine-tuning.

        Uses the target (EMA) encoder, not the online encoder, because:
        1. Target encoder always sees full (unmasked) input — matches Stage 3 usage
        2. EMA smoothing produces more stable representations
        3. This follows I-JEPA/V-JEPA convention for downstream evaluation

        Predictor, online encoder, and masker are discarded.

        Args:
            pretrain_model: PretrainModel instance to receive weights.
        """
        pretrain_model.readin.load_state_dict(self.target_readin.state_dict())
        pretrain_model.backbone.layernorm.load_state_dict(
            self.target_ln.state_dict()
        )
        pretrain_model.backbone.temporal_conv.load_state_dict(
            self.target_temporal_conv.state_dict()
        )
        pretrain_model.backbone.gru.load_state_dict(self.target_gru.state_dict())
