"""Shared backbone: LayerNorm → Conv1d temporal downsampling → BiGRU.

Shared across all patients — the common representational space.
Includes feature dropout (spatial) and smooth time masking as
training-time augmentations.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SharedBackbone(nn.Module):
    """Shared backbone with temporal downsampling and bidirectional GRU.

    Input:  (B, D, T) from read-in
    Output: (B, T//K, 2H) bidirectional GRU hidden states
    """

    def __init__(
        self,
        D: int = 64,
        H: int = 64,
        temporal_stride: int = 5,
        gru_layers: int = 2,
        gru_dropout: float = 0.2,
        feat_drop_max: float = 0.3,
        time_mask_min: int = 2,
        time_mask_max: int = 4,
    ):
        super().__init__()
        self.layernorm = nn.LayerNorm(D)
        self.feat_drop_max = feat_drop_max
        self.time_mask_min = time_mask_min
        self.time_mask_max = time_mask_max

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(D, D, kernel_size=temporal_stride, stride=temporal_stride),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            D, H, num_layers=gru_layers, batch_first=True,
            bidirectional=True, dropout=gru_dropout if gru_layers > 1 else 0.0,
        )

    def _feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial dropout: zero entire feature channels, p ~ U[0, max]."""
        if not self.training:
            return x
        p = torch.rand(1).item() * self.feat_drop_max
        mask = (torch.rand(x.shape[-1], device=x.device) > p).float()
        return x * mask / (1 - p + 1e-8)

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Zero 2-4 contiguous frames with cosine taper."""
        if not self.training:
            return x
        T = x.shape[1]
        mask_len = torch.randint(self.time_mask_min, self.time_mask_max + 1, (1,)).item()
        start = torch.randint(0, max(T - mask_len, 1), (1,)).item()
        taper = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, mask_len, device=x.device)))
        x = x.clone()
        x[:, start:start + mask_len, :] *= (1 - taper).unsqueeze(-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T) from read-in
        x = self.layernorm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self._feature_dropout(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.temporal_conv(x)  # (B, D, T//K)
        x = x.permute(0, 2, 1)    # (B, T//K, D)
        x = self._time_mask(x)
        h, _ = self.gru(x)         # (B, T//K, 2H)
        return h
