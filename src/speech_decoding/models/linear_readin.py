"""Per-patient linear read-in layer (E1 field standard).

Projects zero-padded flattened channels to shared feature dimension.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearReadIn(nn.Module):
    """Per-patient: one instance per patient.

    Input:  (B, H, W, T) — grid-shaped HGA data (same interface as SpatialConvReadIn)
         or (B, D_padded, T) — pre-flattened channels
    Output: (B, D_shared, T)
    """

    def __init__(self, d_padded: int = 208, d_shared: int = 64):
        super().__init__()
        self.d_padded = d_padded
        self.linear = nn.Linear(d_padded, d_shared)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            # (B, H, W, T) → flatten spatial dims → (B, H*W, T)
            B, H, W, T = x.shape
            x = x.reshape(B, H * W, T)
        # (B, D_padded, T) → permute → (B, T, D_padded) → linear → (B, T, D_shared) → permute
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
