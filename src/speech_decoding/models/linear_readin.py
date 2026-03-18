"""Per-patient linear read-in layer (E1 field standard).

Projects zero-padded flattened channels to shared feature dimension.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearReadIn(nn.Module):
    """Per-patient: one instance per patient.

    Input:  (B, D_padded, T) — flattened, zero-padded channels
    Output: (B, D_shared, T)
    """

    def __init__(self, d_padded: int = 208, d_shared: int = 64):
        super().__init__()
        self.linear = nn.Linear(d_padded, d_shared)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, D_padded, T) → permute → (B, T, D_padded) → linear → (B, T, D_shared) → permute
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
