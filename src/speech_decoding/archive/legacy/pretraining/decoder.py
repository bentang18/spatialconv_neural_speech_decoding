"""Linear reconstruction decoder for masked prediction.

Collapse mode: Linear(2H, d_flat) → predicts pooled spatial representation.
Preserve mode: Linear(2H, 1) → predicts per-electrode values.
Discarded after pretraining.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionDecoder(nn.Module):
    """Shared linear decoder for masked frame reconstruction."""

    def __init__(self, input_dim: int, output_dim: int, mode: str = "collapse"):
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Project temporal model output back to input space.

        Args:
            h: (B, T, 2H) temporal model output.

        Returns:
            (B, T, output_dim) reconstructed frames.
        """
        return self.proj(h)
