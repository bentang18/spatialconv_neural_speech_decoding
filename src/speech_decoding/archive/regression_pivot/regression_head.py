"""Regression head for speech embedding prediction."""
from __future__ import annotations

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """Linear projection from backbone states to framewise embeddings."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)
