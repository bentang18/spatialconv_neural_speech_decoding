"""Flat CTC head: Linear(2H, num_classes) → log_softmax.

Standard projection from backbone hidden states to phoneme log-probabilities.
Used in E1 (field standard) and as the baseline for articulatory head comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlatCTCHead(nn.Module):
    """Flat CTC projection head.

    Input:  (B, T, 2H)
    Output: (B, T, num_classes) log probabilities
    """

    def __init__(self, input_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.linear(h), dim=-1)
