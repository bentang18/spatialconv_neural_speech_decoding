"""Per-patient spatial conv read-in layer (E2 full model).

Exploits the 2D grid structure of uECOG arrays. Conv2d's weight sharing
matches rigid-array physics: uniform intra-array, variable inter-array.
AdaptiveAvgPool2d handles different grid sizes (8x16, 12x22).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpatialConvReadIn(nn.Module):
    """Per-patient: one instance per patient.

    Input:  (B, H, W, T) — grid-shaped HGA data
    Output: (B, out_dim, T) where out_dim = C * pool_h * pool_w (default 64)
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        C: int = 8,
        num_layers: int = 1,
        kernel_size: int = 3,
        pool_h: int = 2,
        pool_w: int = 4,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(1, C, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(C, C, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
            ]
        self.convs = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        self.out_dim = C * pool_h * pool_w
        # Store for reference
        self.grid_h = grid_h
        self.grid_w = grid_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, T)
        B, H, W, T = x.shape
        # Reshape to treat each time frame as a separate image
        x = x.permute(0, 3, 1, 2).reshape(B * T, 1, H, W)
        x = self.convs(x)  # (B*T, C, H, W)
        x = self.pool(x)   # (B*T, C, pool_h, pool_w)
        x = x.reshape(B, T, -1)  # (B, T, out_dim)
        return x.permute(0, 2, 1)  # (B, out_dim, T)
