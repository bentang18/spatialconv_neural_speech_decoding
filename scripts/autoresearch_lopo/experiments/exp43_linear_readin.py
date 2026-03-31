#!/usr/bin/env python3
"""Exp 43: Linear read-in (NO Conv2d — control experiment).

Flatten grid -> Linear projection. Tests whether Conv2d spatial processing
actually helps or if the model is just doing dimensionality reduction.
If Linear matches Conv2d, spatial structure isn't being exploited.

Baseline (exp33): Conv2d(1,8,k=3) -> Pool(4,8) -> 256-dim, 80 params
Change: Flatten(H*W) -> Linear(H*W, 256), ~32k-67k params depending on grid
"""
import torch
import torch.nn as nn

from arch_ablation_base import run, Backbone, ArticulatoryBottleneckHead


class LinearReadIn(nn.Module):
    """Flatten + Linear: (B, H, W, T) -> (B, 256, T)."""

    def __init__(self, grid_h, grid_w, d_out=256):
        super().__init__()
        self.linear = nn.Linear(grid_h * grid_w, d_out)
        self.d_flat = d_out

    def forward(self, x):
        B, H, W, T = x.shape
        x = x.reshape(B, H * W, T)       # (B, H*W, T)
        x = x.permute(0, 2, 1)           # (B, T, H*W)
        x = self.linear(x)               # (B, T, d_out)
        return x.permute(0, 2, 1)        # (B, d_out, T)


if __name__ == "__main__":
    run(ReadInCls=LinearReadIn, BackboneCls=Backbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp43_linear_readin")
