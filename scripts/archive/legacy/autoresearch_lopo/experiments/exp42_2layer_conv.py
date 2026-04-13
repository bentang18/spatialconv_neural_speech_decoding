#!/usr/bin/env python3
"""Exp 42: 2-layer Conv2d read-in.

Deeper spatial processing: Conv2d(1,8,k=3) -> ReLU -> Conv2d(8,16,k=3) -> ReLU -> Pool.
Can learn gradient/Laplacian-like filters (layer 1) composed with higher-order
spatial features (layer 2). Currently at 1-layer (80 params).

Baseline (exp33): 1-layer Conv2d, 80 params, d_flat=256
Change: 2-layer Conv2d, ~1,300 params, d_flat=512
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

from arch_ablation_base import run, Backbone, ArticulatoryBottleneckHead, DEVICE


class TwoLayerReadIn(nn.Module):
    """2-layer Conv2d: (B, H, W, T) -> (B, d_flat, T)."""

    def __init__(self, grid_h, grid_w, C1=8, C2=16, pool_h=4, pool_w=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, C1, 3, padding=1)
        self.conv2 = nn.Conv2d(C1, C2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        self.d_flat = C2 * pool_h * pool_w

    def forward(self, x):
        B, H, W, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B * T, 1, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if x.device.type == "mps":
            x = self.pool(x.cpu()).to("mps")
        else:
            x = self.pool(x)
        return x.reshape(B, T, -1).permute(0, 2, 1)


class BackboneFor512(Backbone):
    def __init__(self, d_in=512):
        super().__init__(d_in=d_in)


if __name__ == "__main__":
    run(ReadInCls=TwoLayerReadIn, BackboneCls=BackboneFor512,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp42_2layer_conv")
