#!/usr/bin/env python3
"""Exp 50: Wide everything — 2x wider at every layer.

C=16 read-in (512-dim) + d=64 Conv1d + H=64 GRU = largest model tested.
Total params ~30K+ per read-in (vs ~5K baseline). Tests whether the
architecture was simply too small.

If this wins, capacity was the bottleneck. If it loses, overfitting
dominates at this data scale.

Baseline (exp33): C=8/d_flat=256, d=32, H=32, out=64
Change: C=16/d_flat=512, d=64, H=64, out=128
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

from arch_ablation_base import run, ArticulatoryBottleneckHead, Backbone, DEVICE


class WideReadIn(nn.Module):
    def __init__(self, grid_h, grid_w, C=16, pool_h=4, pool_w=8):
        super().__init__()
        self.conv = nn.Conv2d(1, C, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        self.d_flat = C * pool_h * pool_w

    def forward(self, x):
        B, H, W, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B * T, 1, H, W)
        x = F.relu(self.conv(x))
        if x.device.type == "mps":
            x = self.pool(x.cpu()).to("mps")
        else:
            x = self.pool(x)
        return x.reshape(B, T, -1).permute(0, 2, 1)


class WideBackbone(Backbone):
    def __init__(self, d_in=512):
        super().__init__(d_in=d_in, d=64, gru_hidden=64)


class WideHead(ArticulatoryBottleneckHead):
    def __init__(self, d_in=128):
        super().__init__(d_in=d_in)


if __name__ == "__main__":
    run(ReadInCls=WideReadIn, BackboneCls=WideBackbone,
        HeadCls=WideHead,
        experiment_name="exp50_wide_everything_2x")
