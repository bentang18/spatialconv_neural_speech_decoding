#!/usr/bin/env python3
"""Exp 74: stride=5 + MaxPool + C=16 (all 3 winners combined without GRU change).

exp51 (stride5+C16+1layer) = 0.760, exp67 (stride5+MaxPool) = 0.767,
exp68 (stride5+MaxPool+1layer) = 0.758. Try without GRU layer change.
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

from arch_ablation_base import run, Backbone, ArticulatoryBottleneckHead

class WideMaxPoolReadIn(nn.Module):
    def __init__(self, grid_h, grid_w, C=16, pool_h=4, pool_w=8):
        super().__init__()
        self.conv = nn.Conv2d(1, C, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool2d((pool_h, pool_w))
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

class Stride5WideBackbone(Backbone):
    def __init__(self, d_in=512):
        super().__init__(d_in=d_in, stride=5)

if __name__ == "__main__":
    run(ReadInCls=WideMaxPoolReadIn, BackboneCls=Stride5WideBackbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp74_stride5_maxpool_c16")
