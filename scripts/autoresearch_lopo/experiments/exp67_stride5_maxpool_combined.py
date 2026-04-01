#!/usr/bin/env python3
"""Exp 67: Combine stride=5 + MaxPool (both independently hit 0.749).

Both stride=5 and MaxPool independently achieved 0.749. Are they capturing
the same thing (temporal resolution + spatial sharpness) or complementary
improvements? If complementary, this could break below 0.749.

Baseline: stride=10 + AvgPool, PER=0.762
Best singles: stride=5 = 0.749, MaxPool = 0.749
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

from arch_ablation_base import run, Backbone, ArticulatoryBottleneckHead


class MaxPoolReadIn(nn.Module):
    def __init__(self, grid_h, grid_w, C=8, pool_h=4, pool_w=8):
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


class Stride5Backbone(Backbone):
    def __init__(self, d_in=256):
        super().__init__(d_in=d_in, stride=5)


if __name__ == "__main__":
    run(ReadInCls=MaxPoolReadIn, BackboneCls=Stride5Backbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp67_stride5_maxpool")
