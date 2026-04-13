#!/usr/bin/env python3
"""Exp 51: Combined best architecture changes (stride5 + C16 + 1-layer GRU).

stride=5 gave 0.749, C=16 gave 0.754, 1-layer GRU gave 0.757.
Combining all three — the question is whether they're additive or redundant.

Baseline (exp33): stride=10, C=8, 2-layer GRU, PER=0.762
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

from arch_ablation_base import run, ArticulatoryBottleneckHead


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


from arch_ablation_base import Backbone

class CombinedBackbone(Backbone):
    def __init__(self, d_in=512):
        super().__init__(d_in=d_in, stride=5, gru_layers=1)


if __name__ == "__main__":
    run(ReadInCls=WideReadIn, BackboneCls=CombinedBackbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp51_combined_stride5_C16_1layer")
