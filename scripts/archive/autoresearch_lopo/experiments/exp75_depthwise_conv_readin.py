#!/usr/bin/env python3
"""Exp 75: Depthwise separable Conv2d read-in.

Instead of standard Conv2d(1,C,3), use depthwise-separable: per-position
1x1 conv that weights channels, then spatial 3x3 conv. More parameter
efficient — separates "which channels matter" from "what spatial pattern".

This is what MobileNet uses — efficient spatial processing.
"""
import torch.nn as nn
import torch.nn.functional as F

from arch_ablation_base import run, Backbone, ArticulatoryBottleneckHead

class DepthwiseReadIn(nn.Module):
    def __init__(self, grid_h, grid_w, C=8, pool_h=4, pool_w=8):
        super().__init__()
        # Depthwise: each input channel gets its own 3x3 filter
        # But we only have 1 input channel, so this is just Conv2d(1,1,3)
        # Pointwise: 1x1 conv to expand channels
        # More useful: standard conv + pointwise expansion
        self.spatial = nn.Conv2d(1, 1, 3, padding=1)  # spatial filter
        self.pointwise = nn.Conv2d(1, C, 1)  # channel expansion
        self.pool = nn.AdaptiveMaxPool2d((pool_h, pool_w))  # MaxPool since it won
        self.d_flat = C * pool_h * pool_w

    def forward(self, x):
        B, H, W, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B * T, 1, H, W)
        x = F.relu(self.spatial(x))
        x = F.relu(self.pointwise(x))
        if x.device.type == "mps":
            x = self.pool(x.cpu()).to("mps")
        else:
            x = self.pool(x)
        return x.reshape(B, T, -1).permute(0, 2, 1)

if __name__ == "__main__":
    run(ReadInCls=DepthwiseReadIn, BackboneCls=Backbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp75_depthwise_conv_readin")
