#!/usr/bin/env python3
"""Exp 94: Multi-scale temporal + MaxPool read-in (combine 2 architecture wins)."""
import torch.nn as nn
import torch.nn.functional as F
from arch_ablation_base import run, ArticulatoryBottleneckHead
from exp86_multiscale import MultiScaleBackbone


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


if __name__ == "__main__":
    run(ReadInCls=MaxPoolReadIn, BackboneCls=MultiScaleBackbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp94_multiscale_maxpool")
