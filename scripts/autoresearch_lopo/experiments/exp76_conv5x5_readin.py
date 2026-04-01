#!/usr/bin/env python3
"""Exp 76: Larger spatial kernel (5x5 instead of 3x3) with MaxPool.

A 5x5 kernel covers ~10mm at the electrode spacing (~2mm pitch),
spanning more of the somatotopic organization in a single convolution.
The 3x3 kernel covers ~6mm. Somatotopic organization spans 3-5mm for
individual articulators — 5x5 may capture multi-articulator interactions.
"""
import torch.nn as nn
import torch.nn.functional as F

from arch_ablation_base import run, Backbone, ArticulatoryBottleneckHead

class Conv5x5ReadIn(nn.Module):
    def __init__(self, grid_h, grid_w, C=8, pool_h=4, pool_w=8):
        super().__init__()
        self.conv = nn.Conv2d(1, C, 5, padding=2)  # 5x5 kernel
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
    run(ReadInCls=Conv5x5ReadIn, BackboneCls=Backbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp76_conv5x5_maxpool")
