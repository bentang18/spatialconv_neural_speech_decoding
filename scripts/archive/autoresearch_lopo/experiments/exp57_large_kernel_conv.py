#!/usr/bin/env python3
"""Exp 57: Large Conv1d kernel (k=20, s=5) — overlapping temporal windows.

Current Conv1d uses k=stride (non-overlapping). A larger kernel with smaller
stride creates overlapping windows that can capture longer temporal patterns
while maintaining 40Hz resolution.

Baseline: Conv1d(k=10, s=10) -> 20 frames at 20Hz
Change: Conv1d(k=20, s=5) -> 40 frames at 40Hz with 75% overlap
"""
from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead, Backbone
import torch.nn as nn


class LargeKernelBackbone(Backbone):
    def __init__(self, d_in=256):
        super().__init__(d_in=d_in, stride=5)
        # Override temporal_conv with larger kernel
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_in, 32, kernel_size=20, stride=5, padding=10), nn.GELU(),
        )


if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=LargeKernelBackbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp57_large_kernel_k20_s5")
