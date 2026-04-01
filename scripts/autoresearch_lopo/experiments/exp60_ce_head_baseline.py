#!/usr/bin/env python3
"""Exp 60: CE head (no articulatory bottleneck) with stride=5.

The articulatory bottleneck won at stride=10. But with stride=5 giving
more temporal resolution, maybe the CE head can now match it — the extra
time steps may provide enough discriminative power without needing the
articulatory inductive bias.

Also serves as a clean CE baseline for stride=5 comparisons.

Baseline: Articulatory head + stride=10, PER=0.762 | stride=5 + artic, PER=0.749
Change: CE head + stride=5
"""
import torch.nn as nn

from arch_ablation_base import run, SpatialReadIn, Backbone


class Stride5Backbone(Backbone):
    def __init__(self, d_in=256):
        super().__init__(d_in=d_in, stride=5)


class CEHead(nn.Module):
    """Mean-pool -> dropout -> per-position Linear."""
    def __init__(self, d_in=64, n_positions=3, n_classes=9, dropout=0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Linear(d_in, n_classes) for _ in range(n_positions)
        ])

    def forward(self, h):
        pooled = self.drop(h.mean(dim=1))
        return torch.stack([head(pooled) for head in self.heads], dim=1)


import torch

if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=Stride5Backbone,
        HeadCls=CEHead,
        experiment_name="exp60_ce_head_stride5")
