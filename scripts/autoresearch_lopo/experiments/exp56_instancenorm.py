#!/usr/bin/env python3
"""Exp 56: InstanceNorm instead of LayerNorm.

LayerNorm normalizes across features for each sample. InstanceNorm normalizes
each feature channel independently across time — this removes per-channel
scale/offset, which may remove patient-specific amplitude differences.
InstanceNorm is used in style transfer to remove "style" — here, patient
identity IS the "style" we want to remove.

Baseline: LayerNorm(d_in), PER=0.762
Change: InstanceNorm1d(d_in)
"""
import math
import torch
import torch.nn as nn

from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead


class InstanceNormBackbone(nn.Module):
    """InstanceNorm -> Conv1d(stride) -> GELU -> BiGRU."""

    def __init__(self, d_in=256, d=32, gru_hidden=32, gru_layers=2,
                 stride=10, gru_dropout=0.3, feat_drop_max=0.3,
                 time_mask_min=2, time_mask_max=5):
        super().__init__()
        # InstanceNorm1d: normalizes each channel across time dimension
        self.norm = nn.InstanceNorm1d(d_in, affine=True)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_in, d, kernel_size=stride, stride=stride), nn.GELU(),
        )
        self.gru = nn.GRU(
            d, gru_hidden, num_layers=gru_layers, batch_first=True,
            bidirectional=True, dropout=gru_dropout if gru_layers > 1 else 0.0,
        )
        self.out_dim = gru_hidden * 2
        self.feat_drop_max = feat_drop_max
        self.time_mask_min = time_mask_min
        self.time_mask_max = time_mask_max

    def forward(self, x):
        # x: (B, d_in, T) — InstanceNorm1d expects (B, C, T)
        x = self.norm(x)
        if self.training and self.feat_drop_max > 0:
            p = torch.rand(1).item() * self.feat_drop_max
            mask = (torch.rand(x.shape[1], device=x.device) > p).float()
            x = x * mask.unsqueeze(0).unsqueeze(-1) / (1 - p + 1e-8)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)
        if self.training and self.time_mask_max > 0:
            T = x.shape[1]
            ml = torch.randint(self.time_mask_min, self.time_mask_max + 1, (1,)).item()
            st = torch.randint(0, max(T - ml, 1), (1,)).item()
            taper = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, ml, device=x.device)))
            x = x.clone()
            x[:, st:st + ml, :] *= (1 - taper).unsqueeze(-1)
        h, _ = self.gru(x)
        return h


if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=InstanceNormBackbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp56_instancenorm")
