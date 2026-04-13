#!/usr/bin/env python3
"""Exp 59: Transformer + stride 5 (40Hz).

Transformer matched GRU at stride=10 (0.765 vs 0.762). Stride=5 was the
biggest win (0.749). Combining: Transformer with 40 time steps should
benefit even MORE from self-attention (more positions to attend to).

Baseline: BiGRU stride=10, PER=0.762 | Transformer stride=10, PER=0.765
Change: Transformer stride=5
"""
import math
import torch
import torch.nn as nn

from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead


class TransformerStride5(nn.Module):
    def __init__(self, d_in=256, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, stride=5, dropout=0.3,
                 feat_drop_max=0.3, time_mask_min=2, time_mask_max=5):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_in, d_model, kernel_size=stride, stride=stride), nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_dim = d_model
        self.feat_drop_max = feat_drop_max
        self.time_mask_min = time_mask_min
        self.time_mask_max = time_mask_max

    def forward(self, x):
        x = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.training and self.feat_drop_max > 0:
            p = torch.rand(1).item() * self.feat_drop_max
            mask = (torch.rand(x.shape[1], device=x.device) > p).float()
            x = x * mask.unsqueeze(0).unsqueeze(-1) / (1 - p + 1e-8)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)
        T = x.shape[1]
        x = x + self.pos_embed[:, :T, :]
        if self.training and self.time_mask_max > 0:
            ml = torch.randint(self.time_mask_min, self.time_mask_max + 1, (1,)).item()
            st = torch.randint(0, max(T - ml, 1), (1,)).item()
            taper = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, ml, device=x.device)))
            x = x.clone()
            x[:, st:st + ml, :] *= (1 - taper).unsqueeze(-1)
        x = self.transformer(x)
        return x


if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=TransformerStride5,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp59_transformer_stride5")
