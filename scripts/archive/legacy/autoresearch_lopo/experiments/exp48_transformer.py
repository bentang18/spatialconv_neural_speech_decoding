#!/usr/bin/env python3
"""Exp 48: Transformer encoder replacing BiGRU.

Different inductive bias: self-attention can capture non-local temporal
dependencies directly (no sequential bottleneck). Risk: transformers
typically need more data than RNNs, and our sequences are short (~20 steps).

Architecture: LN -> Conv1d(stride=10) -> GELU -> 2-layer TransformerEncoder
with 4 heads, d_model=64, feedforward=128.

Baseline (exp33): 2-layer BiGRU(32), out_dim=64
Change: 2-layer TransformerEncoder, d_model=64, out_dim=64
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead


class TransformerBackbone(nn.Module):
    """LN -> Conv1d(stride) -> GELU -> positional encoding -> TransformerEncoder."""

    def __init__(self, d_in=256, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, stride=10, dropout=0.3,
                 feat_drop_max=0.3, time_mask_min=2, time_mask_max=5):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_in, d_model, kernel_size=stride, stride=stride), nn.GELU(),
        )
        # Learnable positional encoding (max 100 positions)
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
        x = x.permute(0, 2, 1)  # (B, T, d_model)
        T = x.shape[1]
        x = x + self.pos_embed[:, :T, :]
        if self.training and self.time_mask_max > 0:
            ml = torch.randint(self.time_mask_min, self.time_mask_max + 1, (1,)).item()
            st = torch.randint(0, max(T - ml, 1), (1,)).item()
            taper = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, ml, device=x.device)))
            x = x.clone()
            x[:, st:st + ml, :] *= (1 - taper).unsqueeze(-1)
        x = self.transformer(x)
        return x  # (B, T, d_model)


class TransformerHead(ArticulatoryBottleneckHead):
    def __init__(self, d_in=64):
        super().__init__(d_in=d_in)


if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=TransformerBackbone,
        HeadCls=TransformerHead,
        experiment_name="exp48_transformer_2L_4H")
