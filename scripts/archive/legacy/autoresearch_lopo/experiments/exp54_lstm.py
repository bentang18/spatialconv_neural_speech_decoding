#!/usr/bin/env python3
"""Exp 54: LSTM instead of GRU.

LSTM has a cell state gate that GRU lacks — may better preserve long-range
temporal dependencies across the 20-frame sequence. Or GRU's simpler
gating may be better for this small data regime. Empirical test.

Baseline: 2-layer BiGRU(32), PER=0.762
Change: 2-layer BiLSTM(32)
"""
import math
import torch
import torch.nn as nn

from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead


class LSTMBackbone(nn.Module):
    """LN -> Conv1d(stride) -> GELU -> BiLSTM."""

    def __init__(self, d_in=256, d=32, hidden=32, layers=2, stride=10,
                 dropout=0.3, feat_drop_max=0.3, time_mask_min=2, time_mask_max=5):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_in, d, kernel_size=stride, stride=stride), nn.GELU(),
        )
        self.lstm = nn.LSTM(
            d, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.out_dim = hidden * 2
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
        if self.training and self.time_mask_max > 0:
            T = x.shape[1]
            ml = torch.randint(self.time_mask_min, self.time_mask_max + 1, (1,)).item()
            st = torch.randint(0, max(T - ml, 1), (1,)).item()
            taper = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, ml, device=x.device)))
            x = x.clone()
            x[:, st:st + ml, :] *= (1 - taper).unsqueeze(-1)
        h, _ = self.lstm(x)
        return h


if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=LSTMBackbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp54_lstm_vs_gru")
