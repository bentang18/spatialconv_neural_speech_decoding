#!/usr/bin/env python3
"""Exp 86: Multi-scale temporal backbone (stride 3+5+10 concatenated).

Different temporal scales capture different aspects of speech:
  - stride=3 (~67Hz): fast articulatory transitions
  - stride=5 (~40Hz): syllable-level dynamics
  - stride=10 (~20Hz): broad phoneme-level patterns

Concatenating features from all scales gives the GRU richer input.
The GRU learns to integrate across scales.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead


class MultiScaleBackbone(nn.Module):
    """Multi-scale temporal encoding with concatenated features."""
    def __init__(self, d_in=256, d=16, gru_hidden=32, gru_layers=2,
                 feat_drop_max=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        # Three temporal scales
        self.conv3 = nn.Sequential(nn.Conv1d(d_in, d, kernel_size=3, stride=3), nn.GELU())
        self.conv5 = nn.Sequential(nn.Conv1d(d_in, d, kernel_size=5, stride=5), nn.GELU())
        self.conv10 = nn.Sequential(nn.Conv1d(d_in, d, kernel_size=10, stride=10), nn.GELU())
        # Adaptive pool to common length (use stride=10 output length as reference)
        self.gru = nn.GRU(
            d * 3, gru_hidden, num_layers=gru_layers, batch_first=True,
            bidirectional=True, dropout=0.3 if gru_layers > 1 else 0.0)
        self.out_dim = gru_hidden * 2
        self.feat_drop_max = feat_drop_max

    def forward(self, x):
        x = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.training and self.feat_drop_max > 0:
            p = torch.rand(1).item() * self.feat_drop_max
            mask = (torch.rand(x.shape[1], device=x.device) > p).float()
            x = x * mask.unsqueeze(0).unsqueeze(-1) / (1 - p + 1e-8)

        h3 = self.conv3(x).permute(0, 2, 1)   # (B, T3, d)
        h5 = self.conv5(x).permute(0, 2, 1)   # (B, T5, d)
        h10 = self.conv10(x).permute(0, 2, 1)  # (B, T10, d)

        # Adaptive pool all to T10 length
        T10 = h10.shape[1]
        if h3.shape[1] != T10:
            h3 = F.adaptive_avg_pool1d(h3.permute(0, 2, 1), T10).permute(0, 2, 1)
        if h5.shape[1] != T10:
            h5 = F.adaptive_avg_pool1d(h5.permute(0, 2, 1), T10).permute(0, 2, 1)

        h_cat = torch.cat([h3, h5, h10], dim=-1)  # (B, T10, 3*d)
        out, _ = self.gru(h_cat)
        return out


if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=MultiScaleBackbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp86_multiscale_temporal")
