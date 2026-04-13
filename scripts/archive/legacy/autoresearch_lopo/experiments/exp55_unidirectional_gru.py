#!/usr/bin/env python3
"""Exp 55: Unidirectional GRU (forward only).

BiGRU sees the full sequence both ways. But for real-time decoding,
only forward direction is available. Also, the backward pass may
add noise rather than signal for short (20-frame) sequences.

Baseline: BiGRU(32) -> out_dim=64, PER=0.762
Change: forward GRU(64) -> out_dim=64 (same output dim, double hidden)
"""
from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead, Backbone
import torch.nn as nn


class UniGRUBackbone(Backbone):
    def __init__(self, d_in=256):
        # Double hidden to match output dim (64) since no bidirectional doubling
        super().__init__(d_in=d_in, gru_hidden=64)
        # Override GRU to be unidirectional
        self.gru = nn.GRU(
            32, 64, num_layers=2, batch_first=True,
            bidirectional=False, dropout=0.3,
        )
        self.out_dim = 64  # was 64 from bi, now 64 from larger hidden


if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=UniGRUBackbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp55_unidirectional_gru")
