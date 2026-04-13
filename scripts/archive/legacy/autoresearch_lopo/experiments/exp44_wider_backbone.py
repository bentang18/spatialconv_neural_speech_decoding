#!/usr/bin/env python3
"""Exp 44: Wider backbone (d=64, gru_hidden=64 -> out_dim=128).

Current backbone is very narrow: Conv1d projects to 32-dim, GRU has 32 hidden.
This doubles capacity everywhere. Total backbone goes from ~5K to ~25K params.
Risk: overfitting with 153 target trials. But S1 has ~1300 source trials.

Baseline (exp33): d=32, gru_hidden=32, out_dim=64, ~5K backbone params
Change: d=64, gru_hidden=64, out_dim=128, ~25K backbone params
"""
from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead, Backbone


class WiderBackbone(Backbone):
    def __init__(self, d_in=256):
        super().__init__(d_in=d_in, d=64, gru_hidden=64)


class WiderHead(ArticulatoryBottleneckHead):
    def __init__(self, d_in=128):
        super().__init__(d_in=d_in)


if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=WiderBackbone,
        HeadCls=WiderHead,
        experiment_name="exp44_wider_backbone_d64_H64")
