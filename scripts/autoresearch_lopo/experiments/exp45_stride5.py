#!/usr/bin/env python3
"""Exp 45: Stride 5 (40Hz) instead of stride 10 (20Hz).

Doubles temporal resolution. The GRU sees 2x more time steps, which means
2x more context for phoneme discrimination. Risk: more parameters in GRU
(longer sequences), potentially slower convergence.

Field standard is 20Hz (stride 10). But phoneme transitions are fast
(~50ms) — 40Hz may capture transition dynamics better.

Baseline (exp33): stride=10, 20 time steps for 200-frame input
Change: stride=5, 40 time steps for 200-frame input
"""
from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead, Backbone


class Stride5Backbone(Backbone):
    def __init__(self, d_in=256):
        super().__init__(d_in=d_in, stride=5)


if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=Stride5Backbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp45_stride5_40Hz")
