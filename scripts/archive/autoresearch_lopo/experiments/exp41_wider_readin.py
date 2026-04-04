#!/usr/bin/env python3
"""Exp 41: Wider read-in (C=16 instead of C=8).

Doubles spatial filter count: 16 filters × pool(4,8) = 512-dim input to backbone.
Current C=8 gives only 80 params and 256-dim — may be undersensing spatial patterns.

Baseline (exp33): C=8, d_flat=256, PER=0.762
Change: C=16, d_flat=512
"""
from arch_ablation_base import run, SpatialReadIn, Backbone, ArticulatoryBottleneckHead


class WiderReadIn(SpatialReadIn):
    def __init__(self, grid_h, grid_w):
        super().__init__(grid_h, grid_w, C=16)


class WiderBackbone(Backbone):
    """Backbone that accepts 512-dim input from wider read-in."""
    def __init__(self, d_in=512):
        super().__init__(d_in=d_in)


if __name__ == "__main__":
    run(ReadInCls=WiderReadIn, BackboneCls=WiderBackbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp41_wider_readin_C16")
