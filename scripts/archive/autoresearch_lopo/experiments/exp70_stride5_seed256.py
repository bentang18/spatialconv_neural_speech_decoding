#!/usr/bin/env python3
"""Exp 70: Stride=5 with seed 256 (multi-seed validation of best result)."""
from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead, Backbone

class Stride5Backbone(Backbone):
    def __init__(self, d_in=256):
        super().__init__(d_in=d_in, stride=5)

if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=Stride5Backbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp70_stride5_seed256", seed=256)
