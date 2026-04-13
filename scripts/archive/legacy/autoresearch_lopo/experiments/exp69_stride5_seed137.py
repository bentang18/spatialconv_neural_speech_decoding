#!/usr/bin/env python3
"""Exp 69: Stride=5 with seed 137 (multi-seed validation of best result)."""
from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead, Backbone

class Stride5Backbone(Backbone):
    def __init__(self, d_in=256):
        super().__init__(d_in=d_in, stride=5)

if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=Stride5Backbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp69_stride5_seed137", seed=137)
