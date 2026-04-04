#!/usr/bin/env python3
"""Exp 73: Even finer temporal resolution — stride=3 (~67Hz).

If stride=5 beat stride=10, maybe stride=3 is even better? GRU sees
~67 time steps instead of 40. Risk: too many steps, overfitting.
"""
from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead, Backbone

class Stride3Backbone(Backbone):
    def __init__(self, d_in=256):
        super().__init__(d_in=d_in, stride=3)

if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=Stride3Backbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp73_stride3_67Hz")
