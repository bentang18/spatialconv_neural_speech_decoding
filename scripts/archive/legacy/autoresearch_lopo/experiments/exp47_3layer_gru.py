#!/usr/bin/env python3
"""Exp 47: 3-layer BiGRU (instead of 2-layer).

Deeper temporal processing. More layers can capture longer-range
dependencies but risk overfitting. GRU dropout (0.3) between layers
provides regularization.

Baseline (exp33): 2-layer BiGRU
Change: 3-layer BiGRU
"""
from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead, Backbone


class ThreeLayerBackbone(Backbone):
    def __init__(self, d_in=256):
        super().__init__(d_in=d_in, gru_layers=3)


if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=ThreeLayerBackbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp47_3layer_gru")
