#!/usr/bin/env python3
"""Exp 46: 1-layer BiGRU (instead of 2-layer).

Simpler model — less risk of overfitting with 153 target trials.
With 1 layer there's no inter-layer dropout, so the GRU itself has
zero dropout (only feature dropout and time masking remain).

Baseline (exp33): 2-layer BiGRU, gru_dropout=0.3
Change: 1-layer BiGRU, gru_dropout=0 (automatically, since layers=1)
"""
from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead, Backbone


class OneLayerBackbone(Backbone):
    def __init__(self, d_in=256):
        super().__init__(d_in=d_in, gru_layers=1)


if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=OneLayerBackbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp46_1layer_gru")
