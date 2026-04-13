#!/usr/bin/env python3
"""Exp 49: No regularization (ablation).

Remove ALL stochastic regularization: feature dropout, time masking,
GRU inter-layer dropout, head dropout. Tests whether our current
regularization is helping or hurting.

With S1 having ~1300 source trials, regularization may be unnecessary
and could be destroying signal. Conversely, S2 has only ~120 train
trials per fold — regularization may be critical there.

Baseline (exp33): feat_drop=0.3, time_mask=[2,5], gru_dropout=0.3, head_dropout=0.3
Change: all set to 0
"""
from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead, Backbone


class NoRegBackbone(Backbone):
    def __init__(self, d_in=256):
        super().__init__(d_in=d_in, gru_dropout=0.0, feat_drop_max=0.0,
                         time_mask_min=0, time_mask_max=0)


class NoRegHead(ArticulatoryBottleneckHead):
    def __init__(self, d_in=64):
        super().__init__(d_in=d_in, dropout=0.0)


if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=NoRegBackbone,
        HeadCls=NoRegHead,
        experiment_name="exp49_no_regularization")
