#!/usr/bin/env python3
"""Exp 61: Larger batch size (32 instead of 16).

With ~1300 source trials, batch=16 means ~80 gradient steps per epoch per patient.
Batch=32 halves noise in gradient estimates — may stabilize cross-patient training.
Also doubles effective negative samples for the articulatory bottleneck similarity.

Baseline: batch_size=16, PER=0.762
Change: batch_size=32
"""
import arch_ablation_base as base
from arch_ablation_base import run, SpatialReadIn, Backbone, ArticulatoryBottleneckHead

# Override batch sizes
base.S1_BATCH_SIZE = 32
base.S2_BATCH_SIZE = 32

if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=Backbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp61_batch32")
