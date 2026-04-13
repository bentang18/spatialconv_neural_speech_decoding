#!/usr/bin/env python3
"""Exp 62: No mixup augmentation.

Mixup interpolates between training examples — but in LOPO, mixing
two trials from the same source patient doesn't encourage cross-patient
invariance. It may even hurt by blurring the articulatory features
that the bottleneck head needs to classify sharply.

Baseline: MIXUP_ALPHA=0.2, PER=0.762
Change: MIXUP_ALPHA=0 (no mixup)
"""
import arch_ablation_base as base
from arch_ablation_base import run, SpatialReadIn, Backbone, ArticulatoryBottleneckHead

base.MIXUP_ALPHA = 0.0

if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=Backbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp62_no_mixup")
