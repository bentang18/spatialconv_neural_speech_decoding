#!/usr/bin/env python3
"""Exp 58: No augmentation at all.

Are our augmentations actually helping? With ~1300 source trials in S1,
maybe the data is sufficient without augmentation. Augmentation adds noise
that could hurt when data is not the bottleneck.

Baseline: 5 augmentations (time_shift, temporal_stretch, amplitude_scale,
          channel_dropout, gaussian_noise), PER=0.762
Change: identity augmentation (no changes)
"""
from arch_ablation_base import run, SpatialReadIn, Backbone, ArticulatoryBottleneckHead


def no_augment(x):
    return x


if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=Backbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp58_no_augmentation",
        augment_fn=no_augment)
