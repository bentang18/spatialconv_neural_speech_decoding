#!/usr/bin/env python3
"""Exp 93: Multi-scale temporal with seed 256 (multi-seed validation)."""
from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead
from exp86_multiscale import MultiScaleBackbone

if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=MultiScaleBackbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp93_multiscale_seed256", seed=256)
