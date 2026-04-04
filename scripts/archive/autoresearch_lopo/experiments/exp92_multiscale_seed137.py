#!/usr/bin/env python3
"""Exp 92: Multi-scale temporal with seed 137 (multi-seed validation)."""
from arch_ablation_base import run, SpatialReadIn, ArticulatoryBottleneckHead
from exp86_multiscale import MultiScaleBackbone

if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=MultiScaleBackbone,
        HeadCls=ArticulatoryBottleneckHead,
        experiment_name="exp92_multiscale_seed137", seed=137)
