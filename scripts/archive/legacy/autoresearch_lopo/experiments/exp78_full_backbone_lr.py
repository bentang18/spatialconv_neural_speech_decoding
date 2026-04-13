#!/usr/bin/env python3
"""Exp 78: Full backbone LR in Stage 2 (BACKBONE_LR_MULT=1.0 instead of 0.1).

DIAGNOSTIC: Is the backbone frozen too much in S2? If full fine-tuning
improves PER, the S1 features need more adaptation to the target.
"""
import arch_ablation_base as base

base.S2_BACKBONE_LR_MULT = 1.0

if __name__ == "__main__":
    base.run(experiment_name="exp78_full_backbone_lr")
