#!/usr/bin/env python3
"""Exp 91: Combine the two winners — multi-scale temporal + self-training.

exp86 (multiscale) = 0.757, exp84 (self-training) = 0.758.
If improvements are orthogonal (one is architecture, one is training),
they should stack. If both exploit the same signal, they won't.
"""
from __future__ import annotations
import sys, time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare
from arch_ablation_base import (
    DEVICE, SEED, SpatialReadIn, ArticulatoryBottleneckHead,
    augment, compute_loss, train_eval_fold,
    extract_embeddings, knn_predict,
)
# Import components from the two winning experiments
from exp86_multiscale import MultiScaleBackbone
from exp84_self_training import generate_pseudo_labels


def run():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    all_data = prepare.load_all_patients()
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)

    print("=== exp91_multiscale_self_training ===")
    print(f"Target: {prepare.TARGET_PATIENT} | MultiScale + Self-Training")

    # Round 0: Train with multi-scale backbone
    from arch_ablation_base import train_stage1
    backbone0, head0, read_ins0 = train_stage1(
        all_data, SpatialReadIn, MultiScaleBackbone, ArticulatoryBottleneckHead)

    # Generate pseudo-labels from round 0
    print("\n--- Generating pseudo-labels ---")
    # Quick S2 on first fold to get adapted model
    tr0, va0 = splits[0]
    train_eval_fold(backbone0, head0, grids[tr0], [labels[i] for i in tr0],
                    grids[va0], [labels[i] for i in va0],
                    read_ins=read_ins0, all_data=all_data, ReadInCls=SpatialReadIn)
    pseudo_labels = generate_pseudo_labels(backbone0, head0, read_ins0, all_data)

    # Round 1: Retrain with pseudo-labels + multi-scale backbone
    print("\n--- Round 1: Retrain ---")
    augmented_data = {}
    for pid in prepare.SOURCE_PATIENTS:
        augmented_data[pid] = {
            "grids": all_data[pid]["grids"],
            "labels": pseudo_labels[pid],
            "grid_shape": all_data[pid]["grid_shape"],
        }
    augmented_data[prepare.TARGET_PATIENT] = all_data[prepare.TARGET_PATIENT]

    backbone, head, read_ins = train_stage1(
        augmented_data, SpatialReadIn, MultiScaleBackbone, ArticulatoryBottleneckHead)

    # Evaluate
    print("\n--- Evaluation ---")
    fold_pers = []
    all_preds, all_refs = [], []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        per, preds, methods = train_eval_fold(
            backbone, head, grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx],
            read_ins=read_ins, all_data=augmented_data, ReadInCls=SpatialReadIn)
        fold_pers.append(per)
        all_preds.extend(preds)
        all_refs.extend([labels[i] for i in va_idx])
        best_method = min(methods, key=lambda k: methods[k][0])
        print(f"  Fold {fi+1}: PER={per:.4f} (best={best_method}) ({time.time()-t0:.1f}s)")
        if time.time() - t0 > prepare.TIME_BUDGET: break

    mean_per = float(np.mean(fold_pers))
    collapse = prepare.compute_content_collapse(all_preds)
    print(f"\n---\nval_per:            {mean_per:.6f}")
    print(f"val_per_std:        {float(np.std(fold_pers)):.6f}")
    print(f"fold_pers:          {fold_pers}")
    print(f"collapsed:          {collapse['collapsed']}")
    print(f"mean_entropy:       {collapse['mean_entropy']:.3f}")
    print(f"training_seconds:   {time.time()-t0:.1f}")


if __name__ == "__main__":
    run()
