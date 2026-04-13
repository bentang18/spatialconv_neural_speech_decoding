#!/usr/bin/env python3
"""Exp 84: Self-training with pseudo-labels.

Iterative: S1→S2→predict→relabel→retrain.
1. Train S1+S2 normally
2. Generate pseudo-labels on source patients from target-adapted model
3. Retrain S1 using original + pseudo-labels (confidence-thresholded)
4. Run S2 on retrained model

Self-training creates a feedback loop: S2 adaptation to S14 biases features
toward S14-relevant patterns. Pseudo-labeling source patients with this
model forces S1 to emphasize cross-patient features that generalize to S14.
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
    DEVICE, SEED, SpatialReadIn, Backbone, ArticulatoryBottleneckHead,
    augment, compute_loss, train_stage1, train_eval_fold,
    extract_embeddings, knn_predict, TTA_COPIES,
)


def generate_pseudo_labels(backbone, head, read_ins, all_data, confidence_thresh=0.7):
    """Generate pseudo-labels for source patients using adapted model."""
    backbone.eval(); head.eval()
    pseudo = {}
    total, kept = 0, 0

    for pid in prepare.SOURCE_PATIENTS:
        ri = read_ins[pid]
        ri.eval()
        grids = all_data[pid]["grids"]
        n = len(all_data[pid]["labels"])
        new_labels = list(all_data[pid]["labels"])  # start with real labels
        confidences = []

        with torch.no_grad():
            for start in range(0, n, 32):
                end = min(start + 32, n)
                x = grids[start:end].to(DEVICE)
                feat = ri(x)
                h = backbone(feat)
                logits = head(h)  # (B, 3, 9)

                # Get predictions and confidence
                probs = F.softmax(logits, dim=-1)
                max_probs, preds = probs.max(dim=-1)  # (B, 3)
                mean_conf = max_probs.mean(dim=-1)  # (B,)

                for i in range(end - start):
                    conf = mean_conf[i].item()
                    total += 1
                    if conf >= confidence_thresh:
                        pred = (preds[i] + 1).cpu().tolist()
                        new_labels[start + i] = pred
                        kept += 1
                    confidences.append(conf)

        pseudo[pid] = new_labels

    print(f"  Pseudo-labels: {kept}/{total} above threshold ({100*kept/max(total,1):.1f}%)")
    return pseudo


def run():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    all_data = prepare.load_all_patients()
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)

    print("=== exp84_self_training ===")
    print(f"Target: {prepare.TARGET_PATIENT} | Self-training with pseudo-labels")

    # Round 0: normal S1+S2
    print("\n--- Round 0: Normal training ---")
    backbone, head, read_ins = train_stage1(
        all_data, SpatialReadIn, Backbone, ArticulatoryBottleneckHead)

    # Do one S2 fold to get adapted model for pseudo-labeling
    tr_idx, va_idx = splits[0]
    _, _, _ = train_eval_fold(
        backbone, head, grids[tr_idx], [labels[i] for i in tr_idx],
        grids[va_idx], [labels[i] for i in va_idx],
        read_ins=read_ins, all_data=all_data, ReadInCls=SpatialReadIn)

    # Generate pseudo-labels from round 0 model
    print("\n--- Generating pseudo-labels ---")
    pseudo_labels = generate_pseudo_labels(backbone, head, read_ins, all_data)

    # Round 1: Retrain S1 with pseudo-labels mixed in
    print("\n--- Round 1: Retrain with pseudo-labels ---")
    # Replace source labels with pseudo-labels
    augmented_data = {}
    for pid in prepare.SOURCE_PATIENTS:
        augmented_data[pid] = {
            "grids": all_data[pid]["grids"],
            "labels": pseudo_labels[pid],
            "grid_shape": all_data[pid]["grid_shape"],
        }
    # Keep target data unchanged
    augmented_data[prepare.TARGET_PATIENT] = all_data[prepare.TARGET_PATIENT]

    backbone2, head2, read_ins2 = train_stage1(
        augmented_data, SpatialReadIn, Backbone, ArticulatoryBottleneckHead)

    # Evaluate with retrained model
    print("\n--- Evaluating round 1 model ---")
    fold_pers = []
    all_preds, all_refs = [], []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        per, preds, methods = train_eval_fold(
            backbone2, head2, grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx],
            read_ins=read_ins2, all_data=augmented_data, ReadInCls=SpatialReadIn)
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
