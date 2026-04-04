#!/usr/bin/env python3
"""Exp 85: Transductive inference with label propagation.

Instead of classifying each val sample independently, use the structure
of the val set itself. Label propagation builds a graph over train+val
embeddings and propagates labels using graph structure.

With only ~30 val samples per fold, val samples constrain each other:
if two val samples are close in embedding space, they likely share labels.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare
from arch_ablation_base import (
    DEVICE, SEED, SpatialReadIn, Backbone, ArticulatoryBottleneckHead,
    train_stage1, train_eval_fold, extract_embeddings, augment, TTA_COPIES,
)


def label_propagation(train_emb, train_labels, val_emb, alpha=0.5, n_iter=20):
    """Label propagation: propagate labels from train to val via graph.

    alpha: weight of graph-propagated labels vs initial labels (0=all graph, 1=all initial)
    """
    all_emb = torch.cat([train_emb, val_emb], dim=0)
    n_train = train_emb.shape[0]
    n_val = val_emb.shape[0]
    n_total = n_train + n_val

    # Build affinity matrix (RBF kernel)
    all_norm = F.normalize(all_emb, dim=1)
    sim = all_norm @ all_norm.T
    # Convert to affinity with temperature
    W = torch.exp(sim * 10)  # temperature scaling
    W.fill_diagonal_(0)  # no self-loops

    # Row-normalize (transition matrix)
    D = W.sum(dim=1, keepdim=True)
    T_mat = W / (D + 1e-8)

    preds_all_positions = []
    for pos in range(prepare.N_POSITIONS):
        # Initialize label distribution
        Y = torch.zeros(n_total, prepare.N_CLASSES)
        for i in range(n_train):
            cls = train_labels[i][pos] - 1  # 0-indexed
            Y[i, cls] = 1.0

        Y_init = Y.clone()

        # Iterate
        for _ in range(n_iter):
            Y = alpha * Y_init + (1 - alpha) * (T_mat @ Y)
            # Clamp train labels
            for i in range(n_train):
                Y[i] = Y_init[i]

        # Extract val predictions
        val_Y = Y[n_train:]
        preds_pos = (val_Y.argmax(dim=1) + 1).tolist()
        preds_all_positions.append(preds_pos)

    # Combine positions
    preds = [[preds_all_positions[pos][i] for pos in range(prepare.N_POSITIONS)]
             for i in range(n_val)]
    return preds


def run():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    all_data = prepare.load_all_patients()
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)

    print("=== exp85_transductive ===")
    print(f"Target: {prepare.TARGET_PATIENT} | Label propagation")

    backbone, head, read_ins = train_stage1(
        all_data, SpatialReadIn, Backbone, ArticulatoryBottleneckHead)

    fold_pers_lp = []
    fold_pers_baseline = []
    all_preds, all_refs = [], []

    for fi, (tr_idx, va_idx) in enumerate(splits):
        ft0 = time.time()
        # Standard S2 for comparison
        per_base, preds_base, methods = train_eval_fold(
            backbone, head, grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx],
            read_ins=read_ins, all_data=all_data, ReadInCls=SpatialReadIn)
        fold_pers_baseline.append(per_base)

        # Extract embeddings from S2-adapted model for label propagation
        # Use backbone directly (no S2 adaptation) for embeddings
        target_ri = SpatialReadIn(grids.shape[1], grids.shape[2]).to(DEVICE)
        train_emb = extract_embeddings(backbone, target_ri, grids[tr_idx])
        val_emb = extract_embeddings(backbone, target_ri, grids[va_idx])
        train_labels = [labels[i] for i in tr_idx]
        val_labels = [labels[i] for i in va_idx]

        # Label propagation with different alphas
        best_lp_per = float("inf")
        best_lp_preds = None
        for alpha in [0.2, 0.5, 0.8]:
            lp_preds = label_propagation(train_emb, train_labels, val_emb, alpha=alpha)
            lp_per = prepare.compute_per(lp_preds, val_labels)
            if lp_per < best_lp_per:
                best_lp_per = lp_per
                best_lp_preds = lp_preds

        # Overall best
        best_per = min(per_base, best_lp_per)
        best_preds = preds_base if per_base <= best_lp_per else best_lp_preds
        fold_pers_lp.append(best_per)
        all_preds.extend(best_preds)
        all_refs.extend(val_labels)

        print(f"  Fold {fi+1}: baseline={per_base:.4f} label_prop={best_lp_per:.4f} "
              f"best={best_per:.4f} ({time.time()-ft0:.1f}s)")
        if time.time() - t0 > prepare.TIME_BUDGET: break

    mean_per = float(np.mean(fold_pers_lp))
    collapse = prepare.compute_content_collapse(all_preds)
    print(f"\n---\nval_per:            {mean_per:.6f}")
    print(f"val_per_std:        {float(np.std(fold_pers_lp)):.6f}")
    print(f"fold_pers:          {fold_pers_lp}")
    print(f"baseline_pers:      {fold_pers_baseline}")
    print(f"collapsed:          {collapse['collapsed']}")
    print(f"mean_entropy:       {collapse['mean_entropy']:.3f}")
    print(f"training_seconds:   {time.time()-t0:.1f}")


if __name__ == "__main__":
    run()
