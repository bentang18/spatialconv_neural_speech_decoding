#!/usr/bin/env python3
"""Exp 77: Skip Stage 2 — evaluate S1 backbone directly on target with k-NN.

CRITICAL DIAGNOSTIC: Is S2 adaptation helping or hurting? Per-patient CE=0.700
but LOPO=0.762, meaning cross-patient features + S2 adaptation is worse than
training from scratch. If removing S2 improves PER, S2 is the problem.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import prepare
from arch_ablation_base import (
    DEVICE, SEED, SpatialReadIn, Backbone, ArticulatoryBottleneckHead,
    train_stage1, extract_embeddings, knn_predict, simpleshot_predict,
)


def run():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    all_data = prepare.load_all_patients()
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)
    N, H, W, T = grids.shape

    print("=== exp77_no_s2 ===")
    print(f"Target: {prepare.TARGET_PATIENT} | Trials: {N} | NO STAGE 2")

    backbone, head, read_ins = train_stage1(
        all_data, SpatialReadIn, Backbone, ArticulatoryBottleneckHead)

    # Untrained target read-in (random init)
    target_ri = SpatialReadIn(H, W).to(DEVICE)

    # Source embeddings for cross-patient k-NN
    source_embs, source_labs = [], []
    for pid in prepare.SOURCE_PATIENTS:
        emb = extract_embeddings(backbone, read_ins[pid], all_data[pid]["grids"])
        source_embs.append(emb)
        source_labs.extend(all_data[pid]["labels"])
    source_emb = torch.cat(source_embs, dim=0)

    fold_pers = []
    all_preds, all_refs = [], []

    for fi, (tr_idx, va_idx) in enumerate(splits):
        val_emb = extract_embeddings(backbone, target_ri, grids[va_idx])
        train_emb = extract_embeddings(backbone, target_ri, grids[tr_idx])
        train_labels = [labels[i] for i in tr_idx]
        val_labels = [labels[i] for i in va_idx]

        # k-NN with target-only, source-only, and combined
        knn_tgt = prepare.compute_per(knn_predict(train_emb, train_labels, val_emb), val_labels)
        knn_src = prepare.compute_per(knn_predict(source_emb, source_labs, val_emb), val_labels)
        combined_emb = torch.cat([train_emb, source_emb * 0.5], dim=0)
        combined_labs = train_labels + source_labs
        knn_both = prepare.compute_per(knn_predict(combined_emb, combined_labs, val_emb), val_labels)
        ss = prepare.compute_per(simpleshot_predict(train_emb, train_labels, val_emb), val_labels)

        best_per = min(knn_tgt, knn_src, knn_both, ss)
        best_preds = [knn_predict(train_emb, train_labels, val_emb),
                      knn_predict(source_emb, source_labs, val_emb),
                      knn_predict(combined_emb, combined_labs, val_emb),
                      simpleshot_predict(train_emb, train_labels, val_emb)][
            [knn_tgt, knn_src, knn_both, ss].index(best_per)]
        fold_pers.append(best_per)
        all_preds.extend(best_preds)
        all_refs.extend(val_labels)
        print(f"  Fold {fi+1}: tgt_knn={knn_tgt:.4f} src_knn={knn_src:.4f} "
              f"combined={knn_both:.4f} ss={ss:.4f} best={best_per:.4f}")

    mean_per = float(np.mean(fold_pers))
    collapse = prepare.compute_content_collapse(all_preds)
    print(f"\n---\nval_per:            {mean_per:.6f}")
    print(f"val_per_std:        {float(np.std(fold_pers)):.6f}")
    print(f"fold_pers:          {fold_pers}")
    print(f"collapsed:          {collapse['collapsed']}")
    print(f"mean_entropy:       {collapse['mean_entropy']:.3f}")


if __name__ == "__main__":
    run()
