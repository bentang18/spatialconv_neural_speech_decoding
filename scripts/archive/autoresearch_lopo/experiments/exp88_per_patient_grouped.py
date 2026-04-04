#!/usr/bin/env python3
"""Exp 88: Per-patient training with grouped-by-token CV (TRUE baseline).

CRITICAL: The 0.700 PER baseline was stratified CV. LOPO uses grouped CV.
These are NOT comparable. This establishes the true per-patient baseline
with the SAME grouped-by-token CV used in all LOPO experiments.

If this gives PER ~0.74, then LOPO 0.762 is only 2pp worse (not 6pp).
If this gives PER ~0.70, then the CV type doesn't matter and LOPO is 6pp worse.
"""
from __future__ import annotations
import math, sys, time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare
from arch_ablation_base import (
    DEVICE, SEED, SpatialReadIn, Backbone, ArticulatoryBottleneckHead,
    augment, compute_loss, extract_embeddings, knn_predict,
    simpleshot_predict, decode, TTA_COPIES, MIXUP_ALPHA,
    S2_EPOCHS, S2_BATCH_SIZE, S2_LR, S2_READIN_LR_MULT,
    S2_WEIGHT_DECAY, S2_GRAD_CLIP, S2_WARMUP_EPOCHS, S2_PATIENCE,
    S2_EVAL_EVERY,
)


def train_per_patient_fold(train_grids, train_labels, val_grids, val_labels):
    """Train from scratch on S14 only (no cross-patient data)."""
    H, W = train_grids.shape[1], train_grids.shape[2]
    readin = SpatialReadIn(H, W).to(DEVICE)
    backbone = Backbone(d_in=readin.d_flat).to(DEVICE)
    head = ArticulatoryBottleneckHead(d_in=backbone.out_dim).to(DEVICE)

    n_train = len(train_grids)
    epochs = 300  # more epochs since training from scratch

    optimizer = AdamW([
        {"params": readin.parameters(), "lr": S2_LR * S2_READIN_LR_MULT},
        {"params": backbone.parameters(), "lr": S2_LR},
        {"params": head.parameters(), "lr": S2_LR},
    ], weight_decay=S2_WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < S2_WARMUP_EPOCHS:
            return (epoch + 1) / S2_WARMUP_EPOCHS
        progress = (epoch - S2_WARMUP_EPOCHS) / max(epochs - S2_WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    best_val_loss, best_state, patience_ctr = float("inf"), None, 0

    for epoch in range(epochs):
        backbone.train(); readin.train(); head.train()
        perm = torch.randperm(n_train)
        for start in range(0, n_train, S2_BATCH_SIZE):
            idx = perm[start:start + S2_BATCH_SIZE]
            x = augment(train_grids[idx]).to(DEVICE)
            y = [train_labels[i] for i in idx.tolist()]
            mixup_y, mixup_lam = None, 1.0
            if MIXUP_ALPHA > 0 and len(idx) > 1:
                mixup_lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm_mix = torch.randperm(x.shape[0])
                x = mixup_lam * x + (1 - mixup_lam) * x[perm_mix]
                mixup_y = [y[i] for i in perm_mix.tolist()]
            optimizer.zero_grad()
            feat = readin(x)
            h = backbone(feat)
            logits = head(h)
            loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(readin.parameters()) + list(head.parameters()),
                S2_GRAD_CLIP)
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % S2_EVAL_EVERY == 0:
            backbone.eval(); readin.eval(); head.eval()
            with torch.no_grad():
                feat = readin(val_grids.to(DEVICE))
                h = backbone(feat)
                logits = head(h)
                val_loss = compute_loss(logits, val_labels).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "backbone": deepcopy(backbone.state_dict()),
                    "readin": deepcopy(readin.state_dict()),
                    "head": deepcopy(head.state_dict()),
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= S2_PATIENCE:
                    break

    if best_state:
        backbone.load_state_dict(best_state["backbone"])
        readin.load_state_dict(best_state["readin"])
        head.load_state_dict(best_state["head"])

    backbone.eval(); readin.eval(); head.eval()

    # Linear + TTA
    with torch.no_grad():
        logits_sum = head(backbone(readin(val_grids.to(DEVICE))))
        for _ in range(TTA_COPIES - 1):
            logits_sum = logits_sum + head(backbone(readin(augment(val_grids).to(DEVICE))))
        logits = logits_sum / TTA_COPIES
    linear_preds = decode(logits)
    linear_per = prepare.compute_per(linear_preds, val_labels)

    # k-NN
    train_emb = extract_embeddings(backbone, readin, train_grids)
    val_emb = extract_embeddings(backbone, readin, val_grids)
    knn_preds = knn_predict(train_emb, list(train_labels), val_emb)
    knn_per = prepare.compute_per(knn_preds, val_labels)

    # SimpleShot
    ss_preds = simpleshot_predict(train_emb, list(train_labels), val_emb)
    ss_per = prepare.compute_per(ss_preds, val_labels)

    best_per = min(linear_per, knn_per, ss_per)
    methods = {"linear": linear_per, "knn": knn_per, "ss": ss_per}
    best_name = min(methods, key=methods.get)
    best_preds = {"linear": linear_preds, "knn": knn_preds, "ss": ss_preds}[best_name]

    return best_per, best_preds, methods


def run():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)

    print("=== exp88_per_patient_grouped ===")
    print(f"Target: {prepare.TARGET_PATIENT} | Per-patient only | Grouped-by-token CV")
    print(f"Trials: {grids.shape[0]} | NO cross-patient data")
    print()

    fold_pers = []
    all_preds, all_refs = [], []

    for fi, (tr_idx, va_idx) in enumerate(splits):
        ft0 = time.time()
        per, preds, methods = train_per_patient_fold(
            grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx])
        fold_pers.append(per)
        all_preds.extend(preds)
        all_refs.extend([labels[i] for i in va_idx])
        best_name = min(methods, key=methods.get)
        print(f"  Fold {fi+1}: PER={per:.4f} (lin={methods['linear']:.4f} "
              f"knn={methods['knn']:.4f} ss={methods['ss']:.4f} "
              f"best={best_name}) ({time.time()-ft0:.1f}s)")

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
