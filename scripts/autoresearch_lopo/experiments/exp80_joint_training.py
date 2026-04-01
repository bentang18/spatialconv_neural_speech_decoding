#!/usr/bin/env python3
"""Exp 80: Joint training with S14 INCLUDED in Stage 1 (Singh-style).

Spalding's key insight: cross-patient works because source data is aligned
TO the target patient's space. In our pipeline, S1 excludes S14 entirely,
creating a domain gap. Singh 2025 trains ALL patients jointly with per-
patient layers. This exp includes S14 in S1 with its own read-in.

To avoid data leakage: for each CV fold, S14 train-fold samples go into
S1 training; S14 val-fold samples are held out. S1 is retrained per fold.
After S1, evaluate directly (no S2) since S14 already participated in S1.
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
    simpleshot_predict, decode, focal_ce, TTA_COPIES,
    S1_EPOCHS, S1_BATCH_SIZE, S1_LR, S1_READIN_LR_MULT,
    S1_WEIGHT_DECAY, S1_GRAD_CLIP, S1_WARMUP_EPOCHS, S1_PATIENCE,
    S1_EVAL_EVERY, MIXUP_ALPHA, SOURCE_KNN_WEIGHT,
)


def train_stage1_joint(all_data, target_grids, target_labels, target_train_idx):
    """S1 with target patient included. Only target_train_idx used (no leakage)."""
    torch.manual_seed(SEED); np.random.seed(SEED)

    # All patients including target
    all_pids = prepare.SOURCE_PATIENTS + [prepare.TARGET_PATIENT]
    read_ins = {}
    for pid in all_pids:
        if pid == prepare.TARGET_PATIENT:
            H, W = target_grids.shape[1], target_grids.shape[2]
        else:
            H, W = all_data[pid]["grid_shape"]
        read_ins[pid] = SpatialReadIn(H, W).to(DEVICE)
    d_flat = list(read_ins.values())[0].d_flat

    backbone = Backbone(d_in=d_flat).to(DEVICE)
    head = ArticulatoryBottleneckHead(d_in=backbone.out_dim).to(DEVICE)

    # Split source patients into train/val (20% val)
    source_train, source_val = {}, {}
    for pid in prepare.SOURCE_PATIENTS:
        n = len(all_data[pid]["labels"])
        perm = np.random.permutation(n)
        n_val = max(1, int(round(0.2 * n)))
        source_train[pid] = sorted(perm[n_val:].tolist())
        source_val[pid] = sorted(perm[:n_val].tolist())
    # Target patient: only train indices
    source_train[prepare.TARGET_PATIENT] = list(target_train_idx)
    source_val[prepare.TARGET_PATIENT] = []  # no val for target (evaluated separately)

    readin_params = []
    for ri in read_ins.values():
        readin_params.extend(ri.parameters())
    optimizer = AdamW([
        {"params": readin_params, "lr": S1_LR * S1_READIN_LR_MULT},
        {"params": backbone.parameters(), "lr": S1_LR},
        {"params": head.parameters(), "lr": S1_LR},
    ], weight_decay=S1_WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < S1_WARMUP_EPOCHS:
            return (epoch + 1) / S1_WARMUP_EPOCHS
        progress = (epoch - S1_WARMUP_EPOCHS) / max(S1_EPOCHS - S1_WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    best_val_loss, best_state, patience_ctr = float("inf"), None, 0

    # Build data dict that includes target
    joint_data = dict(all_data)
    joint_data[prepare.TARGET_PATIENT] = {
        "grids": target_grids,
        "labels": target_labels,
        "grid_shape": (target_grids.shape[1], target_grids.shape[2]),
    }

    for epoch in range(S1_EPOCHS):
        backbone.train(); head.train()
        for ri in read_ins.values(): ri.train()
        pids_shuffled = list(all_pids)
        np.random.shuffle(pids_shuffled)
        epoch_loss, n_batches = 0.0, 0

        for pid in pids_shuffled:
            grids = joint_data[pid]["grids"]
            labels = joint_data[pid]["labels"]
            tr_idx = source_train[pid]
            if not tr_idx: continue
            perm = np.random.permutation(len(tr_idx))
            for start in range(0, len(tr_idx), S1_BATCH_SIZE):
                batch_idx = [tr_idx[perm[i]] for i in range(start, min(start + S1_BATCH_SIZE, len(tr_idx)))]
                x = augment(grids[batch_idx]).to(DEVICE)
                y = [labels[i] for i in batch_idx]
                mixup_y, mixup_lam = None, 1.0
                if MIXUP_ALPHA > 0 and len(batch_idx) > 1:
                    mixup_lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                    perm_mix = torch.randperm(x.shape[0])
                    x = mixup_lam * x + (1 - mixup_lam) * x[perm_mix]
                    mixup_y = [y[i] for i in perm_mix.tolist()]
                optimizer.zero_grad()
                feat = read_ins[pid](x)
                h = backbone(feat)
                logits = head(h)
                loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()) + readin_params,
                    S1_GRAD_CLIP)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
        scheduler.step()

        if (epoch + 1) % S1_EVAL_EVERY == 0:
            backbone.eval(); head.eval()
            for ri in read_ins.values(): ri.eval()
            val_loss, val_batches = 0.0, 0
            with torch.no_grad():
                for pid in prepare.SOURCE_PATIENTS:  # only source for val
                    vi = source_val[pid]
                    if not vi: continue
                    x = joint_data[pid]["grids"][vi].to(DEVICE)
                    y = [joint_data[pid]["labels"][i] for i in vi]
                    feat = read_ins[pid](x)
                    h = backbone(feat)
                    logits = head(h)
                    val_loss += compute_loss(logits, y).item()
                    val_batches += 1
            val_loss /= max(val_batches, 1)
            print(f"  S1 epoch {epoch+1}: train={epoch_loss/max(n_batches,1):.4f} val={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "backbone": deepcopy(backbone.state_dict()),
                    "head": deepcopy(head.state_dict()),
                    "read_ins": {pid: deepcopy(ri.state_dict()) for pid, ri in read_ins.items()},
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= S1_PATIENCE:
                    print(f"  S1 early stop at epoch {epoch+1}")
                    break

    if best_state:
        backbone.load_state_dict(best_state["backbone"])
        head.load_state_dict(best_state["head"])
        for pid, ri in read_ins.items():
            if pid in best_state["read_ins"]:
                ri.load_state_dict(best_state["read_ins"][pid])
    return backbone, head, read_ins


def run():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    all_data = prepare.load_all_patients()
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)
    N = grids.shape[0]

    print("=== exp80_joint_training ===")
    print(f"Target: {prepare.TARGET_PATIENT} | Trials: {N} | S14 IN Stage 1")

    fold_pers = []
    all_preds, all_refs = [], []

    for fi, (tr_idx, va_idx) in enumerate(splits):
        ft0 = time.time()
        print(f"\n  Fold {fi+1}/{len(splits)}: training S1 with S14 train fold ({len(tr_idx)} samples)...")

        backbone, head, read_ins = train_stage1_joint(
            all_data, grids, labels, tr_idx)

        # Evaluate directly — no S2 needed since S14 was in S1
        target_ri = read_ins[prepare.TARGET_PATIENT]
        backbone.eval(); target_ri.eval(); head.eval()

        val_grids = grids[va_idx]
        val_labels = [labels[i] for i in va_idx]

        # Linear + TTA
        with torch.no_grad():
            logits_sum = head(backbone(target_ri(val_grids.to(DEVICE))))
            for _ in range(TTA_COPIES - 1):
                logits_sum = logits_sum + head(backbone(target_ri(augment(val_grids).to(DEVICE))))
            logits = logits_sum / TTA_COPIES
        linear_preds = (logits.argmax(dim=-1) + 1).cpu().tolist()
        linear_per = prepare.compute_per(linear_preds, val_labels)

        # k-NN with target train embeddings
        train_emb = extract_embeddings(backbone, target_ri, grids[tr_idx])
        train_labels = [labels[i] for i in tr_idx]
        val_emb = extract_embeddings(backbone, target_ri, val_grids)
        knn_preds = knn_predict(train_emb, train_labels, val_emb)
        knn_per = prepare.compute_per(knn_preds, val_labels)

        # k-NN with source embeddings too
        source_embs, source_labs = [], []
        for pid in prepare.SOURCE_PATIENTS:
            emb = extract_embeddings(backbone, read_ins[pid], all_data[pid]["grids"])
            source_embs.append(emb)
            source_labs.extend(all_data[pid]["labels"])
        source_emb = torch.cat(source_embs, dim=0)
        combined_emb = torch.cat([train_emb, source_emb * SOURCE_KNN_WEIGHT], dim=0)
        combined_labs = train_labels + source_labs
        knn_combined = knn_predict(combined_emb, combined_labs, val_emb)
        knn_combined_per = prepare.compute_per(knn_combined, val_labels)

        # SimpleShot
        ss_preds = simpleshot_predict(train_emb, train_labels, val_emb)
        ss_per = prepare.compute_per(ss_preds, val_labels)

        best_per = min(linear_per, knn_per, knn_combined_per, ss_per)
        methods = {"linear": linear_per, "knn": knn_per, "combined": knn_combined_per, "ss": ss_per}
        best_name = min(methods, key=methods.get)
        best_preds_map = {"linear": linear_preds, "knn": knn_preds,
                          "combined": knn_combined, "ss": ss_preds}
        fold_pers.append(best_per)
        all_preds.extend(best_preds_map[best_name])
        all_refs.extend(val_labels)

        print(f"  Fold {fi+1}: lin={linear_per:.4f} knn={knn_per:.4f} "
              f"combined={knn_combined_per:.4f} ss={ss_per:.4f} "
              f"best={best_per:.4f}({best_name}) ({time.time()-ft0:.1f}s)")
        if time.time() - t0 > 1800:  # 30 min budget (5x S1)
            print("WARNING: time budget exceeded")
            break

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
