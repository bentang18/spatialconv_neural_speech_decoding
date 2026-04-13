#!/usr/bin/env python3
"""Exp 90: CORAL domain alignment loss during Stage 1.

CORAL (Correlation Alignment) minimizes the distance between
covariance matrices of different patients' features. This is
softer than gradient reversal — doesn't flip gradients, just
adds a penalty for patient-specific covariance structure.

Loss = CE + coral_weight * sum(CORAL(patient_i, patient_j))
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
    augment, compute_loss, train_eval_fold,
    S1_EPOCHS, S1_BATCH_SIZE, S1_LR, S1_READIN_LR_MULT,
    S1_WEIGHT_DECAY, S1_GRAD_CLIP, S1_WARMUP_EPOCHS, S1_PATIENCE,
    S1_EVAL_EVERY, S1_VAL_FRACTION, MIXUP_ALPHA,
)


def coral_loss(feat1, feat2):
    """CORAL: Frobenius norm of covariance matrix difference."""
    d = feat1.shape[1]
    n1, n2 = feat1.shape[0], feat2.shape[0]
    if n1 < 2 or n2 < 2:
        return torch.tensor(0.0, device=feat1.device)

    # Covariance matrices
    mean1 = feat1.mean(dim=0, keepdim=True)
    mean2 = feat2.mean(dim=0, keepdim=True)
    cov1 = ((feat1 - mean1).T @ (feat1 - mean1)) / max(n1 - 1, 1)
    cov2 = ((feat2 - mean2).T @ (feat2 - mean2)) / max(n2 - 1, 1)

    # Frobenius norm squared, normalized
    return ((cov1 - cov2) ** 2).sum() / (4 * d * d)


def train_stage1_coral(all_data, ReadInCls, BackboneCls, HeadCls):
    torch.manual_seed(SEED); np.random.seed(SEED)
    pids = prepare.SOURCE_PATIENTS

    read_ins = {}
    for pid in pids:
        H, W = all_data[pid]["grid_shape"]
        read_ins[pid] = ReadInCls(H, W).to(DEVICE)
    d_flat = list(read_ins.values())[0].d_flat

    backbone = BackboneCls(d_in=d_flat).to(DEVICE)
    head = HeadCls(d_in=backbone.out_dim).to(DEVICE)

    source_train, source_val = {}, {}
    for pid in pids:
        n = len(all_data[pid]["labels"])
        perm = np.random.permutation(n)
        n_val = max(1, int(round(S1_VAL_FRACTION * n)))
        source_train[pid] = sorted(perm[n_val:].tolist())
        source_val[pid] = sorted(perm[:n_val].tolist())

    readin_params = [p for ri in read_ins.values() for p in ri.parameters()]
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
    coral_weight = 0.1

    for epoch in range(S1_EPOCHS):
        backbone.train(); head.train()
        for ri in read_ins.values(): ri.train()
        np.random.shuffle(pids)
        epoch_ce, epoch_coral, n_batches = 0.0, 0.0, 0

        # Collect embeddings from each patient for CORAL
        patient_feats = {}

        for pid in pids:
            grids = all_data[pid]["grids"]
            labels_list = all_data[pid]["labels"]
            tr_idx = source_train[pid]
            perm = np.random.permutation(len(tr_idx))
            for start in range(0, len(tr_idx), S1_BATCH_SIZE):
                batch_idx = [tr_idx[perm[i]] for i in range(start, min(start + S1_BATCH_SIZE, len(tr_idx)))]
                x = augment(grids[batch_idx]).to(DEVICE)
                y = [labels_list[i] for i in batch_idx]
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
                ce_loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)

                # Store mean-pooled features for CORAL
                h_pooled = h.mean(dim=1)
                patient_feats[pid] = h_pooled

                # Compute CORAL loss against other stored patients
                c_loss = torch.tensor(0.0, device=DEVICE)
                n_pairs = 0
                for other_pid, other_feat in patient_feats.items():
                    if other_pid != pid:
                        c_loss = c_loss + coral_loss(h_pooled, other_feat.detach())
                        n_pairs += 1
                if n_pairs > 0:
                    c_loss = c_loss / n_pairs

                loss = ce_loss + coral_weight * c_loss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()) + readin_params,
                    S1_GRAD_CLIP)
                optimizer.step()
                epoch_ce += ce_loss.item()
                epoch_coral += c_loss.item()
                n_batches += 1
        scheduler.step()

        if (epoch + 1) % S1_EVAL_EVERY == 0:
            backbone.eval(); head.eval()
            for ri in read_ins.values(): ri.eval()
            val_loss, val_batches = 0.0, 0
            with torch.no_grad():
                for pid in pids:
                    vi = source_val[pid]
                    if not vi: continue
                    x = all_data[pid]["grids"][vi].to(DEVICE)
                    y = [all_data[pid]["labels"][i] for i in vi]
                    logits = head(backbone(read_ins[pid](x)))
                    val_loss += compute_loss(logits, y).item()
                    val_batches += 1
            val_loss /= max(val_batches, 1)
            print(f"  S1 epoch {epoch+1}: ce={epoch_ce/max(n_batches,1):.4f} "
                  f"coral={epoch_coral/max(n_batches,1):.4f} val={val_loss:.4f}")
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

    print("=== exp90_coral_align ===")
    print(f"Target: {prepare.TARGET_PATIENT} | CORAL domain alignment")

    backbone, head, read_ins = train_stage1_coral(
        all_data, SpatialReadIn, Backbone, ArticulatoryBottleneckHead)

    fold_pers = []
    all_preds, all_refs = [], []
    for fi, (tr_idx, va_idx) in enumerate(splits):
        per, preds, methods = train_eval_fold(
            backbone, head, grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx],
            read_ins=read_ins, all_data=all_data, ReadInCls=SpatialReadIn)
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
