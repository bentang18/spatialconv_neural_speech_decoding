#!/usr/bin/env python3
"""Exp 87: Weight source patients by embedding similarity to target.

Not all source patients are equally useful. Patients with similar
neural representations to S14 should contribute more to training.
We compute similarity based on randomly-initialized backbone embeddings,
then weight each patient's loss contribution proportionally.
"""
from __future__ import annotations
import math, sys, time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare
from arch_ablation_base import (
    DEVICE, SEED, SpatialReadIn, Backbone, ArticulatoryBottleneckHead,
    augment, compute_loss, train_eval_fold, extract_embeddings,
    S1_EPOCHS, S1_BATCH_SIZE, S1_LR, S1_READIN_LR_MULT,
    S1_WEIGHT_DECAY, S1_GRAD_CLIP, S1_WARMUP_EPOCHS, S1_PATIENCE,
    S1_EVAL_EVERY, S1_VAL_FRACTION, MIXUP_ALPHA,
)


def compute_patient_weights(all_data, target_grids, read_ins, backbone):
    """Compute patient weights based on embedding similarity to target."""
    backbone.eval()
    target_ri = SpatialReadIn(target_grids.shape[1], target_grids.shape[2]).to(DEVICE)
    target_emb = extract_embeddings(backbone, target_ri, target_grids)
    target_mean = F.normalize(target_emb.mean(dim=0, keepdim=True), dim=1)

    weights = {}
    for pid in prepare.SOURCE_PATIENTS:
        ri = read_ins[pid]
        ri.eval()
        emb = extract_embeddings(backbone, ri, all_data[pid]["grids"])
        source_mean = F.normalize(emb.mean(dim=0, keepdim=True), dim=1)
        sim = (target_mean @ source_mean.T).item()
        weights[pid] = max(sim, 0.1)  # floor at 0.1

    # Normalize to sum to len(patients)
    total = sum(weights.values())
    n = len(weights)
    for pid in weights:
        weights[pid] = weights[pid] / total * n

    return weights


def train_stage1_weighted(all_data, target_grids, ReadInCls, BackboneCls, HeadCls):
    torch.manual_seed(SEED); np.random.seed(SEED)
    pids = prepare.SOURCE_PATIENTS

    read_ins = {}
    for pid in pids:
        H, W = all_data[pid]["grid_shape"]
        read_ins[pid] = ReadInCls(H, W).to(DEVICE)
    d_flat = list(read_ins.values())[0].d_flat

    backbone = BackboneCls(d_in=d_flat).to(DEVICE)
    head = HeadCls(d_in=backbone.out_dim).to(DEVICE)

    # Compute weights based on initial (random) embeddings
    patient_weights = compute_patient_weights(all_data, target_grids, read_ins, backbone)
    print(f"  Patient weights: {', '.join(f'{p}={w:.2f}' for p, w in patient_weights.items())}")

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

    for epoch in range(S1_EPOCHS):
        backbone.train(); head.train()
        for ri in read_ins.values(): ri.train()
        np.random.shuffle(pids)
        epoch_loss, n_batches = 0.0, 0

        for pid in pids:
            w = patient_weights[pid]
            grids = all_data[pid]["grids"]
            labels = all_data[pid]["labels"]
            tr_idx = source_train[pid]
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
                loss = w * compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)
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
                for pid in pids:
                    vi = source_val[pid]
                    if not vi: continue
                    x = all_data[pid]["grids"][vi].to(DEVICE)
                    y = [all_data[pid]["labels"][i] for i in vi]
                    logits = head(backbone(read_ins[pid](x)))
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

    print("=== exp87_patient_weight ===")
    print(f"Target: {prepare.TARGET_PATIENT} | Patient similarity weighting")

    backbone, head, read_ins = train_stage1_weighted(
        all_data, grids, SpatialReadIn, Backbone, ArticulatoryBottleneckHead)

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
