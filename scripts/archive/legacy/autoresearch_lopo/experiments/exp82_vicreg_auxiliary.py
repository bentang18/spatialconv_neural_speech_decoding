#!/usr/bin/env python3
"""Exp 82: VICReg auxiliary loss during Stage 1.

VICReg (Variance-Invariance-Covariance regularization) encourages:
  - Variance: each feature dimension has high variance (prevents collapse)
  - Invariance: augmented views have similar representations
  - Covariance: decorrelate feature dimensions (reduce redundancy)

Added as auxiliary to CE loss during S1. NOT standalone SSL — this is
supervised CE + self-supervised VICReg on the same backbone features.
Previous SSL attempts failed as standalone; this tests SSL as regularizer.
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
    augment, compute_loss, train_eval_fold,
    S1_EPOCHS, S1_BATCH_SIZE, S1_LR, S1_READIN_LR_MULT,
    S1_WEIGHT_DECAY, S1_GRAD_CLIP, S1_WARMUP_EPOCHS, S1_PATIENCE,
    S1_EVAL_EVERY, S1_VAL_FRACTION, MIXUP_ALPHA,
)


class VICRegProjector(nn.Module):
    def __init__(self, d_in=64, d_proj=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_proj), nn.BatchNorm1d(d_proj), nn.ReLU(),
            nn.Linear(d_proj, d_proj),
        )

    def forward(self, x):
        return self.net(x)


def vicreg_loss(z1, z2, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    """VICReg loss between two batches of projections."""
    # Invariance
    sim_loss = F.mse_loss(z1, z2)

    # Variance
    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std1)) + torch.mean(F.relu(1.0 - std2))

    # Covariance
    B, D = z1.shape
    z1c = z1 - z1.mean(dim=0)
    z2c = z2 - z2.mean(dim=0)
    cov1 = (z1c.T @ z1c) / max(B - 1, 1)
    cov2 = (z2c.T @ z2c) / max(B - 1, 1)
    # Off-diagonal elements
    mask = ~torch.eye(D, dtype=torch.bool, device=z1.device)
    cov_loss = (cov1[mask].pow(2).sum() + cov2[mask].pow(2).sum()) / D

    return sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss


def train_stage1_vicreg(all_data, ReadInCls, BackboneCls, HeadCls):
    torch.manual_seed(SEED); np.random.seed(SEED)
    pids = prepare.SOURCE_PATIENTS

    read_ins = {}
    for pid in pids:
        H, W = all_data[pid]["grid_shape"]
        read_ins[pid] = ReadInCls(H, W).to(DEVICE)
    d_flat = list(read_ins.values())[0].d_flat

    backbone = BackboneCls(d_in=d_flat).to(DEVICE)
    head = HeadCls(d_in=backbone.out_dim).to(DEVICE)
    projector = VICRegProjector(d_in=backbone.out_dim, d_proj=128).to(DEVICE)

    source_train, source_val = {}, {}
    for pid in pids:
        n = len(all_data[pid]["labels"])
        perm = np.random.permutation(n)
        n_val = max(1, int(round(S1_VAL_FRACTION * n)))
        source_train[pid] = sorted(perm[n_val:].tolist())
        source_val[pid] = sorted(perm[:n_val].tolist())

    readin_params = []
    for ri in read_ins.values():
        readin_params.extend(ri.parameters())
    optimizer = AdamW([
        {"params": readin_params, "lr": S1_LR * S1_READIN_LR_MULT},
        {"params": backbone.parameters(), "lr": S1_LR},
        {"params": head.parameters(), "lr": S1_LR},
        {"params": projector.parameters(), "lr": S1_LR},
    ], weight_decay=S1_WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < S1_WARMUP_EPOCHS:
            return (epoch + 1) / S1_WARMUP_EPOCHS
        progress = (epoch - S1_WARMUP_EPOCHS) / max(S1_EPOCHS - S1_WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    best_val_loss, best_state, patience_ctr = float("inf"), None, 0
    vicreg_weight = 0.1  # auxiliary weight

    for epoch in range(S1_EPOCHS):
        backbone.train(); head.train(); projector.train()
        for ri in read_ins.values(): ri.train()
        np.random.shuffle(pids)
        epoch_ce, epoch_vic, n_batches = 0.0, 0.0, 0

        for pid in pids:
            grids = all_data[pid]["grids"]
            labels_list = all_data[pid]["labels"]
            tr_idx = source_train[pid]
            perm = np.random.permutation(len(tr_idx))
            for start in range(0, len(tr_idx), S1_BATCH_SIZE):
                batch_idx = [tr_idx[perm[i]] for i in range(start, min(start + S1_BATCH_SIZE, len(tr_idx)))]
                raw_grids = grids[batch_idx]
                x1 = augment(raw_grids).to(DEVICE)
                x2 = augment(raw_grids).to(DEVICE)  # second augmented view
                y = [labels_list[i] for i in batch_idx]

                optimizer.zero_grad()

                # CE path (with mixup on first view)
                mixup_y, mixup_lam = None, 1.0
                if MIXUP_ALPHA > 0 and len(batch_idx) > 1:
                    mixup_lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                    perm_mix = torch.randperm(x1.shape[0])
                    x1_mix = mixup_lam * x1 + (1 - mixup_lam) * x1[perm_mix]
                    mixup_y = [y[i] for i in perm_mix.tolist()]
                else:
                    x1_mix = x1

                feat1 = read_ins[pid](x1_mix)
                h1 = backbone(feat1)
                logits = head(h1)
                ce_loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)

                # VICReg path (no mixup, both views)
                with torch.no_grad():
                    feat1_clean = read_ins[pid](x1)
                feat2 = read_ins[pid](x2)
                h1_clean = backbone(feat1_clean) if MIXUP_ALPHA > 0 else h1
                h2 = backbone(feat2)
                z1 = projector(h1_clean.mean(dim=1))
                z2 = projector(h2.mean(dim=1))

                if z1.shape[0] > 2:  # need enough samples for covariance
                    vic_loss = vicreg_loss(z1, z2)
                else:
                    vic_loss = torch.tensor(0.0, device=DEVICE)

                loss = ce_loss + vicreg_weight * vic_loss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()) +
                    readin_params + list(projector.parameters()), S1_GRAD_CLIP)
                optimizer.step()
                epoch_ce += ce_loss.item()
                epoch_vic += vic_loss.item()
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
                  f"vic={epoch_vic/max(n_batches,1):.4f} val={val_loss:.4f}")
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

    print("=== exp82_vicreg_auxiliary ===")
    print(f"Target: {prepare.TARGET_PATIENT} | VICReg aux during S1")

    backbone, head, read_ins = train_stage1_vicreg(
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
