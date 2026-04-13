#!/usr/bin/env python3
"""Exp 66: Supervised Contrastive (SupCon) pretraining on source patients.

Stage 0 (NEW): SupCon pretrain backbone on source patients.
  - Positives: same phoneme at same position from ANY patient
  - Negatives: different phonemes in the batch
  - Mixed-patient batching (3 patients per step)
Stage 1: Standard CE fine-tune with LP-FT (linear probe first, then unfreeze)
Stage 2: Standard target adaptation

Baseline (exp33): CE-only S1, PER=0.762
Change: SupCon pretrain S0 -> LP-FT S1 -> S2
"""
from __future__ import annotations

import math
import os
import sys
import time
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
    augment, SpatialReadIn, Backbone, ArticulatoryBottleneckHead,
    focal_ce, compute_loss, decode, extract_embeddings,
    knn_predict, simpleshot_predict, prototype_predict,
    DEVICE, SEED, S2_EPOCHS, S2_BATCH_SIZE, S2_LR,
    S2_BACKBONE_LR_MULT, S2_READIN_LR_MULT, S2_WEIGHT_DECAY, S2_GRAD_CLIP,
    S2_WARMUP_EPOCHS, S2_PATIENCE, S2_EVAL_EVERY, MIXUP_ALPHA,
    TTA_COPIES, KNN_K, SOURCE_KNN_WEIGHT,
    train_eval_fold as _base_fold,
)

# SupCon hyperparams
SUPCON_EPOCHS = 100
SUPCON_LR = 1e-3
SUPCON_TEMP = 0.07
SUPCON_PROJ_DIM = 64
PATIENTS_PER_STEP = 3

# LP-FT hyperparams
LP_EPOCHS = 50
FT_EPOCHS = 100
FT_BACKBONE_LR_MULT = 0.01


class Projector(nn.Module):
    """MLP projector for contrastive learning."""
    def __init__(self, d_in=64, d_out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(),
            nn.Linear(128, d_out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def supcon_loss(embeddings, labels_pos, temperature=SUPCON_TEMP):
    """Supervised contrastive loss. Labels_pos is per-position phoneme IDs.
    Positives = same phoneme at same position."""
    B = embeddings.shape[0]
    sim = embeddings @ embeddings.T / temperature  # (B, B)
    # Mask: positives share at least one position-phoneme match
    pos_mask = torch.zeros(B, B, device=embeddings.device)
    for i in range(B):
        for j in range(B):
            if i == j:
                continue
            # Positive if any position matches
            for p in range(len(labels_pos[i])):
                if labels_pos[i][p] == labels_pos[j][p]:
                    pos_mask[i, j] = 1.0
                    break

    # If no positives for an anchor, skip it
    has_pos = pos_mask.sum(dim=1) > 0
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    # Log-sum-exp trick
    logits_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - logits_max.detach()

    # Exclude self
    self_mask = 1.0 - torch.eye(B, device=embeddings.device)
    exp_logits = torch.exp(logits) * self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # Mean over positives
    mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
    loss = -mean_log_prob[has_pos].mean()
    return loss


def train_supcon(all_data):
    """Stage 0: SupCon pretraining on source patients."""
    torch.manual_seed(SEED); np.random.seed(SEED)

    read_ins = {}
    for pid in prepare.SOURCE_PATIENTS:
        H, W = all_data[pid]["grid_shape"]
        read_ins[pid] = SpatialReadIn(H, W).to(DEVICE)
    d_flat = list(read_ins.values())[0].d_flat

    backbone = Backbone(d_in=d_flat).to(DEVICE)
    projector = Projector(d_in=backbone.out_dim, d_out=SUPCON_PROJ_DIM).to(DEVICE)

    readin_params = []
    for ri in read_ins.values():
        readin_params.extend(ri.parameters())
    optimizer = AdamW([
        {"params": readin_params, "lr": SUPCON_LR * 3.0},
        {"params": backbone.parameters(), "lr": SUPCON_LR},
        {"params": projector.parameters(), "lr": SUPCON_LR},
    ], weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < 10:
            return (epoch + 1) / 10
        progress = (epoch - 10) / max(SUPCON_EPOCHS - 10, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    pids = prepare.SOURCE_PATIENTS

    for epoch in range(SUPCON_EPOCHS):
        backbone.train(); projector.train()
        for ri in read_ins.values(): ri.train()
        np.random.shuffle(pids)
        epoch_loss = 0.0; n_batches = 0

        # Mixed-patient batching
        for step in range(50):  # ~50 mixed-patient steps per epoch
            selected = np.random.choice(pids, size=min(PATIENTS_PER_STEP, len(pids)), replace=False)
            batch_feats = []
            batch_labels = []
            for pid in selected:
                grids = all_data[pid]["grids"]; labels = all_data[pid]["labels"]
                n = len(labels)
                idx = np.random.choice(n, size=min(6, n), replace=False)
                x = augment(grids[idx]).to(DEVICE)
                feat = read_ins[pid](x)
                h = backbone(feat)
                pooled = h.mean(dim=1)  # (batch, d)
                batch_feats.append(pooled)
                batch_labels.extend([labels[i] for i in idx])

            all_feats = torch.cat(batch_feats, dim=0)  # (B_total, d)
            z = projector(all_feats)  # (B_total, proj_dim)

            loss = supcon_loss(z, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(projector.parameters()) + readin_params, 5.0)
            optimizer.step()
            epoch_loss += loss.item(); n_batches += 1

        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"  SupCon epoch {epoch+1}: loss={epoch_loss/max(n_batches,1):.4f}")

    return backbone, read_ins


def train_stage1_lpft(backbone, read_ins, all_data):
    """Stage 1: LP-FT — linear probe then fine-tune."""
    head = ArticulatoryBottleneckHead(d_in=backbone.out_dim).to(DEVICE)

    source_train, source_val = {}, {}
    for pid in prepare.SOURCE_PATIENTS:
        n = len(all_data[pid]["labels"])
        perm = np.random.permutation(n)
        n_val = max(1, int(round(0.2 * n)))
        source_train[pid] = sorted(perm[n_val:].tolist())
        source_val[pid] = sorted(perm[:n_val].tolist())

    # Phase 1: Linear probe (freeze backbone)
    for p in backbone.parameters(): p.requires_grad = False
    for ri in read_ins.values():
        for p in ri.parameters(): p.requires_grad = False

    optimizer_lp = AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(LP_EPOCHS):
        head.train(); backbone.eval()
        pids = prepare.SOURCE_PATIENTS; np.random.shuffle(pids)
        for pid in pids:
            grids = all_data[pid]["grids"]; labels = all_data[pid]["labels"]
            tr_idx = source_train[pid]
            perm = np.random.permutation(len(tr_idx))
            for start in range(0, len(tr_idx), 16):
                batch_idx = [tr_idx[perm[i]] for i in range(start, min(start + 16, len(tr_idx)))]
                x = augment(grids[batch_idx]).to(DEVICE)
                y = [labels[i] for i in batch_idx]
                optimizer_lp.zero_grad()
                with torch.no_grad():
                    feat = read_ins[pid](x); h = backbone(feat)
                logits = head(h.detach())
                loss = compute_loss(logits, y)
                loss.backward(); optimizer_lp.step()

    # Phase 2: Fine-tune (unfreeze backbone at low LR)
    for p in backbone.parameters(): p.requires_grad = True
    for ri in read_ins.values():
        for p in ri.parameters(): p.requires_grad = True

    readin_params = []
    for ri in read_ins.values(): readin_params.extend(ri.parameters())
    optimizer_ft = AdamW([
        {"params": readin_params, "lr": 1e-3 * 3.0},
        {"params": backbone.parameters(), "lr": 1e-3 * FT_BACKBONE_LR_MULT},
        {"params": head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)

    best_val_loss = float("inf"); best_state = None; patience_ctr = 0
    for epoch in range(FT_EPOCHS):
        backbone.train(); head.train()
        for ri in read_ins.values(): ri.train()
        pids = prepare.SOURCE_PATIENTS; np.random.shuffle(pids)
        for pid in pids:
            grids = all_data[pid]["grids"]; labels = all_data[pid]["labels"]
            tr_idx = source_train[pid]
            perm = np.random.permutation(len(tr_idx))
            for start in range(0, len(tr_idx), 16):
                batch_idx = [tr_idx[perm[i]] for i in range(start, min(start + 16, len(tr_idx)))]
                x = augment(grids[batch_idx]).to(DEVICE)
                y = [labels[i] for i in batch_idx]
                optimizer_ft.zero_grad()
                feat = read_ins[pid](x); h = backbone(feat); logits = head(h)
                loss = compute_loss(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()) + readin_params, 5.0)
                optimizer_ft.step()

        if (epoch + 1) % 10 == 0:
            backbone.eval(); head.eval()
            for ri in read_ins.values(): ri.eval()
            val_loss = 0.0; val_batches = 0
            with torch.no_grad():
                for pid in prepare.SOURCE_PATIENTS:
                    vi = source_val[pid]
                    if not vi: continue
                    x = all_data[pid]["grids"][vi].to(DEVICE)
                    y = [all_data[pid]["labels"][i] for i in vi]
                    feat = read_ins[pid](x); h = backbone(feat); logits = head(h)
                    val_loss += compute_loss(logits, y).item(); val_batches += 1
            val_loss /= max(val_batches, 1)
            print(f"  LP-FT epoch {epoch+1}: val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "backbone": deepcopy(backbone.state_dict()),
                    "head": deepcopy(head.state_dict()),
                    "read_ins": {pid: deepcopy(ri.state_dict()) for pid, ri in read_ins.items()},
                }; patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= 5: break

    if best_state:
        backbone.load_state_dict(best_state["backbone"])
        head.load_state_dict(best_state["head"])
        for pid, ri in read_ins.items():
            if pid in best_state["read_ins"]: ri.load_state_dict(best_state["read_ins"][pid])
    return backbone, head, read_ins


if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)

    all_data = prepare.load_all_patients()
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)

    print(f"=== exp66_supcon_pretrain ===")
    print(f"Stage 0: SupCon pretrain ({SUPCON_EPOCHS} epochs, temp={SUPCON_TEMP})")
    print(f"Stage 1: LP-FT (LP {LP_EPOCHS} epochs, FT {FT_EPOCHS} epochs)")
    print(f"Device: {DEVICE}\n")

    print("=== Stage 0: SupCon pretraining ===")
    s0_start = time.time()
    backbone, read_ins = train_supcon(all_data)
    s0_time = time.time() - s0_start
    print(f"SupCon done in {s0_time:.1f}s\n")

    print("=== Stage 1: LP-FT ===")
    s1_start = time.time()
    backbone, head, read_ins = train_stage1_lpft(backbone, read_ins, all_data)
    s1_time = time.time() - s1_start
    print(f"LP-FT done in {s1_time:.1f}s\n")

    print("=== Stage 2: Target adaptation + evaluation ===")
    fold_pers = []; all_preds = []
    method_pers = {m: [] for m in ["linear", "knn", "simpleshot", "prototype", "simpleshot_src"]}

    for fi, (tr_idx, va_idx) in enumerate(splits):
        ft0 = time.time()
        per, preds, methods = _base_fold(
            backbone, head, grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx],
            read_ins=read_ins, all_data=all_data, ReadInCls=SpatialReadIn)
        fold_pers.append(per); all_preds.extend(preds)
        for m, (p, _) in methods.items(): method_pers[m].append(p)
        best_method = min(methods, key=lambda k: methods[k][0])
        print(f"  Fold {fi+1}: PER={per:.4f} ({best_method}) ({time.time()-ft0:.1f}s)")
        if time.time() - t0 > prepare.TIME_BUDGET: break

    mean_per = float(np.mean(fold_pers))
    collapse = prepare.compute_content_collapse(all_preds)
    print(f"\n---")
    print(f"val_per:            {mean_per:.6f}")
    print(f"val_per_std:        {float(np.std(fold_pers)):.6f}")
    print(f"fold_pers:          {fold_pers}")
    print(f"supcon_seconds:     {s0_time:.1f}")
    print(f"lpft_seconds:       {s1_time:.1f}")
    print(f"training_seconds:   {time.time()-t0:.1f}")
    print(f"collapsed:          {collapse['collapsed']}")
    print(f"mean_entropy:       {collapse['mean_entropy']:.3f}")
    print(f"stereotypy:         {collapse['stereotypy']:.3f}")
    print(f"unique_ratio:       {collapse['unique_ratio']:.3f}")
