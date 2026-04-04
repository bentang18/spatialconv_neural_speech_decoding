#!/usr/bin/env python3
"""Exp 64: Articulatory feature prediction as auxiliary loss in Stage 1.

Add an explicit articulatory feature prediction head (15-dim BCE) alongside
the articulatory bottleneck classification head. The auxiliary head directly
predicts articulatory features from the backbone, providing a dense,
structured, patient-invariant supervision signal.

Unlike the bottleneck head (which routes through articulatory space for
classification), this auxiliary head provides a separate gradient path
that explicitly rewards articulatory feature encoding.

Baseline (exp33): CE via articulatory bottleneck only, PER=0.762
Change: CE + alpha * BCE_articulatory, alpha=0.3
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

DEVICE = os.environ.get("DEVICE", "mps")
SEED = 42

S1_EPOCHS = 200; S1_BATCH_SIZE = 16; S1_LR = 1e-3; S1_READIN_LR_MULT = 3.0
S1_WEIGHT_DECAY = 1e-4; S1_GRAD_CLIP = 5.0; S1_WARMUP_EPOCHS = 20
S1_PATIENCE = 7; S1_EVAL_EVERY = 10; S1_VAL_FRACTION = 0.2

S2_EPOCHS = 150; S2_BATCH_SIZE = 16; S2_LR = 1e-3; S2_BACKBONE_LR_MULT = 0.1
S2_READIN_LR_MULT = 3.0; S2_WEIGHT_DECAY = 1e-4; S2_GRAD_CLIP = 5.0
S2_WARMUP_EPOCHS = 10; S2_PATIENCE = 7; S2_EVAL_EVERY = 5

LABEL_SMOOTHING = 0.1; FOCAL_GAMMA = 2.0; MIXUP_ALPHA = 0.2
TTA_COPIES = 16; KNN_K = 10; SOURCE_KNN_WEIGHT = 0.5

ARTIC_ALPHA = 0.3  # weight for articulatory auxiliary loss

from arch_ablation_base import (
    augment, SpatialReadIn, Backbone, ArticulatoryBottleneckHead,
    ARTICULATORY_MATRIX, focal_ce, compute_loss, decode,
    extract_embeddings, knn_predict, simpleshot_predict, prototype_predict,
)


class ArticulatoryAuxHead(nn.Module):
    """Auxiliary head: predict 15-dim articulatory features via BCE."""
    def __init__(self, d_in=64, n_positions=3, n_features=15):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(d_in, n_features) for _ in range(n_positions)
        ])
        self.register_buffer('A', torch.tensor(ARTICULATORY_MATRIX, dtype=torch.float32))

    def forward(self, h, labels):
        """Returns BCE loss for articulatory feature prediction."""
        pooled = h.mean(dim=1)
        loss = torch.tensor(0.0, device=h.device)
        for pos_idx, head in enumerate(self.heads):
            pred = head(pooled)  # (B, 15)
            # Target: articulatory features for the phoneme at this position
            targets = torch.stack([self.A[l[pos_idx] - 1] for l in labels]).to(h.device)
            loss = loss + F.binary_cross_entropy_with_logits(pred, targets)
        return loss / len(self.heads)


# Use the base training infrastructure but add aux head
# This requires a custom training loop, so inline it

def train_stage1(all_data):
    torch.manual_seed(SEED); np.random.seed(SEED)
    read_ins = {}
    for pid in prepare.SOURCE_PATIENTS:
        H, W = all_data[pid]["grid_shape"]
        read_ins[pid] = SpatialReadIn(H, W).to(DEVICE)
    d_flat = list(read_ins.values())[0].d_flat

    backbone = Backbone(d_in=d_flat).to(DEVICE)
    head = ArticulatoryBottleneckHead(d_in=backbone.out_dim).to(DEVICE)
    aux_head = ArticulatoryAuxHead(d_in=backbone.out_dim).to(DEVICE)

    source_train, source_val = {}, {}
    for pid in prepare.SOURCE_PATIENTS:
        n = len(all_data[pid]["labels"])
        perm = np.random.permutation(n)
        n_val = max(1, int(round(S1_VAL_FRACTION * n)))
        source_train[pid] = sorted(perm[n_val:].tolist())
        source_val[pid] = sorted(perm[:n_val].tolist())

    readin_params = []
    for ri in read_ins.values(): readin_params.extend(ri.parameters())
    optimizer = AdamW([
        {"params": readin_params, "lr": S1_LR * S1_READIN_LR_MULT},
        {"params": backbone.parameters(), "lr": S1_LR},
        {"params": head.parameters(), "lr": S1_LR},
        {"params": aux_head.parameters(), "lr": S1_LR},
    ], weight_decay=S1_WEIGHT_DECAY)

    def lr_lambda(epoch):
        if S1_WARMUP_EPOCHS > 0 and epoch < S1_WARMUP_EPOCHS:
            return (epoch + 1) / S1_WARMUP_EPOCHS
        progress = (epoch - S1_WARMUP_EPOCHS) / max(S1_EPOCHS - S1_WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    best_val_loss = float("inf"); best_state = None; patience_ctr = 0
    pids = prepare.SOURCE_PATIENTS

    for epoch in range(S1_EPOCHS):
        backbone.train(); head.train(); aux_head.train()
        for ri in read_ins.values(): ri.train()
        np.random.shuffle(pids)
        epoch_ce = 0.0; epoch_aux = 0.0; n_batches = 0

        for pid in pids:
            grids = all_data[pid]["grids"]; labels = all_data[pid]["labels"]
            tr_idx = source_train[pid]
            perm = np.random.permutation(len(tr_idx))
            for start in range(0, len(tr_idx), S1_BATCH_SIZE):
                batch_idx = [tr_idx[perm[i]] for i in range(start, min(start + S1_BATCH_SIZE, len(tr_idx)))]
                x = augment(grids[batch_idx]).to(DEVICE)
                y = [labels[i] for i in batch_idx]
                # No mixup for aux loss (needs clean labels)
                optimizer.zero_grad()
                feat = read_ins[pid](x); h = backbone(feat)
                logits = head(h)
                ce_loss = compute_loss(logits, y)
                aux_loss = aux_head(h, y)
                loss = ce_loss + ARTIC_ALPHA * aux_loss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()) +
                    list(aux_head.parameters()) + readin_params, S1_GRAD_CLIP)
                optimizer.step()
                epoch_ce += ce_loss.item(); epoch_aux += aux_loss.item(); n_batches += 1
        scheduler.step()

        if (epoch + 1) % S1_EVAL_EVERY == 0:
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
            print(f"  S1 epoch {epoch+1}: ce={epoch_ce/max(n_batches,1):.4f}, "
                  f"aux={epoch_aux/max(n_batches,1):.4f}, val={val_loss:.4f}")
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
                    print(f"  S1 early stop at epoch {epoch+1}"); break

    if best_state:
        backbone.load_state_dict(best_state["backbone"])
        head.load_state_dict(best_state["head"])
        for pid, ri in read_ins.items():
            if pid in best_state["read_ins"]: ri.load_state_dict(best_state["read_ins"][pid])
    return backbone, head, read_ins


# Stage 2 + eval: use standard from arch_ablation_base (no aux in S2)
from arch_ablation_base import train_eval_fold as _base_train_eval_fold


if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)

    all_data = prepare.load_all_patients()
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)

    N, H, W, T = grids.shape
    total_source = sum(len(all_data[p]["labels"]) for p in prepare.SOURCE_PATIENTS)
    print(f"=== exp64_articulatory_auxiliary ===")
    print(f"Target: {prepare.TARGET_PATIENT}  |  Trials: {N}  |  Source: {total_source}")
    print(f"CHANGE: CE + {ARTIC_ALPHA} * BCE_articulatory auxiliary")
    print(f"Device: {DEVICE}\n")

    print("=== Stage 1: Source training with articulatory auxiliary ===")
    s1_start = time.time()
    backbone, head, read_ins = train_stage1(all_data)
    s1_time = time.time() - s1_start
    print(f"Stage 1 done in {s1_time:.1f}s\n")

    print("=== Stage 2: Target adaptation + evaluation ===")
    fold_pers = []; all_preds = []; all_refs = []
    method_pers = {m: [] for m in ["linear", "knn", "simpleshot", "prototype", "simpleshot_src"]}

    for fi, (tr_idx, va_idx) in enumerate(splits):
        ft0 = time.time()
        per, preds, methods = _base_train_eval_fold(
            backbone, head, grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx],
            read_ins=read_ins, all_data=all_data, ReadInCls=SpatialReadIn)
        fold_pers.append(per); all_preds.extend(preds)
        all_refs.extend([labels[i] for i in va_idx])
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
    print(f"stage1_seconds:     {s1_time:.1f}")
    print(f"training_seconds:   {time.time()-t0:.1f}")
    print(f"collapsed:          {collapse['collapsed']}")
    print(f"mean_entropy:       {collapse['mean_entropy']:.3f}")
    print(f"stereotypy:         {collapse['stereotypy']:.3f}")
    print(f"unique_ratio:       {collapse['unique_ratio']:.3f}")
