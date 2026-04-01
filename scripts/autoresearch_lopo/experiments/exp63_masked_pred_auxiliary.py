#!/usr/bin/env python3
"""Exp 63: Masked temporal prediction as auxiliary loss during Stage 1.

Add a reconstruction head that predicts masked temporal frames from context.
The backbone simultaneously learns phoneme discrimination (CE) and temporal
dynamics prediction (MSE on masked positions). The reconstruction objective
regularizes toward patient-invariant temporal dynamics.

Uses existing masking approach: 40-60% of temporal frames masked after Conv1d.
Reconstruction target: Conv1d output features at masked positions.

Baseline (exp33): CE-only, PER=0.762
Change: CE + lambda * MSE_masked, lambda anneals 1.0 -> 0.0
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

# Hyperparams (identical to baseline)
S1_EPOCHS = 200
S1_BATCH_SIZE = 16
S1_LR = 1e-3
S1_READIN_LR_MULT = 3.0
S1_WEIGHT_DECAY = 1e-4
S1_GRAD_CLIP = 5.0
S1_WARMUP_EPOCHS = 20
S1_PATIENCE = 7
S1_EVAL_EVERY = 10
S1_VAL_FRACTION = 0.2

S2_EPOCHS = 150
S2_BATCH_SIZE = 16
S2_LR = 1e-3
S2_BACKBONE_LR_MULT = 0.1
S2_READIN_LR_MULT = 3.0
S2_WEIGHT_DECAY = 1e-4
S2_GRAD_CLIP = 5.0
S2_WARMUP_EPOCHS = 10
S2_PATIENCE = 7
S2_EVAL_EVERY = 5

LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0
MIXUP_ALPHA = 0.2
TTA_COPIES = 16
KNN_K = 10
SOURCE_KNN_WEIGHT = 0.5

# SSL-specific
MASK_RATIO = 0.5
RECON_LAMBDA_START = 1.0
RECON_LAMBDA_END = 0.0

# Import shared components
from arch_ablation_base import (
    augment, SpatialReadIn, ArticulatoryBottleneckHead, ARTICULATORY_MATRIX,
    focal_ce, compute_loss, decode, extract_embeddings,
    knn_predict, simpleshot_predict, prototype_predict,
)


class MaskedBackbone(nn.Module):
    """Backbone with optional masked prediction. When mask=True, returns
    both GRU output and reconstruction targets/predictions for masked positions."""

    def __init__(self, d_in=256, d=32, gru_hidden=32, gru_layers=2,
                 stride=10, gru_dropout=0.3, feat_drop_max=0.3,
                 time_mask_min=2, time_mask_max=5):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_in, d, kernel_size=stride, stride=stride), nn.GELU(),
        )
        self.gru = nn.GRU(
            d, gru_hidden, num_layers=gru_layers, batch_first=True,
            bidirectional=True, dropout=gru_dropout if gru_layers > 1 else 0.0,
        )
        self.out_dim = gru_hidden * 2
        self.d_conv = d
        self.feat_drop_max = feat_drop_max
        self.time_mask_min = time_mask_min
        self.time_mask_max = time_mask_max

        # Reconstruction decoder: GRU output -> Conv1d feature space
        self.recon_head = nn.Linear(gru_hidden * 2, d)
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, x, do_mask=False):
        x = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.training and self.feat_drop_max > 0:
            p = torch.rand(1).item() * self.feat_drop_max
            mask = (torch.rand(x.shape[1], device=x.device) > p).float()
            x = x * mask.unsqueeze(0).unsqueeze(-1) / (1 - p + 1e-8)

        # Conv1d features (reconstruction targets)
        conv_out = self.temporal_conv(x)  # (B, d, T')
        conv_features = conv_out.permute(0, 2, 1)  # (B, T', d)
        T = conv_features.shape[1]

        if do_mask and self.training:
            # Generate random mask (True = masked)
            n_mask = max(1, int(T * MASK_RATIO))
            mask_indices = torch.randperm(T, device=conv_features.device)[:n_mask]
            span_mask = torch.zeros(T, dtype=torch.bool, device=conv_features.device)
            span_mask[mask_indices] = True

            # Save targets before masking
            targets = conv_features[:, span_mask, :].detach()  # (B, n_mask, d)

            # Replace masked positions with learnable token
            masked_input = conv_features.clone()
            masked_input[:, span_mask, :] = self.mask_token.unsqueeze(0).expand(
                conv_features.shape[0], -1, -1)[:, :n_mask, :]

            # GRU on masked input
            if self.training and self.time_mask_max > 0:
                ml = torch.randint(self.time_mask_min, self.time_mask_max + 1, (1,)).item()
                st = torch.randint(0, max(T - ml, 1), (1,)).item()
                taper = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, ml, device=x.device)))
                masked_input = masked_input.clone()
                masked_input[:, st:st + ml, :] *= (1 - taper).unsqueeze(-1)

            h, _ = self.gru(masked_input)

            # Reconstruct masked positions
            preds = self.recon_head(h[:, span_mask, :])  # (B, n_mask, d)
            recon_loss = F.mse_loss(preds, targets)

            return h, recon_loss
        else:
            # Normal forward (no masking)
            gru_input = conv_features
            if self.training and self.time_mask_max > 0:
                ml = torch.randint(self.time_mask_min, self.time_mask_max + 1, (1,)).item()
                st = torch.randint(0, max(T - ml, 1), (1,)).item()
                taper = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, ml, device=x.device)))
                gru_input = gru_input.clone()
                gru_input[:, st:st + ml, :] *= (1 - taper).unsqueeze(-1)
            h, _ = self.gru(gru_input)
            return h, torch.tensor(0.0, device=x.device)


def train_stage1(all_data):
    torch.manual_seed(SEED); np.random.seed(SEED)
    read_ins = {}
    for pid in prepare.SOURCE_PATIENTS:
        H, W = all_data[pid]["grid_shape"]
        read_ins[pid] = SpatialReadIn(H, W).to(DEVICE)
    d_flat = list(read_ins.values())[0].d_flat

    backbone = MaskedBackbone(d_in=d_flat).to(DEVICE)
    head = ArticulatoryBottleneckHead(d_in=backbone.out_dim).to(DEVICE)

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
        backbone.train(); head.train()
        for ri in read_ins.values(): ri.train()
        np.random.shuffle(pids)
        epoch_loss = 0.0; epoch_recon = 0.0; n_batches = 0

        # Anneal reconstruction lambda
        recon_lambda = RECON_LAMBDA_START + (RECON_LAMBDA_END - RECON_LAMBDA_START) * (epoch / S1_EPOCHS)

        for pid in pids:
            grids = all_data[pid]["grids"]; labels = all_data[pid]["labels"]
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
                h, recon_loss = backbone(feat, do_mask=True)
                logits = head(h)
                ce_loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)
                loss = ce_loss + recon_lambda * recon_loss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()) + readin_params, S1_GRAD_CLIP)
                optimizer.step()
                epoch_loss += ce_loss.item(); epoch_recon += recon_loss.item(); n_batches += 1
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
                    feat = read_ins[pid](x)
                    h, _ = backbone(feat, do_mask=False)
                    logits = head(h)
                    val_loss += compute_loss(logits, y).item(); val_batches += 1
            val_loss /= max(val_batches, 1)
            print(f"  S1 epoch {epoch+1}: ce={epoch_loss/max(n_batches,1):.4f}, "
                  f"recon={epoch_recon/max(n_batches,1):.4f}, val={val_loss:.4f}, lam={recon_lambda:.3f}")
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


def train_eval_fold(backbone, head_init, train_grids, train_labels,
                    val_grids, val_labels, read_ins, all_data):
    """Stage 2 — no masking, standard CE adaptation."""
    backbone_copy = deepcopy(backbone)
    target_ri = SpatialReadIn(
        prepare.PATIENT_GRIDS[prepare.TARGET_PATIENT][0],
        prepare.PATIENT_GRIDS[prepare.TARGET_PATIENT][1]).to(DEVICE)
    head = deepcopy(head_init).to(DEVICE)

    n_train = len(train_grids)
    optimizer = AdamW([
        {"params": target_ri.parameters(), "lr": S2_LR * S2_READIN_LR_MULT},
        {"params": backbone_copy.parameters(), "lr": S2_LR * S2_BACKBONE_LR_MULT},
        {"params": head.parameters(), "lr": S2_LR},
    ], weight_decay=S2_WEIGHT_DECAY)

    def lr_lambda(epoch):
        if S2_WARMUP_EPOCHS > 0 and epoch < S2_WARMUP_EPOCHS:
            return (epoch + 1) / S2_WARMUP_EPOCHS
        progress = (epoch - S2_WARMUP_EPOCHS) / max(S2_EPOCHS - S2_WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    best_val_loss = float("inf"); best_state = None; patience_ctr = 0

    for epoch in range(S2_EPOCHS):
        backbone_copy.train(); target_ri.train(); head.train()
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
            feat = target_ri(x)
            h, _ = backbone_copy(feat, do_mask=False)
            logits = head(h)
            loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)
            if math.isnan(loss.item()): print("FAIL"); sys.exit(1)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(backbone_copy.parameters()) + list(target_ri.parameters()) + list(head.parameters()),
                S2_GRAD_CLIP)
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % S2_EVAL_EVERY == 0:
            backbone_copy.eval(); target_ri.eval(); head.eval()
            with torch.no_grad():
                feat = target_ri(val_grids.to(DEVICE))
                h, _ = backbone_copy(feat, do_mask=False)
                logits = head(h)
                val_loss = compute_loss(logits, val_labels).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "backbone": deepcopy(backbone_copy.state_dict()),
                    "readin": deepcopy(target_ri.state_dict()),
                    "head": deepcopy(head.state_dict()),
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= S2_PATIENCE: break

    if best_state:
        backbone_copy.load_state_dict(best_state["backbone"])
        target_ri.load_state_dict(best_state["readin"])
        head.load_state_dict(best_state["head"])

    backbone_copy.eval(); target_ri.eval(); head.eval()

    def _extract(bb, ri, g):
        bb.eval(); ri.eval()
        with torch.no_grad():
            x = ri(g.to(DEVICE))
            h, _ = bb(x, do_mask=False)
            return h.mean(dim=1).cpu()

    with torch.no_grad():
        if TTA_COPIES > 1:
            logits_sum = torch.zeros(len(val_grids), prepare.N_POSITIONS, prepare.N_CLASSES, device=DEVICE)
            feat = target_ri(val_grids.to(DEVICE))
            h, _ = backbone_copy(feat, do_mask=False)
            logits_sum += head(h)
            for _ in range(TTA_COPIES - 1):
                feat = target_ri(augment(val_grids).to(DEVICE))
                h, _ = backbone_copy(feat, do_mask=False)
                logits_sum += head(h)
            logits = logits_sum / TTA_COPIES
        else:
            feat = target_ri(val_grids.to(DEVICE))
            h, _ = backbone_copy(feat, do_mask=False)
            logits = head(h)
    linear_preds = decode(logits)
    linear_per = prepare.compute_per(linear_preds, val_labels)

    train_emb = _extract(backbone_copy, target_ri, train_grids)
    train_emb_all = train_emb
    train_labels_all = list(train_labels)
    if read_ins is not None and all_data is not None:
        source_embs, source_labs = [], []
        for pid in prepare.SOURCE_PATIENTS:
            ri = read_ins[pid]; ri.eval()
            emb = _extract(backbone_copy, ri, all_data[pid]["grids"])
            source_embs.append(emb); source_labs.extend(all_data[pid]["labels"])
        source_emb = torch.cat(source_embs, dim=0)
        train_emb_all = torch.cat([train_emb, source_emb * SOURCE_KNN_WEIGHT], dim=0)
        train_labels_all = list(train_labels) + source_labs

    if TTA_COPIES > 1:
        val_emb_sum = _extract(backbone_copy, target_ri, val_grids)
        for _ in range(TTA_COPIES - 1):
            val_emb_sum = val_emb_sum + _extract(backbone_copy, target_ri, augment(val_grids))
        val_emb = val_emb_sum / TTA_COPIES
    else:
        val_emb = _extract(backbone_copy, target_ri, val_grids)

    knn_preds = knn_predict(train_emb_all, train_labels_all, val_emb)
    knn_per = prepare.compute_per(knn_preds, val_labels)
    ss_preds = simpleshot_predict(train_emb, list(train_labels), val_emb)
    ss_per = prepare.compute_per(ss_preds, val_labels)
    proto_preds = prototype_predict(train_emb, list(train_labels), val_emb)
    proto_per = prepare.compute_per(proto_preds, val_labels)

    methods = {"linear": (linear_per, linear_preds), "knn": (knn_per, knn_preds),
               "simpleshot": (ss_per, ss_preds), "prototype": (proto_per, proto_preds)}
    best_name = min(methods, key=lambda k: methods[k][0])
    best_per, best_preds = methods[best_name]
    return best_per, best_preds, methods


if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)

    all_data = prepare.load_all_patients()
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)

    N, H, W, T = grids.shape
    total_source = sum(len(all_data[p]["labels"]) for p in prepare.SOURCE_PATIENTS)
    print(f"=== exp63_masked_pred_auxiliary ===")
    print(f"Target: {prepare.TARGET_PATIENT}  |  Trials: {N}  |  Grid: {H}x{W}  |  T: {T}")
    print(f"Source: {len(prepare.SOURCE_PATIENTS)} patients, {total_source} trials")
    print(f"CHANGE: CE + annealed masked prediction auxiliary loss (lambda {RECON_LAMBDA_START}->{RECON_LAMBDA_END})")
    print(f"Device: {DEVICE}")
    print()

    print("=== Stage 1: Source training with masked prediction ===")
    s1_start = time.time()
    backbone, head, read_ins = train_stage1(all_data)
    s1_time = time.time() - s1_start
    print(f"Stage 1 done in {s1_time:.1f}s\n")

    print("=== Stage 2: Target adaptation + evaluation ===")
    fold_pers = []; all_preds = []; all_refs = []
    method_pers = {m: [] for m in ["linear", "knn", "simpleshot", "prototype"]}

    for fi, (tr_idx, va_idx) in enumerate(splits):
        ft0 = time.time()
        per, preds, methods = train_eval_fold(
            backbone, head, grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx],
            read_ins=read_ins, all_data=all_data)
        fold_pers.append(per); all_preds.extend(preds)
        all_refs.extend([labels[i] for i in va_idx])
        for m, (p, _) in methods.items(): method_pers[m].append(p)
        best_method = min(methods, key=lambda k: methods[k][0])
        print(f"  Fold {fi+1}/{len(splits)}: PER={per:.4f} ({best_method})  ({time.time()-ft0:.1f}s)")
        if time.time() - t0 > prepare.TIME_BUDGET: break

    mean_per = float(np.mean(fold_pers))
    collapse = prepare.compute_content_collapse(all_preds)
    print(f"\n--- Per-method mean PER ---")
    for m, pers in method_pers.items(): print(f"  {m}: {np.mean(pers):.4f}")
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
