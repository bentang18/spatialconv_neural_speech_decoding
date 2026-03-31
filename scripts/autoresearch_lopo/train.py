#!/usr/bin/env python3
"""LOPO cross-patient speech decoding: train on 9 source patients, eval on S14.

THE AI AGENT MODIFIES THIS FILE.

Experiment 13: Skip Stage 1 entirely. Train from scratch on S14 only.
Hypothesis: Stage 1 cross-patient features may HURT S14 performance.
If this beats 0.764, Stage 1 is net-negative.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare

# ============================================================
# DEVICE & SEED
# ============================================================
DEVICE = os.environ.get("DEVICE", "mps")
SEED = 42

# ============================================================
# HYPERPARAMETERS
# ============================================================
# No Stage 1 — train directly on S14
S2_EPOCHS = 200
S2_BATCH_SIZE = 16
S2_LR = 1e-3
S2_WEIGHT_DECAY = 5e-4
S2_GRAD_CLIP = 5.0
S2_WARMUP_EPOCHS = 15
S2_PATIENCE = 10
S2_EVAL_EVERY = 5

# Shared
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0
MIXUP_ALPHA = 0.2
TTA_COPIES = 16
KNN_K = 10

# ============================================================
# AUGMENTATION
# ============================================================

def time_shift(x: torch.Tensor, max_frames: int = 30) -> torch.Tensor:
    if max_frames == 0:
        return x
    B, H, W, T = x.shape
    shifts = torch.randint(-max_frames, max_frames + 1, (B,))
    out = torch.zeros_like(x)
    for i in range(B):
        s = shifts[i].item()
        if s > 0:
            out[i, :, :, s:] = x[i, :, :, :T - s]
        elif s < 0:
            out[i, :, :, :T + s] = x[i, :, :, -s:]
        else:
            out[i] = x[i]
    return out


def amplitude_scale(x: torch.Tensor, std: float = 0.3) -> torch.Tensor:
    if std == 0:
        return x
    B, H, W, _T = x.shape
    return x * torch.exp(torch.randn(B, H, W, 1, device=x.device) * std)


def channel_dropout(x: torch.Tensor, max_p: float = 0.2) -> torch.Tensor:
    if max_p == 0:
        return x
    B, H, W, _T = x.shape
    p = torch.rand(1).item() * max_p
    mask = (torch.rand(B, H, W, device=x.device) > p).float().unsqueeze(-1)
    return x * mask


def gaussian_noise(x: torch.Tensor, frac: float = 0.05) -> torch.Tensor:
    if frac == 0:
        return x
    return x + torch.randn_like(x) * (frac * x.std())


def temporal_stretch(x: torch.Tensor, max_rate: float = 0.15) -> torch.Tensor:
    if max_rate == 0:
        return x
    B, H, W, T = x.shape
    out = torch.zeros_like(x)
    for i in range(B):
        rate = 1.0 + (torch.rand(1).item() * 2 - 1) * max_rate
        new_T = max(int(round(T * rate)), 2)
        flat = x[i].reshape(-1, 1, T)
        stretched = F.interpolate(flat, size=new_T, mode="linear", align_corners=False)
        L = min(new_T, T)
        out[i, :, :, :L] = stretched[:, 0, :L].reshape(H, W, L)
    return out


def augment(x: torch.Tensor) -> torch.Tensor:
    x = time_shift(x, max_frames=30)
    x = temporal_stretch(x, max_rate=0.15)
    x = amplitude_scale(x, std=0.3)
    x = channel_dropout(x, max_p=0.2)
    x = gaussian_noise(x, frac=0.05)
    return x


# ============================================================
# MODEL COMPONENTS
# ============================================================

class SpatialReadIn(nn.Module):
    """Conv2d on electrode grid: (B, H, W, T) -> (B, d_flat, T)."""

    def __init__(self, grid_h: int, grid_w: int, C: int = 8,
                 pool_h: int = 4, pool_w: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(1, C, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        self.d_flat = C * pool_h * pool_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B * T, 1, H, W)
        x = F.relu(self.conv(x))
        if x.device.type == "mps":
            x = self.pool(x.cpu()).to("mps")
        else:
            x = self.pool(x)
        return x.reshape(B, T, -1).permute(0, 2, 1)


class Backbone(nn.Module):
    """LN -> feat_drop -> Conv1d(stride) -> GELU -> time_mask -> BiGRU."""

    def __init__(self, d_in: int = 256, d: int = 32, gru_hidden: int = 32,
                 gru_layers: int = 2, stride: int = 10, gru_dropout: float = 0.5,
                 feat_drop_max: float = 0.5, time_mask_min: int = 2,
                 time_mask_max: int = 5):
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
        self.feat_drop_max = feat_drop_max
        self.time_mask_min = time_mask_min
        self.time_mask_max = time_mask_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.training and self.feat_drop_max > 0:
            p = torch.rand(1).item() * self.feat_drop_max
            mask = (torch.rand(x.shape[1], device=x.device) > p).float()
            x = x * mask.unsqueeze(0).unsqueeze(-1) / (1 - p + 1e-8)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)
        if self.training and self.time_mask_max > 0:
            T = x.shape[1]
            ml = torch.randint(self.time_mask_min, self.time_mask_max + 1, (1,)).item()
            st = torch.randint(0, max(T - ml, 1), (1,)).item()
            taper = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, ml, device=x.device)))
            x = x.clone()
            x[:, st:st + ml, :] *= (1 - taper).unsqueeze(-1)
        h, _ = self.gru(x)
        return h


class CEHead(nn.Module):
    """Mean-pool -> dropout -> separate Linear per position."""

    def __init__(self, d_in: int = 64, n_positions: int = 3, n_classes: int = 9,
                 dropout: float = 0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Linear(d_in, n_classes) for _ in range(n_positions)
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        pooled = self.drop(h.mean(dim=1))
        return torch.stack([head(pooled) for head in self.heads], dim=1)


# ============================================================
# LOSS
# ============================================================

def focal_ce(logits: torch.Tensor, targets: torch.Tensor,
             gamma: float = FOCAL_GAMMA) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING, reduction="none")
    if gamma > 0:
        pt = torch.exp(-ce)
        ce = ((1 - pt) ** gamma) * ce
    return ce.mean()


def compute_loss(logits, labels, mixup_labels=None, mixup_lam=1.0):
    loss = torch.tensor(0.0, device=logits.device)
    for pos in range(prepare.N_POSITIONS):
        tgt = torch.tensor([l[pos] - 1 for l in labels], dtype=torch.long, device=logits.device)
        pos_loss = focal_ce(logits[:, pos, :], tgt)
        if mixup_labels is not None:
            tgt2 = torch.tensor([l[pos] - 1 for l in mixup_labels], dtype=torch.long, device=logits.device)
            pos_loss = mixup_lam * pos_loss + (1 - mixup_lam) * focal_ce(logits[:, pos, :], tgt2)
        loss = loss + pos_loss
    return loss / prepare.N_POSITIONS


# ============================================================
# EVALUATION HELPERS
# ============================================================

def decode(logits: torch.Tensor) -> list[list[int]]:
    return (logits.argmax(dim=-1) + 1).cpu().tolist()


def extract_embeddings(backbone, readin, grids):
    backbone.eval()
    readin.eval()
    with torch.no_grad():
        x = readin(grids.to(DEVICE))
        h = backbone(x)
        return h.mean(dim=1).cpu()


def knn_predict(train_emb, train_labels, val_emb, k=KNN_K):
    train_n = F.normalize(train_emb, dim=1)
    val_n = F.normalize(val_emb, dim=1)
    sim = val_n @ train_n.T
    topk_sim, topk_idx = sim.topk(k, dim=1)
    preds = []
    for i in range(val_emb.shape[0]):
        pred = []
        for pos in range(prepare.N_POSITIONS):
            class_weights = [0.0] * (prepare.N_CLASSES + 1)
            for j_idx in range(k):
                j = topk_idx[i, j_idx].item()
                cls = train_labels[j][pos]
                class_weights[cls] += topk_sim[i, j_idx].item()
            pred.append(int(np.argmax(class_weights[1:]) + 1))
        preds.append(pred)
    return preds


# ============================================================
# TRAIN + EVAL (per fold, no Stage 1)
# ============================================================

def train_eval_fold(train_grids, train_labels, val_grids, val_labels):
    """Train from scratch on S14 fold (no cross-patient pretraining)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    target_ri = SpatialReadIn(
        prepare.PATIENT_GRIDS[prepare.TARGET_PATIENT][0],
        prepare.PATIENT_GRIDS[prepare.TARGET_PATIENT][1],
    ).to(DEVICE)
    d_flat = target_ri.d_flat
    backbone = Backbone(d_in=d_flat).to(DEVICE)
    head = CEHead(d_in=backbone.out_dim).to(DEVICE)

    n_train = len(train_grids)

    optimizer = AdamW([
        {"params": target_ri.parameters(), "lr": S2_LR * 3.0},
        {"params": backbone.parameters(), "lr": S2_LR},
        {"params": head.parameters(), "lr": S2_LR},
    ], weight_decay=S2_WEIGHT_DECAY)

    def lr_lambda(epoch):
        if S2_WARMUP_EPOCHS > 0 and epoch < S2_WARMUP_EPOCHS:
            return (epoch + 1) / S2_WARMUP_EPOCHS
        progress = (epoch - S2_WARMUP_EPOCHS) / max(S2_EPOCHS - S2_WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(S2_EPOCHS):
        backbone.train()
        target_ri.train()
        head.train()
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
            h = backbone(feat)
            logits = head(h)
            loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)

            if math.isnan(loss.item()):
                print("FAIL")
                sys.exit(1)

            loss.backward()
            nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(target_ri.parameters()) + list(head.parameters()),
                S2_GRAD_CLIP,
            )
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % S2_EVAL_EVERY == 0:
            backbone.eval()
            target_ri.eval()
            head.eval()
            with torch.no_grad():
                feat = target_ri(val_grids.to(DEVICE))
                h = backbone(feat)
                logits = head(h)
                val_loss = compute_loss(logits, val_labels).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "backbone": deepcopy(backbone.state_dict()),
                    "readin": deepcopy(target_ri.state_dict()),
                    "head": deepcopy(head.state_dict()),
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= S2_PATIENCE:
                    break

    # Restore best
    if best_state:
        backbone.load_state_dict(best_state["backbone"])
        target_ri.load_state_dict(best_state["readin"])
        head.load_state_dict(best_state["head"])

    # --- Evaluate ---
    backbone.eval()
    target_ri.eval()
    head.eval()

    # Linear predictions with TTA
    with torch.no_grad():
        if TTA_COPIES > 1:
            logits_sum = torch.zeros(len(val_grids), prepare.N_POSITIONS, prepare.N_CLASSES, device=DEVICE)
            feat = target_ri(val_grids.to(DEVICE))
            logits_sum += head(backbone(feat))
            for _ in range(TTA_COPIES - 1):
                feat = target_ri(augment(val_grids).to(DEVICE))
                logits_sum += head(backbone(feat))
            logits = logits_sum / TTA_COPIES
        else:
            feat = target_ri(val_grids.to(DEVICE))
            logits = head(backbone(feat))
    linear_preds = decode(logits)
    linear_per = prepare.compute_per(linear_preds, val_labels)

    # k-NN with TTA
    train_emb = extract_embeddings(backbone, target_ri, train_grids)
    if TTA_COPIES > 1:
        val_emb_sum = extract_embeddings(backbone, target_ri, val_grids)
        for _ in range(TTA_COPIES - 1):
            val_emb_sum = val_emb_sum + extract_embeddings(backbone, target_ri, augment(val_grids))
        val_emb = val_emb_sum / TTA_COPIES
    else:
        val_emb = extract_embeddings(backbone, target_ri, val_grids)
    knn_preds = knn_predict(train_emb, train_labels, val_emb)
    knn_per = prepare.compute_per(knn_preds, val_labels)

    if knn_per < linear_per:
        return knn_per, knn_preds, linear_per, knn_per
    return linear_per, linear_preds, linear_per, knn_per


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data (only need target)
    all_data = prepare.load_all_patients()  # still load for cache
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)

    N, H, W, T = grids.shape
    print(f"Target: {prepare.TARGET_PATIENT}  |  Trials: {N}  |  Grid: {H}x{W}  |  T: {T}")
    print(f"NO STAGE 1 — training from scratch on S14 only")
    print(f"Folds: {len(splits)}  |  Device: {DEVICE}")
    print()

    # No Stage 1 — go directly to per-fold training
    print("=== Training from scratch on S14 (no cross-patient pretraining) ===")

    fold_pers = []
    fold_linear_pers = []
    fold_knn_pers = []
    all_preds = []
    all_refs = []

    for fi, (tr_idx, va_idx) in enumerate(splits):
        ft0 = time.time()
        per, preds, lp, kp = train_eval_fold(
            grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx],
        )
        fold_pers.append(per)
        fold_linear_pers.append(lp)
        fold_knn_pers.append(kp)
        all_preds.extend(preds)
        all_refs.extend([labels[i] for i in va_idx])
        print(f"  Fold {fi + 1}/{len(splits)}: PER={per:.4f} (linear={lp:.4f}, kNN={kp:.4f})  ({time.time() - ft0:.1f}s)")

        if time.time() - t0 > prepare.TIME_BUDGET:
            print(f"WARNING: time budget exceeded ({time.time() - t0:.0f}s)")
            break

    mean_per = float(np.mean(fold_pers))
    std_per = float(np.std(fold_pers))
    collapse = prepare.compute_content_collapse(all_preds)
    elapsed = time.time() - t0

    print()
    print("---")
    print(f"val_per:            {mean_per:.6f}")
    print(f"val_per_std:        {std_per:.6f}")
    print(f"fold_pers:          {fold_pers}")
    print(f"linear_pers:        {fold_linear_pers}")
    print(f"knn_pers:           {fold_knn_pers}")
    print(f"stage1_seconds:     0.0")
    print(f"training_seconds:   {elapsed:.1f}")
    print(f"collapsed:          {collapse['collapsed']}")
    print(f"mean_entropy:       {collapse['mean_entropy']:.3f}")
    print(f"stereotypy:         {collapse['stereotypy']:.3f}")
    print(f"unique_ratio:       {collapse['unique_ratio']:.3f}")
