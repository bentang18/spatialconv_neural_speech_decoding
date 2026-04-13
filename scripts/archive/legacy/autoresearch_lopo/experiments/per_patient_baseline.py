#!/usr/bin/env python3
"""Per-patient CE baseline on S14 — same arch, eval, and CV splits as LOPO.

No Stage 1, no source patients. Just train on S14 with same model + eval.
This gives an apples-to-apples comparison with LOPO experiments.

Includes all eval methods: linear, k-NN, prototype, simpleshot.
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

EPOCHS = 200
BATCH_SIZE = 16
LR = 1e-3
READIN_LR_MULT = 3.0
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
WARMUP_EPOCHS = 20
PATIENCE = 7
EVAL_EVERY = 5

LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0
MIXUP_ALPHA = 0.2
TTA_COPIES = 16
KNN_K = 10

# ============================================================
# AUGMENTATION (same as LOPO)
# ============================================================

def time_shift(x, max_frames=30):
    if max_frames == 0: return x
    B, H, W, T = x.shape
    shifts = torch.randint(-max_frames, max_frames + 1, (B,))
    out = torch.zeros_like(x)
    for i in range(B):
        s = shifts[i].item()
        if s > 0: out[i, :, :, s:] = x[i, :, :, :T - s]
        elif s < 0: out[i, :, :, :T + s] = x[i, :, :, -s:]
        else: out[i] = x[i]
    return out

def amplitude_scale(x, std=0.3):
    if std == 0: return x
    return x * torch.exp(torch.randn(x.shape[0], x.shape[1], x.shape[2], 1, device=x.device) * std)

def channel_dropout(x, max_p=0.2):
    if max_p == 0: return x
    B, H, W, _T = x.shape
    p = torch.rand(1).item() * max_p
    return x * (torch.rand(B, H, W, device=x.device) > p).float().unsqueeze(-1)

def gaussian_noise(x, frac=0.05):
    if frac == 0: return x
    return x + torch.randn_like(x) * (frac * x.std())

def temporal_stretch(x, max_rate=0.15):
    if max_rate == 0: return x
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

def augment(x):
    x = time_shift(x); x = temporal_stretch(x); x = amplitude_scale(x)
    x = channel_dropout(x); x = gaussian_noise(x)
    return x

# ============================================================
# MODEL (same as LOPO)
# ============================================================

class SpatialReadIn(nn.Module):
    def __init__(self, grid_h, grid_w, C=8, pool_h=4, pool_w=8):
        super().__init__()
        self.conv = nn.Conv2d(1, C, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        self.d_flat = C * pool_h * pool_w
    def forward(self, x):
        B, H, W, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B * T, 1, H, W)
        x = F.relu(self.conv(x))
        if x.device.type == "mps": x = self.pool(x.cpu()).to("mps")
        else: x = self.pool(x)
        return x.reshape(B, T, -1).permute(0, 2, 1)

class Backbone(nn.Module):
    def __init__(self, d_in=256, d=32, gru_hidden=32, gru_layers=2,
                 stride=10, gru_dropout=0.3, feat_drop_max=0.3,
                 time_mask_min=2, time_mask_max=5):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_in, d, kernel_size=stride, stride=stride), nn.GELU())
        self.gru = nn.GRU(d, gru_hidden, num_layers=gru_layers, batch_first=True,
                          bidirectional=True, dropout=gru_dropout if gru_layers > 1 else 0.0)
        self.out_dim = gru_hidden * 2
        self.feat_drop_max = feat_drop_max
        self.time_mask_min = time_mask_min
        self.time_mask_max = time_mask_max
    def forward(self, x):
        x = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.training and self.feat_drop_max > 0:
            p = torch.rand(1).item() * self.feat_drop_max
            mask = (torch.rand(x.shape[1], device=x.device) > p).float()
            x = x * mask.unsqueeze(0).unsqueeze(-1) / (1 - p + 1e-8)
        x = self.temporal_conv(x).permute(0, 2, 1)
        if self.training and self.time_mask_max > 0:
            T = x.shape[1]
            ml = torch.randint(self.time_mask_min, self.time_mask_max + 1, (1,)).item()
            st = torch.randint(0, max(T - ml, 1), (1,)).item()
            taper = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, ml, device=x.device)))
            x = x.clone(); x[:, st:st + ml, :] *= (1 - taper).unsqueeze(-1)
        h, _ = self.gru(x)
        return h

class CEHead(nn.Module):
    def __init__(self, d_in=64, n_positions=3, n_classes=9, dropout=0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(d_in, n_classes) for _ in range(n_positions)])
    def forward(self, h):
        pooled = self.drop(h.mean(dim=1))
        return torch.stack([head(pooled) for head in self.heads], dim=1)

# ============================================================
# LOSS
# ============================================================

def focal_ce(logits, targets, gamma=FOCAL_GAMMA):
    ce = F.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING, reduction="none")
    if gamma > 0: pt = torch.exp(-ce); ce = ((1 - pt) ** gamma) * ce
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
# EVAL
# ============================================================

def decode(logits):
    return (logits.argmax(dim=-1) + 1).cpu().tolist()

def extract_embeddings(backbone, readin, grids):
    backbone.eval(); readin.eval()
    with torch.no_grad():
        return backbone(readin(grids.to(DEVICE))).mean(dim=1).cpu()

def knn_predict(train_emb, train_labels, val_emb, k=KNN_K):
    sim = F.normalize(val_emb, dim=1) @ F.normalize(train_emb, dim=1).T
    topk_sim, topk_idx = sim.topk(k, dim=1)
    preds = []
    for i in range(val_emb.shape[0]):
        pred = []
        for pos in range(prepare.N_POSITIONS):
            cw = [0.0] * (prepare.N_CLASSES + 1)
            for j in range(k):
                cw[train_labels[topk_idx[i, j].item()][pos]] += topk_sim[i, j].item()
            pred.append(int(np.argmax(cw[1:]) + 1))
        preds.append(pred)
    return preds

def prototype_predict(train_emb, train_labels, val_emb):
    train_norm = F.normalize(train_emb, dim=1)
    val_norm = F.normalize(val_emb, dim=1)
    preds = []
    for i in range(val_norm.shape[0]):
        pred = []
        for pos in range(prepare.N_POSITIONS):
            best_sim, best_cls = -float("inf"), 1
            for cls in range(1, prepare.N_CLASSES + 1):
                mask = [j for j, l in enumerate(train_labels) if l[pos] == cls]
                if not mask: continue
                centroid = F.normalize(train_norm[mask].mean(dim=0), dim=0)
                sim = (val_norm[i] * centroid).sum().item()
                if sim > best_sim: best_sim, best_cls = sim, cls
            pred.append(best_cls)
        preds.append(pred)
    return preds

def simpleshot_predict(train_emb, train_labels, val_emb):
    train_mean = train_emb.mean(dim=0, keepdim=True)
    train_norm = F.normalize(train_emb - train_mean, dim=1)
    val_norm = F.normalize(val_emb - train_mean, dim=1)
    preds = []
    for i in range(val_norm.shape[0]):
        pred = []
        for pos in range(prepare.N_POSITIONS):
            best_sim, best_cls = -float("inf"), 1
            for cls in range(1, prepare.N_CLASSES + 1):
                mask = [j for j, l in enumerate(train_labels) if l[pos] == cls]
                if not mask: continue
                centroid = F.normalize(train_norm[mask].mean(dim=0), dim=0)
                sim = (val_norm[i] * centroid).sum().item()
                if sim > best_sim: best_sim, best_cls = sim, cls
            pred.append(best_cls)
        preds.append(pred)
    return preds


# ============================================================
# TRAIN + EVAL PER FOLD
# ============================================================

def train_eval_fold(train_grids, train_labels, val_grids, val_labels):
    readin = SpatialReadIn(
        prepare.PATIENT_GRIDS[prepare.TARGET_PATIENT][0],
        prepare.PATIENT_GRIDS[prepare.TARGET_PATIENT][1]).to(DEVICE)
    backbone = Backbone(d_in=readin.d_flat).to(DEVICE)
    head = CEHead(d_in=backbone.out_dim).to(DEVICE)
    n_train = len(train_grids)

    optimizer = AdamW([
        {"params": readin.parameters(), "lr": LR * READIN_LR_MULT},
        {"params": backbone.parameters(), "lr": LR},
        {"params": head.parameters(), "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if WARMUP_EPOCHS > 0 and epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)
    best_val_loss = float("inf"); best_state = None; patience_ctr = 0

    for epoch in range(EPOCHS):
        backbone.train(); readin.train(); head.train()
        perm = torch.randperm(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            x = augment(train_grids[idx]).to(DEVICE)
            y = [train_labels[i] for i in idx.tolist()]
            mixup_y, mixup_lam = None, 1.0
            if MIXUP_ALPHA > 0 and len(idx) > 1:
                mixup_lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                pm = torch.randperm(x.shape[0])
                x = mixup_lam * x + (1 - mixup_lam) * x[pm]
                mixup_y = [y[i] for i in pm.tolist()]
            optimizer.zero_grad()
            loss = compute_loss(head(backbone(readin(x))), y, mixup_labels=mixup_y, mixup_lam=mixup_lam)
            if math.isnan(loss.item()): print("FAIL"); sys.exit(1)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(readin.parameters()) + list(head.parameters()),
                GRAD_CLIP)
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % EVAL_EVERY == 0:
            backbone.eval(); readin.eval(); head.eval()
            with torch.no_grad():
                val_loss = compute_loss(head(backbone(readin(val_grids.to(DEVICE)))), val_labels).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {"backbone": deepcopy(backbone.state_dict()),
                              "readin": deepcopy(readin.state_dict()),
                              "head": deepcopy(head.state_dict())}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE: break

    if best_state:
        backbone.load_state_dict(best_state["backbone"])
        readin.load_state_dict(best_state["readin"])
        head.load_state_dict(best_state["head"])

    backbone.eval(); readin.eval(); head.eval()

    # Linear with TTA
    with torch.no_grad():
        logits_sum = torch.zeros(len(val_grids), prepare.N_POSITIONS, prepare.N_CLASSES, device=DEVICE)
        logits_sum += head(backbone(readin(val_grids.to(DEVICE))))
        for _ in range(TTA_COPIES - 1):
            logits_sum += head(backbone(readin(augment(val_grids).to(DEVICE))))
        logits = logits_sum / TTA_COPIES
    linear_preds = decode(logits)
    linear_per = prepare.compute_per(linear_preds, val_labels)

    # Embeddings with TTA
    train_emb = extract_embeddings(backbone, readin, train_grids)
    ve = extract_embeddings(backbone, readin, val_grids)
    for _ in range(TTA_COPIES - 1):
        ve = ve + extract_embeddings(backbone, readin, augment(val_grids))
    val_emb = ve / TTA_COPIES

    knn_preds = knn_predict(train_emb, list(train_labels), val_emb)
    knn_per = prepare.compute_per(knn_preds, val_labels)
    proto_preds = prototype_predict(train_emb, list(train_labels), val_emb)
    proto_per = prepare.compute_per(proto_preds, val_labels)
    ss_preds = simpleshot_predict(train_emb, list(train_labels), val_emb)
    ss_per = prepare.compute_per(ss_preds, val_labels)

    methods = {"linear": (linear_per, linear_preds), "knn": (knn_per, knn_preds),
               "prototype": (proto_per, proto_preds), "simpleshot": (ss_per, ss_preds)}
    best_name = min(methods, key=lambda k: methods[k][0])
    return methods[best_name][0], methods[best_name][1], methods


# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)

    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)

    N, H, W, T = grids.shape
    print(f"=== Per-patient CE baseline (NO cross-patient transfer) ===")
    print(f"Target: {prepare.TARGET_PATIENT}  |  Trials: {N}  |  Grid: {H}x{W}  |  T: {T}")
    print(f"Folds: {len(splits)}  |  Device: {DEVICE}")
    print(f"Train from scratch each fold — no Stage 1, no source patients")
    print()

    fold_pers = []; all_preds = []; all_refs = []
    method_pers = {m: [] for m in ["linear", "knn", "prototype", "simpleshot"]}

    for fi, (tr_idx, va_idx) in enumerate(splits):
        ft0 = time.time()
        # Reset seed per fold for reproducibility
        torch.manual_seed(SEED + fi); np.random.seed(SEED + fi)
        per, preds, methods = train_eval_fold(
            grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx])
        fold_pers.append(per)
        all_preds.extend(preds); all_refs.extend([labels[i] for i in va_idx])
        for m, (p, _) in methods.items(): method_pers[m].append(p)
        best_m = min(methods, key=lambda k: methods[k][0])
        print(f"  Fold {fi+1}/{len(splits)}: PER={per:.4f} "
              f"(lin={methods['linear'][0]:.4f} knn={methods['knn'][0]:.4f} "
              f"proto={methods['prototype'][0]:.4f} ss={methods['simpleshot'][0]:.4f} "
              f"best={best_m})  ({time.time()-ft0:.1f}s)")

    mean_per = float(np.mean(fold_pers))
    collapse = prepare.compute_content_collapse(all_preds)

    print()
    print("--- Per-method mean PER ---")
    for m, pers in method_pers.items():
        print(f"  {m}: {np.mean(pers):.4f} ± {np.std(pers):.4f}")
    print()
    print("---")
    print(f"val_per:            {mean_per:.6f}")
    print(f"val_per_std:        {float(np.std(fold_pers)):.6f}")
    print(f"fold_pers:          {fold_pers}")
    for m, pers in method_pers.items(): print(f"{m}_pers:       {pers}")
    print(f"training_seconds:   {time.time()-t0:.1f}")
    print(f"collapsed:          {collapse['collapsed']}")
    print(f"mean_entropy:       {collapse['mean_entropy']:.3f}")
    print(f"stereotypy:         {collapse['stereotypy']:.3f}")
    print(f"unique_ratio:       {collapse['unique_ratio']:.3f}")
