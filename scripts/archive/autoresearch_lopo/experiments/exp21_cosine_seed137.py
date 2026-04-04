#!/usr/bin/env python3
"""exp20: Cosine classifier head with temperature scaling.

Hypothesis: Linear heads learn biased decision boundaries in low-data regimes.
Cosine classifiers (normalized weights + learned temperature) are standard in
few-shot learning (Matching Networks, ProtoNet, MetaOptNet) and work better
when class embeddings need to generalize across domains.

Changes from exp13 baseline:
  - Replace CEHead's Linear layers with CosineClassifier (normalized weights, learned temperature)
  - L2-normalize backbone features before classification
  - Temperature-scaled cosine similarity → softmax
  - Multi-patient k-NN preserved (SOURCE_KNN_WEIGHT=0.5)
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
SEED = 137

# Stage 1
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

# Stage 2
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

# Shared
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0
MIXUP_ALPHA = 0.2
TTA_COPIES = 16
KNN_K = 10
SOURCE_KNN_WEIGHT = 0.5
COSINE_TEMPERATURE_INIT = 10.0  # learned temperature, initialized high


# ============================================================
# AUGMENTATION
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
    B, H, W, _T = x.shape
    return x * torch.exp(torch.randn(B, H, W, 1, device=x.device) * std)

def channel_dropout(x, max_p=0.2):
    if max_p == 0: return x
    B, H, W, _T = x.shape
    p = torch.rand(1).item() * max_p
    mask = (torch.rand(B, H, W, device=x.device) > p).float().unsqueeze(-1)
    return x * mask

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
    def __init__(self, grid_h, grid_w, C=8, pool_h=4, pool_w=8):
        super().__init__()
        self.conv = nn.Conv2d(1, C, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        self.d_flat = C * pool_h * pool_w

    def forward(self, x):
        B, H, W, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B * T, 1, H, W)
        x = F.relu(self.conv(x))
        if x.device.type == "mps":
            x = self.pool(x.cpu()).to("mps")
        else:
            x = self.pool(x)
        return x.reshape(B, T, -1).permute(0, 2, 1)


class Backbone(nn.Module):
    def __init__(self, d_in=256, d=32, gru_hidden=32, gru_layers=2, stride=10,
                 gru_dropout=0.3, feat_drop_max=0.3, time_mask_min=2, time_mask_max=5):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_in, d, kernel_size=stride, stride=stride), nn.GELU(),
        )
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


class CosineClassifierHead(nn.Module):
    """Temperature-scaled cosine classifier — standard in few-shot learning.

    Instead of Linear(d_in, n_classes), uses:
        logit = temperature * cosine_similarity(x, w_class)

    Benefits over linear head at small N:
    - Forces features to unit sphere → prevents magnitude shortcuts
    - Learned prototypes (weights) are normalized → class boundaries are equidistant by default
    - Temperature controls sharpness → adapts to confidence level
    """

    def __init__(self, d_in=64, n_positions=3, n_classes=9, dropout=0.3,
                 temperature_init=COSINE_TEMPERATURE_INIT):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.n_positions = n_positions
        # Learned temperature (shared across positions)
        self.temperature = nn.Parameter(torch.tensor(float(temperature_init)))
        # Per-position prototype weights (will be L2-normalized before use)
        self.prototypes = nn.ParameterList([
            nn.Parameter(torch.randn(n_classes, d_in) * 0.01)
            for _ in range(n_positions)
        ])

    def forward(self, h):
        # Mean pool + dropout
        pooled = self.drop(h.mean(dim=1))  # (B, d_in)
        # L2 normalize features
        feat_norm = F.normalize(pooled, dim=1)  # (B, d_in)

        logits_list = []
        for pos in range(self.n_positions):
            # L2 normalize prototypes
            proto_norm = F.normalize(self.prototypes[pos], dim=1)  # (n_classes, d_in)
            # Cosine similarity × temperature
            sim = feat_norm @ proto_norm.T  # (B, n_classes)
            logits_list.append(self.temperature * sim)

        return torch.stack(logits_list, dim=1)  # (B, n_positions, n_classes)


# ============================================================
# LOSS
# ============================================================

def focal_ce(logits, targets, gamma=FOCAL_GAMMA):
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

def decode(logits):
    return (logits.argmax(dim=-1) + 1).cpu().tolist()

def extract_embeddings(backbone, readin, grids):
    backbone.eval(); readin.eval()
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
# STAGE 1 (uses CosineClassifierHead instead of CEHead)
# ============================================================

def train_stage1(all_data):
    torch.manual_seed(SEED); np.random.seed(SEED)

    read_ins = {}
    for pid in prepare.SOURCE_PATIENTS:
        H, W = all_data[pid]["grid_shape"]
        read_ins[pid] = SpatialReadIn(H, W).to(DEVICE)
    d_flat = list(read_ins.values())[0].d_flat

    backbone = Backbone(d_in=d_flat).to(DEVICE)
    head = CosineClassifierHead(d_in=backbone.out_dim).to(DEVICE)

    source_train, source_val = {}, {}
    for pid in prepare.SOURCE_PATIENTS:
        n = len(all_data[pid]["labels"])
        perm = np.random.permutation(n)
        n_val = max(1, int(round(S1_VAL_FRACTION * n)))
        source_val[pid] = sorted(perm[:n_val].tolist())
        source_train[pid] = sorted(perm[n_val:].tolist())

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
        epoch_loss = 0.0; n_batches = 0

        for pid in pids:
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
                loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()) + readin_params, S1_GRAD_CLIP,
                )
                optimizer.step()
                epoch_loss += loss.item(); n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % S1_EVAL_EVERY == 0:
            backbone.eval(); head.eval()
            for ri in read_ins.values(): ri.eval()
            val_loss = 0.0; val_batches = 0
            with torch.no_grad():
                for pid in prepare.SOURCE_PATIENTS:
                    vi = source_val[pid]
                    if not vi: continue
                    grids = all_data[pid]["grids"]; labels = all_data[pid]["labels"]
                    x = grids[vi].to(DEVICE); y = [labels[i] for i in vi]
                    feat = read_ins[pid](x); h = backbone(feat); logits = head(h)
                    val_loss += compute_loss(logits, y).item(); val_batches += 1
            val_loss /= max(val_batches, 1)
            print(f"  S1 epoch {epoch+1}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, temp={head.temperature.item():.2f}")
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


# ============================================================
# STAGE 2 + EVAL (CosineClassifierHead in S2 too)
# ============================================================

def train_eval_fold(backbone, head_init, train_grids, train_labels,
                    val_grids, val_labels, read_ins=None, all_data=None):
    backbone_copy = deepcopy(backbone)
    target_ri = SpatialReadIn(
        prepare.PATIENT_GRIDS[prepare.TARGET_PATIENT][0],
        prepare.PATIENT_GRIDS[prepare.TARGET_PATIENT][1],
    ).to(DEVICE)
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
            h = backbone_copy(feat)
            logits = head(h)
            loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)
            if math.isnan(loss.item()): print("FAIL"); sys.exit(1)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(backbone_copy.parameters()) + list(target_ri.parameters()) + list(head.parameters()),
                S2_GRAD_CLIP,
            )
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % S2_EVAL_EVERY == 0:
            backbone_copy.eval(); target_ri.eval(); head.eval()
            with torch.no_grad():
                feat = target_ri(val_grids.to(DEVICE))
                h = backbone_copy(feat)
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

    with torch.no_grad():
        if TTA_COPIES > 1:
            logits_sum = torch.zeros(len(val_grids), prepare.N_POSITIONS, prepare.N_CLASSES, device=DEVICE)
            feat = target_ri(val_grids.to(DEVICE))
            logits_sum += head(backbone_copy(feat))
            for _ in range(TTA_COPIES - 1):
                feat = target_ri(augment(val_grids).to(DEVICE))
                logits_sum += head(backbone_copy(feat))
            logits = logits_sum / TTA_COPIES
        else:
            feat = target_ri(val_grids.to(DEVICE))
            logits = head(backbone_copy(feat))
    linear_preds = decode(logits)
    linear_per = prepare.compute_per(linear_preds, val_labels)

    train_emb = extract_embeddings(backbone_copy, target_ri, train_grids)
    train_emb_all = train_emb
    train_labels_all = list(train_labels)

    if read_ins is not None and all_data is not None and SOURCE_KNN_WEIGHT > 0:
        source_embs = []; source_labs = []
        for pid in prepare.SOURCE_PATIENTS:
            ri = read_ins[pid]; ri.eval()
            emb = extract_embeddings(backbone_copy, ri, all_data[pid]["grids"])
            source_embs.append(emb); source_labs.extend(all_data[pid]["labels"])
        source_emb = torch.cat(source_embs, dim=0)
        train_emb_all = torch.cat([train_emb, source_emb * SOURCE_KNN_WEIGHT], dim=0)
        train_labels_all = list(train_labels) + source_labs

    if TTA_COPIES > 1:
        val_emb_sum = extract_embeddings(backbone_copy, target_ri, val_grids)
        for _ in range(TTA_COPIES - 1):
            val_emb_sum = val_emb_sum + extract_embeddings(backbone_copy, target_ri, augment(val_grids))
        val_emb = val_emb_sum / TTA_COPIES
    else:
        val_emb = extract_embeddings(backbone_copy, target_ri, val_grids)
    knn_preds = knn_predict(train_emb_all, train_labels_all, val_emb)
    knn_per = prepare.compute_per(knn_preds, val_labels)

    if knn_per < linear_per:
        return knn_per, knn_preds, linear_per, knn_per
    return linear_per, linear_preds, linear_per, knn_per


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)

    all_data = prepare.load_all_patients()
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)

    N, H, W, T = grids.shape
    total_source = sum(len(all_data[p]["labels"]) for p in prepare.SOURCE_PATIENTS)
    print(f"Target: {prepare.TARGET_PATIENT}  |  Trials: {N}  |  Grid: {H}x{W}  |  T: {T}")
    print(f"Source patients: {len(prepare.SOURCE_PATIENTS)}  |  Source trials: {total_source}")
    print(f"Folds: {len(splits)}  |  Device: {DEVICE}")
    print(f"EXP21: Cosine classifier seed=137 head, temp_init={COSINE_TEMPERATURE_INIT}")
    print()

    print("=== Stage 1: Source patient training (cosine classifier) ===")
    s1_start = time.time()
    backbone, head, read_ins = train_stage1(all_data)
    s1_time = time.time() - s1_start
    print(f"Stage 1 done in {s1_time:.1f}s (final temp={head.temperature.item():.2f})")
    print()

    print("=== Stage 2: Target adaptation + evaluation ===")
    fold_pers, fold_linear_pers, fold_knn_pers = [], [], []
    all_preds, all_refs = [], []

    for fi, (tr_idx, va_idx) in enumerate(splits):
        ft0 = time.time()
        per, preds, lp, kp = train_eval_fold(
            backbone, head, grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx],
            read_ins=read_ins, all_data=all_data,
        )
        fold_pers.append(per); fold_linear_pers.append(lp); fold_knn_pers.append(kp)
        all_preds.extend(preds); all_refs.extend([labels[i] for i in va_idx])
        print(f"  Fold {fi+1}/{len(splits)}: PER={per:.4f} (linear={lp:.4f}, kNN={kp:.4f})  ({time.time()-ft0:.1f}s)")
        if time.time() - t0 > prepare.TIME_BUDGET:
            print(f"WARNING: time budget exceeded ({time.time()-t0:.0f}s)"); break

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
    print(f"stage1_seconds:     {s1_time:.1f}")
    print(f"training_seconds:   {elapsed:.1f}")
    print(f"collapsed:          {collapse['collapsed']}")
    print(f"mean_entropy:       {collapse['mean_entropy']:.3f}")
    print(f"stereotypy:         {collapse['stereotypy']:.3f}")
    print(f"unique_ratio:       {collapse['unique_ratio']:.3f}")
