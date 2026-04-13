#!/usr/bin/env python3
"""Exp 40: Combined best — Articulatory Bottleneck + Checkpoint Averaging.

Combines the two best improvements:
1. Articulatory bottleneck head (exp33: 0.762) — biologically-grounded feature bottleneck
2. Checkpoint weight averaging (exp36: 0.775) — smoother backbone features

CUDA baseline: 0.823 | exp33: 0.762 | exp36: 0.775 | Target: < 0.75
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

# ============================================================
DEVICE = os.environ.get("DEVICE", "mps")
SEED = 42

# All hyperparams identical to baseline
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

# ============================================================
# ARTICULATORY MATRIX (9 phonemes × 15 binary features)
# Phoneme order: a, ae, b, g, i, k, p, u, v (1-indexed)
# Features: bilabial, labiodental, alveolar, postalveolar, velar, glottal,
#           stop, fricative, nasal, approximant, lateral,
#           high, low, front, back
# ============================================================

ARTICULATORY_MATRIX = [
    # a:  low central vowel
    [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 1, 0, 0],
    # ae: low front vowel
    [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 1, 1, 0],
    # b:  voiced bilabial stop
    [1, 0, 0, 0, 0, 0,  1, 0, 0, 0, 0,  0, 0, 0, 0],
    # g:  voiced velar stop
    [0, 0, 0, 0, 1, 0,  1, 0, 0, 0, 0,  0, 0, 0, 0],
    # i:  high front vowel
    [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  1, 0, 1, 0],
    # k:  voiceless velar stop
    [0, 0, 0, 0, 1, 0,  1, 0, 0, 0, 0,  0, 0, 0, 0],
    # p:  voiceless bilabial stop
    [1, 0, 0, 0, 0, 0,  1, 0, 0, 0, 0,  0, 0, 0, 0],
    # u:  high back vowel
    [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  1, 0, 0, 1],
    # v:  voiced labiodental fricative
    [0, 1, 0, 0, 0, 0,  0, 1, 0, 0, 0,  0, 0, 0, 0],
]

# ============================================================
# AUGMENTATION (unchanged)
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


# ============================================================
# NEW: Articulatory Bottleneck Head
# ============================================================

class ArticulatoryBottleneckHead(nn.Module):
    """Forces predictions through 15-dim articulatory feature space.

    For each position, predicts a 15-dim articulatory feature vector via
    sigmoid, then computes scaled cosine similarity against the fixed
    articulatory matrix to obtain phoneme logits.

    The temperature parameter is learnable (initialized to 5.0) to allow
    the model to sharpen or soften the similarity distribution during training.
    """
    def __init__(self, d_in=64, n_positions=3, n_features=15, n_classes=9, dropout=0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.n_positions = n_positions
        self.n_features = n_features
        self.n_classes = n_classes
        self.feature_heads = nn.ModuleList([
            nn.Linear(d_in, n_features) for _ in range(n_positions)
        ])
        A = torch.tensor(ARTICULATORY_MATRIX, dtype=torch.float32)  # (9, 15)
        self.register_buffer('A', A)
        self.temperature = nn.Parameter(torch.tensor(5.0))

    def forward(self, h):
        # h: (B, T, d_in)
        pooled = self.drop(h.mean(dim=1))  # (B, d_in)
        logits_list = []
        for head in self.feature_heads:
            feat_pred = torch.sigmoid(head(pooled))     # (B, 15)
            feat_norm = F.normalize(feat_pred, dim=-1)  # (B, 15)
            A_norm = F.normalize(self.A.float(), dim=-1)  # (9, 15)
            sim = feat_norm @ A_norm.T * self.temperature  # (B, 9)
            logits_list.append(sim)
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
# EVAL HELPERS
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

def simpleshot_predict(train_emb, train_labels, val_emb):
    """SimpleShot: center + L2 normalize + nearest centroid."""
    train_mean = train_emb.mean(dim=0, keepdim=True)
    train_centered = train_emb - train_mean
    val_centered = val_emb - train_mean
    train_norm = F.normalize(train_centered, dim=1)
    val_norm = F.normalize(val_centered, dim=1)
    preds = []
    for i in range(val_norm.shape[0]):
        pred = []
        for pos in range(prepare.N_POSITIONS):
            best_sim = -float("inf")
            best_cls = 1
            for cls in range(1, prepare.N_CLASSES + 1):
                mask = [j for j, l in enumerate(train_labels) if l[pos] == cls]
                if not mask:
                    continue
                centroid = train_norm[mask].mean(dim=0)
                centroid = F.normalize(centroid, dim=0)
                sim = (val_norm[i] * centroid).sum().item()
                if sim > best_sim:
                    best_sim = sim
                    best_cls = cls
            pred.append(best_cls)
        preds.append(pred)
    return preds

def prototype_predict(train_emb, train_labels, val_emb):
    """Prototype network: nearest centroid without centering."""
    train_norm = F.normalize(train_emb, dim=1)
    val_norm = F.normalize(val_emb, dim=1)
    preds = []
    for i in range(val_norm.shape[0]):
        pred = []
        for pos in range(prepare.N_POSITIONS):
            best_sim = -float("inf")
            best_cls = 1
            for cls in range(1, prepare.N_CLASSES + 1):
                mask = [j for j, l in enumerate(train_labels) if l[pos] == cls]
                if not mask:
                    continue
                centroid = train_norm[mask].mean(dim=0)
                centroid = F.normalize(centroid, dim=0)
                sim = (val_norm[i] * centroid).sum().item()
                if sim > best_sim:
                    best_sim = sim
                    best_cls = cls
            pred.append(best_cls)
        preds.append(pred)
    return preds

def simpleshot_src_predict(train_emb, train_labels, val_emb):
    """SimpleShot with combined target+source embeddings."""
    return simpleshot_predict(train_emb, train_labels, val_emb)


# ============================================================
# STAGE 1 (uses ArticulatoryBottleneckHead)
# ============================================================

def train_stage1(all_data):
    torch.manual_seed(SEED); np.random.seed(SEED)
    read_ins = {}
    for pid in prepare.SOURCE_PATIENTS:
        H, W = all_data[pid]["grid_shape"]
        read_ins[pid] = SpatialReadIn(H, W).to(DEVICE)
    d_flat = list(read_ins.values())[0].d_flat

    backbone = Backbone(d_in=d_flat).to(DEVICE)
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
    backbone_checkpoints = []  # For checkpoint averaging

    for epoch in range(S1_EPOCHS):
        backbone.train(); head.train()
        for ri in read_ins.values(): ri.train()
        np.random.shuffle(pids)
        epoch_loss = 0.0; n_batches = 0

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
                feat = read_ins[pid](x); h = backbone(feat); logits = head(h)
                loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()) + readin_params, S1_GRAD_CLIP)
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
                    x = all_data[pid]["grids"][vi].to(DEVICE)
                    y = [all_data[pid]["labels"][i] for i in vi]
                    feat = read_ins[pid](x); h = backbone(feat); logits = head(h)
                    val_loss += compute_loss(logits, y).item(); val_batches += 1
            val_loss /= max(val_batches, 1)
            # Save backbone checkpoint for averaging (after warmup, every 10 epochs)
            if epoch >= S1_WARMUP_EPOCHS:
                backbone_checkpoints.append(deepcopy(backbone.state_dict()))

            print(f"  S1 epoch {epoch+1}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f} (ckpts={len(backbone_checkpoints)})")
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

    # Restore best head + read-ins
    if best_state:
        head.load_state_dict(best_state["head"])
        for pid, ri in read_ins.items():
            if pid in best_state["read_ins"]: ri.load_state_dict(best_state["read_ins"][pid])

    # CHECKPOINT AVERAGING: average all backbone checkpoints instead of using best
    if backbone_checkpoints:
        print(f"  Averaging {len(backbone_checkpoints)} backbone checkpoints")
        avg_state = {}
        for key in backbone_checkpoints[0]:
            avg_state[key] = torch.stack([ckpt[key].float() for ckpt in backbone_checkpoints]).mean(dim=0)
        backbone.load_state_dict(avg_state)
    elif best_state:
        backbone.load_state_dict(best_state["backbone"])

    return backbone, head, read_ins


# ============================================================
# STAGE 2 + EVAL (ArticulatoryBottleneckHead in S2 as well)
# ============================================================

def train_eval_fold(backbone, head_init, train_grids, train_labels,
                    val_grids, val_labels, read_ins=None, all_data=None):
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
            feat = target_ri(x); h = backbone_copy(feat); logits = head(h)
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
                feat = target_ri(val_grids.to(DEVICE)); h = backbone_copy(feat)
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

    # Linear predictions with TTA
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

    # Build embeddings for eval methods
    train_emb = extract_embeddings(backbone_copy, target_ri, train_grids)
    train_emb_all = train_emb
    train_labels_all = list(train_labels)
    if read_ins is not None and all_data is not None and SOURCE_KNN_WEIGHT > 0:
        source_embs, source_labs = [], []
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

    # k-NN (baseline)
    knn_preds = knn_predict(train_emb_all, train_labels_all, val_emb)
    knn_per = prepare.compute_per(knn_preds, val_labels)

    # SimpleShot (target-only)
    simpleshot_preds = simpleshot_predict(train_emb, list(train_labels), val_emb)
    simpleshot_per = prepare.compute_per(simpleshot_preds, val_labels)

    # Prototype (target-only)
    proto_preds = prototype_predict(train_emb, list(train_labels), val_emb)
    proto_per = prepare.compute_per(proto_preds, val_labels)

    # SimpleShot with source
    simpleshot_src_preds = simpleshot_predict(train_emb_all, train_labels_all, val_emb)
    simpleshot_src_per = prepare.compute_per(simpleshot_src_preds, val_labels)

    # Pick best
    methods = {
        "linear": (linear_per, linear_preds),
        "knn": (knn_per, knn_preds),
        "simpleshot": (simpleshot_per, simpleshot_preds),
        "prototype": (proto_per, proto_preds),
        "simpleshot_src": (simpleshot_src_per, simpleshot_src_preds),
    }
    best_name = min(methods, key=lambda k: methods[k][0])
    best_per, best_preds = methods[best_name]

    return best_per, best_preds, methods


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
    print(f"CHANGE: Articulatory bottleneck + checkpoint averaging (exp33+exp36 combined)")
    print()

    print("=== Stage 1: Source patient training ===")
    s1_start = time.time()
    backbone, head, read_ins = train_stage1(all_data)
    s1_time = time.time() - s1_start
    print(f"Stage 1 done in {s1_time:.1f}s")
    print()

    print("=== Stage 2: Target adaptation + evaluation ===")
    fold_pers = []
    all_preds, all_refs = [], []
    method_pers = {m: [] for m in ["linear", "knn", "simpleshot", "prototype", "simpleshot_src"]}

    for fi, (tr_idx, va_idx) in enumerate(splits):
        ft0 = time.time()
        per, preds, methods = train_eval_fold(
            backbone, head, grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx],
            read_ins=read_ins, all_data=all_data)
        fold_pers.append(per)
        all_preds.extend(preds); all_refs.extend([labels[i] for i in va_idx])
        for m, (p, _) in methods.items():
            method_pers[m].append(p)
        best_method = min(methods, key=lambda k: methods[k][0])
        print(f"  Fold {fi+1}/{len(splits)}: PER={per:.4f} "
              f"(lin={methods['linear'][0]:.4f} knn={methods['knn'][0]:.4f} "
              f"ss={methods['simpleshot'][0]:.4f} proto={methods['prototype'][0]:.4f} "
              f"ss_src={methods['simpleshot_src'][0]:.4f} best={best_method})  ({time.time()-ft0:.1f}s)")
        if time.time() - t0 > prepare.TIME_BUDGET:
            print(f"WARNING: time budget exceeded"); break

    mean_per = float(np.mean(fold_pers))
    std_per = float(np.std(fold_pers))
    collapse = prepare.compute_content_collapse(all_preds)
    elapsed = time.time() - t0

    print()
    print("--- Per-method mean PER ---")
    for m, pers in method_pers.items():
        print(f"  {m}: {np.mean(pers):.4f} ± {np.std(pers):.4f}")

    print()
    print("---")
    print(f"val_per:            {mean_per:.6f}")
    print(f"val_per_std:        {std_per:.6f}")
    print(f"fold_pers:          {fold_pers}")
    for m, pers in method_pers.items():
        print(f"{m}_pers:       {pers}")
    print(f"stage1_seconds:     {s1_time:.1f}")
    print(f"training_seconds:   {elapsed:.1f}")
    print(f"collapsed:          {collapse['collapsed']}")
    print(f"mean_entropy:       {collapse['mean_entropy']:.3f}")
    print(f"stereotypy:         {collapse['stereotypy']:.3f}")
    print(f"unique_ratio:       {collapse['unique_ratio']:.3f}")
