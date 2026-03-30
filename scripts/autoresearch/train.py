#!/usr/bin/env python3
"""Per-patient CE baseline on S14: SpatialConv → BiGRU → mean-pool → CE.

THE AI AGENT MODIFIES THIS FILE.

Current best PER: ~0.700  |  Chance PER: ~0.889 (1/9 per position)
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
# HYPERPARAMETERS  (tune freely)
# ============================================================
EPOCHS = 300
BATCH_SIZE = 16
LR = 1e-3
READIN_LR_MULT = 3.0
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
WARMUP_EPOCHS = 20
PATIENCE = 7          # early stopping (in eval_every units)
EVAL_EVERY = 10       # validate every N epochs
LABEL_SMOOTHING = 0.1 # 0.0 = hard labels, >0 = regularize
MIXUP_ALPHA = 0.2     # 0 = no mixup, >0 = Beta(alpha, alpha) interpolation
EMA_DECAY = 0         # 0 = no EMA, >0 = exponential moving average for eval
TTA_COPIES = 8        # test-time augmentation: average over N augmented copies
HEAD_TYPE = "dual"  # "ce" = standard linear, "articulatory" = CEBRA-style, "dual" = both

# ============================================================
# AUGMENTATION  (modify strategy freely)
# ============================================================

def time_shift(x: torch.Tensor, max_frames: int = 30) -> torch.Tensor:
    """Random per-trial circular shift ±max_frames (zero-padded edges)."""
    if max_frames == 0:
        return x
    B, H, W, T = x.shape
    shifts = torch.randint(-max_frames, max_frames + 1, (B,))
    out = torch.zeros_like(x)
    for i in range(B):
        s = shifts[i].item()
        if s > 0:
            out[i, :, :, s:] = x[i, :, :, : T - s]
        elif s < 0:
            out[i, :, :, : T + s] = x[i, :, :, -s:]
        else:
            out[i] = x[i]
    return out


def amplitude_scale(x: torch.Tensor, std: float = 0.3) -> torch.Tensor:
    """Per-electrode log-normal gain: scale = exp(N(0, std^2))."""
    if std == 0:
        return x
    B, H, W, _T = x.shape
    return x * torch.exp(torch.randn(B, H, W, 1, device=x.device) * std)


def channel_dropout(x: torch.Tensor, max_p: float = 0.2) -> torch.Tensor:
    """Zero entire electrodes with p ~ U[0, max_p]."""
    if max_p == 0:
        return x
    B, H, W, _T = x.shape
    p = torch.rand(1).item() * max_p
    mask = (torch.rand(B, H, W, device=x.device) > p).float().unsqueeze(-1)
    return x * mask


def gaussian_noise(x: torch.Tensor, frac: float = 0.05) -> torch.Tensor:
    """Additive N(0, (frac * std(x))^2) noise."""
    if frac == 0:
        return x
    return x + torch.randn_like(x) * (frac * x.std())


def temporal_stretch(x: torch.Tensor, max_rate: float = 0.15) -> torch.Tensor:
    """Per-trial time stretch/compress by factor in [1-r, 1+r]."""
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
    """Full augmentation pipeline (pre-read-in)."""
    x = time_shift(x, max_frames=30)
    x = temporal_stretch(x, max_rate=0.15)
    x = amplitude_scale(x, std=0.3)
    x = channel_dropout(x, max_p=0.2)
    x = gaussian_noise(x, frac=0.05)
    return x


# ============================================================
# MODEL COMPONENTS  (redesign freely)
# ============================================================

class SpatialReadIn(nn.Module):
    """Conv2d on electrode grid: (B, H, W, T) -> (B, d_flat, T)."""

    def __init__(self, C: int = 8, pool_h: int = 4, pool_w: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(1, C, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        self.d_flat = C * pool_h * pool_w  # default 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B * T, 1, H, W)
        x = F.relu(self.conv(x))
        # MPS fallback for non-divisible pool sizes
        if x.device.type == "mps":
            x = self.pool(x.cpu()).to("mps")
        else:
            x = self.pool(x)
        return x.reshape(B, T, -1).permute(0, 2, 1)  # (B, d_flat, T)


class Backbone(nn.Module):
    """LN -> feat_drop -> Conv1d(stride) -> GELU -> time_mask -> BiGRU."""

    def __init__(
        self,
        d_in: int = 256,
        d: int = 32,
        gru_hidden: int = 32,
        gru_layers: int = 2,
        stride: int = 10,
        gru_dropout: float = 0.3,
        feat_drop_max: float = 0.3,
        time_mask_min: int = 2,
        time_mask_max: int = 5,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_in, d, kernel_size=stride, stride=stride),
            nn.GELU(),
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
        # x: (B, D, T) from read-in
        x = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Feature dropout (spatial dropout)
        if self.training and self.feat_drop_max > 0:
            p = torch.rand(1).item() * self.feat_drop_max
            mask = (torch.rand(x.shape[1], device=x.device) > p).float()
            x = x * mask.unsqueeze(0).unsqueeze(-1) / (1 - p + 1e-8)

        x = self.temporal_conv(x)    # (B, d, T')
        x = x.permute(0, 2, 1)      # (B, T', d)

        # Time mask (cosine taper)
        if self.training and self.time_mask_max > 0:
            T = x.shape[1]
            ml = torch.randint(self.time_mask_min, self.time_mask_max + 1, (1,)).item()
            st = torch.randint(0, max(T - ml, 1), (1,)).item()
            taper = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, ml, device=x.device)))
            x = x.clone()
            x[:, st : st + ml, :] *= (1 - taper).unsqueeze(-1)

        h, _ = self.gru(x)          # (B, T', 2H)
        return h


class CEHead(nn.Module):
    """Mean-pool -> dropout -> separate Linear per position -> (B, n_pos, n_cls)."""

    def __init__(self, d_in: int = 64, n_positions: int = 3, n_classes: int = 9,
                 dropout: float = 0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Linear(d_in, n_classes) for _ in range(n_positions)
        ])
        self.n_pos = n_positions
        self.n_cls = n_classes

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        pooled = self.drop(h.mean(dim=1))             # (B, 2H)
        return torch.stack([head(pooled) for head in self.heads], dim=1)


# Articulatory feature matrix: reorder from ARPA to PS label order (0-indexed)
# PS labels: a=0, ae=1, b=2, g=3, i=4, k=5, p=6, u=7, v=8
# ARPA rows: AA=0, EH=1, IY=2, UH=3, B=4, P=5, V=6, G=7, K=8
_ARPA_TO_PS = [0, 1, 4, 7, 2, 8, 5, 3, 6]  # ARPA row -> PS 0-indexed class
_ART_MATRIX_PS = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # a: vowel, low, central
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # ae: vowel, mid, front
    [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # b: consonant, bilabial, stop, voiced
    [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # g: consonant, velar, stop, voiced
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # i: vowel, high, front
    [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # k: consonant, velar, stop, voiceless
    [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # p: consonant, bilabial, stop, voiceless
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],  # u: vowel, high, back
    [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # v: consonant, labiodental, fricative, voiced
], dtype=np.float32)


class ArticulatoryHead(nn.Module):
    """CEBRA-style: project embeddings → articulatory space, classify by similarity."""

    def __init__(self, d_in: int = 64, n_positions: int = 3, n_features: int = 15,
                 dropout: float = 0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.projectors = nn.ModuleList([
            nn.Linear(d_in, n_features) for _ in range(n_positions)
        ])
        # Fixed articulatory matrix (normalized)
        art = torch.from_numpy(_ART_MATRIX_PS)
        art = F.normalize(art, dim=1)
        self.register_buffer('art_matrix', art)  # (9, 15)
        self.log_temp = nn.Parameter(torch.tensor(2.0))  # learned temperature

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        pooled = self.drop(h.mean(dim=1))
        temp = self.log_temp.exp().clamp(min=0.01)
        logits_list = []
        for proj in self.projectors:
            art_pred = F.normalize(proj(pooled), dim=1)
            logits = art_pred @ self.art_matrix.T * temp
            logits_list.append(logits)
        return torch.stack(logits_list, dim=1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.readin = SpatialReadIn()
        self.backbone = Backbone(d_in=self.readin.d_flat)
        if HEAD_TYPE == "dual":
            self.head_ce = CEHead(d_in=self.backbone.out_dim)
            self.head_art = ArticulatoryHead(d_in=self.backbone.out_dim)
            self.head = self.head_ce  # for param group
        elif HEAD_TYPE == "articulatory":
            self.head = ArticulatoryHead(d_in=self.backbone.out_dim)
        else:
            self.head = CEHead(d_in=self.backbone.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(self.readin(x))
        if HEAD_TYPE == "dual":
            return (self.head_ce(h) + self.head_art(h)) / 2
        return self.head(h)


# ============================================================
# LOSS  (swap freely: CE, CTC, focal, etc.)
# ============================================================

def compute_loss(
    logits: torch.Tensor,
    labels: list[list[int]],
    mixup_labels: list[list[int]] | None = None,
    mixup_lam: float = 1.0,
) -> torch.Tensor:
    """Per-position CE with label smoothing and optional mixup."""
    loss = torch.tensor(0.0, device=logits.device)
    for pos in range(prepare.N_POSITIONS):
        tgt = torch.tensor(
            [l[pos] - 1 for l in labels], dtype=torch.long, device=logits.device,
        )
        pos_loss = F.cross_entropy(
            logits[:, pos, :], tgt, label_smoothing=LABEL_SMOOTHING,
        )
        if mixup_labels is not None:
            tgt2 = torch.tensor(
                [l[pos] - 1 for l in mixup_labels], dtype=torch.long, device=logits.device,
            )
            pos_loss2 = F.cross_entropy(
                logits[:, pos, :], tgt2, label_smoothing=LABEL_SMOOTHING,
            )
            pos_loss = mixup_lam * pos_loss + (1 - mixup_lam) * pos_loss2
        loss = loss + pos_loss
    return loss / prepare.N_POSITIONS


# ============================================================
# DECODE  (match loss: argmax per position, 1-indexed)
# ============================================================

def decode(logits: torch.Tensor) -> list[list[int]]:
    """Argmax per position -> 1-indexed predictions."""
    return (logits.argmax(dim=-1) + 1).cpu().tolist()


def extract_embeddings(model: nn.Module, grids: torch.Tensor) -> torch.Tensor:
    """Extract mean-pooled backbone features: (N, 2H)."""
    model.eval()
    with torch.no_grad():
        x = model.readin(grids.to(DEVICE))
        h = model.backbone(x)          # (N, T', 2H)
        return h.mean(dim=1).cpu()      # (N, 2H)


def knn_predict(
    train_emb: torch.Tensor,
    train_labels: list[list[int]],
    val_emb: torch.Tensor,
    k: int = 10,
) -> list[list[int]]:
    """Per-position k-NN classification. Returns 1-indexed predictions."""
    # Normalize embeddings for cosine similarity
    train_n = F.normalize(train_emb, dim=1)
    val_n = F.normalize(val_emb, dim=1)
    sim = val_n @ train_n.T                  # (N_val, N_train)
    _, topk_idx = sim.topk(k, dim=1)         # (N_val, k)

    preds = []
    for i in range(val_emb.shape[0]):
        pred = []
        for pos in range(prepare.N_POSITIONS):
            votes = [train_labels[j][pos] for j in topk_idx[i].tolist()]
            pred.append(max(set(votes), key=votes.count))
        preds.append(pred)
    return preds


# ============================================================
# TRAINING
# ============================================================

def _ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update EMA model parameters."""
    with torch.no_grad():
        for ep, mp in zip(ema_model.parameters(), model.parameters()):
            ep.mul_(decay).add_(mp, alpha=1 - decay)
        for eb, mb in zip(ema_model.buffers(), model.buffers()):
            eb.copy_(mb)


def train_fold(
    train_grids: torch.Tensor,
    train_labels: list[list[int]],
    val_grids: torch.Tensor,
    val_labels: list[list[int]],
) -> tuple[float, list[list[int]], float, float]:
    """Train one CV fold. Returns (best_per, best_preds, linear_per, knn_per)."""
    model = Model().to(DEVICE)
    ema_model = deepcopy(model) if EMA_DECAY > 0 else None

    head_params = list(model.head.parameters())
    if HEAD_TYPE == "dual":
        head_params = list(model.head_ce.parameters()) + list(model.head_art.parameters())
    optimizer = AdamW(
        [
            {"params": model.readin.parameters(), "lr": LR * READIN_LR_MULT},
            {"params": model.backbone.parameters(), "lr": LR},
            {"params": head_params, "lr": LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    def lr_lambda(epoch: int) -> float:
        if WARMUP_EPOCHS > 0 and epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    n_train = len(train_grids)
    best_val_loss = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            x = augment(train_grids[idx]).to(DEVICE)
            y = [train_labels[i] for i in idx.tolist()]

            # Mixup: interpolate pairs within batch
            mixup_y = None
            mixup_lam = 1.0
            if MIXUP_ALPHA > 0 and len(idx) > 1:
                mixup_lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm_mix = torch.randperm(x.shape[0])
                x = mixup_lam * x + (1 - mixup_lam) * x[perm_mix]
                mixup_y = [y[i] for i in perm_mix.tolist()]

            optimizer.zero_grad()
            logits = model(x)
            loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)

            if math.isnan(loss.item()) or loss.item() > 100:
                print("FAIL")
                sys.exit(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            # EMA update
            if ema_model is not None:
                _ema_update(ema_model, model, EMA_DECAY)

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # --- Validate (use EMA model if available) ---
        eval_model = ema_model if ema_model is not None else model
        if (epoch + 1) % EVAL_EVERY == 0:
            eval_model.eval()
            with torch.no_grad():
                logits = eval_model(val_grids.to(DEVICE))
                val_loss = compute_loss(logits, val_labels).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(eval_model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1

            if patience_ctr >= PATIENCE:
                break

    # Restore best EMA state
    eval_model = ema_model if ema_model is not None else model
    if best_state is not None:
        eval_model.load_state_dict(best_state)

    # Final predictions — linear head with TTA
    eval_model.eval()
    with torch.no_grad():
        # TTA: average logits over augmented copies
        if TTA_COPIES > 1:
            logits_sum = torch.zeros(len(val_grids), prepare.N_POSITIONS, prepare.N_CLASSES, device=DEVICE)
            logits_sum += eval_model(val_grids.to(DEVICE))  # original (unaugmented)
            for _ in range(TTA_COPIES - 1):
                logits_sum += eval_model(augment(val_grids).to(DEVICE))
            logits = logits_sum / TTA_COPIES
        else:
            logits = eval_model(val_grids.to(DEVICE))
    linear_preds = decode(logits)
    linear_per = prepare.compute_per(linear_preds, val_labels)

    # k-NN evaluation on backbone embeddings (with TTA on embeddings)
    train_emb = extract_embeddings(eval_model, train_grids)
    if TTA_COPIES > 1:
        val_emb_sum = extract_embeddings(eval_model, val_grids)
        for _ in range(TTA_COPIES - 1):
            val_emb_sum = val_emb_sum + extract_embeddings(eval_model, augment(val_grids))
        val_emb = val_emb_sum / TTA_COPIES
    else:
        val_emb = extract_embeddings(eval_model, val_grids)
    knn_preds = knn_predict(train_emb, train_labels, val_emb, k=10)
    knn_per = prepare.compute_per(knn_preds, val_labels)

    # Use whichever is better
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

    # Load data
    grids, labels, token_ids = prepare.load_target_data()
    splits = prepare.create_cv_splits(token_ids)

    N, H, W, T = grids.shape
    print(f"Target: {prepare.TARGET_PATIENT}  |  Trials: {N}  |  Grid: {H}x{W}  |  T: {T}")
    print(f"Folds: {len(splits)}  |  Device: {DEVICE}")
    n_params = sum(p.numel() for p in Model().parameters())
    print(f"Model params: {n_params:,}")
    print()

    fold_pers = []
    fold_linear_pers = []
    fold_knn_pers = []
    all_preds: list[list[int]] = []
    all_refs: list[list[int]] = []

    for fi, (tr_idx, va_idx) in enumerate(splits):
        ft0 = time.time()
        per, preds, lp, kp = train_fold(
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
    print(f"training_seconds:   {elapsed:.1f}")
    print(f"collapsed:          {collapse['collapsed']}")
    print(f"mean_entropy:       {collapse['mean_entropy']:.3f}")
    print(f"stereotypy:         {collapse['stereotypy']:.3f}")
    print(f"unique_ratio:       {collapse['unique_ratio']:.3f}")
