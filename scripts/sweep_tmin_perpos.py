#!/usr/bin/env python3
"""Quick sweep: tmin × loss_type on S14, per-patient, grouped-by-token CV.

Conditions:
  1. tmin=0.0, full-trial CE (3 equal windows) — current baseline
  2. tmin=-0.5, full-trial CE (3 equal windows) — pre-production included
  3. tmin=0.0, per-phoneme MFA epochs — one phoneme per forward pass
  4. tmin=-0.15, per-phoneme MFA epochs — 150ms padding for MFA noise

All use: SpatialConv(8ch, pool 4×8) + BiGRU(32,2L) + stride=10
         Grouped-by-token 5-fold CV, seed=42, focal CE + label smoothing
         NO k-NN, NO TTA, NO mixup (speed — just test the data question)

Usage:
    python scripts/sweep_tmin_perpos.py --paths configs/paths.yaml --device cuda
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.data.bids_dataset import load_patient_data, load_per_position_data  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================
PATIENT = "S14"
DEVICE = os.environ.get("DEVICE", "mps")
SEED = 42
N_FOLDS = 5
N_POSITIONS = 3
N_CLASSES = 9

# Training
EPOCHS = 300
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
WARMUP_EPOCHS = 20
PATIENCE = 7
EVAL_EVERY = 10
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0


# ============================================================
# DATA LOADING
# ============================================================

def load_full_trial(bids_root, tmin, tmax=1.0):
    """Load full-trial epochs (position-1 extraction, 3 phonemes per sample)."""
    ds = load_patient_data(
        PATIENT, bids_root, task="PhonemeSequence", n_phons=3,
        tmin=tmin, tmax=tmax,
    )
    grids, labels = [], []
    for i in range(len(ds)):
        g, l, _ = ds[i]
        grids.append(g)
        labels.append(l)
    grids = torch.tensor(np.stack(grids), dtype=torch.float32)
    return grids, labels, ds.grid_shape


def load_per_phoneme(bids_root, tmin, tmax=0.5):
    """Load per-phoneme MFA epochs (each phoneme separately)."""
    ds = load_per_position_data(
        PATIENT, bids_root, task="PhonemeSequence", n_phons=3,
        tmin=tmin, tmax=tmax,
    )
    grids, labels = [], []
    for i in range(len(ds)):
        g, l, _ = ds[i]
        grids.append(g)
        labels.append(l)
    grids = torch.tensor(np.stack(grids), dtype=torch.float32)
    return grids, labels, ds.grid_shape


# ============================================================
# CV SPLITS
# ============================================================

def create_grouped_splits(labels):
    """Grouped-by-token CV. For per-phoneme, group by original trial token."""
    from sklearn.model_selection import GroupKFold

    # Token = the 3-phoneme sequence this sample belongs to
    # For full-trial: labels are [[p1,p2,p3], ...] — token IS the label
    # For per-phoneme: labels are [[p], ...] — need to reconstruct token groups
    # Since per-phoneme data is ordered pos1, pos1, ..., pos2, pos2, ..., pos3, pos3, ...
    # each position has n_trials entries. Group by trial index within position.

    n = len(labels)
    if all(len(l) == 3 for l in labels):
        # Full-trial: group by token identity
        token_ids = [tuple(l) for l in labels]
    else:
        # Per-phoneme: n = n_trials * 3. Group by trial index.
        n_trials = n // 3
        # trial i across positions should be same fold
        token_ids = [i % n_trials for i in range(n)]
        token_ids = [tuple([token_ids[i]]) for i in range(n)]

    unique = sorted(set(token_ids))
    token_to_group = {t: i for i, t in enumerate(unique)}
    groups = np.array([token_to_group[t] for t in token_ids])

    gkf = GroupKFold(n_splits=min(N_FOLDS, len(unique)))
    X = np.zeros(n)
    rng = np.random.RandomState(42)

    for _ in range(512):
        perm = rng.permutation(len(unique))
        shuffled = np.array([perm[g] for g in groups])
        splits = []
        valid = True
        for train_idx, val_idx in gkf.split(X, groups=shuffled):
            splits.append((sorted(train_idx.tolist()), sorted(val_idx.tolist())))
        if splits:
            return splits

    raise RuntimeError("Failed to create CV splits")


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
                 gru_dropout=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_in, d, kernel_size=stride, stride=stride), nn.GELU(),
        )
        self.gru = nn.GRU(d, gru_hidden, num_layers=gru_layers, batch_first=True,
                          bidirectional=True, dropout=gru_dropout if gru_layers > 1 else 0.0)
        self.out_dim = gru_hidden * 2

    def forward(self, x):
        x = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)
        h, _ = self.gru(x)
        return h


class EqualWindowHead(nn.Module):
    """Mean-pool into 3 equal windows -> per-position CE."""
    def __init__(self, d_in=64, n_positions=3, n_classes=9):
        super().__init__()
        self.n_positions = n_positions
        self.heads = nn.ModuleList([nn.Linear(d_in, n_classes) for _ in range(n_positions)])

    def forward(self, h):
        # h: (B, T, D)
        B, T, D = h.shape
        chunk = T // self.n_positions
        logits = []
        for p in range(self.n_positions):
            start = p * chunk
            end = start + chunk if p < self.n_positions - 1 else T
            pooled = h[:, start:end, :].mean(dim=1)
            logits.append(self.heads[p](pooled))
        return torch.stack(logits, dim=1)  # (B, 3, 9)


class MeanPoolHead(nn.Module):
    """Global mean-pool all time -> 3×9 logits (positions share temporal info)."""
    def __init__(self, d_in=64, n_positions=3, n_classes=9):
        super().__init__()
        self.n_positions = n_positions
        self.heads = nn.ModuleList([nn.Linear(d_in, n_classes) for _ in range(n_positions)])

    def forward(self, h):
        # h: (B, T, D)
        pooled = h.mean(dim=1)  # (B, D)
        logits = [head(pooled) for head in self.heads]
        return torch.stack(logits, dim=1)  # (B, 3, 9)


class LearnedAttnHead(nn.Module):
    """Learned temporal attention queries per position (= current CEPositionHead)."""
    def __init__(self, d_in=64, n_positions=3, n_classes=9):
        super().__init__()
        self.n_positions = n_positions
        self.scale = math.sqrt(d_in)
        self.queries = nn.Parameter(torch.randn(n_positions, d_in) * 0.02)
        self.heads = nn.ModuleList([nn.Linear(d_in, n_classes) for _ in range(n_positions)])

    def forward(self, h):
        # h: (B, T, D)
        # Dot-product attention: each query attends to temporal frames
        attn_scores = torch.matmul(h, self.queries.T) / self.scale  # (B, T, n_pos)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T, n_pos)
        pooled = torch.einsum("btp,btd->bpd", attn_weights, h)  # (B, n_pos, D)
        logits = [self.heads[p](pooled[:, p, :]) for p in range(self.n_positions)]
        return torch.stack(logits, dim=1)  # (B, 3, 9)


class PerPhonemeHead(nn.Module):
    """Mean-pool all time -> single 9-way CE."""
    def __init__(self, d_in=64, n_classes=9):
        super().__init__()
        self.head = nn.Linear(d_in, n_classes)

    def forward(self, h):
        # h: (B, T, D)
        pooled = h.mean(dim=1)
        return self.head(pooled)  # (B, 9)


# ============================================================
# AUGMENTATION (minimal — just time shift + noise)
# ============================================================

def augment(x):
    B, H, W, T = x.shape
    # Time shift ±20 frames
    shifts = torch.randint(-20, 21, (B,))
    out = torch.zeros_like(x)
    for i in range(B):
        s = shifts[i].item()
        if s > 0:
            out[i, :, :, s:] = x[i, :, :, :T - s]
        elif s < 0:
            out[i, :, :, :T + s] = x[i, :, :, -s:]
        else:
            out[i] = x[i]
    # Amplitude scale
    out = out * torch.exp(torch.randn(B, H, W, 1) * 0.2)
    # Small noise
    out = out + torch.randn_like(out) * 0.03 * out.std()
    return out


# ============================================================
# LOSS
# ============================================================

def focal_ce(logits, targets, gamma=FOCAL_GAMMA):
    ce = F.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING, reduction="none")
    if gamma > 0:
        pt = torch.exp(-ce)
        ce = ((1 - pt) ** gamma) * ce
    return ce.mean()


# ============================================================
# TRAIN + EVAL ONE FOLD
# ============================================================

def train_fold(grids, labels, train_idx, val_idx, grid_shape, mode, head_type="equal_window"):
    """Train one fold. mode = 'full_trial' or 'per_phoneme'. head_type for full_trial."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    H, W = grid_shape
    readin = SpatialReadIn(H, W).to(DEVICE)
    d_flat = readin.d_flat
    backbone = Backbone(d_in=d_flat).to(DEVICE)

    if mode == "per_phoneme":
        head = PerPhonemeHead(d_in=backbone.out_dim).to(DEVICE)
    elif head_type == "equal_window":
        head = EqualWindowHead(d_in=backbone.out_dim).to(DEVICE)
    elif head_type == "mean_pool":
        head = MeanPoolHead(d_in=backbone.out_dim).to(DEVICE)
    elif head_type == "learned_attn":
        head = LearnedAttnHead(d_in=backbone.out_dim).to(DEVICE)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    params = list(readin.parameters()) + list(backbone.parameters()) + list(head.parameters())
    optimizer = AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    train_grids = grids[train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_grids = grids[val_idx]
    val_labels = [labels[i] for i in val_idx]

    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(EPOCHS):
        readin.train(); backbone.train(); head.train()
        perm = torch.randperm(len(train_grids))

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(train_grids), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            x = augment(train_grids[idx]).to(DEVICE)
            y_batch = [train_labels[i] for i in idx.tolist()]

            optimizer.zero_grad()
            feat = readin(x)
            h = backbone(feat)
            logits = head(h)

            if mode == "full_trial":
                # logits: (B, 3, 9), labels: [[p1,p2,p3], ...]
                loss = torch.tensor(0.0, device=DEVICE)
                for pos in range(N_POSITIONS):
                    tgt = torch.tensor([l[pos] - 1 for l in y_batch], dtype=torch.long, device=DEVICE)
                    loss = loss + focal_ce(logits[:, pos, :], tgt)
                loss = loss / N_POSITIONS
            else:
                # logits: (B, 9), labels: [[p], ...]
                tgt = torch.tensor([l[0] - 1 for l in y_batch], dtype=torch.long, device=DEVICE)
                loss = focal_ce(logits, tgt)

            loss.backward()
            nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % EVAL_EVERY == 0:
            readin.eval(); backbone.eval(); head.eval()
            with torch.no_grad():
                feat = readin(val_grids.to(DEVICE))
                h = backbone(feat)
                logits = head(h)

                if mode == "full_trial":
                    val_loss = torch.tensor(0.0, device=DEVICE)
                    for pos in range(N_POSITIONS):
                        tgt = torch.tensor([l[pos] - 1 for l in val_labels], dtype=torch.long, device=DEVICE)
                        val_loss = val_loss + focal_ce(logits[:, pos, :], tgt)
                    val_loss = (val_loss / N_POSITIONS).item()
                else:
                    tgt = torch.tensor([l[0] - 1 for l in val_labels], dtype=torch.long, device=DEVICE)
                    val_loss = focal_ce(logits, tgt).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "readin": deepcopy(readin.state_dict()),
                    "backbone": deepcopy(backbone.state_dict()),
                    "head": deepcopy(head.state_dict()),
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

    # Restore best
    if best_state:
        readin.load_state_dict(best_state["readin"])
        backbone.load_state_dict(best_state["backbone"])
        head.load_state_dict(best_state["head"])

    # Evaluate
    readin.eval(); backbone.eval(); head.eval()
    with torch.no_grad():
        feat = readin(val_grids.to(DEVICE))
        h = backbone(feat)
        logits = head(h)

    if mode == "full_trial":
        preds = (logits.argmax(dim=-1) + 1).cpu().tolist()  # (B, 3)
    else:
        preds_flat = (logits.argmax(dim=-1) + 1).cpu().tolist()  # (B,) single phonemes
        preds = [[p] for p in preds_flat]

    return preds, val_labels


def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[m]


def compute_per(preds, refs):
    total_dist = sum(edit_distance(p, r) for p, r in zip(preds, refs))
    total_len = sum(len(r) for r in refs)
    return total_dist / max(total_len, 1)


# ============================================================
# MAIN
# ============================================================

# (name, mode, head_type, tmin, tmax)
CONDITIONS = [
    # Full-trial with different heads (tmin=0.0)
    ("equal_window_t0.0",    "full_trial", "equal_window",  0.0,  1.0),
    ("mean_pool_t0.0",       "full_trial", "mean_pool",     0.0,  1.0),
    ("learned_attn_t0.0",    "full_trial", "learned_attn",  0.0,  1.0),
    # Full-trial with tmin=-0.5
    ("learned_attn_t-0.5",   "full_trial", "learned_attn",  -0.5, 1.0),
    # Per-phoneme MFA epochs
    ("per_phoneme_t0.0",     "per_phoneme", None,            0.0,  0.5),
    ("per_phoneme_t-0.15",   "per_phoneme", None,           -0.15, 0.5),
]


def main():
    import argparse, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Subset of condition names to run")
    args = parser.parse_args()

    if args.device:
        global DEVICE
        DEVICE = args.device

    with open(PROJECT_ROOT / args.paths) as f:
        paths = yaml.safe_load(f)
    bids_root = paths.get("ps_bids_root") or paths["bids_root"]

    print(f"Patient: {PATIENT}  |  Device: {DEVICE}  |  Seed: {SEED}")
    print(f"Epochs: {EPOCHS}  |  Folds: {N_FOLDS}")
    print("=" * 70)

    results = {}
    for cond_name, mode, head_type, tmin, tmax in CONDITIONS:
        if args.conditions and cond_name not in args.conditions:
            continue

        print(f"\n>>> {cond_name} (mode={mode}, head={head_type}, tmin={tmin}, tmax={tmax})")
        t0 = time.time()

        # Load data
        if mode == "full_trial":
            grids, labels, grid_shape = load_full_trial(bids_root, tmin=tmin, tmax=tmax)
        else:
            grids, labels, grid_shape = load_per_phoneme(bids_root, tmin=tmin, tmax=tmax)

        print(f"  Loaded: {len(labels)} samples, grid {grid_shape}, T={grids.shape[-1]}")

        # Create splits
        splits = create_grouped_splits(labels)

        # Train + eval each fold
        all_preds, all_refs = [], []
        fold_pers = []
        for fi, (tr_idx, va_idx) in enumerate(splits):
            ft0 = time.time()
            preds, refs = train_fold(grids, labels, tr_idx, va_idx, grid_shape, mode, head_type=head_type or "equal_window")
            per = compute_per(preds, refs)
            fold_pers.append(per)
            all_preds.extend(preds)
            all_refs.extend(refs)
            print(f"  Fold {fi+1}: PER={per:.4f}  ({time.time()-ft0:.1f}s)")

        mean_per = float(np.mean(fold_pers))
        std_per = float(np.std(fold_pers))
        overall_per = compute_per(all_preds, all_refs)
        elapsed = time.time() - t0

        results[cond_name] = {
            "mean_per": mean_per,
            "std_per": std_per,
            "overall_per": overall_per,
            "fold_pers": fold_pers,
            "time": elapsed,
        }
        print(f"  >>> {cond_name}: PER={mean_per:.4f}±{std_per:.4f} (overall={overall_per:.4f})  [{elapsed:.0f}s]")

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Condition':<30} {'Mean PER':<12} {'Std':<10} {'Overall':<12} {'Time':<8}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<30} {r['mean_per']:.4f}       {r['std_per']:.4f}     {r['overall_per']:.4f}       {r['time']:.0f}s")


if __name__ == "__main__":
    main()
