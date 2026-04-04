#!/usr/bin/env python3
"""Full-recipe sweep: per-phoneme MFA epochs vs full-trial with k-NN, TTA, mixup.

Builds on sweep_tmin_perpos.py findings (per-phoneme > full-trial by ~4pp).
Now tests whether full recipe (mixup, k-NN, TTA, articulatory head) stacks.

Conditions:
  1. per_phoneme_t-0.15_full     — per-phoneme + full recipe (headline)
  2. per_phoneme_t-0.15_stride5  — shorter windows benefit from finer stride?
  3. per_phoneme_t0.0_full       — production-only control
  4. mean_pool_t0.0_full         — full-trial control (best full-trial head from sweep 1)
  5. per_phoneme_t-0.15_flat     — isolate articulatory head contribution

Each condition runs 3 seeds × 5 folds. Per-position PER breakdown included.

Usage:
    python scripts/sweep_full_recipe.py --paths configs/paths.yaml --device cuda
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
from speech_decoding.data.phoneme_map import ARTICULATORY_MATRIX  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================
PATIENT = "S14"
DEVICE = os.environ.get("DEVICE", "mps")
SEEDS = [42, 137, 256]
N_FOLDS = 5
N_POSITIONS = 3
N_CLASSES = 9
N_FEATURES = 15  # articulatory feature dimensions

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
MIXUP_ALPHA = 0.2

# Eval
KNN_K = 10
TTA_N = 16


# ============================================================
# DATA LOADING
# ============================================================

def load_full_trial(bids_root, tmin, tmax=1.0):
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
# CV SPLITS (same as sweep_tmin_perpos.py)
# ============================================================

def create_grouped_splits(labels):
    from sklearn.model_selection import GroupKFold

    n = len(labels)
    if all(len(l) == 3 for l in labels):
        token_ids = [tuple(l) for l in labels]
    else:
        n_trials = n // 3
        token_ids = [tuple([i % n_trials]) for i in range(n)]

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


class ArticulatoryHead(nn.Module):
    """6 articulatory feature heads -> fixed composition matrix -> 9 phonemes."""
    def __init__(self, d_in=64, n_features=N_FEATURES, n_classes=N_CLASSES):
        super().__init__()
        self.feature_proj = nn.Linear(d_in, n_features)
        A = torch.tensor(ARTICULATORY_MATRIX, dtype=torch.float32)  # (9, 15)
        self.register_buffer("A", A)

    def forward(self, h):
        # h: (B, D)
        feat = self.feature_proj(h)  # (B, 15)
        # Cosine similarity with each phoneme's articulatory vector
        feat_norm = F.normalize(feat, dim=-1)
        A_norm = F.normalize(self.A, dim=-1)
        logits = feat_norm @ A_norm.T  # (B, 9)
        return logits * 10.0  # scale for softmax


class FlatHead(nn.Module):
    def __init__(self, d_in=64, n_classes=N_CLASSES):
        super().__init__()
        self.head = nn.Linear(d_in, n_classes)

    def forward(self, h):
        return self.head(h)


class MeanPoolDecoder(nn.Module):
    """Mean pool time -> per-position heads. For full-trial mode."""
    def __init__(self, d_in=64, n_positions=3, head_cls=ArticulatoryHead):
        super().__init__()
        self.n_positions = n_positions
        self.heads = nn.ModuleList([head_cls(d_in=d_in) for _ in range(n_positions)])

    def forward(self, h):
        pooled = h.mean(dim=1)
        logits = [head(pooled) for head in self.heads]
        return torch.stack(logits, dim=1)  # (B, 3, 9)


class PerPhonemeDecoder(nn.Module):
    """Mean pool time -> single head. For per-phoneme mode."""
    def __init__(self, d_in=64, head_cls=ArticulatoryHead):
        super().__init__()
        self.head = head_cls(d_in=d_in)

    def forward(self, h):
        pooled = h.mean(dim=1)
        return self.head(pooled)  # (B, 9)


# ============================================================
# AUGMENTATION + MIXUP
# ============================================================

def augment(x):
    B, H, W, T = x.shape
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
    out = out * torch.exp(torch.randn(B, H, W, 1) * 0.2)
    out = out + torch.randn_like(out) * 0.03 * out.std()
    return out


def mixup_batch(x, y_onehot, alpha=MIXUP_ALPHA):
    """Mixup on grid data and one-hot labels."""
    if alpha <= 0:
        return x, y_onehot
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5
    perm = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[perm]
    y_mix = lam * y_onehot + (1 - lam) * y_onehot[perm]
    return x_mix, y_mix


# ============================================================
# LOSS
# ============================================================

def focal_ce(logits, targets, gamma=FOCAL_GAMMA):
    """Focal CE with label smoothing. targets can be hard (long) or soft (float)."""
    if targets.dtype == torch.long:
        ce = F.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING, reduction="none")
    else:
        # Soft targets from mixup
        log_probs = F.log_softmax(logits, dim=-1)
        ce = -(targets * log_probs).sum(dim=-1)
    if gamma > 0:
        pt = torch.exp(-ce)
        ce = ((1 - pt) ** gamma) * ce
    return ce.mean()


# ============================================================
# K-NN EVAL
# ============================================================

def knn_predict(train_feats, train_labels, val_feats, k=KNN_K):
    """Weighted k-NN on L2-normalized features."""
    train_feats = F.normalize(train_feats, dim=-1)
    val_feats = F.normalize(val_feats, dim=-1)
    sim = val_feats @ train_feats.T  # (V, Tr)
    topk_sim, topk_idx = sim.topk(k, dim=-1)

    # Distance-weighted voting
    weights = F.softmax(topk_sim * 10.0, dim=-1)  # (V, k)
    votes = torch.zeros(val_feats.size(0), N_CLASSES, device=val_feats.device)
    for i in range(val_feats.size(0)):
        for j in range(k):
            votes[i, train_labels[topk_idx[i, j]]] += weights[i, j]
    return votes  # (V, 9) soft predictions


# ============================================================
# TTA
# ============================================================

def tta_predict(readin, backbone, decoder, x_grid, n_aug=TTA_N, mode="per_phoneme"):
    """Test-time augmentation: average predictions over augmented copies."""
    all_logits = []
    for _ in range(n_aug):
        x_aug = augment(x_grid).to(next(readin.parameters()).device)
        feat = readin(x_aug)
        h = backbone(feat)
        logits = decoder(h)
        all_logits.append(logits)
    # Also do clean pass
    feat = readin(x_grid.to(next(readin.parameters()).device))
    h = backbone(feat)
    logits = decoder(h)
    all_logits.append(logits)
    return torch.stack(all_logits).mean(dim=0)  # (B, ...) averaged


# ============================================================
# TRAIN + EVAL ONE FOLD
# ============================================================

def train_fold(grids, labels, train_idx, val_idx, grid_shape, mode, stride, head_type, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    H, W = grid_shape
    readin = SpatialReadIn(H, W).to(DEVICE)
    d_flat = readin.d_flat
    backbone = Backbone(d_in=d_flat, stride=stride).to(DEVICE)

    head_cls = ArticulatoryHead if head_type == "articulatory" else FlatHead

    if mode == "per_phoneme":
        decoder = PerPhonemeDecoder(d_in=backbone.out_dim, head_cls=head_cls).to(DEVICE)
    else:
        decoder = MeanPoolDecoder(d_in=backbone.out_dim, head_cls=head_cls).to(DEVICE)

    params = list(readin.parameters()) + list(backbone.parameters()) + list(decoder.parameters())
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
        readin.train(); backbone.train(); decoder.train()
        perm = torch.randperm(len(train_grids))

        for start in range(0, len(train_grids), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            x = augment(train_grids[idx]).to(DEVICE)
            y_batch = [train_labels[i] for i in idx.tolist()]

            if mode == "full_trial":
                # Mixup on grids
                y_onehot = torch.zeros(len(y_batch), N_POSITIONS, N_CLASSES)
                for b, lab in enumerate(y_batch):
                    for p in range(N_POSITIONS):
                        y_onehot[b, p, lab[p] - 1] = 1.0
                x, y_onehot = mixup_batch(x, y_onehot.to(DEVICE))

                optimizer.zero_grad()
                feat = readin(x)
                h = backbone(feat)
                logits = decoder(h)  # (B, 3, 9)
                loss = sum(focal_ce(logits[:, p], y_onehot[:, p]) for p in range(N_POSITIONS)) / N_POSITIONS
            else:
                # Per-phoneme mixup
                y_onehot = torch.zeros(len(y_batch), N_CLASSES)
                for b, lab in enumerate(y_batch):
                    y_onehot[b, lab[0] - 1] = 1.0
                x, y_onehot = mixup_batch(x, y_onehot.to(DEVICE))

                optimizer.zero_grad()
                feat = readin(x)
                h = backbone(feat)
                logits = decoder(h)  # (B, 9)
                loss = focal_ce(logits, y_onehot)

            loss.backward()
            nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % EVAL_EVERY == 0:
            readin.eval(); backbone.eval(); decoder.eval()
            with torch.no_grad():
                feat = readin(val_grids.to(DEVICE))
                h = backbone(feat)
                logits = decoder(h)

                if mode == "full_trial":
                    val_loss = sum(
                        F.cross_entropy(logits[:, p], torch.tensor([l[p]-1 for l in val_labels], device=DEVICE))
                        for p in range(N_POSITIONS)
                    ).item() / N_POSITIONS
                else:
                    tgt = torch.tensor([l[0]-1 for l in val_labels], device=DEVICE)
                    val_loss = F.cross_entropy(logits, tgt).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "readin": deepcopy(readin.state_dict()),
                    "backbone": deepcopy(backbone.state_dict()),
                    "decoder": deepcopy(decoder.state_dict()),
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
        decoder.load_state_dict(best_state["decoder"])

    readin.eval(); backbone.eval(); decoder.eval()

    # === EVAL: TTA + k-NN ensemble ===
    with torch.no_grad():
        # Get train features for k-NN
        train_feat = readin(train_grids.to(DEVICE))
        train_h = backbone(train_feat)
        train_pooled = train_h.mean(dim=1)  # (Ntr, D)

        if mode == "per_phoneme":
            train_y = torch.tensor([l[0]-1 for l in train_labels], device=DEVICE)
        else:
            # For full-trial k-NN, pool across positions
            train_y = None  # k-NN per position below

        # TTA predictions
        tta_logits = tta_predict(readin, backbone, decoder, val_grids, n_aug=TTA_N, mode=mode)

        # k-NN predictions
        val_feat = readin(val_grids.to(DEVICE))
        val_h = backbone(val_feat)
        val_pooled = val_h.mean(dim=1)

    if mode == "per_phoneme":
        # Ensemble: 0.5 * classifier + 0.5 * k-NN
        knn_votes = knn_predict(train_pooled, train_y, val_pooled)
        tta_probs = F.softmax(tta_logits, dim=-1)
        ensemble = 0.5 * tta_probs + 0.5 * knn_votes
        preds_flat = (ensemble.argmax(dim=-1) + 1).cpu().tolist()
        preds = [[p] for p in preds_flat]
    else:
        # Full-trial: per-position k-NN + TTA
        preds_all = []
        for pos in range(N_POSITIONS):
            pos_train_y = torch.tensor([l[pos]-1 for l in train_labels], device=DEVICE)
            knn_votes = knn_predict(train_pooled, pos_train_y, val_pooled)
            tta_probs = F.softmax(tta_logits[:, pos], dim=-1)
            ensemble = 0.5 * tta_probs + 0.5 * knn_votes
            preds_all.append((ensemble.argmax(dim=-1) + 1).cpu().tolist())
        preds = [[preds_all[p][i] for p in range(N_POSITIONS)] for i in range(len(val_labels))]

    return preds, val_labels


# ============================================================
# METRICS
# ============================================================

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


def per_position_accuracy(preds, refs):
    """Per-position balanced accuracy for per-phoneme mode."""
    if len(preds[0]) == 1:
        # Per-phoneme: samples ordered pos1, pos1, ..., pos2, ...
        n_trials = len(preds) // 3
        accs = []
        for pos in range(3):
            start = pos * n_trials
            end = start + n_trials
            correct = sum(1 for i in range(start, end) if preds[i] == refs[i])
            accs.append(correct / n_trials)
        return accs
    else:
        # Full-trial
        accs = []
        for pos in range(3):
            correct = sum(1 for p, r in zip(preds, refs) if p[pos] == r[pos])
            accs.append(correct / len(preds))
        return accs


# ============================================================
# CONDITIONS
# ============================================================

# (name, mode, head_type, stride, tmin, tmax, seeds)
CONDITIONS = [
    # Per-phoneme with full recipe — the headline
    ("perphon_t-0.15_artic_s10", "per_phoneme", "articulatory", 10, -0.15, 0.5, SEEDS),
    # Stride=5 for shorter windows
    ("perphon_t-0.15_artic_s5",  "per_phoneme", "articulatory", 5,  -0.15, 0.5, SEEDS),
    # Production-only
    ("perphon_t0.0_artic_s10",   "per_phoneme", "articulatory", 10, 0.0,   0.5, [42]),
    # Full-trial control
    ("fulltrial_meanpool_artic",  "full_trial",  "articulatory", 10, 0.0,   1.0, [42]),
    # Flat head (isolate articulatory contribution)
    ("perphon_t-0.15_flat_s10",  "per_phoneme", "flat",         10, -0.15, 0.5, [42]),
]


def main():
    import argparse, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    args = parser.parse_args()

    if args.device:
        global DEVICE
        DEVICE = args.device

    with open(PROJECT_ROOT / args.paths) as f:
        paths = yaml.safe_load(f)
    bids_root = paths.get("ps_bids_root") or paths["bids_root"]

    print(f"Patient: {PATIENT}  |  Device: {DEVICE}")
    print(f"Full recipe: mixup={MIXUP_ALPHA}, k-NN k={KNN_K}, TTA n={TTA_N}")
    print(f"Epochs: {EPOCHS}  |  Folds: {N_FOLDS}")
    print("=" * 80)

    # Cache data loads
    data_cache = {}

    all_results = {}
    for cond_name, mode, head_type, stride, tmin, tmax, seeds in CONDITIONS:
        if args.conditions and cond_name not in args.conditions:
            continue

        print(f"\n>>> {cond_name} (mode={mode}, head={head_type}, stride={stride}, "
              f"tmin={tmin}, tmax={tmax}, seeds={seeds})")
        t0 = time.time()

        # Load data (cached)
        cache_key = (mode, tmin, tmax)
        if cache_key not in data_cache:
            if mode == "full_trial":
                data_cache[cache_key] = load_full_trial(bids_root, tmin=tmin, tmax=tmax)
            else:
                data_cache[cache_key] = load_per_phoneme(bids_root, tmin=tmin, tmax=tmax)
        grids, labels, grid_shape = data_cache[cache_key]
        print(f"  Data: {len(labels)} samples, grid {grid_shape}, T={grids.shape[-1]}")

        splits = create_grouped_splits(labels)

        seed_results = []
        for seed in seeds:
            all_preds, all_refs = [], []
            fold_pers = []
            for fi, (tr_idx, va_idx) in enumerate(splits):
                ft0 = time.time()
                preds, refs = train_fold(grids, labels, tr_idx, va_idx, grid_shape,
                                         mode, stride, head_type, seed)
                per = compute_per(preds, refs)
                fold_pers.append(per)
                all_preds.extend(preds)
                all_refs.extend(refs)
                print(f"    Seed {seed} Fold {fi+1}: PER={per:.4f}  ({time.time()-ft0:.1f}s)")

            mean_per = float(np.mean(fold_pers))
            overall_per = compute_per(all_preds, all_refs)
            pos_acc = per_position_accuracy(all_preds, all_refs)
            seed_results.append({
                "seed": seed, "mean_per": mean_per, "overall_per": overall_per,
                "fold_pers": fold_pers, "pos_acc": pos_acc,
            })
            print(f"  Seed {seed}: PER={mean_per:.4f} (overall={overall_per:.4f}) "
                  f"pos_acc=[{pos_acc[0]:.3f}, {pos_acc[1]:.3f}, {pos_acc[2]:.3f}]")

        # Aggregate across seeds
        mean_across_seeds = float(np.mean([r["mean_per"] for r in seed_results]))
        std_across_seeds = float(np.std([r["mean_per"] for r in seed_results]))
        elapsed = time.time() - t0

        all_results[cond_name] = {
            "mean_per": mean_across_seeds,
            "std_per": std_across_seeds,
            "seed_results": seed_results,
            "time": elapsed,
        }
        print(f"  >>> {cond_name}: PER={mean_across_seeds:.4f}±{std_across_seeds:.4f}  [{elapsed:.0f}s]")

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Condition':<35} {'Mean PER':<12} {'Std':<10} {'Seeds':<8} {'Time':<8}")
    print("-" * 80)
    for name, r in all_results.items():
        n_seeds = len(r["seed_results"])
        print(f"{name:<35} {r['mean_per']:.4f}       {r['std_per']:.4f}     {n_seeds:<8} {r['time']:.0f}s")

    # Per-position breakdown
    print("\nPer-position accuracy (seed-averaged):")
    print(f"{'Condition':<35} {'Pos 1':<10} {'Pos 2':<10} {'Pos 3':<10}")
    print("-" * 65)
    for name, r in all_results.items():
        avg_pos = [float(np.mean([sr["pos_acc"][p] for sr in r["seed_results"]])) for p in range(3)]
        print(f"{name:<35} {avg_pos[0]:.3f}      {avg_pos[1]:.3f}      {avg_pos[2]:.3f}")


if __name__ == "__main__":
    main()
