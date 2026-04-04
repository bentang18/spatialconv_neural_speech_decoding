#!/usr/bin/env python3
"""Head-to-head: learned temporal attention vs per-phoneme MFA epochs.

Fair comparison — identical training loop, recipe, and evaluation for both.
3 seeds each, full recipe (mixup, k-NN, TTA).

Tests both flat and articulatory heads for each approach.

Conditions:
  1. learned_attn_flat     — full-trial, learned temporal attention, flat head, 3 seeds
  2. learned_attn_artic    — full-trial, learned temporal attention, articulatory head, 3 seeds
  3. perphon_flat          — per-phoneme MFA epochs, flat head, 3 seeds
  4. perphon_artic         — per-phoneme MFA epochs, articulatory head, 3 seeds

Usage:
    python scripts/sweep_head_to_head.py --paths configs/paths.yaml --device cuda
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
N_FEATURES = 15

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
KNN_K = 10
TTA_N = 16


# ============================================================
# DATA
# ============================================================

def load_full_trial(bids_root):
    ds = load_patient_data(PATIENT, bids_root, task="PhonemeSequence", n_phons=3, tmin=0.0, tmax=1.0)
    grids, labels = [], []
    for i in range(len(ds)):
        g, l, _ = ds[i]
        grids.append(g)
        labels.append(l)
    return torch.tensor(np.stack(grids), dtype=torch.float32), labels, ds.grid_shape


def load_per_phoneme(bids_root):
    ds = load_per_position_data(PATIENT, bids_root, task="PhonemeSequence", n_phons=3, tmin=-0.15, tmax=0.5)
    grids, labels = [], []
    for i in range(len(ds)):
        g, l, _ = ds[i]
        grids.append(g)
        labels.append(l)
    return torch.tensor(np.stack(grids), dtype=torch.float32), labels, ds.grid_shape


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
        splits = list(gkf.split(X, groups=shuffled))
        if splits:
            return [(sorted(tr.tolist()), sorted(va.tolist())) for tr, va in splits]
    raise RuntimeError("Failed")


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
    def __init__(self, d_in=256, d=32, gru_hidden=32, gru_layers=2, stride=10, gru_dropout=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.temporal_conv = nn.Sequential(nn.Conv1d(d_in, d, kernel_size=stride, stride=stride), nn.GELU())
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
    def __init__(self, d_in=64):
        super().__init__()
        self.feature_proj = nn.Linear(d_in, N_FEATURES)
        A = torch.tensor(ARTICULATORY_MATRIX, dtype=torch.float32)
        self.register_buffer("A", A)

    def forward(self, h):
        feat = F.normalize(self.feature_proj(h), dim=-1)
        A_norm = F.normalize(self.A, dim=-1)
        return (feat @ A_norm.T) * 10.0


class FlatHead(nn.Module):
    def __init__(self, d_in=64):
        super().__init__()
        self.head = nn.Linear(d_in, N_CLASSES)

    def forward(self, h):
        return self.head(h)


# ============================================================
# DECODERS
# ============================================================

class LearnedAttnDecoder(nn.Module):
    """Learned temporal attention queries per phoneme position (= CEPositionHead)."""
    def __init__(self, d_in=64, n_positions=3, head_cls=FlatHead):
        super().__init__()
        self.n_positions = n_positions
        self.scale = math.sqrt(d_in)
        self.queries = nn.Parameter(torch.randn(n_positions, d_in) * 0.02)
        self.heads = nn.ModuleList([head_cls(d_in=d_in) for _ in range(n_positions)])

    def forward(self, h):
        # h: (B, T, D)
        attn_scores = torch.matmul(h, self.queries.T) / self.scale
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.einsum("btp,btd->bpd", attn_weights, h)
        logits = [self.heads[p](pooled[:, p]) for p in range(self.n_positions)]
        return torch.stack(logits, dim=1)  # (B, 3, 9)


class MeanPoolDecoder(nn.Module):
    """Global mean pool -> per-position heads. For full-trial baseline."""
    def __init__(self, d_in=64, n_positions=3, head_cls=FlatHead):
        super().__init__()
        self.n_positions = n_positions
        self.heads = nn.ModuleList([head_cls(d_in=d_in) for _ in range(n_positions)])

    def forward(self, h):
        pooled = h.mean(dim=1)
        return torch.stack([head(pooled) for head in self.heads], dim=1)


class PerPhonemeDecoder(nn.Module):
    """Mean pool -> single head. For per-phoneme mode."""
    def __init__(self, d_in=64, head_cls=FlatHead):
        super().__init__()
        self.head = head_cls(d_in=d_in)

    def forward(self, h):
        return self.head(h.mean(dim=1))


# ============================================================
# AUGMENTATION + MIXUP + LOSS
# ============================================================

def augment(x):
    B, H, W, T = x.shape
    shifts = torch.randint(-20, 21, (B,))
    out = torch.zeros_like(x)
    for i in range(B):
        s = shifts[i].item()
        if s > 0: out[i, :, :, s:] = x[i, :, :, :T - s]
        elif s < 0: out[i, :, :, :T + s] = x[i, :, :, -s:]
        else: out[i] = x[i]
    out = out * torch.exp(torch.randn(B, H, W, 1) * 0.2)
    return out + torch.randn_like(out) * 0.03 * out.std()


def mixup_batch(x, y_onehot, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y_onehot
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    perm = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[perm], lam * y_onehot + (1 - lam) * y_onehot[perm]


def focal_ce(logits, targets, gamma=FOCAL_GAMMA):
    if targets.dtype == torch.long:
        ce = F.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING, reduction="none")
    else:
        log_probs = F.log_softmax(logits, dim=-1)
        ce = -(targets * log_probs).sum(dim=-1)
    if gamma > 0:
        pt = torch.exp(-ce)
        ce = ((1 - pt) ** gamma) * ce
    return ce.mean()


# ============================================================
# K-NN + TTA
# ============================================================

def knn_predict(train_feats, train_labels, val_feats, k=KNN_K):
    train_feats = F.normalize(train_feats, dim=-1)
    val_feats = F.normalize(val_feats, dim=-1)
    sim = val_feats @ train_feats.T
    topk_sim, topk_idx = sim.topk(k, dim=-1)
    weights = F.softmax(topk_sim * 10.0, dim=-1)
    votes = torch.zeros(val_feats.size(0), N_CLASSES, device=val_feats.device)
    for i in range(val_feats.size(0)):
        for j in range(k):
            votes[i, train_labels[topk_idx[i, j]]] += weights[i, j]
    return votes


def tta_predict(readin, backbone, decoder, x_grid, n_aug=TTA_N):
    all_logits = []
    dev = next(readin.parameters()).device
    for _ in range(n_aug):
        x_aug = augment(x_grid).to(dev)
        all_logits.append(decoder(backbone(readin(x_aug))))
    all_logits.append(decoder(backbone(readin(x_grid.to(dev)))))
    return torch.stack(all_logits).mean(dim=0)


# ============================================================
# TRAIN + EVAL ONE FOLD
# ============================================================

def train_fold(grids, labels, train_idx, val_idx, grid_shape, mode, decoder_cls, head_cls, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    H, W = grid_shape
    readin = SpatialReadIn(H, W).to(DEVICE)
    backbone = Backbone(d_in=readin.d_flat).to(DEVICE)

    if mode == "per_phoneme":
        decoder = PerPhonemeDecoder(d_in=backbone.out_dim, head_cls=head_cls).to(DEVICE)
    else:
        decoder = decoder_cls(d_in=backbone.out_dim, head_cls=head_cls).to(DEVICE)

    params = list(readin.parameters()) + list(backbone.parameters()) + list(decoder.parameters())
    optimizer = AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        return 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    train_grids, val_grids = grids[train_idx], grids[val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]

    best_val_loss, best_state, patience_ctr = float("inf"), None, 0

    for epoch in range(EPOCHS):
        readin.train(); backbone.train(); decoder.train()
        perm = torch.randperm(len(train_grids))

        for start in range(0, len(train_grids), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            x = augment(train_grids[idx]).to(DEVICE)
            y_batch = [train_labels[i] for i in idx.tolist()]

            if mode == "full_trial":
                y_onehot = torch.zeros(len(y_batch), N_POSITIONS, N_CLASSES)
                for b, lab in enumerate(y_batch):
                    for p in range(N_POSITIONS):
                        y_onehot[b, p, lab[p] - 1] = 1.0
                x, y_onehot = mixup_batch(x, y_onehot.to(DEVICE))
                optimizer.zero_grad()
                logits = decoder(backbone(readin(x)))
                loss = sum(focal_ce(logits[:, p], y_onehot[:, p]) for p in range(N_POSITIONS)) / N_POSITIONS
            else:
                y_onehot = torch.zeros(len(y_batch), N_CLASSES)
                for b, lab in enumerate(y_batch):
                    y_onehot[b, lab[0] - 1] = 1.0
                x, y_onehot = mixup_batch(x, y_onehot.to(DEVICE))
                optimizer.zero_grad()
                logits = decoder(backbone(readin(x)))
                loss = focal_ce(logits, y_onehot)

            loss.backward()
            nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % EVAL_EVERY == 0:
            readin.eval(); backbone.eval(); decoder.eval()
            with torch.no_grad():
                logits = decoder(backbone(readin(val_grids.to(DEVICE))))
                if mode == "full_trial":
                    val_loss = sum(
                        F.cross_entropy(logits[:, p], torch.tensor([l[p]-1 for l in val_labels], device=DEVICE)).item()
                        for p in range(N_POSITIONS)
                    ) / N_POSITIONS
                else:
                    val_loss = F.cross_entropy(logits, torch.tensor([l[0]-1 for l in val_labels], device=DEVICE)).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: deepcopy(v.state_dict()) for k, v in
                              [("r", readin), ("b", backbone), ("d", decoder)]}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

    if best_state:
        readin.load_state_dict(best_state["r"])
        backbone.load_state_dict(best_state["b"])
        decoder.load_state_dict(best_state["d"])

    readin.eval(); backbone.eval(); decoder.eval()

    # === EVAL: TTA + k-NN ===
    with torch.no_grad():
        train_h = backbone(readin(train_grids.to(DEVICE)))
        train_pooled = train_h.mean(dim=1)
        val_h = backbone(readin(val_grids.to(DEVICE)))
        val_pooled = val_h.mean(dim=1)
        tta_logits = tta_predict(readin, backbone, decoder, val_grids)

    if mode == "per_phoneme":
        train_y = torch.tensor([l[0]-1 for l in train_labels], device=DEVICE)
        knn_votes = knn_predict(train_pooled, train_y, val_pooled)
        ensemble = 0.5 * F.softmax(tta_logits, dim=-1) + 0.5 * knn_votes
        preds = [[int(p)+1] for p in ensemble.argmax(dim=-1).cpu().tolist()]
    else:
        preds_all = []
        for pos in range(N_POSITIONS):
            train_y = torch.tensor([l[pos]-1 for l in train_labels], device=DEVICE)
            knn_votes = knn_predict(train_pooled, train_y, val_pooled)
            ensemble = 0.5 * F.softmax(tta_logits[:, pos], dim=-1) + 0.5 * knn_votes
            preds_all.append((ensemble.argmax(dim=-1)+1).cpu().tolist())
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
    return sum(edit_distance(p, r) for p, r in zip(preds, refs)) / max(sum(len(r) for r in refs), 1)


# ============================================================
# CONDITIONS
# ============================================================

CONDITIONS = [
    # (name, mode, decoder_cls, head_cls, seeds)
    ("learned_attn_flat",   "full_trial",  LearnedAttnDecoder, FlatHead,         SEEDS),
    ("learned_attn_artic",  "full_trial",  LearnedAttnDecoder, ArticulatoryHead,  SEEDS),
    ("meanpool_flat",       "full_trial",  MeanPoolDecoder,    FlatHead,         SEEDS),
    ("perphon_flat",        "per_phoneme", None,               FlatHead,         SEEDS),
    ("perphon_artic",       "per_phoneme", None,               ArticulatoryHead,  SEEDS),
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
    print(f"Full recipe: mixup={MIXUP_ALPHA}, k-NN k={KNN_K}, TTA n={TTA_N}, focal γ={FOCAL_GAMMA}")
    print(f"3 seeds × 5 folds per condition")
    print("=" * 80)

    # Pre-load both data types
    print("Loading full-trial data...")
    grids_ft, labels_ft, gs_ft = load_full_trial(bids_root)
    splits_ft = create_grouped_splits(labels_ft)
    print(f"  {len(labels_ft)} trials, grid {gs_ft}, T={grids_ft.shape[-1]}")

    print("Loading per-phoneme data...")
    grids_pp, labels_pp, gs_pp = load_per_phoneme(bids_root)
    splits_pp = create_grouped_splits(labels_pp)
    print(f"  {len(labels_pp)} samples, grid {gs_pp}, T={grids_pp.shape[-1]}")

    all_results = {}
    for cond_name, mode, decoder_cls, head_cls, seeds in CONDITIONS:
        if args.conditions and cond_name not in args.conditions:
            continue

        grids = grids_ft if mode == "full_trial" else grids_pp
        labels = labels_ft if mode == "full_trial" else labels_pp
        splits = splits_ft if mode == "full_trial" else splits_pp

        print(f"\n>>> {cond_name} (mode={mode}, head={head_cls.__name__})")
        t0 = time.time()

        seed_pers = []
        for seed in seeds:
            all_preds, all_refs, fold_pers = [], [], []
            for fi, (tr, va) in enumerate(splits):
                ft0 = time.time()
                preds, refs = train_fold(grids, labels, tr, va,
                                         gs_ft if mode == "full_trial" else gs_pp,
                                         mode, decoder_cls, head_cls, seed)
                per = compute_per(preds, refs)
                fold_pers.append(per)
                all_preds.extend(preds); all_refs.extend(refs)
                print(f"    Seed {seed} Fold {fi+1}: PER={per:.4f}  ({time.time()-ft0:.1f}s)")

            mean_per = float(np.mean(fold_pers))
            seed_pers.append(mean_per)
            print(f"  Seed {seed}: PER={mean_per:.4f}")

        mean_all = float(np.mean(seed_pers))
        std_all = float(np.std(seed_pers))
        elapsed = time.time() - t0
        all_results[cond_name] = {"mean": mean_all, "std": std_all, "seeds": seed_pers, "time": elapsed}
        print(f"  >>> {cond_name}: PER={mean_all:.4f}±{std_all:.4f}  [{elapsed:.0f}s]")

    # Summary
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD COMPARISON (same training loop, same recipe)")
    print("=" * 80)
    print(f"{'Condition':<25} {'Mean PER':<12} {'Std':<10} {'Seed 42':<10} {'Seed 137':<10} {'Seed 256':<10}")
    print("-" * 80)
    for name, r in all_results.items():
        seeds_str = "  ".join(f"{s:.4f}" for s in r["seeds"])
        print(f"{name:<25} {r['mean']:.4f}       {r['std']:.4f}     {seeds_str}")

    # Winner
    if all_results:
        best = min(all_results.items(), key=lambda x: x[1]["mean"])
        print(f"\nBest: {best[0]} at PER={best[1]['mean']:.4f}±{best[1]['std']:.4f}")


if __name__ == "__main__":
    main()
