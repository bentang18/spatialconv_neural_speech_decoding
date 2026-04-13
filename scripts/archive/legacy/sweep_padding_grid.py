#!/usr/bin/env python3
"""Sweep tmin/tmax padding for per-phoneme MFA epochs.

Fine grid around the sweet spot found in sweep 1 (tmin=-0.15 best).
Also tests tmax variation (how much post-phoneme signal matters).

Simplified recipe (no mixup/k-NN/TTA) for speed — just the data question.

Usage:
    python scripts/sweep_padding_grid.py --paths configs/paths.yaml --device cuda
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

from speech_decoding.data.bids_dataset import load_per_position_data  # noqa: E402

PATIENT = "S14"
DEVICE = os.environ.get("DEVICE", "mps")
SEED = 42
N_FOLDS = 5
N_CLASSES = 9
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


def load_per_phoneme(bids_root, tmin, tmax):
    ds = load_per_position_data(
        PATIENT, bids_root, task="PhonemeSequence", n_phons=3, tmin=tmin, tmax=tmax,
    )
    grids, labels = [], []
    for i in range(len(ds)):
        g, l, _ = ds[i]
        grids.append(g)
        labels.append(l)
    return torch.tensor(np.stack(grids), dtype=torch.float32), labels, ds.grid_shape


def create_grouped_splits(labels):
    from sklearn.model_selection import GroupKFold
    n = len(labels)
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
    raise RuntimeError("Failed")


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
        x, _ = self.gru(x.permute(0, 2, 1))
        return x


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


def focal_ce(logits, targets, gamma=FOCAL_GAMMA):
    ce = F.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING, reduction="none")
    if gamma > 0:
        pt = torch.exp(-ce)
        ce = ((1 - pt) ** gamma) * ce
    return ce.mean()


def train_eval_fold(grids, labels, train_idx, val_idx, grid_shape):
    torch.manual_seed(SEED); np.random.seed(SEED)
    H, W = grid_shape
    readin = SpatialReadIn(H, W).to(DEVICE)
    backbone = Backbone(d_in=readin.d_flat).to(DEVICE)
    head = nn.Linear(backbone.out_dim, N_CLASSES).to(DEVICE)

    params = list(readin.parameters()) + list(backbone.parameters()) + list(head.parameters())
    optimizer = AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = LambdaLR(optimizer, lambda e: ((e+1)/WARMUP_EPOCHS) if e < WARMUP_EPOCHS
                         else 0.5*(1+math.cos(math.pi*(e-WARMUP_EPOCHS)/max(EPOCHS-WARMUP_EPOCHS,1))))

    train_grids, val_grids = grids[train_idx], grids[val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]

    best_val_loss, best_state, patience_ctr = float("inf"), None, 0

    for epoch in range(EPOCHS):
        readin.train(); backbone.train(); head.train()
        perm = torch.randperm(len(train_grids))
        for start in range(0, len(train_grids), BATCH_SIZE):
            idx = perm[start:start+BATCH_SIZE]
            x = augment(train_grids[idx]).to(DEVICE)
            tgt = torch.tensor([train_labels[i][0]-1 for i in idx.tolist()], device=DEVICE)
            optimizer.zero_grad()
            h = backbone(readin(x))
            loss = focal_ce(head(h.mean(dim=1)), tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step()
        scheduler.step()

        if (epoch+1) % EVAL_EVERY == 0:
            readin.eval(); backbone.eval(); head.eval()
            with torch.no_grad():
                h = backbone(readin(val_grids.to(DEVICE)))
                tgt = torch.tensor([l[0]-1 for l in val_labels], device=DEVICE)
                vl = F.cross_entropy(head(h.mean(dim=1)), tgt).item()
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: deepcopy(v.state_dict()) for k, v in
                              [("r", readin), ("b", backbone), ("h", head)]}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE: break

    if best_state:
        readin.load_state_dict(best_state["r"]); backbone.load_state_dict(best_state["b"])
        head.load_state_dict(best_state["h"])

    readin.eval(); backbone.eval(); head.eval()
    with torch.no_grad():
        h = backbone(readin(val_grids.to(DEVICE)))
        preds = (head(h.mean(dim=1)).argmax(-1)+1).cpu().tolist()
    return [[p] for p in preds], val_labels


def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev = dp[0]; dp[0] = i
        for j in range(1, m+1):
            temp = dp[j]
            dp[j] = prev if a[i-1]==b[j-1] else 1+min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[m]


def compute_per(preds, refs):
    return sum(edit_distance(p, r) for p, r in zip(preds, refs)) / max(sum(len(r) for r in refs), 1)


CONDITIONS = [
    # tmin sweep (tmax=0.5 fixed)
    ("tmin=-0.25_tmax=0.5", -0.25, 0.5),
    ("tmin=-0.20_tmax=0.5", -0.20, 0.5),
    ("tmin=-0.15_tmax=0.5", -0.15, 0.5),
    ("tmin=-0.10_tmax=0.5", -0.10, 0.5),
    ("tmin=-0.05_tmax=0.5", -0.05, 0.5),
    ("tmin=0.00_tmax=0.5",   0.00, 0.5),
    # tmax sweep (tmin=-0.15 fixed)
    ("tmin=-0.15_tmax=0.3", -0.15, 0.3),
    ("tmin=-0.15_tmax=0.4", -0.15, 0.4),
    ("tmin=-0.15_tmax=0.6", -0.15, 0.6),
    ("tmin=-0.15_tmax=0.7", -0.15, 0.7),
]


def main():
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        global DEVICE
        DEVICE = args.device

    with open(PROJECT_ROOT / args.paths) as f:
        paths = yaml.safe_load(f)
    bids_root = paths.get("ps_bids_root") or paths["bids_root"]

    print(f"Patient: {PATIENT}  |  Device: {DEVICE}  |  Seed: {SEED}")
    print(f"Per-phoneme MFA epochs — padding grid sweep")
    print("=" * 70)

    results = {}
    for name, tmin, tmax in CONDITIONS:
        print(f"\n>>> {name}")
        t0 = time.time()
        grids, labels, grid_shape = load_per_phoneme(bids_root, tmin, tmax)
        print(f"  {len(labels)} samples, T={grids.shape[-1]}")
        splits = create_grouped_splits(labels)

        all_preds, all_refs, fold_pers = [], [], []
        for fi, (tr, va) in enumerate(splits):
            preds, refs = train_eval_fold(grids, labels, tr, va, grid_shape)
            per = compute_per(preds, refs)
            fold_pers.append(per)
            all_preds.extend(preds); all_refs.extend(refs)
            print(f"  Fold {fi+1}: PER={per:.4f}")

        mean_per = float(np.mean(fold_pers))
        results[name] = {"mean_per": mean_per, "std": float(np.std(fold_pers)),
                         "overall": compute_per(all_preds, all_refs), "time": time.time()-t0}
        print(f"  >>> {name}: PER={mean_per:.4f}±{results[name]['std']:.4f}  [{results[name]['time']:.0f}s]")

    print("\n" + "=" * 70)
    print(f"{'Condition':<30} {'Mean PER':<12} {'Std':<10} {'Time':<8}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<30} {r['mean_per']:.4f}       {r['std']:.4f}     {r['time']:.0f}s")


if __name__ == "__main__":
    main()
