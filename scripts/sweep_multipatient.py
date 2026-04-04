#!/usr/bin/env python3
"""Multi-patient sweep: does per-phoneme MFA advantage hold beyond S14?

Tests per-phoneme vs full-trial (simplified recipe) on all 11 PS patients.
Key question: S14 may have good MFA alignment; is the finding general?

Usage:
    python scripts/sweep_multipatient.py --paths configs/paths.yaml --device cuda
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

# All 11 unique PS patients (S36 excluded — duplicate of S32)
PS_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S39", "S57", "S58", "S62"]


def load_full_trial(patient, bids_root, tmin=0.0, tmax=1.0):
    ds = load_patient_data(patient, bids_root, task="PhonemeSequence", n_phons=3, tmin=tmin, tmax=tmax)
    grids, labels = [], []
    for i in range(len(ds)):
        g, l, _ = ds[i]
        grids.append(g)
        labels.append(l)
    return torch.tensor(np.stack(grids), dtype=torch.float32), labels, ds.grid_shape


def load_per_phoneme(patient, bids_root, tmin=-0.15, tmax=0.5):
    ds = load_per_position_data(patient, bids_root, task="PhonemeSequence", n_phons=3, tmin=tmin, tmax=tmax)
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


def train_eval(grids, labels, splits, grid_shape, mode):
    """Train + eval all folds. Returns mean PER."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    H, W = grid_shape
    n_positions = 3 if mode == "full_trial" else 1

    all_preds, all_refs = [], []
    fold_pers = []

    for fi, (tr_idx, va_idx) in enumerate(splits):
        torch.manual_seed(SEED + fi)

        readin = SpatialReadIn(H, W).to(DEVICE)
        backbone = Backbone(d_in=readin.d_flat).to(DEVICE)

        if mode == "full_trial":
            heads = nn.ModuleList([nn.Linear(backbone.out_dim, N_CLASSES) for _ in range(3)]).to(DEVICE)
            params = list(readin.parameters()) + list(backbone.parameters()) + list(heads.parameters())
        else:
            head = nn.Linear(backbone.out_dim, N_CLASSES).to(DEVICE)
            params = list(readin.parameters()) + list(backbone.parameters()) + list(head.parameters())

        optimizer = AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = LambdaLR(optimizer, lambda e: ((e+1)/WARMUP_EPOCHS) if e < WARMUP_EPOCHS
                             else 0.5*(1+math.cos(math.pi*(e-WARMUP_EPOCHS)/max(EPOCHS-WARMUP_EPOCHS,1))))

        train_grids, val_grids = grids[tr_idx], grids[va_idx]
        train_labels = [labels[i] for i in tr_idx]
        val_labels = [labels[i] for i in va_idx]

        best_vl, best_state, pat = float("inf"), None, 0

        for epoch in range(EPOCHS):
            readin.train(); backbone.train()
            if mode == "full_trial": heads.train()
            else: head.train()

            perm = torch.randperm(len(train_grids))
            for start in range(0, len(train_grids), BATCH_SIZE):
                idx = perm[start:start+BATCH_SIZE]
                x = augment(train_grids[idx]).to(DEVICE)
                optimizer.zero_grad()
                h = backbone(readin(x))
                pooled = h.mean(dim=1)

                if mode == "full_trial":
                    loss = sum(
                        focal_ce(heads[p](pooled), torch.tensor([l[p]-1 for l in [train_labels[i] for i in idx.tolist()]], device=DEVICE))
                        for p in range(3)
                    ) / 3
                else:
                    tgt = torch.tensor([train_labels[i][0]-1 for i in idx.tolist()], device=DEVICE)
                    loss = focal_ce(head(pooled), tgt)

                loss.backward()
                nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                optimizer.step()
            scheduler.step()

            if (epoch+1) % EVAL_EVERY == 0:
                readin.eval(); backbone.eval()
                if mode == "full_trial": heads.eval()
                else: head.eval()
                with torch.no_grad():
                    h = backbone(readin(val_grids.to(DEVICE)))
                    pooled = h.mean(dim=1)
                    if mode == "full_trial":
                        vl = sum(
                            F.cross_entropy(heads[p](pooled), torch.tensor([l[p]-1 for l in val_labels], device=DEVICE)).item()
                            for p in range(3)
                        ) / 3
                    else:
                        vl = F.cross_entropy(head(pooled), torch.tensor([l[0]-1 for l in val_labels], device=DEVICE)).item()

                if vl < best_vl:
                    best_vl = vl
                    best_state = {
                        "r": deepcopy(readin.state_dict()),
                        "b": deepcopy(backbone.state_dict()),
                        "h": deepcopy((heads if mode == "full_trial" else head).state_dict()),
                    }
                    pat = 0
                else:
                    pat += 1
                    if pat >= PATIENCE: break

        if best_state:
            readin.load_state_dict(best_state["r"])
            backbone.load_state_dict(best_state["b"])
            (heads if mode == "full_trial" else head).load_state_dict(best_state["h"])

        readin.eval(); backbone.eval()
        with torch.no_grad():
            h = backbone(readin(val_grids.to(DEVICE)))
            pooled = h.mean(dim=1)
            if mode == "full_trial":
                preds = [[int(heads[p](pooled)[i].argmax()+1) for p in range(3)] for i in range(len(val_labels))]
            else:
                preds_flat = (head(pooled).argmax(-1)+1).cpu().tolist()
                preds = [[p] for p in preds_flat]

        per = compute_per(preds, val_labels)
        fold_pers.append(per)
        all_preds.extend(preds)
        all_refs.extend(val_labels)

    return float(np.mean(fold_pers)), fold_pers, all_preds, all_refs


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


def main():
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--patients", nargs="+", default=None)
    args = parser.parse_args()

    if args.device:
        global DEVICE
        DEVICE = args.device

    with open(PROJECT_ROOT / args.paths) as f:
        paths = yaml.safe_load(f)
    bids_root = paths.get("ps_bids_root") or paths["bids_root"]

    patients = args.patients or PS_PATIENTS

    print(f"Device: {DEVICE}  |  Seed: {SEED}")
    print(f"Patients: {patients}")
    print(f"Comparing: per-phoneme (tmin=-0.15, tmax=0.5) vs full-trial (tmin=0.0, tmax=1.0)")
    print("=" * 80)

    results = []
    for patient in patients:
        print(f"\n{'='*40} {patient} {'='*40}")
        t0 = time.time()

        # Full trial
        try:
            grids_ft, labels_ft, gs_ft = load_full_trial(patient, bids_root)
            splits_ft = create_grouped_splits(labels_ft)
            ft_per, ft_folds, _, _ = train_eval(grids_ft, labels_ft, splits_ft, gs_ft, "full_trial")
            print(f"  Full-trial:  PER={ft_per:.4f}  folds={[f'{f:.3f}' for f in ft_folds]}  "
                  f"({len(labels_ft)} trials, grid {gs_ft})")
        except Exception as e:
            print(f"  Full-trial:  FAILED — {e}")
            ft_per = None

        # Per-phoneme
        try:
            grids_pp, labels_pp, gs_pp = load_per_phoneme(patient, bids_root)
            splits_pp = create_grouped_splits(labels_pp)
            pp_per, pp_folds, _, _ = train_eval(grids_pp, labels_pp, splits_pp, gs_pp, "per_phoneme")
            print(f"  Per-phoneme: PER={pp_per:.4f}  folds={[f'{f:.3f}' for f in pp_folds]}  "
                  f"({len(labels_pp)} samples)")
        except Exception as e:
            print(f"  Per-phoneme: FAILED — {e}")
            pp_per = None

        delta = None
        if ft_per is not None and pp_per is not None:
            delta = ft_per - pp_per
            winner = "per-phoneme" if delta > 0 else "full-trial"
            print(f"  >>> Δ = {delta:+.4f} ({winner} wins by {abs(delta)*100:.1f}pp)")

        results.append({"patient": patient, "ft_per": ft_per, "pp_per": pp_per, "delta": delta,
                         "time": time.time() - t0})

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Patient':<10} {'Trials':<8} {'Full-trial PER':<18} {'Per-phoneme PER':<18} {'Δ':<10} {'Winner'}")
    print("-" * 80)

    ft_wins, pp_wins = 0, 0
    for r in results:
        p = r["patient"]
        ft = f"{r['ft_per']:.4f}" if r['ft_per'] is not None else "FAIL"
        pp = f"{r['pp_per']:.4f}" if r['pp_per'] is not None else "FAIL"
        if r["delta"] is not None:
            d = f"{r['delta']:+.4f}"
            w = "per-phon" if r["delta"] > 0 else "full-trial"
            if r["delta"] > 0: pp_wins += 1
            else: ft_wins += 1
        else:
            d, w = "—", "—"
        print(f"{p:<10} {'—':<8} {ft:<18} {pp:<18} {d:<10} {w}")

    if results:
        ft_pers = [r["ft_per"] for r in results if r["ft_per"] is not None]
        pp_pers = [r["pp_per"] for r in results if r["pp_per"] is not None]
        deltas = [r["delta"] for r in results if r["delta"] is not None]
        if ft_pers and pp_pers:
            print(f"\nPopulation mean: full-trial={np.mean(ft_pers):.4f}  per-phoneme={np.mean(pp_pers):.4f}  "
                  f"Δ={np.mean(deltas):+.4f}")
            print(f"Per-phoneme wins: {pp_wins}/{pp_wins+ft_wins} patients")


if __name__ == "__main__":
    main()
