#!/usr/bin/env python3
"""Sig channel ANOVA + decode comparison.

For each patient:
1. Compute ANOVA-based sig channels (which channels discriminate phonemes)
2. Report sig channel counts vs Spalding expected counts

For S14:
3. Train decoder with all channels vs sig-only vs various thresholds
4. Report PER impact

Run on DCC:
    /work/ht203/miniconda3/envs/speech/bin/python scripts/sweep_sig_channels.py \
        --bids-root /work/ht203/data/BIDS
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech_decoding.data.bids_dataset import load_per_position_data
from speech_decoding.data.grid import load_grid_mapping, channels_to_grid
from speech_decoding.data.phoneme_map import ARPA_PHONEMES
from speech_decoding.data.sig_channels import compute_sig_channels
from speech_decoding.evaluation.grouped_cv import (
    build_token_groups,
    create_grouped_splits,
)

import mne


PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S39", "S57", "S58", "S62"]
N_CLASSES = len(ARPA_PHONEMES)  # 9
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# --- Model (self-contained, from sweep scripts) ---

class PerPhonemeDecoder(nn.Module):
    """Per-phoneme decoder: Conv2d → pool → Conv1d → BiGRU → mean pool → Linear."""

    def __init__(self, grid_h: int, grid_w: int, n_classes: int = 9):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 8))
        d_spatial = 8 * 4 * 8  # 256
        self.ln = nn.LayerNorm(d_spatial)
        self.temporal_conv = nn.Conv1d(d_spatial, 32, kernel_size=3, stride=10, padding=1)
        self.gru = nn.GRU(32, 32, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B * T, 1, H, W)
        x = F.relu(self.conv2d(x))
        x = self.pool(x)
        x = x.reshape(B, T, -1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.temporal_conv(x))
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        return self.head(x)


# --- Sig channel computation for raw epochs ---

def compute_sig_channels_from_fif(
    bids_root: Path, subject: str, alpha: float = 0.05,
    method: str = "baseline",
) -> dict:
    """Load raw epochs, compute sig channels via baseline or ANOVA test."""
    from speech_decoding.data.bids_dataset import _find_fif_path

    fif_path = _find_fif_path(bids_root, subject, "PhonemeSequence")
    epochs = mne.read_epochs(str(fif_path), preload=True, verbose=False)
    all_data = epochs.get_data()  # (n_total_epochs, n_ch, n_times)
    ch_names = epochs.ch_names

    if method == "anova":
        from speech_decoding.data.phoneme_map import normalize_label, encode_ctc_label
        inv_event_id = {v: k for k, v in epochs.event_id.items()}
        labels = []
        for i in range(len(all_data)):
            raw_label = inv_event_id[epochs.events[i, 2]]
            normed = normalize_label(raw_label)
            labels.append(encode_ctc_label([normed])[0])
        sig_mask, stat_values, p_values = compute_sig_channels(
            all_data, labels, alpha=alpha, method="anova"
        )
    else:
        # Baseline: no labels needed
        sig_mask, stat_values, p_values = compute_sig_channels(
            all_data, alpha=alpha, method="baseline"
        )

    return {
        "sig_mask": sig_mask,
        "stat_values": stat_values,
        "p_values": p_values,
        "n_sig": int(sig_mask.sum()),
        "n_total": len(sig_mask),
        "ch_names": ch_names,
        "top_channels": sorted(
            zip(ch_names, stat_values), key=lambda x: x[1], reverse=True
        )[:10],
    }


# --- Training + eval ---

def train_and_eval(
    dataset, grid_shape: tuple[int, int], channel_mask: np.ndarray | None,
    n_seeds: int = 3, n_epochs: int = 80, lr: float = 1e-3,
) -> dict:
    """Train per-phoneme decoder with optional channel masking.

    Uses grouped-by-token CV: all phonemes from the same trial token
    stay in the same fold to prevent leakage.
    """
    all_pers = []

    # Build trial-level groups for CV (per-phoneme has 3× samples)
    # Epochs ordered: all pos-1, all pos-2, all pos-3
    n_total = len(dataset)
    n_trials = n_total // 3  # each trial produces 3 per-phoneme epochs

    # Reconstruct 3-phoneme trial labels for grouping
    trial_labels = []
    for i in range(n_trials):
        phons = [dataset.ctc_labels[i + pos * n_trials][0]
                 for pos in range(3)]
        trial_labels.append(phons)

    # Create trial-level splits
    trial_groups = build_token_groups(trial_labels)
    trial_splits = create_grouped_splits(trial_labels, trial_groups, n_folds=5, seed=42)

    for seed in range(n_seeds):
        torch.manual_seed(seed + 42)
        np.random.seed(seed + 42)

        # Get data
        grid_data = dataset.grid_data.copy()  # (N, H, W, T)

        # Apply channel mask: zero out non-sig channels
        if channel_mask is not None:
            grid_data[:, ~channel_mask, :] = 0.0

        labels = [lbl[0] for lbl in dataset.ctc_labels]  # single-phoneme labels

        fold_pers = []

        for split in trial_splits:
            # Map trial-level indices to per-phoneme indices
            train_trial = split["train_indices"]
            val_trial = split["val_indices"]
            train_idx = []
            val_idx = []
            for pos in range(3):
                train_idx.extend([t + pos * n_trials for t in train_trial])
                val_idx.extend([t + pos * n_trials for t in val_trial])

            model = PerPhonemeDecoder(grid_shape[0], grid_shape[1]).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

            # Train
            X_train = torch.tensor(grid_data[train_idx], dtype=torch.float32)
            y_train = torch.tensor([labels[i] - 1 for i in train_idx], dtype=torch.long)
            X_val = torch.tensor(grid_data[val_idx], dtype=torch.float32)
            y_val = torch.tensor([labels[i] - 1 for i in val_idx], dtype=torch.long)

            train_ds = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

            model.train()
            for epoch in range(n_epochs):
                for xb, yb in train_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    loss = F.cross_entropy(model(xb), yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Eval
            model.eval()
            with torch.no_grad():
                logits = model(X_val.to(DEVICE))
                preds = logits.argmax(dim=-1).cpu()
                correct = (preds == y_val).float().mean().item()
                per = 1.0 - correct  # PER = 1 - accuracy for single phoneme

            fold_pers.append(per)

        all_pers.append(np.mean(fold_pers))

    return {
        "per_mean": np.mean(all_pers),
        "per_std": np.std(all_pers),
        "per_seeds": all_pers,
    }


def create_grid_channel_mask(
    sig_mask_channels: np.ndarray,
    ch_names: list[str],
    grid_info,
) -> np.ndarray:
    """Convert channel-level sig mask to grid-level mask.

    Args:
        sig_mask_channels: (n_channels,) bool — True for sig channels.
        ch_names: Channel names matching the mask.
        grid_info: GridInfo from load_grid_mapping().

    Returns:
        (H, W) bool mask — True for grid positions with sig channels.
    """
    H, W = grid_info.grid_shape
    grid_mask = np.zeros((H, W), dtype=bool)

    for i, name in enumerate(ch_names):
        if str(name) in grid_info.ch_to_pos and sig_mask_channels[i]:
            r, c = grid_info.ch_to_pos[str(name)]
            grid_mask[r, c] = True

    return grid_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--target", default="S14", help="Patient for decode comparison")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    bids_root = Path(args.bids_root)

    # =============================================
    # Part 1: Compute sig channels for all patients
    # =============================================
    print("=" * 70)
    print("PART 1: Sig Channel Detection (all patients)")
    print("=" * 70)

    EXPECTED_SIG = {
        "S14": 111, "S16": 111, "S22": 149, "S23": 74, "S26": 63,
        "S32": 144, "S33": 171, "S39": 201,
    }

    all_results = {}
    for patient in PATIENTS:
        try:
            t0 = time.time()
            # Run both methods
            result_base = compute_sig_channels_from_fif(
                bids_root, patient, method="baseline"
            )
            result_anova = compute_sig_channels_from_fif(
                bids_root, patient, method="anova"
            )
            elapsed = time.time() - t0

            # Store baseline as primary (matches upstream)
            all_results[patient] = result_base
            all_results[patient]["n_sig_anova"] = result_anova["n_sig"]

            expected = EXPECTED_SIG.get(patient, "?")
            print(f"\n{patient}: [{elapsed:.1f}s]")
            print(f"  Baseline t-test: {result_base['n_sig']}/{result_base['n_total']} sig "
                  f"(expected: {expected})")
            print(f"  ANOVA (phoneme): {result_anova['n_sig']}/{result_anova['n_total']} sig")
            print(f"  Top 5 by t-stat: {result_base['top_channels'][:5]}")
        except Exception as e:
            print(f"\n{patient}: ERROR — {e}")

    # Summary table
    print("\n" + "=" * 70)
    print("SIG CHANNEL SUMMARY")
    print("=" * 70)
    print(f"{'Patient':>8} {'Total':>6} {'Baseline':>9} {'ANOVA':>7} {'Spalding':>9} {'%base':>6}")
    print("-" * 55)
    for patient in PATIENTS:
        if patient not in all_results:
            continue
        r = all_results[patient]
        expected = EXPECTED_SIG.get(patient, "?")
        pct = 100.0 * r["n_sig"] / r["n_total"]
        print(f"{patient:>8} {r['n_total']:>6} {r['n_sig']:>9} "
              f"{r.get('n_sig_anova', '?'):>7} {str(expected):>9} {pct:>5.1f}%")

    # =============================================
    # Part 2: Decode comparison on target patient
    # =============================================
    target = args.target
    if target not in all_results:
        print(f"\nERROR: {target} not in results")
        return

    print(f"\n\n{'=' * 70}")
    print(f"PART 2: Decode Comparison on {target}")
    print(f"{'=' * 70}")

    # Load dataset
    ds = load_per_position_data(
        target, bids_root, tmin=-0.15, tmax=0.5,
    )
    grid_shape = ds.grid_shape

    # Get grid info for mask creation
    from speech_decoding.data.bids_dataset import _find_electrodes_tsv
    electrodes_tsv = _find_electrodes_tsv(bids_root, target)
    grid_info = load_grid_mapping(electrodes_tsv)

    sig_result = all_results[target]
    sig_mask_ch = sig_result["sig_mask"]  # baseline method
    stat_values = sig_result["stat_values"]

    # Create grid masks at different thresholds
    conditions = {}

    # 1. All channels (current baseline)
    conditions["all_channels"] = None

    # 2. Baseline sig (alpha=0.05 FDR) — matches upstream
    grid_mask_sig = create_grid_channel_mask(sig_mask_ch, sig_result["ch_names"], grid_info)
    conditions["baseline_sig_005"] = grid_mask_sig
    print(f"\nBaseline sig (α=0.05): {grid_mask_sig.sum()}/{grid_mask_sig.size} grid positions")

    # 3. Top-K channels by t-stat (try several percentiles)
    for top_k_pct in [25, 50, 75]:
        k = int(len(stat_values) * top_k_pct / 100)
        top_k_idx = np.argsort(stat_values)[-k:]
        top_k_mask = np.zeros(len(stat_values), dtype=bool)
        top_k_mask[top_k_idx] = True
        grid_mask_topk = create_grid_channel_mask(top_k_mask, sig_result["ch_names"], grid_info)
        conditions[f"top_{top_k_pct}pct"] = grid_mask_topk
        print(f"Top {top_k_pct}%: {grid_mask_topk.sum()}/{grid_mask_topk.size} grid positions")

    # 4. Stricter baseline (alpha=0.01)
    from speech_decoding.data.bids_dataset import _find_fif_path
    fif_path = _find_fif_path(bids_root, target, "PhonemeSequence")
    all_data = mne.read_epochs(str(fif_path), preload=True, verbose=False).get_data()
    sig_mask_strict, _, _ = compute_sig_channels(all_data, alpha=0.01, method="baseline")
    grid_mask_strict = create_grid_channel_mask(
        sig_mask_strict, sig_result["ch_names"], grid_info
    )
    conditions["baseline_sig_001"] = grid_mask_strict
    print(f"Baseline sig (α=0.01): {grid_mask_strict.sum()}/{grid_mask_strict.size} grid positions")

    # Run comparisons
    print(f"\nTraining decoder ({args.seeds} seeds × 5 folds × {len(conditions)} conditions)...")
    print(f"Device: {DEVICE}")

    results_table = {}
    for name, mask in conditions.items():
        t0 = time.time()
        n_active = mask.sum() if mask is not None else grid_shape[0] * grid_shape[1]
        print(f"\n  {name} ({n_active} active positions)...", end=" ", flush=True)

        result = train_and_eval(ds, grid_shape, mask, n_seeds=args.seeds)
        elapsed = time.time() - t0

        results_table[name] = result
        print(f"PER = {result['per_mean']:.3f} ± {result['per_std']:.3f} [{elapsed:.1f}s]")

    # Final summary
    print(f"\n\n{'=' * 70}")
    print(f"RESULTS: {target} Sig Channel Impact")
    print(f"{'=' * 70}")
    print(f"{'Condition':>20} {'Active':>7} {'PER':>10} {'Δ vs all':>10}")
    print("-" * 55)

    baseline_per = results_table["all_channels"]["per_mean"]
    for name, result in sorted(results_table.items(), key=lambda x: x[1]["per_mean"]):
        mask = conditions[name]
        n_active = mask.sum() if mask is not None else grid_shape[0] * grid_shape[1]
        delta = result["per_mean"] - baseline_per
        sign = "+" if delta > 0 else ""
        print(f"{name:>20} {n_active:>7} "
              f"{result['per_mean']:.3f}±{result['per_std']:.3f} "
              f"{sign}{delta:>+.3f}")


if __name__ == "__main__":
    main()
