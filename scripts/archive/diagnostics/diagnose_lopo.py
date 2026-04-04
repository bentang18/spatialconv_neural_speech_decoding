#!/usr/bin/env python3
"""Diagnose LOPO failure mode on a single target patient.

Runs Stage 1 (multi-patient) → Stage 2 (adaptation) on 1 target,
then logs detailed diagnostics:
  - Blank ratio
  - Decoded sequence lengths
  - Fraction all-blank predictions
  - Per-position balanced accuracy
  - Sample decoded vs target sequences
  - Stage 1 source validation loss trajectory
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.models.assembler import assemble_model
from speech_decoding.training.adaptor import adapt_stage2
from speech_decoding.training.ctc_utils import (
    blank_ratio,
    greedy_decode,
    per_position_ce_decode,
)
from speech_decoding.training.lopo_trainer import train_stage1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def diagnose_predictions(
    predictions: list[list[int]],
    targets: list[list[int]],
    log_probs: torch.Tensor,
) -> None:
    """Print detailed diagnostics for a set of predictions."""
    n = len(predictions)

    # Blank ratio
    br = blank_ratio(log_probs)
    print(f"  Blank ratio: {br:.1%}")

    # Decoded lengths
    pred_lens = [len(p) for p in predictions]
    len_counts = Counter(pred_lens)
    print(f"  Decoded lengths: {dict(sorted(len_counts.items()))}")
    print(f"  Length=0 (all blank): {len_counts.get(0, 0)}/{n} ({len_counts.get(0, 0)/n:.1%})")
    print(f"  Length=3 (correct):   {len_counts.get(3, 0)}/{n} ({len_counts.get(3, 0)/n:.1%})")

    # Per-position accuracy (only for length-3 predictions)
    for pos in range(3):
        correct = 0
        total = 0
        for pred, tgt in zip(predictions, targets):
            if len(pred) >= pos + 1 and len(tgt) >= pos + 1:
                total += 1
                if pred[pos] == tgt[pos]:
                    correct += 1
        if total > 0:
            print(f"  Position {pos+1} accuracy: {correct}/{total} ({correct/total:.1%})")
        else:
            print(f"  Position {pos+1}: no valid predictions")

    # Phoneme distribution in predictions vs targets
    pred_phonemes = [p for seq in predictions for p in seq]
    tgt_phonemes = [t for seq in targets for t in seq]
    pred_dist = Counter(pred_phonemes)
    tgt_dist = Counter(tgt_phonemes)
    print(f"  Predicted phoneme distribution: {dict(sorted(pred_dist.items()))}")
    print(f"  Target phoneme distribution:    {dict(sorted(tgt_dist.items()))}")

    # Sample predictions
    print("  Sample decoded vs target (first 10):")
    for i in range(min(10, n)):
        print(f"    pred={str(predictions[i]):<20s} target={targets[i]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LOPO failure mode diagnosis")
    parser.add_argument("--config", default="configs/lopo_pilot.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--target", default="S14", help="Target patient for diagnosis")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Source patients (default: all Spalding except target)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tmin", type=float, default=-0.5)
    parser.add_argument("--tmax", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    bids_root = Path(paths["ps_bids_root"])

    # Default: 8 Spalding patients
    spalding_pts = ["S14", "S22", "S23", "S26", "S33", "S39", "S58", "S62"]
    source_ids = args.sources or [p for p in spalding_pts if p != args.target]

    # Load data
    all_datasets = {}
    for pid in [args.target] + source_ids:
        try:
            ds = load_patient_data(
                pid, bids_root, task="PhonemeSequence", n_phons=3,
                tmin=args.tmin, tmax=args.tmax,
            )
            all_datasets[pid] = ds
            logger.info("Loaded %s: %d trials, grid %s", pid, len(ds), ds.grid_shape)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", pid, e)

    target_ds = all_datasets[args.target]
    source_datasets = {pid: ds for pid, ds in all_datasets.items() if pid != args.target}

    print(f"\n{'='*60}")
    print(f"LOPO DIAGNOSIS: target={args.target}, {len(source_datasets)} sources, seed={args.seed}")
    print(f"{'='*60}")

    # ========== Stage 1: Multi-patient training ==========
    print(f"\n--- Stage 1: Training on {len(source_datasets)} source patients ---")
    checkpoint = train_stage1(source_datasets, config, seed=args.seed, device=args.device)

    # Inspect Stage 1 losses
    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    if train_losses:
        print(f"  Train loss: start={train_losses[0]:.4f}, end={train_losses[-1]:.4f}, "
              f"min={min(train_losses):.4f}")
    if val_losses:
        print(f"  Val loss:   start={val_losses[0]:.4f}, end={val_losses[-1]:.4f}, "
              f"min={min(val_losses):.4f}, stopped at eval {len(val_losses)}")

    # ========== Diagnose Stage 1 backbone on SOURCE data ==========
    loss_type = config["training"].get("loss_type", "ctc")
    n_segments = config["training"].get("ce_segments", 3)

    print(f"\n--- Stage 1 backbone on SOURCE patients (loss={loss_type}) ---")
    backbone, head, read_ins = assemble_model(
        config, {pid: ds.grid_shape for pid, ds in source_datasets.items()}
    )

    # Replace head for CE
    if loss_type == "ce":
        import torch.nn as nn
        input_dim = config["model"]["hidden_size"] * 2
        n_phonemes = config["model"]["num_classes"] - 1
        head = nn.Linear(input_dim, n_segments * n_phonemes)

    backbone.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])
    for pid in source_datasets:
        read_ins[pid].load_state_dict(checkpoint["read_ins"][pid])

    backbone = backbone.to(args.device).eval()
    head = head.to(args.device).eval()

    for pid, ds in list(source_datasets.items())[:3]:  # first 3 sources
        ri = read_ins[pid].to(args.device).eval()
        with torch.no_grad():
            x = torch.from_numpy(ds.grid_data).to(args.device)
            shared = ri(x)
            h = backbone(shared)
            lp = head(h)
            if loss_type == "ce":
                preds = per_position_ce_decode(lp, n_segments)
            else:
                preds = greedy_decode(lp)

        print(f"\n  Source {pid} ({len(ds)} trials):")
        diagnose_predictions(preds, ds.ctc_labels, lp)

    # ========== Diagnose Stage 1 backbone on TARGET (before adaptation) ==========
    print(f"\n--- Stage 1 backbone on TARGET {args.target} (before adaptation, fresh read-in) ---")
    all_patients = {pid: ds.grid_shape for pid, ds in all_datasets.items()}
    _, _, all_ri = assemble_model(config, all_patients)
    target_ri_fresh = all_ri[args.target].to(args.device).eval()

    with torch.no_grad():
        x = torch.from_numpy(target_ds.grid_data).to(args.device)
        shared = target_ri_fresh(x)
        h = backbone(shared)
        lp = head(h)
        if loss_type == "ce":
            preds = per_position_ce_decode(lp, n_segments)
        else:
            preds = greedy_decode(lp)

    print(f"  Target {args.target} (fresh read-in, no adaptation):")
    diagnose_predictions(preds, target_ds.ctc_labels, lp)

    # ========== Stage 2: Adaptation ==========
    print(f"\n--- Stage 2: Adapting to {args.target} ---")
    result = adapt_stage2(
        checkpoint, target_ds, source_datasets, config,
        seed=args.seed, device=args.device,
    )

    print(f"\n--- Stage 2 Results (after adaptation) ---")
    print(f"  PER:      {result['per_mean']:.3f} ± {result['per_std']:.3f}")
    print(f"  Bal Acc:  {result['bal_acc_mean_mean']:.3f} ± {result['bal_acc_mean_std']:.3f}")
    print(f"  Blank:    {result['blank_ratio_mean']:.1%}")
    print(f"  Len Acc:  {result['length_accuracy_mean']:.3f}")

    # Per-fold breakdown
    print(f"\n  Per-fold:")
    for i, fr in enumerate(result["fold_results"]):
        print(f"    Fold {i+1}: PER={fr['per']:.3f}, bal_acc={fr['bal_acc_mean']:.3f}, "
              f"blank={fr['blank_ratio']:.1%}, len_acc={fr['length_accuracy']:.3f}")


if __name__ == "__main__":
    main()
