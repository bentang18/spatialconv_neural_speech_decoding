#!/usr/bin/env python3
"""LOPO with backbone warm-start from per-patient pre-training.

1. Train per-patient CE on the best source patient → save backbone
2. LOPO Stage 1 with warm backbone initialization
3. Stage 2 adaptation on target patient
4. Print detailed diagnostics
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import torch
import yaml

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.models.assembler import assemble_model
from speech_decoding.training.adaptor import adapt_stage2
from speech_decoding.training.ctc_utils import per_position_ce_decode
from speech_decoding.training.lopo_trainer import train_stage1
from speech_decoding.training.trainer import train_per_patient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SPALDING_PTS = ["S14", "S22", "S23", "S26", "S33", "S39", "S58", "S62"]


def diagnose(predictions, targets):
    """Print compact diagnostics."""
    n = len(predictions)
    pred_lens = [len(p) for p in predictions]
    len3 = sum(1 for l in pred_lens if l == 3)

    for pos in range(3):
        correct = sum(
            1 for p, t in zip(predictions, targets)
            if len(p) > pos and len(t) > pos and p[pos] == t[pos]
        )
        total = sum(1 for p, t in zip(predictions, targets) if len(p) > pos and len(t) > pos)
        pct = correct / total * 100 if total else 0
        print(f"    P{pos+1}: {correct}/{total} ({pct:.0f}%)", end="")
    print()

    pred_phon = Counter(p for seq in predictions for p in seq)
    print(f"    Phoneme dist: {dict(sorted(pred_phon.items()))}")

    for i in range(min(5, n)):
        print(f"    {str(predictions[i]):<15s} ← {targets[i]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lopo_ce.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--target", default="S14")
    parser.add_argument("--warmstart-patient", default=None,
                        help="Patient to pre-train backbone on. Default: best source.")
    parser.add_argument("--sources", nargs="+", default=None)
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
    source_ids = args.sources or [p for p in SPALDING_PTS if p != args.target]

    # Load all datasets
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
    warmstart_pid = args.warmstart_patient or source_ids[0]

    print(f"\n{'='*60}")
    print(f"LOPO WARM-START: target={args.target}, warmstart={warmstart_pid}")
    print(f"Sources: {list(source_datasets.keys())}")
    print(f"{'='*60}")

    # ========== Step 1: Per-patient pre-training ==========
    print(f"\n--- Step 1: Per-patient CE on {warmstart_pid} ---")
    warmstart_ds = all_datasets[warmstart_pid]
    pp_result = train_per_patient(warmstart_ds, config, seed=args.seed, device=args.device)
    print(f"  {warmstart_pid} per-patient: PER={pp_result['per_mean']:.3f}, "
          f"bal_acc={pp_result['bal_acc_mean_mean']:.3f}")

    # Extract backbone + head from the best fold
    # Re-train one fold to get the actual state dicts
    # (train_per_patient doesn't return state dicts, so we do a quick re-extraction)
    print(f"\n  Re-training single fold for weight extraction...")
    import numpy as np
    from copy import deepcopy
    from sklearn.model_selection import StratifiedKFold
    import torch.nn as nn
    from speech_decoding.data.augmentation import augment_from_config
    from speech_decoding.training.ctc_utils import per_position_ce_loss

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tc = config["training"]["stage1"]
    ac = config["training"].get("augmentation", tc.get("augmentation", {}))
    loss_type = config["training"].get("loss_type", "ctc")
    n_segments = config["training"].get("ce_segments", 3)

    patients_map = {warmstart_pid: warmstart_ds.grid_shape}
    backbone, head, readins = assemble_model(config, patients_map)
    readin = readins[warmstart_pid]

    if loss_type == "ce":
        input_dim = config["model"]["hidden_size"] * 2
        n_phonemes = config["model"]["num_classes"] - 1
        head = nn.Linear(input_dim, n_segments * n_phonemes)

    backbone = backbone.to(args.device)
    head = head.to(args.device)
    readin = readin.to(args.device)

    # Train on full dataset (no CV split) for warmstart weights
    x_all = torch.from_numpy(warmstart_ds.grid_data)
    y_all = warmstart_ds.ctc_labels
    B = tc["batch_size"]
    n_train = len(x_all)

    import math
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    optimizer = AdamW(
        [
            {"params": readin.parameters(), "lr": tc["lr"] * tc["readin_lr_mult"]},
            {"params": backbone.parameters(), "lr": tc["lr"]},
            {"params": head.parameters(), "lr": tc["lr"]},
        ],
        weight_decay=tc["weight_decay"],
    )
    total_epochs = tc.get("epochs", 300)

    def lr_lambda(epoch):
        progress = epoch / max(total_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    all_params = list(backbone.parameters()) + list(head.parameters()) + list(readin.parameters())

    best_loss = float("inf")
    best_backbone_sd = None
    best_head_sd = None

    for epoch in range(total_epochs):
        backbone.train(); head.train(); readin.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, B):
            idx = perm[start:start + B]
            x_batch = augment_from_config(x_all[idx], ac, training=True).to(args.device)
            y_batch = [y_all[i] for i in idx.tolist()]

            optimizer.zero_grad()
            out = head(backbone(readin(x_batch)))
            loss = per_position_ce_loss(out, y_batch, n_segments)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, tc["grad_clip"])
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)

        if avg < best_loss:
            best_loss = avg
            best_backbone_sd = deepcopy(backbone.state_dict())
            best_head_sd = deepcopy(head.state_dict())

        if (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1}: loss={avg:.4f}")

    print(f"  Best training loss: {best_loss:.4f}")

    # ========== Step 2: LOPO Stage 1 with warm backbone ==========
    print(f"\n--- Step 2: LOPO Stage 1 (warm-start from {warmstart_pid}) ---")
    checkpoint = train_stage1(
        source_datasets, config, seed=args.seed, device=args.device,
        backbone_init=best_backbone_sd,
        head_init=best_head_sd,
    )

    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    if val_losses:
        print(f"  Val loss: start={val_losses[0]:.4f}, min={min(val_losses):.4f}, "
              f"end={val_losses[-1]:.4f}, evals={len(val_losses)}")

    # Quick source-patient check
    print(f"\n  Source patient check:")
    bb, hd, ris = assemble_model(config, {pid: ds.grid_shape for pid, ds in source_datasets.items()})
    if loss_type == "ce":
        hd = nn.Linear(config["model"]["hidden_size"] * 2,
                        n_segments * (config["model"]["num_classes"] - 1))
    bb.load_state_dict(checkpoint["backbone"])
    hd.load_state_dict(checkpoint["head"])
    bb = bb.to(args.device).eval()
    hd = hd.to(args.device).eval()

    for pid, ds in list(source_datasets.items())[:3]:
        ri = ris[pid].to(args.device).eval()
        ri.load_state_dict(checkpoint["read_ins"][pid])
        with torch.no_grad():
            x = torch.from_numpy(ds.grid_data).to(args.device)
            lp = hd(bb(ri(x)))
            preds = per_position_ce_decode(lp, n_segments)

        correct_p1 = sum(1 for p, t in zip(preds, ds.ctc_labels)
                         if len(p) >= 1 and p[0] == t[0])
        print(f"    {pid}: P1 acc={correct_p1}/{len(ds)} ({correct_p1/len(ds):.0%})")

    # ========== Step 3: Stage 2 adaptation ==========
    print(f"\n--- Step 3: Stage 2 adaptation on {args.target} ---")
    result = adapt_stage2(
        checkpoint, target_ds, source_datasets, config,
        seed=args.seed, device=args.device,
    )

    print(f"\n{'='*60}")
    print(f"RESULTS: target={args.target}")
    print(f"  PER:      {result['per_mean']:.3f} ± {result['per_std']:.3f}")
    print(f"  Bal Acc:  {result['bal_acc_mean_mean']:.3f} ± {result['bal_acc_mean_std']:.3f}")
    print(f"  Len Acc:  {result['length_accuracy_mean']:.3f}")
    fold_pers = [f"{r['per']:.3f}" for r in result['fold_results']]
    print(f"  Per-fold PER: {fold_pers}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
