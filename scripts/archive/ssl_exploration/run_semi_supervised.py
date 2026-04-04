#!/usr/bin/env python3
"""Semi-supervised SSL with proper nested CV evaluation.

The key insight: joint SSL + CE produces dramatically better features (PER 0.040)
but the previous result was contaminated — CE saw all S14 tokens during Stage 2,
then Stage 3 evaluated on those same tokens.

Fix: nested CV. For each fold, train a fresh model with:
  - SSL loss on ALL patients (no labels, S14 included for its neural data)
  - CE loss on S14 TRAIN-fold tokens ONLY
Then evaluate on val-fold tokens with frozen backbone + linear probe.

This is fair because val tokens were never seen by the CE component.

Usage:
  python scripts/run_semi_supervised.py --ssl-mode byol --target S14 --steps 5000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.evaluation.grouped_cv import (
    _patient_seed,
    build_token_groups,
    create_grouped_splits,
)
from speech_decoding.pretraining.pretrain_model import PretrainModel
from speech_decoding.pretraining.semi_supervised_trainer import (
    SemiSupervisedConfig,
    SemiSupervisedStage2Trainer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PS_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S33", "S39", "S57", "S58", "S62"]
DEV_PATIENT = "S26"


def build_ssl_model(mode: str, config: dict, grid_shape: tuple[int, int]):
    """Instantiate SSL model by mode name."""
    if mode == "byol":
        from speech_decoding.pretraining.byol_model import BYOLModel
        return BYOLModel(config, grid_shape)
    elif mode == "jepa":
        from speech_decoding.pretraining.jepa_model import JEPAModel
        return JEPAModel(config, grid_shape)
    elif mode == "dino":
        from speech_decoding.pretraining.dino_model import DINOModel
        return DINOModel(config, grid_shape)
    else:
        raise ValueError(f"Unknown SSL mode: {mode}")


def eval_frozen_fold(model, train_grids, train_labels, val_grids, val_labels,
                     device, n_positions=3, n_classes=9, epochs=100, patience=10):
    """Freeze backbone, train linear head on train fold, eval on val fold."""
    model.eval()
    gru_hidden = model.config["gru_hidden"]
    head = nn.Linear(gru_hidden * 2, n_positions * n_classes).to(device)

    # Only train the head + readin (same as Stage3Evaluator)
    readin_params = [p for n, p in model.named_parameters()
                     if "readin" in n]
    for p in model.parameters():
        p.requires_grad = False
    for p in readin_params:
        p.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": head.parameters(), "lr": 1e-3},
        {"params": readin_params, "lr": 3e-3},
    ], weight_decay=1e-4)

    best_loss = float("inf")
    best_head_state = None
    best_readin_state = None
    patience_ctr = 0

    for epoch in range(epochs):
        head.train()
        model.train()  # for dropout etc, but backbone frozen
        with torch.no_grad():
            feat = model.encode(train_grids.to(device))
        pooled = feat.mean(dim=1)
        logits = head(pooled).view(-1, n_positions, n_classes)
        targets = torch.tensor(train_labels, device=device, dtype=torch.long)
        loss = sum(
            F.cross_entropy(logits[:, p], targets[:, p] - 1)
            for p in range(n_positions)
        ) / n_positions
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validate
        head.eval()
        model.eval()
        with torch.no_grad():
            val_feat = model.encode(val_grids.to(device)).mean(dim=1)
            val_logits = head(val_feat).view(-1, n_positions, n_classes)
            val_tgt = torch.tensor(val_labels, device=device, dtype=torch.long)
            val_loss = sum(
                F.cross_entropy(val_logits[:, p], val_tgt[:, p] - 1)
                for p in range(n_positions)
            ) / n_positions

        if val_loss < best_loss:
            best_loss = val_loss
            best_head_state = deepcopy(head.state_dict())
            best_readin_state = {n: p.clone() for n, p in model.named_parameters()
                                 if "readin" in n}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    # Restore best
    if best_head_state:
        head.load_state_dict(best_head_state)
    if best_readin_state:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_readin_state:
                    p.copy_(best_readin_state[n])

    head.eval()
    model.eval()
    with torch.no_grad():
        val_feat = model.encode(val_grids.to(device)).mean(dim=1)
        logits = head(val_feat).view(-1, n_positions, n_classes)
        preds = (logits.argmax(dim=-1) + 1).cpu().numpy()
        refs = np.array(val_labels)

    total, errors = 0, 0
    for pred, ref in zip(preds, refs):
        for p, r in zip(pred, ref):
            total += 1
            if p != r:
                errors += 1
    return errors / total if total > 0 else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ssl-mode", required=True, choices=["byol", "jepa", "dino"])
    p.add_argument("--target", default="S14")
    p.add_argument("--paths", default="configs/paths.yaml")
    p.add_argument("--config", default="configs/pretrain_base.yaml")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=1.0, help="CE weight relative to SSL")
    p.add_argument("--device", default="mps")
    p.add_argument("--output-dir", default="results/semi_supervised")
    args = p.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)
    with open(args.config) as f:
        model_config = yaml.safe_load(f)

    bids_root = paths.get("ps_bids_root") or paths["bids_root"]
    output_dir = Path(args.output_dir) / f"{args.ssl_mode}" / args.target
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ALL source data for SSL (includes target for SSL, just not labels)
    all_patients = [pid for pid in PS_PATIENTS if pid != DEV_PATIENT]
    logger.info("Loading data for %d patients...", len(all_patients))
    patient_data = {}
    for pid in all_patients:
        ds = load_patient_data(pid, bids_root, task="PhonemeSequence",
                               n_phons=3, tmin=0.0, tmax=1.0)
        grids = []
        for i in range(len(ds)):
            g, _, _ = ds[i]
            grids.append(g)
        patient_data[pid] = torch.tensor(np.stack(grids), dtype=torch.float32)
        logger.info("  %s: %d trials", pid, len(grids))

    # Load target data with labels
    ds_target = load_patient_data(args.target, bids_root, task="PhonemeSequence",
                                  n_phons=3, tmin=0.0, tmax=1.0)
    target_grids, target_labels = [], []
    for i in range(len(ds_target)):
        g, l, _ = ds_target[i]
        target_grids.append(g)
        target_labels.append(l)
    target_grids = torch.tensor(np.stack(target_grids), dtype=torch.float32)
    logger.info("Target %s: %d trials", args.target, len(target_grids))

    # Grouped-by-token CV splits
    groups = build_token_groups(target_labels)
    seed = _patient_seed(args.target)
    splits = create_grouped_splits(target_labels, groups, n_folds=5, seed=seed)

    grid_shape = (target_grids.shape[1], target_grids.shape[2])
    model_config["ema_total_steps"] = args.steps

    fold_pers = []
    for fold_idx, fold in enumerate(splits):
        logger.info("=" * 60)
        logger.info("FOLD %d/5", fold_idx + 1)
        train_idx = fold["train_indices"]
        val_idx = fold["val_indices"]
        train_grids = target_grids[train_idx]
        train_labels = [target_labels[i] for i in train_idx]
        val_grids = target_grids[val_idx]
        val_labels = [target_labels[i] for i in val_idx]
        logger.info("  Train: %d trials, Val: %d trials", len(train_idx), len(val_idx))

        # Fresh model per fold
        ssl_model = build_ssl_model(args.ssl_mode, model_config, grid_shape)

        # Semi-supervised training: SSL on all patients + CE on train fold only
        ss_cfg = SemiSupervisedConfig(
            steps=args.steps, batch_size=8, ce_batch_size=8,
            alpha=args.alpha, lr=1e-3,
        )
        trainer = SemiSupervisedStage2Trainer(ssl_model, ss_cfg, device=args.device)
        logger.info("  Stage 2: %d steps (SSL on %d patients + CE on %d train trials)",
                     args.steps, len(patient_data), len(train_grids))
        trainer.train(patient_data, train_grids, train_labels)

        # Transfer to PretrainModel
        pretrain_model = PretrainModel(model_config, grid_shape)
        ssl_model.transfer_encoder_weights(pretrain_model)
        pretrain_model = pretrain_model.to(args.device)

        # Evaluate: frozen backbone + linear probe on val fold
        per = eval_frozen_fold(
            pretrain_model, train_grids, train_labels,
            val_grids, val_labels, args.device,
        )
        fold_pers.append(per)
        logger.info("  Fold %d PER: %.3f", fold_idx + 1, per)

    # Results
    mean_per = float(np.mean(fold_pers))
    std_per = float(np.std(fold_pers))
    logger.info("=" * 60)
    logger.info("Semi-supervised %s → %s: PER %.3f ± %.3f",
                args.ssl_mode, args.target, mean_per, std_per)
    logger.info("Fold PERs: %s", fold_pers)

    results = {
        "ssl_mode": args.ssl_mode,
        "target": args.target,
        "steps": args.steps,
        "alpha": args.alpha,
        "mean_per": mean_per,
        "std_per": std_per,
        "fold_pers": fold_pers,
        "method": "semi_supervised_nested_cv",
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", output_dir / "results.json")


if __name__ == "__main__":
    main()
