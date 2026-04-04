#!/usr/bin/env python3
"""Evaluate SSL features with temporal heads (not mean-pooled).

Tests two temporal decoders on frozen SSL backbone features:

1. Per-segment CE: divide T' frames into 3 segments, pool within each,
   independent Linear(2H, 9) per position. Preserves temporal ordering.

2. CTC: Linear(2H, 10) at every frame → CTC loss → greedy decode.
   Full temporal decoding with alignment-free sequence output.

Compared to Stage3Evaluator's mean-pool → Linear(2H, 27) which crushes
all temporal information into a single vector.

Usage:
  python scripts/eval_ssl_temporal.py \
    --checkpoint results/pretrain/method_B_jepa/S14/stage2_checkpoint.pt \
    --mode jepa --target S14
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
from speech_decoding.training.ctc_utils import ctc_loss, greedy_decode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_POSITIONS = 3
N_CLASSES = 9


def load_model(checkpoint_path, mode, config, grid_shape):
    """Load SSL checkpoint into PretrainModel."""
    pretrain_model = PretrainModel(config, grid_shape)
    if mode == "jepa":
        from speech_decoding.pretraining.jepa_model import JEPAModel
        ssl_model = JEPAModel(config, grid_shape)
        ssl_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        ssl_model.transfer_encoder_weights(pretrain_model)
    elif mode == "dino":
        from speech_decoding.pretraining.dino_model import DINOModel
        ssl_model = DINOModel(config, grid_shape)
        ssl_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        ssl_model.transfer_encoder_weights(pretrain_model)
    elif mode == "byol":
        from speech_decoding.pretraining.byol_model import BYOLModel
        ssl_model = BYOLModel(config, grid_shape)
        ssl_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        ssl_model.transfer_encoder_weights(pretrain_model)
    elif mode == "masked":
        pretrain_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return pretrain_model


# ─── Per-segment CE head ───────────────────────────────────────────

class SegmentCEHead(nn.Module):
    """Divide T' frames into N segments, pool each, classify independently."""

    def __init__(self, d_in, n_positions=3, n_classes=9):
        super().__init__()
        self.n_positions = n_positions
        self.heads = nn.ModuleList([
            nn.Linear(d_in, n_classes) for _ in range(n_positions)
        ])

    def forward(self, features):
        """
        Args:
            features: (B, T', D) temporal features.
        Returns:
            (B, n_positions, n_classes) logits.
        """
        B, T, D = features.shape
        seg_len = T // self.n_positions
        logits = []
        for p in range(self.n_positions):
            start = p * seg_len
            end = start + seg_len if p < self.n_positions - 1 else T
            seg_pooled = features[:, start:end, :].mean(dim=1)  # (B, D)
            logits.append(self.heads[p](seg_pooled))  # (B, n_classes)
        return torch.stack(logits, dim=1)  # (B, n_pos, n_cls)


def train_segment_ce_fold(model, head, train_grids, train_labels,
                          val_grids, val_labels, device,
                          epochs=100, patience=10, lr=1e-3):
    """Train per-segment CE head on frozen features."""
    readin_params = [p for n, p in model.named_parameters() if "readin" in n]
    for p in model.parameters():
        p.requires_grad = False
    for p in readin_params:
        p.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": head.parameters(), "lr": lr},
        {"params": readin_params, "lr": lr * 3},
    ], weight_decay=1e-4)

    best_loss = float("inf")
    best_state = (None, None)
    patience_ctr = 0

    for epoch in range(epochs):
        head.train()
        with torch.no_grad():
            feat = model.encode(train_grids.to(device))  # (B, T', D)
        logits = head(feat)  # (B, 3, 9)
        targets = torch.tensor(train_labels, device=device, dtype=torch.long)
        loss = sum(
            F.cross_entropy(logits[:, p], targets[:, p] - 1)
            for p in range(N_POSITIONS)
        ) / N_POSITIONS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            val_feat = model.encode(val_grids.to(device))
            val_logits = head(val_feat)
            val_tgt = torch.tensor(val_labels, device=device, dtype=torch.long)
            val_loss = sum(
                F.cross_entropy(val_logits[:, p], val_tgt[:, p] - 1)
                for p in range(N_POSITIONS)
            ) / N_POSITIONS

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = (deepcopy(head.state_dict()),
                          {n: p.clone() for n, p in model.named_parameters() if "readin" in n})
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state[0]:
        head.load_state_dict(best_state[0])
    if best_state[1]:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_state[1]:
                    p.copy_(best_state[1][n])

    head.eval()
    with torch.no_grad():
        val_feat = model.encode(val_grids.to(device))
        logits = head(val_feat)
        preds = (logits.argmax(dim=-1) + 1).cpu().numpy()  # (B, 3), 1-indexed
        refs = np.array(val_labels)
    total, errors = 0, 0
    for pred, ref in zip(preds, refs):
        for p, r in zip(pred, ref):
            total += 1
            if p != r:
                errors += 1
    return errors / total if total > 0 else 1.0


# ─── CTC head ─────────────────────────────────────────────────────

class CTCHead(nn.Module):
    """Per-frame classification: Linear(D, 10) → log_softmax."""

    def __init__(self, d_in, n_classes_with_blank=10):
        super().__init__()
        self.proj = nn.Linear(d_in, n_classes_with_blank)

    def forward(self, features):
        """
        Args:
            features: (B, T', D) temporal features.
        Returns:
            (B, T', C) log probabilities.
        """
        return F.log_softmax(self.proj(features), dim=-1)


def train_ctc_fold(model, head, train_grids, train_labels,
                   val_grids, val_labels, device,
                   epochs=200, patience=15, lr=1e-3):
    """Train CTC head on frozen features."""
    readin_params = [p for n, p in model.named_parameters() if "readin" in n]
    for p in model.parameters():
        p.requires_grad = False
    for p in readin_params:
        p.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": head.parameters(), "lr": lr},
        {"params": readin_params, "lr": lr * 3},
    ], weight_decay=1e-4)

    best_loss = float("inf")
    best_state = (None, None)
    patience_ctr = 0

    for epoch in range(epochs):
        head.train()
        with torch.no_grad():
            feat = model.encode(train_grids.to(device))  # (B, T', D)
        log_probs = head(feat)  # (B, T', C)
        # CTC loss needs CPU for MPS
        loss = ctc_loss(log_probs.cpu(), train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            val_feat = model.encode(val_grids.to(device))
            val_log_probs = head(val_feat)
            val_loss = ctc_loss(val_log_probs.cpu(), val_labels)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = (deepcopy(head.state_dict()),
                          {n: p.clone() for n, p in model.named_parameters() if "readin" in n})
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state[0]:
        head.load_state_dict(best_state[0])
    if best_state[1]:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_state[1]:
                    p.copy_(best_state[1][n])

    head.eval()
    with torch.no_grad():
        val_feat = model.encode(val_grids.to(device))
        val_log_probs = head(val_feat)
        preds = greedy_decode(val_log_probs)
        refs = val_labels

    # PER via edit distance
    from speech_decoding.training.ctc_utils import compute_per
    return compute_per(preds, refs)


# ─── Main ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", required=True, choices=["jepa", "byol", "dino", "masked"])
    p.add_argument("--target", default="S14")
    p.add_argument("--paths", default="configs/paths.yaml")
    p.add_argument("--config", default="configs/pretrain_base.yaml")
    p.add_argument("--device", default="mps")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    bids_root = paths.get("ps_bids_root") or paths["bids_root"]

    # Load target
    ds = load_patient_data(args.target, bids_root, task="PhonemeSequence",
                           n_phons=3, tmin=0.0, tmax=1.0)
    grids, labels = [], []
    for i in range(len(ds)):
        g, l, _ = ds[i]
        grids.append(g)
        labels.append(l)
    grids = torch.tensor(np.stack(grids), dtype=torch.float32)
    logger.info("Target %s: %d trials", args.target, len(grids))

    # Load model
    config["ema_total_steps"] = 5000
    model = load_model(args.checkpoint, args.mode, config, (8, 16))
    model = model.to(args.device)

    # Check feature dimensions
    with torch.no_grad():
        test_feat = model.encode(grids[:2].to(args.device))
    B, T_prime, D = test_feat.shape
    logger.info("Feature shape: (B, %d, %d) — %d temporal frames", T_prime, D, T_prime)

    # Grouped CV
    groups = build_token_groups(labels)
    seed = _patient_seed(args.target)
    splits = create_grouped_splits(labels, groups, n_folds=5, seed=seed)

    base_state = deepcopy(model.state_dict())

    # === Per-segment CE ===
    seg_pers = []
    for fold_idx, fold in enumerate(splits):
        model.load_state_dict(base_state)
        tr_idx, va_idx = fold["train_indices"], fold["val_indices"]
        head = SegmentCEHead(D, N_POSITIONS, N_CLASSES).to(args.device)
        per = train_segment_ce_fold(
            model, head, grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx], args.device,
        )
        seg_pers.append(per)
        logger.info("  Segment CE fold %d: PER=%.3f", fold_idx, per)
    logger.info("Segment CE: PER %.3f ± %.3f", np.mean(seg_pers), np.std(seg_pers))

    # === CTC ===
    ctc_pers = []
    for fold_idx, fold in enumerate(splits):
        model.load_state_dict(base_state)
        tr_idx, va_idx = fold["train_indices"], fold["val_indices"]
        head = CTCHead(D, N_CLASSES + 1).to(args.device)  # +1 for blank
        per = train_ctc_fold(
            model, head, grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx], args.device,
        )
        ctc_pers.append(per)
        logger.info("  CTC fold %d: PER=%.3f", fold_idx, per)
    logger.info("CTC: PER %.3f ± %.3f", np.mean(ctc_pers), np.std(ctc_pers))

    # Results
    results = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "target": args.target,
        "feature_shape": [int(T_prime), int(D)],
        "segment_ce": {
            "mean_per": float(np.mean(seg_pers)),
            "std_per": float(np.std(seg_pers)),
            "fold_pers": seg_pers,
        },
        "ctc": {
            "mean_per": float(np.mean(ctc_pers)),
            "std_per": float(np.std(ctc_pers)),
            "fold_pers": ctc_pers,
        },
    }

    logger.info("=" * 60)
    logger.info("SUMMARY for %s:", args.mode)
    logger.info("  Segment CE: %.3f ± %.3f", results["segment_ce"]["mean_per"], results["segment_ce"]["std_per"])
    logger.info("  CTC:        %.3f ± %.3f", results["ctc"]["mean_per"], results["ctc"]["std_per"])

    out_path = args.output or str(Path(args.checkpoint).parent / "temporal_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
