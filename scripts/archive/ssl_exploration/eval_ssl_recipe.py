#!/usr/bin/env python3
"""Apply autoresearch eval recipe to SSL checkpoints.

Loads a pretrained SSL model, extracts features, and evaluates with:
- Weighted k-NN (k=10, cosine similarity weights)
- TTA (16 augmented copies of val embeddings)
- Grouped-by-token CV (fair evaluation)

Usage:
  python scripts/eval_ssl_recipe.py --checkpoint results/pretrain/method_B_jepa/S14/stage2_checkpoint.pt \
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
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from speech_decoding.data.augmentation import augment_from_config
from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.evaluation.grouped_cv import (
    _patient_seed,
    build_token_groups,
    create_grouped_splits,
)
from speech_decoding.pretraining.pretrain_model import PretrainModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_POSITIONS = 3
N_CLASSES = 9
AUG_CONFIG = {
    "time_shift_frames": 30,
    "amp_scale_std": 0.3,
    "channel_dropout_max": 0.2,
    "noise_frac": 0.05,
    "temporal_stretch": True,
    "temporal_stretch_max_rate": 0.15,
}


def load_model(checkpoint_path: str, mode: str, config: dict, grid_shape):
    """Load SSL model from checkpoint and transfer to PretrainModel."""
    pretrain_model = PretrainModel(config, grid_shape)

    if mode == "jepa":
        from speech_decoding.pretraining.jepa_model import JEPAModel
        ssl_model = JEPAModel(config, grid_shape)
    elif mode == "byol":
        from speech_decoding.pretraining.byol_model import BYOLModel
        ssl_model = BYOLModel(config, grid_shape)
    elif mode == "dino":
        from speech_decoding.pretraining.dino_model import DINOModel
        ssl_model = DINOModel(config, grid_shape)
    elif mode == "masked":
        pretrain_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        return pretrain_model
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ssl_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    ssl_model.transfer_encoder_weights(pretrain_model)
    return pretrain_model


def extract_embeddings(model, grids, device, tta_copies=1):
    """Extract mean-pooled backbone features with optional TTA.

    Args:
        model: PretrainModel with encode() method.
        grids: (N, H, W, T) tensor.
        device: torch device.
        tta_copies: number of augmented copies to average (1=no TTA).

    Returns:
        (N, D) embeddings.
    """
    model.eval()
    with torch.no_grad():
        # Original (unaugmented)
        emb_sum = model.encode(grids.to(device)).mean(dim=1).cpu()

        # TTA: augmented copies
        for _ in range(tta_copies - 1):
            aug_grids = augment_from_config(grids, AUG_CONFIG, training=True)
            emb = model.encode(aug_grids.to(device)).mean(dim=1).cpu()
            emb_sum = emb_sum + emb

    return emb_sum / tta_copies


def weighted_knn_predict(train_emb, train_labels, val_emb, k=10):
    """Per-position weighted k-NN classification. Returns 1-indexed predictions."""
    train_n = F.normalize(train_emb, dim=1)
    val_n = F.normalize(val_emb, dim=1)
    sim = val_n @ train_n.T  # (N_val, N_train)
    topk_sim, topk_idx = sim.topk(k, dim=1)

    preds = []
    for i in range(val_emb.shape[0]):
        pred = []
        for pos in range(N_POSITIONS):
            class_weights = [0.0] * (N_CLASSES + 1)
            for j_idx in range(k):
                j = topk_idx[i, j_idx].item()
                cls = train_labels[j][pos]
                class_weights[cls] += topk_sim[i, j_idx].item()
            pred.append(int(np.argmax(class_weights[1:]) + 1))
        preds.append(pred)
    return preds


def compute_per(preds, refs):
    """Phoneme error rate: fraction of positions with wrong phoneme."""
    total, errors = 0, 0
    for pred, ref in zip(preds, refs):
        for p, r in zip(pred, ref):
            total += 1
            if p != r:
                errors += 1
    return errors / total if total > 0 else 1.0


def linear_probe_fold(model, train_grids, train_labels, val_grids, val_labels,
                      device, epochs=100, patience=10, lr=1e-3):
    """Train linear probe on frozen features, return predictions."""
    model.eval()
    gru_hidden = model.config["gru_hidden"]
    head = torch.nn.Linear(gru_hidden * 2, N_POSITIONS * N_CLASSES).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(epochs):
        head.train()
        with torch.no_grad():
            feat = model.encode(train_grids.to(device)).mean(dim=1)
        logits = head(feat).view(-1, N_POSITIONS, N_CLASSES)
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
            val_feat = model.encode(val_grids.to(device)).mean(dim=1)
            val_logits = head(val_feat).view(-1, N_POSITIONS, N_CLASSES)
            val_tgt = torch.tensor(val_labels, device=device, dtype=torch.long)
            val_loss = sum(
                F.cross_entropy(val_logits[:, p], val_tgt[:, p] - 1)
                for p in range(N_POSITIONS)
            ) / N_POSITIONS

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(head.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        val_feat = model.encode(val_grids.to(device)).mean(dim=1)
        logits = head(val_feat).view(-1, N_POSITIONS, N_CLASSES)
    return (logits.argmax(dim=-1) + 1).cpu().tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to stage2_checkpoint.pt")
    p.add_argument("--mode", required=True, choices=["jepa", "byol", "dino", "masked"])
    p.add_argument("--target", default="S14")
    p.add_argument("--paths", default="configs/paths.yaml")
    p.add_argument("--config", default="configs/pretrain_base.yaml")
    p.add_argument("--device", default="mps")
    p.add_argument("--tta", type=int, default=16, help="TTA copies (1=no TTA)")
    p.add_argument("--k", type=int, default=10, help="k-NN k value")
    p.add_argument("--output", default=None, help="Output JSON path")
    args = p.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    bids_root = paths.get("ps_bids_root") or paths["bids_root"]

    # Load target data
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
    logger.info("Loaded %s checkpoint from %s", args.mode, args.checkpoint)

    # Grouped CV
    groups = build_token_groups(labels)
    seed = _patient_seed(args.target)
    splits = create_grouped_splits(labels, groups, n_folds=5, seed=seed)

    fold_results = []
    base_state = deepcopy(model.state_dict())

    for fold_idx, fold in enumerate(splits):
        model.load_state_dict(base_state)
        train_idx = fold["train_indices"]
        val_idx = fold["val_indices"]

        train_grids = grids[train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_grids = grids[val_idx]
        val_labels = [labels[i] for i in val_idx]

        # Extract embeddings with TTA
        train_emb = extract_embeddings(model, train_grids, args.device, tta_copies=1)
        val_emb = extract_embeddings(model, val_grids, args.device, tta_copies=args.tta)

        # Weighted k-NN
        knn_preds = weighted_knn_predict(train_emb, train_labels, val_emb, k=args.k)
        knn_per = compute_per(knn_preds, val_labels)

        # Linear probe
        linear_preds = linear_probe_fold(
            model, train_grids, train_labels, val_grids, val_labels, args.device,
        )
        linear_per = compute_per(linear_preds, val_labels)

        # Best-of
        best_per = min(knn_per, linear_per)
        best_source = "knn" if knn_per <= linear_per else "linear"

        fold_results.append({
            "fold": fold_idx,
            "knn_per": knn_per,
            "linear_per": linear_per,
            "best_per": best_per,
            "best_source": best_source,
        })
        logger.info("  Fold %d: kNN=%.3f linear=%.3f → best=%.3f (%s)",
                     fold_idx, knn_per, linear_per, best_per, best_source)

    # Aggregate
    knn_pers = [r["knn_per"] for r in fold_results]
    linear_pers = [r["linear_per"] for r in fold_results]
    best_pers = [r["best_per"] for r in fold_results]

    results = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "target": args.target,
        "tta_copies": args.tta,
        "k": args.k,
        "knn_mean_per": float(np.mean(knn_pers)),
        "knn_std_per": float(np.std(knn_pers)),
        "linear_mean_per": float(np.mean(linear_pers)),
        "linear_std_per": float(np.std(linear_pers)),
        "best_mean_per": float(np.mean(best_pers)),
        "best_std_per": float(np.std(best_pers)),
        "fold_results": fold_results,
    }

    logger.info("=" * 60)
    logger.info("Results for %s (%s):", args.mode, args.checkpoint)
    logger.info("  k-NN PER:    %.3f ± %.3f", results["knn_mean_per"], results["knn_std_per"])
    logger.info("  Linear PER:  %.3f ± %.3f", results["linear_mean_per"], results["linear_std_per"])
    logger.info("  Best-of PER: %.3f ± %.3f", results["best_mean_per"], results["best_std_per"])

    out_path = args.output or str(Path(args.checkpoint).parent / "eval_recipe_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
