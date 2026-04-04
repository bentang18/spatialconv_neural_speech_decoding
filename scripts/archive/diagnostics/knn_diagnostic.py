#!/usr/bin/env python3
"""k-NN diagnostic on frozen pretrained features.

No head training — tests whether the embedding space groups
same-phoneme trials together. The standard SSL feature quality
check from DINO/I-JEPA.

Usage:
  python scripts/knn_diagnostic.py --checkpoint results/pretrain/method_B_jepa/S14/stage2_checkpoint.pt \
    --encoder-type jepa --paths configs/paths.yaml --target S14 --device mps
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.pretraining.pretrain_model import PretrainModel
from speech_decoding.pretraining.jepa_model import JEPAModel
from speech_decoding.pretraining.lewm_model import LeWMModel
from speech_decoding.evaluation.grouped_cv import (
    _patient_seed,
    build_token_groups,
    create_grouped_splits,
)
from speech_decoding.evaluation.content_collapse import content_collapse_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def knn_classify(train_features, train_labels, val_features, k=5):
    """Classify val samples by k nearest train neighbors (cosine similarity).

    Args:
        train_features: (N_train, D) normalized feature vectors.
        train_labels: list of label sequences [[p1,p2,p3], ...].
        val_features: (N_val, D) normalized feature vectors.
        k: number of neighbors.

    Returns:
        List of predicted label sequences.
    """
    # Cosine similarity (features assumed normalized)
    sim = val_features @ train_features.T  # (N_val, N_train)
    topk_idx = sim.topk(k, dim=1).indices  # (N_val, k)

    predictions = []
    n_positions = len(train_labels[0])
    for i in range(len(val_features)):
        neighbor_idx = topk_idx[i].tolist()
        neighbor_labels = [train_labels[j] for j in neighbor_idx]
        # Majority vote per position
        pred = []
        for pos in range(n_positions):
            votes = [nl[pos] for nl in neighbor_labels]
            pred.append(max(set(votes), key=votes.count))
        predictions.append(pred)
    return predictions


def compute_per_from_preds(predictions, targets):
    """Simple PER: fraction of incorrect phoneme predictions."""
    total, errors = 0, 0
    for pred, tgt in zip(predictions, targets):
        for p, t in zip(pred, tgt):
            total += 1
            if p != t:
                errors += 1
    return errors / total if total > 0 else 1.0


def main():
    p = argparse.ArgumentParser(description="k-NN diagnostic on pretrained features")
    p.add_argument("--checkpoint", required=True, help="Stage 2 checkpoint path")
    p.add_argument("--encoder-type", default="jepa", choices=["jepa", "lewm", "masked"],
                   help="Which encoder produced the checkpoint")
    p.add_argument("--paths", required=True, help="paths.yaml")
    p.add_argument("--config", default="configs/pretrain_base.yaml")
    p.add_argument("--target", required=True, help="Target patient")
    p.add_argument("--device", default="mps")
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10, 20],
                   help="k values to evaluate")
    p.add_argument("--pooling", default="mean", choices=["mean", "mean_max", "mean_max_std"],
                   help="Temporal pooling strategy")
    p.add_argument("--cv-type", default="grouped", choices=["grouped", "stratified"],
                   help="CV split strategy: grouped-by-token or stratified on position-1 phoneme")
    args = p.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    bids_root = paths.get("ps_bids_root") or paths["bids_root"]

    # Load model and checkpoint
    if args.encoder_type == "jepa":
        config["ema_total_steps"] = 5000
        ssl_model = JEPAModel(config, grid_shape=(8, 16))
        ssl_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        # Transfer target encoder to PretrainModel
        model = PretrainModel(config, grid_shape=(8, 16))
        ssl_model.transfer_encoder_weights(model)
        logger.info("Loaded JEPA checkpoint, transferred target encoder")
    elif args.encoder_type == "lewm":
        ssl_model = LeWMModel(config, grid_shape=(8, 16))
        ssl_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        model = PretrainModel(config, grid_shape=(8, 16))
        ssl_model.transfer_encoder_weights(model)
        logger.info("Loaded LeWM checkpoint, transferred encoder")
    else:
        model = PretrainModel(config, grid_shape=(8, 16))
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        logger.info("Loaded masked PretrainModel checkpoint")

    model = model.to(args.device)
    model.eval()

    # Load target patient data
    ds = load_patient_data(args.target, bids_root, task="PhonemeSequence",
                           n_phons=3, tmin=-0.5, tmax=1.0)
    grids, labels = [], []
    for i in range(len(ds)):
        g, l, _ = ds[i]
        grids.append(g)
        labels.append(l)
    grids = torch.tensor(np.stack(grids), dtype=torch.float32)
    logger.info("%s: %d trials", args.target, len(grids))

    # Extract features
    with torch.no_grad():
        features = model.encode(grids.to(args.device))  # (N, T', 2H)

    if args.pooling == "mean":
        pooled = features.mean(dim=1)  # (N, 2H)
    elif args.pooling == "mean_max":
        pooled = torch.cat([features.mean(dim=1), features.max(dim=1).values], dim=-1)
    elif args.pooling == "mean_max_std":
        pooled = torch.cat([
            features.mean(dim=1),
            features.max(dim=1).values,
            features.std(dim=1),
        ], dim=-1)

    # L2 normalize for cosine similarity
    pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    pooled = pooled.cpu()

    # CV splits
    seed = _patient_seed(args.target)
    if args.cv_type == "grouped":
        groups = build_token_groups(labels)
        splits = create_grouped_splits(labels, groups, n_folds=5, seed=seed)
        logger.info("Using grouped-by-token CV (5 folds)")
    else:
        from sklearn.model_selection import StratifiedKFold
        y_strat = np.array([lab[0] for lab in labels])  # position-1 phoneme
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = []
        for train_idx, val_idx in skf.split(np.zeros(len(labels)), y_strat):
            splits.append({"train_indices": train_idx, "val_indices": val_idx})
        logger.info("Using stratified CV on position-1 phoneme (5 folds)")

    results = {}
    for k in args.k:
        fold_pers = []
        all_preds = []
        all_targets = []
        for fold_idx, fold in enumerate(splits):
            train_idx = fold["train_indices"]
            val_idx = fold["val_indices"]

            train_feat = pooled[train_idx]
            train_lab = [labels[i] for i in train_idx]
            val_feat = pooled[val_idx]
            val_lab = [labels[i] for i in val_idx]

            preds = knn_classify(train_feat, train_lab, val_feat, k=k)
            per = compute_per_from_preds(preds, val_lab)
            fold_pers.append(per)
            all_preds.extend(preds)
            all_targets.extend(val_lab)

        mean_per = float(np.mean(fold_pers))
        std_per = float(np.std(fold_pers))
        results[f"k={k}"] = {
            "mean_per": mean_per,
            "std_per": std_per,
            "fold_pers": fold_pers,
        }
        logger.info("k=%d: PER %.3f ± %.3f  folds=%s",
                     k, mean_per, std_per, [f"{p:.3f}" for p in fold_pers])

    # Also run with random features as control
    torch.manual_seed(42)
    random_pooled = torch.randn_like(pooled)
    random_pooled = random_pooled / random_pooled.norm(dim=-1, keepdim=True)
    random_pers = []
    for fold in splits:
        train_idx = fold["train_indices"]
        val_idx = fold["val_indices"]
        preds = knn_classify(random_pooled[train_idx], [labels[i] for i in train_idx],
                             random_pooled[val_idx], k=5)
        random_pers.append(compute_per_from_preds(preds, [labels[i] for i in val_idx]))
    results["random_k=5"] = {
        "mean_per": float(np.mean(random_pers)),
        "std_per": float(np.std(random_pers)),
        "fold_pers": random_pers,
    }
    logger.info("Random features k=5: PER %.3f ± %.3f", np.mean(random_pers), np.std(random_pers))

    # Summary
    logger.info("\n--- Summary (pooling=%s, cv=%s) ---", args.pooling, args.cv_type)
    logger.info("%-15s %s", "Method", "PER")
    for name, r in results.items():
        logger.info("%-15s %.3f ± %.3f", name, r["mean_per"], r["std_per"])
    logger.info("%-15s %.3f", "Chance", 8/9)  # 1 - 1/9
    logger.info("%-15s %.3f", "Supervised (D)", 0.825)

    # Save
    out_path = Path(args.checkpoint).parent / f"knn_diagnostic_{args.pooling}_{args.cv_type}.json"
    with open(out_path, "w") as f:
        json.dump({"pooling": args.pooling, "cv_type": args.cv_type, "results": results}, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
