#!/usr/bin/env python3
"""LP-FT (Linear Probe then Fine-Tune) on SSL checkpoints.

Kumar et al. 2022: two-phase training prevents feature distortion during fine-tuning.

Phase 1 (LP): Freeze backbone, train linear head to convergence on frozen features.
Phase 2 (FT): Unfreeze backbone with 10× lower LR, continue training with initialized head.

The key insight: a random head sends noisy gradients into the backbone during early training,
destroying pretrained representations. LP-FT initializes the head first, so fine-tuning
gradients are meaningful from the start.

Usage:
  python scripts/run_lpft.py \
    --checkpoint results/pretrain/method_B_jepa/S14/stage2_checkpoint.pt \
    --mode jepa --target S14
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


class CEHeadPerPos(nn.Module):
    """Per-position linear head with dropout."""

    def __init__(self, d_in, n_positions=3, n_classes=9, dropout=0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Linear(d_in, n_classes) for _ in range(n_positions)
        ])

    def forward(self, h):
        """h: (B, T', D) → mean-pool → (B, n_pos, n_cls)."""
        pooled = self.drop(h.mean(dim=1))
        return torch.stack([head(pooled) for head in self.heads], dim=1)


def focal_ce(logits, targets, gamma=2.0, label_smoothing=0.1):
    """Focal cross-entropy with label smoothing."""
    ce = F.cross_entropy(logits, targets, label_smoothing=label_smoothing, reduction="none")
    if gamma > 0:
        pt = torch.exp(-ce)
        ce = ((1 - pt) ** gamma) * ce
    return ce.mean()


def compute_loss(logits, labels, mixup_labels=None, mixup_lam=1.0):
    """Per-position focal CE with optional mixup."""
    loss = torch.tensor(0.0, device=logits.device)
    for pos in range(N_POSITIONS):
        tgt = torch.tensor([l[pos] - 1 for l in labels], dtype=torch.long, device=logits.device)
        pos_loss = focal_ce(logits[:, pos, :], tgt)
        if mixup_labels is not None:
            tgt2 = torch.tensor([l[pos] - 1 for l in mixup_labels], dtype=torch.long, device=logits.device)
            pos_loss2 = focal_ce(logits[:, pos, :], tgt2)
            pos_loss = mixup_lam * pos_loss + (1 - mixup_lam) * pos_loss2
        loss = loss + pos_loss
    return loss / N_POSITIONS


def weighted_knn(train_emb, train_labels, val_emb, k=10):
    """Per-position weighted k-NN classification."""
    train_n = F.normalize(train_emb, dim=1)
    val_n = F.normalize(val_emb, dim=1)
    sim = val_n @ train_n.T
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
    total, errors = 0, 0
    for pred, ref in zip(preds, refs):
        for p, r in zip(pred, ref):
            total += 1
            if p != r:
                errors += 1
    return errors / total if total > 0 else 1.0


def augment(grids):
    return augment_from_config(grids, AUG_CONFIG, training=True)


def extract_embeddings(model, grids, device):
    """Extract mean-pooled backbone features."""
    model.eval()
    with torch.no_grad():
        return model.encode(grids.to(device)).mean(dim=1).cpu()


def lpft_fold(model, train_grids, train_labels, val_grids, val_labels, device,
              lp_epochs=150, lp_patience=10, lp_lr=1e-3,
              ft_epochs=100, ft_patience=7, ft_backbone_lr=1e-4,
              ft_head_lr=3e-4, ft_readin_lr=3e-4,
              batch_size=16, mixup_alpha=0.2, warmup_epochs=20,
              grad_clip=5.0):
    """Run LP-FT on one fold. Returns dict with PER results."""
    gru_hidden = model.config["gru_hidden"]
    d_out = gru_hidden * 2
    head = CEHeadPerPos(d_out, N_POSITIONS, N_CLASSES, dropout=0.3).to(device)
    n_train = len(train_grids)

    # ──────────────────────────────────────────────────────────
    # Phase 1: Linear Probe (frozen backbone)
    # ──────────────────────────────────────────────────────────
    logger.info("    Phase 1: Linear Probe (%d epochs, lr=%.1e)", lp_epochs, lp_lr)
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze readin (following Stage3Evaluator convention)
    readin_params = [p for n, p in model.named_parameters() if "readin" in n]
    for p in readin_params:
        p.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": head.parameters(), "lr": lp_lr},
        {"params": readin_params, "lr": lp_lr * 3},
    ], weight_decay=1e-4)

    def make_scheduler(opt, total_epochs, warmup):
        def lr_lambda(epoch):
            if warmup > 0 and epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(total_epochs - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    scheduler = make_scheduler(optimizer, lp_epochs, warmup_epochs)

    best_val_loss = float("inf")
    best_head_state = None
    best_readin_state = None
    patience_ctr = 0

    for epoch in range(lp_epochs):
        head.train()
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            x = augment(train_grids[idx]).to(device)
            y = [train_labels[i] for i in idx.tolist()]

            mixup_y = None
            mixup_lam = 1.0
            if mixup_alpha > 0 and len(idx) > 1:
                mixup_lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm_mix = torch.randperm(x.shape[0])
                x = mixup_lam * x + (1 - mixup_lam) * x[perm_mix]
                mixup_y = [y[i] for i in perm_mix.tolist()]

            optimizer.zero_grad()
            with torch.no_grad():
                feat = model.encode(x)
            logits = head(feat)
            loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(head.parameters()) + readin_params, grad_clip
            )
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            head.eval()
            model.eval()
            with torch.no_grad():
                feat = model.encode(val_grids.to(device))
                logits = head(feat)
                val_loss = compute_loss(logits, val_labels).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_head_state = deepcopy(head.state_dict())
                best_readin_state = {n: p.clone() for n, p in model.named_parameters()
                                     if "readin" in n}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= lp_patience:
                    logger.info("    LP early stop at epoch %d", epoch + 1)
                    break

    # Restore best LP state
    if best_head_state:
        head.load_state_dict(best_head_state)
    if best_readin_state:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_readin_state:
                    p.copy_(best_readin_state[n])

    # Evaluate LP-only results
    head.eval()
    model.eval()
    with torch.no_grad():
        feat = model.encode(val_grids.to(device))
        logits = head(feat)
        lp_preds = (logits.argmax(dim=-1) + 1).cpu().tolist()
    lp_per = compute_per(lp_preds, val_labels)

    train_emb = extract_embeddings(model, train_grids, device)
    val_emb = extract_embeddings(model, val_grids, device)
    lp_knn_preds = weighted_knn(train_emb, train_labels, val_emb)
    lp_knn_per = compute_per(lp_knn_preds, val_labels)
    logger.info("    LP result: linear=%.3f, kNN=%.3f", lp_per, lp_knn_per)

    # Save LP state for comparison
    lp_model_state = deepcopy(model.state_dict())

    # ──────────────────────────────────────────────────────────
    # Phase 2: Fine-Tune (unfreeze backbone with low LR)
    # ──────────────────────────────────────────────────────────
    logger.info("    Phase 2: Fine-Tune (backbone_lr=%.1e, head_lr=%.1e)", ft_backbone_lr, ft_head_lr)

    # Unfreeze ALL backbone params with differential LR
    backbone_params = []
    readin_params_ft = []
    for n, p in model.named_parameters():
        p.requires_grad = True
        if "readin" in n:
            readin_params_ft.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": head.parameters(), "lr": ft_head_lr},
        {"params": backbone_params, "lr": ft_backbone_lr},
        {"params": readin_params_ft, "lr": ft_readin_lr},
    ], weight_decay=1e-4)

    scheduler = make_scheduler(optimizer, ft_epochs, warmup=10)

    best_val_loss = float("inf")
    best_ft_head_state = None
    best_ft_model_state = None
    patience_ctr = 0

    for epoch in range(ft_epochs):
        head.train()
        model.train()
        perm = torch.randperm(n_train)

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            x = augment(train_grids[idx]).to(device)
            y = [train_labels[i] for i in idx.tolist()]

            mixup_y = None
            mixup_lam = 1.0
            if mixup_alpha > 0 and len(idx) > 1:
                mixup_lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm_mix = torch.randperm(x.shape[0])
                x = mixup_lam * x + (1 - mixup_lam) * x[perm_mix]
                mixup_y = [y[i] for i in perm_mix.tolist()]

            optimizer.zero_grad()
            feat = model.encode(x)  # backbone gets gradients now
            logits = head(feat)
            loss = compute_loss(logits, y, mixup_labels=mixup_y, mixup_lam=mixup_lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            head.eval()
            model.eval()
            with torch.no_grad():
                feat = model.encode(val_grids.to(device))
                logits = head(feat)
                val_loss = compute_loss(logits, val_labels).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ft_head_state = deepcopy(head.state_dict())
                best_ft_model_state = deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= ft_patience:
                    logger.info("    FT early stop at epoch %d", epoch + 1)
                    break

    # Restore best FT state
    if best_ft_head_state:
        head.load_state_dict(best_ft_head_state)
    if best_ft_model_state:
        model.load_state_dict(best_ft_model_state)

    # Evaluate FT results
    head.eval()
    model.eval()
    with torch.no_grad():
        feat = model.encode(val_grids.to(device))
        logits = head(feat)
        ft_preds = (logits.argmax(dim=-1) + 1).cpu().tolist()
    ft_per = compute_per(ft_preds, val_labels)

    train_emb = extract_embeddings(model, train_grids, device)
    val_emb = extract_embeddings(model, val_grids, device)
    ft_knn_preds = weighted_knn(train_emb, train_labels, val_emb)
    ft_knn_per = compute_per(ft_knn_preds, val_labels)
    logger.info("    FT result: linear=%.3f, kNN=%.3f", ft_per, ft_knn_per)

    # Best overall
    best_per = min(lp_per, lp_knn_per, ft_per, ft_knn_per)
    best_source = ["lp_linear", "lp_knn", "ft_linear", "ft_knn"][
        [lp_per, lp_knn_per, ft_per, ft_knn_per].index(best_per)
    ]

    return {
        "lp_linear_per": lp_per,
        "lp_knn_per": lp_knn_per,
        "ft_linear_per": ft_per,
        "ft_knn_per": ft_knn_per,
        "best_per": best_per,
        "best_source": best_source,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", required=True, choices=["jepa", "byol", "dino", "masked"])
    p.add_argument("--target", default="S14")
    p.add_argument("--paths", default="configs/paths.yaml")
    p.add_argument("--config", default="configs/pretrain_base.yaml")
    p.add_argument("--device", default="mps")
    p.add_argument("--output", default=None)
    # LP-FT hyperparameters
    p.add_argument("--lp-epochs", type=int, default=150)
    p.add_argument("--lp-lr", type=float, default=1e-3)
    p.add_argument("--ft-epochs", type=int, default=100)
    p.add_argument("--ft-backbone-lr", type=float, default=1e-4,
                   help="Backbone LR during fine-tuning (10× lower than LP)")
    p.add_argument("--ft-head-lr", type=float, default=3e-4)
    p.add_argument("--ft-readin-lr", type=float, default=3e-4)
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

    # Grouped CV splits
    groups = build_token_groups(labels)
    seed = _patient_seed(args.target)
    splits = create_grouped_splits(labels, groups, n_folds=5, seed=seed)

    config["ema_total_steps"] = 5000

    fold_results = []
    for fold_idx, fold in enumerate(splits):
        logger.info("=" * 60)
        logger.info("FOLD %d/5", fold_idx + 1)
        tr_idx, va_idx = fold["train_indices"], fold["val_indices"]
        logger.info("  Train: %d, Val: %d", len(tr_idx), len(va_idx))

        # Fresh model from checkpoint each fold
        model = load_model(args.checkpoint, args.mode, config,
                           (grids.shape[1], grids.shape[2]))
        model = model.to(args.device)

        result = lpft_fold(
            model,
            grids[tr_idx], [labels[i] for i in tr_idx],
            grids[va_idx], [labels[i] for i in va_idx],
            args.device,
            lp_epochs=args.lp_epochs, lp_lr=args.lp_lr,
            ft_epochs=args.ft_epochs,
            ft_backbone_lr=args.ft_backbone_lr,
            ft_head_lr=args.ft_head_lr,
            ft_readin_lr=args.ft_readin_lr,
        )
        fold_results.append(result)
        logger.info("  Fold %d: LP=%.3f/%s, FT=%.3f/%s → best=%.3f (%s)",
                     fold_idx + 1,
                     min(result["lp_linear_per"], result["lp_knn_per"]),
                     "knn" if result["lp_knn_per"] < result["lp_linear_per"] else "lin",
                     min(result["ft_linear_per"], result["ft_knn_per"]),
                     "knn" if result["ft_knn_per"] < result["ft_linear_per"] else "lin",
                     result["best_per"], result["best_source"])

    # Aggregate
    lp_linear_pers = [r["lp_linear_per"] for r in fold_results]
    lp_knn_pers = [r["lp_knn_per"] for r in fold_results]
    ft_linear_pers = [r["ft_linear_per"] for r in fold_results]
    ft_knn_pers = [r["ft_knn_per"] for r in fold_results]
    best_pers = [r["best_per"] for r in fold_results]

    results = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "target": args.target,
        "method": "lpft",
        "lp_linear_mean_per": float(np.mean(lp_linear_pers)),
        "lp_knn_mean_per": float(np.mean(lp_knn_pers)),
        "ft_linear_mean_per": float(np.mean(ft_linear_pers)),
        "ft_knn_mean_per": float(np.mean(ft_knn_pers)),
        "best_mean_per": float(np.mean(best_pers)),
        "best_std_per": float(np.std(best_pers)),
        "fold_results": fold_results,
        "hyperparams": {
            "lp_epochs": args.lp_epochs, "lp_lr": args.lp_lr,
            "ft_epochs": args.ft_epochs,
            "ft_backbone_lr": args.ft_backbone_lr,
            "ft_head_lr": args.ft_head_lr,
            "ft_readin_lr": args.ft_readin_lr,
        },
    }

    logger.info("=" * 60)
    logger.info("LP-FT RESULTS for %s (%s):", args.mode, args.checkpoint)
    logger.info("  LP linear: %.3f ± %.3f", np.mean(lp_linear_pers), np.std(lp_linear_pers))
    logger.info("  LP k-NN:   %.3f ± %.3f", np.mean(lp_knn_pers), np.std(lp_knn_pers))
    logger.info("  FT linear: %.3f ± %.3f", np.mean(ft_linear_pers), np.std(ft_linear_pers))
    logger.info("  FT k-NN:   %.3f ± %.3f", np.mean(ft_knn_pers), np.std(ft_knn_pers))
    logger.info("  Best-of:   %.3f ± %.3f", results["best_mean_per"], results["best_std_per"])

    out_path = args.output or str(Path(args.checkpoint).parent / "lpft_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
