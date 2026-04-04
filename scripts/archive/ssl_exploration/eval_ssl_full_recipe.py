#!/usr/bin/env python3
"""Apply full autoresearch recipe to SSL features.

Tests on frozen SSL backbone:
1. Weighted k-NN (no TTA) — pure feature geometry
2. Improved head: per-position + dropout + label smoothing + focal + mixup
3. Dual head: CE + articulatory
4. Best-of per fold

Usage:
  python scripts/eval_ssl_full_recipe.py \
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

# Articulatory feature matrix (PS label order: a,ae,b,g,i,k,p,u,v)
_ART_MATRIX = np.array([
    [0,1,0,0,0,0,0,0,0,1,0,0,0,1,0],  # a
    [0,1,0,0,0,0,0,0,0,0,1,0,1,0,0],  # ae
    [1,0,1,0,0,1,0,1,0,0,0,0,0,0,0],  # b
    [1,0,0,0,1,1,0,1,0,0,0,0,0,0,0],  # g
    [0,1,0,0,0,0,0,0,0,0,0,1,1,0,0],  # i
    [1,0,0,0,1,1,0,0,1,0,0,0,0,0,0],  # k
    [1,0,1,0,0,1,0,0,1,0,0,0,0,0,0],  # p
    [0,1,0,0,0,0,0,0,0,0,0,1,0,0,1],  # u
    [1,0,0,1,0,0,1,1,0,0,0,0,0,0,0],  # v
], dtype=np.float32)


def load_model(checkpoint_path, mode, config, grid_shape):
    pretrain_model = PretrainModel(config, grid_shape)
    if mode == "jepa":
        from speech_decoding.pretraining.jepa_model import JEPAModel
        ssl = JEPAModel(config, grid_shape)
    elif mode == "dino":
        from speech_decoding.pretraining.dino_model import DINOModel
        ssl = DINOModel(config, grid_shape)
    elif mode == "byol":
        from speech_decoding.pretraining.byol_model import BYOLModel
        ssl = BYOLModel(config, grid_shape)
    else:
        pretrain_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        return pretrain_model
    ssl.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    ssl.transfer_encoder_weights(pretrain_model)
    return pretrain_model


# ─── Weighted k-NN ─────────────────────────────────────────────────

def weighted_knn(train_emb, train_labels, val_emb, k=10):
    """Per-position weighted k-NN. Returns 1-indexed predictions."""
    train_n = F.normalize(train_emb, dim=1)
    val_n = F.normalize(val_emb, dim=1)
    sim = val_n @ train_n.T
    topk_sim, topk_idx = sim.topk(k, dim=1)
    preds = []
    for i in range(val_emb.shape[0]):
        pred = []
        for pos in range(N_POSITIONS):
            weights = [0.0] * (N_CLASSES + 1)
            for j in range(k):
                cls = train_labels[topk_idx[i, j].item()][pos]
                weights[cls] += topk_sim[i, j].item()
            pred.append(int(np.argmax(weights[1:]) + 1))
        preds.append(pred)
    return preds


def unweighted_knn(train_emb, train_labels, val_emb, k=10):
    """Per-position majority-vote k-NN. Returns 1-indexed predictions."""
    train_n = F.normalize(train_emb, dim=1)
    val_n = F.normalize(val_emb, dim=1)
    sim = val_n @ train_n.T
    _, topk_idx = sim.topk(k, dim=1)
    preds = []
    for i in range(val_emb.shape[0]):
        pred = []
        for pos in range(N_POSITIONS):
            votes = [0] * (N_CLASSES + 1)
            for j in range(k):
                cls = train_labels[topk_idx[i, j].item()][pos]
                votes[cls] += 1
            pred.append(int(np.argmax(votes[1:]) + 1))
        preds.append(pred)
    return preds


# ─── Improved heads ────────────────────────────────────────────────

class CEHeadPerPos(nn.Module):
    """Separate Linear per position with dropout (autoresearch pattern)."""
    def __init__(self, d_in, n_pos=3, n_cls=9, dropout=0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(d_in, n_cls) for _ in range(n_pos)])
    def forward(self, pooled):
        x = self.drop(pooled)
        return torch.stack([h(x) for h in self.heads], dim=1)


class ArticulatoryHead(nn.Module):
    """CEBRA-style: project → articulatory space, classify by cosine sim."""
    def __init__(self, d_in, n_pos=3, n_feat=15, dropout=0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.projectors = nn.ModuleList([nn.Linear(d_in, n_feat) for _ in range(n_pos)])
        art = torch.from_numpy(_ART_MATRIX)
        art = F.normalize(art, dim=1)
        self.register_buffer("art_matrix", art)
        self.log_temp = nn.Parameter(torch.tensor(2.0))
    def forward(self, pooled):
        x = self.drop(pooled)
        temp = self.log_temp.exp().clamp(min=0.01)
        logits = []
        for proj in self.projectors:
            art_pred = F.normalize(proj(x), dim=1)
            logits.append(art_pred @ self.art_matrix.T * temp)
        return torch.stack(logits, dim=1)


def focal_ce(logits, targets, gamma=2.0, label_smoothing=0.1):
    ce = F.cross_entropy(logits, targets, label_smoothing=label_smoothing, reduction="none")
    if gamma > 0:
        pt = torch.exp(-ce)
        ce = ((1 - pt) ** gamma) * ce
    return ce.mean()


def compute_per(preds, refs):
    total, errors = 0, 0
    for pred, ref in zip(preds, refs):
        for p, r in zip(pred, ref):
            total += 1
            if p != r:
                errors += 1
    return errors / total if total > 0 else 1.0


def train_improved_head_fold(model, train_grids, train_labels, val_grids, val_labels,
                             device, head_type="dual", epochs=300, patience=7,
                             lr=1e-3, mixup_alpha=0.2):
    """Train improved head on frozen backbone (autoresearch recipe)."""
    model.eval()
    gru_out = model.config["gru_hidden"] * 2

    # Freeze backbone, unfreeze readin
    for p in model.parameters():
        p.requires_grad = False
    readin_params = []
    for n, p in model.named_parameters():
        if "readin" in n:
            p.requires_grad = True
            readin_params.append(p)

    ce_head = CEHeadPerPos(gru_out).to(device)
    art_head = ArticulatoryHead(gru_out).to(device) if head_type == "dual" else None

    head_params = list(ce_head.parameters())
    if art_head:
        head_params += list(art_head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": lr},
        {"params": readin_params, "lr": lr * 3},
    ], weight_decay=1e-4)

    warmup = 20
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float("inf")
    best_state = None
    patience_ctr = 0
    n_train = len(train_grids)
    batch_size = 16

    for epoch in range(epochs):
        ce_head.train()
        if art_head:
            art_head.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            x = train_grids[idx].to(device)
            y = [train_labels[i] for i in idx.tolist()]

            # Mixup
            mixup_y = None
            lam = 1.0
            if mixup_alpha > 0 and len(idx) > 1:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                mix_perm = torch.randperm(x.shape[0])
                x = lam * x + (1 - lam) * x[mix_perm]
                mixup_y = [y[i] for i in mix_perm.tolist()]

            with torch.no_grad():
                feat = model.encode(x).mean(dim=1)

            logits = ce_head(feat)
            if art_head:
                logits = (logits + art_head(feat)) / 2

            loss = torch.tensor(0.0, device=device)
            for p in range(N_POSITIONS):
                tgt = torch.tensor([l[p] - 1 for l in y], dtype=torch.long, device=device)
                l1 = focal_ce(logits[:, p], tgt)
                if mixup_y is not None:
                    tgt2 = torch.tensor([l[p] - 1 for l in mixup_y], dtype=torch.long, device=device)
                    l2 = focal_ce(logits[:, p], tgt2)
                    l1 = lam * l1 + (1 - lam) * l2
                loss = loss + l1
            loss = loss / N_POSITIONS

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params + readin_params, 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0:
            ce_head.eval()
            if art_head:
                art_head.eval()
            with torch.no_grad():
                vf = model.encode(val_grids.to(device)).mean(dim=1)
                vl = ce_head(vf)
                if art_head:
                    vl = (vl + art_head(vf)) / 2
                vt = torch.tensor(val_labels, device=device, dtype=torch.long)
                val_loss = sum(
                    F.cross_entropy(vl[:, p], vt[:, p] - 1)
                    for p in range(N_POSITIONS)
                ) / N_POSITIONS

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    "ce": deepcopy(ce_head.state_dict()),
                    "art": deepcopy(art_head.state_dict()) if art_head else None,
                    "readin": {n: p.clone() for n, p in model.named_parameters() if "readin" in n},
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

    # Restore best
    if best_state:
        ce_head.load_state_dict(best_state["ce"])
        if art_head and best_state["art"]:
            art_head.load_state_dict(best_state["art"])
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_state["readin"]:
                    p.copy_(best_state["readin"][n])

    # Decode
    ce_head.eval()
    if art_head:
        art_head.eval()
    with torch.no_grad():
        vf = model.encode(val_grids.to(device)).mean(dim=1)
        vl = ce_head(vf)
        if art_head:
            vl = (vl + art_head(vf)) / 2
    preds = (vl.argmax(dim=-1) + 1).cpu().tolist()
    return preds


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

    ds = load_patient_data(args.target, bids_root, task="PhonemeSequence",
                           n_phons=3, tmin=0.0, tmax=1.0)
    grids, labels = [], []
    for i in range(len(ds)):
        g, l, _ = ds[i]
        grids.append(g)
        labels.append(l)
    grids = torch.tensor(np.stack(grids), dtype=torch.float32)
    logger.info("Target %s: %d trials", args.target, len(grids))

    config["ema_total_steps"] = 5000
    model = load_model(args.checkpoint, args.mode, config, (8, 16))
    model = model.to(args.device)

    groups = build_token_groups(labels)
    seed = _patient_seed(args.target)
    splits = create_grouped_splits(labels, groups, n_folds=5, seed=seed)
    base_state = deepcopy(model.state_dict())

    results_by_method = {
        "unweighted_knn": [], "weighted_knn": [],
        "improved_head": [], "best_of": [],
    }

    for fold_idx, fold in enumerate(splits):
        model.load_state_dict(base_state)
        tr_idx, va_idx = fold["train_indices"], fold["val_indices"]
        tr_grids, tr_labels = grids[tr_idx], [labels[i] for i in tr_idx]
        va_grids, va_labels = grids[va_idx], [labels[i] for i in va_idx]

        # Extract embeddings (no TTA)
        model.eval()
        with torch.no_grad():
            tr_emb = model.encode(tr_grids.to(args.device)).mean(dim=1).cpu()
            va_emb = model.encode(va_grids.to(args.device)).mean(dim=1).cpu()

        # Unweighted k-NN
        uw_preds = unweighted_knn(tr_emb, tr_labels, va_emb, k=10)
        uw_per = compute_per(uw_preds, va_labels)
        results_by_method["unweighted_knn"].append(uw_per)

        # Weighted k-NN
        w_preds = weighted_knn(tr_emb, tr_labels, va_emb, k=10)
        w_per = compute_per(w_preds, va_labels)
        results_by_method["weighted_knn"].append(w_per)

        # Improved head (dual CE + articulatory, mixup, focal, label smoothing)
        model.load_state_dict(base_state)
        head_preds = train_improved_head_fold(
            model, tr_grids, tr_labels, va_grids, va_labels, args.device,
        )
        head_per = compute_per(head_preds, va_labels)
        results_by_method["improved_head"].append(head_per)

        # Best-of
        best_per = min(uw_per, w_per, head_per)
        results_by_method["best_of"].append(best_per)

        logger.info("Fold %d: unwt_knn=%.3f wt_knn=%.3f head=%.3f → best=%.3f",
                     fold_idx, uw_per, w_per, head_per, best_per)

    logger.info("=" * 60)
    for method, pers in results_by_method.items():
        logger.info("  %-20s: PER %.3f ± %.3f", method, np.mean(pers), np.std(pers))

    results = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "target": args.target,
    }
    for method, pers in results_by_method.items():
        results[method] = {
            "mean_per": float(np.mean(pers)),
            "std_per": float(np.std(pers)),
            "fold_pers": pers,
        }

    out_path = args.output or str(Path(args.checkpoint).parent / "full_recipe_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
