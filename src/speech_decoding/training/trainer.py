"""Per-patient trainer: end-to-end CTC training on a single patient.

Stratified 5-fold CV on the patient's trials. Used for Sprint 3
(per-patient baseline) before adding cross-patient LOPO in Sprint 5.
"""
from __future__ import annotations

import logging
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from speech_decoding.data.augmentation import augment_from_config
from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.evaluation.metrics import evaluate_predictions
from speech_decoding.models.assembler import assemble_model
from speech_decoding.training.ctc_utils import (
    blank_ratio,
    ce_pooled_decode,
    ce_pooled_loss,
    ctc_loss,
    greedy_decode,
    per_position_ce_decode,
    per_position_ce_loss,
)

logger = logging.getLogger(__name__)


def train_per_patient(
    dataset: BIDSDataset,
    config: dict,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Train and evaluate a model on a single patient with stratified CV.

    Args:
        dataset: BIDSDataset for one patient.
        config: Full YAML config dict.
        seed: Random seed.
        device: Device string.

    Returns:
        Dict with per-fold and mean metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tc = config["training"]["stage1"]
    ec = config["evaluation"]
    n_folds = ec["cv_folds"]

    # Get stratification labels (first phoneme of each trial)
    strat_labels = [y[0] for y in dataset.ctc_labels]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(
        np.zeros(len(dataset)), strat_labels
    )):
        logger.info("Fold %d/%d", fold_idx + 1, n_folds)
        result = _train_fold(
            dataset, config, train_idx, val_idx, seed, device,
        )
        fold_results.append(result)
        logger.info(
            "Fold %d: PER=%.3f, bal_acc=%.3f, blank_ratio=%.2f",
            fold_idx + 1, result["per"], result["bal_acc_mean"],
            result.get("blank_ratio", -1),
        )

    # Aggregate
    mean_metrics = {}
    for key in fold_results[0]:
        vals = [r[key] for r in fold_results]
        mean_metrics[f"{key}_mean"] = np.mean(vals).item()
        mean_metrics[f"{key}_std"] = np.std(vals).item()
    mean_metrics["fold_results"] = fold_results
    mean_metrics["patient_id"] = dataset.patient_id
    mean_metrics["seed"] = seed

    return mean_metrics


def _train_fold(
    dataset: BIDSDataset,
    config: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    device: str,
) -> dict:
    """Train one CV fold."""
    tc = config["training"]["stage1"]
    ac = config["training"].get(
        "augmentation",
        config["training"].get("stage1", {}).get("augmentation", {}),
    )
    loss_type = config["training"].get("loss_type", "ctc")
    n_segments = config["training"].get("ce_segments", 3)

    # Build model
    patients = {dataset.patient_id: dataset.grid_shape}
    backbone, head, readins = assemble_model(config, patients)
    readin = readins[dataset.patient_id]

    # For CE mode, replace head with a raw linear projecting to
    # n_positions * n_phonemes logits (3 independent 9-class classifiers
    # applied to globally pooled backbone output — matches Zac's per-position SVM)
    if loss_type == "ce":
        input_dim = config["model"]["hidden_size"] * 2  # bidirectional
        n_phonemes = config["model"]["num_classes"] - 1  # exclude blank
        head = nn.Linear(input_dim, n_segments * n_phonemes)
    elif loss_type == "ce_attn":
        from speech_decoding.models.ce_position_head import CEPositionHead

        input_dim = config["model"]["hidden_size"] * 2
        n_phonemes = config["model"]["num_classes"] - 1
        head = CEPositionHead(input_dim, n_positions=n_segments, n_phonemes=n_phonemes)
    elif loss_type == "ce_perpos":
        # Per-position epochs: each sample is a single phoneme, 9-way CE
        input_dim = config["model"]["hidden_size"] * 2
        n_phonemes = config["model"]["num_classes"] - 1
        head = nn.Linear(input_dim, n_phonemes)

    backbone = backbone.to(device)
    head = head.to(device)
    readin = readin.to(device)

    # Optimizer
    optimizer = AdamW(
        [
            {"params": readin.parameters(), "lr": tc["lr"] * tc["readin_lr_mult"]},
            {"params": backbone.parameters(), "lr": tc["lr"]},
            {"params": head.parameters(), "lr": tc["lr"]},
        ],
        weight_decay=tc["weight_decay"],
    )
    warmup_epochs = tc.get("warmup_epochs", 0)
    total_epochs = tc.get("epochs", tc.get("steps", 0))

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear ramp from ~0 to 1
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))  # cosine decay

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Split data — keep on CPU for augmentation, move to device in forward pass
    train_x = torch.from_numpy(dataset.grid_data[train_idx])
    train_y = [dataset.ctc_labels[i] for i in train_idx]
    val_x = torch.from_numpy(dataset.grid_data[val_idx])
    val_y = [dataset.ctc_labels[i] for i in val_idx]

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    B = tc["batch_size"]
    n_train = len(train_x)
    all_params = (
        list(backbone.parameters()) + list(head.parameters())
        + list(readin.parameters())
    )

    for epoch in range(tc["epochs"]):
        # --- Train (full epoch: iterate over all training data) ---
        backbone.train()
        head.train()
        readin.train()

        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, B):
            idx = perm[start : start + B]
            x_batch = train_x[idx]
            y_batch = [train_y[i] for i in idx.tolist()]

            # Augment on CPU (avoids MPS↔CPU sync stalls from per-trial loops)
            x_batch = augment_from_config(x_batch, ac, training=True)
            x_batch = x_batch.to(device)

            optimizer.zero_grad()
            shared = readin(x_batch)
            h = backbone(shared)
            out = head(h)
            if loss_type == "ce":
                loss = per_position_ce_loss(out, y_batch, n_segments)
            elif loss_type == "ce_attn":
                loss = ce_pooled_loss(out, y_batch, n_segments)
            elif loss_type == "ce_perpos":
                # out: (B, T, 9), single-phoneme labels [[k], ...]
                pooled = out.mean(dim=1)  # (B, 9)
                tgt = torch.tensor(
                    [y[0] - 1 for y in y_batch], dtype=torch.long, device=device
                )
                loss = F.cross_entropy(pooled, tgt)
            else:
                loss = ctc_loss(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, tc["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # --- Validate ---
        if (epoch + 1) % tc["eval_every"] == 0:
            backbone.eval()
            head.eval()
            readin.eval()
            with torch.no_grad():
                val_x_dev = val_x.to(device)
                shared = readin(val_x_dev)
                h = backbone(shared)
                out = head(h)
                if loss_type == "ce":
                    val_loss = per_position_ce_loss(out, val_y, n_segments).item()
                    br = 0.0
                elif loss_type == "ce_attn":
                    val_loss = ce_pooled_loss(out, val_y, n_segments).item()
                    br = 0.0
                elif loss_type == "ce_perpos":
                    pooled = out.mean(dim=1)
                    tgt = torch.tensor(
                        [y[0] - 1 for y in val_y], dtype=torch.long, device=device
                    )
                    val_loss = F.cross_entropy(pooled, tgt).item()
                    br = 0.0
                else:
                    val_loss = ctc_loss(out, val_y).item()
                    br = blank_ratio(out)

            logger.info(
                "  epoch %d: train_loss=%.4f val_loss=%.4f blank=%.0f%%",
                epoch + 1, avg_train_loss, val_loss, br * 100,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "backbone": deepcopy(backbone.state_dict()),
                    "head": deepcopy(head.state_dict()),
                    "readin": deepcopy(readin.state_dict()),
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= tc["patience"]:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # Restore best checkpoint
    if best_state is not None:
        backbone.load_state_dict(best_state["backbone"])
        head.load_state_dict(best_state["head"])
        readin.load_state_dict(best_state["readin"])

    # Final evaluation
    backbone.eval()
    head.eval()
    readin.eval()
    with torch.no_grad():
        val_x_dev = val_x.to(device)
        shared = readin(val_x_dev)
        h = backbone(shared)
        out = head(h)
        if loss_type == "ce":
            predictions = per_position_ce_decode(out, n_segments)
            br = 0.0
        elif loss_type == "ce_attn":
            predictions = ce_pooled_decode(out, n_segments)
            br = 0.0
        elif loss_type == "ce_perpos":
            # Single-phoneme predictions: argmax of mean-pooled logits
            pooled = out.mean(dim=1)  # (B, 9)
            predictions = [[p.item() + 1] for p in pooled.argmax(dim=-1)]
            br = 0.0
        else:
            predictions = greedy_decode(out)
            br = blank_ratio(out)

    eval_n_positions = 1 if loss_type == "ce_perpos" else 3
    metrics = evaluate_predictions(predictions, val_y, n_positions=eval_n_positions)
    metrics["blank_ratio"] = br
    return metrics
