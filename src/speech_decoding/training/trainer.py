"""Per-patient trainer: end-to-end CTC training on a single patient.

Stratified 5-fold CV on the patient's trials. Used for Sprint 3
(per-patient baseline) before adding cross-patient LOPO in Sprint 5.
"""
from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from speech_decoding.data.augmentation import augment_batch
from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.evaluation.metrics import evaluate_predictions
from speech_decoding.models.assembler import assemble_model
from speech_decoding.training.ctc_utils import (
    blank_ratio,
    ctc_loss,
    greedy_decode,
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
    ac = config["training"]["augmentation"]

    # Build model
    patients = {dataset.patient_id: dataset.grid_shape}
    backbone, head, readins = assemble_model(config, patients)
    readin = readins[dataset.patient_id]

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
    scheduler = CosineAnnealingLR(optimizer, T_max=tc["epochs"])

    # Split data
    train_x = torch.from_numpy(dataset.grid_data[train_idx]).to(device)
    train_y = [dataset.ctc_labels[i] for i in train_idx]
    val_x = torch.from_numpy(dataset.grid_data[val_idx]).to(device)
    val_y = [dataset.ctc_labels[i] for i in val_idx]

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(tc["epochs"]):
        # --- Train ---
        backbone.train()
        head.train()
        readin.train()

        # Sample batch (or use all if small enough)
        B = min(tc["batch_size"], len(train_x))
        perm = torch.randperm(len(train_x))[:B]
        x_batch = train_x[perm]
        y_batch = [train_y[i] for i in perm]

        # Augment
        x_batch = augment_batch(
            x_batch,
            training=True,
            time_shift_frames=ac["time_shift_frames"],
            amp_scale_std=ac["amp_scale_std"],
            channel_dropout_max=ac["channel_dropout_max"],
            noise_frac=ac["noise_frac"],
        )

        optimizer.zero_grad()
        shared = readin(x_batch)
        h = backbone(shared)
        log_probs = head(h)
        loss = ctc_loss(log_probs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(head.parameters()) + list(readin.parameters()),
            tc["grad_clip"],
        )
        optimizer.step()
        scheduler.step()

        # --- Validate ---
        if (epoch + 1) % tc["eval_every"] == 0:
            backbone.eval()
            head.eval()
            readin.eval()
            with torch.no_grad():
                shared = readin(val_x)
                h = backbone(shared)
                log_probs = head(h)
                val_loss = ctc_loss(log_probs, val_y).item()

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
        shared = readin(val_x)
        h = backbone(shared)
        log_probs = head(h)
        predictions = greedy_decode(log_probs)
        br = blank_ratio(log_probs)

    metrics = evaluate_predictions(predictions, val_y, n_positions=3)
    metrics["blank_ratio"] = br
    return metrics
