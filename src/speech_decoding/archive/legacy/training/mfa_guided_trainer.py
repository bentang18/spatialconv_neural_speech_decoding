"""Per-patient trainer with MFA-guided per-position CE loss.

Instead of mean-pooling all backbone frames into one vector, each
phoneme position classifier pools only over the frames corresponding
to its phoneme (defined by MFA boundaries). This gives each classifier
a cleaner signal without requiring the model to learn temporal alignment.

Optionally combines with phonological auxiliary BCE loss.
"""
from __future__ import annotations

import logging
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from speech_decoding.data.augmentation import augment_from_config
from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.evaluation.metrics import evaluate_predictions
from speech_decoding.models.assembler import assemble_model
from speech_decoding.training.ctc_utils import mfa_guided_ce_decode, mfa_guided_ce_loss
from speech_decoding.training.phonological_aux import (
    per_position_feature_bce_loss,
    per_position_feature_metrics,
)

logger = logging.getLogger(__name__)


def train_per_patient_mfa_guided(
    dataset: BIDSDataset,
    segment_masks: np.ndarray,
    config: dict,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Train with MFA-guided per-position CE on one patient.

    Args:
        dataset: BIDSDataset with grid data and CTC labels.
        segment_masks: (n_trials, n_positions, n_backbone_frames) per-phoneme
            frame weights from MFA boundaries.
        config: Full YAML config dict.
        seed: Random seed.
        device: Device string.

    Returns:
        Dict with per-fold and mean metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if len(dataset) != len(segment_masks):
        raise ValueError(
            f"Dataset ({len(dataset)}) and segment_masks ({len(segment_masks)}) "
            "must have matching trial counts."
        )

    n_folds = config["evaluation"]["cv_folds"]
    strat_labels = [y[0] for y in dataset.ctc_labels]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(dataset)), strat_labels)
    ):
        logger.info("MFA-guided fold %d/%d", fold_idx + 1, n_folds)
        result = _train_mfa_fold(
            dataset, segment_masks, config, train_idx, val_idx, seed, device
        )
        fold_results.append(result)
        logger.info(
            "Fold %d: PER=%.3f, bal_acc=%.3f",
            fold_idx + 1, result["per"], result["bal_acc_mean"],
        )

    mean_metrics: dict = {}
    keys = sorted({k for r in fold_results for k in r})
    for key in keys:
        vals = [r[key] for r in fold_results if key in r]
        arr = np.asarray(vals, dtype=float)
        mean_metrics[f"{key}_mean"] = np.nanmean(arr).item()
        mean_metrics[f"{key}_std"] = np.nanstd(arr).item()
    mean_metrics["fold_results"] = fold_results
    mean_metrics["patient_id"] = dataset.patient_id
    mean_metrics["seed"] = seed
    return mean_metrics


def _train_mfa_fold(
    dataset: BIDSDataset,
    segment_masks: np.ndarray,
    config: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    device: str,
) -> dict:
    tc = config["training"]["stage1"]
    ac = config["training"].get(
        "augmentation",
        config["training"].get("stage1", {}).get("augmentation", {}),
    )
    mc = config["model"]
    n_positions = config["training"].get("ce_segments", 3)
    n_phonemes = mc["num_classes"] - 1
    input_dim = mc["hidden_size"] * 2
    aux_lambda = config["training"].get("phonological_aux_lambda", 0.0)
    n_features = config["training"].get("phonological_num_features", 15)

    # Build model — one shared 9-way head (not 27-way)
    patients = {dataset.patient_id: dataset.grid_shape}
    backbone, _, readins = assemble_model(config, patients)
    readin = readins[dataset.patient_id]
    ce_head = nn.Linear(input_dim, n_phonemes)

    # Optional phonological aux head
    aux_head = None
    if aux_lambda > 0:
        aux_head = nn.Linear(input_dim, n_positions * n_features)

    backbone = backbone.to(device)
    readin = readin.to(device)
    ce_head = ce_head.to(device)
    if aux_head is not None:
        aux_head = aux_head.to(device)

    param_groups = [
        {"params": readin.parameters(), "lr": tc["lr"] * tc["readin_lr_mult"]},
        {"params": backbone.parameters(), "lr": tc["lr"]},
        {"params": ce_head.parameters(), "lr": tc["lr"]},
    ]
    if aux_head is not None:
        param_groups.append({"params": aux_head.parameters(), "lr": tc["lr"]})

    optimizer = AdamW(param_groups, weight_decay=tc["weight_decay"])
    warmup_epochs = tc.get("warmup_epochs", 0)
    total_epochs = tc.get("epochs", 0)

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Split data
    train_x = torch.from_numpy(dataset.grid_data[train_idx])
    train_y = [dataset.ctc_labels[i] for i in train_idx]
    train_seg = torch.from_numpy(segment_masks[train_idx].astype(np.float32))

    val_x = torch.from_numpy(dataset.grid_data[val_idx])
    val_y = [dataset.ctc_labels[i] for i in val_idx]
    val_seg = torch.from_numpy(segment_masks[val_idx].astype(np.float32))

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    all_params = (
        list(backbone.parameters()) + list(readin.parameters())
        + list(ce_head.parameters())
    )
    if aux_head is not None:
        all_params += list(aux_head.parameters())

    batch_size = tc["batch_size"]
    n_train = len(train_x)

    for epoch in range(total_epochs):
        backbone.train()
        readin.train()
        ce_head.train()
        if aux_head is not None:
            aux_head.train()

        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            x_batch = augment_from_config(train_x[idx], ac, training=True).to(device)
            y_batch = [train_y[i] for i in idx.tolist()]
            seg_batch = train_seg[idx].to(device)

            optimizer.zero_grad()
            h = backbone(readin(x_batch))  # (B, T_down, 2H)
            logits = ce_head(h)  # (B, T_down, 9)
            ce_loss = mfa_guided_ce_loss(logits, y_batch, seg_batch)

            loss = ce_loss
            if aux_head is not None and aux_lambda > 0:
                aux_logits = aux_head(h)
                aux_loss = per_position_feature_bce_loss(
                    aux_logits, y_batch, n_positions=n_positions
                )
                loss = loss + aux_lambda * aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, tc["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validate
        if (epoch + 1) % tc["eval_every"] == 0:
            backbone.eval()
            readin.eval()
            ce_head.eval()
            if aux_head is not None:
                aux_head.eval()

            with torch.no_grad():
                h = backbone(readin(val_x.to(device)))
                logits = ce_head(h)
                val_ce = mfa_guided_ce_loss(logits, val_y, val_seg.to(device))
                val_loss = val_ce
                if aux_head is not None and aux_lambda > 0:
                    val_aux = per_position_feature_bce_loss(
                        aux_head(h), val_y, n_positions=n_positions
                    )
                    val_loss = val_loss + aux_lambda * val_aux

            logger.info(
                "  epoch %d: train_loss=%.4f val_loss=%.4f",
                epoch + 1, avg_train_loss, val_loss.item(),
            )

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                state = {
                    "backbone": deepcopy(backbone.state_dict()),
                    "readin": deepcopy(readin.state_dict()),
                    "ce_head": deepcopy(ce_head.state_dict()),
                }
                if aux_head is not None:
                    state["aux_head"] = deepcopy(aux_head.state_dict())
                best_state = state
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= tc["patience"]:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # Restore best
    if best_state is not None:
        backbone.load_state_dict(best_state["backbone"])
        readin.load_state_dict(best_state["readin"])
        ce_head.load_state_dict(best_state["ce_head"])
        if aux_head is not None and "aux_head" in best_state:
            aux_head.load_state_dict(best_state["aux_head"])

    # Final eval
    backbone.eval()
    readin.eval()
    ce_head.eval()
    with torch.no_grad():
        h = backbone(readin(val_x.to(device)))
        logits = ce_head(h)
        predictions = mfa_guided_ce_decode(logits, val_seg.to(device))
        metrics = evaluate_predictions(predictions, val_y, n_positions=n_positions)
        if aux_head is not None and aux_lambda > 0:
            aux_head.eval()
            metrics.update(
                per_position_feature_metrics(aux_head(h), val_y, n_positions=n_positions)
            )
    return metrics
