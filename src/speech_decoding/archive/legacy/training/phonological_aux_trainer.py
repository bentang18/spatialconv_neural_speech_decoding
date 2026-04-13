"""Per-patient trainer with CE + phonological auxiliary loss."""
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
from speech_decoding.training.ctc_utils import per_position_ce_decode, per_position_ce_loss
from speech_decoding.training.phonological_aux import (
    per_position_feature_bce_loss,
    per_position_feature_metrics,
)

logger = logging.getLogger(__name__)


def train_per_patient_phonological_aux(
    dataset: BIDSDataset,
    config: dict,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Train a CE model with an auxiliary phonological feature loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_folds = config["evaluation"]["cv_folds"]
    strat_labels = [y[0] for y in dataset.ctc_labels]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), strat_labels)):
        logger.info("Aux fold %d/%d", fold_idx + 1, n_folds)
        fold_results.append(
            _train_aux_fold(dataset, config, train_idx, val_idx, seed, device)
        )

    summary = _aggregate_fold_results(fold_results)
    summary["fold_results"] = fold_results
    summary["patient_id"] = dataset.patient_id
    summary["seed"] = seed
    return summary


def _aggregate_fold_results(fold_results: list[dict]) -> dict:
    result: dict[str, float] = {}
    keys = sorted({key for fold in fold_results for key in fold})
    for key in keys:
        vals = [fold[key] for fold in fold_results if key in fold]
        arr = np.asarray(vals, dtype=float)
        result[f"{key}_mean"] = np.nanmean(arr).item()
        result[f"{key}_std"] = np.nanstd(arr).item()
    return result


def _train_aux_fold(
    dataset: BIDSDataset,
    config: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    device: str,
) -> dict:
    tc = config["training"]["stage1"]
    ac = config["training"].get("augmentation", config["training"].get("stage1", {}).get("augmentation", {}))
    n_segments = config["training"].get("ce_segments", 3)
    aux_lambda = config["training"].get("phonological_aux_lambda", 0.3)
    n_features = config["training"].get("phonological_num_features", 15)

    patients = {dataset.patient_id: dataset.grid_shape}
    backbone, _, readins = assemble_model(config, patients)
    readin = readins[dataset.patient_id]
    input_dim = config["model"]["hidden_size"] * 2
    n_phonemes = config["model"]["num_classes"] - 1
    ce_head = nn.Linear(input_dim, n_segments * n_phonemes)
    aux_head = nn.Linear(input_dim, n_segments * n_features)

    backbone = backbone.to(device)
    readin = readin.to(device)
    ce_head = ce_head.to(device)
    aux_head = aux_head.to(device)

    optimizer = AdamW(
        [
            {"params": readin.parameters(), "lr": tc["lr"] * tc["readin_lr_mult"]},
            {"params": backbone.parameters(), "lr": tc["lr"]},
            {"params": ce_head.parameters(), "lr": tc["lr"]},
            {"params": aux_head.parameters(), "lr": tc["lr"]},
        ],
        weight_decay=tc["weight_decay"],
    )
    warmup_epochs = tc.get("warmup_epochs", 0)
    total_epochs = tc.get("epochs", tc.get("steps", 0))

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    train_x = torch.from_numpy(dataset.grid_data[train_idx])
    train_y = [dataset.ctc_labels[i] for i in train_idx]
    val_x = torch.from_numpy(dataset.grid_data[val_idx])
    val_y = [dataset.ctc_labels[i] for i in val_idx]

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    all_params = list(backbone.parameters()) + list(readin.parameters()) + list(ce_head.parameters()) + list(aux_head.parameters())

    batch_size = tc["batch_size"]
    n_train = len(train_x)

    for epoch in range(tc["epochs"]):
        backbone.train()
        readin.train()
        ce_head.train()
        aux_head.train()

        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            x_batch = augment_from_config(train_x[idx], ac, training=True).to(device)
            y_batch = [train_y[i] for i in idx.tolist()]

            optimizer.zero_grad()
            h = backbone(readin(x_batch))
            ce_logits = ce_head(h)
            ce_loss = per_position_ce_loss(ce_logits, y_batch, n_segments)
            if aux_lambda > 0:
                aux_logits = aux_head(h)
                aux_loss = per_position_feature_bce_loss(aux_logits, y_batch, n_positions=n_segments)
                loss = ce_loss + aux_lambda * aux_loss
            else:
                aux_loss = ce_loss.new_tensor(0.0)
                loss = ce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, tc["grad_clip"])
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % tc["eval_every"] == 0:
            backbone.eval()
            readin.eval()
            ce_head.eval()
            aux_head.eval()
            with torch.no_grad():
                h = backbone(readin(val_x.to(device)))
                val_ce_logits = ce_head(h)
                val_ce = per_position_ce_loss(val_ce_logits, val_y, n_segments)
                if aux_lambda > 0:
                    val_aux_logits = aux_head(h)
                    val_aux = per_position_feature_bce_loss(val_aux_logits, val_y, n_positions=n_segments)
                    val_loss = val_ce + aux_lambda * val_aux
                    logger.info(
                        "  epoch %d: train_loss=%.4f val_ce=%.4f val_aux=%.4f val_loss=%.4f",
                        epoch + 1, avg_train_loss, val_ce.item(), val_aux.item(), val_loss.item(),
                    )
                else:
                    val_loss = val_ce
                    logger.info(
                        "  epoch %d: train_loss=%.4f val_ce=%.4f",
                        epoch + 1, avg_train_loss, val_ce.item(),
                    )

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {
                    "backbone": deepcopy(backbone.state_dict()),
                    "readin": deepcopy(readin.state_dict()),
                    "ce_head": deepcopy(ce_head.state_dict()),
                    "aux_head": deepcopy(aux_head.state_dict()),
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= tc["patience"]:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        backbone.load_state_dict(best_state["backbone"])
        readin.load_state_dict(best_state["readin"])
        ce_head.load_state_dict(best_state["ce_head"])
        aux_head.load_state_dict(best_state["aux_head"])

    backbone.eval()
    readin.eval()
    ce_head.eval()
    aux_head.eval()
    with torch.no_grad():
        h = backbone(readin(val_x.to(device)))
        ce_logits = ce_head(h)
        predictions = per_position_ce_decode(ce_logits, n_segments)
        metrics = evaluate_predictions(predictions, val_y, n_positions=n_segments)
        metrics["aux_lambda"] = aux_lambda
        if aux_lambda > 0:
            metrics.update(per_position_feature_metrics(aux_head(h), val_y, n_positions=n_segments))
    return metrics
