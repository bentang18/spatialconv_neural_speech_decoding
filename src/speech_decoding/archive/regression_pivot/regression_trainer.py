"""Per-patient trainer with joint CE + masked MSE regression loss."""
from __future__ import annotations

import logging
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from speech_decoding.data.augmentation import augment_from_config
from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.evaluation.metrics import (
    evaluate_predictions,
    framewise_r2_diagnostics,
    segment_r2_diagnostics,
)
from speech_decoding.models.assembler import assemble_model
from speech_decoding.models.regression_head import RegressionHead
from speech_decoding.training.ctc_utils import per_position_ce_decode, per_position_ce_loss
from speech_decoding.training.regression_loss import masked_mse_loss, segment_mse_loss

logger = logging.getLogger(__name__)


def train_per_patient_regression(
    dataset: BIDSDataset,
    embeddings: np.ndarray,
    speech_mask: np.ndarray,
    config: dict,
    seed: int = 42,
    device: str = "cpu",
    segment_mask: np.ndarray | None = None,
) -> dict:
    """Train and evaluate CE + regression on one patient with CV."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if len(dataset) != len(embeddings) or len(dataset) != len(speech_mask):
        raise ValueError(
            "Dataset, embeddings, and speech_mask must have matching trial counts."
        )
    target_mode = config["training"].get("regression_target_mode", "frame")
    if target_mode == "segment":
        if segment_mask is None:
            raise ValueError("segment_mask is required when regression_target_mode='segment'.")
        if len(dataset) != len(segment_mask):
            raise ValueError("Dataset and segment_mask must have matching trial counts.")

    n_folds = config["evaluation"]["cv_folds"]
    strat_labels = [y[0] for y in dataset.ctc_labels]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), strat_labels)):
        logger.info("Regression fold %d/%d", fold_idx + 1, n_folds)
        fold_results.append(
            _train_regression_fold(
                dataset=dataset,
                embeddings=embeddings,
                speech_mask=speech_mask,
                config=config,
                train_idx=train_idx,
                val_idx=val_idx,
                seed=seed,
                device=device,
                segment_mask=segment_mask,
            )
        )

    mean_metrics = _aggregate_fold_results(fold_results)
    mean_metrics["fold_results"] = fold_results
    mean_metrics["patient_id"] = dataset.patient_id
    mean_metrics["seed"] = seed
    return mean_metrics


def _aggregate_fold_results(fold_results: list[dict]) -> dict:
    mean_metrics: dict[str, float | list[dict]] = {}
    keys = sorted({k for result in fold_results for k in result})
    for key in keys:
        vals = [result[key] for result in fold_results if key in result]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        mean_metrics[f"{key}_mean"] = np.nanmean(arr).item()
        mean_metrics[f"{key}_std"] = np.nanstd(arr).item()
    return mean_metrics


def _fit_fold_pca(
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    d_emb: int,
    train_mask: np.ndarray | None = None,
    pca_scope: str = "all_frames",
) -> tuple[np.ndarray, np.ndarray, int]:
    flat_train_all = train_embeddings.reshape(-1, train_embeddings.shape[-1])
    flat_val = val_embeddings.reshape(-1, val_embeddings.shape[-1])
    if pca_scope == "speech_only" and train_mask is not None:
        selector = train_mask.reshape(-1) > 0.5
        flat_train = flat_train_all[selector]
        if len(flat_train) == 0:
            flat_train = flat_train_all
    else:
        flat_train = flat_train_all
    n_components = min(d_emb, flat_train.shape[0], flat_train.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(flat_train)
    train_red = pca.transform(flat_train_all).reshape(
        train_embeddings.shape[0], train_embeddings.shape[1], n_components
    )
    val_red = pca.transform(flat_val).reshape(val_embeddings.shape[0], val_embeddings.shape[1], n_components)
    return train_red.astype(np.float32), val_red.astype(np.float32), n_components


def _train_regression_fold(
    dataset: BIDSDataset,
    embeddings: np.ndarray,
    speech_mask: np.ndarray,
    config: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    device: str,
    segment_mask: np.ndarray | None = None,
) -> dict:
    tc = config["training"]["stage1"]
    ac = config["training"].get("augmentation", config["training"].get("stage1", {}).get("augmentation", {}))
    n_segments = config["training"].get("ce_segments", 3)
    reg_lambda = config["training"].get("regression_lambda", 0.3)
    d_emb = config["model"].get("d_emb", 64)
    target_mode = config["training"].get("regression_target_mode", "frame")
    pca_scope = config["training"].get("regression_pca_scope", "all_frames")

    patients = {dataset.patient_id: dataset.grid_shape}
    backbone, _, readins = assemble_model(config, patients)
    readin = readins[dataset.patient_id]
    input_dim = config["model"]["hidden_size"] * 2
    n_phonemes = config["model"]["num_classes"] - 1
    ce_head = nn.Linear(input_dim, n_segments * n_phonemes)

    train_emb, val_emb, d_emb_used = _fit_fold_pca(
        embeddings[train_idx],
        embeddings[val_idx],
        d_emb,
        train_mask=speech_mask[train_idx],
        pca_scope=pca_scope,
    )
    reg_head = RegressionHead(input_dim=input_dim, output_dim=d_emb_used)

    backbone = backbone.to(device)
    readin = readin.to(device)
    ce_head = ce_head.to(device)
    reg_head = reg_head.to(device)

    optimizer = AdamW(
        [
            {"params": readin.parameters(), "lr": tc["lr"] * tc["readin_lr_mult"]},
            {"params": backbone.parameters(), "lr": tc["lr"]},
            {"params": ce_head.parameters(), "lr": tc["lr"]},
            {"params": reg_head.parameters(), "lr": tc["lr"]},
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
    train_emb_t = torch.from_numpy(train_emb)
    train_mask = torch.from_numpy(speech_mask[train_idx].astype(np.float32))
    train_seg_mask = None
    val_seg_mask = None
    train_seg_target = None
    val_seg_target = None

    val_x = torch.from_numpy(dataset.grid_data[val_idx])
    val_y = [dataset.ctc_labels[i] for i in val_idx]
    val_emb_t = torch.from_numpy(val_emb)
    val_mask = torch.from_numpy(speech_mask[val_idx].astype(np.float32))

    if target_mode == "segment":
        if segment_mask is None:
            raise ValueError("segment_mask is required for segment target mode.")
        train_seg_mask = torch.from_numpy(segment_mask[train_idx].astype(np.float32))
        val_seg_mask = torch.from_numpy(segment_mask[val_idx].astype(np.float32))
        train_seg_target = torch.einsum(
            "ntd,nst->nsd",
            torch.from_numpy(train_emb),
            train_seg_mask,
        ) / train_seg_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        val_seg_target = torch.einsum(
            "ntd,nst->nsd",
            torch.from_numpy(val_emb),
            val_seg_mask,
        ) / val_seg_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    all_params = list(backbone.parameters()) + list(readin.parameters()) + list(ce_head.parameters()) + list(reg_head.parameters())

    batch_size = tc["batch_size"]
    n_train = len(train_x)

    for epoch in range(tc["epochs"]):
        backbone.train()
        readin.train()
        ce_head.train()
        reg_head.train()

        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            x_batch = augment_from_config(train_x[idx], ac, training=True).to(device)
            y_batch = [train_y[i] for i in idx.tolist()]
            emb_batch = train_emb_t[idx].to(device)
            mask_batch = train_mask[idx].to(device)
            seg_mask_batch = train_seg_mask[idx].to(device) if train_seg_mask is not None else None
            seg_target_batch = train_seg_target[idx].to(device) if train_seg_target is not None else None

            optimizer.zero_grad()
            shared = readin(x_batch)
            h = backbone(shared)
            ce_logits = ce_head(h)
            ce_loss = per_position_ce_loss(ce_logits, y_batch, n_segments)

            if reg_lambda > 0:
                reg_out = reg_head(h)
                if target_mode == "segment":
                    mse_loss = segment_mse_loss(reg_out, seg_target_batch, seg_mask_batch)
                else:
                    mse_loss = masked_mse_loss(reg_out, emb_batch, mask_batch)
                loss = ce_loss + reg_lambda * mse_loss
            else:
                mse_loss = ce_loss.new_tensor(0.0)
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
            reg_head.eval()
            with torch.no_grad():
                val_x_dev = val_x.to(device)
                h = backbone(readin(val_x_dev))
                val_ce_logits = ce_head(h)
                val_ce = per_position_ce_loss(val_ce_logits, val_y, n_segments)
                if reg_lambda > 0:
                    val_reg = reg_head(h)
                    if target_mode == "segment":
                        val_mse = segment_mse_loss(
                            val_reg,
                            val_seg_target.to(device),
                            val_seg_mask.to(device),
                        )
                    else:
                        val_mse = masked_mse_loss(val_reg, val_emb_t.to(device), val_mask.to(device))
                    val_loss = val_ce + reg_lambda * val_mse
                    logger.info(
                        "  epoch %d: train_loss=%.4f val_ce=%.4f val_mse=%.4f val_loss=%.4f",
                        epoch + 1, avg_train_loss, val_ce.item(), val_mse.item(), val_loss.item(),
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
                    "reg_head": deepcopy(reg_head.state_dict()),
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
        reg_head.load_state_dict(best_state["reg_head"])

    backbone.eval()
    readin.eval()
    ce_head.eval()
    reg_head.eval()
    with torch.no_grad():
        val_x_dev = val_x.to(device)
        h = backbone(readin(val_x_dev))
        ce_logits = ce_head(h)
        predictions = per_position_ce_decode(ce_logits, n_segments)
        metrics = evaluate_predictions(predictions, val_y, n_positions=n_segments)
        if reg_lambda > 0:
            reg_pred = reg_head(h).cpu().numpy()
            if target_mode == "segment":
                pred_seg = (
                    np.einsum("ntd,nst->nsd", reg_pred, segment_mask[val_idx])
                    / np.clip(segment_mask[val_idx].sum(axis=-1, keepdims=True), 1.0, None)
                )
                target_seg = val_seg_target.numpy()
                metrics.update(segment_r2_diagnostics(pred_seg, target_seg))
                metrics["masked_mse"] = segment_mse_loss(
                    torch.from_numpy(reg_pred),
                    val_seg_target,
                    val_seg_mask,
                ).item()
            else:
                metrics.update(framewise_r2_diagnostics(reg_pred, val_emb, speech_mask[val_idx]))
                metrics["masked_mse"] = masked_mse_loss(
                    torch.from_numpy(reg_pred),
                    torch.from_numpy(val_emb),
                    torch.from_numpy(speech_mask[val_idx].astype(np.float32)),
                ).item()
        metrics["regression_lambda"] = reg_lambda
        metrics["d_emb_used"] = float(d_emb_used)
        metrics["target_mode"] = 0.0 if target_mode == "frame" else 1.0
    return metrics
