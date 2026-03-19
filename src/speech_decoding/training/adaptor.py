"""Stage 2 LOPO target adaptation with source replay."""
from __future__ import annotations

import logging
import math
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from speech_decoding.data.augmentation import augment_from_config
from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.evaluation.metrics import evaluate_predictions
from speech_decoding.models.assembler import assemble_model
import torch.nn as nn

from speech_decoding.training.ctc_utils import (
    blank_ratio,
    ctc_loss,
    greedy_decode,
    per_position_ce_decode,
    per_position_ce_loss,
)

logger = logging.getLogger(__name__)


def _safe_stage2_splits(
    labels: list[int],
    requested_folds: int,
    min_inner_class_count: int = 2,
) -> tuple[int, bool]:
    """Choose a safe outer fold count and whether inner stratification is safe."""
    counts = Counter(labels)
    if not counts:
        raise ValueError("Cannot split empty label set")
    min_count = min(counts.values())
    if min_count < 2:
        outer_folds = 0
        can_inner_stratify = False
    else:
        outer_folds = min(requested_folds, min_count)
        can_inner_stratify = (min_count - 1) >= min_inner_class_count
    return outer_folds, can_inner_stratify


def adapt_stage2(
    checkpoint: dict,
    target_dataset: BIDSDataset,
    source_datasets: dict[str, BIDSDataset],
    config: dict,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Adapt a Stage 1 model to a held-out target patient."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    tc = config["training"]["stage2"]
    requested_folds = tc.get("cv_folds", config["evaluation"]["cv_folds"])
    strat_labels = [y[0] for y in target_dataset.ctc_labels]
    outer_folds, can_inner_stratify = _safe_stage2_splits(
        strat_labels,
        requested_folds=requested_folds,
        min_inner_class_count=tc.get("min_inner_class_count", 2),
    )
    if outer_folds == 0:
        logger.warning(
            "Target %s: insufficient class counts for stratified outer CV; using a single unstratified holdout split",
            target_dataset.patient_id,
        )
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(target_dataset))
        n_test = max(1, int(round(1 / max(requested_folds, 2) * len(target_dataset))))
        test_idx = np.sort(perm[:n_test])
        train_idx = np.sort(perm[n_test:])
        fold_results = [
            _adapt_fold(
                checkpoint,
                target_dataset,
                source_datasets,
                config,
                train_idx,
                test_idx,
                seed,
                device,
                can_inner_stratify=False,
            )
        ]
    else:
        if outer_folds < requested_folds:
            logger.warning(
                "Target %s: reducing Stage 2 cv_folds from %d to %d due to low class counts",
                target_dataset.patient_id, requested_folds, outer_folds,
            )
        skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(np.zeros(len(target_dataset)), strat_labels)
        ):
            logger.info("Stage 2 fold %d/%d", fold_idx + 1, outer_folds)
            result = _adapt_fold(
                checkpoint,
                target_dataset,
                source_datasets,
                config,
                train_idx,
                test_idx,
                seed,
                device,
                can_inner_stratify=can_inner_stratify,
            )
            fold_results.append(result)

    mean_metrics = {}
    for key in fold_results[0]:
        vals = [r[key] for r in fold_results]
        mean_metrics[f"{key}_mean"] = np.mean(vals).item()
        mean_metrics[f"{key}_std"] = np.std(vals).item()
    mean_metrics["fold_results"] = fold_results
    mean_metrics["patient_id"] = target_dataset.patient_id
    mean_metrics["seed"] = seed
    return mean_metrics


def _adapt_fold(
    checkpoint: dict,
    target_dataset: BIDSDataset,
    source_datasets: dict[str, BIDSDataset],
    config: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    device: str,
    can_inner_stratify: bool,
) -> dict:
    """Run one adaptation fold."""
    tc = config["training"]["stage2"]
    ac = tc.get("augmentation", config.get("training", {}).get("augmentation", {}))

    loss_type = config["training"].get("loss_type", "ctc")
    n_segments = config["training"].get("ce_segments", 3)

    target_pid = target_dataset.patient_id
    all_patients = {pid: ds.grid_shape for pid, ds in source_datasets.items()}
    all_patients[target_pid] = target_dataset.grid_shape
    backbone, head, all_read_ins = assemble_model(config, all_patients)

    # Replace head for CE mode
    if loss_type == "ce":
        input_dim = config["model"]["hidden_size"] * 2
        n_phonemes = config["model"]["num_classes"] - 1
        head = nn.Linear(input_dim, n_segments * n_phonemes)

    backbone.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])
    backbone.feat_drop_max = ac.get("feat_dropout_max", 0.2)
    backbone.time_mask_min = ac.get("time_mask_min", 2)
    backbone.time_mask_max = ac.get("time_mask_max", 4)

    backbone = backbone.to(device)
    head = head.to(device)

    source_read_ins = {}
    for pid in source_datasets:
        ri = all_read_ins[pid].to(device)
        ri.load_state_dict(checkpoint["read_ins"][pid])
        ri.eval()
        for param in ri.parameters():
            param.requires_grad = False
        source_read_ins[pid] = ri

    target_read_in = all_read_ins[target_pid].to(device)

    for name, param in backbone.named_parameters():
        param.requires_grad = "layernorm" in name

    target_x = torch.from_numpy(target_dataset.grid_data[train_idx])
    target_y = [target_dataset.ctc_labels[i] for i in train_idx]
    test_x = torch.from_numpy(target_dataset.grid_data[test_idx])
    test_y = [target_dataset.ctc_labels[i] for i in test_idx]

    val_frac = tc.get("val_fraction", 0.2)
    if can_inner_stratify:
        strat = [y[0] for y in target_y]
        n_val = max(1, int(round(val_frac * len(target_y))))
        n_classes = len(set(strat))
        # StratifiedShuffleSplit needs test_size >= n_classes
        if n_val >= n_classes:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            inner_train_idx, inner_val_idx = next(sss.split(np.zeros(len(target_y)), strat))
        else:
            can_inner_stratify = False  # fall through to unstratified
    if not can_inner_stratify:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(target_y))
        n_val = max(1, int(round(val_frac * len(target_y))))
        inner_val_idx = np.sort(perm[:n_val])
        inner_train_idx = np.sort(perm[n_val:])
        logger.warning(
            "Target %s fold has insufficient class counts for inner stratification; using unstratified val split",
            target_dataset.patient_id,
        )

    train_x = target_x[inner_train_idx]
    train_y = [target_y[i] for i in inner_train_idx]
    val_x = target_x[inner_val_idx]
    val_y = [target_y[i] for i in inner_val_idx]

    source_data = {
        pid: {"x": torch.from_numpy(ds.grid_data), "y": ds.ctc_labels}
        for pid, ds in source_datasets.items()
    }
    source_pids = list(source_data)

    optimizer = AdamW(
        [
            {
                "params": target_read_in.parameters(),
                "lr": tc["lr"] * tc.get("readin_lr_mult", 1.0),
            },
            {
                "params": [p for p in backbone.parameters() if p.requires_grad],
                "lr": tc["lr"],
            },
            {"params": head.parameters(), "lr": tc["lr"]},
        ],
        weight_decay=tc["weight_decay"],
    )

    warmup_steps = tc.get("warmup_epochs", 0)
    total_steps = tc.get("steps", tc.get("epochs", 0))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    B = tc["batch_size"]
    replay_frac = tc.get("source_replay_frac", 0.3)
    n_target_batch = max(int((1 - replay_frac) * B), 1)
    n_source_batch = B - n_target_batch
    all_trainable = [
        p
        for p in list(target_read_in.parameters()) + list(backbone.parameters()) + list(head.parameters())
        if p.requires_grad
    ]

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for step in range(total_steps):
        backbone.train()
        head.train()
        target_read_in.train()
        optimizer.zero_grad()

        tgt_idx = torch.randint(0, len(train_x), (n_target_batch,))
        x_tgt = augment_from_config(train_x[tgt_idx], ac, training=True).to(device)
        y_tgt = [train_y[i] for i in tgt_idx.tolist()]

        shared_tgt = target_read_in(x_tgt)
        h_tgt = backbone(shared_tgt)
        lp_tgt = head(h_tgt)
        if loss_type == "ce":
            loss_tgt = per_position_ce_loss(lp_tgt, y_tgt, n_segments)
        else:
            loss_tgt = ctc_loss(lp_tgt, y_tgt)

        loss_src = torch.tensor(0.0, device=device)
        if n_source_batch > 0 and source_pids:
            src_pid = source_pids[step % len(source_pids)]
            sd = source_data[src_pid]
            src_idx = torch.randint(0, len(sd["x"]), (n_source_batch,))
            x_src = sd["x"][src_idx].to(device)
            y_src = [sd["y"][i] for i in src_idx.tolist()]
            with torch.no_grad():
                shared_src = source_read_ins[src_pid](x_src)
            h_src = backbone(shared_src)
            lp_src = head(h_src)
            if loss_type == "ce":
                loss_src = per_position_ce_loss(lp_src, y_src, n_segments)
            else:
                loss_src = ctc_loss(lp_src, y_src)

        loss = (1 - replay_frac) * loss_tgt + replay_frac * loss_src
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_trainable, tc["grad_clip"])
        optimizer.step()
        scheduler.step()

        if (step + 1) % tc.get("eval_every", 1) == 0:
            backbone.eval()
            head.eval()
            target_read_in.eval()
            with torch.no_grad():
                vx = val_x.to(device)
                shared = target_read_in(vx)
                h = backbone(shared)
                lp = head(h)
                if loss_type == "ce":
                    val_loss = per_position_ce_loss(lp, val_y, n_segments).item()
                else:
                    val_loss = ctc_loss(lp, val_y).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "target_read_in": deepcopy(target_read_in.state_dict()),
                    "head": deepcopy(head.state_dict()),
                    "backbone": deepcopy(backbone.state_dict()),
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= tc["patience"]:
                logger.info("Stage 2 early stopping at step %d", step + 1)
                break

    if best_state is not None:
        target_read_in.load_state_dict(best_state["target_read_in"])
        head.load_state_dict(best_state["head"])
        backbone.load_state_dict(best_state["backbone"])

    backbone.eval()
    head.eval()
    target_read_in.eval()
    with torch.no_grad():
        tx = test_x.to(device)
        shared = target_read_in(tx)
        h = backbone(shared)
        lp = head(h)
        if loss_type == "ce":
            predictions = per_position_ce_decode(lp, n_segments)
            br = 0.0
        else:
            predictions = greedy_decode(lp)
            br = blank_ratio(lp)

    metrics = evaluate_predictions(predictions, test_y, n_positions=3)
    metrics["blank_ratio"] = br
    return metrics
