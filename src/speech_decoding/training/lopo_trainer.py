"""Stage 1 multi-patient LOPO training."""
from __future__ import annotations

import logging
import math
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from speech_decoding.data.augmentation import augment_from_config
from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.models.assembler import assemble_model
import torch.nn as nn
import torch.nn.functional as F

from speech_decoding.training.ctc_utils import ctc_loss, per_position_ce_loss

logger = logging.getLogger(__name__)


def _safe_stage1_val_split(
    labels: list[int],
    val_fraction: float,
) -> bool:
    """Return whether a stratified Stage 1 train/val split is safe."""
    counts = Counter(labels)
    if not counts:
        return False
    min_count = min(counts.values())
    n_val = max(1, int(round(val_fraction * len(labels))))
    # StratifiedShuffleSplit needs at least 2 samples in each class, and the
    # validation split must be able to contain at least one example per class.
    return min_count >= 2 and n_val >= len(counts)


def train_stage1(
    source_datasets: dict[str, BIDSDataset],
    config: dict,
    seed: int = 42,
    device: str = "cpu",
    backbone_init: dict | None = None,
    head_init: dict | None = None,
) -> dict:
    """Train shared backbone/head with per-patient read-ins on source patients.

    Args:
        backbone_init: Optional state_dict to warm-start backbone weights
            (e.g. from per-patient pre-training).
        head_init: Optional state_dict to warm-start head weights.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tc = config["training"]["stage1"]
    ac = tc.get("augmentation", config.get("training", {}).get("augmentation", {}))

    loss_type = config["training"].get("loss_type", "ctc")
    n_segments = config["training"].get("ce_segments", 3)

    patients = {pid: ds.grid_shape for pid, ds in source_datasets.items()}
    backbone, head, read_ins = assemble_model(config, patients)

    # Replace head for CE mode
    if loss_type == "ce":
        input_dim = config["model"]["hidden_size"] * 2
        n_phonemes = config["model"]["num_classes"] - 1
        head = nn.Linear(input_dim, n_segments * n_phonemes)

    # Warm-start backbone/head from pre-trained weights
    if backbone_init is not None:
        backbone.load_state_dict(backbone_init)
        logger.info("Backbone warm-started from pre-trained weights")
    if head_init is not None:
        head.load_state_dict(head_init)
        logger.info("Head warm-started from pre-trained weights")

    backbone.feat_drop_max = ac.get("feat_dropout_max", 0.3)
    backbone.time_mask_min = ac.get("time_mask_min", 2)
    backbone.time_mask_max = ac.get("time_mask_max", 4)

    backbone = backbone.to(device)
    head = head.to(device)
    for pid in read_ins:
        read_ins[pid] = read_ins[pid].to(device)

    train_data: dict[str, dict[str, object]] = {}
    val_data: dict[str, dict[str, object]] = {}
    val_fraction = tc.get("val_fraction", 0.2)
    for pid, ds in source_datasets.items():
        strat_labels = [y[0] for y in ds.ctc_labels]
        if _safe_stage1_val_split(strat_labels, val_fraction):
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_fraction,
                random_state=seed,
            )
            train_idx, val_idx = next(sss.split(np.zeros(len(ds)), strat_labels))
        else:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(ds))
            n_val = max(1, int(round(val_fraction * len(ds))))
            val_idx = np.sort(perm[:n_val])
            train_idx = np.sort(perm[n_val:])
            logger.warning(
                "Source %s has insufficient class counts for stratified Stage 1 val split; using unstratified split",
                pid,
            )
        train_data[pid] = {
            "x": torch.from_numpy(ds.grid_data[train_idx]),
            "y": [ds.ctc_labels[i] for i in train_idx],
        }
        val_data[pid] = {
            "x": torch.from_numpy(ds.grid_data[val_idx]),
            "y": [ds.ctc_labels[i] for i in val_idx],
        }

    param_groups = [
        {"params": read_ins[pid].parameters(), "lr": tc["lr"] * tc["readin_lr_mult"]}
        for pid in read_ins
    ]
    param_groups.append({"params": backbone.parameters(), "lr": tc["lr"]})
    param_groups.append({"params": head.parameters(), "lr": tc["lr"]})
    optimizer = AdamW(param_groups, weight_decay=tc["weight_decay"])

    warmup_steps = tc.get("warmup_epochs", 0)
    total_steps = tc.get("steps", tc.get("epochs", 0))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    all_params = []
    for pid in read_ins:
        all_params.extend(read_ins[pid].parameters())
    all_params.extend(backbone.parameters())
    all_params.extend(head.parameters())

    B = tc["batch_size"]
    n_source = len(source_datasets)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    train_losses: list[float] = []
    val_losses: list[float] = []

    for step in range(total_steps):
        backbone.train()
        head.train()
        for pid in read_ins:
            read_ins[pid].train()

        optimizer.zero_grad()
        step_loss = 0.0

        for pid in source_datasets:
            td = train_data[pid]
            x_cpu = td["x"]
            y_all = td["y"]
            idx = torch.randint(0, len(x_cpu), (B,))
            x_batch = x_cpu[idx]
            y_batch = [y_all[i] for i in idx.tolist()]

            x_batch = augment_from_config(x_batch, ac, training=True).to(device)
            shared = read_ins[pid](x_batch)
            h = backbone(shared)
            out = head(h)
            if loss_type == "ce":
                loss = per_position_ce_loss(out, y_batch, n_segments) / n_source
            else:
                loss = ctc_loss(out, y_batch) / n_source
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(all_params, tc["grad_clip"])
        optimizer.step()
        scheduler.step()
        train_losses.append(step_loss)

        if (step + 1) % tc["eval_every"] == 0:
            backbone.eval()
            head.eval()
            for pid in read_ins:
                read_ins[pid].eval()

            val_loss = 0.0
            with torch.no_grad():
                for pid in source_datasets:
                    vd = val_data[pid]
                    vx = vd["x"].to(device)
                    shared = read_ins[pid](vx)
                    h = backbone(shared)
                    out = head(h)
                    if loss_type == "ce":
                        val_loss += per_position_ce_loss(out, vd["y"], n_segments).item() / n_source
                    else:
                        val_loss += ctc_loss(out, vd["y"]).item() / n_source

            val_losses.append(val_loss)
            logger.info(
                "Stage 1 step %d: train_loss=%.4f val_loss=%.4f",
                step + 1, step_loss, val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "backbone": deepcopy(backbone.state_dict()),
                    "head": deepcopy(head.state_dict()),
                    "read_ins": {
                        pid: deepcopy(ri.state_dict()) for pid, ri in read_ins.items()
                    },
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= tc["patience"]:
                logger.info("Stage 1 early stopping at step %d", step + 1)
                break

    if best_state is not None:
        backbone.load_state_dict(best_state["backbone"])
        head.load_state_dict(best_state["head"])
        for pid in read_ins:
            read_ins[pid].load_state_dict(best_state["read_ins"][pid])

    return {
        "backbone": deepcopy(backbone.state_dict()),
        "head": deepcopy(head.state_dict()),
        "read_ins": {pid: deepcopy(ri.state_dict()) for pid, ri in read_ins.items()},
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
