"""Orchestrate one `(patient, fold, seed, depth)` training+eval run.

Pure function — no path resolution, no argparse. The CLI
(`scripts/v14_core/train_v14_core.py`) builds the dataset and passes it here.
Phase F ablation drivers reuse the same entry point with different `V14Config`
overrides.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from speech_decoding.v14.config import V14Config
from speech_decoding.v14.cv import make_outer_folds, make_val_split
from speech_decoding.v14.dataset import collate_v14_batch
from speech_decoding.v14.model import NeuralFieldPerceiver
from speech_decoding.v14.train import TRIALS_PER_BATCH, evaluate, train_one_fold


OUTER_FOLD_SEED: int = 0
VAL_SPLIT_SEED: int = 7


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_config(grid_mixer_depth: int) -> V14Config:
    cfg = V14Config()
    return replace(cfg, grid_mixer=replace(cfg.grid_mixer, num_layers=grid_mixer_depth))


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_one_fold(
    dataset,  # V14TrialDataset or protocol-compatible: .tokens, __len__, __getitem__
    *,
    fold_idx: int,
    seed: int,
    grid_mixer_depth: int,
    out_dir: Path,
    patient_id: str,
    max_epochs: int | None = None,
    val_every: int | None = None,
    patience: int | None = None,
) -> dict:
    """Train one fold at one seed, evaluate on the held-out fold, return a result row.

    Fold assignment uses a *fixed* outer-fold seed and val-split seed per the
    First-Run Protocol so all three training seeds of a given outer fold see
    the same train/val/test partition.
    """

    if grid_mixer_depth < 1:
        raise ValueError(f"grid_mixer_depth must be >= 1, got {grid_mixer_depth}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{patient_id}_fold{fold_idx}_seed{seed}_depth{grid_mixer_depth}"
    log_path = out_dir / f"{tag}.log.jsonl"
    result_path = out_dir / f"{tag}.result.json"

    tokens = dataset.tokens
    folds = make_outer_folds(tokens, n_folds=5, seed=OUTER_FOLD_SEED)
    train_idx, test_idx = folds[fold_idx]
    train_idx, val_idx = make_val_split(
        tokens, train_idx, val_frac=0.2, seed=VAL_SPLIT_SEED
    )

    set_seed(seed)
    device = _select_device()
    cfg = _build_config(grid_mixer_depth)
    model = NeuralFieldPerceiver(cfg).to(device)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=TRIALS_PER_BATCH,
        shuffle=True,
        collate_fn=collate_v14_batch,
        generator=g,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=TRIALS_PER_BATCH,
        shuffle=False,
        collate_fn=collate_v14_batch,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=TRIALS_PER_BATCH,
        shuffle=False,
        collate_fn=collate_v14_batch,
    )

    kw: dict = {}
    if max_epochs is not None:
        kw["max_epochs"] = max_epochs
    if val_every is not None:
        kw["val_every"] = val_every
    if patience is not None:
        kw["patience"] = patience

    with log_path.open("w") as f:
        def _log(row: dict) -> None:
            f.write(json.dumps(row) + "\n")
            f.flush()

        fold_result = train_one_fold(
            model, train_loader, val_loader, log_callback=_log, **kw
        )

    test_per = evaluate(model, test_loader)

    result = {
        "patient": patient_id,
        "fold": fold_idx,
        "seed": seed,
        "grid_mixer_depth": grid_mixer_depth,
        "best_val_per": fold_result.best_val_per,
        "test_per": test_per,
        "final_epoch": fold_result.final_epoch,
        "early_stopped": fold_result.early_stopped,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "device": str(device),
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result
