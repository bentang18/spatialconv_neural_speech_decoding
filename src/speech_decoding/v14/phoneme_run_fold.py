"""Per-phoneme fold runner (plan P6).

Analogue of `run_fold.run_one_fold` for the baseline-aligned per-phoneme path.
Builds token-disjoint folds on trial tokens (via the phoneme dataset's
`trial_tokens()` helper), then expands each trial index into the 3 phoneme
indices `{3t, 3t+1, 3t+2}` that index into `V14PhonemeDataset`.

Emits per-phoneme PER and exhaustive-AR slot-averaged PER on the held-out
fold, in the same result-row shape the existing run-fold driver emits.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from speech_decoding.v14.config import PerPhonemeConfig
from speech_decoding.v14.cv import make_outer_folds, make_val_split
from speech_decoding.v14.eval import (
    evaluate_per_phoneme,
    exhaustive_ar_per_from_loader,
)
from speech_decoding.v14.phoneme_dataset import (
    V14PhonemeDataset,
    collate_v14_phoneme_batch,
)
from speech_decoding.v14.phoneme_model import NeuralFieldPerceiverPerPhoneme
from speech_decoding.v14.train import (
    TRIALS_PER_BATCH,
    per_phoneme_ce_loss,
    train_one_fold,
)


OUTER_FOLD_SEED: int = 0
VAL_SPLIT_SEED: int = 7


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_config(backbone_depth: int) -> PerPhonemeConfig:
    """Depth = number of backbone blocks. Default in the plan is 3; the P8
    array sweeps depth ∈ {1, 3} as the two cheapest ablation rungs."""

    cfg = PerPhonemeConfig()
    return replace(cfg, backbone=replace(cfg.backbone, num_blocks=backbone_depth))


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _expand_trial_indices_to_phoneme_indices(trial_idx: list[int]) -> list[int]:
    out: list[int] = []
    for t in trial_idx:
        out.extend([3 * t, 3 * t + 1, 3 * t + 2])
    return out


def run_one_fold(
    dataset: V14PhonemeDataset,
    *,
    fold_idx: int,
    seed: int,
    backbone_depth: int,
    out_dir: Path,
    patient_id: str,
    max_epochs: int | None = None,
    val_every: int | None = None,
    patience: int | None = None,
    warmup_epochs: int | None = None,
) -> dict:
    """Train + eval one `(fold, seed, depth)` per-phoneme run."""

    if backbone_depth < 1:
        raise ValueError(f"backbone_depth must be >= 1, got {backbone_depth}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{patient_id}_fold{fold_idx}_seed{seed}_depth{backbone_depth}"
    log_path = out_dir / f"{tag}.log.jsonl"
    result_path = out_dir / f"{tag}.result.json"

    trial_tokens = dataset.trial_tokens()
    folds = make_outer_folds(trial_tokens, n_folds=5, seed=OUTER_FOLD_SEED)
    train_trial_idx, test_trial_idx = folds[fold_idx]
    train_trial_idx, val_trial_idx = make_val_split(
        trial_tokens, train_trial_idx, val_frac=0.2, seed=VAL_SPLIT_SEED
    )

    train_idx = _expand_trial_indices_to_phoneme_indices(train_trial_idx)
    val_idx = _expand_trial_indices_to_phoneme_indices(val_trial_idx)
    test_idx = _expand_trial_indices_to_phoneme_indices(test_trial_idx)

    set_seed(seed)
    device = _select_device()
    cfg = _build_config(backbone_depth)
    model = NeuralFieldPerceiverPerPhoneme(cfg).to(device)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=TRIALS_PER_BATCH,
        shuffle=True,
        collate_fn=collate_v14_phoneme_batch,
        generator=g,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=TRIALS_PER_BATCH,
        shuffle=False,
        collate_fn=collate_v14_phoneme_batch,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=TRIALS_PER_BATCH,
        shuffle=False,
        collate_fn=collate_v14_phoneme_batch,
    )

    kw: dict = {
        "loss_fn": per_phoneme_ce_loss,
        "evaluate_fn": evaluate_per_phoneme,
    }
    if max_epochs is not None:
        kw["max_epochs"] = max_epochs
    if val_every is not None:
        kw["val_every"] = val_every
    if patience is not None:
        kw["patience"] = patience
    if warmup_epochs is not None:
        kw["warmup_epochs"] = warmup_epochs

    with log_path.open("w") as f:
        def _log(row: dict) -> None:
            f.write(json.dumps(row) + "\n")
            f.flush()

        fold_result = train_one_fold(
            model, train_loader, val_loader, log_callback=_log, **kw
        )

    test_per_phoneme = evaluate_per_phoneme(model, test_loader)
    test_slot_per = exhaustive_ar_per_from_loader(model, test_loader)

    result = {
        "patient": patient_id,
        "fold": fold_idx,
        "seed": seed,
        "backbone_depth": backbone_depth,
        "best_val_per_phoneme": fold_result.best_val_per,
        "test_per_phoneme": test_per_phoneme,
        "test_slot_averaged_per": test_slot_per,
        "final_epoch": fold_result.final_epoch,
        "early_stopped": fold_result.early_stopped,
        "n_train_trials": len(train_trial_idx),
        "n_val_trials": len(val_trial_idx),
        "n_test_trials": len(test_trial_idx),
        "device": str(device),
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result
