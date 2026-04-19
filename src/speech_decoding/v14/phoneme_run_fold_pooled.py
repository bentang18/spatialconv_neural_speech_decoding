"""Pooled multi-patient fold runner (Q1a — pooled held-in).

One model is trained on a pool of patients with the same `#31` `same-patient-
per-batch` invariant: each optimizer step sees one micro-batch from each
patient, enforced by a round-robin interleave over per-patient `DataLoader`s.
`GRAD_ACCUM_STEPS` is overridden to ``n_patients`` so effective batch size is
``n_patients · TRIALS_PER_BATCH`` and every patient contributes equal gradient
weight per step, independent of trial count. Smaller patients' loaders wrap
and reshuffle so the number of cycles per epoch matches the largest patient.

CV: `make_outer_folds_pooled` — same token→fold partition across all
patients. `make_val_split_pooled` — same global val tokens.

Eval: per-patient PER on each patient's test fold (argmax + exhaustive AR),
aggregated as a population mean ± std across patients.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from speech_decoding.v14.config import PerPhonemeConfig
from speech_decoding.v14.cv import make_outer_folds_pooled, make_val_split_pooled
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_config(
    backbone_depth: int,
    d_model: int = 32,
    conv2d_kernel: int = 3,
    pool_shape: tuple[int, int] = (4, 8),
    use_parcel_embedding: bool = True,
    temporal_frontend: str = "per_cell",
    pool_method: str = "masked_mean",
    masking_mode: str = "zero_fill",
    readout_mode: str = "mean_pool",
    spatial_pe_mode: str = "none",
    spatial_path: str = "pool",
    electrode_pe_mode: str = "none",
    num_heads: int | None = None,
) -> PerPhonemeConfig:
    from speech_decoding.v14.config import BackboneConfig, D1DecoderConfig, PoolConfig

    if d_model not in (16, 32, 64):
        raise ValueError(f"d_model must be 16, 32, or 64, got {d_model}")
    if conv2d_kernel not in (1, 3, 5):
        raise ValueError(f"conv2d_kernel must be 1, 3, or 5, got {conv2d_kernel}")
    if num_heads is None:
        num_heads = {16: 1, 32: 2, 64: 4}[d_model]
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
    head_dim = d_model // num_heads
    if head_dim % 2 != 0:
        raise ValueError(
            f"head_dim = d_model/num_heads = {head_dim} must be even for RoPE"
        )
    base = PerPhonemeConfig()
    return replace(
        base,
        d_model=d_model,
        conv2d_kernel_size=conv2d_kernel,
        conv2d_padding=conv2d_kernel // 2,
        pool=PoolConfig(pool_shape=pool_shape),
        per_cell_temporal=replace(base.per_cell_temporal, out_channels=d_model),
        use_parcel_embedding=use_parcel_embedding,
        temporal_frontend=temporal_frontend,
        pool_method=pool_method,
        masking_mode=masking_mode,
        readout_mode=readout_mode,
        spatial_pe_mode=spatial_pe_mode,
        spatial_path=spatial_path,
        electrode_pe_mode=electrode_pe_mode,
        backbone=BackboneConfig(
            d_model=d_model, num_heads=num_heads, head_dim=head_dim,
            ffn_hidden=4 * d_model, num_blocks=backbone_depth, dropout=0.1,
        ),
        decoder=D1DecoderConfig(d_model=d_model, vocab_size=9, prev_embedding_size=10),
    )


def _select_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _expand_trials(trial_idx: list[int]) -> list[int]:
    out: list[int] = []
    for t in trial_idx:
        out.extend([3 * t, 3 * t + 1, 3 * t + 2])
    return out


class _RoundRobinInterleaveLoader:
    """Iterates round-robin over per-patient loaders.

    Each "cycle" yields exactly one micro-batch per patient in a fixed patient
    order. Per-patient loaders are wrapped to restart on exhaustion, so the
    loader with the most batches defines the cycle count per epoch; shorter
    loaders wrap with reshuffle.

    ``len(self) == n_patients * max_patient_batches`` so the training loop's
    grad-accum schedule (``GRAD_ACCUM_STEPS == n_patients``) results in
    ``max_patient_batches`` optimizer steps per epoch.
    """

    def __init__(self, loaders: dict[str, DataLoader]) -> None:
        if not loaders:
            raise ValueError("need at least one per-patient loader")
        self._loaders = loaders
        self._patient_order: list[str] = sorted(loaders.keys())
        self._cycles_per_epoch = max(len(loader) for loader in loaders.values())

    def __len__(self) -> int:
        return self._cycles_per_epoch * len(self._patient_order)

    def __iter__(self) -> Iterator[dict]:
        # Fresh per-patient iterators each epoch (re-shuffles if shuffle=True).
        # Wrap with itertools.cycle(iter(loader)) semantics manually so we can
        # restart iteration without losing a ref.
        iters: dict[str, Iterator[dict]] = {pt: iter(loader) for pt, loader in self._loaders.items()}
        for _ in range(self._cycles_per_epoch):
            for pt in self._patient_order:
                try:
                    yield next(iters[pt])
                except StopIteration:
                    iters[pt] = iter(self._loaders[pt])
                    yield next(iters[pt])


def _pooled_evaluate(model: nn.Module, val_loaders: dict[str, DataLoader]) -> float:
    """Population-mean of per-patient `evaluate_per_phoneme` (early-stop signal)."""

    pers = [evaluate_per_phoneme(model, loader) for loader in val_loaders.values()]
    return float(mean(pers))


def run_one_fold_pooled(
    datasets: dict[str, V14PhonemeDataset],
    *,
    fold_idx: int,
    seed: int,
    backbone_depth: int,
    out_dir: Path,
    tag_prefix: str = "pooled",
    d_model: int = 32,
    conv2d_kernel: int = 3,
    pool_shape: tuple[int, int] = (4, 8),
    use_parcel_embedding: bool = True,
    temporal_frontend: str = "per_cell",
    pool_method: str = "masked_mean",
    masking_mode: str = "zero_fill",
    readout_mode: str = "mean_pool",
    spatial_pe_mode: str = "none",
    spatial_path: str = "pool",
    electrode_pe_mode: str = "none",
    num_heads: int | None = None,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    aug_preset: str = "none",
    max_epochs: int | None = None,
    val_every: int | None = None,
    patience: int | None = None,
    warmup_epochs: int | None = None,
    grad_accum_steps: int | None = None,
    init_from: Path | None = None,
    save_checkpoint: bool = False,
) -> dict:
    """Train one pooled model on ``datasets`` for one (fold, seed, depth).

    Each dataset is one patient's `V14PhonemeDataset`. The token→fold
    partition is built over the union of trial tokens across patients, so the
    held-out fold is globally consistent.

    Returns a result dict with per-patient test PER (argmax + exhaustive AR)
    and the population mean ± std across patients.
    """

    if backbone_depth < 1:
        raise ValueError(f"backbone_depth must be >= 1, got {backbone_depth}")
    if not datasets:
        raise ValueError("need at least one patient dataset")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_slug = "_".join(sorted(datasets.keys()))
    tag = f"{tag_prefix}_{patient_slug}_fold{fold_idx}_seed{seed}_depth{backbone_depth}"
    log_path = out_dir / f"{tag}.log.jsonl"
    result_path = out_dir / f"{tag}.result.json"

    # Pooled token-level CV over the union of trial tokens.
    patient_trial_tokens = {pt: ds.trial_tokens() for pt, ds in datasets.items()}
    folds = make_outer_folds_pooled(
        patient_trial_tokens, n_folds=5, seed=OUTER_FOLD_SEED
    )
    train_by_pt: dict[str, list[int]] = {}
    test_by_pt: dict[str, list[int]] = {}
    for pt, pairs in folds.items():
        train_by_pt[pt], test_by_pt[pt] = pairs[fold_idx]

    val_split = make_val_split_pooled(
        patient_trial_tokens, train_by_pt, val_frac=0.2, seed=VAL_SPLIT_SEED
    )
    train_by_pt = {pt: tr for pt, (tr, _) in val_split.items()}
    val_by_pt = {pt: vl for pt, (_, vl) in val_split.items()}

    train_phon: dict[str, list[int]] = {pt: _expand_trials(v) for pt, v in train_by_pt.items()}
    val_phon: dict[str, list[int]] = {pt: _expand_trials(v) for pt, v in val_by_pt.items()}
    test_phon: dict[str, list[int]] = {pt: _expand_trials(v) for pt, v in test_by_pt.items()}

    _set_seed(seed)
    device = _select_device()
    cfg = _build_config(
        backbone_depth,
        d_model=d_model,
        conv2d_kernel=conv2d_kernel,
        pool_shape=pool_shape,
        use_parcel_embedding=use_parcel_embedding,
        temporal_frontend=temporal_frontend,
        pool_method=pool_method,
        masking_mode=masking_mode,
        readout_mode=readout_mode,
        spatial_pe_mode=spatial_pe_mode,
        spatial_path=spatial_path,
        electrode_pe_mode=electrode_pe_mode,
        num_heads=num_heads,
    )
    model = NeuralFieldPerceiverPerPhoneme(cfg).to(device)

    if init_from is not None:
        state = torch.load(init_from, map_location=device)
        model.load_state_dict(state, strict=True)

    g = torch.Generator()
    g.manual_seed(seed)

    def _make_loader(ds: V14PhonemeDataset, idx: list[int], *, shuffle: bool) -> DataLoader:
        return DataLoader(
            Subset(ds, idx),
            batch_size=TRIALS_PER_BATCH,
            shuffle=shuffle,
            collate_fn=collate_v14_phoneme_batch,
            generator=g if shuffle else None,
        )

    train_loaders = {
        pt: _make_loader(datasets[pt], train_phon[pt], shuffle=True)
        for pt in sorted(datasets.keys())
    }
    val_loaders = {
        pt: _make_loader(datasets[pt], val_phon[pt], shuffle=False)
        for pt in sorted(datasets.keys())
    }
    test_loaders = {
        pt: _make_loader(datasets[pt], test_phon[pt], shuffle=False)
        for pt in sorted(datasets.keys())
    }

    train_loader = _RoundRobinInterleaveLoader(train_loaders)
    val_obj = val_loaders
    effective_accum = grad_accum_steps if grad_accum_steps is not None else len(datasets)

    from speech_decoding.v14.augmentation import PRESETS, AugmentationConfig
    if aug_preset not in PRESETS:
        raise ValueError(
            f"aug_preset must be one of {tuple(PRESETS)}, got {aug_preset!r}"
        )
    aug_cfg: AugmentationConfig | None = None if aug_preset == "none" else PRESETS[aug_preset]
    if label_smoothing > 0.0 or mixup_alpha > 0.0 or aug_cfg is not None:
        from speech_decoding.v14.train import make_per_phoneme_ce_loss
        loss_fn = make_per_phoneme_ce_loss(
            label_smoothing=label_smoothing,
            mixup_alpha=mixup_alpha,
            aug_cfg=aug_cfg,
        )
    else:
        loss_fn = per_phoneme_ce_loss
    kw: dict = {
        "loss_fn": loss_fn,
        "evaluate_fn": _pooled_evaluate,
        "grad_accum_steps": effective_accum,
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
            model, train_loader, val_obj, log_callback=_log, **kw
        )

    if save_checkpoint:
        torch.save(model.state_dict(), out_dir / f"{tag}.ckpt.pt")

    # Per-patient test eval.
    per_patient_test: dict[str, dict[str, float]] = {}
    for pt, loader in test_loaders.items():
        per_patient_test[pt] = {
            "test_per_phoneme": evaluate_per_phoneme(model, loader),
            "test_slot_averaged_per": exhaustive_ar_per_from_loader(model, loader),
        }

    pop_per = [v["test_per_phoneme"] for v in per_patient_test.values()]
    pop_slot = [v["test_slot_averaged_per"] for v in per_patient_test.values()]

    result = {
        "mode": "pooled",
        "patients": sorted(datasets.keys()),
        "fold": fold_idx,
        "seed": seed,
        "backbone_depth": backbone_depth,
        "d_model": d_model,
        "conv2d_kernel": conv2d_kernel,
        "pool_shape": list(pool_shape),
        "use_parcel_embedding": use_parcel_embedding,
        "temporal_frontend": temporal_frontend,
        "pool_method": pool_method,
        "masking_mode": masking_mode,
        "readout_mode": readout_mode,
        "spatial_pe_mode": spatial_pe_mode,
        "spatial_path": spatial_path,
        "electrode_pe_mode": electrode_pe_mode,
        "num_heads": cfg.backbone.num_heads,
        "head_dim": cfg.backbone.head_dim,
        "label_smoothing": label_smoothing,
        "mixup_alpha": mixup_alpha,
        "aug_preset": aug_preset,
        "grad_accum_steps": effective_accum,
        "best_val_per_phoneme": fold_result.best_val_per,
        "per_patient_test": per_patient_test,
        "pop_test_per_phoneme_mean": float(mean(pop_per)),
        "pop_test_per_phoneme_std": float(pstdev(pop_per)) if len(pop_per) > 1 else 0.0,
        "pop_test_slot_averaged_per_mean": float(mean(pop_slot)),
        "pop_test_slot_averaged_per_std": float(pstdev(pop_slot)) if len(pop_slot) > 1 else 0.0,
        "final_epoch": fold_result.final_epoch,
        "early_stopped": fold_result.early_stopped,
        "n_train_trials_by_pt": {pt: len(v) for pt, v in train_by_pt.items()},
        "n_val_trials_by_pt": {pt: len(v) for pt, v in val_by_pt.items()},
        "n_test_trials_by_pt": {pt: len(v) for pt, v in test_by_pt.items()},
        "device": str(device),
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result
