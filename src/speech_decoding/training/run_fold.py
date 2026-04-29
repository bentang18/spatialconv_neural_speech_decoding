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

from speech_decoding.training.config import PerPhonemeConfig
from speech_decoding.training.cv import make_outer_folds, make_val_split
from speech_decoding.training.eval import (
    evaluate_per_phoneme,
    exhaustive_ar_per_from_loader,
)
from speech_decoding.studies.cogan_ps.dataset import (
    V14PhonemeDataset,
    collate_v14_phoneme_batch,
)
from speech_decoding.models.phoneme import NeuralFieldPerceiverPerPhoneme
from speech_decoding.training.train import (
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


def _build_config(
    backbone_depth: int,
    d_model: int = 32,
    conv2d_kernel: int = 3,
    pool_shape: tuple[int, int] = (4, 8),
    temporal_frontend: str = "per_cell",
    pool_method: str = "masked_mean",
    masking_mode: str = "zero_fill",
    readout_mode: str = "mean_pool",
    spatial_pe_mode: str = "none",
    spatial_path: str = "pool",
    electrode_pe_mode: str = "none",
    num_heads: int | None = None,
    use_parcel_embedding: bool = True,
) -> PerPhonemeConfig:
    """Build a consistent config at the given width, depth, conv2d, and pool.

    The model enforces `d_model == per_cell_temporal.out_channels ==
    backbone.d_model == decoder.d_model`, so we rebuild every nested config
    whose width depends on `d_model`. Default heads follow the B-1 contract:
    `d=32 → 2 heads × 16`, `d=64 → 4 heads × 16`. Override with `num_heads`
    for the heads ablation (d=32, heads ∈ {1, 4} → head_dim ∈ {32, 8}).
    FFN stays at 4·d. Conv2d padding auto-derives from kernel
    (k=1→0, k=3→1, k=5→2).
    """

    from speech_decoding.training.config import BackboneConfig, D1DecoderConfig, PoolConfig

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
        temporal_frontend=temporal_frontend,
        pool_method=pool_method,
        masking_mode=masking_mode,
        readout_mode=readout_mode,
        spatial_pe_mode=spatial_pe_mode,
        spatial_path=spatial_path,
        electrode_pe_mode=electrode_pe_mode,
        use_parcel_embedding=use_parcel_embedding,
        backbone=BackboneConfig(
            d_model=d_model, num_heads=num_heads, head_dim=head_dim,
            ffn_hidden=4 * d_model, num_blocks=backbone_depth, dropout=0.1,
        ),
        decoder=D1DecoderConfig(d_model=d_model, vocab_size=9, prev_embedding_size=10),
    )


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
    d_model: int = 32,
    conv2d_kernel: int = 3,
    pool_shape: tuple[int, int] = (4, 8),
    temporal_frontend: str = "per_cell",
    pool_method: str = "masked_mean",
    masking_mode: str = "zero_fill",
    readout_mode: str = "mean_pool",
    spatial_pe_mode: str = "none",
    spatial_path: str = "pool",
    electrode_pe_mode: str = "none",
    num_heads: int | None = None,
    use_parcel_embedding: bool = True,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    aug_preset: str = "none",
    max_epochs: int | None = None,
    val_every: int | None = None,
    patience: int | None = None,
    warmup_epochs: int | None = None,
    init_from: Path | None = None,
    save_checkpoint: bool = False,
) -> dict:
    """Train + eval one `(fold, seed, depth, d_model)` per-phoneme run."""

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
    cfg = _build_config(
        backbone_depth,
        d_model=d_model,
        conv2d_kernel=conv2d_kernel,
        pool_shape=pool_shape,
        temporal_frontend=temporal_frontend,
        pool_method=pool_method,
        masking_mode=masking_mode,
        readout_mode=readout_mode,
        spatial_pe_mode=spatial_pe_mode,
        spatial_path=spatial_path,
        electrode_pe_mode=electrode_pe_mode,
        num_heads=num_heads,
        use_parcel_embedding=use_parcel_embedding,
    )
    model = NeuralFieldPerceiverPerPhoneme(cfg).to(device)

    if init_from is not None:
        state = torch.load(init_from, map_location=device)
        model.load_state_dict(state, strict=True)

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

    from speech_decoding.training.augmentation import PRESETS, AugmentationConfig
    if aug_preset not in PRESETS:
        raise ValueError(
            f"aug_preset must be one of {tuple(PRESETS)}, got {aug_preset!r}"
        )
    aug_cfg: AugmentationConfig | None = None if aug_preset == "none" else PRESETS[aug_preset]
    if label_smoothing > 0.0 or mixup_alpha > 0.0 or aug_cfg is not None:
        from speech_decoding.training.train import make_per_phoneme_ce_loss
        loss_fn = make_per_phoneme_ce_loss(
            label_smoothing=label_smoothing,
            mixup_alpha=mixup_alpha,
            aug_cfg=aug_cfg,
        )
    else:
        loss_fn = per_phoneme_ce_loss
    kw: dict = {
        "loss_fn": loss_fn,
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

    if save_checkpoint:
        torch.save(model.state_dict(), out_dir / f"{tag}.ckpt.pt")

    test_per_phoneme = evaluate_per_phoneme(model, test_loader)
    test_slot_per = exhaustive_ar_per_from_loader(model, test_loader)

    result = {
        "patient": patient_id,
        "fold": fold_idx,
        "seed": seed,
        "backbone_depth": backbone_depth,
        "d_model": d_model,
        "conv2d_kernel": conv2d_kernel,
        "pool_shape": list(pool_shape),
        "temporal_frontend": temporal_frontend,
        "pool_method": pool_method,
        "masking_mode": masking_mode,
        "readout_mode": readout_mode,
        "spatial_pe_mode": spatial_pe_mode,
        "spatial_path": spatial_path,
        "electrode_pe_mode": electrode_pe_mode,
        "use_parcel_embedding": use_parcel_embedding,
        "num_heads": cfg.backbone.num_heads,
        "head_dim": cfg.backbone.head_dim,
        "label_smoothing": label_smoothing,
        "mixup_alpha": mixup_alpha,
        "aug_preset": aug_preset,
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
