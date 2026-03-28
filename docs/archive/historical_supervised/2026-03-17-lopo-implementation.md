# LOPO Cross-Patient Training — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Leave-One-Patient-Out cross-validation with two-stage training: Stage 1 multi-patient SGD with gradient accumulation, Stage 2 target adaptation with source replay and 5-fold stratified CV.

**Architecture:** Three modules — `lopo_trainer.py` (Stage 1), `adaptor.py` (Stage 2), `lopo.py` (orchestrator). Stage 1 trains shared backbone + head + per-patient read-ins on N-1 source patients. Stage 2 freezes backbone, adapts read-in + LayerNorm + head on the target patient with 30% source replay. Orchestrator loops over target patients × seeds, collects metrics, runs Wilcoxon.

**Tech Stack:** PyTorch 2.x, scikit-learn (StratifiedKFold, balanced_accuracy_score), scipy (Wilcoxon), existing `assembler.py`, `augment_batch()`, `ctc_utils.py`, `metrics.py`.

**Design doc:** `docs/plans/2026-03-17-lopo-cross-patient-design.md`

---

## Task 0: Update `default.yaml` config

**Files:**
- Modify: `configs/default.yaml`

**Step 1: Add stage-specific augmentation and missing Stage 2 keys**

```yaml
# E2: Full model — spatial conv + articulatory CTC
experiment: E2_full_model

model:
  readin_type: spatial_conv
  head_type: articulatory
  d_shared: 64
  hidden_size: 64
  gru_layers: 2
  gru_dropout: 0.2
  temporal_stride: 5  # 200Hz → 40Hz
  num_classes: 10  # 9 phonemes + blank

  spatial_conv:
    channels: 8
    num_layers: 1
    kernel_size: 3
    pool_h: 2
    pool_w: 4

training:
  stage1:
    steps: 2000          # 1 batch/patient/step; ~2400 grad steps matches per-patient budget
    lr: 1.0e-3
    warmup_epochs: 20
    readin_lr_mult: 3.0
    weight_decay: 1.0e-4
    batch_size: 16
    grad_clip: 5.0
    patience: 5           # × eval_every = 500 steps without improvement
    eval_every: 100
    augmentation:
      time_shift_frames: 20
      amp_scale_std: 0.15
      channel_dropout_max: 0.2
      noise_frac: 0.02
      feat_dropout_max: 0.3
      time_mask_min: 2
      time_mask_max: 4
      spatial_cutout: false
      temporal_stretch: false

  stage2:
    steps: 100            # per-fold; eval_every=1 so patience=10 = 10 steps
    lr: 1.0e-3
    warmup_epochs: 0
    readin_lr_mult: 3.0
    weight_decay: 1.0e-3
    batch_size: 16
    grad_clip: 5.0
    patience: 10
    eval_every: 1
    cv_folds: 5
    val_fraction: 0.2
    min_outer_class_count: 5   # need >= n_splits samples/class for StratifiedKFold
    min_inner_class_count: 2   # need >= 2 samples/class for StratifiedShuffleSplit
    source_replay_frac: 0.3
    augmentation:
      time_shift_frames: 10
      amp_scale_std: 0.1
      channel_dropout_max: 0.1
      noise_frac: 0.02
      feat_dropout_max: 0.2
      time_mask_min: 2
      time_mask_max: 4
      spatial_cutout: false
      temporal_stretch: false

  # Flat augmentation block for per-patient trainer (backward compat)
  augmentation:
    time_shift_frames: 20
    amp_scale_std: 0.15
    channel_dropout_max: 0.2
    noise_frac: 0.02
    feat_dropout_max: 0.3
    time_mask_min: 2
    time_mask_max: 4

evaluation:
  seeds: [42, 137, 256]
  cv_folds: 5
  primary_metric: per
```

**Step 2: Do NOT change `assembler.py`**

The backbone's `feat_drop_max`, `time_mask_min`, `time_mask_max` are set at construction time from the flat `training.augmentation` block (or defaults). Each training function (`train_stage1`, `_adapt_fold`) overrides these attributes after construction to use stage-specific values. This keeps `assemble_model` simple — it builds from `model` config only.

**Step 3: Commit**

---

## Task 1: Augmentation helper

**Files:**
- Create: `src/speech_decoding/data/augmentation.py` (modify — add helper)

**Step 1: Add `augment_from_config` helper to bottom of `augmentation.py`**

Both Stage 1 and Stage 2 call `augment_batch` by unpacking augmentation config dicts. Extract a helper to avoid repeating the kwarg unpacking everywhere:

```python
def augment_from_config(x: torch.Tensor, ac: dict, training: bool = True) -> torch.Tensor:
    """Call augment_batch using an augmentation config dict."""
    return augment_batch(
        x,
        training=training,
        time_shift_frames=ac.get("time_shift_frames", 20),
        amp_scale_std=ac.get("amp_scale_std", 0.15),
        channel_dropout_max=ac.get("channel_dropout_max", 0.2),
        noise_frac=ac.get("noise_frac", 0.02),
        do_spatial_cutout=ac.get("spatial_cutout", False),
        spatial_cutout_max_h=ac.get("spatial_cutout_max_h", 3),
        spatial_cutout_max_w=ac.get("spatial_cutout_max_w", 6),
        do_temporal_stretch=ac.get("temporal_stretch", False),
        temporal_stretch_max_rate=ac.get("temporal_stretch_max_rate", 0.15),
    )
```

**Step 2: Update per-patient `trainer.py` to use `augment_from_config`**

Replace the 12-line `augment_batch(...)` call in `_train_fold` (lines 163-175) with:

```python
from speech_decoding.data.augmentation import augment_from_config
# ...
x_batch = augment_from_config(x_batch, ac, training=True)
```

**Step 3: Run existing tests**

Run: `pytest tests/test_trainer.py tests/test_augmentation.py -v --tb=short -m "not slow"`
Expected: All pass (behavioral change is zero — just extracted a helper).

**Step 4: Commit**

---

## Task 2: Stage 1 multi-patient trainer — tests

**Files:**
- Create: `tests/test_lopo_trainer.py`

**Step 1: Write Stage 1 tests using synthetic data**

```python
"""Tests for Stage 1 multi-patient training."""
import numpy as np
import pytest
import torch
import yaml

from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.training.lopo_trainer import train_stage1


def _make_synthetic_dataset(pid: str, grid_shape: tuple[int, int], n_trials: int = 40) -> BIDSDataset:
    """Create a synthetic dataset for testing.

    Uses deterministic labels cycling through all 9 phonemes to avoid
    StratifiedKFold failures from missing classes.
    """
    H, W = grid_shape
    T = 100
    # Stable seed from patient ID (hash() is process-unstable in Python 3.3+)
    np.random.seed(int(pid.replace("P", "").replace("T", "")) + 42)
    data = np.random.randn(n_trials, H, W, T).astype(np.float32)
    # Cycle through all 9 phonemes deterministically
    labels = [[((i * 3 + j) % 9) + 1 for j in range(3)] for i in range(n_trials)]
    return BIDSDataset(data, labels, pid, grid_shape)


def _make_config() -> dict:
    """Minimal LOPO config for testing."""
    return {
        "model": {
            "readin_type": "spatial_conv",
            "head_type": "articulatory",
            "d_shared": 64,
            "hidden_size": 32,
            "gru_layers": 1,
            "gru_dropout": 0.0,
            "temporal_stride": 5,
            "num_classes": 10,
            "spatial_conv": {
                "channels": 8,
                "num_layers": 1,
                "kernel_size": 3,
                "pool_h": 2,
                "pool_w": 4,
            },
        },
        "training": {
            "stage1": {
                "steps": 5,
                "lr": 1e-3,
                "warmup_epochs": 0,
                "readin_lr_mult": 3.0,
                "weight_decay": 1e-4,
                "batch_size": 8,
                "grad_clip": 5.0,
                "patience": 3,
                "eval_every": 2,
                "augmentation": {
                    "time_shift_frames": 0,
                    "amp_scale_std": 0.0,
                    "channel_dropout_max": 0.0,
                    "noise_frac": 0.0,
                    "feat_dropout_max": 0.0,
                    "time_mask_min": 2,
                    "time_mask_max": 4,
                    "spatial_cutout": False,
                    "temporal_stretch": False,
                },
            },
        },
        "evaluation": {"seeds": [42], "cv_folds": 5, "primary_metric": "per"},
    }


class TestTrainStage1:
    def test_returns_checkpoint_dict(self):
        """Stage 1 should return a checkpoint with backbone, head, and read-in state dicts."""
        sources = {
            "P1": _make_synthetic_dataset("P1", (8, 16)),
            "P2": _make_synthetic_dataset("P2", (8, 16)),
        }
        config = _make_config()
        checkpoint = train_stage1(sources, config, seed=42, device="cpu")

        assert "backbone" in checkpoint
        assert "head" in checkpoint
        assert "read_ins" in checkpoint
        assert "P1" in checkpoint["read_ins"]
        assert "P2" in checkpoint["read_ins"]

    def test_loss_decreases(self):
        """Training loss should decrease over steps (no augmentation, overfit-friendly)."""
        sources = {
            "P1": _make_synthetic_dataset("P1", (8, 16), n_trials=20),
            "P2": _make_synthetic_dataset("P2", (8, 16), n_trials=20),
        }
        config = _make_config()
        config["training"]["stage1"]["steps"] = 10
        config["training"]["stage1"]["eval_every"] = 5
        checkpoint = train_stage1(sources, config, seed=42, device="cpu")

        assert "train_losses" in checkpoint
        assert len(checkpoint["train_losses"]) > 1
        # Loss should decrease from first to last recorded
        assert checkpoint["train_losses"][-1] < checkpoint["train_losses"][0]

    def test_handles_different_grid_shapes(self):
        """Stage 1 should work with patients that have different grid shapes."""
        sources = {
            "P1": _make_synthetic_dataset("P1", (8, 16)),
            "P2": _make_synthetic_dataset("P2", (12, 22)),
        }
        config = _make_config()
        checkpoint = train_stage1(sources, config, seed=42, device="cpu")

        assert "P1" in checkpoint["read_ins"]
        assert "P2" in checkpoint["read_ins"]

    def test_val_split_applied(self):
        """Should hold out val_fraction of each patient's data."""
        sources = {
            "P1": _make_synthetic_dataset("P1", (8, 16), n_trials=50),
        }
        config = _make_config()
        config["training"]["stage1"]["steps"] = 3
        config["training"]["stage1"]["eval_every"] = 1
        checkpoint = train_stage1(sources, config, seed=42, device="cpu")

        # Checkpoint should record val losses
        assert "val_losses" in checkpoint
        assert len(checkpoint["val_losses"]) > 0

    def test_gradient_accumulation_across_patients(self):
        """Each step should accumulate gradients from all source patients."""
        sources = {
            "P1": _make_synthetic_dataset("P1", (8, 16), n_trials=20),
            "P2": _make_synthetic_dataset("P2", (8, 16), n_trials=20),
            "P3": _make_synthetic_dataset("P3", (8, 16), n_trials=20),
        }
        config = _make_config()
        config["training"]["stage1"]["steps"] = 2
        checkpoint = train_stage1(sources, config, seed=42, device="cpu")

        # All 3 read-ins should be in checkpoint
        assert len(checkpoint["read_ins"]) == 3
```

**Step 2: Run to verify tests fail**

Run: `pytest tests/test_lopo_trainer.py -v --tb=short`
Expected: FAIL — `ImportError: cannot import name 'train_stage1' from 'speech_decoding.training.lopo_trainer'`

---

## Task 3: Stage 1 multi-patient trainer — implementation

**Files:**
- Create: `src/speech_decoding/training/lopo_trainer.py`

**Step 1: Write `train_stage1`**

```python
"""Stage 1: Multi-patient training with gradient accumulation.

Trains shared backbone + head + per-patient read-ins on N-1 source
patients. One optimizer step per step: sample one batch from each
patient, accumulate gradients normalized by number of sources.
steps: 2000 gives ~2400 gradient updates, matching per-patient budget.
"""
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
from speech_decoding.training.ctc_utils import blank_ratio, ctc_loss

logger = logging.getLogger(__name__)


def _safe_stage2_splits(
    labels: list[int],
    requested_folds: int,
    min_inner_class_count: int = 2,
) -> tuple[int, bool]:
    """Choose a safe outer fold count and whether inner stratification is possible.

    Outer StratifiedKFold requires at least `n_splits` samples in every class.
    The nested inner train/val split is stricter: after one outer split, each
    class in the training fold should still have at least 2 examples to support
    StratifiedShuffleSplit. For low-trial patients, reduce outer folds first;
    if that is still insufficient, keep the outer stratification and fall back
    to a reproducible unstratified inner validation split.
    """
    counts = Counter(labels)
    min_count = min(counts.values())
    outer_folds = max(2, min(requested_folds, min_count))
    can_inner_stratify = (min_count - 1) >= min_inner_class_count
    return outer_folds, can_inner_stratify


def train_stage1(
    source_datasets: dict[str, BIDSDataset],
    config: dict,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Train backbone + head + per-patient read-ins on source patients.

    Args:
        source_datasets: {patient_id: BIDSDataset} for each source patient.
        config: Full YAML config dict.
        seed: Random seed.
        device: Device string.

    Returns:
        Checkpoint dict with keys: backbone, head, read_ins, train_losses, val_losses.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tc = config["training"]["stage1"]
    ac = tc.get("augmentation", config.get("training", {}).get("augmentation", {}))

    # Build model with all source patients
    patients = {pid: ds.grid_shape for pid, ds in source_datasets.items()}
    backbone, head, read_ins = assemble_model(config, patients)

    # Override backbone augmentation attrs from stage-specific config
    backbone.feat_drop_max = ac.get("feat_dropout_max", 0.3)
    backbone.time_mask_min = ac.get("time_mask_min", 2)
    backbone.time_mask_max = ac.get("time_mask_max", 4)

    backbone = backbone.to(device)
    head = head.to(device)
    for pid in read_ins:
        read_ins[pid] = read_ins[pid].to(device)

    # Split each patient 80/20 stratified for validation
    # Stratify on y[0] (first phoneme): with 9 phonemes and ~150 trials,
    # this gives ~17 trials/class — fine. Full-tuple stratification would
    # give 729 classes (~0.2 trials/class) — too sparse.
    train_data, val_data = {}, {}
    for pid, ds in source_datasets.items():
        strat_labels = [y[0] for y in ds.ctc_labels]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(sss.split(np.zeros(len(ds)), strat_labels))
        train_data[pid] = {
            "x": torch.from_numpy(ds.grid_data[train_idx]),
            "y": [ds.ctc_labels[i] for i in train_idx],
        }
        val_data[pid] = {
            "x": torch.from_numpy(ds.grid_data[val_idx]),
            "y": [ds.ctc_labels[i] for i in val_idx],
        }

    # Optimizer — one group per read-in (3× LR) + backbone + head (1× LR)
    param_groups = [
        {"params": read_ins[pid].parameters(), "lr": tc["lr"] * tc["readin_lr_mult"]}
        for pid in read_ins
    ]
    param_groups.append({"params": backbone.parameters(), "lr": tc["lr"]})
    param_groups.append({"params": head.parameters(), "lr": tc["lr"]})
    optimizer = AdamW(param_groups, weight_decay=tc["weight_decay"])

    warmup = tc.get("warmup_epochs", 0)
    total = tc["steps"]

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
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
    train_losses = []
    val_losses = []

    for step in range(total):
        # --- Train: one batch per patient, accumulate gradients ---
        backbone.train()
        head.train()
        for pid in read_ins:
            read_ins[pid].train()

        optimizer.zero_grad()
        step_loss = 0.0

        for pid in source_datasets:
            td = train_data[pid]
            idx = torch.randint(0, len(td["x"]), (B,))
            x_batch = td["x"][idx]
            y_batch = [td["y"][i] for i in idx.tolist()]

            x_batch = augment_from_config(x_batch, ac, training=True)
            x_batch = x_batch.to(device)

            shared = read_ins[pid](x_batch)
            h = backbone(shared)
            log_probs = head(h)
            loss = ctc_loss(log_probs, y_batch) / n_source
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(all_params, tc["grad_clip"])
        optimizer.step()
        scheduler.step()
        train_losses.append(step_loss)

        # --- Validate ---
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
                    log_probs = head(h)
                    val_loss += ctc_loss(log_probs, vd["y"]).item() / n_source

            val_losses.append(val_loss)
            logger.info(
                "  step %d: train_loss=%.4f val_loss=%.4f",
                step + 1, step_loss, val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "backbone": deepcopy(backbone.state_dict()),
                    "head": deepcopy(head.state_dict()),
                    "read_ins": {pid: deepcopy(ri.state_dict()) for pid, ri in read_ins.items()},
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= tc["patience"]:
                logger.info("Stage 1 early stopping at step %d", step + 1)
                break

    # Restore best
    if best_state is not None:
        backbone.load_state_dict(best_state["backbone"])
        head.load_state_dict(best_state["head"])
        for pid in read_ins:
            read_ins[pid].load_state_dict(best_state["read_ins"][pid])

    checkpoint = {
        "backbone": deepcopy(backbone.state_dict()),
        "head": deepcopy(head.state_dict()),
        "read_ins": {pid: deepcopy(ri.state_dict()) for pid, ri in read_ins.items()},
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    return checkpoint
```

**Step 2: Run tests**

Run: `pytest tests/test_lopo_trainer.py -v --tb=short`
Expected: All 5 tests pass.

**Step 3: Commit**

---

## Task 4: Stage 2 target adaptation — tests

**Files:**
- Create: `tests/test_adaptor.py`

**Step 1: Write Stage 2 tests**

```python
"""Tests for Stage 2 target adaptation."""
import numpy as np
import pytest
import torch

from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.training.adaptor import adapt_stage2
from speech_decoding.training.lopo_trainer import train_stage1


def _make_synthetic_dataset(pid, grid_shape=(8, 16), n_trials=40):
    """Deterministic labels + stable seed (no hash())."""
    H, W = grid_shape
    # Stable seed: extract digits from pid string
    np.random.seed(int(pid.replace("P", "").replace("T", "")) + 42)
    data = np.random.randn(n_trials, H, W, 100).astype(np.float32)
    # Cycle through all 9 phonemes to guarantee all classes present
    labels = [[((i * 3 + j) % 9) + 1 for j in range(3)] for i in range(n_trials)]
    return BIDSDataset(data, labels, pid, grid_shape)


def _make_config():
    return {
        "model": {
            "readin_type": "spatial_conv",
            "head_type": "articulatory",
            "d_shared": 64,
            "hidden_size": 32,
            "gru_layers": 1,
            "gru_dropout": 0.0,
            "temporal_stride": 5,
            "num_classes": 10,
            "spatial_conv": {
                "channels": 8, "num_layers": 1, "kernel_size": 3,
                "pool_h": 2, "pool_w": 4,
            },
        },
        "training": {
            "stage1": {
                "steps": 3, "lr": 1e-3, "warmup_epochs": 0,
                "readin_lr_mult": 3.0, "weight_decay": 1e-4,
                "batch_size": 8, "grad_clip": 5.0, "patience": 3,
                "eval_every": 1,
                "augmentation": {
                    "time_shift_frames": 0, "amp_scale_std": 0.0,
                    "channel_dropout_max": 0.0, "noise_frac": 0.0,
                    "feat_dropout_max": 0.0, "time_mask_min": 2,
                    "time_mask_max": 4, "spatial_cutout": False,
                    "temporal_stretch": False,
                },
            },
            "stage2": {
                "steps": 3, "lr": 1e-3, "warmup_epochs": 0,
                "readin_lr_mult": 3.0, "weight_decay": 1e-3,
                "batch_size": 8, "grad_clip": 5.0, "patience": 3,
                "eval_every": 1, "cv_folds": 2, "val_fraction": 0.2,
                "source_replay_frac": 0.3,
                "augmentation": {
                    "time_shift_frames": 0, "amp_scale_std": 0.0,
                    "channel_dropout_max": 0.0, "noise_frac": 0.0,
                    "feat_dropout_max": 0.0, "time_mask_min": 2,
                    "time_mask_max": 4, "spatial_cutout": False,
                    "temporal_stretch": False,
                },
            },
        },
        "evaluation": {"seeds": [42], "cv_folds": 2, "primary_metric": "per"},
    }


class TestAdaptStage2:
    def test_returns_metrics_dict(self):
        sources = {
            "P1": _make_synthetic_dataset("P1"),
            "P2": _make_synthetic_dataset("P2"),
        }
        target = _make_synthetic_dataset("T1")
        config = _make_config()

        checkpoint = train_stage1(sources, config, seed=42, device="cpu")
        result = adapt_stage2(checkpoint, target, sources, config, seed=42, device="cpu")

        assert "per_mean" in result
        assert "fold_results" in result
        assert len(result["fold_results"]) == 2  # cv_folds=2

    def test_backbone_is_frozen(self):
        """Backbone state dict should not change during Stage 2."""
        sources = {"P1": _make_synthetic_dataset("P1")}
        target = _make_synthetic_dataset("T1")
        config = _make_config()

        checkpoint = train_stage1(sources, config, seed=42, device="cpu")
        bb_before = {k: v.clone() for k, v in checkpoint["backbone"].items()}

        # Stage 2 should NOT modify the checkpoint's backbone
        result = adapt_stage2(checkpoint, target, sources, config, seed=42, device="cpu")

        # Verify checkpoint backbone unchanged
        for k in bb_before:
            assert torch.equal(bb_before[k], checkpoint["backbone"][k])

    def test_source_replay_present(self):
        """Stage 2 with source_replay_frac > 0 should use source data."""
        sources = {"P1": _make_synthetic_dataset("P1")}
        target = _make_synthetic_dataset("T1")
        config = _make_config()
        config["training"]["stage2"]["source_replay_frac"] = 0.3

        checkpoint = train_stage1(sources, config, seed=42, device="cpu")
        result = adapt_stage2(checkpoint, target, sources, config, seed=42, device="cpu")

        # Should complete without error and return metrics
        assert result["per_mean"] >= 0

    def test_handles_different_target_grid(self):
        """Target can have a different grid shape than sources."""
        sources = {"P1": _make_synthetic_dataset("P1", (8, 16))}
        target = _make_synthetic_dataset("T1", (12, 22))
        config = _make_config()

        checkpoint = train_stage1(sources, config, seed=42, device="cpu")
        result = adapt_stage2(checkpoint, target, sources, config, seed=42, device="cpu")

        assert "per_mean" in result

    def test_low_trial_target_falls_back_to_safe_split(self):
        """Low-trial patients should reduce outer folds and disable inner stratification if needed."""
        sources = {"P1": _make_synthetic_dataset("P1", n_trials=40)}
        target = _make_synthetic_dataset("T1", n_trials=18)  # 2 trials/class
        config = _make_config()
        config["training"]["stage2"]["cv_folds"] = 5

        checkpoint = train_stage1(sources, config, seed=42, device="cpu")
        result = adapt_stage2(checkpoint, target, sources, config, seed=42, device="cpu")

        # Should complete without StratifiedKFold/ShuffleSplit errors
        assert "per_mean" in result
```

**Step 2: Run to verify tests fail**

Run: `pytest tests/test_adaptor.py -v --tb=short`
Expected: FAIL — `ImportError: cannot import name 'adapt_stage2'`

---

## Task 5: Stage 2 target adaptation — implementation

**Files:**
- Create: `src/speech_decoding/training/adaptor.py`

**Step 1: Write `adapt_stage2`**

```python
"""Stage 2: Target adaptation with source replay.

Freezes backbone (Conv1d + BiGRU). Trains fresh target read-in +
LayerNorm + head using target data with 30% source replay.
5-fold stratified CV on target patient.
"""
from __future__ import annotations

import logging
import math
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
from speech_decoding.training.ctc_utils import blank_ratio, ctc_loss, greedy_decode

logger = logging.getLogger(__name__)


def adapt_stage2(
    checkpoint: dict,
    target_dataset: BIDSDataset,
    source_datasets: dict[str, BIDSDataset],
    config: dict,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Adapt to target patient using Stage 1 checkpoint + source replay.

    Args:
        checkpoint: Stage 1 checkpoint dict (backbone, head, read_ins).
        target_dataset: BIDSDataset for the target patient.
        source_datasets: {pid: BIDSDataset} for source patients.
        config: Full YAML config dict.
        seed: Random seed.
        device: Device string.

    Returns:
        Dict with per-fold and mean metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tc = config["training"]["stage2"]
    n_folds = tc.get("cv_folds", config["evaluation"]["cv_folds"])

    # Stratify on y[0] (first phoneme): 9 classes × ~150 trials = ~17/class.
    # Full-tuple (y[0],y[1],y[2]) would give 729 classes — too sparse.
    strat_labels = [y[0] for y in target_dataset.ctc_labels]
    outer_folds, can_inner_stratify = _safe_stage2_splits(
        strat_labels,
        requested_folds=n_folds,
        min_inner_class_count=tc.get("min_inner_class_count", 2),
    )
    if outer_folds < n_folds:
        logger.warning(
            "Target %s: reducing Stage 2 cv_folds from %d to %d due to low class counts",
            target_dataset.patient_id, n_folds, outer_folds,
        )
    skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(
        np.zeros(len(target_dataset)), strat_labels
    )):
        logger.info("  Stage 2 fold %d/%d", fold_idx + 1, n_folds)
        result = _adapt_fold(
            checkpoint, target_dataset, source_datasets, config,
            train_idx, test_idx, seed, device, can_inner_stratify,
        )
        fold_results.append(result)
        logger.info(
            "  Fold %d: PER=%.3f, bal_acc=%.3f",
            fold_idx + 1, result["per"], result["bal_acc_mean"],
        )

    # Aggregate
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
    """Train one Stage 2 CV fold."""
    tc = config["training"]["stage2"]
    ac = tc.get("augmentation", config.get("training", {}).get("augmentation", {}))

    # Build model — backbone + head from checkpoint, fresh target read-in
    target_pid = target_dataset.patient_id
    all_patients = {pid: ds.grid_shape for pid, ds in source_datasets.items()}
    all_patients[target_pid] = target_dataset.grid_shape

    backbone, head, all_read_ins = assemble_model(config, all_patients)
    backbone = backbone.to(device)
    head = head.to(device)

    # Load Stage 1 weights for backbone + head
    backbone.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])

    # Override backbone augmentation attrs for Stage 2 (lighter than Stage 1)
    backbone.feat_drop_max = ac.get("feat_dropout_max", 0.2)
    backbone.time_mask_min = ac.get("time_mask_min", 2)
    backbone.time_mask_max = ac.get("time_mask_max", 4)

    # Load source read-ins from checkpoint (frozen)
    source_read_ins = {}
    for pid in source_datasets:
        ri = all_read_ins[pid].to(device)
        ri.load_state_dict(checkpoint["read_ins"][pid])
        ri.eval()
        for p in ri.parameters():
            p.requires_grad = False
        source_read_ins[pid] = ri

    # Target read-in — fresh init (Kaiming), trainable
    target_read_in = all_read_ins[target_pid].to(device)

    # Freeze backbone internals (Conv1d + GRU), unfreeze LayerNorm
    for name, param in backbone.named_parameters():
        if "layernorm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Split target train into train/val for early stopping
    target_x = torch.from_numpy(target_dataset.grid_data[train_idx])
    target_y = [target_dataset.ctc_labels[i] for i in train_idx]
    test_x = torch.from_numpy(target_dataset.grid_data[test_idx])
    test_y = [target_dataset.ctc_labels[i] for i in test_idx]

    val_frac = tc.get("val_fraction", 0.2)
    if can_inner_stratify:
        strat = [y[0] for y in target_y]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        t_idx, v_idx = next(sss.split(np.zeros(len(target_y)), strat))
    else:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(target_y))
        n_val = max(1, int(round(val_frac * len(target_y))))
        v_idx = np.sort(perm[:n_val])
        t_idx = np.sort(perm[n_val:])
        logger.warning(
            "Target %s fold has insufficient class counts for inner stratification; using unstratified val split",
            target_dataset.patient_id,
        )
    train_x = target_x[t_idx]
    train_y = [target_y[i] for i in t_idx]
    val_x = target_x[v_idx]
    val_y = [target_y[i] for i in v_idx]

    # Prepare source data for replay (CPU tensors)
    source_data = {}
    for pid, ds in source_datasets.items():
        source_data[pid] = {
            "x": torch.from_numpy(ds.grid_data),
            "y": ds.ctc_labels,
        }
    source_pids = list(source_data.keys())

    # Optimizer — only trainable params
    trainable_params = (
        [{"params": target_read_in.parameters(), "lr": tc["lr"] * tc.get("readin_lr_mult", 1.0)}]
        + [{"params": [p for p in backbone.parameters() if p.requires_grad], "lr": tc["lr"]}]
        + [{"params": head.parameters(), "lr": tc["lr"]}]
    )
    optimizer = AdamW(trainable_params, weight_decay=tc["weight_decay"])

    warmup = tc.get("warmup_epochs", 0)
    total = tc["steps"]

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    B = tc["batch_size"]
    replay_frac = tc.get("source_replay_frac", 0.3)
    n_target_batch = max(int((1 - replay_frac) * B), 1)
    n_source_batch = B - n_target_batch

    all_trainable = [p for p in (
        list(target_read_in.parameters())
        + list(backbone.parameters())
        + list(head.parameters())
    ) if p.requires_grad]

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for step in range(total):
        backbone.train()
        head.train()
        target_read_in.train()

        optimizer.zero_grad()

        # --- Target portion ---
        t_idx_batch = torch.randint(0, len(train_x), (n_target_batch,))
        x_tgt = augment_from_config(train_x[t_idx_batch], ac, training=True)
        x_tgt = x_tgt.to(device)
        y_tgt = [train_y[i] for i in t_idx_batch.tolist()]

        shared_tgt = target_read_in(x_tgt)
        h_tgt = backbone(shared_tgt)
        lp_tgt = head(h_tgt)
        loss_tgt = ctc_loss(lp_tgt, y_tgt)

        # --- Source replay portion ---
        loss_src = torch.tensor(0.0, device=device)
        if n_source_batch > 0 and source_pids:
            # Round-robin: pick one source patient per step
            src_pid = source_pids[step % len(source_pids)]
            sd = source_data[src_pid]
            s_idx = torch.randint(0, len(sd["x"]), (n_source_batch,))
            x_src = sd["x"][s_idx].to(device)
            y_src = [sd["y"][i] for i in s_idx.tolist()]

            with torch.no_grad():
                shared_src = source_read_ins[src_pid](x_src)
            # backbone outside no_grad: LayerNorm gets gradients from replay,
            # Conv1d+GRU don't (requires_grad=False on those params)
            h_src = backbone(shared_src)
            lp_src = head(h_src)  # grad flows to head + LayerNorm
            loss_src = ctc_loss(lp_src, y_src)

        loss = (1 - replay_frac) * loss_tgt + replay_frac * loss_src
        loss.backward()

        torch.nn.utils.clip_grad_norm_(all_trainable, tc["grad_clip"])
        optimizer.step()
        scheduler.step()

        # --- Validate ---
        if (step + 1) % tc.get("eval_every", 1) == 0:
            backbone.eval()
            head.eval()
            target_read_in.eval()
            with torch.no_grad():
                vx = val_x.to(device)
                shared = target_read_in(vx)
                h = backbone(shared)
                lp = head(h)
                vl = ctc_loss(lp, val_y).item()

            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {
                    "target_read_in": deepcopy(target_read_in.state_dict()),
                    "head": deepcopy(head.state_dict()),
                    "backbone": deepcopy(backbone.state_dict()),
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= tc["patience"]:
                logger.info("  Stage 2 early stopping at step %d", step + 1)
                break

    # Restore best
    if best_state is not None:
        target_read_in.load_state_dict(best_state["target_read_in"])
        head.load_state_dict(best_state["head"])
        backbone.load_state_dict(best_state["backbone"])

    # Final evaluation on test fold
    backbone.eval()
    head.eval()
    target_read_in.eval()
    with torch.no_grad():
        tx = test_x.to(device)
        shared = target_read_in(tx)
        h = backbone(shared)
        lp = head(h)
        predictions = greedy_decode(lp)
        br = blank_ratio(lp)

    metrics = evaluate_predictions(predictions, test_y, n_positions=3)
    metrics["blank_ratio"] = br
    return metrics
```

**Step 2: Run tests**

Run: `pytest tests/test_adaptor.py -v --tb=short`
Expected: All 4 tests pass.

**Step 3: Commit**

---

## Task 6: LOPO orchestrator — tests

**Files:**
- Create: `tests/test_lopo.py`

**Step 1: Write orchestrator tests**

```python
"""Tests for LOPO orchestrator."""
import numpy as np
import pytest

from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.training.lopo import run_lopo


def _make_ds(pid, grid=(8, 16), n=30):
    """Deterministic labels + stable seed."""
    H, W = grid
    np.random.seed(int(pid.replace("P", "")) + 42)
    data = np.random.randn(n, H, W, 100).astype(np.float32)
    labels = [[((i * 3 + j) % 9) + 1 for j in range(3)] for i in range(n)]
    return BIDSDataset(data, labels, pid, grid)


def _make_config():
    aug = {
        "time_shift_frames": 0, "amp_scale_std": 0.0,
        "channel_dropout_max": 0.0, "noise_frac": 0.0,
        "feat_dropout_max": 0.0, "time_mask_min": 2,
        "time_mask_max": 4, "spatial_cutout": False,
        "temporal_stretch": False,
    }
    return {
        "model": {
            "readin_type": "spatial_conv", "head_type": "articulatory",
            "d_shared": 64, "hidden_size": 32, "gru_layers": 1,
            "gru_dropout": 0.0, "temporal_stride": 5, "num_classes": 10,
            "spatial_conv": {
                "channels": 8, "num_layers": 1, "kernel_size": 3,
                "pool_h": 2, "pool_w": 4,
            },
        },
        "training": {
            "stage1": {
                "steps": 3, "lr": 1e-3, "warmup_epochs": 0,
                "readin_lr_mult": 3.0, "weight_decay": 1e-4,
                "batch_size": 8, "grad_clip": 5.0, "patience": 3,
                "eval_every": 1, "augmentation": aug,
            },
            "stage2": {
                "steps": 3, "lr": 1e-3, "warmup_epochs": 0,
                "readin_lr_mult": 3.0, "weight_decay": 1e-3,
                "batch_size": 8, "grad_clip": 5.0, "patience": 3,
                "eval_every": 1, "cv_folds": 2, "val_fraction": 0.2,
                "source_replay_frac": 0.3, "augmentation": aug,
            },
        },
        "evaluation": {"seeds": [42], "cv_folds": 2, "primary_metric": "per"},
    }


class TestRunLopo:
    def test_returns_results_for_all_patients(self):
        datasets = {
            "P1": _make_ds("P1"),
            "P2": _make_ds("P2"),
            "P3": _make_ds("P3"),
        }
        config = _make_config()
        results = run_lopo(datasets, config, seeds=[42], device="cpu")

        assert "per_patient" in results
        for pid in ["P1", "P2", "P3"]:
            assert pid in results["per_patient"]

    def test_population_stats(self):
        datasets = {
            "P1": _make_ds("P1"),
            "P2": _make_ds("P2"),
            "P3": _make_ds("P3"),
        }
        config = _make_config()
        results = run_lopo(datasets, config, seeds=[42], device="cpu")

        assert "population_per_mean" in results
        assert "population_per_std" in results

    def test_wilcoxon_with_baseline(self):
        """When baseline_pers provided, Wilcoxon stats are computed."""
        datasets = {
            "P1": _make_ds("P1"),
            "P2": _make_ds("P2"),
            "P3": _make_ds("P3"),
        }
        config = _make_config()
        # Fake baseline: high PER for all patients
        baseline_pers = {"P1": 1.0, "P2": 1.0, "P3": 1.0}
        results = run_lopo(datasets, config, seeds=[42], device="cpu",
                           baseline_pers=baseline_pers)

        assert "wilcoxon_stat" in results
        assert "wilcoxon_p" in results
```

**Step 2: Run to verify fail**

Run: `pytest tests/test_lopo.py -v --tb=short`
Expected: FAIL — `ImportError`

---

## Task 7: LOPO orchestrator — implementation

**Files:**
- Create: `src/speech_decoding/training/lopo.py`

**Step 1: Write `run_lopo`**

```python
"""LOPO orchestrator: Leave-One-Patient-Out cross-validation.

For each target patient:
  Stage 1 → train on remaining patients
  Stage 2 → adapt to target with source replay + 5-fold CV
Aggregate metrics and run Wilcoxon signed-rank test vs baseline.
"""
from __future__ import annotations

import logging

import numpy as np

from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.training.adaptor import adapt_stage2
from speech_decoding.training.lopo_trainer import train_stage1

logger = logging.getLogger(__name__)


def run_lopo(
    all_datasets: dict[str, BIDSDataset],
    config: dict,
    seeds: list[int],
    device: str = "cpu",
    baseline_pers: dict[str, float] | None = None,
) -> dict:
    """Run full LOPO cross-validation.

    Args:
        all_datasets: {patient_id: BIDSDataset} for all patients.
        config: Full YAML config dict.
        seeds: List of random seeds.
        device: Device string.
        baseline_pers: Optional {patient_id: PER} from per-patient baseline.
            If provided, Wilcoxon signed-rank test is computed.

    Returns:
        Dict with per_patient results, population stats, optional Wilcoxon.
    """
    patient_ids = sorted(all_datasets.keys())
    per_patient: dict[str, list[dict]] = {pid: [] for pid in patient_ids}

    for target_pid in patient_ids:
        logger.info("=" * 60)
        logger.info("LOPO target: %s", target_pid)

        source_datasets = {
            pid: ds for pid, ds in all_datasets.items() if pid != target_pid
        }

        for seed in seeds:
            logger.info("  Seed %d: Stage 1 (%d source patients)", seed, len(source_datasets))
            checkpoint = train_stage1(source_datasets, config, seed=seed, device=device)

            logger.info("  Seed %d: Stage 2 (target %s)", seed, target_pid)
            result = adapt_stage2(
                checkpoint, all_datasets[target_pid], source_datasets,
                config, seed=seed, device=device,
            )
            per_patient[target_pid].append(result)

            logger.info(
                "  %s seed=%d: PER=%.3f±%.3f",
                target_pid, seed,
                result["per_mean"], result["per_std"],
            )

    # Aggregate: per-patient mean PER across seeds
    patient_pers = {}
    for pid in patient_ids:
        seed_pers = [r["per_mean"] for r in per_patient[pid]]
        patient_pers[pid] = np.mean(seed_pers).item()

    all_pers = list(patient_pers.values())
    summary = {
        "per_patient": per_patient,
        "patient_pers": patient_pers,
        "population_per_mean": np.mean(all_pers).item(),
        "population_per_std": np.std(all_pers).item(),
    }

    # Wilcoxon signed-rank test vs baseline (if provided)
    if baseline_pers is not None:
        from scipy.stats import wilcoxon

        paired_lopo = [patient_pers[pid] for pid in patient_ids]
        paired_base = [baseline_pers[pid] for pid in patient_ids]
        stat, p = wilcoxon(paired_lopo, paired_base, alternative="less")
        summary["wilcoxon_stat"] = float(stat)
        summary["wilcoxon_p"] = float(p)
        logger.info(
            "Wilcoxon vs baseline: stat=%.3f, p=%.4f", stat, p,
        )

    return summary
```

**Step 2: Run tests**

Run: `pytest tests/test_lopo.py -v --tb=short`
Expected: All pass.

**Step 3: Commit**

---

## Task 8: CLI script

**Files:**
- Create: `scripts/train_lopo.py`

**Step 1: Write CLI**

```python
#!/usr/bin/env python3
"""LOPO cross-patient training script.

Usage:
    python scripts/train_lopo.py [--config configs/default.yaml]
                                  [--patients S14 S22 ...]
                                  [--device cpu]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.training.lopo import run_lopo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PS_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S36", "S39", "S57", "S58", "S62"]


def main():
    parser = argparse.ArgumentParser(description="LOPO cross-patient training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--patients", nargs="+", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tmin", type=float, default=-0.5)
    parser.add_argument("--tmax", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    bids_root = Path(paths["ps_bids_root"])
    patient_ids = args.patients or PS_PATIENTS
    seeds = config["evaluation"]["seeds"]

    # Load all patient datasets
    all_datasets = {}
    for pid in patient_ids:
        try:
            ds = load_patient_data(pid, bids_root, task="PhonemeSequence", n_phons=3,
                                   tmin=args.tmin, tmax=args.tmax)
            all_datasets[pid] = ds
            logger.info("Loaded %s: %d trials, grid %s", pid, len(ds), ds.grid_shape)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", pid, e)

    if len(all_datasets) < 3:
        logger.error("Need at least 3 patients for LOPO, got %d", len(all_datasets))
        return

    logger.info("Running LOPO with %d patients, %d seeds", len(all_datasets), len(seeds))
    results = run_lopo(all_datasets, config, seeds=seeds, device=args.device)

    # Summary
    print("\n" + "=" * 50)
    print(f"{'Patient':<10} {'Mean PER':<12}")
    print("-" * 50)
    for pid in sorted(results["patient_pers"]):
        print(f"{pid:<10} {results['patient_pers'][pid]:.3f}")
    print("-" * 50)
    print(f"{'Population':<10} {results['population_per_mean']:.3f} ± {results['population_per_std']:.3f}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

---

## Task 9: Full test run + final commit

**Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short -m "not slow"`
Expected: All pass (existing + new LOPO tests).

**Step 2: Final commit**

```bash
git add src/speech_decoding/training/lopo_trainer.py \
        src/speech_decoding/training/adaptor.py \
        src/speech_decoding/training/lopo.py \
        scripts/train_lopo.py \
        tests/test_lopo_trainer.py \
        tests/test_adaptor.py \
        tests/test_lopo.py \
        configs/default.yaml \
        src/speech_decoding/data/augmentation.py \
        src/speech_decoding/models/assembler.py \
        src/speech_decoding/training/trainer.py \
        docs/plans/
git commit -m "feat: add LOPO cross-patient training (Stage 1 + Stage 2 + orchestrator)"
```
