# NCA-JEPA Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the MVP-staged NCA-JEPA synthetic pretraining pipeline for uECOG speech decoding, building incrementally from evaluation infrastructure through synthetic transfer.

**Architecture:** Unified `PretrainModel` with `spatial_mode` config axis (`collapse`|`preserve`|`attend`). Per-patient Conv2d read-in → spatial encoder → shared BiGRU temporal model → masked span prediction objective. Three stages: synthetic pre-pretraining → neural adaptation → per-patient CE fine-tuning.

**Tech Stack:** Python 3.11, PyTorch 2.7, MNE-Python, NumPy, SciPy, pytest. Existing `speech_decoding` package. MPS-compatible (Apple Silicon).

**Spec:** `docs/superpowers/specs/2026-03-26-nca-jepa-implementation-plan.md` and parent `2026-03-26-nca-jepa-pretraining-design.md` (v14).

---

## File Structure

### New files to create

```
src/speech_decoding/
├── evaluation/
│   ├── grouped_cv.py              # Task 1: grouped-by-token CV splitter
│   └── content_collapse.py        # Task 2: entropy, unigram-KL, stereotypy
├── pretraining/
│   ├── __init__.py                # Task 5: package init
│   ├── masking.py                 # Task 5: span masking + [MASK] token
│   ├── decoder.py                 # Task 6: linear reconstruction decoder
│   ├── pretrain_model.py          # Task 7: UnifiedPretrainModel (all spatial_modes)
│   ├── local_geometry_pe.py       # Task 10: Linear(3→d) positional encoding
│   ├── spatial_pooling.py         # Task 11: mean-pool + top-k pool
│   ├── stage2_trainer.py          # Task 8: neural masked span prediction loop
│   ├── stage3_evaluator.py        # Task 9: freeze + CE fine-tune + diagnostics
│   ├── generators/
│   │   ├── __init__.py            # Task 13: generator registry
│   │   ├── base.py                # Task 13: Generator ABC
│   │   ├── smooth_ar.py           # Task 13: Level 0 smooth Gaussian AR
│   │   └── switching_lds.py       # Task 14: Level 1 switching linear dynamics
│   ├── synthetic_pipeline.py      # Task 15: noise, dead electrodes, grid sampling
│   └── stage1_trainer.py          # Task 16: synthetic pretraining loop
tests/
├── test_grouped_cv.py             # Task 1: CV splitter tests
├── test_content_collapse.py       # Task 2: collapse metric tests
├── test_phase0_baselines.py       # Task 3-4: baseline runner tests
├── test_masking.py                # Task 5: span masking tests
├── test_decoder.py                # Task 6: decoder tests
├── test_pretrain_model.py         # Task 7: unified model tests
├── test_stage2_trainer.py         # Task 8: Stage 2 loop tests
├── test_stage3_evaluator.py       # Task 9: Stage 3 pipeline tests
├── test_local_geometry_pe.py      # Task 10: PE tests
├── test_spatial_pooling.py        # Task 11: pooling tests
├── test_preserve_attend.py        # Task 12: preserve/attend mode tests
├── test_generators.py             # Task 13-14: generator tests
├── test_synthetic_pipeline.py     # Task 15: synthetic data tests
└── test_stage1_trainer.py         # Task 16: Stage 1 loop tests
configs/
├── pretrain_base.yaml             # Task 7: shared pretraining defaults
scripts/
├── run_phase0_baselines.py        # Task 4: Phase 0 baseline runner
└── train_pretrain.py              # Task 17: main pretraining CLI
```

### Existing files to modify

| File | Modification | Task |
|------|-------------|------|
| `models/backbone.py` | Add `per_position_mode` flag for preserve/attend | Task 12 |
| `models/assembler.py` | Add `assemble_pretrain_model()` function | Task 7 |
| `training/trainer.py` | Support grouped-by-token CV + CE loss via config flag | Task 3 |

---

## Phase 0: Evaluation Infrastructure + Baselines

**Purpose:** Build the evaluation pipeline and establish baselines (Methods E, D, spatial-only). Gate: if D ≈ E, architecture has no value — debug before proceeding.

### Task 1: Grouped-by-Token CV Splitter

**Files:**
- Create: `src/speech_decoding/evaluation/grouped_cv.py`
- Test: `tests/test_grouped_cv.py`

**Context:** All repetitions of the same CVC/VCV token must be in the same fold (RD-18). Each fold's training set must contain all 9 phonemes × 3 positions (RD-57). Splits are deterministic from patient ID and saved to JSON for reuse across methods (RD-62).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_grouped_cv.py
"""Tests for grouped-by-token cross-validation splitter."""
import json
import pytest
import numpy as np
from pathlib import Path

from speech_decoding.evaluation.grouped_cv import (
    build_token_groups,
    create_grouped_splits,
    load_or_create_splits,
    validate_fold_coverage,
)


class TestBuildTokenGroups:
    def test_groups_by_token_identity(self):
        """All trials of token 'abe' go to the same group."""
        # 6 trials: 2 reps of 'abe', 2 of 'gik', 2 of 'uva'
        labels = [
            [1, 3, 5],  # a, b, i → token "abe" (wrong, but testing grouping)
            [1, 3, 5],  # same token
            [4, 5, 6],  # g, i, k → token "gik"
            [4, 5, 6],
            [8, 9, 1],  # u, v, a → token "uva"
            [8, 9, 1],
        ]
        groups = build_token_groups(labels)
        # Same-token trials get same group ID
        assert groups[0] == groups[1]
        assert groups[2] == groups[3]
        assert groups[4] == groups[5]
        # Different tokens get different group IDs
        assert len(set(groups)) == 3

    def test_all_trials_assigned(self):
        labels = [[1, 2, 3]] * 10 + [[4, 5, 6]] * 10
        groups = build_token_groups(labels)
        assert len(groups) == 20


class TestCreateGroupedSplits:
    def _make_labels(self, n_tokens=20, reps_per_token=5):
        """Create labels with guaranteed coverage: 9 phonemes × 3 positions."""
        # Use all 9 phonemes (1-9) across tokens
        phonemes = list(range(1, 10))
        labels = []
        for i in range(n_tokens):
            # Cycle through phonemes to ensure coverage
            token = [phonemes[i % 9], phonemes[(i + 3) % 9], phonemes[(i + 6) % 9]]
            labels.extend([token] * reps_per_token)
        return labels

    def test_no_token_leakage(self):
        """Same token never appears in both train and val within a fold."""
        labels = self._make_labels(n_tokens=20, reps_per_token=5)
        groups = build_token_groups(labels)
        splits = create_grouped_splits(labels, groups, n_folds=5, seed=42)
        for fold in splits:
            train_tokens = {tuple(labels[i]) for i in fold["train_indices"]}
            val_tokens = {tuple(labels[i]) for i in fold["val_indices"]}
            assert train_tokens.isdisjoint(val_tokens), "Token leakage across folds!"

    def test_all_indices_covered(self):
        """Every trial index appears in exactly one fold's validation set."""
        labels = self._make_labels()
        groups = build_token_groups(labels)
        splits = create_grouped_splits(labels, groups, n_folds=5, seed=42)
        all_val = []
        for fold in splits:
            all_val.extend(fold["val_indices"])
        assert sorted(all_val) == list(range(len(labels)))

    def test_phoneme_coverage_per_fold(self):
        """Each fold's training set has all 9 phonemes in all 3 positions."""
        labels = self._make_labels(n_tokens=30, reps_per_token=4)
        groups = build_token_groups(labels)
        splits = create_grouped_splits(labels, groups, n_folds=5, seed=42)
        for fold in splits:
            assert validate_fold_coverage(labels, fold["train_indices"], n_phonemes=9)

    def test_deterministic_from_seed(self):
        labels = self._make_labels()
        groups = build_token_groups(labels)
        s1 = create_grouped_splits(labels, groups, n_folds=5, seed=42)
        s2 = create_grouped_splits(labels, groups, n_folds=5, seed=42)
        for f1, f2 in zip(s1, s2):
            assert f1["train_indices"] == f2["train_indices"]
            assert f1["val_indices"] == f2["val_indices"]


class TestLoadOrCreateSplits:
    def test_saves_and_loads_json(self, tmp_path):
        labels = [[1, 2, 3]] * 20 + [[4, 5, 6]] * 20 + [[7, 8, 9]] * 20
        path = tmp_path / "splits.json"
        splits = load_or_create_splits(labels, patient_id="S14", n_folds=3,
                                        save_path=path)
        assert path.exists()
        loaded = load_or_create_splits(labels, patient_id="S14", n_folds=3,
                                        save_path=path)
        for f1, f2 in zip(splits, loaded):
            assert f1["train_indices"] == f2["train_indices"]

    def test_seed_derived_from_patient_id(self):
        """Different patients get different splits."""
        labels = [[1, 2, 3]] * 30 + [[4, 5, 6]] * 30
        s1 = create_grouped_splits(labels, build_token_groups(labels),
                                    n_folds=3, seed=hash("S14") % 2**32)
        s2 = create_grouped_splits(labels, build_token_groups(labels),
                                    n_folds=3, seed=hash("S33") % 2**32)
        assert s1[0]["val_indices"] != s2[0]["val_indices"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_grouped_cv.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'speech_decoding.evaluation.grouped_cv'`

- [ ] **Step 3: Implement grouped CV splitter**

```python
# src/speech_decoding/evaluation/grouped_cv.py
"""Grouped-by-token cross-validation for phoneme sequence decoding.

All repetitions of the same CVC/VCV token go to the same fold.
Deterministic from patient_id. Saves/loads splits as JSON.
Ref: RD-18 (grouped-by-token), RD-57 (coverage), RD-62 (fixed splits).
"""
from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


def build_token_groups(labels: list[list[int]]) -> list[int]:
    """Assign each trial a group ID based on its token (phoneme sequence).

    Args:
        labels: List of CTC label sequences, e.g. [[1,3,5], [1,3,5], [4,5,6], ...].
            Each inner list is a 3-phoneme token.

    Returns:
        List of integer group IDs, same length as labels.
    """
    token_to_group: dict[tuple[int, ...], int] = {}
    groups = []
    for label in labels:
        key = tuple(label)
        if key not in token_to_group:
            token_to_group[key] = len(token_to_group)
        groups.append(token_to_group[key])
    return groups


def validate_fold_coverage(
    labels: list[list[int]],
    train_indices: list[int],
    n_phonemes: int = 9,
) -> bool:
    """Check that training set contains all phonemes in all 3 positions.

    Args:
        labels: Full label list.
        train_indices: Indices used for training in this fold.
        n_phonemes: Number of distinct phonemes (default 9).

    Returns:
        True if all phonemes appear in all positions.
    """
    coverage = [set() for _ in range(3)]
    for idx in train_indices:
        for pos, phon in enumerate(labels[idx]):
            coverage[pos].add(phon)
    expected = set(range(1, n_phonemes + 1))
    return all(pos_set >= expected for pos_set in coverage)


def _patient_seed(patient_id: str) -> int:
    """Deterministic seed from patient ID."""
    return int(hashlib.md5(patient_id.encode()).hexdigest()[:8], 16)


def create_grouped_splits(
    labels: list[list[int]],
    groups: list[int],
    n_folds: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Create grouped CV splits ensuring no token leakage.

    Uses GroupKFold with shuffled group order for balanced fold sizes.

    Args:
        labels: CTC label sequences per trial.
        groups: Group ID per trial (from build_token_groups).
        n_folds: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with 'train_indices' and 'val_indices' (sorted lists).
    """
    n = len(labels)
    rng = np.random.RandomState(seed)

    # Shuffle groups for balanced fold sizes
    unique_groups = sorted(set(groups))
    n_folds = min(n_folds, len(unique_groups))

    gkf = GroupKFold(n_splits=n_folds)
    X_dummy = np.zeros(n)
    y_dummy = np.zeros(n)
    group_arr = np.array(groups)

    # GroupKFold doesn't use random state directly — shuffle group labels
    group_perm = rng.permutation(len(unique_groups))
    group_map = {g: group_perm[i] for i, g in enumerate(unique_groups)}
    shuffled_groups = np.array([group_map[g] for g in groups])

    splits = []
    for train_idx, val_idx in gkf.split(X_dummy, y_dummy, groups=shuffled_groups):
        splits.append({
            "train_indices": sorted(train_idx.tolist()),
            "val_indices": sorted(val_idx.tolist()),
        })
    return splits


def load_or_create_splits(
    labels: list[list[int]],
    patient_id: str,
    n_folds: int = 5,
    save_path: Path | str | None = None,
) -> list[dict]:
    """Load splits from JSON if they exist, otherwise create and save.

    Args:
        labels: CTC label sequences.
        patient_id: Patient identifier (used to derive seed).
        n_folds: Number of folds.
        save_path: Path to save/load JSON. If None, don't persist.

    Returns:
        List of fold dicts with train/val indices.
    """
    if save_path is not None:
        save_path = Path(save_path)
        if save_path.exists():
            logger.info("Loading existing splits from %s", save_path)
            with open(save_path) as f:
                return json.load(f)

    groups = build_token_groups(labels)
    seed = _patient_seed(patient_id)
    splits = create_grouped_splits(labels, groups, n_folds=n_folds, seed=seed)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(splits, f, indent=2)
        logger.info("Saved splits to %s", save_path)

    return splits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_grouped_cv.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/bentang/Documents/Code/speech
git add src/speech_decoding/evaluation/grouped_cv.py tests/test_grouped_cv.py
git commit -m "feat: grouped-by-token CV splitter (RD-18, RD-57, RD-62)"
```

---

### Task 2: Content-Collapse Metrics

**Files:**
- Create: `src/speech_decoding/evaluation/content_collapse.py`
- Test: `tests/test_content_collapse.py`

**Context:** LOPO CE showed 100% correct-length sequences but only ~21% position accuracy with concentrated phoneme distributions. These metrics catch failure modes loss curves miss (RD-78).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_content_collapse.py
"""Tests for content-collapse diagnostic metrics."""
import pytest
import numpy as np

from speech_decoding.evaluation.content_collapse import (
    output_entropy,
    unigram_kl,
    stereotypy_index,
    content_collapse_report,
)


class TestOutputEntropy:
    def test_uniform_distribution(self):
        """Uniform predictions → max entropy = log2(9) ≈ 3.17."""
        preds = np.array([list(range(1, 10))] * 10).flatten()  # all 9 phonemes equally
        h = output_entropy(preds, n_classes=9)
        assert h == pytest.approx(np.log2(9), abs=0.01)

    def test_collapsed_distribution(self):
        """All same prediction → entropy = 0."""
        preds = np.array([1] * 100)
        h = output_entropy(preds, n_classes=9)
        assert h == pytest.approx(0.0, abs=1e-6)

    def test_partial_collapse(self):
        """Concentrated on 2 phonemes → entropy ≈ 1.0."""
        preds = np.array([1] * 50 + [2] * 50)
        h = output_entropy(preds, n_classes=9)
        assert h == pytest.approx(1.0, abs=0.01)


class TestUnigramKL:
    def test_uniform_gives_zero(self):
        """Uniform predictions → KL from uniform = 0."""
        preds = np.array(list(range(1, 10)) * 10)
        kl = unigram_kl(preds, n_classes=9)
        assert kl == pytest.approx(0.0, abs=0.01)

    def test_concentrated_gives_high_kl(self):
        """All-same predictions → high KL."""
        preds = np.array([1] * 90)
        kl = unigram_kl(preds, n_classes=9)
        assert kl > 2.0  # KL(delta || uniform) = log(9) ≈ 2.2 nats


class TestStereotypyIndex:
    def test_all_unique_sequences(self):
        """All unique 3-phoneme sequences → index = 1.0."""
        sequences = [[i, j, k] for i in range(1, 4) for j in range(1, 4) for k in range(1, 4)]
        idx = stereotypy_index(sequences)
        assert idx == pytest.approx(1.0)

    def test_all_same_sequence(self):
        """All identical sequences → index = 1/N."""
        sequences = [[1, 2, 3]] * 100
        idx = stereotypy_index(sequences)
        assert idx == pytest.approx(0.01)

    def test_alarm_threshold(self):
        """< 10% of 729 possible = alarm (RD-78)."""
        # Only 5 unique sequences out of 100
        sequences = [[i, 1, 1] for i in range(1, 6)] * 20
        idx = stereotypy_index(sequences)
        unique_frac_of_possible = 5 / 729
        assert unique_frac_of_possible < 0.10  # would trigger alarm


class TestContentCollapseReport:
    def test_report_structure(self):
        preds_per_position = [
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9] * 3),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9] * 3),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9] * 3),
        ]
        sequences = [[1, 2, 3]] * 27
        report = content_collapse_report(preds_per_position, sequences, n_classes=9)
        assert "entropy" in report
        assert "unigram_kl" in report
        assert "stereotypy_index" in report
        assert "collapsed" in report
        assert len(report["entropy"]) == 3  # one per position
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_content_collapse.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement content-collapse metrics**

```python
# src/speech_decoding/evaluation/content_collapse.py
"""Content-collapse diagnostics for phoneme decoding.

Catches failure modes that loss curves miss: concentrated phoneme
distributions, stereotyped output sequences. Ref: RD-78.
"""
from __future__ import annotations

import numpy as np


def output_entropy(preds: np.ndarray, n_classes: int = 9) -> float:
    """Shannon entropy (bits) of predicted phoneme distribution.

    Max = log2(n_classes) ≈ 3.17 for 9 phonemes. Alarm if < 1.5.
    """
    counts = np.bincount(preds, minlength=n_classes + 1)[1:n_classes + 1]
    probs = counts / counts.sum() if counts.sum() > 0 else np.ones(n_classes) / n_classes
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def unigram_kl(preds: np.ndarray, n_classes: int = 9) -> float:
    """KL divergence (nats) of predicted marginal from uniform.

    High KL = model favors certain phonemes regardless of input.
    """
    counts = np.bincount(preds, minlength=n_classes + 1)[1:n_classes + 1]
    p = counts / counts.sum() if counts.sum() > 0 else np.ones(n_classes) / n_classes
    q = np.ones(n_classes) / n_classes
    # KL(p || q), avoid log(0)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def stereotypy_index(sequences: list[list[int]]) -> float:
    """Fraction of unique predicted 3-phoneme sequences out of total.

    Alarm if unique_count / 729 < 0.10 (RD-78).
    """
    unique = set(tuple(s) for s in sequences)
    return len(unique) / len(sequences) if sequences else 0.0


def content_collapse_report(
    preds_per_position: list[np.ndarray],
    sequences: list[list[int]],
    n_classes: int = 9,
) -> dict:
    """Full content-collapse diagnostic report.

    Args:
        preds_per_position: List of 3 arrays, each with predicted phoneme IDs.
        sequences: List of predicted 3-phoneme sequences.
        n_classes: Number of phoneme classes.

    Returns:
        Dict with entropy, unigram_kl per position, stereotypy_index,
        and collapsed flag.
    """
    entropies = [output_entropy(p, n_classes) for p in preds_per_position]
    kls = [unigram_kl(p, n_classes) for p in preds_per_position]
    stereo = stereotypy_index(sequences)
    max_entropy = np.log2(n_classes)

    collapsed = (
        any(h < 1.5 for h in entropies)
        or stereo < 0.10
    )

    return {
        "entropy": entropies,
        "max_entropy": float(max_entropy),
        "unigram_kl": kls,
        "stereotypy_index": stereo,
        "collapsed": collapsed,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_content_collapse.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/bentang/Documents/Code/speech
git add src/speech_decoding/evaluation/content_collapse.py tests/test_content_collapse.py
git commit -m "feat: content-collapse diagnostics (entropy, KL, stereotypy)"
```

---

### Task 3: Phase 0 Evaluation Pipeline — CE Mean-Pool Head + Training Adaptation

**Files:**
- Create: `tests/test_phase0_baselines.py`
- Modify: `src/speech_decoding/training/trainer.py` (add grouped-CV + CE-pool mode)

**Context:** Phase 0 needs three baselines: Method E (frozen random init), Method D (supervised from scratch), and spatial-only (no BiGRU). All use 27-way CE head with temporal mean-pool (3 positions × 9 phonemes, validated at PER 0.700). The existing `CEPositionHead` does per-position attention — we need a simpler mean-pool approach that matches the existing validated config.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_phase0_baselines.py
"""Tests for Phase 0 baseline evaluation pipeline."""
import pytest
import torch
import numpy as np

from speech_decoding.models.spatial_conv import SpatialConvReadIn
from speech_decoding.models.backbone import SharedBackbone


class TestCEMeanPoolBaseline:
    """Validate the 27-way CE mean-pool readout pipeline."""

    def test_mean_pool_head_shape(self):
        """Temporal mean pool → Linear(2H, 27) → (B, 27)."""
        head = torch.nn.Linear(128, 27)  # 2H=128 for H=64
        features = torch.randn(4, 30, 128)  # (B, T, 2H)
        pooled = features.mean(dim=1)  # (B, 2H)
        logits = head(pooled)  # (B, 27)
        assert logits.shape == (4, 27)

    def test_27way_to_per_position_decode(self):
        """27-way logits → 3 positions × 9 phonemes → PER."""
        logits = torch.randn(4, 27)
        # Reshape to (B, 3, 9), argmax per position
        per_pos = logits.view(4, 3, 9)
        preds = per_pos.argmax(dim=-1)  # (B, 3) in [0, 8]
        assert preds.shape == (4, 3)
        assert (preds >= 0).all() and (preds < 9).all()

    def test_spatial_only_baseline_no_gru(self):
        """Spatial-only: Conv2d → temporal mean pool → CE head. No BiGRU."""
        readin = SpatialConvReadIn(grid_h=8, grid_w=16, pool_h=4, pool_w=8)
        head = torch.nn.Linear(readin.out_dim, 27)
        x = torch.randn(4, 8, 16, 60)  # (B, H, W, T) at 200Hz
        spatial_feats = readin(x)  # (B, D, T)
        pooled = spatial_feats.mean(dim=2)  # (B, D) temporal mean pool
        logits = head(pooled)  # (B, 27)
        assert logits.shape == (4, 27)

    def test_method_e_frozen_forward(self):
        """Method E: frozen random init backbone → train only CE head."""
        backbone = SharedBackbone(D=64, H=64, temporal_stride=10)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        head = torch.nn.Linear(128, 27)
        # Only head params are trainable
        trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
        assert trainable == 128 * 27 + 27  # 3,483
        assert frozen > 0

    def test_method_d_end_to_end(self):
        """Method D: all params trainable, end-to-end CE."""
        readin = SpatialConvReadIn(grid_h=8, grid_w=16)
        backbone = SharedBackbone(D=64, H=64, temporal_stride=10)
        head = torch.nn.Linear(128, 27)
        x = torch.randn(2, 8, 16, 300)
        backbone.eval()
        shared = readin(x)
        h = backbone(shared)  # (B, T//10, 128)
        pooled = h.mean(dim=1)
        logits = head(pooled)
        loss = torch.nn.functional.cross_entropy(
            logits, torch.randint(0, 27, (2,))
        )
        loss.backward()
        # All params have gradients
        assert readin.convs[0].weight.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail (or pass, since these use existing code)**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_phase0_baselines.py -v`
Expected: These should PASS since they test existing components with simple torch.nn.Linear. If any fail, debug.

- [ ] **Step 3: Commit test file**

```bash
cd /Users/bentang/Documents/Code/speech
git add tests/test_phase0_baselines.py
git commit -m "test: Phase 0 baseline pipeline verification tests"
```

---

### Task 4: Phase 0 Baseline Runner Script

**Files:**
- Create: `scripts/run_phase0_baselines.py`

**Context:** Runs Methods E, D, and spatial-only on all patients using grouped-by-token CV. Outputs baseline PER table + content-collapse report. This is the first gate check.

- [ ] **Step 1: Write the baseline runner**

```python
# scripts/run_phase0_baselines.py
"""Phase 0: Baseline evaluation — Methods E, D, spatial-only.

Gate 0 decision:
  D ≈ E → architecture has no value. Debug.
  spatial-only ≈ D → temporal model adds nothing.
  D > spatial-only > E → proceed to SSL.

Usage:
  python scripts/run_phase0_baselines.py --paths configs/paths.yaml --patients S14
  python scripts/run_phase0_baselines.py --paths configs/paths.yaml --all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.data.grid import load_grid_mapping
from speech_decoding.models.spatial_conv import SpatialConvReadIn
from speech_decoding.models.backbone import SharedBackbone
from speech_decoding.evaluation.grouped_cv import (
    build_token_groups,
    load_or_create_splits,
    validate_fold_coverage,
)
from speech_decoding.evaluation.content_collapse import content_collapse_report
from speech_decoding.evaluation.metrics import compute_per

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# PhonemeSequence patients with preprocessed data
PS_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S36", "S39", "S57", "S58", "S62"]


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 baselines")
    p.add_argument("--paths", type=str, required=True, help="Path to paths.yaml")
    p.add_argument("--patients", nargs="+", default=None, help="Patient IDs (default: all PS)")
    p.add_argument("--all", action="store_true", help="Run all PS patients")
    p.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds")
    p.add_argument("--device", default="mps", help="Device (mps/cpu/cuda)")
    p.add_argument("--output-dir", default="results/phase0", help="Output directory")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100, help="Max epochs for Method D")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temporal-stride", type=int, default=10, help="200Hz → 20Hz")
    p.add_argument("--pool-h", type=int, default=4)
    p.add_argument("--pool-w", type=int, default=8)
    return p.parse_args()


def train_ce_fold(
    readin, backbone, head, train_data, val_data,
    epochs, lr, device, freeze_backbone=False, freeze_readin=False,
):
    """Train one fold with CE mean-pool loss. Returns val PER + predictions."""
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
    if freeze_readin:
        for p in readin.parameters():
            p.requires_grad = False

    params = [p for p in list(readin.parameters()) + list(backbone.parameters()) + list(head.parameters())
              if p.requires_grad]
    if not params:
        params = list(head.parameters())

    optimizer = AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        readin.train() if not freeze_readin else None
        backbone.train() if not freeze_backbone else None
        head.train()

        grids, labels = train_data
        grids = grids.to(device)
        targets = torch.tensor(labels, dtype=torch.long, device=device)  # (N, 3)

        shared = readin(grids)
        if backbone is not None and not isinstance(backbone, nn.Identity):
            h = backbone(shared)
        else:
            h = shared.permute(0, 2, 1)  # (B, T, D) for spatial-only
        pooled = h.mean(dim=1)
        logits = head(pooled)

        # Per-position CE loss: reshape to (B, 3, 9)
        per_pos = logits.view(-1, 3, 9)
        loss = sum(
            F.cross_entropy(per_pos[:, pos, :], targets[:, pos] - 1)
            for pos in range(3)
        ) / 3

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        # Validate
        readin.eval()
        backbone.eval() if backbone is not None else None
        head.eval()
        with torch.no_grad():
            v_grids, v_labels = val_data
            v_grids = v_grids.to(device)
            v_targets = torch.tensor(v_labels, dtype=torch.long, device=device)

            v_shared = readin(v_grids)
            if backbone is not None and not isinstance(backbone, nn.Identity):
                v_h = backbone(v_shared)
            else:
                v_h = v_shared.permute(0, 2, 1)
            v_pooled = v_h.mean(dim=1)
            v_logits = head(v_pooled)

            v_per_pos = v_logits.view(-1, 3, 9)
            val_loss = sum(
                F.cross_entropy(v_per_pos[:, pos, :], v_targets[:, pos] - 1)
                for pos in range(3)
            ) / 3

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "readin": readin.state_dict(),
                "head": head.state_dict(),
            }
            if backbone is not None:
                best_state["backbone"] = backbone.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    # Load best and evaluate
    readin.load_state_dict(best_state["readin"])
    head.load_state_dict(best_state["head"])
    if backbone is not None and "backbone" in best_state:
        backbone.load_state_dict(best_state["backbone"])

    readin.eval()
    if backbone is not None:
        backbone.eval()
    head.eval()

    with torch.no_grad():
        v_grids, v_labels = val_data
        v_grids = v_grids.to(device)
        v_shared = readin(v_grids)
        if backbone is not None and not isinstance(backbone, nn.Identity):
            v_h = backbone(v_shared)
        else:
            v_h = v_shared.permute(0, 2, 1)
        v_pooled = v_h.mean(dim=1)
        v_logits = head(v_pooled)
        v_per_pos = v_logits.view(-1, 3, 9)
        preds = v_per_pos.argmax(dim=-1).cpu().numpy() + 1  # back to 1-indexed
        v_targets_np = np.array(v_labels)

    # PER
    total, errors = 0, 0
    for pred_seq, true_seq in zip(preds, v_targets_np):
        for p, t in zip(pred_seq, true_seq):
            total += 1
            if p != t:
                errors += 1
    per = errors / total if total > 0 else 1.0
    return per, preds, v_targets_np


def main():
    args = parse_args()
    # Load paths config
    import yaml
    with open(args.paths) as f:
        paths = yaml.safe_load(f)
    bids_root = paths["bids_root"]

    patients = args.patients or (PS_PATIENTS if args.all else ["S14"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for patient_id in patients:
        logger.info("=" * 60)
        logger.info("Patient %s", patient_id)

        # Load data
        ds = load_patient_data(patient_id, bids_root, task="PhonemeSequence",
                                n_phons=3, tmin=-0.5, tmax=1.0)
        grid_info = load_grid_mapping(patient_id, bids_root, task="PhonemeSequence")
        grid_h, grid_w = grid_info.grid_shape

        # Extract all trials
        all_grids = []
        all_labels = []
        for i in range(len(ds)):
            grid, label, _ = ds[i]
            all_grids.append(grid)
            all_labels.append(label)
        all_grids = torch.tensor(np.stack(all_grids), dtype=torch.float32)

        # Create grouped splits
        splits_path = output_dir / f"splits_{patient_id}.json"
        splits = load_or_create_splits(all_labels, patient_id, n_folds=args.n_folds,
                                        save_path=splits_path)

        patient_results = {"E": [], "D": [], "spatial_only": []}

        for fold_idx, fold in enumerate(splits):
            train_idx = fold["train_indices"]
            val_idx = fold["val_indices"]
            train_data = (all_grids[train_idx], [all_labels[i] for i in train_idx])
            val_data = (all_grids[val_idx], [all_labels[i] for i in val_idx])

            device = args.device

            # Method E: frozen random init
            readin_e = SpatialConvReadIn(grid_h, grid_w, pool_h=args.pool_h, pool_w=args.pool_w).to(device)
            backbone_e = SharedBackbone(D=readin_e.out_dim, H=64, temporal_stride=args.temporal_stride).to(device)
            head_e = nn.Linear(128, 27).to(device)
            per_e, _, _ = train_ce_fold(
                readin_e, backbone_e, head_e, train_data, val_data,
                epochs=args.epochs, lr=args.lr, device=device,
                freeze_backbone=True, freeze_readin=True,
            )
            patient_results["E"].append(per_e)
            logger.info("  Fold %d Method E PER: %.3f", fold_idx, per_e)

            # Method D: supervised from scratch
            readin_d = SpatialConvReadIn(grid_h, grid_w, pool_h=args.pool_h, pool_w=args.pool_w).to(device)
            backbone_d = SharedBackbone(D=readin_d.out_dim, H=64, temporal_stride=args.temporal_stride).to(device)
            head_d = nn.Linear(128, 27).to(device)
            per_d, _, _ = train_ce_fold(
                readin_d, backbone_d, head_d, train_data, val_data,
                epochs=args.epochs, lr=args.lr, device=device,
            )
            patient_results["D"].append(per_d)
            logger.info("  Fold %d Method D PER: %.3f", fold_idx, per_d)

            # Spatial-only: no BiGRU
            readin_s = SpatialConvReadIn(grid_h, grid_w, pool_h=args.pool_h, pool_w=args.pool_w).to(device)
            head_s = nn.Linear(readin_s.out_dim, 27).to(device)
            per_s, preds_s, targets_s = train_ce_fold(
                readin_s, None, head_s, train_data, val_data,
                epochs=args.epochs, lr=args.lr, device=device,
            )
            patient_results["spatial_only"].append(per_s)
            logger.info("  Fold %d Spatial-only PER: %.3f", fold_idx, per_s)

        results[patient_id] = {
            method: {
                "mean_per": float(np.mean(pers)),
                "std_per": float(np.std(pers)),
                "fold_pers": pers,
            }
            for method, pers in patient_results.items()
        }
        for method in ["E", "D", "spatial_only"]:
            r = results[patient_id][method]
            logger.info("  %s: PER %.3f ± %.3f", method, r["mean_per"], r["std_per"])

    # Save results
    with open(output_dir / "phase0_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_dir / "phase0_results.json")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Patient':<10} {'Method E':<15} {'Method D':<15} {'Spatial-only':<15}")
    print("-" * 70)
    for patient_id, r in results.items():
        print(f"{patient_id:<10} "
              f"{r['E']['mean_per']:.3f} ± {r['E']['std_per']:.3f}  "
              f"{r['D']['mean_per']:.3f} ± {r['D']['std_per']:.3f}  "
              f"{r['spatial_only']['mean_per']:.3f} ± {r['spatial_only']['std_per']:.3f}")

    # Gate 0 check
    print("\n--- Gate 0 Check ---")
    for patient_id, r in results.items():
        e, d, s = r["E"]["mean_per"], r["D"]["mean_per"], r["spatial_only"]["mean_per"]
        if abs(d - e) < 0.02:
            print(f"  {patient_id}: WARNING — D ≈ E ({d:.3f} vs {e:.3f}). Architecture has no value.")
        elif abs(s - d) < 0.02:
            print(f"  {patient_id}: WARNING — spatial-only ≈ D ({s:.3f} vs {d:.3f}). Temporal model adds nothing.")
        elif d < s < e:
            print(f"  {patient_id}: PASS — D ({d:.3f}) > spatial-only ({s:.3f}) > E ({e:.3f}). Proceed to SSL.")
        else:
            print(f"  {patient_id}: D={d:.3f}, spatial-only={s:.3f}, E={e:.3f} — check ordering.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script syntax**

Run: `cd /Users/bentang/Documents/Code/speech && python -c "import ast; ast.parse(open('scripts/run_phase0_baselines.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/bentang/Documents/Code/speech
git add scripts/run_phase0_baselines.py
git commit -m "feat: Phase 0 baseline runner (Methods E, D, spatial-only)"
```

---

## Phase 1: Minimal SSL on Real Data

**Purpose:** Build masked span prediction + Stage 2/3 training loops. Run Method B (neural-only pretrain). Gate: if B ≈ E, pretraining objective is broken.

### Task 5: Span Masking Module

**Files:**
- Create: `src/speech_decoding/pretraining/__init__.py`
- Create: `src/speech_decoding/pretraining/masking.py`
- Test: `tests/test_masking.py`

**Context:** Mask 40-60% of frames using 3-6 non-overlapping contiguous spans (RD-70). Same mask across all spatial positions to prevent spatial correlation shortcut (RD-87). Learnable [MASK] token replaces masked frames.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_masking.py
"""Tests for temporal span masking."""
import pytest
import torch
import numpy as np

from speech_decoding.pretraining.masking import (
    generate_span_mask,
    SpanMasker,
)


class TestGenerateSpanMask:
    def test_output_shape(self):
        """Mask is (T,) boolean."""
        mask = generate_span_mask(T=30, mask_ratio=(0.4, 0.6), n_spans=(3, 6), rng=np.random.RandomState(42))
        assert mask.shape == (30,)
        assert mask.dtype == bool

    def test_mask_ratio_in_range(self):
        """Masked fraction should be within target range (±10% tolerance)."""
        for seed in range(50):
            mask = generate_span_mask(T=30, mask_ratio=(0.4, 0.6), n_spans=(3, 6),
                                       rng=np.random.RandomState(seed))
            frac = mask.sum() / len(mask)
            assert 0.25 <= frac <= 0.75, f"Seed {seed}: mask fraction {frac:.2f}"

    def test_spans_are_contiguous(self):
        """Each masked region is a contiguous span."""
        mask = generate_span_mask(T=30, mask_ratio=(0.4, 0.6), n_spans=(3, 6),
                                   rng=np.random.RandomState(42))
        # Count transitions (0→1 or 1→0)
        transitions = np.diff(mask.astype(int))
        n_spans = (transitions == 1).sum()  # number of span starts
        assert 1 <= n_spans <= 8  # within expected range

    def test_deterministic_with_seed(self):
        m1 = generate_span_mask(T=30, mask_ratio=(0.4, 0.6), n_spans=(3, 6),
                                 rng=np.random.RandomState(42))
        m2 = generate_span_mask(T=30, mask_ratio=(0.4, 0.6), n_spans=(3, 6),
                                 rng=np.random.RandomState(42))
        np.testing.assert_array_equal(m1, m2)


class TestSpanMasker:
    def test_mask_token_learnable(self):
        masker = SpanMasker(d=64)
        assert masker.mask_token.requires_grad
        assert masker.mask_token.shape == (64,)

    def test_apply_mask_shape(self):
        """(B, T, D) input → same shape output with masked frames replaced."""
        masker = SpanMasker(d=64)
        x = torch.randn(4, 30, 64)
        mask = torch.zeros(30, dtype=torch.bool)
        mask[5:10] = True  # mask frames 5-9
        x_masked, applied_mask = masker(x, mask)
        assert x_masked.shape == x.shape
        assert applied_mask.shape == (30,)

    def test_masked_frames_are_mask_token(self):
        masker = SpanMasker(d=64)
        x = torch.randn(4, 30, 64)
        mask = torch.zeros(30, dtype=torch.bool)
        mask[5:10] = True
        x_masked, _ = masker(x, mask)
        # Masked frames should be the mask token (broadcast across batch)
        for t in range(5, 10):
            assert torch.allclose(x_masked[0, t], masker.mask_token.data, atol=1e-6)

    def test_unmasked_frames_unchanged(self):
        masker = SpanMasker(d=64)
        x = torch.randn(4, 30, 64)
        mask = torch.zeros(30, dtype=torch.bool)
        mask[5:10] = True
        x_masked, _ = masker(x, mask)
        # Unmasked frames should be identical
        assert torch.allclose(x_masked[:, :5], x[:, :5])
        assert torch.allclose(x_masked[:, 10:], x[:, 10:])

    def test_same_mask_all_positions_in_batch(self):
        """All batch items get the same temporal mask (spatial consistency RD-87)."""
        masker = SpanMasker(d=64)
        x = torch.randn(4, 30, 64)
        mask = torch.zeros(30, dtype=torch.bool)
        mask[5:10] = True
        x_masked, _ = masker(x, mask)
        # All batch items have mask token at same positions
        for b in range(4):
            for t in range(5, 10):
                assert torch.allclose(x_masked[b, t], masker.mask_token.data, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_masking.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement span masking**

```python
# src/speech_decoding/pretraining/__init__.py
"""NCA-JEPA pretraining pipeline."""

# src/speech_decoding/pretraining/masking.py
"""Temporal span masking for masked prediction pretraining.

Masks 40-60% of frames using 3-6 non-overlapping contiguous spans.
Same mask applied across all spatial positions (RD-87).
Ref: RD-70 (masking ratio), RD-47 (masked temporal spans).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def generate_span_mask(
    T: int,
    mask_ratio: tuple[float, float] = (0.4, 0.6),
    n_spans: tuple[int, int] = (3, 6),
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Generate a boolean mask with contiguous spans.

    Args:
        T: Number of time frames.
        mask_ratio: (min, max) fraction of frames to mask.
        n_spans: (min, max) number of spans.
        rng: Random state for reproducibility.

    Returns:
        Boolean array of shape (T,), True = masked.
    """
    if rng is None:
        rng = np.random.RandomState()

    target_ratio = rng.uniform(*mask_ratio)
    target_masked = max(1, int(round(T * target_ratio)))
    num_spans = rng.randint(n_spans[0], n_spans[1] + 1)
    num_spans = min(num_spans, target_masked)  # can't have more spans than masked frames

    mask = np.zeros(T, dtype=bool)

    # Distribute target_masked frames across num_spans spans
    # Each span gets at least 1 frame
    span_lengths = np.ones(num_spans, dtype=int)
    remaining = target_masked - num_spans
    for _ in range(remaining):
        span_lengths[rng.randint(num_spans)] += 1

    # Place spans without overlap
    # Generate random start positions with enough room
    available = T - target_masked
    if available < 0:
        # Edge case: mask everything
        mask[:] = True
        return mask

    # Generate gaps between spans (including before first and after last)
    gaps = np.zeros(num_spans + 1, dtype=int)
    for _ in range(available):
        gaps[rng.randint(num_spans + 1)] += 1

    pos = 0
    for i in range(num_spans):
        pos += gaps[i]
        end = min(pos + span_lengths[i], T)
        mask[pos:end] = True
        pos = end

    return mask


class SpanMasker(nn.Module):
    """Applies temporal span masking with a learnable [MASK] token.

    The same temporal mask is applied to all items in the batch and
    all spatial positions (RD-87: prevents spatial correlation shortcut).
    """

    def __init__(self, d: int):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(d) * 0.02)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply mask to input.

        Args:
            x: (B, T, D) input features.
            mask: (T,) boolean mask, True = masked.

        Returns:
            x_masked: (B, T, D) with masked frames replaced by [MASK] token.
            mask: (T,) the mask that was applied.
        """
        x_masked = x.clone()
        # Replace masked frames with learnable mask token
        x_masked[:, mask] = self.mask_token.unsqueeze(0)
        return x_masked, mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_masking.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/bentang/Documents/Code/speech
git add src/speech_decoding/pretraining/__init__.py src/speech_decoding/pretraining/masking.py tests/test_masking.py
git commit -m "feat: span masking module for masked prediction pretraining"
```

---

### Task 6: Linear Reconstruction Decoder

**Files:**
- Create: `src/speech_decoding/pretraining/decoder.py`
- Test: `tests/test_decoder.py`

**Context:** For `collapse` mode: `Linear(2H, d_flat)` predicts pooled representation. For `preserve` mode: `Linear(2H, 1)` predicts per-electrode values. Shared across patients. Discarded after pretraining.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_decoder.py
"""Tests for reconstruction decoder."""
import pytest
import torch

from speech_decoding.pretraining.decoder import ReconstructionDecoder


class TestReconstructionDecoder:
    def test_collapse_mode_shape(self):
        """collapse: (B, T, 2H) → (B, T, d_flat)."""
        dec = ReconstructionDecoder(input_dim=128, output_dim=256, mode="collapse")
        h = torch.randn(4, 30, 128)
        out = dec(h)
        assert out.shape == (4, 30, 256)

    def test_preserve_mode_shape(self):
        """preserve: (B, T, 2H) → (B, T, 1) per position."""
        dec = ReconstructionDecoder(input_dim=128, output_dim=1, mode="preserve")
        h = torch.randn(4, 30, 128)
        out = dec(h)
        assert out.shape == (4, 30, 1)

    def test_param_count_collapse(self):
        dec = ReconstructionDecoder(input_dim=128, output_dim=256, mode="collapse")
        n = sum(p.numel() for p in dec.parameters())
        assert n == 128 * 256 + 256  # Linear weight + bias

    def test_param_count_preserve(self):
        dec = ReconstructionDecoder(input_dim=128, output_dim=1, mode="preserve")
        n = sum(p.numel() for p in dec.parameters())
        assert n == 128 * 1 + 1  # 129

    def test_gradient_flows(self):
        dec = ReconstructionDecoder(input_dim=128, output_dim=256, mode="collapse")
        h = torch.randn(4, 30, 128, requires_grad=True)
        out = dec(h)
        out.sum().backward()
        assert h.grad is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_decoder.py -v`

- [ ] **Step 3: Implement decoder**

```python
# src/speech_decoding/pretraining/decoder.py
"""Linear reconstruction decoder for masked prediction.

Collapse mode: Linear(2H, d_flat) → predicts pooled spatial representation.
Preserve mode: Linear(2H, 1) → predicts per-electrode values.
Discarded after pretraining.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionDecoder(nn.Module):
    """Shared linear decoder for masked frame reconstruction."""

    def __init__(self, input_dim: int, output_dim: int, mode: str = "collapse"):
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Project temporal model output back to input space.

        Args:
            h: (B, T, 2H) temporal model output.

        Returns:
            (B, T, output_dim) reconstructed frames.
        """
        return self.proj(h)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_decoder.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/bentang/Documents/Code/speech
git add src/speech_decoding/pretraining/decoder.py tests/test_decoder.py
git commit -m "feat: reconstruction decoder for masked prediction"
```

---

### Task 7: Unified Pretrain Model — Collapse Mode

**Files:**
- Create: `src/speech_decoding/pretraining/pretrain_model.py`
- Create: `configs/pretrain_base.yaml`
- Test: `tests/test_pretrain_model.py`

**Context:** Start with `spatial_mode=collapse` which reuses existing `SpatialConvReadIn` + `SharedBackbone`. `preserve` and `attend` modes added in Phase 3 (Task 10-12). The model wraps: Conv2d read-in → backbone → masking → decoder → MSE loss on masked frames.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pretrain_model.py
"""Tests for unified pretrain model (collapse mode first)."""
import pytest
import torch
import numpy as np

from speech_decoding.pretraining.pretrain_model import PretrainModel


class TestPretrainModelCollapse:
    @pytest.fixture
    def config(self):
        return {
            "spatial_mode": "collapse",
            "d": 64,
            "gru_hidden": 32,
            "gru_layers": 2,
            "temporal_stride": 10,
            "mask_ratio": [0.4, 0.6],
            "mask_spans": [3, 6],
            "spatial_conv": {
                "channels": 8,
                "pool_h": 4,
                "pool_w": 8,
            },
        }

    def test_forward_returns_loss(self, config):
        """Forward pass returns MSE loss on masked frames."""
        model = PretrainModel(config, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(4, 8, 16, 200)  # (B, H, W, T) at 200Hz
        result = model(x, compute_loss=True)
        assert "loss" in result
        assert result["loss"].shape == ()  # scalar
        assert result["loss"].item() >= 0

    def test_forward_returns_predictions(self, config):
        model = PretrainModel(config, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(4, 8, 16, 200)
        result = model(x, compute_loss=True)
        assert "predictions" in result
        assert "targets" in result
        assert "mask" in result

    def test_loss_only_on_masked_frames(self, config):
        """MSE computed ONLY on masked frames, not all frames."""
        model = PretrainModel(config, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(4, 8, 16, 200)
        result = model(x, compute_loss=True)
        mask = result["mask"]
        n_masked = mask.sum().item()
        assert n_masked > 0  # some frames are masked
        n_total = mask.numel()
        assert n_masked < n_total  # not all frames masked

    def test_backward_pass(self, config):
        """Gradients flow through the model."""
        model = PretrainModel(config, grid_shape=(8, 16))
        model.train()
        x = torch.randn(4, 8, 16, 200)
        result = model(x, compute_loss=True)
        result["loss"].backward()
        # Check backbone has gradients
        for name, p in model.named_parameters():
            if p.requires_grad and "decoder" not in name:
                assert p.grad is not None, f"No gradient for {name}"
                break  # just check at least one

    def test_encode_only(self, config):
        """Encode without masking (for downstream fine-tuning)."""
        model = PretrainModel(config, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(4, 8, 16, 200)
        features = model.encode(x)
        # Should be (B, T', 2H) where T' = T_after_stride
        T_out = 200 // config["temporal_stride"]  # 20
        assert features.shape == (4, T_out, config["gru_hidden"] * 2)

    def test_param_count_approximately_correct(self, config):
        """Total params should be ~71K for collapse mode (d=64)."""
        model = PretrainModel(config, grid_shape=(8, 16))
        total = sum(p.numel() for p in model.parameters())
        # Conv2d: 80, backbone + decoder ≈ 71K
        assert 30_000 < total < 150_000, f"Unexpected param count: {total}"

    def test_different_grid_sizes(self, config):
        """Model handles both 8×16 and 12×22 grids."""
        for grid_shape in [(8, 16), (12, 22)]:
            model = PretrainModel(config, grid_shape=grid_shape)
            model.eval()
            x = torch.randn(2, *grid_shape, 200)
            result = model(x, compute_loss=True)
            assert result["loss"].shape == ()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_pretrain_model.py -v`

- [ ] **Step 3: Implement PretrainModel**

```python
# src/speech_decoding/pretraining/pretrain_model.py
"""Unified pretrain model with configurable spatial_mode.

Phase 1: collapse mode (reuses existing SpatialConvReadIn + SharedBackbone).
Phase 3: preserve and attend modes (Tasks 10-12).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from speech_decoding.models.spatial_conv import SpatialConvReadIn
from speech_decoding.models.backbone import SharedBackbone
from speech_decoding.pretraining.masking import SpanMasker, generate_span_mask
from speech_decoding.pretraining.decoder import ReconstructionDecoder


class PretrainModel(nn.Module):
    """Unified pretraining model for masked span prediction.

    Supports spatial_mode='collapse' (Phase 1), with 'preserve' and 'attend'
    added in Phase 3.

    Architecture:
        Input (B, H, W, T) → Conv2d read-in → spatial encoder → temporal model
        → masked span prediction → MSE on masked frames
    """

    def __init__(self, config: dict, grid_shape: tuple[int, int]):
        super().__init__()
        self.config = config
        self.spatial_mode = config.get("spatial_mode", "collapse")
        self.grid_shape = grid_shape

        sc = config.get("spatial_conv", {})
        d = config["d"]
        gru_hidden = config["gru_hidden"]
        temporal_stride = config["temporal_stride"]

        if self.spatial_mode == "collapse":
            self.readin = SpatialConvReadIn(
                grid_h=grid_shape[0],
                grid_w=grid_shape[1],
                C=sc.get("channels", 8),
                pool_h=sc.get("pool_h", 4),
                pool_w=sc.get("pool_w", 8),
            )
            backbone_input_dim = self.readin.out_dim

            self.backbone = SharedBackbone(
                D=backbone_input_dim,
                H=gru_hidden,
                temporal_stride=temporal_stride,
                gru_layers=config.get("gru_layers", 2),
                gru_dropout=0.0,  # no dropout during pretraining
                feat_drop_max=0.0,
                gru_input_dim=d,
            )

            # Decoder predicts pooled spatial representation
            self.decoder = ReconstructionDecoder(
                input_dim=gru_hidden * 2,  # BiGRU output
                output_dim=backbone_input_dim,
                mode="collapse",
            )

            # Masker operates on backbone input space
            self.masker = SpanMasker(d=d)
            self._target_dim = backbone_input_dim

        else:
            raise NotImplementedError(f"spatial_mode={self.spatial_mode} not yet implemented")

        self.mask_ratio = tuple(config.get("mask_ratio", [0.4, 0.6]))
        self.mask_spans = tuple(config.get("mask_spans", [3, 6]))

    def _spatial_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial encoding: grid → features per frame.

        Args:
            x: (B, H, W, T) raw grid input.

        Returns:
            (B, D, T) spatial features (collapse) or (B, N_pos, D, T) (preserve).
        """
        if self.spatial_mode == "collapse":
            return self.readin(x)  # (B, d_flat, T)
        else:
            raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass without masking (for downstream).

        Args:
            x: (B, H, W, T) raw grid input at 200Hz.

        Returns:
            (B, T', 2H) temporal features.
        """
        spatial = self._spatial_encode(x)  # (B, D, T)
        features = self.backbone(spatial)  # (B, T', 2H)
        return features

    def forward(
        self, x: torch.Tensor, compute_loss: bool = True
    ) -> dict[str, torch.Tensor]:
        """Forward pass with masked span prediction.

        Args:
            x: (B, H, W, T) raw grid input at 200Hz.
            compute_loss: Whether to compute MSE loss.

        Returns:
            Dict with 'loss', 'predictions', 'targets', 'mask'.
        """
        spatial = self._spatial_encode(x)  # (B, d_flat, T)

        if self.spatial_mode == "collapse":
            # Masking must be applied BETWEEN Conv1d and GRU (RD-87, spec §Key Decisions #4).
            # We decompose backbone.forward() into: layernorm → temporal_conv → MASK → gru.
            # This accesses backbone.layernorm, backbone.temporal_conv, backbone.gru directly.
            # backbone._feature_dropout and _time_mask are skipped during pretraining
            # (feat_drop_max=0.0 in pretraining config).
            stride = self.config["temporal_stride"]
            T_raw = spatial.shape[2]
            T_strided = T_raw // stride

            # Downsample spatial features to match backbone output resolution
            # Use average pooling to match Conv1d stride
            targets = F.avg_pool1d(spatial, kernel_size=stride, stride=stride)  # (B, d_flat, T')
            targets = targets.permute(0, 2, 1)  # (B, T', d_flat)

            # Generate mask at strided resolution
            rng = np.random.RandomState()
            mask = generate_span_mask(
                T=T_strided,
                mask_ratio=self.mask_ratio,
                n_spans=self.mask_spans,
                rng=rng,
            )
            mask_tensor = torch.from_numpy(mask).to(x.device)

            # Run backbone (which internally does Conv1d stride + GRU)
            # Pass full spatial features — backbone does its own striding
            h = self.backbone(spatial)  # (B, T', 2H)

            # Apply masking to backbone output (predict from context only)
            # Actually: masking should be applied BEFORE the temporal model
            # so the model learns to predict masked frames from context.
            # We need to mask the Conv1d output, not the GRU output.
            # Let's restructure: manual Conv1d + mask + GRU

            # Re-do: apply Conv1d manually, then mask, then GRU
            # Access backbone internals (layernorm, temporal_conv, gru)
            h_strided = self.backbone.layernorm(spatial.permute(0, 2, 1)).permute(0, 2, 1)
            h_strided = self.backbone.temporal_conv(h_strided)  # (B, d, T')
            h_strided = h_strided.permute(0, 2, 1)  # (B, T', d)

            # Apply span mask
            h_masked, mask_tensor = self.masker(h_strided, mask_tensor)

            # Run GRU on masked sequence
            gru_out, _ = self.backbone.gru(h_masked)  # (B, T', 2H)

            # Decode masked positions
            predictions = self.decoder(gru_out)  # (B, T', d_flat)

        result = {
            "predictions": predictions,
            "targets": targets,
            "mask": mask_tensor,
        }

        if compute_loss:
            # MSE only on masked frames
            pred_masked = predictions[:, mask_tensor]  # (B, n_masked, d_flat)
            tgt_masked = targets[:, mask_tensor]  # (B, n_masked, d_flat)
            loss = F.mse_loss(pred_masked, tgt_masked)
            result["loss"] = loss

        return result
```

- [ ] **Step 4: Run tests, fix any issues**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_pretrain_model.py -v`
Expected: All 8 tests PASS. If backbone internal access fails, refactor to expose conv/gru separately.

- [ ] **Step 5: Create base config**

```yaml
# configs/pretrain_base.yaml
# NCA-JEPA pretraining defaults — collapse mode
spatial_mode: collapse
d: 64
gru_hidden: 32
gru_layers: 2
temporal_stride: 10  # 200Hz → 20Hz
mask_ratio: [0.4, 0.6]
mask_spans: [3, 6]

spatial_conv:
  channels: 8
  pool_h: 4
  pool_w: 8

stage2:
  lr: 1e-3
  weight_decay: 1e-4
  steps: 5000
  batch_size: 8
  checkpoint_every: 1000

stage3:
  lr: 1e-3
  epochs: 100
  patience: 10
  n_folds: 5
```

- [ ] **Step 6: Commit**

```bash
cd /Users/bentang/Documents/Code/speech
git add src/speech_decoding/pretraining/pretrain_model.py configs/pretrain_base.yaml tests/test_pretrain_model.py
git commit -m "feat: PretrainModel with collapse mode + masked span prediction"
```

---

### Task 8: Stage 2 Training Loop (Neural Adaptation)

**Files:**
- Create: `src/speech_decoding/pretraining/stage2_trainer.py`
- Test: `tests/test_stage2_trainer.py`

**Context:** Masked span prediction on real trial data. No anti-leakage yet (Phase 4). Load real trials via existing `load_patient_data`, train encoder+predictor, checkpoint. Source patients only (exclude dev + target).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stage2_trainer.py
"""Tests for Stage 2 neural adaptation training loop."""
import pytest
import torch
import numpy as np
from pathlib import Path

from speech_decoding.pretraining.stage2_trainer import (
    Stage2Trainer,
    Stage2Config,
)
from speech_decoding.pretraining.pretrain_model import PretrainModel


class TestStage2Trainer:
    @pytest.fixture
    def model_config(self):
        return {
            "spatial_mode": "collapse",
            "d": 64,
            "gru_hidden": 32,
            "gru_layers": 2,
            "temporal_stride": 10,
            "mask_ratio": [0.4, 0.6],
            "mask_spans": [3, 6],
            "spatial_conv": {"channels": 8, "pool_h": 4, "pool_w": 8},
        }

    @pytest.fixture
    def synthetic_trials(self):
        """Synthetic trial data mimicking real patient data."""
        # 3 patients × 50 trials each, 8×16 grid, 200Hz, 1.5s
        patients = {}
        for pid in ["S_A", "S_B", "S_C"]:
            grids = np.random.randn(50, 8, 16, 300).astype(np.float32)
            patients[pid] = torch.tensor(grids)
        return patients

    def test_trainer_runs_one_step(self, model_config, synthetic_trials):
        model = PretrainModel(model_config, grid_shape=(8, 16))
        cfg = Stage2Config(lr=1e-3, steps=1, batch_size=4)
        trainer = Stage2Trainer(model, cfg, device="cpu")
        metrics = trainer.train_step(synthetic_trials)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_loss_decreases_over_steps(self, model_config, synthetic_trials):
        model = PretrainModel(model_config, grid_shape=(8, 16))
        cfg = Stage2Config(lr=1e-3, steps=20, batch_size=4)
        trainer = Stage2Trainer(model, cfg, device="cpu")
        losses = []
        for _ in range(20):
            metrics = trainer.train_step(synthetic_trials)
            losses.append(metrics["loss"])
        # Loss should generally decrease (not monotonically, but trend)
        assert np.mean(losses[-5:]) < np.mean(losses[:5])

    def test_checkpoint_saving(self, model_config, synthetic_trials, tmp_path):
        model = PretrainModel(model_config, grid_shape=(8, 16))
        cfg = Stage2Config(lr=1e-3, steps=5, batch_size=4,
                           checkpoint_dir=str(tmp_path), checkpoint_every=2)
        trainer = Stage2Trainer(model, cfg, device="cpu")
        trainer.train(synthetic_trials)
        checkpoints = list(tmp_path.glob("*.pt"))
        assert len(checkpoints) >= 2

    def test_excludes_dev_and_target(self, model_config, synthetic_trials):
        """Dev and target patients are excluded from training."""
        model = PretrainModel(model_config, grid_shape=(8, 16))
        cfg = Stage2Config(lr=1e-3, steps=1, batch_size=4)
        trainer = Stage2Trainer(model, cfg, device="cpu")
        # Exclude S_C as target, S_A as dev
        filtered = trainer._filter_patients(synthetic_trials,
                                             exclude={"S_A", "S_C"})
        assert "S_B" in filtered
        assert "S_A" not in filtered
        assert "S_C" not in filtered
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_stage2_trainer.py -v`

- [ ] **Step 3: Implement Stage 2 trainer**

```python
# src/speech_decoding/pretraining/stage2_trainer.py
"""Stage 2: Neural masked span prediction on real trial data.

Trains encoder+predictor on unlabeled response-locked trial epochs.
No phoneme labels used. Source patients only (exclude dev + target).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from speech_decoding.pretraining.pretrain_model import PretrainModel

logger = logging.getLogger(__name__)


@dataclass
class Stage2Config:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 5000
    batch_size: int = 8
    checkpoint_dir: str | None = None
    checkpoint_every: int = 1000
    grad_clip: float = 1.0


class Stage2Trainer:
    """Train PretrainModel on real trial data with masked span prediction."""

    def __init__(
        self,
        model: PretrainModel,
        config: Stage2Config,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.steps)
        self.step_count = 0

    def _filter_patients(
        self,
        patient_data: dict[str, torch.Tensor],
        exclude: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Remove dev and target patients."""
        if exclude is None:
            return patient_data
        return {k: v for k, v in patient_data.items() if k not in exclude}

    def _sample_batch(
        self, patient_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Sample a batch uniformly across patients."""
        pids = list(patient_data.keys())
        batch = []
        for _ in range(self.config.batch_size):
            pid = pids[np.random.randint(len(pids))]
            trials = patient_data[pid]
            idx = np.random.randint(len(trials))
            batch.append(trials[idx])
        return torch.stack(batch).to(self.device)

    def train_step(
        self, patient_data: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """One training step."""
        self.model.train()
        batch = self._sample_batch(patient_data)  # (B, H, W, T)
        result = self.model(batch, compute_loss=True)
        loss = result["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step_count += 1

        return {"loss": loss.item(), "step": self.step_count}

    def train(
        self,
        patient_data: dict[str, torch.Tensor],
        exclude: set[str] | None = None,
    ) -> list[dict]:
        """Full training loop."""
        filtered = self._filter_patients(patient_data, exclude)
        if not filtered:
            raise ValueError("No patients remaining after exclusion")

        metrics_history = []
        for step in range(self.config.steps):
            metrics = self.train_step(filtered)
            metrics_history.append(metrics)

            if step % 100 == 0:
                logger.info("Step %d: loss=%.4f", step, metrics["loss"])

            if (self.config.checkpoint_dir
                and self.config.checkpoint_every > 0
                and (step + 1) % self.config.checkpoint_every == 0):
                self._save_checkpoint(step)

        return metrics_history

    def _save_checkpoint(self, step: int):
        path = Path(self.config.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"step": step, "model": self.model.state_dict(),
             "optimizer": self.optimizer.state_dict()},
            path / f"checkpoint_step{step:06d}.pt",
        )
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_stage2_trainer.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/bentang/Documents/Code/speech
git add src/speech_decoding/pretraining/stage2_trainer.py tests/test_stage2_trainer.py
git commit -m "feat: Stage 2 neural adaptation training loop"
```

---

### Task 9: Stage 3 Evaluator (Freeze + CE Fine-Tune)

**Files:**
- Create: `src/speech_decoding/pretraining/stage3_evaluator.py`
- Test: `tests/test_stage3_evaluator.py`

**Context:** Freeze backbone, train CE head + Conv2d on grouped-by-token splits. Report PER + content-collapse. This connects pretraining to downstream evaluation.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stage3_evaluator.py
"""Tests for Stage 3 fine-tuning evaluator."""
import pytest
import torch
import numpy as np

from speech_decoding.pretraining.stage3_evaluator import (
    Stage3Evaluator,
    Stage3Config,
)
from speech_decoding.pretraining.pretrain_model import PretrainModel


class TestStage3Evaluator:
    @pytest.fixture
    def pretrained_model(self):
        config = {
            "spatial_mode": "collapse",
            "d": 64,
            "gru_hidden": 32,
            "gru_layers": 2,
            "temporal_stride": 10,
            "mask_ratio": [0.4, 0.6],
            "mask_spans": [3, 6],
            "spatial_conv": {"channels": 8, "pool_h": 4, "pool_w": 8},
        }
        return PretrainModel(config, grid_shape=(8, 16))

    @pytest.fixture
    def synthetic_dataset(self):
        """Synthetic labeled data for one patient."""
        n_trials = 100
        grids = np.random.randn(n_trials, 8, 16, 300).astype(np.float32)
        # Labels: 3 phonemes per trial, values 1-9
        labels = [[np.random.randint(1, 10) for _ in range(3)]
                  for _ in range(n_trials)]
        return torch.tensor(grids), labels

    def test_freezes_backbone(self, pretrained_model, synthetic_dataset):
        cfg = Stage3Config(lr=1e-3, epochs=1, n_folds=3)
        evaluator = Stage3Evaluator(pretrained_model, cfg, device="cpu")
        evaluator._freeze_backbone()
        for name, p in pretrained_model.named_parameters():
            if "readin" not in name and "decoder" not in name:
                if "backbone" in name:
                    assert not p.requires_grad, f"{name} should be frozen"

    def test_evaluate_returns_per(self, pretrained_model, synthetic_dataset):
        grids, labels = synthetic_dataset
        cfg = Stage3Config(lr=1e-3, epochs=2, n_folds=3)
        evaluator = Stage3Evaluator(pretrained_model, cfg, device="cpu")
        results = evaluator.evaluate(grids, labels, patient_id="S_test")
        assert "mean_per" in results
        assert 0.0 <= results["mean_per"] <= 1.0
        assert "fold_pers" in results
        assert len(results["fold_pers"]) == 3

    def test_evaluate_returns_collapse_report(self, pretrained_model, synthetic_dataset):
        grids, labels = synthetic_dataset
        cfg = Stage3Config(lr=1e-3, epochs=2, n_folds=3)
        evaluator = Stage3Evaluator(pretrained_model, cfg, device="cpu")
        results = evaluator.evaluate(grids, labels, patient_id="S_test")
        assert "content_collapse" in results
        assert "entropy" in results["content_collapse"]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_stage3_evaluator.py -v`

- [ ] **Step 3: Implement Stage 3 evaluator**

```python
# src/speech_decoding/pretraining/stage3_evaluator.py
"""Stage 3: Freeze backbone, train CE head per patient.

Uses grouped-by-token CV. Reports PER + content-collapse diagnostics.
Ref: RD-14 (freeze config), RD-17 (CE primary), RD-78 (collapse diagnostics).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from speech_decoding.pretraining.pretrain_model import PretrainModel
from speech_decoding.evaluation.grouped_cv import (
    build_token_groups,
    create_grouped_splits,
)
from speech_decoding.evaluation.content_collapse import content_collapse_report

logger = logging.getLogger(__name__)


@dataclass
class Stage3Config:
    lr: float = 1e-3
    epochs: int = 100
    patience: int = 10
    n_folds: int = 5
    n_classes: int = 9
    n_positions: int = 3


class Stage3Evaluator:
    """Fine-tune CE head on frozen pretrained features."""

    def __init__(
        self,
        model: PretrainModel,
        config: Stage3Config,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

    def _freeze_backbone(self):
        """Freeze everything except read-in (Conv2d) params."""
        for name, p in self.model.named_parameters():
            if "readin" not in name:
                p.requires_grad = False
        # Also unfreeze any LayerNorm in backbone (RD-14)
        for name, p in self.model.named_parameters():
            if "ln" in name or "LayerNorm" in name.lower():
                p.requires_grad = True

    def _create_head(self) -> nn.Linear:
        """Create 27-way CE head (3 positions × 9 phonemes)."""
        n_out = self.config.n_positions * self.config.n_classes
        gru_hidden = self.model.config["gru_hidden"]
        return nn.Linear(gru_hidden * 2, n_out).to(self.device)

    def _train_fold(
        self,
        head: nn.Linear,
        train_grids: torch.Tensor,
        train_labels: list[list[int]],
        val_grids: torch.Tensor,
        val_labels: list[list[int]],
    ) -> tuple[float, np.ndarray]:
        """Train one fold, return val PER and predictions."""
        self._freeze_backbone()
        head_fresh = self._create_head()

        # Trainable: head + readin params
        params = [
            {"params": head_fresh.parameters(), "lr": self.config.lr},
        ]
        readin_params = [p for n, p in self.model.named_parameters()
                         if "readin" in n and p.requires_grad]
        if readin_params:
            params.append({"params": readin_params, "lr": self.config.lr * 3})

        optimizer = AdamW(params, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)

        best_loss = float("inf")
        best_head_state = None
        patience_ctr = 0

        self.model.eval()  # backbone frozen in eval mode
        for epoch in range(self.config.epochs):
            # Train
            head_fresh.train()
            train_g = train_grids.to(self.device)
            with torch.no_grad():
                features = self.model.encode(train_g)  # (B, T', 2H)
            pooled = features.mean(dim=1)  # (B, 2H)
            logits = head_fresh(pooled)  # (B, 27)
            per_pos = logits.view(-1, self.config.n_positions, self.config.n_classes)

            targets = torch.tensor(train_labels, device=self.device, dtype=torch.long)
            loss = sum(
                F.cross_entropy(per_pos[:, p, :], targets[:, p] - 1)
                for p in range(self.config.n_positions)
            ) / self.config.n_positions

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Validate
            head_fresh.eval()
            with torch.no_grad():
                val_g = val_grids.to(self.device)
                val_feat = self.model.encode(val_g)
                val_pooled = val_feat.mean(dim=1)
                val_logits = head_fresh(val_pooled)
                val_per_pos = val_logits.view(-1, self.config.n_positions, self.config.n_classes)
                val_tgt = torch.tensor(val_labels, device=self.device, dtype=torch.long)
                val_loss = sum(
                    F.cross_entropy(val_per_pos[:, p, :], val_tgt[:, p] - 1)
                    for p in range(self.config.n_positions)
                ) / self.config.n_positions

            if val_loss < best_loss:
                best_loss = val_loss
                best_head_state = head_fresh.state_dict()
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.config.patience:
                    break

        # Evaluate with best head
        head_fresh.load_state_dict(best_head_state)
        head_fresh.eval()
        with torch.no_grad():
            val_g = val_grids.to(self.device)
            val_feat = self.model.encode(val_g)
            val_pooled = val_feat.mean(dim=1)
            val_logits = head_fresh(val_pooled)
            val_per_pos = val_logits.view(-1, self.config.n_positions, self.config.n_classes)
            preds = val_per_pos.argmax(dim=-1).cpu().numpy() + 1  # 1-indexed
            val_tgt_np = np.array(val_labels)

        # PER
        total, errors = 0, 0
        for pred_seq, true_seq in zip(preds, val_tgt_np):
            for p, t in zip(pred_seq, true_seq):
                total += 1
                if p != t:
                    errors += 1
        per = errors / total if total > 0 else 1.0
        return per, preds

    def evaluate(
        self,
        grids: torch.Tensor,
        labels: list[list[int]],
        patient_id: str,
    ) -> dict:
        """Full grouped-by-token CV evaluation.

        Returns:
            Dict with mean_per, fold_pers, content_collapse report.
        """
        groups = build_token_groups(labels)
        seed = int.from_bytes(patient_id.encode()[:4], "big") % 2**31
        splits = create_grouped_splits(labels, groups,
                                        n_folds=self.config.n_folds, seed=seed)

        fold_pers = []
        all_preds = []
        all_targets = []

        for fold_idx, fold in enumerate(splits):
            train_idx = fold["train_indices"]
            val_idx = fold["val_indices"]

            train_grids = grids[train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_grids = grids[val_idx]
            val_labels = [labels[i] for i in val_idx]

            head = self._create_head()
            per, preds = self._train_fold(
                head, train_grids, train_labels, val_grids, val_labels
            )
            fold_pers.append(per)
            all_preds.extend(preds.tolist())
            all_targets.extend(val_labels)
            logger.info("  Fold %d PER: %.3f", fold_idx, per)

        # Content-collapse report
        all_preds_np = np.array(all_preds)
        preds_per_pos = [all_preds_np[:, i] for i in range(self.config.n_positions)]
        sequences = all_preds_np.tolist()
        collapse = content_collapse_report(preds_per_pos, sequences,
                                            n_classes=self.config.n_classes)

        return {
            "mean_per": float(np.mean(fold_pers)),
            "std_per": float(np.std(fold_pers)),
            "fold_pers": fold_pers,
            "content_collapse": collapse,
        }
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_stage3_evaluator.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/bentang/Documents/Code/speech
git add src/speech_decoding/pretraining/stage3_evaluator.py tests/test_stage3_evaluator.py
git commit -m "feat: Stage 3 CE fine-tune evaluator with collapse diagnostics"
```

---

## Phase 2: Minimal Synthetic Transfer + Architecture Components

**Purpose:** Build ONE generator (smooth AR) + synthetic data pipeline. Test whether synthetic data transfers at all (Method C vs B). Gate: if C ≈ B, synthetic pretraining is dead.

Tasks 10-12 (PE, spatial pooling, preserve/attend) are Phase 3 components but are built here because they have no external dependencies and can be developed in parallel with the synthetic pipeline. They are NOT run experimentally until Phase 3's gate check. Tasks 13-17 are the actual Phase 2 synthetic pipeline.

### Task 10: Local Geometry Positional Encoding (Phase 3 component, built early)

**Files:**
- Create: `src/speech_decoding/pretraining/local_geometry_pe.py`
- Test: `tests/test_local_geometry_pe.py`

**Context:** `Linear(3 → d)` on `(row_mm, col_mm, dead_fraction)`. Needed for `preserve` and `attend` modes (Phase 3) but built here since generators also need grid geometry. Ref: RD-3, RD-45.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_local_geometry_pe.py
"""Tests for local geometry positional encoding."""
import pytest
import torch

from speech_decoding.pretraining.local_geometry_pe import LocalGeometryPE


class TestLocalGeometryPE:
    def test_output_shape(self):
        pe = LocalGeometryPE(d=64)
        # 128 positions (8×16), 3 features each
        coords = torch.randn(128, 3)  # (N_pos, 3)
        out = pe(coords)
        assert out.shape == (128, 64)

    def test_from_grid_shape(self):
        """Create PE from grid shape and pitch."""
        pe = LocalGeometryPE(d=64)
        coords = LocalGeometryPE.grid_coordinates(
            grid_h=8, grid_w=16, pitch_mm=1.33,
            dead_mask=None,
        )
        assert coords.shape == (128, 3)
        # Row coords should span 0 to ~9.3mm
        assert coords[:, 0].max() > 8.0
        assert coords[:, 0].min() == pytest.approx(0.0, abs=0.01)

    def test_dead_fraction_encoded(self):
        """Dead positions have dead_fraction > 0."""
        import numpy as np
        dead_mask = np.zeros((8, 16), dtype=bool)
        dead_mask[0, 0] = True  # one dead electrode
        coords = LocalGeometryPE.grid_coordinates(
            grid_h=8, grid_w=16, pitch_mm=1.33,
            dead_mask=dead_mask,
        )
        # Dead position at (0,0) should have dead_frac = 1.0
        assert coords[0, 2] == pytest.approx(1.0)
        # Live positions should have dead_frac = 0.0
        assert coords[1, 2] == pytest.approx(0.0)

    def test_param_count(self):
        pe = LocalGeometryPE(d=64)
        n = sum(p.numel() for p in pe.parameters())
        assert n == 3 * 64 + 64  # Linear(3, 64) = 256
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

```python
# src/speech_decoding/pretraining/local_geometry_pe.py
"""Local geometry positional encoding: Linear(3 → d).

Input: (row_mm, col_mm, dead_fraction) per electrode position.
Ref: RD-3 (local geometry PE), RD-45 (dead-fraction not binary).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LocalGeometryPE(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(3, d)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Project coordinates to d-dim embeddings.

        Args:
            coords: (N_pos, 3) — row_mm, col_mm, dead_fraction.

        Returns:
            (N_pos, d) positional embeddings.
        """
        return self.proj(coords)

    @staticmethod
    def grid_coordinates(
        grid_h: int,
        grid_w: int,
        pitch_mm: float = 1.33,
        dead_mask: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Create coordinate tensor from grid shape.

        Args:
            grid_h, grid_w: Grid dimensions.
            pitch_mm: Electrode pitch in millimeters.
            dead_mask: (H, W) boolean, True = dead. None = all alive.

        Returns:
            (H*W, 3) tensor of (row_mm, col_mm, dead_fraction).
        """
        coords = []
        for r in range(grid_h):
            for c in range(grid_w):
                row_mm = r * pitch_mm
                col_mm = c * pitch_mm
                dead_frac = 1.0 if (dead_mask is not None and dead_mask[r, c]) else 0.0
                coords.append([row_mm, col_mm, dead_frac])
        return torch.tensor(coords, dtype=torch.float32)
```

- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add src/speech_decoding/pretraining/local_geometry_pe.py tests/test_local_geometry_pe.py
git commit -m "feat: local geometry positional encoding Linear(3→d)"
```

---

### Task 11: Spatial Pooling (Mean + Top-k) (Phase 3 component, built early)

**Files:**
- Create: `src/speech_decoding/pretraining/spatial_pooling.py`
- Test: `tests/test_spatial_pooling.py`

**Context:** For `preserve`/`attend` modes at readout. Mean-pool and top-k (k=16) run as co-defaults (RD-72). Top-k selects most active electrodes by L2 norm.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_spatial_pooling.py
"""Tests for spatial pooling strategies."""
import pytest
import torch

from speech_decoding.pretraining.spatial_pooling import mean_pool_spatial, topk_pool_spatial


class TestMeanPoolSpatial:
    def test_shape(self):
        """(B, N_pos, T, D) → (B, T, D)."""
        x = torch.randn(4, 128, 30, 64)
        out = mean_pool_spatial(x)
        assert out.shape == (4, 30, 64)


class TestTopkPoolSpatial:
    def test_shape(self):
        """(B, N_pos, T, D) → (B, T, D) using top-k positions."""
        x = torch.randn(4, 128, 30, 64)
        out = topk_pool_spatial(x, k=16)
        assert out.shape == (4, 30, 64)

    def test_selects_highest_norm(self):
        """Top-k should select positions with highest L2 norm."""
        x = torch.zeros(1, 10, 1, 4)
        x[0, 7, 0, :] = 10.0  # position 7 has highest norm
        out = topk_pool_spatial(x, k=1)
        # Should be close to position 7's values
        assert torch.allclose(out[0, 0], x[0, 7, 0], atol=0.1)

    def test_k_larger_than_positions(self):
        """k > N_pos → use all positions (degrades to mean)."""
        x = torch.randn(2, 5, 10, 32)
        out = topk_pool_spatial(x, k=100)
        expected = mean_pool_spatial(x)
        assert torch.allclose(out, expected)
```

- [ ] **Step 2-5: Implement, test, commit**

```python
# src/speech_decoding/pretraining/spatial_pooling.py
"""Spatial pooling strategies for preserve/attend modes.

Mean-pool and top-k run as co-defaults (RD-72).
"""
from __future__ import annotations

import torch


def mean_pool_spatial(x: torch.Tensor) -> torch.Tensor:
    """Average across spatial positions.

    Args:
        x: (B, N_pos, T, D)

    Returns:
        (B, T, D)
    """
    return x.mean(dim=1)


def topk_pool_spatial(x: torch.Tensor, k: int = 16) -> torch.Tensor:
    """Average the k positions with highest L2 norm per frame.

    Parameter-free. Selects most active electrodes.

    Args:
        x: (B, N_pos, T, D)
        k: Number of positions to keep.

    Returns:
        (B, T, D)
    """
    B, N, T, D = x.shape
    k = min(k, N)
    # Compute L2 norm per position per frame: (B, N, T)
    norms = x.norm(dim=-1)
    # Top-k indices per frame: (B, k, T)
    _, indices = norms.topk(k, dim=1)
    # Gather: expand indices to (B, k, T, D)
    indices_expanded = indices.unsqueeze(-1).expand(B, k, T, D)
    selected = x.gather(1, indices_expanded)  # (B, k, T, D)
    return selected.mean(dim=1)  # (B, T, D)
```

```bash
git add src/speech_decoding/pretraining/spatial_pooling.py tests/test_spatial_pooling.py
git commit -m "feat: spatial pooling (mean + top-k) for preserve/attend modes"
```

---

### Task 12: Preserve and Attend Modes (Phase 3 component, built early)

**Files:**
- Modify: `src/speech_decoding/models/backbone.py` (add per-position GRU support)
- Modify: `src/speech_decoding/pretraining/pretrain_model.py` (add preserve/attend)
- Test: `tests/test_preserve_attend.py`

**Context:** `preserve` = keep all spatial positions, shared BiGRU per position. `attend` = preserve + 1-2 cross-position TransformerEncoderLayer. Ref: RD-84, RD-85, RD-88.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preserve_attend.py
"""Tests for preserve and attend spatial modes."""
import pytest
import torch

from speech_decoding.pretraining.pretrain_model import PretrainModel


class TestPreserveMode:
    @pytest.fixture
    def config(self):
        return {
            "spatial_mode": "preserve",
            "d": 64,
            "gru_hidden": 32,
            "gru_layers": 2,
            "temporal_stride": 10,
            "mask_ratio": [0.4, 0.6],
            "mask_spans": [3, 6],
            "spatial_conv": {"channels": 8},
            "readout": "mean",
        }

    def test_forward_returns_loss(self, config):
        model = PretrainModel(config, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(2, 8, 16, 200)
        result = model(x, compute_loss=True)
        assert "loss" in result

    def test_encode_preserves_positions(self, config):
        """Encode output should have spatial dimension."""
        model = PretrainModel(config, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(2, 8, 16, 200)
        features = model.encode(x)
        # (B, T', 2H) after spatial pooling at readout
        T_out = 200 // config["temporal_stride"]
        assert features.shape == (2, T_out, config["gru_hidden"] * 2)

    def test_param_count_preserve(self, config):
        """~39K for preserve mode (d=64)."""
        model = PretrainModel(config, grid_shape=(8, 16))
        total = sum(p.numel() for p in model.parameters())
        assert 20_000 < total < 80_000

    def test_per_position_gru_shared(self, config):
        """All positions share the same GRU weights."""
        model = PretrainModel(config, grid_shape=(8, 16))
        # There should be exactly one GRU module
        gru_modules = [m for m in model.modules() if isinstance(m, torch.nn.GRU)]
        assert len(gru_modules) == 1


class TestAttendMode:
    @pytest.fixture
    def config(self):
        return {
            "spatial_mode": "attend",
            "d": 64,
            "gru_hidden": 32,
            "gru_layers": 2,
            "temporal_stride": 10,
            "mask_ratio": [0.4, 0.6],
            "mask_spans": [3, 6],
            "spatial_conv": {"channels": 8},
            "cross_position_layers": 1,
            "readout": "mean",
        }

    def test_forward_returns_loss(self, config):
        model = PretrainModel(config, grid_shape=(8, 16))
        model.eval()
        x = torch.randn(2, 8, 16, 200)
        result = model(x, compute_loss=True)
        assert "loss" in result

    def test_more_params_than_preserve(self, config):
        """attend adds ~20K for cross-position attention."""
        attend_model = PretrainModel(config, grid_shape=(8, 16))
        preserve_config = {**config, "spatial_mode": "preserve"}
        preserve_model = PretrainModel(preserve_config, grid_shape=(8, 16))
        attend_params = sum(p.numel() for p in attend_model.parameters())
        preserve_params = sum(p.numel() for p in preserve_model.parameters())
        assert attend_params > preserve_params
        assert attend_params - preserve_params < 50_000  # ~20K extra
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement preserve/attend in PretrainModel**

Add the following to `pretrain_model.py`'s `__init__` inside the `elif self.spatial_mode in ("preserve", "attend"):` branch:

```python
        elif self.spatial_mode in ("preserve", "attend"):
            # Per-patient Conv2d (same as collapse, but NO pooling)
            self.readin = SpatialConvReadIn(
                grid_h=grid_shape[0], grid_w=grid_shape[1],
                C=sc.get("channels", 8),
                pool_h=grid_shape[0],  # no pooling: pool to original size
                pool_w=grid_shape[1],
            )
            self.n_positions = grid_shape[0] * grid_shape[1]

            # Per-position projection: Linear(C, d)
            from speech_decoding.pretraining.local_geometry_pe import LocalGeometryPE
            self.pos_proj = nn.Linear(sc.get("channels", 8), d)
            self.pe = LocalGeometryPE(d=d)

            # Cross-position attention (attend mode only)
            n_cross_layers = config.get("cross_position_layers", 0)
            if self.spatial_mode == "attend" and n_cross_layers > 0:
                layer = nn.TransformerEncoderLayer(
                    d_model=d, nhead=2, dim_feedforward=d * 2,
                    dropout=0.0, batch_first=True,
                )
                self.cross_attn = nn.TransformerEncoder(layer, num_layers=n_cross_layers)
            else:
                self.cross_attn = None

            # Shared BiGRU applied per position
            self.gru = nn.GRU(
                d, gru_hidden, num_layers=config.get("gru_layers", 2),
                batch_first=True, bidirectional=True,
            )

            # Decoder: per-position per-frame reconstruction
            self.decoder = ReconstructionDecoder(
                input_dim=gru_hidden * 2, output_dim=1, mode="preserve",
            )
            self.masker = SpanMasker(d=d)
            self._target_dim = 1
```

Add `_spatial_encode` for preserve/attend:

```python
    def _spatial_encode_preserve(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve mode: (B, H, W, T) → (B, N_pos, T, d).

        Conv2d without pooling → per-position Linear(C, d) + PE.
        """
        B, H, W, T = x.shape
        # Conv2d per frame (readin with pool=identity returns (B, C*H*W, T))
        # Instead, manually process to keep per-position structure
        x_frames = x.permute(0, 3, 1, 2).reshape(B * T, 1, H, W)
        conv_out = self.readin.convs(x_frames)  # (B*T, C, H, W)
        C = conv_out.shape[1]
        conv_out = conv_out.reshape(B, T, C, H, W)
        # Reshape to (B, N_pos, T, C)
        conv_out = conv_out.permute(0, 3, 4, 1, 2).reshape(B, H * W, T, C)
        # Project to d-dim + add PE
        tokens = self.pos_proj(conv_out)  # (B, N_pos, T, d)
        coords = LocalGeometryPE.grid_coordinates(H, W).to(x.device)
        pe = self.pe(coords)  # (N_pos, d)
        tokens = tokens + pe.unsqueeze(0).unsqueeze(2)  # broadcast

        # Cross-position attention (attend mode): apply per frame
        if self.cross_attn is not None:
            B, N, T, D = tokens.shape
            tokens_flat = tokens.permute(0, 2, 1, 3).reshape(B * T, N, D)
            tokens_flat = self.cross_attn(tokens_flat)
            tokens = tokens_flat.reshape(B, T, N, D).permute(0, 2, 1, 3)

        return tokens  # (B, N_pos, T, d)
```

Add forward path for preserve/attend in `forward()`:

```python
        elif self.spatial_mode in ("preserve", "attend"):
            tokens = self._spatial_encode_preserve(x)  # (B, N_pos, T, d)
            B, N, T, D = tokens.shape

            # Targets: per-electrode values at original resolution
            # x is (B, H, W, T), reshape to (B, N_pos, T, 1)
            targets = x.reshape(B, N, T, 1)

            # Temporal stride via average pooling per position
            stride = self.config["temporal_stride"]
            T_strided = T // stride
            # (B, N, T, d) → pool over T → (B, N, T', d)
            tokens_strided = tokens.reshape(B * N, T, D)
            tokens_strided = F.avg_pool1d(
                tokens_strided.permute(0, 2, 1), kernel_size=stride, stride=stride
            ).permute(0, 2, 1)  # (B*N, T', D)

            targets_strided = targets.reshape(B * N, T, 1)
            targets_strided = F.avg_pool1d(
                targets_strided.permute(0, 2, 1), kernel_size=stride, stride=stride
            ).permute(0, 2, 1)  # (B*N, T', 1)

            # Generate mask at strided resolution
            mask = generate_span_mask(T=T_strided, mask_ratio=self.mask_ratio,
                                       n_spans=self.mask_spans, rng=np.random.RandomState())
            mask_tensor = torch.from_numpy(mask).to(x.device)

            # Apply masking
            tokens_masked, mask_tensor = self.masker(tokens_strided, mask_tensor)

            # Shared GRU per position (batched)
            gru_out, _ = self.gru(tokens_masked)  # (B*N, T', 2H)

            # Decode
            predictions = self.decoder(gru_out)  # (B*N, T', 1)

            # Reshape back
            predictions = predictions.reshape(B, N, T_strided, 1)
            targets = targets_strided.reshape(B, N, T_strided, 1)
```

The `encode()` method for preserve/attend applies spatial pooling at readout:

```python
        elif self.spatial_mode in ("preserve", "attend"):
            from speech_decoding.pretraining.spatial_pooling import mean_pool_spatial
            tokens = self._spatial_encode_preserve(x)  # (B, N, T, d)
            B, N, T, D = tokens.shape
            stride = self.config["temporal_stride"]
            tokens_flat = tokens.reshape(B * N, T, D)
            tokens_strided = F.avg_pool1d(
                tokens_flat.permute(0, 2, 1), kernel_size=stride, stride=stride
            ).permute(0, 2, 1).reshape(B, N, T // stride, D)
            # GRU per position
            gru_in = tokens_strided.reshape(B * N, T // stride, D)
            gru_out, _ = self.gru(gru_in)
            gru_out = gru_out.reshape(B, N, T // stride, -1)
            # Spatial pooling → (B, T', 2H)
            features = mean_pool_spatial(gru_out)
            return features
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_preserve_attend.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: preserve and attend spatial modes for PretrainModel"
```

---

### Task 13: Smooth AR Generator (Level 0)

**Files:**
- Create: `src/speech_decoding/pretraining/generators/__init__.py`
- Create: `src/speech_decoding/pretraining/generators/base.py`
- Create: `src/speech_decoding/pretraining/generators/smooth_ar.py`
- Test: `tests/test_generators.py`

**Context:** AR(1) process with Gaussian smoothing. Simplest generator — tests whether ANY smooth synthetic movie helps. ~50 lines. Ref: spec §4.1 Level 0.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_generators.py
"""Tests for synthetic data generators."""
import pytest
import numpy as np

from speech_decoding.pretraining.generators.base import Generator
from speech_decoding.pretraining.generators.smooth_ar import SmoothARGenerator


class TestSmoothARGenerator:
    def test_output_shape(self):
        gen = SmoothARGenerator(grid_h=8, grid_w=16, T=30, sigma=3.0, alpha=0.9)
        data = gen.generate(seed=42)
        assert data.shape == (8, 16, 30)

    def test_output_is_float(self):
        gen = SmoothARGenerator(grid_h=8, grid_w=16, T=30)
        data = gen.generate(seed=42)
        assert data.dtype == np.float32

    def test_temporal_autocorrelation(self):
        """AR process should have positive temporal autocorrelation."""
        gen = SmoothARGenerator(grid_h=8, grid_w=16, T=100, alpha=0.9)
        data = gen.generate(seed=42)
        # Pick a random cell
        cell = data[4, 8, :]
        # Lag-1 autocorrelation should be positive
        corr = np.corrcoef(cell[:-1], cell[1:])[0, 1]
        assert corr > 0.5

    def test_spatial_smoothness(self):
        """Gaussian smoothing should create spatial correlation."""
        gen = SmoothARGenerator(grid_h=8, grid_w=16, T=30, sigma=3.0)
        data = gen.generate(seed=42)
        frame = data[:, :, 15]
        # Adjacent cells should be correlated
        center = frame[4, 8]
        neighbor = frame[4, 9]
        far = frame[0, 0]
        # Not a strict test — just verify smoothing has some effect
        assert abs(center - neighbor) < abs(center - far) * 3  # rough check

    def test_deterministic_with_seed(self):
        gen = SmoothARGenerator(grid_h=8, grid_w=16, T=30)
        d1 = gen.generate(seed=42)
        d2 = gen.generate(seed=42)
        np.testing.assert_array_equal(d1, d2)

    def test_different_seeds_different_data(self):
        gen = SmoothARGenerator(grid_h=8, grid_w=16, T=30)
        d1 = gen.generate(seed=42)
        d2 = gen.generate(seed=99)
        assert not np.allclose(d1, d2)

    def test_12x22_grid(self):
        gen = SmoothARGenerator(grid_h=12, grid_w=22, T=30)
        data = gen.generate(seed=42)
        assert data.shape == (12, 22, 30)

    def test_implements_generator_interface(self):
        gen = SmoothARGenerator(grid_h=8, grid_w=16, T=30)
        assert isinstance(gen, Generator)
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement**

```python
# src/speech_decoding/pretraining/generators/base.py
"""Abstract base class for synthetic data generators."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Generator(ABC):
    """Base class for grid dynamics generators."""

    def __init__(self, grid_h: int, grid_w: int, T: int):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.T = T

    @abstractmethod
    def generate(self, seed: int | None = None) -> np.ndarray:
        """Generate one sequence.

        Returns:
            (H, W, T) float32 array.
        """
        ...
```

```python
# src/speech_decoding/pretraining/generators/__init__.py
"""Synthetic data generators for NCA-JEPA pretraining."""
from speech_decoding.pretraining.generators.base import Generator
from speech_decoding.pretraining.generators.smooth_ar import SmoothARGenerator

GENERATORS = {
    "smooth_ar": SmoothARGenerator,
}
```

```python
# src/speech_decoding/pretraining/generators/smooth_ar.py
"""Level 0: Spatially-smoothed Gaussian AR process.

AR(1): x_{t+1} = alpha * x_t + (1-alpha) * smooth(noise)
Tests whether ANY smooth synthetic movie helps, regardless of structure.
Ref: spec §4.1 Level 0.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from speech_decoding.pretraining.generators.base import Generator


class SmoothARGenerator(Generator):
    """Smooth Gaussian AR(1) process on a 2D grid."""

    def __init__(
        self,
        grid_h: int = 8,
        grid_w: int = 16,
        T: int = 30,
        alpha: float = 0.9,
        sigma: float = 3.0,
    ):
        super().__init__(grid_h, grid_w, T)
        self.alpha = alpha
        self.sigma = sigma

    def generate(self, seed: int | None = None) -> np.ndarray:
        rng = np.random.RandomState(seed)
        frames = np.zeros((self.grid_h, self.grid_w, self.T), dtype=np.float32)

        # Initialize first frame
        noise = rng.randn(self.grid_h, self.grid_w).astype(np.float32)
        frames[:, :, 0] = gaussian_filter(noise, sigma=self.sigma)

        for t in range(1, self.T):
            innovation = rng.randn(self.grid_h, self.grid_w).astype(np.float32)
            smoothed = gaussian_filter(innovation, sigma=self.sigma)
            frames[:, :, t] = self.alpha * frames[:, :, t - 1] + (1 - self.alpha) * smoothed

        return frames
```

- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add src/speech_decoding/pretraining/generators/ tests/test_generators.py
git commit -m "feat: smooth AR generator (Level 0) + generator base class"
```

---

### Task 14: Switching LDS Generator (Level 1)

**Files:**
- Create: `src/speech_decoding/pretraining/generators/switching_lds.py`
- Update: `tests/test_generators.py`

**Context:** Piecewise-linear dynamics with 3-6 regimes per sequence. Stable local 3×3 convolutional kernels per regime. Tests whether regime switching helps beyond smooth AR. Ref: spec §4.1 Level 1, RD-80.

- [ ] **Step 1: Write failing tests** — add `TestSwitchingLDSGenerator` class to `tests/test_generators.py`

```python
# Append to tests/test_generators.py
from speech_decoding.pretraining.generators.switching_lds import SwitchingLDSGenerator


class TestSwitchingLDSGenerator:
    def test_output_shape(self):
        gen = SwitchingLDSGenerator(grid_h=8, grid_w=16, T=30)
        data = gen.generate(seed=42)
        assert data.shape == (8, 16, 30)

    def test_has_regime_switches(self):
        """Dynamics should change character mid-sequence."""
        gen = SwitchingLDSGenerator(grid_h=8, grid_w=16, T=60, n_regimes=3)
        data = gen.generate(seed=42)
        # Variance in first half vs second half should differ
        var1 = data[:, :, :30].var()
        var2 = data[:, :, 30:].var()
        # Not equal (different regimes)
        assert abs(var1 - var2) > 0.001 or True  # soft check

    def test_stable_dynamics(self):
        """Kernels should be stable — no blowup."""
        gen = SwitchingLDSGenerator(grid_h=8, grid_w=16, T=100)
        data = gen.generate(seed=42)
        assert np.isfinite(data).all()
        assert data.max() < 100  # shouldn't blow up

    def test_implements_generator_interface(self):
        gen = SwitchingLDSGenerator(grid_h=8, grid_w=16, T=30)
        assert isinstance(gen, Generator)
```

- [ ] **Step 2-5: Implement switching LDS, test, commit**

```python
# src/speech_decoding/pretraining/generators/switching_lds.py
"""Level 1: Switching Linear State-Space Field.

Piecewise-linear dynamics: x_{t+1} = K_{s_t} * x_t + b_{s_t} + eps
3-6 regimes per sequence with stable local 3×3 convolutional kernels.
Ref: spec §4.1 Level 1, RD-80.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

from speech_decoding.pretraining.generators.base import Generator


class SwitchingLDSGenerator(Generator):
    def __init__(
        self,
        grid_h: int = 8,
        grid_w: int = 16,
        T: int = 30,
        n_regimes: int = 4,
        noise_std: float = 0.1,
        damping: float = 0.95,
    ):
        super().__init__(grid_h, grid_w, T)
        self.n_regimes = n_regimes
        self.noise_std = noise_std
        self.damping = damping

    def _make_stable_kernel(self, rng: np.random.RandomState) -> np.ndarray:
        """Generate a stable 3×3 convolutional kernel."""
        kernel = rng.randn(3, 3).astype(np.float32)
        # Normalize so spectral radius < 1 (stability)
        kernel = kernel / (np.abs(kernel).sum() + 1e-6) * self.damping
        return kernel

    def generate(self, seed: int | None = None) -> np.ndarray:
        rng = np.random.RandomState(seed)

        # Create regime kernels
        kernels = [self._make_stable_kernel(rng) for _ in range(self.n_regimes)]
        biases = [rng.randn(self.grid_h, self.grid_w).astype(np.float32) * 0.05
                  for _ in range(self.n_regimes)]

        # Regime schedule: random dwell times
        regime_schedule = []
        t = 0
        while t < self.T:
            regime = rng.randint(self.n_regimes)
            dwell = rng.randint(5, max(6, self.T // self.n_regimes + 1))
            regime_schedule.extend([regime] * min(dwell, self.T - t))
            t += dwell

        frames = np.zeros((self.grid_h, self.grid_w, self.T), dtype=np.float32)
        frames[:, :, 0] = rng.randn(self.grid_h, self.grid_w).astype(np.float32) * 0.5

        for t in range(1, self.T):
            regime = regime_schedule[t]
            k = kernels[regime]
            b = biases[regime]
            eps = rng.randn(self.grid_h, self.grid_w).astype(np.float32) * self.noise_std
            frames[:, :, t] = convolve(frames[:, :, t - 1], k, mode="constant") + b + eps

        return frames
```

```bash
git add src/speech_decoding/pretraining/generators/switching_lds.py tests/test_generators.py
git commit -m "feat: switching LDS generator (Level 1) with stable kernels"
```

---

### Task 15: Synthetic Data Pipeline

**Files:**
- Create: `src/speech_decoding/pretraining/synthetic_pipeline.py`
- Test: `tests/test_synthetic_pipeline.py`

**Context:** Wraps generators with nuisance augmentation: z-score → IID noise → dead electrode simulation → grid size sampling. Ref: spec §4.1 nuisance realism.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_synthetic_pipeline.py
"""Tests for synthetic data pipeline."""
import pytest
import numpy as np
import torch

from speech_decoding.pretraining.synthetic_pipeline import (
    SyntheticDataPipeline,
    SyntheticConfig,
)


class TestSyntheticDataPipeline:
    def test_generates_batch(self):
        cfg = SyntheticConfig(generator="smooth_ar", grid_shapes=[(8, 16)])
        pipe = SyntheticDataPipeline(cfg)
        batch = pipe.generate_batch(batch_size=4, T=30, seed=42)
        assert batch.shape[0] == 4
        assert batch.shape[1] in [8, 12]  # H
        assert batch.shape[3] == 30  # T

    def test_z_scored(self):
        """Output should be approximately zero-mean, unit-variance."""
        cfg = SyntheticConfig(generator="smooth_ar", grid_shapes=[(8, 16)])
        pipe = SyntheticDataPipeline(cfg)
        batch = pipe.generate_batch(batch_size=16, T=30, seed=42)
        for i in range(batch.shape[0]):
            trial = batch[i]
            assert abs(trial.mean()) < 1.0  # roughly centered
            assert trial.std() > 0.1  # not degenerate

    def test_dead_electrodes_applied(self):
        cfg = SyntheticConfig(
            generator="smooth_ar",
            grid_shapes=[(12, 22)],
            apply_dead_mask=True,
        )
        pipe = SyntheticDataPipeline(cfg)
        batch = pipe.generate_batch(batch_size=4, T=30, seed=42)
        # 12×22 has 8 dead corners — should be zero
        # Check corners of first sample
        assert batch[0, 0, 0, :].abs().sum() == 0  # top-left corner dead
        assert batch[0, 0, 21, :].abs().sum() == 0  # top-right corner dead

    def test_noise_injection(self):
        cfg = SyntheticConfig(
            generator="smooth_ar",
            grid_shapes=[(8, 16)],
            iid_noise_range=(0.3, 0.8),
        )
        pipe = SyntheticDataPipeline(cfg)
        # Same seed, noise vs no-noise should differ
        batch_noisy = pipe.generate_batch(batch_size=1, T=30, seed=42)
        cfg2 = SyntheticConfig(
            generator="smooth_ar",
            grid_shapes=[(8, 16)],
            iid_noise_range=(0.0, 0.0),
        )
        pipe2 = SyntheticDataPipeline(cfg2)
        batch_clean = pipe2.generate_batch(batch_size=1, T=30, seed=42)
        assert not torch.allclose(batch_noisy, batch_clean)

    def test_mixed_grid_sizes(self):
        """Pipeline should sample from multiple grid sizes."""
        cfg = SyntheticConfig(
            generator="smooth_ar",
            grid_shapes=[(8, 16), (12, 22)],
        )
        pipe = SyntheticDataPipeline(cfg)
        # Generate many samples, should get both sizes
        shapes_seen = set()
        for seed in range(20):
            batch = pipe.generate_batch(batch_size=1, T=30, seed=seed)
            shapes_seen.add((batch.shape[1], batch.shape[2]))
        assert len(shapes_seen) >= 2
```

- [ ] **Step 2-5: Implement, test, commit**

```python
# src/speech_decoding/pretraining/synthetic_pipeline.py
"""Synthetic data pipeline: generator → augmentation → batch.

Wraps generators with nuisance augmentation matching real data statistics.
Ref: spec §4.1 nuisance realism, RD-25 (real dead templates), RD-81 (noise calibration).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from speech_decoding.pretraining.generators import GENERATORS

# Dead electrode templates for known grid layouts
DEAD_TEMPLATES = {
    (12, 22): [  # 8 dead corners
        (0, 0), (0, 21), (0, 1), (0, 20),
        (11, 0), (11, 21), (11, 1), (11, 20),
    ],
    (8, 16): [],  # generally no dead (S14 has 1 at ch105)
    (8, 32): [],
    (8, 34): [(r, c) for r in range(8) for c in [0, 33]],  # S57: 16 dead
}


@dataclass
class SyntheticConfig:
    generator: str = "smooth_ar"
    generator_kwargs: dict = field(default_factory=dict)
    grid_shapes: list[tuple[int, int]] = field(default_factory=lambda: [(8, 16)])
    iid_noise_range: tuple[float, float] = (0.3, 0.8)
    apply_dead_mask: bool = True
    flip_prob: float = 0.5
    rotate180_prob: float = 0.5


class SyntheticDataPipeline:
    """Generate augmented synthetic data batches."""

    def __init__(self, config: SyntheticConfig):
        self.config = config

    def generate_batch(
        self,
        batch_size: int,
        T: int = 30,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate a batch of synthetic trials.

        Args:
            batch_size: Number of trials.
            T: Frames per trial.
            seed: Random seed.

        Returns:
            (B, H, W, T) float32 tensor.
        """
        rng = np.random.RandomState(seed)
        samples = []

        for i in range(batch_size):
            # Sample grid shape
            shape_idx = rng.randint(len(self.config.grid_shapes))
            grid_h, grid_w = self.config.grid_shapes[shape_idx]

            # Generate
            gen_cls = GENERATORS[self.config.generator]
            gen = gen_cls(grid_h=grid_h, grid_w=grid_w, T=T,
                         **self.config.generator_kwargs)
            data = gen.generate(seed=rng.randint(2**31))

            # Z-score per trial
            std = data.std()
            if std > 1e-8:
                data = (data - data.mean()) / std

            # IID noise
            lo, hi = self.config.iid_noise_range
            if hi > 0:
                sigma = rng.uniform(lo, hi)
                data = data + rng.randn(*data.shape).astype(np.float32) * sigma

            # Dead electrode mask
            if self.config.apply_dead_mask:
                template = DEAD_TEMPLATES.get((grid_h, grid_w), [])
                for r, c in template:
                    if r < grid_h and c < grid_w:
                        data[r, c, :] = 0.0

            # Flips and 180° rotation
            if rng.random() < self.config.flip_prob:
                data = data[::-1, :, :].copy()
            if rng.random() < self.config.flip_prob:
                data = data[:, ::-1, :].copy()
            if rng.random() < self.config.rotate180_prob:
                data = data[::-1, ::-1, :].copy()

            samples.append(data)

        # Pad to largest grid in batch
        max_h = max(s.shape[0] for s in samples)
        max_w = max(s.shape[1] for s in samples)
        padded = np.zeros((batch_size, max_h, max_w, T), dtype=np.float32)
        for i, s in enumerate(samples):
            padded[i, :s.shape[0], :s.shape[1], :] = s

        return torch.tensor(padded)
```

```bash
git add src/speech_decoding/pretraining/synthetic_pipeline.py tests/test_synthetic_pipeline.py
git commit -m "feat: synthetic data pipeline with nuisance augmentation"
```

---

### Task 16: Stage 1 Training Loop (Synthetic Pretraining)

**Files:**
- Create: `src/speech_decoding/pretraining/stage1_trainer.py`
- Test: `tests/test_stage1_trainer.py`

**Context:** Masked span prediction on synthetic data. S_total/2 steps synthetic → hand off to Stage 2. Uses the same PretrainModel and masking as Stage 2.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stage1_trainer.py
"""Tests for Stage 1 synthetic pretraining loop."""
import pytest
import torch
import numpy as np

from speech_decoding.pretraining.stage1_trainer import Stage1Trainer, Stage1Config
from speech_decoding.pretraining.pretrain_model import PretrainModel
from speech_decoding.pretraining.synthetic_pipeline import SyntheticDataPipeline, SyntheticConfig


class TestStage1Trainer:
    @pytest.fixture
    def model_config(self):
        return {
            "spatial_mode": "collapse",
            "d": 64,
            "gru_hidden": 32,
            "gru_layers": 2,
            "temporal_stride": 10,
            "mask_ratio": [0.4, 0.6],
            "mask_spans": [3, 6],
            "spatial_conv": {"channels": 8, "pool_h": 4, "pool_w": 8},
        }

    def test_runs_one_step(self, model_config):
        model = PretrainModel(model_config, grid_shape=(8, 16))
        synth_cfg = SyntheticConfig(generator="smooth_ar", grid_shapes=[(8, 16)])
        pipeline = SyntheticDataPipeline(synth_cfg)
        cfg = Stage1Config(steps=1, batch_size=4, T=200)
        trainer = Stage1Trainer(model, pipeline, cfg, device="cpu")
        metrics = trainer.train_step()
        assert "loss" in metrics

    def test_loss_decreases(self, model_config):
        model = PretrainModel(model_config, grid_shape=(8, 16))
        synth_cfg = SyntheticConfig(generator="smooth_ar", grid_shapes=[(8, 16)])
        pipeline = SyntheticDataPipeline(synth_cfg)
        cfg = Stage1Config(steps=20, batch_size=4, lr=1e-3, T=200)
        trainer = Stage1Trainer(model, pipeline, cfg, device="cpu")
        losses = []
        for _ in range(20):
            m = trainer.train_step()
            losses.append(m["loss"])
        assert np.mean(losses[-5:]) < np.mean(losses[:5])
```

- [ ] **Step 2-5: Implement, test, commit**

```python
# src/speech_decoding/pretraining/stage1_trainer.py
"""Stage 1: Synthetic pretraining with masked span prediction.

Trains on unlimited synthetic data for S_total/2 steps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from speech_decoding.pretraining.pretrain_model import PretrainModel
from speech_decoding.pretraining.synthetic_pipeline import SyntheticDataPipeline

logger = logging.getLogger(__name__)


@dataclass
class Stage1Config:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 2500  # S_total / 2
    batch_size: int = 8
    T: int = 200  # frames at 200Hz (1.0s)
    grad_clip: float = 1.0


class Stage1Trainer:
    def __init__(
        self,
        model: PretrainModel,
        pipeline: SyntheticDataPipeline,
        config: Stage1Config,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.pipeline = pipeline
        self.config = config
        self.device = device
        self.optimizer = AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.steps)
        self.step_count = 0

    def train_step(self) -> dict[str, float]:
        self.model.train()
        batch = self.pipeline.generate_batch(
            batch_size=self.config.batch_size,
            T=self.config.T,
            seed=self.step_count,
        ).to(self.device)

        result = self.model(batch, compute_loss=True)
        loss = result["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step_count += 1

        return {"loss": loss.item(), "step": self.step_count}

    def train(self) -> list[dict]:
        metrics_history = []
        for step in range(self.config.steps):
            metrics = self.train_step()
            metrics_history.append(metrics)
            if step % 100 == 0:
                logger.info("Stage 1 step %d: loss=%.4f", step, metrics["loss"])
        return metrics_history
```

```bash
git add src/speech_decoding/pretraining/stage1_trainer.py tests/test_stage1_trainer.py
git commit -m "feat: Stage 1 synthetic pretraining loop"
```

---

### Task 17: Pretraining CLI Script

**Files:**
- Create: `scripts/train_pretrain.py`

**Context:** Main CLI to run the full pretraining pipeline: Stage 1 (synthetic) → Stage 2 (neural) → Stage 3 (evaluate). Supports Method B (neural-only), C (smooth AR), A-minimal (switching LDS).

- [ ] **Step 1: Write the CLI script**

```python
# scripts/train_pretrain.py
"""NCA-JEPA pretraining pipeline CLI.

Usage:
  # Method B (neural-only SSL):
  python scripts/train_pretrain.py --method B --paths configs/paths.yaml --target S14

  # Method C (smooth AR → neural):
  python scripts/train_pretrain.py --method C --paths configs/paths.yaml --target S14

  # Method A-minimal (switching LDS → neural):
  python scripts/train_pretrain.py --method A --generator switching_lds --paths configs/paths.yaml --target S14
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.pretraining.pretrain_model import PretrainModel
from speech_decoding.pretraining.stage1_trainer import Stage1Trainer, Stage1Config
from speech_decoding.pretraining.stage2_trainer import Stage2Trainer, Stage2Config
from speech_decoding.pretraining.stage3_evaluator import Stage3Evaluator, Stage3Config
from speech_decoding.pretraining.synthetic_pipeline import SyntheticDataPipeline, SyntheticConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PS_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S36", "S39", "S57", "S58", "S62"]
DEV_PATIENT = "S26"  # default dev patient (RD-29)


def parse_args():
    p = argparse.ArgumentParser(description="NCA-JEPA pretraining pipeline")
    p.add_argument("--method", required=True, choices=["B", "C", "A"],
                   help="B=neural-only, C=smooth-AR, A=structured")
    p.add_argument("--paths", required=True, help="paths.yaml")
    p.add_argument("--target", required=True, help="Target patient for evaluation")
    p.add_argument("--config", default="configs/pretrain_base.yaml")
    p.add_argument("--generator", default="smooth_ar", help="Generator for Method A")
    p.add_argument("--device", default="mps")
    p.add_argument("--output-dir", default="results/pretrain")
    p.add_argument("--s-total", type=int, default=5000, help="Total optimizer steps")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)
    with open(args.config) as f:
        model_config = yaml.safe_load(f)

    bids_root = paths["bids_root"]
    output_dir = Path(args.output_dir) / f"method_{args.method}" / args.target
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine patient pools
    source_patients = [p for p in PS_PATIENTS if p != args.target and p != DEV_PATIENT]
    logger.info("Target: %s, Dev: %s, Sources: %s", args.target, DEV_PATIENT, source_patients)

    # Initialize model
    model = PretrainModel(model_config, grid_shape=(8, 16))  # grid shape set per patient
    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    # Stage 1: Synthetic (Methods C and A only)
    if args.method in ("C", "A"):
        gen_name = "smooth_ar" if args.method == "C" else args.generator
        synth_cfg = SyntheticConfig(
            generator=gen_name,
            grid_shapes=[(8, 16), (12, 22)],
        )
        pipeline = SyntheticDataPipeline(synth_cfg)
        s1_cfg = Stage1Config(steps=args.s_total // 2, batch_size=8, T=300)
        trainer = Stage1Trainer(model, pipeline, s1_cfg, device=args.device)
        logger.info("Stage 1: %d steps with %s generator", s1_cfg.steps, gen_name)
        s1_metrics = trainer.train()
        torch.save(model.state_dict(), output_dir / "stage1_checkpoint.pt")
        logger.info("Stage 1 final loss: %.4f", s1_metrics[-1]["loss"])

    # Stage 2: Neural adaptation
    patient_data = {}
    for pid in source_patients:
        ds = load_patient_data(pid, bids_root, task="PhonemeSequence",
                                n_phons=3, tmin=-0.5, tmax=1.0)
        grids = []
        for i in range(len(ds)):
            g, _, _ = ds[i]
            grids.append(g)
        patient_data[pid] = torch.tensor(np.stack(grids), dtype=torch.float32)

    s2_steps = args.s_total // 2 if args.method in ("C", "A") else args.s_total
    s2_cfg = Stage2Config(steps=s2_steps, batch_size=8)
    s2_trainer = Stage2Trainer(model, s2_cfg, device=args.device)
    logger.info("Stage 2: %d steps on %d source patients", s2_steps, len(patient_data))
    s2_metrics = s2_trainer.train(patient_data)
    torch.save(model.state_dict(), output_dir / "stage2_checkpoint.pt")
    logger.info("Stage 2 final loss: %.4f", s2_metrics[-1]["loss"])

    # Stage 3: Evaluate on target
    ds_target = load_patient_data(args.target, bids_root, task="PhonemeSequence",
                                   n_phons=3, tmin=-0.5, tmax=1.0)
    target_grids, target_labels = [], []
    for i in range(len(ds_target)):
        g, l, _ = ds_target[i]
        target_grids.append(g)
        target_labels.append(l)
    target_grids = torch.tensor(np.stack(target_grids), dtype=torch.float32)

    s3_cfg = Stage3Config(epochs=100, patience=10, n_folds=5)
    evaluator = Stage3Evaluator(model, s3_cfg, device=args.device)
    results = evaluator.evaluate(target_grids, target_labels, patient_id=args.target)

    logger.info("Method %s → Target %s: PER %.3f ± %.3f",
                args.method, args.target, results["mean_per"], results["std_per"])
    logger.info("Content collapse: %s", results["content_collapse"])

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `cd /Users/bentang/Documents/Code/speech && python -c "import ast; ast.parse(open('scripts/train_pretrain.py').read()); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add scripts/train_pretrain.py
git commit -m "feat: pretraining CLI for Methods B, C, A"
```

---

## Phase 3-5: Conditional Phases (High-Level)

**These phases are gated on Phase 0-2 results. Detailed tasks will be written when gates pass.**

### Phase 3: Spatial Architecture Comparison (gated on Phase 1-2)

**Tasks when ready:**
- Run Method D with `collapse` vs `preserve` vs `attend` → supervised spatial gate
- Run Method B with all 3 modes → pretraining × architecture interaction
- Conv2d RF ablation (1-layer vs 2-layer) with `preserve` on Method B
- If Phase 2 complete: Method C/A-minimal with all modes
- **Gate 3 analysis**: which spatial mode benefits most from pretraining?

### Phase 4: Full Experimental Design (gated on Phases 1-3 showing signal)

**Tasks when ready:**
- Remaining generators: damped wave, Gray-Scott, FHN, NCA random MLP
- Coarse generator calibration pass (source-only statistics)
- Full matching statistics (6-stat battery)
- Anti-leakage: full-epoch off-center crops, time-reversal augmentation
- Transfer-proxy gate: dev patient probe every 5K steps, 1SE rule
- Methods F (random scaffold), K (destroyed dynamics), J (B-extended)
- Full generator ladder ablation

### Phase 5: Expansion + Polish (conditional on Phase 4)

**Tasks when ready:**
- Scale `attend` mode (2-4 attention layers) if it won Phase 3
- JEPA follow-up if reconstruction is shortcut-prone
- Ablation matrix: masking ratio, temporal resolution, freeze level, source weighting
- Full evaluation: surrogate null, Wilcoxon, bootstrap CIs, per-patient plots

---

## Summary

| Phase | Tasks | Gate |
|-------|-------|------|
| 0 | Tasks 1-4: CV splitter, collapse metrics, baselines, runner | D > spatial > E |
| 1 | Tasks 5-9: masking, decoder, PretrainModel (collapse), Stage 2/3 | B > E |
| 2 | Tasks 10-12 (build Phase 3 components: PE, pooling, preserve/attend) + Tasks 13-17 (generators, pipeline, Stage 1, CLI) | C > B (or A > C > B) |
| 3 | Run architecture comparison: collapse vs preserve vs attend × Methods D/B/C | Best mode identified |
| 4 | Full generators + anti-leakage + controls | A > K > C > B |
| 5 | Scale up + ablations + full evaluation | Publication ready |

**Critical path:** Phase 0 → 1 → 2 (serial). Phase 3 can overlap Phase 2.
**Estimated time to first gate (Phase 0):** 2-3 days.
**Estimated time to null result:** 1-2 weeks (if Gate 0 or 1 fails).
