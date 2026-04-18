"""Unit tests for `run_one_fold` (Task E1).

Uses a minimal fake dataset that satisfies the `V14TrialDataset` protocol
(`tokens`, `__len__`, `__getitem__` -> `V14Sample`) so the test runs without
a real `.fif`, channel map, or support cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from speech_decoding.v14.dataset import (
    N_TIER1_PARCELS,
    T_SAMPLES,
    V14Sample,
    build_sample_from_epoch,
)
from speech_decoding.v14.run_fold import OUTER_FOLD_SEED, VAL_SPLIT_SEED, run_one_fold


class FakeDataset:
    """Deterministic fake dataset — one `V14Sample` per trial.

    Tokens repeat across trials so grouped CV sees multiple unique tokens per
    fold; signals / support are reproducible from a numpy seed.
    """

    def __init__(self, n_trials: int = 40, n_e: int = 16, grid_shape: tuple[int, int] = (8, 16)) -> None:
        rng = np.random.default_rng(0)
        tokens_pool = ["bak", "gub", "pik", "vap", "abi", "uga", "iga", "aeba"]
        _H_p, W_p = grid_shape
        layout = np.stack(
            [np.array([i // W_p, i % W_p], dtype=np.int64) for i in range(n_e)]
        )
        active = np.ones(n_e, dtype=bool)

        self._samples: list[V14Sample] = []
        self._tokens: list[str] = []
        for t in range(n_trials):
            token = tokens_pool[t % len(tokens_pool)]
            signal = rng.standard_normal((n_e, T_SAMPLES)).astype(np.float32)
            support = rng.random((n_e, N_TIER1_PARCELS)).astype(np.float32)
            self._samples.append(
                build_sample_from_epoch(
                    signal=signal,
                    token=token,
                    patient_id="SXX",
                    electrode_grid_layout=layout,
                    electrode_grid_shape=grid_shape,
                    electrode_active_mask=active,
                    support=support,
                )
            )
            self._tokens.append(token)

    @property
    def tokens(self) -> list[str]:
        return list(self._tokens)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, i: int) -> V14Sample:
        return self._samples[i]


class TestRunOneFold:
    def test_writes_expected_files_and_fields(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        dataset = FakeDataset()
        result = run_one_fold(
            dataset,
            fold_idx=0,
            seed=0,
            grid_mixer_depth=1,
            out_dir=tmp_path,
            patient_id="SXX",
            max_epochs=3,
            val_every=1,
            patience=10,
        )
        required = {
            "patient", "fold", "seed", "grid_mixer_depth",
            "best_val_per", "test_per", "final_epoch",
        }
        assert required.issubset(result.keys())
        assert 0.0 <= result["best_val_per"] <= 1.0
        assert 0.0 <= result["test_per"] <= 1.0

        tag = "SXX_fold0_seed0_depth1"
        assert (tmp_path / f"{tag}.result.json").exists()
        assert (tmp_path / f"{tag}.log.jsonl").exists()

        log_rows = [
            json.loads(line)
            for line in (tmp_path / f"{tag}.log.jsonl").read_text().splitlines()
        ]
        assert len(log_rows) == 3
        assert all("train_loss" in r and "lr" in r for r in log_rows)
        assert all("val_per" in r for r in log_rows)  # val_every=1

    def test_fold_and_val_splits_are_deterministic(self, tmp_path: Path) -> None:
        """Outer-fold seed and val-split seed are fixed across training seeds."""

        from speech_decoding.v14.cv import make_outer_folds, make_val_split

        dataset = FakeDataset()
        folds = make_outer_folds(dataset.tokens, n_folds=5, seed=OUTER_FOLD_SEED)
        train_idx_0, _ = folds[0]
        _, val_a = make_val_split(dataset.tokens, train_idx_0, val_frac=0.2, seed=VAL_SPLIT_SEED)
        _, val_b = make_val_split(dataset.tokens, train_idx_0, val_frac=0.2, seed=VAL_SPLIT_SEED)
        assert val_a == val_b

    def test_depth_override_reaches_model(self, tmp_path: Path) -> None:
        from speech_decoding.v14.run_fold import _build_config

        assert _build_config(1).grid_mixer.num_layers == 1
        assert _build_config(2).grid_mixer.num_layers == 2

    def test_rejects_bad_depth(self, tmp_path: Path) -> None:
        import pytest as _pt

        dataset = FakeDataset(n_trials=10)
        with _pt.raises(ValueError, match="grid_mixer_depth"):
            run_one_fold(
                dataset, fold_idx=0, seed=0, grid_mixer_depth=0,
                out_dir=tmp_path, patient_id="SXX",
            )
