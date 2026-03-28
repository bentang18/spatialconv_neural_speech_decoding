"""Tests for grouped-by-token cross-validation splitter."""
import json
import pytest
import numpy as np
from pathlib import Path

from speech_decoding.evaluation.grouped_cv import (
    _patient_seed,
    build_token_groups,
    create_grouped_splits,
    load_or_create_splits,
    validate_fold_coverage,
)


class TestBuildTokenGroups:
    def test_groups_by_token_identity(self):
        """All trials of token 'abe' go to the same group."""
        labels = [
            [1, 3, 5], [1, 3, 5],
            [4, 5, 6], [4, 5, 6],
            [8, 9, 1], [8, 9, 1],
        ]
        groups = build_token_groups(labels)
        assert groups[0] == groups[1]
        assert groups[2] == groups[3]
        assert groups[4] == groups[5]
        assert len(set(groups)) == 3

    def test_all_trials_assigned(self):
        labels = [[1, 2, 3]] * 10 + [[4, 5, 6]] * 10
        groups = build_token_groups(labels)
        assert len(groups) == 20


class TestCreateGroupedSplits:
    def _make_labels(self, n_tokens=20, reps_per_token=5):
        phonemes = list(range(1, 10))
        labels = []
        for i in range(n_tokens):
            token = [phonemes[i % 9], phonemes[(i + 3) % 9], phonemes[(i + 6) % 9]]
            labels.extend([token] * reps_per_token)
        return labels

    def test_no_token_leakage(self):
        labels = self._make_labels(n_tokens=20, reps_per_token=5)
        groups = build_token_groups(labels)
        splits = create_grouped_splits(labels, groups, n_folds=5, seed=42)
        for fold in splits:
            train_tokens = {tuple(labels[i]) for i in fold["train_indices"]}
            val_tokens = {tuple(labels[i]) for i in fold["val_indices"]}
            assert train_tokens.isdisjoint(val_tokens), "Token leakage across folds!"

    def test_all_indices_covered(self):
        labels = self._make_labels()
        groups = build_token_groups(labels)
        splits = create_grouped_splits(labels, groups, n_folds=5, seed=42)
        all_val = []
        for fold in splits:
            all_val.extend(fold["val_indices"])
        assert sorted(all_val) == list(range(len(labels)))

    def test_phoneme_coverage_per_fold(self):
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
        splits = load_or_create_splits(labels, patient_id="S14", n_folds=3, save_path=path)
        assert path.exists()
        loaded = load_or_create_splits(labels, patient_id="S14", n_folds=3, save_path=path)
        for f1, f2 in zip(splits, loaded):
            assert f1["train_indices"] == f2["train_indices"]

    def test_seed_derived_from_patient_id(self):
        labels = [[1, 2, 3]] * 30 + [[4, 5, 6]] * 30
        s1 = create_grouped_splits(labels, build_token_groups(labels),
                                    n_folds=3, seed=_patient_seed("S14"))
        s2 = create_grouped_splits(labels, build_token_groups(labels),
                                    n_folds=3, seed=_patient_seed("S33"))
        assert s1[0]["val_indices"] != s2[0]["val_indices"]
