"""Unit tests for grouped-by-token CV (Task D1)."""

from __future__ import annotations

import pytest

from speech_decoding.training.cv import (
    make_outer_folds,
    make_outer_folds_pooled,
    make_val_split,
    make_val_split_pooled,
)


def _tokens(n_tokens: int, per_token: int) -> list[str]:
    return [f"tok{j:02d}" for j in range(n_tokens) for _ in range(per_token)]


class TestMakeOuterFolds:
    def test_returns_n_folds_pairs(self) -> None:
        folds = make_outer_folds(_tokens(20, 3), n_folds=5, seed=0)
        assert len(folds) == 5

    def test_folds_cover_all_trials_exactly_once_as_test(self) -> None:
        tokens = _tokens(20, 3)
        folds = make_outer_folds(tokens, n_folds=5, seed=0)
        seen_in_test: list[int] = []
        for _, test_idx in folds:
            seen_in_test.extend(test_idx)
        assert sorted(seen_in_test) == list(range(len(tokens)))

    def test_train_test_disjoint_and_token_disjoint(self) -> None:
        tokens = _tokens(20, 3)
        folds = make_outer_folds(tokens, n_folds=5, seed=0)
        for train_idx, test_idx in folds:
            assert set(train_idx).isdisjoint(test_idx)
            train_toks = {tokens[i] for i in train_idx}
            test_toks = {tokens[i] for i in test_idx}
            assert train_toks.isdisjoint(test_toks)

    def test_seed_is_deterministic(self) -> None:
        tokens = _tokens(20, 3)
        a = make_outer_folds(tokens, n_folds=5, seed=0)
        b = make_outer_folds(tokens, n_folds=5, seed=0)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        tokens = _tokens(20, 3)
        a = make_outer_folds(tokens, n_folds=5, seed=0)
        b = make_outer_folds(tokens, n_folds=5, seed=1)
        assert a != b

    def test_rejects_fewer_tokens_than_folds(self) -> None:
        with pytest.raises(ValueError, match="unique tokens"):
            make_outer_folds(_tokens(4, 3), n_folds=5, seed=0)

    def test_rejects_tiny_n_folds(self) -> None:
        with pytest.raises(ValueError, match="n_folds"):
            make_outer_folds(_tokens(20, 3), n_folds=1, seed=0)


class TestMakeValSplit:
    def test_val_token_disjoint_from_remaining_train(self) -> None:
        tokens = _tokens(20, 3)
        train_idx, _ = make_outer_folds(tokens, n_folds=5, seed=0)[0]
        new_train, val = make_val_split(tokens, train_idx, val_frac=0.2, seed=7)
        assert set(new_train).isdisjoint(val)
        train_toks = {tokens[i] for i in new_train}
        val_toks = {tokens[i] for i in val}
        assert train_toks.isdisjoint(val_toks)

    def test_val_covers_roughly_val_frac_of_tokens(self) -> None:
        tokens = _tokens(20, 3)
        train_idx, _ = make_outer_folds(tokens, n_folds=5, seed=0)[0]
        _, val = make_val_split(tokens, train_idx, val_frac=0.2, seed=7)
        n_val_toks = len({tokens[i] for i in val})
        n_train_toks = len({tokens[i] for i in train_idx})
        assert n_val_toks == round(0.2 * n_train_toks)

    def test_partition_preserves_all_train_trials(self) -> None:
        tokens = _tokens(20, 3)
        train_idx, _ = make_outer_folds(tokens, n_folds=5, seed=0)[0]
        new_train, val = make_val_split(tokens, train_idx, val_frac=0.2, seed=7)
        assert sorted(new_train + val) == sorted(train_idx)

    def test_seed_is_deterministic(self) -> None:
        tokens = _tokens(20, 3)
        train_idx, _ = make_outer_folds(tokens, n_folds=5, seed=0)[0]
        a = make_val_split(tokens, train_idx, val_frac=0.2, seed=7)
        b = make_val_split(tokens, train_idx, val_frac=0.2, seed=7)
        assert a == b

    def test_rejects_out_of_range_val_frac(self) -> None:
        tokens = _tokens(20, 3)
        train_idx, _ = make_outer_folds(tokens, n_folds=5, seed=0)[0]
        with pytest.raises(ValueError, match="val_frac"):
            make_val_split(tokens, train_idx, val_frac=0.0, seed=7)
        with pytest.raises(ValueError, match="val_frac"):
            make_val_split(tokens, train_idx, val_frac=1.0, seed=7)


class TestMakeOuterFoldsPooled:
    def _patients(self) -> dict[str, list[str]]:
        # Overlapping but not identical token coverage across three patients.
        return {
            "A": [f"tok{j:02d}" for j in range(20) for _ in range(3)],
            "B": [f"tok{j:02d}" for j in range(5, 20) for _ in range(2)],
            "C": [f"tok{j:02d}" for j in range(0, 15) for _ in range(4)],
        }

    def test_same_token_same_fold_across_patients(self) -> None:
        pt_tokens = self._patients()
        folds = make_outer_folds_pooled(pt_tokens, n_folds=5, seed=0)
        # Build per-patient {token: fold_k}. Must agree across patients.
        per_patient_maps: dict[str, dict[str, int]] = {}
        for pt, fold_pairs in folds.items():
            m: dict[str, int] = {}
            for k, (_, test_idx) in enumerate(fold_pairs):
                for i in test_idx:
                    m[pt_tokens[pt][i]] = k
            per_patient_maps[pt] = m
        all_tokens: set[str] = set()
        for m in per_patient_maps.values():
            all_tokens.update(m.keys())
        for tok in all_tokens:
            seen = {m[tok] for m in per_patient_maps.values() if tok in m}
            assert len(seen) == 1, f"token {tok} in multiple folds: {seen}"

    def test_train_test_disjoint_per_patient(self) -> None:
        pt_tokens = self._patients()
        folds = make_outer_folds_pooled(pt_tokens, n_folds=5, seed=0)
        for pt, fold_pairs in folds.items():
            tokens = pt_tokens[pt]
            for train_idx, test_idx in fold_pairs:
                assert set(train_idx).isdisjoint(test_idx)
                train_toks = {tokens[i] for i in train_idx}
                test_toks = {tokens[i] for i in test_idx}
                assert train_toks.isdisjoint(test_toks)

    def test_each_patient_folds_partition_trials(self) -> None:
        pt_tokens = self._patients()
        folds = make_outer_folds_pooled(pt_tokens, n_folds=5, seed=0)
        for pt, fold_pairs in folds.items():
            n = len(pt_tokens[pt])
            seen: list[int] = []
            for _, test_idx in fold_pairs:
                seen.extend(test_idx)
            assert sorted(seen) == list(range(n))

    def test_seed_is_deterministic(self) -> None:
        pt_tokens = self._patients()
        a = make_outer_folds_pooled(pt_tokens, n_folds=5, seed=0)
        b = make_outer_folds_pooled(pt_tokens, n_folds=5, seed=0)
        assert a == b

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            make_outer_folds_pooled({}, n_folds=5, seed=0)

    def test_rejects_too_few_union_tokens(self) -> None:
        with pytest.raises(ValueError, match="unique tokens"):
            make_outer_folds_pooled(
                {"A": ["t0", "t1", "t2"], "B": ["t1", "t2"]}, n_folds=5, seed=0
            )


class TestMakeValSplitPooled:
    def _setup(self) -> tuple[dict[str, list[str]], dict[str, list[int]]]:
        pt_tokens = {
            "A": [f"tok{j:02d}" for j in range(20) for _ in range(3)],
            "B": [f"tok{j:02d}" for j in range(5, 20) for _ in range(2)],
        }
        folds = make_outer_folds_pooled(pt_tokens, n_folds=5, seed=0)
        pt_train = {pt: fold_pairs[0][0] for pt, fold_pairs in folds.items()}
        return pt_tokens, pt_train

    def test_val_tokens_agree_across_patients(self) -> None:
        pt_tokens, pt_train = self._setup()
        out = make_val_split_pooled(pt_tokens, pt_train, val_frac=0.2, seed=7)
        val_toks: dict[str, set[str]] = {}
        for pt, (_, val_idx) in out.items():
            val_toks[pt] = {pt_tokens[pt][i] for i in val_idx}
        # A's val tokens ⊇ B's val tokens ∩ A's tokens (they share a single
        # global val-token set; B just doesn't have every token A has).
        a_set = val_toks["A"]
        b_set = val_toks["B"]
        shared_tokens = set(pt_tokens["A"]) & set(pt_tokens["B"])
        assert (a_set & shared_tokens) == (b_set & shared_tokens)

    def test_partition_preserves_all_train_trials(self) -> None:
        pt_tokens, pt_train = self._setup()
        out = make_val_split_pooled(pt_tokens, pt_train, val_frac=0.2, seed=7)
        for pt, (new_train, val) in out.items():
            assert sorted(new_train + val) == sorted(pt_train[pt])

    def test_seed_is_deterministic(self) -> None:
        pt_tokens, pt_train = self._setup()
        a = make_val_split_pooled(pt_tokens, pt_train, val_frac=0.2, seed=7)
        b = make_val_split_pooled(pt_tokens, pt_train, val_frac=0.2, seed=7)
        assert a == b

    def test_rejects_key_mismatch(self) -> None:
        pt_tokens, pt_train = self._setup()
        del pt_train["B"]
        with pytest.raises(ValueError, match="keys must match"):
            make_val_split_pooled(pt_tokens, pt_train, val_frac=0.2, seed=7)
