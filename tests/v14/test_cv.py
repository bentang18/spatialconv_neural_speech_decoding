"""Unit tests for grouped-by-token CV (Task D1)."""

from __future__ import annotations

import pytest

from speech_decoding.v14.cv import make_outer_folds, make_val_split


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
