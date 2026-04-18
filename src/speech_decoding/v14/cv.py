"""Grouped-by-token cross-validation for v14-core (Task D1).

Tokens — not trials — are the CV grouping unit. A token that appears in the
training set never appears in validation or test for the same fold, so the
model cannot memorize token-specific patterns.

`make_outer_folds` assigns unique tokens to folds via round-robin on a seeded
shuffle (stable across the 3 training seeds of a given outer fold per the
First-Run Protocol). `make_val_split` carves ``val_frac`` of the training
*tokens* into a disjoint val set.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def _tokens_to_trials(tokens_per_trial: Sequence[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, t in enumerate(tokens_per_trial):
        out.setdefault(t, []).append(i)
    return out


def make_outer_folds(
    tokens_per_trial: Sequence[str],
    n_folds: int = 5,
    seed: int = 0,
) -> list[tuple[list[int], list[int]]]:
    """Return ``n_folds`` ``(train_idx, test_idx)`` pairs, token-disjoint."""

    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    token_to_trials = _tokens_to_trials(tokens_per_trial)
    tokens = sorted(token_to_trials.keys())
    if len(tokens) < n_folds:
        raise ValueError(
            f"need at least n_folds={n_folds} unique tokens, got {len(tokens)}"
        )
    rng = random.Random(seed)
    rng.shuffle(tokens)

    fold_tokens: list[list[str]] = [[] for _ in range(n_folds)]
    for i, tok in enumerate(tokens):
        fold_tokens[i % n_folds].append(tok)

    folds: list[tuple[list[int], list[int]]] = []
    for k in range(n_folds):
        test_tokens = set(fold_tokens[k])
        train_idx: list[int] = []
        test_idx: list[int] = []
        for tok, idxs in token_to_trials.items():
            (test_idx if tok in test_tokens else train_idx).extend(idxs)
        folds.append((sorted(train_idx), sorted(test_idx)))
    return folds


def make_val_split(
    tokens_per_trial: Sequence[str],
    train_idx: Sequence[int],
    val_frac: float = 0.2,
    seed: int = 7,
) -> tuple[list[int], list[int]]:
    """Carve ``val_frac`` of the *tokens* in ``train_idx`` into a val set."""

    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

    train_tokens = sorted({tokens_per_trial[i] for i in train_idx})
    if len(train_tokens) < 2:
        raise ValueError(
            f"need at least 2 unique training tokens, got {len(train_tokens)}"
        )
    rng = random.Random(seed)
    rng.shuffle(train_tokens)
    n_val = max(1, round(val_frac * len(train_tokens)))
    val_tokens = set(train_tokens[:n_val])

    new_train: list[int] = []
    val: list[int] = []
    for i in train_idx:
        (val if tokens_per_trial[i] in val_tokens else new_train).append(i)
    return sorted(new_train), sorted(val)
