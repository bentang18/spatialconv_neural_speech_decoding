"""Grouped-by-token cross-validation for v14-core (Task D1).

Tokens — not trials — are the CV grouping unit. A token that appears in the
training set never appears in validation or test for the same fold, so the
model cannot memorize token-specific patterns.

`make_outer_folds` assigns unique tokens to folds via round-robin on a seeded
shuffle (stable across the 3 training seeds of a given outer fold per the
First-Run Protocol). `make_val_split` carves ``val_frac`` of the training
*tokens* into a disjoint val set.

The ``_pooled`` variants extend the contract to multi-patient training: the
token→fold map is partitioned on the **union** of tokens across patients once,
then applied per-patient so a given token is in the same fold globally.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def _tokens_to_trials(tokens_per_trial: Sequence[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, t in enumerate(tokens_per_trial):
        out.setdefault(t, []).append(i)
    return out


def _partition_tokens(tokens: Sequence[str], n_folds: int, seed: int) -> list[list[str]]:
    """Shuffle unique tokens under ``seed`` and round-robin into ``n_folds`` bins."""

    uniq = sorted(set(tokens))
    if len(uniq) < n_folds:
        raise ValueError(
            f"need at least n_folds={n_folds} unique tokens, got {len(uniq)}"
        )
    rng = random.Random(seed)
    rng.shuffle(uniq)
    fold_tokens: list[list[str]] = [[] for _ in range(n_folds)]
    for i, tok in enumerate(uniq):
        fold_tokens[i % n_folds].append(tok)
    return fold_tokens


def make_outer_folds(
    tokens_per_trial: Sequence[str],
    n_folds: int = 5,
    seed: int = 0,
) -> list[tuple[list[int], list[int]]]:
    """Return ``n_folds`` ``(train_idx, test_idx)`` pairs, token-disjoint."""

    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    token_to_trials = _tokens_to_trials(tokens_per_trial)
    fold_tokens = _partition_tokens(list(token_to_trials.keys()), n_folds, seed)

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


def make_outer_folds_pooled(
    patient_tokens: dict[str, Sequence[str]],
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, list[tuple[list[int], list[int]]]]:
    """Multi-patient token-level CV.

    Partitions the union of tokens across ``patient_tokens.values()`` into
    ``n_folds`` bins once under ``seed``, then expands to per-patient trial
    indices. A given token lives in the same fold for every patient, so the
    pooled-training eval slice is globally held out.

    Returns a dict ``{patient: [(train_idx, test_idx)] * n_folds}``.
    """

    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if not patient_tokens:
        raise ValueError("patient_tokens must be non-empty")

    union: list[str] = []
    for tokens in patient_tokens.values():
        union.extend(tokens)
    fold_tokens = _partition_tokens(union, n_folds, seed)
    fold_of: dict[str, int] = {}
    for k, toks in enumerate(fold_tokens):
        for t in toks:
            fold_of[t] = k

    out: dict[str, list[tuple[list[int], list[int]]]] = {}
    for pt, tokens in patient_tokens.items():
        per_patient: list[tuple[list[int], list[int]]] = []
        for k in range(n_folds):
            train_idx: list[int] = []
            test_idx: list[int] = []
            for i, t in enumerate(tokens):
                if t not in fold_of:
                    raise KeyError(f"token {t!r} for patient {pt!r} not in partition")
                (test_idx if fold_of[t] == k else train_idx).append(i)
            per_patient.append((sorted(train_idx), sorted(test_idx)))
        out[pt] = per_patient
    return out


def make_val_split_pooled(
    patient_tokens: dict[str, Sequence[str]],
    patient_train_idx: dict[str, Sequence[int]],
    val_frac: float = 0.2,
    seed: int = 7,
) -> dict[str, tuple[list[int], list[int]]]:
    """Multi-patient val-split. Picks the same val tokens for every patient.

    Carves ``val_frac`` of the union of *training* tokens into a val set, then
    applies the same token→val membership to each patient's train indices.
    """

    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if set(patient_train_idx.keys()) != set(patient_tokens.keys()):
        raise ValueError("patient_train_idx keys must match patient_tokens keys")

    union_train: set[str] = set()
    for pt, train_idx in patient_train_idx.items():
        tokens = patient_tokens[pt]
        union_train.update(tokens[i] for i in train_idx)
    train_tokens = sorted(union_train)
    if len(train_tokens) < 2:
        raise ValueError(
            f"need at least 2 unique training tokens in union, got {len(train_tokens)}"
        )
    rng = random.Random(seed)
    rng.shuffle(train_tokens)
    n_val = max(1, round(val_frac * len(train_tokens)))
    val_tokens = set(train_tokens[:n_val])

    out: dict[str, tuple[list[int], list[int]]] = {}
    for pt, train_idx in patient_train_idx.items():
        tokens = patient_tokens[pt]
        new_train: list[int] = []
        val: list[int] = []
        for i in train_idx:
            (val if tokens[i] in val_tokens else new_train).append(i)
        out[pt] = (sorted(new_train), sorted(val))
    return out
