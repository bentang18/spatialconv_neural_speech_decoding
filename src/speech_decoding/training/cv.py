"""Grouped-by-token cross-validation.

Two routines, merged from `v14/cv.py` + `evaluation/grouped_cv.py` during
the pre-Stage-0 reorg:

1. **v14 token-fold split** — `make_outer_folds` / `make_val_split` and
   their `_pooled` variants. Tokens (not trials) are the grouping unit:
   a token in train never appears in val or test for the same fold.
   Round-robin assignment on a seeded shuffle (stable across the 3
   training seeds of a given outer fold per the First-Run Protocol).
   `_pooled` extends the contract to multi-patient training: the
   token→fold map is partitioned on the **union** of tokens across
   patients once, then applied per-patient so a given token is in the
   same fold globally.

2. **Legacy multi-label CV** — `build_token_groups` /
   `validate_fold_coverage` / `create_grouped_splits` /
   `load_or_create_splits`. Uses sklearn `GroupKFold` over deterministic
   per-patient seeds (md5 of patient_id). Retries permutations until
   every training fold has phoneme/position coverage. Refs RD-18, RD-57,
   RD-62.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold


logger = logging.getLogger(__name__)


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


def build_token_groups(labels: list[list[int]]) -> list[int]:
    """Assign each trial a group ID based on its token (phoneme sequence)."""
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

    Only enforces coverage for phonemes that appear in more than one
    token at a given position.  Singleton tokens (only one token in the
    whole dataset carries a particular phoneme at a particular position)
    cannot be required in training without data leakage — if that token
    lands in validation, coverage is lost regardless of the assignment.
    This mirrors real-data constraints where 52 PS tokens give each
    phoneme/position pair many repetitions across distinct tokens.
    """
    n_positions = max(len(l) for l in labels) if labels else 3

    # For each (position, phoneme), collect the set of distinct tokens
    # that carry it.  A pair is "enforceable" only when it appears in
    # ≥2 distinct tokens, so at least one can always remain in training
    # regardless of which token ends up in the validation fold.
    token_set_pos_phon: list[dict[int, set]] = [{} for _ in range(n_positions)]
    for label in labels:
        key = tuple(label)
        for pos, phon in enumerate(label):
            if phon not in token_set_pos_phon[pos]:
                token_set_pos_phon[pos][phon] = set()
            token_set_pos_phon[pos][phon].add(key)

    # Enforceable: phoneme appears in ≥2 distinct tokens at that position.
    enforceable: list[set[int]] = [
        {ph for ph, toks in token_set_pos_phon[pos].items() if len(toks) >= 2}
        for pos in range(n_positions)
    ]

    # Check training coverage only for enforceable phoneme/position pairs.
    train_coverage = [set() for _ in range(n_positions)]
    for idx in train_indices:
        for pos, phon in enumerate(labels[idx]):
            train_coverage[pos].add(phon)

    return all(
        enforceable[pos] <= train_coverage[pos]
        for pos in range(n_positions)
    )


def _patient_seed(patient_id: str) -> int:
    """Deterministic seed from patient ID."""
    return int(hashlib.md5(patient_id.encode()).hexdigest()[:8], 16)


def create_grouped_splits(
    labels: list[list[int]],
    groups: list[int],
    n_folds: int = 5,
    seed: int = 42,
    max_attempts: int = 512,
) -> list[dict]:
    """Create grouped CV splits ensuring no token leakage.

    Uses GroupKFold over shuffled group IDs, retrying deterministic
    permutations until every training fold passes coverage validation.
    """
    n = len(labels)
    rng = np.random.RandomState(seed)

    unique_groups = sorted(set(groups))
    n_folds = min(n_folds, len(unique_groups))

    gkf = GroupKFold(n_splits=n_folds)
    X_dummy = np.zeros(n)
    y_dummy = np.zeros(n)

    for _ in range(max_attempts):
        group_perm = rng.permutation(len(unique_groups))
        group_map = {g: group_perm[i] for i, g in enumerate(unique_groups)}
        shuffled_groups = np.array([group_map[g] for g in groups])

        splits = []
        valid = True
        for train_idx, val_idx in gkf.split(X_dummy, y_dummy, groups=shuffled_groups):
            train_list = sorted(train_idx.tolist())
            val_list = sorted(val_idx.tolist())
            if not validate_fold_coverage(labels, train_list):
                valid = False
                break
            splits.append({
                "train_indices": train_list,
                "val_indices": val_list,
            })

        if valid:
            return splits

    raise RuntimeError(
        f"Failed to find grouped splits with full training-fold coverage after {max_attempts} attempts"
    )


def load_or_create_splits(
    labels: list[list[int]],
    patient_id: str,
    n_folds: int = 5,
    save_path: Path | str | None = None,
) -> list[dict]:
    """Load splits from JSON if they exist, otherwise create and save."""
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
