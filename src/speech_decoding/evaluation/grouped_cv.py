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
