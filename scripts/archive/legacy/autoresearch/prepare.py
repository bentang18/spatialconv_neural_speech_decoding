#!/usr/bin/env python3
"""Fixed evaluation harness for autoresearch. DO NOT MODIFY THIS FILE.

Provides:
  - Data loading for target patient (S14) and source patients (cached)
  - Grouped-by-token cross-validation splits (no token leakage)
  - PER (Phoneme Error Rate) via edit distance
  - Content collapse diagnostics

The AI agent modifies train.py only. This file defines ground truth.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import GroupKFold

# === Project setup ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.data.bids_dataset import load_patient_data  # noqa: E402

# ============================================================
# CONSTANTS — DO NOT CHANGE
# ============================================================
TARGET_PATIENT = "S14"
DEV_PATIENT = "S26"
PS_PATIENTS = [
    "S14", "S16", "S22", "S23", "S26",
    "S32", "S33", "S39", "S57", "S58", "S62",
]
SOURCE_PATIENTS = [p for p in PS_PATIENTS if p != TARGET_PATIENT and p != DEV_PATIENT]
TIME_BUDGET = 300  # 5 minutes wall clock per experiment
N_FOLDS = 5
N_POSITIONS = 3
N_CLASSES = 9       # phonemes: a=1, ae=2, b=3, g=4, i=5, k=6, p=7, u=8, v=9

CACHE_DIR = PROJECT_ROOT / ".cache" / "autoresearch"

# Grid shapes per patient (from electrode TSV inspection)
PATIENT_GRIDS = {
    "S14": (8, 16), "S16": (8, 16), "S22": (8, 16), "S23": (8, 16), "S26": (8, 16),
    "S32": (12, 22), "S33": (12, 22), "S39": (12, 22),
    "S57": (8, 34),
    "S58": (12, 22), "S62": (12, 22),
}


# ============================================================
# DATA LOADING (cached after first call)
# ============================================================

def _load_paths() -> dict:
    with open(PROJECT_ROOT / "configs" / "paths.yaml") as f:
        return yaml.safe_load(f)


def load_target_data() -> tuple[torch.Tensor, list[list[int]], list[tuple]]:
    """Load target patient (S14). Cached to disk.

    Returns:
        grids:     (N, 8, 16, 300) float32 — HGA grid data, 200 Hz
        labels:    list of N lists [p1, p2, p3] — 1-indexed phoneme IDs
        token_ids: list of N tuples (p1, p2, p3) — for CV grouping
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"target_{TARGET_PATIENT}.pt"

    if cache.exists():
        d = torch.load(cache, weights_only=False)
        return d["grids"], d["labels"], d["token_ids"]

    paths = _load_paths()
    bids_root = paths.get("ps_bids_root") or paths["bids_root"]
    ds = load_patient_data(
        TARGET_PATIENT, bids_root,
        task="PhonemeSequence", n_phons=3, tmin=-0.5, tmax=1.0,
    )

    grids, labels, token_ids = [], [], []
    for i in range(len(ds)):
        g, l, _ = ds[i]
        grids.append(g)
        labels.append(l)
        token_ids.append(tuple(l))

    grids = torch.tensor(np.stack(grids), dtype=torch.float32)
    torch.save({"grids": grids, "labels": labels, "token_ids": token_ids}, cache)
    return grids, labels, token_ids


def load_source_patients() -> dict[str, torch.Tensor]:
    """Load all source patient grids (no labels). Cached to disk.

    Returns:
        {patient_id: (N_i, H_i, W_i, T) float32}
        Grid shapes vary: (8,16), (12,22), (8,34).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "source_patients.pt"

    if cache.exists():
        return torch.load(cache, weights_only=False)

    paths = _load_paths()
    bids_root = paths.get("ps_bids_root") or paths["bids_root"]
    data = {}
    for pid in SOURCE_PATIENTS:
        ds = load_patient_data(
            pid, bids_root,
            task="PhonemeSequence", n_phons=3, tmin=-0.5, tmax=1.0,
        )
        gs = [ds[i][0] for i in range(len(ds))]
        data[pid] = torch.tensor(np.stack(gs), dtype=torch.float32)

    torch.save(data, cache)
    return data


# ============================================================
# CROSS-VALIDATION
# ============================================================

def create_cv_splits(
    token_ids: list[tuple],
    n_folds: int = N_FOLDS,
) -> list[tuple[list[int], list[int]]]:
    """Grouped-by-token CV: all repetitions of a token in the same fold.

    Retries permutations until every training fold covers all 9 phonemes
    at all 3 positions (prevents unseen-class issues during training).

    Returns list of (train_indices, val_indices).
    """
    unique_tokens = sorted(set(token_ids))
    token_to_group = {t: i for i, t in enumerate(unique_tokens)}
    groups = np.array([token_to_group[t] for t in token_ids])
    n_folds = min(n_folds, len(unique_tokens))

    gkf = GroupKFold(n_splits=n_folds)
    X = np.zeros(len(token_ids))
    rng = np.random.RandomState(42)

    for _ in range(512):
        perm = rng.permutation(len(unique_tokens))
        shuffled = np.array([perm[g] for g in groups])

        splits, valid = [], True
        for train_idx, val_idx in gkf.split(X, groups=shuffled):
            # Check: training has all phonemes at all positions
            for pos in range(N_POSITIONS):
                seen = {token_ids[i][pos] for i in train_idx}
                if len(seen) < N_CLASSES:
                    valid = False
                    break
            if not valid:
                break
            splits.append((sorted(train_idx.tolist()), sorted(val_idx.tolist())))

        if valid:
            return splits

    raise RuntimeError("Failed to create CV splits with full phoneme coverage")


# ============================================================
# EVALUATION METRICS
# ============================================================

def _edit_distance(a: list[int], b: list[int]) -> int:
    """Levenshtein distance."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def compute_per(
    predictions: list[list[int]],
    references: list[list[int]],
) -> float:
    """Phoneme Error Rate = sum(edit_distance) / sum(ref_length).

    Both predictions and references use 1-indexed phoneme IDs.
    Works for variable-length predictions (CTC) and fixed-length (CE).
    """
    total_dist = 0
    total_len = 0
    for pred, ref in zip(predictions, references):
        total_dist += _edit_distance(pred, ref)
        total_len += len(ref)
    return total_dist / max(total_len, 1)


def compute_content_collapse(predictions: list[list[int]]) -> dict:
    """Diagnose content collapse in decoded predictions.

    Returns:
        collapsed:      True if predictions are degenerate
        mean_entropy:   avg per-position entropy (max 3.17 for 9 classes)
        stereotypy:     fraction of most-common complete prediction
        unique_ratio:   unique predictions / total
    """
    if not predictions:
        return {"collapsed": True, "mean_entropy": 0.0, "stereotypy": 1.0, "unique_ratio": 0.0}

    entropies = []
    for pos in range(N_POSITIONS):
        counts = Counter(p[pos] for p in predictions if len(p) > pos)
        total = sum(counts.values())
        if total == 0:
            entropies.append(0.0)
            continue
        probs = np.array([counts.get(c, 0) / total for c in range(1, N_CLASSES + 1)])
        probs = probs[probs > 0]
        entropies.append(float(-np.sum(probs * np.log2(probs))))

    max_ent = float(np.log2(N_CLASSES))  # 3.17
    mean_ent = float(np.mean(entropies))

    full = Counter(tuple(p) for p in predictions)
    stereotypy = full.most_common(1)[0][1] / len(predictions)
    unique_ratio = len(full) / len(predictions)

    return {
        "collapsed": mean_ent < max_ent * 0.6 or stereotypy > 0.3,
        "mean_entropy": mean_ent,
        "max_entropy": max_ent,
        "stereotypy": stereotypy,
        "unique_ratio": unique_ratio,
        "per_position_entropy": entropies,
    }
