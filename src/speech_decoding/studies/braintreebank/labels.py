"""Label metadata for BrainTreebank's 15-task Neuroprobe suite.

Full per-window label parity lands in Stage-0 E3. This module owns the stable
task names and Neuroprobe column remapping so downstream configs do not depend
on importing Neuroprobe internals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NEUROPROBE_TASKS: tuple[str, ...] = (
    "onset",
    "speech",
    "volume",
    "delta_volume",
    "pitch",
    "word_index",
    "word_gap",
    "gpt2_surprisal",
    "word_head_pos",
    "word_part_speech",
    "word_length",
    "global_flow",
    "local_flow",
    "frame_brightness",
    "face_num",
)
"""Canonical task order from `neuroprobe.config.NEUROPROBE_TASKS`."""

SINGLE_FLOAT_TASK_COLUMNS: dict[str, str] = {
    "pitch": "enhanced_pitch",
    "volume": "rms",
    "frame_brightness": "mean_pixel_brightness",
    "global_flow": "max_global_magnitude",
    "local_flow": "max_vector_magnitude",
    "delta_volume": "delta_rms",
    "gpt2_surprisal": "gpt2_surprisal",
    "word_length": "word_length",
}
"""Continuous Neuroprobe tasks and their BrainTreebank feature columns."""

CLASSIFICATION_TASK_COLUMNS: dict[str, str] = {
    "word_head_pos": "bin_head",
    "word_part_speech": "pos",
}
"""Categorical Neuroprobe tasks and their BrainTreebank feature columns."""

BINARY_TASKS: tuple[str, ...] = NEUROPROBE_TASKS
"""All Neuroprobe tasks support the binary leaderboard formulation."""

NEW_PITCH_VOLUME_COLUMNS: tuple[str, ...] = (
    "enhanced_pitch",
    "enhanced_volume",
    "delta_enhanced_pitch",
    "delta_enhanced_volume",
    "raw_pitch",
    "raw_volume",
    "delta_raw_pitch",
    "delta_raw_volume",
)
"""Pitch/volume columns loaded from Neuroprobe's packaged JSON features."""

LITE_MAX_SAMPLES = 3500
NANO_MAX_SAMPLES = 1000
GLOBAL_RANDOM_SEED = 42


def remap_task_column(task: str) -> str:
    """Return the BrainTreebank feature column used by a Neuroprobe task."""
    if task in SINGLE_FLOAT_TASK_COLUMNS:
        return SINGLE_FLOAT_TASK_COLUMNS[task]
    if task in CLASSIFICATION_TASK_COLUMNS:
        return CLASSIFICATION_TASK_COLUMNS[task]
    if task in NEUROPROBE_TASKS:
        return task
    raise KeyError(f"unknown Neuroprobe task: {task}")


def enrich_words_with_transcript_features(
    words_df: pd.DataFrame,
    transcript_features: pd.DataFrame,
) -> pd.DataFrame:
    """Mirror Neuroprobe's transcript-feature enrichment by original_index."""
    enriched = words_df.copy()
    features = transcript_features.set_index("Unnamed: 0")
    new_columns = [column for column in features.columns if column not in enriched.columns]
    for column in new_columns:
        enriched[column] = enriched["original_index"].map(features[column])
    return enriched


def load_pitch_volume_features(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load Neuroprobe pitch/volume JSON with normalized 5-decimal keys."""
    raw = json.loads(Path(path).read_text())
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        normalized[f"{float(key):.5f}"] = value
    return normalized


def derive_label_indices(
    *,
    words_df: pd.DataFrame,
    nonverbal_df: pd.DataFrame,
    task: str,
    binary_tasks: bool = True,
    pitch_volume_features: dict[str, dict[str, Any]] | None = None,
    random_seed: int = GLOBAL_RANDOM_SEED,
    lite: bool = True,
    nano: bool = False,
    max_samples: int | None = None,
    balance: bool = True,
) -> dict[int, np.ndarray]:
    """Return Neuroprobe-compatible balanced label indices for one task.

    ``balance=False`` (label-free SSL): keep EVERY index per class — no
    cross-class ``min()`` down-sampling. The lite/nano cap, if set, still bounds
    each class so a smoke budget stays small. P4 eval keeps ``balance=True`` for
    leaderboard parity.
    """
    if task not in NEUROPROBE_TASKS:
        raise KeyError(f"unknown Neuroprobe task: {task}")
    task_column = remap_task_column(task)

    if task in SINGLE_FLOAT_TASK_COLUMNS:
        all_labels = _continuous_labels(
            words_df=words_df,
            task_column=task_column,
            pitch_volume_features=pitch_volume_features,
        )
        label_percentiles = np.array([np.mean(all_labels < value) for value in all_labels])
        if binary_tasks:
            label_indices = {
                1: np.where(label_percentiles > 0.75)[0],
                0: np.where(label_percentiles < 0.25)[0],
            }
        else:
            label_indices = {
                2: np.where(label_percentiles >= 0.75)[0],
                1: np.where((label_percentiles < 0.625) & (label_percentiles >= 0.375))[0],
                0: np.where(label_percentiles < 0.25)[0],
            }
    elif task in {"onset", "speech"}:
        label_indices = {
            1: np.where(words_df["is_onset"].to_numpy() == 1)[0]
            if task == "onset"
            else np.arange(len(words_df)),
            0: np.arange(len(nonverbal_df)),
        }
    elif task == "face_num":
        face_nums = words_df["face_num"].to_numpy().astype(int)
        if binary_tasks:
            label_indices = {
                1: np.where(face_nums > 0)[0],
                0: np.where(face_nums == 0)[0],
            }
        else:
            label_indices = {
                2: np.where(face_nums > 1)[0],
                1: np.where(face_nums == 1)[0],
                0: np.where(face_nums == 0)[0],
            }
    elif task == "word_index":
        word_indices = words_df["idx_in_sentence"].to_numpy().astype(int)
        if binary_tasks:
            label_indices = {
                1: np.where(word_indices == 0)[0],
                0: np.where(word_indices == 1)[0],
            }
        else:
            label_indices = {
                2: np.where(word_indices >= 2)[0],
                1: np.where(word_indices == 1)[0],
                0: np.where(word_indices == 0)[0],
            }
    elif task == "word_head_pos":
        head_pos = words_df[task_column].to_numpy().astype(int)
        label_indices = {
            1: np.where(head_pos == 0)[0],
            0: np.where(head_pos == 1)[0],
        }
    elif task == "word_part_speech":
        pos = words_df[task_column].to_numpy()
        if binary_tasks:
            label_indices = {
                1: np.where(pos == "VERB")[0],
                0: np.where(pos == "NOUN")[0],
            }
        else:
            label_indices = {
                5: np.where(pos == "ADV")[0],
                4: np.where(pos == "ADJ")[0],
                3: np.where(pos == "DET")[0],
                2: np.where(pos == "PRON")[0],
                1: np.where(pos == "VERB")[0],
                0: np.where(pos == "NOUN")[0],
            }
    elif task == "word_gap":
        label_indices = _word_gap_indices(words_df, binary_tasks=binary_tasks)
    else:
        raise KeyError(f"unknown Neuroprobe task: {task}")

    return _balance_label_indices(
        label_indices=label_indices,
        random_seed=random_seed,
        lite=lite,
        nano=nano,
        max_samples=max_samples,
        balance=balance,
    )


def ordered_dataset_labels(label_indices: dict[int, np.ndarray]) -> np.ndarray:
    """Return labels in Neuroprobe's Dataset.__getitem__ order."""
    n_classes = len(label_indices)
    n_samples = sum(len(indices) for indices in label_indices.values())
    return np.array([(idx + 1) % n_classes for idx in range(n_samples)], dtype=int)


def ordered_dataset_source_indices(label_indices: dict[int, np.ndarray]) -> np.ndarray:
    """Return source word/nonverbal row indices in Neuroprobe item order."""
    n_classes = len(label_indices)
    n_samples = sum(len(indices) for indices in label_indices.values())
    return np.array(
        [label_indices[(idx + 1) % n_classes][idx // n_classes] for idx in range(n_samples)],
        dtype=int,
    )


def _continuous_labels(
    *,
    words_df: pd.DataFrame,
    task_column: str,
    pitch_volume_features: dict[str, dict[str, Any]] | None,
) -> np.ndarray:
    if task_column in NEW_PITCH_VOLUME_COLUMNS:
        if pitch_volume_features is None:
            raise ValueError(f"{task_column} requires pitch_volume_features")
        labels = []
        for start_time in words_df["start"].to_list():
            labels.append(pitch_volume_features[f"{start_time:.5f}"][task_column])
        return np.asarray(labels)
    return words_df[task_column].to_numpy()


def _word_gap_indices(words_df: pd.DataFrame, *, binary_tasks: bool) -> dict[int, np.ndarray]:
    word_gap_distribution = []
    for idx in range(1, len(words_df)):
        if words_df.iloc[idx]["sentence"] != words_df.iloc[idx - 1]["sentence"]:
            continue
        word_gap_distribution.append(words_df.iloc[idx]["start"] - words_df.iloc[idx - 1]["end"])
    word_gap_distribution = np.array(word_gap_distribution)

    positive_indices = []
    negative_indices = []
    middle_indices = []
    for idx in range(1, len(words_df)):
        if words_df.iloc[idx]["sentence"] != words_df.iloc[idx - 1]["sentence"]:
            continue
        gap = words_df.iloc[idx]["start"] - words_df.iloc[idx - 1]["end"]
        gap_percentile = np.mean(word_gap_distribution < gap)
        if gap_percentile >= 0.75:
            positive_indices.append(idx)
        elif (gap_percentile >= 0.375) and (gap_percentile < 0.625):
            middle_indices.append(idx)
        elif gap_percentile < 0.25:
            negative_indices.append(idx)
    if binary_tasks:
        return {
            1: np.asarray(positive_indices),
            0: np.asarray(negative_indices),
        }
    return {
        2: np.asarray(positive_indices),
        1: np.asarray(middle_indices),
        0: np.asarray(negative_indices),
    }


def _balance_label_indices(
    *,
    label_indices: dict[int, np.ndarray],
    random_seed: int,
    lite: bool,
    nano: bool,
    max_samples: int | None,
    balance: bool = True,
) -> dict[int, np.ndarray]:
    rng = np.random.RandomState(random_seed)
    n_classes = len(label_indices)
    cap: int | None = None
    if lite:
        cap = LITE_MAX_SAMPLES // n_classes
    elif nano:
        cap = NANO_MAX_SAMPLES // n_classes

    if not balance:
        # Label-free SSL: keep EVERY index per class (no cross-class min). A
        # set lite/nano cap still bounds each class so smoke budgets stay small.
        out: dict[int, np.ndarray] = {}
        for label in list(label_indices.keys()):
            idxs = label_indices[label]
            if cap is not None and len(idxs) > cap:
                idxs = rng.choice(idxs, size=cap, replace=False)
            idxs = np.sort(idxs)
            if max_samples is not None:
                idxs = idxs[: max_samples // n_classes]
            out[label] = idxs
        return out

    n_samples_each = min(len(indices) for indices in label_indices.values())
    if cap is not None:
        n_samples_each = min(n_samples_each, cap)

    balanced: dict[int, np.ndarray] = {}
    for label in list(label_indices.keys()):
        chosen = np.sort(rng.choice(label_indices[label], size=n_samples_each, replace=False))
        if max_samples is not None:
            chosen = chosen[: max_samples // n_classes]
        balanced[label] = chosen
    return balanced
