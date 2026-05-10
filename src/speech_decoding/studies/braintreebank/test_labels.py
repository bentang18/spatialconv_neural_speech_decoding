from __future__ import annotations

import pytest
import pandas as pd

from speech_decoding.studies.braintreebank.labels import (
    CLASSIFICATION_TASK_COLUMNS,
    NEUROPROBE_TASKS,
    SINGLE_FLOAT_TASK_COLUMNS,
    derive_label_indices,
    ordered_dataset_labels,
    ordered_dataset_source_indices,
    remap_task_column,
)


def test_neuroprobe_task_order_is_stable() -> None:
    assert NEUROPROBE_TASKS == (
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


def test_task_column_remaps_match_neuroprobe_contract() -> None:
    assert remap_task_column("pitch") == "enhanced_pitch"
    assert remap_task_column("volume") == "rms"
    assert remap_task_column("word_head_pos") == "bin_head"
    assert remap_task_column("word_part_speech") == "pos"
    assert remap_task_column("onset") == "onset"
    assert set(SINGLE_FLOAT_TASK_COLUMNS).isdisjoint(CLASSIFICATION_TASK_COLUMNS)


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown Neuroprobe task"):
        remap_task_column("not_a_task")


def test_continuous_binary_label_indices_match_neuroprobe_percentiles() -> None:
    words = _words(value=[0.0, 1.0, 2.0, 3.0, 4.0], enhanced_pitch=[10, 11, 12, 13, 14])
    labels = derive_label_indices(
        words_df=words,
        nonverbal_df=_nonverbal(),
        task="gpt2_surprisal",
        binary_tasks=True,
        lite=False,
    )
    assert labels[1].tolist() == [4]
    assert labels[0].tolist() == [1]
    assert ordered_dataset_labels(labels).tolist() == [1, 0]
    assert ordered_dataset_source_indices(labels).tolist() == [4, 1]


def test_continuous_multiclass_label_indices_match_neuroprobe_bins() -> None:
    words = _words(value=list(range(9)), enhanced_pitch=list(range(9)))
    labels = derive_label_indices(
        words_df=words,
        nonverbal_df=_nonverbal(),
        task="gpt2_surprisal",
        binary_tasks=False,
        lite=False,
    )
    assert labels[2].tolist() == [7, 8]
    assert labels[1].tolist() == [4, 5]
    assert labels[0].tolist() == [1, 2]
    assert ordered_dataset_labels(labels).tolist() == [1, 2, 0, 1, 2, 0]


def test_pitch_volume_features_are_loaded_from_normalized_start_keys() -> None:
    words = _words(value=list(range(5)), enhanced_pitch=[0] * 5)
    pitch_features = {
        f"{float(i):.5f}": {"enhanced_pitch": float(i)}
        for i in range(5)
    }
    labels = derive_label_indices(
        words_df=words,
        nonverbal_df=_nonverbal(),
        task="pitch",
        binary_tasks=True,
        pitch_volume_features=pitch_features,
        lite=False,
    )
    assert labels[1].tolist() == [4]
    assert labels[0].tolist() == [1]


def test_word_gap_indices_skip_sentence_boundaries() -> None:
    words = _words(value=list(range(6)), enhanced_pitch=list(range(6)))
    words["sentence"] = ["a", "a", "a", "b", "b", "b"]
    words["start"] = [0.0, 1.0, 3.0, 0.0, 1.5, 4.5]
    words["end"] = [0.5, 1.5, 3.5, 0.5, 2.0, 5.0]
    labels = derive_label_indices(
        words_df=words,
        nonverbal_df=_nonverbal(),
        task="word_gap",
        binary_tasks=False,
        lite=False,
    )
    assert set(labels) == {0, 1, 2}


def _words(**columns: list[object]) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    return pd.DataFrame(
        {
            "original_index": list(range(n)),
            "start": [float(i) for i in range(n)],
            "end": [float(i) + 0.5 for i in range(n)],
            "est_idx": [i * 2048 for i in range(n)],
            "is_onset": [1] + [0] * (n - 1),
            "sentence": ["s"] * n,
            "idx_in_sentence": list(range(n)),
            "face_num": [0, 1, 2, 0, 1, 2, 0, 1, 2][:n],
            "bin_head": [0, 1, 0, 1, 0, 1, 0, 1, 0][:n],
            "pos": ["NOUN", "VERB", "PRON", "DET", "ADJ", "ADV", "NOUN", "VERB", "PRON"][:n],
            "gpt2_surprisal": columns["value"],
            "enhanced_pitch": columns["enhanced_pitch"],
        }
    )


def _nonverbal() -> pd.DataFrame:
    return pd.DataFrame({"est_idx": [0, 2048, 4096]})
