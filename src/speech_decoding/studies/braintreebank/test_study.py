"""Schema tests for the local `Wang2024Treebank` NeuralSet Study."""

from __future__ import annotations

import ujson
import neuralset as ns

from speech_decoding.studies.braintreebank.manifest import (
    BT_FULL_SESSIONS,
    BT_LITE_SESSIONS,
)
from speech_decoding.studies.braintreebank.study import Wang2024Treebank


def test_wang2024treebank_registers_with_neuralset_catalog() -> None:
    catalog = ns.Study.catalog()

    assert catalog["Wang2024Treebank"] is Wang2024Treebank


def test_wang2024treebank_iter_timelines_matches_full_manifest(tmp_path) -> None:
    study = Wang2024Treebank(path=tmp_path, mode="full")
    timelines = list(study.iter_timelines())

    assert len(timelines) == len(BT_FULL_SESSIONS)
    assert timelines[0] == {"subject": "btbank1", "subject_id": 1, "trial_id": 0}
    assert timelines[-1] == {"subject": "btbank10", "subject_id": 10, "trial_id": 1}


def test_wang2024treebank_defaults_to_lite_manifest(tmp_path) -> None:
    study = Wang2024Treebank(path=tmp_path)
    timelines = list(study.iter_timelines())

    assert len(timelines) == len(BT_LITE_SESSIONS)
    assert timelines[0] == {"subject": "btbank1", "subject_id": 1, "trial_id": 1}


def test_wang2024treebank_emits_ieeg_special_loader_without_reading_raw(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        Wang2024Treebank,
        "_trial_duration_seconds",
        lambda self, timeline: 10449.71142578125,
    )
    study = ns.Study(
        name="Wang2024Treebank",
        path=tmp_path,
        query="timeline_index < 1",
        infra_timelines={"cluster": None},
    )

    events = study.run()

    assert len(events) == 1
    event = events.iloc[0]
    assert event["type"] == "Ieeg"
    assert event["study"] == "Wang2024Treebank"
    assert event["subject"] == "Wang2024Treebank/btbank1"
    assert event["timeline"] == (
        "Wang2024Treebank:subject=btbank1,subject_id=1,trial_id=1"
    )
    assert event["start"] == 0.0
    assert event["duration"] == 10449.71142578125
    assert event["frequency"] == 2048.0
    assert event["subject_id"] == "1"
    assert event["trial_id"] == "1"

    loader = ujson.loads(event["filepath"])
    assert loader["cls"] == "Wang2024Treebank"
    assert loader["method"] == "_load_raw"
    assert loader["timeline"] == {"subject": "btbank1", "subject_id": 1, "trial_id": 1}
