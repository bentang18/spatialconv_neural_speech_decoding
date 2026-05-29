"""Schema tests for the local `Wang2024Treebank` NeuralSet Study.

Collision policy: see study.py docstring + CLAUDE.md "Wang2024Treebank
upgrade-path". NeuralSet's ``__init_subclass__`` already raises on duplicate
``Wang2024Treebank`` registration, and Pydantic's discriminated-union
registry independently detects the duplicate via its own error path. We
don't add a redundant test here because forcing the duplicate via ``type()``
or ``class`` statements pollutes Pydantic's discriminator registry and
breaks later tests in this module.
"""

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


def test_wang2024treebank_mode_excluded_from_cls_kwargs(tmp_path) -> None:
    """``mode='nano'`` would otherwise be flagged by NeuralSet's
    ``_cls_kwargs`` as an unsupported class parameter, blocking dispatch."""
    study = Wang2024Treebank(path=tmp_path, mode="nano")
    assert study._cls_kwargs() == {}
    timeline_str = study._to_timeline_string(
        {"subject": "btbank1", "subject_id": 1, "trial_id": 1}
    )
    assert "mode" not in timeline_str


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
