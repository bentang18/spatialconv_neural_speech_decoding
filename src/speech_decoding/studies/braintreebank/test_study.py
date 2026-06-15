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
    V14_PRETRAIN_SESSIONS,
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


def test_wang2024treebank_pretrain_mode_emits_exactly_legal_corpus(tmp_path) -> None:
    """LC2/LC5: mode='pretrain' emits EXACTLY V14_PRETRAIN_SESSIONS (13 sessions,
    subjects {1,2,3,4,6,8,9}, disjoint from BT_LITE) — the disk->emission wiring
    the runtime leakage guard cannot see (it only checks realized loaders for the
    LEAK). Re-pointing _SESSIONS_BY_MODE['pretrain'] to the eval-containing
    V14_TRAIN_SESSIONS would fail here immediately. The subject set IS the
    S7/S10-zero-shot pin (both absent from the standard legal corpus)."""
    study = Wang2024Treebank(path=tmp_path, mode="pretrain")
    emitted = {(tl["subject_id"], tl["trial_id"]) for tl in study.iter_timelines()}
    assert emitted == set(V14_PRETRAIN_SESSIONS)
    assert len(emitted) == 13
    assert sorted({s for s, _ in emitted}) == [1, 2, 3, 4, 6, 8, 9]
    # S7/S10 contribute zero standard legal sessions (zero-shot, deferred).
    assert 7 not in {s for s, _ in emitted}
    assert 10 not in {s for s, _ in emitted}
    # Disjoint from the eval set — the leakage boundary at the emission layer.
    assert emitted.isdisjoint(set(BT_LITE_SESSIONS))


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


# --- electrode_set / extra_bad folded into the RAW exca cache uid -------------
#
# The per-session timeline IS the SpecialLoader payload, and the raw exca cache
# uid = _splittable_event_uid() over study_relative_path() == that JSON. So
# anything injected into the timeline distinguishes the cache (full-CAR vs
# Lite-CAR) AND auto-invalidates it on a bad-electrode edit — and because the
# cache KEY and the data LOAD both read the SAME dict, they can never disagree
# (the 2026-06-15 static-cache stale-raw trap, structurally closed). Injection
# is CONDITIONAL so a clean subject's existing "all" cache stays byte-valid.


def test_iter_timelines_injects_electrode_set_and_extra_bad_conditionally(
    tmp_path,
) -> None:
    from speech_decoding.studies.braintreebank.anatomy import extra_bad_electrodes

    all_tls = list(Wang2024Treebank(path=tmp_path, mode="lite").iter_timelines())
    lite_tls = list(
        Wang2024Treebank(path=tmp_path, mode="lite", electrode_set="lite").iter_timelines()
    )

    # electrode_set: absent under the "all" default (keeps the existing cache
    # valid); present on every "lite" timeline.
    assert all("electrode_set" not in tl for tl in all_tls)
    assert all(tl.get("electrode_set") == "lite" for tl in lite_tls)

    # extra_bad: present (sorted list) iff the subject has STATIC bad contacts,
    # absent otherwise — same conditional under both montages.
    for tl in all_tls + lite_tls:
        bad = sorted(extra_bad_electrodes(tl["subject_id"]))
        if bad:
            assert tl["extra_bad"] == bad
        else:
            assert "extra_bad" not in tl


def test_lite_special_loader_uid_differs_from_all(monkeypatch, tmp_path) -> None:
    """full-CAR and Lite-CAR can never collide in the raw exca cache: the
    SpecialLoader filepath (= the per-item cache uid) embeds the timeline, so
    electrode_set='lite' yields a distinct key for the same session."""
    monkeypatch.setattr(
        Wang2024Treebank, "_trial_duration_seconds", lambda self, tl: 10.0
    )

    def first_filepath(electrode_set: str) -> str:
        study = Wang2024Treebank(path=tmp_path, mode="lite", electrode_set=electrode_set)
        tl = next(iter(study.iter_timelines()))
        return study._load_timeline_events(tl).iloc[0]["filepath"]

    all_uid = first_filepath("all")
    lite_uid = first_filepath("lite")
    assert all_uid != lite_uid
    assert "electrode_set" not in ujson.loads(all_uid)["timeline"]
    assert ujson.loads(lite_uid)["timeline"]["electrode_set"] == "lite"


def test_extra_bad_edit_changes_special_loader_uid(monkeypatch, tmp_path) -> None:
    """Editing the STATIC bad-electrode set auto-invalidates the raw cache: the
    per-item SpecialLoader uid changes when extra_bad changes (closes the
    2026-06-15 stale-raw-cache trap — KEY and LOAD share one timeline dict)."""
    import speech_decoding.studies.braintreebank.study as study_mod

    monkeypatch.setattr(
        Wang2024Treebank, "_trial_duration_seconds", lambda self, tl: 10.0
    )

    def first_filepath(bad: set[str]) -> str:
        monkeypatch.setattr(
            study_mod, "extra_bad_electrodes", lambda subject_id: frozenset(bad)
        )
        study = Wang2024Treebank(path=tmp_path, mode="lite")
        tl = next(iter(study.iter_timelines()))
        return study._load_timeline_events(tl).iloc[0]["filepath"]

    clean = first_filepath(set())
    with_bad = first_filepath({"E9"})
    assert clean != with_bad
    assert "extra_bad" not in ujson.loads(clean)["timeline"]
    assert ujson.loads(with_bad)["timeline"]["extra_bad"] == ["E9"]
