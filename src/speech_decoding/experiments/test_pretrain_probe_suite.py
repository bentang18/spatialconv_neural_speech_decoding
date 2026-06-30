"""Tests for the firewall-legal cohort + cell enumeration (WALL OF FIRE)."""

from __future__ import annotations

import dataclasses

import pytest

from speech_decoding.experiments.pretrain_probe_suite import (
    CS_TRAIN_ANCHOR_CANDIDATES,
    CS_TRAIN_SUBJECT,
    DEFAULT_CS_TRAIN_ANCHOR,
    NEUROPROBE_TASKS,
    PRETRAIN_UNIVERSE,
    PROBE_COHORT_7,
    WS_N_FOLDS,
    assert_cells_firewall_legal,
    enumerate_cells,
)
from speech_decoding.studies.braintreebank.manifest import (
    BT_LITE_SESSIONS,
    V14_EXCLUDED_SUBJECT_IDS,
)

_LITE = set(BT_LITE_SESSIONS)


def test_universe_disjoint_from_eval_set():
    """🔥 The cohort never intersects the Neuroprobe eval set."""
    assert set(PRETRAIN_UNIVERSE).isdisjoint(_LITE)


def test_universe_excludes_v14_excluded_subjects():
    subjects = {s for (s, _) in PRETRAIN_UNIVERSE}
    assert subjects.isdisjoint(set(V14_EXCLUDED_SUBJECT_IDS))
    assert subjects == {1, 2, 3, 4, 6, 8, 9}


def test_cs_train_anchor_is_firewall_legal_subject_2():
    assert DEFAULT_CS_TRAIN_ANCHOR == (2, 1)
    assert DEFAULT_CS_TRAIN_ANCHOR[0] == CS_TRAIN_SUBJECT
    assert DEFAULT_CS_TRAIN_ANCHOR not in _LITE
    for anchor in CS_TRAIN_ANCHOR_CANDIDATES:
        assert anchor[0] == 2
        assert anchor not in _LITE


def test_within_session_cell_count():
    cells = enumerate_cells(("WithinSession",), NEUROPROBE_TASKS)
    assert len(cells) == len(PRETRAIN_UNIVERSE) * len(NEUROPROBE_TASKS) * WS_N_FOLDS
    assert {c.eval_mode for c in cells} == {"WithinSession"}
    assert {c.fold_index for c in cells} == set(range(WS_N_FOLDS))


def test_cross_subject_drops_train_subject_and_uses_anchor():
    cells = enumerate_cells(("CrossSubject",), NEUROPROBE_TASKS)
    assert {c.test_subject_id for c in cells} == {1, 3, 4, 6, 8, 9}
    for c in cells:
        assert (c.train_subject_id, c.train_trial_id) == DEFAULT_CS_TRAIN_ANCHOR
    n_test_sessions = sum(1 for (s, _) in PRETRAIN_UNIVERSE if s != 2)
    assert len(cells) == n_test_sessions * len(NEUROPROBE_TASKS)


def test_probe_cohort_7_is_one_firewall_legal_trial_per_subject():
    """The 13→7 cut: one trial per subject, the CS anchor included, all firewall-legal."""
    assert len(PROBE_COHORT_7) == 7
    assert {s for (s, _) in PROBE_COHORT_7} == {1, 2, 3, 4, 6, 8, 9}
    assert len({s for (s, _) in PROBE_COHORT_7}) == 7          # one trial per subject
    assert set(PROBE_COHORT_7).issubset(set(PRETRAIN_UNIVERSE))
    assert DEFAULT_CS_TRAIN_ANCHOR in PROBE_COHORT_7
    assert set(PROBE_COHORT_7).isdisjoint(_LITE)


def test_enumerate_cells_restricts_to_cohort():
    """Passing the 7-cohort enumerates cells only over those sessions (no missing-cache
    cells): WS test sessions ⊆ cohort, CS test subjects = cohort minus the anchor subject."""
    ws = enumerate_cells(("WithinSession",), NEUROPROBE_TASKS, cohort=PROBE_COHORT_7)
    assert {(c.test_subject_id, c.test_trial_id) for c in ws} == set(PROBE_COHORT_7)
    assert len(ws) == len(PROBE_COHORT_7) * len(NEUROPROBE_TASKS) * WS_N_FOLDS
    cs = enumerate_cells(("CrossSubject",), NEUROPROBE_TASKS, cohort=PROBE_COHORT_7)
    assert {c.test_subject_id for c in cs} == {1, 3, 4, 6, 8, 9}
    assert len(cs) == 6 * len(NEUROPROBE_TASKS)
    assert_cells_firewall_legal(cs)


def test_enumerate_rejects_crosssession():
    with pytest.raises(ValueError, match="WithinSession"):
        enumerate_cells(("CrossSession",), NEUROPROBE_TASKS)


def test_cross_subject_rejects_lite_anchor():
    # (2, 4) is a leaderboard eval cell — never a legal train anchor.
    with pytest.raises(ValueError, match="PRETRAIN_UNIVERSE"):
        enumerate_cells(("CrossSubject",), NEUROPROBE_TASKS, cs_train_anchor=(2, 4))


def test_firewall_guard_catches_injected_eval_cell():
    legal = enumerate_cells(("WithinSession",), ("onset",))
    leaked = dataclasses.replace(
        legal[0], test_subject_id=BT_LITE_SESSIONS[0][0],
        test_trial_id=BT_LITE_SESSIONS[0][1],
    )
    with pytest.raises(AssertionError, match="WALL OF FIRE"):
        assert_cells_firewall_legal([leaked])


def test_firewall_guard_catches_injected_eval_anchor():
    legal = enumerate_cells(("CrossSubject",), ("onset",))
    leaked = dataclasses.replace(legal[0], train_subject_id=2, train_trial_id=4)
    with pytest.raises(AssertionError, match="WALL OF FIRE"):
        assert_cells_firewall_legal([leaked])
