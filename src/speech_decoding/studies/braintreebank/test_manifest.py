from __future__ import annotations

import importlib.util

import pytest

from speech_decoding.studies.braintreebank.manifest import (
    BT_FULL_SESSIONS,
    BT_LITE_SESSIONS,
    BT_NANO_SESSIONS,
    BT_PRETRAIN_ALLOWED_SESSIONS,
    BT_PRETRAIN_PARTIAL_SESSIONS,
    NO_TEACHER_CACHE_SESSIONS,
    V14_EXCLUDED_SUBJECT_IDS,
    V14_LEADERBOARD_SUBJECT_IDS,
    V14_P3_DISTILL_SESSIONS,
    V14_PRETRAIN_SESSIONS,
    V14_TRAIN_SESSIONS,
    V14_TRAIN_SUBJECT_IDS,
)


def _load_upstream_neuroprobe_config(monkeypatch, tmp_path):
    """Load the installed upstream ``neuroprobe.config`` module in isolation.

    ``neuroprobe/__init__`` pulls in ``braintreebank_subject`` (heavy, data-
    bound); ``config.py`` itself only needs ``ROOT_DIR_BRAINTREEBANK`` set at
    import and otherwise defines pure literals. We load the file directly via
    importlib so the parity check depends on the pinned upstream constant, not on
    BT data being present. importorskip when neuroprobe isn't installed."""
    spec = importlib.util.find_spec("neuroprobe")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("upstream neuroprobe not installed")
    cfg_path = next(
        p for loc in spec.submodule_search_locations
        for p in [__import__("pathlib").Path(loc) / "config.py"]
        if p.exists()
    )
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    cfg_spec = importlib.util.spec_from_file_location(
        "_neuroprobe_config_parity_probe", cfg_path
    )
    module = importlib.util.module_from_spec(cfg_spec)
    cfg_spec.loader.exec_module(module)
    return module


def test_bt_lite_sessions_match_upstream_neuroprobe_lite(monkeypatch, tmp_path) -> None:
    """LC6: BT_LITE_SESSIONS is bit-identical to the pinned upstream
    NEUROPROBE_LITE_SUBJECT_TRIALS (12 sessions). This is the ONLY place that
    ties our internal off-limits set to the external eval set every guard
    subtracts against — an upstream re-pin or a local literal drift would
    otherwise pass all internal-consistency tests while silently changing the
    leakage boundary."""
    cfg = _load_upstream_neuroprobe_config(monkeypatch, tmp_path)
    upstream = {(int(s), int(t)) for s, t in cfg.NEUROPROBE_LITE_SUBJECT_TRIALS}
    assert upstream == set(BT_LITE_SESSIONS)
    assert len(BT_LITE_SESSIONS) == 12 == len(upstream)


def test_v14_cohort_excludes_s5() -> None:
    assert V14_EXCLUDED_SUBJECT_IDS == (5,)
    assert 5 not in V14_TRAIN_SUBJECT_IDS
    assert V14_TRAIN_SUBJECT_IDS == (1, 2, 3, 4, 6, 7, 8, 9, 10)
    assert len(V14_TRAIN_SUBJECT_IDS) == 9


def test_v14_leaderboard_subset_matches_neuroprobe_lite() -> None:
    lite_subjects = sorted({s for s, _ in BT_LITE_SESSIONS})
    assert V14_LEADERBOARD_SUBJECT_IDS == tuple(lite_subjects)
    assert set(V14_LEADERBOARD_SUBJECT_IDS).issubset(set(V14_TRAIN_SUBJECT_IDS))


def test_v14_train_sessions_filter_excluded_subjects() -> None:
    assert all(
        s in V14_TRAIN_SUBJECT_IDS for s, _ in V14_TRAIN_SESSIONS
    )
    expected = tuple(
        (s, t) for s, t in BT_FULL_SESSIONS if s in V14_TRAIN_SUBJECT_IDS
    )
    assert V14_TRAIN_SESSIONS == expected
    # S5's single session (5, 0) is the only drop from BT_FULL_SESSIONS.
    assert (5, 0) not in V14_TRAIN_SESSIONS
    assert len(V14_TRAIN_SESSIONS) == len(BT_FULL_SESSIONS) - 1


def test_pretrain_allowed_is_full_minus_eval_and_disjoint() -> None:
    """Neuroprobe SUBMIT.md: the pretraining-allowed standard set is exactly
    BT_FULL_SESSIONS minus the 12 off-limits eval sessions (== BT_LITE_SESSIONS).
    The legality of any SSL corpus rests on this disjointness."""
    allowed = set(BT_PRETRAIN_ALLOWED_SESSIONS)
    eval_sessions = set(BT_LITE_SESSIONS)
    # Disjoint from the eval set — the core leakage invariant.
    assert allowed.isdisjoint(eval_sessions)
    # Exactly the non-eval remainder of the full corpus.
    assert allowed == set(BT_FULL_SESSIONS) - eval_sessions
    assert len(BT_PRETRAIN_ALLOWED_SESSIONS) == 14
    # Subjects 7 and 10 have no standard allowed session (both trials are eval).
    assert all(s not in (7, 10) for s, _ in BT_PRETRAIN_ALLOWED_SESSIONS)
    # Partials cover exactly subjects 7 and 10, with non-eval pseudo-trial ids.
    assert sorted({s for s, _ in BT_PRETRAIN_PARTIAL_SESSIONS}) == [7, 10]
    assert all(t >= 100 for _, t in BT_PRETRAIN_PARTIAL_SESSIONS)
    assert set(BT_PRETRAIN_PARTIAL_SESSIONS).isdisjoint(eval_sessions)


def test_v14_pretrain_sessions_legal_and_cohort_scoped() -> None:
    """The cohort-restricted legal SSL set drops S5 and stays disjoint from the
    eval set. Subjects 7/10 contribute nothing here (partials, not on DCC)."""
    legal = set(V14_PRETRAIN_SESSIONS)
    assert legal.isdisjoint(set(BT_LITE_SESSIONS))
    assert all(s != 5 for s, _ in V14_PRETRAIN_SESSIONS)
    assert legal == {
        (s, t) for (s, t) in BT_PRETRAIN_ALLOWED_SESSIONS
        if s in V14_TRAIN_SUBJECT_IDS
    }
    assert len(V14_PRETRAIN_SESSIONS) == 13
    # Standard legal coverage spans {1,2,3,4,6,8,9}; 7 and 10 are blind here.
    assert sorted({s for s, _ in V14_PRETRAIN_SESSIONS}) == [1, 2, 3, 4, 6, 8, 9]


def test_v14_p3_distill_sessions_drop_no_teacher_session() -> None:
    """LC4: the P3 distill corpus is the pretrain corpus minus the no-teacher
    sessions ((8,0)). 12 sessions over {1,2,3,4,6,9}; subject 8 drops because its
    only session has no Whisper teacher cache. Still leakage-free (a subset of
    the legal pretrain set, disjoint from the eval set)."""
    assert NO_TEACHER_CACHE_SESSIONS == ((8, 0),)
    distill = set(V14_P3_DISTILL_SESSIONS)
    assert distill == set(V14_PRETRAIN_SESSIONS) - set(NO_TEACHER_CACHE_SESSIONS)
    assert distill.isdisjoint(set(BT_LITE_SESSIONS))
    assert len(V14_P3_DISTILL_SESSIONS) == 12
    assert sorted({s for s, _ in V14_P3_DISTILL_SESSIONS}) == [1, 2, 3, 4, 6, 9]
    assert (8, 0) not in distill


def test_bt_nano_carries_two_trials_for_default_test_subject() -> None:
    """The default CrossSession contract picks ``test_subject_id=2,
    test_trial_id=4`` and trains on *other* trials of the same subject.
    Nano must therefore carry a second trial of subject 2 (``(2, 0)``)
    or the train split is structurally empty and the nano smoke fails
    inside ``SegmentsMixin.select`` with ``ValueError: Empty
    subselection`` before the first batch.
    """
    subject_2_trials = sorted({t for s, t in BT_NANO_SESSIONS if s == 2})
    assert len(subject_2_trials) >= 2, (
        f"Nano needs ≥2 trials of subject 2 for the default CrossSession "
        f"contract; got {subject_2_trials}"
    )
    assert 4 in subject_2_trials, "Nano must include the test trial (2, 4)"
    assert 0 in subject_2_trials, "Nano must include a train trial (2, 0)"
