from __future__ import annotations

from speech_decoding.studies.braintreebank.manifest import (
    BT_FULL_SESSIONS,
    BT_LITE_SESSIONS,
    BT_NANO_SESSIONS,
    BT_PRETRAIN_ALLOWED_SESSIONS,
    BT_PRETRAIN_PARTIAL_SESSIONS,
    V14_EXCLUDED_SUBJECT_IDS,
    V14_LEADERBOARD_SUBJECT_IDS,
    V14_PRETRAIN_SESSIONS,
    V14_TRAIN_SESSIONS,
    V14_TRAIN_SUBJECT_IDS,
)


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
