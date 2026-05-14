from __future__ import annotations

from speech_decoding.studies.braintreebank.manifest import (
    BT_FULL_SESSIONS,
    BT_LITE_SESSIONS,
    V14_EXCLUDED_SUBJECT_IDS,
    V14_LEADERBOARD_SUBJECT_IDS,
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
