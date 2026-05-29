"""BrainTreebank cohort manifest constants.

The session lists mirror Neuroprobe's published BrainTreebank splits. v14 parcel
coverage filters remain separate: `TIER1_WHITELIST` is populated only after the
BT support caches are baked.
"""

from __future__ import annotations

BT_NANO_SESSIONS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 0),
    (2, 4),
    (3, 1),
    (4, 0),
    (7, 1),
    (10, 1),
)
"""Tiny BrainTreebank subset used by Neuroprobe for CI-scale smoke tests.

Subject 2 carries both trial 0 and trial 4 so the default CrossSession
contract — test on ``(test_subject_id=2, test_trial_id=4)`` and train on
*other* trials of the same subject — has a non-empty train split. With
trial 4 alone the train mask is structurally empty and
:class:`neuralset.dataloader.SegmentsMixin.select` raises
``ValueError: Empty subselection`` before the first batch.
"""

BT_LITE_SESSIONS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 4),
    (3, 0),
    (3, 1),
    (4, 0),
    (4, 1),
    (7, 0),
    (7, 1),
    (10, 0),
    (10, 1),
)
"""Lite BrainTreebank sessions exposed by Neuroprobe."""

BT_FULL_SESSIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 0),
    (3, 1),
    (3, 2),
    (4, 0),
    (4, 1),
    (4, 2),
    (5, 0),
    (6, 0),
    (6, 1),
    (6, 4),
    (7, 0),
    (7, 1),
    (8, 0),
    (9, 0),
    (10, 0),
    (10, 1),
)
"""Full BrainTreebank subject/trial sessions used by Neuroprobe."""

TIER1_WHITELIST: tuple[tuple[int, int], ...] = ()
"""(subject_id, trial_id) pairs kept after Tier-1 parcel coverage filtering."""

TIER1_BNA_PARCEL_INDICES_1BASED: tuple[int, ...] = ()
"""BT-Tier-1 parcel index list (1-based BNA), populated by Stage-0 A0."""


V14_EXCLUDED_SUBJECT_IDS: tuple[int, ...] = (5,)
"""Subjects excluded from the v14 training cohort.

S5: BT Appendix A.5 reports a large right-frontal lesion preventing atlas
alignment; BT themselves exclude S5 from region analyses. The DK-first v14
contract (`project_v14_dk_first_pass_2026_05_13.md`) consumes `depth-wm.csv`
labels whose right-frontal entries are unreliable for this subject. S5 is not
in `NEUROPROBE_LITE_SUBJECT_TRIALS`, so exclusion costs zero leaderboard
coverage.
"""

V14_TRAIN_SUBJECT_IDS: tuple[int, ...] = tuple(
    s for s in range(1, 11) if s not in V14_EXCLUDED_SUBJECT_IDS
)
"""Subject ids used for v14 training (9 subjects: 1-10 minus S5)."""

V14_LEADERBOARD_SUBJECT_IDS: tuple[int, ...] = (1, 2, 3, 4, 7, 10)
"""Leaderboard-eval subset of `V14_TRAIN_SUBJECT_IDS`.

Derived from Neuroprobe's `NEUROPROBE_LITE_SUBJECT_TRIALS` (see
`.cache/neuroprobe_upstream/neuroprobe/config.py`). Subjects {6, 8, 9}
contribute pretraining coverage only.
"""

V14_TRAIN_SESSIONS: tuple[tuple[int, int], ...] = tuple(
    (subject_id, trial_id)
    for subject_id, trial_id in BT_FULL_SESSIONS
    if subject_id in V14_TRAIN_SUBJECT_IDS
)
"""BT full-session list restricted to the v14 training cohort."""
