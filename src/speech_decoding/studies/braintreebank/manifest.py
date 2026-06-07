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
"""BT full-session list restricted to the v14 training cohort.

WARNING: this list is NOT leaderboard-legal for SSL pretraining — it still
contains the 12 Neuroprobe eval sessions (== BT_LITE_SESSIONS). For a
leaderboard-legal foundation-model pretraining corpus use
``V14_PRETRAIN_SESSIONS`` below, which is the Neuroprobe pretraining-allowed
set restricted to the cohort.
"""


# --- Neuroprobe leaderboard pretraining contract -------------------------
# Source of truth: upstream `.cache/neuroprobe_upstream/SUBMIT.md` section
# "Pretraining guidelines". The 12 off-limits eval sessions are EXACTLY
# BT_LITE_SESSIONS (== neuroprobe.config.NEUROPROBE_LITE_SUBJECT_TRIALS); a
# leaderboard-legal model "was not pretrained on any data that intersects with
# any data of Neuroprobe", so SSL/pretraining may use ONLY the disjoint
# allowed set below. The dispatch's eval-split data path (BTWordEvents +
# Wang2024Treebank mode) does NOT yet read these — the SSL corpus is currently
# coupled to the eval split (see docs/neuroprobe/v14_blockers.md). These
# constants codify the legal contract for the decoupling fix + a leakage guard.

BT_PRETRAIN_ALLOWED_SESSIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (2, 1), (2, 2), (2, 3), (2, 5), (2, 6),
    (3, 2),
    (4, 2),
    (5, 0),
    (6, 0), (6, 1), (6, 4),
    (8, 0),
    (9, 0),
)
"""Neuroprobe pretraining-allowed STANDARD sessions (SUBMIT.md). Exactly
``BT_FULL_SESSIONS - BT_LITE_SESSIONS``: 14 sessions over subjects
{1,2,3,4,5,6,8,9}. Disjoint from BT_LITE_SESSIONS (the eval set). Subjects 7
and 10 have NO standard allowed session — both their trials are eval trials."""

BT_PRETRAIN_PARTIAL_SESSIONS: tuple[tuple[int, int], ...] = (
    (7, 100), (7, 101), (7, 102),
    (10, 100), (10, 101),
)
"""Special non-eval-intersecting "partial" pretraining sessions for subjects 7
and 10 (SUBMIT.md): ~20 min each, carved from the times of their two trials
that do NOT intersect the Neuroprobe eval indices. NOT bundled with the corpus
— distributed via a separate Google-Drive download and ABSENT on DCC as of
2026-06-06. Without these, subjects 7 and 10 have zero legal pretraining data."""

V14_PRETRAIN_SESSIONS: tuple[tuple[int, int], ...] = tuple(
    (s, t) for (s, t) in BT_PRETRAIN_ALLOWED_SESSIONS
    if s not in V14_EXCLUDED_SUBJECT_IDS
)
"""Leaderboard-legal SSL pretraining sessions for the v14 cohort (S5 dropped).
13 standard sessions over subjects {1,2,3,4,6,8,9}. Subjects 7 and 10
contribute zero sessions here (their only legal data is
``BT_PRETRAIN_PARTIAL_SESSIONS``, not yet on DCC).

DCC data availability (verified 2026-06-06, #26 closed):
  - Raw h5: all 13 sessions present (incl. 6/8/9).
  - Transcripts (word-event clip boundaries → P1/P2): all 13 present.
  → P1 (front-end) + P2 (parcel) run on the full 13-session legal corpus.
  - Whisper teacher cache (P3 distill target): 12/13 movies present. The
    sole gap is session (8,0) "Sesame Street Episode 3990" — no teacher .pt.
  → P3 runs on 12 sessions / subjects {1,2,3,4,6,9}; subject 8 drops from P3
    ONLY until its teacher target is backfilled (port 16 kHz audio + Whisper).
    P4 readout is unaffected (it evals on BT_LITE_SESSIONS, not this corpus)."""


NO_TEACHER_CACHE_SESSIONS: tuple[tuple[int, int], ...] = ((8, 0),)
"""Pretrain-legal sessions with NO Whisper teacher cache on DCC (data gap, not a
contract change). (8,0) "Sesame Street Episode 3990" is subject 8's only session
and has no teacher .pt as of 2026-06-06. These are excluded from the P3 distill
corpus (P3 needs a teacher target per clip); P1/P2 — which need no teacher — keep
the full ``V14_PRETRAIN_SESSIONS``. Backfilling (8,0)'s teacher restores S8 to P3.
"""

V14_P3_DISTILL_SESSIONS: tuple[tuple[int, int], ...] = tuple(
    (s, t) for (s, t) in V14_PRETRAIN_SESSIONS if (s, t) not in NO_TEACHER_CACHE_SESSIONS
)
"""Leaderboard-legal P3 (Whisper-distill) corpus: ``V14_PRETRAIN_SESSIONS`` minus
the no-teacher sessions. 12 sessions over subjects {1,2,3,4,6,9} (subject 8 drops
because its only session (8,0) has no teacher cache). Without an explicit P3
corpus the distill phase routes to the full 13-session pretrain set and crashes
lazily on the first (8,0) clip (whisper_target FileNotFoundError)."""
