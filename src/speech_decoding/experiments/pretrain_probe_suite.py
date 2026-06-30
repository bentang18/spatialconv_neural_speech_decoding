"""Firewall-legal cohort + cell enumeration for the pretrain probe suite.

The per-checkpoint full-suite eval (project_pretrain_probe_suite_contract_2026_06_30)
is a leaderboard-FAITHFUL proxy run ENTIRELY on pretraining data: it selects the
best of ~5-10 checkpoints, which is then cooled and submitted via the real
Neuroprobe leaderboard eval (a separate path).

🔥 WALL OF FIRE (Ben, IRONCLAD): every cell here — train, val, test, and the
CrossSubject train anchor — is drawn ONLY from :data:`PRETRAIN_UNIVERSE`
(``BT_PRETRAIN_ALLOWED_SESSIONS`` minus the v14-excluded subjects). NO
``BT_LITE_SESSIONS`` clip is ever touched. This is asserted at import and again
per-cell by :func:`assert_cells_firewall_legal`.

Modes (Ben 2026-06-30): WithinSession + CrossSubject ONLY (CrossSession skipped to
save compute). CrossSubject trains on subject 2 — the SAME person the leaderboard
DS-DM split trains on — but on a NON-LITE trial of subject 2, never the (2,4)/(2,0)
eval cells.
"""

from __future__ import annotations

from dataclasses import dataclass

from speech_decoding.studies.braintreebank.labels import NEUROPROBE_TASKS
from speech_decoding.studies.braintreebank.manifest import (
    BT_LITE_SESSIONS,
    BT_PRETRAIN_ALLOWED_SESSIONS,
    V14_EXCLUDED_SUBJECT_IDS,
)

__all__ = [
    "NEUROPROBE_TASKS",
    "PRETRAIN_UNIVERSE",
    "CS_TRAIN_SUBJECT",
    "CS_TRAIN_ANCHOR_CANDIDATES",
    "DEFAULT_CS_TRAIN_ANCHOR",
    "WS_N_FOLDS",
    "ProbeCell",
    "enumerate_cells",
    "assert_cells_firewall_legal",
]

# Firewall-legal universe: pretrain-allowed sessions, minus the v14-excluded
# subjects (S5: right-frontal lesion, no atlas alignment). Subjects {1,2,3,4,6,8,9}.
PRETRAIN_UNIVERSE: tuple[tuple[int, int], ...] = tuple(
    (subject_id, trial_id)
    for (subject_id, trial_id) in BT_PRETRAIN_ALLOWED_SESSIONS
    if subject_id not in V14_EXCLUDED_SUBJECT_IDS
)

# 🔥 Hard import-time WALL OF FIRE guard: the cohort must never intersect the eval set.
_LEAK = set(PRETRAIN_UNIVERSE) & set(BT_LITE_SESSIONS)
if _LEAK:
    raise AssertionError(
        f"WALL OF FIRE breach: pretrain probe cohort intersects the Neuroprobe eval "
        f"set (BT_LITE_SESSIONS) at {sorted(_LEAK)}. The probe must NEVER touch eval data."
    )

# CrossSubject trains on subject 2 (same person as the leaderboard DS-DM train
# subject) but on a FIREWALL-LEGAL trial — never (2,4)/(2,0).
CS_TRAIN_SUBJECT: int = 2
CS_TRAIN_ANCHOR_CANDIDATES: tuple[tuple[int, int], ...] = tuple(
    (subject_id, trial_id)
    for (subject_id, trial_id) in PRETRAIN_UNIVERSE
    if subject_id == CS_TRAIN_SUBJECT
)
if not CS_TRAIN_ANCHOR_CANDIDATES:
    raise AssertionError(
        f"no firewall-legal CrossSubject train anchor for subject {CS_TRAIN_SUBJECT}: "
        "every subject-2 session is a leaderboard eval cell."
    )
# Deterministic default: the first non-lite subject-2 trial ((2,1)).
DEFAULT_CS_TRAIN_ANCHOR: tuple[int, int] = CS_TRAIN_ANCHOR_CANDIDATES[0]

# WithinSession uses KFold(2, shuffle=False) — mirrors upstream Lite WS; score both folds.
WS_N_FOLDS: int = 2


@dataclass(frozen=True)
class ProbeCell:
    """One eval cell: a (mode, task, test session[, fold][, CS train anchor]) tuple."""

    eval_mode: str            # "WithinSession" | "CrossSubject"
    task: str
    test_subject_id: int
    test_trial_id: int
    fold_index: int = 0       # WS KFold fold in [0, WS_N_FOLDS); always 0 for CS
    train_subject_id: int | None = None   # CS train anchor subject
    train_trial_id: int | None = None     # CS train anchor trial

    def label(self) -> str:
        base = f"{self.eval_mode}_{self.task}_S{self.test_subject_id}T{self.test_trial_id}"
        if self.eval_mode == "WithinSession":
            base += f"_f{self.fold_index}"
        return base


def _within_session_cells(tasks: tuple[str, ...]) -> list[ProbeCell]:
    cells: list[ProbeCell] = []
    for (subject_id, trial_id) in PRETRAIN_UNIVERSE:
        for task in tasks:
            for fold in range(WS_N_FOLDS):
                cells.append(
                    ProbeCell("WithinSession", task, subject_id, trial_id, fold)
                )
    return cells


def _cross_subject_cells(
    tasks: tuple[str, ...], anchor: tuple[int, int]
) -> list[ProbeCell]:
    anchor_subject, anchor_trial = anchor
    if anchor not in PRETRAIN_UNIVERSE:
        raise ValueError(
            f"CrossSubject train anchor {anchor} is not in the firewall-legal "
            f"PRETRAIN_UNIVERSE; legal subject-{CS_TRAIN_SUBJECT} anchors: "
            f"{CS_TRAIN_ANCHOR_CANDIDATES}."
        )
    cells: list[ProbeCell] = []
    for (subject_id, trial_id) in PRETRAIN_UNIVERSE:
        if subject_id == anchor_subject:
            # Drop the entire train SUBJECT from test (upstream CrossSubject contract).
            continue
        for task in tasks:
            cells.append(
                ProbeCell(
                    "CrossSubject", task, subject_id, trial_id, 0,
                    anchor_subject, anchor_trial,
                )
            )
    return cells


def enumerate_cells(
    modes: tuple[str, ...],
    tasks: tuple[str, ...] = NEUROPROBE_TASKS,
    *,
    cs_train_anchor: tuple[int, int] = DEFAULT_CS_TRAIN_ANCHOR,
) -> list[ProbeCell]:
    """All probe cells for the requested modes. Modes ⊆ {WithinSession, CrossSubject}."""
    cells: list[ProbeCell] = []
    for mode in modes:
        if mode == "WithinSession":
            cells.extend(_within_session_cells(tasks))
        elif mode == "CrossSubject":
            cells.extend(_cross_subject_cells(tasks, cs_train_anchor))
        else:
            raise ValueError(
                f"unsupported eval mode {mode!r}; the pretrain probe suite runs only "
                "WithinSession + CrossSubject (CrossSession is skipped)."
            )
    assert_cells_firewall_legal(cells)
    return cells


def assert_cells_firewall_legal(cells: list[ProbeCell]) -> None:
    """🔥 Fail loud if ANY cell's train/test session is a Neuroprobe eval cell."""
    lite = set(BT_LITE_SESSIONS)
    for cell in cells:
        test = (cell.test_subject_id, cell.test_trial_id)
        if test in lite:
            raise AssertionError(
                f"WALL OF FIRE breach: cell {cell.label()} tests on eval cell {test}."
            )
        if cell.train_subject_id is not None:
            train = (cell.train_subject_id, cell.train_trial_id)
            if train in lite:
                raise AssertionError(
                    f"WALL OF FIRE breach: cell {cell.label()} trains on eval cell {train}."
                )
