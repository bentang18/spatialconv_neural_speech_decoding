"""Collector + firewall self-test for the pretrain probe suite.

Aggregates the per-cell :class:`CellScore` rows (from :func:`fixed_hp_eval`) into the
checkpoint-selection signal: mean held-out test AUROC per (eval_mode, tap, readout),
per task, and the best cross-subject (tap, readout) vs the raw_tok ridge floor
(``delta_volume`` 0.666). The best checkpoint is the one whose cross-subject number is
highest; it is then cooled and submitted via the separate Neuroprobe leaderboard path.

The firewall self-test is the last line of the WALL OF FIRE: before any results are
trusted, assert every scored cell drew train/val/test/anchor ONLY from the firewall-legal
cohort — no ``BT_LITE_SESSIONS`` clip ever touched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from speech_decoding.experiments.pretrain_probe_suite import (
    PRETRAIN_UNIVERSE,
    ProbeCell,
    assert_cells_firewall_legal,
)
from speech_decoding.experiments.pretrain_probe_sweep import CellScore
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS

__all__ = [
    "RIDGE_FLOOR_DELTA_VOLUME",
    "ScoreSummary",
    "aggregate_scores",
    "firewall_self_test",
]

# The raw_tok cross-subject ridge floor the pooled encoder taps must beat
# (project_probe_bench_keepS_ridge_lock_2026_06_28).
RIDGE_FLOOR_DELTA_VOLUME: float = 0.666


def _mean(vals: list[float]) -> float:
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


@dataclass(frozen=True)
class ScoreSummary:
    """Aggregated checkpoint-selection signal."""

    mean_by_mode_tap: dict[tuple[str, str, str], float]    # (mode, tap, readout) -> mean AUROC
    mean_by_task: dict[tuple[str, str, str], float]        # (mode, task, readout) -> mean AUROC
    best_cross_subject: tuple[str, str, float]             # (tap, readout, mean AUROC)
    n_cells: int


def aggregate_scores(scores: list[CellScore]) -> ScoreSummary:
    """Roll cell scores up into the selection signal. Both readouts ('linear',
    'attentive') are aggregated side by side."""
    by_mode_tap: dict[tuple[str, str, str], list[float]] = {}
    by_task: dict[tuple[str, str, str], list[float]] = {}
    for s in scores:
        for readout, val in (("linear", s.linear), ("attentive", s.attentive)):
            by_mode_tap.setdefault((s.eval_mode, s.tap, readout), []).append(val)
            by_task.setdefault((s.eval_mode, s.task, readout), []).append(val)

    mean_by_mode_tap = {k: _mean(v) for k, v in by_mode_tap.items()}
    mean_by_task = {k: _mean(v) for k, v in by_task.items()}

    cs = {
        (tap, readout): m
        for (mode, tap, readout), m in mean_by_mode_tap.items()
        if mode == "CrossSubject" and np.isfinite(m)
    }
    if cs:
        (best_tap, best_readout), best_m = max(cs.items(), key=lambda kv: kv[1])
        best = (best_tap, best_readout, best_m)
    else:
        best = ("", "", float("nan"))
    return ScoreSummary(
        mean_by_mode_tap=mean_by_mode_tap,
        mean_by_task=mean_by_task,
        best_cross_subject=best,
        n_cells=len({s.label for s in scores}),
    )


def firewall_self_test(cells: list[ProbeCell]) -> None:
    """🔥 Belt-and-suspenders WALL OF FIRE: the cohort is disjoint from the eval set AND
    every cell is firewall-legal. Raises ``AssertionError`` on any breach."""
    lite = set(BT_LITE_SESSIONS)
    leak = set(PRETRAIN_UNIVERSE) & lite
    if leak:
        raise AssertionError(
            f"WALL OF FIRE breach: cohort intersects the eval set at {sorted(leak)}."
        )
    assert_cells_firewall_legal(cells)
