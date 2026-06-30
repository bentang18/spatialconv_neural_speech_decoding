"""Checkpoint-selection orchestrator for the pretrain probe suite (Stage-2 top level).

The vision (project_pretrain_probe_suite_contract_2026_06_30): per checkpoint, Stage 1
forwards the encoder once and caches the tap grids; then this module SELECTS the best of
~5-10 checkpoints on pretrain data, so that one can be cooled and submitted via the real
Neuroprobe leaderboard path. The cadence is sweep-ONCE: pick the attentive HP on a
representative checkpoint, then eval EVERY checkpoint at that fixed HP and rank by the
cross-subject number.

This orchestrator is pure CPU and fully testable — it consumes already-built caches
(``caches_by_ckpt``) and runs cells → sweep → fixed-HP eval → aggregate → firewall. The
only gated pieces are upstream: building each checkpoint's caches (Stage-1 GPU forward,
:func:`build_caches`) and the real ``forward_fn`` / ``word_events`` reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from speech_decoding.experiments.pretrain_probe_collect import (
    ScoreSummary,
    aggregate_scores,
    firewall_self_test,
)
from speech_decoding.experiments.pretrain_probe_driver import SessionTapCache
from speech_decoding.experiments.pretrain_probe_suite import (
    DEFAULT_CS_TRAIN_ANCHOR,
    NEUROPROBE_TASKS,
    ProbeCell,
    enumerate_cells,
)
from speech_decoding.experiments.pretrain_probe_sweep import (
    TAPS,
    CellScore,
    HPCombo,
    SweepResult,
    default_hp_grid,
    fixed_hp_eval,
    sweep_hp,
)
from speech_decoding.experiments.v2_attentive_train import HeadTrainConfig

__all__ = [
    "CheckpointSelection",
    "select_best_checkpoint",
]

# Caches for a set of checkpoints: ckpt tag -> {(subject,trial) -> SessionTapCache}.
CachesByCkpt = dict[str, dict[tuple[int, int], SessionTapCache]]


@dataclass(frozen=True)
class CheckpointSelection:
    """The suite's output: the swept HP, every checkpoint's scores + summary, and the
    winner (highest cross-subject AUROC) to cool down and submit."""

    cells: list[ProbeCell]
    sweep: SweepResult
    scores_by_ckpt: dict[str, list[CellScore]]
    summary_by_ckpt: dict[str, ScoreSummary]
    best_ckpt: str
    best_cross_subject: float


def select_best_checkpoint(
    caches_by_ckpt: CachesByCkpt,
    base_cfg: HeadTrainConfig,
    *,
    representative_ckpt: str,
    modes: tuple[str, ...] = ("WithinSession", "CrossSubject"),
    tasks: tuple[str, ...] = NEUROPROBE_TASKS,
    cs_train_anchor: tuple[int, int] = DEFAULT_CS_TRAIN_ANCHOR,
    hp_grid: list[HPCombo] | None = None,
    sweep_cells: list[ProbeCell] | None = None,
    taps: tuple[str, ...] = TAPS,
    device: torch.device | None = None,
) -> CheckpointSelection:
    """Run the full suite and pick the best checkpoint by cross-subject AUROC.

    1. enumerate firewall-legal cells (+ self-test); 2. sweep the attentive HP ONCE on
    ``representative_ckpt`` by mean val AUROC; 3. eval every checkpoint at the fixed HP;
    4. aggregate; 5. rank by ``best_cross_subject``. ``sweep_cells`` defaults to all cells
    (pass a representative subset to cut sweep cost)."""
    if representative_ckpt not in caches_by_ckpt:
        raise ValueError(
            f"representative_ckpt {representative_ckpt!r} not in caches_by_ckpt "
            f"{sorted(caches_by_ckpt)}"
        )
    cells = enumerate_cells(modes, tasks, cs_train_anchor=cs_train_anchor)
    firewall_self_test(cells)

    grid = hp_grid if hp_grid is not None else default_hp_grid()
    sweep = sweep_hp(
        sweep_cells if sweep_cells is not None else cells,
        caches_by_ckpt[representative_ckpt],
        base_cfg, grid, taps=taps, device=device,
    )

    scores_by_ckpt: dict[str, list[CellScore]] = {}
    summary_by_ckpt: dict[str, ScoreSummary] = {}
    for tag, caches in caches_by_ckpt.items():
        scores = fixed_hp_eval(
            cells, caches, base_cfg, sweep.best_hp, taps=taps, device=device
        )
        scores_by_ckpt[tag] = scores
        summary_by_ckpt[tag] = aggregate_scores(scores)

    ranked = {
        tag: s.best_cross_subject[2]
        for tag, s in summary_by_ckpt.items()
        if np.isfinite(s.best_cross_subject[2])
    }
    if ranked:
        best_ckpt = max(ranked, key=lambda t: ranked[t])
        best_cs = ranked[best_ckpt]
    else:
        best_ckpt, best_cs = representative_ckpt, float("nan")

    return CheckpointSelection(
        cells=cells,
        sweep=sweep,
        scores_by_ckpt=scores_by_ckpt,
        summary_by_ckpt=summary_by_ckpt,
        best_ckpt=best_ckpt,
        best_cross_subject=best_cs,
    )
