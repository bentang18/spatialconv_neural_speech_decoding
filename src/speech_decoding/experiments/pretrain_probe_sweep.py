"""Sweep-once HP harness + fixed-HP per-checkpoint eval (Stage-2 orchestration).

Ben's cadence (project_pretrain_probe_suite_contract_2026_06_30): do NOT sweep every
checkpoint. When the 60k checkpoints arrive, sweep the attentive HPs ONCE on a
representative checkpoint, pick the best by mean held-out VAL AUROC (never test), fix
those HPs, then eval all ~5-10 checkpoints with the fixed HPs.

  - swept (attentive head only): ``weight_decay`` (≤3.0), ``parcel_dropout`` (primary
    structured knob), standard ``dropout`` (attn/mlp/residual share one rate).
  - kept on (unswept): SWAD, val-AUROC early-stop, learnable time tag (forced by the
    readout). The linear ridge has no HP sweep — its λ is selected per-cell on val.

Both stages consume the Stage-1 caches (one :class:`SessionTapCache` per cohort session,
keyed ``(subject_id, trial_id)``). CrossSubject cells pull their subject-2 anchor cache
from the same dict by the cell's ``train_*`` ids.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from speech_decoding.experiments.pretrain_probe_driver import (
    SessionTapCache,
    run_attentive_cell,
    run_attentive_cell_result,
    run_linear_cell,
)
from speech_decoding.experiments.pretrain_probe_readout import DEFAULT_LAM_GRID
from speech_decoding.experiments.pretrain_probe_suite import ProbeCell
from speech_decoding.experiments.v2_attentive_train import HeadTrainConfig

__all__ = [
    "HPCombo",
    "default_hp_grid",
    "cfg_for_hp",
    "SweepResult",
    "sweep_hp",
    "CellScore",
    "fixed_hp_eval",
    "TAPS",
]

TAPS: tuple[str, ...] = ("M2", "M3", "M4")


@dataclass(frozen=True)
class HPCombo:
    weight_decay: float
    parcel_dropout: float
    dropout: float            # shared attn/mlp/residual rate


def default_hp_grid() -> list[HPCombo]:
    """Ben-approved grid: wd {0.1,1.0,3.0} × parcel_dropout {0.0,0.2} × dropout {0.1,0.5}."""
    return [
        HPCombo(wd, pd, dr)
        for wd in (0.1, 1.0, 3.0)
        for pd in (0.0, 0.2)
        for dr in (0.1, 0.5)
    ]


def cfg_for_hp(base_cfg: HeadTrainConfig, hp: HPCombo) -> HeadTrainConfig:
    """Stamp one HP combo onto the base config (the shared dropout rate hits all three)."""
    return replace(
        base_cfg,
        weight_decay=hp.weight_decay,
        parcel_dropout=hp.parcel_dropout,
        attn_dropout=hp.dropout,
        mlp_dropout=hp.dropout,
        residual_dropout=hp.dropout,
    )


def _anchor_for(cell: ProbeCell, caches: dict) -> SessionTapCache | None:
    if cell.eval_mode != "CrossSubject":
        return None
    return caches.get((cell.train_subject_id, cell.train_trial_id))


@dataclass(frozen=True)
class SweepResult:
    best_hp: HPCombo
    mean_val_by_hp: dict[HPCombo, float]
    n_cells_scored: int


def sweep_hp(
    cells: list[ProbeCell],
    caches: dict,
    base_cfg: HeadTrainConfig,
    hp_grid: list[HPCombo],
    *,
    taps: tuple[str, ...] = TAPS,
    device: torch.device | None = None,
) -> SweepResult:
    """Pick the attentive HP with the best mean VAL AUROC over the sweep cells × taps.

    VAL only — the test halves never enter HP selection. Ties → the earliest grid entry
    (so the grid order encodes the tie-break preference, e.g. lower wd / less dropout)."""
    if not hp_grid:
        raise ValueError("hp_grid must be non-empty")
    mean_val: dict[HPCombo, float] = {}
    n_scored = 0
    for hp in hp_grid:
        cfg = cfg_for_hp(base_cfg, hp)
        vals: list[float] = []
        for cell in cells:
            tc = caches[(cell.test_subject_id, cell.test_trial_id)]
            anchor = _anchor_for(cell, caches)
            for tap in taps:
                r = run_attentive_cell_result(
                    cell, tap, cfg, test_cache=tc, anchor_cache=anchor, device=device
                )
                if np.isfinite(r.val):
                    vals.append(r.val)
        mean_val[hp] = float(np.mean(vals)) if vals else float("nan")
        n_scored = max(n_scored, len(vals))
    best_hp = max(
        hp_grid,
        key=lambda h: (mean_val[h] if np.isfinite(mean_val[h]) else -np.inf),
    )
    return SweepResult(best_hp=best_hp, mean_val_by_hp=mean_val, n_cells_scored=n_scored)


@dataclass(frozen=True)
class CellScore:
    """One cell × tap, both readouts' held-out TEST AUROC."""

    label: str
    eval_mode: str
    task: str
    tap: str
    linear: float
    attentive: float


def fixed_hp_eval(
    cells: list[ProbeCell],
    caches: dict,
    base_cfg: HeadTrainConfig,
    hp: HPCombo,
    *,
    taps: tuple[str, ...] = TAPS,
    lam_grid: tuple[float, ...] = DEFAULT_LAM_GRID,
    readouts: tuple[str, ...] = ("ridge", "attentive"),
    device: torch.device | None = None,
) -> list[CellScore]:
    """Eval every (cell, tap) at the FIXED swept HP — linear ridge + attentive test AUROC.
    Run per checkpoint; no per-checkpoint sweep.

    ``readouts`` selects which heads to actually fit: a ridge-only shard skips the (much
    costlier) attentive training and leaves ``attentive=nan``; an attentive-only shard
    skips the ridge solve and leaves ``linear=nan``. The fan-out uses this to put ridge
    shards on cheap CPU/1-GPU holes and attentive shards on the RAM-heavy 2-GPU holes."""
    if not set(readouts) <= {"ridge", "attentive"}:
        raise ValueError(f"readouts must be a subset of {{ridge,attentive}}, got {readouts}")
    cfg = cfg_for_hp(base_cfg, hp)
    scores: list[CellScore] = []
    for cell in cells:
        tc = caches[(cell.test_subject_id, cell.test_trial_id)]
        anchor = _anchor_for(cell, caches)
        for tap in taps:
            lin = (
                run_linear_cell(cell, tap, test_cache=tc, anchor_cache=anchor,
                                lam_grid=lam_grid)
                if "ridge" in readouts else float("nan")
            )
            att = (
                run_attentive_cell(cell, tap, cfg, test_cache=tc, anchor_cache=anchor,
                                   device=device)
                if "attentive" in readouts else float("nan")
            )
            scores.append(
                CellScore(cell.label(), cell.eval_mode, cell.task, tap, lin, att)
            )
    return scores
