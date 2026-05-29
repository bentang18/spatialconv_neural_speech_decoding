"""Submission-gate harness (T3.3).

Aggregates per-task Phase-4 readout results into the dual-prong gate:

  - Prong A: CrossSession **multiclass** ≥ ``csession_multiclass_threshold``
  - Prong B: CrossSubject **binary AUROC** ≥ ``csubject_binary_threshold``

Plus the bookkeeping clauses: ≥ ``min_passing_tasks`` tasks pass *both*
prongs, and the model param count is ≤ ``param_count_limit``.

Per ``docs/neuroprobe/v14_blockers.md``:

  - EV01 — baseline CI (what is the preprint max + 0.05?). Caller supplies
    the resulting threshold floats.
  - EV07 — statistical testing protocol (bootstrap CI, seed count). Caller
    supplies a ``ci_protocol`` callable that returns ``(low, high)``.
  - EV08 — which 4 tasks to report. Caller supplies ``task_ids``.
  - EV09 — dual-prong independence check. Caller supplies a boolean
    ``independence_check_passed`` flag plus a free-form note.

CR06 (per-task probe-init seed isolation): the gate harness records the
seed used for *each* task's Phase-4 probe so a single rogue seed can be
audited / rerun without touching the others. The ``per_task_seed_fn``
callable is required.

Everything else (per-task score, AUROC, etc.) is data passed in from the
upstream Phase-4 dispatch loop. This module is pure decision logic.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal


CiInterval = tuple[float, float]


@dataclasses.dataclass(frozen=True, slots=True)
class TaskProngResult:
    """One (task, prong) cell: point estimate, CI, and per-task seed (CR06)."""

    task_id: str
    prong: Literal["csession_multiclass", "csubject_binary"]
    score: float
    ci_low: float
    ci_high: float
    threshold: float
    seed: int

    @property
    def passes(self) -> bool:
        return self.score >= self.threshold


@dataclasses.dataclass(frozen=True, slots=True)
class TaskRow:
    """All prong cells for a single task."""

    task_id: str
    csession_multiclass: TaskProngResult
    csubject_binary: TaskProngResult

    @property
    def both_prongs_pass(self) -> bool:
        return self.csession_multiclass.passes and self.csubject_binary.passes


@dataclasses.dataclass(frozen=True, slots=True)
class GateDecision:
    """Final dual-prong gate verdict + provenance for the submission artifact."""

    passed: bool
    rows: tuple[TaskRow, ...]
    n_passing_tasks: int
    min_passing_tasks: int
    param_count: int
    param_count_limit: int
    csession_multiclass_threshold: float
    csubject_binary_threshold: float
    independence_check_passed: bool
    independence_note: str
    failure_reasons: tuple[str, ...]

    def to_json_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["rows"] = [dataclasses.asdict(r) for r in self.rows]
        return d

    def write_artifact(self, path: Path) -> None:
        """Persist as JSON. Caller decides where (typically the report dir)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2))


def _build_prong(
    *, task_id: str,
    prong: Literal["csession_multiclass", "csubject_binary"],
    score: float, threshold: float, seed: int,
    ci_protocol: Callable[[str, str], CiInterval],
) -> TaskProngResult:
    ci_low, ci_high = ci_protocol(task_id, prong)
    if ci_low > ci_high:
        raise ValueError(
            f"ci_protocol returned inverted CI for ({task_id}, {prong}): "
            f"low={ci_low} > high={ci_high}"
        )
    return TaskProngResult(
        task_id=task_id, prong=prong, score=score, threshold=threshold,
        ci_low=ci_low, ci_high=ci_high, seed=seed,
    )


def apply_submission_gate(
    *,
    task_ids: tuple[str, ...],
    csession_multiclass_scores: Mapping[str, float],
    csubject_binary_scores: Mapping[str, float],
    csession_multiclass_threshold: float,
    csubject_binary_threshold: float,
    min_passing_tasks: int,
    param_count: int,
    param_count_limit: int,
    ci_protocol: Callable[[str, str], CiInterval],
    per_task_seed_fn: Callable[[str, str], int],
    independence_check_passed: bool,
    independence_note: str,
) -> GateDecision:
    """Apply the dual-prong submission gate to per-task Phase-4 results.

    Inputs ``csession_multiclass_scores`` / ``csubject_binary_scores`` are
    keyed by ``task_id``; missing keys raise. All thresholds, task IDs, and
    the param-count cap are required (no defaults — see EV01/EV08).

    Returns a :class:`GateDecision` with a boolean ``passed`` plus the full
    per-task breakdown and a list of any failure reasons (helpful for the
    rerun loop)."""
    if not task_ids:
        raise ValueError("task_ids must be non-empty (EV08)")
    if min_passing_tasks <= 0:
        raise ValueError("min_passing_tasks must be > 0")
    if min_passing_tasks > len(task_ids):
        raise ValueError(
            f"min_passing_tasks={min_passing_tasks} exceeds task_ids count={len(task_ids)}"
        )

    rows: list[TaskRow] = []
    for task_id in task_ids:
        if task_id not in csession_multiclass_scores:
            raise KeyError(f"csession_multiclass_scores missing task {task_id!r}")
        if task_id not in csubject_binary_scores:
            raise KeyError(f"csubject_binary_scores missing task {task_id!r}")
        csess = _build_prong(
            task_id=task_id, prong="csession_multiclass",
            score=csession_multiclass_scores[task_id],
            threshold=csession_multiclass_threshold,
            seed=per_task_seed_fn(task_id, "csession_multiclass"),
            ci_protocol=ci_protocol,
        )
        csubj = _build_prong(
            task_id=task_id, prong="csubject_binary",
            score=csubject_binary_scores[task_id],
            threshold=csubject_binary_threshold,
            seed=per_task_seed_fn(task_id, "csubject_binary"),
            ci_protocol=ci_protocol,
        )
        rows.append(TaskRow(task_id=task_id, csession_multiclass=csess, csubject_binary=csubj))

    n_passing = sum(1 for r in rows if r.both_prongs_pass)
    failure_reasons: list[str] = []
    if n_passing < min_passing_tasks:
        failure_reasons.append(
            f"only {n_passing} of {len(rows)} tasks passed both prongs "
            f"(need ≥ {min_passing_tasks})"
        )
    if param_count > param_count_limit:
        failure_reasons.append(
            f"param_count={param_count} exceeds limit={param_count_limit}"
        )
    if not independence_check_passed:
        failure_reasons.append(f"independence check failed: {independence_note}")

    return GateDecision(
        passed=not failure_reasons,
        rows=tuple(rows),
        n_passing_tasks=n_passing,
        min_passing_tasks=min_passing_tasks,
        param_count=param_count,
        param_count_limit=param_count_limit,
        csession_multiclass_threshold=csession_multiclass_threshold,
        csubject_binary_threshold=csubject_binary_threshold,
        independence_check_passed=independence_check_passed,
        independence_note=independence_note,
        failure_reasons=tuple(failure_reasons),
    )
