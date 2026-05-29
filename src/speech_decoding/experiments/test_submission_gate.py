"""T3.3 + CR06 tests: dual-prong gate decision + per-task seed isolation."""

from __future__ import annotations

import json

import pytest

from speech_decoding.experiments.submission_gate import (
    GateDecision,
    apply_submission_gate,
)


def _const_ci(low: float = -0.05, high: float = 0.05):
    def _f(task_id: str, prong: str) -> tuple[float, float]:
        return (low, high)
    return _f


def _task_seed_table(table: dict[tuple[str, str], int]):
    """CR06 helper: per-(task, prong) seed lookup."""
    def _f(task_id: str, prong: str) -> int:
        return table[(task_id, prong)]
    return _f


def test_all_tasks_pass_both_prongs_gate_passes() -> None:
    tasks = ("speech", "onset", "delta_volume", "word_index")
    csess = {t: 0.70 for t in tasks}  # > 0.667
    csubj = {t: 0.65 for t in tasks}  # > 0.628
    seed_table = {(t, p): hash((t, p)) & 0xFFFF for t in tasks for p in ("csession_multiclass", "csubject_binary")}

    decision = apply_submission_gate(
        task_ids=tasks,
        csession_multiclass_scores=csess,
        csubject_binary_scores=csubj,
        csession_multiclass_threshold=0.667,
        csubject_binary_threshold=0.628,
        min_passing_tasks=4,
        param_count=13_000_000,
        param_count_limit=30_000_000,
        ci_protocol=_const_ci(),
        per_task_seed_fn=_task_seed_table(seed_table),
        independence_check_passed=True,
        independence_note="EV09: prong scores computed on disjoint label sets.",
    )
    assert isinstance(decision, GateDecision)
    assert decision.passed is True
    assert decision.n_passing_tasks == 4
    assert decision.failure_reasons == ()


def test_per_task_seed_recorded_for_each_prong_cell() -> None:
    """CR06: each (task, prong) records its own seed so a single rogue
    cell can be rerun without disturbing the rest."""
    tasks = ("speech", "onset")
    seed_table = {
        ("speech", "csession_multiclass"): 11,
        ("speech", "csubject_binary"): 12,
        ("onset", "csession_multiclass"): 21,
        ("onset", "csubject_binary"): 22,
    }
    decision = apply_submission_gate(
        task_ids=tasks,
        csession_multiclass_scores={"speech": 0.7, "onset": 0.7},
        csubject_binary_scores={"speech": 0.65, "onset": 0.65},
        csession_multiclass_threshold=0.5,
        csubject_binary_threshold=0.5,
        min_passing_tasks=2,
        param_count=10_000_000,
        param_count_limit=30_000_000,
        ci_protocol=_const_ci(),
        per_task_seed_fn=_task_seed_table(seed_table),
        independence_check_passed=True,
        independence_note="",
    )
    seeds_seen = {(r.csession_multiclass.seed, r.csubject_binary.seed) for r in decision.rows}
    assert (11, 12) in seeds_seen
    assert (21, 22) in seeds_seen


def test_param_count_over_limit_fails_gate() -> None:
    tasks = ("speech",)
    decision = apply_submission_gate(
        task_ids=tasks,
        csession_multiclass_scores={"speech": 0.8},
        csubject_binary_scores={"speech": 0.8},
        csession_multiclass_threshold=0.667,
        csubject_binary_threshold=0.628,
        min_passing_tasks=1,
        param_count=40_000_000,
        param_count_limit=30_000_000,
        ci_protocol=_const_ci(),
        per_task_seed_fn=_task_seed_table({("speech", "csession_multiclass"): 1, ("speech", "csubject_binary"): 2}),
        independence_check_passed=True,
        independence_note="",
    )
    assert decision.passed is False
    assert any("param_count" in r for r in decision.failure_reasons)


def test_independence_failure_blocks_gate() -> None:
    tasks = ("speech",)
    decision = apply_submission_gate(
        task_ids=tasks,
        csession_multiclass_scores={"speech": 0.8},
        csubject_binary_scores={"speech": 0.8},
        csession_multiclass_threshold=0.667,
        csubject_binary_threshold=0.628,
        min_passing_tasks=1,
        param_count=10_000_000,
        param_count_limit=30_000_000,
        ci_protocol=_const_ci(),
        per_task_seed_fn=_task_seed_table({("speech", "csession_multiclass"): 1, ("speech", "csubject_binary"): 2}),
        independence_check_passed=False,
        independence_note="EV09: prong scores share trials — leakage suspected.",
    )
    assert decision.passed is False
    assert any("independence" in r for r in decision.failure_reasons)


def test_insufficient_passing_tasks_blocks_gate() -> None:
    tasks = ("speech", "onset", "delta_volume", "word_index")
    csess = {"speech": 0.7, "onset": 0.4, "delta_volume": 0.4, "word_index": 0.4}
    csubj = {t: 0.7 for t in tasks}
    seed_table = {(t, p): 0 for t in tasks for p in ("csession_multiclass", "csubject_binary")}
    decision = apply_submission_gate(
        task_ids=tasks,
        csession_multiclass_scores=csess,
        csubject_binary_scores=csubj,
        csession_multiclass_threshold=0.667,
        csubject_binary_threshold=0.628,
        min_passing_tasks=4,
        param_count=10_000_000,
        param_count_limit=30_000_000,
        ci_protocol=_const_ci(),
        per_task_seed_fn=_task_seed_table(seed_table),
        independence_check_passed=True,
        independence_note="",
    )
    assert decision.passed is False
    assert decision.n_passing_tasks == 1
    assert any("passed both prongs" in r for r in decision.failure_reasons)


def test_missing_task_score_raises() -> None:
    with pytest.raises(KeyError, match="csubject"):
        apply_submission_gate(
            task_ids=("speech", "onset"),
            csession_multiclass_scores={"speech": 0.7, "onset": 0.7},
            csubject_binary_scores={"speech": 0.7},  # onset missing
            csession_multiclass_threshold=0.5,
            csubject_binary_threshold=0.5,
            min_passing_tasks=2,
            param_count=10_000_000,
            param_count_limit=30_000_000,
            ci_protocol=_const_ci(),
            per_task_seed_fn=lambda t, p: 0,
            independence_check_passed=True,
            independence_note="",
        )


def test_empty_task_ids_rejected() -> None:
    with pytest.raises(ValueError, match="task_ids"):
        apply_submission_gate(
            task_ids=(),
            csession_multiclass_scores={},
            csubject_binary_scores={},
            csession_multiclass_threshold=0.5,
            csubject_binary_threshold=0.5,
            min_passing_tasks=1,
            param_count=10_000_000,
            param_count_limit=30_000_000,
            ci_protocol=_const_ci(),
            per_task_seed_fn=lambda t, p: 0,
            independence_check_passed=True,
            independence_note="",
        )


def test_min_passing_tasks_exceeds_count_rejected() -> None:
    with pytest.raises(ValueError, match="min_passing_tasks"):
        apply_submission_gate(
            task_ids=("speech",),
            csession_multiclass_scores={"speech": 0.9},
            csubject_binary_scores={"speech": 0.9},
            csession_multiclass_threshold=0.5,
            csubject_binary_threshold=0.5,
            min_passing_tasks=5,
            param_count=10_000_000,
            param_count_limit=30_000_000,
            ci_protocol=_const_ci(),
            per_task_seed_fn=lambda t, p: 0,
            independence_check_passed=True,
            independence_note="",
        )


def test_inverted_ci_rejected() -> None:
    def bad_ci(task_id: str, prong: str) -> tuple[float, float]:
        return (0.1, -0.1)  # inverted
    with pytest.raises(ValueError, match="inverted CI"):
        apply_submission_gate(
            task_ids=("speech",),
            csession_multiclass_scores={"speech": 0.9},
            csubject_binary_scores={"speech": 0.9},
            csession_multiclass_threshold=0.5,
            csubject_binary_threshold=0.5,
            min_passing_tasks=1,
            param_count=10_000_000,
            param_count_limit=30_000_000,
            ci_protocol=bad_ci,
            per_task_seed_fn=lambda t, p: 0,
            independence_check_passed=True,
            independence_note="",
        )


def test_artifact_roundtrip(tmp_path) -> None:
    decision = apply_submission_gate(
        task_ids=("speech",),
        csession_multiclass_scores={"speech": 0.8},
        csubject_binary_scores={"speech": 0.7},
        csession_multiclass_threshold=0.667,
        csubject_binary_threshold=0.628,
        min_passing_tasks=1,
        param_count=13_000_000,
        param_count_limit=30_000_000,
        ci_protocol=_const_ci(),
        per_task_seed_fn=lambda t, p: 33,
        independence_check_passed=True,
        independence_note="",
    )
    out = tmp_path / "gate.json"
    decision.write_artifact(out)
    loaded = json.loads(out.read_text())
    assert loaded["passed"] is True
    assert loaded["n_passing_tasks"] == 1
    assert loaded["rows"][0]["task_id"] == "speech"
