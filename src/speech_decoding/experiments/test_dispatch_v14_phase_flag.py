"""T3.1: --phase flag scaffold.

Phases 1/2/3 must raise NotImplementedError citing the gating blocker IDs;
Phase 4 (default) must fall through to the existing Phase-4 dispatch path
(here exercised with --dry-run so no Experiment is built)."""

from __future__ import annotations

import pytest

from speech_decoding.experiments.dispatch_v14 import main


@pytest.mark.parametrize(
    "phase, expected_substrings",
    [
        # B29 Item 1 (2026-05-27 PM-late) collapsed P1 + P2 into a single
        # joint SSL phase routed through V14JointExperiment. B2.1 (#96)
        # closed 2026-05-28: phase=1 now constructs V14JointExperiment;
        # the SSL training-step blockers (B2.2-B2.5) raise from inside
        # ``V14JointExperiment._train_and_test`` instead. Phase 2 stays
        # gated with the B29 Item 1 redirect message; Phase 3 stays
        # gated on a frozen SSL checkpoint.
        (2, ("B29 Item 1", "joint phase", "V14JointExperiment")),
        (3, ("Phase-3 distillation", "frozen SSL checkpoint", "B29 Item 1")),
    ],
)
def test_phase_2_3_raise_with_blocker_ids(
    phase: int, expected_substrings: tuple[str, ...]
) -> None:
    with pytest.raises(NotImplementedError) as exc_info:
        main(["--phase", str(phase), "--dry-run"])
    message = str(exc_info.value)
    for token in expected_substrings:
        assert token in message, f"phase {phase}: missing blocker id {token}"


def test_phase_1_dry_run_constructs_joint_experiment_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """B2.1 (#96): phase=1 dispatches through the joint construction path
    (no NotImplementedError from the dispatch); the SSL training-step
    blockers (B2.2-B2.5) fire from inside V14JointExperiment.run().

    --dry-run short-circuits *before* the Experiment is built, so this
    test just confirms the dispatch path no longer raises at the
    phase-switch and that the V14 dispatch summary prints normally."""
    rc = main(["--phase", "1", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "V14 dispatch" in out


def test_phase_1_build_brain_module_uses_joint_ssl_module() -> None:
    """B2.2 (#97): the joint Experiment now overrides
    :meth:`_build_brain_module` to construct
    :class:`V14JointBrainModule` (EMA teacher + 3 LN heads + PMA + 4-term
    aggregator) instead of the parent CE-classifier ``BrainModule``.
    Confirms the surface exists and is grep'able; the actual training
    loop is exercised end-to-end by the synthetic-batch test in
    :mod:`test_v14_joint_module`."""
    import inspect

    from speech_decoding.experiments.v14_joint import (
        JOINT_PHASE_VALUE,
        V14JointExperiment,
    )
    from speech_decoding.experiments.v14_joint_module import V14JointBrainModule

    src = inspect.getsource(V14JointExperiment._build_brain_module)
    assert "V14JointBrainModule" in src, (
        "V14JointExperiment._build_brain_module must construct a "
        "V14JointBrainModule (B2.2 wiring)."
    )
    # Sanity: the override is grep'able from the symbol surface too.
    assert V14JointBrainModule.__name__ == "V14JointBrainModule"
    assert JOINT_PHASE_VALUE == 1


def test_phase_4_is_default_and_falls_through_to_dispatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--phase 4 is the current Phase-4 downstream path; --dry-run exits
    cleanly (no Experiment built)."""
    rc = main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "V14 dispatch" in out


def test_phase_4_explicit_falls_through_to_dispatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["--phase", "4", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "V14 dispatch" in out


def test_invalid_phase_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit):
        main(["--phase", "0", "--dry-run"])
    with pytest.raises(SystemExit):
        main(["--phase", "5", "--dry-run"])
