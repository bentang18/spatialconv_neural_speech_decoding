"""T3.1: --phase flag scaffold.

Phases 1/2/3 must raise NotImplementedError citing the gating blocker IDs;
Phase 4 (default) must fall through to the existing Phase-4 dispatch path
(here exercised with --dry-run so no Experiment is built)."""

from __future__ import annotations

import pytest

from speech_decoding.experiments.dispatch_v14 import main


def test_phase_2_raises_with_blocker_ids() -> None:
    """Phase 2 is the legacy split-P2 entry-point, collapsed into the joint
    phase by B29 Item 1; it stays gated at the dispatch level with the
    redirect-to-``--phase 1`` message."""
    with pytest.raises(NotImplementedError) as exc_info:
        main(["--phase", "2", "--dry-run"])
    message = str(exc_info.value)
    for token in ("B29 Item 1", "joint phase", "V14JointExperiment"):
        assert token in message, f"phase 2: missing blocker id {token}"


def test_phase_3_dry_run_no_longer_gated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """WS-F: --phase 3 is no longer blanket-gated. --dry-run short-circuits
    before any build, so it exits 0 (the module/experiment are wired); the
    live (non-dry-run) path raises the precise whisper_target data blocker —
    see :func:`test_phase_3_live_raises_whisper_target_data_blocker`."""
    rc = main(["--phase", "3", "--dry-run"])
    assert rc == 0
    assert "V14 dispatch" in capsys.readouterr().out


def test_phase_3_live_without_cache_raises_operator_error() -> None:
    """#21 (WS-H landed): --phase 3 is no longer blanket-gated. A live run
    without --whisper-target-cache-dir fails fast with a clear operator error
    (the P3 SmoothL1 loss has no target stream), NOT the old WS-H blocker."""
    with pytest.raises(ValueError) as exc_info:
        main(["--phase", "3"])
    message = str(exc_info.value)
    assert "--whisper-target-cache-dir" in message
    assert "Whisper distillation" in message


def test_phase_3_live_with_cache_routes_into_p3_build(monkeypatch) -> None:
    """#21: with the teacher cache supplied, --phase 3 routes into the P3 build
    (V14Phase3Experiment) rather than raising at the phase switch. With
    ROOT_DIR_BRAINTREEBANK unset the build raises the early data-root error —
    proof the dispatch reached build_v14_experiment, i.e. P3 is wired, not
    gated. (Full P3 construction is exercised by the synthetic-BT capstone.)"""
    monkeypatch.delenv("ROOT_DIR_BRAINTREEBANK", raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        main([
            "--phase", "3",
            "--whisper-target-cache-dir", "/nonexistent/teacher_cache",
            "--no-target-standardize",
        ])
    assert "ROOT_DIR_BRAINTREEBANK" in str(exc_info.value)


def test_chain_without_work_dir_raises() -> None:
    """#21: --chain needs --work-dir for the per-phase ckpt handoff; fail fast
    at the operator boundary before any (data-bound) build."""
    with pytest.raises(ValueError) as exc_info:
        main(["--chain", "--whisper-target-cache-dir", "/x", "--no-target-standardize"])
    assert "--work-dir" in str(exc_info.value)


def test_chain_without_whisper_cache_raises(tmp_path) -> None:
    """#21: --chain runs the P3 distill stages, so the teacher cache is required."""
    with pytest.raises(ValueError) as exc_info:
        main(["--chain", "--work-dir", str(tmp_path), "--no-target-standardize"])
    assert "--whisper-target-cache-dir" in str(exc_info.value)


def test_chain_standardize_without_channel_stats_raises(tmp_path) -> None:
    """#21: --chain with B33 default standardization needs --channel-stats-path."""
    with pytest.raises(ValueError) as exc_info:
        main([
            "--chain", "--work-dir", str(tmp_path),
            "--whisper-target-cache-dir", "/x",
        ])
    assert "--channel-stats-path" in str(exc_info.value)


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


# --- B36 (2026-06-03 H4) --jepa-phase staged masked-JEPA sub-phase ----------


def test_jepa_phase_default_p1_in_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No flag → the run summary records the default ``jepa_phase=p1``
    (front-end M2). --dry-run short-circuits before any build."""
    rc = main(["--dry-run"])
    assert rc == 0
    assert "jepa_phase=p1" in capsys.readouterr().out


def test_jepa_phase_p2_dry_run_prints_in_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """B36 H4: ``--phase 1 --jepa-phase p2`` selects the staged parcel-M4
    stage; the run summary records it so the persisted run record never
    silently rides the wrong stage. --dry-run exits 0 before the build."""
    rc = main(["--phase", "1", "--jepa-phase", "p2", "--dry-run"])
    assert rc == 0
    assert "jepa_phase=p2" in capsys.readouterr().out


def test_invalid_jepa_phase_rejected_by_argparse() -> None:
    """argparse ``choices`` rejects an unknown stage so the run record YAML
    never drifts to a typo'd sub-phase."""
    with pytest.raises(SystemExit):
        main(["--jepa-phase", "p3", "--dry-run"])
