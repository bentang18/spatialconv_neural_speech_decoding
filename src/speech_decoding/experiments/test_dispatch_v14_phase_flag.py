"""--phase flag + --chain dispatch routing (T3.1 scaffold → #21 chain driver).

Phase 2 stays gated (collapsed into the joint phase, B29 Item 1). Phases 1/3/4
route to their experiment classes (V14JointExperiment / V14Phase3Experiment /
base Experiment or V14Phase4ReadoutExperiment). --chain assembles the staged
P1→P2→P3a→P3b→P4 pipeline. Most tests use --dry-run (short-circuits before any
Experiment is built); the chain-assembly tests monkeypatch build_v14_experiment
to capture per-phase kwargs without touching BT data."""

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
    live (non-dry-run) path raises the precise operator error —
    see :func:`test_phase_3_live_without_cache_raises_operator_error`."""
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


# --- Gate-D fixes: chain assembly + sister-flag threading + P4 guards --------
#
# These monkeypatch build_v14_experiment to capture per-phase kwargs without
# touching BT data, closing the coverage gap the Gate-D audit found (the chain
# assembly + the common-dict flag drift were both untested).


class _StubXp:
    def run(self):  # noqa: D401 - the dispatch only calls .run()
        return {}


def _capture_builds(monkeypatch) -> list[dict]:
    """Replace dispatch_v14.build_v14_experiment with a kwargs-capturing stub."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls: list[dict] = []

    def fake_build(**kw):
        calls.append(kw)
        return _StubXp()

    monkeypatch.setattr(dv, "build_v14_experiment", fake_build)
    return calls


def _parse(argv: list[str]):
    import speech_decoding.experiments.dispatch_v14 as dv

    return dv._parser().parse_args(argv)


def test_chain_assembly_shape_and_handoff(monkeypatch, tmp_path) -> None:
    """#21 / Gate-D: --chain assembles exactly [P1, P2, P3a, P3b, P4] with the
    right phase selectors, 5 s SSL/distill + 1 s P4 clip windows, zero P4 lag,
    and the Whisper teacher stream on ONLY the two P3 stages. This mirrors the
    must-pass capstone ordering; without it a `common`-dict edit could ship a
    wrong maiden run undetected."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
    ])
    phases = dv._build_v14_chain(args, cross_attn_positions=None)
    assert len(calls) == len(phases) == 5
    assert calls[0]["joint_phase"] and calls[0]["jepa_phase"] == "p1"
    assert calls[1]["joint_phase"] and calls[1]["jepa_phase"] == "p2"
    assert calls[2]["p3_distill"] and calls[2]["p3_stage"] == "3a"
    assert calls[3]["p3_distill"] and calls[3]["p3_stage"] == "3b"
    assert calls[4]["phase4_frozen_probe"]
    assert [c["clip_len"] for c in calls] == [5.0, 5.0, 5.0, 5.0, 1.0]
    assert calls[4]["neural_lag_s"] == 0.0
    assert calls[2]["whisper_target_cache_dir"] == "/c"
    assert calls[3]["whisper_target_cache_dir"] == "/c"
    for non_p3 in (0, 1, 4):
        assert "whisper_target_cache_dir" not in calls[non_p3]


def test_chain_threads_sister_flags(monkeypatch, tmp_path) -> None:
    """Gate-D HIGH regression: loss_variant / latent_valid_override /
    sa_mask_mode reach the chain's joint phases, and binary_tasks reaches EVERY
    phase (so the SSL/distill clip population matches the P4 eval set). Before
    the _common_build_kwargs refactor these silently fell back to defaults while
    the run summary printed the sister as applied."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
        "--loss-variant", "b31_plus_m3",
        "--latent-valid-override", "all_true",
        "--sa-mask-mode", "key_only",
        "--no-binary-tasks",
    ])
    dv._build_v14_chain(args, cross_attn_positions=None)
    for i in (0, 1):  # the joint SSL phases carry the sisters
        assert calls[i]["loss_variant"] == "b31_plus_m3"
        assert calls[i]["latent_valid_override"] == "all_true"
        assert calls[i]["sa_mask_mode"] == "key_only"
    assert all(c["binary_tasks"] is False for c in calls)  # all 5 phases


def test_single_phase_passes_sister_flags(monkeypatch) -> None:
    """The single-phase build must keep passing the same sisters (the shared
    _common_build_kwargs helper feeds both call sites)."""
    calls = _capture_builds(monkeypatch)
    main([
        "--phase", "1", "--loss-variant", "b31_plus_utt",
        "--latent-valid-override", "all_true", "--no-binary-tasks",
    ])
    assert len(calls) == 1
    assert calls[0]["loss_variant"] == "b31_plus_utt"
    assert calls[0]["latent_valid_override"] == "all_true"
    assert calls[0]["binary_tasks"] is False
    assert calls[0]["joint_phase"] is True


def test_p4_clip_len_defaults_to_one_second(monkeypatch, capsys) -> None:
    """Gate-B flag 3 / Gate-D: a single --phase 4 with no --clip-len resolves to
    the 1 s leaderboard-parity window, not the 5 s SSL default."""
    calls = _capture_builds(monkeypatch)
    main(["--phase", "4", "--frozen-probe"])
    assert calls[0]["clip_len"] == 1.0
    assert "clip_len=1.0" in capsys.readouterr().out


def test_non_p4_clip_len_defaults_to_five_seconds(monkeypatch) -> None:
    """SSL/distill phases keep the 5 s default when --clip-len is unset."""
    calls = _capture_builds(monkeypatch)
    main(["--phase", "1"])
    assert calls[0]["clip_len"] == 5.0


def test_explicit_clip_len_overrides_phase_default(monkeypatch) -> None:
    """An explicit --clip-len is honored on any phase."""
    calls = _capture_builds(monkeypatch)
    main(["--phase", "4", "--frozen-probe", "--clip-len", "2.0"])
    assert calls[0]["clip_len"] == 2.0


def test_snapshot_ckpt_to_implies_frozen_probe_on_p4(monkeypatch) -> None:
    """Gate-D: --snapshot-ckpt-to needs the transferable protocol, which only
    the frozen-probe readout carries; on P4 it must select that experiment so it
    doesn't TypeError at runtime after a full train."""
    calls = _capture_builds(monkeypatch)
    main(["--phase", "4", "--snapshot-ckpt-to", "/tmp/p4.ckpt"])
    assert calls[0]["phase4_frozen_probe"] is True
    assert calls[0]["snapshot_ckpt_to"] == "/tmp/p4.ckpt"


def test_resume_from_implies_frozen_probe_on_p4(monkeypatch) -> None:
    calls = _capture_builds(monkeypatch)
    main(["--phase", "4", "--resume-from", "/tmp/p3b.ckpt"])
    assert calls[0]["phase4_frozen_probe"] is True
    assert calls[0]["pretrained_ckpt"] == "/tmp/p3b.ckpt"


def test_plain_p4_is_not_frozen_probe(monkeypatch) -> None:
    """A bare --phase 4 (no resume / snapshot / frozen flag) stays the base
    supervised path."""
    calls = _capture_builds(monkeypatch)
    main(["--phase", "4"])
    assert calls[0]["phase4_frozen_probe"] is False
