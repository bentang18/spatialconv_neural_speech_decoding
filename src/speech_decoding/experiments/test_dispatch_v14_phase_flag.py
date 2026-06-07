"""--phase flag + --chain dispatch routing (T3.1 scaffold → #21 chain driver).

Phase 2 stays gated (collapsed into the joint phase, B29 Item 1). Phases 1/3/4
route to their experiment classes (V14JointExperiment / V14Phase3Experiment /
base Experiment or V14Phase4ReadoutExperiment). --chain assembles the staged
P1→P2→P3a→P3b→P4 pipeline. Most tests use --dry-run (short-circuits before any
Experiment is built); the chain-assembly tests monkeypatch build_v14_experiment
to capture per-phase kwargs without touching BT data."""

from __future__ import annotations

import pytest

from speech_decoding.experiments.dispatch_v14 import _resolve_corpus_mode, main
from speech_decoding.studies.braintreebank.manifest import (
    NO_TEACHER_CACHE_SESSIONS,
    V14_P3_DISTILL_SESSIONS,
    V14_PRETRAIN_SESSIONS,
)
from speech_decoding.studies.braintreebank.study import Wang2024Treebank


# --- leakage decouple (#82): study/eval corpus-mode routing -----------------

def test_resolve_corpus_mode_joint_forces_pretrain() -> None:
    # P1/P2 (joint) ALWAYS pretrain on the full legal corpus ("pretrain" =
    # V14_PRETRAIN_SESSIONS), even when the CLI passes an eval mode/split — this
    # is what prevents leakage. (No teacher is needed, so all 13 sessions stay.)
    assert _resolve_corpus_mode(
        joint_phase=True, p3_distill=False,
        mode="full", eval_mode="CrossSession",
    ) == ("pretrain", "Pretrain")
    assert _resolve_corpus_mode(
        joint_phase=True, p3_distill=False,
        mode="lite", eval_mode="CrossSubject",
    ) == ("pretrain", "Pretrain")


def test_resolve_corpus_mode_p3_routes_to_distill_corpus() -> None:
    # LC4: P3 (distill) routes to "p3_distill" (V14_P3_DISTILL_SESSIONS = the 13
    # pretrain sessions minus (8,0), which has no Whisper teacher cache), NOT the
    # full "pretrain" corpus — routing P3 to "pretrain" crashes lazily on the
    # first (8,0) clip. Still the leakage-free "Pretrain" split, any CLI mode.
    assert _resolve_corpus_mode(
        joint_phase=False, p3_distill=True,
        mode="full", eval_mode="CrossSession",
    ) == ("p3_distill", "Pretrain")
    assert _resolve_corpus_mode(
        joint_phase=False, p3_distill=True,
        mode="lite", eval_mode="CrossSubject",
    ) == ("p3_distill", "Pretrain")


def test_p3_distill_corpus_excludes_session_8_0_no_teacher() -> None:
    # LC4: the P3 distill corpus EXCLUDES (8,0) / subject 8 (no teacher cache):
    # 12 sessions over subjects {1,2,3,4,6,9}. Was the confirmed defect — P3 ran
    # on the full 13-session set and crashed on (8,0).
    assert (8, 0) in NO_TEACHER_CACHE_SESSIONS
    assert (8, 0) in V14_PRETRAIN_SESSIONS  # legal to pretrain on (P1/P2)...
    assert (8, 0) not in V14_P3_DISTILL_SESSIONS  # ...but no teacher for P3.
    assert 8 not in {s for s, _ in V14_P3_DISTILL_SESSIONS}
    assert set(V14_P3_DISTILL_SESSIONS) == set(V14_PRETRAIN_SESSIONS) - {(8, 0)}
    assert len(V14_P3_DISTILL_SESSIONS) == 12
    assert sorted({s for s, _ in V14_P3_DISTILL_SESSIONS}) == [1, 2, 3, 4, 6, 9]


def test_p3_study_mode_emits_exactly_distill_corpus(tmp_path) -> None:
    # LC4 end-to-end: the study with mode="p3_distill" emits exactly the 12
    # teacher-backed sessions — the disk->emission wiring the runtime guard
    # cannot see (it only checks the realized loaders don't LEAK). iter_timelines
    # yields plain dicts (no data read); a tmp dir satisfies the study's path.
    study = Wang2024Treebank(path=str(tmp_path), mode="p3_distill")
    emitted = {
        (int(tl["subject_id"]), int(tl["trial_id"]))
        for tl in study.iter_timelines()
    }
    assert emitted == set(V14_P3_DISTILL_SESSIONS)
    assert (8, 0) not in emitted
    assert len(emitted) == 12


def test_ssl_study_sessions_invariant_to_cli_mode(tmp_path) -> None:
    # LC9: under SSL the realized study session set is the legal corpus
    # regardless of --mode (the #82 decouple). _resolve_corpus_mode returns the
    # same study mode for every CLI mode, and iter_timelines is identical across
    # them (it cannot depend on the CLI clip-sampling --mode).
    for cli_mode in ("lite", "full", "nano"):
        joint_study, _ = _resolve_corpus_mode(
            joint_phase=True, p3_distill=False,
            mode=cli_mode, eval_mode="CrossSession",
        )
        assert joint_study == "pretrain"
        p3_study, _ = _resolve_corpus_mode(
            joint_phase=False, p3_distill=True,
            mode=cli_mode, eval_mode="CrossSession",
        )
        assert p3_study == "p3_distill"
    # The emitted session set is identical across CLI modes for each SSL study.
    for study_mode, expected in (
        ("pretrain", set(V14_PRETRAIN_SESSIONS)),
        ("p3_distill", set(V14_P3_DISTILL_SESSIONS)),
    ):
        study = Wang2024Treebank(path=str(tmp_path), mode=study_mode)
        emitted = {
            (int(tl["subject_id"]), int(tl["trial_id"]))
            for tl in study.iter_timelines()
        }
        assert emitted == expected


def test_resolve_corpus_mode_p4_keeps_eval_split() -> None:
    # The supervised P4 probe (neither joint nor distill) IS the Neuroprobe
    # probe and must keep the CLI eval mode/split. A regression that forced P4
    # to "pretrain" would silently evaluate on the wrong data and the leakage
    # guard (P4-exempt) would NOT catch it — this is the only guard against it.
    for mode in ("lite", "full", "nano"):
        for ev in ("CrossSession", "CrossSubject"):
            assert _resolve_corpus_mode(
                joint_phase=False, p3_distill=False, mode=mode, eval_mode=ev,
            ) == (mode, ev)


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
        main(["--phase", "3", "--warmup-steps", "100"])
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
            "--no-target-standardize", "--warmup-steps", "100",
        ])
    assert "ROOT_DIR_BRAINTREEBANK" in str(exc_info.value)


def test_chain_without_work_dir_raises() -> None:
    """#21: --chain needs --work-dir for the per-phase ckpt handoff; fail fast
    at the operator boundary before any (data-bound) build."""
    with pytest.raises(ValueError) as exc_info:
        main(["--chain", "--whisper-target-cache-dir", "/x", "--no-target-standardize",
              "--warmup-steps", "100"])
    assert "--work-dir" in str(exc_info.value)


def test_chain_without_whisper_cache_raises(tmp_path) -> None:
    """#21: --chain runs the P3 distill stages, so the teacher cache is required."""
    with pytest.raises(ValueError) as exc_info:
        main(["--chain", "--work-dir", str(tmp_path), "--no-target-standardize",
              "--warmup-steps", "100"])
    assert "--whisper-target-cache-dir" in str(exc_info.value)


def test_chain_standardize_without_channel_stats_raises(tmp_path) -> None:
    """#21: --chain with B33 default standardization needs --channel-stats-path."""
    with pytest.raises(ValueError) as exc_info:
        main([
            "--chain", "--work-dir", str(tmp_path),
            "--whisper-target-cache-dir", "/x", "--warmup-steps", "100",
        ])
    assert "--channel-stats-path" in str(exc_info.value)


def test_chain_standardize_with_missing_channel_stats_path_raises(tmp_path) -> None:
    """B33 default standardization + a --channel-stats-path that doesn't exist
    must fast-fail at dispatch (audit A4-HIGH#2b), NOT hours into the chain at
    P3a's torch.load. Pins the _validate_channel_stats_path guard so a refactor
    can't silently drop it."""
    with pytest.raises(ValueError) as exc_info:
        main([
            "--chain", "--work-dir", str(tmp_path),
            "--whisper-target-cache-dir", "/x",
            "--channel-stats-path", str(tmp_path / "missing.pt"),
            "--warmup-steps", "100",
        ])
    assert "is not a file" in str(exc_info.value)


def test_chain_standardize_rejects_channel_stats_directory(tmp_path) -> None:
    """A directory passed where the .pt is expected (operator passes the cache
    dir, not the file) must ALSO fast-fail — .is_file() catches it where
    .exists() would let it through to a late torch.load crash."""
    with pytest.raises(ValueError) as exc_info:
        main([
            "--chain", "--work-dir", str(tmp_path),
            "--whisper-target-cache-dir", "/x",
            "--channel-stats-path", str(tmp_path),  # a directory, not a .pt
            "--warmup-steps", "100",
        ])
    assert "is not a file" in str(exc_info.value)


def test_chain_no_standardize_skips_channel_stats_check(monkeypatch, tmp_path) -> None:
    """The fast-fail is gated on target-std ON: --no-target-standardize with a
    bad channel-stats path must NOT raise (the path is unused). Pins the
    exemption so the guard can't over-fire."""
    import speech_decoding.experiments.dispatch_v14 as dv

    _capture_builds(monkeypatch)
    # Reaches _build_v14_chain directly (bypasses main()'s launch guard) and must
    # assemble all 5 phases without raising on the nonexistent path.
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/x", "--no-target-standardize",
        "--channel-stats-path", str(tmp_path / "missing.pt"),
    ])
    dv._build_v14_chain(args, cross_attn_positions=None)  # no raise


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


def test_chain_p1_always_armed_default(monkeypatch, tmp_path) -> None:
    """Default chain (guard armed): all four SSL/distill phases carry the guard;
    P4 keeps its own collapse_guard=False (frozen probe)."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
    ])
    dv._build_v14_chain(args, cross_attn_positions=None)
    assert [c["collapse_guard"] for c in calls] == [True, True, True, True, False]


def test_chain_no_collapse_guard_keeps_p1_armed(monkeypatch, tmp_path) -> None:
    """--no-collapse-guard disarms ONLY the parcel/distill phases (P2/P3a/P3b).
    P1 STAYS armed: its M2 front-end RankMe floor (~0.31) is above the 0.25 hard
    alarm so it can't false-positive, and keeping it armed pins P1's exca UID so
    a guard-off relaunch reuses the cached P1 instead of re-running it. The M4
    parcel phases (floor ~0.05) are the ones that false-positive, so only they
    are disarmed. P4 is always collapse_guard=False (frozen probe, #54 M1)."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
        "--no-collapse-guard",
    ])
    dv._build_v14_chain(args, cross_attn_positions=None)
    # P1 armed; P2/P3a/P3b disarmed; P4 always off.
    assert [c["collapse_guard"] for c in calls] == [True, False, False, False, False]


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


def test_chain_budget_and_early_stop_wiring(monkeypatch, tmp_path) -> None:
    """#32: the SSL/distill phases (P1/P2/P3a/P3b) ride a fixed budget and NEVER
    early-stop on val loss; only the P4 supervised probe early-stops (val_loss-min
    IS correct there). ``--ssl-max-steps`` step-budgets the SSL phases; P4 ignores
    it. This is the "NOT naive val_loss-min" guarantee, asserted at the wiring."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
        "--ssl-max-steps", "5000",
    ])
    dv._build_v14_chain(args, cross_attn_positions=None)
    # SSL/distill phases: step-budgeted, never early-stop.
    for i in (0, 1, 2, 3):
        assert calls[i]["max_steps"] == 5000
        assert "early_stopping_patience" not in calls[i]
    # P4: early-stops on val_loss at the default patience; ignores --ssl-max-steps.
    assert calls[4]["early_stopping_patience"] == dv.DEFAULT_P4_EARLY_STOP_PATIENCE
    assert "max_steps" not in calls[4]


def test_chain_default_budget_and_patience_disable(monkeypatch, tmp_path) -> None:
    """#32 defaults: no SSL step budget (epoch budget via --n-epochs); a P4
    patience <= 0 disables early-stop and runs P4 to the --n-epochs cap."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
        "--p4-early-stop-patience", "0",
    ])
    dv._build_v14_chain(args, cross_attn_positions=None)
    for i in (0, 1, 2, 3):
        assert calls[i]["max_steps"] is None  # default -> epoch budget
    assert calls[4]["early_stopping_patience"] is None  # 0 disables


def test_chain_default_optim_is_locked_b01_config(monkeypatch, tmp_path) -> None:
    """Default FLIPPED 2026-06-04 (project_v14_optimizer_default_b01_config): with
    no optim flags every chain phase gets the §7/B01-locked config — lr=1e-3,
    warmup_cosine, AdamW, β2=0.95, grad_clip=1.0, wd=0.0. constant-Adam/β2=0.999
    was the unimplemented-default *bug*, never a chosen config, so it is no longer
    the default (and a real SSL/distill run refuses it — see
    test_launch_guard_refuses_real_ssl_on_unimplemented_default_optim)."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
    ])
    dv._build_v14_chain(args, cross_attn_positions=None)
    for c in calls:  # all 5 phases
        assert c["lr"] == 1e-3
        assert c["lr_schedule"] == "warmup_cosine"
        assert c["optimizer_name"] == "AdamW"
        assert c["weight_decay"] == 0.0
        assert c["adam_betas"] == (0.9, 0.95)
        assert c["gradient_clip_val"] == 1.0


def test_launch_guard_refuses_real_ssl_on_unimplemented_default_optim(monkeypatch) -> None:
    """§7 launch guard (2026-06-04): a real (non-fast-dev-run) SSL/distill run
    (P1/P2/P3 or --chain) refuses constant LR / plain Adam / β2≥0.999 so the
    unimplemented-default config can't ship silently on a multi-hour run.
    --fast-dev-run is the smoke escape hatch; --phase 4 is the supervised probe,
    not SSL/distill, so it is exempt."""
    _capture_builds(monkeypatch)
    # Each optim-clause case carries a valid --warmup-steps so it isolates ONE
    # clause (otherwise the default warmup_steps=0 co-fires the warmup-floor
    # clause and the match= could pass on the wrong message). --lr-schedule
    # constant ignores warmup, so its case needs no warmup flag. The match= uses
    # the clause-specific _bad fragment (not "AdamW"/"warmup_cosine"/"0.95", which
    # also appear in the guard's unconditional boilerplate) so each case pins THAT
    # clause, not merely "some clause fired".
    with pytest.raises(SystemExit, match="--lr-schedule constant"):
        main(["--phase", "1", "--lr-schedule", "constant"])
    with pytest.raises(SystemExit, match="--optimizer Adam"):
        main(["--phase", "1", "--optimizer", "Adam", "--warmup-steps", "100"])
    with pytest.raises(SystemExit, match=r"β2=0\.999"):
        main(["--phase", "1", "--adam-beta2", "0.999", "--warmup-steps", "100"])
    # warmup_cosine + a too-short warmup ramps to peak LR within the first step
    # of a cold-init SSL model (readiness-audit finding b). The default schedule
    # IS warmup_cosine, so the bare default (--warmup-steps 0) is refused...
    with pytest.raises(SystemExit, match="too short to ramp"):
        main(["--phase", "1", "--warmup-steps", "0"])
    # ...AND --warmup-steps 1 must ALSO be refused: the (step+1)/warmup_steps
    # ramp puts it at peak on step 0 too — a bare `> 0` check would let it
    # through (audit finding A1). The 1%-of-budget floor catches it.
    with pytest.raises(SystemExit, match="too short to ramp"):
        main(["--phase", "1", "--warmup-steps", "1", "--ssl-max-steps", "1500"])
    # The locked 150/1500 (=10%) is well above the floor and must NOT raise.
    main(["--phase", "1", "--warmup-steps", "150", "--ssl-max-steps", "1500"])
    # --fast-dev-run exempts the smoke even on constant+Adam (must NOT raise).
    main(["--phase", "1", "--lr-schedule", "constant", "--optimizer", "Adam",
          "--fast-dev-run"])
    # --fast-dev-run also exempts warmup_steps=0 (smoke needs no LR ramp).
    main(["--phase", "1", "--warmup-steps", "0", "--fast-dev-run"])


def test_dry_run_exempts_real_ssl_default_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The §7 launch guard exempts --dry-run, same as --fast-dev-run: a dry-run
    prints the run summary and short-circuits BEFORE any build/train, so it is a
    config preview, not a real launch. A real SSL phase / --chain on the bare
    default (warmup_cosine + warmup_steps=0) must dry-run to rc=0, NOT raise
    SystemExit — else the operator can't even preview the locked config. This
    pins the exemption directly (it was only covered incidentally before)."""
    assert main(["--phase", "1", "--dry-run"]) == 0
    assert "V14 dispatch" in capsys.readouterr().out
    # --chain exercises the other half of `_is_ssl_distill`; --dry-run short-
    # circuits before the chain's --work-dir/cache/channel-stats validation too.
    assert main(["--chain", "--dry-run"]) == 0


def test_chain_lof_off_by_default(monkeypatch, tmp_path) -> None:
    """#17: no LOF flags → every phase builds with LOF off (cache uid untouched)."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
    ])
    dv._build_v14_chain(args, cross_attn_positions=None)
    for c in calls:  # all 5 phases
        assert c["lof_bad_channels"] is False
        assert c["lof_report_path"] is None


def test_chain_lof_flags_reach_every_phase(monkeypatch, tmp_path) -> None:
    """#17: --lof-bad-channels + threshold/n_neighbors/report-path thread to all
    phases via _common_build_kwargs, so the chain's SSL/distill/probe data all
    share one bad-channel set."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
        "--lof-bad-channels", "--lof-threshold", "1.8",
        "--lof-n-neighbors", "15", "--lof-report-path", str(tmp_path / "lof.json"),
    ])
    dv._build_v14_chain(args, cross_attn_positions=None)
    for c in calls:
        assert c["lof_bad_channels"] is True
        assert c["lof_threshold"] == 1.8
        assert c["lof_n_neighbors"] == 15
        assert c["lof_report_path"] == str(tmp_path / "lof.json")


def test_chain_c_max_default_and_override(monkeypatch, tmp_path) -> None:
    """#35: c_max defaults to 384 on every phase, and --c-max 256 threads to all
    five (so the chain's c_max-padded extractors stay shape-consistent)."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    base = ["--chain", "--work-dir", str(tmp_path),
            "--whisper-target-cache-dir", "/c", "--no-target-standardize"]
    dv._build_v14_chain(_parse(base), cross_attn_positions=None)
    assert all(c["c_max"] == dv.DEFAULT_C_MAX for c in calls)

    calls.clear()
    dv._build_v14_chain(_parse(base + ["--c-max", "256"]), cross_attn_positions=None)
    assert all(c["c_max"] == 256 for c in calls)


def test_chain_warmup_cosine_optim_reaches_every_phase(monkeypatch, tmp_path) -> None:
    """#37: the locked warmup-cosine + AdamW + β2=0.95 path threads to all phases
    via _common_build_kwargs; --adam-beta2 becomes a (0.9, β2) tuple. weight_decay
    is held at 0.0 here purely to isolate the threading check — wd>0 is now
    accepted (the §7 no-WD param groups landed, #40; see
    test_build_optim_cfg_forwards_weight_decay_no_wd_groups_landed)."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
        "--lr-schedule", "warmup_cosine", "--optimizer", "AdamW",
        "--lr", "5e-4", "--warmup-steps", "2000", "--weight-decay", "0.0",
        "--adam-beta2", "0.95", "--min-lr-ratio", "0.0", "--grad-clip", "1.0",
    ])
    dv._build_v14_chain(args, cross_attn_positions=None)
    for c in calls:
        assert c["lr_schedule"] == "warmup_cosine"
        assert c["optimizer_name"] == "AdamW"
        assert c["lr"] == 5e-4
        assert c["warmup_steps"] == 2000
        assert c["weight_decay"] == 0.0
        assert c["adam_betas"] == (0.9, 0.95)
        assert c["gradient_clip_val"] == 1.0


def test_grad_clip_zero_disables(monkeypatch) -> None:
    """--grad-clip <= 0 → gradient_clip_val=None (clipping off)."""
    calls = _capture_builds(monkeypatch)
    # --warmup-steps is required on a real (non-fast-dev-run) SSL phase by the §7
    # launch guard (warmup_cosine + warmup_steps=0 is refused). The single-phase
    # config-capture tests below all carry a valid nonzero warmup so the build is
    # reached; the value itself is irrelevant to what each test asserts.
    main(["--phase", "1", "--grad-clip", "0", "--warmup-steps", "100"])
    assert calls[0]["gradient_clip_val"] is None


def test_accumulate_grad_batches_reaches_every_phase(monkeypatch, tmp_path) -> None:
    """#42: --accumulate-grad-batches threads to all chain phases via
    _common_build_kwargs (effective batch = --batch-size * N)."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
        "--accumulate-grad-batches", "8",
    ])
    dv._build_v14_chain(args, cross_attn_positions=None)
    assert calls and all(c["accumulate_grad_batches"] == 8 for c in calls)


def test_accumulate_grad_batches_default_is_one(monkeypatch) -> None:
    """No flag → accumulate_grad_batches=1 (no accumulation, prior behavior)."""
    calls = _capture_builds(monkeypatch)
    main(["--phase", "1", "--warmup-steps", "100"])
    assert calls[0]["accumulate_grad_batches"] == 1


def test_wd_exclude_norms_reaches_every_phase(monkeypatch, tmp_path) -> None:
    """#40: --wd-exclude-norms / --no-wd-exclude-norms threads to ALL chain
    phases via _common_build_kwargs (the no-WD split is uniform across the
    stack). Default is True."""
    import speech_decoding.experiments.dispatch_v14 as dv

    base = ["--chain", "--work-dir", str(tmp_path),
            "--whisper-target-cache-dir", "/c", "--no-target-standardize"]

    calls = _capture_builds(monkeypatch)
    dv._build_v14_chain(_parse(base), cross_attn_positions=None)
    assert calls and all(c["wd_exclude_norms"] is True for c in calls)

    calls.clear()
    dv._build_v14_chain(_parse(base + ["--no-wd-exclude-norms"]),
                        cross_attn_positions=None)
    assert calls and all(c["wd_exclude_norms"] is False for c in calls)


def test_wd_exclude_norms_default_single_phase(monkeypatch) -> None:
    """No flag → wd_exclude_norms=True on a single-phase build."""
    calls = _capture_builds(monkeypatch)
    main(["--phase", "1", "--warmup-steps", "100"])
    assert calls[0]["wd_exclude_norms"] is True


def test_parcel_lr_scale_reaches_p3_phases(monkeypatch, tmp_path) -> None:
    """#78: --parcel-lr-scale (P3-3b parcel-side discriminative LR) threads to the
    two chain P3 stages; the default is 1/3. Mirrors --p2-frontend-lr-scale —
    set explicitly on the P3 call sites, not in the shared `common` dict."""
    import speech_decoding.experiments.dispatch_v14 as dv

    base = ["--chain", "--work-dir", str(tmp_path),
            "--whisper-target-cache-dir", "/c", "--no-target-standardize"]

    calls = _capture_builds(monkeypatch)
    dv._build_v14_chain(_parse(base + ["--parcel-lr-scale", "0.25"]),
                        cross_attn_positions=None)
    # calls[2]=P3a, calls[3]=P3b carry the override.
    assert calls[2]["parcel_lr_scale"] == 0.25
    assert calls[3]["parcel_lr_scale"] == 0.25

    calls.clear()
    dv._build_v14_chain(_parse(base), cross_attn_positions=None)
    assert calls[2]["parcel_lr_scale"] == 1.0 / 3.0


def test_parcel_lr_scale_reaches_single_phase(monkeypatch) -> None:
    """--parcel-lr-scale reaches a single --phase 3 build too."""
    calls = _capture_builds(monkeypatch)
    main(["--phase", "3", "--p3-stage", "3b",
          "--whisper-target-cache-dir", "/c", "--no-target-standardize",
          "--parcel-lr-scale", "0.5", "--warmup-steps", "100"])
    assert calls[0]["parcel_lr_scale"] == 0.5


def test_rankme_thresholds_reach_every_phase(monkeypatch, tmp_path) -> None:
    """#74: --rankme-warn/alarm-threshold thread to ALL chain builds via
    _common_build_kwargs (build_v14_experiment uses them only on the joint
    branch). Default is None (→ canonical 0.5/0.25 resolved in the module)."""
    import speech_decoding.experiments.dispatch_v14 as dv

    base = ["--chain", "--work-dir", str(tmp_path),
            "--whisper-target-cache-dir", "/c", "--no-target-standardize"]

    calls = _capture_builds(monkeypatch)
    dv._build_v14_chain(_parse(base), cross_attn_positions=None)
    assert calls and all(c["rankme_warn_threshold"] is None for c in calls)
    assert all(c["rankme_alarm_threshold"] is None for c in calls)

    calls.clear()
    dv._build_v14_chain(
        _parse(base + ["--rankme-warn-threshold", "0.4",
                       "--rankme-alarm-threshold", "0.3"]),
        cross_attn_positions=None,
    )
    assert calls and all(c["rankme_warn_threshold"] == 0.4 for c in calls)
    assert all(c["rankme_alarm_threshold"] == 0.3 for c in calls)


def test_rankme_thresholds_default_single_phase(monkeypatch) -> None:
    """No flag → both thresholds None on a single-phase build (joint P1)."""
    calls = _capture_builds(monkeypatch)
    main(["--phase", "1", "--warmup-steps", "100"])
    assert calls[0]["rankme_warn_threshold"] is None
    assert calls[0]["rankme_alarm_threshold"] is None


def test_build_optim_cfg_default_matches_prior_and_validates() -> None:
    """The default optim dict is the prior plain-Adam (no scheduler, no β/wd, and
    — post live-crash fix 2026-06-04 — NOT fused; see
    test_optim_not_amp_scaling_so_grad_clip_works) and validates as a
    LightningOptimizer with scheduler is None."""
    import speech_decoding.experiments.dispatch_v14 as dv
    from neuraltrain.optimizers.base import LightningOptimizer

    cfg = dv._build_optim_cfg(
        lr=1e-3, lr_schedule="constant", warmup_steps=0, min_lr_ratio=0.0,
        weight_decay=0.0, optimizer_name="Adam", adam_betas=None,
    )
    assert cfg == {"optimizer": {"name": "Adam", "lr": 1e-3, "kwargs": {}}}
    assert LightningOptimizer.model_validate(cfg).scheduler is None


def test_build_optim_cfg_warmup_cosine_validates_with_scheduler() -> None:
    """warmup_cosine attaches a top-level WarmupCosine scheduler + AdamW β/wd
    and validates (catches the 'kwargs-wrapper would fail extra=forbid' trap)."""
    import speech_decoding.experiments.dispatch_v14 as dv
    from neuraltrain.optimizers.base import LightningOptimizer

    from speech_decoding.experiments.lr_schedule import WarmupCosine

    cfg = dv._build_optim_cfg(
        lr=5e-4, lr_schedule="warmup_cosine", warmup_steps=2000, min_lr_ratio=0.0,
        weight_decay=0.0, optimizer_name="AdamW", adam_betas=(0.9, 0.95),
    )
    assert cfg["optimizer"]["kwargs"]["betas"] == [0.9, 0.95]
    assert cfg["scheduler"] == {"name": "WarmupCosine", "warmup_steps": 2000, "min_lr_ratio": 0.0}
    assert cfg["interval"] == "step"
    lopt = LightningOptimizer.model_validate(cfg)
    assert isinstance(lopt.scheduler, WarmupCosine)


def test_build_optim_cfg_forwards_weight_decay_no_wd_groups_landed() -> None:
    """§7 no-WD param-group split (#40) LANDED: a non-zero weight_decay is now
    FORWARDED as the top-level optimizer wd, and the phase modules'
    configure_optimizers ride the no-WD-group override (biases / LN γβ / embeds
    in a weight_decay:0.0 group) — see test_optim_param_groups +
    test_v14_joint_module::test_configure_optimizers_no_wd_split_applied_at_positive_wd.
    The earlier guard that refused wd>0 is removed."""
    import speech_decoding.experiments.dispatch_v14 as dv

    cfg = dv._build_optim_cfg(
        lr=5e-4, lr_schedule="warmup_cosine", warmup_steps=2000,
        min_lr_ratio=0.0, weight_decay=0.05, optimizer_name="AdamW",
        adam_betas=(0.9, 0.95),
    )
    assert cfg["optimizer"]["kwargs"]["weight_decay"] == 0.05


def test_build_optim_cfg_adamw_pins_weight_decay_zero_no_torch_default_leak() -> None:
    """§7/B01 lock, re-audit 2026-06-03: torch ``AdamW`` defaults weight_decay to
    0.01. On the lock-recommended ``--optimizer AdamW --weight-decay 0`` path the
    build sees a falsy flag and would otherwise OMIT weight_decay, letting AdamW's
    0.01 decay biases / LN γβ / freq_embed uniformly — the exact silent violation
    the §7 no-WD lock forbids. So AdamW must pin weight_decay=0.0 explicitly; Adam
    (torch default already 0.0) carries empty kwargs (non-fused, post-crash fix)."""
    import speech_decoding.experiments.dispatch_v14 as dv

    adamw = dv._build_optim_cfg(
        lr=5e-4, lr_schedule="constant", warmup_steps=0, min_lr_ratio=0.0,
        weight_decay=0.0, optimizer_name="AdamW", adam_betas=None,
    )
    assert adamw["optimizer"]["kwargs"]["weight_decay"] == 0.0
    assert "fused" not in adamw["optimizer"]["kwargs"]
    adam = dv._build_optim_cfg(
        lr=1e-3, lr_schedule="constant", warmup_steps=0, min_lr_ratio=0.0,
        weight_decay=0.0, optimizer_name="Adam", adam_betas=None,
    )
    assert adam["optimizer"]["kwargs"] == {}


def test_optim_not_amp_scaling_so_grad_clip_works() -> None:
    """Regression — live BT crash 2026-06-04 (job 47686013): a *fused* torch
    optimizer sets ``_step_supports_amp_scaling=True``, which makes Lightning's
    bf16-mixed AMP precision plugin REFUSE gradient clipping (``precision/amp.py``:
    "does not allow for gradient clipping because it performs unscaling of
    gradients internally"). The §7 recipe locks BOTH ``grad_clip=1.0`` and
    ``bf16-mixed``, so ``_build_optim_cfg`` must build a NON-fused optimizer.
    Assert the actual torch optimizer our cfg builds clears the exact flag
    Lightning checks, for both families on the locked warmup-cosine + β2 path."""
    import torch
    import speech_decoding.experiments.dispatch_v14 as dv
    from neuraltrain.optimizers.base import LightningOptimizer

    for name in ("Adam", "AdamW"):
        cfg = dv._build_optim_cfg(
            lr=5e-4, lr_schedule="warmup_cosine", warmup_steps=10,
            min_lr_ratio=0.1, weight_decay=0.0, optimizer_name=name,
            adam_betas=(0.9, 0.95),
        )
        assert cfg["optimizer"]["kwargs"].get("fused") is not True
        out = LightningOptimizer.model_validate(cfg).build(
            [torch.zeros(1, requires_grad=True)], total_steps=100
        )
        opt = out["optimizer"]
        # The exact check Lightning's AMP clip guard performs.
        assert getattr(opt, "_step_supports_amp_scaling", False) is False, (
            f"{name} would trip the bf16-mixed AMP grad-clip guard"
        )


def test_single_phase_passes_sister_flags(monkeypatch) -> None:
    """The single-phase build must keep passing the same sisters (the shared
    _common_build_kwargs helper feeds both call sites)."""
    calls = _capture_builds(monkeypatch)
    main([
        "--phase", "1", "--loss-variant", "b31_plus_utt",
        "--latent-valid-override", "all_true", "--no-binary-tasks",
        "--warmup-steps", "100",
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
    main(["--phase", "1", "--warmup-steps", "100"])
    assert calls[0]["clip_len"] == 5.0


def test_single_phase_p1_threads_ssl_max_steps(monkeypatch) -> None:
    """#39 (audit 2026-06-03): --ssl-max-steps reaches the single-phase P1 build
    (it previously reached only --chain), and P1 carries no P4 early-stop."""
    calls = _capture_builds(monkeypatch)
    main(["--phase", "1", "--ssl-max-steps", "5000", "--warmup-steps", "100"])
    assert calls[0]["max_steps"] == 5000
    assert calls[0]["early_stopping_patience"] is None


def test_single_phase_p3_threads_ssl_max_steps(monkeypatch) -> None:
    """#39: --ssl-max-steps reaches the single-phase P3 distill build too."""
    calls = _capture_builds(monkeypatch)
    main([
        "--phase", "3", "--ssl-max-steps", "5000",
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
        "--warmup-steps", "100",
    ])
    assert calls[0]["max_steps"] == 5000
    assert calls[0]["early_stopping_patience"] is None


def test_single_phase_p4_threads_early_stop_and_ignores_ssl_max_steps(
    monkeypatch,
) -> None:
    """#39: the next gate is a single-phase --phase 4 rerun. --p4-early-stop-patience
    must reach it (was silently ignored), and --ssl-max-steps must NOT (P4 uses the
    --n-epochs cap + val_loss early-stop, mirroring the chain's per-phase gating)."""
    calls = _capture_builds(monkeypatch)
    main([
        "--phase", "4", "--frozen-probe",
        "--p4-early-stop-patience", "7", "--ssl-max-steps", "5000",
    ])
    assert calls[0]["early_stopping_patience"] == 7
    assert calls[0]["max_steps"] is None


def test_single_phase_budget_defaults_match_chain(monkeypatch) -> None:
    """#39: with no budget flags, P1 gets no step budget / no early-stop, and a
    default --phase 4 inherits DEFAULT_P4_EARLY_STOP_PATIENCE (=10) — the same
    values the chain applies, so single-phase and chain stay in lock-step."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    main(["--phase", "1", "--warmup-steps", "100"])
    assert calls[0]["max_steps"] is None
    assert calls[0]["early_stopping_patience"] is None
    calls.clear()
    main(["--phase", "4", "--frozen-probe"])
    assert calls[0]["max_steps"] is None
    assert calls[0]["early_stopping_patience"] == dv.DEFAULT_P4_EARLY_STOP_PATIENCE


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


# --- C5 resilience: --slurm-requeue (preemption auto-resubmit) ---------------


def test_slurm_requeue_threads_to_every_chain_phase(monkeypatch, tmp_path) -> None:
    """C5: --slurm-requeue reaches EVERY chain phase via _common_build_kwargs, so
    the whole P1→P4 chain is requeue-resilient (not just the first phase). Default
    is False (smokes / single-shot runs don't requeue)."""
    import speech_decoding.experiments.dispatch_v14 as dv

    base = ["--chain", "--work-dir", str(tmp_path),
            "--whisper-target-cache-dir", "/c", "--no-target-standardize"]

    calls = _capture_builds(monkeypatch)
    dv._build_v14_chain(_parse(base), cross_attn_positions=None)
    assert calls and all(c["requeue"] is False for c in calls)

    calls.clear()
    dv._build_v14_chain(_parse(base + ["--slurm-requeue"]), cross_attn_positions=None)
    assert calls and all(c["requeue"] is True for c in calls)


def test_slurm_requeue_default_false_single_phase(monkeypatch) -> None:
    """No flag → requeue=False on a single-phase build."""
    calls = _capture_builds(monkeypatch)
    main(["--phase", "1", "--warmup-steps", "100"])
    assert calls[0]["requeue"] is False


def test_requeue_emits_slurm_additional_parameters(monkeypatch, tmp_path) -> None:
    """C5 end-to-end: requeue=True on a slurm build sets
    infra.slurm_additional_parameters={'requeue': True}, which submitit renders as
    a bare '#SBATCH --requeue' (bool True -> flag, verified against submitit's
    _as_sbatch_flag). False leaves it None so a smoke never auto-resubmits. Pins
    the flag→infra mapping the _capture_builds threading tests can't see."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    xp_on = build_v14_experiment(
        joint_phase=True, cluster="slurm", exca_folder=str(tmp_path),
        requeue=True, warmup_steps=100, max_steps=1500,
    )
    assert xp_on.infra.slurm_additional_parameters == {"requeue": True}

    xp_off = build_v14_experiment(
        joint_phase=True, cluster="slurm", exca_folder=str(tmp_path),
        requeue=False, warmup_steps=100, max_steps=1500,
    )
    assert xp_off.infra.slurm_additional_parameters is None
