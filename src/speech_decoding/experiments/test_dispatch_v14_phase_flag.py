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


def test_chain_default_optim_is_prior_constant_adam_plus_gradclip(monkeypatch, tmp_path) -> None:
    """#37: with no optim flags every phase gets the prior constant-Adam config
    (lr=1e-3, schedule=constant, Adam, no β/wd override) — EXCEPT grad_clip,
    which defaults to 1.0 (restoring the §7 locked-universal that was silently
    OFF). This is the "nothing silently changes but the bug-fix" guarantee."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
    ])
    dv._build_v14_chain(args, cross_attn_positions=None)
    for c in calls:  # all 5 phases
        assert c["lr"] == 1e-3
        assert c["lr_schedule"] == "constant"
        assert c["optimizer_name"] == "Adam"
        assert c["weight_decay"] == 0.0
        assert c["adam_betas"] is None
        assert c["gradient_clip_val"] == 1.0


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
    is held at 0.0 — wd>0 is guarded at the optim-build until the §7 no-WD param
    groups land (see test_build_optim_cfg_rejects_weight_decay_until_no_wd_groups);
    the threading itself is exercised here."""
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
    main(["--phase", "1", "--grad-clip", "0"])
    assert calls[0]["gradient_clip_val"] is None


def test_build_optim_cfg_default_matches_prior_and_validates() -> None:
    """The default optim dict is the prior plain-fused-Adam (no scheduler, no
    β/wd) and validates as a LightningOptimizer with scheduler is None."""
    import speech_decoding.experiments.dispatch_v14 as dv
    from neuraltrain.optimizers.base import LightningOptimizer

    cfg = dv._build_optim_cfg(
        lr=1e-3, lr_schedule="constant", warmup_steps=0, min_lr_ratio=0.0,
        weight_decay=0.0, optimizer_name="Adam", adam_betas=None,
    )
    assert cfg == {"optimizer": {"name": "Adam", "lr": 1e-3, "kwargs": {"fused": True}}}
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


def test_build_optim_cfg_rejects_weight_decay_until_no_wd_groups() -> None:
    """§7 lock guard (audit 2026-06-03): a non-zero weight_decay is refused loudly
    because the no-WD param-group split (biases / LN γβ / freq_embed exempt) is not
    yet implemented — uniform wd would violate the B01 lock. wd=0.0 is allowed."""
    import speech_decoding.experiments.dispatch_v14 as dv

    with pytest.raises(ValueError, match="no-WD param-group"):
        dv._build_optim_cfg(
            lr=5e-4, lr_schedule="warmup_cosine", warmup_steps=2000,
            min_lr_ratio=0.0, weight_decay=0.05, optimizer_name="AdamW",
            adam_betas=(0.9, 0.95),
        )


def test_build_optim_cfg_adamw_pins_weight_decay_zero_no_torch_default_leak() -> None:
    """§7 lock guard, re-audit 2026-06-03: torch ``AdamW`` defaults weight_decay to
    0.01. On the lock-recommended ``--optimizer AdamW --weight-decay 0`` path the
    guard sees a falsy flag and would otherwise OMIT weight_decay, letting AdamW's
    0.01 decay biases / LN γβ / freq_embed uniformly — the exact silent violation
    the guard exists to stop. So AdamW must pin weight_decay=0.0 explicitly; Adam
    (torch default already 0.0) must stay ``{"fused": True}`` bit-for-bit."""
    import speech_decoding.experiments.dispatch_v14 as dv

    adamw = dv._build_optim_cfg(
        lr=5e-4, lr_schedule="constant", warmup_steps=0, min_lr_ratio=0.0,
        weight_decay=0.0, optimizer_name="AdamW", adam_betas=None,
    )
    assert adamw["optimizer"]["kwargs"]["weight_decay"] == 0.0
    adam = dv._build_optim_cfg(
        lr=1e-3, lr_schedule="constant", warmup_steps=0, min_lr_ratio=0.0,
        weight_decay=0.0, optimizer_name="Adam", adam_betas=None,
    )
    assert adam["optimizer"]["kwargs"] == {"fused": True}


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


def test_single_phase_p1_threads_ssl_max_steps(monkeypatch) -> None:
    """#39 (audit 2026-06-03): --ssl-max-steps reaches the single-phase P1 build
    (it previously reached only --chain), and P1 carries no P4 early-stop."""
    calls = _capture_builds(monkeypatch)
    main(["--phase", "1", "--ssl-max-steps", "5000"])
    assert calls[0]["max_steps"] == 5000
    assert calls[0]["early_stopping_patience"] is None


def test_single_phase_p3_threads_ssl_max_steps(monkeypatch) -> None:
    """#39: --ssl-max-steps reaches the single-phase P3 distill build too."""
    calls = _capture_builds(monkeypatch)
    main([
        "--phase", "3", "--ssl-max-steps", "5000",
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
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
    main(["--phase", "1"])
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
