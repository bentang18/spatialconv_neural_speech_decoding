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
    BT_LITE_SESSIONS,
    BT_NANO_SESSIONS,
    NO_TEACHER_CACHE_SESSIONS,
    V14_P3_DISTILL_SESSIONS,
    V14_PRETRAIN_SESSIONS,
)
from speech_decoding.studies.braintreebank.study import Wang2024Treebank


@pytest.fixture(autouse=True)
def _restore_v14_compile_env():
    """main() sets global os.environ flags for its submitit subprocess (default
    compile-ON; --session-z-winsor sets V14_SESSION_Z_WINSOR). The --dry-run /
    --fast-dev-run dispatch tests call main() in-process, so without cleanup those
    flags leak into later tests in the same pytest process — V14_COMPILE forces
    torch.compile on the numerical-correctness joint tests (cross-test inductor
    symbolic-shape conflict); a stray V14_SESSION_Z_WINSOR silently clamps every
    later session-robust-z read. Snapshot + restore both so each dispatch test
    leaves the env exactly as it found it."""
    import os

    sentinel = object()
    saved = {k: os.environ.get(k, sentinel) for k in ("V14_COMPILE", "V14_SESSION_Z_WINSOR")}
    try:
        yield
    finally:
        for k, prev in saved.items():
            if prev is sentinel:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev  # type: ignore[assignment]


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


def test_resolve_corpus_mode_p4_routes_to_leaderboard_eval() -> None:
    # The supervised P4 probe (neither joint nor distill) IS the Neuroprobe eval,
    # so it ALWAYS runs the leaderboard protocol: study universe = BT_LITE (the 12
    # eval sessions, S5-free), with "nano" reserved for tiny smokes. --mode picks
    # the SSL clip budget ONLY; routing P4 to "full" would (a) load BT_FULL incl
    # excluded subject 5 (no data → crash) and (b) make CrossSession train on the
    # subject's *pretrain* trials — a silently wrong, non-leaderboard eval the
    # (P4-exempt) leakage guard would NOT catch. The eval split TYPE is preserved.
    for ev in ("CrossSession", "CrossSubject"):
        assert _resolve_corpus_mode(
            joint_phase=False, p3_distill=False, mode="full", eval_mode=ev,
        ) == ("lite", ev)
        assert _resolve_corpus_mode(
            joint_phase=False, p3_distill=False, mode="lite", eval_mode=ev,
        ) == ("lite", ev)
        assert _resolve_corpus_mode(
            joint_phase=False, p3_distill=False, mode="nano", eval_mode=ev,
        ) == ("nano", ev)


def test_p4_study_universe_is_bt_lite_for_all_cli_modes(tmp_path) -> None:
    # Positive guard: whatever clip budget --mode selects for SSL, the P4 probe's
    # realized study universe is exactly BT_LITE (the 12 Neuroprobe eval sessions,
    # S5-free) — never BT_FULL. This is the regression net for the sub_5 crash +
    # the CrossSession-trains-on-pretrain leaderboard contamination.
    for cli_mode, expected in (
        ("full", set(BT_LITE_SESSIONS)),
        ("lite", set(BT_LITE_SESSIONS)),
        ("nano", set(BT_NANO_SESSIONS)),
    ):
        p4_study, _ = _resolve_corpus_mode(
            joint_phase=False, p3_distill=False,
            mode=cli_mode, eval_mode="CrossSession",
        )
        study = Wang2024Treebank(path=str(tmp_path), mode=p4_study)
        emitted = {
            (int(tl["subject_id"]), int(tl["trial_id"]))
            for tl in study.iter_timelines()
        }
        assert emitted == expected, f"P4 universe wrong for --mode {cli_mode}"


def _p4_word_events(tmp_path, **kw):
    """Build a P4 frozen-probe experiment and return its BTWordEvents step."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    xp = build_v14_experiment(
        bt_root=str(tmp_path), mode="lite", phase4_frozen_probe=True, **kw
    )
    return xp.data.study.steps[1]


def test_build_threads_within_session_fold_index(tmp_path) -> None:
    # The P4 full-eval submitter scores WithinSession by KFold fold; build_v14_experiment
    # must forward eval_mode + fold_index to BTWordEvents (the only WithinSession
    # plumbing gap — eval_mode already flows through _resolve_corpus_mode).
    we0 = _p4_word_events(
        tmp_path, eval_mode="WithinSession", test_subject_id=3, test_trial_id=0,
        fold_index=0,
    )
    we1 = _p4_word_events(
        tmp_path, eval_mode="WithinSession", test_subject_id=3, test_trial_id=0,
        fold_index=1,
    )
    assert we0.eval_mode == "WithinSession" and we0.fold_index == 0
    assert we1.eval_mode == "WithinSession" and we1.fold_index == 1
    # Distinct folds must be distinct exca cells (different serialized config).
    assert we0.model_dump() != we1.model_dump()


def test_build_fold_index_default_is_uid_transparent(tmp_path) -> None:
    # Default fold_index=0 is the BTWordEvents default, so a CrossSession build is
    # byte-identical to one that never knew about fold_index (cache uid unchanged).
    we = _p4_word_events(
        tmp_path, eval_mode="CrossSession", test_subject_id=2, test_trial_id=4,
    )
    assert we.fold_index == 0


def test_phase_2_choice_removed() -> None:
    """Phase 2 was the legacy split-P2 entry-point, collapsed into the joint
    phase by B29 Item 1. Its --phase choice was culled (2026-06-13): argparse
    now only accepts {1, 3, 4}, so --phase 2 is rejected at parse time
    (SystemExit code 2)."""
    with pytest.raises(SystemExit):
        main(["--phase", "2", "--dry-run"])


def test_phase_3_dry_run_no_longer_gated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """WS-F: --phase 3 is no longer blanket-gated. --dry-run short-circuits
    before any build, so it exits 0 (the module/experiment are wired); the
    live (non-dry-run) path raises the precise operator error —
    see :func:`test_phase_3_live_without_cache_raises_operator_error`."""
    rc = main(["--phase", "3", "--pool", "cross_attn", "--dry-run"])
    assert rc == 0
    assert "V14 dispatch" in capsys.readouterr().out


def test_phase_3_live_without_cache_raises_operator_error() -> None:
    """#21 (WS-H landed): --phase 3 is no longer blanket-gated. A live run
    without --whisper-target-cache-dir fails fast with a clear operator error
    (the P3 SmoothL1 loss has no target stream), NOT the old WS-H blocker."""
    with pytest.raises(ValueError) as exc_info:
        main(["--phase", "3", "--pool", "cross_attn", "--warmup-steps", "100"])
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
            "--phase", "3", "--pool", "cross_attn",
            "--whisper-target-cache-dir", "/nonexistent/teacher_cache",
            "--no-target-standardize", "--warmup-steps", "100",
        ])
    assert "ROOT_DIR_BRAINTREEBANK" in str(exc_info.value)


def test_chain_without_work_dir_raises() -> None:
    """#21: --chain needs --work-dir for the per-phase ckpt handoff; fail fast
    at the operator boundary before any (data-bound) build. (Asserted against
    _build_v14_chain directly: main() gates the B37 chain SSL path upstream
    — chain+joint is "not wired yet" — before reaching the builder's own
    operator-boundary validation, which is what this test pins.)"""
    import speech_decoding.experiments.dispatch_v14 as dv

    with pytest.raises(ValueError) as exc_info:
        dv._build_v14_chain(_parse([
            "--chain", "--whisper-target-cache-dir", "/x",
            "--no-target-standardize", "--warmup-steps", "100",
        ]))
    assert "--work-dir" in str(exc_info.value)


def test_chain_without_whisper_cache_raises(tmp_path) -> None:
    """#21: --chain runs the P3 distill stages, so the teacher cache is
    required. (Asserted against _build_v14_chain directly — see
    test_chain_without_work_dir_raises for why main() can't reach it.)"""
    import speech_decoding.experiments.dispatch_v14 as dv

    with pytest.raises(ValueError) as exc_info:
        dv._build_v14_chain(_parse([
            "--chain", "--work-dir", str(tmp_path),
            "--no-target-standardize", "--warmup-steps", "100",
        ]))
    assert "--whisper-target-cache-dir" in str(exc_info.value)


def test_chain_standardize_without_channel_stats_raises(tmp_path) -> None:
    """#21: --chain with B33 default standardization needs --channel-stats-path.
    (Asserted against _build_v14_chain directly — see
    test_chain_without_work_dir_raises for why main() can't reach it.)"""
    import speech_decoding.experiments.dispatch_v14 as dv

    with pytest.raises(ValueError) as exc_info:
        dv._build_v14_chain(_parse([
            "--chain", "--work-dir", str(tmp_path),
            "--whisper-target-cache-dir", "/x", "--warmup-steps", "100",
        ]))
    assert "--channel-stats-path" in str(exc_info.value)


def test_chain_standardize_with_missing_channel_stats_path_raises(tmp_path) -> None:
    """B33 default standardization + a --channel-stats-path that doesn't exist
    must fast-fail at dispatch (audit A4-HIGH#2b), NOT hours into the chain at
    P3a's torch.load. Pins the _validate_channel_stats_path guard so a refactor
    can't silently drop it. (Asserted against _build_v14_chain directly — see
    test_chain_without_work_dir_raises for why main() can't reach it.)"""
    import speech_decoding.experiments.dispatch_v14 as dv

    with pytest.raises(ValueError) as exc_info:
        dv._build_v14_chain(_parse([
            "--chain", "--work-dir", str(tmp_path),
            "--whisper-target-cache-dir", "/x",
            "--channel-stats-path", str(tmp_path / "missing.pt"),
            "--warmup-steps", "100",
        ]))
    assert "is not a file" in str(exc_info.value)


def test_chain_standardize_rejects_channel_stats_directory(tmp_path) -> None:
    """A directory passed where the .pt is expected (operator passes the cache
    dir, not the file) must ALSO fast-fail — .is_file() catches it where
    .exists() would let it through to a late torch.load crash. (Asserted against
    _build_v14_chain directly — see test_chain_without_work_dir_raises for why
    main() can't reach it.)"""
    import speech_decoding.experiments.dispatch_v14 as dv

    with pytest.raises(ValueError) as exc_info:
        dv._build_v14_chain(_parse([
            "--chain", "--work-dir", str(tmp_path),
            "--whisper-target-cache-dir", "/x",
            "--channel-stats-path", str(tmp_path),  # a directory, not a .pt
            "--warmup-steps", "100",
        ]))
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
    dv._build_v14_chain(args)  # no raise


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


# --- B37 (2026-06-10) --pool / --ssl-mode joint SSL + D8 discriminative LR ----


def test_b37_joint_threads_pool_ssl_mode_and_d8_scales(tmp_path) -> None:
    """--pool mean joint build threads pool → brain config, ssl_mode/lambda_m4/
    per-tap predictor depths → the joint experiment, and the D8 joint LR scales
    OVERRIDE the staged frontend scale + are distinct from the B33 P3
    parcel_lr_scale."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    xp = build_v14_experiment(
        bt_root=str(tmp_path), mode="lite", joint_phase=True,
        pool="mean", ssl_mode="joint", latent_depth=3,
        lambda_m4=0.7, m2_predictor_depth=4, m4_predictor_depth=3,
        joint_frontend_lr_scale=0.5, joint_parcel_lr_scale=0.25,
        m2_mask_ratio=0.4, m4_mask_ratio=0.3,
        m2_mask_type="random", m4_mask_type="tube",
        predictor_hidden=256, predictor_n_heads=8,
    )
    assert xp.ssl_mode == "joint"
    assert xp.lambda_m4 == 0.7
    assert xp.m2_predictor_depth == 4 and xp.m4_predictor_depth == 3
    # D8: the joint scales win (the staged frontend scale + P3 parcel scale do
    # NOT apply in joint mode). The joint field is named ``joint_parcel_lr_scale``
    # to stay distinct from V14Phase3Experiment.parcel_lr_scale (the B33 knob).
    assert xp.frontend_lr_scale == 0.5
    assert xp.joint_parcel_lr_scale == 0.25
    # masking + predictor sizing thread through
    assert xp.m2_mask_ratio == 0.4 and xp.m4_mask_ratio == 0.3
    assert xp.m2_mask_type == "random"
    assert xp.predictor_hidden == 256 and xp.predictor_n_heads == 8
    # encoder config carries the mean pool + thin latent depth
    assert xp.brain_model_config.pool == "mean"
    assert xp.brain_model_config.latent_parcel_depth == 3


# --- atlas (DK / DKT) + single-electrode-parcel exclusion (2026-06-13) -------


def test_atlas_default_is_dk_k80(tmp_path) -> None:
    """No-flag default keeps the canonical DK atlas → encoder k_parcels = 80."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    xp = build_v14_experiment(bt_root=str(tmp_path), mode="lite", joint_phase=True)
    assert xp.brain_model_config.k_parcels == 80


def test_atlas_dkt_sets_encoder_k_parcels_74(tmp_path) -> None:
    """--atlas dkt threads anatomy.atlas_spec → encoder k_parcels = 74 (DKT)."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    xp = build_v14_experiment(
        bt_root=str(tmp_path), mode="lite", joint_phase=True, atlas="dkt"
    )
    assert xp.brain_model_config.k_parcels == 74


def test_atlas_and_exclusion_reach_support_and_valid_mask_extractors(tmp_path) -> None:
    """atlas + exclude-single-electrode-parcels thread to BOTH the support and the
    valid-mask extractors with the MATCHED (column, vocabulary) pair, so support /
    valid_mask stay row-aligned (the C1/C2 desync poisoning guard)."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment
    from speech_decoding.extractors.dk_support import V14DKHardSupportExtractor
    from speech_decoding.extractors.valid_mask import ElectrodeValidMask
    from speech_decoding.studies.braintreebank.anatomy import V14_DKT_PARCEL_LABELS

    xp = build_v14_experiment(
        bt_root=str(tmp_path), mode="lite", joint_phase=True,
        atlas="dkt", exclude_single_electrode_parcels=True,
    )
    extractors = xp.data.segmenter.extractors
    support = extractors["support"]
    valid_mask = extractors["valid_mask"]
    assert isinstance(support, V14DKHardSupportExtractor)
    assert isinstance(valid_mask, ElectrodeValidMask)
    for e in (support, valid_mask):
        assert e.parcel_labels == V14_DKT_PARCEL_LABELS
        assert e.label_column == "DKT"
        assert e.exclude_single_electrode_parcels is True


# --- 2STFT dual-band front end (--frontend 2stft) ----------------------------


def test_frontend_default_is_raw_single_grid(tmp_path) -> None:
    """No-flag default keeps the single-grid raw F=50 front end — one batch key,
    per_band_stem False."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    xp = build_v14_experiment(bt_root=str(tmp_path), mode="lite", joint_phase=True)
    assert xp.brain_model_config.per_band_stem is False
    assert "electrode_tokens_high" not in xp.data.segmenter.extractors


def test_frontend_2stft_builds_dual_band_views_and_per_band_stem(tmp_path) -> None:
    """--frontend 2stft builds TWO single-band views (low → electrode_tokens, high
    → electrode_tokens_high) at the locked band configs, and threads per_band_stem
    + view-derived band geometry to the encoder config."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment
    from speech_decoding.extractors.view import MultiStftView

    xp = build_v14_experiment(
        bt_root=str(tmp_path), mode="lite", joint_phase=True,
        frontend="2stft", pool="mean", clip_len=5.0,
    )
    ext = xp.data.segmenter.extractors
    low, high = ext["electrode_tokens"], ext["electrode_tokens_high"]
    assert isinstance(low, MultiStftView) and isinstance(high, MultiStftView)
    # band configs (committed STFT_2BAND_LOW / HIGH)
    assert (low.front_end, low.band_nperseg, low.band_hop) == ("band", 512, 256)
    assert (high.front_end, high.band_nperseg, high.band_hop) == ("band", 128, 64)
    assert low.hop_length == 256 and high.hop_length == 64
    # encoder config: per_band_stem + view-DERIVED bin counts (no drift)
    c = xp.brain_model_config
    assert c.per_band_stem is True
    assert c.band_low_n_freq_bins == 14 and c.band_high_n_freq_bins == 9
    # 5 s @ 2048 Hz: low 1+10240//256=41, high 1+10240//64=161
    assert c.band_low_n_time_bins == 41 and c.band_high_n_time_bins == 161


def test_frontend_2stft_rejects_non_mean_pool(tmp_path) -> None:
    """The dual-band stem is the B37 mean-pool path; cross_attn must fail loud."""
    import pytest

    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    with pytest.raises(ValueError, match="requires --pool mean"):
        build_v14_experiment(
            bt_root=str(tmp_path), mode="lite", joint_phase=True,
            frontend="2stft", pool="cross_attn",
        )


def test_cache_session_index_subsets_study_to_one_pretrain_session(tmp_path) -> None:
    """--cache-session-index N restricts the SSL study to exactly the Nth session
    of the resolved corpus (so a SLURM array builds one session's spec cache per
    task). The subset rides the study's timeline list only — k_parcels and the
    rest of the build are unchanged."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment
    from speech_decoding.studies.braintreebank.study import _SESSIONS_BY_MODE

    xp = build_v14_experiment(
        bt_root=str(tmp_path), mode="nano", joint_phase=True,
        frontend="2stft", pool="mean", atlas="dkt",
        cache_session_index=5,
    )
    # study is a Chain; the BT study is its first step
    bt = xp.data.study.steps[0]
    assert bt.session_subset == (tuple(_SESSIONS_BY_MODE["pretrain"][5]),)
    assert xp.brain_model_config.k_parcels == 74


def test_cache_session_index_out_of_range_raises(tmp_path) -> None:
    """An index past the corpus size fails loud (one source of truth for N)."""
    import pytest

    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    with pytest.raises(ValueError, match="out of range"):
        build_v14_experiment(
            bt_root=str(tmp_path), mode="nano", joint_phase=True,
            frontend="2stft", pool="mean", cache_session_index=99,
        )


def test_cache_only_flags_parse_via_main(monkeypatch, tmp_path) -> None:
    """--cache-only + --cache-session-index are accepted by argparse and reach
    args. (--dry-run short-circuits before the build, so this only proves the
    CLI surface exists; the build-path subsetting is covered above.)"""
    rc = main(
        ["--phase", "1", "--frontend", "2stft", "--cache-only",
         "--cache-session-index", "3", "--dry-run"]
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# SSL-sweep override flags: --ema-tau / --m2-mask-ratio / --m4-mask-ratio
# (None sentinels resolve to the locked constants; non-default values thread
# onto V14JointExperiment → V14JointBrainModule). These knobs feed tonight's GPU
# sweep, so a default-changing regression would silently invalidate the baseline.
# ---------------------------------------------------------------------------


def test_ssl_sweep_defaults_reproduce_locked_values(tmp_path) -> None:
    """Unset --ema-tau / --m{2,4}-mask-ratio (None) resolve to the locked
    constants (0.99925 / 0.50 / 0.20) on the joint experiment — byte-identical
    to the prior hardcoded values."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    xp = build_v14_experiment(
        bt_root=str(tmp_path), mode="lite", joint_phase=True, jepa_phase="p1",
    )
    assert xp.ema_tau == 0.99925
    assert xp.m2_mask_ratio == 0.50
    assert xp.m4_mask_ratio == 0.20


def test_ssl_sweep_overrides_thread_onto_joint_experiment(tmp_path) -> None:
    """Passed --ema-tau / --m{2,4}-mask-ratio reach the V14JointExperiment
    fields the BrainModule reads (EMA schedule + mask samplers)."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    xp = build_v14_experiment(
        bt_root=str(tmp_path), mode="lite", joint_phase=True, jepa_phase="p1",
        ema_tau=0.9999, m2_mask_ratio=0.65, m4_mask_ratio=0.30,
    )
    assert xp.ema_tau == 0.9999
    assert xp.m2_mask_ratio == 0.65
    assert xp.m4_mask_ratio == 0.30


def test_ssl_sweep_out_of_range_raises_at_dispatch(tmp_path) -> None:
    """A bad swept value fails at build (fail-at-dispatch, not hours into a run):
    τ is open (0, 1); held-out ratios are [0, 1)."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    base = dict(bt_root=str(tmp_path), mode="lite", joint_phase=True, jepa_phase="p1")
    for bad in (dict(ema_tau=1.0), dict(ema_tau=0.0)):
        with pytest.raises(ValueError, match="ema_tau"):
            build_v14_experiment(**base, **bad)
    with pytest.raises(ValueError, match="m2_mask_ratio"):
        build_v14_experiment(**base, m2_mask_ratio=1.0)
    with pytest.raises(ValueError, match="m4_mask_ratio"):
        build_v14_experiment(**base, m4_mask_ratio=-0.1)


def test_ssl_sweep_cli_flags_default_none_and_parse(monkeypatch) -> None:
    """The CLI exposes --ema-tau / --m{2,4}-mask-ratio (default None) and a
    passed value reaches build_v14_experiment as a float; unset → None (the
    builder resolves it to the locked constant). Mirrors the example invocation
    ``--phase 1 ... --ema-tau 0.9999 --m2-mask-ratio 0.65``."""
    # default: the three flags arrive at the builder as None (resolved inside)
    calls = _capture_builds(monkeypatch)
    rc = main(["--phase", "1", "--fast-dev-run"])
    assert rc == 0 and len(calls) == 1
    assert calls[0]["ema_tau"] is None
    assert calls[0]["m2_mask_ratio"] is None
    assert calls[0]["m4_mask_ratio"] is None

    # override: the parsed floats reach the builder verbatim
    calls2 = _capture_builds(monkeypatch)
    rc2 = main([
        "--phase", "1", "--fast-dev-run",
        "--ema-tau", "0.9999", "--m2-mask-ratio", "0.65", "--m4-mask-ratio", "0.3",
    ])
    assert rc2 == 0 and len(calls2) == 1
    assert calls2[0]["ema_tau"] == 0.9999
    assert calls2[0]["m2_mask_ratio"] == 0.65
    assert calls2[0]["m4_mask_ratio"] == 0.3


# ---------------------------------------------------------------------------
# #127 M2 band-WIDTH knobs: --m2-time-band-floor / --m2-freq-band-floor. Unlike
# the ratio/τ flags these carry a non-None default (the 6/03 lock: time 2,
# freq 1) and thread through V14JointExperiment → ssl/mask.py::sample_m2_mask.
# Widening the bands (larger floor) is the "1D bands too narrow" sweep lever.
# ---------------------------------------------------------------------------


def test_band_floor_defaults_reproduce_lock(tmp_path) -> None:
    """Unset band floors resolve to the 6/03 lock (time 2 / freq 1) on the
    joint experiment — byte-identical to the prior hardcoded mask grid."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    xp = build_v14_experiment(
        bt_root=str(tmp_path), mode="lite", joint_phase=True, jepa_phase="p1",
    )
    assert xp.m2_time_band_floor == 2
    assert xp.m2_freq_band_floor == 1


def test_band_floor_overrides_thread_onto_joint_experiment(tmp_path) -> None:
    """Wider band floors reach the V14JointExperiment fields sample_m2_mask
    reads, so the swept (harder) mask grid actually takes effect."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    xp = build_v14_experiment(
        bt_root=str(tmp_path), mode="lite", joint_phase=True, jepa_phase="p1",
        m2_time_band_floor=4, m2_freq_band_floor=2,
    )
    assert xp.m2_time_band_floor == 4
    assert xp.m2_freq_band_floor == 2


def test_band_floor_below_one_raises_at_dispatch(tmp_path) -> None:
    """A < 1 band width fails at build (fail-at-dispatch), not by silently
    snapping to a 1-cell band hours into a run."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    base = dict(bt_root=str(tmp_path), mode="lite", joint_phase=True, jepa_phase="p1")
    with pytest.raises(ValueError, match="m2_time_band_floor"):
        build_v14_experiment(**base, m2_time_band_floor=0)
    with pytest.raises(ValueError, match="m2_freq_band_floor"):
        build_v14_experiment(**base, m2_freq_band_floor=0)


def test_band_floor_cli_flags_default_and_parse(monkeypatch) -> None:
    """The CLI exposes --m2-time-band-floor / --m2-freq-band-floor; unset →
    the locked defaults (2 / 1) reach the builder, a passed int reaches it
    verbatim. These are non-None-default ints (cf. the None-sentinel ratio
    flags), so the default itself is asserted."""
    calls = _capture_builds(monkeypatch)
    rc = main(["--phase", "1", "--fast-dev-run"])
    assert rc == 0 and len(calls) == 1
    assert calls[0]["m2_time_band_floor"] == 2
    assert calls[0]["m2_freq_band_floor"] == 1

    calls2 = _capture_builds(monkeypatch)
    rc2 = main([
        "--phase", "1", "--fast-dev-run",
        "--m2-time-band-floor", "4", "--m2-freq-band-floor", "2",
    ])
    assert rc2 == 0 and len(calls2) == 1
    assert calls2[0]["m2_time_band_floor"] == 4
    assert calls2[0]["m2_freq_band_floor"] == 2


# ---------------------------------------------------------------------------
# #180 bad-electrode defense CLI surface: --bad-window-dir (Layer-2 clip filter,
# SSL-only) + --session-z-winsor (Layer-3 read-time cap, cache-neutral env knob).
# ---------------------------------------------------------------------------


def test_bad_window_dir_cli_flag_default_and_parse(monkeypatch) -> None:
    """--bad-window-dir defaults None (no filtering) and a passed path reaches the
    builder verbatim via _common_build_kwargs (so every phase sees it; the builder
    self-gates it to the SSL phases — see test_bad_window_dir_ssl_only_gate)."""
    calls = _capture_builds(monkeypatch)
    rc = main(["--phase", "1", "--fast-dev-run"])
    assert rc == 0 and len(calls) == 1
    assert calls[0]["bad_window_dir"] is None

    calls2 = _capture_builds(monkeypatch)
    rc2 = main(["--phase", "1", "--fast-dev-run", "--bad-window-dir", "/w/bad"])
    assert rc2 == 0 and len(calls2) == 1
    assert calls2[0]["bad_window_dir"] == "/w/bad"


def test_bad_window_dir_ssl_only_gate(tmp_path) -> None:
    """The builder gates the clip filter to SSL phases: a P1 (joint) Data carries
    the dir, a P4 (frozen probe) Data zeroes it to None so the Neuroprobe-Lite
    parity clip set is never altered. DP4: removes rows only, never electrodes."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    p1 = build_v14_experiment(
        bt_root=str(tmp_path), mode="lite", joint_phase=True, jepa_phase="p1",
        bad_window_dir="/w/bad",
    )
    assert p1.data.bad_window_dir == "/w/bad"

    p4 = build_v14_experiment(
        bt_root=str(tmp_path), mode="lite", phase4_frozen_probe=True,
        bad_window_dir="/w/bad",
    )
    assert p4.data.bad_window_dir is None


def test_session_z_winsor_cli_sets_and_clears_env(monkeypatch) -> None:
    """--session-z-winsor is Layer-3 and cache-NEUTRAL: it must set the read-time
    env knob V14_SESSION_Z_WINSOR (which normalize.SessionRobustZNormalizer reads),
    NOT enter the build kwargs (a serialized field would re-fork the spec cache).
    Unset → env popped (no clamp, no stale leak in a warm worker)."""
    import os

    # set: a passed cap reaches the env as a parseable float; absent from build
    calls = _capture_builds(monkeypatch)
    rc = main(["--phase", "1", "--fast-dev-run", "--session-z-winsor", "2500"])
    assert rc == 0 and len(calls) == 1
    assert os.environ["V14_SESSION_Z_WINSOR"] == "2500.0"
    assert float(os.environ["V14_SESSION_Z_WINSOR"]) == 2500.0
    # cache-neutrality contract: NOT threaded as a build/view field
    assert "session_z_winsor" not in calls[0]
    assert "winsor" not in calls[0]

    # unset: env is explicitly popped (no clamp, no warm-worker leak)
    monkeypatch.setenv("V14_SESSION_Z_WINSOR", "999")  # pretend a stale value lingers
    calls2 = _capture_builds(monkeypatch)
    rc2 = main(["--phase", "1", "--fast-dev-run"])
    assert rc2 == 0 and len(calls2) == 1
    assert "V14_SESSION_Z_WINSOR" not in os.environ


def test_b37_pool_mean_auto_resolves_ssl_mode_joint(monkeypatch) -> None:
    """--pool mean with no --ssl-mode resolves to joint and reaches the build
    with pool=mean + ssl_mode=joint (the #108 nano-launch default path)."""
    calls = _capture_builds(monkeypatch)
    rc = main(["--phase", "1", "--pool", "mean", "--fast-dev-run"])
    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["pool"] == "mean"
    assert calls[0]["ssl_mode"] == "joint"
    assert calls[0]["joint_phase"] is True
    # joint LR scales ride along (default 1.0 = no discrimination)
    assert calls[0]["joint_frontend_lr_scale"] == 1.0
    assert calls[0]["joint_parcel_lr_scale"] == 1.0


def test_b37_pool_cross_attn_ssl_phase_rejected(monkeypatch) -> None:
    """The B36 'staged' SSL CLI surface was culled (2026-06-13): 'auto' now
    always resolves to 'joint', and the joint objective is the freq-preserving
    mean-pool path, so an SSL phase on a cross_attn encoder fails fast at
    dispatch (before any build). (cross_attn remains a valid encoder for the P4
    probe, where ssl_mode is irrelevant — see test_b37_phase4_pool_cross_attn.)"""
    _capture_builds(monkeypatch)
    with pytest.raises(SystemExit, match="requires --pool mean"):
        main(["--phase", "1", "--pool", "cross_attn", "--fast-dev-run"])


def test_b37_ssl_mode_joint_requires_mean_pool() -> None:
    """ssl_mode=joint on a cross_attn encoder fails fast at dispatch (the module
    rejects it too, but the dispatch catches it before any build)."""
    with pytest.raises(SystemExit, match="requires --pool mean"):
        main(["--phase", "1", "--ssl-mode", "joint", "--pool", "cross_attn",
              "--fast-dev-run"])


def test_b37_ssl_mode_staged_choice_removed() -> None:
    """The B36 'staged' --ssl-mode choice was culled (2026-06-13): argparse now
    only accepts {auto, joint}, so an explicit --ssl-mode staged is rejected at
    parse time (SystemExit code 2). The mean-pool SSL phase has only the joint
    objective now."""
    with pytest.raises(SystemExit):
        main(["--phase", "1", "--pool", "mean", "--ssl-mode", "staged",
              "--fast-dev-run"])


def test_b37_chain_with_joint_rejected(tmp_path) -> None:
    """--chain --pool mean (→ ssl_mode joint) is rejected: the chain builder
    still uses the staged p1→p2 split and would run the joint objective twice."""
    with pytest.raises(SystemExit, match="not wired yet"):
        main(["--chain", "--pool", "mean", "--work-dir", str(tmp_path),
              "--fast-dev-run"])


def test_b37_phase3_pool_mean_rejected() -> None:
    """--phase 3 --pool mean fails loud at config: the P3 distill student
    readout unpacks a 4-D latent and has no freq-preserving variant for the 5-D
    mean-pool latent yet (else it crashes one batch into the forward)."""
    with pytest.raises(SystemExit, match="phase 3 is not wired"):
        main(["--phase", "3", "--pool", "mean", "--fast-dev-run"])


def test_b37_config_layer_rejects_staged_mean(tmp_path) -> None:
    """The config-layer guard holds regardless of entry point: a programmatic
    build_v14_experiment(pool='mean') that takes the staged ssl_mode default
    must raise at construction (not crash one batch into the forward)."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    with pytest.raises(ValueError, match="only the joint SSL path consumes"):
        build_v14_experiment(
            bt_root=str(tmp_path), mode="lite", joint_phase=True,
            pool="mean", ssl_mode="staged",
        )


def test_b37_joint_honors_time_block_m4_mask(tmp_path) -> None:
    """The 6/03 tube↔cross_time coupling gate is STAGED-only: joint's
    freq-carrying M4 predictor has full visible-context attention, so the
    R-time-block joint sister (--m4-mask-type time_block) builds without the
    H1-leak rejection (matching the module gate + the --m4-mask-type help)."""
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    xp = build_v14_experiment(
        bt_root=str(tmp_path), mode="lite", joint_phase=True,
        pool="mean", ssl_mode="joint", m4_mask_type="time_block",
    )
    assert xp.ssl_mode == "joint"
    assert xp.m4_mask_type == "time_block"


def test_b37_invalid_pool_and_ssl_mode_rejected_by_argparse() -> None:
    """argparse ``choices`` reject typo'd --pool / --ssl-mode so the run record
    never drifts."""
    with pytest.raises(SystemExit):
        main(["--pool", "meanpool", "--dry-run"])
    with pytest.raises(SystemExit):
        main(["--ssl-mode", "jointly", "--dry-run"])


def test_b37_summary_prints_pool_and_ssl_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The dispatch summary records pool + the resolved ssl_mode so the run
    record never silently rides the wrong encoder / SSL objective."""
    rc = main(["--phase", "1", "--pool", "mean", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "pool=mean" in out
    assert "ssl_mode=joint" in out


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
    phases = dv._build_v14_chain(args)
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


def test_chain_guard_off_by_default(monkeypatch, tmp_path) -> None:
    """Never-kill default ([[feedback_never_kill_runs_save_every_500_2026_06_11]]):
    the collapse guard is OFF for EVERY phase by default. No phase auto-kills a run
    — Ben stops diverging runs himself; the monitors still log for post-hoc
    checkpoint selection. P4 is always off (frozen probe, #54 M1). Supersedes the
    pre-run-ops 'P1 always armed by default' contract."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
    ])
    dv._build_v14_chain(args)
    assert [c["collapse_guard"] for c in calls] == [False, False, False, False, False]


def test_chain_collapse_guard_opt_in_arms_ssl_phases(monkeypatch, tmp_path) -> None:
    """--collapse-guard is strictly opt-IN (never-kill directive). When armed it
    enables the guard on all four SSL/distill phases (P1/P2/P3a/P3b); P4 is always
    collapse_guard=False (frozen probe, #54 M1). The old 'P1 specially armed even
    under --no-collapse-guard' asymmetry is gone — P1 now follows the flag like
    every other phase, so a guard-off relaunch keeps a single stable exca UID."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
        "--collapse-guard",
    ])
    dv._build_v14_chain(args)
    # Opt-in arms P1/P2/P3a/P3b; P4 always off.
    assert [c["collapse_guard"] for c in calls] == [True, True, True, True, False]


def test_chain_threads_sister_flags(monkeypatch, tmp_path) -> None:
    """Gate-D HIGH regression: loss_variant reaches the chain's joint phases,
    and binary_tasks reaches EVERY phase (so the SSL/distill clip population
    matches the P4 eval set). Before the _common_build_kwargs refactor these
    silently fell back to defaults while the run summary printed the sister as
    applied. (The --latent-valid-override / --sa-mask-mode CLI sisters were
    culled 2026-06-13; both are now hardcoded to "support" / "bidirectional"
    inside build_v14_experiment, off the dispatch threading surface.)"""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
        "--loss-variant", "b31_plus_m3",
        "--no-binary-tasks",
    ])
    dv._build_v14_chain(args)
    for i in (0, 1):  # the joint SSL phases carry the sister
        assert calls[i]["loss_variant"] == "b31_plus_m3"
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
    dv._build_v14_chain(args)
    # SSL/distill phases: step-budgeted, never early-stop.
    for i in (0, 1, 2, 3):
        assert calls[i]["max_steps"] == 5000
        assert "early_stopping_patience" not in calls[i]
    # P4: early-stops on val_loss at the default patience; ignores --ssl-max-steps.
    assert calls[4]["early_stopping_patience"] == dv.DEFAULT_P4_EARLY_STOP_PATIENCE
    assert "max_steps" not in calls[4]


def test_chain_default_budget_and_patience_disable(monkeypatch, tmp_path) -> None:
    """Default SSL budget = 1500 steps/phase (2026-06-07 production default); a P4
    patience <= 0 disables early-stop and runs P4 to the --n-epochs cap."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
        "--p4-early-stop-patience", "0",
    ])
    dv._build_v14_chain(args)
    for i in (0, 1, 2, 3):
        assert calls[i]["max_steps"] == 1500  # default step budget
    assert calls[4]["early_stopping_patience"] is None  # 0 disables


def test_chain_ssl_max_steps_zero_falls_back_to_epoch_budget(
    monkeypatch, tmp_path,
) -> None:
    """--ssl-max-steps <=0 is the documented epoch-budget escape: max_steps→None
    so the SSL/distill phases fall back to the --n-epochs budget."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
        "--ssl-max-steps", "0",
    ])
    dv._build_v14_chain(args)
    for i in (0, 1, 2, 3):
        assert calls[i]["max_steps"] is None  # <=0 -> epoch budget


def test_chain_default_optim_is_locked_b01_config(monkeypatch, tmp_path) -> None:
    """Production default: with no optim flags every chain phase gets the
    validated config — lr=AUTO √-rule (5e-4·√(eff/1024); default eff = bs 4 ×
    accum 8 × 1 gpu = 32 → 8.8e-5), warmup_cosine, AdamW, β2=0.95, grad_clip=1.0,
    wd=0.05.
    constant-Adam/β2=0.999 was the unimplemented-default *bug*, never a chosen
    config, so it is refused on a real SSL/distill run — see
    test_launch_guard_refuses_real_ssl_on_unimplemented_default_optim."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--pool", "mean", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
    ])
    dv._build_v14_chain(args)
    expected_lr = 5e-4 * (4 * 8 * 1 / 1024) ** 0.5
    for c in calls:  # all 5 phases
        assert c["lr"] == pytest.approx(expected_lr)
        assert c["lr_schedule"] == "warmup_cosine"
        assert c["optimizer_name"] == "AdamW"
        assert c["weight_decay"] == 0.05
        assert c["adam_betas"] == (0.9, 0.95)
        assert c["gradient_clip_val"] == 1.0


def test_resolve_lr_auto_sqrt_rule() -> None:
    """--lr default None → auto √-rule lr = 5e-4·√(eff/1024), eff = bs × accum ×
    gpus_per_node. An explicit --lr overrides. Pure function so the chain, the
    single-phase build, and direct-call tests all resolve identically."""
    import speech_decoding.experiments.dispatch_v14 as dv

    a = _parse(["--phase", "1"])
    assert a.lr is None  # default = auto
    assert dv._resolve_lr(a) == pytest.approx(5e-4 * (4 * 8 * 1 / 1024) ** 0.5)
    # eff scales with gpus_per_node (4-GPU production = eff-128 → 1.76e-4)
    a = _parse(["--phase", "1", "--gpus-per-node", "4"])
    assert dv._resolve_lr(a) == pytest.approx(5e-4 * (4 * 8 * 4 / 1024) ** 0.5)
    # and with bs/accum
    a = _parse(["--phase", "1", "--batch-size", "2",
                "--accumulate-grad-batches", "16"])
    assert dv._resolve_lr(a) == pytest.approx(5e-4 * (32 / 1024) ** 0.5)
    # explicit override wins
    a = _parse(["--phase", "1", "--lr", "3e-4"])
    assert dv._resolve_lr(a) == pytest.approx(3e-4)


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
    # --phase 3 exercises the other branch of `_is_ssl_distill` (the distill
    # phase); --dry-run short-circuits before any build/train there too. P3 is
    # the cross_attn-encoder distill phase (the 5-D mean-pool readout is not
    # wired), so request --pool cross_attn. (The B37 chain SSL path is gated
    # upstream — chain+joint is "not wired yet" — so the chain can no longer
    # reach the dry-run short-circuit; the distill phase is the remaining
    # `_is_ssl_distill` arm whose exemption is observable.)
    assert main(["--phase", "3", "--pool", "cross_attn", "--dry-run"]) == 0


def test_chain_lof_off_by_default(monkeypatch, tmp_path) -> None:
    """#17: no LOF flags → every phase builds with LOF off (cache uid untouched)."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    args = _parse([
        "--chain", "--work-dir", str(tmp_path),
        "--whisper-target-cache-dir", "/c", "--no-target-standardize",
    ])
    dv._build_v14_chain(args)
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
    dv._build_v14_chain(args)
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
    dv._build_v14_chain(_parse(base))
    assert all(c["c_max"] == dv.DEFAULT_C_MAX for c in calls)

    calls.clear()
    dv._build_v14_chain(_parse(base + ["--c-max", "256"]))
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
    dv._build_v14_chain(args)
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
    dv._build_v14_chain(args)
    assert calls and all(c["accumulate_grad_batches"] == 8 for c in calls)


def test_accumulate_grad_batches_default_is_eight(monkeypatch) -> None:
    """No flag → accumulate_grad_batches=8 (2026-06-07 production default: bs 4 ×
    accum 8 × 4 gpu = eff-128)."""
    calls = _capture_builds(monkeypatch)
    main(["--phase", "1", "--warmup-steps", "100"])
    assert calls[0]["accumulate_grad_batches"] == 8


def test_wd_exclude_norms_reaches_every_phase(monkeypatch, tmp_path) -> None:
    """#40: --wd-exclude-norms / --no-wd-exclude-norms threads to ALL chain
    phases via _common_build_kwargs (the no-WD split is uniform across the
    stack). Default is True."""
    import speech_decoding.experiments.dispatch_v14 as dv

    base = ["--chain", "--work-dir", str(tmp_path),
            "--whisper-target-cache-dir", "/c", "--no-target-standardize"]

    calls = _capture_builds(monkeypatch)
    dv._build_v14_chain(_parse(base))
    assert calls and all(c["wd_exclude_norms"] is True for c in calls)

    calls.clear()
    dv._build_v14_chain(_parse(base + ["--no-wd-exclude-norms"]))
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
    dv._build_v14_chain(_parse(base + ["--parcel-lr-scale", "0.25"]))
    # calls[2]=P3a, calls[3]=P3b carry the override.
    assert calls[2]["parcel_lr_scale"] == 0.25
    assert calls[3]["parcel_lr_scale"] == 0.25

    calls.clear()
    dv._build_v14_chain(_parse(base))
    assert calls[2]["parcel_lr_scale"] == 1.0 / 3.0


def test_parcel_lr_scale_reaches_single_phase(monkeypatch) -> None:
    """--parcel-lr-scale reaches a single --phase 3 build too. (P3 is the
    cross_attn distill path; mean is the default now, so request it explicitly.)"""
    calls = _capture_builds(monkeypatch)
    main(["--phase", "3", "--pool", "cross_attn", "--p3-stage", "3b",
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
    dv._build_v14_chain(_parse(base))
    assert calls and all(c["rankme_warn_threshold"] is None for c in calls)
    assert all(c["rankme_alarm_threshold"] is None for c in calls)

    calls.clear()
    dv._build_v14_chain(
        _parse(base + ["--rankme-warn-threshold", "0.4",
                       "--rankme-alarm-threshold", "0.3"]),
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
        "--no-binary-tasks",
        "--warmup-steps", "100",
    ])
    assert len(calls) == 1
    assert calls[0]["loss_variant"] == "b31_plus_utt"
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
    """#39: --ssl-max-steps reaches the single-phase P3 distill build too.
    (P3 is the cross_attn distill path; mean is the default now.)"""
    calls = _capture_builds(monkeypatch)
    main([
        "--phase", "3", "--pool", "cross_attn", "--ssl-max-steps", "5000",
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
    """#39: with no budget flags, P1 gets the default 1500-step budget / no
    early-stop, and a default --phase 4 inherits DEFAULT_P4_EARLY_STOP_PATIENCE
    (=10) and ignores the SSL step budget — the same values the chain applies, so
    single-phase and chain stay in lock-step."""
    import speech_decoding.experiments.dispatch_v14 as dv

    calls = _capture_builds(monkeypatch)
    main(["--phase", "1", "--warmup-steps", "100"])
    assert calls[0]["max_steps"] == 1500
    assert calls[0]["early_stopping_patience"] is None
    calls.clear()
    main(["--phase", "4", "--frozen-probe"])
    assert calls[0]["max_steps"] is None  # P4 ignores the SSL step budget
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
    dv._build_v14_chain(_parse(base))
    assert calls and all(c["requeue"] is False for c in calls)

    calls.clear()
    dv._build_v14_chain(_parse(base + ["--slurm-requeue"]))
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


def test_live_flag_threads_wandb_and_step_cadence() -> None:
    """--live wires the near-free nano-dashboard graphs (loss/val-loss, RankMe,
    LR-curve): a WandbLoggerConfig + per-step LR cadence + log_every_n_steps=1,
    reaching every phase via _common_build_kwargs. Off (default) → no wandb,
    epoch cadence, log_every_n_steps=10 (prior behavior, so the exca cache uid is
    unchanged). reports/nano_dynamics_dashboard_handoff_2026_06_07.md."""
    import speech_decoding.experiments.dispatch_v14 as dv

    live = dv._common_build_kwargs(
        _parse(["--phase", "1", "--mode", "nano", "--live"]),
    )
    assert live["lr_log_interval"] == "step"
    assert live["log_every_n_steps"] == 1
    assert live["wandb_config"] is not None
    assert live["wandb_config"].project == "v14-nano-dynamics"

    off = dv._common_build_kwargs(
        _parse(["--phase", "1", "--mode", "nano"]),
    )
    assert off["lr_log_interval"] == "epoch"
    assert off["log_every_n_steps"] == 10
    assert off["wandb_config"] is None


def test_live_run_name_defaults_and_override() -> None:
    """Overlay legend label defaults to <task>_<mode>_p<phase>; --wandb-run-name /
    --wandb-group override per predict-then-check rung. _build_wandb_config returns
    None unless --live is set."""
    import speech_decoding.experiments.dispatch_v14 as dv

    default = dv._build_wandb_config(_parse(["--phase", "1", "--mode", "nano", "--live"]))
    assert default is not None
    assert default.name == "speech_nano_p1"
    assert default.group == "speech_nano"

    named = dv._build_wandb_config(_parse([
        "--phase", "1", "--mode", "nano", "--live",
        "--wandb-run-name", "p1-lr1e-4", "--wandb-group", "ladderA",
        "--wandb-offline",
    ]))
    assert named is not None
    assert named.name == "p1-lr1e-4" and named.group == "ladderA"
    assert named.offline is True

    assert dv._build_wandb_config(_parse(["--phase", "1", "--mode", "nano"])) is None
