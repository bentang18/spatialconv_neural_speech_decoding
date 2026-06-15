"""Integration tests for ``build_v14_experiment`` after the view wrapper landed.

Replaces the placeholder ``test_v14_dispatch_raises_until_electrode_tokens_extractor_wired``
in ``test_v14_wiring.py`` — that gap is now closed by the default
:class:`speech_decoding.extractors.view.LogStftView` plus
:class:`speech_decoding.extractors.valid_mask.ElectrodeValidMask` plus
``c_max``-padded :class:`V14DKHardSupportExtractor`.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.experiments import dispatch_v14
from speech_decoding.extractors.dk_support import V14DKHardSupportExtractor
from speech_decoding.extractors.valid_mask import ElectrodeValidMask
from speech_decoding.extractors.view import LogStftView, MultiStftView
from speech_decoding.extractors.whisper_target import WhisperTargetExtractor


def test_dispatch_default_wires_multi_stft_view(tmp_path, monkeypatch) -> None:
    """WS-C / C2: without ``electrode_tokens_extractor`` the dispatch now picks
    :class:`MultiStftView` (was ``LogStftView`` pre-B36). Keeps the v14 chain:
    shaft-CAR, per-corpus mains notch, abs-magnitude (``apply_log=False``), and
    the C4 0.5 Hz HPF (``filter=(0.5, None)``)."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")

    extractors = xp.data.segmenter.extractors
    ext = extractors["electrode_tokens"]
    assert isinstance(ext, MultiStftView)
    assert ext.car == "shaft"
    assert ext.notch_filter == 60.0
    assert ext.hop_length == 128  # hop=128 re-lock 2026-06-03 (iMINDBench-standard)
    # FE-RAW-1 (2026-06-04): the default front end is the raw |STFT| F=50 path
    # (the const-Q F=30 filterbank is the demoted front_end="fbank" sister).
    assert ext.front_end == "raw"
    assert ext.apply_log is False
    # C4 HPF: 0.5 Hz high-pass (None upper edge → high-pass only).
    assert ext.filter == (0.5, None)
    # C3: StandardScaler dropped from the default view (robust-z replaces it).
    assert ext.scaler is None


def test_dispatch_lof_off_by_default_leaves_view_unperturbed(tmp_path, monkeypatch) -> None:
    """#17: with LOF off (the default) the Multi-STFT view carries NO lof config —
    ``lof_bad_channels``/``drop_bads`` False, no report path — so the multi-TB STFT
    cache uid is bit-identical to pre-#17."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    ext = dispatch_v14.build_v14_experiment(mode="nano").data.segmenter.extractors[
        "electrode_tokens"
    ]
    assert ext.lof_bad_channels is False
    assert ext.drop_bads is False
    assert ext.lof_report_path is None


def test_dispatch_lof_threshold_inert_when_lof_off(tmp_path, monkeypatch) -> None:
    """#17 cache-uid footgun: a non-default ``lof_threshold`` passed WITHOUT
    enabling LOF must NOT reach the view (else it lands in the cache uid and forces
    a needless multi-TB rebuild while LOF does nothing). The view keeps its
    default threshold."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    ext = dispatch_v14.build_v14_experiment(
        mode="nano", lof_bad_channels=False, lof_threshold=2.0, lof_n_neighbors=42,
    ).data.segmenter.extractors["electrode_tokens"]
    assert ext.lof_bad_channels is False
    assert ext.lof_threshold == 1.5  # view default, NOT the passed 2.0
    assert ext.lof_n_neighbors == 20  # view default, NOT the passed 42


def test_dispatch_lof_on_is_gated_off(tmp_path, monkeypatch) -> None:
    """DP4 (2026-06-09): enabling LOF at dispatch is a hard build error. LOF drops
    electrodes per-trial before the scatter, but support/valid_mask are per-subject
    static, so electrode_tokens would desync from them. The dispatch gates LOF off
    until per-event support/valid_mask plumbing lands. (The view-level LOF
    mechanism itself stays covered by extractors/test_lof_wiring.py.)"""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    report = tmp_path / "lof_drops.json"
    with pytest.raises(ValueError, match="gated OFF"):
        dispatch_v14.build_v14_experiment(
            mode="nano",
            lof_bad_channels=True,
            lof_threshold=1.8,
            lof_n_neighbors=15,
            lof_report_path=str(report),
        )


def test_dispatch_lof_on_gate_fires_before_report_path_check(tmp_path, monkeypatch) -> None:
    """DP4: the gate fires for LOF-on regardless of report_path (it precedes the
    view's report-path validator), so no LOF config can slip a per-trial drop into
    a scored build."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    with pytest.raises(ValueError, match="gated OFF"):
        dispatch_v14.build_v14_experiment(mode="nano", lof_bad_channels=True)


def test_dispatch_warmup_cosine_reaches_experiment_optim(tmp_path, monkeypatch) -> None:
    """#37 (audit follow-up): the warmup_cosine optim config must survive the real
    Experiment construction — not just _build_optim_cfg / stubbed-chain capture. A
    regression in the ``optim=optim_cfg`` hand-off would otherwise go unseen."""
    from speech_decoding.experiments.lr_schedule import WarmupCosine

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", lr_schedule="warmup_cosine", warmup_steps=2000,
        min_lr_ratio=0.0, optimizer_name="AdamW",
    )
    assert isinstance(xp.optim.scheduler, WarmupCosine)
    assert xp.optim.scheduler.warmup_steps == 2000
    assert xp.optim.interval == "step"
    # optimizer is a BaseTorchOptimizer subclass auto-named by class (`name` is
    # the discriminator, resolved from __name__, not a stored attribute).
    assert type(xp.optim.optimizer).__name__ == "AdamW"


def test_dispatch_default_optim_has_no_scheduler(tmp_path, monkeypatch) -> None:
    """#37: the default (constant) build leaves scheduler None — prior plain-Adam."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    assert xp.optim.scheduler is None
    assert type(xp.optim.optimizer).__name__ == "Adam"


def test_dispatch_adamw_does_not_leak_torch_default_weight_decay(
    tmp_path, monkeypatch
) -> None:
    """Re-audit 2026-06-03 (chunk-gate FAIL): torch ``AdamW`` defaults weight_decay
    to 0.01. Building the REAL optimizer for an ``--optimizer AdamW`` run (wd guarded
    to 0) must yield param_groups with weight_decay 0.0 — not the implicit 0.01,
    which would silently decay the §7 no-WD params (biases / LN γβ / freq_embed).
    A cfg-level kwargs check would miss a regression that re-omits the pin, so this
    builds the actual torch optimizer and inspects its param_groups."""
    import torch

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano", optimizer_name="AdamW")
    built = xp.optim.build([torch.nn.Parameter(torch.zeros(2))])
    optim = built["optimizer"] if isinstance(built, dict) else built
    assert type(optim).__name__ == "AdamW"
    assert all(g["weight_decay"] == 0.0 for g in optim.param_groups)


def test_dispatch_c_max_defaults_to_256_across_extractors(tmp_path, monkeypatch) -> None:
    """Default c_max = 256 (2026-06-07 BT-only default; BT max electrodes = 256)
    reaches every c_max-padded extractor. The joint corpus uses --c-max 384."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    exts = dispatch_v14.build_v14_experiment(mode="nano").data.segmenter.extractors
    assert exts["electrode_tokens"].c_max == 256
    assert exts["support"].c_max == 256
    assert exts["valid_mask"].c_max == 256


def test_dispatch_c_max_256_threads_to_all_padded_extractors(tmp_path, monkeypatch) -> None:
    """#35: --c-max 256 must reach ALL four c_max-padded extractors — including the
    joint-phase shaft_mask — or the per-electrode tensors mismatch in the encoder.
    256 is BT's exact safe floor (raw max=256); the build must not raise."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    exts = dispatch_v14.build_v14_experiment(
        mode="nano", c_max=256, joint_phase=True, jepa_phase="p1",
    ).data.segmenter.extractors
    assert exts["electrode_tokens"].c_max == 256
    assert exts["support"].c_max == 256
    assert exts["valid_mask"].c_max == 256
    assert exts["shaft_mask"].c_max == 256  # joint-phase extractor must match


def test_dispatch_omits_whisper_target_by_default(tmp_path, monkeypatch) -> None:
    """WS-H: P1/P2/P4 never carry the 1280-d teacher stream — the extractor is
    built only when a teacher-cache dir is passed (P3)."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    assert "whisper_target" not in xp.data.segmenter.extractors


def test_dispatch_wires_whisper_target_when_cache_dir_set(tmp_path, monkeypatch) -> None:
    """WS-H / T20: ``whisper_target_cache_dir`` inserts a WhisperTargetExtractor
    into the segmenter, with its clip window pinned to the dispatch ``clip_len``
    (5 s SSL → 250 teacher frames) and the locked mean_all merge."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", whisper_target_cache_dir="/cache/bt_teacher",
    )
    ext = xp.data.segmenter.extractors["whisper_target"]
    assert isinstance(ext, WhisperTargetExtractor)
    assert ext.cache_dir == "/cache/bt_teacher"
    assert ext.layer_merge == "mean_all"
    assert ext.clip_s == dispatch_v14.DEFAULT_CLIP_LEN_S  # 5.0 -> n_frames 250
    assert ext.n_frames == 250
    assert ext.event_types == "Word"
    assert ext.aggregation == "trigger"


def test_dispatch_whisper_layer_merge_threads_to_extractor(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", whisper_target_cache_dir="/cache/bt_teacher",
        whisper_layer_merge="8",
    )
    assert xp.data.segmenter.extractors["whisper_target"].layer_merge == "8"


def test_dispatch_default_n_freq_bins_50_gives_encoder_f_p_10(
    tmp_path, monkeypatch,
) -> None:
    """WS-C / C2 + FE-RAW-1 (2026-06-04): ``DEFAULT_N_FREQ_BINS`` flips 30 → 50
    (raw |STFT| front end); the kernel-5-freq-stride patch stem maps F=50 →
    F_p=10 — byte-shape-identical to the old F=30/kernel-3 grid."""
    from speech_decoding.models.v14_encoder import _PatchStem

    assert dispatch_v14.DEFAULT_N_FREQ_BINS == 50
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    assert xp.brain_model_config.n_freq_bins == 50
    assert _PatchStem(8).n_freq_patches(50) == 10  # kernel-5 default


def test_dispatch_clip_len_threads_to_segmenter_duration_and_time_bins(
    tmp_path, monkeypatch,
) -> None:
    """WS-C / C1 (hop=128 re-lock 2026-06-03): phase-conditional ``clip_len``
    sets the segmenter window and sizes the encoder's ``n_time_bins`` (RoPE
    ceiling) from the Multi-STFT frame geometry. 5 s (P1/P2/P3) → T_bin 81 →
    T_p 40; 1 s (P4) → T_bin 17 → T_p 8. (The nominal 16 Hz counts 80/16 floor
    to the same T_p — D8.)"""
    from speech_decoding.models.v14_encoder import _PatchStem

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    stem = _PatchStem(8)

    xp5 = dispatch_v14.build_v14_experiment(mode="nano", clip_len=5.0)
    assert xp5.data.segmenter.duration == 5.0
    assert xp5.brain_model_config.n_time_bins == 81
    assert stem.n_time_patches(xp5.brain_model_config.n_time_bins) == 40

    xp1 = dispatch_v14.build_v14_experiment(mode="nano", clip_len=1.0)
    assert xp1.data.segmenter.duration == 1.0
    assert xp1.brain_model_config.n_time_bins == 17
    assert stem.n_time_patches(xp1.brain_model_config.n_time_bins) == 8


def test_dispatch_default_clip_len_is_5s(tmp_path, monkeypatch) -> None:
    """The default ``clip_len`` is the 5 s SSL window (P1/P2/P3); the P4 driver
    passes ``clip_len=1.0`` for the 1 s readout window."""
    assert dispatch_v14.DEFAULT_CLIP_LEN_S == 5.0
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    assert xp.data.segmenter.duration == 5.0


def test_b36_default_neural_lag_is_zero_1to1_baseline(tmp_path, monkeypatch) -> None:
    """Δlag default = 0.0: the 1:1 stimulus-onset baseline / falsifier null.
    The segmenter clip ``start`` is 0.0 so the neural clip begins exactly at
    the word onset (current behavior)."""
    assert dispatch_v14.DEFAULT_NEURAL_LAG_S == 0.0
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    assert xp.data.segmenter.start == 0.0


def test_b36_neural_lag_threads_to_segmenter_start(tmp_path, monkeypatch) -> None:
    """A non-zero Δlag slides the neural clip ``start`` forward by that lag
    (joint/SSL path) so the lagged response aligns to the stimulus-time
    teacher. The teacher cache is audio-keyed and untouched — a lag sweep is a
    pure re-slice, no recache."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", joint_phase=True, neural_lag_s=0.15,
    )
    assert xp.data.segmenter.start == 0.15
    # duration (clip window) is unchanged — only the start offset moves.
    assert xp.data.segmenter.duration == dispatch_v14.DEFAULT_CLIP_LEN_S


def test_b36_neural_lag_out_of_range_raises(tmp_path, monkeypatch) -> None:
    """Δlag must be causal and bounded: negative (acausal — neural window
    before the stimulus) and > 1 s (unphysical) both raise."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    # match= pins the RANGE guard specifically (not the P4 supervised guard,
    # which also mentions "neural_lag_s").
    with pytest.raises(ValueError, match=r"must lie in \[0\.0, 1\.0\]"):
        dispatch_v14.build_v14_experiment(
            mode="nano", joint_phase=True, neural_lag_s=-0.05,
        )
    with pytest.raises(ValueError, match=r"must lie in \[0\.0, 1\.0\]"):
        dispatch_v14.build_v14_experiment(
            mode="nano", joint_phase=True, neural_lag_s=1.5,
        )


def test_b36_neural_lag_rejected_on_supervised_p4_path(tmp_path, monkeypatch) -> None:
    """A non-zero Δlag on the supervised Phase-4 path raises: it WOULD shift the
    probe window, breaking the leaderboard-parity [onset, onset+1 s] window."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    # match= pins the SUPERVISED-P4 guard specifically ("joint_phase=False"
    # never appears in the range-check message).
    with pytest.raises(ValueError, match=r"joint_phase=False"):
        dispatch_v14.build_v14_experiment(
            mode="nano", joint_phase=False, neural_lag_s=0.15,
        )


def test_b36_neural_lag_rejected_on_frozen_probe_p4_path(tmp_path, monkeypatch) -> None:
    """Gate-D fix: the frozen-probe Phase-4 readout (the canonical P4 path —
    chain P4 + every --resume-from / --frozen-probe run) must ALSO reject a
    non-zero Δlag. Its branch short-circuits the base-P4 guard, so a separate
    guard re-asserts the leaderboard-parity [onset, onset+1 s] window."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    with pytest.raises(ValueError, match=r"frozen-probe path"):
        dispatch_v14.build_v14_experiment(
            # P4 eval is always-lite (the leaderboard cell); pass it so we reach
            # the neural-lag guard rather than the eval-always-lite guard.
            mode="nano", phase4_frozen_probe=True, neural_lag_s=0.15,
            electrode_set="lite",
        )


def test_b36_frozen_probe_p4_accepts_zero_lag(tmp_path, monkeypatch) -> None:
    """The zero-lag frozen probe (the maiden-run config) builds cleanly through
    the new guard."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", phase4_frozen_probe=True, neural_lag_s=0.0, clip_len=1.0,
        electrode_set="lite",  # P4 eval is always-lite (the leaderboard cell)
    )
    assert xp.data.segmenter.start == 0.0
    assert xp.data.segmenter.duration == 1.0


def test_p4_frozen_probe_rejects_non_lite_montage(tmp_path, monkeypatch) -> None:
    """Eval-always-lite invariant (Ben 2026-06-15): the P4 frozen-probe IS the
    Neuroprobe-Lite leaderboard cell, so building it with electrode_set != "lite"
    must fail loud — a full-montage eval CARs Lite contacts against out-of-budget
    shaft-mates, making the result non-reproducible from the Lite budget."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    with pytest.raises(ValueError, match=r"must run electrode_set='lite'"):
        dispatch_v14.build_v14_experiment(
            mode="nano", phase4_frozen_probe=True, neural_lag_s=0.0, clip_len=1.0,
            electrode_set="all",
        )


def test_ssl_path_accepts_all_montage(tmp_path, monkeypatch) -> None:
    """The SSL (pretrain) path runs the full BT montage — electrode_set="all" is
    NOT gated when phase4_frozen_probe is False (the eval-always-lite guard fires
    on the eval path only)."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", joint_phase=True, electrode_set="all",
    )
    # chain = ns.Chain(steps=[study, word_events]); the Wang study is steps[0].
    assert xp.data.study.steps[0].electrode_set == "all"


def test_b36_neural_lag_cli_arg_parses() -> None:
    """The ``--neural-lag-s`` CLI flag exists, defaults to the 0.0 baseline, and
    parses an explicit lag through to ``args.neural_lag_s`` (covers the argparse
    wiring + ``main`` build-call that the builder-level tests above skip)."""
    parser = dispatch_v14._parser()
    assert parser.parse_args([]).neural_lag_s == dispatch_v14.DEFAULT_NEURAL_LAG_S
    assert parser.parse_args([]).neural_lag_s == 0.0
    assert parser.parse_args(["--neural-lag-s", "0.15"]).neural_lag_s == 0.15


def test_b36_neural_lag_inclusive_upper_bound_accepted(tmp_path, monkeypatch) -> None:
    """The range is INCLUSIVE [0.0, 1.0]: the 1.0 s upper edge is accepted and
    threads through to the segmenter (guards a ``<=`` → ``<`` mutation that
    would reject the boundary)."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", joint_phase=True, neural_lag_s=1.0,
    )
    assert xp.data.segmenter.start == 1.0


def test_c4_hpf_0_5hz_removes_dc_and_slow_drift_keeps_passband() -> None:
    """WS-C / C4: the dispatch wires ``filter=(0.5, None)`` → ``raw.filter(0.5,
    None)`` (MNE default FIR HPF). Faithful spec of THAT filter:

      * DC offset → fully removed (the dominant slow-drift concern).
      * deep slow drift (≤ 0.05 Hz) → > 35 dB attenuated; 0.02 Hz → > 50 dB.
      * passband (≥ 1 Hz) → flat within 0.1 dB.

    Note on the plan's "> 40 dB at 0.1 Hz": with the plain ``(0.5, None)`` tuple
    MNE picks an auto transition band whose stopband edge sits at ~0 Hz, so
    0.1 Hz lands inside the transition band (~23 dB), not the stopband. Hitting
    40 dB *at 0.1 Hz* would need a custom ``l_trans_bandwidth`` the MneRaw
    ``filter`` tuple can't express — out of scope for the C4 config. The HPF's
    actual job (kill DC + sub-0.05 Hz drift, preserve the band) is met."""
    import mne
    import numpy as np

    fs = 2048.0
    dur_s = 32.0  # long enough to resolve 0.02 Hz
    n = int(fs * dur_s)
    t = np.arange(n) / fs

    def amp_at(sig: np.ndarray, freq_hz: float) -> float:
        # Single-bin DFT magnitude normalized to sinusoid amplitude.
        w = np.exp(-2j * np.pi * freq_hz * t)
        return 2.0 * np.abs(np.dot(sig, w)) / n

    def atten_db(freq_hz: float) -> float:
        sig = np.sin(2 * np.pi * freq_hz * t)
        flt = mne.filter.filter_data(
            sig.astype(np.float64), sfreq=fs, l_freq=0.5, h_freq=None,
            verbose=False,
        )
        return 20.0 * np.log10(max(amp_at(flt, freq_hz), 1e-12))

    # Deep stopband: slow drift is crushed.
    assert atten_db(0.02) < -50.0
    assert atten_db(0.05) < -35.0
    # Transition band: 0.1 Hz is partially attenuated (not the full stopband).
    assert atten_db(0.1) < -20.0
    # Passband: ≥ 1 Hz preserved within 0.1 dB.
    for f in (1.0, 2.0, 10.0):
        assert abs(atten_db(f)) < 0.1, f"{f} Hz passband not flat"

    # DC offset: a pure constant is removed to numerical zero.
    dc = np.full(n, 5.0)
    dc_filtered = mne.filter.filter_data(
        dc, sfreq=fs, l_freq=0.5, h_freq=None, verbose=False,
    )
    assert np.abs(dc_filtered).max() < 1e-6, "DC offset not removed by HPF"


def test_dispatch_default_wires_valid_mask_and_support_with_c_max(
    tmp_path, monkeypatch,
) -> None:
    """Support and valid-mask both align to ``c_max`` so per-batch collation works."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")

    extractors = xp.data.segmenter.extractors
    assert isinstance(extractors["support"], V14DKHardSupportExtractor)
    assert isinstance(extractors["valid_mask"], ElectrodeValidMask)
    assert extractors["support"].c_max == 256
    assert extractors["valid_mask"].c_max == 256


def test_dispatch_default_sets_x_name_tuple_with_mask(
    tmp_path, monkeypatch,
) -> None:
    """v14 BrainModule x_name unpacks the base 3-tuple plus the
    per-clip metadata kwargs (subject_subtype, ref_idx). The B28
    ``lambda_anat`` soft-gate kwarg was removed at B36 — the hard
    block-diagonal pool consumes the one-hot DK support directly."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")

    assert tuple(xp.x_name) == (
        "electrode_tokens", "support", "valid_mask",
        "subject_subtype", "ref_idx",
    )


def test_b_m2_dispatch_registers_metadata_extractors_in_segmenter(
    tmp_path, monkeypatch,
) -> None:
    """Per-clip metadata extractors are wired to NeuralSet's segmenter
    so the dataloader produces the encoder's B29 Item 11/12 kwargs from
    the actual events."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    extractors = xp.data.segmenter.extractors
    assert "ref_idx" in extractors
    assert "subject_subtype" in extractors
    # B36: no lambda_anat extractor — the hard pool needs no soft anatomy gate.
    assert "lambda_anat" not in extractors


def test_dispatch_default_sets_time_last_input_true(tmp_path, monkeypatch) -> None:
    """The encoder must transpose NS time-last input (B, C, F, T) -> (B, C, T, F)."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")

    assert xp.brain_model_config.time_last_input is True


def test_dispatch_accepts_custom_electrode_tokens_extractor(
    tmp_path, monkeypatch,
) -> None:
    """Caller can still pass an explicit extractor; default is just a default."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    custom = LogStftView(
        event_types="Ieeg",
        car="shaft",
        notch_filter=60.0,
        stft_nperseg=256,  # different from default to confirm caller-override
    )
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", electrode_tokens_extractor=custom,
    )
    assert xp.data.segmenter.extractors["electrode_tokens"] is custom


def test_dispatch_dry_run_no_longer_mentions_missing_extractor(
    capsys, tmp_path, monkeypatch,
) -> None:
    """``--dry-run`` output should no longer claim the extractor is unwired."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    rc = dispatch_v14.main(["--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "not wired yet" not in out
    assert "MultiStftView" in out or "electrode-tokens extractor wired" in out


def test_dk_support_c_max_pads_output(tmp_path) -> None:
    """When ``c_max`` is set, the support tensor pads to (c_max, K=80) zero-rows."""
    import json

    labels_dir = tmp_path / "electrode_labels" / "sub_1"
    labels_dir.mkdir(parents=True)
    (labels_dir / "electrode_labels.json").write_text(json.dumps(["E1", "E2"]))
    csv_dir = tmp_path / "localization" / "sub_1"
    csv_dir.mkdir(parents=True)
    (csv_dir / "depth-wm.csv").write_text(
        "Electrode,DesikanKilliany\nE1,ctx-lh-precentral\nE2,ctx-rh-bankssts\n"
    )

    ext = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=120,
    )
    from types import SimpleNamespace
    out = ext.get_static(SimpleNamespace(subject="1"))  # type: ignore[arg-type]

    assert out.shape == (120, 80)
    assert out[:2].sum().item() == 2.0
    assert out[2:].sum().item() == 0.0


def test_dispatch_default_wires_btwordevents_chain(tmp_path, monkeypatch) -> None:
    """``build_v14_experiment`` wraps Wang2024Treebank + BTWordEvents in an
    ``ns.Chain`` so split-aware Word events flow into the segmenter."""
    import neuralset as ns
    from speech_decoding.studies.braintreebank.study import Wang2024Treebank
    from speech_decoding.studies.braintreebank.word_events import BTWordEvents

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")

    chain = xp.data.study
    assert isinstance(chain, ns.Chain), f"expected ns.Chain, got {type(chain)!r}"
    assert len(chain.steps) == 2
    assert isinstance(chain.steps[0], Wang2024Treebank)
    assert isinstance(chain.steps[1], BTWordEvents)
    assert chain.steps[1].tasks == (dispatch_v14.DEFAULT_TASK,)
    assert chain.steps[1].eval_mode == "CrossSession"
    assert xp.data.segmenter.trigger_query == "type == 'Word'"


def test_dispatch_default_target_pulls_label_from_word(tmp_path, monkeypatch) -> None:
    """Target extractor must read ``label`` off the Word events, not from Ieeg."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")

    target = xp.data.segmenter.extractors["target"]
    assert target.event_types == "Word"
    assert target.event_field == "label"


def test_log_stft_view_pads_to_c_max(monkeypatch) -> None:
    """When ``c_max`` is set, LogStftView pads (C_event, F, T) -> (c_max, F, T)."""
    from neuralset.base import TimedArray
    from speech_decoding.extractors.reference import CARIeegExtractor
    import numpy as np

    rng = np.random.default_rng(0)
    fake_ta = TimedArray(
        frequency=2048.0, start=0.0, duration=1.0,
        data=rng.standard_normal((5, 2048)).astype(np.float32),
    )

    def _fake(self, event, start, duration):
        return fake_ta

    monkeypatch.setattr(CARIeegExtractor, "_get_timed_array", _fake, raising=False)

    view = LogStftView(
        event_types="Ieeg", car="shaft", notch_filter=60.0, c_max=120,
    )

    class _Ev:
        start = 0.0
        duration = 1.0
        frequency = 2048.0

    out = view._get_timed_array(_Ev(), start=0.0, duration=1.0)
    assert out.data.shape == (120, 38, 17)
    # padding rows are exactly zero (log_eps spec from real waveform is finite,
    # so any all-zero row is from padding).
    assert (out.data[5:] == 0.0).all()


def test_encoder_time_last_input_transposes_correctly() -> None:
    """`time_last_input=True` transposes (B, C, F, T) -> (B, C, T, F) so the
    forward body sees the canonical layout regardless of NeuralSet time-last."""
    from speech_decoding.models import V14ParcelPerceiver

    cfg = V14ParcelPerceiver(
        n_freq_bins=3, n_time_bins=5, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=1, m_sub_slots=2,
        patch_kernel_freq=3,  # FE-RAW-1: kernel-3 path for the tiny F=3 config
        time_last_input=True,
    )
    model = cfg.build(n_outputs=2)
    # NS-style time-last input: (B, C, F=3, T=5)
    x = torch.randn(2, 4, 3, 5)
    support = torch.zeros(2, 4, 6)
    support[..., 0] = 1.0
    out = model(x, support)
    assert out.shape == (2, 2)


# --- B28 (2026-05-27 PM) dispatch flags ------------------------------


def test_b28_dispatch_default_dkoleo_mode_is_off(tmp_path, monkeypatch) -> None:
    """B28 default: DKoleo @ M4 is off in dispatch; the 4-term composer fires."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    # The dispatch records its dkoleo selection through the build kwarg; the
    # composer-side wiring is downstream (LOSS-01), so we just verify the
    # function accepts the default and runs end-to-end.
    assert xp is not None


def test_b28_dkoleo_mode_build_kwarg_removed(tmp_path, monkeypatch) -> None:
    """The ``--dkoleo-mode`` CLI selector + the build_v14_experiment kwarg were
    culled 2026-06-13 (B28-demoted). Passing it now raises TypeError; the
    encoder keeps its own ``dkoleo_mode="off"`` internal default."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    import pytest
    with pytest.raises(TypeError):
        dispatch_v14.build_v14_experiment(mode="nano", dkoleo_mode="off")


def test_b28_cross_attn_positions_build_kwarg_removed(tmp_path, monkeypatch) -> None:
    """The ``--cross-attn-positions`` CLI flag + the build_v14_experiment kwarg
    were culled 2026-06-13 (Perceiver, thrown away). Passing it now raises
    TypeError; the encoder keeps its own ``cross_attn_positions`` (→ [0])
    internal default (see test_b28_dispatch_default_cross_attn_positions_is_none)."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    import pytest
    with pytest.raises(TypeError):
        dispatch_v14.build_v14_experiment(
            mode="nano", cross_attn_positions=[0, 3], depth=6,
        )


def test_b28_dispatch_default_cross_attn_positions_is_none(
    tmp_path, monkeypatch,
) -> None:
    """Default: ``cross_attn_positions`` passes through as None → encoder
    constructor's default of [0] applies."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    assert xp.brain_model_config.cross_attn_positions is None


def test_b28_dispatch_swec_corpus_uses_50hz_notch() -> None:
    """``MAINS_NOTCH_BY_CORPUS['swec'] == 50.0`` (Swiss site)."""
    assert dispatch_v14.MAINS_NOTCH_BY_CORPUS["swec"] == 50.0
    # US sites stay at 60 Hz.
    for us_corpus in ("braintreebank", "d_cohort", "ajile12"):
        assert dispatch_v14.MAINS_NOTCH_BY_CORPUS[us_corpus] == 60.0


def test_b28_dispatch_mains_notch_kwarg_overrides_default(
    tmp_path, monkeypatch,
) -> None:
    """SWEC dispatch passes ``mains_notch_hz=50.0`` → the default view gets 50 Hz."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano", mains_notch_hz=50.0)
    ext = xp.data.segmenter.extractors["electrode_tokens"]
    assert ext.notch_filter == 50.0


def test_b28_cli_cross_attn_positions_flag_removed() -> None:
    """`--cross-attn-positions` was culled 2026-06-13 (Perceiver, thrown away);
    argparse now rejects it (SystemExit code 2)."""
    import pytest
    parser = dispatch_v14._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--cross-attn-positions", "0,3"])


def test_b28_cli_dkoleo_mode_flag_removed() -> None:
    """`--dkoleo-mode` was culled 2026-06-13 (B28-demoted); argparse now rejects
    it (SystemExit code 2). The encoder keeps its own ``dkoleo_mode="off"``
    internal default."""
    import pytest
    parser = dispatch_v14._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dkoleo-mode", "intra_clip_slots"])


def test_b28_cli_default_mains_notch_is_60() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert args.mains_notch_hz == 60.0


def test_b28_cli_mains_notch_accepts_swec_50() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args(["--mains-notch-hz", "50.0"])
    assert args.mains_notch_hz == 50.0


def test_c3_dispatch_rejects_corpus_mix_with_missing_notch_entry(
    tmp_path, monkeypatch,
) -> None:
    """Every corpus in ``corpus_mix`` must have a
    ``notch_filter_hz_by_corpus`` entry so the per-corpus extractor
    builds with the right mains frequency."""
    import pytest

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    bad_mix = {"braintreebank": 0.5, "fake_eu_corpus": 0.5}
    with pytest.raises(ValueError, match="notch_filter_hz_by_corpus"):
        dispatch_v14.build_v14_experiment(
            mode="nano",
            corpus_mix=bad_mix,
            # MAINS_NOTCH_BY_CORPUS does NOT contain "fake_eu_corpus"
            # so the validation must trip even with the default map.
        )


def test_c3_dispatch_per_corpus_notch_routes_into_bt_extractor(
    tmp_path, monkeypatch,
) -> None:
    """When ``mains_notch_hz`` is left at default, the BT extractor
    reads its notch from ``notch_filter_hz_by_corpus`` — so an
    override-via-map for BT actually reaches the extractor."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano",
        notch_filter_hz_by_corpus={
            "braintreebank": 50.0,
            "wang2024treebank": 60.0,
            "d_cohort": 60.0,
            "cogan_dcohort": 60.0,
            "ajile12": 60.0,
            "swec": 50.0,
        },
    )
    ext = xp.data.segmenter.extractors["electrode_tokens"]
    assert ext.notch_filter == 50.0


# ---------------------------------------------------------------------------
# B29 lock 2026-05-27 PM-late — dispatch surface
# ---------------------------------------------------------------------------


def test_b29_dispatch_default_m_sub_slots_is_one(tmp_path, monkeypatch) -> None:
    """B29 Item 13: default M=1 (was 4) propagates to brain_model_config."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    assert xp.brain_model_config.m_sub_slots == 1


def test_b29_dispatch_r_m4_slots_sister_restores_m_eq_4(tmp_path, monkeypatch) -> None:
    """Sister R-m4-slots flips ``m_sub_slots`` back to 4 via dispatch."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano", m_sub_slots=4)
    assert xp.brain_model_config.m_sub_slots == 4


def test_b29_dispatch_default_disables_subtype_and_ref_embed(
    tmp_path, monkeypatch,
) -> None:
    """B29 Item 11 + 5/28 PM precedent-audit flip: subtype default OFF.
    B32 5/28 PM-late first-pass-no-input-aug lock: ref_embed default
    also OFF (was ON). vocab=2 (binary) stays."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    cfg = xp.brain_model_config
    assert cfg.subtype_embed_enabled is False
    assert cfg.subtype_embed_reuse_kv is True
    # B32 5/28 PM-late first-pass-no-input-aug lock: ref_embed default OFF.
    assert cfg.ref_embed_enabled is False
    assert cfg.ref_embed_reuse_kv is True
    assert cfg.subtype_vocab == 2  # binary default


def test_b29_dispatch_subtype_three_way_vocab_propagates(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", subtype_embed_vocab="three_way",
    )
    assert xp.brain_model_config.subtype_vocab == 3


def test_b29_dispatch_r_subtype_embed_on_with_kv_reuse_sister(
    tmp_path, monkeypatch,
) -> None:
    """Sister ``R-subtype-embed-on-with-kv-reuse`` P0 (NEW, prior default
    → sister): enabling subtype_embed restores the binary embed + reuse."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", subtype_embed_enabled=True,
    )
    assert xp.brain_model_config.subtype_embed_enabled is True
    assert xp.brain_model_config.subtype_embed_reuse_kv is True


def test_b29_dispatch_r_subtype_embed_input_only_sister(
    tmp_path, monkeypatch,
) -> None:
    """Sister ``R-subtype-embed-input-only`` P0 (PROMOTED M3AE-faithful):
    enabling subtype with reuse_kv=False adds at A1 only."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano",
        subtype_embed_enabled=True,
        subtype_embed_reuse_kv=False,
    )
    assert xp.brain_model_config.subtype_embed_enabled is True
    assert xp.brain_model_config.subtype_embed_reuse_kv is False


def test_b29_dispatch_ref_embed_input_only_sister(tmp_path, monkeypatch) -> None:
    """Post-B32 the default is ``ref_embed_enabled=False``; the input-only
    sister cell now needs callers to opt-in both flags explicitly."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", ref_embed_enabled=True, ref_embed_reuse_kv=False,
    )
    assert xp.brain_model_config.ref_embed_enabled is True
    assert xp.brain_model_config.ref_embed_reuse_kv is False


def test_b29_dispatch_default_corpus_mix_sums_to_one(tmp_path, monkeypatch) -> None:
    """B29 corpus mix default normalizes to sum == 1.0 ± 1e-4."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    # Just calling build_v14_experiment with mode='nano' must not raise the
    # sum-to-1 assertion.
    dispatch_v14.build_v14_experiment(mode="nano")
    # Direct numerical check.
    total = sum(dispatch_v14.DEFAULT_CORPUS_MIX.values())
    assert abs(total - 1.0) < 1e-4


def test_b29_dispatch_corpus_mix_must_sum_to_one(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    bad = {"swec": 0.5, "ajile12": 0.5, "d_cohort": 0.5, "braintreebank": 0.5}
    import pytest

    # include_ajile12=True keeps all four corpora so the sum-to-1 guard fires;
    # with the default (now False) the ajile12-drop renormalizes the remaining
    # three (covered by test_b29_dispatch_corpus_mix_renormalizes_when_ajile12_excluded).
    with pytest.raises(ValueError, match="corpus_mix must sum to 1.0"):
        dispatch_v14.build_v14_experiment(
            mode="nano", corpus_mix=bad, include_ajile12=True,
        )


def test_b29_dispatch_corpus_mix_renormalizes_when_ajile12_excluded(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    # Default corpus_mix includes ajile12; with include_ajile12=False the
    # build re-normalizes over the remaining three corpora and does NOT
    # raise the sum-to-1 assertion.
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", include_ajile12=False,
    )
    assert xp is not None  # build succeeded with renormalized mix


def test_b29_dispatch_ref_operator_alpha_must_be_in_open_unit_interval(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    import pytest

    with pytest.raises(ValueError, match="ref_operator_alpha"):
        dispatch_v14.build_v14_experiment(mode="nano", ref_operator_alpha=0.0)
    with pytest.raises(ValueError, match="ref_operator_alpha"):
        dispatch_v14.build_v14_experiment(mode="nano", ref_operator_alpha=1.0)


def test_b36_dispatch_phase_mode_flag_removed() -> None:
    """The ``--phase-mode`` CLI axis was culled 2026-06-13; argparse no longer
    exposes it (the parsed Namespace has no ``phase_mode``), and passing the flag
    is rejected (SystemExit code 2). The encoder keeps its own ``phase_mode``
    internal default."""
    import pytest
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert not hasattr(args, "phase_mode")
    with pytest.raises(SystemExit):
        parser.parse_args(["--phase-mode", "split_p1_p2"])


def test_dispatch_default_excludes_ajile12() -> None:
    """Default OFF (2026-06-07): the active chain is BT-only. --include-ajile12
    restores the B29 joint mix. (The redundant --no-include-ajile12 store_false
    alias was culled 2026-06-13 — the default is already OFF.)"""
    parser = dispatch_v14._parser()
    assert parser.parse_args([]).include_ajile12 is False
    assert parser.parse_args(["--include-ajile12"]).include_ajile12 is True


def test_b29_dispatch_default_ref_operator_alpha_is_0p3() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert args.ref_operator_alpha == 0.3


def test_b29_dispatch_ffn_variant_flag_removed() -> None:
    """The ``--ffn-variant`` CLI flag + the build_v14_experiment kwarg were
    culled 2026-06-13 (MoE audit-rejected; the dense FFN is hardcoded). argparse
    no longer exposes it and the build kwarg is gone."""
    import pytest
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert not hasattr(args, "ffn_variant")
    with pytest.raises(SystemExit):
        parser.parse_args(["--ffn-variant", "dense"])


def test_b29_dispatch_ffn_variant_build_kwarg_removed(
    tmp_path, monkeypatch,
) -> None:
    """B29 MoE audit 2026-05-28: MoE was rejected, the dense FFN preserved. The
    ``ffn_variant`` build_v14_experiment kwarg was culled 2026-06-13, so passing
    it (e.g. soft_moe_4) now raises TypeError instead of NotImplementedError."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    import pytest

    with pytest.raises(TypeError):
        dispatch_v14.build_v14_experiment(mode="nano", ffn_variant="soft_moe_4")


def test_b29_dispatch_dkoleo_mode_build_kwarg_removed(
    tmp_path, monkeypatch,
) -> None:
    """The ``dkoleo_mode`` build_v14_experiment kwarg (+ its CLI flag and
    back-compat alias map) was culled 2026-06-13 (B28-demoted), so passing it
    now raises TypeError. The encoder keeps its own ``dkoleo_mode="off"``
    internal default."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    import pytest

    with pytest.raises(TypeError):
        dispatch_v14.build_v14_experiment(
            mode="nano", dkoleo_mode="vicreg_slot_variance",
        )


def test_b29_dispatch_invalid_subtype_vocab_rejected(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    import pytest

    with pytest.raises(ValueError, match="subtype_embed_vocab"):
        dispatch_v14.build_v14_experiment(
            mode="nano", subtype_embed_vocab="quaternary",
        )


def test_b29_cli_parses_subtype_three_way_vocab() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args(["--subtype-embed-vocab", "three_way"])
    assert args.subtype_embed_vocab == "three_way"


def test_b29_cli_subtype_embed_flag_enables() -> None:
    """Post 5/28 PM flip: subtype default is OFF; the CLI flag opts in."""
    parser = dispatch_v14._parser()
    args = parser.parse_args(["--subtype-embed"])
    assert args.subtype_embed_enabled is True


def test_b29_cli_subtype_embed_default_is_disabled() -> None:
    """No flag → default OFF per 5/28 PM precedent-audit flip."""
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert args.subtype_embed_enabled is False


def test_b29_cli_no_ref_embed_reuse_kv_flag_disables_reuse() -> None:
    """`--no-ref-embed-reuse-kv` only flips reuse_kv. Post-B32 the
    ref_embed default is OFF, so this flag in isolation does NOT enable
    the embed — callers must combine it with ``--ref-embed`` to land the
    input-only sister."""
    parser = dispatch_v14._parser()
    args = parser.parse_args(["--no-ref-embed-reuse-kv"])
    assert args.ref_embed_reuse_kv is False
    assert args.ref_embed_enabled is False  # B32 default; ``--ref-embed`` not passed


def test_b29_cli_no_include_ajile12_flag_removed() -> None:
    """The redundant ``--no-include-ajile12`` store_false alias was culled
    2026-06-13 (the default is already OFF); argparse now rejects it."""
    import pytest
    parser = dispatch_v14._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--no-include-ajile12"])


# ---------------------------------------------------------------------------
# B31 V-JEPA-2-canonical 2-term lock 2026-05-28 PM-late
# ([[project_v14_b31_vjepa2_canonical_loss_2026_05_28]]). CLI surface +
# dispatch wiring of the ``--loss-variant`` selector.
# ---------------------------------------------------------------------------


def test_b31_cli_loss_variant_default_is_b31_default() -> None:
    """B31 default: ``--loss-variant`` omitted → ``"b31_default"`` (the
    V-JEPA-2-canonical 2-term joint SSL surface)."""
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert args.loss_variant == "b31_default"


@pytest.mark.parametrize(
    "variant",
    ["b31_default", "b31_plus_m3", "b31_plus_utt", "b31_plus_both"],
)
def test_b31_cli_loss_variant_accepts_all_four_arms(variant: str) -> None:
    """All four B31 ``loss_variant`` arms parse without error."""
    parser = dispatch_v14._parser()
    args = parser.parse_args(["--loss-variant", variant])
    assert args.loss_variant == variant


def test_b31_cli_loss_variant_rejects_bogus_value() -> None:
    """argparse rejects an unknown variant (defends the run record
    YAML from drift)."""
    parser = dispatch_v14._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--loss-variant", "bogus"])


def test_b31_dispatch_supervised_phase_rejects_non_default_loss_variant(
    tmp_path, monkeypatch,
) -> None:
    """``loss_variant`` is a joint-phase-only selector. Passing a
    sister value with ``joint_phase=False`` (the default supervised
    Phase-4 path) raises so the run record never silently mis-records
    the sister."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    with pytest.raises(ValueError, match="loss_variant"):
        dispatch_v14.build_v14_experiment(
            mode="nano", loss_variant="b31_plus_utt",
        )


def test_b31_dispatch_supervised_phase_accepts_default_loss_variant(
    tmp_path, monkeypatch,
) -> None:
    """Sanity: supervised Phase-4 path WITH the default
    ``loss_variant`` constructs cleanly."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", loss_variant="b31_default",
    )
    # Supervised path uses vanilla Experiment, not V14JointExperiment.
    assert type(xp).__name__ == "Experiment"


def test_b31_dispatch_joint_phase_propagates_default_loss_variant_to_experiment(
    tmp_path, monkeypatch,
) -> None:
    """Joint-phase dispatch threads the default ``loss_variant`` onto the
    :class:`V14JointExperiment` instance. The brain-model config does NOT
    carry it (its Pydantic schema is ``extra='forbid'``); the
    Experiment-level snapshot records the choice."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", joint_phase=True, loss_variant="b31_default",
    )
    assert type(xp).__name__ == "V14JointExperiment"
    assert xp.loss_variant == "b31_default"
    # The brain-model config must NOT carry ``loss_variant`` (would
    # crash Pydantic ``extra='forbid'``).
    assert "loss_variant" not in xp.brain_model_config.model_dump()


@pytest.mark.parametrize("variant", ["b31_plus_m3", "b31_plus_utt", "b31_plus_both"])
def test_b31_dispatch_joint_phase_quarantines_multiterm_loss_variants(
    tmp_path, monkeypatch, variant: str,
) -> None:
    """B9 (B36 WS-B): the masked-JEPA default is single-term, so the B31
    multi-term ``b31_plus_*`` sisters are quarantined — joint-phase dispatch
    construction raises :class:`NotImplementedError` (the CLI still parses
    them; the run-record gate fires at Experiment construction)."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    with pytest.raises(NotImplementedError, match="single-term"):
        dispatch_v14.build_v14_experiment(
            mode="nano", joint_phase=True, loss_variant=variant,
        )


def test_b31_dispatch_invalid_loss_variant_rejected(
    tmp_path, monkeypatch,
) -> None:
    """``build_v14_experiment`` validates the variant string up front,
    before any Experiment construction."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    with pytest.raises(ValueError, match="loss_variant must be one of"):
        dispatch_v14.build_v14_experiment(
            mode="nano", joint_phase=True, loss_variant="bogus",
        )


# ---------------------------------------------------------------------------
# B36 staged masked-JEPA sub-phase (2026-06-03 H4). ``--jepa-phase`` threads
# ``V14JointExperiment.jepa_phase`` so the parcel-M4 stage (p2) is launchable
# from the entrypoint; previously only the default front-end-M2 stage (p1)
# could run. Mirrors the B31 ``loss_variant`` joint-only-selector pattern.
# ---------------------------------------------------------------------------


def test_b36_dispatch_default_jepa_phase_is_p1() -> None:
    """B36 default: ``--jepa-phase`` omitted → ``"p1"`` (front-end M2)."""
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert args.jepa_phase == "p1"
    assert dispatch_v14.DEFAULT_JEPA_PHASE == "p1"
    assert dispatch_v14.JEPA_PHASES == ("p1", "p2")


def test_b36_dispatch_joint_phase_threads_jepa_phase_to_experiment(
    tmp_path, monkeypatch,
) -> None:
    """Joint-phase dispatch threads ``jepa_phase`` onto the
    :class:`V14JointExperiment` instance so the staged P1->P2 split is
    recorded and the parcel-M4 stage is reachable."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", joint_phase=True, jepa_phase="p2",
    )
    assert type(xp).__name__ == "V14JointExperiment"
    assert xp.jepa_phase == "p2"


def test_b36_dispatch_joint_phase_default_jepa_phase_is_p1(
    tmp_path, monkeypatch,
) -> None:
    """Default joint-phase dispatch lands the front-end-M2 stage (p1)."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano", joint_phase=True)
    assert xp.jepa_phase == "p1"


def test_b36_dispatch_supervised_phase_rejects_non_default_jepa_phase(
    tmp_path, monkeypatch,
) -> None:
    """``jepa_phase`` is a joint-phase-only selector. A non-default value on
    the supervised Phase-4 path (``joint_phase=False``) raises so the run
    record never silently mis-records the stage — same guard as the B30/B31
    sister flags."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    with pytest.raises(ValueError, match="jepa_phase"):
        dispatch_v14.build_v14_experiment(mode="nano", jepa_phase="p2")


def test_b36_dispatch_invalid_jepa_phase_rejected_by_build(
    tmp_path, monkeypatch,
) -> None:
    """``build_v14_experiment`` validates the stage string up front, before
    any Experiment construction (defends a programmatic caller that bypasses
    argparse)."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    with pytest.raises(ValueError, match="jepa_phase"):
        dispatch_v14.build_v14_experiment(
            mode="nano", joint_phase=True, jepa_phase="bogus",
        )


def test_b_bug_4_default_log_stft_view_pins_ref_idx_to_shaft_car(
    tmp_path, monkeypatch,
) -> None:
    """When the dispatch builds the static ``LogStftView(car="shaft")``
    electrode-tokens path, ``RefIdxExtractor`` must collapse to a single
    mode so the ``ref_idx`` label matches the operator the waveform
    actually saw — otherwise the encoder learns to condition on a label
    that lies.
    """
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    ref_idx_ext = xp.data.segmenter.extractors["ref_idx"]
    assert tuple(ref_idx_ext.ref_modes) == ("shaft_car",)


def test_b_bug_4_ref_aug_multi_stft_view_lifts_ref_modes_to_label(
    tmp_path, monkeypatch,
) -> None:
    """When the caller supplies a :class:`RefAugMultiStftView` (REF-aug
    3-cell), the dispatch lifts its ``ref_modes`` onto the label
    extractor so both halves of the REF-aug contract draw from the same
    set.
    """
    from speech_decoding.extractors.ref_aug import RefAugMultiStftView

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    view = RefAugMultiStftView(
        event_types="Ieeg",
        ref_modes=("shaft_car", "bipolar"),
        seed=42,
    )
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", electrode_tokens_extractor=view, seed=7,
    )
    ref_idx_ext = xp.data.segmenter.extractors["ref_idx"]
    assert tuple(ref_idx_ext.ref_modes) == ("shaft_car", "bipolar")
    # ``seed`` is lifted from the view at construction so the shared
    # ``_event_key`` SHA keeps draws aligned regardless of the
    # dispatch's ``seed`` argument.
    assert int(ref_idx_ext.seed) == 42


def test_r3_bug_3_dispatch_rejects_global_car_log_stft_view(
    tmp_path, monkeypatch,
) -> None:
    """A non-RefAugMultiStftView extractor whose CAR config does not
    map cleanly into ``REF_MODES`` would otherwise silently train the
    encoder against a ``ref_idx`` label that lies about the operator.
    Raise instead so the caller picks a supported config (shaft-CAR
    LogStftView) or opts into RefAugMultiStftView.
    """
    import pytest

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    bad_view = LogStftView(
        event_types="Ieeg",
        car="global",
        notch_filter=60.0,
        scaler="StandardScaler",
        channel_order="original",
        c_max=384,
    )
    with pytest.raises(ValueError, match="does not map into REF_MODES"):
        dispatch_v14.build_v14_experiment(
            mode="nano", electrode_tokens_extractor=bad_view,
        )


def test_dispatch_rejects_extractor_without_car_attribute(
    tmp_path, monkeypatch,
) -> None:
    """When the caller hands in an electrode-tokens extractor that
    exposes no ``car`` attribute at all (a future raw-waveform view
    that doesn't inherit ``CARIeegExtractor``), the dispatch must raise
    a clear ValueError naming the extractor class instead of falling
    through with a misleading "does not map into REF_MODES" message.
    """
    import pytest

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))

    class _NoCarExtractor:
        pass

    with pytest.raises(ValueError, match="exposes no 'car' attribute"):
        dispatch_v14.build_v14_experiment(
            mode="nano",
            electrode_tokens_extractor=_NoCarExtractor(),
        )


def test_r3_bug_4_v14_parcel_perceiver_rejects_bogus_ssl_modes() -> None:
    """``dkoleo_mode`` / ``phase_mode`` on :class:`V14ParcelPerceiver` are
    typed as :class:`Literal` so reconstruction from a persisted YAML rejects
    typos at deserialization, rather than silently riding a bogus string
    through to the (absent) SSL trainer. (``anatomy_bias_mode`` was removed
    with the soft routing bias by the B36 hard pool, 2026-06-01.)
    """
    import pytest
    from pydantic import ValidationError

    from speech_decoding.models.v14_encoder import V14ParcelPerceiver

    with pytest.raises(ValidationError):
        V14ParcelPerceiver(
            n_freq_bins=12, n_time_bins=8, k_parcels=80,
            dkoleo_mode="bogus",  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        V14ParcelPerceiver(
            n_freq_bins=12, n_time_bins=8, k_parcels=80,
            phase_mode="p1_only",  # type: ignore[arg-type]
        )


def test_b_cr_3_brain_model_config_keeps_internal_ssl_defaults(
    tmp_path, monkeypatch,
) -> None:
    """The ``--dkoleo-mode`` (B28-demoted) / ``--phase-mode`` (B36) CLI selectors
    and their ``build_v14_experiment`` kwargs were culled 2026-06-13: passing
    either now raises TypeError, and a default build records the
    ``V14ParcelPerceiver`` config's own internal defaults
    (``dkoleo_mode="off"`` / ``phase_mode="joint_b29"``) on the run record.
    (``anatomy_bias_mode`` was removed with the soft routing bias by the B36
    hard pool, 2026-06-01.)
    """
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    import pytest

    with pytest.raises(TypeError):
        dispatch_v14.build_v14_experiment(
            mode="nano", dkoleo_mode="vicreg_slot_variance",
        )
    with pytest.raises(TypeError):
        dispatch_v14.build_v14_experiment(mode="nano", phase_mode="split_p1_p2")

    xp = dispatch_v14.build_v14_experiment(mode="nano")
    cfg = xp.brain_model_config
    assert cfg.dkoleo_mode == "off"
    assert cfg.phase_mode == "joint_b29"


def test_b23_dispatch_joint_phase_wires_shaft_mask_extractor(
    tmp_path, monkeypatch,
) -> None:
    """B2.3 / B03 lock 2026-05-27 PM: the joint SSL dispatch wires a
    :class:`BTShaftMaskExtractor` under the ``shaft_mask`` key. The
    student-only routing (teacher full-input contract) is enforced by
    :class:`V14JointBrainModule._extract_student_kwargs` — this test
    locks the segmenter-side wiring."""
    from speech_decoding.extractors.shaft_mask import BTShaftMaskExtractor

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano", joint_phase=True)
    extractors = xp.data.segmenter.extractors
    assert "shaft_mask" in extractors, (
        "joint phase must wire the BT shaft-mask extractor"
    )
    assert isinstance(extractors["shaft_mask"], BTShaftMaskExtractor)


def test_b23_dispatch_supervised_phase_4_omits_shaft_mask(
    tmp_path, monkeypatch,
) -> None:
    """Phase-4 supervised dispatch does NOT run the joint BrainModule, so
    the shaft-mask extractor stays absent (avoids a no-op extractor +
    cache cost on the downstream classifier path)."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")  # joint_phase=False
    extractors = xp.data.segmenter.extractors
    assert "shaft_mask" not in extractors, (
        "supervised Phase-4 path must NOT wire the shaft-mask extractor"
    )


# B1.5 (task #120, 2026-05-29) two-tier extractor cache wiring.
# ``EXCA_EXTRACTOR_CACHE_FOLDER`` env var or
# ``--extractor-cache-folder`` CLI flag points each extractor's
# ``infra.folder`` at ``{root}/{extractor_name}/`` so outputs survive
# across Experiment-config changes (precision, batch_size, model knobs).
# CLAUDE.md storage tiering: belongs on /work/ (75-day purge,
# regenerate cheap), separate from the Experiment slot on
# /hpc/group/coganlab/.


def test_b15_extractor_cache_default_no_caching(tmp_path, monkeypatch) -> None:
    """Without ``EXCA_EXTRACTOR_CACHE_FOLDER`` or the CLI flag, the
    extractor cache stays off — laptop dry-runs / tests pay zero infra
    overhead and never write to disk."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    monkeypatch.delenv("EXCA_EXTRACTOR_CACHE_FOLDER", raising=False)
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    for name, extractor in xp.data.segmenter.extractors.items():
        if name == "target":
            continue
        if not hasattr(extractor, "infra"):
            continue
        assert extractor.infra.folder is None, (
            f"extractor {name!r} has infra.folder set without explicit "
            f"cache root: {extractor.infra.folder!r}"
        )


def test_b15_extractor_cache_arg_points_each_infra_folder(
    tmp_path, monkeypatch,
) -> None:
    """With ``extractor_cache_folder=...`` set, each extractor's
    ``infra.folder`` points at ``{root}/{name}/`` — independent of
    Experiment-config so the cache survives precision/batch_size
    changes."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    cache_root = tmp_path / "v14_extractors"
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", extractor_cache_folder=str(cache_root),
    )
    extractors = xp.data.segmenter.extractors
    # MapInfra-bearing extractors get their folder set; BaseStatic
    # subclasses (DK support, valid mask) carry no infra and are skipped.
    for name in ("electrode_tokens", "ref_idx", "subject_subtype"):
        ext = extractors[name]
        assert hasattr(ext, "infra"), (
            f"extractor {name!r} unexpectedly missing infra field"
        )
        expected = cache_root / name
        assert str(ext.infra.folder) == str(expected), (
            f"extractor {name!r}: expected infra.folder={expected!r}, "
            f"got {ext.infra.folder!r}"
        )


def test_b15_extractor_cache_env_var_fallback(tmp_path, monkeypatch) -> None:
    """``EXCA_EXTRACTOR_CACHE_FOLDER`` env var picks up when no explicit
    argument is passed — matches the ``scripts/dcc/dispatch`` injection
    path."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    cache_root = tmp_path / "via_env"
    monkeypatch.setenv("EXCA_EXTRACTOR_CACHE_FOLDER", str(cache_root))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    ext = xp.data.segmenter.extractors["electrode_tokens"]
    assert str(ext.infra.folder) == str(cache_root / "electrode_tokens")


def test_b15_extractor_cache_joint_phase_wires_shaft_mask(
    tmp_path, monkeypatch,
) -> None:
    """Joint-phase shaft-mask extractor also gets the cache folder so
    the joint dispatch never silently bypasses caching for one of its
    most expensive extractors."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    cache_root = tmp_path / "joint_cache"
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", joint_phase=True,
        extractor_cache_folder=str(cache_root),
    )
    shaft = xp.data.segmenter.extractors["shaft_mask"]
    if hasattr(shaft, "infra"):
        assert str(shaft.infra.folder) == str(cache_root / "shaft_mask")


# ---------------------------------------------------------------------------
# DP9 — support/valid_mask electrode-config agreement (data-integrity L4).
# These two extractors are joined by bare positional electrode index
# downstream, so a config divergence silently desyncs row identity. The
# pairing-site guard makes that fail loud. Pre-dispatch gate.
# ---------------------------------------------------------------------------


def _dk(**kw):
    base = dict(event_types="Ieeg", bt_root="/x", unmapped_policy="zero", c_max=384)
    base.update(kw)
    return V14DKHardSupportExtractor(**base)


def _vm(**kw):
    base = dict(
        event_types="Ieeg", bt_root="/x", unmapped_policy="zero", c_max=384,
        electrode_set="all",
    )
    base.update(kw)
    return ElectrodeValidMask(**base)


@pytest.mark.must_pass_before_dispatch
def test_dp9_guard_passes_on_matched_config() -> None:
    """The production pairing (identical parcel_labels / unmapped_policy / c_max
    / bt_root / event_types) must pass — the guard is false-positive-free on the
    convention dispatch follows today."""
    dispatch_v14._assert_support_valid_config_agree(_dk(), _vm())  # no raise


@pytest.mark.must_pass_before_dispatch
def test_dp9_guard_raises_on_unmapped_policy_divergence() -> None:
    with pytest.raises(ValueError, match="DP9"):
        dispatch_v14._assert_support_valid_config_agree(
            _dk(unmapped_policy="zero"), _vm(unmapped_policy="raise"),
        )


@pytest.mark.must_pass_before_dispatch
def test_dp9_guard_raises_on_c_max_divergence() -> None:
    with pytest.raises(ValueError, match="DP9"):
        dispatch_v14._assert_support_valid_config_agree(_dk(c_max=384), _vm(c_max=256))


@pytest.mark.must_pass_before_dispatch
def test_dp9_guard_raises_on_parcel_labels_divergence() -> None:
    with pytest.raises(ValueError, match="DP9"):
        dispatch_v14._assert_support_valid_config_agree(
            _dk(), _vm(parcel_labels=("ctx-lh-bankssts",)),
        )


@pytest.mark.must_pass_before_dispatch
def test_dp9_guard_raises_on_label_column_divergence() -> None:
    """Atlas-column-only divergence (one extractor on the DK column, the other on
    DKT) must fail loud. The two K-vocabularies can even share a length, so a
    silent column mismatch would route row ``c`` through different anatomy with no
    error — exactly the C1/C2 desync the guard exists to catch. Pins the
    ``label_column`` axis the DKT swap (#155) introduced. (audit gap, 2026-06-13)"""
    with pytest.raises(ValueError, match="DP9"):
        dispatch_v14._assert_support_valid_config_agree(
            _dk(label_column="DKT"), _vm(label_column="DesikanKilliany"),
        )


@pytest.mark.must_pass_before_dispatch
def test_dp9_guard_raises_on_exclude_single_electrode_divergence() -> None:
    """The single-electrode-parcel drop (#154) zeroes a support row AND flips its
    valid flag; if only ONE extractor excludes, that row's support and valid_mask
    describe different coverage. The shared-config tuple includes the exclusion
    flag, so a one-sided setting must fail loud. (audit gap, 2026-06-13)"""
    with pytest.raises(ValueError, match="DP9"):
        dispatch_v14._assert_support_valid_config_agree(
            _dk(exclude_single_electrode_parcels=True),
            _vm(exclude_single_electrode_parcels=False),
        )


@pytest.mark.must_pass_before_dispatch
def test_dp9_guard_wired_into_build_v14_experiment(tmp_path, monkeypatch) -> None:
    """The guard runs on the real dispatch path — a nano build (matched config)
    succeeds, proving the call site is reached without raising."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    assert xp is not None


# --- CN-3: channel-stats (model, layer_merge) provenance guard ----------------

def _stats_args(path, whisper_layer_merge):
    from types import SimpleNamespace

    return SimpleNamespace(
        target_standardize=True,
        channel_stats_path=str(path),
        whisper_layer_merge=whisper_layer_merge,
    )


@pytest.mark.must_pass_before_dispatch
def test_channel_stats_merge_mismatch_raises(tmp_path) -> None:
    """CN-3: a stats file fit on layer_merge=8 wired to --whisper-layer-merge
    mean_all must fail loud — the 1280-d affine is the same width across merges,
    so a mismatch would otherwise silently z-score the distill target wrong."""
    p = tmp_path / "channel_stats.pt"
    torch.save(
        {"mean": torch.zeros(1280), "inv_std": torch.ones(1280),
         "model": "openai/whisper-large-v3", "layer_merge": 8}, p,
    )
    with pytest.raises(ValueError, match="CN-3"):
        dispatch_v14._validate_channel_stats_path(_stats_args(p, "mean_all"))


@pytest.mark.must_pass_before_dispatch
def test_channel_stats_merge_match_passes(tmp_path) -> None:
    p = tmp_path / "channel_stats.pt"
    torch.save(
        {"mean": torch.zeros(1280), "inv_std": torch.ones(1280),
         "model": "openai/whisper-large-v3", "layer_merge": "mean_all"}, p,
    )
    dispatch_v14._validate_channel_stats_path(_stats_args(p, "mean_all"))  # no raise


@pytest.mark.must_pass_before_dispatch
def test_channel_stats_legacy_no_provenance_is_not_blocked(tmp_path) -> None:
    """A legacy stats file (no model/layer_merge keys) must NOT be invalidated —
    assert-if-present, so existing /work caches keep working (no recompute)."""
    p = tmp_path / "channel_stats.pt"
    torch.save({"mean": torch.zeros(1280), "inv_std": torch.ones(1280)}, p)
    dispatch_v14._validate_channel_stats_path(_stats_args(p, "8"))  # no raise
