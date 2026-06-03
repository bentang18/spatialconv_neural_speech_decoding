"""Tests for the v14 I2 STFT view extractor (5/25 default: abs magnitude).

Contract:
- Emit ``(C, F_bin=38, T_bin=17)`` per Ieeg trigger window, NeuralSet time-last
  (``frequency = T_bin / duration = 17 Hz`` for a 1-s window @ 2048 Hz).
- Default ``apply_log=False`` → byte parity with upstream
  ``preprocess_stft(..., "stft_abs")``. ``apply_log=True`` recovers the
  pre-5/25 ``log(|X| + eps)`` behavior.
- STFT spec from Neuroprobe Section D: nperseg=512, poverlap=0.75, hann,
  0-150 Hz, torch.stft center=True.
- Inherit ``CARIeegExtractor`` so v14's R2 shaftCAR + F1 notch + N1 z-score
  preprocessing chain stays in front of the STFT.
"""

from __future__ import annotations

import numpy as np
import torch

from speech_decoding.extractors.view import LogStftView


def test_log_stft_view_inherits_car_ieeg_extractor() -> None:
    """LogStftView is a CARIeegExtractor subclass — keeps R2/F1/N1 chain via config."""
    from speech_decoding.extractors.reference import CARIeegExtractor

    assert issubclass(LogStftView, CARIeegExtractor)


def test_log_stft_view_accepts_v14_recipe_kwargs() -> None:
    """v14 recipe = N1 train-set z-score + R2 shaftCAR + F1 60-Hz notch + I2L log-STFT.
    Pydantic must accept all relevant kwargs in one constructor call."""
    view = LogStftView(
        event_types="Ieeg",
        car="shaft",
        notch_filter=60.0,
        scaler="StandardScaler",
        stft_nperseg=512,
        stft_poverlap=0.75,
        stft_max_freq_hz=150.0,
        stft_min_freq_hz=0.0,
        stft_log_eps=1e-6,
    )
    assert view.car == "shaft"
    assert view.notch_filter == 60.0
    assert view.scaler == "StandardScaler"
    assert view.stft_nperseg == 512
    assert view.stft_poverlap == 0.75
    assert view.stft_max_freq_hz == 150.0
    assert view.stft_log_eps == 1e-6


def test_log_stft_view_helper_matches_upstream_preprocess_stft() -> None:
    """Default (``apply_log=False``) yields byte-exact ``|X|`` per upstream
    ``preprocess_stft(..., 'stft_abs')``. ``apply_log=True`` adds
    ``log(|X| + eps)`` byte-exact."""
    from speech_decoding.extractors.view import _log_stft_view

    torch.manual_seed(0)
    sample_rate = 2048
    duration_s = 1.0
    n_channels = 3
    n_samples = int(sample_rate * duration_s)
    waveform = torch.randn(n_channels, n_samples)

    nperseg = 512
    poverlap = 0.75
    noverlap = int(nperseg * poverlap)
    hop_length = nperseg - noverlap
    window = torch.hann_window(nperseg)
    x_complex = torch.stft(
        waveform,
        n_fft=nperseg,
        hop_length=hop_length,
        win_length=nperseg,
        window=window,
        return_complex=True,
        normalized=False,
        center=True,
    )
    freqs = torch.fft.rfftfreq(nperseg, d=1.0 / sample_rate)
    keep = (freqs >= 0.0) & (freqs <= 150.0)
    x_filtered = x_complex[:, keep]

    out_abs = _log_stft_view(
        waveform,
        sample_rate=sample_rate,
        nperseg=512,
        poverlap=0.75,
        min_freq_hz=0.0,
        max_freq_hz=150.0,
        log_eps=1e-6,
    )
    torch.testing.assert_close(out_abs, torch.abs(x_filtered), rtol=0, atol=0)

    out_log = _log_stft_view(
        waveform,
        sample_rate=sample_rate,
        nperseg=512,
        poverlap=0.75,
        min_freq_hz=0.0,
        max_freq_hz=150.0,
        log_eps=1e-6,
        apply_log=True,
    )
    torch.testing.assert_close(
        out_log, torch.log(torch.abs(x_filtered) + 1e-6), rtol=0, atol=0,
    )


def test_log_stft_view_helper_accepts_probe_window_shorter_than_nperseg() -> None:
    """NeuralSet's ``prepare()`` probes the extractor with a 0.001s window
    (~2 samples @ 2048 Hz). ``torch.stft(center=True)`` reflect-pads by
    ``nperseg//2`` and chokes on inputs that short. The helper must
    zero-pad short inputs so introspection runs without RuntimeError."""
    from speech_decoding.extractors.view import _log_stft_view

    # 135 channels × 2 samples — the literal shape that crashed
    # in the DCC nano smoke (job 46929894).
    waveform = torch.zeros(135, 2)
    out = _log_stft_view(
        waveform,
        sample_rate=2048,
        nperseg=512,
        poverlap=0.75,
        min_freq_hz=0.0,
        max_freq_hz=150.0,
        log_eps=1e-6,
    )
    assert out.shape[0] == 135
    assert out.shape[1] == 38
    assert torch.isfinite(out).all()


def test_log_stft_view_helper_shape_38_freq_17_time() -> None:
    """For 1-s @ 2048 Hz input with nperseg=512, hop=128, center=True:
    F_bin = 38 (bins from 0 to 148 Hz at 4 Hz/bin),
    T_bin = 17 (frames over the centered-padded 2560-sample window)."""
    from speech_decoding.extractors.view import _log_stft_view

    waveform = torch.zeros(4, 2048)
    out = _log_stft_view(
        waveform,
        sample_rate=2048,
        nperseg=512,
        poverlap=0.75,
        min_freq_hz=0.0,
        max_freq_hz=150.0,
        log_eps=1e-6,
    )
    assert out.shape == (4, 38, 17)


def test_log_stft_view_get_timed_array_returns_time_last_neuralset_shape(
    monkeypatch,
) -> None:
    """``_get_timed_array(event, start, duration)`` returns a TimedArray with
    ``data.shape == (C, F_bin=38, T_bin=17)`` and ``frequency = 17 Hz`` for a
    1-s window @ 2048 Hz."""
    from neuralset.base import TimedArray

    view = LogStftView(
        event_types="Ieeg",
        car="shaft",
        notch_filter=60.0,
        stft_nperseg=512,
        stft_poverlap=0.75,
        stft_max_freq_hz=150.0,
        stft_min_freq_hz=0.0,
        stft_log_eps=1e-6,
    )

    fake_waveform_ta = TimedArray(
        frequency=2048.0,
        start=0.0,
        duration=1.0,
        data=np.random.default_rng(0).standard_normal((6, 2048)).astype(np.float32),
    )

    def _fake_super_get_timed_array(self, event, start, duration):  # type: ignore[no-redef]
        return fake_waveform_ta

    from speech_decoding.extractors.reference import CARIeegExtractor

    monkeypatch.setattr(
        CARIeegExtractor, "_get_timed_array", _fake_super_get_timed_array,
        raising=False,
    )

    class _FakeEvent:
        start = 0.0
        duration = 1.0
        frequency = 2048.0

    out = view._get_timed_array(_FakeEvent(), start=0.0, duration=1.0)
    assert isinstance(out, TimedArray)
    assert out.data.shape == (6, 38, 17)
    assert abs(float(out.frequency) - 17.0) < 1e-6
    assert out.start == 0.0


def test_log_stft_view_get_timed_array_preserves_log_stft_values(monkeypatch) -> None:
    """Values returned by ``_get_timed_array`` agree byte-exact with the helper."""
    from neuralset.base import TimedArray
    from speech_decoding.extractors.view import _log_stft_view

    view = LogStftView(
        event_types="Ieeg",
        car="shaft",
        notch_filter=60.0,
        stft_nperseg=512,
        stft_poverlap=0.75,
        stft_max_freq_hz=150.0,
        stft_min_freq_hz=0.0,
        stft_log_eps=1e-6,
    )

    rng = np.random.default_rng(7)
    waveform_np = rng.standard_normal((3, 2048)).astype(np.float32)
    fake_waveform_ta = TimedArray(
        frequency=2048.0, start=0.0, duration=1.0, data=waveform_np,
    )

    def _fake_super_get_timed_array(self, event, start, duration):  # type: ignore[no-redef]
        return fake_waveform_ta

    from speech_decoding.extractors.reference import CARIeegExtractor

    monkeypatch.setattr(
        CARIeegExtractor, "_get_timed_array", _fake_super_get_timed_array,
        raising=False,
    )

    class _FakeEvent:
        start = 0.0
        duration = 1.0
        frequency = 2048.0

    out = view._get_timed_array(_FakeEvent(), start=0.0, duration=1.0)

    expected = _log_stft_view(
        torch.from_numpy(waveform_np),
        sample_rate=2048,
        nperseg=512,
        poverlap=0.75,
        min_freq_hz=0.0,
        max_freq_hz=150.0,
        log_eps=1e-6,
    )
    np.testing.assert_allclose(out.data, expected.numpy(), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Multi-STFT view (T1.5)
# ---------------------------------------------------------------------------


def test_multi_stft_bin_centers_span_1hz_to_813hz() -> None:
    """30 log ⅓-octave bins from 2^0=1 Hz to 2^(29/3)≈813 Hz (5/22 spec)."""
    from speech_decoding.extractors.view import multi_stft_bin_centers_hz

    centers = multi_stft_bin_centers_hz()
    assert centers.shape == (30,)
    assert abs(centers[0].item() - 1.0) < 1e-6
    assert abs(centers[-1].item() - 2 ** (29 / 3)) < 1e-3
    # Monotonic increasing.
    assert (centers[1:] > centers[:-1]).all()


def test_multi_stft_valid_bin_mask_swec_passband_22_bins() -> None:
    """5/19 SWEC audit (0.5–120 Hz): low-edge-overlap criterion gives the
    SWEC-trainable bins — bins k=21 (center 128 Hz) and k=22 (center 161 Hz)
    sit just inside the upper edge by their half-octave skirts."""
    from speech_decoding.extractors.view import multi_stft_valid_bin_mask

    mask = multi_stft_valid_bin_mask(passband_low_hz=0.5, passband_high_hz=120.0)
    assert mask.shape == (30,)
    # The exact cutoff hovers around k=21 / k=22 depending on the threshold
    # convention. Either of these is acceptable; the spec gives k0–k21.
    n_valid = int(mask.sum())
    assert n_valid in (22, 23), f"expected 22–23 valid bins, got {n_valid}"
    assert mask[:22].all(), "first 22 bins must be valid"
    assert not mask[25:].any(), "bins 25+ must be invalid for a 120 Hz cap"


def test_multi_stft_view_shape_is_30_freq_17_time_at_1s_2048hz() -> None:
    """FE-01 (hop=128 re-lock 2026-06-03): Common hop=128 @ 2048 Hz with
    Nperseg ∈ {1024, 512, 256} and 1-s input yields 17 frames at a 16 Hz
    front-end rate (1 + 2048/128 with center=True padding). Filterbank flattens
    to 30 output bins. iMINDBench-standard hop; the old hop=256 default is the
    R-hop-256 sister."""
    from speech_decoding.extractors.view import (
        MULTI_STFT_ROUTING,
        _multi_stft_view,
    )

    waveform = torch.zeros(4, 2048)
    out = _multi_stft_view(
        waveform,
        sample_rate=2048,
        hop_length=128,
        nperseg_low=1024,
        nperseg_mid=512,
        nperseg_hi=256,
        n_bins=30,
        f0_hz=1.0,
        octave_step=1.0 / 3.0,
        half_bw_octaves=0.5,
        routing=MULTI_STFT_ROUTING,
        log_eps=1e-6,
    )
    assert out.shape == (4, 30, 17)


def test_multi_stft_view_routes_tone_to_correct_band() -> None:
    """A pure 100-Hz tone must peak in a filterbank bin centered near 100 Hz
    (k ≈ 20, center ≈ 101.6 Hz). This bin is routed from STFT_mid per the
    5/22 routing — implicitly validates the cross-STFT plumbing."""
    from speech_decoding.extractors.view import (
        MULTI_STFT_ROUTING,
        _multi_stft_view,
        multi_stft_bin_centers_hz,
    )

    sr = 2048
    t = torch.arange(sr).float() / sr
    tone_100hz = torch.sin(2 * torch.pi * 100.0 * t).unsqueeze(0)  # (1, 2048)
    out = _multi_stft_view(
        tone_100hz,
        sample_rate=sr,
        hop_length=128,
        nperseg_low=1024,
        nperseg_mid=512,
        nperseg_hi=256,
        n_bins=30,
        f0_hz=1.0,
        octave_step=1.0 / 3.0,
        half_bw_octaves=0.5,
        routing=MULTI_STFT_ROUTING,
        log_eps=1e-6,
    )
    centers = multi_stft_bin_centers_hz()
    peak_bin = int(out.mean(dim=-1)[0].argmax().item())
    assert abs(centers[peak_bin].item() - 100.0) < 20.0, (
        f"100 Hz tone peaked at bin {peak_bin} (center {centers[peak_bin].item():.1f} Hz)"
    )


def test_multi_stft_view_inherits_car_ieeg_extractor() -> None:
    """MultiStftView keeps the v14 CAR/notch/scaler chain (CARIeegExtractor)."""
    from speech_decoding.extractors.reference import CARIeegExtractor
    from speech_decoding.extractors.view import MultiStftView

    assert issubclass(MultiStftView, CARIeegExtractor)


def test_multi_stft_view_accepts_v14_recipe_kwargs() -> None:
    """Constructor accepts the v14 preprocessing chain alongside Multi-STFT params."""
    from speech_decoding.extractors.view import MultiStftView

    view = MultiStftView(
        event_types="Ieeg",
        car="shaft",
        notch_filter=60.0,
        scaler="StandardScaler",
        hop_length=256,  # explicit override = the R-hop-256 sister (default is 128)
        nperseg_low=1024,
        nperseg_mid=512,
        nperseg_hi=256,
    )
    assert view.car == "shaft"
    assert view.hop_length == 256  # override took (proves hop is settable off-default)
    assert view.nperseg_low == 1024
    assert view.n_fbank_bins == 30


def test_multi_stft_view_n_time_bins_for_duration_maps_to_t_p_40_and_8() -> None:
    """WS-C / C1 (hop=128 re-lock 2026-06-03): phase-conditional ``clip_len`` →
    T_bin → T_p. MultiStftView (hop=128 @ 2048 Hz, ``torch.stft(center=True)``)
    emits ``1 + L // hop`` frames: 5 s → 81, 1 s → 17. These differ from the
    nominal 16 Hz × duration (80 / 16) by the single center-pad frame, but the
    (3,2)-stride patch stem floors both onto the load-bearing patch counts
    **T_p = 40 / 8** (8 Hz latent → matches the 8 Hz Whisper teacher)."""
    from speech_decoding.extractors.view import MultiStftView
    from speech_decoding.models.v14_encoder import _PatchStem

    view = MultiStftView(event_types="Ieeg", car="shaft")
    assert view.n_time_bins_for_duration(5.0) == 81
    assert view.n_time_bins_for_duration(1.0) == 17

    stem = _PatchStem(8)  # default kernel/stride (3, 2)
    assert stem.n_time_patches(81) == 40  # 5 s SSL clip
    assert stem.n_time_patches(17) == 8   # 1 s P4 readout clip
    # The nominal 16 Hz frame counts land on the same T_p (floor-division
    # absorbs the center-pad off-by-one) — so RoPE downward-generalizes (D8).
    assert stem.n_time_patches(80) == 40
    assert stem.n_time_patches(16) == 8


def test_log_stft_view_n_time_bins_for_duration_matches_17_at_1s() -> None:
    """The sister single-STFT view (hop = nperseg·(1−poverlap) = 128 @ 2048 Hz)
    emits 17 frames for a 1-s window — the historical ``DEFAULT_N_TIME_BINS``."""
    view = LogStftView(event_types="Ieeg", car="shaft")
    assert view.n_time_bins_for_duration(1.0) == 17


# ---------------------------------------------------------------------------
# C3 (WS-C, B13): session-level robust-z wired into MultiStftView
# ---------------------------------------------------------------------------


def _make_multi_stft_view(**kw):
    from speech_decoding.extractors.view import MultiStftView

    base = dict(event_types="Ieeg", car="shaft", notch_filter=60.0, apply_log=False)
    base.update(kw)
    return MultiStftView(**base)


def test_c3_session_robust_z_default_off_leaves_output_unchanged(monkeypatch) -> None:
    """Default ``session_robust_z=False`` → the view emits the raw filterbank,
    byte-identical to ``_spec_from_waveform`` (regression guard: the capstone /
    smoke path must be untouched)."""
    from neuralset.base import TimedArray
    from speech_decoding.extractors.reference import CARIeegExtractor

    view = _make_multi_stft_view()  # session_robust_z defaults False
    assert view.session_robust_z is False

    rng = np.random.default_rng(3)
    wf = rng.standard_normal((5, 2048)).astype(np.float32)
    ta = TimedArray(frequency=2048.0, start=0.0, duration=1.0, data=wf)
    monkeypatch.setattr(
        CARIeegExtractor, "_get_timed_array",
        lambda self, e, start, duration: ta, raising=False,
    )

    class _FakeEvent:
        def _splittable_event_uid(self):
            return "sess-A"

    out = view._get_timed_array(_FakeEvent(), start=0.0, duration=1.0)
    expected = view._spec_from_waveform(torch.from_numpy(wf), 2048)
    np.testing.assert_allclose(out.data, expected.numpy(), rtol=0, atol=0)


def test_c3_apply_uses_frozen_session_stats(monkeypatch) -> None:
    """With stats fitted, ``_get_timed_array`` returns the normalizer-transformed
    spec for the clip's session key."""
    from neuralset.base import TimedArray
    from speech_decoding.extractors.normalize import SessionRobustZNormalizer
    from speech_decoding.extractors.reference import CARIeegExtractor

    view = _make_multi_stft_view(session_robust_z=True)

    rng = np.random.default_rng(11)
    recording = rng.standard_normal((4, 8192)).astype(np.float32)
    session_frames = view._spec_from_waveform(torch.from_numpy(recording), 2048)
    norm = SessionRobustZNormalizer().fit(session_frames)
    view._session_stats["sess-A"] = norm
    view._stats_ready = True  # prepare() would set this; here we inject stats directly

    clip = recording[:, :2048]
    ta = TimedArray(frequency=2048.0, start=0.0, duration=1.0, data=clip)
    monkeypatch.setattr(
        CARIeegExtractor, "_get_timed_array",
        lambda self, e, start, duration: ta, raising=False,
    )

    class _FakeEvent:
        def _splittable_event_uid(self):
            return "sess-A"

    out = view._get_timed_array(_FakeEvent(), start=0.0, duration=1.0)
    expected = norm.transform(view._spec_from_waveform(torch.from_numpy(clip), 2048))
    np.testing.assert_allclose(out.data, expected.numpy(), rtol=1e-6, atol=1e-6)


def test_c3_gain_invariance_rho_one(monkeypatch) -> None:
    """C3 spec test: a pure ×k gain on a channel leaves the normalized clip
    output unchanged (median and MAD both scale by k → cancel). Fit + apply the
    full wired path at gain 1 and gain k; outputs must match (ρ=1)."""
    from neuralset.base import TimedArray
    from speech_decoding.extractors.normalize import SessionRobustZNormalizer
    from speech_decoding.extractors.reference import CARIeegExtractor

    rng = np.random.default_rng(23)
    recording = rng.standard_normal((4, 8192)).astype(np.float32)
    gains = np.array([1.0, 7.0, 0.2, 50.0], dtype=np.float32)[:, None]

    def _normalized_clip(scale: float):
        view = _make_multi_stft_view(session_robust_z=True)
        rec = (recording * gains * scale).astype(np.float32)
        frames = view._spec_from_waveform(torch.from_numpy(rec), 2048)
        view._session_stats["s"] = SessionRobustZNormalizer().fit(frames)
        view._stats_ready = True
        clip = rec[:, :2048]
        ta = TimedArray(frequency=2048.0, start=0.0, duration=1.0, data=clip)
        monkeypatch.setattr(
            CARIeegExtractor, "_get_timed_array",
            lambda self, e, start, duration: ta, raising=False,
        )

        class _E:
            def _splittable_event_uid(self):
                return "s"

        return view._get_timed_array(_E(), start=0.0, duration=1.0).data

    base = _normalized_clip(1.0)
    scaled = _normalized_clip(13.0)
    # Per-channel gain already baked into both; the extra ×13 must wash out.
    np.testing.assert_allclose(scaled, base, rtol=1e-4, atol=1e-4)


def test_c3_chunked_fit_matches_whole_recording_fit(monkeypatch) -> None:
    """The PRODUCTION ``_fit_session_robust_z`` chunked path must equal a single
    whole-recording median/MAD fit, on NON-STATIONARY data.

    This routes through the real ``_fit_session_robust_z`` (not an inline
    re-implementation of the concat), on a recording with per-second gain wobble
    + slow offset drift, so a fit that takes the median PER CHUNK and averages
    (instead of concatenating all time then taking one median) diverges sharply
    (≳100% on non-stationary input) and is caught. The only legitimate
    difference vs a single STFT is the handful of ``center=True`` reflect-pad
    frames at each 60-s chunk seam (<0.5% of frames)."""
    import types

    from speech_decoding.extractors.normalize import SessionRobustZNormalizer
    from speech_decoding.extractors.view import MultiStftView

    view = _make_multi_stft_view(session_robust_z=True)  # chunk_s defaults 60.0
    ch = ["e0", "e1", "e2"]
    rng = np.random.default_rng(31)
    n = 2048 * 300  # 300 s → 5 chunks at the 60-s default
    t = (np.arange(n, dtype=np.float32) / 2048.0)
    base = rng.standard_normal((3, n)).astype(np.float32)
    # Strong non-stationarity: slow per-channel gain wobble + offset drift, so
    # per-chunk statistics differ materially from the whole-recording statistic.
    gain = (1.0 + 0.5 * np.sin(2 * np.pi * t / 90.0))[None, :]
    offset = (4.0 * np.tanh(t / 120.0))[None, :]
    rec = (base * gain + offset).astype(np.float32)

    class _E:
        start = 0.0

        def _splittable_event_uid(self):
            return "s"

    def _fake_get_data(self, evs):
        for _e in evs:
            yield types.SimpleNamespace(
                frequency=2048.0, data=rec, ch_names=list(ch),
            )

    monkeypatch.setattr(
        MultiStftView, "_get_data",
        property(lambda self: types.MethodType(_fake_get_data, self)), raising=False,
    )
    monkeypatch.setattr(
        MultiStftView, "_event_types_helper",
        property(lambda self: types.SimpleNamespace(extract=lambda obj: obj)),
        raising=False,
    )
    view._channels.update({c: i for i, c in enumerate(ch)})  # global == session here
    view._fit_session_robust_z([_E()])  # the real chunked path + global scatter
    got = view._session_stats["s"]

    # Independent reference: one STFT over the whole recording, one median/MAD.
    whole = view._spec_from_waveform(torch.from_numpy(rec), 2048)
    ref = SessionRobustZNormalizer(sigma_floor=view.session_z_sigma_floor).fit(whole)

    sw = ref.sigma.squeeze(-1).numpy()
    valid = sw > 1e-3
    np.testing.assert_allclose(
        got.median.squeeze(-1).numpy()[:3][valid],
        ref.median.squeeze(-1).numpy()[valid],
        rtol=0.05, atol=0.05,
    )
    np.testing.assert_allclose(
        got.sigma.squeeze(-1).numpy()[:3][valid], sw[valid], rtol=0.05, atol=0.05,
    )


def test_c3_missing_session_stats_raises(monkeypatch) -> None:
    """Fail loud: ``session_robust_z=True`` but no fitted stats for the clip's
    session → KeyError (never silently emit un-normalized features)."""
    from neuralset.base import TimedArray
    from speech_decoding.extractors.reference import CARIeegExtractor

    view = _make_multi_stft_view(session_robust_z=True)
    view._stats_ready = True  # past prepare; a genuinely-missing session must fail loud
    ta = TimedArray(
        frequency=2048.0, start=0.0, duration=1.0,
        data=np.random.default_rng(0).standard_normal((4, 2048)).astype(np.float32),
    )
    monkeypatch.setattr(
        CARIeegExtractor, "_get_timed_array",
        lambda self, e, start, duration: ta, raising=False,
    )

    class _FakeEvent:
        def _splittable_event_uid(self):
            return "never-fitted"

    import pytest

    with pytest.raises(KeyError, match="never-fitted"):
        view._get_timed_array(_FakeEvent(), start=0.0, duration=1.0)


def test_c3_prepare_fits_per_session(monkeypatch) -> None:
    """``prepare`` fits one normalizer per session AND survives the shape-probe.

    The fake ``super().prepare`` faithfully models ``MneRaw.prepare``'s two jobs
    — (1) populate ``_channels`` from each session's raw, (2) fire the 0.001 s
    shape-probe THROUGH the view's robust-z apply — so this test exercises the
    real ordering instead of a no-op that hid BUG #2 (probe-before-fit KeyError).
    With the ``_stats_ready`` gate the probe must pass the spec through, then the
    fit scatters per-session stats into the global channel index and arms
    ``_stats_ready``."""
    import types

    from speech_decoding.extractors.view import MultiStftView

    view = _make_multi_stft_view(session_robust_z=True, session_z_chunk_s=2.0)

    rng = np.random.default_rng(41)
    ch = ["e0", "e1", "e2"]
    raws = {
        "sess-A": rng.standard_normal((3, 2048 * 6)).astype(np.float32),
        "sess-B": (rng.standard_normal((3, 2048 * 6)) * 9.0).astype(np.float32),
    }

    class _SessEvent:
        start = 0.0

        def __init__(self, key):
            self.key = key

        def _splittable_event_uid(self):
            return self.key

    events = [_SessEvent("sess-A"), _SessEvent("sess-B"), _SessEvent("sess-A")]

    # Fake the session-level cached raw read (now carries ch_names, which the
    # global-index scatter and _update_channels both need) + event extraction.
    def _fake_get_data(self, evs):
        for e in evs:
            yield types.SimpleNamespace(
                frequency=2048.0, start=0.0, duration=6.0,
                data=raws[e.key], ch_names=list(ch),
            )

    monkeypatch.setattr(
        MultiStftView, "_get_data", property(lambda self: types.MethodType(_fake_get_data, self)),
        raising=False,
    )
    monkeypatch.setattr(
        MultiStftView, "_event_types_helper",
        property(lambda self: types.SimpleNamespace(extract=lambda obj: obj)),
        raising=False,
    )

    probe_log: dict = {}

    def _fake_super_prepare(self, obj):
        evs = self._event_types_helper.extract(obj)
        for ta in self._get_data(evs):
            self._update_channels(ta.ch_names)
        # The shape-probe: route a spec through robust-z BEFORE the fit runs.
        probe_log["ready_at_probe"] = self._stats_ready
        probe_log["out"] = self._apply_session_robust_z(
            evs[0], torch.zeros(len(ch), self.n_fbank_bins, 1),
        )

    monkeypatch.setattr(
        MultiStftView.__mro__[1], "prepare", _fake_super_prepare, raising=False,
    )

    view.prepare(events)

    # BUG #2 regression: the probe ran before the fit, with no stats and
    # _stats_ready False, and did NOT raise — it passed the spec through.
    assert probe_log["ready_at_probe"] is False
    assert probe_log["out"] is not None
    # After prepare: stats armed + one normalizer per session.
    assert view._stats_ready is True
    assert set(view._session_stats) == {"sess-A", "sess-B"}
    # Stats were scattered to the global channel index (3 channels here).
    assert view._session_stats["sess-A"].sigma.shape[0] == len(view._channels)
    # B's recording is 9× A's → B's σ ≈ 9× A's (gain shows up in the stats).
    sig_a = view._session_stats["sess-A"].sigma.flatten()
    sig_b = view._session_stats["sess-B"].sigma.flatten()
    # Constant (σ≈0) filterbank bins give 0/0 → drop them before the ratio.
    keep = sig_a > 1e-3
    ratio = (sig_b[keep] / sig_a[keep]).median()
    assert 6.0 < float(ratio) < 13.0


def test_c3_scatter_to_noncontiguous_global_indices(monkeypatch) -> None:
    """BUG #1 regression, hardened against a wrong-index scatter.

    A session with FEWER electrodes than the cohort-global dimension must apply
    (the apply path scatters every clip into a ``(C_global, F, T)`` array; fit-side
    stats are session-indexed). Critically, this session's channels map to
    NON-CONTIGUOUS, NON-IDENTITY global rows (``_get_channels`` from a shuffled
    cohort), so a scatter that used ``range(len)`` or reversed indices instead of
    ``_get_channels`` would place stats on the WRONG electrodes and be caught.

    The expected output is built INDEPENDENTLY — each global row's stats come
    from that channel's own session-indexed median/σ at the row ``_get_channels``
    assigns — NOT by re-running ``norm.transform`` (which would be tautological
    against a wrong-index scatter)."""
    from neuralset.base import TimedArray
    from speech_decoding.extractors.normalize import SessionRobustZNormalizer
    from speech_decoding.extractors.reference import CARIeegExtractor

    view = _make_multi_stft_view(session_robust_z=True)
    view._stats_ready = True
    floor = view.session_z_sigma_floor
    # Cohort-global = 8 rows; this session owns 4 channels at scrambled rows.
    cohort = {"e0": 0, "e9": 1, "e2": 2, "e7": 3, "e1": 4, "e5": 5, "e3": 6, "e8": 7}
    view._channels.update(cohort)
    sess_ch = ["e7", "e2", "e8", "e1"]  # → global rows [3, 2, 7, 4]
    g_idx = view._get_channels(sess_ch)
    assert g_idx == [3, 2, 7, 4]  # non-contiguous, non-identity

    rng = np.random.default_rng(7)
    # Distinct per-channel scale so a mis-routed scatter changes the numbers.
    session_rec = (
        rng.standard_normal((4, 8192)) * np.array([1.0, 5.0, 0.3, 12.0])[:, None]
    ).astype(np.float32)
    session_frames = view._spec_from_waveform(torch.from_numpy(session_rec), 2048)
    norm = SessionRobustZNormalizer().fit(session_frames)
    sess_med = norm.median.clone()  # (4, F, 1), session order — saved BEFORE scatter
    sess_sig = norm.sigma.clone()
    view._scatter_stats_to_global(norm, sess_ch)
    assert norm.median.shape[0] == 8 and norm.sigma.shape[0] == 8
    view._session_stats["sess-A"] = norm

    # Apply path hands a GLOBAL (8, T) clip: this session's channels at g_idx,
    # the other 4 rows are pad zeros.
    clip = np.zeros((8, 2048), dtype=np.float32)
    for i, g in enumerate(g_idx):
        clip[g] = session_rec[i, :2048]
    ta = TimedArray(frequency=2048.0, start=0.0, duration=1.0, data=clip)
    monkeypatch.setattr(
        CARIeegExtractor, "_get_timed_array",
        lambda self, e, start, duration: ta, raising=False,
    )

    class _E:
        def _splittable_event_uid(self):
            return "sess-A"

    out = view._get_timed_array(_E(), start=0.0, duration=1.0)  # must not raise
    assert out.data.shape[0] == 8

    # Independent expected: transform each real row with ITS OWN session stats at
    # the row _get_channels assigned; pad rows stay 0.
    clip_spec = view._spec_from_waveform(torch.from_numpy(clip), 2048)
    expected = torch.zeros_like(clip_spec)
    for i, g in enumerate(g_idx):
        safe = sess_sig[i].clamp(min=floor)
        z = (clip_spec[g] - sess_med[i]) / safe
        z = torch.where(sess_sig[i] >= floor, z, torch.zeros_like(z))
        expected[g] = z
    np.testing.assert_allclose(out.data, expected.numpy(), rtol=1e-6, atol=1e-6)
    # Pad rows (not owned by this session) come out exactly 0.
    pad_rows = [r for r in range(8) if r not in g_idx]
    np.testing.assert_allclose(out.data[pad_rows], 0.0, atol=0)


def test_c3_normalizer_is_robust_median_mad_not_mean_std() -> None:
    """Pin the statistic: the C3 fit must be median + 1.4826·MAD, NOT mean/std.
    On a heavy-tailed signal (a few large outliers) median≠mean and MAD≪std, so a
    mean/std substitution would visibly diverge. Guards the gain-invariance and
    delegation tests above, which a mean/std swap would silently still pass."""
    from speech_decoding.extractors.normalize import SCALE_TO_SIGMA, SessionRobustZNormalizer

    rng = np.random.default_rng(99)
    # (C=2, F=3, T=4000): mostly N(0,1) with a handful of ±60 spikes per bin.
    frames = rng.standard_normal((2, 3, 4000)).astype(np.float32)
    frames[..., :40] = 60.0  # outliers that wreck mean/std but barely move median/MAD
    t = torch.from_numpy(frames)

    norm = SessionRobustZNormalizer().fit(t)

    exp_median = t.median(dim=-1, keepdim=True).values
    exp_sigma = SCALE_TO_SIGMA * (t - exp_median).abs().median(dim=-1, keepdim=True).values
    torch.testing.assert_close(norm.median, exp_median)
    torch.testing.assert_close(norm.sigma, exp_sigma)
    # And it must NOT be mean/std: the outliers pull mean/std far from median/MAD.
    assert (norm.median.abs() < 0.2).all()  # robust center near 0
    assert (t.mean(dim=-1) > 0.4).all()  # mean dragged up by the +60 spikes
    assert (norm.sigma < 2.0 * SCALE_TO_SIGMA).all()  # MAD-σ stays ~O(1)
    assert (t.std(dim=-1) > 5.0).all()  # std blown out by the spikes
