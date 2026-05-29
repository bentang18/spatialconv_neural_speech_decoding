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


def test_multi_stft_view_shape_is_30_freq_9_time_at_1s_2048hz() -> None:
    """FE-01 (B20 v4 lock 2026-05-24): Common hop=256 @ 2048 Hz with
    Nperseg ∈ {1024, 512, 256} and 1-s input yields 9 frames at 8 Hz frame rate
    (1 + 2048/256 with center=True padding). Filterbank flattens to 30 output
    bins. Previous default hop=128 (14.7 Hz, 17 frames) is retired."""
    from speech_decoding.extractors.view import (
        MULTI_STFT_ROUTING,
        _multi_stft_view,
    )

    waveform = torch.zeros(4, 2048)
    out = _multi_stft_view(
        waveform,
        sample_rate=2048,
        hop_length=256,
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
    assert out.shape == (4, 30, 9)


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
        hop_length=256,
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
        hop_length=256,
        nperseg_low=1024,
        nperseg_mid=512,
        nperseg_hi=256,
    )
    assert view.car == "shaft"
    assert view.hop_length == 256
    assert view.nperseg_low == 1024
    assert view.n_fbank_bins == 30
