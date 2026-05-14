"""Tests for the v14 I2L log-STFT view extractor.

Contract:
- Emit ``(C, F_bin=38, T_bin=17)`` per Ieeg trigger window, NeuralSet time-last
  (``frequency = T_bin / duration = 17 Hz`` for a 1-s window @ 2048 Hz).
- Byte parity with upstream ``preprocess_stft(..., "stft_abs")`` + log(x + eps),
  matching the linear-baseline STFT spec from Neuroprobe Section D
  (nperseg=512, poverlap=0.75, hann, 0-150 Hz, torch.stft center=True).
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
    """The internal STFT helper produces byte-exact agreement with the upstream
    ``preprocess_stft(..., 'stft_abs')`` then ``log(x + eps)`` recipe."""
    from speech_decoding.extractors.view import _log_stft_view

    torch.manual_seed(0)
    sample_rate = 2048
    duration_s = 1.0
    n_channels = 3
    n_samples = int(sample_rate * duration_s)
    waveform = torch.randn(n_channels, n_samples)

    out = _log_stft_view(
        waveform,
        sample_rate=sample_rate,
        nperseg=512,
        poverlap=0.75,
        min_freq_hz=0.0,
        max_freq_hz=150.0,
        log_eps=1e-6,
    )

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
    expected = torch.log(torch.abs(x_filtered) + 1e-6)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)


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
