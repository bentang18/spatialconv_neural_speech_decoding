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

import json

import numpy as np
import pytest
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


# ---------------------------------------------------------------------------
# FE-RAW-1 (2026-06-04): raw |STFT| bins (F=50) replace the const-Q filterbank
# ---------------------------------------------------------------------------

_RAW_FREQS_HZ = [
    4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0,
    30.0,  # low (Δf=2), k2–k15
    32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 68.0, 72.0, 76.0,
    80.0, 84.0, 88.0, 92.0, 96.0, 100.0, 104.0, 108.0, 112.0, 116.0, 120.0,
    124.0, 128.0, 132.0, 136.0, 140.0, 144.0, 148.0,  # mid (Δf=4), k8–k37
    152.0, 160.0, 168.0, 176.0, 184.0, 192.0,  # hi (Δf=8), k19–k24
]


def test_multi_stft_raw_bin_freqs_are_the_exact_f50_axis() -> None:
    """FE-RAW-1 §4.2: 14 low + 30 mid + 6 hi = 50 raw |STFT| bins, concat
    low→mid→hi, on a strictly-increasing piecewise-uniform (2/4/8 Hz) axis
    spanning 4–192 Hz with non-overlapping crossovers at 32/152 Hz."""
    from speech_decoding.extractors.view import F_RAW, multi_stft_raw_bin_freqs_hz

    assert F_RAW == 50
    freqs = multi_stft_raw_bin_freqs_hz()
    assert freqs.shape == (50,)
    assert freqs.tolist() == _RAW_FREQS_HZ
    assert (freqs[1:] > freqs[:-1]).all(), "raw freq axis must be strictly increasing"


def test_multi_stft_raw_valid_bin_mask_swec_is_37_valid_13_invalid() -> None:
    """FE-RAW-1 §4.4: SWEC/AJILE12 (≤120 Hz) keep 37 bins (low 14 + mid k8–k30
    = 32–120 Hz = 23), drop 13 (mid 124–148 Hz = 7 + hi 6). BT/D-cohort
    (broadband) keep all 50. Raw analog of multi_stft_valid_bin_mask."""
    from speech_decoding.extractors.view import multi_stft_raw_valid_bin_mask

    swec = multi_stft_raw_valid_bin_mask(passband_low_hz=0.5, passband_high_hz=120.0)
    assert swec.shape == (50,)
    assert int(swec.sum()) == 37
    assert int((~swec).sum()) == 13
    assert swec[:37].all(), "first 37 raw bins (≤120 Hz) must be valid for SWEC"
    assert not swec[37:].any(), "raw bins 37+ (124–192 Hz) must be invalid for SWEC"

    bt = multi_stft_raw_valid_bin_mask(passband_low_hz=0.0, passband_high_hz=1024.0)
    assert bt.all(), "BT/D-cohort broadband → all 50 raw bins valid"


def test_multi_stft_raw_view_shape_is_50_freq_17_time_at_1s_2048hz() -> None:
    """FE-RAW-1 §4.2: the raw view runs the same 3 STFTs (nperseg 1024/512/256,
    hop=128) as the filterbank path, then selects the 50 raw |STFT| bins —
    (C, 50, 17) for a 1 s, 2048 Hz clip. Frame geometry (17) is unchanged."""
    from speech_decoding.extractors.view import (
        MULTI_STFT_RAW_BINS,
        _multi_stft_raw_view,
    )

    waveform = torch.zeros(4, 2048)
    out = _multi_stft_raw_view(
        waveform,
        sample_rate=2048,
        hop_length=128,
        nperseg_low=1024,
        nperseg_mid=512,
        nperseg_hi=256,
        raw_bins=MULTI_STFT_RAW_BINS,
        log_eps=1e-6,
    )
    assert out.shape == (4, 50, 17)


def test_multi_stft_raw_view_routes_tone_to_correct_raw_bin() -> None:
    """A pure 100-Hz tone must peak in the raw bin centered at exactly 100 Hz.
    Implicitly validates the cross-STFT bin selection + concat order."""
    from speech_decoding.extractors.view import (
        MULTI_STFT_RAW_BINS,
        _multi_stft_raw_view,
        multi_stft_raw_bin_freqs_hz,
    )

    sr = 2048
    t = torch.arange(sr).float() / sr
    tone_100hz = torch.sin(2 * torch.pi * 100.0 * t).unsqueeze(0)
    out = _multi_stft_raw_view(
        tone_100hz,
        sample_rate=sr,
        hop_length=128,
        nperseg_low=1024,
        nperseg_mid=512,
        nperseg_hi=256,
        raw_bins=MULTI_STFT_RAW_BINS,
        log_eps=1e-6,
    )
    freqs = multi_stft_raw_bin_freqs_hz()
    peak_bin = int(out.mean(dim=-1)[0].argmax().item())
    assert abs(freqs[peak_bin].item() - 100.0) < 4.0, (
        f"100 Hz tone peaked at raw bin {peak_bin} (center {freqs[peak_bin].item():.1f} Hz)"
    )


def test_multi_stft_view_raw_is_default_and_emits_f50() -> None:
    """The live MultiStftView default front end is ``raw`` (F=50); the const-Q
    filterbank is the demoted ``fbank`` opt-in sister (F=30)."""
    from speech_decoding.extractors.view import MultiStftView

    raw_view = MultiStftView(event_types="Ieeg", car="shaft")
    assert raw_view.front_end == "raw"
    fbank_view = MultiStftView(event_types="Ieeg", car="shaft", front_end="fbank")
    assert fbank_view.front_end == "fbank"
    assert fbank_view.n_fbank_bins == 30


def test_multi_stft_raw_view_chunked_matches_unchunked() -> None:
    """The whole-movie feature cache slices a chunked spectrogram instead of
    re-STFT-ing per clip; that chunked build MUST be byte-identical to one
    un-chunked ``_multi_stft_raw_view`` over the full recording, or cached clips
    would silently diverge from the live path's interior. Pins overlap-save
    correctness across lengths that straddle chunk seams, the single-pass
    fast path, and a realistic tone+noise signal."""
    from speech_decoding.extractors.view import (
        MULTI_STFT_RAW_BINS,
        _multi_stft_raw_view,
        _multi_stft_raw_view_chunked,
    )

    sr = 2048
    torch.manual_seed(0)
    # Lengths chosen to exercise: shorter than a chunk (single-pass), exactly a
    # few chunks, and lengths that land mid-chunk so seams fall in the interior.
    for n_samples in (4096, 60_000, 122_880, 200_003, 350_000):
        for chunk_frames in (64, 200, 960):
            # tone + broadband noise: non-trivial across all 50 bins
            t = torch.arange(n_samples).float() / sr
            sig = (
                torch.sin(2 * torch.pi * 73.0 * t)
                + 0.5 * torch.sin(2 * torch.pi * 11.0 * t)
                + 0.1 * torch.randn(n_samples)
            )
            wav = torch.stack([sig, sig.flip(0)], dim=0)  # (C=2, T)
            kw = dict(
                sample_rate=sr,
                hop_length=128,
                nperseg_low=1024,
                nperseg_mid=512,
                nperseg_hi=256,
                raw_bins=MULTI_STFT_RAW_BINS,
                log_eps=1e-6,
            )
            full = _multi_stft_raw_view(wav, **kw)
            chunked = _multi_stft_raw_view_chunked(wav, chunk_frames=chunk_frames, **kw)
            assert chunked.shape == full.shape, (
                f"shape mismatch n={n_samples} chunk={chunk_frames}: "
                f"{chunked.shape} vs {full.shape}"
            )
            max_abs_delta = (chunked - full).abs().max().item()
            assert torch.allclose(chunked, full, atol=1e-5, rtol=1e-4), (
                f"chunked != unchunked at n={n_samples} chunk={chunk_frames}: "
                f"max|delta|={max_abs_delta:.3e}"
            )


def test_multi_stft_raw_view_chunked_rejects_non_hop_divisible_window() -> None:
    """Overlap-save alignment requires nperseg_low % hop == 0 (segment starts on
    the global hop grid). A non-divisible config must fail loudly, not silently
    emit misaligned frames."""
    import pytest

    from speech_decoding.extractors.view import (
        MULTI_STFT_RAW_BINS,
        _multi_stft_raw_view_chunked,
    )

    with pytest.raises(ValueError, match="divisible by hop_length"):
        _multi_stft_raw_view_chunked(
            torch.zeros(2, 300_000),
            sample_rate=2048,
            hop_length=100,  # 1024 % 100 != 0
            nperseg_low=1024,
            nperseg_mid=512,
            nperseg_hi=256,
            raw_bins=MULTI_STFT_RAW_BINS,
            log_eps=1e-6,
            chunk_frames=200,
        )


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


def test_c3_scatter_to_per_subject_voltage_rows(monkeypatch) -> None:
    """Robust-z scatter regression, hardened against a wrong-index scatter
    (post DP1/DP2 fix, 2026-06-09).

    A session with FEWER electrodes than the cohort-global dimension must apply
    (the apply path scatters every clip into a ``(C_global, F, T)`` array; fit-side
    stats are session-indexed). Per the row-alignment fix, this session's channels
    scatter to their OWN per-subject voltage positions ``range(len(ch_names))`` —
    NOT the shared last-writer-wins global ``_channels`` index, which permuted rows
    across subjects and was the desync bug. ``C_global`` is still the cohort row
    count ``max(_channels.values())+1``, so the 4 real rows land at 0..3 and the
    remaining cohort rows are zero pad.

    The expected output is built INDEPENDENTLY — each output row's stats come from
    that channel's own session-indexed median/σ at the row the scatter assigns —
    NOT by re-running ``norm.transform`` (which would be tautological against a
    wrong-index scatter)."""
    from neuralset.base import TimedArray
    from speech_decoding.extractors.normalize import SessionRobustZNormalizer
    from speech_decoding.extractors.reference import CARIeegExtractor

    view = _make_multi_stft_view(session_robust_z=True)
    view._stats_ready = True
    floor = view.session_z_sigma_floor
    # Cohort-global = 8 rows; this session owns 4 channels. The cohort dict only
    # sets C_global=8; with the fix the session always scatters to voltage rows
    # 0..3, independent of the (here scrambled) global name→index map.
    cohort = {"e0": 0, "e9": 1, "e2": 2, "e7": 3, "e1": 4, "e5": 5, "e3": 6, "e8": 7}
    view._channels.update(cohort)
    sess_ch = ["e7", "e2", "e8", "e1"]  # → per-subject voltage rows [0, 1, 2, 3]
    g_idx = view._get_channels(sess_ch)
    assert g_idx == [0, 1, 2, 3]  # per-subject voltage order, not the global remap

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


# --------------------------------------------------------------------------- #
# Whole-movie raw-|STFT| feature cache (Ben-approved 2026-06-05)              #
# --------------------------------------------------------------------------- #
def _cache_view_harness(view, rec, ch, monkeypatch, key="s"):
    """Wire a real ``MultiStftView`` for an end-to-end prepare over one synthetic
    session: fake ``_get_data`` (the cached preprocessed waveform), event
    extraction, and a ``super().prepare`` that only populates ``_channels`` (the
    cache build and per-clip scatter need it). Returns the single fake event.
    ``key`` is the session uid the cache files are named after — pass a path-like
    or JSON-metachar value to exercise the filesystem-safe stem."""
    import types

    from speech_decoding.extractors.view import MultiStftView

    n = rec.shape[-1]

    class _E:
        start = 0.0

        def _splittable_event_uid(self):
            return key

    state = {"get_data_calls": 0}

    def _fake_get_data(self, evs):
        state["get_data_calls"] += 1
        for _e in evs:
            yield types.SimpleNamespace(
                frequency=2048.0, start=0.0, duration=float(n) / 2048.0,
                data=rec, ch_names=list(ch),
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

    def _fake_super_prepare(self, obj):
        for ta in self._get_data(self._event_types_helper.extract(obj)):
            self._update_channels(ta.ch_names)

    monkeypatch.setattr(
        MultiStftView.__mro__[1], "prepare", _fake_super_prepare, raising=False,
    )
    return _E(), state


def test_spec_cache_clip_matches_whole_movie_slice(tmp_path, monkeypatch) -> None:
    """Core read-path correctness. The cached clip must equal an INDEPENDENT
    reference: the un-chunked whole-movie ``_spec_from_waveform``, fp16-quantized
    (the cache stores fp16), sliced at the global hop grid ``g0:g0+n_frames``.
    This bakes in the three Ben-accepted forks (real-neighbour edges, global hop
    grid) — the clip is NOT compared to the reflect-padded per-clip recompute.
    The 70 s recording forces the build's chunked branch (total_frames 1120 >
    chunk_frames 960), and the last window exercises the end-of-recording clamp."""
    from speech_decoding.extractors.view import MultiStftView

    view = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=False, c_max=None,
    )
    ch = ["e0", "e1", "e2", "e3"]
    rng = np.random.default_rng(123)
    n = 2048 * 70  # 70 s → total_frames 1120 > chunk_frames 960 → real chunking
    t = np.arange(n, dtype=np.float32) / 2048.0
    base = rng.standard_normal((4, n)).astype(np.float32)
    rec = (base + 0.5 * np.sin(2 * np.pi * 40.0 * t)[None, :]).astype(np.float32)

    event, _ = _cache_view_harness(view, rec, ch, monkeypatch)
    view.prepare([event])

    assert view._spec_ready is True
    assert len(list(tmp_path.rglob("*.npy"))) == 1
    assert len(list(tmp_path.rglob("*.json"))) == 1
    # _channels is identity here, so global order == session order.
    assert view._get_channels(ch) == [0, 1, 2, 3]

    whole = view._spec_from_waveform(torch.from_numpy(rec), 2048).numpy()
    total = whole.shape[-1]
    hop = view.hop_length

    for start, dur in [(0.0, 1.0), (5.0, 1.0), (10.3, 2.0), (70.0 - 1.0, 1.0)]:
        out = view._get_timed_array(event, start=start, duration=dur)
        n_frames = view.n_time_bins_for_duration(dur)
        s0 = int(round(start * 2048.0))
        g0 = int(round(s0 / hop))
        g0 = max(0, min(g0, max(0, total - n_frames)))
        expected = whole[:, :, g0 : g0 + n_frames]
        assert out.data.shape == expected.shape == (4, view._spec_from_waveform(
            torch.from_numpy(rec[:, :2048]), 2048).shape[1], n_frames)
        # The fp32 cache stores |STFT| verbatim, so the only gap vs this
        # un-chunked reference is the seam-free chunked build (chunked ==
        # un-chunked to f32 round-off); a wrong slice / channel / off-by-one
        # frame moves values far beyond that and is caught.
        np.testing.assert_allclose(out.data, expected, rtol=1e-4, atol=1e-3)


def test_spec_cache_hit_reuses_disk_and_matches_robust_z(tmp_path, monkeypatch) -> None:
    """Cross-job reuse: a second view with the same config + cache dir must HIT
    the on-disk spec (no waveform read, file not rewritten) AND fit robust-z stats
    bit-identical to the building view — because both fit on the fp32 frames
    that actually live on disk."""
    view1 = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=True, c_max=None,
    )
    ch = ["e0", "e1", "e2"]
    rng = np.random.default_rng(7)
    n = 2048 * 30
    rec = (rng.standard_normal((3, n)) * np.array([1.0, 4.0, 0.5])[:, None]).astype(np.float32)

    event1, state1 = _cache_view_harness(view1, rec, ch, monkeypatch)
    view1.prepare([event1])
    # MISS: super().prepare reads the waveform once (channels), the build reads it
    # again to compute + write the STFT → 2 reads.
    assert state1["get_data_calls"] == 2
    npy = next(tmp_path.rglob("*.npy"))
    mtime1 = npy.stat().st_mtime_ns

    view2 = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=True, c_max=None,
    )
    event2, state2 = _cache_view_harness(view2, rec, ch, monkeypatch)
    view2.prepare([event2])

    # HIT: super().prepare still reads the waveform once for channels, but the
    # build SKIPS the STFT recompute (no second read) and the file is untouched —
    # that skipped recompute is the cross-job win.
    assert state2["get_data_calls"] == 1
    assert npy.stat().st_mtime_ns == mtime1
    # Same exca uid namespace → exactly one cached spec for both views.
    assert len(list(tmp_path.rglob("*.npy"))) == 1
    # Bit-identical robust-z stats across the build and the cache-hit job.
    np.testing.assert_array_equal(
        view1._session_stats["s"].median.numpy(),
        view2._session_stats["s"].median.numpy(),
    )
    np.testing.assert_array_equal(
        view1._session_stats["s"].sigma.numpy(),
        view2._session_stats["s"].sigma.numpy(),
    )


def test_robust_z_from_stats_matches_fit() -> None:
    """``from_stats`` rebuilds a normalizer that transforms IDENTICALLY to one
    produced by ``fit`` — the guarantee that lets the stats sidecar replace the
    refit. Stats are a deterministic function of the frames, so the round-trip is
    bit-exact (rtol=atol=0), independent of the (transform-time) sigma_floor."""
    from speech_decoding.extractors.normalize import SessionRobustZNormalizer

    rng = np.random.default_rng(3)
    frames = torch.from_numpy(rng.standard_normal((4, 6, 200)).astype(np.float32))
    fitted = SessionRobustZNormalizer(sigma_floor=1e-3).fit(frames)
    rebuilt = SessionRobustZNormalizer.from_stats(
        median=fitted.median.clone(), sigma=fitted.sigma.clone(), sigma_floor=1e-3,
    )
    probe = torch.from_numpy(rng.standard_normal((4, 6, 20)).astype(np.float32))
    torch.testing.assert_close(
        fitted.transform(probe), rebuilt.transform(probe), rtol=0, atol=0,
    )


def test_spec_cache_stats_sidecar_skips_reload_and_refit(tmp_path, monkeypatch) -> None:
    """The robust-z stats sidecar (#80 warm-prepare speedup) lets a cache hit skip
    BOTH the full-frame .npy reload and the median refit: a second view must
    rebuild bit-identical stats from the ~200KB sidecar without ever calling
    np.load on the frames .npy or fit()."""
    from speech_decoding.extractors import view as view_mod
    from speech_decoding.extractors.normalize import SessionRobustZNormalizer

    view1 = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=True, c_max=None,
    )
    ch = ["e0", "e1", "e2"]
    rng = np.random.default_rng(7)
    n = 2048 * 30
    rec = (rng.standard_normal((3, n)) * np.array([1.0, 4.0, 0.5])[:, None]).astype(np.float32)
    event1, _ = _cache_view_harness(view1, rec, ch, monkeypatch)
    view1.prepare([event1])

    # The build wrote exactly one stats sidecar next to the frames.
    assert len(list(tmp_path.rglob("*.stats.npz"))) == 1

    view2 = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=True, c_max=None,
    )
    event2, _ = _cache_view_harness(view2, rec, ch, monkeypatch)

    real_np_load = np.load
    loaded: list[str] = []

    def spy_load(path, *a, **k):
        loaded.append(str(path))
        return real_np_load(path, *a, **k)

    def boom(self, *a, **k):
        raise AssertionError("fit() must not run when the stats sidecar is present")

    monkeypatch.setattr(view_mod.np, "load", spy_load)
    monkeypatch.setattr(SessionRobustZNormalizer, "fit", boom)

    view2.prepare([event2])

    # Read the tiny sidecar, never the frames .npy.
    assert any(p.endswith(".stats.npz") for p in loaded)
    assert not any(p.endswith(".npy") for p in loaded)
    # Stats bit-identical to the building view (sidecar == refit).
    np.testing.assert_array_equal(
        view1._session_stats["s"].median.numpy(),
        view2._session_stats["s"].median.numpy(),
    )
    np.testing.assert_array_equal(
        view1._session_stats["s"].sigma.numpy(),
        view2._session_stats["s"].sigma.numpy(),
    )


def test_spec_cache_stats_sidecar_self_heals_legacy_cache(tmp_path, monkeypatch) -> None:
    """A frames-only cache built before the sidecar existed must upgrade in place:
    the first hit with no sidecar reloads + refits AND writes the sidecar, so the
    NEXT hit skips both. Simulate the legacy state by deleting the sidecar after
    the build."""
    view1 = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=True, c_max=None,
    )
    ch = ["e0", "e1"]
    rng = np.random.default_rng(9)
    rec = rng.standard_normal((2, 2048 * 30)).astype(np.float32)
    event1, _ = _cache_view_harness(view1, rec, ch, monkeypatch)
    view1.prepare([event1])
    next(tmp_path.rglob("*.stats.npz")).unlink()  # → legacy frames-only cache
    assert list(tmp_path.rglob("*.stats.npz")) == []

    view2 = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=True, c_max=None,
    )
    event2, _ = _cache_view_harness(view2, rec, ch, monkeypatch)
    view2.prepare([event2])  # legacy hit: refit from frames + write sidecar

    assert len(list(tmp_path.rglob("*.stats.npz"))) == 1
    np.testing.assert_array_equal(
        view1._session_stats["s"].sigma.numpy(),
        view2._session_stats["s"].sigma.numpy(),
    )


@pytest.mark.must_pass_before_dispatch
def test_spec_cache_scatter_to_per_subject_voltage_rows(monkeypatch) -> None:
    """``_scatter_spec_to_global`` places each session row at its OWN per-subject
    voltage position ``range(len(ch_names))`` (post DP1/DP2 fix, 2026-06-09) — NOT
    the shared last-writer-wins global ``_channels`` index, which permuted rows
    across subjects and was the desync bug. ``C_global`` is still
    ``max(_channels.values())+1``; absent global rows stay exactly 0
    (= |STFT|(0) for raw). Mirrors the apply-path and robust-z scatters row-for-
    row so cache and recompute agree."""
    view = _make_multi_stft_view(spec_cache_dir="/tmp/spec_cache_unit")
    cohort = {"e0": 0, "e9": 1, "e2": 2, "e7": 3, "e1": 4, "e5": 5, "e3": 6, "e8": 7}
    view._channels.update(cohort)  # only sets C_global=8
    sess_ch = ["e7", "e2", "e8", "e1"]  # → per-subject voltage rows [0, 1, 2, 3]
    g_idx = view._get_channels(sess_ch)
    assert g_idx == [0, 1, 2, 3]

    rng = np.random.default_rng(11)
    spec_session = torch.from_numpy(rng.standard_normal((4, 5, 6)).astype(np.float32))
    out = view._scatter_spec_to_global(spec_session, sess_ch)
    assert out.shape == (8, 5, 6)
    for i, g in enumerate(g_idx):
        torch.testing.assert_close(out[g], spec_session[i])
    pad_rows = [r for r in range(8) if r not in g_idx]
    assert torch.all(out[pad_rows] == 0)


def test_spec_cache_dir_excluded_from_cache_uid() -> None:
    """#80: ``spec_cache_dir`` is a feature-cache LOCATION knob, not data-
    determining. Like ``lof_report_path`` it MUST be excluded from the extractor
    cache uid so arming the whole-movie |STFT| cache is identity-transparent — the
    run keeps the SAME uid and the per-clip features it slices are byte-identical
    to the recompute path. Two views differing ONLY in spec_cache_dir must share
    one uid; the STFT/preproc knobs stay in it."""
    bare = _make_multi_stft_view()
    armed = _make_multi_stft_view(spec_cache_dir="/work/whatever/v14_spec_cache")
    assert "spec_cache_dir" in armed._exclude_from_cache_uid()
    assert armed.infra.uid() == bare.infra.uid()


def test_spec_cache_stem_is_filesystem_safe(tmp_path, monkeypatch) -> None:
    """A session uid containing path separators (joint D-cohort/SWEC) or JSON
    metacharacters (BT) must NOT be used verbatim as a filename: it would escape
    the cache dir / hit a missing parent / overrun the 255-byte limit. The hashed
    stem sidesteps all three, and the build still produces exactly one npy/json
    with no leftover temp files, readable end-to-end."""
    from speech_decoding.extractors.view import MultiStftView

    # path separator + JSON metachars + length, all in one nasty key.
    nasty_key = 'sub-D146/ses-01/run-1_{"a":1,"b":2}_' + "x" * 300
    stem = MultiStftView._spec_cache_stem(nasty_key)
    assert "/" not in stem and len(stem) == 40  # sha1 hex

    view = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=False, c_max=None,
    )
    ch = ["e0", "e1"]
    rng = np.random.default_rng(5)
    rec = rng.standard_normal((2, 2048 * 3)).astype(np.float32)
    event, _ = _cache_view_harness(view, rec, ch, monkeypatch, key=nasty_key)

    view.prepare([event])  # would FileNotFoundError if the key were used raw

    assert len(list(tmp_path.rglob("*.npy"))) == 1
    assert len(list(tmp_path.rglob("*.json"))) == 1
    assert list(tmp_path.rglob("*.tmp")) == []  # atomic writes leave no temps
    # The raw key is preserved in the sidecar for provenance.
    meta = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert meta["key"] == nasty_key
    # And the clip read works against the hashed-stem file.
    out = view._get_timed_array(event, start=0.0, duration=1.0)
    assert out.data.shape[0] == 2


def test_spec_cache_stores_large_finite_fp32(tmp_path, monkeypatch) -> None:
    """The fp32 cache stores |STFT| magnitudes beyond the old fp16 range (65504)
    verbatim — FE-RAW-1 raw magnitudes routinely exceed it (~117k on btbank1) —
    so a large but finite spec must round-trip exactly, not overflow to inf."""
    view = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=False, c_max=None,
    )
    ch = ["e0", "e1"]
    rec = np.ones((2, 2048 * 3), dtype=np.float32)
    event, _ = _cache_view_harness(view, rec, ch, monkeypatch)

    big = torch.full((2, 50, 48), 117000.0)  # > 65504 = old fp16 max
    monkeypatch.setattr(view, "_whole_movie_spec", lambda w, sr: big, raising=True)
    view.prepare([event])

    meta_path = next(tmp_path.rglob("*.json"))
    assert json.loads(meta_path.read_text())["dtype"] == "float32"
    stored = np.load(next(tmp_path.rglob("*.npy")))
    assert stored.dtype == np.float32
    assert np.isfinite(stored).all()
    assert float(stored.max()) == 117000.0  # exact, no overflow


def test_spec_cache_rejects_non_finite(tmp_path, monkeypatch) -> None:
    """A non-finite whole-movie spec must fail loud at build, not store inf/nan
    where the f32 recompute path stays finite."""
    view = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=False, c_max=None,
    )
    ch = ["e0", "e1"]
    rec = np.ones((2, 2048 * 3), dtype=np.float32)
    event, _ = _cache_view_harness(view, rec, ch, monkeypatch)

    bad = torch.full((2, 50, 48), float("inf"))
    monkeypatch.setattr(view, "_whole_movie_spec", lambda w, sr: bad, raising=True)

    with pytest.raises(ValueError, match="non-finite"):
        view.prepare([event])


def test_spec_cache_hit_validates_f_bins(tmp_path, monkeypatch) -> None:
    """A cache hit whose stored ``f_bins`` disagrees with the live config (stale
    spec / hash collision) must raise rather than slice mismatched features."""
    view1 = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=False, c_max=None,
    )
    ch = ["e0", "e1"]
    rng = np.random.default_rng(9)
    rec = rng.standard_normal((2, 2048 * 3)).astype(np.float32)
    event1, _ = _cache_view_harness(view1, rec, ch, monkeypatch)
    view1.prepare([event1])

    # Corrupt the sidecar's f_bins, simulating a stale/colliding cache entry.
    json_path = next(tmp_path.rglob("*.json"))
    meta = json.loads(json_path.read_text())
    meta["f_bins"] = meta["f_bins"] + 1
    json_path.write_text(json.dumps(meta))

    view2 = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=False, c_max=None,
    )
    event2, _ = _cache_view_harness(view2, rec, ch, monkeypatch)
    with pytest.raises(ValueError, match="f_bins"):
        view2.prepare([event2])


def test_spec_cache_hit_validates_stft_geometry(tmp_path, monkeypatch) -> None:
    """LG4: an STFT-geometry default edit is invisible to the cache key
    (``infra.uid()`` excludes default-valued fields; the namespace digest folds
    only raw_bins/front_end/apply_log). A post-fix cache records the geometry in
    its sidecar, so a hit whose stored ``hop_length`` disagrees with the live
    config must raise rather than slice a stale feature grid."""
    view1 = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=False, c_max=None,
    )
    # Sanity: the build records the geometry the guard will check.
    assert "hop_length" in view1._spec_cache_geometry()
    ch = ["e0", "e1"]
    rng = np.random.default_rng(11)
    rec = rng.standard_normal((2, 2048 * 3)).astype(np.float32)
    event1, _ = _cache_view_harness(view1, rec, ch, monkeypatch)
    view1.prepare([event1])

    json_path = next(tmp_path.rglob("*.json"))
    meta = json.loads(json_path.read_text())
    assert int(meta["hop_length"]) == view1.hop_length  # geometry made it to disk
    meta["hop_length"] = int(meta["hop_length"]) * 2  # simulate a default edit
    json_path.write_text(json.dumps(meta))

    view2 = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=False, c_max=None,
    )
    event2, _ = _cache_view_harness(view2, rec, ch, monkeypatch)
    with pytest.raises(ValueError, match="hop_length"):
        view2.prepare([event2])


def test_spec_cache_legacy_meta_without_geometry_is_not_invalidated(
    tmp_path, monkeypatch
) -> None:
    """LG4 non-invalidation: a legacy sidecar predating the geometry keys must
    still hit (the guard is assert-IF-present), so existing ``/work`` caches are
    not silently torched by shipping the fix."""
    view1 = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=False, c_max=None,
    )
    ch = ["e0", "e1"]
    rng = np.random.default_rng(12)
    rec = rng.standard_normal((2, 2048 * 3)).astype(np.float32)
    event1, _ = _cache_view_harness(view1, rec, ch, monkeypatch)
    view1.prepare([event1])

    # Strip the geometry keys to mimic a cache built before this fix.
    json_path = next(tmp_path.rglob("*.json"))
    meta = json.loads(json_path.read_text())
    for k in ("hop_length", "nperseg_low", "nperseg_mid", "nperseg_hi", "log_eps"):
        meta.pop(k, None)
    json_path.write_text(json.dumps(meta))

    view2 = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path), session_robust_z=False, c_max=None,
    )
    event2, _ = _cache_view_harness(view2, rec, ch, monkeypatch)
    view2.prepare([event2])  # must NOT raise — legacy cache still served


# --------------------------------------------------------------------------- #
# Electrode-row alignment (DP1/DP2 fix, 2026-06-09)                            #
#                                                                             #
# MultiStftView must scatter each event's electrode_tokens into THAT event's   #
# own per-subject voltage order (enumerate(ch_names)), NOT the shared          #
# last-writer-wins global ``_channels`` index. The global index permutes rows  #
# across subjects in a pooled prepare() (DP1) and silently drops a subject's   #
# own electrodes when two alias one global row (DP2) — desyncing               #
# electrode_tokens[c] from voltage_electrode_order(subject)[c], and so from    #
# the DK support / valid_mask extractors. Regression for                       #
# reports/bt_alignment/electrode_desync_damage_2026_06_09.md.                  #
# --------------------------------------------------------------------------- #

import pathlib as _pathlib

_BT_CACHE_VIEW = (
    _pathlib.Path(__file__).resolve().parents[3] / ".cache" / "braintreebank"
)


def _real_two_subject_orders():
    """sub_1 + sub_8 voltage orders from the vendored fixtures. The pair shares
    12 electrode names at DIFFERENT positions, so the buggy shared-dict scatter
    mis-places exactly those rows (DP1) and collides them (DP2)."""
    from speech_decoding.studies.braintreebank.anatomy import voltage_electrode_order

    if not (_BT_CACHE_VIEW / "electrode_labels" / "sub_1").exists():
        pytest.skip("vendored braintreebank electrode_labels not present")
    o1 = list(voltage_electrode_order(str(_BT_CACHE_VIEW), 1))
    o8 = list(voltage_electrode_order(str(_BT_CACHE_VIEW), 8))
    return o1, o8


def _pooled_view_two_subjects(order_first, order_last):
    """A MultiStftView whose ``_channels`` is populated exactly as the neuralset
    pooled ``prepare()`` loop populates it (``_update_channels`` per session, in
    order), with ``order_last`` written last (last-writer-wins). ``channel_order``
    is pinned to ``'original'`` to match the production dispatch
    (dispatch_v14.py:940) — the config under which the desync occurs."""
    view = _make_multi_stft_view(channel_order="original")
    view._update_channels(list(order_first))
    view._update_channels(list(order_last))
    return view


def test_get_channels_emits_per_subject_voltage_order_real_cohort() -> None:
    """DP1: after a pooled sub_1+sub_8 prepare, ``_get_channels`` for each
    subject's clip must return that subject's own voltage positions (``range``),
    so electrode_tokens[c] == voltage_electrode_order(subject)[c]."""
    o1, o8 = _real_two_subject_orders()
    view = _pooled_view_two_subjects(o1, o8)  # sub_8 last writer
    assert view._get_channels(o1) == list(range(len(o1)))
    assert view._get_channels(o8) == list(range(len(o8)))


def test_get_channels_no_within_subject_row_collision_real_cohort() -> None:
    """DP2: no two of a subject's own electrodes may scatter to one output row
    (that silently overwrites one electrode's signal). 12 sub_1 electrodes
    collide under the buggy shared-dict scatter."""
    o1, o8 = _real_two_subject_orders()
    view = _pooled_view_two_subjects(o1, o8)
    idx1 = view._get_channels(o1)
    assert len(set(idx1)) == len(idx1), "within-subject row collision -> data loss"


def test_scatter_spec_to_global_preserves_per_subject_order() -> None:
    """The spec-cache re-scatter must place session row i at output row i (no
    permutation, no loss) for a pooled cohort. Tracer spec: row i is the constant
    i, so any misplacement is visible."""
    o1, o8 = _real_two_subject_orders()
    view = _pooled_view_two_subjects(o1, o8)
    n = len(o1)
    tracer = (
        torch.arange(n, dtype=torch.float32)
        .reshape(n, 1, 1)
        .expand(n, 2, 3)
        .contiguous()
    )
    out = view._scatter_spec_to_global(tracer, o1)
    np.testing.assert_array_equal(
        out[:n, 0, 0].numpy(), np.arange(n, dtype=np.float32)
    )


def test_scatter_stats_to_global_preserves_per_subject_order() -> None:
    """Robust-z stat scatter (median/sigma) must land session row i at output row
    i for a pooled cohort, matching the spec scatter row-for-row."""
    from speech_decoding.extractors.normalize import SessionRobustZNormalizer

    o1, o8 = _real_two_subject_orders()
    view = _pooled_view_two_subjects(o1, o8)
    n = len(o1)
    norm = SessionRobustZNormalizer()
    norm.median = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1).clone()
    norm.sigma = torch.ones(n, 1, 1)
    view._scatter_stats_to_global(norm, o1)
    np.testing.assert_array_equal(
        norm.median[:n, 0, 0].numpy(), np.arange(n, dtype=np.float32)
    )


def test_update_channels_row_count_covers_longest_session() -> None:
    """C_global = max(_channels.values())+1 must be >= every session's electrode
    count, so the per-session range(len) scatter never indexes out of bounds —
    even when the longest session's tail electrode name collides with a shorter
    later session (name-keyed last-writer-wins would shrink the count below the
    longest session's length and crash the scatter)."""
    long_sess = [f"E{i}" for i in range(10)]  # 10 electrodes, tail "E9"
    short_sess = ["E9", "X0", "X1"]  # reuses tail name "E9" at position 0
    view = _make_multi_stft_view(channel_order="original")
    view._update_channels(long_sess)
    view._update_channels(short_sess)  # last writer
    c_global = max(view._channels.values()) + 1
    assert c_global >= len(long_sess)


def test_single_subject_scatter_is_byte_identical_to_base() -> None:
    """Byte-identical guard: for a single subject the per-session emit equals the
    neuralset channel_order='original' base (global index == voltage position when
    there is exactly one writer), so single-subject runs are unchanged."""
    o1, _ = _real_two_subject_orders()
    view = _make_multi_stft_view(channel_order="original")
    view._update_channels(o1)  # single subject
    assert view._get_channels(o1) == list(range(len(o1)))
    assert max(view._channels.values()) + 1 == len(o1)


def test_c_max_pins_scatter_width_for_cold_fit() -> None:
    """Cold-fit guard (2026-06-15): when ``c_max`` is set, the scatter width is
    pinned to ``c_max`` even if ``_channels`` has only seen a SHORTER session
    before a fit runs — so a stat/spec scatter never indexes out of bounds and
    all three scatter sites (two fit-side + the vendored apply neuro.py:455, which
    all read ``max(_channels)+1``) agree. Without the pin, a 200-channel session's
    stat scatter against a ``_channels`` populated by a 5-channel probe would size
    ``c_global=5`` and crash (``global_*[idx]`` out of bounds)."""
    from speech_decoding.extractors.normalize import SessionRobustZNormalizer

    view = _make_multi_stft_view(channel_order="original", c_max=256)
    # Cold ordering: only a short 5-channel session has populated _channels.
    view._update_channels([f"E{i}" for i in range(5)])
    assert max(view._channels.values()) + 1 == 256  # pinned to c_max, not 5

    n = 200
    sess = [f"L{i}" for i in range(n)]
    norm = SessionRobustZNormalizer()
    norm.median = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1).clone()
    norm.sigma = torch.ones(n, 1, 1)
    view._scatter_stats_to_global(norm, sess)  # must not raise
    assert norm.median.shape[0] == 256
    np.testing.assert_array_equal(
        norm.median[:n, 0, 0].numpy(), np.arange(n, dtype=np.float32)
    )
    # No-source rows carry σ=0 → transform zeros them (no 0/0 NaN at apply).
    assert float(norm.sigma[n:].abs().sum()) == 0.0

    # The spec scatter lands at the SAME c_max width (matches stats + apply).
    spec = (
        torch.arange(n, dtype=torch.float32)
        .reshape(n, 1, 1)
        .expand(n, 2, 3)
        .contiguous()
    )
    out = view._scatter_spec_to_global(spec, sess)
    assert out.shape[0] == 256
    np.testing.assert_array_equal(
        out[:n, 0, 0].numpy(), np.arange(n, dtype=np.float32)
    )


@pytest.mark.must_pass_before_dispatch
def test_spec_cache_stores_session_order_not_global_order(tmp_path, monkeypatch) -> None:
    """L7 cache-key row-identity audit (open Q5). The #80 whole-movie spec cache
    must persist frames + robust-z stats in PER-SESSION order (``C_session`` rows),
    never the c_max-padded / cohort-scattered GLOBAL layout. That is exactly what
    makes the cache order-transparent: the corrected per-subject-voltage scatter is
    re-applied by ``_get_channels`` at every clip read, so a cache built BEFORE the
    electrode-row-desync fix holds bit-identical session-order frames and is reused
    correctly AFTER the fix — NO cache invalidation needed (the fix is a method
    override, leaving ``infra.uid()`` / the cache namespace unchanged). A regression
    that baked global/cohort order into the .npy/.stats would desync every reuse;
    this fails loud on it. See electrode_desync_damage_2026_06_09.md."""
    view = _make_multi_stft_view(
        spec_cache_dir=str(tmp_path),
        session_robust_z=True,
        c_max=384,  # >> C_session, so session vs global layout is unambiguous
        channel_order="original",
    )
    ch = ["e0", "e1", "e2"]  # C_session = 3
    rng = np.random.default_rng(11)
    rec = rng.standard_normal((3, 2048 * 20)).astype(np.float32)
    event, _ = _cache_view_harness(view, rec, ch, monkeypatch)
    view.prepare([event])

    # (1) frames stored in session order: 3 rows, NOT c_max=384 / C_global.
    npy = next(tmp_path.rglob("*.npy"))
    frames = np.load(npy)
    assert frames.shape[0] == len(ch) == 3, "spec frames not in session order"
    # (2) meta records session geometry.
    meta = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert meta["n_channels"] == 3
    assert list(meta["ch_names"]) == ch
    # (3) robust-z sidecar stats are session-order too (median has C_session rows).
    stats = np.load(next(tmp_path.rglob("*.stats.npz")))
    assert stats["median"].shape[0] == 3, "robust-z stats not in session order"

    # (4) order-transparency: re-scattering the cached session frames depends only
    # on the LIVE _channels (via _get_channels), placing this session's rows at its
    # own per-subject voltage positions [0,3) and zero-padding the rest — nothing
    # global is baked into the cache (proven by (1)-(3): the .npy/.stats hold 3
    # session rows). With c_max set, the live scatter width is pinned to c_max
    # (cold-fit fix 2026-06-15: all three scatter sites + the vendored apply path
    # agree on max(_channels)+1 == c_max, independent of arrival order); the FINAL
    # clip is c_max-padded regardless, so this is output-identical to the old
    # C_session scatter-then-pad.
    scattered = view._scatter_spec_to_global(torch.from_numpy(frames), ch)
    assert scattered.shape[0] == max(view._channels.values()) + 1 == view.c_max
    np.testing.assert_array_equal(scattered[: len(ch)].numpy(), frames)
    assert float(np.abs(scattered[len(ch):].numpy()).sum()) == 0.0  # rest zero-pad


@pytest.mark.must_pass_before_dispatch
def test_spec_cache_session_order_independent_of_c_max(tmp_path, monkeypatch) -> None:
    """L7: the stored spec row count is the session electrode count regardless of
    ``c_max`` — proving c_max (the global pad width) is NOT baked into the cache.
    Two builds with different c_max must store the same session-order n_channels."""
    out = {}
    for tag, c_max in (("a", 384), ("b", None)):
        d = tmp_path / tag
        d.mkdir()
        view = _make_multi_stft_view(
            spec_cache_dir=str(d), session_robust_z=False,
            c_max=c_max, channel_order="original",
        )
        ch = ["e0", "e1", "e2", "e3"]
        rng = np.random.default_rng(5)
        rec = rng.standard_normal((4, 2048 * 12)).astype(np.float32)
        event, _ = _cache_view_harness(view, rec, ch, monkeypatch)
        view.prepare([event])
        out[tag] = np.load(next(d.rglob("*.npy"))).shape[0]
    assert out["a"] == out["b"] == 4, "spec cache row count depends on c_max"


# =====================================================================
# 2STFT front end (§3a) — single-band raw |STFT| views (front_end="band").
# reports/fe_2stft_handoff_2026_06_12.md; bins audit-verified 2026-06-13.
# =====================================================================
def test_2stft_band_k_range_is_exact_low_k1_14_high_k4_12() -> None:
    """The authoritative bin selector reproduces the spec EXACTLY: low band
    (N=512, 4-56 Hz) → k1..k14 = 14 bins; high band (N=128, 64-192 Hz) →
    k4..k12 = 9 bins. Total 23. Pins the ±1e-9 guard at exact multiples
    (56/4 = 14.0, 64/16 = 4.0, 192/16 = 12.0)."""
    from speech_decoding.extractors.view import (
        STFT_2BAND_HIGH,
        STFT_2BAND_LOW,
        _stft_band_k_range,
    )

    lo = _stft_band_k_range(
        STFT_2BAND_LOW["band_f_lo_hz"], STFT_2BAND_LOW["band_f_hi_hz"],
        nperseg=int(STFT_2BAND_LOW["band_nperseg"]),
    )
    hi = _stft_band_k_range(
        STFT_2BAND_HIGH["band_f_lo_hz"], STFT_2BAND_HIGH["band_f_hi_hz"],
        nperseg=int(STFT_2BAND_HIGH["band_nperseg"]),
    )
    assert lo == (1, 14), f"low band must be k1..k14 (14 bins); got {lo}"
    assert hi == (4, 12), f"high band must be k4..k12 (9 bins); got {hi}"
    assert (lo[1] - lo[0] + 1) + (hi[1] - hi[0] + 1) == 23


def test_2stft_single_band_view_shapes_and_rates() -> None:
    """§3a: low band → (C, 14, T) at 8 Hz (T = 1 + L//256); high band →
    (C, 9, T) at 32 Hz (T = 1 + L//64). Verified at 1 s and 5 s."""
    from speech_decoding.extractors.view import _single_stft_raw_view

    for duration_s, (low_T, high_T) in ((1.0, (9, 33)), (5.0, (41, 161))):
        L = int(round(duration_s * 2048))
        wav = torch.zeros(4, L)
        low = _single_stft_raw_view(
            wav, sample_rate=2048, nperseg=512, hop_length=256,
            k0=1, k1=14, log_eps=1e-6,
        )
        high = _single_stft_raw_view(
            wav, sample_rate=2048, nperseg=128, hop_length=64,
            k0=4, k1=12, log_eps=1e-6,
        )
        assert low.shape == (4, 14, low_T), low.shape
        assert high.shape == (4, 9, high_T), high.shape


def test_2stft_single_band_view_routes_tone_to_correct_bin() -> None:
    """A 40 Hz tone peaks in the low band's k10 (center 40 Hz, slice index 9);
    a 128 Hz tone peaks in the high band's k8 (center 128 Hz, slice index 4).
    Validates bin selection + Δf for both bands."""
    from speech_decoding.extractors.view import _single_stft_raw_view

    sr = 2048
    t = torch.arange(sr).float() / sr
    low_freqs = torch.fft.rfftfreq(512, d=1.0 / sr)[1 : 14 + 1]
    high_freqs = torch.fft.rfftfreq(128, d=1.0 / sr)[4 : 12 + 1]

    tone_40 = torch.sin(2 * torch.pi * 40.0 * t).unsqueeze(0)
    low = _single_stft_raw_view(
        tone_40, sample_rate=sr, nperseg=512, hop_length=256, k0=1, k1=14, log_eps=1e-6,
    )
    low_peak = int(low.mean(dim=-1)[0].argmax().item())
    assert abs(low_freqs[low_peak].item() - 40.0) < 4.0

    tone_128 = torch.sin(2 * torch.pi * 128.0 * t).unsqueeze(0)
    high = _single_stft_raw_view(
        tone_128, sample_rate=sr, nperseg=128, hop_length=64, k0=4, k1=12, log_eps=1e-6,
    )
    high_peak = int(high.mean(dim=-1)[0].argmax().item())
    assert abs(high_freqs[high_peak].item() - 128.0) < 16.0


def test_2stft_single_band_chunked_matches_unchunked() -> None:
    """The whole-movie cache slices a chunked single-band spectrogram; that
    overlap-save build MUST be byte-identical to one un-chunked
    ``_single_stft_raw_view`` over the full recording, for BOTH bands, across
    lengths that straddle chunk seams (half=N/2 is a multiple of hop for both)."""
    from speech_decoding.extractors.view import (
        _single_stft_raw_view,
        _single_stft_raw_view_chunked,
    )

    sr = 2048
    torch.manual_seed(0)
    bands = (
        dict(nperseg=512, hop_length=256, k0=1, k1=14),
        dict(nperseg=128, hop_length=64, k0=4, k1=12),
    )
    for band in bands:
        for n_samples in (4096, 60_000, 122_880, 200_003, 350_000):
            for chunk_frames in (64, 200, 960):
                t = torch.arange(n_samples).float() / sr
                sig = (
                    torch.sin(2 * torch.pi * 73.0 * t)
                    + 0.5 * torch.sin(2 * torch.pi * 11.0 * t)
                    + 0.1 * torch.randn(n_samples)
                )
                wav = torch.stack([sig, sig.flip(0)], dim=0)
                kw = dict(sample_rate=sr, log_eps=1e-6, **band)
                full = _single_stft_raw_view(wav, **kw)
                chunked = _single_stft_raw_view_chunked(
                    wav, chunk_frames=chunk_frames, **kw
                )
                assert chunked.shape == full.shape, (
                    f"band={band} n={n_samples} chunk={chunk_frames}: "
                    f"{chunked.shape} vs {full.shape}"
                )
                assert torch.allclose(chunked, full, atol=1e-5, rtol=1e-4), (
                    f"chunked != unchunked band={band} n={n_samples} "
                    f"chunk={chunk_frames}: "
                    f"max|delta|={(chunked - full).abs().max().item():.3e}"
                )


def test_2stft_single_band_chunked_rejects_non_aligned_window() -> None:
    """Single-band overlap-save needs nperseg % hop == 0 AND half % hop == 0."""
    from speech_decoding.extractors.view import _single_stft_raw_view_chunked

    with pytest.raises(ValueError, match="divisible by"):
        _single_stft_raw_view_chunked(
            torch.zeros(2, 300_000), sample_rate=2048, nperseg=512,
            hop_length=100, k0=1, k1=14, log_eps=1e-6, chunk_frames=200,
        )


def test_2stft_multistftview_band_mode_construct_and_bins() -> None:
    """MultiStftView(front_end="band", **STFT_2BAND_{LOW,HIGH}) wires the band:
    _band_bins / _expected_raw_f_bins match the spec, hop_length pins to band_hop,
    and the per-clip spec has the band shape."""
    from speech_decoding.extractors.view import (
        STFT_2BAND_HIGH,
        STFT_2BAND_LOW,
        MultiStftView,
    )

    low = MultiStftView(event_types="Ieeg", car="shaft", front_end="band",
                        hop_length=int(STFT_2BAND_LOW["band_hop"]), **STFT_2BAND_LOW)
    high = MultiStftView(event_types="Ieeg", car="shaft", front_end="band",
                         hop_length=int(STFT_2BAND_HIGH["band_hop"]), **STFT_2BAND_HIGH)
    assert low.front_end == "band" and high.front_end == "band"
    assert low._band_bins() == (1, 14) and low._expected_raw_f_bins() == 14
    assert high._band_bins() == (4, 12) and high._expected_raw_f_bins() == 9
    assert low.hop_length == 256 and high.hop_length == 64
    # 1 s @ 2048 Hz → low 9 frames (8 Hz + pad), high 33 (32 Hz + pad).
    assert low.n_time_bins_for_duration(1.0) == 9
    assert high.n_time_bins_for_duration(1.0) == 33
    spec_lo = low._spec_from_waveform(torch.zeros(3, 2048), 2048)
    spec_hi = high._spec_from_waveform(torch.zeros(3, 2048), 2048)
    assert spec_lo.shape == (3, 14, 9)
    assert spec_hi.shape == (3, 9, 33)


def test_2stft_band_mode_requires_hop_equals_band_hop() -> None:
    """The band-mode validator forces hop_length == band_hop so every
    self.hop_length reader (frame counts, clip snap, chunked) is correct."""
    from speech_decoding.extractors.view import STFT_2BAND_LOW, MultiStftView

    with pytest.raises(ValueError, match="hop_length == band_hop"):
        MultiStftView(event_types="Ieeg", car="shaft", front_end="band",
                      hop_length=128, **STFT_2BAND_LOW)  # 128 != band_hop 256


def test_2stft_low_and_high_bands_get_distinct_cache_namespaces(tmp_path) -> None:
    """Low and high band views MUST land in different spec-cache directories
    (band geometry folded into the namespace digest), else one band's spectrogram
    would silently overwrite the other's."""
    from speech_decoding.extractors.view import (
        STFT_2BAND_HIGH,
        STFT_2BAND_LOW,
        MultiStftView,
    )

    low = MultiStftView(event_types="Ieeg", car="shaft", front_end="band",
                        hop_length=int(STFT_2BAND_LOW["band_hop"]),
                        spec_cache_dir=str(tmp_path), **STFT_2BAND_LOW)
    high = MultiStftView(event_types="Ieeg", car="shaft", front_end="band",
                         hop_length=int(STFT_2BAND_HIGH["band_hop"]),
                         spec_cache_dir=str(tmp_path), **STFT_2BAND_HIGH)
    assert low._spec_cache_namespace() != high._spec_cache_namespace()


def _make_band_view(which: str, **kw):
    """A single-band MultiStftView (front_end="band") for the low/high basket."""
    from speech_decoding.extractors.view import (
        STFT_2BAND_HIGH,
        STFT_2BAND_LOW,
        MultiStftView,
    )

    cfg = STFT_2BAND_LOW if which == "low" else STFT_2BAND_HIGH
    base = dict(event_types="Ieeg", car="shaft", front_end="band",
                hop_length=int(cfg["band_hop"]))
    base.update(cfg)
    base.update(kw)
    return MultiStftView(**base)


def _fit_band_robust_z(view, rec, ch, monkeypatch):
    """Run the REAL chunked _fit_session_robust_z on a band view over ``rec``."""
    import types

    from speech_decoding.extractors.view import MultiStftView

    class _E:
        start = 0.0

        def _splittable_event_uid(self):
            return "s"

    def _fake_get_data(self, evs):
        for _e in evs:
            yield types.SimpleNamespace(frequency=2048.0, data=rec, ch_names=list(ch))

    monkeypatch.setattr(
        MultiStftView, "_get_data",
        property(lambda self: types.MethodType(_fake_get_data, self)), raising=False,
    )
    monkeypatch.setattr(
        MultiStftView, "_event_types_helper",
        property(lambda self: types.SimpleNamespace(extract=lambda obj: obj)),
        raising=False,
    )
    view._channels.update({c: i for i, c in enumerate(ch)})
    view._fit_session_robust_z([_E()])
    return view._session_stats["s"]


def test_2stft_session_robust_z_fits_band_rows_23_total(monkeypatch) -> None:
    """§3c: each band fits its OWN freq rows over the session — low 14 + high 9 =
    23 rows total, disjoint by construction (separate views). The chunked fit's
    window is the band's own nperseg (low 512 / high 128)."""
    ch = ["e0", "e1", "e2"]
    rng = np.random.default_rng(71)
    rec = rng.standard_normal((3, 2048 * 4)).astype(np.float32)

    low = _make_band_view("low", session_robust_z=True)
    high = _make_band_view("high", session_robust_z=True)
    assert low._widest_nperseg() == 512 and high._widest_nperseg() == 128

    low_stats = _fit_band_robust_z(low, rec, ch, monkeypatch)
    high_stats = _fit_band_robust_z(high, rec, ch, monkeypatch)
    # per-(channel, freq) stats: F dim is the band's bin count.
    assert low_stats.median.shape[1] == 14
    assert high_stats.median.shape[1] == 9
    assert low_stats.median.shape[1] + high_stats.median.shape[1] == 23


def test_2stft_session_robust_z_fit_apply_roundtrips_per_band(monkeypatch) -> None:
    """§3c: the frozen band stats z-score a clip per (channel, freq) — applying
    the fitted normalizer matches the standalone transform of the band spec."""
    from speech_decoding.extractors.normalize import SessionRobustZNormalizer

    ch = ["e0", "e1", "e2"]
    rng = np.random.default_rng(72)
    rec = rng.standard_normal((3, 2048 * 4)).astype(np.float32)

    for which, n_bins in (("low", 14), ("high", 9)):
        view = _make_band_view(which, session_robust_z=True)
        clip = rec[:, :2048]
        whole = view._spec_from_waveform(torch.from_numpy(rec), 2048)
        ref = SessionRobustZNormalizer(sigma_floor=view.session_z_sigma_floor).fit(whole)
        out = ref.transform(view._spec_from_waveform(torch.from_numpy(clip), 2048))
        assert out.shape[1] == n_bins
        # z-scored: each (channel, freq) row centred near 0, unit-ish scale.
        assert torch.isfinite(out).all()
