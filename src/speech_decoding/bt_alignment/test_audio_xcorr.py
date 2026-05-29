"""Tests for per-trial audio cross-correlation + drift estimation."""
import numpy as np
import pandas as pd

from speech_decoding.bt_alignment.audio_xcorr import (
    compute_rms_envelope_at_anchors,
    pick_anchors,
    estimate_drift,
    DRIFT_GATE_MS_PER_MIN,
    MIN_CORRELATION,
)


def test_rms_envelope_constant_amplitude_pure_tone():
    sr = 16000
    dur = 10.0
    t = np.arange(int(sr * dur)) / sr
    wav = 0.5 * np.sin(2 * np.pi * 440 * t)
    anchors = np.array([1.0, 3.0, 5.0, 8.0])
    rms = compute_rms_envelope_at_anchors(wav, sr, anchors, window_s=0.5)
    expected = 0.5 / np.sqrt(2)
    assert np.allclose(rms, expected, atol=1e-3)


def test_rms_envelope_zero_signal():
    sr = 16000
    wav = np.zeros(sr * 5)
    rms = compute_rms_envelope_at_anchors(wav, sr, np.array([1.0, 2.0]))
    assert np.all(rms == 0)


def test_rms_envelope_anchor_outside_wav_returns_nan():
    sr = 16000
    wav = np.ones(sr * 2)
    rms = compute_rms_envelope_at_anchors(wav, sr, np.array([0.5, 50.0]))
    assert np.isfinite(rms[0])
    assert np.isnan(rms[1])


def test_pick_anchors_evenly_spaced():
    df = pd.DataFrame({
        "start": np.arange(0, 1000.0, 1.0),
        "rms": np.linspace(0.1, 0.5, 1000),
    })
    picks = pick_anchors(df, n=10)
    assert len(picks) == 10
    starts = picks["start"].to_numpy()
    assert starts[0] == 0
    # picks should span the full range
    assert starts[-1] > 800


def test_pick_anchors_filters_nan_rms():
    df = pd.DataFrame({
        "start": [1, 2, 3, 4, 5],
        "rms": [0.1, np.nan, 0.3, np.nan, 0.5],
    })
    picks = pick_anchors(df, n=5)
    assert len(picks) == 3
    assert all(picks["rms"].notna())


def test_estimate_drift_no_drift_perfect_correlation():
    # rip_rms perfectly tracks bt_rms with no time-dependent slope
    rng = np.random.default_rng(0)
    n = 50
    t = np.linspace(60, 6000, n)
    bt = rng.uniform(0.05, 0.3, size=n)
    rip = 1.05 * bt  # constant 5% gain, no drift
    r = estimate_drift(t, rip, bt)
    assert r.pearson_r > 0.99
    assert abs(r.drift_slope_ms_per_min) < 1.0
    assert r.pass_correlation
    assert r.pass_drift


def test_estimate_drift_synthetic_linear_drift_caught():
    # log(rip) - log(bt) = slope*t + c; simulate slope big enough to fail gate
    n = 50
    t = np.linspace(60, 6000, n)  # 99 minutes
    bt = np.full(n, 0.1)
    # 200 ms/min drift => slope_per_s = 200/60/1000 = 3.33e-3 nat units / s
    rip = bt * np.exp(3.33e-3 * t)
    r = estimate_drift(t, rip, bt)
    assert r.drift_slope_ms_per_min > 100
    assert not r.pass_drift


def test_estimate_drift_too_few_anchors_returns_nan_no_pass():
    r = estimate_drift(np.array([1.0, 2.0]), np.array([0.1, 0.1]), np.array([0.1, 0.1]))
    assert not r.pass_correlation
    assert not r.pass_drift
    assert np.isnan(r.pearson_r)


def test_drift_gate_constants():
    assert DRIFT_GATE_MS_PER_MIN == 50.0
    assert MIN_CORRELATION == 0.5
