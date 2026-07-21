"""Tests for LowLfpView — the raw-waveform 1-30 Hz LFS band producer.

TDD for the Chang 2-stream frontend redesign. The load-bearing test is
``test_frame_grid_alignment``: it proves the LFS 64 Hz frame grid rides the SAME
window-center clock as an HGA STFT (N=64, hop=16), so the two streams are exactly
alignable (one LFS frame per two HGA frames).
"""

from __future__ import annotations

import numpy as np
import scipy.signal
import torch

from speech_decoding.extractors.lowlfp_view import LowLfpView
from speech_decoding.extractors.normalize import SessionRobustZNormalizer

FS = 2048
C = 4
DUR_S = 6.0
T = int(FS * DUR_S)  # 12288


def _make_view(tmp_path=None, **overrides):
    kwargs = dict(
        front_end="band",
        band_nperseg=64,
        band_hop=32,
        hop_length=32,
        band_f_lo_hz=1.0,
        band_f_hi_hz=30.0,
        session_robust_z=True,
    )
    kwargs.update(overrides)
    if tmp_path is not None:
        kwargs["spec_cache_dir"] = str(tmp_path)
    return LowLfpView(**kwargs)


def _sinusoid_mix(n_ch=C, n_samples=T, fs=FS, freqs=(2.0, 20.0, 100.0)):
    """(C, T) mix of known sinusoids so we can verify filtering per-frequency."""
    t = torch.arange(n_samples).float() / fs
    wave = torch.zeros(n_ch, n_samples)
    for f in freqs:
        # small per-channel phase jitter so channels aren't identical
        for c in range(n_ch):
            wave[c] += torch.sin(2 * np.pi * f * t + 0.1 * c)
    return wave


# --------------------------------------------------------------------------- #
def test_shape_and_f_bins() -> None:
    view = _make_view()
    wave = _sinusoid_mix()
    frames = view._lfs_frames(wave, FS)
    assert frames.shape == (C, 1, 1 + T // 32)
    assert view._expected_raw_f_bins() == 1
    print(f"[check] OK shape={tuple(frames.shape)} f_bins={view._expected_raw_f_bins()}")


def test_clock_matches_n_time_bins() -> None:
    """LFS frame count == inherited n_time_bins_for_duration (same hop=32 clock)."""
    view = _make_view()
    for dur in (0.5, 1.0, 2.0, 5.0, 6.0):
        n_samples = int(round(dur * FS))
        wave = _sinusoid_mix(n_samples=n_samples)
        frames = view._lfs_frames(wave, FS)
        expected = view.n_time_bins_for_duration(dur)
        assert frames.shape[-1] == expected, (dur, frames.shape[-1], expected)
    print("[check] OK LFS frame count == n_time_bins_for_duration for all durations")


def test_prepare_geometry_probe_short_input() -> None:
    """REGRESSION: neuralset's prepare() probes geometry with a ~1 ms (duration=0.001,
    ~2-sample) clip that is shorter than the SOS filter's padlen (~27). _lfs_frames
    must survive it (shape-only), not raise ValueError('greater than padlen'). Real
    clips are >>padlen so this path never touches them."""
    view = _make_view()
    # 0.001 s at 2048 Hz ≈ 2 samples (the exact prepare() probe), plus other
    # sub-padlen lengths down to 1 sample.
    for n_samples in (1, 2, 3, int(round(0.001 * FS)), 10, 27):
        wave = _sinusoid_mix(n_samples=n_samples)
        frames = view._lfs_frames(wave, FS)  # must NOT raise
        assert frames.shape == (C, 1, 1 + n_samples // 32), (n_samples, frames.shape)
    print("[check] OK short-input geometry probe survives (no padlen ValueError)")


def test_frame_grid_alignment() -> None:
    """CRITICAL: LFS frame i centered at raw-sample i*32 == HGA frame 2i center (2i*16).

    Proves the LFS 64 Hz grid is exactly the even sub-grid of an HGA STFT
    (N=64, hop=16, center=True), so the two streams are alignable frame-for-frame.
    """
    view = _make_view()
    wave = _sinusoid_mix()
    frames = view._lfs_frames(wave, FS)
    n_lfs = frames.shape[-1]

    # HGA STFT frame count (torch.stft(center=True), hop=16).
    n_hga = 1 + T // 16
    # LFS count is the even sub-grid of the HGA grid.
    assert n_lfs == 1 + T // 32
    assert n_lfs == 1 + (n_hga - 1) // 2, (n_lfs, n_hga)

    lfs_hop, hga_hop = 32, 16
    for i in range(n_lfs):
        lfs_center = i * lfs_hop
        hga_center = (2 * i) * hga_hop
        assert lfs_center == hga_center, (i, lfs_center, hga_center)
    print(f"[check] OK grid alignment n_lfs={n_lfs} n_hga={n_hga}; "
          f"LFS[i] center == HGA[2i] center for all i")


def test_sos_filter_removes_high_freq() -> None:
    """100 Hz attenuated to <1% of input; 2 Hz and 20 Hz survive (>50%)."""
    view = _make_view()
    fs = FS
    n = T
    t = torch.arange(n).float() / fs

    def amp_at(sig_np, f, fs_grid, n_grid):
        """Goertzel-ish: project onto sin/cos at f, return amplitude."""
        tt = np.arange(n_grid) / fs_grid
        s = np.sin(2 * np.pi * f * tt)
        co = np.cos(2 * np.pi * f * tt)
        a = 2.0 * (sig_np * s).mean()
        b = 2.0 * (sig_np * co).mean()
        return float(np.hypot(a, b))

    for f in (2.0, 20.0, 100.0):
        wave = torch.sin(2 * np.pi * f * t).unsqueeze(0)  # (1, T), unit amplitude
        frames = view._lfs_frames(wave, fs)  # (1, 1, n_frames) at 64 Hz
        out = frames[0, 0].numpy()
        n_frames = out.shape[0]
        if f == 100.0:
            # 100 Hz is -86 dB after zero-phase filtfilt; the only residual is the
            # sosfiltfilt edge transient, so measure the central steady-state region
            # (drop the outer 20% of frames) — there the stop-band rejection shows.
            c0, c1 = n_frames // 5, 4 * n_frames // 5
            frac = float(np.sqrt((out[c0:c1] ** 2).mean()) / (1.0 / np.sqrt(2)))
            assert frac < 0.01, f"100 Hz not removed: central rms frac {frac}"
            print(f"[check] OK 100 Hz central rms/input = {frac:.4f} < 0.01")
        else:
            out_amp = amp_at(out, f, 64.0, n_frames)
            assert out_amp > 0.5, f"{f} Hz not preserved: amp {out_amp}"
            print(f"[check] OK {f:g} Hz amplitude {out_amp:.3f} > 0.5")


def test_filter_sanity_no_fake_sine() -> None:
    """Viewer-bug guard: output is NOT a collapsed near-pure-~1 Hz sine (the b,a
    instability signature). Passband content survives and no spurious 1 Hz spike
    dominates a broadband input; confirm the code uses SOS."""
    import inspect

    from speech_decoding.extractors import lowlfp_view

    src = inspect.getsource(lowlfp_view.LowLfpView._lfs_frames)
    assert 'output="sos"' in src and "sosfiltfilt" in src, "must use SOS, not b,a"

    view = _make_view()

    # (1) A 20 Hz signal: >16 Hz power fraction must be > 0 and delta not ~all.
    t = torch.arange(T).float() / FS
    wave20 = torch.sin(2 * np.pi * 20.0 * t).unsqueeze(0)
    out20 = view._lfs_frames(wave20, FS)[0, 0].numpy()
    f, pxx = scipy.signal.welch(out20, fs=64.0, nperseg=min(256, out20.shape[0]))
    total = pxx.sum()
    high_frac = pxx[f > 16.0].sum() / total
    delta_frac = pxx[(f >= 1.0) & (f <= 4.0)].sum() / total
    assert high_frac > 0.0, "no >16 Hz power — 20 Hz signal collapsed"
    assert delta_frac < 0.99, f"delta dominates ({delta_frac}) — fake ~1 Hz sine"
    print(f"[check] OK 20 Hz: high_frac={high_frac:.3f}>0, delta_frac={delta_frac:.3f}<0.99")

    # (2) Broadband white noise: no single ~1 Hz spectral spike should dominate
    # (a b,a corner instability rings into a huge 1 Hz peak).
    torch.manual_seed(0)
    noise = torch.randn(1, T)
    out_n = view._lfs_frames(noise, FS)[0, 0].numpy()
    fn, pxxn = scipy.signal.welch(out_n, fs=64.0, nperseg=min(256, out_n.shape[0]))
    peak_idx = int(np.argmax(pxxn))
    peak_frac = pxxn[peak_idx] / pxxn.sum()
    assert peak_frac < 0.5, (
        f"a single {fn[peak_idx]:.2f} Hz bin holds {peak_frac:.2f} of power — "
        "b,a-instability fake-sine signature"
    )
    print(f"[check] OK broadband noise: peak bin {fn[peak_idx]:.2f} Hz "
          f"holds {peak_frac:.3f} < 0.5 of power")


def test_robust_z_f1() -> None:
    """_fit_session_stats on (C,1,T) frames → per-(electrode) robust-z, median/sigma (C,1)."""
    view = _make_view()
    wave = _sinusoid_mix()
    frames = view._lfs_frames(wave, FS)  # (C, 1, T)
    norm = view._fit_session_stats(frames)
    assert isinstance(norm, SessionRobustZNormalizer)
    assert norm.median is not None and norm.sigma is not None
    assert tuple(norm.median.shape[:2]) == (C, 1)
    assert tuple(norm.sigma.shape[:2]) == (C, 1)
    print(f"[check] OK robust-z median/sigma shape[:2]=({C},1)")


def test_construction_passes_validators(tmp_path) -> None:
    """LowLfpView constructs with the dummy-but-valid band params + a spec cache dir."""
    view = LowLfpView(
        front_end="band",
        band_nperseg=64,
        band_hop=32,
        hop_length=32,
        band_f_lo_hz=1.0,
        band_f_hi_hz=30.0,
        spec_cache_dir=str(tmp_path),
        session_robust_z=True,
    )
    assert view.front_end == "band"
    assert view._winsor_band_name() is None
    # Namespace carries the -lfp marker so it never collides with an |STFT| band dir.
    assert view._spec_cache_namespace().endswith("-lfp")
    print("[check] OK construction passes validators; namespace ends with -lfp")
