"""Tests for the FE-filterbank-sweep views (`multi_stft`, `raw_multi_stft`).

Covers the only novel logic in the sweep harness: octave bin-count math,
§6 routing derivation, byte-parity of the filterbank wrapper against the live
`_multi_stft_view`, and the raw-bins control. All synthetic — no DCC data.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import signal

from speech_decoding.extractors.view import (
    _multi_stft_view,
    multi_stft_bin_centers_hz,
)
from speech_decoding.views import (
    _FILTER_CHUNK_CH,
    _derive_multi_stft_routing,
    _multi_stft_band,
    _multi_stft_n_bins,
    _raw_multi_stft_bins,
    _resolve_multi_stft_params,
    apply_temporal_filter_inplace,
    apply_view,
    multi_stft_dead_bin_count,
)

SR = 2048
T_SAMPLES = 2048  # 1 s window
T_BINS = 1 + T_SAMPLES // 128  # center=True, hop=128 → 17


def _wave(c: int = 3, t: int = T_SAMPLES) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(c, t, generator=g)


def test_n_bins_matches_build_doc_grid():
    # (f0, cap, step) -> expected n_bins (build doc §4 grid).
    cases = [
        (2.0, 256.0, 0.5, 15),   # cell 1 default
        (4.0, 256.0, 0.5, 13),   # cell 2 drop-delta
        (2.0, 128.0, 0.5, 13),   # cell 3 trim-octave
        (2.0, 256.0, 1.0 / 3.0, 22),  # cell 4 finer-⅓
        (2.0, 256.0, 0.25, 29),  # cell 5 finer-¼
    ]
    for f0, cap, step, expected in cases:
        assert _multi_stft_n_bins(f0, cap, step) == expected, (f0, cap, step)


def test_routing_default_crossovers():
    # f0=2, step=0.5, 15 bins: centers 2..256; <32 low, 32..<152 mid, >=152 hi.
    centers = multi_stft_bin_centers_hz(n_bins=15, f0_hz=2.0, octave_step=0.5)
    routing = _derive_multi_stft_routing(centers)
    assert routing == tuple([0] * 8 + [1] * 5 + [2] * 2)


def test_band_view_shape_2d_and_3d():
    params = _resolve_multi_stft_params(None)  # cell 1 default
    out2d = _multi_stft_band(_wave(c=3), SR, params)
    assert out2d.shape == (3, 15, T_BINS)
    out3d = _multi_stft_band(_wave(c=3).unsqueeze(0), SR, params)  # (1,3,T)
    assert out3d.shape == (1, 3, 15, T_BINS)
    # batch dim is just a reshape — content identical to the 2D path.
    assert torch.allclose(out3d[0], out2d, atol=0, rtol=0)


def test_band_view_is_byte_identical_to_live_filterbank():
    """`_multi_stft_band` must reproduce `_multi_stft_view` for the same knobs.

    This is the parity contract: the sweep tunes the *actual* v14 front-end,
    not a re-implementation.
    """
    params = _resolve_multi_stft_params(None)
    data = _wave(c=4)
    centers = multi_stft_bin_centers_hz(n_bins=15, f0_hz=2.0, octave_step=0.5)
    routing = _derive_multi_stft_routing(centers)
    direct = _multi_stft_view(
        data,
        sample_rate=SR,
        hop_length=128,
        nperseg_low=1024,
        nperseg_mid=512,
        nperseg_hi=256,
        n_bins=15,
        f0_hz=2.0,
        octave_step=0.5,
        half_bw_octaves=0.5,
        routing=routing,
        log_eps=1e-6,
        apply_log=False,
    )
    via_view = _multi_stft_band(data, SR, params)
    assert via_view.shape == direct.shape
    assert torch.equal(via_view, direct)


def test_raw_bins_count_and_shape():
    params = _resolve_multi_stft_params(None)
    out = _raw_multi_stft_bins(_wave(c=3), SR, params)
    # Expected count = sum of in-band rfftfreq bins per source STFT.
    expected = 0
    for nperseg, band in (
        (1024, params["raw_low_band"]),
        (512, params["raw_mid_band"]),
        (256, params["raw_hi_band"]),
    ):
        freqs = torch.fft.rfftfreq(nperseg, d=1.0 / SR)
        expected += int(((freqs >= band[0]) & (freqs <= band[1])).sum())
    assert expected == 75  # 20 (low) + 33 (mid) + 22 (hi)
    assert out.shape == (3, expected, T_BINS)


def test_apply_view_dispatch_registers_both():
    helpers = {"multi_stft_params": None}
    data = _wave(c=2).unsqueeze(0)  # (1,2,T) — runner passes (B,C,T)
    fb = apply_view(data, "multi_stft", sampling_rate=SR, upstream_helpers=helpers)
    rb = apply_view(data, "raw_multi_stft", sampling_rate=SR, upstream_helpers=helpers)
    assert fb.shape == (1, 2, 15, T_BINS)
    assert rb.shape == (1, 2, 75, T_BINS)


def test_dead_bin_counts_match_grid():
    """Lock FE-MULTISTFT-1: constant-Q bins that fall between 2 Hz rfft bins.

    The finer-resolution cells (4: step=⅓, 5: step=¼) accumulate the most
    all-zero bins — this is the resolution-axis confound the manifest surfaces.
    """
    third = 1.0 / 3.0
    cases = [
        # (f0, cap, step, half_bw) -> expected dead bins
        ((2.0, 256.0, 0.5, 0.5), 1),    # cell 1 default
        ((4.0, 256.0, 0.5, 0.5), 0),    # cell 2 drop-delta
        ((2.0, 128.0, 0.5, 0.5), 1),    # cell 3 trim-octave
        ((2.0, 256.0, third, third), 2),  # cell 4 finer-⅓
        ((2.0, 256.0, 0.25, 0.25), 4),  # cell 5 finer-¼
        ((2.0, 256.0, 0.5, 0.75), 0),   # cell 6 smoother-0.75
        ((2.0, 256.0, 0.5, 1.0), 0),    # cell 7 smoother-1.0
    ]
    for (f0, cap, step, hbw), expected in cases:
        n = multi_stft_dead_bin_count(
            SR, {"f0_hz": f0, "cap_hz": cap, "octave_step": step, "half_bw_octaves": hbw}
        )
        assert n == expected, (f0, cap, step, hbw, n)


def test_filterbank_changes_with_half_bw():
    # half_bw is a real knob: widening the triangle changes the output.
    narrow = _multi_stft_band(_wave(c=2), SR, _resolve_multi_stft_params({"half_bw_octaves": 0.5}))
    wide = _multi_stft_band(_wave(c=2), SR, _resolve_multi_stft_params({"half_bw_octaves": 1.0}))
    assert narrow.shape == wide.shape
    assert not torch.allclose(narrow, wide)


def _ref_filtered(arr: np.ndarray, *, notch_freqs, notch_q, hpf_hz, hpf_order) -> np.ndarray:
    # Whole-array float64 reference: exactly the pre-chunking code path.
    nyq = 0.5 * SR
    out = arr.astype(np.float64)
    for f0 in notch_freqs:
        if f0 <= 0 or f0 >= nyq:
            continue
        b, a = signal.iirnotch(f0 / nyq, notch_q)
        out = np.asarray(signal.sosfiltfilt(signal.tf2sos(b, a), out, axis=-1))
    if hpf_hz > 0:
        sos = signal.butter(hpf_order, hpf_hz / nyq, btype="highpass", output="sos")
        out = np.asarray(signal.sosfiltfilt(sos, out, axis=-1))
    return out.astype(np.float32)


def test_chunked_filter_is_bit_identical_to_whole_array():
    # Channel chunking is a pure memory knob: sosfiltfilt along axis=-1 is
    # independent per channel, so filtering in _FILTER_CHUNK_CH-row blocks must
    # equal filtering the whole tensor at once. Use >2 chunks worth of channels.
    n_ch = 2 * _FILTER_CHUNK_CH + 5
    g = torch.Generator().manual_seed(7)
    base = torch.randn(n_ch, 6000, generator=g)
    notch_freqs = (60.0, 120.0, 180.0)

    ref = _ref_filtered(
        base.numpy(), notch_freqs=notch_freqs, notch_q=30.0, hpf_hz=0.5, hpf_order=4
    )
    got = base.clone()
    apply_temporal_filter_inplace(
        got, sampling_rate=SR, notch_freqs=notch_freqs, notch_q=30.0, hpf_hz=0.5, hpf_order=4
    )

    assert got.shape == base.shape
    np.testing.assert_array_equal(got.numpy(), ref)


def test_filter_noop_when_nothing_requested():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 1000, generator=g)
    before = x.clone()
    apply_temporal_filter_inplace(x, sampling_rate=SR, notch_freqs=(), hpf_hz=0.0)
    assert torch.equal(x, before)
