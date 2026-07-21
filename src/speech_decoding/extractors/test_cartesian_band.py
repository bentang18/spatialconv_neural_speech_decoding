"""Tests for the 3-band STFT (2/2/2) frontend cache machinery.

Spec: ``reports/fe_3stft_2of2of2_spec_2026_06_17.md``. Two things are guarded:

1. **The 38-token ladder geometry** (§2) — slow/beta_lowg/HG bin selection,
   real ``torch.stft(center=True)`` frame counts, ``n_patches`` token counts,
   and the window-tiling invariant ``tk·hop == N``. Reproduced from the repo's
   OWN bin selector (``_stft_band_k_range``), not hard-coded.

2. **The slow-band Cartesian (Re/Im) cache path** (§4) — the slow band stores
   ``[Re ++ Im]`` (no magnitude; |z|=√(Re²+Im²) is redundant), fits the
   ``.stats.npz`` scale on |STFT| MAD shared across Re/Im, and applies a
   scale-only transform ``(Re/σ_p, Im/σ_p)`` that preserves the phase angle.
   The beta/HG magnitude bands are untouched.

Hermetic: pure band math + view machinery on a synthetic clip. No BT data, no
DCC, no cache I/O.
"""

from __future__ import annotations

import torch

from speech_decoding.extractors.normalize import SCALE_TO_SIGMA
from speech_decoding.extractors.view import (
    STFT_2BAND_HGA,
    STFT_2BAND_LFS,
    STFT_3BAND_BETA,
    STFT_3BAND_HG,
    STFT_3BAND_SLOW,
    STFT_V3_HGA,
    _WINSOR_BAND_TAG,
    MultiStftView,
    _single_stft_raw_view,
    _single_stft_raw_view_chunked,
    _stft_band_k_range,
)

FS = 2048
CLIP = 2048  # 1 s @ 2048 Hz


def _n_patches(n_bins: int, kernel: int) -> int:
    return 0 if n_bins < kernel else (n_bins - kernel) // kernel + 1


def _real_frames(n_fft: int, hop: int) -> int:
    spec = torch.stft(
        torch.zeros(CLIP), n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=torch.hann_window(n_fft), return_complex=True, center=True,
    )
    return spec.shape[-1]


# band: (name, N, hop, lo, hi, fk, tk, channels, expected_tokens, expected_bins)
_BANDS = [
    ("slow", 1024, 512, 2.0, 12.0, 2, 2, 2, 6, [2, 4, 6, 8, 10, 12]),
    ("beta_lowg", 256, 128, 16.0, 56.0, 3, 2, 1, 16, [16, 24, 32, 40, 48, 56]),
    ("HG", 128, 64, 64.0, 192.0, 9, 2, 1, 16,
     [64, 80, 96, 112, 128, 144, 160, 176, 192]),
]


def test_3stft_ladder_geometry_totals_38_tokens() -> None:
    """slow 6 + beta 16 + HG 16 = 38 tokens; bins + tk·hop==N reproduced from src."""
    total = 0
    for name, N, hop, lo, hi, fk, tk, _ch, exp_tok, exp_bins in _BANDS:
        df = FS / N
        k0, k1 = _stft_band_k_range(lo, hi, nperseg=N, sample_rate=FS)
        bins = [round(k * df, 3) for k in range(k0, k1 + 1)]
        assert bins == exp_bins, f"{name}: bins {bins} != {exp_bins}"
        n_bins = k1 - k0 + 1
        frames = _real_frames(N, hop)
        tok = _n_patches(n_bins, fk) * _n_patches(frames, tk)
        assert tok == exp_tok, f"{name}: tokens {tok} != {exp_tok}"
        assert tk * hop == N, f"{name}: window-tiling tk·hop {tk * hop} != N {N}"
        total += tok
    assert total == 38


def test_beta_lo_13_and_16_select_same_first_bin() -> None:
    """Designer STARTING_POINT lo=13 and spec table lo=16 both snap to k0=2 (16 Hz)."""
    k0_13, _ = _stft_band_k_range(13.0, 56.0, nperseg=256, sample_rate=FS)
    k0_16, _ = _stft_band_k_range(16.0, 56.0, nperseg=256, sample_rate=FS)
    assert k0_13 == k0_16 == 2


def test_cartesian_view_is_re_im_concat_exact() -> None:
    """Slow-band cartesian view = [Re(6) ++ Im(6)] on the freq axis, exact vs torch.stft."""
    torch.manual_seed(0)
    x = torch.randn(4, CLIP)
    k0, k1 = _stft_band_k_range(2.0, 12.0, nperseg=1024, sample_rate=FS)
    n = k1 - k0 + 1
    cart = _single_stft_raw_view(
        x, sample_rate=FS, nperseg=1024, hop_length=512, k0=k0, k1=k1,
        log_eps=1e-6, cartesian=True,
    )
    spec = torch.stft(
        x, n_fft=1024, hop_length=512, win_length=1024,
        window=torch.hann_window(1024), return_complex=True, center=True,
    )
    band = spec[..., k0:k1 + 1, :]
    expected = torch.cat([band.real, band.imag], dim=-2)
    assert n == 6 and cart.shape == (4, 12, spec.shape[-1])
    assert torch.allclose(cart, expected)


def test_mag_view_unchanged_by_cartesian_flag() -> None:
    """cartesian=False is the original |STFT| path, byte-identical."""
    torch.manual_seed(0)
    x = torch.randn(4, CLIP)
    k0, k1 = _stft_band_k_range(2.0, 12.0, nperseg=1024, sample_rate=FS)
    mag = _single_stft_raw_view(
        x, sample_rate=FS, nperseg=1024, hop_length=512, k0=k0, k1=k1,
        log_eps=1e-6, cartesian=False,
    )
    spec = torch.stft(
        x, n_fft=1024, hop_length=512, win_length=1024,
        window=torch.hann_window(1024), return_complex=True, center=True,
    )
    assert torch.allclose(mag, spec[..., k0:k1 + 1, :].abs())


def test_native_rate_slow_equals_decimated_32hz() -> None:
    """Native SLOW (hop=512 → 4 Hz) == 32 Hz cache (hop=64) then ::8.

    The load-bearing invariant of the native-rate rebake: PerBandStem decimates
    arm0's 32 Hz SLOW cache ::8 (stem.py:95,164), so the model sees SLOW@4 Hz.
    Extracting natively at hop=512 hits the SAME window centers (k·512) with the
    SAME N=1024 window ⇒ bit-identical |STFT| at the retained frames. Skips the
    decimate + saves 8× storage. Same for MID ::2 below."""
    torch.manual_seed(0)
    x = torch.randn(4, 8192)  # 4 s @ 2048 Hz — many SLOW frames
    k0, k1 = _stft_band_k_range(2.0, 14.0, nperseg=1024, sample_rate=FS)
    dense = _single_stft_raw_view(
        x, sample_rate=FS, nperseg=1024, hop_length=64, k0=k0, k1=k1,
        log_eps=1e-6, cartesian=False,
    )
    native = _single_stft_raw_view(
        x, sample_rate=FS, nperseg=1024, hop_length=512, k0=k0, k1=k1,
        log_eps=1e-6, cartesian=False,
    )
    dec = dense[..., ::8]
    assert dec.shape == native.shape, f"[check] SLOW frames {dec.shape} != {native.shape}"
    assert torch.allclose(dec, native, atol=1e-5), "[check] native SLOW != decimated 32 Hz"


def test_native_rate_mid_equals_decimated_32hz() -> None:
    """Native MID (hop=128 → 16 Hz) == 32 Hz cache (hop=64) then ::2 (PerBandStem MID stride 2)."""
    torch.manual_seed(1)
    x = torch.randn(4, 8192)
    k0, k1 = _stft_band_k_range(16.0, 56.0, nperseg=256, sample_rate=FS)
    dense = _single_stft_raw_view(
        x, sample_rate=FS, nperseg=256, hop_length=64, k0=k0, k1=k1,
        log_eps=1e-6, cartesian=False,
    )
    native = _single_stft_raw_view(
        x, sample_rate=FS, nperseg=256, hop_length=128, k0=k0, k1=k1,
        log_eps=1e-6, cartesian=False,
    )
    dec = dense[..., ::2]
    assert dec.shape == native.shape, f"[check] MID frames {dec.shape} != {native.shape}"
    assert torch.allclose(dec, native, atol=1e-5), "[check] native MID != decimated 32 Hz"


def test_chunked_cartesian_equals_unchunked() -> None:
    """The streaming (chunked) cartesian view matches the whole-clip view."""
    torch.manual_seed(0)
    x = torch.randn(4, CLIP)
    k0, k1 = _stft_band_k_range(2.0, 12.0, nperseg=1024, sample_rate=FS)
    cart = _single_stft_raw_view(
        x, sample_rate=FS, nperseg=1024, hop_length=512, k0=k0, k1=k1,
        log_eps=1e-6, cartesian=True,
    )
    chunked = _single_stft_raw_view_chunked(
        x, sample_rate=FS, nperseg=1024, hop_length=512, k0=k0, k1=k1,
        log_eps=1e-6, cartesian=True, chunk_frames=2,
    )
    assert torch.allclose(chunked, cart, atol=1e-5)


def test_cartesian_fit_stats_median0_sigma_shared_mad_of_mag() -> None:
    """Slow-band fit: median=0, sigma=[s++s], s=1.4826·MAD(|STFT|) per bin (shared Re/Im)."""
    torch.manual_seed(0)
    x = torch.randn(4, CLIP)
    slow = MultiStftView(front_end="band", hop_length=512, session_robust_z=True,
                         **STFT_3BAND_SLOW)
    k0, k1 = _stft_band_k_range(2.0, 12.0, nperseg=1024, sample_rate=FS)
    frames = _single_stft_raw_view(
        x, sample_rate=FS, nperseg=1024, hop_length=512, k0=k0, k1=k1,
        log_eps=1e-6, cartesian=True,
    )  # (4, 12, T) = [Re(6) ++ Im(6)]
    norm = slow._fit_session_stats(frames)
    assert norm.median is not None and norm.sigma is not None
    assert torch.allclose(norm.median, torch.zeros_like(norm.median))
    assert norm.sigma.shape == (4, 12, 1)
    assert torch.allclose(norm.sigma[:, :6], norm.sigma[:, 6:])  # shared Re/Im scale
    re, im = frames[:, :6], frames[:, 6:]
    mag = torch.sqrt(re * re + im * im)
    med = mag.median(dim=-1, keepdim=True).values
    mad = (mag - med).abs().median(dim=-1, keepdim=True).values
    assert torch.allclose(norm.sigma[:, :6], SCALE_TO_SIGMA * mad, atol=1e-5)


def test_cartesian_transform_scale_only_preserves_phase() -> None:
    """transform = (Re/σ, Im/σ): no centering, phase angle invariant."""
    torch.manual_seed(0)
    x = torch.randn(4, CLIP)
    slow = MultiStftView(front_end="band", hop_length=512, session_robust_z=True,
                         **STFT_3BAND_SLOW)
    k0, k1 = _stft_band_k_range(2.0, 12.0, nperseg=1024, sample_rate=FS)
    frames = _single_stft_raw_view(
        x, sample_rate=FS, nperseg=1024, hop_length=512, k0=k0, k1=k1,
        log_eps=1e-6, cartesian=True,
    )
    norm = slow._fit_session_stats(frames)
    z = norm.transform(frames)
    re, im = frames[:, :6], frames[:, 6:]
    assert norm.sigma is not None
    sig = norm.sigma[:, :6]
    assert torch.allclose(z, torch.cat([re / sig, im / sig], dim=-2), atol=1e-4)
    ang_in = torch.atan2(im, re)
    ang_out = torch.atan2(z[:, 6:], z[:, :6])
    assert torch.allclose(ang_in, ang_out, atol=1e-4)


def test_mag_bands_fit_unchanged_and_bin_counts() -> None:
    """beta/HG magnitude bands keep standard robust-z (median≠0); f-bins 6/9; slow 12."""
    torch.manual_seed(0)
    x = torch.randn(4, CLIP)
    beta = MultiStftView(front_end="band", hop_length=128, session_robust_z=True,
                         **STFT_3BAND_BETA)
    kb0, kb1 = beta._band_bins()
    beta_frames = _single_stft_raw_view(
        x, sample_rate=FS, nperseg=256, hop_length=128, k0=kb0, k1=kb1,
        log_eps=1e-6, cartesian=False,
    )
    bnorm = beta._fit_session_stats(beta_frames)
    assert bnorm.median is not None
    assert not torch.allclose(bnorm.median, torch.zeros_like(bnorm.median))
    assert beta._expected_raw_f_bins() == (kb1 - kb0 + 1) == 6
    hg = MultiStftView(front_end="band", hop_length=64, session_robust_z=True,
                       **STFT_3BAND_HG)
    assert hg._expected_raw_f_bins() == 9
    slow = MultiStftView(front_end="band", hop_length=512, session_robust_z=True,
                         **STFT_3BAND_SLOW)
    assert slow._expected_raw_f_bins() == 12  # 2·6 (Re++Im)


def test_cartesian_and_mag_have_distinct_cache_namespaces() -> None:
    """Same geometry, different channelization → different spec-cache dir (no collision)."""
    common = dict(front_end="band", hop_length=512, band_nperseg=1024,
                  band_hop=512, band_f_lo_hz=2.0, band_f_hi_hz=12.0,
                  spec_cache_dir="/tmp/x")
    mag = MultiStftView(**common, band_channelization="mag")
    cart = MultiStftView(**common, band_channelization="cartesian")
    assert mag._spec_cache_namespace() != cart._spec_cache_namespace()


def test_cartesian_validators() -> None:
    """cartesian requires front_end='band' and rejects apply_log=True."""
    import pytest

    with pytest.raises(ValueError):
        MultiStftView(front_end="raw", band_channelization="cartesian")
    with pytest.raises(ValueError):
        MultiStftView(front_end="band", hop_length=512, apply_log=True,
                      **STFT_3BAND_SLOW)


# ------------------------------------------------------- per-band winsor tag
def test_winsor_band_name_per_band() -> None:
    """Each band view reports its canonical winsor-band tag, keyed off
    ``(band_nperseg, band_channelization, f_hi)``; a non-band (raw) view has no
    tag → global scalar cap."""
    slow = MultiStftView(front_end="band", hop_length=512, **STFT_3BAND_SLOW)
    beta = MultiStftView(front_end="band", hop_length=128, **STFT_3BAND_BETA)
    hg = MultiStftView(front_end="band", hop_length=64, **STFT_3BAND_HG)
    lfs = MultiStftView(front_end="band", hop_length=512, **STFT_2BAND_LFS)
    hga = MultiStftView(front_end="band", hop_length=64, **STFT_2BAND_HGA)
    vhga = MultiStftView(front_end="band", hop_length=16, **STFT_V3_HGA)
    assert slow._winsor_band_name() == "slow"
    assert beta._winsor_band_name() == "beta"
    assert hg._winsor_band_name() == "hg"
    assert lfs._winsor_band_name() == "lfs"
    assert hga._winsor_band_name() == "hga"
    # fine-HGA (N=64/hop=16) is the SAME physical 64-160 band as the 32 Hz HGA
    # (N=128), so it REUSES the "hga" winsor cap by design (distinct tuple key,
    # same value) — the |z| family is the same; parity spec keeps HGA winsor 20.
    assert vhga._winsor_band_name() == "hga"
    assert MultiStftView(front_end="raw")._winsor_band_name() is None


def test_winsor_band_tags_do_not_collide() -> None:
    """The physical bands must map to DISTINCT tags. ``band_nperseg`` alone
    collides (2-band LFS 1024 == 3STFT SLOW == v3 SLOW; HGA 128 == HG); the
    ``(nperseg, channelization, f_hi)`` key separates them (v3 SLOW is 1024/mag/14,
    distinct from LFS 1024/mag/56 and cartesian SLOW 1024/cart/12). Guard against a
    future band whose tuple aliases an existing band — that would silently clamp
    the wrong band's training signal."""
    tags = list(_WINSOR_BAND_TAG.values())
    # "hga" appears TWICE by design: the 32 Hz HGA (128,mag,160) and the fine-HGA
    # (64,mag,160) are the SAME physical 64-160 band at two window/hop resolutions,
    # so they share the "hga" cap. The guard that matters is KEY uniqueness (no two
    # physical bands aliasing one tuple → silently clamping the wrong signal), which
    # holds: 7 DISTINCT keys. Value-sharing across resolutions of one band is fine.
    assert sorted(tags) == ["beta", "hg", "hga", "hga", "lfs", "slow", "vslow"]
    assert len(_WINSOR_BAND_TAG) == len(set(_WINSOR_BAND_TAG)) == 7
    # The two real-world nperseg aliases resolve to different tags via the tuple.
    lfs = MultiStftView(front_end="band", hop_length=512, **STFT_2BAND_LFS)
    slow = MultiStftView(front_end="band", hop_length=512, **STFT_3BAND_SLOW)
    assert lfs.band_nperseg == slow.band_nperseg == 1024
    assert lfs._winsor_band_name() != slow._winsor_band_name()
    hga = MultiStftView(front_end="band", hop_length=64, **STFT_2BAND_HGA)
    hg = MultiStftView(front_end="band", hop_length=64, **STFT_3BAND_HG)
    assert hga.band_nperseg == hg.band_nperseg == 128
    assert hga._winsor_band_name() != hg._winsor_band_name()


def test_winsor_band_tag_is_cache_neutral() -> None:
    """The band tag is DERIVED from band_nperseg (already a cache-key field), so
    it adds nothing to the spec-cache namespace — the per-band winsor must never
    fork the multi-TB cache."""
    common = dict(front_end="band", hop_length=512, band_nperseg=1024,
                  band_hop=512, band_f_lo_hz=2.0, band_f_hi_hz=12.0,
                  spec_cache_dir="/tmp/x", band_channelization="cartesian")
    a = MultiStftView(**common)
    b = MultiStftView(**common)
    assert a._winsor_band_name() == b._winsor_band_name() == "slow"
    assert a._spec_cache_namespace() == b._spec_cache_namespace()


def test_winsor_band_reaches_fitted_normalizer(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Integration: a per-band env cap reaches the view's fitted normalizer — the
    slow band's normalizer clamps at the SLOW cap while a beta normalizer (no beta
    var here) falls back to the global cap."""
    monkeypatch.setenv("V14_SESSION_Z_WINSOR", "2500")
    monkeypatch.setenv("V14_SESSION_Z_WINSOR_SLOW", "300")
    torch.manual_seed(0)
    x = torch.randn(4, CLIP)
    slow = MultiStftView(front_end="band", hop_length=512, session_robust_z=True,
                         **STFT_3BAND_SLOW)
    ks0, ks1 = slow._band_bins()
    slow_frames = _single_stft_raw_view(
        x, sample_rate=FS, nperseg=1024, hop_length=512, k0=ks0, k1=ks1,
        log_eps=1e-6, cartesian=True,
    )
    assert slow._fit_session_stats(slow_frames).winsor == 300.0  # slow override
    beta = MultiStftView(front_end="band", hop_length=128, session_robust_z=True,
                         **STFT_3BAND_BETA)
    kb0, kb1 = beta._band_bins()
    beta_frames = _single_stft_raw_view(
        x, sample_rate=FS, nperseg=256, hop_length=128, k0=kb0, k1=kb1,
        log_eps=1e-6, cartesian=False,
    )
    assert beta._fit_session_stats(beta_frames).winsor == 2500.0  # global
