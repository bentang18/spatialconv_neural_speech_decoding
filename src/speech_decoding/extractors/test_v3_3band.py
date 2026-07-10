"""v14_converged_v3 3-band frontend geometry (Phase 0).

The v3 front-end is 3 multi-resolution |STFT| magnitude bands broadcast to a
common 32 Hz token clock (memo project-v14-converged-v3-sensor-architecture):

    SLOW  N=1024 hop=512  2–14 Hz   → k1..k7  = 7 bins   (NEW: mag, was cartesian 2–12)
    MID   N=256  hop=128  16–56 Hz  → k2..k7  = 6 bins   (= STFT_3BAND_BETA, reused)
    HGA   N=128  hop=64   64–160 Hz → k4..k10 = 7 bins   (= STFT_2BAND_HGA, reused)

Total 20 bins → Linear(20→256). Only SLOW is a new dict; MID/HGA are reused
verbatim (defining new same-param dicts would collide in _WINSOR_BAND_TAG). All
three are magnitude (band_channelization default "mag"); cutoffs land on clean
integer bin centers.
"""

from __future__ import annotations

from speech_decoding.extractors.view import (
    STFT_2BAND_HGA,
    STFT_3BAND_BETA,
    STFT_V3_SLOW,
    _DEFAULT_BAND_CHANNELIZATION,
    _WINSOR_BAND_TAG,
    _band_tag_key,
    _stft_band_k_range,
)

# v3 band → (nperseg, f_lo, f_hi, expected inclusive k0, k1, n_bins)
V3_SLOW_EXPECT = (1024, 2.0, 14.0, 1, 7, 7)
V3_MID_EXPECT = (256, 16.0, 56.0, 2, 7, 6)
V3_HGA_EXPECT = (128, 64.0, 160.0, 4, 10, 7)


def _check(band: dict, expect: tuple) -> None:
    nperseg, f_lo, f_hi, k0_e, k1_e, nb_e = expect
    assert int(band["band_nperseg"]) == nperseg
    assert float(band["band_f_lo_hz"]) == f_lo
    assert float(band["band_f_hi_hz"]) == f_hi
    k0, k1 = _stft_band_k_range(f_lo, f_hi, nperseg=nperseg, sample_rate=2048)
    assert (k0, k1) == (k0_e, k1_e), f"{band}: got k[{k0},{k1}] want [{k0_e},{k1_e}]"
    assert k1 - k0 + 1 == nb_e


def test_slow_band_is_new_mag_2_14_seven_bins() -> None:
    _check(STFT_V3_SLOW, V3_SLOW_EXPECT)
    # NEW band is magnitude, not the legacy cartesian 2–12 SLOW.
    assert STFT_V3_SLOW.get("band_channelization", _DEFAULT_BAND_CHANNELIZATION) == "mag"


def test_mid_reuses_beta_six_bins() -> None:
    _check(STFT_3BAND_BETA, V3_MID_EXPECT)


def test_hga_reuses_2band_hga_seven_bins() -> None:
    _check(STFT_2BAND_HGA, V3_HGA_EXPECT)


def test_total_is_twenty_bins() -> None:
    total = sum(e[5] for e in (V3_SLOW_EXPECT, V3_MID_EXPECT, V3_HGA_EXPECT))
    assert total == 20


def test_slow_winsor_tag_registered_and_no_collision() -> None:
    # The new SLOW key (1024, "mag", 14) must be a distinct winsor tag — it must
    # NOT alias the legacy cartesian SLOW (1024,"cartesian",12) or LFS (1024,"mag",56).
    key = _band_tag_key(STFT_V3_SLOW)
    assert key == (1024, "mag", 14)
    assert key in _WINSOR_BAND_TAG
    # every registered band key is unique (the reuse discipline holds).
    keys = list(_WINSOR_BAND_TAG.keys())
    assert len(keys) == len(set(keys))
    # MID/HGA reuse keeps the existing tags.
    assert _WINSOR_BAND_TAG[_band_tag_key(STFT_3BAND_BETA)] == "beta"
    assert _WINSOR_BAND_TAG[_band_tag_key(STFT_2BAND_HGA)] == "hga"
