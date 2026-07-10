"""v14_converged_v3 3-band frontend geometry (Phase 0).

The v3 front-end is 3 multi-resolution |STFT| magnitude bands on a common 32 Hz
token clock (memo project-v14-converged-v3-sensor-architecture). UNIFORM hop=64
(fixed 2026-07-10): every band is extracted at 32 Hz natively — window sets
frequency resolution, hop sets frame rate (decoupled). No stem hold/duplication.

    SLOW  N=1024 hop=64  2–14 Hz   → k1..k7  = 7 bins   (v3-only mag)
    MID   N=256  hop=64  16–56 Hz  → k2..k7  = 6 bins   (v3-only; forked from beta)
    HGA   N=128  hop=64  64–160 Hz → k4..k10 = 7 bins   (= STFT_2BAND_HGA, already 32 Hz)

Total 20 bins → Linear(20→256). SLOW+MID are v3-exclusive dicts at hop 64; the
SHARED STFT_3BAND_BETA (hop 128) is left untouched for the 3stft ladder. MID's
winsor key (256,"mag",56) equals beta's, so it correctly reuses beta's cap with no
new registration. All three magnitude; cutoffs land on clean integer bin centers.
"""

from __future__ import annotations

from speech_decoding.extractors.view import (
    STFT_2BAND_HGA,
    STFT_3BAND_BETA,
    STFT_V3_MID,
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


def test_mid_is_v3_mag_16_56_six_bins() -> None:
    _check(STFT_V3_MID, V3_MID_EXPECT)
    assert STFT_V3_MID.get("band_channelization", _DEFAULT_BAND_CHANNELIZATION) == "mag"


def test_hga_reuses_2band_hga_seven_bins() -> None:
    _check(STFT_2BAND_HGA, V3_HGA_EXPECT)


def test_all_three_v3_bands_hop_at_64_for_a_32hz_clock() -> None:
    # THE fix: every v3 band hops at 64 samples (2048/64 = 32 Hz) so each emits one
    # frame per 31.25 ms slot natively — no 4/16 Hz extraction + stem hold.
    for band in (STFT_V3_SLOW, STFT_V3_MID, STFT_2BAND_HGA):
        assert int(band["band_hop"]) == 64


def test_shared_beta_is_untouched_at_hop_128() -> None:
    # v3 MID must be a SEPARATE dict — the shared 3stft beta keeps its hop 128 so
    # forking v3 MID to hop 64 never re-hops the cartesian 3stft ladder.
    assert int(STFT_3BAND_BETA["band_hop"]) == 128
    assert int(STFT_V3_MID["band_hop"]) == 64


def test_total_is_twenty_bins() -> None:
    total = sum(e[5] for e in (V3_SLOW_EXPECT, V3_MID_EXPECT, V3_HGA_EXPECT))
    assert total == 20


def test_v3_winsor_tags_registered_and_mid_reuses_beta_cap() -> None:
    # SLOW key (1024,"mag",14) is a distinct registered tag (not legacy cartesian
    # SLOW (1024,"cartesian",12) nor LFS (1024,"mag",56)).
    slow_key = _band_tag_key(STFT_V3_SLOW)
    assert slow_key == (1024, "mag", 14)
    assert _WINSOR_BAND_TAG[slow_key] == "vslow"
    # v3 MID is hop-forked from beta but shares beta's winsor key (hop-independent),
    # so it correctly resolves to the "beta" cap WITHOUT a second registration.
    assert _band_tag_key(STFT_V3_MID) == _band_tag_key(STFT_3BAND_BETA) == (256, "mag", 56)
    assert _WINSOR_BAND_TAG[_band_tag_key(STFT_V3_MID)] == "beta"
    # every registered key is still unique (no collision introduced).
    keys = list(_WINSOR_BAND_TAG.keys())
    assert len(keys) == len(set(keys))
    assert _WINSOR_BAND_TAG[_band_tag_key(STFT_2BAND_HGA)] == "hga"
