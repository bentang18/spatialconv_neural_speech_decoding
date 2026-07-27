"""The reduction is a 90 GB -> 20 MB funnel, so a wrong slice is unfalsifiable downstream.

Everything here pins the two places that can silently produce a plausible-looking wrong
answer: which columns are the HGA band (the taps do NOT share a layout), and whether the
trial averages are the means they claim to be.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.neuroprobe.viz_reduce import _band_slice, _halves, reduce_session

BL = (4, 16, 32)      # band_lengths at the 1 s board window
FD = (3, 5, 8)        # per-band frequency bins -> enc0 width 4*3 + 16*5 + 32*8 = 348


def test_band_slice_encoder_tap_is_the_trailing_k_block() -> None:
    # encoder taps are (k_full, d), k band-major [SLOW; MID; HGA] -> HGA is k in [20, 52)
    cols, t, c = _band_slice("enc12", "hga", BL, FD, 52 * 256)
    assert (t, c) == (32, 256)
    np.testing.assert_array_equal(cols, np.arange(20 * 256, 52 * 256))


def test_band_slice_encoder_tap_earlier_bands() -> None:
    cols, t, c = _band_slice("enc12", "slow", BL, FD, 52 * 256)
    assert (t, c) == (4, 256)
    np.testing.assert_array_equal(cols, np.arange(0, 4 * 256))
    cols, t, c = _band_slice("enc12", "mid", BL, FD, 52 * 256)
    assert (t, c) == (16, 256)
    np.testing.assert_array_equal(cols, np.arange(4 * 256, 20 * 256))


def test_band_slice_enc0_uses_running_offset_of_unequal_blocks() -> None:
    # enc0 blocks are (T_b, F_b) and F_b differs per band, so the offset is NOT k-major
    cols, t, c = _band_slice("enc0", "hga", BL, FD, 348)
    assert (t, c) == (32, 8)
    off = 4 * 3 + 16 * 5
    np.testing.assert_array_equal(cols, np.arange(off, off + 32 * 8))
    assert cols[-1] == 347


def test_band_slice_enc0_without_band_fdims_refuses_rather_than_guesses() -> None:
    with pytest.raises(KeyError, match="band_fdims"):
        _band_slice("enc0", "hga", BL, None, 348)


def test_band_slice_enc0_checks_the_width_it_was_told() -> None:
    with pytest.raises(ValueError, match="!= stored width"):
        _band_slice("enc0", "hga", BL, FD, 349)


def test_band_slice_rejects_width_that_is_not_a_multiple_of_k_full() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        _band_slice("enc12", "hga", BL, FD, 52 * 256 + 1)


def test_halves_are_disjoint_interleaved_and_cover_the_input() -> None:
    idx = np.array([9, 1, 5, 3, 7, 2])
    h0, h1 = _halves(idx)
    np.testing.assert_array_equal(h0, [1, 3, 7])   # sorted 1,2,3,5,7,9 -> alternating
    np.testing.assert_array_equal(h1, [2, 5, 9])
    assert not set(h0) & set(h1)
    assert set(h0) | set(h1) == set(idx)


def _synthetic(tmp_path, n=40, n_p=3, d=4):
    """Feature value == trial index, so every average has a closed form."""
    k_full = sum(BL)
    feat = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1) * torch.ones(1, n_p, k_full * d)
    y = np.full(n, np.nan)
    y[:20] = 0.0          # class 0 = trials 0..19
    y[20:] = 1.0          # class 1 = trials 20..39
    rec = {
        "subject_id": 7, "trial_id": 1, "ckpt_tag": "t",
        "present_parcels": np.array([4, 6, 7], dtype=np.int64),
        "parcel_canon": np.array([4, 4, 6, 7, 7, 7], dtype=np.int64),
        "band_lengths": BL, "band_fdims": FD,
        "feats": {"enc12": {"raw": feat.to(torch.float16)}},
        "clip_starts": np.arange(n, dtype=float),
        "labels": {"onset": y},
        "ws_split": {}, "cs_split": {}, "n_windows": n,
    }
    p = str(tmp_path / "enc_s7_t1_x.pt")
    torch.save(rec, p)
    return p, n, n_p, d


def test_reduce_session_class_means_are_the_actual_means(tmp_path) -> None:
    p, n, n_p, d = _synthetic(tmp_path)
    out = reduce_session(p, taps=("enc12",), tasks=("onset",), band="hga",
                         chunk=7, verbose=False)  # chunk deliberately not a divisor of n
    assert out["enc12/shape"].tolist() == [n_p, 32, d]
    # class 0 = trials 0..19 -> mean 9.5, class 1 = 20..39 -> mean 29.5
    np.testing.assert_allclose(out["enc12/onset/c0/all"], 9.5, rtol=1e-5)
    np.testing.assert_allclose(out["enc12/onset/c1/all"], 29.5, rtol=1e-5)
    assert int(out["n/onset/c0/all"]) == 20 and int(out["n/onset/c1/all"]) == 20


def test_reduce_session_halves_average_their_own_trials(tmp_path) -> None:
    p, _, _, _ = _synthetic(tmp_path)
    out = reduce_session(p, taps=("enc12",), tasks=("onset",), band="hga",
                         chunk=256, verbose=False)
    # class 0 trials 0..19 -> h0 = evens (mean 9), h1 = odds (mean 10)
    np.testing.assert_allclose(out["enc12/onset/c0/h0"], 9.0, rtol=1e-5)
    np.testing.assert_allclose(out["enc12/onset/c0/h1"], 10.0, rtol=1e-5)
    assert int(out["n/onset/c0/h0"]) == 10 and int(out["n/onset/c0/h1"]) == 10
    # and the halves must average back to the grand mean
    mid = 0.5 * (out["enc12/onset/c0/h0"] + out["enc12/onset/c0/h1"])
    np.testing.assert_allclose(mid, out["enc12/onset/c0/all"], rtol=1e-5)


def test_reduce_session_column_moments_recover_mean_and_variance(tmp_path) -> None:
    p, n, _, _ = _synthetic(tmp_path)
    out = reduce_session(p, taps=("enc12",), tasks=("onset",), band="hga",
                         chunk=9, verbose=False)
    vals = np.arange(n, dtype=np.float64)
    np.testing.assert_allclose(out["enc12/col_sum"], vals.mean(), rtol=1e-4)
    var = out["enc12/col_sq"] - np.square(out["enc12/col_sum"])
    np.testing.assert_allclose(var, vals.var(), rtol=1e-3)


def test_reduce_session_parcel_counts_partition_the_montage(tmp_path) -> None:
    p, _, _, _ = _synthetic(tmp_path)
    out = reduce_session(p, taps=("enc12",), tasks=("onset",), band="hga",
                         chunk=256, verbose=False)
    np.testing.assert_array_equal(out["parcel_counts"], [2, 1, 3])
    assert out["parcel_counts"].sum() == 6


def test_band_fdims_override_only_fills_a_missing_key(tmp_path) -> None:
    p, _, _, _ = _synthetic(tmp_path)
    rec = torch.load(p, map_location="cpu", weights_only=False)
    del rec["band_fdims"]
    rec["feats"]["enc0"] = {"raw": torch.zeros(rec["n_windows"], 3, 348, dtype=torch.float16)}
    q = str(tmp_path / "enc_s7_t2_x.pt")
    torch.save(rec, q)

    with pytest.raises(KeyError, match="band_fdims"):
        reduce_session(q, taps=("enc0",), tasks=("onset",), band="hga", chunk=256,
                       verbose=False)

    out = reduce_session(q, taps=("enc0",), tasks=("onset",), band="hga", chunk=256,
                         verbose=False, band_fdims_override=(3, 5, 8))
    assert out["enc0/shape"].tolist() == [3, 32, 8]


def test_band_fdims_override_that_does_not_match_the_width_raises(tmp_path) -> None:
    p, _, _, _ = _synthetic(tmp_path)
    rec = torch.load(p, map_location="cpu", weights_only=False)
    del rec["band_fdims"]
    rec["feats"]["enc0"] = {"raw": torch.zeros(rec["n_windows"], 3, 348, dtype=torch.float16)}
    q = str(tmp_path / "enc_s7_t3_x.pt")
    torch.save(rec, q)
    with pytest.raises(ValueError, match="!= stored width"):
        reduce_session(q, taps=("enc0",), tasks=("onset",), band="hga", chunk=256,
                       verbose=False, band_fdims_override=(3, 5, 9))


def test_every_band_slice_is_contiguous_so_the_fast_path_is_the_only_path() -> None:
    """The reduction slices instead of fancy-indexing when cols are contiguous. If a layout
    ever stopped being contiguous the slow path would silently take over, so pin it."""
    for tap, width in (("enc12", 52 * 256), ("enc0", 348)):
        for band in ("slow", "mid", "hga"):
            cols, _, _ = _band_slice(tap, band, BL, FD, width)
            assert cols[-1] - cols[0] == len(cols) - 1, f"{tap}/{band} is not contiguous"


def test_contiguous_and_fancy_paths_agree(tmp_path) -> None:
    p, _, _, _ = _synthetic(tmp_path)
    rec = torch.load(p, map_location="cpu", weights_only=False)
    # make the feature vary across columns so a wrong slice cannot pass by symmetry
    n, n_p, w = rec["n_windows"], 3, sum(BL) * 4
    rec["feats"]["enc12"]["raw"] = (
        torch.arange(w, dtype=torch.float32).reshape(1, 1, w)
        + torch.arange(n, dtype=torch.float32).reshape(n, 1, 1) * 1000.0
    ).expand(n, n_p, w).contiguous().to(torch.float16)
    q = str(tmp_path / "enc_s7_t4_x.pt")
    torch.save(rec, q)
    out = reduce_session(q, taps=("enc12",), tasks=("onset",), band="hga", chunk=8,
                         verbose=False)
    # class 0 = trials 0..19 (mean index 9.5) and HGA cols start at k=20 -> column 20*4=80
    got = out["enc12/onset/c0/all"]
    assert got.shape == (3, 32, 4)
    expected_first = 9.5 * 1000.0 + 80.0
    np.testing.assert_allclose(got[0, 0, 0], expected_first, rtol=1e-3)
    np.testing.assert_allclose(got[0, 0, 1], expected_first + 1, rtol=1e-3)
