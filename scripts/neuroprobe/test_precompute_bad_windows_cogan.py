"""Tests for the Cogan guard-2 cache-reading slider.

Covers the NEW code (cache magnitudes + baked stats → the detector's ewm/n_flat
grid, and the end-to-end session scan over a synthetic cache tree). The DETECTOR
itself (_decide_bad_windows / _merge_bad_windows) is imported verbatim from
precompute_bad_windows and already tested there, so we only assert the slider
feeds it correctly and that each rule fires on an injected artifact.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import precompute_bad_windows_cogan as cog
from precompute_bad_windows import CAT_MULT_BY_BAND

# tiny v3-shaped session: 3 bands, a handful of freq bins each, a few seconds
_C = 40  # electrodes
_FB = {"v3slow": 5, "v3mid": 4, "hga": 4}  # freq bins per band
_FPS = 32.0
_CLIP_S = 1.0


def _clean_bands(n_frames, seed=0):
    """3 bands of magnitude ~ median + σ·N(0,1) so z ≈ N(0,1) (no artifact)."""
    rng = np.random.default_rng(seed)
    mags, stats = [], []
    for b in cog.BAND_NAMES:
        f = _FB[b]
        median = np.full((_C, f, 1), 10.0, np.float32)
        sigma = np.full((_C, f, 1), 2.0, np.float32)
        mag = median + sigma * rng.standard_normal((_C, f, n_frames)).astype(np.float32)
        mags.append(mag.astype(np.float32))
        stats.append((median[:, :, 0].copy(), sigma[:, :, 0].copy()))
    return mags, stats


def _ewm(mags, stats):
    return cog.compute_ewm_from_cache(
        mags, stats, cog.BAND_NAMES, flat_band=cog.FLAT_BAND,
        clip_s=_CLIP_S, fps=_FPS,
    )


def test_clean_session_has_no_bad_windows():
    mags, stats = _clean_bands(n_frames=int(5 * _FPS), seed=1)
    ewm_by_band, n_flat, n_elec, total_s, n_windows = _ewm(mags, stats)
    assert n_windows == 5
    assert set(ewm_by_band) == set(cog.BAND_NAMES)
    assert ewm_by_band["v3slow"].shape == (_C, n_windows)
    from precompute_bad_windows import _decide_bad_windows
    bad_idx, _ = _decide_bad_windows(ewm_by_band, n_flat, n_elec)
    assert bad_idx == []  # clean Gaussian z → nothing fires


def test_catastrophic_single_cell_flags_its_window():
    mags, stats = _clean_bands(n_frames=int(5 * _FPS), seed=2)
    # one electrode, one HGA bin, one frame in window 3 → z ~ 300 (> 8·q and > ABS 200)
    frame = int(3.5 * _FPS)
    mags[2][7, 1, frame] = 10.0 + 2.0 * 300.0
    ewm_by_band, n_flat, n_elec, _, _ = _ewm(mags, stats)
    from precompute_bad_windows import _decide_bad_windows
    bad_idx, decision = _decide_bad_windows(
        ewm_by_band, n_flat, n_elec, cat_mult_by_band=CAT_MULT_BY_BAND,
    )
    assert 3 in bad_idx
    assert decision["n_cat_windows"] >= 1


def test_common_mode_hot_flags_without_cat():
    # 20 windows so the 8 injected hot cells stay ~1% of the grid (P99 q_b is not
    # swamped by the injection — the self-calibration would otherwise ride up and
    # defeat the fence). Inject relative to the measured clean q_b so the level
    # sits between the 4·q hot fence and the 8·q cat fence regardless of scale.
    mags, stats = _clean_bands(n_frames=int(20 * _FPS), seed=3)
    q_mid = float(np.percentile(_ewm(mags, stats)[0]["v3mid"], 99.0))
    win = 10
    frame = int((win + 0.5) * _FPS)
    hot = np.arange(0, _C, 5)  # 8 electrodes = 20%
    mags[1][hot, 0, frame] = 10.0 + 2.0 * (6.0 * q_mid)  # ~6·q: > 4·q hot, < 8·q cat
    ewm_by_band, n_flat, n_elec, _, _ = _ewm(mags, stats)
    from precompute_bad_windows import _decide_bad_windows
    bad_idx, decision = _decide_bad_windows(
        ewm_by_band, n_flat, n_elec, cat_mult_by_band=CAT_MULT_BY_BAND,
    )
    assert win in bad_idx
    assert decision["n_hot_windows"] >= 1
    assert decision["n_cat_windows"] == 0  # 6·q < 8·q → hot, not catastrophic


def test_flat_electrodes_flag_dropout():
    mags, stats = _clean_bands(n_frames=int(5 * _FPS), seed=4)
    # freeze the slow band of 20% of electrodes to the median over window 1 → z-std ~ 0
    lo, hi = int(1 * _FPS), int(2 * _FPS)
    flat = np.arange(0, _C, 5)
    for e in flat:
        mags[0][e, :, lo:hi] = 10.0  # == median → z ≈ 0, std ≈ 0
    ewm_by_band, n_flat, n_elec, _, _ = _ewm(mags, stats)
    assert n_flat[1] >= len(flat)  # those electrodes read flat in window 1
    from precompute_bad_windows import _decide_bad_windows
    bad_idx, decision = _decide_bad_windows(ewm_by_band, n_flat, n_elec)
    assert 1 in bad_idx
    assert decision["n_flat_windows"] >= 1


def test_hga_uses_stricter_cat_multiplier_than_low_bands():
    """A cell at 7·q fires CAT in HGA (fence 6·q) but NOT in v3slow (fence 8·q)."""
    from precompute_bad_windows import _decide_bad_windows

    n_windows = 5
    base = np.ones((_C, n_windows), np.float32)  # q≈1 in every band
    n_flat = np.zeros(n_windows, np.int32)

    def one_hot(band):
        ewm = {b: base.copy() for b in cog.BAND_NAMES}
        ewm[band][0, 2] = 7.0  # single cell at 7×
        return ewm

    _, d_hga = _decide_bad_windows(one_hot("hga"), n_flat, _C, cat_mult_by_band=CAT_MULT_BY_BAND)
    _, d_slow = _decide_bad_windows(one_hot("v3slow"), n_flat, _C, cat_mult_by_band=CAT_MULT_BY_BAND)
    assert d_hga["n_cat_windows"] >= 1   # 7 > 6·q → HGA fires
    assert d_slow["n_cat_windows"] == 0  # 7 < 8·q → v3slow does not


# ---- end-to-end over a synthetic on-disk cache tree ----------------------------


def _write_band_cache(band_dir, key, ch_names, mag, median, sigma, sample_rate=32):
    band_dir.mkdir(parents=True, exist_ok=True)
    stem = band_dir / "sess"
    np.save(str(stem) + ".npy", mag)
    np.savez(str(stem) + ".stats.npz", median=median, sigma=sigma)
    (band_dir.parent / band_dir.name / "sess.json").write_text(json.dumps({
        "key": key,
        "ch_names": list(ch_names),
        "total_frames": int(mag.shape[2]),
        "sample_rate": sample_rate,
    }))


def test_scan_session_cache_end_to_end(tmp_path):
    sid, tid = 1019, 0
    key = f'{{"cls":"DCohortStudy","subject_id":{sid},"trial_id":{tid}}}_0.0_5.0'
    ch_names = [f"LA{i}" for i in range(_C)]
    n_frames = int(5 * _FPS)
    mags, stats = _clean_bands(n_frames, seed=7)
    # inject a catastrophic HGA cell at window 4
    mags[2][3, 1, int(4.5 * _FPS)] = 10.0 + 2.0 * 400.0

    spec = tmp_path / "cogan_spec_cache"
    for b, mag, (med, sig) in zip(cog.BAND_NAMES, mags, stats):
        _write_band_cache(spec / f"band_{b}", key, ch_names, mag, med, sig)

    result = cog.scan_session_cache(sid, tid, cog._band_dirs(str(spec)))
    assert result["session"] == "cogan1019_t0"
    assert result["subject_id"] == sid and result["trial_id"] == tid
    assert isinstance(result["subject_id"], int)
    assert result["n_elec"] == _C
    assert result["n_windows"] == 5
    assert any(lo <= 4.0 < hi for lo, hi in result["bad_windows_s"])

    # consumable by the v3 span reader (keys on integer subject_id/trial_id)
    out = tmp_path / "spans"
    out.mkdir()
    (out / f"{result['session']}.json").write_text(json.dumps(result))
    from speech_decoding.models.v14_converged_v3.cache_index import index_bad_windows
    spans = index_bad_windows(str(out))
    assert (sid, tid) in spans
    assert spans[(sid, tid)]  # non-empty


def test_scan_fails_loud_on_missing_session(tmp_path):
    sid, tid = 1019, 0
    key = f'{{"subject_id":{sid},"trial_id":{tid}}}_0.0_5.0'
    ch_names = [f"LA{i}" for i in range(_C)]
    mags, stats = _clean_bands(int(2 * _FPS), seed=8)
    spec = tmp_path / "spec"
    for b, mag, (med, sig) in zip(cog.BAND_NAMES, mags, stats):
        _write_band_cache(spec / f"band_{b}", key, ch_names, mag, med, sig)
    with pytest.raises(ValueError, match="expected exactly one cache entry"):
        cog.scan_session_cache(9999, 9, cog._band_dirs(str(spec)))
