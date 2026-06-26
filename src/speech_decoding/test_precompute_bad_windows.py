"""Quantitative TDD for the CLIP bad-window scan's pure decision core (#231).

The scan script (``scripts/neuroprobe/precompute_bad_windows.py``) is DCC-only —
``scan_session`` needs BT voltage — but the load-bearing logic is the PURE
``_decide_bad_windows`` per-band rule, which is testable with synthetic |z|-max
arrays. The decisive check is that going per-band (one self-calibrating ``q_b``
per band) catches a transient in a quiet band that the old POOLED ``q`` (dominated
by the loudest band) would have missed.

The script is imported by path (scripts/ is not a package) with a skip fallback
when its heavy module-level deps (mne/scipy/torch) are absent — same pattern as
``test_run_stage0_normalize``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "neuroprobe" / "precompute_bad_windows.py"
)
_MOD = None


def _mod():
    global _MOD
    if _MOD is not None:
        return _MOD
    if not _SCRIPT.exists():
        pytest.skip(f"scan script not found at {_SCRIPT}")
    spec = importlib.util.spec_from_file_location("_pbw_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # heavy module deps (mne/scipy/torch) absent
        pytest.skip(f"scan script not importable in this env: {e}")
    _MOD = mod
    return mod


# --------------------------------------------------------------- band geometry
def test_band_specs_are_the_three_3stft_bands() -> None:
    """The scan runs the 3 production bands in slow→beta→hg order with the locked
    geometry (slow 6 bins, beta 6, hg 9) — derived from STFT_3BAND_* via the bin
    selector, not hand-typed."""
    m = _mod()
    names = [name for (name, *_) in m._BAND_SPECS]
    assert names == ["slow", "beta", "hg"]
    by_name = {name: (npseg, hop, k0, k1) for (name, npseg, hop, k0, k1) in m._BAND_SPECS}
    assert by_name["slow"][:2] == (1024, 512) and by_name["slow"][3] - by_name["slow"][2] + 1 == 6
    assert by_name["beta"][:2] == (256, 128) and by_name["beta"][3] - by_name["beta"][2] + 1 == 6
    assert by_name["hg"][:2] == (128, 64) and by_name["hg"][3] - by_name["hg"][2] + 1 == 9


def test_2band_frontend_specs() -> None:
    """The --frontend 2band selector scans the converged-v2 LFS/HGA magnitude bands
    (LFS k1..k28 = 28 bins, HGA k4..k10 = 7 bins), LFS first as the dropout sentinel
    — geometry derived from STFT_2BAND_* via the bin selector, not hand-typed."""
    m = _mod()
    specs = m._make_band_specs(m._FRONTEND_BANDS["2band"])
    names = [name for (name, *_) in specs]
    assert names == ["lfs", "hga"]
    by_name = {name: (npseg, hop, k0, k1) for (name, npseg, hop, k0, k1) in specs}
    assert by_name["lfs"][:2] == (1024, 512) and by_name["lfs"][3] - by_name["lfs"][2] + 1 == 28
    assert by_name["hga"][:2] == (128, 64) and by_name["hga"][3] - by_name["hga"][2] + 1 == 7


# ------------------------------------------------- per-band q (the #231 crux)
def _quiet_loud(ne: int = 100, nw: int = 10):
    """A quiet band (q≈1) carrying a real transient + a loud uniform band (q≈100)
    with nothing wrong. Under a single POOLED q (≈100) the quiet transient is
    invisible (10 ≪ 5·100); under per-band q it fires."""
    quiet = np.ones((ne, nw), np.float32)
    quiet[:6, 3] = 10.0   # 6 electrodes hot in window 3  (10 > the slow-band hot fence)
    quiet[0, 5] = 20.0    # one catastrophic cell window 5 (20 > the slow-band cat fence)
    loud = np.full((ne, nw), 100.0, np.float32)
    return quiet, loud


def test_per_band_q_catches_quiet_band_transient() -> None:
    """Decisive #231 check: a transient in the quiet band is flagged because that
    band self-calibrates its own q_b — the loud band does not mask it. Pins the
    multipliers (5/12) so the rule is tested independent of the production defaults:
    at q=1 the cell 10 is hot-only (>5, <12) and only the cell 20 is cat (>12)."""
    m = _mod()
    quiet, loud = _quiet_loud()
    n_flat = np.zeros(quiet.shape[1], np.int32)
    bad, diag = m._decide_bad_windows(
        {"slow": quiet, "beta": loud, "hg": loud}, n_flat, quiet.shape[0],
        hot_mult=5.0, cat_mult=12.0,
    )
    assert diag["per_band"]["slow"]["q"] == pytest.approx(1.0)
    assert diag["per_band"]["beta"]["q"] == pytest.approx(100.0)
    assert bad == [3, 5]            # window 3 (hot) + window 5 (cat), both quiet-band
    assert diag["n_hot_windows"] == 1 and diag["n_cat_windows"] == 1


def test_pooled_q_would_miss_what_per_band_catches() -> None:
    """Control: collapse both bands into ONE band (= the old pooled scan). The
    pooled q is dominated by the loud band, so its fences sit far above the quiet
    transient and NOTHING is flagged — proving the per-band split is load-bearing,
    not cosmetic."""
    m = _mod()
    quiet, loud = _quiet_loud()
    pooled = np.concatenate([quiet, loud], axis=0)  # one "band" of 2·ne electrodes
    n_flat = np.zeros(quiet.shape[1], np.int32)
    bad, diag = m._decide_bad_windows({"pooled": pooled}, n_flat, pooled.shape[0])
    assert diag["per_band"]["pooled"]["q"] == pytest.approx(100.0)
    assert bad == []               # the quiet transient slips the pooled fence


# ----------------------------------------------------------------- each rule
def test_hot_rule_needs_enough_electrodes() -> None:
    """Common-mode: a window is hot only when >= max(N_FLOOR, FRAC_HOT·C) electrodes
    exceed the fence together. One lone hot electrode is NOT enough. The session is
    wide (nw=20) so the 9 elevated cells stay <1% of the band — keeping q at the
    quiet baseline (q≈1) instead of letting the transient inflate its own fence."""
    m = _mod()
    ne, nw = 100, 20                      # thresh = max(3, ceil(0.05·100)) = 5
    band = np.ones((ne, nw), np.float32)
    band[:4, 1] = 10.0                    # 4 hot < 5 → window 1 stays clean
    band[:5, 2] = 10.0                    # 5 hot == thresh → window 2 bad
    n_flat = np.zeros(nw, np.int32)
    # cat_mult=12 keeps the cell 10 below the cat fence (10 < 12·1) so this isolates
    # the HOT rule — the production cat fence (8·1) would otherwise also flag it.
    bad, diag = m._decide_bad_windows({"slow": band}, n_flat, ne, cat_mult=12.0)
    assert diag["per_band"]["slow"]["q"] == pytest.approx(1.0)  # transient didn't inflate q
    assert bad == [2]
    assert diag["hot_count_thresh"] == 5


def test_cat_rule_one_giant_cell() -> None:
    """A single cell above CAT_MULT·q poisons its window alone (no electrode-count
    floor) — distinct from the common-mode rule."""
    m = _mod()
    ne, nw = 100, 6
    band = np.ones((ne, nw), np.float32)
    band[7, 4] = 50.0                     # 1 cell, 50 > 8·1 → window 4 bad on cat
    n_flat = np.zeros(nw, np.int32)
    bad, diag = m._decide_bad_windows({"slow": band}, n_flat, ne)
    assert bad == [4]
    assert diag["n_cat_windows"] == 1 and diag["n_hot_windows"] == 0


def test_flat_rule_from_slow_band_count() -> None:
    """Dropout: a window with >= flat_count_thresh slow-band-flat electrodes is bad
    even when every amplitude is tiny (the loudest cell stays invisible)."""
    m = _mod()
    ne, nw = 100, 6
    band = np.ones((ne, nw), np.float32)  # nothing hot/cat anywhere
    n_flat = np.zeros(nw, np.int32)
    n_flat[3] = 5                         # 5 flat == thresh → window 3 bad
    n_flat[4] = 4                         # 4 flat < thresh → window 4 clean
    bad, diag = m._decide_bad_windows({"slow": band}, n_flat, ne)
    assert bad == [3]
    assert diag["n_flat_windows"] == 1 and diag["flat_count_thresh"] == 5


def test_clean_session_drops_nothing() -> None:
    """A uniform session (no transient, no dropout) yields an empty bad list — the
    rule does not fabricate artifacts."""
    m = _mod()
    ne, nw = 100, 8
    rng = np.random.default_rng(0)
    bands = {b: (1.0 + 0.01 * rng.standard_normal((ne, nw))).astype(np.float32)
             for b in ("slow", "beta", "hg")}
    bad, _ = m._decide_bad_windows(bands, np.zeros(nw, np.int32), ne)
    assert bad == []


def test_per_band_multiplier_override() -> None:
    """A per-band HOT_MULT override changes only that band's fence: raising slow's
    multiplier above the transient's ratio un-flags it, leaving other bands' fences
    untouched."""
    m = _mod()
    ne, nw = 100, 6
    band = np.ones((ne, nw), np.float32)
    band[:6, 2] = 10.0                    # ratio 10 vs q=1
    n_flat = np.zeros(nw, np.int32)
    # cat_mult=12 keeps the cell 10 under the cat fence in both calls, so only the
    # HOT-override is under test (the production cat fence 8·1 would mask it).
    base, _ = m._decide_bad_windows({"slow": band}, n_flat, ne, cat_mult=12.0)
    assert base == [2]                    # default HOT_MULT=4 → 10 > 4 → flagged
    loosened, diag = m._decide_bad_windows(
        {"slow": band}, n_flat, ne, cat_mult=12.0, hot_mult_by_band={"slow": 12.0},
    )
    assert loosened == []                 # 10 < 12 → un-flagged
    assert diag["per_band"]["slow"]["hot_mult"] == 12.0


# --------------------------------------------------------------- locked defaults
def test_production_defaults_are_locked() -> None:
    """The CLIP-aggressiveness sweep (job 48519544) locked these — guard against an
    accidental drift back to the pre-sweep 5/12 fences or a dropped abs-floor."""
    m = _mod()
    assert m.HOT_MULT == 4.0
    assert m.CAT_MULT == 8.0           # 3STFT scalar fence
    assert m.ABS_FLOOR_MAD == 200.0
    assert m.FRAC_HOT == 0.05
    assert m.N_FLOOR == 3
    # converged-v2 2-band finals (locked 2026-06-26, proposal §6): cat 8→6 on lfs/hga
    # only, via the band-keyed override; hot/frac/abs held. 3STFT keeps scalar cat8.
    assert m.CAT_MULT_BY_BAND == {"lfs": 6.0, "hga": 6.0}
    assert m.HOT_MULT_BY_BAND == {}


# --------------------------------------------------------------- absolute floor
def test_abs_floor_catches_giant_the_relative_fence_misses() -> None:
    """The q-INDEPENDENT backstop: on an artifact-inflated band (q≈100) a single
    250-MAD cell sits UNDER the relative fences (hot 4·100=400, cat 8·100=800) but
    ABOVE the absolute floor (200), so only the abs rule flags it. This is exactly
    the failure mode the floor exists for — a ballooned q hiding a real giant."""
    m = _mod()
    ne, nw = 100, 10
    band = np.full((ne, nw), 100.0, np.float32)   # inflated band → q≈100
    band[0, 3] = 250.0                             # one giant: < 8·q but > 200
    n_flat = np.zeros(nw, np.int32)
    bad, diag = m._decide_bad_windows({"slow": band}, n_flat, ne)
    assert diag["per_band"]["slow"]["q"] == pytest.approx(100.0)
    assert bad == [3]
    assert diag["n_cat_windows"] == 0 and diag["n_hot_windows"] == 0
    assert diag["n_abs_windows"] == 1 and diag["abs_floor_mad"] == 200.0
    # raising the floor above the giant un-flags it (nothing else fires)
    clean, _ = m._decide_bad_windows({"slow": band}, n_flat, ne, abs_floor_mad=300.0)
    assert clean == []


# ----------------------------------------------------------- span merging
def test_merge_contiguous_windows_into_spans() -> None:
    """Adjacent bad window indices fuse into one neural-second span; a gap splits
    them; the final partial window is capped at the session duration."""
    m = _mod()
    # windows 1-3 fuse into [5,20); window 7 is [35,40) but the session ends at 38,
    # so its tail is capped to the session duration → [35,38).
    spans = m._merge_bad_windows([1, 2, 3, 7], clip_s=5.0, total_s=38.0)
    assert spans == [(5.0, 20.0), (35.0, 38.0)]
