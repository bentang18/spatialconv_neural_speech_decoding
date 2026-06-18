"""Quantitative TDD for the CLIP-aggressiveness sweep's pure core (#236).

``scripts/neuroprobe/sweep_clip_aggressiveness.py`` pays the expensive streaming
pass once per session (DCC-only, needs BT voltage) then re-runs the pure
``_decide_bad_windows`` over a one-factor-at-a-time grid. The load-bearing logic
is ``sweep_decision`` — pure numpy on synthetic per-band |z|-max arrays — so the
marginal-drop behaviour of every lever is checked here with hand-computed
expectations, no BT.

Imported by path (scripts/ is not a package) with a skip fallback when the heavy
module-level deps (mne/scipy/torch via precompute_bad_windows) are absent — same
pattern as ``test_precompute_bad_windows``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "neuroprobe" / "sweep_clip_aggressiveness.py"
)
_MOD = None


def _mod():
    global _MOD
    if _MOD is not None:
        return _MOD
    if not _SCRIPT.exists():
        pytest.skip(f"sweep script not found at {_SCRIPT}")
    spec = importlib.util.spec_from_file_location("_sweep_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # heavy deps (mne/scipy/torch) absent
        pytest.skip(f"sweep script not importable in this env: {e}")
    _MOD = mod
    return mod


# ----------------------------------------------------------------- synthetic ewm
def _synth(n_elec: int = 100, n_windows: int = 60):
    """All-ones per-band |z|-max so q (P99) = 1.0 → hot fence = 5, cat fence = 12.
    Inject events on top to drive specific rules. n_flat all-zero (no dropout)."""
    ewm = {b: np.ones((n_elec, n_windows), np.float32)
           for b in ("slow", "beta", "hg")}
    return ewm, np.zeros(n_windows, np.int32), n_elec


def _events(ewm) -> None:
    """A fixed scenario with hand-known expected drops (see per-test asserts):
      win10/11/12: 6/4/3 electrodes hot at |z|=8 (hot, not cat) — common-mode count
      win30:       6 electrodes at |z|=4.5 (hot only below fence 5)
      win20:       one cell at 20 (always cat); win21: one at 11 (cat ≤10×q only)
      win22:       one cell at 9 (sub-cat, sub-threshold — abs-floor only)"""
    ewm["beta"][0:6, 10] = 8.0
    ewm["beta"][0:4, 11] = 8.0
    ewm["beta"][0:3, 12] = 8.0
    ewm["beta"][0:6, 30] = 4.5
    ewm["slow"][0, 20] = 20.0
    ewm["slow"][0, 21] = 11.0
    ewm["slow"][0, 22] = 9.0


# ---------------------------------------------------------- baseline consistency
def test_baseline_equals_decide_bad_windows() -> None:
    """The sweep's baseline count must be exactly what the production decision
    returns at the production fences — same code path, no drift."""
    m = _mod()
    ewm, n_flat, n_elec = _synth()
    _events(ewm)
    res = m.sweep_decision(ewm, n_flat, n_elec)
    bad_idx, _ = m._decide_bad_windows(
        ewm, n_flat, n_elec, **m.BASELINE, abs_floor_mad=float("inf")
    )
    assert res["baseline"] == len(bad_idx)
    # hand-known: win10 (6≥5 hot) + win20 (20>12 cat)
    assert res["baseline"] == 2


def test_locked_combined_is_real_union_not_ofat_sum() -> None:
    """``locked_combined`` is the production decision at the LOCKED fences
    (hot4/cat8/abs200) — a UNION, strictly ≥ baseline. Hand-known on _events:
    hot@4 fires win10 (6@8) + win30 (6@4.5); cat@8 fires win20/21/22 (20/11/9 > 8);
    beta's 8.0 cells are NOT > 8, abs@200 fires nothing → {10,20,21,22,30} = 5."""
    m = _mod()
    ewm, n_flat, n_elec = _synth()
    _events(ewm)
    res = m.sweep_decision(ewm, n_flat, n_elec)
    locked_idx, _ = m._decide_bad_windows(ewm, n_flat, n_elec, **m.LOCKED)
    assert res["locked_combined"] == len(locked_idx)
    assert res["locked_combined"] == 5
    assert res["locked_combined"] >= res["baseline"]


def test_clean_session_zero_everywhere() -> None:
    """No events → zero drops at every grid point, and the histogram is empty."""
    m = _mod()
    ewm, n_flat, n_elec = _synth()
    res = m.sweep_decision(ewm, n_flat, n_elec)
    assert res["baseline"] == 0
    for lev in m.GRID:
        assert all(v == 0 for v in res["levers"][lev].values())
    assert all(v == 0 for v in res["abs_floor"].values())
    assert all(v == 0 for v in res["common_mode_ge_k"].values())


# ----------------------------------------------------------------- lever margins
def test_hot_mult_marginal_and_monotone() -> None:
    """Lowering hot_mult past 4.5 pulls in win30 (6 electrodes at |z|=4.5): only a
    fence ≤ 4 (4.5 > 4) makes them hot. Drops are non-increasing as the fence
    rises."""
    m = _mod()
    ewm, n_flat, n_elec = _synth()
    _events(ewm)
    hm = m.sweep_decision(ewm, n_flat, n_elec)["levers"]["hot_mult"]
    assert hm == {"3.5": 3, "4": 3, "4.5": 2, "5": 2}
    seq = [hm[m._key(v)] for v in m.GRID["hot_mult"]]
    assert seq == sorted(seq, reverse=True)


def test_cat_mult_marginal_and_monotone() -> None:
    """Lowering cat_mult pulls in the single mid cells: 11 trips at ≤10×q, 9 at
    ≤8×q. Non-increasing as the cat fence rises."""
    m = _mod()
    ewm, n_flat, n_elec = _synth()
    _events(ewm)
    cm = m.sweep_decision(ewm, n_flat, n_elec)["levers"]["cat_mult"]
    assert cm == {"8": 4, "10": 3, "12": 2}
    seq = [cm[m._key(v)] for v in m.GRID["cat_mult"]]
    assert seq == sorted(seq, reverse=True)


def test_frac_hot_marginal_floored_by_n_floor() -> None:
    """Lowering frac_hot lowers the common-mode count threshold (n_elec=100):
    0.05→5, 0.04→4, 0.03→3, but 0.02→ceil(2)=2 is FLOORED back to n_floor=3 — so
    0.02 and 0.03 tie. Pulls in win11 (4 hot) then win12 (3 hot)."""
    m = _mod()
    ewm, n_flat, n_elec = _synth(n_elec=100)
    _events(ewm)
    fh = m.sweep_decision(ewm, n_flat, n_elec)["levers"]["frac_hot"]
    assert fh == {"0.02": 4, "0.03": 4, "0.04": 3, "0.05": 2}


def test_n_floor_marginal_small_array() -> None:
    """n_floor only bites when frac_hot·n_elec < n_floor. With n_elec=40,
    frac 0.05→ceil(2)=2: n_floor 2 keeps thresh 2 (a 2-electrode window drops),
    n_floor 3 raises it to 3 (that window survives)."""
    m = _mod()
    ewm, n_flat, n_elec = _synth(n_elec=40)
    ewm["beta"][0:2, 5] = 8.0  # exactly 2 electrodes hot
    nf = m.sweep_decision(ewm, n_flat, n_elec)["levers"]["n_floor"]
    assert nf == {"2": 1, "3": 0}


# --------------------------------------------------------------- absolute floor
def test_abs_floor_union_and_monotone() -> None:
    """Absolute floor = baseline ∪ (any band cell > K). Always ≥ baseline, and
    non-increasing as K rises. Custom small K probes the controlled mid cells."""
    m = _mod()
    ewm, n_flat, n_elec = _synth()
    _events(ewm)
    res = m.sweep_decision(ewm, n_flat, n_elec, abs_floor_k=[6, 8, 10, 12, 15])
    af = res["abs_floor"]
    assert af["baseline"] == 2
    # K=6 catches the |z|=8 common-mode cells + 11 + 9; K=15 catches only the 20
    assert af == {"baseline": 2, "6": 6, "8": 4, "10": 3, "12": 2, "15": 2}
    ks = [6, 8, 10, 12, 15]
    seq = [af[m._key(k)] for k in ks]
    assert all(x >= af["baseline"] for x in seq)
    assert seq == sorted(seq, reverse=True)


# ------------------------------------------------------- common-mode histogram
def test_common_mode_ge_k_counts_and_monotone() -> None:
    """ge_k[k] = #windows with ≥ k electrodes hot at the 5×q fence — the
    distribution Ben wants to see near the FRAC_HOT threshold. Hand-known from the
    6/4/3-hot windows (+ the three single-cell windows at k=1)."""
    m = _mod()
    ewm, n_flat, n_elec = _synth()
    _events(ewm)
    ge = m.sweep_decision(ewm, n_flat, n_elec)["common_mode_ge_k"]
    assert ge["1"] == 6 and ge["2"] == 3 and ge["3"] == 3
    assert ge["4"] == 2 and ge["5"] == 1 and ge["6"] == 1 and ge["7"] == 0
    seq = [ge[str(k)] for k in range(1, m.GE_K_MAX + 1)]
    assert seq == sorted(seq, reverse=True)


# --------------------------------------------------------------- collector logic
def test_aggregate_sums_cohort_and_picks_canary() -> None:
    """The collector sums drop counts across sessions (cohort) and reads the
    canary session's own counts. Two synthetic records, one of them the canary."""
    m = _mod()
    ewm, n_flat, n_elec = _synth()
    _events(ewm)
    base = m.sweep_decision(ewm, n_flat, n_elec)
    canary = dict(base, session=m.CANARY)
    other = dict(base, session="btbank1_t0")
    agg = m._aggregate([canary, other])
    assert agg["n_sessions"] == 2
    assert agg["canary_baseline"] == base["baseline"]
    # cohort = 2× a single record; canary = exactly the canary record
    hm = agg["levers"]["hot_mult"]["4"]
    assert hm["cohort"] == 2 * base["levers"]["hot_mult"]["4"]
    assert hm["canary"] == base["levers"]["hot_mult"]["4"]


def test_most_aggressive_safe_picks_smallest_at_canary_baseline() -> None:
    """The decision rule: smallest (most aggressive) lever value whose canary
    count still equals the canary baseline. Here baseline canary=0; values 4.5/5
    hold 0, 4.0/3.5 break it → answer is 4.5."""
    m = _mod()
    curve = {
        "3.5": {"cohort": 9, "canary": 2},
        "4": {"cohort": 5, "canary": 1},
        "4.5": {"cohort": 3, "canary": 0},
        "5": {"cohort": 2, "canary": 0},
    }
    assert m._most_aggressive_safe(curve, m.GRID["hot_mult"], 0) == "4.5"
    # if even the gentlest breaks the canary → 'none'
    curve_all_bad = {k: {"cohort": 1, "canary": 1} for k in ["3.5", "4", "4.5", "5"]}
    assert m._most_aggressive_safe(curve_all_bad, m.GRID["hot_mult"], 0) == "none"
