"""TDD for the CLIP bad-window collector's pure aggregation core.

``analyze_bad_windows.summarize_bad_windows`` reads the per-session sidecar dicts
``precompute_bad_windows.py`` emits and rolls them into cohort + per-band knee
stats. It has no I/O, so it is testable from synthetic sidecar payloads — the same
shape the scan writes (``rule.per_band[band] = {q, max_cell_max, n_*_windows, …}``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "neuroprobe" / "analyze_bad_windows.py"
)
_MOD = None


def _mod():
    global _MOD
    if _MOD is not None:
        return _MOD
    if not _SCRIPT.exists():
        pytest.skip(f"analyzer not found at {_SCRIPT}")
    spec = importlib.util.spec_from_file_location("_abw_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MOD = mod
    return mod


def _sidecar(session, n_elec, n_windows, n_bad, *, per_band, by_rule=None):
    """Build a sidecar dict in the exact shape scan_session writes."""
    by_rule = by_rule or {"hot": 0, "cat": 0, "flat": 0}
    return {
        "session": session,
        "n_elec": n_elec,
        "duration_s": n_windows * 5.0,
        "n_windows": n_windows,
        "n_bad_windows": n_bad,
        "frac_bad": (n_bad / n_windows) if n_windows else 0.0,
        "n_hot_windows": by_rule["hot"],
        "n_cat_windows": by_rule["cat"],
        "n_flat_windows": by_rule["flat"],
        "rule": {"clip_s": 5.0, "per_band": per_band},
    }


def test_cohort_frac_bad_pools_across_sessions() -> None:
    m = _mod()
    recs = [
        _sidecar("btbank1_t0", 120, 100, 2, per_band={
            "slow": {"q": 3.0, "max_cell_max": 30.0, "max_n_hot": 1, "n_cat_windows": 1, "n_hot_windows": 1},
        }),
        _sidecar("btbank2_t1", 133, 300, 8, per_band={
            "slow": {"q": 4.0, "max_cell_max": 60.0, "max_n_hot": 2, "n_cat_windows": 3, "n_hot_windows": 5},
        }),
    ]
    s = m.summarize_bad_windows(recs)
    assert s["n_sessions"] == 2
    assert s["n_windows_total"] == 400
    assert s["n_bad_total"] == 10
    assert s["frac_bad_cohort"] == pytest.approx(10 / 400)


def test_per_band_headroom_is_worst_cell_over_q() -> None:
    """headroom = max_cell_max / q — the ratio the CAT fence (12·q) must clear.
    This is the per-band knee Ben locks CAT_MULT_BY_BAND against."""
    m = _mod()
    recs = [
        _sidecar("btbank1_t0", 120, 100, 0, per_band={
            "slow": {"q": 2.0, "max_cell_max": 20.0, "max_n_hot": 0, "n_cat_windows": 0, "n_hot_windows": 0},
            "hg": {"q": 5.0, "max_cell_max": 100.0, "max_n_hot": 0, "n_cat_windows": 0, "n_hot_windows": 0},
        }),
    ]
    s = m.summarize_bad_windows(recs)
    assert s["sessions"][0]["per_band"]["slow"]["headroom"] == pytest.approx(10.0)  # 20/2
    assert s["sessions"][0]["per_band"]["hg"]["headroom"] == pytest.approx(20.0)     # 100/5
    assert s["per_band"]["slow"]["headroom_max"] == pytest.approx(10.0)
    assert s["per_band"]["hg"]["headroom_max"] == pytest.approx(20.0)


def test_per_band_q_quantiles_across_sessions() -> None:
    m = _mod()
    recs = [
        _sidecar(f"btbank{i}_t0", 120, 100, 0, per_band={
            "beta": {"q": q, "max_cell_max": q * 4, "max_n_hot": 0, "n_cat_windows": 0, "n_hot_windows": 0},
        })
        for i, q in enumerate([2.0, 6.0, 10.0])
    ]
    s = m.summarize_bad_windows(recs)
    assert s["per_band"]["beta"]["q_min"] == pytest.approx(2.0)
    assert s["per_band"]["beta"]["q_med"] == pytest.approx(6.0)
    assert s["per_band"]["beta"]["q_max"] == pytest.approx(10.0)


def test_band_absent_from_some_sessions_is_skipped() -> None:
    """A band only present in some sessions is aggregated over just those — no
    KeyError, no zero-padding that would corrupt the q quantiles."""
    m = _mod()
    recs = [
        _sidecar("btbank1_t0", 120, 100, 0, per_band={
            "slow": {"q": 3.0, "max_cell_max": 12.0, "max_n_hot": 0, "n_cat_windows": 0, "n_hot_windows": 0},
        }),
        _sidecar("btbank2_t0", 120, 100, 0, per_band={
            "slow": {"q": 5.0, "max_cell_max": 20.0, "max_n_hot": 0, "n_cat_windows": 0, "n_hot_windows": 0},
            "hg": {"q": 7.0, "max_cell_max": 28.0, "max_n_hot": 0, "n_cat_windows": 0, "n_hot_windows": 0},
        }),
    ]
    s = m.summarize_bad_windows(recs)
    assert set(s["per_band"]) == {"slow", "hg"}
    assert len([x for x in s["sessions"] if "hg" in x["per_band"]]) == 1
    assert s["per_band"]["hg"]["q_min"] == pytest.approx(7.0)


def test_format_runs_without_error() -> None:
    """The text formatter is exercised end-to-end so a column/format bug surfaces
    in CI, not on DCC."""
    m = _mod()
    recs = [
        _sidecar("btbank1_t0", 120, 100, 2, per_band={
            "slow": {"q": 3.0, "max_cell_max": 30.0, "max_n_hot": 1, "n_cat_windows": 1, "n_hot_windows": 1},
            "beta": {"q": 8.0, "max_cell_max": 40.0, "max_n_hot": 0, "n_cat_windows": 0, "n_hot_windows": 0},
            "hg": {"q": 12.0, "max_cell_max": 70.0, "max_n_hot": 0, "n_cat_windows": 0, "n_hot_windows": 0},
        }, by_rule={"hot": 1, "cat": 1, "flat": 0}),
    ]
    txt = m._fmt(m.summarize_bad_windows(recs))
    assert "btbank1_t0" in txt
    assert "per-band cohort knee" in txt
