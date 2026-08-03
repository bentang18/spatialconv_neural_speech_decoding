"""Tests for the curve inversion behind the exchange-rate reconstruction.

These two functions already produced one FALSE VERDICT: the first version divided the gap by a
global linear slope, which understated the doublings on a curve that flattens near the top and made
the WS reconstruction print 🔴 FAILS on data that in fact reconstructs exactly. So they are tested
against curves whose answer is known by construction, including a deliberately non-linear one --
the case the original got wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.neuroprobe.r31_exchange_rate import _cross, _shift_doublings


def test_cross_finds_an_exact_grid_point() -> None:
    lg, y = np.arange(5.0), np.array([.50, .52, .54, .56, .58])
    assert _cross(lg, y, .54) == pytest.approx(2.0)


def test_cross_interpolates_between_grid_points() -> None:
    lg, y = np.arange(5.0), np.array([.50, .52, .54, .56, .58])
    assert _cross(lg, y, .53) == pytest.approx(1.5)


def test_cross_is_nan_outside_the_measured_range() -> None:
    """Never extrapolate: 'did not reach it on this grid' is a result, not a number to invent."""
    lg, y = np.arange(5.0), np.array([.50, .52, .54, .56, .58])
    assert np.isnan(_cross(lg, y, .60))
    assert np.isnan(_cross(lg, y, .49))


def test_shift_doublings_on_a_straight_line_equals_a_over_slope() -> None:
    """Where the curve IS linear in log2 N, curve inversion must agree with the old a/s shortcut —
    otherwise the fix would have changed answers it had no business changing."""
    s, a = 0.02, 0.05
    lg = np.arange(8.0)
    y = .50 + s * lg
    assert _shift_doublings(lg, y, a) == pytest.approx(a / s, rel=1e-9)


def test_shift_doublings_exceeds_a_over_slope_on_a_SATURATING_curve() -> None:
    """THE REGRESSION TEST. A curve that flattens near the top is worth MORE doublings than the
    global slope implies, because the doublings are spent where the curve is shallow. The global-fit
    version underestimated exactly here and fired a false 🔴 on the WS data."""
    lg = np.arange(8.0)
    y = .50 + 0.10 * np.sqrt(lg / 7.0)              # concave, flattening in log2 N
    a = 0.02
    got = _shift_doublings(lg, y, a)
    s_global = float(np.polyfit(lg, y, 1)[0])
    assert got > a / s_global, "saturating curve must cost MORE doublings than the linear shortcut"


def test_shift_doublings_recovers_a_planted_offset_exactly() -> None:
    """If enc12 is enc0 shifted up by a, the predicted crossing IS enc12's crossing. This is the
    whole content of the reconstruction check, on data where it cannot fail."""
    lg = np.arange(8.0)
    y0 = .50 + 0.10 * np.sqrt(lg / 7.0)
    a = 0.015
    y12 = y0 + a
    pred = _shift_doublings(lg, y0, a)
    meas = lg[-1] - _cross(lg, y12, float(y0[-1]))
    assert pred == pytest.approx(meas, abs=1e-9)


def test_shift_doublings_is_nan_when_the_gap_clears_the_whole_grid() -> None:
    """A gap bigger than the curve's entire range means the grid cannot measure the saving."""
    lg = np.arange(5.0)
    y = .50 + 0.002 * lg
    assert np.isnan(_shift_doublings(lg, y, 0.5))
