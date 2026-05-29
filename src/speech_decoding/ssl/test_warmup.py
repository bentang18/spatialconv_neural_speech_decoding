"""Tests for the λ warmup schedulers (T2.6)."""

from __future__ import annotations

import math

import pytest

from speech_decoding.ssl.warmup import (
    linear_warmup,
    linear_warmup_then_cosine,
)


def test_linear_warmup_zero_at_start() -> None:
    sched = linear_warmup(peak=0.5, warmup_steps=100)
    assert sched(0) == 0.0


def test_linear_warmup_peak_at_end() -> None:
    sched = linear_warmup(peak=0.5, warmup_steps=100)
    assert sched(100) == 0.5


def test_linear_warmup_holds_at_peak_after_warmup() -> None:
    sched = linear_warmup(peak=0.5, warmup_steps=100)
    assert sched(1_000) == 0.5
    assert sched(10_000) == 0.5


def test_linear_warmup_midpoint_is_half_peak() -> None:
    sched = linear_warmup(peak=0.5, warmup_steps=100)
    assert abs(sched(50) - 0.25) < 1e-9


def test_linear_warmup_zero_warmup_holds_at_peak_immediately() -> None:
    sched = linear_warmup(peak=0.5, warmup_steps=0)
    assert sched(0) == 0.5
    assert sched(100) == 0.5


def test_linear_warmup_rejects_negative_warmup_steps() -> None:
    with pytest.raises(ValueError):
        linear_warmup(peak=0.5, warmup_steps=-1)


def test_warmup_then_cosine_passes_through_warmup() -> None:
    sched = linear_warmup_then_cosine(
        peak=1.0, end=0.0, warmup_steps=10, decay_steps=100
    )
    assert sched(0) == 0.0
    assert sched(5) == 0.5
    assert sched(10) == 1.0


def test_warmup_then_cosine_decays_to_end() -> None:
    sched = linear_warmup_then_cosine(
        peak=1.0, end=0.0, warmup_steps=10, decay_steps=100
    )
    # Just past the decay endpoint should equal `end` (within 1e-9).
    assert abs(sched(10 + 100) - 0.0) < 1e-9
    # Past the end clamps at `end`.
    assert abs(sched(10 + 200) - 0.0) < 1e-9


def test_warmup_then_cosine_midpoint_is_half_peak_plus_end() -> None:
    sched = linear_warmup_then_cosine(
        peak=1.0, end=0.0, warmup_steps=10, decay_steps=100
    )
    midpoint = 10 + 50
    expected = 0.0 + (1.0 - 0.0) * 0.5 * (1.0 + math.cos(math.pi * 0.5))
    # cos(π/2) = 0 → expected = 0.5
    assert abs(sched(midpoint) - expected) < 1e-9
    assert abs(sched(midpoint) - 0.5) < 1e-9


def test_warmup_then_cosine_rejects_invalid_decay_steps() -> None:
    with pytest.raises(ValueError):
        linear_warmup_then_cosine(peak=1.0, end=0.0, warmup_steps=10, decay_steps=0)


# --- B28 anatomy-bias warmup ------------------------------------------

from speech_decoding.ssl.warmup import anatomy_bias_warmup_schedule


def test_b28_anatomy_bias_p1_holds_at_zero_before_window() -> None:
    """Early P1 (before the last 25% of P1 begins) → λ_anat = 0."""
    sched = anatomy_bias_warmup_schedule(
        "P1", total_p1_steps=1000, total_p2_steps=2000
    )
    assert sched(0) == 0.0
    assert sched(500) == 0.0
    assert sched(750) == 0.0  # exactly at window start (last-25%-of-P1)


def test_b28_anatomy_bias_p1_ramps_to_half_at_p1_end() -> None:
    """λ_anat hits exactly 0.5 at the P1→P2 boundary.

    Window spans 25% of P1 (=250 steps) + 25% of P2 (=500 steps) = 750 total;
    at P1 end (step=1000), 250/750 of the window is traversed → λ = 1/3.

    Correction: with equal warmup_fraction_p1 and warmup_fraction_p2, the
    halfway point of the window does NOT land at the P1→P2 boundary unless
    P1 and P2 have the same step count. We just check that 0 < λ(P1-end) < 1.
    """
    sched = anatomy_bias_warmup_schedule(
        "P1", total_p1_steps=1000, total_p2_steps=1000  # equal totals
    )
    # Equal totals: window = 250 + 250 = 500; halfway = 250 P1-steps after
    # window start = exactly at P1 end → λ = 0.5.
    assert abs(sched(1000) - 0.5) < 1e-9


def test_b28_anatomy_bias_p2_picks_up_from_half() -> None:
    sched = anatomy_bias_warmup_schedule(
        "P2", total_p1_steps=1000, total_p2_steps=1000
    )
    # At P2 step 0 (= P1→P2 boundary), λ = 0.5 (continuing from P1).
    assert abs(sched(0) - 0.5) < 1e-9
    # At P2 step 250 (end of warmup window), λ = 1.0.
    assert abs(sched(250) - 1.0) < 1e-9
    # After window end, λ stays at 1.0.
    assert sched(500) == 1.0
    assert sched(10_000) == 1.0


def test_b28_anatomy_bias_p1_linear_inside_window() -> None:
    """Inside the warmup window the schedule is linear."""
    sched = anatomy_bias_warmup_schedule(
        "P1", total_p1_steps=1000, total_p2_steps=1000
    )
    # Window starts at P1 step 750, length 500.
    assert abs(sched(750) - 0.0) < 1e-9
    assert abs(sched(875) - 0.25) < 1e-9   # 125/500
    assert abs(sched(1000) - 0.5) < 1e-9   # 250/500


def test_b28_anatomy_bias_p3_and_p4_are_constant_one() -> None:
    """Downstream phases see anatomy bias fully on at every step."""
    sched_p3 = anatomy_bias_warmup_schedule(
        "P3", total_p1_steps=1000, total_p2_steps=2000
    )
    sched_p4 = anatomy_bias_warmup_schedule(
        "P4", total_p1_steps=1000, total_p2_steps=2000
    )
    for step in [0, 100, 10_000]:
        assert sched_p3(step) == 1.0
        assert sched_p4(step) == 1.0


def test_b28_anatomy_bias_zero_warmup_fraction_collapses_to_step_function() -> None:
    """``warmup_fraction_*=0`` recovers the prior discrete OFF→ON toggle —
    this is the ``R-anatomy-bias-step`` sister."""
    sched_p1 = anatomy_bias_warmup_schedule(
        "P1", total_p1_steps=1000, total_p2_steps=1000,
        warmup_fraction_p1=0.0, warmup_fraction_p2=0.0,
    )
    sched_p2 = anatomy_bias_warmup_schedule(
        "P2", total_p1_steps=1000, total_p2_steps=1000,
        warmup_fraction_p1=0.0, warmup_fraction_p2=0.0,
    )
    # In P1, λ stays 0 throughout (window length 0).
    assert sched_p1(0) == 0.0
    assert sched_p1(500) == 0.0
    assert sched_p1(1000) == 0.0
    # In P2, λ jumps to 1.0 immediately (no warmup).
    assert sched_p2(0) == 1.0
    assert sched_p2(500) == 1.0


def test_b28_anatomy_bias_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError):
        anatomy_bias_warmup_schedule(
            "P1", total_p1_steps=1000, total_p2_steps=1000,
            warmup_fraction_p1=-0.1,
        )
    with pytest.raises(ValueError):
        anatomy_bias_warmup_schedule(
            "P1", total_p1_steps=1000, total_p2_steps=1000,
            warmup_fraction_p2=1.5,
        )


def test_b28_anatomy_bias_rejects_negative_step_totals() -> None:
    with pytest.raises(ValueError):
        anatomy_bias_warmup_schedule(
            "P1", total_p1_steps=-1, total_p2_steps=1000,
        )


def test_b28_anatomy_bias_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError, match="unknown phase"):
        anatomy_bias_warmup_schedule(
            "P0", total_p1_steps=1000, total_p2_steps=1000,  # type: ignore[arg-type]
        )
