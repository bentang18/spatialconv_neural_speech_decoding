"""Tests for MON-GRAD-SPIKE-DIVERGENCE."""

from __future__ import annotations

import math

import pytest

from speech_decoding.experiments.monitors import (
    GRAD_EMA_ALPHA,
    GRAD_SPIKE_THRESHOLD,
    grad_spike_monitor,
)


def test_grad_spike_defaults_match_spam() -> None:
    """SPAM (Huang 2025) defaults: α=0.99, k=5."""
    assert GRAD_EMA_ALPHA == 0.99
    assert GRAD_SPIKE_THRESHOLD == 5.0


def test_grad_spike_first_step_seeds_ema_no_spike() -> None:
    """With no prior history (prior_ema = 0), the first step seeds the
    EMA to the current grad and does NOT flag a spike — there's no
    baseline to spike against."""
    v = grad_spike_monitor(grad_l2=3.0, prior_ema_l2=0.0)
    assert v.grad_l2 == pytest.approx(3.0)
    assert v.grad_ema_l2 == pytest.approx(0.0)
    assert v.spike_ratio == pytest.approx(0.0)
    assert v.is_spike is False
    assert v.is_diverged is False
    # First-step seeding: new EMA equals current.
    assert v.new_grad_ema_l2 == pytest.approx(3.0)


def test_grad_spike_steady_state_holds_ema_steady() -> None:
    """When current ≈ prior EMA, the ratio stays near 1.0 and the new
    EMA barely moves (0.99 decay)."""
    v = grad_spike_monitor(grad_l2=2.0, prior_ema_l2=2.0)
    assert v.spike_ratio == pytest.approx(1.0)
    assert v.is_spike is False
    assert v.new_grad_ema_l2 == pytest.approx(2.0)


def test_grad_spike_detects_5x_spike() -> None:
    """A 6x spike on top of an EMA = 1.0 trips the default threshold."""
    v = grad_spike_monitor(grad_l2=6.0, prior_ema_l2=1.0)
    assert v.spike_ratio == pytest.approx(6.0)
    assert v.is_spike is True
    assert v.is_diverged is False
    # EMA tracks slowly so it doesn't get pulled to the spike: new
    # EMA ≈ 0.99*1 + 0.01*6 = 1.05.
    assert v.new_grad_ema_l2 == pytest.approx(1.05)


def test_grad_spike_3x_does_not_trip_default_threshold() -> None:
    """3x is the Wortsman et al. 2024 sister-cell threshold but is below
    SPAM default 5x."""
    v = grad_spike_monitor(grad_l2=3.0, prior_ema_l2=1.0)
    assert v.spike_ratio == pytest.approx(3.0)
    assert v.is_spike is False


def test_grad_spike_3x_trips_when_threshold_lowered() -> None:
    """Lower threshold (Wortsman variant) → same 3x trips."""
    v = grad_spike_monitor(
        grad_l2=3.0, prior_ema_l2=1.0, spike_threshold=2.5,
    )
    assert v.is_spike is True


def test_grad_spike_inf_flags_divergence_and_protects_ema() -> None:
    """An Inf grad-L2 (AdamW divergence) is flagged, but the persistent
    EMA is NOT updated — otherwise a single bad step would NaN-poison
    the buffer for all subsequent comparisons."""
    v = grad_spike_monitor(grad_l2=float("inf"), prior_ema_l2=2.0)
    assert v.is_diverged is True
    assert v.is_spike is True
    assert v.new_grad_ema_l2 == pytest.approx(2.0)  # EMA preserved
    assert v.spike_ratio == float("inf")


def test_grad_spike_nan_flags_divergence_and_protects_ema() -> None:
    """NaN grad-L2 behaves identically to Inf for divergence + protection."""
    v = grad_spike_monitor(grad_l2=float("nan"), prior_ema_l2=2.0)
    assert v.is_diverged is True
    assert v.is_spike is True
    assert v.new_grad_ema_l2 == pytest.approx(2.0)
    assert math.isnan(v.grad_l2)


def test_grad_spike_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\)"):
        grad_spike_monitor(grad_l2=1.0, prior_ema_l2=1.0, alpha=1.5)


def test_grad_spike_rejects_threshold_at_or_below_one() -> None:
    with pytest.raises(ValueError, match="spike_threshold must be > 1"):
        grad_spike_monitor(grad_l2=1.0, prior_ema_l2=1.0, spike_threshold=1.0)


def test_grad_spike_ema_decays_under_repeated_steady_steps() -> None:
    """A 0.99-decay EMA seeded at 10 and then fed 1.0 for many steps
    should converge to 1.0; demonstrate the SPAM EMA is well-behaved
    over a few hundred steps."""
    ema = 10.0
    for _ in range(500):
        v = grad_spike_monitor(grad_l2=1.0, prior_ema_l2=ema)
        ema = v.new_grad_ema_l2
    # 0.99^500 ≈ 0.0066 → starting at 10, converges to ~1 + 9*0.0066 ≈ 1.06
    assert ema == pytest.approx(1.06, abs=0.02)
