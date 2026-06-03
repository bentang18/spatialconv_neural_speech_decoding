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


# The B28 ``anatomy_bias_warmup_schedule`` tests were removed with the
# scheduler itself by the B36 hard per-parcel pool (2026-06-01): there is no
# soft ``λ_anat`` routing bias left to warm up. See
# ``project_v14_b36_perparcel_pool_structured_jepa_2026_06_01``.
