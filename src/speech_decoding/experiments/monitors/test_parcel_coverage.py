"""Tests for MON-PARCEL-COVERAGE-VARIANCE."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.experiments.monitors import (
    DEGENERATE_CLIP_FRACTION_ALARM,
    DEGENERATE_SLOT_COUNT_MAX,
    parcel_coverage_monitor,
)
from speech_decoding.experiments.monitors.parcel_coverage import (
    _SLOT_USAGE_VARIANCE_REFERENCE,
)


def test_parcel_coverage_thresholds_match_lock() -> None:
    assert DEGENERATE_CLIP_FRACTION_ALARM == 0.10
    assert DEGENERATE_SLOT_COUNT_MAX == 1
    assert _SLOT_USAGE_VARIANCE_REFERENCE == 0.05


def _make_latent_valid(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.bool)


def test_parcel_coverage_healthy_uniform_batch_no_alarm() -> None:
    """All clips have ~32 out of 80 active slots uniformly distributed →
    no alarm. Modest CV, low slot-usage variance."""
    torch.manual_seed(0)
    B, L = 16, 80
    # Each clip picks 32 of 80 slots uniformly at random.
    latent_valid = torch.zeros(B, L, dtype=torch.bool)
    for b in range(B):
        perm = torch.randperm(L)[:32]
        latent_valid[b, perm] = True
    v = parcel_coverage_monitor(latent_valid)
    assert v.is_alarm is False
    assert v.active_slots_per_clip_mean == pytest.approx(32.0)
    # CV should be 0 here (every clip has exactly 32 active).
    assert v.active_slots_per_clip_cv == pytest.approx(0.0)
    assert v.degenerate_clip_fraction == 0.0
    assert v.front_end_only_clip_fraction == 0.0


def test_parcel_coverage_degenerate_batch_triggers_alarm() -> None:
    """5/10 clips have only 1 active slot → degenerate fraction 0.5 >
    0.10 → alarm."""
    rows = [
        [1, 0, 0, 0],  # 1 active → degenerate
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 1, 1],  # 4 active → healthy
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    v = parcel_coverage_monitor(_make_latent_valid(rows))
    assert v.degenerate_clip_fraction == pytest.approx(0.5)
    assert v.is_alarm is True


def test_parcel_coverage_swec_only_batch_emits_zero_alarms() -> None:
    """An all-front-end-only batch (zero anatomy in every clip) MUST
    NOT trip an alarm — SWEC is the legitimate B30 zero-anatomy case."""
    latent_valid = torch.zeros(8, 80, dtype=torch.bool)
    v = parcel_coverage_monitor(latent_valid)
    assert v.front_end_only_clip_fraction == pytest.approx(1.0)
    assert v.is_alarm is False
    assert v.active_slots_per_clip_mean == 0.0


def test_parcel_coverage_mixed_swec_and_bt_excludes_swec_from_stats() -> None:
    """4 SWEC clips (all-False) + 4 BT clips, each BT clip activating a
    *different* random ~30-slot subset → per-clip stats computed on the
    BT 4 only; front_end_only fraction reports the SWEC presence; slot-
    usage variance stays moderate because each slot is active in only
    some of the BT clips."""
    torch.manual_seed(11)
    B, L = 8, 80
    latent_valid = torch.zeros(B, L, dtype=torch.bool)
    for b in range(4, 8):
        perm = torch.randperm(L)[:30]
        latent_valid[b, perm] = True
    v = parcel_coverage_monitor(latent_valid)
    assert v.front_end_only_clip_fraction == pytest.approx(0.5)
    assert v.active_slots_per_clip_mean == pytest.approx(30.0)
    assert v.degenerate_clip_fraction == 0.0
    # Per-slot usage fraction across only 4 anatomy-bearing clips with
    # random ~30/80 selection should land well below the alarm.
    assert v.slot_usage_fraction_var < _SLOT_USAGE_VARIANCE_REFERENCE
    assert v.is_alarm is False


def test_parcel_coverage_high_slot_usage_variance_logged_but_no_alarm() -> None:
    """If every clip activates the same 10 slots (slots 0..9), per-slot
    usage is 1.0 for slots 0..9, 0.0 for slots 10..79. With p = 0.125 the
    bimodal variance ≈ p(1-p) = 0.109 > 0.05 — this is the structural
    K=80-sparse regime. The variance is STILL computed and logged, but it
    does NOT trip the alarm: every clip has 10 active slots (> 1) so
    degenerate_clip_fraction is 0.0 and is_alarm is False."""
    B, L = 16, 80
    latent_valid = torch.zeros(B, L, dtype=torch.bool)
    latent_valid[:, :10] = True
    v = parcel_coverage_monitor(latent_valid)
    assert v.slot_usage_fraction_var > _SLOT_USAGE_VARIANCE_REFERENCE
    assert v.degenerate_clip_fraction == 0.0
    assert v.is_alarm is False


def test_parcel_coverage_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match=r"\(B, L\)"):
        parcel_coverage_monitor(torch.zeros(4, 5, 3, dtype=torch.bool))


def test_parcel_coverage_rejects_non_bool_input() -> None:
    with pytest.raises(ValueError, match="bool"):
        parcel_coverage_monitor(torch.zeros(4, 5))


def test_parcel_coverage_single_anatomy_clip_returns_zero_std() -> None:
    """Edge: only 1 clip carries anatomy. std/CV gracefully return 0
    rather than raising or producing NaN."""
    latent_valid = torch.zeros(4, 80, dtype=torch.bool)
    latent_valid[0, :30] = True
    v = parcel_coverage_monitor(latent_valid)
    assert v.active_slots_per_clip_mean == pytest.approx(30.0)
    assert v.active_slots_per_clip_std == 0.0
    assert v.active_slots_per_clip_cv == 0.0
