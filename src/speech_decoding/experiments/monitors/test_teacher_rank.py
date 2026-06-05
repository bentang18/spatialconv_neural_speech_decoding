"""Tests for MON-TEACHER-FEATURE-RANK (RankMe)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.experiments.monitors import (
    RANKME_M4_NORMALISED_ALARM,
    RANKME_M4_NORMALISED_WARN,
    RANKME_NORMALISED_ALARM,
    RANKME_NORMALISED_WARN,
    teacher_rank_monitor,
)


def test_teacher_rank_thresholds_match_dinov3_band() -> None:
    """DINOv3 §3.3 + RankMe §4.2 band: warn at 0.5, alarm at 0.25 (M2
    front-end, |STFT| floor ~0.31)."""
    assert RANKME_NORMALISED_WARN == 0.5
    assert RANKME_NORMALISED_ALARM == 0.25


def test_teacher_rank_m4_band_below_parcel_floor() -> None:
    """M4 parcel-token band sits below its structural floor (~0.05, raw rank
    ≈ active-parcel count). warn 0.04 / alarm 0.02, with the 2:1 warn:alarm
    structure preserved and both strictly below the M2 band."""
    assert RANKME_M4_NORMALISED_WARN == 0.04
    assert RANKME_M4_NORMALISED_ALARM == 0.02
    assert 0.0 < RANKME_M4_NORMALISED_ALARM < RANKME_M4_NORMALISED_WARN
    assert RANKME_M4_NORMALISED_WARN < RANKME_NORMALISED_WARN
    assert RANKME_M4_NORMALISED_ALARM < RANKME_NORMALISED_ALARM


def test_teacher_rank_isotropic_gaussian_near_full_rank() -> None:
    """Standard-normal feature matrix N >> d → RankMe close to d."""
    torch.manual_seed(0)
    N, d = 4096, 64
    Z = torch.randn(N, d)
    v = teacher_rank_monitor(Z)
    # For isotropic Gaussian + large N, RankMe / d should be ≥ 0.8.
    assert v.rankme_normalised > 0.8
    assert v.is_warn is False
    assert v.is_alarm is False
    assert v.n_samples == N
    assert v.d_feature == d


def test_teacher_rank_rank_one_collapse_alarm() -> None:
    """All rows = same vector → rank 1 → RankMe == 1 → normalised
    = 1/d << 0.25 → hard alarm."""
    N, d = 256, 64
    v0 = torch.randn(d)
    Z = v0.unsqueeze(0).expand(N, d).contiguous()
    v = teacher_rank_monitor(Z)
    assert v.rankme == pytest.approx(1.0, abs=0.05)
    assert v.rankme_normalised < 0.05
    assert v.is_alarm is True
    assert v.is_warn is True


def test_teacher_rank_intermediate_rank_warn_not_alarm() -> None:
    """Z = U @ V where U is N×r, V is r×d with r = d/4 → effective rank
    ≈ r → normalised ≈ 0.25 (borderline between warn and alarm). Pick
    r = d/3 so we cleanly land in the warn band."""
    torch.manual_seed(1)
    N, d, r = 1024, 60, 20      # r/d ≈ 0.33 → in warn band, above alarm
    U = torch.randn(N, r)
    V = torch.randn(r, d)
    Z = U @ V
    v = teacher_rank_monitor(Z)
    assert v.rankme_normalised < RANKME_NORMALISED_WARN
    assert v.rankme_normalised > RANKME_NORMALISED_ALARM
    assert v.is_warn is True
    assert v.is_alarm is False


def test_teacher_rank_handles_4d_m4_tensor_with_valid_mask() -> None:
    """The realistic input is a (B, L, T, d) M4 tap with a (B, L)
    latent_valid mask. The function flattens correctly and masks out
    inactive slots so a half-SWEC batch doesn't spoof the rank."""
    torch.manual_seed(2)
    B, L, T, d = 8, 80, 40, 32
    M4 = torch.randn(B, L, T, d)
    # Half the batch is "SWEC" (all-False); other half is anatomy-bearing.
    latent_valid = torch.zeros(B, L, dtype=torch.bool)
    latent_valid[B // 2 :, :32] = True
    v = teacher_rank_monitor(M4, valid_mask=latent_valid)
    assert v.n_samples == (B // 2) * 32 * T  # only active rows kept
    assert v.d_feature == d
    assert v.rankme_normalised > 0.5    # full-rank random subspace


def test_teacher_rank_3d_slot_bank_no_mask() -> None:
    """(B, L, d) shape with no mask should flatten to (B*L, d)."""
    torch.manual_seed(3)
    B, L, d = 16, 80, 32
    slot_bank = torch.randn(B, L, d)
    v = teacher_rank_monitor(slot_bank)
    assert v.n_samples == B * L
    assert v.d_feature == d


def test_teacher_rank_3d_slot_bank_with_valid_mask() -> None:
    torch.manual_seed(4)
    B, L, d = 8, 80, 32
    slot_bank = torch.randn(B, L, d)
    valid = torch.zeros(B, L, dtype=torch.bool)
    valid[:, :10] = True
    v = teacher_rank_monitor(slot_bank, valid_mask=valid)
    assert v.n_samples == B * 10


def test_teacher_rank_empty_mask_returns_degenerate_healthy_verdict() -> None:
    """When fewer than 2 samples survive the mask, the monitor returns
    a degenerate-healthy verdict (rankme=1, no alarm) rather than crashing."""
    Z = torch.randn(8, 80, 32)
    valid = torch.zeros(8, 80, dtype=torch.bool)  # zero True entries
    v = teacher_rank_monitor(Z, valid_mask=valid)
    assert v.n_samples == 0
    assert v.is_alarm is False
    assert v.is_warn is False


def test_teacher_rank_rejects_unsupported_dims() -> None:
    with pytest.raises(ValueError, match="2-/3-/4-D"):
        teacher_rank_monitor(torch.randn(1, 2, 3, 4, 5))


def test_teacher_rank_rejects_mismatched_mask_shape_3d() -> None:
    Z = torch.randn(8, 80, 32)
    bad_mask = torch.ones(8, 40, dtype=torch.bool)
    with pytest.raises(ValueError, match="must match"):
        teacher_rank_monitor(Z, valid_mask=bad_mask)


def test_teacher_rank_rejects_mismatched_mask_shape_4d() -> None:
    Z = torch.randn(8, 80, 40, 32)
    bad_mask = torch.ones(8, 40, dtype=torch.bool)
    with pytest.raises(ValueError, match="not.*compatible"):
        teacher_rank_monitor(Z, valid_mask=bad_mask)


def test_teacher_rank_rejects_invalid_thresholds() -> None:
    Z = torch.randn(64, 32)
    with pytest.raises(ValueError, match="alarm < warn"):
        teacher_rank_monitor(Z, warn_threshold=0.2, alarm_threshold=0.5)
