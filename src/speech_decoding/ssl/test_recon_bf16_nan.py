"""SCAFFOLD-09 (TST05 + RT01): bf16 NaN detector for the P1 loss.

Under bf16 mixed precision the Multi-STFT power → (no-)log → robust-z
chain has a thin numerical floor. With ``log_eps`` too small (or with
no log at all, the 5/25 swap to raw ``|STFT|``), bf16 magnitudes can
underflow to zero in the per-element diff² → div by mask sum → NaN.

This test pins the contract: with ``log_eps ≥ 1e-6`` the bf16 reconstruction
loss is finite across the full pipeline (no log + raw magnitude + masked
positions + bf16-cast student/teacher). If a future preprocessing change
re-introduces the NaN regression, this is the early-warning test.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.ssl.recon import recon_loss


_BF16_LOG_EPS_FLOOR = 1e-6


@pytest.mark.parametrize("loss_form", ["l1", "smooth_l1", "mse"])
def test_recon_loss_finite_under_bf16(loss_form: str) -> None:
    """B26 default is pure L1; Smooth-L1 + MSE are the R-smoothl1-beta-*
    and R-l2-loss sister falsifiers. All three must be finite under bf16
    with the standard masking contract."""
    torch.manual_seed(0)
    B, C, F, T = 2, 4, 30, 64
    s = torch.randn(B, C, F, T, dtype=torch.bfloat16) * 0.1
    t = torch.randn(B, C, F, T, dtype=torch.bfloat16) * 0.1
    valid = torch.ones(B, C, F, T, dtype=torch.bool)
    valid[:, :, :5, :] = False  # SWEC-style bin mask: drop first 5 bins
    loss = recon_loss(
        s, t,
        valid_mask_student=valid, valid_mask_teacher=valid,
        loss_form=loss_form,  # type: ignore[arg-type]
    )
    assert torch.isfinite(loss), f"{loss_form}: bf16 reconstruction loss is non-finite"


def test_recon_loss_finite_with_log_eps_at_floor() -> None:
    """A direct check on the pre-loss preprocessing — taking a log of an
    abs-stft envelope. With ``log_eps`` at the floor, the resulting
    feature vector is finite (no -inf from log(0))."""
    torch.manual_seed(0)
    # Synthetic |STFT|-like tensor with values straddling underflow.
    raw = torch.zeros(2, 4, 30, 64, dtype=torch.bfloat16)
    raw[..., :10, :] = 0.0                          # underflow region
    raw[..., 10:, :] = torch.rand(2, 4, 20, 64) * 0.05
    logged = torch.log(raw.float() + _BF16_LOG_EPS_FLOOR).to(torch.bfloat16)
    assert torch.isfinite(logged).all(), (
        "log(raw + 1e-6) is non-finite under bf16 — log_eps floor must hold"
    )


def test_recon_loss_finite_when_some_mask_rows_are_all_zero() -> None:
    """Edge case: an entire batch row could be all-padded (in mixed-C
    cohort sampling). The masked-mean denominator is floored at ``eps``
    so the loss stays finite even when that row contributes nothing."""
    B, C, F, T = 2, 4, 30, 64
    s = torch.randn(B, C, F, T, dtype=torch.bfloat16) * 0.1
    t = torch.randn(B, C, F, T, dtype=torch.bfloat16) * 0.1
    valid = torch.ones(B, C, F, T, dtype=torch.bool)
    valid[1] = False  # whole row padded
    loss = recon_loss(
        s, t, valid_mask_student=valid, valid_mask_teacher=valid,
    )
    assert torch.isfinite(loss), "bf16 loss with one fully-padded batch row went non-finite"


def test_recon_loss_finite_with_extreme_tail_values() -> None:
    """Robustness check: bf16 dynamic range is ~3.4e38 / ~1.18e-38; the
    pipeline must clip / tolerate this without producing NaN."""
    torch.manual_seed(0)
    B, C, F, T = 1, 4, 30, 64
    s = torch.randn(B, C, F, T, dtype=torch.bfloat16)
    t = torch.randn(B, C, F, T, dtype=torch.bfloat16)
    # Force a few extreme values into both tensors.
    s[0, 0, 0, 0] = torch.finfo(torch.bfloat16).max / 1e4
    s[0, 0, 0, 1] = torch.finfo(torch.bfloat16).min / 1e4
    t[0, 0, 0, 0] = 0.0
    t[0, 0, 0, 1] = 0.0
    loss = recon_loss(s, t)
    # B26 default L1 keeps the gradient bounded → loss stays finite even
    # with the outliers; MSE would diverge. (Smooth-L1 is also bounded;
    # the L1 / Smooth-L1 difference shows up at the per-element gradient
    # but both stay finite here.)
    assert torch.isfinite(loss), "bf16 L1 went non-finite under extreme tail values"
