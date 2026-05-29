"""Tests for the Phase-3 distillation loss (T2.4)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.ssl.distill import (
    PhaseThreeDistillationConfig,
    phase3_distillation_loss,
)


def test_smooth_l1_matches_torch_reference() -> None:
    torch.manual_seed(0)
    s = torch.randn(4, 50, 256)
    t = torch.randn(4, 50, 256)
    expected = torch.nn.functional.smooth_l1_loss(s, t, beta=1.0)
    got = phase3_distillation_loss(s, t, loss_form="smooth_l1", beta=1.0)
    assert torch.allclose(got, expected, atol=1e-7)


def test_mse_form_matches_torch_reference() -> None:
    torch.manual_seed(1)
    s = torch.randn(4, 50, 256)
    t = torch.randn(4, 50, 256)
    expected = torch.nn.functional.mse_loss(s, t)
    got = phase3_distillation_loss(s, t, loss_form="mse", beta=1.0)
    assert torch.allclose(got, expected, atol=1e-7)


def test_cosine_form_is_zero_for_identical_inputs() -> None:
    torch.manual_seed(2)
    x = torch.randn(4, 50, 256)
    got = phase3_distillation_loss(x, x, loss_form="cosine", beta=1.0)
    # cosine-similarity of x with itself is 1, so loss = 1 - 1 = 0
    assert torch.allclose(got, torch.zeros_like(got), atol=1e-6)


def test_target_instance_norm_changes_loss() -> None:
    """When teacher is already standardized, instance-norm is a no-op; when
    teacher has a wildly different scale, instance-norm absorbs the offset."""
    torch.manual_seed(3)
    s = torch.randn(4, 50, 256)
    t = torch.randn(4, 50, 256) * 100.0 + 5.0  # large scale + offset
    without_norm = phase3_distillation_loss(s, t, loss_form="smooth_l1", beta=1.0)
    with_norm = phase3_distillation_loss(
        s, t, loss_form="smooth_l1", beta=1.0, target_instance_norm=True
    )
    assert with_norm < without_norm


def test_config_pins_beta_to_one_by_default() -> None:
    """P3-04 (v14_implementation_fix_list.md §A.6): β = 1.0 default.

    M04 unblocked by B01 v3 lock + 5/25 PM revisit; AB10 sweeps {0.5, 1.0, 2.0}
    remain as sister cells (callers can still override).
    """
    cfg_default = PhaseThreeDistillationConfig(loss_form="smooth_l1")
    assert cfg_default.beta == 1.0, "P3-04: Smooth-L1 β default must be 1.0"
    cfg_override = PhaseThreeDistillationConfig(beta=0.5)
    assert cfg_override.beta == 0.5


def test_config_call_delegates_to_function() -> None:
    cfg = PhaseThreeDistillationConfig(beta=0.5, loss_form="smooth_l1")
    s = torch.randn(2, 50, 256)
    t = torch.randn(2, 50, 256)
    a = cfg(s, t)
    b = phase3_distillation_loss(s, t, loss_form="smooth_l1", beta=0.5)
    assert torch.allclose(a, b)


def test_shape_mismatch_raises() -> None:
    s = torch.randn(4, 50, 256)
    t = torch.randn(4, 50, 128)
    with pytest.raises(ValueError):
        phase3_distillation_loss(s, t, beta=1.0)


def test_gradient_flows_to_student_only_when_teacher_detached() -> None:
    s = torch.randn(4, 50, 256, requires_grad=True)
    t = torch.randn(4, 50, 256)  # no grad
    loss = phase3_distillation_loss(s, t, loss_form="smooth_l1", beta=1.0)
    loss.backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()
