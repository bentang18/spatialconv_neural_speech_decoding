"""Tests for the masked reconstruction loss (T2.3)."""

from __future__ import annotations

import torch

from speech_decoding.ssl.recon import recon_loss


def test_recon_loss_default_is_l1() -> None:
    """B26 lock 2026-05-27 PM: default loss_form is pure L1 (V-JEPA 2
    §2.1 Eq 1 + V-JEPA 2.1 §2.3.1 Eqs 1+2 both use L1, NOT Smooth-L1 —
    rolling back the B25 Smooth-L1 default after PM re-read of the
    V-JEPA 2/2.1/data2vec-2.0 PDFs surfaced the prior precedent chain
    was empirically wrong)."""
    torch.manual_seed(0)
    s = torch.randn(4, 8, requires_grad=True)
    t = torch.randn(4, 8)
    expected = torch.nn.functional.l1_loss(s, t)
    got = recon_loss(s, t)
    assert torch.allclose(got, expected, atol=1e-7)


def test_recon_loss_l1_explicit_matches_torch_reference() -> None:
    """Explicit ``loss_form='l1'`` matches ``torch.nn.functional.l1_loss``."""
    torch.manual_seed(0)
    s = torch.randn(4, 8)
    t = torch.randn(4, 8)
    expected = torch.nn.functional.l1_loss(s, t)
    got = recon_loss(s, t, loss_form="l1")
    assert torch.allclose(got, expected, atol=1e-7)


def test_recon_loss_mse_opt_in_still_works() -> None:
    """``R-l2-loss`` / ``R-mse-loss`` sister cell: explicit
    ``loss_form='mse'`` selects MSE (data2vec 2.0 §3.1 convention).
    Falsifies "L1 is load-bearing at v14 scale"."""
    torch.manual_seed(0)
    s = torch.randn(4, 8)
    t = torch.randn(4, 8)
    expected = torch.nn.functional.mse_loss(s, t)
    got = recon_loss(s, t, loss_form="mse")
    assert torch.allclose(got, expected, atol=1e-7)


def test_recon_loss_smooth_l1_opt_in_still_works() -> None:
    """``R-smoothl1-beta-{0.5, 1.0, 2.0}`` sister sweep: Smooth-L1 stays
    available as an opt-in for the bracket sweep around the prior B25
    default. Smooth-L1 reduces to L1 as β→0; the bracket falsifies the
    transition."""
    torch.manual_seed(0)
    s = torch.randn(4, 8)
    t = torch.randn(4, 8)
    expected = torch.nn.functional.smooth_l1_loss(s, t, beta=1.0)
    got = recon_loss(s, t, loss_form="smooth_l1", smooth_l1_beta=1.0)
    assert torch.allclose(got, expected, atol=1e-7)


def test_recon_loss_mask_excludes_positions() -> None:
    """Loss with all-valid mask == loss with no mask; partial mask gives
    a value equal to the mean over surviving positions.

    Uses ``loss_form='mse'`` so the expected-value math stays simple
    ``(s - t)².mean()``."""
    torch.manual_seed(0)
    s = torch.randn(4, 8)
    t = torch.randn(4, 8)

    full_mask = torch.ones(4, 8, dtype=torch.bool)
    got_full = recon_loss(
        s, t, valid_mask_student=full_mask, valid_mask_teacher=full_mask,
        loss_form="mse",
    )
    no_mask = recon_loss(s, t, loss_form="mse")
    assert torch.allclose(got_full, no_mask)

    partial = torch.zeros(4, 8, dtype=torch.bool)
    partial[:, :4] = True
    got_partial = recon_loss(
        s, t, valid_mask_student=partial, valid_mask_teacher=partial,
        loss_form="mse",
    )
    expected = (s[:, :4] - t[:, :4]).pow(2).mean()
    assert torch.allclose(got_partial, expected)


def test_recon_loss_intersection_of_student_and_teacher_masks() -> None:
    """Per EX09, positions invalid on EITHER side must be excluded from
    both numerator and denominator. ``loss_form='mse'`` so the
    closed-form expectation stays trivial."""
    s = torch.zeros(2, 4)
    t = torch.ones(2, 4)  # diff² == 1 everywhere
    sm = torch.tensor([[True, True, False, False], [True, True, False, False]])
    tm = torch.tensor([[True, False, True, False], [True, False, True, False]])
    # Intersection: only (row, col=0). Mean diff² over 2 surviving cells = 1.0.
    got = recon_loss(
        s, t, valid_mask_student=sm, valid_mask_teacher=tm, loss_form="mse",
    )
    assert torch.allclose(got, torch.tensor(1.0))


def test_recon_loss_zero_when_student_equals_teacher() -> None:
    s = torch.randn(4, 8, requires_grad=True)
    got = recon_loss(s, s.detach())
    assert torch.allclose(got, torch.zeros_like(got), atol=1e-7)


def test_recon_loss_gradient_only_flows_to_student() -> None:
    """Teacher is *not* detached inside the function — caller's job — but
    student should still receive gradient when teacher is detached at the
    call site (the recipe pattern)."""
    s = torch.randn(4, 8, requires_grad=True)
    t = torch.randn(4, 8)  # no grad
    loss = recon_loss(s, t)
    loss.backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()


def test_recon_loss_smooth_l1_uses_beta() -> None:
    """SmoothL1 with very large β degenerates to 0.5 * MSE for small diffs;
    test parity vs torch's smooth_l1_loss directly."""
    torch.manual_seed(1)
    s = torch.randn(4, 8)
    t = torch.randn(4, 8)
    expected = torch.nn.functional.smooth_l1_loss(s, t, beta=2.0)
    got = recon_loss(s, t, loss_form="smooth_l1", smooth_l1_beta=2.0)
    assert torch.allclose(got, expected, atol=1e-7)


def test_recon_loss_all_masked_does_not_divide_by_zero() -> None:
    """An all-invalid mask must produce a finite loss (the safety floor
    matters in batches where some samples have no valid positions)."""
    s = torch.randn(4, 8)
    t = torch.randn(4, 8)
    zeros = torch.zeros(4, 8, dtype=torch.bool)
    got = recon_loss(s, t, valid_mask_student=zeros, valid_mask_teacher=zeros)
    assert torch.isfinite(got)


def test_recon_loss_shape_mismatch_raises() -> None:
    s = torch.randn(4, 8)
    t = torch.randn(4, 9)
    try:
        recon_loss(s, t)
    except ValueError as e:
        assert "student.shape" in str(e)
    else:
        raise AssertionError("expected ValueError on shape mismatch")
