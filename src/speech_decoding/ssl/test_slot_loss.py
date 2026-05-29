"""Tests for the slot×time×d masked loss used by LOSS-02 / LOSS-03.

B26 amendment 2026-05-27 PM: default ``loss_form="l1"`` (V-JEPA 2 §2.1
Eq 1 + V-JEPA 2.1 §2.3.1 Eqs 1+2 both use pure L1). MSE is preserved as
the ``R-l2-loss`` / ``R-mse-loss`` sister falsifier via explicit
``loss_form="mse"``.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.ssl.slot_loss import masked_mse_slot_time


def test_masked_loss_default_is_l1() -> None:
    """B26 lock 2026-05-27 PM: default ``loss_form`` is pure L1.
    When every slot is supervised, masked-L1 = plain L1 over the
    ``(B, L, T_p, d)`` tensor."""
    torch.manual_seed(0)
    s = torch.randn(2, 6, 4, 32)
    t = torch.randn(2, 6, 4, 32)
    mask = torch.ones(2, 6, dtype=torch.bool)
    got = masked_mse_slot_time(s, t, latent_valid=mask)
    expected = torch.nn.functional.l1_loss(s, t)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_masked_mse_matches_torch_mse_when_all_slots_supervised() -> None:
    """``R-l2-loss`` / ``R-mse-loss`` sister: explicit ``loss_form="mse"``
    selects MSE (data2vec 2.0 §3.1 convention). When every slot is
    supervised, masked-MSE = plain MSE over the ``(B, L, T_p, d)`` tensor."""
    torch.manual_seed(0)
    s = torch.randn(2, 6, 4, 32)
    t = torch.randn(2, 6, 4, 32)
    mask = torch.ones(2, 6, dtype=torch.bool)
    got = masked_mse_slot_time(s, t, latent_valid=mask, loss_form="mse")
    expected = torch.nn.functional.mse_loss(s, t)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_masked_mse_zero_for_identical_inputs() -> None:
    torch.manual_seed(0)
    s = torch.randn(2, 6, 4, 32)
    mask = torch.ones(2, 6, dtype=torch.bool)
    loss = masked_mse_slot_time(s, s, latent_valid=mask)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_masked_mse_ignores_unsupervised_slot_diffs() -> None:
    """A perturbation confined to an unsupervised slot must not change the
    masked-MSE — that's the whole point of the latent_valid gate."""
    torch.manual_seed(0)
    s = torch.randn(1, 6, 4, 8)
    t = torch.randn(1, 6, 4, 8)
    mask = torch.tensor([[True, True, True, False, False, False]])
    loss_a = masked_mse_slot_time(s, t, latent_valid=mask)

    s_pert = s.clone()
    s_pert[:, 3:] = 100.0  # garbage in unsupervised slots
    loss_b = masked_mse_slot_time(s_pert, t, latent_valid=mask)
    torch.testing.assert_close(loss_a, loss_b, atol=1e-7, rtol=1e-7)


def test_masked_mse_divisor_is_supervised_slots_x_T_p_x_d() -> None:
    """Numerator counts squared error in supervised cells; divisor is
    Σ_b |supervised_slots(b)| × T_p × d. Constant nonzero error in supervised
    slots gives loss = err² regardless of how many slots are supervised.
    Pinned to ``loss_form="mse"`` (R-l2-loss sister) for the closed-form
    math; the same divisor contract holds for L1."""
    B, L, T_p, d = 2, 8, 3, 16
    s = torch.zeros(B, L, T_p, d)
    t = torch.zeros(B, L, T_p, d)
    s[:, : L // 2] = 1.0           # constant error in supervised half
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, : L // 2] = True       # half supervised
    loss = masked_mse_slot_time(s, t, latent_valid=mask, loss_form="mse")
    # err² = 1.0 over supervised cells, divisor normalizes correctly.
    torch.testing.assert_close(loss, torch.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_masked_l1_divisor_is_supervised_slots_x_T_p_x_d() -> None:
    """B26 L1 form: same divisor contract; constant unit error in
    supervised cells gives loss = |err| = 1.0 regardless of how many
    slots are supervised."""
    B, L, T_p, d = 2, 8, 3, 16
    s = torch.zeros(B, L, T_p, d)
    t = torch.zeros(B, L, T_p, d)
    s[:, : L // 2] = 1.0
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, : L // 2] = True
    loss = masked_mse_slot_time(s, t, latent_valid=mask)  # default L1
    torch.testing.assert_close(loss, torch.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_masked_mse_gradient_flows_to_student_only_when_teacher_detached() -> None:
    """LOSS-02 / LOSS-03 require the teacher be detached upstream (EMA target).
    Verifies gradient flows only into the student branch when the caller
    passes ``teacher.detach()``.
    """
    torch.manual_seed(0)
    s = torch.randn(2, 6, 4, 8, requires_grad=True)
    t = torch.randn(2, 6, 4, 8, requires_grad=True)
    mask = torch.ones(2, 6, dtype=torch.bool)
    loss = masked_mse_slot_time(s, t.detach(), latent_valid=mask)
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert t.grad is None


def test_masked_mse_zero_loss_when_no_slots_supervised() -> None:
    """Empty supervised set on every batch row → denominator clamped to 1.0,
    numerator masked to 0 → loss = 0. Edge case for SWEC-style subjects
    when the upstream slot-mask helper did NOT expand to all-True (caller
    contract violation; primitive must not crash)."""
    s = torch.randn(2, 6, 4, 8)
    t = torch.randn(2, 6, 4, 8)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    loss = masked_mse_slot_time(s, t, latent_valid=mask)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_masked_mse_rejects_shape_mismatch() -> None:
    s = torch.randn(2, 6, 4, 8)
    t = torch.randn(2, 6, 4, 16)
    mask = torch.ones(2, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match="shape mismatch"):
        masked_mse_slot_time(s, t, latent_valid=mask)


def test_masked_mse_rejects_wrong_mask_shape() -> None:
    s = torch.randn(2, 6, 4, 8)
    t = torch.randn(2, 6, 4, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="latent_valid shape"):
        masked_mse_slot_time(s, t, latent_valid=mask)


def test_masked_mse_rejects_non_bool_mask() -> None:
    s = torch.randn(2, 6, 4, 8)
    t = torch.randn(2, 6, 4, 8)
    mask = torch.ones(2, 6, dtype=torch.float32)
    with pytest.raises(TypeError, match="latent_valid dtype"):
        masked_mse_slot_time(s, t, latent_valid=mask)


def test_masked_loss_rejects_unknown_loss_form() -> None:
    """B26: unknown ``loss_form`` must fail loudly."""
    s = torch.randn(2, 6, 4, 8)
    t = torch.randn(2, 6, 4, 8)
    mask = torch.ones(2, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match="loss_form"):
        masked_mse_slot_time(s, t, latent_valid=mask, loss_form="huber")  # type: ignore[arg-type]
