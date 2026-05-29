"""Tests for LOSS-04 utterance loss + PMA-then-mean helper.

B26 amendment 2026-05-27 PM: default ``loss_form="l1"`` (V-JEPA 2 §2.1
Eq 1 + V-JEPA 2.1 §2.3.1 Eqs 1+2). MSE retained as ``R-l2-loss`` /
``R-mse-loss`` sister via explicit ``loss_form="mse"``.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_encoder import V14ParcelCollapsePMA
from speech_decoding.ssl.utterance_loss import pma_then_mean, utterance_mse_loss


def test_utterance_loss_default_is_l1() -> None:
    """B26 lock 2026-05-27 PM: default ``loss_form`` is pure L1."""
    torch.manual_seed(0)
    s = torch.randn(4, 256)
    t = torch.randn(4, 256)
    expected = torch.nn.functional.l1_loss(s, t)
    got = utterance_mse_loss(s, t)
    torch.testing.assert_close(got, expected, atol=1e-7, rtol=1e-7)


def test_utterance_mse_matches_torch_reference() -> None:
    """``R-l2-loss`` / ``R-mse-loss`` sister: explicit ``loss_form="mse"``
    selects MSE (data2vec 2.0 §3.1 convention)."""
    torch.manual_seed(0)
    s = torch.randn(4, 256)
    t = torch.randn(4, 256)
    expected = torch.nn.functional.mse_loss(s, t)
    got = utterance_mse_loss(s, t, loss_form="mse")
    torch.testing.assert_close(got, expected, atol=1e-7, rtol=1e-7)


def test_utterance_mse_zero_for_identical_inputs() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 256)
    assert torch.allclose(utterance_mse_loss(x, x), torch.zeros(()), atol=1e-7)


def test_utterance_mse_rejects_shape_mismatch() -> None:
    s = torch.randn(4, 256)
    t = torch.randn(4, 128)
    with pytest.raises(ValueError, match="shape mismatch"):
        utterance_mse_loss(s, t)


def test_utterance_mse_rejects_non_2d_input() -> None:
    s = torch.randn(4, 50, 256)
    t = torch.randn(4, 50, 256)
    with pytest.raises(ValueError, match=r"\(B, d\)"):
        utterance_mse_loss(s, t)


def test_pma_then_mean_shape_and_path() -> None:
    """LOSS-04 path: PMA(M4, latent_valid) → (B, T_p, d) →
    mean(T_p) → (B, d). Trainable PMA (``freeze=False``). Returns the
    canonical ``(out, clip_valid)`` tuple from the B30 aggregator
    contract."""
    torch.manual_seed(0)
    B, L, T_p, d = 2, 12, 5, 32
    pma = V14ParcelCollapsePMA(d_model=d, n_heads=4, freeze=False)
    m4 = torch.randn(B, L, T_p, d)
    latent_valid = torch.ones(B, L, dtype=torch.bool)
    out, clip_valid = pma_then_mean(pma, m4, latent_valid=latent_valid)
    assert out.shape == (B, d)
    assert clip_valid.shape == (B,)
    assert clip_valid.dtype == torch.bool
    assert clip_valid.all()
    assert torch.isfinite(out).all()


def test_pma_then_mean_honors_latent_valid() -> None:
    """B30 single-source-of-truth: when a slot's ``latent_valid`` is False,
    perturbing its row of M4 must not change the PMA-then-mean output."""
    torch.manual_seed(0)
    B, L, T_p, d = 1, 8, 3, 32
    pma = V14ParcelCollapsePMA(d_model=d, n_heads=4, freeze=False)
    pma.eval()
    m4_a = torch.randn(B, L, T_p, d)
    m4_b = m4_a.clone()
    m4_b[:, 5:] = 999.0  # garbage in slots with latent_valid=False
    mask = torch.tensor([[True, True, True, True, True, False, False, False]])
    out_a, _ = pma_then_mean(pma, m4_a, latent_valid=mask)
    out_b, _ = pma_then_mean(pma, m4_b, latent_valid=mask)
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


def test_pma_then_mean_zero_active_returns_zeros() -> None:
    """B30 zero-active guard ([[v14-anatomy-gated-symmetric-2026-05-28]]):
    a clip with ``latent_valid.sum() == 0`` (SWEC default) must return
    a structural zero of shape ``(d,)`` without invoking PMA's softmax
    over an all-False key set. ``clip_valid`` must be all-False so the
    caller knows to skip ``L_post_utterance``."""
    torch.manual_seed(0)
    B, L, T_p, d = 2, 8, 3, 32
    pma = V14ParcelCollapsePMA(d_model=d, n_heads=4, freeze=False)
    m4 = torch.randn(B, L, T_p, d)
    mask = torch.zeros(B, L, dtype=torch.bool)  # all clips no-anatomy
    out, clip_valid = pma_then_mean(pma, m4, latent_valid=mask)
    assert out.shape == (B, d)
    assert torch.allclose(out, torch.zeros_like(out))
    assert clip_valid.shape == (B,)
    assert not clip_valid.any()


def test_pma_then_mean_mixed_batch_routes_active_only() -> None:
    """B30 zero-active guard, mixed batch: active clips go through PMA;
    inactive clips get a zero (d,) without breaking the active rows.
    ``clip_valid`` flags which rows are zero-structural so the caller can
    exclude them from ``L_post_utterance``."""
    torch.manual_seed(0)
    B, L, T_p, d = 3, 8, 3, 32
    pma = V14ParcelCollapsePMA(d_model=d, n_heads=4, freeze=False)
    pma.eval()
    m4 = torch.randn(B, L, T_p, d)
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[0, :4] = True   # clip 0: active
    # clip 1: all False (SWEC-like)
    mask[2, :2] = True   # clip 2: active
    out, clip_valid = pma_then_mean(pma, m4, latent_valid=mask)
    assert out.shape == (B, d)
    assert torch.equal(clip_valid, torch.tensor([True, False, True]))
    assert torch.allclose(out[1], torch.zeros(d))
    assert not torch.allclose(out[0], torch.zeros(d))
    assert not torch.allclose(out[2], torch.zeros(d))
    # Active rows must equal the value PMA would produce when called on
    # the active-subset alone (route equivalence).
    out_subset, _ = pma_then_mean(pma, m4[[0, 2]], latent_valid=mask[[0, 2]])
    torch.testing.assert_close(out[[0, 2]], out_subset, atol=1e-6, rtol=1e-6)


def test_pma_then_mean_rejects_wrong_latent_valid_shape() -> None:
    """BUG-3 (5/28 slot-loss review): ``pma_then_mean`` must validate
    ``latent_valid`` against ``(B, L)`` like ``masked_mse_slot_time`` does
    — silent shape drift would mis-mask under per-subject variable L."""
    pma = V14ParcelCollapsePMA(d_model=32, n_heads=4, freeze=False)
    m4 = torch.randn(2, 6, 4, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="latent_valid shape"):
        pma_then_mean(pma, m4, latent_valid=mask)


def test_pma_then_mean_rejects_non_bool_latent_valid() -> None:
    """BUG-3: ``pma_then_mean`` must reject non-bool ``latent_valid``
    — boolean is the load-bearing dtype contract for the SA-mask /
    loss-gating single source of truth."""
    pma = V14ParcelCollapsePMA(d_model=32, n_heads=4, freeze=False)
    m4 = torch.randn(2, 6, 4, 32)
    mask = torch.ones(2, 6, dtype=torch.float32)
    with pytest.raises(TypeError, match="latent_valid dtype"):
        pma_then_mean(pma, m4, latent_valid=mask)


def test_utterance_loss_clip_valid_zero_dilution_contract() -> None:
    """BUG-1 + CHECK-3 close (5/28 slot-loss review): when the
    aggregator passes ``clip_valid`` derived from ``pma_then_mean``'s
    second return value, inactive rows must contribute zero to BOTH the
    numerator AND the divisor, so the active-clip gradient signal is
    not diluted by SWEC / no-anatomy rows.

    Setup: 4 clips, 2 active. The loss with ``clip_valid`` must equal
    the loss computed on the active subset alone."""
    torch.manual_seed(0)
    B, d = 4, 32
    s = torch.randn(B, d)
    t = torch.randn(B, d)
    s[1] = 0.0
    t[1] = 0.0  # inactive PMA outputs are structural zeros
    s[3] = 0.0
    t[3] = 0.0
    clip_valid = torch.tensor([True, False, True, False])

    got = utterance_mse_loss(s, t, clip_valid=clip_valid)
    expected = utterance_mse_loss(s[[0, 2]], t[[0, 2]])
    torch.testing.assert_close(got, expected, atol=1e-7, rtol=1e-7)


def test_utterance_loss_clip_valid_all_zero_is_zero_with_divisor_clamp() -> None:
    """BUG-1: degenerate all-inactive batch under ``clip_valid`` must
    return zero (divisor clamp at 1.0), NOT NaN."""
    torch.manual_seed(0)
    s = torch.zeros(3, 32)
    t = torch.zeros(3, 32)
    clip_valid = torch.zeros(3, dtype=torch.bool)
    loss = utterance_mse_loss(s, t, clip_valid=clip_valid)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_utterance_loss_rejects_wrong_clip_valid_shape() -> None:
    """BUG-3 mirror: ``clip_valid`` shape validated against ``(B,)``."""
    s = torch.randn(4, 32)
    t = torch.randn(4, 32)
    clip_valid = torch.ones(3, dtype=torch.bool)
    with pytest.raises(ValueError, match="clip_valid shape"):
        utterance_mse_loss(s, t, clip_valid=clip_valid)


def test_utterance_loss_rejects_non_bool_clip_valid() -> None:
    """BUG-3 mirror: ``clip_valid`` dtype must be bool."""
    s = torch.randn(4, 32)
    t = torch.randn(4, 32)
    clip_valid = torch.ones(4, dtype=torch.float32)
    with pytest.raises(TypeError, match="clip_valid dtype"):
        utterance_mse_loss(s, t, clip_valid=clip_valid)


def test_utterance_loss_mse_with_clip_valid_matches_subset() -> None:
    """R-l2-loss sister: ``clip_valid`` contract also holds for MSE form."""
    torch.manual_seed(0)
    B, d = 4, 32
    s = torch.randn(B, d)
    t = torch.randn(B, d)
    s[1] = 0.0
    t[1] = 0.0
    clip_valid = torch.tensor([True, False, True, True])

    got = utterance_mse_loss(s, t, clip_valid=clip_valid, loss_form="mse")
    expected = utterance_mse_loss(
        s[[0, 2, 3]], t[[0, 2, 3]], loss_form="mse"
    )
    torch.testing.assert_close(got, expected, atol=1e-7, rtol=1e-7)


def test_utterance_loss_gradient_flows_only_to_student() -> None:
    """End-to-end gradient gating: detached teacher must not receive grad.
    Works for both default L1 and MSE sister; default suffices here."""
    torch.manual_seed(0)
    s = torch.randn(4, 32, requires_grad=True)
    t = torch.randn(4, 32, requires_grad=True)
    loss = utterance_mse_loss(s, t.detach())
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert t.grad is None


def test_utterance_loss_rejects_unknown_loss_form() -> None:
    """B26: unknown ``loss_form`` must fail loudly."""
    s = torch.randn(4, 32)
    t = torch.randn(4, 32)
    with pytest.raises(ValueError, match="loss_form"):
        utterance_mse_loss(s, t, loss_form="huber")  # type: ignore[arg-type]
