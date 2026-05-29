"""Tests for the KoLeo readout uniformity loss (T2.5)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.ssl.koleo import koleo_loss


def test_koleo_is_zero_for_singleton_batch() -> None:
    x = torch.randn(1, 8)
    got = koleo_loss(x)
    assert torch.equal(got, torch.zeros(()))


def test_koleo_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        koleo_loss(torch.randn(2, 3, 8))


def test_koleo_lower_for_more_spread_out_points() -> None:
    """A uniform-spread embedding bunch should have a lower KoLeo loss
    than a near-collapsed bunch (since collapsed points have tiny NN
    distances → large negative log → high loss)."""
    torch.manual_seed(0)
    # Spread: uniformly random on the sphere
    spread = torch.randn(16, 8)
    # Collapsed: clustered around a single point
    collapsed = torch.randn(16, 8) * 0.001 + torch.tensor([1.0] * 8)
    loss_spread = koleo_loss(spread)
    loss_collapsed = koleo_loss(collapsed)
    assert loss_spread < loss_collapsed


def test_koleo_returns_finite_value() -> None:
    """Even on adversarial near-collinear inputs the clamp(min=eps) must
    keep the result finite."""
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    got = koleo_loss(x)
    assert torch.isfinite(got)


def test_koleo_gradient_flow() -> None:
    x = torch.randn(8, 16, requires_grad=True)
    loss = koleo_loss(x)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
