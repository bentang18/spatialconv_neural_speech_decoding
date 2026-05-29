"""Tests for LOSS-05: slot-bank uniformity over M4."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.ssl.dkoleo import dkoleo_m4_loss
from speech_decoding.ssl.koleo import koleo_loss


def test_dkoleo_m4_matches_per_batch_koleo_on_pooled_slots() -> None:
    """For B=1, the loss is exactly ``koleo_loss(slot_vecs.mean(time))``."""
    torch.manual_seed(0)
    m4 = torch.randn(1, 32, 4, 16)
    expected = koleo_loss(m4.mean(dim=2)[0])
    got = dkoleo_m4_loss(m4)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_dkoleo_m4_averages_across_batch() -> None:
    """For B>1, the loss is the mean of per-batch koleo losses."""
    torch.manual_seed(0)
    m4 = torch.randn(3, 32, 4, 16)
    pooled = m4.mean(dim=2)
    expected = torch.stack([koleo_loss(pooled[b]) for b in range(3)]).mean()
    got = dkoleo_m4_loss(m4)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_dkoleo_m4_penalizes_collapsed_slots() -> None:
    """When every slot collapses to the same vector, koleo loss is large
    (-log of near-zero distance = +inf). A spread-out slot bank gives a
    smaller (more negative-ish) loss."""
    d, L = 16, 32
    # Collapsed: every slot identical.
    collapsed = torch.ones(1, L, 4, d)
    # Spread out: random independent slots.
    torch.manual_seed(0)
    spread = torch.randn(1, L, 4, d)
    loss_collapsed = dkoleo_m4_loss(collapsed)
    loss_spread = dkoleo_m4_loss(spread)
    assert loss_collapsed > loss_spread, (
        f"collapsed slots should have LARGER koleo (worse uniformity); "
        f"got collapsed={loss_collapsed.item():.4f} spread={loss_spread.item():.4f}"
    )


def test_dkoleo_m4_gradient_is_finite_and_nonzero() -> None:
    torch.manual_seed(0)
    m4 = torch.randn(2, 32, 4, 16, requires_grad=True)
    loss = dkoleo_m4_loss(m4)
    loss.backward()
    assert m4.grad is not None and torch.isfinite(m4.grad).all()
    assert m4.grad.abs().sum() > 0, "loss must produce nonzero gradient"


def test_dkoleo_m4_rejects_wrong_ndim() -> None:
    with pytest.raises(ValueError, match=r"\(B, L, T_p, d\)"):
        dkoleo_m4_loss(torch.randn(4, 32, 16))
