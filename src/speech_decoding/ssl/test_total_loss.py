"""Tests for LOSS-01: the B19/B22/B28/B31 composer with locked coefficients.

B31 amendment (2026-05-28 PM-late, V-JEPA-2-canonical loss simplification):
the joint SSL default collapses to a 2-term surface
(``L_pre_frame + L_post_frame``, both pure L1 per V-JEPA 2 §2.1 Eq 1).
``L_mid_slot`` and ``L_post_utterance`` are reconstructed only via the
falsifier sisters and are passed as Optional kwargs.

B28 amendment (2026-05-27 PM): DKoleo @ M4 is demoted to an opt-in sister
term. ``W_DKOLEO_M4 = 0.1`` constant is retained for the
``R-dkoleo-{intra_clip_slots,batch_cls}`` sisters but only applied when
``l_dkoleo_m4`` is passed.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.ssl.total_loss import (
    V14TotalLossBreakdown,
    W_DKOLEO_M3_REACTIVE,
    W_DKOLEO_M4,
    W_GRAM_REACTIVE,
    W_MID_SLOT,
    W_POST_FRAME,
    W_POST_UTTERANCE,
    W_PRE_FRAME,
    v14_total_loss,
)


def test_coefficients_pinned() -> None:
    """LOSS-01 lock + B28 + B31 amendments: coefficients are
    ``(1, 1)`` always-on (B31 2-term default); ``W_MID_SLOT`` and
    ``W_POST_UTTERANCE`` constants retained at ``1.0`` for the B31
    falsifier-sister arms; ``0.1`` for the opt-in DKoleo arm."""
    assert W_PRE_FRAME == 1.0, "L_pre_frame coefficient drifted"
    assert W_POST_FRAME == 1.0, "L_post_frame coefficient drifted"
    # B31 falsifier-sister arms; constants retained at 1.0 because sister
    # arms reuse the historical lock weight when armed.
    assert W_MID_SLOT == 1.0, "L_mid_slot sister coefficient drifted"
    assert W_POST_UTTERANCE == 1.0, "L_post_utterance sister coefficient drifted"
    # B28: DKoleo @ M4 demoted to opt-in sister. Constant retained at the
    # historical B21 weight so sisters can share the coefficient.
    assert W_DKOLEO_M4 == 0.1, "L_DKoleo@M4 sister coefficient drifted"
    # Reactive cousins.
    assert W_DKOLEO_M3_REACTIVE == 0.05, "reactive L_DKoleo@M3 coefficient drifted"
    assert W_GRAM_REACTIVE == 0.1, "reactive L_Gram coefficient drifted"


def test_v14_total_loss_b31_default_is_two_terms() -> None:
    """B31 lock 2026-05-28 PM-late: joint SSL default = L_pre + L_post_frame
    only. ``L_mid_slot`` / ``L_post_utterance`` are None at the breakdown
    level; the coefficient tuple reports ``None`` for the inactive slots."""
    l_pre = torch.tensor(0.5)
    l_post_f = torch.tensor(0.2)
    br = v14_total_loss(l_pre, l_post_f)
    expected = 1.0 * 0.5 + 1.0 * 0.2
    torch.testing.assert_close(br.total, torch.tensor(expected), atol=1e-6, rtol=1e-6)
    # B31 dropped terms are None.
    assert br.l_mid_slot is None
    assert br.l_post_utterance is None
    # DKoleo @ M4 + reactive cousins also None under the joint default.
    assert br.l_dkoleo_m4 is None
    assert br.l_dkoleo_m3_reactive is None
    assert br.l_gram_reactive is None
    # 5-slot coefficient tuple: (pre, post_frame, mid, post_utt, dkoleo_m4).
    assert br.coefficients == (1.0, 1.0, None, None, None)
    # Round-trip the raw per-term losses for dashboards.
    assert br.l_pre_frame is l_pre
    assert br.l_post_frame is l_post_f


def test_v14_total_loss_b31_plus_m3_sister_adds_mid_slot_term() -> None:
    """``R-add-m3-loss`` sister: B31 default + ``l_mid_slot`` at locked
    weight ``W_MID_SLOT = 1.0`` reconstructs the B22 M3 supervision."""
    l_pre = torch.tensor(0.5)
    l_post_f = torch.tensor(0.2)
    l_mid = torch.tensor(0.3)
    br = v14_total_loss(l_pre, l_post_f, l_mid_slot=l_mid)
    expected = 1.0 * 0.5 + 1.0 * 0.2 + 1.0 * 0.3
    torch.testing.assert_close(br.total, torch.tensor(expected), atol=1e-6, rtol=1e-6)
    assert br.l_mid_slot is l_mid
    assert br.l_post_utterance is None
    assert br.coefficients == (1.0, 1.0, 1.0, None, None)


def test_v14_total_loss_b31_plus_utt_sister_adds_utterance_term() -> None:
    """``R-add-utterance-loss`` sister (EAT-faithful comparator): B31
    default + ``l_post_utterance`` at locked weight ``W_POST_UTTERANCE
    = 1.0`` reconstructs the B19 utterance head."""
    l_pre = torch.tensor(0.5)
    l_post_f = torch.tensor(0.2)
    l_post_u = torch.tensor(0.4)
    br = v14_total_loss(l_pre, l_post_f, l_post_utterance=l_post_u)
    expected = 1.0 * 0.5 + 1.0 * 0.2 + 1.0 * 0.4
    torch.testing.assert_close(br.total, torch.tensor(expected), atol=1e-6, rtol=1e-6)
    assert br.l_mid_slot is None
    assert br.l_post_utterance is l_post_u
    assert br.coefficients == (1.0, 1.0, None, 1.0, None)


def test_v14_total_loss_b31_plus_both_sister_adds_both_dropped_terms() -> None:
    """``R-add-both`` sister: B31 default + both dropped terms restored
    at locked weight ``1.0`` each."""
    l_pre = torch.tensor(0.5)
    l_post_f = torch.tensor(0.2)
    l_mid = torch.tensor(0.3)
    l_post_u = torch.tensor(0.4)
    br = v14_total_loss(
        l_pre, l_post_f,
        l_mid_slot=l_mid, l_post_utterance=l_post_u,
    )
    expected = 1.0 * 0.5 + 1.0 * 0.2 + 1.0 * 0.3 + 1.0 * 0.4
    torch.testing.assert_close(br.total, torch.tensor(expected), atol=1e-6, rtol=1e-6)
    assert br.l_mid_slot is l_mid
    assert br.l_post_utterance is l_post_u
    assert br.coefficients == (1.0, 1.0, 1.0, 1.0, None)


def test_v14_total_loss_dkoleo_sister_adds_term_at_locked_weight() -> None:
    """``R-dkoleo-intra-clip-slots`` / ``R-dkoleo-batch-cls-unit`` sisters
    pass ``l_dkoleo_m4`` through; the composer adds ``0.1 · l_dk`` to the
    2-term B31 default and bubbles the per-term tensor + active coefficient."""
    l_pre = torch.tensor(0.5)
    l_post_f = torch.tensor(0.2)
    l_dk = torch.tensor(1.0)
    br = v14_total_loss(l_pre, l_post_f, l_dkoleo_m4=l_dk)
    expected = 1.0 * 0.5 + 1.0 * 0.2 + 0.1 * 1.0
    torch.testing.assert_close(br.total, torch.tensor(expected), atol=1e-6, rtol=1e-6)
    assert br.l_dkoleo_m4 is l_dk
    assert br.coefficients == (1.0, 1.0, None, None, 0.1)


def test_v14_total_loss_reactive_arms_add_with_locked_coefficients() -> None:
    base = torch.zeros(())
    l_dk_m3 = torch.tensor(2.0)
    l_gram = torch.tensor(3.0)
    br = v14_total_loss(
        base, base,
        l_dkoleo_m3_reactive=l_dk_m3,
        l_gram_reactive=l_gram,
    )
    expected = 0.05 * 2.0 + 0.1 * 3.0
    torch.testing.assert_close(br.total, torch.tensor(expected), atol=1e-6, rtol=1e-6)


def test_v14_total_loss_reactive_off_by_default() -> None:
    """Reactive terms must be off-default (``None``) per the spec."""
    base = torch.tensor(0.5)
    br = v14_total_loss(base, base)
    assert br.l_dkoleo_m3_reactive is None
    assert br.l_gram_reactive is None


def test_v14_total_loss_backward_propagates_through_b31_default_two_terms() -> None:
    """B31 default composer: gradient flows through the 2 always-on
    branches at the locked unit coefficients."""
    l_pre = torch.tensor(0.1, requires_grad=True)
    l_post_f = torch.tensor(0.3, requires_grad=True)
    br = v14_total_loss(l_pre, l_post_f)
    br.total.backward()
    assert l_pre.grad is not None and l_pre.grad.item() == 1.0
    assert l_post_f.grad is not None and l_post_f.grad.item() == 1.0


def test_v14_total_loss_backward_through_b31_plus_both_sister() -> None:
    """``R-add-both`` sister: gradient flows through all 4 terms at unit
    coefficients."""
    l_pre = torch.tensor(0.1, requires_grad=True)
    l_post_f = torch.tensor(0.3, requires_grad=True)
    l_mid = torch.tensor(0.2, requires_grad=True)
    l_post_u = torch.tensor(0.4, requires_grad=True)
    br = v14_total_loss(
        l_pre, l_post_f,
        l_mid_slot=l_mid, l_post_utterance=l_post_u,
    )
    br.total.backward()
    assert l_pre.grad is not None and l_pre.grad.item() == 1.0
    assert l_mid.grad is not None and l_mid.grad.item() == 1.0
    assert l_post_f.grad is not None and l_post_f.grad.item() == 1.0
    assert l_post_u.grad is not None and l_post_u.grad.item() == 1.0


def test_v14_total_loss_backward_dkoleo_sister_gradient_uses_locked_weight() -> None:
    """When ``l_dkoleo_m4`` is supplied, gradient w.r.t. it equals ``W_DKOLEO_M4``."""
    base = torch.tensor(0.1)
    l_dk = torch.tensor(0.5, requires_grad=True)
    br = v14_total_loss(base, base, l_dkoleo_m4=l_dk)
    br.total.backward()
    assert l_dk.grad is not None and pytest.approx(l_dk.grad.item()) == 0.1


def test_v14_total_loss_breakdown_is_frozen() -> None:
    """V14TotalLossBreakdown is frozen — once produced, no field can be reassigned."""
    base = torch.zeros(())
    br = v14_total_loss(base, base)
    assert isinstance(br, V14TotalLossBreakdown)
    with pytest.raises((AttributeError, Exception)):
        br.total = torch.tensor(1.0)  # type: ignore[misc]
