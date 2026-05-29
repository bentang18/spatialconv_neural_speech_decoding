"""Tests for the B30 single-source-of-truth aggregator
([[project_v14_anatomy_gated_symmetric_2026_05_28]]) under the B31
V-JEPA-2-canonical 2-term simplification
([[project_v14_b31_vjepa2_canonical_loss_2026_05_28]]).

The B31 default joint SSL surface is::

    L_total = 1.0 · L_pre_frame  @ M2
            + 1.0 · L_post_frame @ LN_frame(M4)

with both pure L1. ``L_mid_slot`` and ``L_post_utterance`` are
reconstructed only via the falsifier sisters (``loss_variant`` ∈
{``b31_plus_m3``, ``b31_plus_utt``, ``b31_plus_both``}) and must be
passed via the matching Optional kwargs.

These tests pin two layers of contract together:
1. B31 — the variant-gated arm semantics (default 2-term + sister adds).
2. B30 — the SAME ``latent_valid`` threaded through every slot-axis
   term whenever that term is active.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_encoder import V14ParcelCollapsePMA
from speech_decoding.ssl.aggregator import compute_v14_ssl_losses
from speech_decoding.ssl.total_loss import (
    W_MID_SLOT,
    W_POST_FRAME,
    W_POST_UTTERANCE,
    W_PRE_FRAME,
)


def _make_pma(d: int = 32) -> V14ParcelCollapsePMA:
    return V14ParcelCollapsePMA(d_model=d, n_heads=4, freeze=False)


def _student_teacher_quad(B: int, L: int, T_p: int, d: int) -> tuple[torch.Tensor, ...]:
    """Build M3/M4/M4 student + teacher tensors for the slot-axis terms.

    Six tensors in canonical order: (s_m3, t_m3, s_m4f, t_m4f, s_m4u, t_m4u).
    Teacher tensors are detached upstream per the EMA contract.
    """
    torch.manual_seed(0)
    return (
        torch.randn(B, L, T_p, d),
        torch.randn(B, L, T_p, d).detach(),
        torch.randn(B, L, T_p, d),
        torch.randn(B, L, T_p, d).detach(),
        torch.randn(B, L, T_p, d),
        torch.randn(B, L, T_p, d).detach(),
    )


# ────────────────────────── B31 default (2 terms) ──────────────────────────

def test_aggregator_b31_default_returns_two_term_breakdown() -> None:
    """B31 default: ``loss_variant="b31_default"`` returns a breakdown with
    ``l_mid_slot=None`` and ``l_post_utterance=None``; the total equals the
    sum of the two B31-canonical terms at unit weight."""
    torch.manual_seed(0)
    B, L, T_p, d = 2, 12, 5, 32
    _, _, s_m4f, t_m4f, _, _ = _student_teacher_quad(B, L, T_p, d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)
    latent_valid = torch.ones(B, L, dtype=torch.bool)

    breakdown = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        latent_valid=latent_valid,
    )
    assert breakdown.total.ndim == 0
    assert torch.isfinite(breakdown.total)
    assert torch.isfinite(breakdown.l_pre_frame)
    assert torch.isfinite(breakdown.l_post_frame)
    # B31 dropped terms must be None under the default.
    assert breakdown.l_mid_slot is None
    assert breakdown.l_post_utterance is None
    assert breakdown.l_dkoleo_m4 is None
    expected = W_PRE_FRAME * breakdown.l_pre_frame + W_POST_FRAME * breakdown.l_post_frame
    torch.testing.assert_close(breakdown.total, expected, atol=1e-7, rtol=1e-7)


def test_aggregator_b31_default_rejects_m3_tensors() -> None:
    """Default variant + M3 tensors = programming error caught at runtime."""
    torch.manual_seed(0)
    B, L, T_p, d = 1, 8, 3, 32
    s_m3, t_m3, s_m4f, t_m4f, *_ = _student_teacher_quad(B, L, T_p, d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)
    with pytest.raises(ValueError, match="does not select the M3 arm"):
        compute_v14_ssl_losses(
            student_m2_pred=m2_pred,
            m2_target=m2_target,
            student_m4_lnframe=s_m4f,
            teacher_m4_lnframe=t_m4f,
            latent_valid=torch.ones(B, L, dtype=torch.bool),
            student_m3_lnmid=s_m3,
            teacher_m3_lnmid=t_m3,
        )


def test_aggregator_b31_default_rejects_utterance_tensors() -> None:
    """Default variant + utterance tensors/PMA = programming error."""
    torch.manual_seed(0)
    B, L, T_p, d = 1, 8, 3, 32
    _, _, s_m4f, t_m4f, s_m4u, t_m4u = _student_teacher_quad(B, L, T_p, d)
    pma_s, pma_t = _make_pma(d), _make_pma(d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)
    with pytest.raises(ValueError, match="does not select the utterance arm"):
        compute_v14_ssl_losses(
            student_m2_pred=m2_pred,
            m2_target=m2_target,
            student_m4_lnframe=s_m4f,
            teacher_m4_lnframe=t_m4f,
            latent_valid=torch.ones(B, L, dtype=torch.bool),
            student_m4_lnutt=s_m4u,
            teacher_m4_lnutt=t_m4u,
            pma_student=pma_s,
            pma_teacher=pma_t,
        )


# ───────────────────── B31 sister arms (3 falsifiers) ─────────────────────

def test_aggregator_b31_plus_m3_sister_adds_mid_slot_term() -> None:
    """``R-add-m3-loss`` (``b31_plus_m3``): default + M3 LN_mid supervision."""
    torch.manual_seed(0)
    B, L, T_p, d = 2, 8, 3, 32
    s_m3, t_m3, s_m4f, t_m4f, *_ = _student_teacher_quad(B, L, T_p, d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)
    latent_valid = torch.ones(B, L, dtype=torch.bool)

    breakdown = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        latent_valid=latent_valid,
        loss_variant="b31_plus_m3",
        student_m3_lnmid=s_m3,
        teacher_m3_lnmid=t_m3,
    )
    assert breakdown.l_mid_slot is not None
    assert torch.isfinite(breakdown.l_mid_slot)
    assert breakdown.l_post_utterance is None
    expected = (
        W_PRE_FRAME * breakdown.l_pre_frame
        + W_POST_FRAME * breakdown.l_post_frame
        + W_MID_SLOT * breakdown.l_mid_slot
    )
    torch.testing.assert_close(breakdown.total, expected, atol=1e-7, rtol=1e-7)


def test_aggregator_b31_plus_m3_requires_m3_tensors() -> None:
    """Selecting the M3 sister without supplying both M3 tensors is a hard error."""
    torch.manual_seed(0)
    B, L, T_p, d = 1, 8, 3, 32
    _, _, s_m4f, t_m4f, *_ = _student_teacher_quad(B, L, T_p, d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)
    with pytest.raises(ValueError, match="student_m3_lnmid \\+ teacher_m3_lnmid"):
        compute_v14_ssl_losses(
            student_m2_pred=m2_pred,
            m2_target=m2_target,
            student_m4_lnframe=s_m4f,
            teacher_m4_lnframe=t_m4f,
            latent_valid=torch.ones(B, L, dtype=torch.bool),
            loss_variant="b31_plus_m3",
        )


def test_aggregator_b31_plus_utt_sister_adds_utterance_term() -> None:
    """``R-add-utterance-loss`` (EAT-faithful comparator; ≥0.02 AUROC
    promotion gate): default + L_post_utterance via PMA pair."""
    torch.manual_seed(0)
    B, L, T_p, d = 2, 8, 3, 32
    _, _, s_m4f, t_m4f, s_m4u, t_m4u = _student_teacher_quad(B, L, T_p, d)
    pma_s, pma_t = _make_pma(d), _make_pma(d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)
    latent_valid = torch.ones(B, L, dtype=torch.bool)

    breakdown = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        latent_valid=latent_valid,
        loss_variant="b31_plus_utt",
        student_m4_lnutt=s_m4u,
        teacher_m4_lnutt=t_m4u,
        pma_student=pma_s,
        pma_teacher=pma_t,
    )
    assert breakdown.l_mid_slot is None
    assert breakdown.l_post_utterance is not None
    assert torch.isfinite(breakdown.l_post_utterance)
    expected = (
        W_PRE_FRAME * breakdown.l_pre_frame
        + W_POST_FRAME * breakdown.l_post_frame
        + W_POST_UTTERANCE * breakdown.l_post_utterance
    )
    torch.testing.assert_close(breakdown.total, expected, atol=1e-7, rtol=1e-7)


def test_aggregator_b31_plus_utt_requires_full_quartet() -> None:
    """The utterance arm needs all four of {student_lnutt, teacher_lnutt,
    pma_student, pma_teacher}; omitting any is a hard error."""
    torch.manual_seed(0)
    B, L, T_p, d = 1, 8, 3, 32
    _, _, s_m4f, t_m4f, s_m4u, t_m4u = _student_teacher_quad(B, L, T_p, d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)
    with pytest.raises(ValueError, match="student_m4_lnutt"):
        compute_v14_ssl_losses(
            student_m2_pred=m2_pred,
            m2_target=m2_target,
            student_m4_lnframe=s_m4f,
            teacher_m4_lnframe=t_m4f,
            latent_valid=torch.ones(B, L, dtype=torch.bool),
            loss_variant="b31_plus_utt",
            student_m4_lnutt=s_m4u,
            teacher_m4_lnutt=t_m4u,
            # pma_student / pma_teacher missing.
        )


def test_aggregator_b31_plus_both_sister_adds_both_dropped_terms() -> None:
    """``R-add-both``: default + L_mid_slot + L_post_utterance, all at unit weight."""
    assert W_PRE_FRAME == W_MID_SLOT == W_POST_FRAME == W_POST_UTTERANCE == 1.0
    torch.manual_seed(0)
    B, L, T_p, d = 2, 8, 3, 32
    s_m3, t_m3, s_m4f, t_m4f, s_m4u, t_m4u = _student_teacher_quad(B, L, T_p, d)
    pma_s, pma_t = _make_pma(d), _make_pma(d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)
    latent_valid = torch.ones(B, L, dtype=torch.bool)

    breakdown = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        latent_valid=latent_valid,
        loss_variant="b31_plus_both",
        student_m3_lnmid=s_m3,
        teacher_m3_lnmid=t_m3,
        student_m4_lnutt=s_m4u,
        teacher_m4_lnutt=t_m4u,
        pma_student=pma_s,
        pma_teacher=pma_t,
    )
    expected = (
        breakdown.l_pre_frame
        + breakdown.l_post_frame
        + breakdown.l_mid_slot
        + breakdown.l_post_utterance
    )
    torch.testing.assert_close(breakdown.total, expected, atol=1e-7, rtol=1e-7)


# ───────── B30 single-source-of-truth invariants under sisters ─────────

def test_aggregator_b30_threads_same_latent_valid_under_b31_plus_both() -> None:
    """B30 single-source-of-truth: perturbing slot-axis tensors at
    invalid slot positions must NOT change ``L_mid_slot`` or
    ``L_post_frame`` when both terms are active. Verifies the SAME
    ``latent_valid`` is threaded to both slot-axis losses, not
    re-derived along the way."""
    torch.manual_seed(0)
    B, L, T_p, d = 1, 8, 3, 32
    s_m3, t_m3, s_m4f, t_m4f, s_m4u, t_m4u = _student_teacher_quad(B, L, T_p, d)
    pma_s, pma_t = _make_pma(d), _make_pma(d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)
    latent_valid = torch.tensor([[True, True, True, True, False, False, False, False]])

    common: dict = dict(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        teacher_m3_lnmid=t_m3,
        teacher_m4_lnframe=t_m4f,
        student_m4_lnutt=s_m4u,
        teacher_m4_lnutt=t_m4u,
        latent_valid=latent_valid,
        loss_variant="b31_plus_both",
        pma_student=pma_s,
        pma_teacher=pma_t,
    )
    base = compute_v14_ssl_losses(
        student_m3_lnmid=s_m3,
        student_m4_lnframe=s_m4f,
        **common,
    )

    s_m3_p = s_m3.clone()
    s_m4f_p = s_m4f.clone()
    s_m3_p[:, 4:, :, :] = 999.0  # garbage at invalid slot positions
    s_m4f_p[:, 4:, :, :] = -999.0
    perturbed = compute_v14_ssl_losses(
        student_m3_lnmid=s_m3_p,
        student_m4_lnframe=s_m4f_p,
        **common,
    )

    torch.testing.assert_close(
        base.l_mid_slot, perturbed.l_mid_slot, atol=1e-7, rtol=1e-7,
    )
    torch.testing.assert_close(
        base.l_post_frame, perturbed.l_post_frame, atol=1e-7, rtol=1e-7,
    )


def test_aggregator_swec_clip_does_not_change_pre_frame_under_b31_default() -> None:
    """``L_pre_frame`` is at M2 (electrode-axis, pre-encoder) and is NOT
    gated by ``latent_valid``; dropping a clip from the active set must
    not change it."""
    torch.manual_seed(0)
    B, L, T_p, d = 2, 8, 3, 32
    _, _, s_m4f, t_m4f, *_ = _student_teacher_quad(B, L, T_p, d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)

    base = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        latent_valid=torch.ones(B, L, dtype=torch.bool),
    )

    lv_one = torch.zeros(B, L, dtype=torch.bool)
    lv_one[0] = True
    swec = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        latent_valid=lv_one,
    )

    torch.testing.assert_close(
        base.l_pre_frame, swec.l_pre_frame, atol=1e-7, rtol=1e-7,
    )


def test_aggregator_all_swec_batch_does_not_nan_under_b31_default() -> None:
    """B30 degenerate: a batch where every clip is SWEC (``latent_valid``
    all-False) must yield finite losses (zero numerators + clamped
    divisors) under the B31 2-term default."""
    torch.manual_seed(0)
    B, L, T_p, d = 3, 8, 3, 32
    _, _, s_m4f, t_m4f, *_ = _student_teacher_quad(B, L, T_p, d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)

    breakdown = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        latent_valid=torch.zeros(B, L, dtype=torch.bool),
    )
    assert torch.isfinite(breakdown.total)
    # All-SWEC: L_post_frame collapses to 0 (numerator zero, divisor
    # clamped). L_pre_frame keeps its M2 value.
    assert torch.allclose(breakdown.l_post_frame, torch.zeros(()))
    assert torch.isfinite(breakdown.l_pre_frame)


def test_aggregator_all_swec_batch_does_not_nan_under_b31_plus_both() -> None:
    """Same degenerate batch under the full ``b31_plus_both`` sister:
    all three B30-gated terms collapse to 0, no NaN from softmax over
    all-False keys in the PMA."""
    torch.manual_seed(0)
    B, L, T_p, d = 3, 8, 3, 32
    s_m3, t_m3, s_m4f, t_m4f, s_m4u, t_m4u = _student_teacher_quad(B, L, T_p, d)
    pma_s, pma_t = _make_pma(d), _make_pma(d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)

    breakdown = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        latent_valid=torch.zeros(B, L, dtype=torch.bool),
        loss_variant="b31_plus_both",
        student_m3_lnmid=s_m3,
        teacher_m3_lnmid=t_m3,
        student_m4_lnutt=s_m4u,
        teacher_m4_lnutt=t_m4u,
        pma_student=pma_s,
        pma_teacher=pma_t,
    )
    assert torch.isfinite(breakdown.total)
    assert breakdown.l_mid_slot is not None
    assert breakdown.l_post_utterance is not None
    assert torch.allclose(breakdown.l_mid_slot, torch.zeros(()))
    assert torch.allclose(breakdown.l_post_frame, torch.zeros(()))
    assert torch.allclose(breakdown.l_post_utterance, torch.zeros(()))
    assert torch.isfinite(breakdown.l_pre_frame)


# ─────────────────────── EMA gradient-flow contract ───────────────────────

def test_aggregator_gradient_flows_only_to_student_under_b31_default() -> None:
    """Teacher branches must be detached upstream (EMA contract); the
    aggregator does not detach for the caller. Verified by running
    ``backward()`` and checking teacher grads are None."""
    torch.manual_seed(0)
    B, L, T_p, d = 1, 6, 2, 32
    s_m4f = torch.randn(B, L, T_p, d, requires_grad=True)
    t_m4f = torch.randn(B, L, T_p, d).detach()
    m2_pred = torch.randn(B, 4, T_p, d, requires_grad=True)
    m2_target = torch.randn(B, 4, T_p, d).detach()
    latent_valid = torch.ones(B, L, dtype=torch.bool)

    breakdown = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        latent_valid=latent_valid,
    )
    breakdown.total.backward()
    assert s_m4f.grad is not None and torch.isfinite(s_m4f.grad).all()
    assert m2_pred.grad is not None and torch.isfinite(m2_pred.grad).all()
    assert t_m4f.grad is None
    assert m2_target.grad is None
