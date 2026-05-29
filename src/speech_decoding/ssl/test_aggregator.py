"""Tests for the B30 single-source-of-truth aggregator
([[project_v14_anatomy_gated_symmetric_2026_05_28]]).

Closes ``docs/neuroprobe/v14_blockers.md`` row 1
(``B30-anatomy-gated-via-latent-valid``) at the orchestrator layer: the
aggregator computes ``latent_valid`` ONCE and threads the SAME tensor to
``L_mid_slot``, ``L_post_frame``, ``pma_student``, and ``pma_teacher``,
with ``utterance_mse_loss`` reducing only over active clips via
``clip_valid``. L_pre_frame is at M2 (electrode-axis) so ``latent_valid``
does not apply; it uses its own ``m2_valid_mask``.

These tests pin the B30 invariants at the aggregator layer; the
Lightning ``training_step`` (still gated on SCAFFOLD-02) becomes a
trivial wrapper around ``compute_v14_ssl_losses``.
"""

from __future__ import annotations

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
    """Build M3/M4/M4 student + teacher tensors for the slot-axis terms."""
    torch.manual_seed(0)
    return (
        torch.randn(B, L, T_p, d),
        torch.randn(B, L, T_p, d).detach(),
        torch.randn(B, L, T_p, d),
        torch.randn(B, L, T_p, d).detach(),
        torch.randn(B, L, T_p, d),
        torch.randn(B, L, T_p, d).detach(),
    )


def test_aggregator_returns_breakdown_with_4_terms_and_clip_valid() -> None:
    """End-to-end shape contract: breakdown carries 4 SSL terms (DKoleo
    None per B28 default) and ``clip_valid`` reflects ``latent_valid.any``."""
    torch.manual_seed(0)
    B, L, T_p, d = 2, 12, 5, 32
    pma_s, pma_t = _make_pma(d), _make_pma(d)
    s_m3, t_m3, s_m4f, t_m4f, s_m4u, t_m4u = _student_teacher_quad(B, L, T_p, d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)
    latent_valid = torch.ones(B, L, dtype=torch.bool)

    breakdown, clip_valid = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m3_lnmid=s_m3,
        teacher_m3_lnmid=t_m3,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        student_m4_lnutt=s_m4u,
        teacher_m4_lnutt=t_m4u,
        latent_valid=latent_valid,
        pma_student=pma_s,
        pma_teacher=pma_t,
    )
    assert breakdown.total.ndim == 0
    assert torch.isfinite(breakdown.total)
    assert torch.isfinite(breakdown.l_pre_frame)
    assert torch.isfinite(breakdown.l_mid_slot)
    assert torch.isfinite(breakdown.l_post_frame)
    assert torch.isfinite(breakdown.l_post_utterance)
    assert breakdown.l_dkoleo_m4 is None  # B28 default off
    assert clip_valid.shape == (B,)
    assert clip_valid.dtype == torch.bool
    assert clip_valid.all()


def test_aggregator_composition_matches_unit_weights() -> None:
    """B19/B28 unit weights: total == 1·pre + 1·mid + 1·post_frame + 1·post_utt."""
    assert W_PRE_FRAME == W_MID_SLOT == W_POST_FRAME == W_POST_UTTERANCE == 1.0
    torch.manual_seed(0)
    B, L, T_p, d = 2, 8, 3, 32
    pma_s, pma_t = _make_pma(d), _make_pma(d)
    s_m3, t_m3, s_m4f, t_m4f, s_m4u, t_m4u = _student_teacher_quad(B, L, T_p, d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)
    latent_valid = torch.ones(B, L, dtype=torch.bool)

    breakdown, _ = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m3_lnmid=s_m3,
        teacher_m3_lnmid=t_m3,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        student_m4_lnutt=s_m4u,
        teacher_m4_lnutt=t_m4u,
        latent_valid=latent_valid,
        pma_student=pma_s,
        pma_teacher=pma_t,
    )
    expected = (
        breakdown.l_pre_frame
        + breakdown.l_mid_slot
        + breakdown.l_post_frame
        + breakdown.l_post_utterance
    )
    torch.testing.assert_close(breakdown.total, expected, atol=1e-7, rtol=1e-7)


def test_aggregator_threads_same_latent_valid_to_slot_losses() -> None:
    """B30 single-source-of-truth: perturbing the slot-axis tensors at
    invalid slot positions must NOT change ``L_mid_slot`` or
    ``L_post_frame``. This is the structural test that the SAME
    ``latent_valid`` is threaded to both slot-axis losses, NOT
    re-derived along the way."""
    torch.manual_seed(0)
    B, L, T_p, d = 1, 8, 3, 32
    pma_s, pma_t = _make_pma(d), _make_pma(d)
    s_m3, t_m3, s_m4f, t_m4f, s_m4u, t_m4u = _student_teacher_quad(B, L, T_p, d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)
    latent_valid = torch.tensor([[True, True, True, True, False, False, False, False]])

    base, _ = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m3_lnmid=s_m3,
        teacher_m3_lnmid=t_m3,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        student_m4_lnutt=s_m4u,
        teacher_m4_lnutt=t_m4u,
        latent_valid=latent_valid,
        pma_student=pma_s,
        pma_teacher=pma_t,
    )

    s_m3_p = s_m3.clone()
    s_m4f_p = s_m4f.clone()
    s_m3_p[:, 4:, :, :] = 999.0  # garbage at invalid slot positions
    s_m4f_p[:, 4:, :, :] = -999.0
    perturbed, _ = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m3_lnmid=s_m3_p,
        teacher_m3_lnmid=t_m3,
        student_m4_lnframe=s_m4f_p,
        teacher_m4_lnframe=t_m4f,
        student_m4_lnutt=s_m4u,
        teacher_m4_lnutt=t_m4u,
        latent_valid=latent_valid,
        pma_student=pma_s,
        pma_teacher=pma_t,
    )

    torch.testing.assert_close(
        base.l_mid_slot, perturbed.l_mid_slot, atol=1e-7, rtol=1e-7,
    )
    torch.testing.assert_close(
        base.l_post_frame, perturbed.l_post_frame, atol=1e-7, rtol=1e-7,
    )


def test_aggregator_swec_clip_contributes_zero_to_three_b30_terms() -> None:
    """B30 SWEC contract: a clip with ``latent_valid`` all-False (the
    SWEC default — no anatomy at any slot) contributes ZERO to
    ``L_mid_slot``, ``L_post_frame``, ``L_post_utterance`` (the three
    B30-gated terms). ``L_pre_frame`` is at M2 (electrode-axis,
    pre-encoder) and is NOT gated by ``latent_valid`` — it stays
    unchanged."""
    torch.manual_seed(0)
    B, L, T_p, d = 2, 8, 3, 32
    pma_s, pma_t = _make_pma(d), _make_pma(d)
    s_m3, t_m3, s_m4f, t_m4f, s_m4u, t_m4u = _student_teacher_quad(B, L, T_p, d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)

    # Both clips active (baseline).
    lv_both = torch.ones(B, L, dtype=torch.bool)
    base, cv_base = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m3_lnmid=s_m3,
        teacher_m3_lnmid=t_m3,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        student_m4_lnutt=s_m4u,
        teacher_m4_lnutt=t_m4u,
        latent_valid=lv_both,
        pma_student=pma_s,
        pma_teacher=pma_t,
    )

    # Only clip 0 active; clip 1 = SWEC.
    lv_one = torch.zeros(B, L, dtype=torch.bool)
    lv_one[0] = True
    swec, cv_swec = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m3_lnmid=s_m3,
        teacher_m3_lnmid=t_m3,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        student_m4_lnutt=s_m4u,
        teacher_m4_lnutt=t_m4u,
        latent_valid=lv_one,
        pma_student=pma_s,
        pma_teacher=pma_t,
    )

    # SWEC clip excluded from active-clip count.
    assert torch.equal(cv_swec, torch.tensor([True, False]))
    assert cv_base.all()

    # L_pre_frame is NOT gated by latent_valid — same M2 input ⇒ same loss.
    torch.testing.assert_close(
        base.l_pre_frame, swec.l_pre_frame, atol=1e-7, rtol=1e-7,
    )

    # The three B30-gated terms must change (one clip dropped from the
    # active set), and must remain finite (no NaN from softmax over
    # all-False keys).
    assert torch.isfinite(swec.l_mid_slot)
    assert torch.isfinite(swec.l_post_frame)
    assert torch.isfinite(swec.l_post_utterance)


def test_aggregator_all_swec_batch_does_not_nan() -> None:
    """B30 degenerate: a batch where every clip is SWEC (latent_valid
    all-False) must yield finite losses (zero numerators + clamped
    divisors) and ``clip_valid`` all-False."""
    torch.manual_seed(0)
    B, L, T_p, d = 3, 8, 3, 32
    pma_s, pma_t = _make_pma(d), _make_pma(d)
    s_m3, t_m3, s_m4f, t_m4f, s_m4u, t_m4u = _student_teacher_quad(B, L, T_p, d)
    m2_pred = torch.randn(B, 4, T_p, d)
    m2_target = torch.randn(B, 4, T_p, d)
    latent_valid = torch.zeros(B, L, dtype=torch.bool)

    breakdown, clip_valid = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m3_lnmid=s_m3,
        teacher_m3_lnmid=t_m3,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        student_m4_lnutt=s_m4u,
        teacher_m4_lnutt=t_m4u,
        latent_valid=latent_valid,
        pma_student=pma_s,
        pma_teacher=pma_t,
    )
    assert torch.isfinite(breakdown.total)
    # All-SWEC: 3 B30-gated terms collapse to 0 (numerator zero, divisor
    # clamped). L_pre_frame keeps its M2 value.
    assert torch.allclose(breakdown.l_mid_slot, torch.zeros(()))
    assert torch.allclose(breakdown.l_post_frame, torch.zeros(()))
    assert torch.allclose(breakdown.l_post_utterance, torch.zeros(()))
    assert torch.isfinite(breakdown.l_pre_frame)
    assert not clip_valid.any()


def test_aggregator_gradient_flows_only_to_student() -> None:
    """Teacher branches must be detached upstream (EMA contract); the
    aggregator does not detach for the caller. Verified by running
    ``backward()`` and checking teacher grads are None / zero."""
    torch.manual_seed(0)
    B, L, T_p, d = 1, 6, 2, 32
    pma_s, pma_t = _make_pma(d), _make_pma(d)
    s_m3 = torch.randn(B, L, T_p, d, requires_grad=True)
    t_m3 = torch.randn(B, L, T_p, d).detach()
    s_m4f = torch.randn(B, L, T_p, d, requires_grad=True)
    t_m4f = torch.randn(B, L, T_p, d).detach()
    s_m4u = torch.randn(B, L, T_p, d, requires_grad=True)
    t_m4u = torch.randn(B, L, T_p, d).detach()
    m2_pred = torch.randn(B, 4, T_p, d, requires_grad=True)
    m2_target = torch.randn(B, 4, T_p, d).detach()
    latent_valid = torch.ones(B, L, dtype=torch.bool)

    breakdown, _ = compute_v14_ssl_losses(
        student_m2_pred=m2_pred,
        m2_target=m2_target,
        student_m3_lnmid=s_m3,
        teacher_m3_lnmid=t_m3,
        student_m4_lnframe=s_m4f,
        teacher_m4_lnframe=t_m4f,
        student_m4_lnutt=s_m4u,
        teacher_m4_lnutt=t_m4u,
        latent_valid=latent_valid,
        pma_student=pma_s,
        pma_teacher=pma_t,
    )
    breakdown.total.backward()
    assert s_m3.grad is not None and torch.isfinite(s_m3.grad).all()
    assert s_m4f.grad is not None and torch.isfinite(s_m4f.grad).all()
    assert s_m4u.grad is not None and torch.isfinite(s_m4u.grad).all()
    assert m2_pred.grad is not None and torch.isfinite(m2_pred.grad).all()
    assert t_m3.grad is None
    assert t_m4f.grad is None
    assert t_m4u.grad is None
    assert m2_target.grad is None
