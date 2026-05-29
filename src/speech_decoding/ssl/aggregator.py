"""B30 single-source-of-truth aggregator for the v14 joint SSL step.

**B30 lock 2026-05-28** ([[project_v14_anatomy_gated_symmetric_2026_05_28]]):
``latent_valid`` is the SAME ``(B, L) bool`` tensor across the encoder's
latent self-attn ``attn_mask``, the slot-axis loss primitives
(``L_mid_slot`` / ``L_post_frame``), and the utterance-level PMA-then-mean
zero-active guard. The caller computes ``latent_valid`` ONCE per batch
from ``(support.sum(dim=1) > 0)`` and threads the SAME tensor through
this function — that's the structural contract that gives the
``latent_valid`` rename its name.

This module's :func:`compute_v14_ssl_losses` orchestrates the 4-term
B28/B29 joint default::

    L_total = 1.0 · L_pre_frame      @ M2
            + 1.0 · L_mid_slot       @ LN_mid(M3)
            + 1.0 · L_post_frame     @ LN_frame(M4)
            + 1.0 · L_post_utterance @ LN_utt(M4) - PMA

threading ``latent_valid`` to the three slot/clip-axis terms (L_pre_frame
operates at M2, electrode-axis, pre-encoder — it uses its own
``m2_valid_mask`` covering freq-bin × time-patch validity, NOT
``latent_valid``).

The function is intentionally a pure orchestrator: it does NOT call the
encoder, the EMA teacher, or any LN module. The caller is responsible
for:
  - forwarding the student encoder with ``return_taps=True``,
  - forwarding the EMA teacher (full input, detached),
  - applying ``LN_mid`` / ``LN_frame`` / ``LN_utt`` (and the mirrored
    teacher LNs) before passing tensors here,
  - computing ``latent_valid`` from ``support`` + ``valid_mask`` via the
    encoder's ``_compute_latent_valid`` helper.

This keeps the aggregator unit-testable WITHOUT needing the SCAFFOLD-02
V14Data pipeline or the production Lightning ``training_step`` — both of
which can be layered on top by simply calling this function from their
``_run_step`` once available (tracked at
``docs/neuroprobe/v14_blockers.md`` row 1 / SCAFFOLD-02).
"""

from __future__ import annotations

import typing as tp

from torch import Tensor, nn

from speech_decoding.ssl.recon import recon_loss
from speech_decoding.ssl.slot_loss import masked_mse_slot_time
from speech_decoding.ssl.total_loss import V14TotalLossBreakdown, v14_total_loss
from speech_decoding.ssl.utterance_loss import pma_then_mean, utterance_mse_loss


_LossForm = tp.Literal["mse", "l1"]


def compute_v14_ssl_losses(
    *,
    student_m2_pred: Tensor,
    m2_target: Tensor,
    m2_valid_mask: tp.Optional[Tensor] = None,
    student_m3_lnmid: Tensor,
    teacher_m3_lnmid: Tensor,
    student_m4_lnframe: Tensor,
    teacher_m4_lnframe: Tensor,
    student_m4_lnutt: Tensor,
    teacher_m4_lnutt: Tensor,
    latent_valid: Tensor,
    pma_student: nn.Module,
    pma_teacher: nn.Module,
    loss_form: _LossForm = "l1",
) -> tp.Tuple[V14TotalLossBreakdown, Tensor]:
    """Compose the B28/B29 4-term v14 joint SSL loss with B30 plumbing.

    Parameters
    ----------
    student_m2_pred, m2_target
        Frame-level (M2) student prediction and reconstruction target.
        Shape is arbitrary (typically ``(B, T_p, F_p, d)`` or
        ``(B, C, T_p, F_p)`` depending on the recipe — the loss is
        elementwise after broadcasting the masks).
    m2_valid_mask
        Optional broadcastable mask for ``L_pre_frame``. Passed to
        ``recon_loss`` as both ``valid_mask_student`` and
        ``valid_mask_teacher`` (B30 keeps the per-side masking
        decision in the caller per EX09; for the joint default we pass
        the same mask to both sides). M2 is pre-encoder so the mask
        covers freq-bin / time-patch validity, NOT ``latent_valid``.
    student_m3_lnmid, teacher_m3_lnmid
        Slot-axis student / teacher tensors AFTER ``LN_mid``. Shape
        ``(B, L, T_p, d)``. Teacher must be detached upstream.
    student_m4_lnframe, teacher_m4_lnframe
        Slot-axis student / teacher tensors AFTER ``LN_frame``. Shape
        ``(B, L, T_p, d)``. Teacher detached upstream.
    student_m4_lnutt, teacher_m4_lnutt
        Slot-axis student / teacher tensors AFTER ``LN_utt`` (for the
        utterance head). Shape ``(B, L, T_p, d)``. Teacher detached
        upstream.
    latent_valid
        ``(B, L) bool`` — the B30 single source of truth, derived from
        ``(support.sum(dim=1) > 0)`` expanded across the M sub-slots.
        The SAME tensor that the encoder's latent-SA ``attn_mask``
        consumes.
    pma_student, pma_teacher
        ``V14ParcelCollapsePMA`` instances. ``pma_student`` participates
        in EMA mirroring (it's a module on the student model);
        ``pma_teacher`` is the EMA mirror (the caller builds it via the
        EMA helper, this function just consumes it).
    loss_form
        ``"l1"`` (B26 default) or ``"mse"`` (R-l2-loss sister). Applied
        to all 4 terms uniformly.

    Returns
    -------
    breakdown
        :class:`V14TotalLossBreakdown` carrying ``.total`` plus the raw
        per-term scalars for logging.
    clip_valid
        ``(B,) bool`` from ``pma_then_mean`` — flags which clips
        contributed to ``L_post_utterance``. Useful for logging
        active-clip ratios (SWEC / no-anatomy clips drop out via the
        zero-active guard).

    Notes
    -----
    The slot-axis terms (``L_mid_slot``, ``L_post_frame``) use the
    ``masked_mse_slot_time`` primitive whose divisor is
    ``Σ_b |active_slots(b)| · T_p · d`` (clamped to 1.0), so a batch of
    all-SWEC clips gives ``L_mid_slot = L_post_frame = 0`` cleanly.
    The utterance term uses ``utterance_mse_loss(..., clip_valid=...)``
    with divisor ``Σ_b active_clip(b) · d`` (clamped to 1.0), so no
    dilution from inactive rows.

    DKoleo (B28 demoted to sister) is intentionally NOT threaded here.
    The ``R-dkoleo-batch-cls-unit`` / ``R-dkoleo-intra-clip-slots``
    sisters should compute the DKoleo scalar externally (the unit choice
    differs across sisters and the embedding source — CLS-analog vs
    per-clip slots — is not available from the tensors this aggregator
    sees) and pass it directly to :func:`v14_total_loss` via its
    ``l_dkoleo_m4`` kwarg, alongside this function's output ``breakdown``
    fields. Do NOT add a DKoleo kwarg to this orchestrator.
    """
    # L_pre_frame @ M2 — electrode-axis, pre-encoder. Uses its own
    # ``m2_valid_mask`` (freq-bin × time-patch validity), NOT
    # ``latent_valid``.
    l_pre_frame = recon_loss(
        student_m2_pred,
        m2_target,
        valid_mask_student=m2_valid_mask,
        valid_mask_teacher=m2_valid_mask,
        loss_form=loss_form,
    )

    # L_mid_slot @ LN_mid(M3) — slot-axis, time-patch-axis. B30
    # ``latent_valid`` gates the slot-axis sum + divisor.
    l_mid_slot = masked_mse_slot_time(
        student_m3_lnmid,
        teacher_m3_lnmid,
        latent_valid=latent_valid,
        loss_form=loss_form,
    )

    # L_post_frame @ LN_frame(M4) — same slot-axis primitive, SAME
    # ``latent_valid`` (single source of truth).
    l_post_frame = masked_mse_slot_time(
        student_m4_lnframe,
        teacher_m4_lnframe,
        latent_valid=latent_valid,
        loss_form=loss_form,
    )

    # L_post_utterance @ LN_utt(M4)-PMA — clip-axis, derived from
    # ``latent_valid.any(dim=1)`` via ``pma_then_mean``'s second return.
    # SWEC clips (no active slots) get a structural zero (d,) that the
    # ``clip_valid`` mask excludes from the divisor → no dilution.
    student_clip, clip_valid = pma_then_mean(
        pma_student, student_m4_lnutt, latent_valid=latent_valid,
    )
    teacher_clip, _ = pma_then_mean(
        pma_teacher, teacher_m4_lnutt, latent_valid=latent_valid,
    )
    l_post_utterance = utterance_mse_loss(
        student_clip,
        teacher_clip,
        clip_valid=clip_valid,
        loss_form=loss_form,
    )

    breakdown = v14_total_loss(
        l_pre_frame=l_pre_frame,
        l_mid_slot=l_mid_slot,
        l_post_frame=l_post_frame,
        l_post_utterance=l_post_utterance,
    )
    return breakdown, clip_valid


__all__ = ["compute_v14_ssl_losses"]
