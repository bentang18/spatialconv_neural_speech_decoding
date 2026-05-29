"""B30 single-source-of-truth aggregator for the v14 joint SSL step.

**B30 lock 2026-05-28** ([[project_v14_anatomy_gated_symmetric_2026_05_28]]):
``latent_valid`` is the SAME ``(B, L) bool`` tensor across the encoder's
latent self-attn ``attn_mask``, the slot-axis loss primitives
(``L_mid_slot`` / ``L_post_frame``), and the utterance-level PMA-then-mean
zero-active guard. The caller computes ``latent_valid`` ONCE per batch
from ``(support.sum(dim=1) > 0)`` and threads the SAME tensor through
this function.

**B31 amendment 2026-05-28 PM** ([[project_v14_b31_vjepa2_canonical_loss_2026_05_28]]):
The joint SSL default collapses to a 2-term V-JEPA-2-canonical surface::

    L_total = 1.0 · L_pre_frame      @ M2
            + 1.0 · L_post_frame     @ LN_frame(M4)

Both pure L1 per V-JEPA 2 §2.1 Eq 1. ``L_mid_slot`` and
``L_post_utterance`` are reconstructed only under the falsifier sisters
selected via ``loss_variant``:

  - ``"b31_default"``     →  2 terms (frame@M2 + frame@M4)
  - ``"b31_plus_m3"``     →  +1.0 · L_mid_slot      @ LN_mid(M3)
  - ``"b31_plus_utt"``    →  +1.0 · L_post_utterance @ LN_utt(M4)-PMA
  - ``"b31_plus_both"``   →  +both

Sister-conditional kwargs (``student_m3_lnmid`` / ``teacher_m3_lnmid``
for the M3 arm; ``student_m4_lnutt`` / ``teacher_m4_lnutt`` /
``pma_student`` / ``pma_teacher`` for the utterance arm) are
``Optional`` and must be supplied iff the corresponding sister fires.
Passing them with ``loss_variant="b31_default"`` is a programming error
(caught at runtime).

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
LossVariant = tp.Literal[
    "b31_default", "b31_plus_m3", "b31_plus_utt", "b31_plus_both",
]
LOSS_VARIANTS: tuple[str, ...] = tp.get_args(LossVariant)


def variant_wants_m3(variant: LossVariant) -> bool:
    """True iff ``variant`` selects the B22-Arm-1 ``L_mid_slot`` term.

    Single source of truth for B31 sister-arm gating — imported by the
    SSL aggregator AND by :class:`V14JointBrainModule` so that loss-path
    selection and head-construction conditioning never drift.
    """
    return variant in ("b31_plus_m3", "b31_plus_both")


def variant_wants_utt(variant: LossVariant) -> bool:
    """True iff ``variant`` selects the B19 ``L_post_utterance`` term."""
    return variant in ("b31_plus_utt", "b31_plus_both")


_wants_m3 = variant_wants_m3
_wants_utt = variant_wants_utt


def compute_v14_ssl_losses(
    *,
    student_m2_pred: Tensor,
    m2_target: Tensor,
    m2_valid_mask: tp.Optional[Tensor] = None,
    student_m4_lnframe: Tensor,
    teacher_m4_lnframe: Tensor,
    latent_valid: Tensor,
    loss_variant: LossVariant = "b31_default",
    student_m3_lnmid: tp.Optional[Tensor] = None,
    teacher_m3_lnmid: tp.Optional[Tensor] = None,
    student_m4_lnutt: tp.Optional[Tensor] = None,
    teacher_m4_lnutt: tp.Optional[Tensor] = None,
    pma_student: tp.Optional[nn.Module] = None,
    pma_teacher: tp.Optional[nn.Module] = None,
    loss_form: _LossForm = "l1",
) -> V14TotalLossBreakdown:
    """Compose the B31 v14 joint SSL loss with B30 plumbing.

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
    student_m4_lnframe, teacher_m4_lnframe
        Slot-axis student / teacher tensors AFTER ``LN_frame``. Shape
        ``(B, L, T_p, d)``. Teacher detached upstream. **Required**
        for every variant — this is the half of the B31 default.
    latent_valid
        ``(B, L) bool`` — the B30 single source of truth, derived from
        ``(support.sum(dim=1) > 0)`` expanded across the M sub-slots.
        The SAME tensor that the encoder's latent-SA ``attn_mask``
        consumes.
    loss_variant
        One of ``"b31_default"`` (frame@M2 + frame@M4 only),
        ``"b31_plus_m3"`` (adds L_mid_slot), ``"b31_plus_utt"`` (adds
        L_post_utterance), ``"b31_plus_both"``. Selects which sister
        arms reconstruct the B28/B29 dropped terms.
    student_m3_lnmid, teacher_m3_lnmid
        Slot-axis student / teacher tensors AFTER ``LN_mid``. Shape
        ``(B, L, T_p, d)``. Teacher must be detached upstream.
        **Required iff** ``loss_variant`` selects the M3 arm.
    student_m4_lnutt, teacher_m4_lnutt
        Slot-axis student / teacher tensors AFTER ``LN_utt`` (for the
        utterance head). Shape ``(B, L, T_p, d)``. Teacher detached
        upstream. **Required iff** ``loss_variant`` selects the
        utterance arm.
    pma_student, pma_teacher
        ``V14ParcelCollapsePMA`` instances. ``pma_student`` participates
        in EMA mirroring (it's a module on the student model);
        ``pma_teacher`` is the EMA mirror (the caller builds it via the
        EMA helper, this function just consumes it). **Required iff**
        ``loss_variant`` selects the utterance arm.
    loss_form
        ``"l1"`` (B26 default) or ``"mse"`` (R-l2-loss sister). Applied
        to all active terms uniformly.

    Returns
    -------
    breakdown
        :class:`V14TotalLossBreakdown` carrying ``.total`` plus the raw
        per-term scalars for logging. Sister-only fields are ``None``
        when ``loss_variant`` did not select them.

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
    sisters should compute the DKoleo scalar externally and pass it
    directly to :func:`v14_total_loss` via its ``l_dkoleo_m4`` kwarg
    alongside this function's output ``breakdown`` fields.
    """
    # L_pre_frame @ M2 — electrode-axis, pre-encoder.
    l_pre_frame = recon_loss(
        student_m2_pred,
        m2_target,
        valid_mask_student=m2_valid_mask,
        valid_mask_teacher=m2_valid_mask,
        loss_form=loss_form,
    )

    # L_post_frame @ LN_frame(M4) — slot-axis, time-patch-axis. B30
    # ``latent_valid`` gates the slot-axis sum + divisor.
    l_post_frame = masked_mse_slot_time(
        student_m4_lnframe,
        teacher_m4_lnframe,
        latent_valid=latent_valid,
        loss_form=loss_form,
    )

    # ── B31 falsifier sister arm: L_mid_slot @ LN_mid(M3) ─────────────
    l_mid_slot: tp.Optional[Tensor] = None
    if _wants_m3(loss_variant):
        if student_m3_lnmid is None or teacher_m3_lnmid is None:
            raise ValueError(
                f"loss_variant={loss_variant!r} requires "
                "student_m3_lnmid + teacher_m3_lnmid; got None"
            )
        l_mid_slot = masked_mse_slot_time(
            student_m3_lnmid,
            teacher_m3_lnmid,
            latent_valid=latent_valid,
            loss_form=loss_form,
        )
    elif student_m3_lnmid is not None or teacher_m3_lnmid is not None:
        raise ValueError(
            f"loss_variant={loss_variant!r} does not select the M3 arm "
            "but student_m3_lnmid / teacher_m3_lnmid were provided"
        )

    # ── B31 falsifier sister arm: L_post_utterance @ LN_utt(M4)-PMA ───
    l_post_utterance: tp.Optional[Tensor] = None
    if _wants_utt(loss_variant):
        if (
            student_m4_lnutt is None
            or teacher_m4_lnutt is None
            or pma_student is None
            or pma_teacher is None
        ):
            raise ValueError(
                f"loss_variant={loss_variant!r} requires student_m4_lnutt "
                "+ teacher_m4_lnutt + pma_student + pma_teacher; got "
                "None for at least one of them"
            )
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
    elif (
        student_m4_lnutt is not None
        or teacher_m4_lnutt is not None
        or pma_student is not None
        or pma_teacher is not None
    ):
        raise ValueError(
            f"loss_variant={loss_variant!r} does not select the utterance "
            "arm but one or more of student_m4_lnutt / teacher_m4_lnutt / "
            "pma_student / pma_teacher were provided"
        )

    return v14_total_loss(
        l_pre_frame=l_pre_frame,
        l_post_frame=l_post_frame,
        l_mid_slot=l_mid_slot,
        l_post_utterance=l_post_utterance,
    )


__all__ = [
    "compute_v14_ssl_losses",
    "LossVariant",
    "LOSS_VARIANTS",
    "variant_wants_m3",
    "variant_wants_utt",
]
