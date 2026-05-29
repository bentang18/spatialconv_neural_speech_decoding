"""Reconstruction loss with valid-bin mask honored on input AND target (T2.3).

Used by Phase 1 (Stage A: time-frequency mask, UFO frame target) and Phase 2
(Stage B: electrode-mask + utterance target) of the v14 SSL recipe.

Two design decisions are *deliberately deferred* and surfaced as constructor
parameters rather than baked in, because they are open blockers in
``docs/neuroprobe/v14_blockers.md``:

  * **IE04** (per-sample valid-bin mask reduction protocol) — caller chooses
    via ``reduction``. The default candidate (per-sample masked-mean over
    freq, then mean over (batch, time)) is implemented as ``"masked_mean"``;
    we also expose ``"sum_over_valid"`` (Σ diff² / Σ mask, used by some EAT
    variants).
  * **EX09** (EMA teacher must mask too) — caller must pass both
    ``valid_mask_student`` and ``valid_mask_teacher`` explicitly. We do not
    silently broadcast one to the other. If the loss should treat invalid
    positions identically on both sides, pass the same mask twice — that's a
    decision the caller makes, not the function.

The function is differentiable on the student side only — the teacher input
should already be detached (e.g. by ``stop_grad`` from :mod:`speech_decoding.ssl.ema`).
"""

from __future__ import annotations

import typing as tp

import torch
from torch import Tensor


_Reduction = tp.Literal["masked_mean", "sum_over_valid"]


def recon_loss(
    student: Tensor,
    teacher: Tensor,
    *,
    valid_mask_student: tp.Optional[Tensor] = None,
    valid_mask_teacher: tp.Optional[Tensor] = None,
    loss_form: tp.Literal["mse", "smooth_l1", "l1"] = "l1",
    smooth_l1_beta: float = 1.0,
    reduction: _Reduction = "masked_mean",
    eps: float = 1e-8,
) -> Tensor:
    """Masked reconstruction loss.

    ``student`` and ``teacher`` are arbitrary-shape tensors; the masks (if
    provided) must be broadcastable to that shape. The combined mask is the
    elementwise AND of the two (positions invalid on either side are excluded
    from both numerator and denominator).

    Returns a scalar tensor.

    Parameters
    ----------
    student, teacher
        Tensors of equal shape. ``teacher`` should be detached if it comes
        from an EMA teacher — this function does not detach for you (per
        EX09: teacher-side masking is the caller's responsibility, but
        ``stop_grad`` is also the caller's responsibility so the intent is
        explicit).
    valid_mask_student, valid_mask_teacher
        Bool (or 0/1 float) tensors, broadcastable to ``student.shape``.
        Positions where the mask is False/0 are excluded from the loss.
        Pass ``None`` to skip masking on that side. Per EX09 the typical
        use is to pass the SAME mask both sides — there is no per-sample
        side mismatch in the v14 recipe today.
    loss_form
        ``"l1"`` (B26 amendment 2026-05-27 PM default; V-JEPA 2 §2.1 Eq 1
        and V-JEPA 2.1 §2.3.1 Eqs 1+2 both use pure L1, NOT Smooth-L1; the
        B25 Smooth-L1 default is rolled back after a re-read of the V-JEPA
        2/2.1/data2vec-2.0 PDFs surfaced that the prior precedent chain
        was empirically wrong). ``"mse"`` is the R-l2-loss / R-mse-loss
        sister-cell falsifier and stays opt-in. ``"smooth_l1"`` is
        retained for the R-smoothl1-beta-{0.5, 1.0, 2.0} sister sweep
        (Smooth-L1 reduces to L1 as β→0; the bracket falsifies the
        L1-vs-Smooth-L1 transition at v14 scale). Phase-3 distillation
        in ``ssl/distill.py`` stays Smooth-L1 β=1.0 (cross-modal
        regression — out of B26 scope).
    smooth_l1_beta
        β parameter for SmoothL1 (only consulted when
        ``loss_form="smooth_l1"``). Defaults to ``1.0``.
    reduction
        How to aggregate the per-element loss over the mask.
        ``"masked_mean"`` divides by the sum of the mask (standard
        per-element mean of valid positions). ``"sum_over_valid"`` is the
        same numerator but used when the caller wants to weight examples
        by their valid-position count (rare).
    eps
        Floor for the mask-sum denominator. Prevents division by zero when
        every position in a batch happens to be masked out.
    """
    if student.shape != teacher.shape:
        raise ValueError(
            f"student.shape ({tuple(student.shape)}) "
            f"!= teacher.shape ({tuple(teacher.shape)})"
        )

    if loss_form == "l1":
        per_elem = (student - teacher).abs()
    elif loss_form == "mse":
        per_elem = (student - teacher).pow(2)
    elif loss_form == "smooth_l1":
        per_elem = torch.nn.functional.smooth_l1_loss(
            student, teacher, beta=smooth_l1_beta, reduction="none"
        )
    else:
        raise ValueError(f"unknown loss_form={loss_form!r}")

    mask = None
    if valid_mask_student is not None and valid_mask_teacher is not None:
        mask = valid_mask_student.to(per_elem.dtype) * valid_mask_teacher.to(per_elem.dtype)
    elif valid_mask_student is not None:
        mask = valid_mask_student.to(per_elem.dtype)
    elif valid_mask_teacher is not None:
        mask = valid_mask_teacher.to(per_elem.dtype)

    if mask is None:
        return per_elem.mean()

    mask = mask.expand_as(per_elem)
    if reduction == "masked_mean":
        num = (per_elem * mask).sum()
        den = mask.sum().clamp(min=eps)
        return num / den
    if reduction == "sum_over_valid":
        return (per_elem * mask).sum() / mask.sum().clamp(min=eps)
    raise ValueError(f"unknown reduction={reduction!r}")
