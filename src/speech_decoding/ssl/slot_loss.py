"""LOSS-02 / LOSS-03 (B19 + B21 + B22 5-term loss lock 2026-05-25;
B26 loss-form amendment 2026-05-27 PM; **B30 mask-discipline lock
2026-05-28**): slot-axis × time-patch × d masked loss used by both
``L_mid_slot @ LN_mid(M3)`` and ``L_post_frame @ LN_frame(M4)``.

Both losses share the same masked shape contract; what differs is the
upstream LayerNorm (``LN_mid`` for M3, ``LN_frame`` for M4) which is
applied by the training Experiment before calling these primitives.

B26 default loss form is **pure L1** (V-JEPA 2 §2.1 Eq 1 + V-JEPA 2.1
§2.3.1 Eqs 1+2). MSE is preserved as the ``R-l2-loss`` /
``R-mse-loss`` sister falsifier — set ``loss_form="mse"`` explicitly.

**B30 mask-discipline lock 2026-05-28** ([[project_v14_anatomy_gated_symmetric_2026_05_28]]):
the validity mask is **`latent_valid`** — derived from
``(support.sum(over electrodes) > 0)`` — the SAME rule across all
subjects. For BT/D/AJILE12 (anatomy known), `latent_valid[X, K]` is
True iff subject X has at least one electrode covering DK parcel K.
For SWEC (zero anatomy), `latent_valid` degenerates naturally to
all-False, contributing zero gradient to slot-axis losses (SWEC trains
only L_pre_frame_masked at M2).

This rule replaces the pre-B30 three-mask floor (``support`` /
``latent_valid`` / ``parcels_supervised``) with a single source of
truth. The encoder also consumes ``latent_valid`` to construct the
bidirectional latent-SA `attn_mask`, so the same tensor controls both
SA participation AND loss gating — single name, single source.

Sisters that override the default mask (passed by the training
aggregator via a `latent_valid_override` dispatch field, NOT by passing
a different mask to this primitive directly): `R-item-12-all-true`
passes `torch.ones_like(latent_valid)` (reinstates B29 Item 12);
`R-parcels-supervised-gating` reconstructs the pre-B30
`parcels_supervised`-derived mask (semantically matches default for
BT-with-anatomy; differs only for SWEC where the old extractor expanded
empty-set to all-True).
"""

from __future__ import annotations

import typing as tp

import torch
from torch import Tensor


def masked_mse_slot_time(
    student: Tensor,
    teacher: Tensor,
    *,
    latent_valid: Tensor,
    loss_form: tp.Literal["mse", "l1"] = "l1",
) -> Tensor:
    """Masked per-element loss over ``(slot ∈ active, time-patch, d)``.

    Used by both ``L_mid_slot`` (student/teacher are M3 streams post-
    ``LN_mid``) and ``L_post_frame`` (student/teacher are M4 streams post-
    ``LN_frame``).

    Inputs::

        student      : (B, L, T_p, d)  — student forward (gradient receivers)
        teacher      : (B, L, T_p, d)  — EMA teacher target (must be detached
                                         by caller; ``no_grad`` is NOT applied
                                         here so the caller can validate
                                         gradient gating with autograd checks)
        latent_valid : (B, L) bool     — per-subject anatomy-coverage mask
                                         from ``(support.sum(dim=1) > 0)``;
                                         single source of truth shared with
                                         the encoder's latent-SA `attn_mask`.
                                         SWEC degenerates to all-False; loss
                                         returns 0 for SWEC clips (denom
                                         clamp at 1.0 prevents NaN).

    Loss (default ``loss_form="l1"``)::

        L = ( Σ_{b, l ∈ active, t, d} |s - t| )
            / ( Σ_b |active_slots(b)| · T_p · d )

    With ``loss_form="mse"`` (``R-l2-loss`` sister), the per-element term
    is ``(s - t)²``. Divisor is unchanged.

    Following LOSS-02 / LOSS-03 "divisor = |latent_valid| × T_p × d".

    The function name is retained as ``masked_mse_slot_time`` for grep /
    call-site stability; the B26 default is L1 (rename deferred to a
    future cleanup PR).
    """
    if student.shape != teacher.shape:
        raise ValueError(
            f"student and teacher shape mismatch: {tuple(student.shape)} vs "
            f"{tuple(teacher.shape)}"
        )
    if student.ndim != 4:
        raise ValueError(
            f"expected (B, L, T_p, d); got {tuple(student.shape)}"
        )
    B, L, T_p, d = student.shape
    if latent_valid.shape != (B, L):
        raise ValueError(
            f"latent_valid shape {tuple(latent_valid.shape)} does not match "
            f"(B, L) = ({B}, {L})"
        )
    if latent_valid.dtype != torch.bool:
        raise TypeError(f"latent_valid dtype must be torch.bool; got {latent_valid.dtype}")

    if loss_form == "l1":
        per_elem = (student - teacher).abs()
    elif loss_form == "mse":
        per_elem = (student - teacher).pow(2)
    else:
        raise ValueError(f"unknown loss_form={loss_form!r}")

    mask = latent_valid.to(per_elem.dtype).unsqueeze(-1).unsqueeze(-1)         # (B, L, 1, 1)
    numerator = (per_elem * mask).sum()
    # Divisor counts ALL (slot, time-patch, d) cells whose slot is active for
    # this subject, summed over the batch. clamp(min=1) avoids divide-by-zero
    # in the degenerate all-empty edge case (e.g. an entire batch of SWEC
    # clips under B30 default → latent_valid all-False → loss = 0 with denom
    # = 1.0 placeholder).
    denom = latent_valid.to(per_elem.dtype).sum() * T_p * d
    denom = denom.clamp(min=1.0)
    return numerator / denom
