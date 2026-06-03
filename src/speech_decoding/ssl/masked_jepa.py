"""B36 WS-B paradigm-B masked-JEPA loss terms (the real SSL objective).

Replaces the inert B31 2-term self-distill (`ssl/aggregator.py`, full input
both sides, no mask, no predictor) with two staged masked-prediction terms,
exactly one of which is active per phase (B9):

* **P1 — front-end M2 (paradigm A, no separate predictor).** The masked
  ``(electrode, freq-patch, time-patch)`` cells were zeroed UPSTREAM of the
  token blocks (`v14_encoder` B5), so a masked position's M2 is a pure
  function of the visible context — the token blocks ARE the predictor. Loss
  = L1 between the student's masked M2 tokens and the EMA teacher's
  full-input M2 (post-``frontend_ln``) at the same positions.

* **P2 — parcel M4 (paradigm B, separate predictor).** The visible-only
  student encoder produces M4 at visible parcel-time cells; the
  :class:`~speech_decoding.models.v14_encoder.JepaPredictor` predicts the
  masked parcel-time cells from those visible tokens (mask-token queries
  tagged by a fixed parcel/time sinusoid); target = EMA teacher full-input
  M4 (post-``encoder_ln``). Loss = L1 at masked positions only.

Both targets are ``detach()``ed (stop-grad on the EMA teacher, V-JEPA 2 §2.1)
and normalized only by the encoder's own terminal LayerNorm — NO separate
``ln_frame`` head (B6 / B36 §4 canonical V-JEPA target-norm).
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import torch
from torch import Tensor

from speech_decoding.models.v14_encoder import factored_sinusoidal_pos_emb

_LossForm = tp.Literal["l1", "mse"]


@dataclass(frozen=True)
class MaskedJepaBreakdown:
    """Single-term masked-JEPA loss (exactly one active term per phase, B9).

    ``total`` is the scalar the optimizer steps on. ``phase`` is ``"p1"`` or
    ``"p2"``; ``n_masked`` is the number of masked cells scored this step
    (0 → ``total`` is an exact 0, no NaN — B6 masked-empty contract).
    """

    total: Tensor
    phase: str
    n_masked: int


def _l1_or_zero(pred: Tensor, target: Tensor, loss_form: _LossForm) -> Tensor:
    """Per-element L1 (default) / MSE over the gathered masked cells; an exact
    0 scalar (graph-connected to ``pred``) when the masked set is empty."""
    if pred.shape != target.shape:
        raise ValueError(
            f"pred.shape {tuple(pred.shape)} != target.shape {tuple(target.shape)}"
        )
    if pred.numel() == 0:
        # Keep the result connected to the predictor graph so the optimizer
        # never sees a detached/zero-grad surprise; ``* 0`` is exact.
        return pred.sum() * 0.0
    if loss_form == "l1":
        return (pred - target).abs().mean()
    if loss_form == "mse":
        return (pred - target).pow(2).mean()
    raise ValueError(f"unknown loss_form={loss_form!r}")


def p1_frontend_m2_loss(
    *,
    student_m2: Tensor,   # (B, C, F_p, T_p, d) post-frontend_ln, masked cells zeroed upstream
    teacher_m2: Tensor,   # (B, C, F_p, T_p, d) EMA full-input, post-frontend_ln
    token_mask: Tensor,   # (B, C, F_p, T_p) bool, True = masked
    loss_form: _LossForm = "l1",
    freq_patch_valid: tp.Optional[Tensor] = None,  # (F_p,) or (B, F_p) bool
) -> MaskedJepaBreakdown:
    """P1 front-end masked JEPA (paradigm A — token blocks self-predict).

    B36 C5: ``freq_patch_valid`` (per-corpus freq-patch validity, True = valid)
    excludes invalid freq-patch cells (e.g. SWEC k22–29 → F-patches 7–9) from
    the L1 target — a masked cell on an invalid freq patch is never a
    reconstruction target. ``None`` (BT, all valid) → every masked cell scored,
    byte-identical to the pre-C5 loss. P2's M4 is parcel-pooled (no freq axis),
    so this exclusion is P1-only.
    """
    if token_mask.shape != student_m2.shape[:-1]:
        raise ValueError(
            f"token_mask {tuple(token_mask.shape)} must match M2 grid "
            f"{tuple(student_m2.shape[:-1])}"
        )
    if freq_patch_valid is not None:
        B, C, F_p, T_p = token_mask.shape
        fpv = freq_patch_valid.to(torch.bool)
        if fpv.shape not in {(F_p,), (B, F_p)}:
            raise ValueError(
                f"freq_patch_valid {tuple(freq_patch_valid.shape)} must be "
                f"(F_p,) or (B, F_p) = ({F_p},) / ({B}, {F_p})"
            )
        if fpv.dim() == 1:
            fpv = fpv.unsqueeze(0).expand(B, F_p)
        # Broadcast (B, F_p) over (C, T_p): a masked cell counts only on a valid
        # freq patch.
        token_mask = token_mask & fpv.view(B, 1, F_p, 1)
    pred = student_m2[token_mask]               # (n_masked, d)
    target = teacher_m2[token_mask].detach()    # (n_masked, d)
    loss = _l1_or_zero(pred, target, loss_form)
    return MaskedJepaBreakdown(total=loss, phase="p1", n_masked=int(token_mask.sum()))


def p2_parcel_m4_loss(
    *,
    predictor: torch.nn.Module,  # JepaPredictor
    student_m4: Tensor,   # (B, L, T_p, d) post-encoder_ln, visible-only encoder
    teacher_m4: Tensor,   # (B, L, T_p, d) EMA full-input, post-encoder_ln
    visible: Tensor,      # (B, L, T_p) bool — covered & ~masked
    target_mask: Tensor,  # (B, L, T_p) bool — covered & masked
    loss_form: _LossForm = "l1",
) -> MaskedJepaBreakdown:
    """P2 parcel masked JEPA (paradigm B — separate predictor).

    The predictor reads the VISIBLE parcel-time tokens as context and predicts
    the masked parcel-time cells; ``query_pos`` tags each masked slot by a
    fixed (slot-id, time) sinusoid. Target = EMA teacher full-input M4.
    """
    B, L, T_p, d = student_m4.shape
    if visible.shape != (B, L, T_p) or target_mask.shape != (B, L, T_p):
        raise ValueError(
            f"visible/target_mask must be (B, L, T_p)=({B},{L},{T_p}); got "
            f"{tuple(visible.shape)} / {tuple(target_mask.shape)}"
        )
    n_masked = int(target_mask.sum())
    N = L * T_p
    device = student_m4.device

    ctx = student_m4.reshape(B, N, d)               # visible used as keys (kpm below)
    visible_flat = visible.reshape(B, N)
    target_flat = target_mask.reshape(B, N)

    # (slot-id, time) position ids for every grid cell, flatten order (L outer,
    # T_p inner) matching the reshape above. Slot id == parcel*M + subslot, so
    # it is the parcel id at M=1 (the B36 default) — the §5 "parcel-id + time"
    # tag, generalized to M>1.
    l_ids = (
        torch.arange(L, device=device).view(L, 1).expand(L, T_p).reshape(N)
    )
    t_ids = (
        torch.arange(T_p, device=device).view(1, T_p).expand(L, T_p).reshape(N)
    )
    l_ids = l_ids.unsqueeze(0).expand(B, N)
    t_ids = t_ids.unsqueeze(0).expand(B, N)
    # The sinusoid is built in fp32; cast to the feature dtype so the
    # predictor's ``mask_token + query_pos`` stays single-dtype under
    # bf16/fp16 autocast (LayerNorm rejects mixed dtype on CPU).
    query_pos = factored_sinusoidal_pos_emb(
        [l_ids, t_ids], predictor.hidden,
    ).to(student_m4.dtype)

    pred = predictor(
        ctx,
        query_pos,
        query_valid=target_flat,
        context_key_padding_mask=~visible_flat,
    )                                                # (n_masked, d)
    target = teacher_m4.reshape(B, N, d)[target_flat].detach()  # (n_masked, d)
    loss = _l1_or_zero(pred, target, loss_form)
    return MaskedJepaBreakdown(total=loss, phase="p2", n_masked=n_masked)


__all__ = [
    "MaskedJepaBreakdown",
    "p1_frontend_m2_loss",
    "p2_parcel_m4_loss",
]
