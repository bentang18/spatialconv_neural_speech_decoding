"""LOSS-04 (B19 + B21 5-term loss lock 2026-05-25; B26 loss-form
amendment 2026-05-27 PM; B30 mask-discipline lock 2026-05-28;
**B31 V-JEPA-2-canonical 2-term lock 2026-05-28
([[project_v14_b31_vjepa2_canonical_loss_2026_05_28]])**):
utterance-level loss between student and teacher PMA-pooled M4
representations.

**B31 scope-shift**: this module survives because :func:`pma_then_mean`
is still needed at **Phase-3** for the Whisper-L8 cross-modal
distillation readout. It is NOT used in the **B31 default joint SSL
aggregator** (``loss_variant="b31_default"``) — the B31 default drops
``L_post_utterance`` to two pure-L1 terms (``L_pre_frame @ M2`` +
``L_post_frame @ M4``). The 5-term path documented below is reached only
via the ``R-add-utterance-loss`` (``loss_variant="b31_plus_utt"``) /
``R-add-both`` (``loss_variant="b31_plus_both"``) sisters, which exist
to empirically re-litigate B31's drop.

Path (P3 distillation, or B31 utterance-sister: recipe §5 +
``project_v14_collapse_prevention_lock_2026_05_25`` +
``project_v14_anatomy_gated_symmetric_2026_05_28``)::

    student_M4   --(LN_utt)--> PMA(latent_valid) → (B, T_p, d)
                                                 → mean(T_p) → (B, d)
    teacher_M4   --(LN_utt_T)-> PMA(latent_valid) → (B, T_p, d)
                                                 → mean(T_p) → (B, d)
    L_post_utt   = L1(student_d, teacher_d.detach())                # B26 default

B26 default is pure L1 (V-JEPA 2 §2.1 Eq 1 + V-JEPA 2.1 §2.3.1 Eqs 1+2);
MSE remains available as the ``R-l2-loss`` / ``R-mse-loss`` sister
falsifier via explicit ``loss_form="mse"``.

The PMA module is owned by the encoder so its query is checkpointable and
participates in EMA-teacher mirroring. Under the **B31 three-phase
training contract**, the PMA query receives gradient only at **P3**
(Whisper-L8 distillation); P1+P2 use no PMA in the default joint SSL
loss path, and P4 freezes PMA explicitly at the Phase-4 construction
site. The ``R-add-utterance-loss`` sister restores PMA training in
P1+P2 to falsify the B31 drop.

The caller passes the SAME PMA module instance for both student and
teacher forward passes; the EMA teacher's PMA mirror is built by the
EMA helper, not here. ``LN_utt`` / ``LN_utt_T`` are also applied by the
caller before passing tensors here — keeps the loss primitive a pure
function over already-normalized inputs.

**B30 zero-active guard** ([[project_v14_anatomy_gated_symmetric_2026_05_28]]):
``pma_then_mean`` returns a structural zero ``(B, d)`` of zeros for
clips whose ``latent_valid`` sums to zero (SWEC default; pathological
BT/D/AJILE12 clips with no anatomy-covered parcel). The training
aggregator is responsible for skipping ``L_post_utterance`` accumulation
for those clips so the gradient signal is exactly zero, NOT a softmax
over an all-False key set (which would be NaN). The parameter name is
``latent_valid`` to match the encoder + slot-loss single-source-of-truth
convention.
"""

from __future__ import annotations

import typing as tp

import torch
from torch import Tensor, nn


def utterance_mse_loss(
    pma_student_out: Tensor,
    pma_teacher_out: Tensor,
    *,
    clip_valid: tp.Optional[Tensor] = None,
    loss_form: tp.Literal["mse", "l1"] = "l1",
) -> Tensor:
    """Utterance-level loss between student and teacher PMA outputs.

    ``pma_student_out`` / ``pma_teacher_out`` shape ``(B, d)`` after the
    full path: PMA → mean(T_p). Caller is responsible for the LN_utt
    pre-norm, the PMA forward pass, the T_p mean reduction, and detaching
    the teacher branch.

    B26 default ``loss_form="l1"``; ``"mse"`` selected by the
    ``R-l2-loss`` / ``R-mse-loss`` sister falsifier.

    **B30 zero-active aggregator contract** ([[v14-anatomy-gated-symmetric-2026-05-28]]):
    pass ``clip_valid: (B,) bool`` derived from
    ``latent_valid.any(dim=1)`` to exclude no-anatomy clips from the
    reduction. The natural source is the second tuple element of
    ``pma_then_mean(...)``. When ``clip_valid`` is omitted, every clip
    contributes (caller-asserts no zero-active rows). When provided,
    inactive rows have a closed-form zero contribution to the numerator
    AND zero contribution to the divisor — dilution-free, gradient-zero.

    Name retained as ``utterance_mse_loss`` for grep / call-site
    stability; rename deferred to a future cleanup PR.
    """
    if pma_student_out.shape != pma_teacher_out.shape:
        raise ValueError(
            f"student/teacher shape mismatch: "
            f"{tuple(pma_student_out.shape)} vs {tuple(pma_teacher_out.shape)}"
        )
    if pma_student_out.ndim != 2:
        raise ValueError(
            f"expected (B, d); got {tuple(pma_student_out.shape)}"
        )
    if clip_valid is not None:
        B, d = pma_student_out.shape
        if clip_valid.shape != (B,):
            raise ValueError(
                f"clip_valid shape {tuple(clip_valid.shape)} does not match "
                f"(B,) = ({B},)"
            )
        if clip_valid.dtype != torch.bool:
            raise TypeError(
                f"clip_valid dtype must be torch.bool; got {clip_valid.dtype}"
            )
        if loss_form == "l1":
            per_elem = (pma_student_out - pma_teacher_out).abs()
        elif loss_form == "mse":
            per_elem = (pma_student_out - pma_teacher_out).pow(2)
        else:
            raise ValueError(f"unknown loss_form={loss_form!r}")
        mask = clip_valid.to(per_elem.dtype).unsqueeze(-1)        # (B, 1)
        numerator = (per_elem * mask).sum()
        denom = clip_valid.to(per_elem.dtype).sum() * d
        denom = denom.clamp(min=1.0)
        return numerator / denom
    if loss_form == "l1":
        return torch.nn.functional.l1_loss(pma_student_out, pma_teacher_out)
    if loss_form == "mse":
        return torch.nn.functional.mse_loss(pma_student_out, pma_teacher_out)
    raise ValueError(f"unknown loss_form={loss_form!r}")


def pma_then_mean(
    pma: nn.Module,
    m4: Tensor,
    *,
    latent_valid: Tensor,
) -> tp.Tuple[Tensor, Tensor]:
    """Helper: ``PMA(m4, latent_valid) → (B, T_p, d) → mean(T_p)``.

    Returns ``(out, clip_valid)`` where ``out`` is ``(B, d)`` and
    ``clip_valid`` is ``(B,) bool`` = ``latent_valid.any(dim=1)``. Pass
    ``clip_valid`` straight into ``utterance_mse_loss(..., clip_valid=...)``
    to honour the B30 zero-active aggregator contract without diluting
    the active-clip gradient.

    ``pma`` must accept the ``(latents, latent_valid)`` calling
    convention of ``V14ParcelCollapsePMA``. ``latent_valid`` is the
    per-subject anatomy-coverage slot mask ``(B, L)`` derived from
    ``(support.sum(over electrodes) > 0)`` per the B30 single-source-
    of-truth convention.

    **B30 zero-active guard** (aggregator-level per recipe §5): clips
    with ``latent_valid.sum() == 0`` (e.g. SWEC, or pathological
    no-anatomy BT/D/AJILE12 clips) would otherwise drive PMA's softmax
    over an all-False key set → NaN. We detect that case per-clip and
    substitute a structural zero ``(d,)``. The returned ``clip_valid``
    tells the caller which rows must be excluded from
    ``L_post_utterance``; passing it to ``utterance_mse_loss``'s
    ``clip_valid`` kwarg is the canonical call pattern.
    """
    if m4.ndim != 4:
        raise ValueError(
            f"expected M4 shape (B, L, T_p, d); got {tuple(m4.shape)}"
        )
    B, L_m4 = m4.shape[0], m4.shape[1]
    if latent_valid.shape != (B, L_m4):
        raise ValueError(
            f"latent_valid shape {tuple(latent_valid.shape)} does not match "
            f"(B, L) = ({B}, {L_m4})"
        )
    if latent_valid.dtype != torch.bool:
        raise TypeError(
            f"latent_valid dtype must be torch.bool; got {latent_valid.dtype}"
        )
    per_clip_active = latent_valid.any(dim=1)             # (B,) bool
    if per_clip_active.all():
        pooled = pma(m4, latent_valid=latent_valid)       # (B, T_p, d)
        if pooled.ndim != 3:
            raise ValueError(
                f"PMA output expected (B, T_p, d); got {tuple(pooled.shape)}"
            )
        return pooled.mean(dim=1), per_clip_active        # (B, d), (B,)
    # Mixed batch (some clips have anatomy, some don't): route only the
    # active clips through PMA and scatter zeros into the inactive rows.
    d = m4.shape[-1]
    out = torch.zeros((B, d), dtype=m4.dtype, device=m4.device)
    if per_clip_active.any():
        idx = per_clip_active.nonzero(as_tuple=False).squeeze(1)
        pooled = pma(m4[idx], latent_valid=latent_valid[idx])  # (B', T_p, d)
        if pooled.ndim != 3:
            raise ValueError(
                f"PMA output expected (B, T_p, d); got {tuple(pooled.shape)}"
            )
        out[idx] = pooled.mean(dim=1)
    return out, per_clip_active
