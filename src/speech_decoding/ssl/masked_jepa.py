"""B36 WS-B paradigm-B masked-JEPA loss terms (the real SSL objective).

Replaces the inert B31 2-term self-distill (`ssl/aggregator.py`, full input
both sides, no mask, no predictor) with two staged masked-prediction terms,
exactly one of which is active per phase (B9). **Both phases are paradigm B**
— a visible-only student, a separate student-only
:class:`~speech_decoding.models.v14_encoder.JepaPredictor`, an EMA full-input
teacher target, and L1 at masked positions only. EXACT PARITY; only the
predictor's attention SCOPE differs, because the mask geometry differs (6/02
masking rederivation M2/M4 contract — ``reports/b36_masking_rederivation_2026_06_02.md``;
the predictor + stop-grad is the canonical anti-collapse mechanism, B36 §5 /
Tian 2021). See ``memory/project_v14_p1_predictor_paradigm_b_regression_2026_06_04.md``
for why P1 was (wrongly) paradigm-A before:

* **P1 — front-end M2 (predictor scope UNCONSTRAINED).** The visible-only
  student front-end produces M2 (masked ``(electrode, freq-patch, time-patch)``
  cells zeroed UPSTREAM of the token blocks, `v14_encoder` B5). The predictor
  predicts the masked cells from the VISIBLE cells of the SAME electrode
  (per-electrode — no cross-electrode path), tagged by a fixed
  (freq-patch, time-patch) sinusoid; target = EMA teacher full-input M2
  (post-``frontend_ln``). Scope is unconstrained because the structured
  whole-row/column band mask is its own shortcut guard.

* **P2 — parcel M4 (predictor scope cross-time).** The visible-only student
  encoder produces M4 at visible parcel-time cells; the predictor predicts the
  masked parcel-time cells from those visible tokens (mask-token queries tagged
  by a fixed parcel/time sinusoid); target = EMA teacher full-input M4
  (post-``encoder_ln``). The tube mask ↔ cross-time scope coupling is
  shortcut-free (``validate_m4_coupling``).

Both targets are ``detach()``ed (stop-grad on the EMA teacher, V-JEPA 2 §2.1)
and normalized only by the encoder's own terminal LayerNorm — NO separate
``ln_frame`` head (B6 / B36 §4 canonical V-JEPA target-norm).
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import torch
from torch import Tensor

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


# Heteroscedastic (precision-weighted) M4 loss — project_v14_heteroscedastic_ssl_loss
# floor on σ²+ε so a near-zero-variance parcel cell can't blow the weight up. With
# the production session-robust-z'd |STFT| the pooled σ lands ~O(1), so ε is purely a
# div-by-zero guard, not a scale knob (the mean-1 normalization sets the scale).
_M4_PRECISION_EPS = 1e-6


def _weighted_l1_or_zero(
    pred: Tensor,            # (n_target, d)
    target: Tensor,          # (n_target, d)
    weight: Tensor,          # (n_target,) detached, mean-1 over the scored cells
    loss_form: _LossForm,
) -> Tensor:
    """Precision-weighted per-cell L1/MSE: weight each masked cell's d-dim error
    by ``weight`` (mean-1 → preserves the loss scale of the unweighted form), then
    mean over cells. Empty masked set → exact graph-connected 0 (B6 contract)."""
    if pred.shape != target.shape:
        raise ValueError(
            f"pred.shape {tuple(pred.shape)} != target.shape {tuple(target.shape)}"
        )
    if pred.numel() == 0:
        return pred.sum() * 0.0
    if weight.shape != (pred.shape[0],):
        raise ValueError(
            f"weight shape {tuple(weight.shape)} != (n_target,)=({pred.shape[0]},)"
        )
    if loss_form == "l1":
        per_cell = (pred - target).abs().mean(dim=-1)        # (n_target,)
    elif loss_form == "mse":
        per_cell = (pred - target).pow(2).mean(dim=-1)       # (n_target,)
    else:
        raise ValueError(f"unknown loss_form={loss_form!r}")
    return (per_cell * weight).mean()


def _m4_precision_weight(
    precision_std: Tensor,   # (B, K, F_p, T_p) per-cell pooled σ (raw |STFT| std)
    precision_n: Tensor,     # (B, K) electrode count per parcel
    target_cell: Tensor,     # (B, N) bool, N = K·F_p·T_p, flat (K outer, F_p, T_p inner)
    alpha: float,
    eps: float,
) -> Tensor:
    """Inverse-variance precision weight ``w = n_k^α / (σ²+ε)`` gathered at the
    masked cells and normalized to mean 1 (DETACHED — σ/n are data-derived, no
    grad path, and detaching guarantees the model can't inflate σ to dodge its
    own hard cells). Mean-1 → uniform-weight limit recovers plain L1."""
    B, K, F_p, T_p = precision_std.shape
    N = K * F_p * T_p
    n_pow = precision_n.clamp(min=1.0).pow(alpha)            # (B, K)
    n_cell = n_pow.unsqueeze(-1).unsqueeze(-1).expand(B, K, F_p, T_p).reshape(B, N)
    sigma2 = precision_std.pow(2).reshape(B, N)              # (B, N)
    w_full = n_cell / (sigma2 + eps)                         # (B, N)
    w = w_full[target_cell].detach()                        # (n_target,)
    if w.numel() == 0:
        return w
    return w / w.mean().clamp(min=eps)


def p1_frontend_m2_loss(
    *,
    predictor: torch.nn.Module,  # JepaPredictor (P1, n_identity = F_p)
    student_m2: Tensor,   # (B, C, F_p, T_p, d) post-frontend_ln, visible-only student
    teacher_m2: Tensor,   # (B, C, F_p, T_p, d) EMA full-input, post-frontend_ln
    token_mask: Tensor,   # (B, C, F_p, T_p) bool, True = masked
    loss_form: _LossForm = "l1",
    freq_patch_valid: tp.Optional[Tensor] = None,  # (F_p,) or (B, F_p) bool
    valid_mask: tp.Optional[Tensor] = None,  # (B, C) bool, True = real electrode
) -> MaskedJepaBreakdown:
    """P1 front-end masked JEPA (paradigm B — separate predictor, UNCONSTRAINED
    scope; exact parity with :func:`p2_parcel_m4_loss`).

    #91 ``valid_mask`` (B, C): True = real electrode. When provided, PAD
    electrodes are dropped from the per-electrode predictor batch entirely — no
    visible context, no target, no contribution to the ``mean`` L1. ``None``
    (the pre-#91 default) reconstructs every ``B·C`` row, INCLUDING pad
    electrodes, whose zero-input M2 dilutes the mean and wastes ~half the
    predictor batch at BT-Lite c_max=256. Pad exclusion is the correct semantics
    (a pad electrode carries no data) AND is REQUIRED for parity with the
    encoder's ragged front-end, which zeroes pad rows: the joint module passes
    ``valid_mask`` here exactly when ``encoder.ragged_frontend`` is on. Because
    the predictor is per-electrode (each row is one electrode), every VALID
    electrode's loss term is bit-identical with or without ``valid_mask``; only
    the pad terms (and hence the mean's denominator) change.

    Per-electrode (electrodes batched into the leading dim — the front-end has
    no cross-electrode path), the predictor reads the VISIBLE ``(freq-patch,
    time-patch)`` M2 cells of an electrode as context and predicts that
    electrode's MASKED cells. Each masked query slot = a learnable mask token +
    a learned freq-patch ``id_embed`` (the unordered freq identity), with the
    time-patch carried by RoPE inside the blocks. Target = EMA teacher
    full-input M2 at the masked cells (detached). Scope is "unconstrained"
    (no cross-time restriction) because the structured whole-row/column band
    mask is its own shortcut guard — vs P2's cross-time scope coupled to the
    tube mask ([[project_v14_predictor_design_rope_lock_2026_06_04]]).

    B36 C5: ``freq_patch_valid`` (per-corpus freq-patch validity, True = valid)
    excludes invalid freq-patch cells (e.g. SWEC k22–29 → F-patches 7–9) from
    BOTH the visible context keys and the L1 target — an invalid freq patch is
    neither attended nor reconstructed. ``None`` (BT, all valid) → every cell
    participates. P2's M4 is parcel-pooled (no freq axis), so this is P1-only.
    """
    if token_mask.shape != student_m2.shape[:-1]:
        raise ValueError(
            f"token_mask {tuple(token_mask.shape)} must match M2 grid "
            f"{tuple(student_m2.shape[:-1])}"
        )
    B, C, F_p, T_p, d = student_m2.shape
    device = student_m2.device
    BC = B * C
    N = F_p * T_p

    mask_flat = token_mask.reshape(BC, N)          # True = masked target
    visible_flat = ~mask_flat                      # True = visible context
    target_flat = mask_flat
    if freq_patch_valid is not None:
        fpv = freq_patch_valid.to(torch.bool)
        if fpv.shape not in {(F_p,), (B, F_p)}:
            raise ValueError(
                f"freq_patch_valid {tuple(freq_patch_valid.shape)} must be "
                f"(F_p,) or (B, F_p) = ({F_p},) / ({B}, {F_p})"
            )
        if fpv.dim() == 1:
            fpv = fpv.unsqueeze(0).expand(B, F_p)
        # (B, F_p) → per-cell (BC, N) in the (F_p outer, T_p inner) flat order.
        fpv_cell = (
            fpv.view(B, 1, F_p, 1)
            .expand(B, C, F_p, T_p)
            .reshape(BC, N)
        )
        visible_flat = visible_flat & fpv_cell
        target_flat = target_flat & fpv_cell

    ctx = student_m2.reshape(BC, N, d)             # visible used as keys (kpm below)
    teacher_ctx = teacher_m2.reshape(BC, N, d)
    # #91: drop PAD electrodes from the per-electrode predictor batch. Each row
    # is one electrode (the front-end has no cross-electrode path), so removing
    # pad rows leaves every valid row's prediction + target bit-identical; only
    # the dropped pad terms (and the mean's denominator) change. None → every
    # B·C row participates (pre-#91 behavior, incl. pad).
    if valid_mask is not None:
        if valid_mask.shape != (B, C):
            raise ValueError(
                f"valid_mask {tuple(valid_mask.shape)} must be (B, C) = "
                f"({B}, {C})"
            )
        keep = valid_mask.reshape(BC).to(torch.bool)       # (BC,) True = real
        ctx = ctx[keep]                                    # (N_valid, N, d)
        teacher_ctx = teacher_ctx[keep]
        visible_flat = visible_flat[keep]                  # (N_valid, N)
        target_flat = target_flat[keep]
    # (freq-patch, time-patch) ids for the (F_p outer, T_p inner) flat order:
    # flat index i = f_p · T_p + t_p. Shared across the batch (grid-identical;
    # one entry per cell, NOT per row — so row dropping above leaves them intact).
    f_ids = torch.arange(N, device=device) // T_p  # freq-patch id ∈ [0, F_p)
    t_ids = torch.arange(N, device=device) % T_p   # time-patch id ∈ [0, T_p)

    pred = predictor(
        ctx,
        context_time_ids=t_ids,
        query_time_ids=t_ids,
        query_id=f_ids,
        context_key_padding_mask=~visible_flat,
        query_valid=target_flat,
    )                                              # (n_masked, d)
    target = teacher_ctx[target_flat].detach()     # (n_masked, d)
    loss = _l1_or_zero(pred, target, loss_form)
    return MaskedJepaBreakdown(total=loss, phase="p1", n_masked=int(target_flat.sum()))


def p2_parcel_m4_loss(
    *,
    predictor: torch.nn.Module,  # JepaPredictor (P2, n_identity = L)
    student_m4: Tensor,   # (B, L, T_p, d) post-encoder_ln, visible-only encoder
    teacher_m4: Tensor,   # (B, L, T_p, d) EMA full-input, post-encoder_ln
    visible: Tensor,      # (B, L, T_p) bool — covered & ~masked
    target_mask: Tensor,  # (B, L, T_p) bool — covered & masked
    loss_form: _LossForm = "l1",
) -> MaskedJepaBreakdown:
    """P2 parcel masked JEPA (paradigm B — separate predictor, CROSS-TIME scope).

    The predictor reads the VISIBLE parcel-time tokens as context and predicts
    the masked parcel-time cells; each masked query slot = a learnable mask
    token + a learned parcel-slot ``id_embed`` (the unordered parcel identity),
    with the time-patch carried by RoPE inside the blocks. Target = EMA teacher
    full-input M4 (detached). Cross-time scope is coupled to the tube mask
    (``validate_m4_coupling``).
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

    # (slot-id, time) ids for the (L outer, T_p inner) flat order: flat index
    # i = l · T_p + t_p. Slot id == parcel*M + subslot, so it is the parcel id
    # at M=1 (the B36 default), generalized to M>1. Shared across the batch.
    l_ids = (
        torch.arange(L, device=device).view(L, 1).expand(L, T_p).reshape(N)
    )
    t_ids = (
        torch.arange(T_p, device=device).view(1, T_p).expand(L, T_p).reshape(N)
    )

    pred = predictor(
        ctx,
        context_time_ids=t_ids,
        query_time_ids=t_ids,
        query_id=l_ids,
        context_key_padding_mask=~visible_flat,
        query_valid=target_flat,
    )                                                # (n_masked, d)
    target = teacher_m4.reshape(B, N, d)[target_flat].detach()  # (n_masked, d)
    loss = _l1_or_zero(pred, target, loss_form)
    return MaskedJepaBreakdown(total=loss, phase="p2", n_masked=n_masked)


def b37_m4_freq_loss(
    *,
    predictor: torch.nn.Module,  # JepaPredictor (n_identity = K·M parcel, n_identity_2 = F_p freq)
    student_m4: Tensor,   # (B, K, F_p, T_p, d) post-encoder_ln, visible-only encoder
    teacher_m4: Tensor,   # (B, K, F_p, T_p, d) EMA full-input, post-encoder_ln
    visible: Tensor,      # (B, K, T_p) bool — surviving (non-tubed) covered parcel-times
    target_mask: Tensor,  # (B, K, T_p) bool — tube-masked covered parcel-times
    loss_form: _LossForm = "l1",
    # Heteroscedastic / inverse-variance precision weighting (OPT-IN; both None →
    # byte-identical to the plain-L1 path). project_v14_heteroscedastic_ssl_loss.
    precision_std: tp.Optional[Tensor] = None,  # (B, K, F_p, T_p) pooled σ (raw |STFT| std)
    precision_n: tp.Optional[Tensor] = None,    # (B, K) electrode count per parcel
    precision_alpha: float = 1.0,
) -> MaskedJepaBreakdown:
    """B37 M4 masked JEPA — the freq-PRESERVING parcel reconstruction (D3/D9).

    Unlike :func:`p2_parcel_m4_loss` (whose M4 latent is parcel-pooled with NO
    freq axis), the B37 mean-pool latent is ``parcel × freq × time``, so a masked
    M4 query token is identified by THREE axes: parcel (learned ``query_id``, the
    unordered parcel identity), freq-patch (sinusoidal ``query_id_2``, the ordered
    freq identity), and time-patch (RoPE inside the blocks). The predictor reads
    the full ``(F_p, T_p)`` field of every VISIBLE (surviving, band-masked) parcel
    as context and reconstructs the full ``(F_p, T_p, d)`` field of every TUBED
    parcel. Target = EMA teacher full-input M4 at the tubed cells (detached).

    Tube is whole-parcel-all-time, so ``visible``/``target_mask`` are constant
    over the time axis within a parcel (a parcel is either surviving → context,
    or tubed → target — mutually exclusive); they are accepted at parcel-time
    granularity ``(B, K, T_p)`` for parity with the encoder's ``latent_valid``
    bookkeeping and broadcast over the freq axis here.

    Cost note (NOT silently capped): the context is every surviving parcel's
    full ``F_p·T_p`` field, so ``n_ctx ≈ K_visible·F_p·T_p`` — heavy at
    production scale (the ``R-pred-2d-rope-freq`` / #112-class ragged-gather
    optimizations are deferred). The predictor is discarded after SSL, so this
    only costs pretraining wall-clock, never inference.

    ``freq_patch_valid`` (per-corpus freq-patch validity, P1's B36-C5) is NOT
    threaded here — inert for BT/capstone (all freq patches valid) and SWEC is
    still ``NotImplementedError``; a follow-up if a partial-freq corpus lands.
    """
    if student_m4.dim() != 5:
        raise ValueError(
            f"student_m4 must be (B, K, F_p, T_p, d); got {tuple(student_m4.shape)}"
        )
    if teacher_m4.shape != student_m4.shape:
        raise ValueError(
            f"teacher_m4 {tuple(teacher_m4.shape)} != student_m4 "
            f"{tuple(student_m4.shape)}"
        )
    B, K, F_p, T_p, d = student_m4.shape
    if visible.shape != (B, K, T_p) or target_mask.shape != (B, K, T_p):
        raise ValueError(
            f"visible/target_mask must be (B, K, T_p)=({B},{K},{T_p}); got "
            f"{tuple(visible.shape)} / {tuple(target_mask.shape)}"
        )
    device = student_m4.device
    N = K * F_p * T_p

    ctx = student_m4.reshape(B, N, d)               # visible used as keys (kpm below)
    # parcel-time visibility broadcast over the freq axis → per-cell (B, N) in the
    # (K outer, F_p mid, T_p inner) flat order: flat index i = k·(F_p·T_p) + f·T_p + t.
    visible_cell = (
        visible.unsqueeze(2).expand(B, K, F_p, T_p).reshape(B, N)
    )
    target_cell = (
        target_mask.unsqueeze(2).expand(B, K, F_p, T_p).reshape(B, N)
    )

    # (parcel, freq, time) ids for the flat order. Shared across the batch
    # (grid-identical; one entry per cell). l_ids == parcel*M + subslot → the
    # parcel id at M=1 (the B37 default), generalized to M>1.
    base = torch.arange(K * F_p * T_p, device=device)
    l_ids = base // (F_p * T_p)              # parcel id ∈ [0, K)  → query_id (learned)
    f_ids = (base // T_p) % F_p              # freq-patch id ∈ [0, F_p) → query_id_2 (sincos)
    t_ids = base % T_p                       # time-patch id ∈ [0, T_p) → RoPE

    pred = predictor(
        ctx,
        context_time_ids=t_ids,
        query_time_ids=t_ids,
        query_id=l_ids,
        query_id_2=f_ids,
        context_key_padding_mask=~visible_cell,
        query_valid=target_cell,
    )                                                # (n_target, d)
    target = teacher_m4.reshape(B, N, d)[target_cell].detach()  # (n_target, d)
    if precision_std is None and precision_n is None:
        loss = _l1_or_zero(pred, target, loss_form)
    else:
        if precision_std is None or precision_n is None:
            raise ValueError(
                "precision weighting needs BOTH precision_std and precision_n; got "
                f"std={precision_std is not None}, n={precision_n is not None}"
            )
        if precision_std.shape != (B, K, F_p, T_p):
            raise ValueError(
                f"precision_std shape {tuple(precision_std.shape)} != "
                f"(B, K, F_p, T_p)=({B},{K},{F_p},{T_p})"
            )
        if precision_n.shape != (B, K):
            raise ValueError(
                f"precision_n shape {tuple(precision_n.shape)} != (B, K)=({B},{K})"
            )
        weight = _m4_precision_weight(
            precision_std=precision_std,
            precision_n=precision_n,
            target_cell=target_cell,
            alpha=precision_alpha,
            eps=_M4_PRECISION_EPS,
        )
        loss = _weighted_l1_or_zero(pred, target, weight, loss_form)
    return MaskedJepaBreakdown(
        total=loss, phase="m4_freq", n_masked=int(target_cell.sum())
    )


__all__ = [
    "MaskedJepaBreakdown",
    "p1_frontend_m2_loss",
    "p2_parcel_m4_loss",
    "b37_m4_freq_loss",
]
