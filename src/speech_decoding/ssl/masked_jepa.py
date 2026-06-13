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
# ``ε`` is ONLY a div-by-zero / NaN guard — NOT the variance floor. The real floor
# is an empirical-Bayes SHRINKAGE prior ``σ²₀`` (an in-batch percentile of the
# scored σ²; see ``_m4_precision_weight``). Measured on a real nano batch
# (2026-06-13): ~12.5% of scored cells are degenerate (σ²≈1e-8 — single-electrode
# parcels and near-equal-electrode cells), seven orders below the typical σ²≈0.5.
# With the old bare ``σ²+ε`` those 12.5% soaked up the loss mass and mean-1
# normalization drove the informative 87.5% of cells to ~3e-5 weight (max/median
# ≈1.07e5 — the loss trained on flat, information-free cells). A max-cap alone does
# NOT fix this (the mass is bulk, not a few outliers); the additive shrinkage floor
# does (max/median → ~4.7). σ²₀ is an in-batch statistic so it auto-recalibrates if
# the front-end scale changes (e.g. the 2-STFT redesign). The cap is cheap insurance
# on top. project_v14_heteroscedastic_ssl_loss.
_M4_PRECISION_EPS = 1e-6
# Default shrinkage percentile (p25 of in-batch scored σ²) and post-norm weight cap.
_M4_PRECISION_FLOOR_PCT = 25.0
_M4_PRECISION_CAP = 10.0


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


def _precision_weight_core(
    sigma2: Tensor,      # (B, N) per-cell σ² in flat order
    n_cell: Tensor,      # (B, N) per-cell n_k^α (electrode-count weight, broadcast)
    target_cell: Tensor, # (B, N) bool, True = scored (masked target) cell
    eps: float,
    floor_pct: float,    # percentile (0-100) of in-batch scored σ² → shrinkage prior σ²₀
    cap: float,          # max weight after mean-1 norm (<=0 disables the cap)
) -> Tensor:
    """Shrinkage-regularized inverse-variance precision weight
    ``w = n_k^α / (σ² + σ²₀)`` gathered at the masked cells and normalized to
    mean 1 (DETACHED — σ/n are data-derived, no grad path, and detaching
    guarantees the model can't inflate σ to dodge its own hard cells). Layout-
    agnostic: callers flatten their (parcel × …) grid to ``(B, N)`` cells.

    ``σ²₀`` is an empirical-Bayes shrinkage prior: the ``floor_pct`` percentile of
    the *scored* σ² in this batch (NOT the bare ``ε``). It pulls degenerate cells
    (σ²→0 at single-electrode / near-equal-electrode parcels — which are the
    SHAKIEST, lowest-coverage targets, yet under bare ``1/(σ²+ε)`` get the HIGHEST
    weight and swamp the informative cells) back toward a typical-cell weight. See
    the module-level note for the measured pathology. Being an in-batch statistic,
    σ²₀ auto-recalibrates if the front-end magnitude scale changes. ``cap`` is a
    cheap post-normalization belt-and-suspenders. Mean-1 ⇒ uniform-weight limit
    recovers plain L1."""
    s2_scored = sigma2[target_cell].detach()                # (n_target,)
    if s2_scored.numel() == 0:
        return s2_scored
    # Empirical-Bayes shrinkage prior σ²₀ = p{floor_pct} of in-batch scored σ²,
    # floored at ε so an all-degenerate batch can't divide by ~0.
    q = max(0.0, min(1.0, floor_pct / 100.0))
    s2_floor = torch.quantile(s2_scored, q).clamp(min=eps)
    w_full = n_cell / (sigma2 + s2_floor)                    # (B, N)
    w = w_full[target_cell].detach()                        # (n_target,)
    w = w / w.mean().clamp(min=eps)                         # mean-1
    if cap > 0.0:
        w = w.clamp(max=cap)
        w = w / w.mean().clamp(min=eps)                    # re-normalize after cap
    return w


def _m4_precision_weight(
    precision_std: Tensor,   # (B, K, F_p, T_p) per-cell pooled σ (raw |STFT| std)
    precision_n: Tensor,     # (B, K) electrode count per parcel
    target_cell: Tensor,     # (B, N) bool, N = K·F_p·T_p, flat (K outer, F_p, T_p inner)
    alpha: float,
    eps: float,
    floor_pct: float,        # percentile (0-100) of in-batch scored σ² → shrinkage prior σ²₀
    cap: float,              # max weight after mean-1 norm (<=0 disables the cap)
) -> Tensor:
    """Inverse-variance precision weight for the B37 ``(F_p, T_p)`` M4 grid —
    builds the flat ``(B, N)`` σ²/n cells (K outer, F_p, T_p inner) and delegates
    to :func:`_precision_weight_core`."""
    B, K, F_p, T_p = precision_std.shape
    N = K * F_p * T_p
    n_pow = precision_n.clamp(min=1.0).pow(alpha)            # (B, K)
    n_cell = n_pow.unsqueeze(-1).unsqueeze(-1).expand(B, K, F_p, T_p).reshape(B, N)
    sigma2 = precision_std.pow(2).reshape(B, N)              # (B, N)
    return _precision_weight_core(sigma2, n_cell, target_cell, eps, floor_pct, cap)


def _m4_precision_weight_flat(
    precision_std: Tensor,   # (B, K, S) per-cell pooled σ (raw |STFT| std), flat dual-band S
    precision_n: Tensor,     # (B, K) electrode count per parcel
    target_cell: Tensor,     # (B, N) bool, N = K·S, flat (K outer, S inner)
    alpha: float,
    eps: float,
    floor_pct: float,
    cap: float,
) -> Tensor:
    """Inverse-variance precision weight for the 2STFT flat-S M4 latent — builds
    the flat ``(B, N=K·S)`` σ²/n cells (K outer, S inner) and delegates to
    :func:`_precision_weight_core`. The whole-parcel tube means a target parcel's
    σ/n weight is shared across all its ``S`` slots (n is per-parcel; σ varies per
    slot)."""
    B, K, S = precision_std.shape
    N = K * S
    n_pow = precision_n.clamp(min=1.0).pow(alpha)            # (B, K)
    n_cell = n_pow.unsqueeze(-1).expand(B, K, S).reshape(B, N)
    sigma2 = precision_std.pow(2).reshape(B, N)              # (B, N)
    return _precision_weight_core(sigma2, n_cell, target_cell, eps, floor_pct, cap)


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
    precision_floor_pct: float = _M4_PRECISION_FLOOR_PCT,
    precision_cap: float = _M4_PRECISION_CAP,
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
            floor_pct=precision_floor_pct,
            cap=precision_cap,
        )
        loss = _weighted_l1_or_zero(pred, target, weight, loss_form)
    return MaskedJepaBreakdown(
        total=loss, phase="m4_freq", n_masked=int(target_cell.sum())
    )


def m2_dual_band_loss(
    *,
    predictor: torch.nn.Module,  # JepaPredictor (n_identity = K parcel, n_identity_2 = F_p freq)
    student_m2: Tensor,   # (B, K, S, d) post-frontend_ln, VISIBLE-only (drop-token) student
    teacher_m2: Tensor,   # (B, K, S, d) post-frontend_ln, EMA full-input teacher
    token_mask: Tensor,   # (B, K, S) bool, True = masked (the §5 dual-band sampler)
    slot_ids: Tensor,     # (S,) long — per-token dual-rate real-time slot (layout["slot"])
    freq_patch_ids: Tensor,  # (S,) long — per-token freq-patch id / "Hz tag" (layout["freq_patch_idx"])
    latent_valid: Tensor,  # (B, K) bool — covered parcels (gate context + targets)
    loss_form: _LossForm = "l1",
) -> MaskedJepaBreakdown:
    """2STFT M2 front-end masked JEPA (FE 2STFT §6 — ONE shared predictor).

    The 2STFT M2 tap is a FLAT per-parcel token sequence ``(B, K, S, d)`` (the two
    STFT bands carry different time rates, so they cannot share an ``(F_p, T_p)``
    grid — ``S = F_p_low·T_low_p + F_p_high·T_high_p``). The drop-token student
    encoded the VISIBLE tokens only (masked tokens key-padded out of its
    token-block self-attention — :meth:`V14ParcelPerceiverModel._forward_dual_band`
    with ``token_mask``); ``teacher_m2`` is the EMA full-input forward.

    **ONE shared predictor** reconstructs ALL masked slots — both baskets, both
    rates — in a single call (the §6 "one predictor" lock). Each masked query slot
    = a learnable mask token + a learned PARCEL tag (``query_id`` ∈ [0, K),
    unordered anatomy → ``id_pos="learned"``) + a sinusoidal FREQ/"Hz" tag
    (``query_id_2`` = freq-patch ∈ [0, F_p), ordered → ``id_pos_2="sinusoidal"``;
    **basket identity is implicit** — low freq-patches index ≤ the low count, high
    above) + the dual-rate real-time **RoPE** slot (``query_time_ids``, the §4
    keying, so a low token at time-patch ``t`` and a high token at ``2t`` share
    rotation). This is structurally :func:`b37_m4_freq_loss` (parcel + freq + time)
    on the FRONT-END M2 tap with the dual-rate slot in place of a single time grid.

    Target = EMA teacher full-input M2 at the masked slots, post the encoder's own
    ``frontend_ln`` (canonical V-JEPA target-norm — no separate head), detached
    (stop-grad). **Pure L1, pooled mean over all masked tokens** (no per-basket
    normalization — the §5 sampler balances the baskets ~24/24 and L1's per-token
    gradient is unit regardless of basket). ``latent_valid`` gates BOTH the visible
    context keys and the targets to COVERED parcels — an uncovered parcel's junk M2
    (zero-pooled) is never a context key nor a reconstruction target.

    Cost note (NOT silently capped): the context is every covered parcel's full
    ``S`` visible tokens, so ``n_ctx ≈ K_visible·S`` — heavy at production scale
    (a ragged-gather optimization is deferred, as in ``b37_m4_freq_loss``). The
    predictor is discarded after SSL, so this is pretraining wall-clock only.
    """
    if student_m2.dim() != 4:
        raise ValueError(
            f"student_m2 must be (B, K, S, d); got {tuple(student_m2.shape)}"
        )
    if teacher_m2.shape != student_m2.shape:
        raise ValueError(
            f"teacher_m2 {tuple(teacher_m2.shape)} != student_m2 "
            f"{tuple(student_m2.shape)}"
        )
    B, K, S, d = student_m2.shape
    if token_mask.shape != (B, K, S):
        raise ValueError(
            f"token_mask {tuple(token_mask.shape)} must be (B, K, S)=({B},{K},{S})"
        )
    if latent_valid.shape != (B, K):
        raise ValueError(
            f"latent_valid {tuple(latent_valid.shape)} must be (B, K)=({B},{K})"
        )
    if slot_ids.shape != (S,) or freq_patch_ids.shape != (S,):
        raise ValueError(
            f"slot_ids / freq_patch_ids must be (S,)=({S},); got "
            f"{tuple(slot_ids.shape)} / {tuple(freq_patch_ids.shape)}"
        )
    device = student_m2.device
    N = K * S

    ctx = student_m2.reshape(B, N, d)               # visible used as keys (kpm below)
    # Gate by covered parcels: an uncovered parcel's M2 is zero-pooled junk, so
    # it must be neither a context key nor a target. covered broadcast over S →
    # per-cell (B, N) in the (K outer, S inner) flat order: flat index i = k·S + s.
    covered_cell = latent_valid.unsqueeze(-1).expand(B, K, S).reshape(B, N)
    mask_flat = token_mask.reshape(B, N)
    target_cell = mask_flat & covered_cell          # True = masked target (covered)
    visible_cell = (~mask_flat) & covered_cell      # True = visible context (covered)

    # (parcel, freq-patch, slot) ids for the flat (K outer, S inner) order. Parcel
    # is per-ROW (k) → learned query_id; freq-patch + slot are per-TOKEN (s),
    # tiled across the K parcels → sinusoidal query_id_2 / RoPE query_time_ids.
    parcel_ids = (
        torch.arange(K, device=device).view(K, 1).expand(K, S).reshape(N)
    )
    freq_ids = freq_patch_ids.to(device=device).view(1, S).expand(K, S).reshape(N)
    time_ids = slot_ids.to(device=device).view(1, S).expand(K, S).reshape(N)

    pred = predictor(
        ctx,
        context_time_ids=time_ids,
        query_time_ids=time_ids,
        query_id=parcel_ids,
        query_id_2=freq_ids,
        context_key_padding_mask=~visible_cell,
        query_valid=target_cell,
    )                                                # (n_masked, d)
    target = teacher_m2.reshape(B, N, d)[target_cell].detach()  # (n_masked, d)
    loss = _l1_or_zero(pred, target, loss_form)
    return MaskedJepaBreakdown(
        total=loss, phase="m2_dual_band", n_masked=int(target_cell.sum())
    )


def m4_dual_band_loss(
    *,
    predictor: torch.nn.Module,  # M4 predictor (SEPARATE from M2; depth 1 locked config)
    student_m4: Tensor,   # (B, K, S, d) post-encoder_ln, VISIBLE-PARCEL-only latent
    teacher_m4: Tensor,   # (B, K, S, d) post-encoder_ln, EMA full-input teacher
    tubed: Tensor,        # (B, K) bool — True = tubed parcel (whole-parcel M4 target)
    latent_valid: Tensor, # (B, K) bool — covered parcels (gate context + targets)
    token_mask: Tensor,   # (B, K, S) bool, True = M2-band-masked (drop from M4 ctx)
    slot_ids: Tensor,     # (S,) long — per-token dual-rate real-time RoPE slot
    freq_patch_ids: Tensor,  # (S,) long — per-token freq-patch id / "Hz tag"
    loss_form: _LossForm = "l1",
    # Heteroscedastic / inverse-variance precision weighting (OPT-IN; both None →
    # byte-identical to the plain-L1 path). project_v14_heteroscedastic_ssl_loss.
    precision_std: tp.Optional[Tensor] = None,  # (B, K, S) pooled σ (raw |STFT| std)
    precision_n: tp.Optional[Tensor] = None,    # (B, K) electrode count per parcel
    precision_alpha: float = 1.0,
    precision_floor_pct: float = _M4_PRECISION_FLOOR_PCT,
    precision_cap: float = _M4_PRECISION_CAP,
) -> MaskedJepaBreakdown:
    """2STFT M4 latent masked JEPA — the WHOLE-PARCEL tube on the flat-S latent.

    The 2STFT M4 tap is a FLAT per-parcel latent sequence ``(B, K, S, d)`` (the
    parcel-only self-attention output, rate-agnostic — the two STFT bands carry
    different time rates so they cannot share an ``(F_p, T_p)`` grid;
    ``S = F_p_low·T_low_p + F_p_high·T_high_p``). This is the dual-band sibling of
    :func:`b37_m4_freq_loss` (whose latent is ``parcel × freq × time``) — the tube
    is the ATOMIC PARCEL: a covered parcel is EITHER surviving (whole ``S`` field is
    context) OR tubed (whole ``S`` field is the target), mutually exclusive. So
    ``tubed`` is per-parcel ``(B, K)`` (no time axis) — Ben's lock: "M4 tube = whole
    parcel, all time AND all freq; reconstruct each tubed parcel from the co-located
    activity of all surviving parcels."

    Drop-token V-JEPA: the visible-parcel-only student encoded M4 of the SURVIVING
    parcels (tubed parcels excluded from the parcel self-attention upstream —
    :meth:`V14ParcelPerceiverModel._forward_dual_band` with ``dual_band_parcel_mask``;
    the tubed rows of ``student_m4`` are junk and are read as NEITHER keys NOR query
    sources). ``teacher_m4`` is the EMA full-input forward (all parcels). **ONE M4
    predictor** (separate from M2, depth 1 in the locked config) reads every
    surviving parcel's full ``S`` field as context and reconstructs every tubed
    parcel's full ``S`` field. Each masked query slot = a learnable mask token + a
    learned PARCEL tag (``query_id`` ∈ [0, K), ``id_pos="learned"``) + a sinusoidal
    FREQ/"Hz" tag (``query_id_2`` = freq-patch, ``id_pos_2="sinusoidal"``; basket
    identity implicit in the Hz order) + the dual-rate real-time **RoPE** slot
    (``query_time_ids``). Target = EMA teacher full-input M4 at the tubed slots,
    post the encoder's own ``encoder_ln`` (canonical V-JEPA target-norm), detached.

    ``latent_valid`` gates BOTH context and targets to COVERED parcels (an uncovered
    parcel's zero-pooled junk M4 is never a key nor a target). Heteroscedastic
    precision weighting is opt-in (``precision_std``/``precision_n`` from the encoder
    taps): both ``None`` → plain pooled-mean L1; both set → inverse-variance weight
    ``w = n_k^α / (σ² + σ²₀)`` over the scored cells (project_v14_heteroscedastic_ssl_loss).

    Cost note (NOT silently capped): the context is every surviving parcel's full
    ``S`` tokens, so ``n_ctx ≈ K_visible·S`` — heavy at scale (a ragged-gather
    optimization is deferred, as in :func:`b37_m4_freq_loss`). The predictor is
    discarded after SSL → pretraining wall-clock only, never inference.
    """
    if student_m4.dim() != 4:
        raise ValueError(
            f"student_m4 must be (B, K, S, d); got {tuple(student_m4.shape)}"
        )
    if teacher_m4.shape != student_m4.shape:
        raise ValueError(
            f"teacher_m4 {tuple(teacher_m4.shape)} != student_m4 "
            f"{tuple(student_m4.shape)}"
        )
    B, K, S, d = student_m4.shape
    if tubed.shape != (B, K):
        raise ValueError(f"tubed {tuple(tubed.shape)} must be (B, K)=({B},{K})")
    if latent_valid.shape != (B, K):
        raise ValueError(
            f"latent_valid {tuple(latent_valid.shape)} must be (B, K)=({B},{K})"
        )
    if token_mask.shape != (B, K, S):
        raise ValueError(
            f"token_mask {tuple(token_mask.shape)} must be (B, K, S)=({B},{K},{S})"
        )
    if slot_ids.shape != (S,) or freq_patch_ids.shape != (S,):
        raise ValueError(
            f"slot_ids / freq_patch_ids must be (S,)=({S},); got "
            f"{tuple(slot_ids.shape)} / {tuple(freq_patch_ids.shape)}"
        )
    device = student_m4.device
    N = K * S

    ctx = student_m4.reshape(B, N, d)               # surviving used as keys (kpm below)
    # A covered parcel is EITHER surviving (context) OR tubed (target) — mutually
    # exclusive. Broadcast each per-parcel bool over S → per-cell (B, N) in the
    # (K outer, S inner) flat order: flat index i = k·S + s.
    target_parcel = latent_valid & tubed            # (B, K) tube targets
    visible_parcel = latent_valid & ~tubed          # (B, K) surviving context
    # Queries = every tubed parcel at ALL S=(freq×time) spots (whole-parcel tube).
    target_cell = target_parcel.unsqueeze(-1).expand(B, K, S).reshape(B, N)
    # Context = the surviving (untubed, covered) parcels' UN-DROPPED cells only:
    # the encoder DROPPED the M2-band-masked tokens to zero in the token blocks, so
    # the M4 predictor must read ONLY the M2-visible cells of the surviving parcels
    # — never a dropped token's zero content (Ben "NEVER READ ZEROS" 2026-06-13;
    # matches the encoder's per-(parcel, slot) M4-latent key mask ``cell_keep``).
    visible_cell = (
        visible_parcel.unsqueeze(-1) & ~token_mask
    ).reshape(B, N)                                  # (B, N) M2-visible surviving

    # (parcel, freq-patch, slot) ids for the flat (K outer, S inner) order. Parcel
    # is per-ROW (k) → learned query_id; freq-patch + slot are per-TOKEN (s), tiled
    # across the K parcels → sinusoidal query_id_2 / dual-rate RoPE query_time_ids.
    parcel_ids = (
        torch.arange(K, device=device).view(K, 1).expand(K, S).reshape(N)
    )
    freq_ids = freq_patch_ids.to(device=device).view(1, S).expand(K, S).reshape(N)
    time_ids = slot_ids.to(device=device).view(1, S).expand(K, S).reshape(N)

    pred = predictor(
        ctx,
        context_time_ids=time_ids,
        query_time_ids=time_ids,
        query_id=parcel_ids,
        query_id_2=freq_ids,
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
        if precision_std.shape != (B, K, S):
            raise ValueError(
                f"precision_std shape {tuple(precision_std.shape)} != "
                f"(B, K, S)=({B},{K},{S})"
            )
        if precision_n.shape != (B, K):
            raise ValueError(
                f"precision_n shape {tuple(precision_n.shape)} != (B, K)=({B},{K})"
            )
        weight = _m4_precision_weight_flat(
            precision_std=precision_std,
            precision_n=precision_n,
            target_cell=target_cell,
            alpha=precision_alpha,
            eps=_M4_PRECISION_EPS,
            floor_pct=precision_floor_pct,
            cap=precision_cap,
        )
        loss = _weighted_l1_or_zero(pred, target, weight, loss_form)
    return MaskedJepaBreakdown(
        total=loss, phase="m4_dual_band", n_masked=int(target_cell.sum())
    )


__all__ = [
    "MaskedJepaBreakdown",
    "p1_frontend_m2_loss",
    "p2_parcel_m4_loss",
    "b37_m4_freq_loss",
    "m2_dual_band_loss",
    "m4_dual_band_loss",
]
