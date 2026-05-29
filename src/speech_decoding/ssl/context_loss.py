"""B25 amendment (2026-05-27) + B26 amendment (2026-05-27 PM):
V-JEPA 2.1 §2.3.1 Eq 2 — context self-supervision loss at
``L_pre_frame``.

Companion to the masked-patch ``L_pre_frame`` loss. Applies the same
per-element loss form to the *visible* (non-masked) M2 patches with
per-position weight ``λ_i = λ_ctx(step) / √d_min(i, M)`` where
``d_min`` is the Chebyshev distance on the per-electrode
``(F_p × T_p)`` grid from visible patch ``i`` to the nearest masked
patch ``M``. Visible patches closer to a masked patch carry more
weight — the model is rewarded for keeping visible-patch reps
consistent with the teacher where they could plausibly inform the
masked-prediction term.

**B26 loss form**: pure L1 default (V-JEPA 2 §2.1 Eq 1 + V-JEPA 2.1
§2.3.1 Eqs 1+2). Smooth-L1 / MSE remain available as opt-in sister-cell
falsifiers via ``loss_form``.

**B26 λ_ctx schedule**: linear 0 → ``λ_ctx_peak`` over the first 25% of
P1 steps, constant ``λ_ctx_peak`` thereafter. V-JEPA 2.1 Table 2 best
config: constant λ=0.5 without warmup is the WORST non-zero λ
(27.5 mIoU); warmed schedule buys +6.3 mIoU absolute. P2 enters at the
peak (warmup already done).

Total objective (P1 + P2, B26)::

    L_total = L_pre_frame_masked + λ_ctx(step) · L_pre_frame_context
            + L_mid_slot + L_post_frame
            + 1.0 · L_post_utterance + 0.1 · L_DKoleo

The masked-patch ``L_pre_frame_masked`` lives in
:mod:`speech_decoding.ssl.recon`; this module owns the context companion.

Sister cells:

* B25 / B26 ``R-no-context-loss`` (``λ_ctx=0``).
* B25 / B26 ``R-ctx-lambda-{0.0, 0.25, 0.5, 1.0}`` — peak value sweep.
* B26 ``R-no-warmup`` — λ_ctx=0.5 constant from step 0 (V-JEPA 2.1
  Table 2 worst non-zero config); falsifies the warmup is load-bearing.
* B26 ``R-l2-loss`` / ``R-mse-loss`` / ``R-smoothl1-beta-*`` —
  per-element loss form falsifiers.
"""

from __future__ import annotations

import typing as tp

import torch
import torch.nn.functional as F
from torch import Tensor


LAMBDA_CTX_DEFAULT: float = 0.5
"""V-JEPA 2.1 §2.3.1 Eq 2 best-config peak: λ_ctx = 0.5 (B25 lock 2026-05-27;
B26 unchanged 2026-05-27 PM)."""

LAMBDA_CTX_WARMUP_FRACTION: float = 0.25
"""B26 lock 2026-05-27 PM: linear-warmup fraction of P1 over which λ_ctx
ramps from 0 → ``LAMBDA_CTX_DEFAULT``. V-JEPA 2.1 Table 2 best schedule
warms over the first quarter of training; constant λ=0.5 without warmup
is the worst non-zero config (loses +6.3 mIoU absolute)."""


def lambda_ctx_p1_warmup_schedule(
    total_p1_steps: int,
    *,
    peak: float = LAMBDA_CTX_DEFAULT,
    warmup_fraction: float = LAMBDA_CTX_WARMUP_FRACTION,
) -> tp.Callable[[int], float]:
    """B26: λ_ctx schedule for Phase-1.

    Returns ``λ(step) → float`` that ramps linearly from 0 at step 0 to
    ``peak`` at ``warmup_fraction * total_p1_steps``, then stays at
    ``peak`` for the remainder of P1. P2 calls into this schedule with
    its own step counter and immediately gets ``peak`` (since P2 enters
    after P1 warmup is already done).

    Parameters
    ----------
    total_p1_steps
        Total P1 training steps; used to compute ``warmup_steps``.
    peak
        Target λ_ctx value; default ``LAMBDA_CTX_DEFAULT = 0.5`` per the
        B25/B26 V-JEPA 2.1 Eq 2 best-config peak.
    warmup_fraction
        Fraction of ``total_p1_steps`` over which to ramp; default
        ``LAMBDA_CTX_WARMUP_FRACTION = 0.25`` per V-JEPA 2.1 Table 2.

    Notes
    -----
    The ``R-no-warmup`` sister cell builds an equivalent constant
    schedule via ``lambda _step: LAMBDA_CTX_DEFAULT`` (no warmup, peak
    from step 0).
    """
    if total_p1_steps <= 0:
        raise ValueError(f"total_p1_steps must be positive, got {total_p1_steps}")
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError(
            f"warmup_fraction must be in [0.0, 1.0], got {warmup_fraction}"
        )
    warmup_steps = int(round(warmup_fraction * total_p1_steps))

    def schedule(step: int) -> float:
        if warmup_steps == 0:
            return float(peak)
        t = max(0, min(step, warmup_steps))
        return float(peak) * (t / float(warmup_steps))

    return schedule


def chebyshev_dmin_to_masked(patch_mask: Tensor) -> Tensor:
    """For each cell on a ``(..., F_p, T_p)`` mask grid, return the
    Chebyshev (``L∞``) distance to the nearest masked cell.

    Parameters
    ----------
    patch_mask
        Bool tensor of shape ``(..., F_p, T_p)``. ``True`` marks a
        masked patch.

    Returns
    -------
    Tensor
        Float tensor of shape ``(..., F_p, T_p)``. Cells with no masked
        neighbor anywhere on the grid receive ``+inf`` (V-JEPA 2.1
        convention; downstream λ-weight is ``1/√∞ = 0``). When every
        cell is masked, the result is all zeros (each cell is at
        distance 0 from itself).
    """
    if patch_mask.ndim < 2:
        raise ValueError(f"patch_mask must have ≥2 dims; got {patch_mask.ndim}")
    F_p, T_p = patch_mask.shape[-2], patch_mask.shape[-1]
    # Per-cell coords on the F_p × T_p grid.
    f_idx = torch.arange(F_p, device=patch_mask.device)
    t_idx = torch.arange(T_p, device=patch_mask.device)
    # Pairwise |Δf| and |Δt|: (F_p, F_p), (T_p, T_p).
    df = (f_idx[:, None] - f_idx[None, :]).abs()
    dt = (t_idx[:, None] - t_idx[None, :]).abs()

    flat_mask = patch_mask.reshape(-1, F_p, T_p)
    B_flat = flat_mask.shape[0]
    out = torch.empty((B_flat, F_p, T_p), dtype=torch.float32, device=patch_mask.device)

    for b in range(B_flat):
        m = flat_mask[b]
        if not m.any():
            out[b].fill_(float("inf"))
            continue
        # Find masked-cell coordinates (k, 2).
        masked_coords = m.nonzero(as_tuple=False)
        mfs = masked_coords[:, 0]  # (k,)
        mts = masked_coords[:, 1]  # (k,)
        # Per (f, t) grid cell, Chebyshev distance to each masked cell.
        df_to_masked = df[:, mfs]            # (F_p, k)
        dt_to_masked = dt[:, mts]            # (T_p, k)
        # Broadcast to (F_p, T_p, k) and take max over the last two grids.
        d = torch.maximum(df_to_masked[:, None, :], dt_to_masked[None, :, :])
        out[b] = d.min(dim=-1).values.to(torch.float32)
    return out.reshape(*patch_mask.shape)


def context_self_supervision_loss(
    *,
    student: Tensor,
    teacher: Tensor,
    patch_mask: Tensor,
    lambda_ctx: float = LAMBDA_CTX_DEFAULT,
    loss_form: tp.Literal["mse", "smooth_l1", "l1"] = "l1",
    smooth_l1_beta: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """V-JEPA 2.1 §2.3.1 Eq 2 context-self-supervision loss.

    Parameters
    ----------
    student, teacher
        Per-electrode-patch M2 reps of shape ``(B, C, F_p, T_p, d)``.
        ``teacher`` is expected to come from the EMA teacher and be
        already detached at the call site.
    patch_mask
        Bool tensor ``(B, C, F_p, T_p)``; ``True`` = masked patch (this
        term operates on visible patches, so masked cells receive zero
        weight here — they are scored by ``L_pre_frame_masked``).
    lambda_ctx
        ``λ_ctx`` per V-JEPA 2.1 Eq 2; default ``0.5`` (B25 / B26 peak
        value). Trainers should pass the warmed value from
        :func:`lambda_ctx_p1_warmup_schedule` each step rather than
        the bare default.
    loss_form
        ``"l1"`` (B26 default; V-JEPA 2 §2.1 Eq 1 + V-JEPA 2.1 §2.3.1
        Eqs 1+2). ``"mse"`` / ``"smooth_l1"`` retained for the
        ``R-l2-loss`` / ``R-mse-loss`` / ``R-smoothl1-beta-*`` sister
        falsifiers.
    smooth_l1_beta
        Smooth-L1 β (only consulted when ``loss_form="smooth_l1"``);
        default ``1.0`` to match the prior B25 lock.
    eps
        Numerical floor for the weight-sum denominator.

    Returns
    -------
    Tensor
        Scalar loss. The weighted Smooth-L1 numerator is summed per
        ``(electrode, clip)`` and divided by the per-clip weight sum;
        the resulting per-clip loss is averaged over ``(B, C)``.

    Notes
    -----
    * When no patch is masked anywhere in a clip, every weight is zero
      and the per-clip loss is ``0/eps ≈ 0`` — the trainer can call
      this primitive unconditionally without a guard.
    * Masked-cell positions contribute zero (they're handled by the
      masked-patch reconstruction loss in :mod:`recon`).
    """
    if student.shape != teacher.shape:
        raise ValueError(
            f"shape mismatch: student={tuple(student.shape)} "
            f"teacher={tuple(teacher.shape)}"
        )
    if student.ndim != 5:
        raise ValueError(
            f"expected (B, C, F_p, T_p, d); got {tuple(student.shape)}"
        )
    if patch_mask.ndim != 4:
        raise ValueError(
            f"patch_mask must be (B, C, F_p, T_p); got {tuple(patch_mask.shape)}"
        )

    B, C, F_p, T_p, d = student.shape
    if patch_mask.shape != (B, C, F_p, T_p):
        raise ValueError(
            f"patch_mask shape {tuple(patch_mask.shape)} must match "
            f"student[..., :-1] {(B, C, F_p, T_p)}"
        )

    # Per-(b, c) Chebyshev d_min on the (F_p, T_p) grid.
    bool_mask = patch_mask.to(torch.bool)
    dmin = chebyshev_dmin_to_masked(bool_mask)               # (B, C, F_p, T_p) float
    finite = torch.isfinite(dmin)
    visible = ~bool_mask
    # Weight λ_i is non-zero only at visible cells with finite d_min ≥ 1.
    # Cells with d_min == 0 are mask cells (already excluded by `visible`).
    sqrt_d = torch.sqrt(dmin.clamp(min=1).to(student.dtype))
    # ``lambda_ctx / √d_min`` at eligible positions, else 0.
    eligible = visible & finite & (dmin > 0)
    weights = torch.where(
        eligible,
        torch.tensor(lambda_ctx, dtype=student.dtype, device=student.device) / sqrt_d,
        torch.zeros((), dtype=student.dtype, device=student.device),
    )                                                          # (B, C, F_p, T_p)

    # Per-element loss; sum over feature axis, then weight per cell.
    if loss_form == "l1":
        per_elem = (student - teacher).abs()
    elif loss_form == "mse":
        per_elem = (student - teacher).pow(2)
    elif loss_form == "smooth_l1":
        per_elem = F.smooth_l1_loss(
            student, teacher, beta=smooth_l1_beta, reduction="none",
        )
    else:
        raise ValueError(f"unknown loss_form={loss_form!r}")
    # (B, C, F_p, T_p, d) → sum over the feature axis.
    per_cell = per_elem.sum(dim=-1)                            # (B, C, F_p, T_p)

    weighted_sum = (per_cell * weights).sum(dim=(-2, -1))      # (B, C)
    # Weight normalizer per clip — multiplied by d so weighted_mean over
    # the feature axis aggregates correctly.
    weight_norm = (weights * d).sum(dim=(-2, -1)).clamp(min=eps)
    per_clip = weighted_sum / weight_norm                      # (B, C)
    return per_clip.mean()
