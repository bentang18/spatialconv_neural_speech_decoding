"""VICReg variance + covariance anti-collapse terms (Bardes et al. 2022, §3).

Why this exists — the P1 collapse (2026-06-04 capstone abort):
    P1 is predictor-free EMA self-distillation (paradigm A): the loss is L1
    between the student's masked front-end M2 and the EMA teacher's full-input
    M2 (post ``frontend_ln``). That objective has a degenerate GLOBAL minimum
    at the constant function — map every (electrode, freq-patch, time-patch)
    token to one shared vector and student ≡ teacher ≡ const ⇒ L1 = 0 exactly.
    ``frontend_ln`` is ``LayerNorm(d_model)`` — it normalizes each token across
    its feature dim, so it fixes per-token scale but places NO constraint
    ACROSS tokens; the all-tokens-identical (rank-1) solution survives it.
    The capstone run descended straight into it: front-end M2 normalised RankMe
    fell from the healthy [0.25, 0.5) band into < 0.25 while val_loss dropped
    0.208 → 0.142 (loss DOWN as rank → 1 — the textbook collapse signature).
    The 4-agent audit found P1 had NO surviving anti-collapse mechanism (no
    predictor like P2, DKoleo demoted to a sister by B28, only the rank-blind
    LayerNorm).

The fix — the full VICReg variance+covariance pair on the student
representation whose rank we protect (the student-side analog of the M2 rows
the EMA-teacher RankMe monitor scores; the term must act on the student since
the teacher target is a detached EMA copy). Two terms:

  * **Variance floor** (one-sided). For each feature dim ``j`` compute the std
    of ``z[:, j]`` across the token rows and hinge it up to ``gamma``:

        L_var = mean_j  relu(gamma - std_j)

    This makes the CONSTANT (rank-1) solution a HIGH-loss point: collapse ⇒
    every per-dim cross-token std → 0 ⇒ the hinge is maximal. One-sided — once
    ``std_j >= gamma`` the term and its gradient are exactly 0, so a generous
    coefficient never distorts a healthy representation. Scaling cannot cheat
    it: identical tokens have per-dim std 0 regardless of the LN affine scale.

  * **Covariance decorrelation.** The off-diagonal of the (token-centered)
    feature covariance, squared and summed:

        L_cov = (1/d) Σ_{i≠j} C_ij²,   C = zᵀz / (N-1),  z token-centered

    The variance floor ALONE is insufficient — it is blind to DIMENSIONAL
    collapse. A rank-k subspace (k ≪ d) can hold every per-dim std ≥ gamma
    (so L_var = 0) while the effective rank sits deep in the RankMe alarm
    band: the audit reproduced this by gradient descent (a rank-8 student in
    d=256 reads RankMe-norm ≈ 0.03 — a hard alarm — with the variance floor
    fully satisfied). The covariance term is exactly what closes that hole:
    a low-rank embedding forces large off-diagonal feature correlations, so
    L_cov penalises it directly. Variance kills constant collapse; covariance
    kills dimensional collapse. (RankMe stays the independent live abort
    backstop in ``collapse_guard`` regardless.)

DEVIATION FROM A LOCK (flag to Ben): B26/B27 lock P1 as "pure L1, no extra
terms". This adds anti-collapse regularisers, motivated by the empirical
refutation of the pure-L1-is-collapse-safe assumption (the capstone collapse)
and the audit's reproduction of the variance-only residual hole. Opt-in via
``--p1-var-coeff`` / ``--p1-cov-coeff`` (BOTH 0.0 ⇒ exact B26/B27 pure-L1, the
collapsing falsifier); both default ON for P1.
"""

from __future__ import annotations

import torch
from torch import Tensor


def variance_floor_loss(
    tokens: Tensor,
    *,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """VICReg variance hinge over the per-dim cross-token std.

    Parameters
    ----------
    tokens:
        ``(N, d)`` representation rows — the exact rows whose effective rank
        we protect (P1: the valid-electrode front-end M2 cells, post
        ``frontend_ln``, flattened over ``B, C, F_p, T_p``). Cast to fp32
        internally so a bf16-autocast std stays accurate (bf16 carries ~3
        significant digits — too few for a variance).
    gamma:
        Target std floor. Post-``frontend_ln`` (affine = 1 at init) the
        per-dim cross-token std of a healthy/diverse representation is ≈ 1,
        so ``gamma=1`` is calibrated to the init scale. Log ``mean_std``
        (returned) on a real run to confirm the healthy value and retune.
    eps:
        Added under the sqrt for a finite gradient at std → 0.

    Returns
    -------
    (loss, mean_std):
        ``loss`` is the scalar hinge (graph-connected to ``tokens``).
        ``mean_std`` is the detached mean per-dim std — a diagnostic logged
        so the collapse can be watched directly (healthy ≈ gamma; collapse
        → 0). ``N < 2`` (std undefined for < 2 rows) → both an exact 0
        connected to ``tokens`` (no signal, no penalty, no NaN).
    """
    if tokens.dim() != 2:
        raise ValueError(
            f"variance_floor_loss expects (N, d) tokens; got {tuple(tokens.shape)}"
        )
    if tokens.shape[0] < 2:
        zero = tokens.sum() * 0.0
        return zero, zero.detach()
    z = tokens.float()
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)  # (d,)
    loss = torch.relu(gamma - std).mean()
    return loss, std.mean().detach()


def covariance_loss(tokens: Tensor) -> Tensor:
    """VICReg covariance decorrelation — the off-diagonal of the token-centered
    feature covariance, squared, summed, and scaled by ``1/d`` (Bardes 2022 §3).

    Parameters
    ----------
    tokens:
        ``(N, d)`` representation rows (the same rows the variance floor uses).
        Cast to fp32 internally (a covariance over bf16 carries too few
        significant digits). If ``tokens`` is already fp32 the cast is a
        no-op, so a caller that pre-casts and shares the tensor with
        :func:`variance_floor_loss` pays the fp32 copy only once.

    Returns
    -------
    The scalar ``(1/d) Σ_{i≠j} C_ij²`` (graph-connected to ``tokens``).
    ``C = zᵀz / (N-1)`` is the covariance of the token-centered features;
    only the OFF-diagonal is penalised (the diagonal — per-dim variance — is
    the variance floor's job, and penalising it would fight that term). A
    decorrelated full-rank representation → ≈ 0; a low-rank (dimensionally
    collapsed) one forces large feature correlations → large loss, which is
    why this term, not the variance floor, is what arrests dimensional
    collapse. ``N < 2`` (covariance undefined) → an exact graph-connected 0.
    """
    if tokens.dim() != 2:
        raise ValueError(
            f"covariance_loss expects (N, d) tokens; got {tuple(tokens.shape)}"
        )
    n, d = tokens.shape
    if n < 2:
        return tokens.sum() * 0.0
    z = tokens.float()
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (n - 1)                       # (d, d)
    off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag_sq / d


__all__ = ["variance_floor_loss", "covariance_loss"]
