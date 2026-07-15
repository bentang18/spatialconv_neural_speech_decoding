"""Compact in-objective monitor reductions for r4 (#40 per-band JEPA, #41 per-band NLL,
#42 predicted-cov entropy vs floor).

The JEPA ``pred``/``tgt`` are ``(B, total, 1024)`` — GB-scale, too big to ship to the
Lightning callback as raw taps. These pure functions reduce them (and the small secondary
``mu``/``cov``) to per-band SCALARS INSIDE the objective's forward, on the monitor cadence
only; the callback then just logs the scalars.

Every split uses WEIGHTED reductions over the STATIC ``(B, total)`` grid (never a boolean
index), matching how the JEPA loss itself is computed — so the compiled forward's shapes
stay static and no data-dependent branch is introduced on the monitor path.
"""

from __future__ import annotations

import torch
from torch import Tensor

from speech_decoding.models.v14_converged_v3.secondary_head import (
    _nll_terms,
    gaussian_entropy,
)

_EPS: float = 1e-8
BAND_NAMES: tuple[str, ...] = ("slow", "mid", "hga")


def per_band_jepa_stats(
    pred: Tensor, tgt: Tensor, w: Tensor, band_ids: Tensor
) -> dict[str, Tensor]:
    """Per-band explained-var / pred-target var-ratio / L1 over the margin-gated scored
    tokens — the r4 (per-band) replacement for the retired whole/intra tier split.

    ``pred``/``tgt`` (B, total, F); ``w`` (B, total) the ``in_loss`` weight (float, 1 at a
    scored token else 0); ``band_ids`` (total,) in {0,1,2}. Matches ``_tier_stats``
    semantics: variance is over the SCORED-TOKEN axis, per feature, averaged over features
    (``mean_f Var_tok``). Computed with weighted reductions (static shape), using

        Var = mean_f( E[x_f²] − E[x_f]² ),  E weighted over the band's scored tokens
            = E[‖x‖²]/F  −  ‖E[x]‖²/F

    so the per-feature mean vector is one matmul (no GB-scale elementwise temporary) and the
    ``E[‖x‖²]`` term reduces the feature axis first (a (B,total) intermediate)."""
    B, total, n_feat = pred.shape
    resid = tgt - pred
    # per-token squared norms — reduce the feature axis first (one GB transient each, freed).
    sqn_t = tgt.pow(2).sum(-1)  # (B, total)
    sqn_p = pred.pow(2).sum(-1)
    sqn_r = resid.pow(2).sum(-1)
    l1_tok = resid.abs().mean(-1)  # (B, total)
    pred_f = pred.reshape(B * total, n_feat)
    tgt_f = tgt.reshape(B * total, n_feat)
    resid_f = resid.reshape(B * total, n_feat)

    out: dict[str, Tensor] = {}
    for b, name in enumerate(BAND_NAMES):
        wb = w * (band_ids == b).to(w.dtype).unsqueeze(0)  # (B, total)
        denom = wb.sum().clamp_min(1.0)
        wf = wb.reshape(1, B * total)

        def _mean_f_var(sqn_tok: Tensor, x_f: Tensor, _wb=wb, _wf=wf, _denom=denom) -> Tensor:
            e_sq = (sqn_tok * _wb).sum() / _denom  # E[‖x‖²]
            mean_vec = (_wf @ x_f).squeeze(0) / _denom  # (F,) weighted per-feature mean
            return (e_sq - mean_vec.pow(2).sum()) / n_feat

        var_t = _mean_f_var(sqn_t, tgt_f)
        var_p = _mean_f_var(sqn_p, pred_f)
        var_r = _mean_f_var(sqn_r, resid_f)
        out[f"jepa_{name}_explained_var"] = 1.0 - var_r / (var_t + _EPS)
        out[f"jepa_{name}_pred_target_var_ratio"] = var_p / (var_t + _EPS)
        out[f"jepa_{name}_l1"] = (l1_tok * wb).sum() / denom
    return out


def per_band_nll(
    mu: Tensor, cov: Tensor, target_q: Tensor, present_q: Tensor
) -> dict[str, Tensor]:
    """Per-band marginal Gaussian NLL, scored on that band's present positions.

    ``mu``/``target_q`` (B, Q, 6), ``cov`` (B, Q, 6, 6), ``present_q`` (B, Q, 6). A joint
    Gaussian's MARGINAL over a dim subset IS the sub-block ``(μ_S, Σ_SS)``; band ``b``'s
    marginal is dims ``[b, b+3]`` (its mean + its std). Scored where the band's std dim is
    present (n_elec ≥ 2 ⇒ the std target is defined), weighted-mean over positions (static
    shape). n=1 positions carry an undefined std and are weighted out — matching the way
    ``present_masked_nll`` drops them from the training loss."""
    n_mean = target_q.shape[-1] // 2  # 3
    pres = present_q.to(mu.dtype)
    out: dict[str, Tensor] = {}
    for b, name in enumerate(BAND_NAMES):
        idx = [b, b + n_mean]
        sub_mu = mu[..., idx]
        sub_x = target_q[..., idx]
        sub_cov = cov[..., idx, :][..., :, idx]  # (B, Q, 2, 2) marginal cov block
        nll_pos = _nll_terms(sub_mu, sub_cov, sub_x)  # (B, Q)
        wpos = pres[..., b + n_mean]  # present iff the band's std dim is on
        out[f"nll_{name}"] = (nll_pos * wpos).sum() / wpos.sum().clamp_min(1.0)
    return out


def cov_entropy_vs_floor(cov: Tensor, noise: Tensor) -> dict[str, Tensor]:
    """Predicted-cov differential entropy vs the noise-floor ceiling, meaned over positions.

    ``cov`` (B, Q, 6, 6) = ``L Lᵀ + diag(noise)``; ``noise`` (B, Q, 6) the per-position PD
    floor. The floor ceiling is ``entropy(diag(noise))`` (the head's L=0 limit — pure
    irreducible measurement noise, the NLL ceiling). ``gap = entropy(cov) − entropy(floor)``
    is ≥ 0 BY CONSTRUCTION (``cov ⪰ diag(noise)`` ⇒ larger log-det), so a healthy head sits
    ABOVE the floor and the gap tracks how much predictive covariance it is still carrying."""
    ent = gaussian_entropy(cov)  # (B, Q)
    ent_floor = gaussian_entropy(torch.diag_embed(noise))  # (B, Q)
    return {
        "cov_entropy": ent.mean(),
        "cov_entropy_floor": ent_floor.mean(),
        "cov_entropy_gap": (ent - ent_floor).mean(),
    }


__all__ = ["per_band_jepa_stats", "per_band_nll", "cov_entropy_vs_floor", "BAND_NAMES"]
