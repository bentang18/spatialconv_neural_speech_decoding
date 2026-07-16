"""MON-PERCEIVER-HEALTH: representation-health reductions for the r4 secondary
Perceiver's processed latents (post-``process`` self-attention, pre-decode).

The perceiver's ``lat`` is a ``(B, n_slots·M, d_perc=128)`` bank — the shared cross-parcel
regime (M10) the Gaussian head decodes from. Two failure modes matter and RankMe alone
(already logged via the shared ``_rank_and_std`` path, same as the encoder's block-12 tap)
does not name either directly:

  * DEAD FEATURE DIMS — a coordinate of the 128-d code whose across-latent std collapses
    to ~0 carries no information; the fraction of such dims is a direct collapse read that
    survives even when RankMe is still moderate.
  * LATENT-LATENT REDUNDANCY — the n_slots·M latents pointing the same way (high pairwise
    cosine) means the bank has fewer effective modes than M10's n@90%≈8.7 asks for. This is
    the write-side twin of the v2 slot-bank redundancy monitor; it REUSES those tested
    pairwise-cosine helpers rather than re-deriving them.

VICReg's variance/covariance terms are DROPPED as separate monitors: RankMe subsumes the
global-collapse signal and these two scalars cover the residual (per-dim death, per-latent
redundancy) more legibly. Everything here is a pure reduction over a DETACHED tap — the
callback owns cadence, DDP gating, and logging.
"""

from __future__ import annotations

import torch
from torch import Tensor

from speech_decoding.experiments.monitors.slot_redundancy import (
    _diag_zeroed_mean,
    _pairwise_cosine,
    _upper_triangular_pct95,
)

# A feature dim is "dead" when its across-latent std is below this fraction of the MEDIAN
# feature std. Relative-to-median (not mean) so a single hyperactive dim can't inflate the
# reference and hide genuinely dead dims. Scale-free: the processed latents are un-normed
# (post-residual), so an absolute std floor would be arbitrary. Flagged for Ben — log-only,
# no science rides on the exact value; it only sets when ``perc_lat_dead_frac`` ticks up.
DEAD_FEAT_STD_REL: float = 0.05


def dead_feature_fraction(feats: Tensor, *, rel: float = DEAD_FEAT_STD_REL) -> float:
    """Fraction of the ``d`` feature dims whose across-sample std < ``rel × median(std)``.

    ``feats`` (N, d) — the flattened latent bank (rows = latents, cols = code dims). Returns
    0.0 when N < 2 (std undefined). A truly dead dim (std == 0) is counted for any positive
    reference; an all-equal-variance code returns ~0 (median ≈ each std, none below rel·median).
    """
    if feats.dim() != 2:
        raise ValueError(f"expected (N, d); got shape {tuple(feats.shape)}")
    if feats.shape[0] < 2:
        return 0.0
    std = feats.std(dim=0, unbiased=False)  # (d,)
    ref = std.median()
    thresh = rel * ref
    dead = (std < thresh).to(torch.float32).mean()
    return float(dead.item())


def latent_redundancy(lat: Tensor) -> tuple[float, float]:
    """Per-clip latent-latent cosine redundancy of the processed latent bank.

    ``lat`` (B, L, d) with L = n_slots·M. For each clip, the L×L pairwise-cosine matrix's
    diagonal-zeroed mean and strict-upper-triangular 95th percentile are computed, then
    averaged over the batch. Returns ``(mean_cos, pct95_cos)``: high = the bank's latents
    are redundant (fewer effective modes). Signed cosine (matches the v2 slot monitor):
    redundancy is same-direction alignment, anti-aligned latents are not redundant.
    """
    if lat.dim() != 3:
        raise ValueError(f"expected (B, L, d); got shape {tuple(lat.shape)}")
    B = lat.shape[0]
    means: list[float] = []
    pct95s: list[float] = []
    for b in range(B):
        mat = _pairwise_cosine(lat[b])  # (L, L)
        means.append(_diag_zeroed_mean(mat))
        pct95s.append(_upper_triangular_pct95(mat))
    n = max(B, 1)
    return sum(means) / n, sum(pct95s) / n


def perceiver_latent_health(
    lat: Tensor, *, dead_rel: float = DEAD_FEAT_STD_REL
) -> dict[str, float]:
    """Bundle the two residual (non-RankMe) perceiver-latent health scalars.

    ``lat`` (B, L, d). Returns ``dead_frac`` (over the flattened (B·L, d) code) and the
    per-clip ``cos_mean`` / ``cos_pct95`` redundancy. RankMe + feat_std_mean/min are logged
    separately by the callback's shared ``_rank_and_std`` on the same tap (no duplication).
    """
    if lat.dim() != 3:
        raise ValueError(f"expected (B, L, d); got shape {tuple(lat.shape)}")
    flat = lat.reshape(-1, lat.shape[-1])
    cos_mean, cos_pct95 = latent_redundancy(lat)
    return {
        "dead_frac": dead_feature_fraction(flat, rel=dead_rel),
        "cos_mean": cos_mean,
        "cos_pct95": cos_pct95,
    }


__all__ = [
    "dead_feature_fraction",
    "latent_redundancy",
    "perceiver_latent_health",
    "DEAD_FEAT_STD_REL",
]
