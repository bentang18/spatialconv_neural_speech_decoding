"""Nv14 robust-z normalization (T1.6).

Spec lock (project_v14_preproc_recipe_2026_05_12.md, 5/22 amendment): the
v14 post-view normalizer is per-(electrode, freq-bin, session) **robust z**:

    μ = median over full-session time at that (electrode, freq, session)
    σ = 1.4826 · MAD over the same
    z = (x − μ) / max(σ, σ_floor)

Robust statistics matter here because Multi-STFT outputs are heavy-tailed
(transients, line-noise residuals after notch) and z-scoring on the raw
mean/std would let outliers compress the rest of the distribution.

Per-corpus valid-bin mask honored: when ``valid_bin_mask`` is provided,
invalid bins are passed through unchanged (z = 0) and *not* used to fit
statistics — SWEC's masked k22–k29 don't contribute pseudo-noise stats.

Where this sits in the NeuralFetch chain is an open discussion item
(``T1.6`` task description) — needs to attach BEFORE the segmenter so
the per-session statistics are computed once over the full recording,
not re-derived per trial window. ``Nv14RobustZTransform`` below is the
shape the wiring will take; the pure-function ``robust_z`` is the
substrate-agnostic primitive and is what the SSL/preflight harnesses
should call until the transform is wired.
"""

from __future__ import annotations

import typing as tp

import torch
from pydantic import BaseModel, ConfigDict


SCALE_TO_SIGMA: float = 1.4826  # consistency constant: σ ≈ k · MAD for Normal


def robust_z(
    x: torch.Tensor,
    *,
    valid_bin_mask: tp.Optional[torch.Tensor] = None,
    sigma_floor: float = 1e-6,
    dim: int = -1,
) -> torch.Tensor:
    """Per-(everything-except-``dim``) robust z-score.

    Inputs
    ------
    x
        Float tensor. Statistics are computed across ``dim`` (default the
        last axis — typically the time axis for ``(C, F, T)`` Multi-STFT
        output). All other axes are kept independent.
    valid_bin_mask
        Optional boolean mask broadcastable to ``x.shape``. Where False:
        return zeros at those positions and skip stat fitting there. The
        caller is responsible for the shape — e.g. for ``x: (C, F, T)``
        with a per-freq mask, pass ``(1, F, 1)``. Auto-alignment was tried
        and dropped because rank-1 broadcasting on multi-axis tensors is
        too ambiguous (which axis is the mask's axis?).
    sigma_floor
        Clamp on the scale denominator. Bins with σ < ``sigma_floor`` are
        treated as constant and their z-output is zeroed (instead of
        blowing up to ±∞).
    dim
        Axis along which median/MAD are computed.

    Output
    ------
    Float tensor of the same shape as ``x``, robust-z-normalized.
    """
    if x.dtype not in (torch.float16, torch.float32, torch.float64):
        x = x.float()
    median = x.median(dim=dim, keepdim=True).values
    centered = x - median
    mad = centered.abs().median(dim=dim, keepdim=True).values
    sigma = SCALE_TO_SIGMA * mad
    safe_sigma = sigma.clamp(min=sigma_floor)
    z = centered / safe_sigma
    # Where σ < floor (constant bins), output zeros — otherwise z = centered / floor
    # which is finite but meaningless. The floor's job is to avoid div-by-zero,
    # not to invent signal. So zero out positions where the underlying σ was below
    # the floor.
    z = torch.where(sigma >= sigma_floor, z, torch.zeros_like(z))

    if valid_bin_mask is not None:
        # Caller-provided mask: must be broadcastable to x.shape.
        z = torch.where(valid_bin_mask.to(dtype=torch.bool), z, torch.zeros_like(z))

    return z


class Nv14RobustZTransform(BaseModel):
    """Pydantic-shaped Nv14 transform skeleton — for later wiring into the
    NeuralFetch chain. The current pure-function entry point is :func:`robust_z`;
    callers (preflight, SSL losses) should use that until the substrate
    wiring is agreed (see T1.6 discussion point).

    Storing the (C, F) per-session statistics is a separate concern handled
    by the Exca cache: this object just describes "what" the transform is,
    not "where" it stores its fit state.
    """

    model_config = ConfigDict(extra="forbid")

    sigma_floor: float = 1e-6
    # `dim` defaults to the time axis (the last axis of a (C, F, T) tensor).
    dim: int = -1

    def __call__(
        self,
        x: torch.Tensor,
        *,
        valid_bin_mask: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return robust_z(
            x,
            valid_bin_mask=valid_bin_mask,
            sigma_floor=self.sigma_floor,
            dim=self.dim,
        )
