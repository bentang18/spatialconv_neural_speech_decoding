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

import os
import typing as tp

import torch
from pydantic import BaseModel, ConfigDict


SCALE_TO_SIGMA: float = 1.4826  # consistency constant: σ ≈ k · MAD for Normal

# Launch-time activation for the read-time winsor clamp. Set ``V14_SESSION_Z_WINSOR``
# in the job environment to the |z| cap (e.g. ``2500``) and every
# ``SessionRobustZNormalizer`` constructed WITHOUT an explicit ``winsor`` picks it up.
# Deliberately an env knob, NOT a ``MultiStftView`` pydantic field: winsor is a pure
# READ-time clamp (the cached |STFT| frames and the fitted median/σ stats are byte-
# identical with or without it), so threading it as a config field would change
# ``infra.uid()`` and fork the multi-TB spec cache for a value that does not alter a
# single cached byte. Reading it here keeps the clamp cache-neutral and out of
# ``view.py``/``dispatch_v14.py``. The launch env is recorded in the run-config ledger,
# so the active value stays auditable.
_WINSOR_ENV_VAR = "V14_SESSION_Z_WINSOR"


def _winsor_from_env() -> tp.Optional[float]:
    """Parse ``V14_SESSION_Z_WINSOR`` → positive ``float`` cap, or ``None`` when unset
    or blank. A set-but-unparseable / non-positive value fails loud rather than
    silently disabling the backstop (a typo'd cap must not pass as 'off')."""
    raw = os.environ.get(_WINSOR_ENV_VAR)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_WINSOR_ENV_VAR}={raw!r} is not a float |z| cap (e.g. '2500')"
        ) from exc
    if not (value > 0):
        raise ValueError(
            f"{_WINSOR_ENV_VAR}={raw!r} must be a positive |z| cap, got {value}"
        )
    return value


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


class SessionRobustZNormalizer:
    """Session-level robust-z: fit per-(electrode, freq-bin) median + MAD-σ once
    over a session/train recording, then apply the FROZEN stats to each clip.

    This is the stateful fit/apply form of :func:`robust_z` (which fits its
    stats per call over its own time axis). Fitting at session granularity is
    what the v14 N_v14 recipe specifies: stable stats (the full recording, not a
    noisy ~20-frame clip). Per B13 (``v14_blockers.md``) this is a
    per-session-OWN-recording contract — ``fit`` over each session's WHOLE
    recording and ``transform`` that session's clips with those stats,
    computed IDENTICALLY at train and eval (NO train/eval split filter). It is
    physical-unit calibration (impedance × amp-gain), so there is no leak:
    stats are per-session-own and never pooled across sessions, so a test
    session's clips are normalized only by that test session's own moments.
    The cohort-pooled variant is the ``R-norm-cohort-pooled`` P1 sister.
    Gain-invariant: a pure ×k per-channel gain scales both median and σ by k,
    so it cancels exactly in linear space.

    Scope note (WS-C / C3): this object only fits/applies the (C, F) stats. The
    "where the stats get fit" wiring — a per-session precompute over each
    session's Multi-STFT frames, cached per (electrode, freq, session) — is the
    data pipeline's job (WS-E/WS-H, multi-corpus). The default dispatch drops
    ``scaler="StandardScaler"`` from the view so this normalizer owns the
    post-STFT scaling once that precompute lands.
    """

    _FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64)

    def __init__(
        self, *, sigma_floor: float = 1e-6, winsor: tp.Optional[float] = None
    ) -> None:
        self.sigma_floor = sigma_floor
        # An explicit ``winsor`` always wins; only fall back to the launch env when
        # the caller passed nothing. The view.py construction sites pass no winsor,
        # so this is where ``V14_SESSION_Z_WINSOR`` reaches the read path.
        if winsor is None:
            winsor = _winsor_from_env()
        if winsor is not None and not (winsor > 0):
            raise ValueError(f"winsor must be a positive |z| cap, got {winsor}")
        self.winsor = winsor
        self.median: tp.Optional[torch.Tensor] = None
        self.sigma: tp.Optional[torch.Tensor] = None
        self._valid_bin_mask: tp.Optional[torch.Tensor] = None

    @classmethod
    def from_stats(
        cls,
        *,
        median: torch.Tensor,
        sigma: torch.Tensor,
        sigma_floor: float = 1e-6,
        winsor: tp.Optional[float] = None,
        valid_bin_mask: tp.Optional[torch.Tensor] = None,
    ) -> "SessionRobustZNormalizer":
        """Rebuild a fitted normalizer from already-computed ``median``/``sigma``
        (e.g. loaded from a stats cache) — skips refitting over the frames. The
        stats are a deterministic function of the frames + reduce axis, so a
        ``from_stats`` normalizer is identical to a ``fit`` one. ``sigma_floor``
        and ``winsor`` are applied at ``transform`` (not baked into ``sigma``), so
        they may differ from the values used when the stats were originally fit."""
        obj = cls(sigma_floor=sigma_floor, winsor=winsor)
        obj.median = median
        obj.sigma = sigma
        obj._valid_bin_mask = valid_bin_mask
        return obj

    def fit(
        self,
        frames: torch.Tensor,
        *,
        valid_bin_mask: tp.Optional[torch.Tensor] = None,
    ) -> "SessionRobustZNormalizer":
        """Fit median + MAD-σ over the last (time) axis of ``frames``.

        ``frames`` is the session's view output ``(..., F, T_session)`` (all
        train clips of a session concatenated along time). Stats reduce over the
        last axis → ``(..., F, 1)``, kept independent per (electrode, freq).
        ``valid_bin_mask`` (broadcastable to a clip's shape, True = valid) is
        stored and applied on ``transform`` so invalid bins return z=0.
        """
        x = frames if frames.dtype in self._FLOAT_DTYPES else frames.float()
        median = x.median(dim=-1, keepdim=True).values
        mad = (x - median).abs().median(dim=-1, keepdim=True).values
        self.median = median
        self.sigma = SCALE_TO_SIGMA * mad
        self._valid_bin_mask = valid_bin_mask
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the frozen stats: ``z = (x − median) / max(σ, floor)``, with
        constant-bin (σ < floor) and invalid-bin positions zeroed, and an optional
        ``winsor`` clamp on ``|z|``.

        The robust-z floors only the DENOMINATOR, so a brief intense ``|STFT|``
        transient on a flaky contact yields a huge *finite* ``z`` (numerator
        unbounded) that spikes the SSL gradient. Static bad-electrode exclusion
        removes the contacts that RECUR across trials; ``winsor`` is the
        complementary backstop for the residual single-trial transient (which the
        recurrence rule deliberately does not exclude). Pick the cap ABOVE the
        legitimate signal range and BELOW the artifact range; default ``None``
        (off) so behavior is unchanged until set."""
        if self.median is None or self.sigma is None:
            raise RuntimeError(
                "SessionRobustZNormalizer.transform called before fit()"
            )
        x = x if x.dtype in self._FLOAT_DTYPES else x.float()
        safe_sigma = self.sigma.clamp(min=self.sigma_floor)
        z = (x - self.median) / safe_sigma
        z = torch.where(self.sigma >= self.sigma_floor, z, torch.zeros_like(z))
        if self.winsor is not None:
            z = z.clamp(min=-self.winsor, max=self.winsor)
        if self._valid_bin_mask is not None:
            z = torch.where(
                self._valid_bin_mask.to(dtype=torch.bool), z, torch.zeros_like(z),
            )
        return z

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)
