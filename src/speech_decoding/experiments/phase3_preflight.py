"""Phase-3 preflight harness (T3.2).

Picks (Whisper-layer k, target-rate r) for the Phase-3 distillation target via
a ridge sweep on one BT-Lite subject. Per ``docs/neuroprobe/v14_blockers.md``:

  - B06 gates the protocol details (ridge α grid, train/val split, tie-break).
  - M19 gates the choice of subject.

Both are REQUIRED kwargs here — the harness deliberately refuses to bake
defaults. The caller supplies the decisions once B06/M19 resolve and runs
the sweep on DCC.

Structure:

  - :class:`PreflightCell` — one (layer, rate) cell with its train+val score.
  - :class:`PreflightResult` — full table + chosen ``(k*, r*)``.
  - :func:`ridge_fit_score` — closed-form ridge regression solver returning
    ``(weights, train_metric, val_metric)`` for one ``(X, Y)`` pair.
  - :func:`run_preflight_sweep` — outer driver: iterates over the
    caller-supplied ``whisper_layers × target_rates`` grid, accumulates
    cells, applies ``tie_break`` to pick the winning cell.

The harness does not load Whisper, does not load a student checkpoint, and
does not touch DCC paths — those belong in the launcher script. The unit
tests exercise the math on synthetic Gaussian features.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import Literal

import torch
from torch import Tensor


@dataclasses.dataclass(frozen=True, slots=True)
class PreflightCell:
    """One ``(layer, rate)`` ridge-fit cell."""

    whisper_layer: int
    target_rate_hz: float
    alpha: float
    train_r2: float
    val_r2: float
    train_cosine: float
    val_cosine: float


@dataclasses.dataclass(frozen=True, slots=True)
class PreflightResult:
    """Full sweep table + chosen winning cell."""

    cells: tuple[PreflightCell, ...]
    winner: PreflightCell
    tie_break_metric: Literal["val_r2", "val_cosine"]


TieBreak = Literal["val_r2", "val_cosine"]


def _r2(pred: Tensor, target: Tensor) -> float:
    """Coefficient of determination R² aggregated across all output dims.

    Uses the standard 1 - SSE/SST form with SST computed per-dim then summed
    — equivalent to flattening when target is centered, but more robust to
    per-dim scale differences."""
    sse = (pred - target).pow(2).sum().item()
    centered = target - target.mean(dim=0, keepdim=True)
    sst = centered.pow(2).sum().item()
    if sst == 0.0:
        return float("nan")
    return 1.0 - sse / sst


def _cosine(pred: Tensor, target: Tensor) -> float:
    """Mean cosine similarity over rows (time steps)."""
    num = (pred * target).sum(dim=-1)
    den = pred.norm(dim=-1) * target.norm(dim=-1) + 1e-12
    return (num / den).mean().item()


def ridge_fit_score(
    *,
    X_train: Tensor,
    Y_train: Tensor,
    X_val: Tensor,
    Y_val: Tensor,
    alpha: float,
) -> tuple[Tensor, float, float, float, float]:
    """Closed-form ridge: ``W = (Xᵀ X + α I)⁻¹ Xᵀ Y``.

    Returns ``(W, train_r2, val_r2, train_cosine, val_cosine)``.

    Shapes:
        X: ``(N_t, D_in)`` brain features at the target rate
        Y: ``(N_t, D_out)`` Whisper-L_k features at the target rate
    """
    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("X must be 2-D (N_t, D_in)")
    if Y_train.ndim != 2 or Y_val.ndim != 2:
        raise ValueError("Y must be 2-D (N_t, D_out)")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("X_train and X_val must share D_in")
    if Y_train.shape[1] != Y_val.shape[1]:
        raise ValueError("Y_train and Y_val must share D_out")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    d_in = X_train.shape[1]
    xt_x = X_train.T @ X_train
    xt_y = X_train.T @ Y_train
    gram = xt_x + alpha * torch.eye(d_in, dtype=X_train.dtype, device=X_train.device)
    W = torch.linalg.solve(gram, xt_y)
    pred_train = X_train @ W
    pred_val = X_val @ W
    return (
        W,
        _r2(pred_train, Y_train),
        _r2(pred_val, Y_val),
        _cosine(pred_train, Y_train),
        _cosine(pred_val, Y_val),
    )


def _select_alpha(
    *,
    X_train: Tensor, Y_train: Tensor,
    X_val: Tensor, Y_val: Tensor,
    alphas: tuple[float, ...],
    tie_break: TieBreak,
) -> tuple[float, float, float, float, float]:
    """For one (layer, rate) cell, sweep α and pick the best by ``tie_break``.

    Returns ``(best_alpha, train_r2, val_r2, train_cosine, val_cosine)``."""
    best: tuple[float, float, float, float, float] | None = None
    for alpha in alphas:
        _, tr2, vr2, tc, vc = ridge_fit_score(
            X_train=X_train, Y_train=Y_train,
            X_val=X_val, Y_val=Y_val, alpha=alpha,
        )
        score = vr2 if tie_break == "val_r2" else vc
        if best is None or score > (
            best[2] if tie_break == "val_r2" else best[4]
        ):
            best = (alpha, tr2, vr2, tc, vc)
    assert best is not None  # alphas is non-empty (validated upstream)
    return best


def run_preflight_sweep(
    *,
    feature_fn: Callable[[int, float], tuple[Tensor, Tensor, Tensor, Tensor]],
    whisper_layers: Iterable[int],
    target_rates_hz: Iterable[float],
    ridge_alphas: tuple[float, ...],
    tie_break: TieBreak,
) -> PreflightResult:
    """Run the preflight sweep over ``whisper_layers × target_rates_hz``.

    ``feature_fn(layer, rate) -> (X_train, Y_train, X_val, Y_val)``: a caller
    callable that returns the brain features ``X`` and Whisper features ``Y``
    already pooled to the target rate, with train/val pre-split per B06.

    All other knobs (alpha grid, tie-break metric) are required — there is
    no default until B06 lands. The caller is also responsible for choosing
    the subject (M19) when constructing ``feature_fn``.

    Returns a :class:`PreflightResult` with the full cell table plus the
    winning ``(k*, r*, α*)`` triple.
    """
    layers = tuple(whisper_layers)
    rates = tuple(target_rates_hz)
    if not layers:
        raise ValueError("whisper_layers must be non-empty")
    if not rates:
        raise ValueError("target_rates_hz must be non-empty")
    if not ridge_alphas:
        raise ValueError("ridge_alphas must be non-empty")
    if tie_break not in ("val_r2", "val_cosine"):
        raise ValueError(f"unknown tie_break: {tie_break!r}")

    cells: list[PreflightCell] = []
    for layer in layers:
        for rate in rates:
            X_train, Y_train, X_val, Y_val = feature_fn(layer, rate)
            alpha, tr2, vr2, tc, vc = _select_alpha(
                X_train=X_train, Y_train=Y_train,
                X_val=X_val, Y_val=Y_val,
                alphas=ridge_alphas, tie_break=tie_break,
            )
            cells.append(PreflightCell(
                whisper_layer=layer,
                target_rate_hz=rate,
                alpha=alpha,
                train_r2=tr2, val_r2=vr2,
                train_cosine=tc, val_cosine=vc,
            ))

    winner = max(
        cells,
        key=lambda c: c.val_r2 if tie_break == "val_r2" else c.val_cosine,
    )
    return PreflightResult(
        cells=tuple(cells), winner=winner, tie_break_metric=tie_break,
    )
