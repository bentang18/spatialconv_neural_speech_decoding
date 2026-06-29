"""v2 raw-feature linear-probe floor — the dev (pretrain-corpus) probe baseline.

RAW RAW (Ben 2026-06-26): the linear probe reads **every ``|STFT|`` bin the v2
encoder is fed**, with no reduction whatsoever — no ``kernel_freq×kernel_time``
patch pooling, no Cartesian→magnitude collapse (v2 bands are already pure
magnitude, ``in_channels=1``). So the floor is a linear read of *genuinely the
model's input*, and the encoder↔raw head-to-head is finally apples-to-apples.

Per electrode, every band ``(N,C,F,T)`` is flattened over its full ``F×T`` grid
(freq-major, time-minor) and the bands are concatenated end-to-end into one raw
vector of ``D_raw = Σ_b F_b·T_b`` scalars (v2 @1 s: LFS 28×5=140 + HGA 7×33=231
= 371). There is no token grid and no ``bands_for_clip_len`` dependency — the
delivered band tensors already carry the clip's ``T``.

Protocol (unchanged from the prior raw baseline, only the feature construction is
new):
  - **WS (within-session)**: per-electrode, un-pooled — ``(N, C_valid·D_raw)``
    features, contiguous 2-fold, ``StandardScaler`` + L2 logistic (``C=1.0``),
    mean AUROC over WS subjects.
  - **CS (cross-subject)**: electrode→DKT-parcel **mean at every single bin
    position** → ``(N, n_parcels, D_raw)``, fit the sub2 anchor / score each test
    subject over the parcel intersection.

Pure torch/numpy/sklearn — no model, no forward, no GPU/DCC — so the whole path is
laptop-TDD'd (``test_v2_raw_probe``). The logistic/AUROC/fold primitives and the
parcel-intersection / finite-row helpers are imported from the already-TDD'd
upstream-parity modules, so this file only owns the raw-bin construction and the
single WS/CS loop.
"""

from __future__ import annotations

import typing as tp
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from speech_decoding.experiments.linear_probe_logistic import (
    cs_auroc_logistic,
    ws_auroc_2fold_logistic,
)
from speech_decoding.experiments.online_probe import (
    _finite_rows,
    cs_auroc as _cs_auroc_ridge,
    feature_matrix,
    parcel_intersection,
    ws_auroc_2fold as _ws_auroc_ridge,
)


def auroc_estimators(estimator: str, max_iter: int, lam_mult: float = 1.0):
    """Return ``(ws_fn(z,y), cs_fn(za,ya,zt,yt))`` for ``"logistic"`` or ``"ridge"``.

    Shared by the raw floor and the encoder-tap bench so both score with one meter.
    ``"ridge"`` = the dimension-robust dual ridge (``online_probe`` primitives; the
    trustworthy meter at p≫n); ``"logistic"`` = fixed-C L2 logistic (the original
    floor). The ridge primitives take no ``max_iter``. ``lam_mult`` scales the ridge λ
    rule (``lam_mult·trace(G)/N``) — default 1; set it to the λ picked by the
    pretrain-eval sweep (:mod:`v2_lambda_sweep`) to apply that λ to the benchmark
    eval. ``lam_mult`` is ridge-only (logistic ignores it)."""
    if estimator == "ridge":
        return (lambda z, y: _ws_auroc_ridge(z, y, lam_mult=lam_mult),
                lambda za, ya, zt, yt: _cs_auroc_ridge(za, ya, zt, yt, lam_mult=lam_mult))
    if estimator == "logistic":
        return (lambda z, y: ws_auroc_2fold_logistic(z, y, max_iter=max_iter),
                lambda za, ya, zt, yt: cs_auroc_logistic(za, ya, zt, yt, max_iter=max_iter))
    raise ValueError(f"estimator must be 'logistic' or 'ridge', got {estimator!r}")

__all__ = [
    "raw_bins_from_bands",
    "raw_pooled_tokens_from_bands",
    "per_electrode_features",
    "pool_electrodes_to_parcels",
    "raw_ws_cs_auroc",
    "run_v2_raw_baseline",
]


def raw_bins_from_bands(bands: Sequence[Tensor]) -> Tensor:
    """Flatten every ``|STFT|`` bin of each magnitude band → ``(N, C, D_raw, 1)``.

    Each band is ``(N, C, F, T)`` magnitude. Per electrode, ``F×T`` is flattened
    freq-major / time-minor and the bands are concatenated in the given order, so
    ``D_raw = Σ_b F_b·T_b`` (no pooling — every bin survives). The trailing ``d=1``
    lets :func:`per_electrode_features` / :func:`pool_electrodes_to_parcels` /
    :func:`feature_matrix` consume it unchanged."""
    if not bands:
        raise ValueError("need at least one band")
    cols: list[Tensor] = []
    n0, c0 = bands[0].shape[0], bands[0].shape[1]
    for b in bands:
        if b.ndim != 4:
            raise ValueError(f"each band must be (N,C,F,T) magnitude; got {tuple(b.shape)}")
        if b.shape[0] != n0 or b.shape[1] != c0:
            raise ValueError(
                f"bands must share (N,C); got {(b.shape[0], b.shape[1])} vs {(n0, c0)}"
            )
        n, c, f, t = b.shape
        cols.append(b.reshape(n, c, f * t))          # freq-major, time-minor
    return torch.cat(cols, dim=2).unsqueeze(-1)      # (N, C, D_raw, 1)


def raw_pooled_tokens_from_bands(
    bands: Sequence[Tensor], specs: Sequence[tp.Any]
) -> Tensor:
    """Mean-pool every ``|STFT|`` bin into the frontend's 22-token patch grid →
    ``(N, C, N_TOK, 1)``.

    The *linear* analogue of the learned :class:`TwoBandTokenizerV2` ``_PatchStem``
    pooling: per band, for each freq-patch group (cumulative ``freq_patch_bins``) ×
    each non-overlapping ``kernel_time`` time window, take the **mean** over that
    patch's ``freq×time`` bins → one scalar token. Token order is band-major,
    freq-group-major, time-minor — identical to the tokenizer's concat order — so
    ``N_TOK = Σ_b n_freq_patches_b · n_tp_b`` (= 22 @1 s). Each ``spec`` supplies the
    clip-independent ``freq_patch_bins`` + ``kernel_time``; the time count is read
    from the delivered band's own ``T``. Trailing ``d=1`` matches
    :func:`raw_bins_from_bands` so the WS/CS path is shared."""
    if not bands:
        raise ValueError("need at least one band")
    if len(bands) != len(specs):
        raise ValueError(f"need one spec per band; got {len(bands)} bands, {len(specs)} specs")
    n0, c0 = bands[0].shape[0], bands[0].shape[1]
    cols: list[Tensor] = []
    for b, spec in zip(bands, specs):
        if b.ndim != 4:
            raise ValueError(f"each band must be (N,C,F,T) magnitude; got {tuple(b.shape)}")
        n, c, f, t = b.shape
        if n != n0 or c != c0:
            raise ValueError(f"bands must share (N,C); got {(n, c)} vs {(n0, c0)}")
        kt = int(spec.kernel_time)
        if sum(spec.freq_patch_bins) != f:
            raise ValueError(
                f"band freq bins {f} != sum(freq_patch_bins) {sum(spec.freq_patch_bins)}"
            )
        n_tp = (t - kt) // kt + 1
        if n_tp < 1:
            raise ValueError(f"time frames {t} < kernel_time {kt}; clip too short")
        lo = 0
        for g in spec.freq_patch_bins:
            fg = b[:, :, lo:lo + g, :]                       # (N, C, g, T)
            lo += g
            for j in range(n_tp):
                w = fg[:, :, :, j * kt:(j + 1) * kt]         # (N, C, g, kt)
                cols.append(w.reshape(n, c, -1).mean(dim=2))  # (N, C)
    return torch.stack(cols, dim=2).unsqueeze(-1)            # (N, C, N_TOK, 1)


def per_electrode_features(raw: Tensor, electrode_mask: Tensor) -> Tensor:
    """WS feature matrix: keep valid electrodes, flatten → ``(N, C_valid·D_raw)``.

    No parcel pooling (within a session the montage is fixed). Column order is
    electrode-major then bin, identical across the subject's windows."""
    valid = electrode_mask.bool()
    sel = raw[:, valid]                               # (N, C_valid, D_raw, 1)
    return sel.reshape(sel.shape[0], -1)


def pool_electrodes_to_parcels(
    raw: Tensor,
    parcel_per_electrode: Tensor,
    electrode_mask: Tensor,
    n_parcels: int,
) -> tuple[Tensor, Tensor]:
    """CS pooling: electrode→DKT-parcel **mean, computed independently at every
    bin** → ``(N, n_parcels, D_raw, 1)`` + ``present`` ``(n_parcels,)``.

    The ONLY reduction (the bin axis is untouched — all ``D_raw`` survive). Empty
    parcels are zero with ``present=False`` so :func:`parcel_intersection` drops
    them."""
    n, c, d_raw, d = raw.shape
    if c != int(parcel_per_electrode.shape[0]) or c != int(electrode_mask.shape[0]):
        raise ValueError("parcel_per_electrode / electrode_mask must be length C")
    valid = electrode_mask.to(raw.dtype)
    pe = parcel_per_electrode.long()
    contrib = raw * valid[None, :, None, None]
    psum = raw.new_zeros(n, n_parcels, d_raw, d)
    psum.index_add_(1, pe, contrib)
    pcount = raw.new_zeros(n_parcels)
    pcount.index_add_(0, pe, valid)
    present = pcount > 0
    pooled = psum / pcount.clamp(min=1.0)[None, :, None, None]
    return pooled, present


def raw_ws_cs_auroc(
    raw: dict[int, Tensor],
    sd: dict[int, tp.Any],
    *,
    ws_subjects: Sequence[int],
    cs_anchor: int,
    cs_test_subjects: Sequence[int],
    tasks: Sequence[str],
    n_parcels: int,
    max_iter: int = 10000,
    tap: str = "raw",
    estimator: str = "logistic",
    lam_mult: float = 1.0,
) -> dict[str, float]:
    """WS (per-electrode) + CS (parcel-pool) AUROC over all tasks.

    ``raw[s]`` is subject ``s``'s ``(N,C,D,1)`` per-electrode feature bins; ``sd[s]``
    carries ``parcel_per_electrode`` / ``electrode_mask`` / ``labels``. ``tap`` names
    the metric family (``raw`` for the input floor; reused by the bench with
    ``tap="frontend"`` on the frontend tap reshaped to ``(N,C,S·d,1)``). ``lam_mult``
    scales the ridge λ (for applying a swept λ; ridge-only). Emits
    ``val_probe/{tap}/{ws,cs,gap}/{task}`` so the floor and the trained taps share the
    metric namespace and drop straight onto a run's ``val_probe/...`` lines."""
    a = cs_anchor
    ws_fn, cs_fn = auroc_estimators(estimator, max_iter, lam_mult)
    pooled_cs: dict[int, tuple[Tensor, Tensor]] = {
        s: pool_electrodes_to_parcels(
            raw[s], sd[s].parcel_per_electrode, sd[s].electrode_mask, n_parcels
        )
        for s in {a, *cs_test_subjects}
    }

    metrics: dict[str, float] = {}
    for task in tasks:
        ws_vals: list[float] = []
        for s in ws_subjects:
            z = per_electrode_features(raw[s], sd[s].electrode_mask).numpy()
            zf, yf = _finite_rows(z, sd[s].labels[task])
            ws_vals.append(ws_fn(zf, yf) if len(yf) >= 4 else float("nan"))
        ws_mean = float(np.nanmean(ws_vals)) if ws_vals else float("nan")

        pa, pres_a = pooled_cs[a]
        cs_vals: list[float] = []
        for t in cs_test_subjects:
            pt, pres_t = pooled_cs[t]
            inter = parcel_intersection(pres_a, pres_t)
            if inter.numel() == 0:
                cs_vals.append(float("nan"))
                continue
            za, ya = _finite_rows(feature_matrix(pa, inter).numpy(), sd[a].labels[task])
            zt, yt = _finite_rows(feature_matrix(pt, inter).numpy(), sd[t].labels[task])
            if len(ya) < 2 or len(yt) < 1:
                cs_vals.append(float("nan"))
                continue
            cs_vals.append(cs_fn(za, ya, zt, yt))
        cs_mean = float(np.nanmean(cs_vals)) if cs_vals else float("nan")

        metrics[f"val_probe/{tap}/ws/{task}"] = ws_mean
        metrics[f"val_probe/{tap}/cs/{task}"] = cs_mean
        metrics[f"val_probe/{tap}/gap/{task}"] = ws_mean - cs_mean
    return metrics


def run_v2_raw_baseline(
    dataset: tp.Any, *, max_iter: int = 10000, estimator: str = "ridge",
    lam_mult: float = 1.0,
) -> dict[str, float]:
    """Raw-feature floor over the dev probe dataset — two representations.

    ``dataset`` is the v2 probe dataset: ``ws_subjects`` / ``cs_anchor`` /
    ``cs_test_subjects`` / ``tasks`` / ``n_parcels`` and ``subject_data(s)`` whose
    result carries ``.bands`` (the list of magnitude band tensors the encoder is
    fed) plus ``.parcel_per_electrode`` / ``.electrode_mask`` / ``.labels``. Pure
    CPU tensor work — no model, no forward.

    Emits **two** floors with the same protocol/estimator (default ``"ridge"`` — the
    dimension-robust dual ridge that the encoder-tap bench scores with, so floor and
    taps share one meter):
      - ``val_probe/raw/...``     — every ``|STFT|`` bin (full ``D_raw``, no pooling).
      - ``val_probe/raw_tok/...`` — the frontend's 22-token mean-pool grid (the
        apples-to-apples floor for the *pooled* encoder taps).

    Band patch specs (``freq_patch_bins`` + ``kernel_time``) default to ``BANDS_V2``
    (clip-independent); a ``dataset.band_specs`` attribute overrides them (used by the
    laptop test, whose synthetic bands don't match the real ladder)."""
    needed = sorted({dataset.cs_anchor, *dataset.ws_subjects, *dataset.cs_test_subjects})
    sd = {s: dataset.subject_data(s) for s in needed}
    specs = getattr(dataset, "band_specs", None)
    if specs is None:
        from speech_decoding.models.v14_converged_v2 import BANDS_V2
        specs = BANDS_V2
    raw = {s: raw_bins_from_bands(sd[s].bands) for s in needed}
    tok = {s: raw_pooled_tokens_from_bands(sd[s].bands, specs) for s in needed}
    common = dict(
        ws_subjects=dataset.ws_subjects,
        cs_anchor=dataset.cs_anchor,
        cs_test_subjects=dataset.cs_test_subjects,
        tasks=dataset.tasks,
        n_parcels=dataset.n_parcels,
        max_iter=max_iter,
        estimator=estimator,
        lam_mult=lam_mult,
    )
    metrics = raw_ws_cs_auroc(raw, sd, tap="raw", **common)
    metrics.update(raw_ws_cs_auroc(tok, sd, tap="raw_tok", **common))
    return metrics
