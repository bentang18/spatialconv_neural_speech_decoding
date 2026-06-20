"""Raw-feature linear-probe baseline (Ben 2026-06-20).

The question: does the trained encoder build a representation that beats a linear
read of the **raw input the model sees**? This computes that floor at parity with
the UPSTREAM benchmark — same window set, same labels, same StandardScaler + L2
**LogisticRegression** (C=1.0) as ``probe_v14_frozen_logistic`` (the "logistic-on-
Multi-STFT 0.663" line), same WS contiguous-2-fold / CS sub2-anchor AUROC — but
replaces the encoder tokens with the raw 3STFT ``|STFT|`` magnitudes, patch-averaged
onto the SAME tokenizer grid (``ThreeBandTokenizer``): **30 tokens/electrode, d=1**
(slow 6 + beta 8 + HG 16 at 1 s), no time pooling.

Two deliberate departures from :func:`online_probe.run_probe`, per Ben:
  - **WS (within-session) keeps electrodes un-pooled.** Within a session the
    montage is fixed, so the probe sees every electrode's 30 tokens directly
    (``C·30`` features) — no DKT parcel averaging, maximal spatial resolution.
  - **CS (cross-subject) pools electrodes → DKT parcels** and fits/scores on the
    sub2-anchor ∩ eval-subject parcel intersection (the only coordinate shared
    across montages), keeping all 30 tokens.

Pure/numpy + torch tensor ops, no model: the baseline needs no forward pass. The
logistic/AUROC/fold primitives are imported from :mod:`linear_probe_logistic`
(upstream-parity, already TDD'd) and the feature/intersection helpers from
:mod:`online_probe`, so this module only adds the raw-token construction and the
un-pooled WS path.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from speech_decoding.experiments.linear_probe_logistic import (
    cs_auroc_logistic,
    ws_auroc_2fold_logistic,
)
from speech_decoding.experiments.online_probe import (
    ProbeDataset,
    SubjectProbeData,
    _finite_rows,
    feature_matrix,
    parcel_intersection,
)
from speech_decoding.models.v14_converged import BandSpec, bands_for_clip_len

__all__ = [
    "raw_tokens_from_bands",
    "pool_electrodes_to_parcels",
    "feature_matrix_per_electrode",
    "run_raw_baseline",
]


def _band_magnitude(x: Tensor, in_channels: int) -> Tensor:
    """Per-electrode ``|STFT|`` ``(B,C,F,T)`` from a delivered band tensor.

    The 1-channel bands (beta/HG) arrive as magnitude already ``(B,C,F,T)``. The
    slow band arrives Cartesian ``(B,C,2,F,T)`` (Re,Im) — collapse to a d=1 raw
    scalar with ``sqrt(Re²+Im²)`` so every band contributes one magnitude per
    (freq,time) cell (the probe is d=1; phase is the model's to exploit, not the
    raw-feature floor's)."""
    if in_channels == 2:
        if x.ndim != 5 or x.shape[2] != 2:
            raise ValueError(f"slow band must be (B,C,2,F,T); got {tuple(x.shape)}")
        return torch.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)
    if x.ndim != 4:
        raise ValueError(f"magnitude band must be (B,C,F,T); got {tuple(x.shape)}")
    return x


def _patch_pool(mag: Tensor, b: BandSpec) -> Tensor:
    """Average ``|STFT|`` ``(B,C,F,T)`` over the band's non-overlapping
    ``kernel_freq × kernel_time`` patches → ``(B,C, F_p·T_p)`` in tokenizer order
    (freq-patch-major, time-minor — matches ``token_metadata``). Mirrors the
    ``_PatchStem`` Conv2d grid (stride == kernel) with a mean instead of a learned
    embedding, so the raw tokens align 1:1 with the encoder's."""
    B, C, F, T = mag.shape
    if F != b.n_freq_bins or T != b.n_time_frames:
        raise ValueError(
            f"band {b.name!r} expected ({b.n_freq_bins},{b.n_time_frames}) bins/frames, "
            f"got ({F},{T})"
        )
    pooled = torch.nn.functional.avg_pool2d(
        mag.reshape(B * C, 1, F, T),
        kernel_size=(b.kernel_freq, b.kernel_time),
        stride=(b.kernel_freq, b.kernel_time),
    )  # (B*C, 1, F_p, T_p)
    F_p, T_p = pooled.shape[-2], pooled.shape[-1]
    if F_p != b.n_freq_patches or T_p != b.n_time_patches:
        raise ValueError(
            f"band {b.name!r} patch grid ({F_p},{T_p}) != expected "
            f"({b.n_freq_patches},{b.n_time_patches})"
        )
    # row-major reshape == freq-patch-major, time-minor (token_metadata order).
    return pooled.reshape(B, C, F_p * T_p)


def raw_tokens_from_bands(
    slow: Tensor, beta: Tensor, hg: Tensor, clip_len_s: float = 1.0
) -> Tensor:
    """Raw ``|STFT|`` tokens ``(B, C, N_TOKENS, 1)`` on the tokenizer grid.

    ``N_TOKENS`` = Σ band freq-patches·time-patches (30 at 1 s: slow 6 + beta 8 +
    HG 16), concatenated band slow→beta→HG, each freq-patch-major/time-minor — the
    exact order of ``token_metadata`` so a raw token and its encoder token share a
    ``(freq_global_id, time_slot)``. Trailing ``d=1`` so the existing pooling /
    ``feature_matrix`` consume it unchanged."""
    bands = bands_for_clip_len(clip_len_s)
    mags = (
        _band_magnitude(slow, bands[0].in_channels),
        _band_magnitude(beta, bands[1].in_channels),
        _band_magnitude(hg, bands[2].in_channels),
    )
    toks = [_patch_pool(m, b) for m, b in zip(mags, bands)]  # each (B,C,n_tok_b)
    return torch.cat(toks, dim=2).unsqueeze(-1)              # (B,C,N,1)


def feature_matrix_per_electrode(raw_tokens: Tensor, electrode_mask: Tensor) -> Tensor:
    """WS feature matrix: keep every valid electrode's tokens, flatten → ``(B, C_valid·N)``.

    No parcel pooling (within-session the montage is fixed; Ben 2026-06-20). Column
    order is electrode-major then token, identical across the subject's windows."""
    valid = electrode_mask.bool()
    sel = raw_tokens[:, valid]                               # (B, C_valid, N, 1)
    return sel.reshape(sel.shape[0], -1)


def pool_electrodes_to_parcels(
    raw_tokens: Tensor,
    parcel_per_electrode: Tensor,
    electrode_mask: Tensor,
    n_parcels: int,
) -> tuple[Tensor, Tensor]:
    """CS pooling: electrode → DKT-parcel mean, keeping all ``N`` tokens →
    ``(B, n_parcels, N, 1)`` + ``present`` ``(n_parcels,)``.

    The ONLY reduction (no freq/time grouping — all 30 tokens survive). Empty
    parcels are zero and ``present=False`` so :func:`parcel_intersection` can drop
    them. Mirrors the electrode→parcel half of ``pool_window_features``."""
    B, C, N, d = raw_tokens.shape
    if C != int(parcel_per_electrode.shape[0]) or C != int(electrode_mask.shape[0]):
        raise ValueError("parcel_per_electrode / electrode_mask must be length C")
    valid = electrode_mask.to(raw_tokens.dtype)
    pe = parcel_per_electrode.long()
    contrib = raw_tokens * valid[None, :, None, None]
    psum = raw_tokens.new_zeros(B, n_parcels, N, d)
    psum.index_add_(1, pe, contrib)
    pcount = raw_tokens.new_zeros(n_parcels)
    pcount.index_add_(0, pe, valid)
    present = pcount > 0
    pooled = psum / pcount.clamp(min=1.0)[None, :, None, None]
    return pooled, present


def run_raw_baseline(
    dataset: ProbeDataset, *, clip_len_s: float = 1.0
) -> dict[str, float]:
    """Raw-feature probe over all tasks (WS per-electrode + CS parcel-pooled).

    Returns a flat ``{val_probe/raw/{ws,cs,gap}/{task}: auroc}`` dict, the same
    naming as :func:`online_probe.run_probe` (tap = ``raw``) so a trained run's
    ``val_probe/latent/ws/<task>`` lines drop straight onto these baselines."""
    tasks = list(dataset.tasks)
    n_parcels = dataset.n_parcels

    # Raw tokens + labels per needed subject, computed once.
    needed = {dataset.cs_anchor, *dataset.ws_subjects, *dataset.cs_test_subjects}
    rt: dict[int, Tensor] = {}
    sd: dict[int, SubjectProbeData] = {}
    for s in needed:
        d = dataset.subject_data(s)
        sd[s] = d
        rt[s] = raw_tokens_from_bands(d.slow, d.beta, d.hg, clip_len_s=clip_len_s)

    # CS: pre-pool the anchor and each test subject to parcels once.
    a = dataset.cs_anchor
    pooled_cs: dict[int, tuple[Tensor, Tensor]] = {}
    for s in {a, *dataset.cs_test_subjects}:
        pooled_cs[s] = pool_electrodes_to_parcels(
            rt[s], sd[s].parcel_per_electrode, sd[s].electrode_mask, n_parcels
        )

    metrics: dict[str, float] = {}
    for task in tasks:
        # WS — per-electrode, contiguous 2-fold, mean over subjects.
        ws_vals: list[float] = []
        for s in dataset.ws_subjects:
            z = feature_matrix_per_electrode(rt[s], sd[s].electrode_mask).numpy()
            zf, yf = _finite_rows(z, sd[s].labels[task])
            ws_vals.append(ws_auroc_2fold_logistic(zf, yf) if len(yf) >= 4 else float("nan"))
        ws_mean = float(np.nanmean(ws_vals)) if ws_vals else float("nan")

        # CS — sub2 anchor → each test subject over the parcel intersection.
        pa, pres_a = pooled_cs[a]
        cs_vals: list[float] = []
        for t in dataset.cs_test_subjects:
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
            cs_vals.append(cs_auroc_logistic(za, ya, zt, yt))
        cs_mean = float(np.nanmean(cs_vals)) if cs_vals else float("nan")

        metrics[f"val_probe/raw/ws/{task}"] = ws_mean
        metrics[f"val_probe/raw/cs/{task}"] = cs_mean
        metrics[f"val_probe/raw/gap/{task}"] = ws_mean - cs_mean
    return metrics
