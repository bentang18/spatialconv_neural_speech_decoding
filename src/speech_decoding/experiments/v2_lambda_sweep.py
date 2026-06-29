"""λ-sweep harness for the v2 probe bench (Run A eval protocol).

Protocol (LOCKED, ``project_probe_bench_keepS_ridge_lock_2026_06_28``):
  - tap = latent keep-S (per-parcel ``k·S·d`` cells kept), dual (kernel) ridge.
  - **Sweep λ on the PRETRAIN dev probe eval only.** The selected multiplier ``m*``
    is then APPLIED to the benchmark (Neuroprobe-Lite) eval via
    ``run_v2_probe_bench(..., lam_mult=m*)`` — so λ never peeks at the benchmark and
    the leaderboard-parity lock is preserved (no test-set tuning).
  - λ = ``mult · trace(G)/N`` — the scale-aware dual-ridge default
    (:func:`online_probe.dual_ridge_scores`); ``mult=1`` = the current fixed rule.

Efficiency: for a fixed ``(z_train, z_test)`` the only λ-dependence is the
``(G+λI)⁻¹`` inverse, so eigendecompose ``G = V diag(σ) Vᵀ`` ONCE and reuse it for
the whole grid — ``α = V (Vᵀy / (σ+λ))``, an O(n²) matvec per λ instead of an
O(n³) solve each. At ``m=1`` this is byte-equivalent to
:func:`online_probe.dual_ridge_scores` (TDD-checked).

Parity-honest selection: :func:`select_lam_mult` argmaxes the pretrain CS curve over
the full test-subject pool to pick the ``m*`` applied to the benchmark (it never sees
the benchmark). :func:`loso_cs_auroc` then reports what that *rule* yields on
held-out pretrain subjects (leave-one-test-subject-out), so the reported pretrain CS
number is not self-selected-optimistic.

The pure cores (everything but the model-side driver) are numpy/torch only and are
TDD-checked with synthetic features in ``test_v2_lambda_sweep.py``.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import torch
from torch import Tensor

from speech_decoding.experiments.online_probe import (
    _finite_rows,
    auroc,
    contiguous_folds,
    feature_matrix,
    parcel_intersection,
)

__all__ = [
    "DEFAULT_LAM_MULTS",
    "ridge_lambda_scores",
    "ridge_lambda_curve",
    "ws_lambda_curve",
    "select_lam_mult",
    "sweep_cs_task",
    "loso_cs_auroc",
    "run_latent_keepS_lambda_sweep",
]

# Geometric grid centred on m=1 (the current fixed rule): 1e-3 … 1e3, 13 points.
DEFAULT_LAM_MULTS: tuple[float, ...] = tuple(float(x) for x in np.logspace(-3, 3, 13))


def ridge_lambda_scores(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    lam_mults: tp.Sequence[float] = DEFAULT_LAM_MULTS,
) -> np.ndarray:
    """Dual-ridge test scores for EVERY ``lam_mult``, eigendecomposing ``G`` once.

    ``G = ZᵀZ``'s gram ``Z Zᵀ`` (n×n) is symmetric PSD; with ``G = V diag(σ) Vᵀ`` the
    ridge solution is ``α = (G+λI)⁻¹y = V (Vᵀy / (σ+λ))`` for any λ, so the
    eigendecomposition and the two heavy products (``Vᵀy`` and ``K_test V``) are shared
    across the grid. ``λ = mult · trace(G)/N`` matches
    :func:`online_probe.dual_ridge_scores`. Returns ``(len(lam_mults), n_test)``."""
    z_train = np.asarray(z_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    z_test = np.asarray(z_test, dtype=np.float64)
    g = z_train @ z_train.T                         # (n, n)
    n = g.shape[0]
    base = float(np.trace(g) / max(n, 1))
    sigma, vecs = np.linalg.eigh(g)                 # σ ascending, G symmetric PSD
    vt_y = vecs.T @ y_train                          # (n,)
    ktest_v = (z_test @ z_train.T) @ vecs            # (n_test, n)
    out = np.empty((len(lam_mults), z_test.shape[0]), dtype=np.float64)
    for i, m in enumerate(lam_mults):
        lam = m * base
        out[i] = ktest_v @ (vt_y / (sigma + lam))
    return out


def ridge_lambda_curve(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    lam_mults: tp.Sequence[float] = DEFAULT_LAM_MULTS,
) -> np.ndarray:
    """AUROC vs ``lam_mults`` for an anchor→test fit (the CS unit). ``(len(lam_mults),)``."""
    scores = ridge_lambda_scores(z_train, y_train, z_test, lam_mults)
    return np.array([auroc(scores[i], y_test) for i in range(scores.shape[0])])


def ws_lambda_curve(
    z: np.ndarray,
    y: np.ndarray,
    lam_mults: tp.Sequence[float] = DEFAULT_LAM_MULTS,
    k_folds: int = 2,
) -> np.ndarray:
    """Within-session AUROC vs ``lam_mults``: mean over contiguous folds. Each fold
    eigendecomposes its own train gram once and sweeps the grid. ``(len(lam_mults),)``."""
    per_fold: list[np.ndarray] = []
    for train, test in contiguous_folds(len(y), k_folds):
        if len(train) == 0 or len(test) == 0:
            continue
        per_fold.append(ridge_lambda_curve(z[train], y[train], z[test], y[test], lam_mults))
    if not per_fold:
        return np.full(len(lam_mults), np.nan)
    return np.nanmean(np.stack(per_fold), axis=0)


def select_lam_mult(
    curve: np.ndarray, lam_mults: tp.Sequence[float], tol: float = 1e-9
) -> tuple[int, float, float] | None:
    """``(index, mult, auroc)`` of the best λ on ``curve``. Ties (within ``tol`` of the
    max) break toward the LARGEST mult — more regularization is the safer pick at
    p≫n. Returns ``None`` if the curve is all-NaN."""
    curve = np.asarray(curve, dtype=np.float64)
    if not np.isfinite(curve).any():
        return None
    best = float(np.nanmax(curve))
    cand = np.where(curve >= best - tol)[0]
    i = int(cand[int(np.argmax(np.asarray(lam_mults, dtype=np.float64)[cand]))])
    return i, float(lam_mults[i]), float(curve[i])


def sweep_cs_task(
    glob: dict[int, Tensor],
    present: dict[int, Tensor],
    label_vecs: dict[int, np.ndarray],
    anchor: int,
    test_subjects: tp.Sequence[int],
    lam_mults: tp.Sequence[float] = DEFAULT_LAM_MULTS,
) -> tuple[np.ndarray, dict[int, np.ndarray | None]]:
    """Cross-subject λ-curves for ONE task. ``glob[s]`` is ``(N_s, n_parcels, D)`` (the
    per-subject keep-S features scattered into the global DKT-id table), ``present[s]``
    the ``(n_parcels,)`` bool mask, ``label_vecs[s]`` the ``(N_s,)`` ±1/NaN task labels.

    For each test subject: fit the anchor / score the test subject over the pair's
    parcel intersection (same intersection the bench CS uses), sweeping λ. Returns
    ``(mean_curve, per_subject)`` where ``per_subject[t]`` is that subject's curve (or
    ``None`` when the pair is unusable)."""
    pa, pres_a = glob[anchor], present[anchor]
    per: dict[int, np.ndarray | None] = {}
    for t in test_subjects:
        inter = parcel_intersection(pres_a, present[t])
        if inter.numel() == 0:
            per[t] = None
            continue
        za, ya = _finite_rows(feature_matrix(pa, inter).numpy(), label_vecs[anchor])
        zt, yt = _finite_rows(feature_matrix(glob[t], inter).numpy(), label_vecs[t])
        if len(ya) < 2 or len(yt) < 1:
            per[t] = None
            continue
        per[t] = ridge_lambda_curve(za, ya, zt, yt, lam_mults)
    usable = [c for c in per.values() if c is not None]
    mean = np.nanmean(np.stack(usable), axis=0) if usable else np.full(len(lam_mults), np.nan)
    return mean, per


def loso_cs_auroc(
    per_subject: dict[int, np.ndarray | None], lam_mults: tp.Sequence[float]
) -> tuple[float, dict[int, float]]:
    """Honest held-out CS estimate of the selection RULE. Leave-one-test-subject-out:
    for each held-out subject, pick ``m*`` on the OTHERS' mean curve, then read the
    held-out subject's AUROC at that ``m*``. Returns ``(mean_auroc, picks)`` — the
    pretrain CS number you can quote without self-selection bias."""
    subs = [t for t, c in per_subject.items() if c is not None]
    if len(subs) < 2:
        return float("nan"), {}
    vals: list[float] = []
    picks: dict[int, float] = {}
    for h in subs:
        others = np.nanmean(np.stack([per_subject[t] for t in subs if t != h]), axis=0)
        sel = select_lam_mult(others, lam_mults)
        if sel is None:
            continue
        i, m, _ = sel
        picks[h] = m
        vals.append(float(per_subject[h][i]))
    return (float(np.nanmean(vals)) if vals else float("nan")), picks


def run_latent_keepS_lambda_sweep(
    dataset: tp.Any,
    model: tp.Any,
    *,
    clip_len_s: float,
    device: torch.device,
    batch_size: int = 64,
    lam_mults: tp.Sequence[float] = DEFAULT_LAM_MULTS,
) -> dict[str, tp.Any]:  # pragma: no cover - model-side; pure cores are TDD-checked
    """Encode the latent keep-S tap for the pretrain dev probe corpus, then sweep λ.

    Per task returns the CS curve (mean over test subjects), the WS curve (mean over
    WS subjects), the per-test-subject CS curves, the argmax ``m*`` to APPLY to the
    benchmark (``selected_lam_mult``), its pretrain CS AUROC, and the LOSO-honest
    pretrain CS estimate (``loso_cs_auroc``). Feed ``selected_lam_mult`` to
    ``run_v2_probe_bench(..., lam_mult=...)`` for the firewalled benchmark eval."""
    from speech_decoding.experiments.v2_probe_bench import encode_subject_taps

    needed = sorted({dataset.cs_anchor, *dataset.ws_subjects, *dataset.cs_test_subjects})
    sd = {s: dataset.subject_data(s) for s in needed}
    latent_keep: dict[int, Tensor] = {}
    labels: dict[int, Tensor] = {}
    for s in needed:
        _, _, _, lat_keep, _, lab = encode_subject_taps(
            model, sd[s].bands, sd[s].parcel_per_electrode,
            sd[s].electrode_mask, dataset.n_parcels,
            clip_len_s=clip_len_s, device=device, batch_size=batch_size,
        )
        latent_keep[s] = lat_keep
        labels[s] = lab

    n_parcels = dataset.n_parcels
    glob: dict[int, Tensor] = {}
    present: dict[int, Tensor] = {}
    for s in {dataset.cs_anchor, *dataset.cs_test_subjects}:
        lat, lab = latent_keep[s], labels[s]
        n, _, dim = lat.shape
        gtab = lat.new_zeros(n, n_parcels, dim)
        gtab[:, lab] = lat
        pmask = torch.zeros(n_parcels, dtype=torch.bool)
        pmask[lab] = True
        glob[s], present[s] = gtab, pmask

    out: dict[str, tp.Any] = {"lam_mults": list(lam_mults), "tasks": {}}
    for task in dataset.tasks:
        label_vecs = {s: sd[s].labels[task] for s in needed}
        cs_mean, cs_per = sweep_cs_task(
            glob, present, label_vecs, dataset.cs_anchor, dataset.cs_test_subjects, lam_mults
        )
        ws_curves: list[np.ndarray] = []
        for s in dataset.ws_subjects:
            z = latent_keep[s].reshape(latent_keep[s].shape[0], -1).numpy()
            zf, yf = _finite_rows(z, sd[s].labels[task])
            if len(yf) >= 4:
                ws_curves.append(ws_lambda_curve(zf, yf, lam_mults))
        ws_mean = (
            np.nanmean(np.stack(ws_curves), axis=0)
            if ws_curves else np.full(len(lam_mults), np.nan)
        )
        sel = select_lam_mult(cs_mean, lam_mults)
        loso_val, loso_picks = loso_cs_auroc(cs_per, lam_mults)
        out["tasks"][task] = {
            "cs_curve": cs_mean.tolist(),
            "ws_curve": ws_mean.tolist(),
            "cs_per_subject": {
                int(t): (c.tolist() if c is not None else None) for t, c in cs_per.items()
            },
            "selected_lam_mult": (sel[1] if sel else None),
            "selected_cs_auroc": (sel[2] if sel else None),
            "loso_cs_auroc": loso_val,
            "loso_picks": {int(k): v for k, v in loso_picks.items()},
        }
    return out
