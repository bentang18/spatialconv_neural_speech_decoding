"""Online frozen-feature linear probe for v14 converged-arch SSL pretraining.

Spec: ``reports/online_probe_spec_2026_06_18.md`` (DESIGN LOCKED). A diagnostic
*callback*, never a model change. Every ``cadence`` steps, on rank 0, it freezes
the in-training encoder, forwards a cached set of word-onset windows through BOTH
taps (``encode_frontend`` = electrode-isolated, ``encode_latent`` = parcel-tagged
cross-electrode), pools to a per-window feature vector, and fits a dual-form
ridge head for three word tasks under two protocols:

- **WS** (within-session): per subject, contiguous 2-fold (``KFold shuffle=False``)
  — the decodability floor.
- **CS** (cross-subject): sub2-anchored, train-on-anchor / score-on-test over the
  pair's parcel intersection — the zero-per-subject transfer goal.

The headline is **AUROC** and the load-bearing trend is the **WS − CS gap**
closing over training (§0). This module holds the PURE, unit-testable core
(pooling + dual ridge + AUROC + folds + intersection + firewall) and the
``OnlineLinearProbe`` callback. The probe DATASET builder (band tensors + labels
from BT) lives in :mod:`speech_decoding.experiments.online_probe_dataset`.

Everything here is numpy/torch only — no BT, no DCC — so the decision logic is
TDD-checked with synthetic features (``test_online_probe.py``). The callback is
**OFF by default** (``enabled=False``): registering it cannot perturb a live run.
"""

from __future__ import annotations

import dataclasses
import typing as tp

import numpy as np
import torch
from torch import Tensor

# --------------------------------------------------------------------------- pooling


def time_bin_of_slot(time_slot: Tensor, k: int) -> Tensor:
    """Map each token's RoPE ``time_slot`` to a physical-time bin in ``[0, k)``.

    ``time_slot`` is the shared HG-stride clock (×8 slow / ×2 beta / ×1 HG); its
    span ``n_slots = max+1`` covers the whole 1 s clip, so binning by
    ``(slot·k)//n_slots`` is a physical-time split — NOT a per-band patch split.
    For ``k=2`` this is ``slot // 8`` (early/late halves at 0.5 s); for ``k=1`` all
    tokens fall in bin 0. Patch counts are read off the clock, never hardcoded
    (spec §4)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n_slots = int(time_slot.max().item()) + 1
    return torch.clamp((time_slot * k) // n_slots, max=k - 1)


def pool_window_features(
    feats: Tensor,
    freq_global_id: Tensor,
    time_slot: Tensor,
    parcel_per_electrode: Tensor,
    electrode_mask: Tensor,
    n_parcels: int,
    k: int,
    n_freq: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Pool ``(B,C,N,d)`` encoder tokens → ``(B, n_parcels, n_freq, k, d)`` (spec §4).

    Three reductions, all means: tokens sharing a ``(freq_global_id, time_bin)``
    group are averaged (keeps all ``n_freq`` freq-patches, bins time into ``k``
    physical bins); then electrodes are averaged into their parcel
    (``parcel_per_electrode``), counting only ``electrode_mask`` electrodes.
    Empty parcels are zero and flagged ``present=False`` so the fit can drop /
    intersect them. ``n_freq`` defaults to ``max(freq_global_id)+1``."""
    if feats.ndim != 4:
        raise ValueError(f"feats must be (B,C,N,d), got {tuple(feats.shape)}")
    b, c, _n, d = feats.shape
    if n_freq is None:
        n_freq = int(freq_global_id.max().item()) + 1
    if c != int(parcel_per_electrode.shape[0]) or c != int(electrode_mask.shape[0]):
        raise ValueError("parcel_per_electrode / electrode_mask must be length C")

    tb = time_bin_of_slot(time_slot, k)                 # (N,)
    group = freq_global_id.long() * k + tb              # (N,) in [0, n_freq*k)
    g = n_freq * k

    # token -> (freq, time-bin) group mean: (B, C, G, d)
    gsum = feats.new_zeros(b, c, g, d)
    gsum.index_add_(2, group, feats)
    gcount = feats.new_zeros(g)
    gcount.index_add_(0, group, torch.ones_like(group, dtype=feats.dtype))
    tok = gsum / gcount.clamp(min=1.0)[None, None, :, None]

    # electrode -> parcel mean, masked: (B, n_parcels, G, d)
    valid = electrode_mask.to(feats.dtype)              # (C,)
    pe = parcel_per_electrode.long()
    contrib = tok * valid[None, :, None, None]
    psum = feats.new_zeros(b, n_parcels, g, d)
    psum.index_add_(1, pe, contrib)
    pcount = feats.new_zeros(n_parcels)
    pcount.index_add_(0, pe, valid)
    present = pcount > 0
    pooled = psum / pcount.clamp(min=1.0)[None, :, None, None]
    return pooled.reshape(b, n_parcels, n_freq, k, d), present


def feature_matrix(pooled: Tensor, parcels: Tensor | tp.Sequence[int]) -> Tensor:
    """Select ``parcels`` from ``(B, n_parcels, F, k, d)`` and flatten → ``(B, P·F·k·d)``."""
    idx = torch.as_tensor(list(parcels), dtype=torch.long, device=pooled.device)
    sel = pooled.index_select(1, idx)
    return sel.reshape(sel.shape[0], -1)


def parcel_intersection(present_a: Tensor, present_b: Tensor) -> Tensor:
    """Parcel ids present in BOTH subjects (spec §5 CS) as a sorted long tensor."""
    inter = present_a.bool() & present_b.bool()
    return torch.nonzero(inter, as_tuple=False).flatten()


# ------------------------------------------------------------------------- dual ridge


def dual_ridge_scores(
    z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, lam: float | None = None
) -> np.ndarray:
    """Dual (kernel) ridge (spec §5). ``G = ZZᵀ`` (N×N), ``α = (G+λI)⁻¹y``,
    ``scores = G_test,train α``. ``lam`` defaults to ``trace(G)/N`` (fixed, no
    per-step tuning). ``d`` enters only through ``G`` so the cost is N²·d — the
    feature dim is free, which is why dual form was chosen."""
    z_train = np.asarray(z_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    z_test = np.asarray(z_test, dtype=np.float64)
    g = z_train @ z_train.T                             # (n, n)
    n = g.shape[0]
    if lam is None:
        lam = float(np.trace(g) / max(n, 1))
    alpha = np.linalg.solve(g + lam * np.eye(n), y_train)
    return (z_test @ z_train.T) @ alpha


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC of continuous ``scores`` vs ±1 ``labels`` (spec §5). NaN if a fold
    is single-class (AUROC undefined) — callers nan-mean over folds/subjects."""
    from sklearn.metrics import roc_auc_score

    y = (np.asarray(labels) > 0).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores)))


def contiguous_folds(n: int, k: int = 2) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous (time-ordered) ``KFold(n_splits=k, shuffle=False)`` index splits
    (spec §5 WS) — test blocks are contiguous stretches, NOT interleaved, so a
    train/test pair never straddles the same autocorrelated stretch (upstream
    ``train_test_splits.py:220`` parity). Matches ``sklearn`` fold sizes: the first
    ``n % k`` folds get one extra sample."""
    idx = np.arange(n)
    sizes = np.full(k, n // k, dtype=int)
    sizes[: n % k] += 1
    out: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for s in sizes:
        test = idx[start : start + s]
        train = np.concatenate([idx[:start], idx[start + s :]])
        out.append((train, test))
        start += s
    return out


def ws_auroc_2fold(z: np.ndarray, y: np.ndarray, k_folds: int = 2) -> float:
    """Within-session AUROC: mean over contiguous folds (spec §5). Fit dual ridge
    on each fold's train half, score the held-out half, nan-mean the per-fold
    AUROCs."""
    scores: list[float] = []
    for train, test in contiguous_folds(len(y), k_folds):
        if len(train) == 0 or len(test) == 0:
            continue
        pred = dual_ridge_scores(z[train], y[train], z[test])
        scores.append(auroc(pred, y[test]))
    return float(np.nanmean(scores)) if scores else float("nan")


def cs_auroc(
    z_anchor: np.ndarray, y_anchor: np.ndarray, z_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Cross-subject AUROC: fit dual ridge on the anchor (sub2), score the test
    subject's windows (spec §5). Caller restricts ``z_*`` to the pair's parcel
    intersection beforehand."""
    pred = dual_ridge_scores(z_anchor, y_anchor, z_test)
    return auroc(pred, y_test)


# --------------------------------------------------------------------------- firewall


def assert_firewall(
    sessions: tp.Iterable[tuple[int, int]], lite_sessions: tp.Iterable[tuple[int, int]]
) -> None:
    """Hard fail if any probe ``(subject, trial)`` is a lite/leaderboard eval cell
    (spec §6). The online CS number must stay a pretrain-sourced PROXY; the honest
    number comes only from the firewalled lite eval."""
    lite = {tuple(s) for s in lite_sessions}
    leaked = sorted({tuple(s) for s in sessions} & lite)
    if leaked:
        raise AssertionError(
            f"online-probe firewall breach: probe sessions overlap BT_LITE_SESSIONS "
            f"{leaked}; the probe must never touch leaderboard eval cells"
        )


# ----------------------------------------------------------------------- probe data


@dataclasses.dataclass
class SubjectProbeData:
    """One subject's cached, encoder-ready probe inputs (spec §2).

    ``slow/beta/hg`` are the 1 s 3STFT band tensors ``(N,C,2,6,5)/(N,C,6,17)/
    (N,C,9,33)``; ``parcel_per_electrode`` / ``electrode_mask`` are ``(C,)`` and
    constant across the subject's windows; ``labels`` maps task → ``(N,)`` ±1;
    ``sessions`` is the set of ``(subject,trial)`` the windows came from (firewall)."""

    subject_id: int
    slow: Tensor
    beta: Tensor
    hg: Tensor
    parcel_per_electrode: Tensor
    electrode_mask: Tensor
    labels: dict[str, np.ndarray]
    sessions: list[tuple[int, int]]


class ProbeDataset(tp.Protocol):
    """Minimal interface the callback consumes (built once, cached, deterministic)."""

    ws_subjects: tp.Sequence[int]
    cs_anchor: int
    cs_test_subjects: tp.Sequence[int]
    tasks: tp.Sequence[str]
    n_parcels: int

    def subject_data(self, subject_id: int) -> SubjectProbeData: ...


# --------------------------------------------------------------------------- callback


def _encode_and_pool(
    model: tp.Any,
    sd: SubjectProbeData,
    k_list: tp.Sequence[int],
    n_parcels: int,
    *,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> dict[str, dict[int, tuple[Tensor, Tensor]]]:
    """Forward one subject's windows through BOTH taps (unmasked, no_grad, batched)
    and pool to ``(N, n_parcels, F, k, d)`` per ``(tap, k)``. Returns
    ``{tap: {k: (pooled, present)}}``. One forward per tap feeds every ``k`` (spec
    §3 cost note: WS and CS reuse the same forwards)."""
    tok = model.student_frontend.tokenizer
    freq_id, time_slot = tok.freq_global_id, tok.time_slot
    dev = device or freq_id.device
    slow, beta, hg = sd.slow.to(dev), sd.beta.to(dev), sd.hg.to(dev)
    ppe = sd.parcel_per_electrode.to(dev)
    emask = sd.electrode_mask.to(dev)

    front_chunks: list[Tensor] = []
    lat_chunks: list[Tensor] = []
    n = slow.shape[0]
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        front_chunks.append(model.encode_frontend(slow[sl], beta[sl], hg[sl]))
        lat_chunks.append(
            model.encode_latent(slow[sl], beta[sl], hg[sl], ppe, electrode_mask=emask)
        )
    feats = {"frontend": torch.cat(front_chunks), "latent": torch.cat(lat_chunks)}

    out: dict[str, dict[int, tuple[Tensor, Tensor]]] = {}
    for tap, f in feats.items():
        out[tap] = {}
        for k in k_list:
            pooled, present = pool_window_features(
                f, freq_id, time_slot, ppe, emask, n_parcels, k
            )
            out[tap][k] = (pooled.float().cpu(), present.cpu())
    return out


def run_probe(
    model: tp.Any, dataset: ProbeDataset, k_list: tp.Sequence[int], *, batch_size: int = 256
) -> dict[str, float]:
    """Compute the full probe (all taps × k × tasks, WS + CS) and return a flat
    ``{metric_name: value}`` dict ready for ``self.log`` (spec §7). Pure given a
    frozen ``model`` + ``dataset`` — the callback only wraps eval()/no_grad/gating
    around it. Names: ``val_probe/{tap}/{ws|cs|gap}/{task}`` (+``_k{1}`` suffix for
    the non-primary k)."""
    taps = ("frontend", "latent")
    k_primary = k_list[0]
    n_parcels = dataset.n_parcels

    # forward + pool every needed subject once (anchor + WS + CS test subjects)
    needed = {dataset.cs_anchor, *dataset.ws_subjects, *dataset.cs_test_subjects}
    pooled: dict[int, dict[str, dict[int, tuple[Tensor, Tensor]]]] = {}
    labels: dict[int, dict[str, np.ndarray]] = {}
    for s in needed:
        sd = dataset.subject_data(s)
        pooled[s] = _encode_and_pool(model, sd, k_list, n_parcels, batch_size=batch_size)
        labels[s] = sd.labels

    metrics: dict[str, float] = {}
    for tap in taps:
        for k in k_list:
            sfx = "" if k == k_primary else f"_k{k}"
            for task in dataset.tasks:
                # WS: per-subject contiguous 2-fold, mean over subjects
                ws_vals: list[float] = []
                for s in dataset.ws_subjects:
                    p, present = pooled[s][tap][k]
                    z = feature_matrix(p, parcel_intersection(present, present)).numpy()
                    ws_vals.append(ws_auroc_2fold(z, labels[s][task]))
                ws_mean = float(np.nanmean(ws_vals)) if ws_vals else float("nan")

                # CS: sub2 anchor -> each test subject over intersection parcels
                a = dataset.cs_anchor
                pa, pres_a = pooled[a][tap][k]
                cs_vals: list[float] = []
                for t in dataset.cs_test_subjects:
                    pt, pres_t = pooled[t][tap][k]
                    inter = parcel_intersection(pres_a, pres_t)
                    if inter.numel() == 0:
                        cs_vals.append(float("nan"))
                        continue
                    za = feature_matrix(pa, inter).numpy()
                    zt = feature_matrix(pt, inter).numpy()
                    cs_vals.append(cs_auroc(za, labels[a][task], zt, labels[t][task]))
                cs_mean = float(np.nanmean(cs_vals)) if cs_vals else float("nan")

                metrics[f"val_probe/{tap}/ws/{task}{sfx}"] = ws_mean
                metrics[f"val_probe/{tap}/cs/{task}{sfx}"] = cs_mean
                metrics[f"val_probe/{tap}/gap/{task}{sfx}"] = ws_mean - cs_mean
    return metrics


try:  # lightning is a runtime dep on DCC; keep import soft for laptop unit tests
    import lightning.pytorch as pl

    _CallbackBase = pl.Callback
except Exception:  # pragma: no cover - exercised only where lightning is absent
    _CallbackBase = object


class OnlineLinearProbe(_CallbackBase):  # type: ignore[misc,valid-type]
    """Lightning callback: frozen linear probe every ``cadence`` steps (spec §1).

    **OFF by default** (``enabled=False``) — registering it is inert until a run
    explicitly turns it on, so it can never perturb a live training run. Rank-0
    only, no ``sync_dist`` / ``all_gather`` (no DDP deadlock, spec §8). Restores
    ``model.train()`` afterwards."""

    def __init__(
        self,
        dataset: ProbeDataset | None = None,
        *,
        enabled: bool = False,
        cadence: int = 1000,
        k_list: tp.Sequence[int] = (2, 1),
        batch_size: int = 256,
    ) -> None:
        self.dataset = dataset
        self.enabled = enabled
        self.cadence = cadence
        self.k_list = tuple(k_list)
        self.batch_size = batch_size

    def _should_fire(self, trainer: tp.Any) -> bool:
        if not self.enabled or self.dataset is None:
            return False
        if getattr(trainer, "global_rank", 0) != 0:
            return False
        step = int(getattr(trainer, "global_step", 0))
        return step > 0 and step % self.cadence == 0

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs) -> None:  # noqa: ANN001
        if not self._should_fire(trainer):
            return
        self.fire(trainer, pl_module)

    def fire(self, trainer: tp.Any, pl_module: tp.Any) -> dict[str, float]:
        """Run one probe and log it. Separated from the gate so tests can drive it
        directly. Freezes the model for the probe, restores train() after."""
        assert self.dataset is not None, "OnlineLinearProbe.fire requires a dataset"
        model = getattr(pl_module, "model", pl_module)
        was_training = bool(getattr(model, "training", True))
        model.eval()
        try:
            with torch.no_grad():
                metrics = run_probe(
                    model, self.dataset, self.k_list, batch_size=self.batch_size
                )
        finally:
            if was_training:
                model.train()
        for name, value in metrics.items():
            if np.isfinite(value):
                pl_module.log(name, value, rank_zero_only=True, sync_dist=False)
        return metrics


__all__ = [
    "OnlineLinearProbe",
    "ProbeDataset",
    "SubjectProbeData",
    "assert_firewall",
    "auroc",
    "contiguous_folds",
    "cs_auroc",
    "dual_ridge_scores",
    "feature_matrix",
    "parcel_intersection",
    "pool_window_features",
    "run_probe",
    "time_bin_of_slot",
    "ws_auroc_2fold",
]
