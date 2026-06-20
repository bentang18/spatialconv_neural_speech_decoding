"""Offline probe bench — pieces 1 + 3 (Ben 2026-06-20), run on a real checkpoint.

Two questions, one loaded encoder, answered together on a 1-GPU DCC job over an
isolated clone (so the live SSL run is never touched):

**Piece 1 — online-probe wall-time.** The online linear probe (:mod:`online_probe`,
dual ridge) is a rank-0 callback that stalls all DDP GPUs for its duration. Before
turning it on in a real run we need its per-fire cost broken into **forward**
(``encode_frontend`` + ``encode_latent``), **pooling** (``pool_window_features``),
and **ridge** (the 240 dual solves: 2 taps × 2 k × 3 tasks × {7 WS·2-fold + 6 CS}).
:func:`bench_ridge_timing` reproduces ``run_probe`` with synchronised timers and
also reports the 5 k AUROC, then prices the trim levers (reuse ``G`` across the 3
tasks, single k, fewer subjects) so we can land under the < 2 min/fire budget.

**Piece 3 — encoder↔raw head-to-head.** Does the trained encoder build a
representation a linear probe reads better than the raw ``|STFT|`` the model is fed?
:func:`bench_logistic_head_to_head` scores three taps — ``raw`` (d=1), ``frontend``
(d=256, electrode-isolated), ``latent`` (d=256, parcel-tagged cross-electrode) —
through ONE protocol: per-electrode within-session (``C·30·d`` features, contiguous
2-fold) and parcel-pooled cross-subject (sub2 anchor → test over the parcel
intersection), all scored with the upstream-parity **logistic C=1.0 + StandardScaler**
(:mod:`linear_probe_logistic`), so the three taps sit on one axis and on the same
axis as the "logistic-on-Multi-STFT 0.663" line.

The model build loads the 5 s-pretrained checkpoint into a **1 s-geometry** model
(``clip_len_s=1.0``): every learned param (conv stems, attention, the
``freq_global_id``-indexed learned freq tag, parcel embedding) is clip-length
invariant in shape; only the ``persistent=False`` geometry buffers differ and are
rebuilt by the 1 s ``__init__`` — so the 5 s weights drop into a valid 1 s encoder
(``strict=False``; the only state_dict gaps are those non-persistent buffers).

The token-source-generic WS/CS logistic loop (:func:`logistic_ws_cs_from_tokens`)
is pure torch/numpy and laptop-TDD'd (``test_offline_probe_bench``); the forward,
ridge-timing, ckpt-load, and driver are DCC-only (need BT voltage + a GPU).
"""

from __future__ import annotations

import time
import typing as tp

import numpy as np
import torch
from torch import Tensor

from speech_decoding.experiments.linear_probe_logistic import (
    cs_auroc_logistic,
    ws_auroc_2fold_logistic,
)
from speech_decoding.experiments.online_probe import (
    SubjectProbeData,
    _finite_rows,
    cs_auroc,
    feature_matrix,
    parcel_intersection,
    pool_window_features,
    ws_auroc_2fold,
)
from speech_decoding.experiments.online_probe_raw_baseline import (
    feature_matrix_per_electrode,
    pool_electrodes_to_parcels,
    raw_tokens_from_bands,
)

__all__ = [
    "logistic_ws_cs_from_tokens",
    "bench_logistic_head_to_head",
    "bench_ridge_timing",
    "encode_tap_tokens",
    "load_converged_model",
    "run_probe_bench",
]


# --------------------------------------------------------------- piece 3: logistic
# A single token-source-generic WS/CS loop. ``tokens`` maps subject -> (N,C,N_tok,d)
# (d=1 raw |STFT| or d=256 encoder tap); the protocol is identical across taps so
# raw and encoder AUROCs are directly comparable. Mirrors
# online_probe_raw_baseline.run_raw_baseline but takes the tokens from ANY source
# and threads max_iter (bounds the p≫n per-electrode d=256 fits, same cap per tap).


def logistic_ws_cs_from_tokens(
    dataset: tp.Any,
    tokens: dict[int, Tensor],
    sd: dict[int, SubjectProbeData],
    *,
    n_parcels: int,
    tap: str,
    max_iter: int = 2000,
) -> dict[str, float]:
    """WS (per-electrode) + CS (parcel-pool) logistic AUROC for one token source.

    Returns ``{val_probe/{tap}/{ws,cs,gap}/{task}: auroc}`` — the same names
    :func:`online_probe.run_probe` emits, so a tap's offline number drops straight
    onto its online counterpart. WS keeps every valid electrode's ``N_tok`` tokens
    (``C·N_tok·d`` features, contiguous 2-fold, mean over WS subjects); CS pools
    electrodes → DKT parcels and fits anchor→test over the parcel intersection."""
    tasks = list(dataset.tasks)
    a = dataset.cs_anchor

    pooled_cs: dict[int, tuple[Tensor, Tensor]] = {}
    for s in {a, *dataset.cs_test_subjects}:
        pooled_cs[s] = pool_electrodes_to_parcels(
            tokens[s], sd[s].parcel_per_electrode, sd[s].electrode_mask, n_parcels
        )

    metrics: dict[str, float] = {}
    for task in tasks:
        ws_vals: list[float] = []
        for s in dataset.ws_subjects:
            z = feature_matrix_per_electrode(tokens[s], sd[s].electrode_mask).numpy()
            zf, yf = _finite_rows(z, sd[s].labels[task])
            ws_vals.append(
                ws_auroc_2fold_logistic(zf, yf, max_iter=max_iter)
                if len(yf) >= 4 else float("nan")
            )
        ws_mean = float(np.nanmean(ws_vals)) if ws_vals else float("nan")

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
            cs_vals.append(cs_auroc_logistic(za, ya, zt, yt, max_iter=max_iter))
        cs_mean = float(np.nanmean(cs_vals)) if cs_vals else float("nan")

        metrics[f"val_probe/{tap}/ws/{task}"] = ws_mean
        metrics[f"val_probe/{tap}/cs/{task}"] = cs_mean
        metrics[f"val_probe/{tap}/gap/{task}"] = ws_mean - cs_mean
    return metrics


@torch.no_grad()
def encode_tap_tokens(
    model: tp.Any,
    sd: SubjectProbeData,
    tap: str,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:  # pragma: no cover - needs a real model + GPU
    """Forward one subject's windows through a tap, return ``(N,C,N_tok,d)`` on CPU.

    ``tap='frontend'`` → :meth:`encode_frontend` (electrode-isolated); ``'latent'``
    → :meth:`encode_latent` (parcel-tagged cross-electrode). Batched + no_grad; the
    band tensors move to ``device`` and the result comes back float CPU so the
    downstream numpy/logistic path is device-free."""
    slow, beta, hg = sd.slow.to(device), sd.beta.to(device), sd.hg.to(device)
    ppe = sd.parcel_per_electrode.to(device)
    emask = sd.electrode_mask.to(device)
    chunks: list[Tensor] = []
    for i in range(0, slow.shape[0], batch_size):
        sl = slice(i, i + batch_size)
        if tap == "frontend":
            f = model.encode_frontend(slow[sl], beta[sl], hg[sl])
        elif tap == "latent":
            f = model.encode_latent(slow[sl], beta[sl], hg[sl], ppe, electrode_mask=emask)
        else:
            raise ValueError(f"tap must be 'frontend' or 'latent', got {tap!r}")
        chunks.append(f.float().cpu())
    return torch.cat(chunks, dim=0)


def bench_logistic_head_to_head(
    model: tp.Any,
    dataset: tp.Any,
    *,
    clip_len_s: float = 1.0,
    batch_size: int = 256,
    device: torch.device | None = None,
    max_iter: int = 2000,
) -> dict[str, tp.Any]:  # pragma: no cover - needs a real model + GPU
    """Piece 3: raw / frontend / latent logistic AUROC head-to-head + per-tap wall
    times. Each tap is built once (raw tokens from the bands; encoder taps from a
    forward) then scored through :func:`logistic_ws_cs_from_tokens`. Returns
    ``{metrics: {...}, timings: {tap: {build_s, fit_s}}}``."""
    dev = device or next(model.parameters()).device
    needed = sorted({dataset.cs_anchor, *dataset.ws_subjects, *dataset.cs_test_subjects})
    sd = {s: dataset.subject_data(s) for s in needed}
    n_parcels = dataset.n_parcels

    metrics: dict[str, float] = {}
    timings: dict[str, dict[str, float]] = {}
    for tap in ("raw", "frontend", "latent"):
        t0 = time.perf_counter()
        if tap == "raw":
            tokens = {
                s: raw_tokens_from_bands(sd[s].slow, sd[s].beta, sd[s].hg,
                                         clip_len_s=clip_len_s)
                for s in needed
            }
        else:
            tokens = {
                s: encode_tap_tokens(model, sd[s], tap, batch_size=batch_size, device=dev)
                for s in needed
            }
        t_build = time.perf_counter() - t0
        t1 = time.perf_counter()
        metrics.update(
            logistic_ws_cs_from_tokens(
                dataset, tokens, sd, n_parcels=n_parcels, tap=tap, max_iter=max_iter
            )
        )
        timings[tap] = {"build_s": t_build, "fit_s": time.perf_counter() - t1}
        del tokens
    return {"metrics": metrics, "timings": timings}


# ------------------------------------------------------------- piece 1: ridge time


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _timed_encode_and_pool(
    model: tp.Any,
    sd: SubjectProbeData,
    k_list: tp.Sequence[int],
    n_parcels: int,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, dict[int, tuple[Tensor, Tensor]]], float, float]:  # pragma: no cover
    """``_encode_and_pool`` split into (forward_s, pool_s) with cuda syncs. Output
    is byte-identical to :func:`online_probe._encode_and_pool` — only timers added."""
    tok = model.student_frontend.tokenizer
    freq_id, time_slot = tok.freq_global_id, tok.time_slot
    slow, beta, hg = sd.slow.to(device), sd.beta.to(device), sd.hg.to(device)
    ppe = sd.parcel_per_electrode.to(device)
    emask = sd.electrode_mask.to(device)

    # Mirrors online_probe._encode_and_pool's chunked-pool memory profile (forward a
    # chunk → pool it → keep only the small CPU pooled result; never hold all N
    # windows' raw feats on the GPU), with the forward / pool times summed across
    # chunks via per-chunk cuda syncs. Output is byte-identical to a single forward.
    taps = ("frontend", "latent")
    pooled_acc: dict[str, dict[int, list[Tensor]]] = {
        tap: {k: [] for k in k_list} for tap in taps
    }
    present_acc: dict[str, dict[int, Tensor]] = {tap: {} for tap in taps}
    t_forward = 0.0
    t_pool = 0.0
    for i in range(0, slow.shape[0], batch_size):
        sl = slice(i, i + batch_size)
        _sync(device)
        t0 = time.perf_counter()
        feats = {
            "frontend": model.encode_frontend(slow[sl], beta[sl], hg[sl]),
            "latent": model.encode_latent(
                slow[sl], beta[sl], hg[sl], ppe, electrode_mask=emask
            ),
        }
        _sync(device)
        t_forward += time.perf_counter() - t0

        t1 = time.perf_counter()
        for tap, f in feats.items():
            for k in k_list:
                pooled, present = pool_window_features(
                    f, freq_id, time_slot, ppe, emask, n_parcels, k
                )
                pooled_acc[tap][k].append(pooled.float().cpu())
                present_acc[tap][k] = present.cpu()
        _sync(device)
        t_pool += time.perf_counter() - t1
    out = {
        tap: {k: (torch.cat(pooled_acc[tap][k]), present_acc[tap][k]) for k in k_list}
        for tap in taps
    }
    return out, t_forward, t_pool


def bench_ridge_timing(
    model: tp.Any,
    dataset: tp.Any,
    k_list: tp.Sequence[int] = (2, 1),
    *,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> dict[str, tp.Any]:  # pragma: no cover - needs a real model + GPU
    """Piece 1: reproduce ``run_probe`` with forward/pool/ridge timers + AUROC.

    Returns ``{metrics, timings: {forward_s, pool_s, ridge_s, total_s,
    n_subjects, n_ridge_solves}, trim: {...}}``. ``metrics`` is the real-checkpoint
    AUROC (identical to ``run_probe`` up to float-order); the trim block prices the
    levers without re-forwarding (the forwards/pools are reused)."""
    dev = device or next(model.parameters()).device
    taps = ("frontend", "latent")
    k_primary = k_list[0]
    n_parcels = dataset.n_parcels
    needed = sorted({dataset.cs_anchor, *dataset.ws_subjects, *dataset.cs_test_subjects})

    pooled: dict[int, dict[str, dict[int, tuple[Tensor, Tensor]]]] = {}
    labels: dict[int, dict[str, np.ndarray]] = {}
    t_forward = t_pool = 0.0
    for s in needed:
        sdata = dataset.subject_data(s)
        out, tf, tp_ = _timed_encode_and_pool(
            model, sdata, k_list, n_parcels, batch_size=batch_size, device=dev
        )
        pooled[s] = out
        labels[s] = sdata.labels
        t_forward += tf
        t_pool += tp_

    metrics: dict[str, float] = {}
    n_solves = 0
    t_ridge = 0.0
    t2 = time.perf_counter()
    for tap in taps:
        for k in k_list:
            sfx = "" if k == k_primary else f"_k{k}"
            for task in dataset.tasks:
                ws_vals: list[float] = []
                for s in dataset.ws_subjects:
                    p, present = pooled[s][tap][k]
                    z = feature_matrix(p, parcel_intersection(present, present)).numpy()
                    zf, yf = _finite_rows(z, labels[s][task])
                    if len(yf) >= 4:
                        ws_vals.append(ws_auroc_2fold(zf, yf))
                        n_solves += 2
                    else:
                        ws_vals.append(float("nan"))
                ws_mean = float(np.nanmean(ws_vals)) if ws_vals else float("nan")

                a = dataset.cs_anchor
                pa, pres_a = pooled[a][tap][k]
                cs_vals: list[float] = []
                for t in dataset.cs_test_subjects:
                    pt, pres_t = pooled[t][tap][k]
                    inter = parcel_intersection(pres_a, pres_t)
                    if inter.numel() == 0:
                        cs_vals.append(float("nan"))
                        continue
                    za, ya = _finite_rows(feature_matrix(pa, inter).numpy(), labels[a][task])
                    zt, yt = _finite_rows(feature_matrix(pt, inter).numpy(), labels[t][task])
                    if len(ya) < 2 or len(yt) < 1:
                        cs_vals.append(float("nan"))
                        continue
                    cs_vals.append(cs_auroc(za, ya, zt, yt))
                    n_solves += 1
                cs_mean = float(np.nanmean(cs_vals)) if cs_vals else float("nan")

                metrics[f"val_probe/{tap}/ws/{task}{sfx}"] = ws_mean
                metrics[f"val_probe/{tap}/cs/{task}{sfx}"] = cs_mean
                metrics[f"val_probe/{tap}/gap/{task}{sfx}"] = ws_mean - cs_mean
    t_ridge = time.perf_counter() - t2

    total = t_forward + t_pool + t_ridge
    # Trim projections (measured fractions; the forwards/pools are reused as-is).
    # The ridge cost scales ~linearly in (#solves); reusing G across the 3 tasks
    # factors once per (subject,tap,k,fold) and solves 3 RHS -> ~/3 on the solve.
    trim = {
        "single_k": {  # drop k=1: half the ridge, forward/pool unchanged
            "ridge_s": t_ridge / len(k_list),
            "total_s": t_forward + t_pool + t_ridge / len(k_list),
        },
        "single_k_reuse_G": {  # + reuse G across 3 tasks: ridge ~/3 more
            "ridge_s": t_ridge / len(k_list) / 3.0,
            "total_s": t_forward + t_pool + t_ridge / len(k_list) / 3.0,
        },
    }
    return {
        "metrics": metrics,
        "timings": {
            "forward_s": t_forward,
            "pool_s": t_pool,
            "ridge_s": t_ridge,
            "total_s": total,
            "n_subjects": len(needed),
            "n_ridge_solves": n_solves,
        },
        "trim": trim,
    }


# ----------------------------------------------------------- model load + driver


def load_converged_model(
    xp: tp.Any, ckpt_path: str, *, clip_len_s: float, device: torch.device
) -> tp.Any:  # pragma: no cover - needs a real ckpt + the model deps
    """Build the converged model at ``clip_len_s`` geometry and load ``ckpt_path``.

    The run's ``brain_model_config`` carries the 5 s ``clip_len_s``; rebuilding it at
    1 s gives a 1 s-token encoder whose LEARNED params match the 5 s checkpoint
    (only ``persistent=False`` geometry buffers differ → not in the state_dict).
    The Lightning checkpoint keys the SSL model under ``model.`` — strip it and load
    ``strict=False``, asserting the only gaps are non-persistent buffers."""
    from speech_decoding.models.v14_converged import V14ConvergedSSL

    cfg = xp.brain_model_config
    try:
        cfg = cfg.model_copy(update={"clip_len_s": float(clip_len_s)})
    except AttributeError:  # not pydantic v2 — mutate the field in place
        cfg.clip_len_s = float(clip_len_s)
    model = cfg.build(n_in_channels=1, n_outputs=1)
    if not isinstance(model, V14ConvergedSSL):
        raise RuntimeError(f"expected V14ConvergedSSL, got {type(model).__name__}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    stripped = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state.items()
    }
    incompat = model.load_state_dict(stripped, strict=False)
    # missing keys must be ONLY non-persistent geometry buffers (rebuilt at init).
    own_persistent = set(model.state_dict().keys())
    real_missing = [k for k in incompat.missing_keys if k in own_persistent]
    if real_missing:
        raise RuntimeError(
            f"checkpoint is missing persistent params {real_missing[:8]} "
            f"({len(real_missing)} total) — the 5s->1s load is not weight-complete."
        )
    if incompat.unexpected_keys:
        print(f"[probe-bench] {len(incompat.unexpected_keys)} unexpected ckpt keys "
              f"(e.g. {incompat.unexpected_keys[:4]}) ignored.")
    return model.to(device).eval()


def run_probe_bench(
    xp: tp.Any,
    *,
    ckpt_path: str,
    out_path: str,
    clip_len_s: float = 1.0,
    batch_size: int = 256,
    k_list: tp.Sequence[int] = (2, 1),
    max_iter: int = 2000,
    do_ridge: bool = True,
    do_headtohead: bool = True,
) -> dict[str, tp.Any]:  # pragma: no cover - DCC-only driver
    """Top-level: build the probe dataset from the run, load the model at 1 s, run
    piece 1 (ridge timing) and piece 3 (logistic head-to-head), write ``out_path``
    incrementally (so a slow/OOM tap still leaves the earlier results on disk)."""
    import json

    from speech_decoding.experiments.online_probe_dataset import build_probe_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe-bench] device={device} ckpt={ckpt_path}")

    print("[probe-bench] building probe dataset (1 s windows) ...")
    t0 = time.perf_counter()
    dataset = build_probe_dataset(xp.data, n_cap=xp.online_probe_n_cap, seed=xp.online_probe_seed)
    print(f"[probe-bench] dataset built in {time.perf_counter()-t0:.1f}s — "
          f"ws={list(dataset.ws_subjects)} cs_anchor={dataset.cs_anchor} "
          f"cs_test={list(dataset.cs_test_subjects)} n_parcels={dataset.n_parcels}")

    model = load_converged_model(xp, ckpt_path, clip_len_s=clip_len_s, device=device)
    print("[probe-bench] model loaded at 1 s geometry.")

    result: dict[str, tp.Any] = {
        "ckpt": ckpt_path, "clip_len_s": clip_len_s, "max_iter": max_iter,
        "cohort": {
            "ws_subjects": list(dataset.ws_subjects),
            "cs_anchor": dataset.cs_anchor,
            "cs_test_subjects": list(dataset.cs_test_subjects),
            "n_parcels": dataset.n_parcels,
        },
    }

    def _flush() -> None:
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=float)

    _flush()
    if do_ridge:
        print("[probe-bench] piece 1: ridge timing ...")
        t = time.perf_counter()
        result["piece1_ridge"] = bench_ridge_timing(
            model, dataset, k_list, batch_size=batch_size, device=device
        )
        print(f"[probe-bench] piece 1 done in {time.perf_counter()-t:.1f}s: "
              f"{result['piece1_ridge']['timings']}")
        _flush()
    if do_headtohead:
        print("[probe-bench] piece 3: logistic head-to-head ...")
        t = time.perf_counter()
        result["piece3_head_to_head"] = bench_logistic_head_to_head(
            model, dataset, clip_len_s=clip_len_s, batch_size=batch_size,
            device=device, max_iter=max_iter,
        )
        print(f"[probe-bench] piece 3 done in {time.perf_counter()-t:.1f}s: "
              f"{result['piece3_head_to_head']['timings']}")
        _flush()
    print(f"[probe-bench] wrote {out_path}")
    return result
