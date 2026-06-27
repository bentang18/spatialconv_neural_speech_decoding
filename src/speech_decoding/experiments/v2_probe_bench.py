"""v2 dev probe-bench driver — raw floor + (when a ckpt exists) encoder taps.

One entrypoint, :func:`run_v2_probe_bench`, over the firewalled pretrain-corpus dev
probe (:mod:`v2_probe_dataset`):

  - **raw** (always, model-free): the RAW RAW |STFT|-bin floor
    (:func:`v2_raw_probe.run_v2_raw_baseline`). Runs with no checkpoint, so the port
    is testable the moment the dataset builds.
  - **frontend / latent** (only when ``ckpt_path`` is given): the trained taps from
    :meth:`V14ConvergedV2.encode_clip_taps` (the clean, mask-free teacher-path encode),
    head-to-head against raw in the SAME ``val_probe/{tap}/{ws,cs,gap}/{task}`` metric
    namespace.

Encoder-tap reduction (PROVISIONAL — confirm with Ben): the frontend ``(N,C,S,d)`` and
latent ``(N,P,k,S,d)`` taps are **mean-pooled over the token grid** (frontend over
``S``; latent over ``k`` and ``S``) to one ``d``-vector per electrode / parcel before
the logistic. This keeps the probe tractable (``d`` features, not ``C·S·d``) and pools
in-loop so host memory stays small. It deliberately differs from the RAW tap, which
stays fully unpooled (every bin) per Ben's directive — the floor reads the literal
input; the encoder taps read a time-pooled representation a downstream decoder would use.

The xp construction is reused from ``dispatch_v14`` (via a dispatch branch), so the 1 s
probe dataset and the 5 s→1 s model load are byte-faithful to the production run.
"""

from __future__ import annotations

import json
import os
import typing as tp

import numpy as np
import torch
from torch import Tensor

from speech_decoding.experiments.linear_probe_logistic import (
    cs_auroc_logistic,
    ws_auroc_2fold_logistic,
)
from speech_decoding.experiments.online_probe import (
    _finite_rows,
    feature_matrix,
    parcel_intersection,
)
from speech_decoding.experiments.online_probe_dataset import N_CAP
from speech_decoding.experiments.v2_raw_probe import raw_ws_cs_auroc, run_v2_raw_baseline

__all__ = [
    "load_v2_converged_model",
    "encode_subject_taps",
    "latent_ws_cs_auroc",
    "run_v2_encoder_taps",
    "run_v2_probe_bench",
]


def load_v2_converged_model(
    xp: tp.Any, ckpt_path: str, *, device: torch.device
) -> tp.Any:  # pragma: no cover - needs a real ckpt + model deps
    """Build :class:`V14ConvergedV2` from the run's config and load ``ckpt_path``.

    The v2 model is clip-len-AGNOSTIC at build (band geometry threads at forward via
    ``clip_len_s``), so no config mutation is needed — :func:`encode_subject_taps`
    passes the 1 s clock. The Lightning checkpoint keys the model under ``model.``
    (``V14ConvergedV2BrainModule.model``); strip it and load ``strict=False``,
    asserting the only gaps are non-persistent geometry buffers."""
    from speech_decoding.models.v14_converged_v2 import V14ConvergedV2

    model = xp.brain_model_config.build(n_in_channels=1, n_outputs=1)
    if not isinstance(model, V14ConvergedV2):
        raise RuntimeError(f"expected V14ConvergedV2, got {type(model).__name__}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    stripped = {
        (k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()
    }
    incompat = model.load_state_dict(stripped, strict=False)
    own_persistent = set(model.state_dict().keys())
    real_missing = [k for k in incompat.missing_keys if k in own_persistent]
    if real_missing:
        raise RuntimeError(
            f"checkpoint is missing persistent params {real_missing[:8]} "
            f"({len(real_missing)} total) — the 5s->1s load is not weight-complete."
        )
    if incompat.unexpected_keys:
        print(f"[v2-probe-bench] {len(incompat.unexpected_keys)} unexpected ckpt keys "
              f"(e.g. {incompat.unexpected_keys[:4]}) ignored.")
    return model.to(device).eval()


@torch.no_grad()
def encode_subject_taps(
    model: tp.Any,
    bands: list[Tensor],
    parcel_per_electrode: Tensor,
    *,
    clip_len_s: float,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[Tensor, Tensor, Tensor]:
    """Forward one subject's clips → token-pooled frontend + latent taps.

    Pools the token grid INSIDE the batch loop (frontend mean over ``S``; latent mean
    over ``k`` and ``S``) so only ``(N,C,d)`` / ``(N,P,d)`` ever land on the host.
    ``labels`` ``(P,)`` are the active-parcel DKT ids (constant across the subject's
    windows; read off the first batch). Returns ``(front (N,C,d), latent (N,P,d),
    labels (P,))`` on CPU."""
    lfs, hga = bands[0], bands[1]
    n = lfs.shape[0]
    ppe = parcel_per_electrode.to(device)
    fronts: list[Tensor] = []
    latents: list[Tensor] = []
    labels: Tensor | None = None
    for i in range(0, n, batch_size):
        taps = model.encode_clip_taps(
            lfs[i : i + batch_size].to(device),
            hga[i : i + batch_size].to(device),
            ppe,
            clip_len_s=clip_len_s,
        )
        fronts.append(taps["frontend"].mean(dim=2).cpu())          # (b,C,d)
        latents.append(taps["latent"].mean(dim=(2, 3)).cpu())      # (b,P,d)
        if labels is None:
            labels = taps["labels"].cpu().long()
    if labels is None:
        raise RuntimeError("subject had no clips to encode")
    return torch.cat(fronts, 0), torch.cat(latents, 0), labels


def latent_ws_cs_auroc(
    latent: dict[int, Tensor],
    labels: dict[int, Tensor],
    sd: dict[int, tp.Any],
    *,
    ws_subjects: tp.Sequence[int],
    cs_anchor: int,
    cs_test_subjects: tp.Sequence[int],
    tasks: tp.Sequence[str],
    n_parcels: int,
    max_iter: int = 10000,
) -> dict[str, float]:
    """Logistic AUROC on the per-parcel latent tap.

    ``latent[s]`` is ``(N, P_s, d)`` (token-pooled), ``labels[s]`` the ``(P_s,)``
    active-parcel DKT ids. **WS** flattens the subject's own active parcels →
    ``(N, P_s·d)``. **CS** scatters each subject's parcels into the global
    ``n_parcels`` table by DKT id, then fits the anchor / scores each test subject
    over the shared (present-in-both) parcels — the same global-id intersection the raw
    CS pool uses. Emits ``val_probe/latent/{ws,cs,gap}/{task}``."""
    def _to_global(s: int) -> tuple[Tensor, Tensor]:
        lat, lab = latent[s], labels[s]
        n, _, d = lat.shape
        glob = lat.new_zeros(n, n_parcels, d)
        glob[:, lab] = lat
        present = torch.zeros(n_parcels, dtype=torch.bool)
        present[lab] = True
        return glob, present

    glob_cs = {s: _to_global(s) for s in {cs_anchor, *cs_test_subjects}}

    metrics: dict[str, float] = {}
    for task in tasks:
        ws_vals: list[float] = []
        for s in ws_subjects:
            z = latent[s].reshape(latent[s].shape[0], -1).numpy()
            zf, yf = _finite_rows(z, sd[s].labels[task])
            ws_vals.append(
                ws_auroc_2fold_logistic(zf, yf, max_iter=max_iter)
                if len(yf) >= 4 else float("nan")
            )
        ws_mean = float(np.nanmean(ws_vals)) if ws_vals else float("nan")

        ga, pres_a = glob_cs[cs_anchor]
        cs_vals: list[float] = []
        for t in cs_test_subjects:
            gt, pres_t = glob_cs[t]
            inter = parcel_intersection(pres_a, pres_t)
            if inter.numel() == 0:
                cs_vals.append(float("nan"))
                continue
            za, ya = _finite_rows(feature_matrix(ga, inter).numpy(), sd[cs_anchor].labels[task])
            zt, yt = _finite_rows(feature_matrix(gt, inter).numpy(), sd[t].labels[task])
            if len(ya) < 2 or len(yt) < 1:
                cs_vals.append(float("nan"))
                continue
            cs_vals.append(cs_auroc_logistic(za, ya, zt, yt, max_iter=max_iter))
        cs_mean = float(np.nanmean(cs_vals)) if cs_vals else float("nan")

        metrics[f"val_probe/latent/ws/{task}"] = ws_mean
        metrics[f"val_probe/latent/cs/{task}"] = cs_mean
        metrics[f"val_probe/latent/gap/{task}"] = ws_mean - cs_mean
    return metrics


def run_v2_encoder_taps(
    dataset: tp.Any,
    model: tp.Any,
    *,
    clip_len_s: float,
    device: torch.device,
    max_iter: int = 10000,
    batch_size: int = 64,
) -> dict[str, float]:
    """Frontend + latent tap AUROC over the dev probe dataset.

    Frontend (per-electrode, token-pooled to ``(N,C,d)``) reuses the raw WS/CS
    machinery under ``tap="frontend"``; latent (per-parcel) uses
    :func:`latent_ws_cs_auroc`."""
    needed = sorted({dataset.cs_anchor, *dataset.ws_subjects, *dataset.cs_test_subjects})
    sd = {s: dataset.subject_data(s) for s in needed}
    front: dict[int, Tensor] = {}
    latent: dict[int, Tensor] = {}
    labels: dict[int, Tensor] = {}
    for s in needed:
        f, lat, lab = encode_subject_taps(
            model, sd[s].bands, sd[s].parcel_per_electrode,
            clip_len_s=clip_len_s, device=device, batch_size=batch_size,
        )
        front[s] = f.unsqueeze(-1)        # (N,C,d,1) for the raw machinery
        latent[s] = lat
        labels[s] = lab

    metrics = raw_ws_cs_auroc(
        front, sd, ws_subjects=dataset.ws_subjects, cs_anchor=dataset.cs_anchor,
        cs_test_subjects=dataset.cs_test_subjects, tasks=dataset.tasks,
        n_parcels=dataset.n_parcels, max_iter=max_iter, tap="frontend",
    )
    metrics.update(latent_ws_cs_auroc(
        latent, labels, sd, ws_subjects=dataset.ws_subjects, cs_anchor=dataset.cs_anchor,
        cs_test_subjects=dataset.cs_test_subjects, tasks=dataset.tasks,
        n_parcels=dataset.n_parcels, max_iter=max_iter,
    ))
    return metrics


def run_v2_probe_bench(
    xp: tp.Any,
    *,
    out_path: str,
    ckpt_path: str | None = None,
    clip_len_s: float = 1.0,
    n_cap: int = N_CAP,
    max_iter: int = 10000,
    device: torch.device | None = None,
) -> dict[str, float]:  # pragma: no cover - needs BT voltage (+ ckpt for taps)
    """Build the dev probe dataset from ``xp.data`` and bench it.

    Always runs the raw floor. If ``ckpt_path`` is given, also loads the v2 model and
    runs the frontend/latent taps head-to-head. Writes ``{metrics, ckpt, n_cap}`` to
    ``out_path`` and returns the metric dict."""
    from speech_decoding.experiments.v2_probe_dataset import build_v2_probe_dataset

    print(f"[v2-probe-bench] building dev probe dataset (n_cap={n_cap}, clip={clip_len_s}s) ...")
    dataset = build_v2_probe_dataset(xp.data, n_cap=n_cap)
    print(f"[v2-probe-bench] cohort ws={dataset.ws_subjects} "
          f"cs_anchor={dataset.cs_anchor} cs_test={dataset.cs_test_subjects} "
          f"n_parcels={dataset.n_parcels}")

    metrics = run_v2_raw_baseline(dataset, max_iter=max_iter)
    print("[v2-probe-bench] raw floor:")
    for k in sorted(metrics):
        print(f"    {k} = {metrics[k]:.4f}")

    if ckpt_path is not None:
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[v2-probe-bench] loading checkpoint {ckpt_path} on {dev} ...")
        model = load_v2_converged_model(xp, ckpt_path, device=dev)
        tap_metrics = run_v2_encoder_taps(
            dataset, model, clip_len_s=clip_len_s, device=dev, max_iter=max_iter
        )
        metrics.update(tap_metrics)
        print("[v2-probe-bench] encoder taps:")
        for k in sorted(tap_metrics):
            print(f"    {k} = {tap_metrics[k]:.4f}")
    else:
        print("[v2-probe-bench] no checkpoint — raw floor only (taps skipped).")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "ckpt": ckpt_path, "n_cap": n_cap}, f, indent=2)
    print(f"[v2-probe-bench] wrote {out_path}")
    return metrics
