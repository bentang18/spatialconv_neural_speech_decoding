"""Stage 1: assemble per-checkpoint token-grid caches (GPU forward → SessionTapCache).

For one checkpoint, Stage 1 forwards the encoder ONCE over each cohort session's union
clip windows (:class:`SessionTargets` from :mod:`pretrain_probe_labels`) and caches the
per-token tap grids. Stage 2 (the readouts) then re-runs on CPU off the cache — so the
expensive encoder forward is paid once per checkpoint, not once per (cell, tap, readout).

The encoder forward + BT clip materialization is GPU/DCC, injected here as ``forward_fn``
(``SessionTargets -> {tap: grid}``) so the assembly/loop/save logic is pure and testable;
the real ``forward_fn`` is the dispatch step. Electrode→parcel metadata per session is
likewise injected (:class:`SessionMeta`) — it comes from the same anatomy the encoder pool
uses, not recomputed here.

Cache key (Ben 2026-06-28): ``(ckpt, session, electrode_set=all, clip_set)`` — full
electrodes (not lite), pretrain cells. Firewall: ``targets_by_session`` must be built only
from the firewall-legal cohort; this module never selects sessions itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor

from speech_decoding.experiments.pretrain_probe_driver import (
    SessionTapCache,
    save_cache,
)
from speech_decoding.experiments.pretrain_probe_labels import SessionTargets

__all__ = [
    "SessionMeta",
    "cache_from_targets",
    "build_caches",
]


@dataclass(frozen=True)
class SessionMeta:
    """Electrode→parcel anatomy for one session (the encoder pool's own mapping).

    ``parcel_labels`` are the ``(P,)`` DKT atlas ids of the COMPACTED parcel-space grids'
    P axis (``encode_clip_taps`` emits ~16 present parcels, not the full atlas). The
    parcel-space readouts map compacted↔atlas lazily off these ids — the grids are NEVER
    densified to the full atlas (that 5×'d memory/disk), so ``parcel_labels`` is the bridge.
    ``None`` for electrode-space-only caches (raw floor / M2), which need no parcel ids."""

    parcel_per_electrode: Tensor      # (C,)
    electrode_mask: Tensor            # (C,)
    n_parcels: int
    parcel_labels: Tensor | None = None   # (P,) DKT ids of the compacted parcel grids


def cache_from_targets(
    targets: SessionTargets,
    grids: dict[str, Tensor],
    meta: SessionMeta,
) -> SessionTapCache:
    """Combine forwarded tap grids with the session's labels/splits/anatomy.

    Every grid's leading axis must equal the union clip count, so labels and split indices
    line up with the forwarded rows."""
    n = len(targets.clip_starts)
    for tap, g in grids.items():
        if g.shape[0] != n:
            raise ValueError(
                f"tap {tap!r} grid has {g.shape[0]} rows but the union clip axis has {n}; "
                "the forward must cover exactly the SessionTargets clips, in order."
            )
    return SessionTapCache(
        subject_id=targets.subject_id,
        trial_id=targets.trial_id,
        grids=grids,
        labels=targets.labels,
        parcel_per_electrode=meta.parcel_per_electrode,
        electrode_mask=meta.electrode_mask,
        n_parcels=meta.n_parcels,
        parcel_labels=meta.parcel_labels,
        ws_split=targets.ws_split,
        cs_split=targets.cs_split,
    )


def build_caches(
    targets_by_session: dict[tuple[int, int], SessionTargets],
    meta_by_session: dict[tuple[int, int], SessionMeta],
    forward_fn: Callable[[SessionTargets], dict[str, Tensor]],
    *,
    save_path_fn: Callable[[tuple[int, int]], str] | None = None,
) -> dict[tuple[int, int], SessionTapCache]:
    """Forward + assemble (+ optionally save) one cache per cohort session.

    ``forward_fn`` runs the GPU encoder over a session's union clips and returns the
    ``{tap: grid}`` dict (the only GPU step). ``save_path_fn(key)`` → a path; when given,
    each cache is ``torch.save``-d there for the CPU-side Stage-2 reruns."""
    caches: dict[tuple[int, int], SessionTapCache] = {}
    for key, targets in targets_by_session.items():
        grids = forward_fn(targets)
        cache = cache_from_targets(targets, grids, meta_by_session[key])
        if save_path_fn is not None:
            save_cache(cache, save_path_fn(key))
        caches[key] = cache
    return caches
