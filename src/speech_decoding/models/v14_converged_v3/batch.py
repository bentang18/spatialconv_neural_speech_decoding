"""v14_converged_v3 — V3Batch + session-homogeneous collate (Phase D3).

A batch is B clips from ONE session (the reused ``_SessionGroupedBatchSampler``
guarantees homogeneity), so the per-session fixtures — ``L1Geometry`` and
``parcel_id`` — are SHARED and carried once, not per-item. ``v3_collate`` stacks
each band's per-clip ``(N, F_band, T)`` into the ``(B, N, F_band, T)`` that
``V3ConvergedModel.forward`` consumes as ``band_inputs``, and fails loud if a batch
ever mixes sessions (the shared geometry would be wrong for the odd clips).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from speech_decoding.models.v14_converged_v3.geometry import L1Geometry
from speech_decoding.models.v14_converged_v3.session_setup import V3SessionSetup


@dataclass(frozen=True)
class V3ClipSample:
    """One random-continuous clip: the 3 |STFT| bands + its session's fixtures."""

    bands: tuple[Tensor, ...]  # 3 × (N, F_band, T)
    setup: V3SessionSetup
    session_key: tuple


@dataclass(frozen=True)
class V3Batch:
    """A session-homogeneous batch ready for ``V3ConvergedModel.forward``."""

    bands: list[Tensor]  # 3 × (B, N, F_band, T)
    geom: L1Geometry
    parcel_id: Tensor  # (N,) long — session-shared
    session_key: tuple


def v3_collate(samples: Sequence[V3ClipSample]) -> V3Batch:
    """Stack a session-homogeneous list of clips into a ``V3Batch``.

    All clips must share one ``session_key``/setup (the sampler's invariant); a
    cross-session batch raises rather than silently corrupt the shared geometry.
    """
    if not samples:
        raise ValueError("v3_collate got an empty batch")
    key = samples[0].session_key
    setup = samples[0].setup
    for s in samples:
        if s.session_key != key:
            raise ValueError(
                f"batch is not session-homogeneous: {key} vs {s.session_key} — the "
                "_SessionGroupedBatchSampler invariant broke"
            )
    n_bands = len(samples[0].bands)
    bands = [
        torch.stack([s.bands[b] for s in samples], dim=0) for b in range(n_bands)
    ]
    return V3Batch(
        bands=bands,
        geom=setup.geom,
        parcel_id=setup.parcel_id,
        session_key=key,
    )
