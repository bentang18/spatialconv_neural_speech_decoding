"""v14_converged_v3 Phase D3 — V3Batch + session-homogeneous collate (TDD).

Memo project-v3-pipeline-build-contract-2026-07-10: B=32 SESSION-HOMOGENEOUS
batches. Every clip in a batch comes from ONE session (the reused
``_SessionGroupedBatchSampler`` guarantees it), so the per-session fixtures — the
``L1Geometry`` and ``parcel_id`` — are SHARED, carried once per batch, never
per-item. That is what makes the static-shape compile-once-per-session contract
hold and what lets the ragged L1 attention pay only this session's true N.

The collate stacks each band's per-clip ``(N, F_band, T)`` into ``(B, N, F_band, T)``
(the shape ``V3ConvergedModel.forward`` consumes as ``band_inputs``), takes the
shared geom/parcel_id from the first clip, and FAILS LOUD if a batch is ever handed
clips from two sessions (the sampler invariant broke) — a silent mix would corrupt
the shared geometry.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.batch import (
    V3ClipSample,
    v3_collate,
)
from speech_decoding.models.v14_converged_v3.session_setup import build_session_setup

F_SLOW, F_MID, F_HGA = 7, 6, 7
T = 96  # clip_len 3.0 s @ 32 Hz


def _setup(shaft_sizes=(4, 3, 3), key=(1, 0)):
    labels, parcels = [], []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            parcels.append(s)
    setup = build_session_setup(labels, torch.tensor(parcels), drop_labels=set())
    return setup, key


def _clip(setup, key):
    n = len(setup.sidecar.labels)
    bands = (
        torch.randn(n, F_SLOW, T),
        torch.randn(n, F_MID, T),
        torch.randn(n, F_HGA, T),
    )
    return V3ClipSample(bands=bands, setup=setup, session_key=key)


def test_collate_stacks_bands_to_B_N_F_T() -> None:
    setup, key = _setup()
    n = len(setup.sidecar.labels)
    batch = v3_collate([_clip(setup, key) for _ in range(5)])
    assert len(batch.bands) == 3
    assert batch.bands[0].shape == (5, n, F_SLOW, T)
    assert batch.bands[1].shape == (5, n, F_MID, T)
    assert batch.bands[2].shape == (5, n, F_HGA, T)


def test_collate_carries_shared_geom_and_parcel_once() -> None:
    setup, key = _setup()
    batch = v3_collate([_clip(setup, key) for _ in range(3)])
    # geom/parcel_id are the session's, not per-item
    assert batch.geom is setup.geom
    assert torch.equal(batch.parcel_id, setup.parcel_id)
    assert batch.parcel_id.shape == (len(setup.sidecar.labels),)


def test_collate_preserves_session_key() -> None:
    setup, key = _setup(key=(6, 4))
    batch = v3_collate([_clip(setup, key) for _ in range(2)])
    assert batch.session_key == (6, 4)


def test_collate_rejects_cross_session_batch() -> None:
    a, ka = _setup(shaft_sizes=(4, 3, 3), key=(1, 0))
    b, kb = _setup(shaft_sizes=(3, 3), key=(2, 1))
    with pytest.raises(ValueError, match="session-homogeneous"):
        v3_collate([_clip(a, ka), _clip(b, kb)])


def test_collate_single_clip_batch() -> None:
    setup, key = _setup()
    n = len(setup.sidecar.labels)
    batch = v3_collate([_clip(setup, key)])
    assert batch.bands[0].shape == (1, n, F_SLOW, T)
    assert batch.session_key == key


def test_collate_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        v3_collate([])


def test_batch_bands_feed_model_forward() -> None:
    # The stacked bands must be exactly what V3ConvergedModel.forward consumes.
    from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel

    setup, key = _setup()
    batch = v3_collate([_clip(setup, key) for _ in range(2)])
    torch.manual_seed(0)
    model = V3ConvergedModel(n_parcels=8).eval()
    g = torch.Generator().manual_seed(0)
    out = model(
        batch.bands, batch.geom, batch.parcel_id, generator=g, backend="reference"
    )
    assert torch.isfinite(out.loss)
