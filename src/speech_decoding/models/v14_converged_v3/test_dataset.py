"""v14_converged_v3 Phase D2/D4 — V3SessionDataset (TDD, synthetic caches).

Memo project-v3-pipeline-build-contract-2026-07-10: the map-style dataset over the
per-session continuous |STFT| spec caches. Each dataset index is one random-
CONTINUOUS clip draw from a session; ``__getitem__`` (1) samples a guard-2-valid t0
(``clip_sampler``), (2) memmap-reads the 3 band ``.npy`` caches at the survivor rows
(``keep_idx``) over ``[t0, t0+T)``, (3) applies the session's FROZEN robust-z
(reused ``SessionRobustZNormalizer.from_stats``), and returns a ``V3ClipSample``.
Session identity per index feeds the reused ``_SessionGroupedBatchSampler``.

Tested against tiny synthetic ``.npy`` caches — the real braintreebank file-format
adapter (labels / LOF / parcel support) is a thin layer validated on DeltaAI at F2.
"""

from __future__ import annotations

import numpy as np
import torch

from speech_decoding.models.v14_converged_v3.batch import v3_collate
from speech_decoding.models.v14_converged_v3.dataset import (
    V3SessionDataset,
    build_session_spec,
)
from speech_decoding.models.v14_converged_v3.session_setup import build_session_setup

F_SLOW, F_MID, F_HGA = 7, 6, 7
T_CLIP = 96
FPS = 32.0


def _write_band(tmp, name, c, f, t_total):
    arr = (np.random.RandomState(abs(hash(name)) % 2**32)
           .randn(c, f, t_total).astype(np.float32))
    path = str(tmp / f"{name}.npy")
    np.save(path, arr)
    return path, arr


def _spec(tmp, *, key=(1, 0), shaft_sizes=(4, 3, 3), t_total=4000,
          bad_spans=(), drop=frozenset()):
    labels, parcels = [], []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            parcels.append(s)
    c_full = len(labels)
    setup = build_session_setup(labels, torch.tensor(parcels), drop_labels=drop)
    band_paths, band_fbins = [], (F_SLOW, F_MID, F_HGA)
    for bname, fb in zip(("v3slow", "v3mid", "hga"), band_fbins):
        p, _ = _write_band(tmp, f"{key[0]}_{key[1]}_{bname}", c_full, fb, t_total)
        band_paths.append(p)
    # frozen robust-z stats over the FULL c_full rows (adapter slices to survivors)
    band_stats = [
        (torch.randn(c_full, fb, 1), torch.rand(c_full, fb, 1) + 0.5)
        for fb in band_fbins
    ]
    return build_session_spec(
        session_key=key,
        band_paths=tuple(band_paths),
        band_stats=tuple(band_stats),
        setup=setup,
        n_frames=t_total,
        bad_spans_s=list(bad_spans),
    )


def test_getitem_returns_three_bands_survivor_rows(tmp_path) -> None:
    spec = _spec(tmp_path)
    n = len(spec.setup.sidecar.labels)
    ds = V3SessionDataset([spec], clips_per_session=5, clip_frames=T_CLIP, fps=FPS)
    sample = ds[0]
    assert len(sample.bands) == 3
    assert sample.bands[0].shape == (n, F_SLOW, T_CLIP)
    assert sample.bands[1].shape == (n, F_MID, T_CLIP)
    assert sample.bands[2].shape == (n, F_HGA, T_CLIP)
    assert sample.session_key == (1, 0)


def test_len_is_total_clip_budget(tmp_path) -> None:
    a = _spec(tmp_path / "a" if False else tmp_path, key=(1, 0))
    b = _spec(tmp_path, key=(2, 1), shaft_sizes=(3, 3))
    ds = V3SessionDataset([a, b], clips_per_session=7, clip_frames=T_CLIP, fps=FPS)
    assert len(ds) == 14


def test_session_key_list_aligns_to_index(tmp_path) -> None:
    a = _spec(tmp_path, key=(1, 0))
    b = _spec(tmp_path, key=(2, 1), shaft_sizes=(3, 3))
    ds = V3SessionDataset([a, b], clips_per_session=3, clip_frames=T_CLIP, fps=FPS)
    keys = ds.session_key_list()
    assert len(keys) == 6
    assert keys[:3] == [(1, 0)] * 3
    assert keys[3:] == [(2, 1)] * 3


def test_getitem_reads_survivor_rows_after_drop(tmp_path) -> None:
    # Drop LB1 → survivors exclude it; the clip must have N-1 rows, in survivor order.
    spec = _spec(tmp_path, drop=frozenset({"LB1"}))
    n = len(spec.setup.sidecar.labels)
    assert "LB1" not in spec.setup.sidecar.labels
    ds = V3SessionDataset([spec], clips_per_session=1, clip_frames=T_CLIP, fps=FPS)
    assert ds[0].bands[0].shape[0] == n


def test_robust_z_matches_reused_normalizer(tmp_path) -> None:
    # The clip the dataset returns must equal a direct read + from_stats transform.
    spec = _spec(tmp_path)
    ds = V3SessionDataset([spec], clips_per_session=1, clip_frames=T_CLIP, fps=FPS)
    ds.set_epoch(0)
    # force a deterministic t0 by making the whole session valid + seeding
    sample = ds[0]
    # recompute band 0 with the same normalizer at the SAME t0 the dataset used
    from speech_decoding.extractors.normalize import SessionRobustZNormalizer

    t0 = ds._last_t0  # test hook: the t0 __getitem__ drew
    mm = np.load(spec.band_paths[0], mmap_mode="r")
    keep = spec.keep_idx.numpy()
    raw = torch.from_numpy(mm[keep, :, t0:t0 + T_CLIP].astype(np.float32))
    med, sig = spec.band_stats[0]
    norm = SessionRobustZNormalizer.from_stats(median=med, sigma=sig)
    assert torch.allclose(sample.bands[0], norm.transform(raw))


def test_guard2_never_reads_a_bad_span(tmp_path) -> None:
    # A bad span in the MIDDLE; over many draws no clip may overlap it.
    a, b = 50.0, 60.0  # seconds → frames [1600,1920)
    spec = _spec(tmp_path, t_total=4000, bad_spans=[(a, b)])
    ds = V3SessionDataset([spec], clips_per_session=1, clip_frames=T_CLIP, fps=FPS)
    fa, fb = round(a * FPS), round(b * FPS)
    for ep in range(60):
        ds.set_epoch(ep)
        ds[0]
        t0 = ds._last_t0
        assert not (t0 < fb and t0 + T_CLIP > fa)


def test_epoch_changes_the_draw(tmp_path) -> None:
    spec = _spec(tmp_path, t_total=8000)
    ds = V3SessionDataset([spec], clips_per_session=1, clip_frames=T_CLIP, fps=FPS)
    draws = set()
    for ep in range(5):
        ds.set_epoch(ep)
        ds[0]
        draws.add(ds._last_t0)
    assert len(draws) > 1  # epochs re-draw the random-continuous window


def test_collate_integration_produces_batch(tmp_path) -> None:
    spec = _spec(tmp_path)
    n = len(spec.setup.sidecar.labels)
    ds = V3SessionDataset([spec], clips_per_session=8, clip_frames=T_CLIP, fps=FPS)
    batch = v3_collate([ds[i] for i in range(4)])
    assert batch.bands[0].shape == (4, n, F_SLOW, T_CLIP)
    assert batch.session_key == (1, 0)


def test_build_session_spec_slices_stats_to_survivors(tmp_path) -> None:
    # band_stats come in at c_full rows; the spec must slice them to keep_idx so they
    # align with the survivor clip rows the mmap read returns.
    spec = _spec(tmp_path, drop=frozenset({"LA1"}))
    n = len(spec.setup.sidecar.labels)
    for med, sig in spec.band_stats:
        assert med.shape[0] == n and sig.shape[0] == n
