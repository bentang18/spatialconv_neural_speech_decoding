"""v14_converged_v3 Phase D4 — V3DataModule (TDD, synthetic sessions).

Wires ``V3SessionDataset`` + the REUSED ``_SessionGroupedBatchSampler`` (session-
homogeneous, B enforced, DDP self-sharding) + ``v3_collate`` into a train loader
(memo project-v3-pipeline-build-contract-2026-07-10). Asserted contracts: the
loader yields ``V3Batch`` objects; every batch is one session (the ragged forward's
static-shape prerequisite); B is honoured; an epoch sees every session; and
``set_epoch`` fans out to BOTH the dataset (re-draw windows) and the sampler
(reshuffle) so the two stay phase-locked.
"""

from __future__ import annotations

import numpy as np
import torch

from speech_decoding.models.v14_converged_v3.batch import V3Batch
from speech_decoding.models.v14_converged_v3.datamodule import V3DataModule
from speech_decoding.models.v14_converged_v3.dataset import build_session_spec
from speech_decoding.models.v14_converged_v3.session_setup import build_session_setup

F_SLOW, F_MID, F_HGA = 7, 6, 7
T_CLIP = 96
FPS = 32.0


def _spec(tmp, *, key, shaft_sizes, t_total=4000):
    labels, parcels = [], []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            parcels.append(s)
    c_full = len(labels)
    setup = build_session_setup(labels, torch.tensor(parcels), drop_labels=set())
    band_paths = []
    for bname, fb in zip(("v3slow", "v3mid", "hga"), (F_SLOW, F_MID, F_HGA)):
        arr = np.random.RandomState(key[0] * 10 + key[1]).randn(c_full, fb, t_total).astype(np.float32)
        p = str(tmp / f"{key[0]}_{key[1]}_{bname}.npy")
        np.save(p, arr)
        band_paths.append(p)
    band_stats = [
        (torch.randn(c_full, fb, 1), torch.rand(c_full, fb, 1) + 0.5)
        for fb in (F_SLOW, F_MID, F_HGA)
    ]
    return build_session_spec(
        session_key=key, band_paths=tuple(band_paths), band_stats=tuple(band_stats),
        setup=setup, n_frames=t_total, bad_spans_s=[],
    )


def _dm(tmp, batch_size=4, clips_per_session=8):
    sessions = [
        _spec(tmp, key=(1, 0), shaft_sizes=(4, 3, 3)),
        _spec(tmp, key=(2, 1), shaft_sizes=(3, 3, 3, 3)),
    ]
    return V3DataModule(
        sessions,
        batch_size=batch_size,
        clips_per_session=clips_per_session,
        clip_frames=T_CLIP,
        fps=FPS,
        num_workers=0,
        seed=7,
    )


def test_train_loader_yields_v3_batches(tmp_path) -> None:
    dm = _dm(tmp_path)
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert isinstance(batch, V3Batch)
    assert len(batch.bands) == 3
    assert batch.bands[0].shape[2] == F_SLOW and batch.bands[0].shape[3] == T_CLIP


def test_every_batch_is_session_homogeneous(tmp_path) -> None:
    dm = _dm(tmp_path)
    for batch in dm.train_dataloader():
        b, n = batch.bands[0].shape[0], batch.bands[0].shape[1]
        # N must equal the batch's session electrode count (shared geom)
        assert batch.parcel_id.shape == (n,)
        assert batch.geom.shaft_of_contact.shape == (n,)
        assert b <= 4


def test_full_batches_have_batch_size(tmp_path) -> None:
    dm = _dm(tmp_path, batch_size=4, clips_per_session=8)
    sizes = [batch.bands[0].shape[0] for batch in dm.train_dataloader()]
    # 8 clips/session, B=4 → two full batches of 4 per session, no remainder
    assert all(s == 4 for s in sizes)
    assert len(sizes) == 4  # 2 sessions × 2 batches


def test_epoch_covers_every_session(tmp_path) -> None:
    dm = _dm(tmp_path)
    keys = {batch.session_key for batch in dm.train_dataloader()}
    assert keys == {(1, 0), (2, 1)}


def test_set_epoch_fans_out_to_dataset_and_sampler(tmp_path) -> None:
    dm = _dm(tmp_path)
    loader = dm.train_dataloader()
    dm.set_epoch(3)
    assert dm.dataset._epoch == 3
    assert loader.batch_sampler._epoch == 3


def test_set_epoch_redraws_windows(tmp_path) -> None:
    dm = _dm(tmp_path, batch_size=2, clips_per_session=2)
    windows = set()
    for ep in range(4):
        dm.set_epoch(ep)
        for _ in dm.train_dataloader():
            pass
        windows.add(dm.dataset._last_t0)
    assert len(windows) > 1


def test_drop_last_default_keeps_partial(tmp_path) -> None:
    # clips_per_session=5, B=4 → one full + one partial(1) per session unless drop_last
    dm = _dm(tmp_path, batch_size=4, clips_per_session=5)
    sizes = sorted(batch.bands[0].shape[0] for batch in dm.train_dataloader())
    assert sizes == [1, 1, 4, 4]  # partials kept
