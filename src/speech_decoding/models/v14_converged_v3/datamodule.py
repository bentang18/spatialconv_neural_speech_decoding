"""v14_converged_v3 — V3DataModule (Phase D4).

Wires ``V3SessionDataset`` + the REUSED ``_SessionGroupedBatchSampler`` +
``v3_collate`` into a train loader (memo project-v3-pipeline-build-contract-
2026-07-10). Session-homogeneous batching is the static-shape prerequisite: every
batch is one session, so the ragged L1 attention pays that session's true N and the
compile caches once per session. The sampler self-shards under DDP (Lightning sets
``use_distributed_sampler=False``); ``session_size`` (per-session electrode count)
is passed so the optional straggler-balancing path is available.

``set_epoch`` fans out to BOTH the dataset (re-seed the random-continuous window
draw) and the sampler (reshuffle) so they stay phase-locked. The training loop
(E1) calls it from ``on_train_epoch_start``. NOTE: with ``persistent_workers`` the
forked workers hold a stale ``dataset._epoch``; the launch trainer refreshes windows
via ``reload_dataloaders_every_n_epochs=1`` (an E2 knob) — moot at ``num_workers=0``.
"""

from __future__ import annotations

from collections.abc import Sequence

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from speech_decoding.experiments.data import _SessionGroupedBatchSampler
from speech_decoding.models.v14_converged_v3.batch import v3_collate
from speech_decoding.models.v14_converged_v3.dataset import (
    V3SessionDataset,
    V3SessionSpec,
)


class V3DataModule(pl.LightningDataModule):
    def __init__(
        self,
        sessions: Sequence[V3SessionSpec],
        *,
        batch_size: int,
        clips_per_session: int,
        clip_frames: int,
        fps: float,
        num_workers: int = 0,
        seed: int = 0,
        drop_last: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        balance_ranks: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = V3SessionDataset(
            sessions,
            clips_per_session=clips_per_session,
            clip_frames=clip_frames,
            fps=fps,
            seed=seed,
        )
        self._session_size = {
            s.session_key: len(s.setup.sidecar.labels) for s in sessions
        }
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.balance_ranks = bool(balance_ranks)
        self._sampler: _SessionGroupedBatchSampler | None = None

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
        if self._sampler is not None:
            self._sampler.set_epoch(epoch)

    def train_dataloader(self) -> DataLoader:
        self._sampler = _SessionGroupedBatchSampler(
            self.dataset.session_key_list(),
            self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            seed=self.seed,
            session_size=self._session_size,
            balance_ranks=self.balance_ranks,
        )
        kwargs: dict = dict(
            batch_sampler=self._sampler,
            collate_fn=v3_collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self.dataset, **kwargs)
