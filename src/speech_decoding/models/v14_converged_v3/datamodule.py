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
(E1) calls it from ``on_train_epoch_start``. ``persistent_workers`` DEFAULTS OFF:
``reload_dataloaders_every_n_epochs=1`` (the launch E2 knob) recreates the loader each
epoch anyway, so persistence buys nothing here AND the smoke warned that
``pin_memory=True`` + ``persistent_workers=True`` + ``reload>0`` hits a documented
PyTorch instability (pytorch/pytorch#91252). Off removes the risk at no throughput cost
(``pin_memory`` stays on). With persistence on, the forked workers would also hold a
stale ``dataset._epoch`` — reload=1 is what refreshes windows regardless. Moot at
``num_workers=0``.
"""

from __future__ import annotations

from collections.abc import Sequence

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from speech_decoding.experiments.data import _SessionGroupedBatchSampler
from speech_decoding.models.v14_converged_v3.batch import v3_collate
from speech_decoding.models.v14_converged_v3.dataset import (
    UNIFORM_BAND_RATES,
    V3SessionDataset,
    V3SessionSpec,
)
from speech_decoding.models.v14_converged_v3.shaft_batch import collate_shaft_pack
from speech_decoding.models.v14_converged_v3.shaft_dataset import ShaftPackDataset


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
        persistent_workers: bool = False,
        prefetch_factor: int = 4,
        balance_ranks: bool = False,
        same_session: bool = False,
        band_rates: Sequence[tuple[int, int]] = UNIFORM_BAND_RATES,
        batch_unit: str = "session",
        contact_budget: int | None = None,
        shaft_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        # batch_unit: "session" (v3 session-homogeneous batching) or "shaft" (cross-patient
        # shaft-level batching — the r5 default; K distinct patients/step from the global
        # shaft pool). The shaft path streams B=1 super-montage packs; the session path keeps
        # the reused _SessionGroupedBatchSampler. Everything else (module, model) is shared.
        self.batch_unit = str(batch_unit)
        if self.batch_unit not in ("session", "shaft"):
            raise ValueError(f"batch_unit must be 'session' or 'shaft', got {batch_unit!r}")
        if self.batch_unit == "shaft":
            if contact_budget is None:
                raise ValueError("batch_unit='shaft' requires contact_budget (the per-pack "
                                 "contact count that pins grid.total to one compiled shape)")
            # packs/epoch: reuse the session clip budget × n_sessions as the per-epoch step
            # count (max_steps bounds training anyway; the stream draws fresh packs each step).
            self.dataset = ShaftPackDataset(
                sessions,
                contact_budget=int(contact_budget),
                clip_frames=clip_frames,
                fps=fps,
                band_rates=band_rates,
                packs_per_epoch=int(clips_per_session) * max(1, len(sessions)),
                alpha=float(shaft_alpha),
                seed=seed,
            )
        else:
            self.dataset = V3SessionDataset(
                sessions,
                clips_per_session=clips_per_session,
                clip_frames=clip_frames,
                fps=fps,
                seed=seed,
                band_rates=band_rates,
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
        # same_session (v2 `--same-session-ranks`): all DDP ranks step the SAME session
        # each step ⇒ identical N/shape across ranks. Under `--compile` this keeps every
        # rank on the SAME compiled variant at the same step (no recompile-desync stall
        # at the all-reduce barrier) AND removes the straggler. Multi-GPU only (world>1).
        self.same_session = bool(same_session)
        self._sampler: _SessionGroupedBatchSampler | None = None

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
        if self._sampler is not None:
            self._sampler.set_epoch(epoch)

    def train_dataloader(self) -> DataLoader:
        if self.batch_unit == "shaft":
            # One pack → one V3Batch super-montage: the IterableDataset yields a
            # list[ShaftClipSample] per step; batch_size=None disables auto-batching so
            # collate_shaft_pack runs on that list directly. The dataset self-shards across
            # DDP ranks/workers (disjoint seeds), so no DistributedSampler is needed.
            kwargs: dict = dict(
                batch_size=None,
                collate_fn=collate_shaft_pack,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            if self.num_workers > 0:
                kwargs["persistent_workers"] = self.persistent_workers
                kwargs["prefetch_factor"] = self.prefetch_factor
            return DataLoader(self.dataset, **kwargs)

        self._sampler = _SessionGroupedBatchSampler(
            self.dataset.session_key_list(),
            self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            seed=self.seed,
            session_size=self._session_size,
            balance_ranks=self.balance_ranks,
            same_session=self.same_session,
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
