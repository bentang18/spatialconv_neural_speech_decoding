from __future__ import annotations

import logging
import os
import random

import numpy as np
import pydantic
import torch
from torch.utils.data import DataLoader

import neuralset as ns

from speech_decoding.experiments.bad_windows import (
    filter_events_by_bad_windows,
    load_bad_windows,
)

logger = logging.getLogger(__name__)


# Warm dataset-OBJECT cache (throughput lever, env-gated by ``V14_WARM_DATASET_CACHE``).
# In-process memo of the built split-DataLoader dict, keyed on the full Data config +
# worker_seed, so a warm worker skips dataset RE-MATERIALIZATION on repeat runs of the
# same data config — ``study.run()`` + ``segmenter.apply()`` + ``dataset.prepare()`` +
# per-split ``select`` + DataLoader/worker construction are the ~tens-of-seconds the
# warm-worker startup can't otherwise amortize. LRU size 1 (the warm 4-GPU worker runs
# one data config repeatedly). FLOPs- and LOSS-neutral: the returned loaders still draw
# a REAL fresh batch every step (nothing is cached at the batch level; the shuffle
# generator continues its RNG across reuse). Off by default → byte-identical to the
# uncached path.
_DATASET_CACHE: dict[str, dict] = {}


class _SessionGroupedBatchSampler(torch.utils.data.Sampler):
    """Yield batches whose samples all share one ``(subject_id, trial_id)`` session.

    Every event in a BrainTreebank session has the **same** electrode count C, so a
    session-homogeneous batch needs zero electrode padding. The ragged converged
    forward then pays that session's true C — not the corpus max-C — and since the
    latent encoder's all-pairs attention is O(C²) in electrode count, padding a
    mixed batch up to the corpus max (179) is the dominant wasted FLOP. Grouping
    removes it (throughput lever ``Data.group_by_session``).

    Science-neutral for the per-sample SSL loss: over an epoch every sample is
    drawn exactly once (no resampling, no dropping unless ``drop_last``); only the
    intra-step batch composition changes, and grad-accum across the step's
    microbatches restores cross-session mixing. The per-sample M2/M4 L1 losses have
    no cross-sample coupling (LayerNorm only, no BatchNorm), so the per-step
    gradient is the same mean-of-per-sample-grads regardless of batch composition.

    ``session_key`` is the positional per-segment session id (len == dataset). With
    ``shuffle`` the within-session order and the cross-session batch order are both
    permuted from ``generator`` each epoch; without it the order is deterministic.
    """

    def __init__(self, session_key, batch_size, *, shuffle, drop_last, generator):
        self._groups: dict = {}
        for idx, key in enumerate(session_key):
            self._groups.setdefault(key, []).append(idx)
        self._batch_size = int(batch_size)
        self._shuffle = bool(shuffle)
        self._drop_last = bool(drop_last)
        self._generator = generator

    def _build_batches(self) -> list[list[int]]:
        g = self._generator
        batches: list[list[int]] = []
        for idxs in self._groups.values():
            order = list(idxs)
            if self._shuffle:
                perm = torch.randperm(len(order), generator=g).tolist()
                order = [order[i] for i in perm]
            for i in range(0, len(order), self._batch_size):
                batch = order[i : i + self._batch_size]
                if self._drop_last and len(batch) < self._batch_size:
                    continue
                batches.append(batch)
        if self._shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]
        return batches

    def __iter__(self):
        return iter(self._build_batches())

    def __len__(self) -> int:
        total = 0
        for idxs in self._groups.values():
            if self._drop_last:
                total += len(idxs) // self._batch_size
            else:
                total += (len(idxs) + self._batch_size - 1) // self._batch_size
        return total


def _make_worker_init_fn(seed: int):
    """Per-worker RNG seeding (CR02). Sets numpy + random + torch RNGs to
    ``seed + worker_id`` so augmentation / masking inside workers is
    deterministic across runs without relying on Lightning's worker-init
    coverage of every RNG family."""

    def init_fn(worker_id: int) -> None:
        local_seed = seed + worker_id
        np.random.seed(local_seed)
        random.seed(local_seed)
        torch.manual_seed(local_seed)
        # 2026-05-30 speedup audit (Tier-1): cap each DataLoader worker's
        # torch intra-op thread pool to 1. The per-__getitem__ STFT runs
        # via torch.stft (torch threads); with num_workers workers each
        # defaulting to all cores, the pool oversubscribes whenever
        # --cpus-per-task > num_workers, slowing the very path the workers
        # exist to overlap. One thread/worker removes the contention.
        # In-process so it works under submitit (login-node env never
        # reaches the GPU job). Numerics-safe: torch.stft is thread-count
        # independent.
        torch.set_num_threads(1)

    return init_fn


class Data(pydantic.BaseModel):
    """Build split DataLoaders from a NeuralSet Step and Segmenter."""

    model_config = pydantic.ConfigDict(extra="forbid")

    study: ns.Step
    segmenter: ns.dataloader.Segmenter
    batch_size: int = 64
    # num_workers=0 runs the per-sample extractor stack (CAR + torch.stft in
    # LogStftView/MultiStftView is recomputed per __getitem__ — only the raw
    # waveform load is MapInfra-cached) single-threaded in the main process,
    # which starves the GPU (1.48 it/s on the 5/29 Lite Phase-4 baseline).
    # Dispatch overrides this; the >0 path also enables persistent workers +
    # prefetch + pinned host buffers so the CPU STFT overlaps GPU compute.
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    split_field: str = "split"
    splits: tuple[str, ...] = ("train", "val", "test")
    prepare: bool = True
    # Layer-2 bad-electrode defense (pretrain-only): directory of per-session
    # bad-time-window sidecars (``scripts/neuroprobe/precompute_bad_windows.py``). When
    # set, clips whose neural window overlaps a glitch span are dropped BEFORE the
    # segmenter, so they are never sampled. Default None = no filtering — the Phase-4
    # eval datamodule leaves it unset so the Neuroprobe parity-locked clip sets are
    # never altered. It removes events (rows), never electrodes, so DP4 row-alignment
    # is untouched. Part of the warm-cache key (a different dir → a different cache).
    bad_window_dir: str | None = None
    # Throughput lever (science-neutral): when True, the TRAIN loader draws
    # session-homogeneous batches (all samples share one subject_id/trial_id), so
    # the ragged forward pays each batch's true electrode count C instead of padding
    # a mixed batch up to the corpus max-C. The latent encoder's attention is O(C²)
    # in electrode count, so this is the dominant wasted FLOP on a mixed batch.
    # Per-sample SSL loss has no cross-sample coupling, so this changes only
    # intra-step batch composition (grad-accum restores cross-session mixing), not
    # the epoch-level sample set. Eval (val/test) splits are never grouped — the
    # Neuroprobe parity-locked clip sets stay byte-identical. See
    # ``_SessionGroupedBatchSampler``. Off by default = byte-identical loader.
    group_by_session: bool = False

    def build(self, *, worker_seed: int | None = None) -> dict[str, DataLoader]:
        """Build split DataLoaders.

        ``worker_seed`` (CR02): when not None, each worker's numpy / random /
        torch RNG is initialized to ``worker_seed + worker_id``. The train
        loader also receives a seeded ``Generator`` so shuffle order is
        deterministic across runs. Pass ``Experiment.seed`` from the caller.
        """
        # Warm dataset-OBJECT cache (env-gated). On a hit, return the already-built
        # loaders so the warm worker pays zero dataset-materialization on repeat runs
        # of the same data config. The loaders still yield fresh batches per step.
        cache_on = os.environ.get("V14_WARM_DATASET_CACHE") == "1"
        cache_key: str | None = None
        if cache_on:
            try:
                cache_key = f"{self.model_dump_json()}|worker_seed={worker_seed}"
            except Exception:
                cache_key = None  # unserializable config → build normally, no cache
            if cache_key is not None and cache_key in _DATASET_CACHE:
                return _DATASET_CACHE[cache_key]

        events = self.study.run()
        if self.bad_window_dir is not None:
            bad_windows = load_bad_windows(self.bad_window_dir)
            n_before = len(events)
            events = filter_events_by_bad_windows(
                events,
                bad_windows,
                clip_start_s=float(self.segmenter.start),
                clip_dur_s=float(self.segmenter.duration),
                # Only the segmenter's trigger (Word) rows are clip anchors. The
                # continuous Ieeg data row (start=0) must survive — dropping it on an
                # early bad span would orphan the whole session (missing-Ieeg error).
                trigger_query=self.segmenter.trigger_query,
            )
            logger.info(
                "bad-window clip filter (%s): dropped %d / %d events overlapping "
                "%d session glitch spans",
                self.bad_window_dir,
                n_before - len(events),
                n_before,
                sum(len(v) for v in bad_windows.values()),
            )
        dataset = self.segmenter.apply(events)
        if self.prepare:
            dataset.prepare()

        if self.split_field not in dataset.triggers.columns:
            raise ValueError(f"Missing trigger split column: {self.split_field!r}")

        worker_init_fn = (
            _make_worker_init_fn(worker_seed) if worker_seed is not None else None
        )
        train_generator = None
        if worker_seed is not None:
            train_generator = torch.Generator().manual_seed(worker_seed)

        loaders: dict[str, DataLoader] = {}
        for split in self.splits:
            selected = dataset.select(dataset.triggers[self.split_field] == split)
            if len(selected) == 0:
                raise ValueError(f"Split {split!r} has no segments")
            loader_kwargs: dict = dict(
                num_workers=self.num_workers,
                collate_fn=selected.collate_fn,
                worker_init_fn=worker_init_fn,
                pin_memory=self.pin_memory,
            )
            if self.group_by_session and split == "train":
                # Session-homogeneous TRAIN batches (zero electrode padding). Eval
                # splits keep the parity-locked plain loader untouched.
                triggers = selected.triggers
                if len(triggers) != len(selected):
                    raise ValueError(
                        "group_by_session: triggers rows "
                        f"({len(triggers)}) != segments ({len(selected)}); "
                        "cannot positionally align session keys."
                    )
                if not {"subject_id", "trial_id"} <= set(triggers.columns):
                    raise ValueError(
                        "group_by_session needs subject_id/trial_id triggers; "
                        f"got columns {list(triggers.columns)}"
                    )
                session_key = list(
                    zip(triggers["subject_id"].tolist(), triggers["trial_id"].tolist())
                )
                loader_kwargs["batch_sampler"] = _SessionGroupedBatchSampler(
                    session_key,
                    self.batch_size,
                    shuffle=True,
                    drop_last=False,
                    generator=train_generator,
                )
            else:
                loader_kwargs["batch_size"] = self.batch_size
                loader_kwargs["shuffle"] = split == "train"
                loader_kwargs["generator"] = (
                    train_generator if split == "train" else None
                )
            # persistent_workers / prefetch_factor are only valid when workers
            # are spawned; torch raises ValueError if they are passed with
            # num_workers=0.
            if self.num_workers > 0:
                loader_kwargs["persistent_workers"] = self.persistent_workers
                loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loaders[split] = DataLoader(selected, **loader_kwargs)

        if cache_on and cache_key is not None:
            _DATASET_CACHE.clear()  # LRU size 1
            _DATASET_CACHE[cache_key] = loaders
        return loaders
