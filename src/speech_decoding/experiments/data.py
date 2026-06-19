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
    permuted each epoch (seeded by ``seed + epoch``); without it the order is
    deterministic.

    **DDP self-sharding.** A custom batch_sampler is NOT a ``BatchSampler`` subclass,
    so Lightning cannot auto-inject a ``DistributedSampler`` on top of it — the owning
    Trainer therefore sets ``use_distributed_sampler=False`` and the sampler shards
    itself. Every rank builds the SAME full batch list (the shuffle is seeded by
    ``seed + epoch``, rank-INDEPENDENT), then takes a disjoint ``[rank::world]`` stride
    of an even-padded copy. So each rank draws an EQUAL count of session-homogeneous
    batches (no DDP uneven-input hang) and every sample is still seen once per epoch
    across the cohort. ``world`` / ``rank`` are read LAZILY from ``torch.distributed``
    at iterate/len time, because the loader is built before the DDP process group
    exists (so they cannot be captured at construction).

    **Rank size-balancing** (``balance_ranks`` + ``session_size``, OFF by default).
    Session-homogeneous batches make compute *vary per batch*: the ragged forward's
    O(C²) electrode attention costs the batch's true C, and sessions span ~94–179
    electrodes. Under DDP the ``W`` ranks step in lockstep at the gradient all-reduce,
    so when the ``[rank::world]`` stride hands different-C batches to the ranks of one
    micro-step, the small-C ranks idle at the barrier (straggler). Balancing removes
    it: sort all batches by cost (the session's electrode count), chunk the sorted
    list into groups of ``W`` near-equal-cost batches, shuffle the GROUP ORDER
    (rank-independent ``seed + epoch`` → SGD randomness preserved at group
    granularity), then the SAME ``[rank::world]`` stride gives each rank one member of
    each group. So the ``W`` batches that run simultaneously at micro-step ``j`` are
    group ``j`` — cost-matched, no straggler. ``__len__`` is unchanged (one batch per
    group == ``ceil(n_batches / W)``). Science-neutral: still every sample once/epoch
    (the per-sample SSL loss has no cross-sample coupling). Needs a true cost proxy —
    ``session_size`` maps ``(subject, trial) -> electrode count``; absent it (or
    ``world <= 1``) the path no-ops back to the plain stride.
    """

    def __init__(
        self, session_key, batch_size, *, shuffle, drop_last,
        seed=0, num_replicas=None, rank=None,
        session_size=None, balance_ranks=False,
    ):
        self._session_key = list(session_key)
        self._groups: dict = {}
        for idx, key in enumerate(session_key):
            self._groups.setdefault(key, []).append(idx)
        self._batch_size = int(batch_size)
        self._shuffle = bool(shuffle)
        self._drop_last = bool(drop_last)
        self._seed = int(seed)
        self._epoch = 0
        # None => detect lazily from torch.distributed at iterate/len time.
        self._num_replicas = num_replicas
        self._rank = rank
        self._session_size = dict(session_size) if session_size is not None else None
        self._balance_ranks = bool(balance_ranks)

    def set_epoch(self, epoch: int) -> None:
        # Lightning calls this at each train-epoch start. Reshuffles
        # deterministically AND identically across ranks (seed + epoch), so the
        # disjoint sharding below stays a clean partition.
        self._epoch = int(epoch)

    def _dist(self) -> tuple[int, int]:
        if self._num_replicas is not None:
            return int(self._num_replicas), int(self._rank or 0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, 0

    def _all_batches(self) -> list[list[int]]:
        # Rank-independent full list: same seed+epoch on every rank.
        g = torch.Generator().manual_seed(self._seed + self._epoch)
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

    def _batch_cost(self, batch: list[int]) -> int:
        # Cost proxy = the batch's session electrode count (drives the O(C²)
        # ragged forward). All samples share one session, so batch[0] suffices.
        # Missing key -> 0 (degrades to the small end; never raises).
        return int(self._session_size.get(self._session_key[batch[0]], 0))

    def _balanced_groups(self, world: int) -> list[list[list[int]]] | None:
        # Returns groups of exactly `world` cost-matched batches, or None to
        # signal "fall back to the plain stride". Each group's W members run
        # simultaneously across the W ranks at one micro-step.
        if not self._balance_ranks or self._session_size is None or world <= 1:
            return None
        batches = self._all_batches()
        n = len(batches)
        if n == 0:
            return []
        # Stable sort by cost; _all_batches already shuffled, so equal-cost
        # batches keep a rank-independent random order (SGD noise within a tier).
        order = sorted(range(n), key=lambda i: self._batch_cost(batches[i]))
        sorted_batches = [batches[i] for i in order]
        groups = [sorted_batches[g : g + world] for g in range(0, n, world)]
        # Pad a partial final group (the largest-cost tail) by wrapping within
        # itself, so it stays near-equal-cost and every rank gets one member.
        last = groups[-1]
        if len(last) < world:
            k = len(last)
            groups[-1] = last + [last[i % k] for i in range(world - k)]
        # Shuffle GROUP ORDER only (rank-independent): preserves SGD randomness
        # at group granularity while keeping each group cost-matched. Distinct
        # generator stream from _all_batches so the two shuffles decorrelate.
        gen = torch.Generator().manual_seed(self._seed + self._epoch + 0x5A1A)
        perm = torch.randperm(len(groups), generator=gen).tolist()
        return [groups[i] for i in perm]

    def _sharded(self) -> list[list[int]]:
        world, rank = self._dist()
        groups = self._balanced_groups(world)
        if groups is not None:
            # Stride is implicit: rank r takes the r-th member of every group.
            return [grp[rank] for grp in groups]
        batches = self._all_batches()
        if world <= 1:
            return batches
        n = len(batches)
        per_rank = (n + world - 1) // world
        padded_total = per_rank * world
        if n > 0 and padded_total > n:
            # wrap-around pad (like DistributedSampler) → equal count per rank.
            batches = batches + [batches[i % n] for i in range(padded_total - n)]
        return batches[rank:padded_total:world]

    def __iter__(self):
        return iter(self._sharded())

    def __len__(self) -> int:
        total = 0
        for idxs in self._groups.values():
            if self._drop_last:
                total += len(idxs) // self._batch_size
            else:
                total += (len(idxs) + self._batch_size - 1) // self._batch_size
        world, _ = self._dist()
        if world <= 1:
            return total
        return (total + world - 1) // world


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
    # Throughput lever (DDP straggler removal; requires group_by_session). When True,
    # the train sampler buckets sessions by electrode count C so the W batches the W
    # DDP ranks run simultaneously at each gradient all-reduce have matched C — the
    # small-C ranks no longer idle at the barrier while a big-C rank finishes its
    # O(C²) forward. Science-neutral (still every sample once/epoch; only which ranks
    # co-run which sessions changes). Needs a true cost proxy: built at loader time
    # from ``voltage_electrode_order`` (the exact per-session post-static-drop count);
    # if BraintreeBank data / ROOT_DIR_BRAINTREEBANK is absent the proxy can't be
    # built and the sampler safely no-ops back to the plain stride. Off by default.
    balance_ranks_by_size: bool = False

    def _session_electrode_counts(self, sessions) -> dict | None:
        """Map each ``(subject_id, trial_id)`` to its post-static-drop electrode
        count C — the O(C²) cost proxy the rank-balancer sorts on. Reads only the
        small per-session electrode tables via ``voltage_electrode_order``. Returns
        ``None`` (balancing no-ops) if BraintreeBank data / the root env is absent,
        so an accidentally-on flag in a non-DCC build degrades instead of crashing."""
        bt_root = os.environ.get("ROOT_DIR_BRAINTREEBANK")
        if not bt_root:
            logger.warning(
                "balance_ranks_by_size: ROOT_DIR_BRAINTREEBANK unset; cannot size "
                "sessions — rank-balancing disabled (plain DDP stride)."
            )
            return None
        from speech_decoding.studies.braintreebank.anatomy import (
            voltage_electrode_order,
        )
        try:
            sizes: dict = {}
            for subject_id, trial_id in sessions:
                sizes[(subject_id, trial_id)] = len(
                    voltage_electrode_order(bt_root, int(subject_id), int(trial_id))
                )
            return sizes
        except Exception as exc:  # missing files, etc. — degrade, don't crash
            logger.warning(
                "balance_ranks_by_size: failed to size sessions (%s) — "
                "rank-balancing disabled (plain DDP stride).", exc,
            )
            return None

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
                session_size = (
                    self._session_electrode_counts(set(session_key))
                    if self.balance_ranks_by_size
                    else None
                )
                loader_kwargs["batch_sampler"] = _SessionGroupedBatchSampler(
                    session_key,
                    self.batch_size,
                    shuffle=True,
                    drop_last=False,
                    # rank-INDEPENDENT seed so every DDP rank builds the same
                    # full batch list before self-sharding (see class docstring).
                    seed=worker_seed if worker_seed is not None else 0,
                    session_size=session_size,
                    balance_ranks=self.balance_ranks_by_size,
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
