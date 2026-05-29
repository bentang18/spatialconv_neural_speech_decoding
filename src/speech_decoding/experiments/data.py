from __future__ import annotations

import random

import numpy as np
import pydantic
import torch
from torch.utils.data import DataLoader

import neuralset as ns


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

    return init_fn


class Data(pydantic.BaseModel):
    """Build split DataLoaders from a NeuralSet Step and Segmenter."""

    model_config = pydantic.ConfigDict(extra="forbid")

    study: ns.Step
    segmenter: ns.dataloader.Segmenter
    batch_size: int = 64
    num_workers: int = 0
    split_field: str = "split"
    splits: tuple[str, ...] = ("train", "val", "test")
    prepare: bool = True

    def build(self, *, worker_seed: int | None = None) -> dict[str, DataLoader]:
        """Build split DataLoaders.

        ``worker_seed`` (CR02): when not None, each worker's numpy / random /
        torch RNG is initialized to ``worker_seed + worker_id``. The train
        loader also receives a seeded ``Generator`` so shuffle order is
        deterministic across runs. Pass ``Experiment.seed`` from the caller.
        """
        events = self.study.run()
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
            loaders[split] = DataLoader(
                selected,
                batch_size=self.batch_size,
                shuffle=split == "train",
                num_workers=self.num_workers,
                collate_fn=selected.collate_fn,
                worker_init_fn=worker_init_fn,
                generator=train_generator if split == "train" else None,
            )
        return loaders
