from __future__ import annotations

import pydantic
from torch.utils.data import DataLoader

import neuralset as ns


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

    def build(self) -> dict[str, DataLoader]:
        events = self.study.run()
        dataset = self.segmenter.apply(events)
        if self.prepare:
            dataset.prepare()

        if self.split_field not in dataset.triggers.columns:
            raise ValueError(f"Missing trigger split column: {self.split_field!r}")

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
            )
        return loaders
