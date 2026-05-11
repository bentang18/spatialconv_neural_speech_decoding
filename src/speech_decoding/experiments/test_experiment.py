from __future__ import annotations

import json

import pandas as pd

import neuralset as ns

from speech_decoding.experiments import Data, Experiment


class TinySplitStudy(ns.Step):
    def _run(self):
        rows = []
        splits = ["train"] * 4 + ["val"] * 2 + ["test"] * 2
        for idx, split in enumerate(splits):
            rows.append(
                {
                    "type": "Stimulus",
                    "start": float(idx * 2),
                    "duration": 0.2,
                    "timeline": "run0",
                    "code": idx % 2,
                    "split": split,
                }
            )
        return ns.events.standardize_events(pd.DataFrame(rows))


def tiny_data(batch_size: int = 2) -> Data:
    return Data(
        study=TinySplitStudy(),
        segmenter={
            "extractors": {
                "input": {
                    "name": "Pulse",
                    "event_types": "Stimulus",
                    "frequency": 16.0,
                    "aggregation": "single",
                },
                "target": {
                    "name": "EventField",
                    "event_types": "Stimulus",
                    "event_field": "code",
                },
            },
            "trigger_query": "type == 'Stimulus'",
            "start": 0.0,
            "duration": 1.0,
        },
        batch_size=batch_size,
    )


def test_data_builds_split_loaders() -> None:
    loaders = tiny_data().build()

    assert set(loaders) == {"train", "val", "test"}
    batch = next(iter(loaders["train"]))
    assert batch.data["input"].shape == (2, 1, 16)
    assert batch.data["target"].shape == (2, 1)


def test_experiment_dry_run(tmp_path) -> None:
    run_root = tmp_path / "runs"
    xp = Experiment(
        data=tiny_data(),
        brain_model_config={
            "name": "SimpleConvTimeAgg",
            "hidden": 4,
            "depth": 1,
            "kernel_size": 3,
            "merger_config": None,
        },
        loss={"name": "CrossEntropyLoss"},
        optim={"optimizer": {"name": "Adam", "lr": 1e-3}},
        metrics=[
            {
                "name": "Accuracy",
                "log_name": "acc",
                "kwargs": {"task": "multiclass", "num_classes": 2},
            }
        ],
        n_epochs=1,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        infra={"folder": str(run_root), "cluster": None},
    )

    result = xp.run()

    assert isinstance(result, dict)
    records = list(run_root.rglob("experiment_record.json"))
    assert len(records) == 1


def _tiny_mlp_xp(run_root, n_epochs: int) -> Experiment:
    return Experiment(
        data=tiny_data(),
        brain_model_config={"name": "TinyMLP", "hidden": 8},
        loss={"name": "CrossEntropyLoss"},
        optim={"optimizer": {"name": "Adam", "lr": 1e-2}},
        metrics=[
            {
                "name": "Accuracy",
                "log_name": "acc",
                "kwargs": {"task": "multiclass", "num_classes": 2},
            }
        ],
        n_epochs=n_epochs,
        accelerator="cpu",
        devices=1,
        log_every_n_steps=1,
        infra={"folder": str(run_root), "cluster": None},
    )


def test_substrate_smoke_full_fit_caches(tmp_path) -> None:
    """End-to-end substrate integration: TinyMLP fits via Lightning, exca caches.

    Asserts (a) Experiment.run returns real test metrics, (b) ExperimentLogger
    writes one record per uid, (c) the second `.run()` call hits the exca cache
    instead of re-fitting (no second record dir, identical return value).
    """
    import speech_decoding.models  # noqa: F401  # registers TinyMLP discriminator
    run_root = tmp_path / "runs"

    first = _tiny_mlp_xp(run_root, n_epochs=2).run()
    assert isinstance(first, dict)
    assert "test_loss" in first

    records = sorted(run_root.rglob("experiment_record.json"))
    assert len(records) == 1, [str(r) for r in records]

    record_payload = json.loads(records[0].read_text())
    assert record_payload["status"] == "succeeded"
    assert record_payload["primary_metric_name"] == "test_loss"
    assert isinstance(record_payload["primary_metric_value"], float)
    assert record_payload["exca_uid"]

    uid_dirs_before = sorted(p for p in run_root.rglob("*") if p.is_dir())

    second = _tiny_mlp_xp(run_root, n_epochs=2).run()
    assert second == first  # exca cache hit
    uid_dirs_after = sorted(p for p in run_root.rglob("*") if p.is_dir())
    assert uid_dirs_before == uid_dirs_after
