from __future__ import annotations

import csv
import json

import pytest

from speech_decoding.experiments.logging import (
    EXPERIMENT_EVENTS_NAME,
    EXPERIMENT_RECORD_NAME,
    RUN_RECORD_FIELDS,
    ExperimentLogger,
    ExperimentRunRecord,
    collect_experiment_records,
)


def test_experiment_logger_writes_success_record(tmp_path) -> None:
    artifact_dir = tmp_path / "success"

    with ExperimentLogger(
        artifact_dir=artifact_dir,
        block="A",
        cell="A0",
        config_json={"threshold": 1},
    ) as logger:
        logger.set_metrics(
            {"accuracy": 0.75},
            primary_metric_name="accuracy",
            primary_metric_value=0.75,
            reference_metric_value=0.7,
        )

    record_path = artifact_dir / EXPERIMENT_RECORD_NAME
    events_path = artifact_dir / EXPERIMENT_EVENTS_NAME

    record = json.loads(record_path.read_text())
    assert record["status"] == "succeeded"
    assert record["block"] == "A"
    assert record["cell"] == "A0"
    assert record["config_json"] == '{"threshold":1}'
    assert record["metrics_json"] == '{"accuracy":0.75}'
    assert record["delta_from_reference"] == pytest.approx(0.05)

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [event["status"] for event in events] == ["started", "succeeded"]


def test_experiment_logger_writes_failed_record(tmp_path) -> None:
    artifact_dir = tmp_path / "failed"

    with pytest.raises(RuntimeError, match="boom"):
        with ExperimentLogger(artifact_dir=artifact_dir, block="B"):
            raise RuntimeError("boom")

    record = json.loads((artifact_dir / EXPERIMENT_RECORD_NAME).read_text())
    assert record["status"] == "failed"
    assert "RuntimeError: boom" in record["notes"]


def test_experiment_logger_writes_skipped_record(tmp_path) -> None:
    record = ExperimentLogger.write_skipped(
        artifact_dir=tmp_path / "skipped",
        block="D",
        cell="D.9",
        notes="A5 did not confirm depth-wm.csv",
    )

    assert record.status == "skipped"
    assert record.notes == "A5 did not confirm depth-wm.csv"
    on_disk = ExperimentRunRecord.model_validate_json(
        (tmp_path / "skipped" / EXPERIMENT_RECORD_NAME).read_text()
    )
    assert on_disk.status == "skipped"


def test_collect_experiment_records_is_idempotent(tmp_path) -> None:
    root = tmp_path / "artifacts"
    with ExperimentLogger(artifact_root=root, block="A", cell="A1"):
        pass
    with ExperimentLogger(artifact_root=root, block="A", cell="A2"):
        pass

    out_csv = tmp_path / "runs.csv"
    first = collect_experiment_records([root], out_csv)
    second = collect_experiment_records([root], out_csv)

    assert [record.run_id for record in first] == [record.run_id for record in second]
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert list(rows[0]) == list(RUN_RECORD_FIELDS)
