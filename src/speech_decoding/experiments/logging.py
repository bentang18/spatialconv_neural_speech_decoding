from __future__ import annotations

import csv
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import traceback
import typing as tp
from datetime import datetime, timezone
from pathlib import Path

import pydantic


SCHEMA_VERSION = 1
EXPERIMENT_RECORD_NAME = "experiment_record.json"
EXPERIMENT_EVENTS_NAME = "experiment_events.jsonl"

RUN_RECORD_FIELDS: tuple[str, ...] = (
    "schema_version",
    "run_id",
    "parent_run_id",
    "created_at_utc",
    "completed_at_utc",
    "status",
    "stage",
    "block",
    "cell",
    "run_kind",
    "eval_mode",
    "split_mode",
    "dataset",
    "dataset_mode",
    "subject_id",
    "trial_id",
    "task",
    "seed",
    "code_commit",
    "host",
    "command",
    "config_json",
    "metrics_json",
    "primary_metric_name",
    "primary_metric_value",
    "reference_metric_value",
    "delta_from_reference",
    "exca_uid",
    "artifact_dir",
    "report_dir",
    "notes",
)


RunStatus = tp.Literal["started", "succeeded", "failed", "skipped"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def current_git_commit() -> str:
    env_commit = os.environ.get("SPEECH_CODE_COMMIT", "").strip()
    if env_commit:
        return env_commit
    sync_commit = Path(".sync_git_commit")
    if sync_commit.exists():
        value = sync_commit.read_text().strip()
        if value:
            return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def current_command() -> str:
    return shlex.join(sys.argv)


def make_run_id(
    *,
    stage: str = "neuroprobe_stage0",
    block: str = "",
    cell: str = "",
    run_kind: str = "run",
    code_commit: str | None = None,
    created_at_utc: str | None = None,
) -> str:
    created = created_at_utc or utc_now()
    stamp = re.sub(r"[^0-9]", "", created)[:14]
    commit = code_commit or current_git_commit()
    parts = [stamp, stage, block, cell or run_kind, commit]
    return "_".join(_slug(p) for p in parts if p)


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return cleaned.strip("-") or "na"


def _json_string(value: tp.Any) -> str:
    if value is None or value == "":
        return "{}"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "{}"
        json.loads(text)
        return json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class ExperimentRunRecord(pydantic.BaseModel):
    """Canonical per-run record written beside artifacts and collected into CSV."""

    model_config = pydantic.ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    parent_run_id: str = ""
    created_at_utc: str = pydantic.Field(default_factory=utc_now)
    completed_at_utc: str = ""
    status: RunStatus = "started"
    stage: str = "neuroprobe_stage0"
    block: str = ""
    cell: str = ""
    run_kind: str = "run"
    eval_mode: str = ""
    split_mode: str = ""
    dataset: str = "Wang2024Treebank"
    dataset_mode: str = "lite"
    subject_id: str = ""
    trial_id: str = ""
    task: str = ""
    seed: str = ""
    code_commit: str = pydantic.Field(default_factory=current_git_commit)
    host: str = pydantic.Field(default_factory=socket.gethostname)
    command: str = pydantic.Field(default_factory=current_command)
    config_json: str = "{}"
    metrics_json: str = "{}"
    primary_metric_name: str = ""
    primary_metric_value: float | None = None
    reference_metric_value: float | None = None
    delta_from_reference: float | None = None
    exca_uid: str = ""
    artifact_dir: str = ""
    report_dir: str = ""
    notes: str = ""

    @pydantic.field_validator("config_json", "metrics_json", mode="before")
    @classmethod
    def _validate_json_field(cls, value: tp.Any) -> str:
        return _json_string(value)

    @pydantic.model_validator(mode="after")
    def _default_run_id(self) -> ExperimentRunRecord:
        if not self.run_id:
            self.run_id = make_run_id(
                stage=self.stage,
                block=self.block,
                cell=self.cell,
                run_kind=self.run_kind,
                code_commit=self.code_commit,
                created_at_utc=self.created_at_utc,
            )
        return self

    def csv_row(self) -> dict[str, str]:
        data = self.model_dump()
        row: dict[str, str] = {}
        for field in RUN_RECORD_FIELDS:
            value = data[field]
            row[field] = "" if value is None else str(value)
        return row


class ExperimentLogger:
    """Write per-run JSON sidecars without shared-file contention."""

    def __init__(
        self,
        *,
        artifact_dir: str | Path | None = None,
        artifact_root: str | Path | None = None,
        **record_kwargs: tp.Any,
    ) -> None:
        if artifact_dir is None and artifact_root is None:
            raise ValueError("Either artifact_dir or artifact_root is required")
        self.record = ExperimentRunRecord(**record_kwargs)
        if artifact_dir is None:
            artifact_dir = Path(tp.cast(str | Path, artifact_root)) / self.record.run_id
        self.artifact_dir = Path(artifact_dir)
        self.record.artifact_dir = str(self.artifact_dir)
        self._closed = False

    @property
    def record_path(self) -> Path:
        return self.artifact_dir / EXPERIMENT_RECORD_NAME

    @property
    def events_path(self) -> Path:
        return self.artifact_dir / EXPERIMENT_EVENTS_NAME

    def __enter__(self) -> ExperimentLogger:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.log_event("started")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: tp.Any,
    ) -> bool:
        if exc is None:
            self.finish("succeeded")
            return False
        summary = "".join(traceback.format_exception_only(exc_type, exc)).strip()
        self.finish("failed", notes=summary)
        return False

    def set_metrics(
        self,
        metrics: dict[str, tp.Any],
        *,
        primary_metric_name: str = "",
        primary_metric_value: float | None = None,
        reference_metric_value: float | None = None,
    ) -> None:
        self.record.metrics_json = _json_string(metrics)
        self.record.primary_metric_name = primary_metric_name
        self.record.primary_metric_value = primary_metric_value
        self.record.reference_metric_value = reference_metric_value
        if primary_metric_value is not None and reference_metric_value is not None:
            self.record.delta_from_reference = primary_metric_value - reference_metric_value

    def log_event(
        self,
        status: RunStatus,
        *,
        notes: str = "",
        extra: dict[str, tp.Any] | None = None,
    ) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp_utc": utc_now(),
            "run_id": self.record.run_id,
            "status": status,
            "notes": notes,
            "extra": extra or {},
        }
        with self.events_path.open("a") as f:
            f.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")

    def finish(self, status: RunStatus, *, notes: str = "") -> None:
        if self._closed:
            return
        self.record.status = status
        self.record.completed_at_utc = utc_now()
        if notes:
            self.record.notes = notes
        self.log_event(status, notes=self.record.notes)
        _atomic_write_json(self.record_path, self.record.model_dump(mode="json"))
        self._closed = True

    @classmethod
    def write_skipped(
        cls,
        *,
        artifact_dir: str | Path,
        notes: str,
        **record_kwargs: tp.Any,
    ) -> ExperimentRunRecord:
        logger = cls(artifact_dir=artifact_dir, **record_kwargs)
        logger.artifact_dir.mkdir(parents=True, exist_ok=True)
        logger.finish("skipped", notes=notes)
        return logger.record


def collect_experiment_records(
    artifact_roots: tp.Iterable[str | Path],
    out_csv: str | Path,
) -> list[ExperimentRunRecord]:
    records_by_id: dict[str, ExperimentRunRecord] = {}
    for root in artifact_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for record_path in sorted(root_path.rglob(EXPERIMENT_RECORD_NAME)):
            record = ExperimentRunRecord.model_validate_json(record_path.read_text())
            records_by_id[record.run_id] = record

    records = sorted(
        records_by_id.values(),
        key=lambda rec: (rec.created_at_utc, rec.run_id),
    )
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_RECORD_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.csv_row())
    return records


def _atomic_write_json(path: Path, payload: dict[str, tp.Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)
