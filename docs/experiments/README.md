# Experiments — durable record of every DCC run

`/work/ht203` auto-purges every 75 days, so this directory is the *only*
surviving trace of finished runs. Every DCC dispatch must write per-run
artifacts and a `experiment_record.json` sidecar; this dir collects them.

## Files

- `runs.csv` — flat one-row-per-run table, regenerated from all
  `experiment_record.json` sidecars by `scripts/neuroprobe/collect_experiment_records.py`.
- `stage0_summary.csv` — Stage-0 freeze table (curated; not auto-generated).
- `SCHEMA.md` — pointer to §Schema below; kept for greppability.
- `v14_ablation_log.csv` — pre-NeuroAI-reset log. Frozen historical reference;
  do not extend. New runs land in `runs.csv` via the schema below.

## Pipeline

```
scripts/dcc/dispatch  →  Experiment.run()  →  ExperimentLogger (ctx mgr)
                                                 │
                                                 ├─ writes  artifact_dir/experiment_record.json
                                                 └─ writes  artifact_dir/experiment_events.jsonl
                                                              │
collect_experiment_records()  ──────────────────────────────────┘
        │
        └─ writes  docs/experiments/runs.csv  (one row per record)
```

- `ExperimentLogger` (`src/speech_decoding/experiments/logging.py`) is the only
  writer of `experiment_record.json`. The `Experiment.run()` `@infra.apply`
  body wraps `_train_and_test()` in the logger context manager so every cell
  in a sweep emits a sidecar without shared-file contention.
- `collect_experiment_records()` is a pure aggregator: it `rglob`s all
  `experiment_record.json` under the artifact roots, validates each via the
  `ExperimentRunRecord` pydantic model, and writes a CSV. Idempotent —
  re-running overwrites `runs.csv` from the JSON ground truth.
- `scripts/neuroprobe/collect_experiment_records.py` is the CLI entrypoint
  (`--artifact-root` repeatable, `--out-csv` overridable). Typical invocation:

  ```bash
  .venv/bin/python scripts/neuroprobe/collect_experiment_records.py \
    --artifact-root reports \
    --artifact-root /hpc/group/coganlab/ht203/cache_neuroai \
    --out-csv docs/experiments/runs.csv
  ```

- Pre-NeuroAI-reset ablation log + run JSONs are archived at
  `docs/archive/experiments/pre_neuroai_reset_2026_04_29/`. Do **not** extend
  that schema — it reflects the retired PS-era training loop and sbatch
  tooling. New runs land in `runs.csv` via the schema below.

## Schema

**Schema version**: `1` (locked 2026-05-13; bump on any breaking change to
field names or types).

Fields are emitted by `ExperimentRunRecord.model_dump()` and mirrored 1:1 as
columns in `runs.csv` via `RUN_RECORD_FIELDS`. Strings default to empty;
numerics default to `None` → empty cell in CSV.

| Field | Type | Default | Source | Meaning |
|---|---|---|---|---|
| `schema_version` | int | `1` | `logging.SCHEMA_VERSION` | Increment on breaking changes. |
| `run_id` | str | derived | `make_run_id(...)` | `{stamp}_{stage}_{block}_{cell}_{commit}`. Auto-derived if blank. |
| `parent_run_id` | str | `""` | caller | For sister runs / replicas referring back to a primary cell. |
| `created_at_utc` | ISO-8601 | `now()` | logger init | Emit time of the `started` event. |
| `completed_at_utc` | ISO-8601 | `""` | `finish()` | Emit time of the terminal event. |
| `status` | enum | `"started"` | `finish()` | One of `started / succeeded / failed / skipped`. |
| `stage` | str | `"neuraltrain"` (in `Experiment`) / `"neuroprobe_stage0"` (default) | caller | Coarse program stage. |
| `block` | str | `""` | caller | Sub-program block within a stage (e.g. `"L.2"`, `"v14_stage1"`). |
| `cell` | str | `""` | caller | Specific cell inside a block (e.g. `"R4xI2L"`, `"d128_depth6_M4_eps1e-2"`). |
| `run_kind` | str | `"run"` | caller | `"train" / "sweep" / "audit"` etc. — free-form. |
| `eval_mode` | str | `""` | caller | `"CrossSession" / "CrossSubject" / "WithinSession"`. |
| `split_mode` | str | `""` | caller | `"K-fold" / "chronological" / "leaderboard"`. |
| `dataset` | str | `"Wang2024Treebank"` | caller | NeuralSet study name. |
| `dataset_mode` | str | `"lite"` | caller | `"nano" / "lite" / "full"`. |
| `subject_id` | str | `""` | caller | Single subject id for per-subject runs; empty for pooled. |
| `trial_id` | str | `""` | caller | Single trial id; empty for pooled. |
| `task` | str | `""` | caller | Neuroprobe task (e.g. `"frame_perception"`). |
| `seed` | str | `""` | caller | Stringified for CSV interop. |
| `code_commit` | str | `git rev-parse --short HEAD` | `current_git_commit()` | Falls back to `$SPEECH_CODE_COMMIT` then `.sync_git_commit` then `"unknown"`. |
| `host` | str | `socket.gethostname()` | logger init | DCC node / laptop. |
| `command` | str | `shlex.join(sys.argv)` | logger init | Full invocation, for replay. |
| `config_json` | str (JSON) | `"{}"` | caller | Full Experiment config; canonicalized (sorted keys, no whitespace). `Experiment.run` passes `model_dump_json(exclude={"infra"})`. |
| `metrics_json` | str (JSON) | `"{}"` | `set_metrics()` | Full Lightning test-loop output. |
| `primary_metric_name` | str | `""` | `set_metrics()` | Name of the headline metric (e.g. `"acc"`). |
| `primary_metric_value` | float | `None` | `set_metrics()` | Headline metric value. |
| `reference_metric_value` | float | `None` | `set_metrics()` | Comparator (linear baseline / SOTA at submission). |
| `delta_from_reference` | float | `None` | derived | `primary − reference` if both present; auto-computed in `set_metrics`. |
| `exca_uid` | str | `""` | `Experiment.run` | exca's `infra.uid_folder().name`. |
| `artifact_dir` | str | `""` | logger init | Absolute path to the per-run dir. |
| `report_dir` | str | `""` | caller | Sweep-level report dir (`reports/<sweep>/...`). |
| `notes` | str | `""` | `finish()` / caller | Failure tracebacks land here; otherwise free-form. |

### Per-run sidecar — what's on disk

Each `Experiment.run()` call writes two files into `artifact_dir`:

- `experiment_record.json` — the canonical record above, written atomically
  on `finish()`. JSON-pretty, sorted keys. Overwritten in-place on retry.
- `experiment_events.jsonl` — append-only event log; one line per state
  transition (`started`, `succeeded` / `failed` / `skipped`). Each line carries
  `timestamp_utc`, `run_id`, `status`, `notes`, and free-form `extra` dict.
  Use for retry forensics; not aggregated into `runs.csv`.

### Versioning policy

- **Non-breaking changes** (adding new optional fields with `default=""` or
  `None`): keep `schema_version=1`. Downstream readers tolerate extra
  columns; `RUN_RECORD_FIELDS` and `ExperimentRunRecord` stay in sync.
- **Breaking changes** (renaming, removing, retyping): bump
  `SCHEMA_VERSION`, branch the reader in `collect_experiment_records`, and
  rebuild `runs.csv` from scratch. Old sidecars stay readable via their own
  `schema_version`.
- **NEVER** mutate a written sidecar after the fact — re-derive via re-running
  the cell instead. `_atomic_write_json` is the only sanctioned writer.

### Caller responsibilities (what `Experiment.run()` already handles)

`speech_decoding.experiments.experiment.Experiment.run` wraps the train/test
loop with `ExperimentLogger(...)` and supplies `stage`, `run_kind`, `seed`,
`exca_uid`, `config_json`, plus `primary_metric_name` / `primary_metric_value`
from the first Lightning test-loop metric. Sweep drivers (e.g.
`scripts/neuroprobe/submit_v14_stage1_grid.py`) do **not** open their own
logger — each cell's Slurm job runs `Experiment.run()` which handles its own
record.

Linear-baseline scripts (e.g. `scripts/neuroprobe/run_stage0_linear_baseline.py`)
that don't go through `Experiment.run()` must instantiate `ExperimentLogger`
directly and populate `block`, `cell`, `eval_mode`, `split_mode`, `subject_id`,
`trial_id`, `task`, `reference_metric_value` themselves — `Experiment.run`
leaves those empty because they're not generic.

### Pointers

- Pydantic model + writer: `src/speech_decoding/experiments/logging.py`.
- `Experiment.run` integration: `src/speech_decoding/experiments/experiment.py`.
- Aggregator CLI: `scripts/neuroprobe/collect_experiment_records.py`.
- Tests: `src/speech_decoding/experiments/test_logging.py`.
