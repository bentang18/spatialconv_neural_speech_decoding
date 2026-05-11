"""Verify EXCA_CACHE_FOLDER convention on DCC + fire one TinyMLP Experiment.

Two checks:

1. The env var `EXCA_CACHE_FOLDER` is set, points under
   `/hpc/group/coganlab/` (persistent), and the directory exists.
2. A trivial `Experiment` (TinyMLP on synthetic `TinySplitStudy`) with
   `infra.folder = $EXCA_CACHE_FOLDER/_substrate_smoke_<ts>` runs end-to-end
   and writes its `experiment_record.json` under the expected path.

Why this exists: catching a misrouted `EXCA_CACHE_FOLDER` *now* (one trivial
synthetic Experiment) is the difference between an annoying check today and a
month of v14 sweeps disappearing into `/work/ht203/` on its 75-day purge.

Run remotely:
    scripts/dcc/dispatch scripts/dcc/verify_exca_cache.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PERSISTENT_PREFIX = "/hpc/group/coganlab/"


def _check_env_var(strict: bool) -> Path | None:
    raw = os.environ.get("EXCA_CACHE_FOLDER", "").strip()
    print(f"EXCA_CACHE_FOLDER = {raw!r}")
    if not raw:
        msg = "EXCA_CACHE_FOLDER not set. Add to ~/.bashrc on DCC."
        if strict:
            raise SystemExit(f"FAIL: {msg}")
        print(f"warn: {msg}")
        return None
    folder = Path(raw)
    if not str(folder).startswith(PERSISTENT_PREFIX):
        msg = (
            f"EXCA_CACHE_FOLDER ({folder}) is NOT under {PERSISTENT_PREFIX} — "
            "/work/ is purged every 75 days, you'll lose all cached runs."
        )
        if strict:
            raise SystemExit(f"FAIL: {msg}")
        print(f"warn: {msg}")
    if not folder.exists():
        msg = f"EXCA_CACHE_FOLDER ({folder}) does not exist."
        if strict:
            raise SystemExit(f"FAIL: {msg}")
        print(f"warn: {msg}")
    else:
        print(f"ok: directory exists at {folder}")
    return folder


def _fire_smoke(cache_root: Path) -> None:
    from speech_decoding.experiments import Experiment
    from speech_decoding.experiments.test_experiment import tiny_data
    import speech_decoding.models  # noqa: F401  # registers TinyMLP

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = cache_root / "_substrate_smoke" / f"verify_{stamp}"
    print(f"\nfiring TinyMLP smoke run at {run_root}")

    xp = Experiment(
        data=tiny_data(),
        brain_model_config={"name": "TinyMLP", "hidden": 8},
        loss={"name": "CrossEntropyLoss"},
        optim={"optimizer": {"name": "Adam", "lr": 1e-2}},
        metrics=[{
            "name": "Accuracy", "log_name": "acc",
            "kwargs": {"task": "multiclass", "num_classes": 2},
        }],
        n_epochs=1,
        accelerator="cpu",
        devices=1,
        log_every_n_steps=1,
        infra={"folder": str(run_root), "cluster": None},
    )
    result = xp.run()
    print(f"\nresult: {result}")

    records = sorted(run_root.rglob("experiment_record.json"))
    if not records:
        raise SystemExit(
            f"FAIL: no experiment_record.json under {run_root}. "
            "Cache write did not land."
        )
    record_path = records[0]
    print(f"ok: experiment_record.json at {record_path}")
    payload = json.loads(record_path.read_text())
    print(f"  status={payload['status']!r} primary={payload['primary_metric_name']!r}")

    if not str(record_path).startswith(str(cache_root)):
        raise SystemExit(
            f"FAIL: record path {record_path} does NOT live under cache_root {cache_root}."
        )
    print(f"\nPASS: cache landed under {cache_root}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--no-fire", action="store_true",
        help="Only check env var + dir; skip the TinyMLP run.",
    )
    p.add_argument(
        "--lax", action="store_true",
        help="Warn instead of failing on missing env / non-persistent path.",
    )
    args = p.parse_args()

    folder = _check_env_var(strict=not args.lax)
    if args.no_fire or folder is None:
        return
    _fire_smoke(folder)


if __name__ == "__main__":
    sys.exit(main())
