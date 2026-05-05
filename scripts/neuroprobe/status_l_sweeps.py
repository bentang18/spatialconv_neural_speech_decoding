"""Print live status of an L-sweep dispatch.

Reads `launch_manifest.csv` from a dated sweep root, queries `sacct` for each
job_id, and reports per-cell counts of {PENDING, RUNNING, COMPLETED, FAILED,
TIMEOUT, CANCELLED, OTHER}. Run on DCC.

Usage:
    .venv/bin/python scripts/neuroprobe/status_l_sweeps.py \\
        reports/neuroprobe_stage0_l1_normalization_2026_05_05/
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    args = _parse_args()
    manifest = args.sweep_root.resolve() / "launch_manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"missing {manifest}")
    rows = list(csv.DictReader(manifest.open()))
    job_ids = [r["job_id"] for r in rows if r.get("job_id")]
    if not job_ids:
        raise SystemExit("no job_ids recorded in manifest")

    states = sacct_states(job_ids)
    by_cell: dict[str, Counter[str]] = defaultdict(Counter)
    by_norm: dict[str, Counter[str]] = defaultdict(Counter)
    overall: Counter[str] = Counter()
    for row in rows:
        st = states.get(row["job_id"], "OTHER")
        by_cell[row["cell"]][st] += 1
        by_norm[row.get("normalization", "")][st] += 1
        overall[st] += 1

    width = max(len(c) for c in by_cell) + 2
    print(f"\nL-sweep status — {args.sweep_root.name}")
    print(f"{'cell':<{width}} {'PENDING':>8} {'RUNNING':>8} {'DONE':>6} {'FAIL':>5} {'OTHER':>6}")
    for cell in sorted(by_cell):
        c = by_cell[cell]
        print(
            f"{cell:<{width}} {c.get('PENDING', 0):>8} {c.get('RUNNING', 0):>8} "
            f"{c.get('COMPLETED', 0):>6} {c.get('FAILED', 0) + c.get('TIMEOUT', 0) + c.get('CANCELLED', 0):>5} "
            f"{sum(v for k, v in c.items() if k not in {'PENDING','RUNNING','COMPLETED','FAILED','TIMEOUT','CANCELLED'}):>6}"
        )
    print(
        f"\noverall: PENDING={overall.get('PENDING',0)} RUNNING={overall.get('RUNNING',0)} "
        f"COMPLETED={overall.get('COMPLETED',0)} "
        f"FAILED={overall.get('FAILED',0) + overall.get('TIMEOUT',0) + overall.get('CANCELLED',0)} "
        f"OTHER={sum(v for k, v in overall.items() if k not in {'PENDING','RUNNING','COMPLETED','FAILED','TIMEOUT','CANCELLED'})}"
    )


def sacct_states(job_ids: list[str]) -> dict[str, str]:
    cmd = ["sacct", "-X", "-n", "-P", "-o", "JobID,State", "-j", ",".join(job_ids)]
    out = subprocess.run(cmd, check=True, text=True, capture_output=True).stdout
    states: dict[str, str] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        jid = parts[0].split(".")[0]
        state = parts[1].split()[0]
        states[jid] = state
    return states


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
