#!/usr/bin/env python3
"""Pull result JSONs from DCC, run the aggregator, print the new CSV rows.

Usage:
    scripts/ablation/collect.py 12345                   # one job
    scripts/ablation/collect.py 12345 12346             # multiple
    scripts/ablation/collect.py --all                   # everything in submission log
    scripts/ablation/collect.py 12345 --no-aggregate    # rsync only

Output: one block per job — number of *.result.json synced, number of new CSV
rows written. The local mirror lives at /tmp/v14_results/<name>/ and the CSV
diff is appended to docs/experiments/v14_ablation_log.csv via update_ablation_log.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _common import DCC_HOST, LOCAL_REPO, SUBMIT_LOG, lookup_submission

LOCAL_MIRROR = Path("/tmp/v14_results")
CSV_PATH = LOCAL_REPO / "docs" / "experiments" / "v14_ablation_log.csv"
AGGREGATOR = LOCAL_REPO / "scripts" / "v14_core" / "update_ablation_log.py"


def _all_known_jobs() -> list[str]:
    if not SUBMIT_LOG.exists():
        return []
    out = []
    seen = set()
    for line in SUBMIT_LOG.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        jid = str(r.get("job_id", ""))
        if jid and jid not in seen:
            seen.add(jid)
            out.append(jid)
    return out


def _rsync_one(rec: dict) -> int:
    """Rsync a job's out_dir to LOCAL_MIRROR/<name>. Return file count."""
    out_dir = rec["out_dir"]
    name = rec["name"]
    local = LOCAL_MIRROR / name
    local.mkdir(parents=True, exist_ok=True)
    args = [
        "rsync", "-az",
        "--include", "*.result.json",
        "--include", "*/",
        "--exclude", "*",
        f"{DCC_HOST}:{out_dir}/",
        f"{local}/",
    ]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"rsync failed for {name}: {r.stderr}\n")
        return 0
    n = sum(1 for _ in local.rglob("*.result.json"))
    return n


def _run_aggregator() -> str:
    """Run the existing aggregator and return its stdout (typically a row count)."""
    if not AGGREGATOR.exists():
        return "(aggregator not found)"
    r = subprocess.run(
        [
            sys.executable, str(AGGREGATOR),
            "--results-root", str(LOCAL_MIRROR),
            "--csv", str(CSV_PATH),
        ],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return f"aggregator failed: {r.stderr.strip()}"
    return r.stdout.strip() or "(no output)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job_ids", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--no-aggregate", action="store_true")
    args = p.parse_args()

    job_ids = _all_known_jobs() if args.all else args.job_ids
    if not job_ids:
        sys.exit("pass job ids or --all (with .ablation_submissions.jsonl populated)")

    total_files = 0
    for jid in job_ids:
        rec = lookup_submission(jid)
        if not rec:
            print(f"{jid}  no submission record — skipping (use --out-dir overrides manually)")
            continue
        n = _rsync_one(rec)
        total_files += n
        print(f"{jid}  {rec['name']:<24} synced {n} result.json")

    if args.no_aggregate:
        return 0

    summary = _run_aggregator()
    print(f"aggregator: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
