#!/usr/bin/env python3
"""Summarize SLURM array status for one or more job IDs. One block per job.

Usage:
    scripts/ablation/status.py 12345
    scripts/ablation/status.py 12345 12346 12347
    scripts/ablation/status.py            # all jobs from .ablation_submissions.jsonl

Output (per job):
    12345  q1e_d64                running   8/15 done  5 run  2 pend  0 fail
        failed: -
        elapsed: 1h12m   est-done: ~38m

If a fold/seed has failed, the failing tasks are listed (decoded back to
fold/seed if the submission spec is on file).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

from _common import SUBMIT_LOG, lookup_submission, ssh


SQUEUE_FMT = "%i %T %M %R"  # JobID(possibly array) State Time Reason


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


def _squeue_state(job_id: str) -> dict[str, int]:
    """Counter over states for the array tasks of one job_id, from live squeue."""
    out = ssh(f'squeue -h -j {job_id} -o "{SQUEUE_FMT}"', check=False)
    counts: Counter[str] = Counter()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        counts[parts[1]] += 1
    return dict(counts)


def _sacct_done(job_id: str) -> dict[str, int]:
    """Counter over completed states from sacct (for finished/failed tasks)."""
    out = ssh(
        f"sacct -j {job_id} -X -n -P -o JobID,State",
        check=False,
    )
    counts: Counter[str] = Counter()
    for line in out.splitlines():
        if "|" not in line:
            continue
        _, state = line.split("|", 1)
        # sacct can include "CANCELLED by ..." — strip suffix
        state = state.split(" ")[0].strip()
        counts[state] += 1
    return dict(counts)


def _failed_task_ids(job_id: str) -> list[str]:
    """Extract array task ids whose final state is FAILED/CANCELLED/TIMEOUT."""
    out = ssh(f"sacct -j {job_id} -X -n -P -o JobID,State", check=False)
    bad = []
    for line in out.splitlines():
        if "|" not in line:
            continue
        jid, state = line.split("|", 1)
        state = state.split(" ")[0].strip()
        if state in {"FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"}:
            # jid like "12345_3" — keep just the index
            m = re.match(r"\d+_(\d+)", jid)
            if m:
                bad.append(m.group(1))
    return bad


def _decode_task(task_id: int, spec: dict | None) -> str:
    if not spec:
        return f"task={task_id}"
    folds = spec.get("folds", [])
    seeds = spec.get("seeds", [])
    n_seeds = len(seeds) or 1
    seed_idx = task_id % n_seeds
    fold_idx = task_id // n_seeds
    fold = folds[fold_idx] if fold_idx < len(folds) else "?"
    seed = seeds[seed_idx] if seed_idx < len(seeds) else "?"
    return f"fold={fold},seed={seed}"


def _summarize(job_id: str) -> None:
    rec = lookup_submission(job_id)
    name = rec["name"] if rec else "?"
    n_total = rec["n_jobs"] if rec else None

    live = _squeue_state(job_id)
    done = _sacct_done(job_id)

    running = live.get("RUNNING", 0)
    pending = live.get("PENDING", 0) + live.get("CONFIGURING", 0)
    completed = done.get("COMPLETED", 0)
    failed = sum(
        done.get(s, 0) for s in ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL")
    )

    if not (live or done):
        print(f"{job_id}  {name}  no record (sacct empty / squeue empty)")
        return

    n_known = max(n_total or 0, completed + failed + running + pending)
    overall = (
        "done"
        if running == 0 and pending == 0 and (completed + failed) > 0
        else "running"
        if running > 0
        else "pending"
    )

    head = (
        f"{job_id}  {name:<24} {overall:<8} "
        f"{completed}/{n_known} done  {running} run  {pending} pend  {failed} fail"
    )
    print(head)

    if failed:
        bad = _failed_task_ids(job_id)
        decoded = ", ".join(_decode_task(int(t), rec.get("spec") if rec else None) for t in bad[:8])
        more = f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""
        print(f"    failed: {decoded}{more}")


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job_ids", nargs="*", help="Job ids. Empty → all known.")
    args = p.parse_args()
    job_ids = args.job_ids or _all_known_jobs()
    if not job_ids:
        print("no job ids given and .ablation_submissions.jsonl is empty")
        return 1
    for jid in job_ids:
        _summarize(jid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
