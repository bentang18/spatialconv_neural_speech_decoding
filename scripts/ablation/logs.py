#!/usr/bin/env python3
"""Peek at job logs without dragging the whole file into context.

Usage:
    scripts/ablation/logs.py 12345                  # tail of task 0 .err + .out
    scripts/ablation/logs.py 12345 --task 7         # specific array task
    scripts/ablation/logs.py 12345 --grep ERROR     # grep across all tasks
    scripts/ablation/logs.py 12345 --failed         # tail logs of failed tasks
    scripts/ablation/logs.py 12345 --tail 50        # last 50 lines (default 30)
    scripts/ablation/logs.py 12345 --stream         # follow .out of task 0

Default behavior is conservative — never dumps a full log to stdout. --stream
runs ssh tail -f directly (you'll see the live tail; ctrl-c to stop).
"""

from __future__ import annotations

import argparse
import re
import sys

from _common import REMOTE_LOGS, lookup_submission, ssh


def _log_paths(name: str, job_id: str, task_id: int) -> tuple[str, str]:
    out = f"{REMOTE_LOGS}/{name}_{job_id}_{task_id}.out"
    err = f"{REMOTE_LOGS}/{name}_{job_id}_{task_id}.err"
    return out, err


def _tail_one(path: str, n: int) -> str:
    return ssh(f"tail -n {n} {path} 2>/dev/null || echo '(missing: {path})'")


def _failed_task_ids(job_id: str) -> list[str]:
    out = ssh(f"sacct -j {job_id} -X -n -P -o JobID,State", check=False)
    bad = []
    for line in out.splitlines():
        if "|" not in line:
            continue
        jid, state = line.split("|", 1)
        state = state.split(" ")[0].strip()
        if state in {"FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"}:
            m = re.match(r"\d+_(\d+)", jid)
            if m:
                bad.append(m.group(1))
    return bad


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job_id")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--tail", type=int, default=30)
    p.add_argument("--grep", default=None, help="ripgrep across all tasks of the job")
    p.add_argument("--failed", action="store_true", help="tail .err of all failed tasks")
    p.add_argument("--stream", action="store_true", help="ssh tail -f the .out of one task")
    p.add_argument("--err-only", action="store_true")
    args = p.parse_args()

    rec = lookup_submission(args.job_id)
    name = rec["name"] if rec else None
    if not name:
        sys.exit(f"no submission record for {args.job_id} — can't construct log paths")

    if args.grep:
        pat = args.grep.replace("'", "'\\''")
        cmd = f"grep -nH -E '{pat}' {REMOTE_LOGS}/{name}_{args.job_id}_*.{{out,err}} | head -200"
        print(ssh(cmd, check=False).rstrip() or "(no matches)")
        return 0

    if args.stream:
        out, _ = _log_paths(name, args.job_id, args.task)
        # exec ssh directly so ctrl-c reaches it cleanly
        import os
        os.execvp("ssh", ["ssh", "-t", "ht203@dcc-login.oit.duke.edu", f"tail -f {out}"])

    if args.failed:
        bad = _failed_task_ids(args.job_id)
        if not bad:
            print("no failed tasks")
            return 0
        for t in bad:
            _, err = _log_paths(name, args.job_id, int(t))
            print(f"--- task {t} (err, last {args.tail}) ---")
            print(_tail_one(err, args.tail).rstrip())
        return 0

    out, err = _log_paths(name, args.job_id, args.task)
    if not args.err_only:
        print(f"--- task {args.task} .out (last {args.tail}) ---")
        print(_tail_one(out, args.tail).rstrip())
    print(f"--- task {args.task} .err (last {args.tail}) ---")
    print(_tail_one(err, args.tail).rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
