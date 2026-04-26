#!/usr/bin/env python3
"""PostToolUse hook: when a Bash tool call ran `sbatch ...`, parse the SLURM
job id from stdout and append a minimal record to .ablation_submissions.jsonl.

CLAUDE.md mandates aggregating every DCC run into v14_ablation_log.csv because
/work/ht203 auto-purges after 75 days. scripts/ablation/submit.py already
records to .ablation_submissions.jsonl, but hand-written `sbatch` calls (LOPO
chains, smoke runs) bypass that. This hook closes the gap so no submission
disappears from the long-term log.
"""
import json
import re
import sys
import time
from pathlib import Path


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    cmd = (payload.get("tool_input") or {}).get("command") or ""
    if not re.search(r"\bsbatch\b", cmd):
        return 0

    stdout = (payload.get("tool_response") or {}).get("stdout") or ""
    m = re.search(r"Submitted batch job (\d+)", stdout)
    if not m:
        return 0
    job_id = m.group(1)

    log = Path(".ablation_submissions.jsonl")
    record = {
        "ts": int(time.time()),
        "job_id": job_id,
        "command": cmd.strip(),
        "source": "hook",
    }
    with log.open("a") as f:
        f.write(json.dumps(record) + "\n")

    print(
        f"[hook] recorded sbatch job {job_id} → .ablation_submissions.jsonl. "
        f"Run `.venv/bin/python scripts/ablation/collect.py {job_id}` after it lands.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
