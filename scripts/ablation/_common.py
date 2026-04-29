"""Shared DCC plumbing for ablation submit / status / collect / logs.

Conventions hard-coded here so the CLIs stay short:

- DCC host         : ht203@dcc-login.oit.duke.edu
- Remote repo      : /work/ht203/repo/speech
- Remote results   : /work/ht203/results
- Remote logs      : /work/ht203/logs/v14core
- Python on DCC    : /work/ht203/repo/speech/.venv/bin/python
- Local repo       : the parent of this file's grandparent
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

DCC_USER = "ht203"
DCC_HOST = f"{DCC_USER}@dcc-login.oit.duke.edu"
REMOTE_REPO = "/work/ht203/repo/speech"
REMOTE_RESULTS = "/work/ht203/results"
REMOTE_LOGS = "/work/ht203/logs/v14core"
REMOTE_PY = "/work/ht203/repo/speech/.venv/bin/python"

LOCAL_REPO = Path(__file__).resolve().parents[2]
SUBMIT_LOG = LOCAL_REPO / ".ablation_submissions.jsonl"


def ssh(cmd: str, *, capture: bool = True, check: bool = True) -> str:
    """Run a command on DCC. Returns stdout (capture=True) or empty string."""
    full = ["ssh", "-o", "BatchMode=yes", DCC_HOST, cmd]
    if capture:
        r = subprocess.run(full, capture_output=True, text=True, check=False)
        if check and r.returncode != 0:
            sys.stderr.write(f"ssh failed: {cmd}\nstderr: {r.stderr}\n")
            sys.exit(r.returncode)
        return r.stdout
    return subprocess.run(full, check=check).stdout or ""  # type: ignore[return-value]


def rsync_repo(extra_excludes: Iterable[str] = ()) -> None:
    """Push local repo to DCC, mirroring text+code only.

    Excludes data/, results/, .venv/, .git/objects, large caches. Uses --delete
    so renames don't leave stale files behind. Loud on stderr only on failure.
    """
    excludes = [
        ".venv",
        ".git",
        "__pycache__",
        "*.pyc",
        "data/atlas/kernel_exploration_2026_04_17",
        "data/mni_coords",
        "data/fsaverage_coords",
        "data/channel_maps",
        "data/transforms",
        "*.fif",
        "*.npz",
        "*.pt",
        "*.ckpt",
        "node_modules",
        ".DS_Store",
        ".pytest_cache",
        ".ablation_submissions.jsonl",
        # Machine-specific paths — DCC holds its own copy. Clobbering it
        # with the Mac paths makes every subsequent job fail with
        # FileNotFoundError on /Users/... paths.
        "configs/paths.yaml",
        *extra_excludes,
    ]
    args = ["rsync", "-az", "--delete"]
    for e in excludes:
        args += ["--exclude", e]
    args += [f"{LOCAL_REPO}/", f"{DCC_HOST}:{REMOTE_REPO}/"]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"rsync failed:\n{r.stderr}\n")
        sys.exit(r.returncode)


def parse_folds(spec: str) -> list[int]:
    """'0-4' or '0,2,4' or '3' → [int]."""
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def parse_flag_pairs(items: list[str]) -> dict[str, str]:
    """['d_model=64', 'backbone_depth=5'] → {'d_model': '64', ...}."""
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            sys.exit(f"bad --flag (need k=v): {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def render_flag_args(flags: dict[str, str]) -> str:
    """{'d_model': '64', 'no-parcel-embedding': ''} → CLI arg string.

    Empty value means a bare flag (action='store_true'). Keys use underscores
    or hyphens; we emit hyphens since that's what argparse expects on this CLI.
    """
    parts: list[str] = []
    for k, v in flags.items():
        cli = "--" + k.replace("_", "-")
        if v == "":
            parts.append(cli)
        else:
            parts.append(f"{cli} {shlex.quote(v)}")
    return " ".join(parts)


def log_submission(record: dict) -> None:
    """Append one submission to .ablation_submissions.jsonl for later lookup."""
    SUBMIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SUBMIT_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


def lookup_submission(job_id: str) -> dict | None:
    if not SUBMIT_LOG.exists():
        return None
    for line in SUBMIT_LOG.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(r.get("job_id")) == str(job_id):
            return r
    return None
