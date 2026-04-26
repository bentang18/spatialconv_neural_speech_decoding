#!/usr/bin/env python3
"""Verify the local repo and DCC repo are in sync before submitting work.

Usage:
    scripts/ablation/dcc_sync_check.py
    scripts/ablation/dcc_sync_check.py --strict   # exit non-zero on any drift

Output (one block):

    local:  73849ec  branch=main  (clean)
    dcc:    73849ec  branch=main  (clean)
    origin: 73849ec
    status: in sync

When out of sync:

    local:  abc1234  branch=main  (3 modified, 2 untracked)
    dcc:    def5678  branch=main  (1 modified)
    origin: abc1234
    status: DRIFT — DCC HEAD ≠ local HEAD; rsync recommended
            DCC dirty: docs/foo.md  scripts/bar.sh

Exits 0 when fully in sync, 1 on drift (always under --strict; otherwise just
prints the warning so you can choose). Designed to be folded into submit.py
as a pre-flight or run on its own.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from _common import REMOTE_REPO, ssh


def _local_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _local_branch() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()


def _local_dirty() -> tuple[int, int, list[str]]:
    out = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).splitlines()
    modified = [l for l in out if l[:2].strip() in {"M", "A", "D", "R", "MM", "AM"}]
    untracked = [l for l in out if l.startswith("??")]
    sample = [l[3:] for l in (modified + untracked)[:5]]
    return len(modified), len(untracked), sample


def _origin_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "origin/main"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def _dcc_state() -> dict:
    cmd = (
        f"cd {REMOTE_REPO} && "
        "echo SHA=$(git rev-parse HEAD) && "
        "echo BR=$(git rev-parse --abbrev-ref HEAD) && "
        "echo --- && "
        "git status --porcelain"
    )
    out = ssh(cmd, check=False).splitlines()
    sha = ""
    branch = ""
    dirty: list[str] = []
    in_dirty = False
    for line in out:
        if line.startswith("SHA="):
            sha = line.split("=", 1)[1].strip()
        elif line.startswith("BR="):
            branch = line.split("=", 1)[1].strip()
        elif line.strip() == "---":
            in_dirty = True
        elif in_dirty and line.strip():
            dirty.append(line)
    return {"sha": sha, "branch": branch, "dirty": dirty}


def _fmt_dirty(modified: int, untracked: int) -> str:
    if not (modified or untracked):
        return "clean"
    bits = []
    if modified:
        bits.append(f"{modified} modified")
    if untracked:
        bits.append(f"{untracked} untracked")
    return ", ".join(bits)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on any drift (HEAD mismatch or DCC dirty).",
    )
    args = p.parse_args()

    local_sha = _local_sha()
    local_branch = _local_branch()
    l_mod, l_unt, l_sample = _local_dirty()

    dcc = _dcc_state()
    dcc_sha = dcc["sha"]
    dcc_branch = dcc["branch"]
    dcc_dirty = dcc["dirty"]

    origin = _origin_sha()

    print(f"local:  {local_sha[:7]}  branch={local_branch}  ({_fmt_dirty(l_mod, l_unt)})")
    print(f"dcc:    {dcc_sha[:7]}  branch={dcc_branch}  ({len(dcc_dirty)} modified)")
    if origin:
        print(f"origin: {origin[:7]}")

    drift_reasons: list[str] = []
    if dcc_sha and local_sha != dcc_sha:
        drift_reasons.append("DCC HEAD ≠ local HEAD; rsync recommended")
    if dcc_dirty:
        drift_reasons.append("DCC has uncommitted changes")
    if l_mod or l_unt:
        drift_reasons.append("local has uncommitted changes (rsync will push them)")

    if not drift_reasons:
        print("status: in sync")
        return 0

    print("status: DRIFT — " + "; ".join(drift_reasons))
    if dcc_dirty:
        sample = "  ".join(d[3:] for d in dcc_dirty[:5])
        more = f" (+{len(dcc_dirty) - 5} more)" if len(dcc_dirty) > 5 else ""
        print(f"        DCC dirty: {sample}{more}")
    if l_sample:
        print(f"        local dirty (first 5): {' '.join(l_sample)}")

    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
