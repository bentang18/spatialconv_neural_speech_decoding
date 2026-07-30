"""Re-derive the pbs double dissociation FROM THE PROBE JSONs, not from a memo line.

WHY THIS EXISTS. The pbs50-vs-pbs25 result is the reason pbs50 is the adoption candidate,
and until now every number in it was transcribed from a one-line index entry — no ledger rows,
no shard read (see project-per-band-spatial-mask-double-dissociation-2026-07-29, "PROVENANCE
GAP"). This script reads the on-disk probe JSONs and recomputes the deltas and the sign test,
so the claim comes out of data.

WHAT IT COMPUTES. Paired over CELLS, which is the board test
(project-paired-over-cells-is-the-board-test-2026-07-27):
  WS cell = (session, task), from ``ws_per_session``  -> n = 7 x 4 = 28
  CS cell = (held-out subject, task), from ``cs_per_test`` -> n = 6 x 4 = 24
Two-sided sign test, ties dropped. enc0 is the CONTROL: a pretraining-arm difference cannot
reach the frontend tap, so anything but ~all-tie at enc0 is a bug, not a result.

Usage:
  python -m scripts.neuroprobe.audit.pbs_ladder_read results/r6_era/pbs
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys

TAPS = ("enc0", "enc3", "enc6", "enc12")
TASKS = ("onset", "delta_volume", "word_index", "gpt2_surprisal")


def _sign_p(k: int, n: int) -> float:
    """Two-sided exact binomial sign test at p=0.5."""
    if n == 0:
        return float("nan")
    k = max(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _load(path: str) -> tuple[str, dict]:
    d = json.load(open(path))
    prefixes = {k.split("|")[0] for k in d}
    if len(prefixes) != 1:
        raise SystemExit(f"{path}: expected one arm prefix, got {sorted(prefixes)}")
    return prefixes.pop(), d


def _cells(d: dict, prefix: str, tap: str, field: str) -> dict[tuple[str, str], float]:
    out = {}
    for task in TASKS:
        rec = d.get(f"{prefix}|{tap}|std|{task}")
        if rec is None:
            continue
        for unit, v in rec[field].items():
            out[(unit, task)] = float(v)
    return out


def _compare(a_name, a, b_name, b, tap, field, label):
    ca, cb = _cells(a, a_name, tap, field), _cells(b, b_name, tap, field)
    keys = sorted(set(ca) & set(cb))
    if not keys:
        return
    deltas = [ca[k] - cb[k] for k in keys]
    pos = sum(d > 0 for d in deltas)
    neg = sum(d < 0 for d in deltas)
    n_eff = pos + neg
    mean = sum(deltas) / len(deltas)
    per_task = {t: sum(ca[k] - cb[k] for k in keys if k[1] == t)
                / max(sum(k[1] == t for k in keys), 1) for t in TASKS}
    tasks_won = sum(v > 0 for v in per_task.values())
    verdict = "ALL-TIE" if n_eff == 0 else f"p={_sign_p(pos, n_eff):.4f}"
    print(f"  {label} {tap:6s} d={mean:+.4f}  {pos}/{len(keys)} cells positive "
          f"(ties {len(keys) - n_eff})  {verdict}  "
          f"tasks won {tasks_won}/{len(per_task)}")
    print("         per task: " + "  ".join(f"{t}={per_task[t]:+.4f}" for t in TASKS))


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "results/r6_era/pbs"
    paths = sorted(glob.glob(os.path.join(root, "results_v3_probe_*.json")))
    if len(paths) < 2:
        raise SystemExit(f"need >=2 probe JSONs in {root}, found {len(paths)}")
    arms = {}
    for p in paths:
        name, d = _load(p)
        arms[name] = d
        step = re.search(r"_(\d+k)\.json$", p)
        print(f"[loaded] {name:14s} step={step.group(1) if step else '?'} "
              f"keys={len(d)}  {os.path.basename(p)}")

    names = sorted(arms)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"\n=== {a} MINUS {b} ===")
            for regime, field in (("WS", "ws_per_session"), ("CS", "cs_per_test")):
                for tap in TAPS:
                    _compare(a, arms[a], b, arms[b], tap, field, regime)
                print()

    print("NOTE: enc0 is the control — it must be ~all-tie. A non-tie there is a bug.")


if __name__ == "__main__":
    main()
