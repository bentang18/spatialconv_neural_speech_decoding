#!/usr/bin/env python
"""Does a 64 Hz HGA stream beat the 32 Hz one at enc0? Read straight off the shards.

THE COMPARISON IS THREE ARMS, NOT TWO. A 64 Hz bake differs from the published 32 Hz enc0 on
three axes at once — rate (64 vs 32), window (31.25 vs 62.5 ms) and bins (4 vs 7) — so
"64 beats published" cannot say WHICH axis paid. The subsampled arm is the control that
separates them:

    native64          HGA at 64 Hz                      (4,16,64) x (7,6,4)
    sub32             the SAME bake, strided back to 32  same window, same bins, half the rate
    published         the shipped 32 Hz enc0             (4,16,32) x (7,6,7)

    native64 - sub32       isolates RATE      (window and bins held fixed)
    sub32    - published   isolates WINDOW+BINS at matched rate

Shards, never a merged JSON: the merge collapses each grid entry to a cohort mean, and a mean
over cells cannot be paired against another arm's mean. The per-cell values are the paired unit
and they only exist here.

NULL, stated before the number. Arms share cells, tasks, splits, ridge and lambda rule; only the
HGA columns differ. So the paired per-cell difference has null mean 0, and the decision map is:
  |mean diff| within the paired 95% CI of 0  -> NO EFFECT. 64 Hz does not pay, do not rebake.
  CI excludes 0 and diff > 0                 -> rate pays; size it against the ~.0095 CS lead
                                                before calling it worth a frontend change.
  CI excludes 0 and diff < 0                 -> 64 Hz HURTS at enc0.
Visual tasks are reported separately and excluded from the headline: they are noise, and a macro
that averages them in dilutes any real effect toward 0.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np

# Copied VERBATIM from v3_board_readout.py:92 — the 15 tasks the board macro is defined over.
# Order is irrelevant to a mean, but a missing or extra name silently redefines the headline
# number, so this is a transcription and not a re-derivation.
BOARD_TASKS = (
    "onset", "speech", "volume", "delta_volume", "pitch", "word_index",
    "word_gap", "gpt2_surprisal", "word_head_pos", "word_part_speech",
    "word_length", "global_flow", "local_flow", "frame_brightness", "face_num",
)
# paper_figs_r6.py:654 — "all visual is noise" (Ben 2026-08-01). Kept in the 15-task macro
# because the BOARD defines it that way, but reported apart so a real effect is not diluted.
VISUAL = ("frame_brightness", "global_flow", "local_flow", "face_num")
MODES = ("ws", "csession", "cs")


def load(shard_dir: str, mode: str, gk: str) -> dict[str, dict[str, float]]:
    """{cell: {task: test AUROC}} for one grid key, straight from the shard files."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for path in sorted(glob.glob(f"{shard_dir}/{mode}_*.json")):
        with open(path) as f:
            sh = json.load(f)
        for task, val in sh["cells"].items():
            cells = val.get("cells") or {}
            if gk in cells:
                out[sh["name"]][task] = float(cells[gk]["test"])
    return dict(out)


def macro(per_cell: dict[str, dict[str, float]], tasks) -> float:
    """Board macro: cohort-mean each task over cells, then mean over tasks (_finalize's order).

    Not the mean of per-cell macros — those differ whenever the grid has holes, and quoting the
    convenient one is how a number stops being comparable to the leaderboard."""
    per_task = [np.nanmean([c[t] for c in per_cell.values() if t in c])
                for t in tasks if any(t in c for c in per_cell.values())]
    return float(np.nanmean(per_task)) if per_task else float("nan")


def cell_macros(per_cell, tasks) -> dict[str, float]:
    """Per-cell mean over tasks — the PAIRED unit."""
    return {name: float(np.nanmean([v[t] for t in tasks if t in v]))
            for name, v in per_cell.items() if any(t in v for t in tasks)}


def paired(a: dict[str, float], b: dict[str, float]) -> tuple:
    """mean diff, 95% CI, n, and the per-cell win count. CI is the normal-approx paired
    interval; with n=10-12 it is indicative, so the win count is printed beside it rather
    than a p-value pretending to more resolution than 12 cells carry."""
    shared = sorted(set(a) & set(b))
    d = np.array([a[c] - b[c] for c in shared], dtype=float)
    if len(d) < 2:
        return float("nan"), (float("nan"), float("nan")), len(d), 0, shared
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return m, (m - 1.96 * se, m + 1.96 * se), len(d), int((d > 0).sum()), shared


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--native", required=True, help="shard dir for the native 64 Hz arm")
    p.add_argument("--sub32", required=True, help="shard dir for the subsampled 32 Hz control")
    p.add_argument("--published", default=None,
                   help="optional shard dir for the shipped 32 Hz enc0")
    p.add_argument("--tap", default="enc0_elec",
                   help="base tap; cs uses the parcel tap, ws/csession the electrode tap")
    p.add_argument("--cs-tap", default="enc0")
    p.add_argument("--fm-arm", default="hga64t32",
                   help="the fm: arm name the sub32 shards were written under")
    a = p.parse_args()

    print(f"{'':10s} {'arm':22s} {'macro15':>9s} {'no-visual':>10s} {'visual':>8s}  cells")
    per_mode = {}
    for mode in MODES:
        base = a.cs_tap if mode == "cs" else a.tap
        arms = {"native64": (a.native, f"{base}|std"),
                "sub32": (a.sub32, f"fm:{a.fm_arm}:{base}|std")}
        if a.published:
            arms["published"] = (a.published, f"{base}|std")
        loaded = {}
        for name, (d, gk) in arms.items():
            pc = load(d, mode, gk)
            if not pc:
                print(f"{mode:10s} {name:22s} {'—':>9s} {'—':>10s} {'—':>8s}  MISSING "
                      f"(no {mode}_*.json with key {gk!r} in {d})")
                continue
            loaded[name] = pc
            nv = [t for t in BOARD_TASKS if t not in VISUAL]
            print(f"{mode:10s} {name:22s} {macro(pc, BOARD_TASKS):9.4f} "
                  f"{macro(pc, nv):10.4f} {macro(pc, VISUAL):8.4f}  {len(pc)}")
        per_mode[mode] = loaded

    nv = [t for t in BOARD_TASKS if t not in VISUAL]
    for label, x, y, what in (("RATE (64 vs 32, window+bins held)", "native64", "sub32",
                               "a positive diff means the RATE pays"),
                              ("WINDOW+BINS (at matched 32 Hz)", "sub32", "published",
                               "a positive diff means the 31.25 ms window pays")):
        print(f"\n=== {label} — paired over cells, null = 0 ===\n    {what}")
        for mode in MODES:
            L = per_mode.get(mode, {})
            if x not in L or y not in L:
                print(f"  {mode:10s} —  need both {x} and {y}")
                continue
            for tag, tasks in (("macro15", BOARD_TASKS), ("no-visual", nv)):
                m, (lo, hi), n, wins, _ = paired(cell_macros(L[x], tasks),
                                                 cell_macros(L[y], tasks))
                verdict = "NO EFFECT (CI spans 0)" if not (lo > 0 or hi < 0) else (
                    "HELPS" if m > 0 else "HURTS")
                print(f"  {mode:10s} {tag:10s} diff={m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  "
                      f"n={n:2d}  won {wins}/{n}  -> {verdict}")


if __name__ == "__main__":
    main()
