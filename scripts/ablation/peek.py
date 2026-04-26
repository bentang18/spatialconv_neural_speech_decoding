#!/usr/bin/env python3
"""Cat one or more result.json files from DCC without rsyncing the tree.

Usage:
    scripts/ablation/peek.py 12345                    # list result.json files for the job
    scripts/ablation/peek.py 12345 --task 0           # show one task's result
    scripts/ablation/peek.py 12345 --fold 2 --seed 1  # decode (fold, seed) → task → result
    scripts/ablation/peek.py 12345 --all              # one-line summary per result.json

Output (single result):
    pooled_S14_S26_S33_S62_fold2_seed1_depth3.result.json
      test_per_phoneme = 0.812
      slot_avg         = 0.853
      best_epoch       = 14
      wall_s           = 412.7

Output (--all):
    fold0_seed0  PER 0.798  slot 0.846  ep 12  410s
    fold0_seed1  PER 0.802  slot 0.851  ep 13  408s
    ...

Designed for quick spot checks while a job is mid-flight, before collect.py
pulls everything down. Falls back gracefully when the result.json doesn't yet
exist (still-running tasks).
"""

from __future__ import annotations

import argparse
import json
import sys

from _common import lookup_submission, ssh

_INTERESTING_KEYS = (
    "test_per_phoneme",
    "slot_avg",
    "best_epoch",
    "best_val_loss",
    "wall_s",
    "n_train_samples",
    "n_test_samples",
)


def _list_results(out_dir: str) -> list[str]:
    out = ssh(f"ls -1 {out_dir}/*.result.json 2>/dev/null", check=False)
    return [l.strip() for l in out.splitlines() if l.strip()]


def _read_result(path: str) -> dict | None:
    out = ssh(f"cat {path} 2>/dev/null", check=False)
    if not out.strip():
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def _decode_to_task(spec: dict, fold: int, seed: int) -> int | None:
    folds = spec.get("folds", [])
    seeds = spec.get("seeds", [])
    if fold not in folds or seed not in seeds:
        return None
    fold_idx = folds.index(fold)
    seed_idx = seeds.index(seed)
    return fold_idx * len(seeds) + seed_idx


def _task_to_filename_glob(spec: dict, task: int) -> str:
    folds = spec.get("folds", [])
    seeds = spec.get("seeds", [])
    n_seeds = len(seeds) or 1
    seed_idx = task % n_seeds
    fold_idx = task // n_seeds
    fold = folds[fold_idx] if fold_idx < len(folds) else "?"
    seed = seeds[seed_idx] if seed_idx < len(seeds) else "?"
    return f"*_fold{fold}_seed{seed}_*.result.json"


def _format_one(name: str, data: dict) -> str:
    lines = [name]
    for k in _INTERESTING_KEYS:
        if k in data:
            v = data[k]
            if isinstance(v, float):
                lines.append(f"  {k:<18} = {v:.4g}")
            else:
                lines.append(f"  {k:<18} = {v}")
    return "\n".join(lines)


def _format_summary(name: str, data: dict) -> str:
    short = name.rsplit("/", 1)[-1].replace(".result.json", "")
    parts = []
    if "test_per_phoneme" in data:
        parts.append(f"PER {data['test_per_phoneme']:.3f}")
    if "slot_avg" in data:
        parts.append(f"slot {data['slot_avg']:.3f}")
    if "best_epoch" in data:
        parts.append(f"ep {data['best_epoch']}")
    if "wall_s" in data:
        parts.append(f"{data['wall_s']:.0f}s")
    return f"{short}  " + "  ".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job_id")
    p.add_argument("--task", type=int, help="Array task id (0-based)")
    p.add_argument("--fold", type=int, help="Used with --seed to decode → task")
    p.add_argument("--seed", type=int, help="Used with --fold to decode → task")
    p.add_argument("--all", action="store_true", help="One-line summary per result.json")
    args = p.parse_args()

    rec = lookup_submission(args.job_id)
    if not rec:
        sys.exit(f"no submission record for {args.job_id}")
    out_dir = rec["out_dir"]
    spec = rec.get("spec", {})

    files = _list_results(out_dir)
    if not files:
        print(f"(no result.json under {out_dir} yet)")
        return 0

    if args.all:
        for path in files:
            data = _read_result(path)
            if data:
                print(_format_summary(path, data))
        return 0

    target_task = args.task
    if target_task is None and args.fold is not None and args.seed is not None:
        target_task = _decode_to_task(spec, args.fold, args.seed)
        if target_task is None:
            sys.exit(f"(fold={args.fold}, seed={args.seed}) not in spec folds/seeds")

    if target_task is None:
        print(f"{len(files)} result.json file(s) under {out_dir}:")
        for path in files:
            print(f"  {path.rsplit('/', 1)[-1]}")
        print("rerun with --task N or --fold F --seed S to inspect one")
        return 0

    glob = _task_to_filename_glob(spec, target_task)
    matches = [p for p in files if _glob_match(p.rsplit("/", 1)[-1], glob)]
    if not matches:
        print(f"(no result.json matching {glob} for task {target_task} yet)")
        return 0
    for path in matches:
        data = _read_result(path)
        if not data:
            print(f"({path}: empty / unreadable)")
            continue
        print(_format_one(path, data))
    return 0


def _glob_match(name: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatch(name, pattern)


if __name__ == "__main__":
    sys.exit(main())
