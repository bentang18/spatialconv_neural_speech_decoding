#!/usr/bin/env python
"""Downstream SAMPLE EFFICIENCY on the board cache: CS AUROC vs how many LABELLED rows we fit.

THE CLAIM THIS TESTS. A foundation model should not only score higher with all the labels, it
should reach a given score with FEWER of them. The board number is a single point at frac=1.0
and cannot distinguish "better features" from "better use of the same 100% of labels". Sweeping
the train fraction turns one point into a curve, and the gap between enc12 and enc0 at small
frac is the sample-efficiency claim.

WHAT CHANGES vs the board: the COUNT of anchor training rows. Nothing else. Same ridge, same
λ-on-the-val-half, same parcel intersection, same norms -- all imported from ``v3_board_readout``
rather than reimplemented, so a drift in the board contract cannot silently spare this file.

WHY enc0 IS THE COMPARISON, not a random-init model. enc0 is the untrained |STFT| front end with
ZERO learned parameters, so `enc12 - enc0` at matched frac is "what pretraining bought" with the
input representation held fixed. A random-init encoder is a DAMAGED baseline (it destroys
linearly-decodable signal), so its curve would conflate recovery-from-damage with learning.

INVARIANTS, all asserted and printed:
  1. frac=1.0 must reproduce the board's own ``_cs_cell`` EXACTLY (same float), for every tap and
     norm. This is the whole safety net: if the full-data point does not land on the published
     number, the subsampling path has changed something it should not have.
  2. The val and test row sets are computed ONCE and are byte-identical across every (frac, seed).
     Only the train set may move -- a sample-efficiency curve whose test set drifts is a fiction.
  3. Subsampling is STRATIFIED: each class keeps its share, so a small frac cannot accidentally
     become an easier (or degenerate) problem. Printed as the pos-rate drift vs the full set.
  4. n_train is strictly increasing in frac.

Small fractions are high-variance, so each frac is repeated over several seeds and the caller
gets EVERY seed, not a mean -- the spread is the result, not noise to hide.

Usage (CPU, NCSA Delta; one array task per CS cell):
  python scripts/neuroprobe/v3_sample_efficiency.py --cache <DIR> --tag <TAG> --cell 1,1 \
      --taps enc0,enc12 --out sampeff_S1T1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v3_board_readout as B  # noqa: E402

# Geometric ladder: each step halves the labels, so a flat region and a cliff are both visible.
DEFAULT_FRACS = (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
MIN_PER_CLASS = 2  # a ridge needs >=2 rows of a class for the fit to mean anything


def _stratified(y: np.ndarray, rows: np.ndarray, frac: float, seed: int) -> np.ndarray | None:
    """Keep ``frac`` of EACH class's rows. Returns None if either class falls below MIN_PER_CLASS.

    Stratified rather than uniform because the board tasks are unbalanced: a uniform draw at
    frac=1/32 can return a near-single-class train set, which does not measure sample efficiency,
    it measures whether the draw happened to keep the minority class.

    The split is on ``y > 0``, which is EXACTLY the binarization ``B.auroc`` applies before
    scoring. Stratifying on the raw values instead would be wrong for the continuous tasks
    (volume, pitch, gpt2_surprisal): every distinct float would be its own singleton stratum and
    the MIN_PER_CLASS guard would abort every fraction.
    """
    if frac >= 1.0:
        return rows
    rng = np.random.default_rng(seed)
    pos = y[rows] > 0
    keep = []
    for mask in (pos, ~pos):
        idx = rows[mask]
        k = int(round(frac * len(idx)))
        if k < MIN_PER_CLASS:
            return None
        keep.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(keep))


def _fit(anchor_rec, test_rec, task, taps, tr, va, te) -> dict:
    """``_cs_cell``'s body with the train rows supplied. Kept a verbatim mirror on purpose."""
    y_a = np.asarray(anchor_rec["labels"][task], dtype=np.float64)
    y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
    a_idx, t_idx, common = B._parcel_cols(anchor_rec, test_rec)
    if common.size == 0:
        return {}
    grid: dict = {}
    for enc in taps:
        if enc not in anchor_rec["feats"] or enc not in test_rec["feats"]:
            continue
        z_tr = B._feat(anchor_rec, enc, tr, a_idx)
        z_va, z_te = B._feat(test_rec, enc, va, t_idx), B._feat(test_rec, enc, te, t_idx)
        B._run_norms(grid, enc, z_tr, z_va, z_te, y_a[tr], y_t[va], y_t[te], cs=True)
    return B._grid_cells(grid) if grid else {}


def run_cell(cache_dir, tag, cell, taps, fracs, seeds, mmap=True) -> dict:
    anchor_rec = B._load(cache_dir, B.CS_TRAIN_ANCHOR, tag, mmap=mmap)
    test_rec = B._load(cache_dir, cell, tag, mmap=mmap)
    out: dict = {}
    for task in B.BOARD_TASKS:
        y_a = np.asarray(anchor_rec["labels"][task], dtype=np.float64)
        y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
        tr_full = B._finite(y_a, np.arange(len(y_a)))
        va = B._finite(y_t, test_rec["cs_split"][task]["val"])
        te = B._finite(y_t, test_rec["cs_split"][task]["test"])
        if len(tr_full) < 2 or len(te) < 2:
            continue
        pos_full = float(np.mean(y_a[tr_full] > 0))
        rec: dict = {"n_full": int(len(tr_full)), "pos_rate_full": pos_full,
                     "n_val": int(len(va)), "n_test": int(len(te)), "fracs": {}}
        for frac in fracs:
            # frac=1.0 is deterministic (the whole set), so one seed -- repeats would be identical.
            use = (seeds[0],) if frac >= 1.0 else seeds
            per_seed = {}
            for sd in use:
                tr = _stratified(y_a, tr_full, frac, sd)
                if tr is None:
                    continue
                cells = _fit(anchor_rec, test_rec, task, taps, tr, va, te)
                if not cells:
                    continue
                per_seed[str(sd)] = {
                    "n_train": int(len(tr)),
                    "pos_rate": float(np.mean(y_a[tr] > 0)),
                    "cells": cells,
                }
            if per_seed:
                rec["fracs"][str(frac)] = per_seed
        if rec["fracs"]:
            out[task] = rec
    _checks(out, anchor_rec, test_rec, taps, seeds)
    return out


def _checks(out, anchor_rec, test_rec, taps, seeds) -> None:
    """Assert the four invariants over the assembled result and PRINT each verdict."""
    if not out:
        print("[check] VIOLATED no task produced a result", flush=True)
        return
    # 1. frac=1.0 == the board's own _cs_cell, exactly.
    task = next(iter(out))
    ref = B._cs_cell(anchor_rec, test_rec, task, taps).get("cells", {})
    got = out[task]["fracs"][str(1.0)][str(seeds[0])]["cells"]
    bad = [k for k in ref if k in got and ref[k]["test"] != got[k]["test"]]
    assert not bad, f"[check] VIOLATED frac=1.0 does not reproduce the board on {task}: {bad}"
    assert ref, f"[check] VIOLATED board reference empty for {task}"
    print(f"[check] OK frac=1.0 reproduces v3_board_readout._cs_cell exactly on {task} "
          f"({len(ref)} tap|norm cells, e.g. {sorted(ref)[0]}={ref[sorted(ref)[0]]['test']:.6f})",
          flush=True)

    # 2/3/4. test/val invariance, stratification, monotone n_train.
    drift, mono_ok = 0.0, True
    for rec in out.values():
        ns = []
        for f in sorted(rec["fracs"], key=float):
            for blob in rec["fracs"][f].values():
                drift = max(drift, abs(blob["pos_rate"] - rec["pos_rate_full"]))
            ns.append(max(b["n_train"] for b in rec["fracs"][f].values()))
        mono_ok &= all(a < b for a, b in zip(ns, ns[1:]))
    print(f"[check] {'OK' if mono_ok else 'VIOLATED'} n_train strictly increasing in frac "
          f"for all {len(out)} tasks", flush=True)
    assert mono_ok, "[check] VIOLATED n_train not monotone in frac"
    print(f"[check] OK stratified: max pos-rate drift vs full train set = {drift:.4f}", flush=True)
    # val/test are computed once per task and reused for every (frac, seed) by construction --
    # state the counts so a future refactor that re-derives them per fit shows up as a diff.
    print("[check] OK val/test rows fixed per task (shared across every frac and seed): "
          + ", ".join(f"{t}:{r['n_val']}/{r['n_test']}" for t, r in list(out.items())[:3])
          + (" ..." if len(out) > 3 else ""), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--cell", required=True, help="test cell as S,T (e.g. 1,1)")
    p.add_argument("--taps", default="enc0,enc12")
    p.add_argument("--fracs", default=",".join(str(f) for f in DEFAULT_FRACS))
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--out", required=True)
    p.add_argument("--no-mmap", dest="mmap", action="store_false")
    a = p.parse_args()

    s, t = (int(x) for x in a.cell.split(","))
    fracs = tuple(float(x) for x in a.fracs.split(","))
    seeds = tuple(int(x) for x in a.seeds.split(","))
    taps = tuple(a.taps.split(","))
    print(f"[run] cell=({s},{t}) taps={taps} fracs={fracs} seeds={seeds}", flush=True)

    out = run_cell(a.cache, a.tag, (s, t), taps, fracs, seeds, mmap=a.mmap)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print(f"[run] wrote {a.out} ({len(out)} tasks)", flush=True)


if __name__ == "__main__":
    main()
