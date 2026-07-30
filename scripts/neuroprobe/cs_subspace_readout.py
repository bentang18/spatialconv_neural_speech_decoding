#!/usr/bin/env python
"""Read out cs_subspace_ablation shards against the PRE-REGISTERED decision map.

GATE FIRST, VERDICT SECOND. Arm `none` must reproduce the board's per-cell CS AUROC before any
projection arm is allowed to mean anything. The board and LEACE pipelines independently agree to
max |diff| 4.1e-4 on 450 shared (tap, cell, task), and the readout's primal/dual fp32 difference is
documented at 2.7e-5, so a `none` deviation above ~1e-3 per cell is a BUG, not noise.

REFUSES a partial macro. All 10 CS cells must be present for a tap, or that tap is reported as
INCOMPLETE and no delta is printed for it -- a 9-cell macro silently reads as if it covered
everything (`perband` at 9 cells is the recorded instance of exactly this).

Paired over CELLS is the board test, not paired over (cell, task): 15 tasks inside a cell are not
independent samples of transfer. Both are printed, the cell-paired one is the claim.

  null          delta 0 => those directions are not load-bearing
  decision map  out_top hurts monotonically in k while out_rand ~ 0
                   => identity and content SHARE the high-variance directions
                out_top ~ 0 too => content is in LOW-variance dirs; retract the shared reading
                out_top HELPS => identity is an active nuisance; removal is a method contribution
  admissibility out_rand hurting as much as out_top => rank alone explains it => NOTHING is shown.
                Checked and printed BEFORE the verdict, not after.
  gradient      if the harm is identity-driven it should track dir_between_frac
                (.0747 enc0 -> .9951 enc12), i.e. GROW with depth. A flat profile across taps means
                tap-dependence, not identity.

Usage:
  python scripts/neuroprobe/cs_subspace_readout.py --shards 'subspace/cs_sub*.json' \
      --baseline cs_baseline_40k.json
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np

TAPS = ("enc0", "enc3", "enc6", "enc12")
N_CELLS = 10
TOL = 1e-3
DIR_BETWEEN = {"enc0": 0.0747, "enc3": None, "enc6": None, "enc12": 0.9951}


def _ckey(c: str) -> str:
    """Canonical cell key. The ablation writes `s1_t1`, the board baseline `S1T1`; same cell.

    Normalizing is safe ONLY because the mapping is a bijection on these 10 cells -- verified by
    asserting the two label sets coincide after normalization, below. A silent non-match here is
    what produced `CANNOT GATE`, and swallowing it would have gated against an empty reference.
    """
    return c.upper().replace("_", "")


def load(pat: str) -> dict:
    """{(cell,tap,task,arm): test_auroc}, deduped. Later shards must AGREE where they overlap."""
    v: dict = {}
    clash = 0
    files = sorted(glob.glob(pat))
    kept = []
    for f in files:
        d = json.load(open(f))
        if "rows" not in d:
            # The verdict this script writes must never be re-ingested as a shard. Announced, not
            # silent: a shard that legitimately lost its `rows` key would otherwise vanish from the
            # macro and the cell count would quietly drop below 10.
            print(f"[skip] {f}: no 'rows' key — not a shard (verdict//summary file?)")
            continue
        kept.append(f)
        for r in d["rows"]:
            k = (_ckey(r["cell"]), r["tap"], r["task"], r["arm"])
            if k in v and abs(v[k] - r["test"]) > 1e-9:
                clash += 1
            v[k] = r["test"]
    print(f"[load] {len(kept)} shards, {len(v)} unique (cell,tap,task,arm)"
          + (f"  ⚠️ {clash} DISAGREEING duplicates" if clash else "  no duplicate disagreement"))
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--baseline", required=True, help="{tap: {cell: board auroc}} json")
    # NOT `cs_sub*` — the shard glob would re-ingest it on the next run.
    ap.add_argument("--out", default="verdict_cs_subspace.json")
    a = ap.parse_args()

    v = load(a.shards)
    raw = json.load(open(a.baseline))
    board = {t: {_ckey(c): x for c, x in d.items()} for t, d in raw.items()}
    for t, d in board.items():
        assert len(d) == len(raw[t]), f"{t}: cell labels collide under normalization"
    mine = {k[0] for k in v}
    for t, d in board.items():
        shared = mine & set(d)
        if mine and not shared:
            raise SystemExit(f"🔴 {t}: shard cells {sorted(mine)[:4]} match NONE of the baseline "
                             f"{sorted(d)[:4]} even after normalization — fix the key map, do not gate")
    arms = sorted({k[3] for k in v})
    ranks = sorted({int(x.replace("out_top", "")) for x in arms if x.startswith("out_top")})
    print(f"[arms] {len(arms)} = none + out_top/keep_top/out_rand at k={ranks}")

    # ── GATE: arm 'none' must reproduce the board per cell ────────────────────────
    print("\n=== GATE: does arm 'none' reproduce the board? (tol 1e-3 per cell) ===")
    ok_taps = []
    for tap in TAPS:
        cells = sorted({k[0] for k in v if k[1] == tap and k[3] == "none"})
        if not cells:
            print(f"  {tap:6s} ABSENT — not run yet")
            continue
        if len(cells) < N_CELLS:
            print(f"  {tap:6s} INCOMPLETE {len(cells)}/{N_CELLS} cells — NO DELTA REPORTED")
            continue
        dev = []
        for c in cells:
            mine = np.mean([v[(c, tap, t, "none")] for t in
                            {k[2] for k in v if k[0] == c and k[1] == tap and k[3] == "none"}])
            ref = board.get(tap, {}).get(c)
            if ref is None:
                continue
            dev.append(abs(mine - ref))
        if not dev:
            print(f"  {tap:6s} no board reference for these cells — CANNOT GATE")
            continue
        mx = float(np.max(dev))
        good = mx <= TOL
        print(f"  {tap:6s} {len(cells)} cells  max|none-board| {mx:.6f}  "
              f"{'PASS' if good else '🔴 FAIL — do not interpret this tap'}")
        if good:
            ok_taps.append(tap)
    if not ok_taps:
        print("\n🔴 NO TAP PASSED THE GATE. Nothing below is interpretable. Stop and diff.")
        return

    # ── ADMISSIBILITY before verdict ──────────────────────────────────────────────
    def macro(tap, arm):
        cells = sorted({k[0] for k in v if k[1] == tap and k[3] == arm})
        if len(cells) < N_CELLS:
            return None, None
        per = {c: float(np.mean([v[(c, tap, t, arm)] for t in
                                 {k[2] for k in v if k[0] == c and k[1] == tap and k[3] == arm}]))
               for c in cells}
        return float(np.mean(list(per.values()))), per

    print("\n=== ADMISSIBILITY: is out_rand as damaging as out_top? ===")
    inadmissible = []
    for tap in ok_taps:
        for k in ranks:
            bt, _ = macro(tap, "none")
            ot, _ = macro(tap, f"out_top{k}")
            rd, _ = macro(tap, f"out_rand{k}")
            if bt is None or ot is None or rd is None:
                continue
            if abs(rd - bt) >= abs(ot - bt) - 1e-6 and abs(ot - bt) > 2e-3:
                inadmissible.append((tap, k))
    print("  " + ("rank-matched random control is HARMLESS relative to out_top — admissible"
                  if not inadmissible else
                  f"🔴 rand >= top at {inadmissible} — RANK ALONE EXPLAINS IT, nothing shown"))

    # ── the table ─────────────────────────────────────────────────────────────────
    print("\n=== DELTA vs arm 'none', paired over CELLS (the board test) ===")
    res = {}
    for tap in ok_taps:
        bt, bper = macro(tap, "none")
        # ok_taps only contains taps whose 'none' arm had all N_CELLS and passed the gate
        assert bt is not None and bper is not None, f"{tap} in ok_taps but macro is None"
        print(f"\n  {tap}  baseline {bt:.4f}   (dir_between_frac "
              f"{DIR_BETWEEN.get(tap) if DIR_BETWEEN.get(tap) else '—'})")
        print(f"    {'arm':12s} {'macro':>7s} {'delta':>8s} {'cells<base':>11s} {'per-task w':>11s}")
        row = {}
        for fam in ("out_top", "out_rand", "keep_top"):
            for k in ranks:
                arm = f"{fam}{k}"
                m, per = macro(tap, arm)
                if m is None or per is None:
                    print(f"    {arm:12s} INCOMPLETE")
                    continue
                worse = sum(1 for c in per if per[c] < bper[c])
                tw = [(c, t) for (c, tp, t, ar) in v if tp == tap and ar == arm]
                pw = sum(1 for c, t in tw if v[(c, tap, t, arm)] < v[(c, tap, t, "none")])
                print(f"    {arm:12s} {m:7.4f} {m - bt:+8.4f} {worse:5d}/{len(per):<5d} "
                      f"{pw:5d}/{len(tw):<5d}")
                row[arm] = {"macro": m, "delta": m - bt, "cells_worse": worse,
                            "n_cells": len(per)}
        res[tap] = {"baseline": bt, "arms": row}

    # ── verdict against the map ───────────────────────────────────────────────────
    print("\n=== VERDICT against the pre-registered map ===")
    kmax = max(ranks)
    for tap in ok_taps:
        r = res[tap]["arms"]
        if f"out_top{kmax}" not in r:
            continue
        dt, dr = r[f"out_top{kmax}"]["delta"], r.get(f"out_rand{kmax}", {}).get("delta", 0.0)
        mono = all(r[f"out_top{ranks[i]}"]["delta"] >= r[f"out_top{ranks[i+1]}"]["delta"] - 1e-9
                   for i in range(len(ranks) - 1) if f"out_top{ranks[i+1]}" in r)
        if dt < -2e-3 and abs(dr) < abs(dt) / 2:
            call = "SHARED — identity and content share the high-variance directions"
        elif abs(dt) <= 2e-3:
            call = "FREE — content is in LOW-variance dirs; retract the shared reading"
        elif dt > 2e-3:
            call = "NUISANCE — removing these directions HELPS; method contribution"
        else:
            call = "AMBIGUOUS"
        print(f"  {tap:6s} out_top{kmax} {dt:+.4f}  out_rand{kmax} {dr:+.4f}  "
              f"monotone_in_k={mono}  => {call}")
    print("\n  GRADIENT CHECK: identity-driven harm should GROW enc0 -> enc12 "
          "(dir_between_frac .0747 -> .9951).")
    if len(ok_taps) >= 2:
        prof = [(t, res[t]["arms"].get(f"out_top{kmax}", {}).get("delta")) for t in ok_taps]
        prof = [(t, d) for t, d in prof if d is not None]
        print("  " + "  ".join(f"{t} {d:+.4f}" for t, d in prof))
        if len(prof) >= 2:
            print(f"  deepest-minus-shallowest {prof[-1][1] - prof[0][1]:+.4f} "
                  "(NEGATIVE = harm grows with depth = identity-consistent)")

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
