"""Turn the controls array into a verdict on the LEACE null.

The question is not "did the score move" -- we know it did not. It is whether that null carries
information. Three readings are live, and the numbers below are chosen so that each one predicts a
DIFFERENT pattern, which is what makes this a test rather than a summary:

  A. Pretraining relocated identity into a high-variance subspace disjoint from content.
     Predicts: dir_between_frac well below 1 (the direction carries real within-session variance),
     cos_pc1 not ~1, AND leace_toppc COSTS something (deleting a comparable amount of variance
     chosen without reference to identity is NOT free).
  B. The erased direction is the between-session offset, which AUROC cannot see.
     Predicts: dir_between_frac ~ 1. Then "21% of variance for 7e-6" is arithmetic, not biology.
  C. Ridge is simply robust to losing any one direction out of ~7000.
     Predicts: leace_shuf is ALSO null -- the test never had power, whatever the geometry says.

B and C are not mutually exclusive and either one is fatal to the headline on its own.

Pairing is over CELLS (average tasks within a cell first), which is the board's own test unit --
see project-paired-over-cells-is-the-board-test-2026-07-27. Only the cell x task intersection where
EVERY arm scored is used: partial cells lie.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from pathlib import Path

GEOM = ("var_removed", "pc1_var_frac", "cos_pc1", "pc_participation", "wt_in_top10_pcs",
        "dir_between_frac", "pc1_between_frac", "cos_common_mode", "cos_domain_mean_shift")


def load(d: Path) -> tuple[dict, dict]:
    """-> scores[(cell, task, tap, arm)] = test AUROC, and checks[(cell, tap)] = the geometry dict."""
    scores: dict = {}
    checks: dict = {}
    for f in sorted(d.glob("*.json")):
        m = re.search(r"_(S\d+T\d+)\.json$", f.name)
        if not m:
            continue
        cell = m.group(1)
        for task, res in json.loads(f.read_text()).items():
            for key, v in res.get("cells", {}).items():
                tap, arm = key.split("|")
                scores[(cell, task, tap, arm)] = v["test"]
            for tap, ck in res.get("checks", {}).items():
                checks.setdefault((cell, tap), []).append(ck)
    return scores, checks


def paired(scores: dict, tap: str, arm: str, base: str = "std") -> dict | None:
    """Delta vs the baseline, averaged within cell, then paired across cells."""
    cells = sorted({c for (c, _, tp, a) in scores if tp == tap and a == arm})
    per_cell, n_pairs = {}, 0
    for c in cells:
        tasks = [t for (cc, t, tp, a) in scores
                 if cc == c and tp == tap and a == arm and (c, t, tap, base) in scores]
        if not tasks:
            continue
        per_cell[c] = st.fmean(scores[(c, t, tap, arm)] - scores[(c, t, tap, base)] for t in tasks)
        n_pairs += len(tasks)
    if len(per_cell) < 2:
        return None
    vals = list(per_cell.values())
    sd = st.stdev(vals)
    return {"mean": st.fmean(vals), "sd": sd, "n_cells": len(vals), "n_pairs": n_pairs,
            "t": st.fmean(vals) / (sd / math.sqrt(len(vals))) if sd > 0 else float("nan"),
            "neg": sum(v < 0 for v in vals), "per_cell": per_cell}


def _fmt(r: dict | None) -> str:
    if r is None:
        return "        --        "
    return f"{r['mean']:+.6f}  sd {r['sd']:.5f}  t {r['t']:+6.2f}  {r['neg']}/{r['n_cells']} neg"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--arms", default="leace,leace_shuf,leace_toppc")
    p.add_argument("--json-out", default="")
    args = p.parse_args()

    scores, checks = load(Path(args.dir))
    taps = sorted({tp for (_, _, tp, _) in scores}, key=lambda t: (len(t), t))
    arms = [a for a in args.arms.split(",") if a]
    if not scores:
        raise SystemExit(f"no scored cells found under {args.dir}")

    out: dict = {"delta": {}, "geometry": {}}
    print(f"\n=== deltas vs std, paired over cells ({args.dir}) ===")
    for tap in taps:
        print(f"\n  {tap}")
        for arm in arms:
            r = paired(scores, tap, arm)
            if r is not None:
                out["delta"][f"{tap}|{arm}"] = {k: v for k, v in r.items() if k != "per_cell"}
            print(f"    {arm:<12} {_fmt(r)}")

    print("\n=== geometry of the erased direction (mean [min, max] over cells) ===")
    for tap in taps:
        rows = [c for (_, tp), cks in checks.items() if tp == tap for c in cks]
        if not rows:
            continue
        print(f"\n  {tap}  ({len(rows)} cell-task fits)")
        out["geometry"][tap] = {}
        for k in GEOM:
            v = [r[k] for r in rows if k in r and r[k] == r[k]]
            if not v:
                continue
            out["geometry"][tap][k] = {"mean": st.fmean(v), "min": min(v), "max": max(v)}
            print(f"    {k:<22} {st.fmean(v):8.4f}   [{min(v):7.4f}, {max(v):7.4f}]")

    print("\n=== verdict ===")
    for tap in taps:
        g = out["geometry"].get(tap, {})
        if not g:
            continue
        bf = g.get("dir_between_frac", {}).get("mean")
        d_shuf = out["delta"].get(f"{tap}|leace_shuf", {}).get("mean")
        d_top = out["delta"].get(f"{tap}|leace_toppc", {}).get("mean")
        d_le = out["delta"].get(f"{tap}|leace", {}).get("mean")
        print(f"\n  {tap}")
        if bf is not None:
            print(f"    [B] {bf:.1%} of the erased direction's variance is the BETWEEN-session "
                  f"offset, which AUROC cannot see."
                  + ("  => the headline is largely arithmetic." if bf > 0.8 else
                     "  => a real within-session share survives." if bf < 0.5 else
                     "  => mixed; report the split, do not quote var_removed alone."))
        if d_shuf is not None and d_le is not None:
            print(f"    [C] shuffled-domain (rank-matched) control costs {d_shuf:+.6f} vs "
                  f"identity {d_le:+.6f}."
                  + ("  => no power: any rank-1 deletion is free." if abs(d_shuf) < 2e-4 else
                     "  => the test HAS power; identity being free is informative."))
        if d_top is not None:
            print(f"    [A] top-PC (variance-matched) control costs {d_top:+.6f}."
                  + ("  => deleting comparable variance is ALSO free; var_removed says nothing "
                     "about identity." if abs(d_top) < 2e-4 else
                     "  => comparable variance is NOT free, so identity's freeness is specific."))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=1, default=float))
        print(f"\n[wrote] {args.json_out}")


if __name__ == "__main__":
    main()
