"""Turn the controls array into a verdict on the LEACE null.

The question is not "did the score move" -- we know it did not. It is whether that null carries
information. Three readings are live, and the numbers below are chosen so that each one predicts a
DIFFERENT pattern, which is what makes this a test rather than a summary:

  A. Pretraining relocated identity into a high-variance subspace disjoint from content.
     Predicts: a large EFFECTIVE (within-session) erased share, cos_pc1 not ~1, AND leace_toppc
     COSTS something -- deleting a comparable amount of variance chosen without reference to
     identity is NOT free.
  B. The erased direction is mostly the between-session offset, which AUROC cannot see.
     Predicts: dir_between_frac ~ 1.
  C. Ridge is simply robust to losing any one direction out of ~7000.
     Predicts: leace_shuf is ALSO null -- the test never had power, whatever the geometry says.

**B and C are not equally damaging, and it matters not to conflate them.** AUROC is invariant to a
constant per-row score offset, so the offset component contributes EXACTLY ZERO to the measured
delta. It therefore cannot be hiding a cost, and the -7e-6 null is already a statement about the
WITHIN-session component alone. What B corrupts is the ADVERTISING, not the null: `var_removed`
counts variance the test never probed. The honest headline is

    erasing  var_removed x (1 - dir_between_frac)  of WITHIN-session variance costs 7e-6

reported below as `effective_within_var`. That number also re-tests the enc0-vs-enc12 story: the
published contrast is "enc0 tiny (0.37%) and ENTANGLED vs enc12 huge (20.7%) and DISJOINT", and if
enc12's effective share collapses toward enc0's, the contrast weakens or inverts on its own terms.

C is the reading that can actually void the null, because it says the instrument reads zero on
everything.

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
    effective: dict[str, float] = {}
    for tap in taps:
        g = out["geometry"].get(tap, {})
        if not g:
            continue
        bf = g.get("dir_between_frac", {}).get("mean")
        d_shuf = out["delta"].get(f"{tap}|leace_shuf", {}).get("mean")
        d_top = out["delta"].get(f"{tap}|leace_toppc", {}).get("mean")
        d_le = out["delta"].get(f"{tap}|leace", {}).get("mean")
        vr = g.get("var_removed", {}).get("mean")
        print(f"\n  {tap}")
        if bf is not None and vr is not None:
            eff = vr * (1.0 - bf)
            effective[tap] = eff
            print(f"    [B] var_removed {vr:.2%}, of which {bf:.1%} is the BETWEEN-session offset "
                  f"AUROC cannot see")
            print(f"        => EFFECTIVE within-session erased variance = {eff:.3%}"
                  + ("   (the 'huge subspace' framing does not survive this)" if eff < 0.02 else
                     "   (a substantial within-session share -- the framing survives)"))
        if d_shuf is not None and d_le is not None:
            print(f"    [C] shuffled-domain (rank-matched) control costs {d_shuf:+.6f} vs "
                  f"identity {d_le:+.6f}."
                  + ("  => NO POWER: any rank-1 deletion is free, so the null is about ridge, "
                     "not geometry. This voids the claim." if abs(d_shuf) < 2e-4 else
                     "  => the test HAS power; identity being free is informative."))
        if d_top is not None:
            print(f"    [A] top-PC (variance-matched) control costs {d_top:+.6f}."
                  + ("  => deleting comparable variance is ALSO free; var_removed says nothing "
                     "about identity." if abs(d_top) < 2e-4 else
                     "  => comparable variance is NOT free, so identity's freeness is specific."))

    if len(effective) > 1 and "enc0" in effective and "enc12" in effective:
        e0, e12 = effective["enc0"], effective["enc12"]
        print(f"\n  enc0 vs enc12 on EFFECTIVE within-session erased variance: "
              f"{e0:.3%} vs {e12:.3%}  (ratio {e12 / e0:.1f}x)" if e0 > 0 else "")
        print("    the published contrast is 'enc0 tiny+entangled vs enc12 huge+disjoint' -- "
              + ("that survives on effective variance too." if e12 > 5 * e0 else
                 "on EFFECTIVE variance the size gap largely CLOSES, so the contrast rests on "
                 "the delta (enc0 hurts, enc12 does not), NOT on subspace size."))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=1, default=float))
        print(f"\n[wrote] {args.json_out}")


if __name__ == "__main__":
    main()
