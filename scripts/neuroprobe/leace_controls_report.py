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

GEOM = ("var_removed", "pc1_var_frac", "cos_pc1", "cos_pc1_excess", "pc_participation",
        "wt_in_top10_pcs", "dir_between_frac", "pc1_between_frac", "cos_common_mode",
        "cos_domain_mean_shift")

# Stats whose raw value means nothing on its own. `pc_participation` cannot be compared across taps
# at all without this: the row-space rank differs (enc0 is d-limited at ~1-2.4k, enc12 is n-limited
# at 7k), so the deep tap simply has more directions to spread over. `leace_shuf` is the same rows,
# the same spectrum and the same rank with the session structure destroyed, so the ratio is the
# comparable quantity and the raw number is only a diagnostic.
NULLED = ("cos_pc1", "pc_participation", "dir_between_frac", "var_removed")

# The gain pretraining delivers cross-subject is speech-selective (k_CS/k_WS: language 1.21,
# acoustic 1.17, visual 0.86), so a shared task axis is only a candidate mechanism if it also
# rises selectively. Pooling `task_cos` over the menu averages the two groups together and the
# contrast disappears, which is why the breakdown is printed beside the pooled line.
VISUAL = ("face_num", "frame_brightness", "global_flow", "local_flow")


def load(d: Path) -> tuple[dict, dict, dict]:
    """-> scores[(cell, task, tap, arm)] = test AUROC, checks[(cell, tap)] = the geometry dicts,
    and by_task[(cell, task, tap)] = the same dict with the task it came from kept."""
    scores: dict = {}
    checks: dict = {}
    by_task: dict = {}
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
                by_task[(cell, task, tap)] = ck
    return scores, checks, by_task


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

    scores, checks, by_task = load(Path(args.dir))
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

    print("\n=== against the matched null (leace_shuf: same rows, spectrum and rank) ===")
    for tap in taps:
        rows = [c for (_, tp), cks in checks.items() if tp == tap for c in cks]
        if not rows:
            continue
        print(f"\n  {tap}")
        out.setdefault("nulled", {})[tap] = {}
        for k in NULLED:
            pair = [(r[k], r[k + "_leace_shuf"]) for r in rows
                    if k in r and k + "_leace_shuf" in r and r[k] == r[k]]
            if not pair:
                continue
            real, null = st.fmean(a for a, _ in pair), st.fmean(b for _, b in pair)
            ratio = real / null if null else float("nan")
            out["nulled"][tap][k] = {"real": real, "null": null, "ratio": ratio,
                                     "n_cells": len(pair)}
            print(f"    {k:<20} real {real:8.4f}   null {null:9.4f}   ratio {ratio:8.4f}"
                  f"   ({len(pair)} fits)")

    print("\n=== do the sessions share a coordinate system, offset aside? ===")
    print("    align frac: overlap of the top-k within-session subspaces as a FRACTION of the")
    print("    shuffled-split ceiling. diag vs rot: 'same axes up to scale' must beat a random")
    print("    rotation of the same spectrum, not an absolute threshold.")
    for tap in taps:
        rows = [c for (_, tp), cks in checks.items() if tp == tap for c in cks]
        ks = sorted({int(q.split("_k")[1]) for r in rows for q in r
                     if q.startswith("align_k") and q.count("_") == 1})
        if not ks:
            continue
        print(f"\n  {tap}")
        for k in ks:
            fr = [r[f"align_k{k}_frac"] for r in rows if f"align_k{k}_frac" in r]
            dg = [r[f"diag_k{k}"] for r in rows if f"diag_k{k}" in r]
            rot = [r[f"diag_k{k}_rot"] for r in rows if f"diag_k{k}_rot" in r]
            if not fr:
                continue
            out.setdefault("alignment", {}).setdefault(tap, {})[k] = {
                "frac": st.fmean(fr), "diag": st.fmean(dg), "diag_rot": st.fmean(rot)}
            print(f"    k={k:<4} align frac {st.fmean(fr):.4f}  [{min(fr):.4f}, {max(fr):.4f}]"
                  f"   |  diag {st.fmean(dg):.4f} vs rotation {st.fmean(rot):.4f}")

    print("\n=== is the TASK axis shared across sessions? ===")
    for tap in taps:
        rows = [c for (_, tp), cks in checks.items() if tp == tap for c in cks]
        got = [r for r in rows if "task_cos" in r]
        if not got:
            continue
        f = lambda k: st.fmean(r[k] for r in got)                          # noqa: E731
        beat = sum(r["task_cos"] > r["task_cos_null_p95"] for r in got)
        out.setdefault("task", {})[tap] = {"cos": f("task_cos"), "null": f("task_cos_null"),
                                           "vs_sess": f("task_vs_sess_t"),
                                           "chance": f("task_vs_sess_chance"),
                                           "beat_p95": beat, "n": len(got)}
        print(f"\n  {tap}   cos {f('task_cos'):.4f}  vs null {f('task_cos_null'):.4f} "
              f"(x{f('task_cos') / f('task_cos_null'):.2f})   {beat}/{len(got)} beat their p95")
        print(f"        overlap with the session offset: {f('task_vs_sess_t'):.4f} "
              f"(chance {f('task_vs_sess_chance'):.4f})")

        groups: dict[str, list] = {}
        for (_, task, tp), ck in by_task.items():
            if tp == tap and "task_cos" in ck:
                groups.setdefault("visual" if task in VISUAL else "speech/language", []).append(ck)
        if set(groups) == {"speech/language", "visual"}:
            for lab in ("speech/language", "visual"):
                sub = groups[lab]
                cos = st.fmean(r["task_cos"] for r in sub)
                nul = st.fmean(r["task_cos_null"] for r in sub)
                beat_g = sum(r["task_cos"] > r["task_cos_null_p95"] for r in sub)
                out["task"][tap][lab] = {"cos": cos, "null": nul, "beat_p95": beat_g, "n": len(sub)}
                print(f"        {lab:<16} cos {cos:.4f}  vs null {nul:.4f}   "
                      f"{beat_g}/{len(sub)} beat their p95")

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
