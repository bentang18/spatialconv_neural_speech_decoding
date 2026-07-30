# PROVENANCE (see PROVENANCE.md in this directory)
#   reads:   cs_leace_r6_40k + cs_leace_r6_40k_enc3
#   run IDs: 20544396 / 20565621
#   arm:     LEACE artifacts are r6_40k, NOT canonical cdlin_45k.
#   audited: 2026-07-29 -- memory/project-number-provenance-audit-2026-07-29.md
"""Does the recorded "erasure is FREE at enc3" claim survive at the FULL 10 cells?

The claim (project-geometry-and-transfer-are-decoupled-in-depth-2026-07-29.md:255-272) was derived
on 35 complete triples spanning only 4 enc3 cells, and its load-bearing step is: drop face_num and
frame_brightness (two single-cell spikes) and enc3 goes to +.000002 with 0/29 pairs hurt, i.e.
identical to enc12, while enc0 stays at -.002615 with 14/29 hurt.

The enc3 array (20565621) is now terminal at 10/10 cells x 15/15 tasks, so the claim can be checked
at full N instead of re-argued. This recomputes THE RECORDED NUMBERS ONLY -- same two dropped tasks,
same "hurt < -1e-4" threshold, same cell-task pair unit. No new slicing.
"""
import glob, json, statistics
from pathlib import Path

DIRS = {"enc0": "cs_leace_r6_40k", "enc12": "cs_leace_r6_40k", "enc3": "cs_leace_r6_40k_enc3"}
DROP = {"face_num", "frame_brightness"}
HURT = -1e-4


def pairs(tap):
    out = {}
    for f in sorted(glob.glob(f"{DIRS[tap]}/leace_*.json")):
        cell = Path(f).stem.split("_", 1)[1]
        for t, blob in json.load(open(f)).items():
            cl = blob["cells"]
            if f"{tap}|std" in cl and f"{tap}|leace" in cl:
                out[(cell, t)] = cl[f"{tap}|leace"]["test"] - cl[f"{tap}|std"]["test"]
    return out


print(f"{'tap':6s} {'n pairs':>8s} {'mean cost':>11s} {'hurt<-1e-4':>11s}   slice")
rows = {}
for tap in ("enc0", "enc3", "enc12"):
    p = pairs(tap)
    rows[tap] = p
    for label, sel in (("ALL 15 tasks", p),
                       (f"DROP {sorted(DROP)}", {k: v for k, v in p.items() if k[1] not in DROP})):
        n = len(sel)
        m = statistics.fmean(sel.values())
        h = sum(1 for v in sel.values() if v < HURT)
        print(f"{tap:6s} {n:8d} {m:+11.6f} {h:6d}/{n:<4d}   {label}")
    print()

print("RECORDED CLAIM (4-cell, 35 triples) vs FULL N (10 cells, 150 pairs):")
print("  enc0  recorded -.002038 all / -.002615 dropped, 15/35 -> 14/29 hurt")
print("  enc3  recorded -.001626 all / +.000002 dropped,  2/35 ->  0/29 hurt   <-- the load-bearing step")
print("  enc12 recorded -.000003 all,                     0/35 hurt")

# Was the enc3 mean really carried by two cells? Count how many DISTINCT cells are hurt.
for tap in ("enc0", "enc3", "enc12"):
    kept = {k: v for k, v in rows[tap].items() if k[1] not in DROP}
    hurt_cells = sorted({c for (c, t), v in kept.items() if v < HURT})
    hurt_tasks = sorted({t for (c, t), v in kept.items() if v < HURT})
    print(f"\n  {tap}: after the drop, hurt pairs span {len(hurt_cells)} cells "
          f"and {len(hurt_tasks)} tasks")
    if hurt_tasks:
        per_t = {t: statistics.fmean(v for (c, tt), v in kept.items() if tt == t)
                 for t in {t for _, t in kept}}
        worst = sorted(per_t.items(), key=lambda kv: kv[1])[:3]
        print(f"     worst tasks by mean cost: " +
              ", ".join(f"{t} {v:+.5f}" for t, v in worst))
