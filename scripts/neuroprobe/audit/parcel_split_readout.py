# PROVENANCE (see PROVENANCE.md in this directory)
#   reads:   board_shards_cdlin45k_PARCELUNIT
#   run IDs: 20580123 (ws, 12/12) / 20580124 (csession)
#   arm:     LEACE artifacts are r6_40k, NOT canonical cdlin_45k.
#   audited: 2026-07-29 -- memory/project-number-provenance-audit-2026-07-29.md
"""PRE-REGISTERED READ-OUT: does the event/level split appear WITHIN-SESSION at the PARCEL unit?

The cross-subject row shows an event/level asymmetry (event gain >> level gain). Two readings:
  UNIT   -- the parcel mean itself makes the split; then it must also appear within-session,
            because ws/csession use the SAME parcel averaging here.
  REGIME -- the split is a property of cross-subject transfer (no sensor correspondence);
            then within-session at the same unit shows NO split.

Pre-registered decision (Ben, 2026-07-29): APPEARS => the split is the UNIT and the change-coding
mechanism claim is an averaging artifact. ABSENT => it is the REGIME and the reading survives.

CSESSION IS THE LOAD-BEARING ARM: _ws_cell does not intersect parcels (`_feat(rec, enc, tr)` takes
no column argument), so the ws arm at --taps enc*|parcel is NOT unit-matched to cs. Only
_csession_cell and _cs_cell go through _parcel_cols atlas-id intersection.

INVARIANTS, asserted not assumed -- a partial macro over cells LIES (ledger, 07-27):
  1. every mode must have ALL 12 Lite cells present, else REFUSE to print a macro;
  2. every cell must carry all 15 tasks;
  3. the reference cs asymmetry must reproduce from the cs artifact in the same run, else the
     comparison has no anchor.
"""
import glob, json, statistics, sys
from pathlib import Path

SHARDS = Path(sys.argv[1] if len(sys.argv) > 1 else
              "/projects/bhqk/htang13/board_shards_cdlin45k_PARCELUNIT")
CS_BOARD = Path(sys.argv[2] if len(sys.argv) > 2 else
                "/projects/bhqk/htang13/results/r6_era/board/results_v3_board_cdlin_45k.json")
LEVEL = {"volume", "pitch", "frame_brightness", "face_num", "global_flow", "local_flow"}
UNSTABLE = {"face_num", "frame_brightness", "word_length"}
N_CELLS = 12
LO, HI = "enc0", "enc12"


def load(mode):
    """Shard schema: {kind, name, cells: {"<tag>|<task>": {"cells": {"<tap>|<norm>": {test,..}}}}}."""
    out = {}
    for f in sorted(glob.glob(str(SHARDS / f"{mode}_*.json"))):
        d = json.load(open(f))
        assert d["kind"] == mode, f"{f} says kind={d['kind']}, expected {mode}"
        out[d["name"]] = {k.split("|")[-1]: v for k, v in d["cells"].items()}
    return out


def gains(per, cell_keys, lo_key, hi_key):
    """per[cell][task]['cells'][tap|norm] -> per-task multiplicative k and additive gain."""
    tasks, rows = None, {}
    for c in cell_keys:
        d = per[c]
        got = {}
        for t, blob in d.items():
            cl = blob["cells"] if "cells" in blob else blob
            if lo_key in cl and hi_key in cl:
                a = cl[lo_key]["test"] if isinstance(cl[lo_key], dict) else cl[lo_key]
                b = cl[hi_key]["test"] if isinstance(cl[hi_key], dict) else cl[hi_key]
                got[t] = (a, b)
        rows[c] = got
        tasks = set(got) if tasks is None else tasks & set(got)
    return rows, sorted(tasks)


def report(label, rows, tasks):
    stable = [t for t in tasks if t not in UNSTABLE]
    ev = sorted(t for t in stable if t not in LEVEL)
    lv = sorted(t for t in stable if t in LEVEL)
    print(f"\n  {label}: {len(rows)} cells x {len(stable)} stable tasks "
          f"(event {len(ev)}, level {len(lv)})")

    def k(a, b):
        return (b - 0.5) / (a - 0.5) if abs(a - 0.5) > 1e-6 else float("nan")

    def grp(sel, fn):
        vals = [fn(*rows[c][t]) for c in rows for t in sel]
        vals = [v for v in vals if v == v]
        return statistics.fmean(vals) if vals else float("nan")

    add_e, add_l = grp(ev, lambda a, b: b - a), grp(lv, lambda a, b: b - a)
    k_e, k_l = grp(ev, k), grp(lv, k)
    print(f"    additive gain: event {add_e:+.4f}  level {add_l:+.4f}  asym {add_e-add_l:+.4f}")
    print(f"    multiplicative k: event {k_e:.3f}  level {k_l:.3f}  "
          f"ratio level/event {k_l/k_e if k_e else float('nan'):.3f}")
    # THREE ratio definitions have been quoted for this comparison and they are NOT the same
    # number. Print all of them so no downstream claim silently swaps one for another:
    #   additive ratio = mean level gain / mean event gain   (this is memory's "cs 0.03")
    #   k ratio        = mean of per-cell-task k, level / event
    print(f"    additive ratio level/event {add_l/add_e if add_e else float('nan'):.3f}"
          f"   <-- the definition MEMORY.md's \"cs 0.03\" uses")
    # cell-level sign counts: a macro can hide a split that only 2 cells carry
    pos_l = sum(1 for c in rows
                if statistics.fmean(rows[c][t][1] - rows[c][t][0] for t in lv) > 0)
    print(f"    level gain POSITIVE in {pos_l}/{len(rows)} cells "
          f"(cross-subject level gain is ~0 by construction of the claim)")
    return add_e - add_l, (k_l / k_e if k_e else float("nan"))


print("=" * 78)
print(f"shards: {SHARDS}")
anchors = {}
for mode in ("ws", "csession"):
    per = load(mode)
    print(f"\n{mode}: {len(per)}/{N_CELLS} cells  {' '.join(sorted(per))}")
    if len(per) < N_CELLS:
        print(f"  ⛔ REFUSING TO REPORT A MACRO — {N_CELLS-len(per)} cell(s) missing. "
              f"A partial cell set lies (ledger 07-27). Re-run when the array finishes.")
        continue
    cells0 = list(per.values())[0]
    keys = list(cells0[next(iter(cells0))]["cells"])
    lo_key = next(k for k in keys if k.split("|")[0] == LO)
    hi_key = next(k for k in keys if k.split("|")[0] == HI)
    print(f"  taps used: {lo_key} -> {hi_key}   (all arm keys: {keys})")
    rows, tasks = gains(per, sorted(per), lo_key, hi_key)
    ntask = {c: len(v) for c, v in rows.items()}
    if len(set(ntask.values())) != 1 or min(ntask.values()) < 15:
        print(f"  ⚠️ RAGGED tasks per cell: {sorted(set(ntask.values()))} — macro is not clean")
    anchors[mode] = report(mode, rows, tasks)

# reference anchor: the cs asymmetry these are being compared against
if CS_BOARD.exists():
    d = json.load(open(CS_BOARD))
    per = {}
    for key, blob in d.items():
        cs = blob.get("cs") or {}
        a, b = cs.get(f"{LO}|std"), cs.get(f"{HI}|std")
        if a and b:
            for c in set(a) & set(b):
                per.setdefault(c, {})[key.split("|")[-1]] = {"cells": {f"{LO}|std": a[c],
                                                                      f"{HI}|std": b[c]}}
    rows, tasks = gains(per, sorted(per), f"{LO}|std", f"{HI}|std")
    print(f"\nREFERENCE cs row ({len(rows)} cells) — the asymmetry under test:")
    anchors["cs"] = report("cs", rows, tasks)
else:
    print(f"\n⚠️ cs anchor not found at {CS_BOARD} — comparison has no reference")

print("\n" + "=" * 78)
print("PRE-REGISTERED VERDICT")
for m in ("ws", "csession", "cs"):
    if m in anchors:
        print(f"  {m:9s} additive asym {anchors[m][0]:+.4f}   level/event k-ratio {anchors[m][1]:.3f}")
if "csession" in anchors and "cs" in anchors:
    # Threshold comes from THIS run's own cs anchor, computed by the same code path. A hardcoded
    # number from another script's k definition would not be comparable.
    cs_a, cs_r = anchors["cs"]
    cse_a, cse_r = anchors["csession"]
    print(f"\n  csession is the load-bearing arm (unit-matched to cs; ws is not).")
    print(f"  cs anchor: additive asym {cs_a:+.4f}, k-ratio {cs_r:.3f}. No-split point: asym 0,"
          f" k-ratio 1.")
    d_cs, d_no = abs(cse_a - cs_a), abs(cse_a - 0.0)
    print(f"  csession additive asym {cse_a:+.4f} is {d_cs:.4f} from cs and {d_no:.4f} from zero"
          f"  =>  {'APPEARS (UNIT)' if d_cs < d_no else 'ABSENT (REGIME)'}")
    d_cs, d_no = abs(cse_r - cs_r), abs(cse_r - 1.0)
    print(f"  csession k-ratio    {cse_r:.3f}  is {d_cs:.3f} from cs and {d_no:.3f} from 1.0"
          f"  =>  {'APPEARS (UNIT)' if d_cs < d_no else 'ABSENT (REGIME)'}")
    print("  Both measures must agree. If they split, report the disagreement, not a verdict.")
else:
    print("\n  ⛔ csession incomplete — NO VERDICT. Do not read the ws arm as the answer;")
    print("     _ws_cell does not intersect parcels, so ws is not unit-matched to cs.")
