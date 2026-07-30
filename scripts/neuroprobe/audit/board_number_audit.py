# PROVENANCE (see PROVENANCE.md in this directory)
#   reads:   results/r6_era/board/*.json
#   run IDs: board readout arrays; baselines from vendored neuroprobe f9b0842
#   arm:     LEACE artifacts are r6_40k, NOT canonical cdlin_45k.
#   audited: 2026-07-29 -- memory/project-number-provenance-audit-2026-07-29.md
"""PROVENANCE AUDIT: recompute every headline number from its source artifact.

Not "I found the file that mentions .6036" -- this RECOMPUTES each quoted number from the raw
board JSON under an explicitly stated aggregation, and prints PASS/FAIL against the value we
have been quoting. A number that cannot be recomputed here has no provenance and must not go in
the paper until it does.

The arXiv rule (Dietterich 2026-05-15) is the reason: every number an LLM touched has to be
verifiable against a real source. This is that check, run mechanically instead of by memory.

AGGREGATION IS STATED, NOT ASSUMED:
  - 15 tasks, ALL of them. Dropping the near-chance tasks is defensible for a multiplicative k
    (near-zero denominator) and NOT defensible for a macro AUROC. Quoting a 12-task subset as a
    macro is the exact error made earlier today.
  - cs: mean over the 10 cross-subject CELLS, then mean over tasks. Cells are per-subject-trial,
    so this is the per-subject average the ledger requires.
  - ws / csession: mean over the 12 Lite session cells, then over tasks.
  - norm: std (raw retired 2026-07-20).
"""
import json
import statistics
from pathlib import Path

B = Path("results/r6_era/board")
BOARD15 = None  # filled from the file; asserted to be 15

# (label, file, regime, tap, quoted value)
CLAIMS = [
    ("CS  enc0  (canonical 45k)", "results_v3_board_cdlin_45k.json", "cs", "enc0|std", 0.5872),
    ("CS  enc12 (canonical 45k)", "results_v3_board_cdlin_45k.json", "cs", "enc12|std", 0.6036),
    ("WS  enc12 (canonical 45k)", "results_v3_board_cdlin_45k.json", "ws", "enc12_elec|std", 0.6897),
    # .6846 was the 40k arm (Finding 5); canonical 45k csession enc12 is .6862.
    ("CSession enc12 (canon 45k)", "results_v3_board_cdlin_45k.json", "csession", "enc12_elec|std", 0.6862),
]


def _from_shards(tag, regime, tap):
    """Merged board JSONs carry csession:{} from a stale merge, so csession is ONLY traceable
    from shards. Same aggregation as macro(): mean over cells, then over tasks."""
    d = B / f"shards_{tag}"
    per = {}
    for p in sorted(d.glob(f"{regime}_*.json")):
        blob = json.load(open(p))
        for key, rec in blob.get("cells", {}).items():
            col = rec.get("cells", {}).get(tap)
            if col:
                per.setdefault(key.split("|")[-1], {})[p.stem] = col["test"]
    if not per:
        return None, 0, 0
    cells = sorted({c for v in per.values() for c in v})
    return (statistics.fmean(statistics.fmean(v.values()) for v in per.values()),
            len(per), len(cells))


def macro(fname, regime, tap):
    """Mean over cells then over tasks, for one regime/tap. Returns (value, n_tasks, n_cells)."""
    d = json.load(open(B / fname))
    per_task = {}
    for key, blob in d.items():
        r = blob.get(regime)
        if not r:
            continue
        col = r.get(tap)
        if col:
            per_task[key.split("|")[-1]] = col
    if not per_task:
        # Fall back to shards; this is the ONLY route for csession.
        tag = fname.replace("results_v3_board_", "").replace(".json", "")
        return _from_shards(tag, regime, tap)
    # Cell set per task can differ; report the union size and average each task over its own cells.
    cells = sorted({c for col in per_task.values() for c in col})
    vals = {t: statistics.fmean(col.values()) for t, col in per_task.items()}
    return statistics.fmean(vals.values()), len(per_task), len(cells)


print("=" * 78)
print("PART A -- OUR NUMBERS, recomputed from board JSONs")
print("=" * 78)
fails = []
for label, fname, regime, tap, quoted in CLAIMS:
    got, ntask, ncell = macro(fname, regime, tap)
    if got is None:
        print(f"  {label:28s} NO DATA for {regime}/{tap} in {fname}  <-- UNTRACEABLE")
        fails.append(label)
        continue
    ok = abs(got - quoted) < 5e-4
    print(f"  {label:28s} quoted {quoted:.4f}  recomputed {got:.4f}  "
          f"({ntask} tasks, {ncell} cells)  {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"{label}: quoted {quoted:.4f} vs recomputed {got:.4f}")

# INVARIANT: the task count must be 15. If it is not, the aggregation is not the board macro.
print("\n  task-count invariant (must be 15 for a board macro):")
for label, fname, regime, tap, _q in CLAIMS:
    _v, ntask, _c = macro(fname, regime, tap)
    flag = "OK" if ntask == 15 else f"<-- NOT 15, this is a SUBSET"
    print(f"    {label:28s} {ntask:2d} tasks  {flag}")

print("\n  cross-check against the file's own stored *_mean fields:")
d = json.load(open(B / "results_v3_board_cdlin_45k.json"))
k0 = sorted(d)[0]
for fld in ("cs_mean", "ws_mean", "csession_mean"):
    v = d[k0].get(fld)
    print(f"    {fld:16s} {'present' if v else 'EMPTY'}"
          + (f"  keys {sorted(v)[:4]}" if isinstance(v, dict) and v else ""))

print()
print("=" * 78)
print("PART B -- claims that are NOT recomputable from these artifacts")
print("=" * 78)
NOT_HERE = [
    ("board baselines CNN .5777 / PopT .5750 / Linear .5392",
     "vendored upstream JSONs + scripts/neuroprobe/leaderboard_baselines.py -- separate artifact"),
    ("identity rank-11-of-256, linear from layer 3, .68 -> 1.0000",
     "viz / LEACE artifacts, not the board JSON"),
    ("erasing identity moves cs r by +/-.002",
     "LEACE arrays under results/r6_era/leace"),
    ("a_event ~1.8 / a_level ~0.0",
     "derived by gate_model.py from these same board JSONs -- derived, not primary"),
    ("r6 masks 75.0%",
     "model config / masking test, not a result artifact"),
]
for claim, where in NOT_HERE:
    print(f"  - {claim}\n      source: {where}")

print()
if fails:
    print(f"AUDIT RESULT: {len(fails)} PROBLEM(S)")
    for f in fails:
        print(f"  ! {f}")
else:
    print("AUDIT RESULT: all Part A claims recomputed and matched.")
