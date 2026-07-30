# PROVENANCE (see PROVENANCE.md in this directory)
#   reads:   cs_leace_r6_40k (enc0,enc12) + cs_leace_r6_40k_enc3 (enc3)
#   run IDs: 20544396 (enc0/enc12, COMPLETED) / 20565621 (enc3, COMPLETED)
#   arm:     LEACE artifacts are r6_40k, NOT canonical cdlin_45k.
#   audited: 2026-07-29 -- memory/project-number-provenance-audit-2026-07-29.md
"""AUDIT: recompute the identity-erasure delta from the LEACE SHARDS, not a merged JSON.

The claim under audit is MEMORY's "+/-.002 identity erasure". Rules applied:
  - shards, never a merge;
  - arms must be PAIRED cell-by-cell (same 10 cells, same 15 tasks) before any difference;
  - lam_pin PAIRING asserted: a leace-vs-std delta is uninterpretable if one arm pinned lambda
    and the other searched it, since then the delta mixes erasure with regularisation strength.
"""
import glob, json, statistics, sys
from collections import defaultdict

DIRS = [(d, tuple(t.split(","))) for d, t in (a.split(":") for a in sys.argv[1:])] or [
    ("results/r6_era/leace", ("enc0", "enc12")),
    ("results/r6_era/leace_enc3", ("enc3",))]
for dirn, taps in DIRS:
    fs = sorted(glob.glob(dirn + "/leace_*.json"))
    cells = [f.split("leace_")[-1].replace(".json", "") for f in fs]
    print("=" * 78)
    print(f"{dirn}   {len(fs)} shards: {' '.join(cells)}")

    # task completeness per shard -- a partial shard silently biases a macro
    per = {}
    for f, c in zip(fs, cells):
        per[c] = json.load(open(f))
    ntasks = {c: len(v) for c, v in per.items()}
    print(f"  tasks per shard: {sorted(set(ntasks.values()))}  "
          f"{'OK all equal' if len(set(ntasks.values())) == 1 else '<-- RAGGED'}")
    tasks = sorted(set.intersection(*(set(v) for v in per.values())))
    print(f"  common tasks: {len(tasks)}")

    for tap in taps:
        std, lea, tgt = f"{tap}|std", f"{tap}|leace", f"{tap}|std_target"
        # lambda pin pairing
        pins = defaultdict(set)
        miss = 0
        for c in cells:
            for t in tasks:
                cl = per[c][t]["cells"]
                if std not in cl or lea not in cl:
                    miss += 1; continue
                pins[(cl[std].get("lam_pinned"), cl[lea].get("lam_pinned"))].add((c, t))
        print(f"\n  --- {tap} ---   missing pairs: {miss}")
        for k, v in sorted(pins.items(), key=lambda kv: -len(kv[1])):
            print(f"    lam_pinned (std,leace)={k}: {len(v)} cell-tasks")
        lm = [(per[c][t]['cells'][std]['lam_mult'], per[c][t]['cells'][lea]['lam_mult'])
              for c in cells for t in tasks
              if std in per[c][t]['cells'] and lea in per[c][t]['cells']]
        same = sum(1 for a, b in lm if abs(a - b) < 1e-9)
        print(f"    lam_mult identical in {same}/{len(lm)} cell-tasks "
              f"{'<-- NOT matched, delta mixes lambda' if same < len(lm) else '(matched)'}")

        # paired macro: mean over cells, then over tasks
        rows = []
        for t in tasks:
            s = [per[c][t]["cells"][std]["test"] for c in cells if std in per[c][t]["cells"]]
            l = [per[c][t]["cells"][lea]["test"] for c in cells if lea in per[c][t]["cells"]]
            g = [per[c][t]["cells"][tgt]["test"] for c in cells if tgt in per[c][t]["cells"]]
            if len(s) != len(cells) or len(l) != len(cells):
                continue
            rows.append((t, statistics.fmean(s), statistics.fmean(l),
                         statistics.fmean(g) if g else float("nan")))
        ms = statistics.fmean(r[1] for r in rows)
        ml = statistics.fmean(r[2] for r in rows)
        print(f"    PAIRED MACRO over {len(rows)} tasks x {len(cells)} cells:")
        print(f"      std {ms:.4f}   leace {ml:.4f}   delta {ml-ms:+.5f}")
        # per-task deltas so a macro cancellation is visible
        ds = sorted(((r[2]-r[1], r[0]) for r in rows), reverse=True)
        print(f"      per-task delta range {ds[-1][0]:+.4f} ({ds[-1][1]}) .. "
              f"{ds[0][0]:+.4f} ({ds[0][1]})   |mean| of |delta| "
              f"{statistics.fmean(abs(d) for d,_ in ds):.5f}")
        pos = sum(1 for d, _ in ds if d > 0)
        print(f"      sign: leace higher in {pos}/{len(ds)} tasks")

        # the eraser's own diagnostics
        ib = [per[c][t]["checks"][tap]["id_auc_before"] for c in cells for t in tasks
              if tap in per[c][t]["checks"]]
        ia = [per[c][t]["checks"][tap]["id_auc_after"] for c in cells for t in tasks
              if tap in per[c][t]["checks"]]
        vr = [per[c][t]["checks"][tap]["var_removed"] for c in cells for t in tasks
              if tap in per[c][t]["checks"]]
        vf = [per[c][t]["checks"][tap]["var_removed_floor"] for c in cells for t in tasks
              if tap in per[c][t]["checks"]]
        print(f"      eraser: id_auc {statistics.fmean(ib):.4f} -> {statistics.fmean(ia):.4f}"
              f" | var_removed {statistics.fmean(vr):.4f} (floor {statistics.fmean(vf):.5f})"
              f" | n={len(ib)}")
