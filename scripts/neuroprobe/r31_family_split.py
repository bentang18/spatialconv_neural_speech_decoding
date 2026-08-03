"""Recompute the event/level gain split ON THE CANONICAL ARM, in every regime that has a curve.

═══ THE DECISION THIS CHANGES ═══════════════════════════════════════════════════════════════════
Claims-table row D6c is currently unassertable for two reasons, and this closes both or fails
loudly trying:

  1. PROVENANCE. The within-session control for the change-coded finding exists only in a memo.
     The claims table bans memo-only provenance, so the number cannot enter the draft. This file
     is the missing file:line.
  2. ARM. The published per-task `k` table is a MEAN OVER 4 BOARDS (`r6_40k`, `cdlin_45k`,
     `nocd_45k`, `r6_20k`), and the within-session control is on `r6_40k` / `cdlin_45k`. None of
     those is the shipped checkpoint. Arm mixing is the ledger's #1 recorded defect, so the
     headline selectivity claim has in fact never been read off the arm we ship.

If the split survives here, D6c clears and M2 gets a canonical-arm citation. If it does not, then
M2 is arm-dependent, which is a far more important thing to learn than a tidier memo.

═══ THE NULLS, STATED BEFORE THE NUMBERS ════════════════════════════════════════════════════════
  RANK (does the split interleave?)  — under random assignment of families to tasks, the chance
    that all `nl` level tasks fall below all event tasks is exactly `1 / C(n, nl)`. That is 1/5005
    on 15 tasks and 1/55 on the 11 non-visual ones: dropping visual costs a factor of 91, so BOTH
    are always printed. This test cannot return anything stronger than its own floor.
  POOLED GAP (how big is it?) — permute the family labels over tasks and refit. Reported next to
    the observed gap so the size is never read without its null.
  ACROSS REGIMES — the interesting quantity is the gap in ws versus the gap in cs. Bootstrapped
    over SUBJECTS, never cells: two sessions of one patient are one draw.

═══ WHY THE FULL-N POINT IS THE BOARD NUMBER ════════════════════════════════════════════════════
`_anchor_check` asserts the N=full point equals the published cell computed by the real
`_ws_cell`, at `max|diff| = 0` — so reading `k` off the curve's anchor is reading it off the board,
not off a re-derivation. At full N there is no subsample, so `trainonly` and `both` coincide there
and the column choice cannot move this number.

🚫 THIS FILE DOES NOT ASSERT AGAINST `LEDGER_K`. Comparing a single canonical checkpoint against a
4-board mean and demanding agreement would BE the arm-mixing bug. The ledger values are printed
alongside for orientation and explicitly not used as a test.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.neuroprobe.r31_two_axes import EVENT, LEDGER_K, LEVEL, VISUAL  # noqa: E402
from scripts.neuroprobe.v3_board_samplecurve import _subject  # noqa: E402

SRC = pathlib.Path("results/showcase/2_what_pretraining_does")
COL = "trainonly"                       # at full N this is identical to `both` (see docstring)
NBOOT = 4000

REGIMES = {
    "ws":       dict(src="samplecurve_pbs50_cd45k.json", taps=("enc0_elec", "enc12_elec")),
    "csession": dict(src="samplecurve_csession_pbs50_cd45k.json", taps=("enc0_elec", "enc12_elec")),
    "cs":       dict(src="samplecurve_cs_pbs50_cd45k.json", taps=("enc0", "enc12")),
}


def anchor_cells(pts, tap0, tap12, col=COL) -> dict:
    """(task, cell) -> (enc0, enc12) at the ANCHOR (N=full), folds and seeds averaged.

    Only cells holding BOTH taps survive: `k` is a slope of one on the other, so a cell missing
    either side is not a partial observation, it is not an observation.
    """
    acc: dict = {}
    for p in pts:
        if p["col"] != col or not p["n_is_full"] or p["tap"] not in (tap0, tap12):
            continue
        acc.setdefault((p["task"], p["cell"]), {}).setdefault(p["tap"], []).append(p["test"])
    out: dict = {}
    for (task, cell), taps in acc.items():
        if tap0 not in taps or tap12 not in taps:
            continue
        x, y = float(np.nanmean(taps[tap0])), float(np.nanmean(taps[tap12]))
        if np.isfinite(x) and np.isfinite(y):
            out.setdefault(task, {})[cell] = (x, y)
    return out


def _k(pairs) -> float:
    """The ledger's estimator, verbatim (`paper_figs_r6.py:679`): no-intercept slope of
    (tap - .5) on (enc0 - .5), which is the per-cell ratio weighted by (enc0 - .5)^2 so that
    near-chance cells contribute in proportion to the signal they actually had. A plain ratio
    explodes when enc0 sits a thousandth above chance."""
    num = sum((x - .5) * (y - .5) for x, y in pairs)
    den = sum((x - .5) ** 2 for x, y in pairs)
    return num / den if den > 0 else float("nan")


def _pooled(cells: dict, tasks) -> float:
    return _k([p for t in tasks for p in cells[t].values()])


# Claims-table M2b, canonical checkpoint, computed FROM THE SHARDS. This file computes k from the
# CURVE ANCHOR instead -- a different code path over different files. They must agree, and checking
# it is what makes this a recomputation of a known quantity rather than a new number nobody can
# cross-check. Values are quoted to 2dp in the table, hence the tolerance.
M2B_CS = {"onset": 1.32, "speech": 1.30, "delta_volume": 1.22, "volume": 1.04, "pitch": 0.64}
M2B_TOL = 0.005


def m2b_parity(per_task: dict) -> list:
    return [(t, v, per_task[t], abs(per_task[t] - v))
            for t, v in M2B_CS.items() if t in per_task]


def split_stats(cells: dict, tasks, nperm=10000, seed=0) -> dict:
    ev = [t for t in tasks if t in EVENT]
    lv = [t for t in tasks if t in LEVEL]
    if not ev or not lv:
        return {"testable": False}
    per = {t: _k(list(cells[t].values())) for t in tasks}
    gap = _pooled(cells, ev) - _pooled(cells, lv)

    clean = min(per[t] for t in ev) > max(per[t] for t in lv)
    p_rank = 1.0 / math.comb(len(tasks), len(lv))

    rng = np.random.default_rng(seed)
    idx, nl = np.arange(len(tasks)), len(lv)
    null = []
    for _ in range(nperm):
        perm = rng.permutation(idx)
        lv_p = [tasks[i] for i in perm[:nl]]
        ev_p = [tasks[i] for i in perm[nl:]]
        null.append(_pooled(cells, ev_p) - _pooled(cells, lv_p))
    null = np.asarray(null)
    return {"testable": True, "per_task": per, "n_event": len(ev), "n_level": len(lv),
            "k_event": _pooled(cells, ev), "k_level": _pooled(cells, lv), "gap": gap,
            "clean_split": clean, "p_rank": p_rank,
            "p_gap": float((np.abs(null) >= abs(gap)).mean()),
            "null_hi": float(np.percentile(np.abs(null), 95)),
            "boundary": (min(ev, key=lambda t: per[t]), max(lv, key=lambda t: per[t]))}


def gap_ci(cells: dict, tasks, nboot=NBOOT, seed=0) -> tuple:
    """Bootstrap the event-level gap over SUBJECTS."""
    ev = [t for t in tasks if t in EVENT]
    lv = [t for t in tasks if t in LEVEL]
    subs = sorted({_subject(c) for t in tasks for c in cells[t]})
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nboot):
        draw = rng.choice(len(subs), len(subs), replace=True)
        picked = [subs[i] for i in draw]
        sub_cells = {t: {f"{c}#{j}": v for j, s in enumerate(picked)
                         for c, v in cells[t].items() if _subject(c) == s} for t in tasks}
        if any(not sub_cells[t] for t in tasks):
            continue
        out.append(_pooled(sub_cells, ev) - _pooled(sub_cells, lv))
    if not out:
        return float("nan"), float("nan"), 0
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(subs)


def did_test(cells_a, cells_b, tasks, nperm=10000, seed=0) -> dict:
    """Difference-in-differences: (event - level)_b  -  (event - level)_a.

    WHY THIS ONE IS QUOTABLE WHEN THE RAW `k` LEVELS ARE NOT. ws reads per-electrode and cs reads
    parcel-mean, so comparing k_event across regimes carries a readout-unit confound and the
    ledger bans that ratio. But each SIDE of this statistic is a contrast computed INSIDE one
    regime, on one unit and one tap pair, so the unit cancels -- which is exactly the ledger's own
    justification for quoting `(lang-vis)_CS - (lang-vis)_WS = +0.337, perm p .0187`. This is the
    same shape of statistic with the event/level cut in place of the modality cut.

    NULL, stated first: the family cut has the SAME effect in both regimes, i.e. the difference is
    0. Permute the family assignment over tasks and apply the SAME permutation to both regimes, so
    the null preserves each regime's own task difficulty profile and only destroys the family
    labelling. n_eff is the number of TASKS, not the number of cells.
    """
    shared = [t for t in tasks if t in cells_a and t in cells_b]
    ev = [t for t in shared if t in EVENT]
    lv = [t for t in shared if t in LEVEL]
    if len(ev) < 2 or len(lv) < 2:
        return {"testable": False, "n_event": len(ev), "n_level": len(lv)}

    def gap(cells, e, l):
        return _pooled(cells, e) - _pooled(cells, l)

    obs = gap(cells_b, ev, lv) - gap(cells_a, ev, lv)
    rng = np.random.default_rng(seed)
    idx, nl = np.arange(len(shared)), len(lv)
    null = []
    for _ in range(nperm):
        perm = rng.permutation(idx)
        lv_p = [shared[i] for i in perm[:nl]]
        ev_p = [shared[i] for i in perm[nl:]]
        null.append(gap(cells_b, ev_p, lv_p) - gap(cells_a, ev_p, lv_p))
    null = np.asarray(null)
    return {"testable": True, "did": obs, "n_tasks": len(shared),
            "p": float((np.abs(null) >= abs(obs)).mean()),
            "null_hi": float(np.percentile(np.abs(null), 95)), "nperm": nperm}


def report(name, cells) -> dict:
    print(f"\n{'─' * 96}\n{name.upper()}   canonical arm pbs50_cd45k, anchor (N=full)")
    got = sorted(cells)
    ncell = len({c for t in got for c in cells[t]})
    print(f"  {len(got)} tasks, {ncell} cells, "
          f"{len({_subject(c) for t in got for c in cells[t]})} subjects")
    res = {}
    for lab, tasks in (("15 tasks (submission macro)", got),
                       ("11 tasks (visual dropped — THE EVIDENCE)",
                        [t for t in got if t not in VISUAL])):
        s = split_stats(cells, tasks)
        if not s["testable"]:
            continue
        res[len(tasks)] = s
        lo, hi, nsub = gap_ci(cells, tasks)
        print(f"\n  {lab}   {s['n_event']} event / {s['n_level']} level")
        print(f"    pooled k   event {s['k_event']:.3f}   level {s['k_level']:.3f}   "
              f"gap {s['gap']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  ({nsub} subjects)")
        print(f"    NULLS  rank 1/C({len(tasks)},{s['n_level']}) = {s['p_rank']:.4f}   ·   "
              f"gap permutation p = {s['p_gap']:.4f}  (null |gap| 95th {s['null_hi']:.3f})")
        print(f"    zero interleaving: {'YES' if s['clean_split'] else 'NO'}   "
              f"boundary pair  event-floor {s['boundary'][0]} "
              f"{s['per_task'][s['boundary'][0]]:.3f}  vs  level-ceiling {s['boundary'][1]} "
              f"{s['per_task'][s['boundary'][1]]:.3f}")
    if res and name == "cs":
        rows = m2b_parity(res[max(res)]["per_task"])
        if rows:
            worst = max(r[3] for r in rows)
            print(f"\n    ✅ PARITY vs claims-table M2b (computed FROM SHARDS, a different code "
                  f"path): {len(rows)}/{len(M2B_CS)} tasks, max |diff| {worst:.4f}")
            for t, want, got, d in rows:
                print(f"       {t:>14}  M2b {want:.2f}   anchor {got:.3f}   |diff| {d:.4f}"
                      f"{'' if d < M2B_TOL else '   ❌ DRIFT'}")
            assert worst < M2B_TOL, (
                f"anchor-derived k disagrees with M2b's shard-derived k by {worst:.4f} — one of the "
                "two paths has drifted and neither number is safe to quote until that is resolved")

    if res:
        s = res[max(res)]
        print("\n    per task (canonical arm; ledger column is a 4-BOARD MEAN, orientation only, "
              "NOT a test):")
        for t in sorted(s["per_task"], key=lambda t: -s["per_task"][t]):
            fam = "EVENT" if t in EVENT else "level"
            led = f"{LEDGER_K[t]:.3f}" if t in LEDGER_K else "  —  "
            print(f"      {t:>18} {fam}  k {s['per_task'][t]:+.3f}     ledger(4-board) {led}")
    return res


def main() -> None:
    print("=" * 96)
    print("R31 · THE EVENT/LEVEL SPLIT, RECOMPUTED ON THE CANONICAL ARM")
    print("=" * 96)
    print("  Closes claims-table D6c: memo-only provenance AND a non-canonical arm.")
    print("  k = no-intercept slope of (tap-.5) on (enc0-.5) over CELLS at the anchor.")
    print("  🚫 The ledger's table is a 4-board mean — printed for orientation, never asserted.")

    got, raw_cells = {}, {}
    for name, spec in REGIMES.items():
        f = SRC / spec["src"]
        if not f.exists():
            print(f"\n[skip] {name}: {spec['src']} not on disk yet")
            continue
        raw = json.loads(f.read_text())
        pts = raw["points"] if isinstance(raw, dict) else raw
        cells = anchor_cells(pts, *spec["taps"])
        if not cells:
            print(f"\n[skip] {name}: no anchor cells at col={COL}")
            continue
        raw_cells[name] = cells
        got[name] = report(name, cells)

    for a, b in (("ws", "cs"), ("ws", "csession"), ("csession", "cs")):
        if a not in raw_cells or b not in raw_cells:
            continue
        print(f"\n{'─' * 96}\nDIFFERENCE-IN-DIFFERENCES  ({b} − {a}) of the event−level gap")
        print("  Each side is a contrast computed INSIDE one regime on one unit, so the readout")
        print("  unit cancels — this is the statistic the raw k levels cannot be compared as.")
        for lab, tasks in (("15 tasks", sorted(raw_cells[a])),
                           ("11 non-visual", [t for t in sorted(raw_cells[a]) if t not in VISUAL])):
            d = did_test(raw_cells[a], raw_cells[b], tasks)
            if not d["testable"]:
                print(f"    {lab}: not testable ({d['n_event']} event / {d['n_level']} level)")
                continue
            print(f"    {lab:>14}:  DiD = {d['did']:+.3f}   permutation p = {d['p']:.4f}   "
                  f"(null |DiD| 95th {d['null_hi']:.3f}, {d['nperm']} perms, "
                  f"n_eff = {d['n_tasks']} TASKS)")

    if "ws" in got and "cs" in got and 11 in got["ws"] and 11 in got["cs"]:
        gw, gc = got["ws"][11]["gap"], got["cs"][11]["gap"]
        print(f"\n{'─' * 96}\nWS vs CS, 11 non-visual tasks")
        print(f"  event-level gap:  ws {gw:+.3f}   cs {gc:+.3f}   difference {gc - gw:+.3f}")
        print("  ⚠️ TAPS DIFFER (enc*_elec vs parcel-mean enc*) — the readout unit moved with the")
        print("     brain, so the two k LEVELS are not comparable. Their DIFFERENCE above is")
        print("     (the unit cancels inside each regime's own contrast), but on the 11 non-visual")
        print("     tasks it does NOT clear its null — so on the argument unit claim PRESENCE vs")
        print("     ABSENCE off the interleaving test, not a magnitude. csession is the tap-matched rung.")
        if "csession" in got and 11 in got["csession"]:
            print(f"  ✅ TAP-MATCHED rung available: csession gap {got['csession'][11]['gap']:+.3f}")
        else:
            print("  [csession curve not on disk — the tap-matched rung is still missing]")


if __name__ == "__main__":
    main()
