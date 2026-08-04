"""Paired arm-vs-arm read of two board MERGED jsons at the PUBLISHED cell of each regime.

    python -m scripts.neuroprobe.board_arm_compare BASE.json ARM.json [--base-tag T] [--arm-tag T]

The published cell differs by regime and is not a preference (memory: reference-sota-claim-protocol-
parity-and-decoder-confound). WS and CSession read the per-ELECTRODE tap because the same electrodes
are available in both; CSubject reads the parcel-mean tap because electrode identity is not shared
across subjects. Reading one tap for all three would silently change what is being compared.

The null and the decision map print BEFORE the numbers so the read cannot be rationalised after the
fact. The floor row is the control: enc0 never reads weights (v3_probe_encode_r4.py:477), so it must
tie across arms. A floor that MOVES means the two caches are not on the same features and the enc12
delta is uninterpretable -- that check comes first and it is fatal, not advisory.

An arm encoded with --elec-taps 12 has no enc0_elec, so WS/CSession lose the floor control and only
CSubject keeps it. That is reported as SKIPPED, never silently passed.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict

# regime -> (published tap, its depth-0 floor)
PUBLISHED = OrderedDict(
    ws=("enc12_elec", "enc0_elec"),
    csession=("enc12_elec", "enc0_elec"),
    cs=("enc12", "enc0"),
)
NORM = "std"


def _cells(merged: dict, tag: str, regime: str, tap: str) -> dict:
    """{(cell, task): auc} for one regime/tap, over every task in the file."""
    out = {}
    for key, per_task in merged.items():
        t, task = key.split("|", 1)
        if t != tag:
            continue
        block = per_task.get(regime) or {}
        vals = block.get(f"{tap}|{NORM}")
        if not vals:
            continue
        for cell, v in vals.items():
            if v is not None:
                out[(cell, task)] = float(v)
    return out


def _sign_test(pairs) -> tuple[int, int, float]:
    """Two-sided exact sign test over paired differences; ties dropped (standard)."""
    pos = sum(1 for d in pairs if d > 0)
    neg = sum(1 for d in pairs if d < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return pos, neg, min(1.0, 2 * tail)


def _only_tag(merged: dict) -> str:
    tags = {k.split("|", 1)[0] for k in merged}
    if len(tags) != 1:
        raise SystemExit(f"expected exactly one tag in the merged json, found {sorted(tags)}")
    return tags.pop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_json")
    ap.add_argument("arm_json")
    ap.add_argument("--base-tag", default=None)
    ap.add_argument("--arm-tag", default=None)
    a = ap.parse_args()

    base = json.load(open(a.base_json))
    arm = json.load(open(a.arm_json))
    bt = a.base_tag or _only_tag(base)
    at = a.arm_tag or _only_tag(arm)

    print("=" * 78)
    print(f"BASE {bt}  <-  {a.base_json}")
    print(f"ARM  {at}  <-  {a.arm_json}")
    print("=" * 78)
    print("HYPOTHESIS  the ablated arm scores LOWER than the baseline at enc12, in all 3 regimes.")
    print("NULL        the two arms are draws from the same distribution => each paired (cell,task)")
    print("            difference is positive with p=0.5 => an exact two-sided sign test.")
    print("CONTROL     the depth-0 floor MUST tie exactly (enc0 never reads weights). A moving")
    print("            floor is FATAL: the caches are not on the same features.")
    print("DECISION    |delta| < .005 with p > .05  -> a WASH at this power")
    print("            delta < 0 with p <= .05      -> the component is load-bearing on that regime")
    print("            delta > 0 with p <= .05      -> the ablation HELPS; report it, do not bury it")
    print("UNITS       ws 12 cells x 15 tasks = 180 | csession 180 | cs 10 x 15 = 150")
    print("=" * 78)

    print("\n--- CONTROL: depth-0 floor ---")
    fatal = False
    for regime, (_tap, floor) in PUBLISHED.items():
        b = _cells(base, bt, regime, floor)
        m = _cells(arm, at, regime, floor)
        common = sorted(set(b) & set(m))
        if not common:
            why = "arm has no such tap (encoded --elec-taps 12)" if not m else "no overlap"
            print(f"  {regime:9s} {floor:11s} SKIPPED — {why}")
            continue
        d = [m[k] - b[k] for k in common]
        worst = max(abs(x) for x in d)
        ties = sum(1 for x in d if x == 0)
        ok = worst < 1e-9
        fatal |= not ok
        print(f"  {regime:9s} {floor:11s} n={len(common):3d} ties={ties:3d}/{len(common)} "
              f"max|delta|={worst:.2e}  {'OK' if ok else 'FAIL — CACHES DIFFER'}")
    if fatal:
        print("\n  !! FLOOR MOVED. The enc12 comparison below is NOT interpretable. Stop here.")

    print("\n--- enc12: published cell per regime ---")
    print(f"  {'regime':9s} {'tap':11s} {'n':>4s} {'base':>7s} {'arm':>7s} {'delta':>8s} "
          f"{'arm+':>7s} {'p':>8s}")
    for regime, (tap, _floor) in PUBLISHED.items():
        b = _cells(base, bt, regime, tap)
        m = _cells(arm, at, regime, tap)
        common = sorted(set(b) & set(m))
        if not common:
            print(f"  {regime:9s} {tap:11s}  --  no common (cell,task); arm cells="
                  f"{len(m)} base cells={len(b)}")
            continue
        d = [m[k] - b[k] for k in common]
        pos, neg, p = _sign_test(d)
        bm = sum(b[k] for k in common) / len(common)
        mm = sum(m[k] for k in common) / len(common)
        print(f"  {regime:9s} {tap:11s} {len(common):4d} {bm:7.4f} {mm:7.4f} {mm - bm:+8.4f} "
              f"{pos:3d}/{pos + neg:<3d} {p:8.4f}")
        if len(common) != len(b) or len(common) != len(m):
            print(f"  {'':9s} !! PARTIAL: base={len(b)} arm={len(m)} common={len(common)} — "
                  f"partial cells lie, do not quote this row")


if __name__ == "__main__":
    main()
