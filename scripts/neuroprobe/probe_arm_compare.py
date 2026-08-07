#!/usr/bin/env python
"""Paired per-cell comparison of two r6 pretrain-probe arms, tap by tap.

USAGE
  python -m scripts.neuroprobe.probe_arm_compare ARM_JSON BASE_JSON [--taps enc0,enc3,enc6,enc12]

THE PAIRED UNIT IS A CELL, NOT A COHORT MEAN. The probe merge keeps per-cell values
(``ws_per_session`` = 7 sessions, ``cs_per_test`` = 6 held-out subjects), so a cell is
(task, session) in WS and (task, subject) in CS. Both arms share cells, tasks, splits, the
ridge and the lambda rule; only the pretrained tower differs. Pairing on the cell removes the
cell-difficulty variance that dominates the raw spread, which is why a ~.005 arm effect is
readable at all. 🚫 Never pair two cohort means -- a mean over cells cannot be paired.

NULL, STATED BEFORE THE NUMBERS
  Under "the arm changes nothing", each paired per-cell difference has mean 0 and the sign is
  a fair coin. Reported per tap: n, mean delta, positive count, an EXACT two-sided sign test,
  and a Wilcoxon signed-rank. Nothing here is one-sided.

THE BUILT-IN CONTROL IS enc0
  enc0 is the frozen 3-band |STFT| floor. It is computed without the encoder blocks and is
  ARM-INDEPENDENT by construction -- R34 measured the identical floor (.5872) on two different
  arms. So enc0 MUST come back a tie. 🔴 If enc0 moves, the comparison is wrong (mismatched
  cache, mismatched cohort, or the wrong ckpt) and NO other row may be read. This is a
  precondition, not a finding, and it is checked before anything else is printed.

DECISION MAP (agreed shape, not a p-value ritual)
  enc0 not a tie                      -> ABORT. Fix the pipeline; read nothing.
  enc12 delta within noise            -> the arm does not pay BY THIS STEP COUNT.
  enc12 delta > 0, signs consistent   -> the arm pays; size it against the CS lead (+.0095)
                                         before it justifies a contract change.
  enc12 delta < 0, signs consistent   -> the arm HURTS.

⚠️ A WIDTH ARM IS ASYMMETRIC AND THAT IS NOT FIXABLE HERE. At a fixed step count a 2.25x model
is less converged per parameter, so a null reads "384 does not pay off BY step N", NEVER
"width does not help". Same caveat the ofat45k launcher carries.

⚠️ WS and CS are different regimes on different cohorts. 🚫 Do not compare a WS delta to a CS
delta as if they were the same quantity, and do not average them into one macro.
"""
from __future__ import annotations

import argparse
import json
from math import comb


def _sign_test(diffs: list[float]) -> tuple[int, int, float]:
    """Exact two-sided sign test. Zeros are dropped, which is the conservative convention."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 0, 0, 1.0
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return pos, n, min(1.0, 2 * tail)


def _cells(blob: dict, tag: str, tap: str, norm: str = "std") -> dict[tuple[str, str], float]:
    """Flatten one arm/tap into {(regime_cell, task): auroc}."""
    out: dict[tuple[str, str], float] = {}
    for key, rec in blob.items():
        parts = key.split("|")
        if len(parts) != 4:
            continue
        k_tag, k_tap, k_norm, k_task = parts
        if k_tag != tag or k_tap != tap or k_norm != norm:
            continue
        for cell, v in (rec.get("ws_per_session") or {}).items():
            out[(f"ws:{cell}", k_task)] = float(v)
        for cell, v in (rec.get("cs_per_test") or {}).items():
            out[(f"cs:{cell}", k_task)] = float(v)
    return out


def _tag_of(blob: dict) -> str:
    tags = {k.split("|")[0] for k in blob if len(k.split("|")) == 4}
    if len(tags) != 1:
        raise SystemExit(f"expected exactly one tag in the json, found {sorted(tags)}")
    return tags.pop()


def _row(name: str, diffs: list[float]) -> str:
    if not diffs:
        return f"  {name:<18} (no common cells)"
    pos, n, p = _sign_test(diffs)
    mean = sum(diffs) / len(diffs)
    try:
        from scipy.stats import wilcoxon
        w = wilcoxon(diffs).pvalue if any(d != 0 for d in diffs) else 1.0
        wtxt = f"  wilcoxon={w:.4g}"
    except Exception:
        wtxt = ""
    return (f"  {name:<18} n={len(diffs):>3}  mean={mean:+.4f}  "
            f"pos={pos}/{n}  sign_p={p:.4g}{wtxt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_json")
    ap.add_argument("base_json")
    ap.add_argument("--taps", default="enc0,enc3,enc6,enc12")
    ap.add_argument("--norm", default="std")
    ap.add_argument("--control-tap", default="enc0")
    ap.add_argument("--control-tol", type=float, default=1e-6,
                    help="max |mean delta| at the control tap that still counts as a tie")
    args = ap.parse_args()

    arm = json.load(open(args.arm_json))
    base = json.load(open(args.base_json))
    arm_tag, base_tag = _tag_of(arm), _tag_of(base)
    taps = args.taps.split(",")

    print(f"ARM  = {arm_tag}   ({args.arm_json})")
    print(f"BASE = {base_tag}   ({args.base_json})")
    print(f"delta = ARM - BASE, paired per cell, norm={args.norm}\n")

    # ---- the control is checked FIRST and can abort the read -------------------------
    a0, b0 = _cells(arm, arm_tag, args.control_tap, args.norm), _cells(base, base_tag, args.control_tap, args.norm)
    common0 = sorted(set(a0) & set(b0))
    d0 = [a0[c] - b0[c] for c in common0]
    m0 = (sum(d0) / len(d0)) if d0 else float("nan")
    print(f"[control] {args.control_tap} must tie (arm-independent |STFT| floor): "
          f"n={len(d0)} mean={m0:+.6f} max|d|={max((abs(x) for x in d0), default=float('nan')):.6f}")
    if not d0:
        raise SystemExit("[ABORT] no common cells at the control tap")
    if abs(m0) > args.control_tol:
        print(f"[ABORT] control tap moved by {m0:+.6f} > tol {args.control_tol}. "
              f"The arms are not on the same cells/cache/cohort. Read nothing below.")
        raise SystemExit(1)
    print("[control] OK -- floor is identical, the taps below are comparable.\n")

    for tap in taps:
        a, b = _cells(arm, arm_tag, tap, args.norm), _cells(base, base_tag, tap, args.norm)
        common = sorted(set(a) & set(b))
        if not common:
            print(f"{tap}: (no common cells)")
            continue
        ws = [a[c] - b[c] for c in common if c[0].startswith("ws:")]
        cs = [a[c] - b[c] for c in common if c[0].startswith("cs:")]
        am = sum(a[c] for c in common) / len(common)
        bm = sum(b[c] for c in common) / len(common)
        print(f"{tap}   arm={am:.4f}  base={bm:.4f}")
        print(_row("ws", ws))
        print(_row("cs", cs))
        print()


if __name__ == "__main__":
    main()
