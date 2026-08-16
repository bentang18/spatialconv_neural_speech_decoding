#!/usr/bin/env python
"""Does pretraining move the cross-subject readout toward speech cortex? enc0 vs enc12.

SIZE GATE RUNS FIRST AND CAN VETO EVERYTHING. Dropping a parcel removes its contacts, so a large
AUROC drop is also what "I deleted most of the input" predicts. This regresses normalized drop on
contacts removed across every (cell, task, parcel) point and reports the residual for the a-priori
speech parcels. If superior temporal carries no positive residual, parcel size explains the
profile and there is NO anatomical claim to make -- printed as a verdict, not left to the reader.

THE UNIT IS THE SUBJECT, NOT THE CELL. The 10 CS cells come from 5 subjects (two sessions each),
so a cell-paired test double-counts. Both are printed; the SUBJECT-level one is the claim.

VISUAL TASKS ARE REPORTED, NEVER DROPPED, AND NEVER USED FOR THE CLAIM. They sit at chance, and a
leave-one-out drop measured against a chance-level fit is uninterpretable in either direction --
not evidence of absence.

Usage:
  python scripts/neuroprobe/cs_parcel_readout.py --glob 'parcelattr_vits384_cd55k/*.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from speech_decoding.studies.braintreebank.anatomy import (  # noqa: E402
    V14_DKT_PARCEL_LABELS as DKT,
)

# The a-priori speech parcels, named BEFORE the numbers are read. Superior temporal is the single
# most robust speech locus in the literature; middle temporal is the second rung of the same
# prediction and is reported separately so the two are never silently pooled.
STG = tuple(i for i, n in enumerate(DKT) if n.endswith("superiortemporal"))
MTG = tuple(i for i, n in enumerate(DKT) if n.endswith("middletemporal"))

FAMILY = {
    "acoustic": ("onset", "speech", "volume", "delta_volume", "pitch"),
    "lexical": ("word_index", "word_gap", "gpt2_surprisal", "word_head_pos",
                "word_part_speech", "word_length"),
    "visual": ("global_flow", "local_flow", "frame_brightness", "face_num"),
}
CLAIM_FAMILIES = ("acoustic", "lexical")


def name(p):
    return DKT[p] if p < len(DKT) else f"atlas{p}"


def subject(cell):
    return cell.split("T")[0]


def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f))
        for tap, tasks in d["taps"].items():
            for task, r in tasks.items():
                fam = next((k for k, v in FAMILY.items() if task in v), "other")
                for p, lo in r["lopo"].items():
                    if lo["drop_norm"] is None:
                        continue
                    rows.append(dict(cell=d["cell"], subj=subject(d["cell"]), tap=tap, task=task,
                                     family=fam, parcel=int(p), drop=lo["drop"],
                                     dn=lo["drop_norm"], n_contacts=lo["contacts_removed"],
                                     full=r["full_test"]))
    return rows


def _by(rows, keyf, valf):
    out: dict = {}
    for r in rows:
        out.setdefault(keyf(r), []).append(valf(r))
    return {k: float(np.mean(v)) for k, v in out.items()}


def size_gate(rows):
    """Regress normalized drop on contacts removed. Returns residual-by-parcel per tap."""
    print("=" * 86)
    print("SIZE GATE -- does parcel SIZE already explain the drop profile?")
    print("=" * 86)
    resid = {}
    for tap in sorted({r["tap"] for r in rows}):
        sub = [r for r in rows if r["tap"] == tap and r["family"] in CLAIM_FAMILIES]
        if len(sub) < 3:
            continue
        x = np.array([r["n_contacts"] for r in sub], dtype=float)
        y = np.array([r["dn"] for r in sub], dtype=float)
        b, a = np.polyfit(x, y, 1)
        pred = a + b * x
        ss = 1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
        print(f"\n  [{tap}] drop_norm = {a:+.4f} {b:+.5f}*contacts   R2 = {ss:.3f}   n = {len(sub)}")
        res = {r["parcel"]: [] for r in sub}
        for r, pr in zip(sub, pred):
            res[r["parcel"]].append(r["dn"] - pr)
        resid[tap] = {p: float(np.mean(v)) for p, v in res.items()}
        for p, v in sorted(resid[tap].items(), key=lambda kv: -kv[1]):
            flag = "  <- a-priori speech" if p in STG + MTG else ""
            print(f"      {name(p):32s} residual {v:+.4f}{flag}")
    for tap, rr in resid.items():
        stg = [v for p, v in rr.items() if p in STG]
        if stg and np.mean(stg) <= 0:
            print(f"\n  VERDICT [{tap}]: superior temporal sits AT OR BELOW the size trend. "
                  f"Parcel size explains the profile; NO anatomical claim at this tap.")
        elif stg:
            print(f"\n  VERDICT [{tap}]: superior temporal is {np.mean(stg):+.4f} ABOVE the size "
                  f"trend -- anatomy survives the size explanation.")
    return resid


def profile(rows):
    print("\n" + "=" * 86)
    print("PARCEL SENSITIVITY (mean normalized LOPO drop), claim families only")
    print("=" * 86)
    for fam in CLAIM_FAMILIES + ("visual",):
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        note = "   [AT CHANCE -- reported, NOT used for the claim]" if fam == "visual" else ""
        print(f"\n  {fam.upper()}{note}")
        taps = sorted({r["tap"] for r in sub})
        parcels = sorted({r["parcel"] for r in sub})
        print(f"    {'parcel':32s}" + "".join(f"{t:>12s}" for t in taps) + f"{'delta':>12s}")
        for p in parcels:
            vals = [_by([r for r in sub if r["tap"] == t and r["parcel"] == p],
                        lambda r: 0, lambda r: r["dn"]).get(0, np.nan) for t in taps]
            dl = (vals[-1] - vals[0]) if len(vals) == 2 else np.nan
            star = " *" if p in STG else ""
            print(f"    {name(p):32s}" + "".join(f"{v:12.4f}" for v in vals)
                  + f"{dl:12.4f}{star}")


def paired(rows, taps=("enc0", "enc12")):
    """The claim. enc12 - enc0 STG sensitivity, paired over SUBJECTS (and cells, for reference)."""
    print("\n" + "=" * 86)
    print(f"THE CLAIM -- superior temporal sensitivity, {taps[1]} minus {taps[0]}")
    print("=" * 86)
    sub = [r for r in rows if r["family"] in CLAIM_FAMILIES and r["parcel"] in STG]
    if not sub:
        print("  no superior-temporal rows; nothing to test")
        return
    for unit in ("subj", "cell"):
        per = {}
        for t in taps:
            per[t] = _by([r for r in sub if r["tap"] == t], lambda r: r[unit], lambda r: r["dn"])
        keys = sorted(set(per[taps[0]]) & set(per[taps[1]]))
        d = np.array([per[taps[1]][k] - per[taps[0]][k] for k in keys])
        if not len(d):
            continue
        lab = "SUBJECT (the claim)" if unit == "subj" else "cell (double-counts; reference only)"
        print(f"\n  paired over {lab}   n = {len(d)}")
        for k, v in zip(keys, d):
            print(f"    {k:8s} {per[taps[0]][k]:+.4f} -> {per[taps[1]][k]:+.4f}   {v:+.4f}")
        pos = int((d > 0).sum())
        print(f"    mean {d.mean():+.4f}   median {np.median(d):+.4f}   "
              f"{pos}/{len(d)} positive")
        # ⚠ THE SUBJECT-LEVEL TEST CANNOT REACH p < .05. Two-sided Wilcoxon on n pairs has a
        # minimum attainable p of 2/2^n = .0625 at n=5, so a unanimous 5/5 result is the FLOOR,
        # not a null. Printed next to the p-value so nobody reads .0625 as a failure to replicate.
        floor = 2.0 / (2 ** len(d))
        try:
            from scipy.stats import wilcoxon
            if len(d) >= 5 and np.any(d != 0):
                p = float(np.asarray(wilcoxon(d).pvalue))
                note = "  <- THE FLOOR at this n; significance is unreachable" \
                    if abs(p - floor) < 1e-9 else ""
                print(f"    Wilcoxon p = {p:.4f}  (min attainable {floor:.4f}){note}")
        except ImportError:
            pass
        if unit == "subj":
            if d.mean() > 0 and pos >= len(d) - 1:
                print("    => H1: pretraining CONCENTRATES the readout on speech cortex.")
            elif abs(d.mean()) < 0.02:
                print("    => H0: pretraining AMPLIFIES without RELOCATING. Not alignment.")
            else:
                print("    => H2: the added signal sits OUTSIDE classical speech cortex. "
                      "Do NOT write this up as improved physiological alignment.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    a = ap.parse_args()
    rows = load(a.glob)
    if not rows:
        raise SystemExit(f"no rows matched {a.glob!r}")
    cells = sorted({r["cell"] for r in rows})
    print(f"loaded {len(rows)} (cell, tap, task, parcel) rows over {len(cells)} cells "
          f"{cells}, {len(set(r['subj'] for r in rows))} subjects, "
          f"taps {sorted(set(r['tap'] for r in rows))}\n")
    if len(cells) < 10:
        print(f"⚠ INCOMPLETE: {len(cells)}/10 CS cells. A partial macro reads as if it covered "
              f"everything; treat every number below as provisional.\n")
    size_gate(rows)
    profile(rows)
    paired(rows)


if __name__ == "__main__":
    main()
