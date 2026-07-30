#!/usr/bin/env python
"""Does the task axis avoid the subject-identity subspace, and does that grow with depth?

THE ONE MEASUREMENT THAT CAN SETTLE THE SEPARABILITY CLAIM. Every earlier attempt was blind by
construction: the subspace rank is pinned at min(n_sessions-1, d) by algebra, the identity probe
saturates at AUROC 1.000, and LEACE erasure of a binary concept is free BY THEOREM because AUROC
cannot see a rigid score offset. This asks the construction-side question instead -- what the model
BUILT, not what a decoder chooses to use -- and it has an exact analytic null.

Reads ONLY the reduced .npz shards, calls viz_figures.task_identity_overlap, prints the table.
Nothing is recomputed and no figure is drawn, so it cannot disagree with the suite.

NULL-FIRST CONTRACT (printed at runtime before any result, per
feedback-compute-the-null-before-you-run-the-measurement-2026-07-29):
  null value      chance_sq = keep/d -- a direction with no preference for a keep-dim subspace of a
                  d-dim space still leaves keep/d of its squared norm inside it. At keep=11, d=256
                  that is 0.043, so a raw 0.04 is ORTHOGONALITY, not "4% rides on identity".
  read            ratio_to_chance ONLY. The raw overlap is uninterpretable.
  decision map    ratio < 1  -> the task axis AVOIDS identity  => learned separation, claim revives
                  ratio ~ 1  -> generic orientation            => NO learned separation (a real null)
                  ratio > 1  -> task axis lies INSIDE identity => entangled
  admissibility   identity_subspace_is_complete == False. A full-rank span contains every direction
                  by algebra, so a complete subspace makes the question vacuous and returns nan.
                  This is why enc0 (d=7, 12 session means) cannot be measured at all.

WINDOW: whatever --red-dir is. 1 s is the protocol; the 2 s shards are a DIFFERENT measurement and
must never be pooled with them (identity erasure flips sign at enc3 between the two windows).

Usage:
  python scripts/neuroprobe/viz_ident_overlap.py --red-dir red_1s_cdlin_45k --out ident_1s.json
"""
from __future__ import annotations

import argparse
import json

from viz_common import load_all, shared_lobes
from viz_figures import task_identity_overlap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--red-dir", required=True)
    ap.add_argument("--taps", default="enc3,enc6,enc12")
    ap.add_argument("--tasks", default="onset,speech,delta_volume,word_index,"
                                      "word_part_speech,frame_brightness")
    ap.add_argument("--out", default="ident_overlap.json")
    a = ap.parse_args()

    sessions = load_all(a.red_dir)
    lobes = shared_lobes(sessions)
    taps = [t for t in a.taps.split(",") if t and any(t in s.shapes for s in sessions)]
    tasks = [t for t in a.tasks.split(",") if t]
    assert lobes, "no lobe shared by every subject"

    print(f"[load]  {len(sessions)} sessions, subjects "
          f"{sorted({s.subject_id for s in sessions})}")
    print(f"[load]  red_dir={a.red_dir}  taps={taps}")
    print(f"[check] shared lobes: {lobes}")
    print(f"[check] feature dims: "
          f"{ {t: s.shapes[t][-1] for t in taps for s in sessions[:1] if t in s.shapes} }")
    print("[null]  chance_sq = identity_rank / feature_dim -- READ ratio_to_chance, NEVER overlap")
    print("[null]  ratio <1 => separation | ~1 => generic orientation (null) | >1 => entangled")
    print("[null]  vacuous when the identity subspace is complete (enc0: d=7 < n_sessions-1)")
    print()

    rows = []
    for tap in taps:
        for task in tasks:
            o = task_identity_overlap(sessions, lobes, tap, task)
            if not o:
                print(f"[skip]  {tap:6s} {task:18s} <3 sessions with both classes")
                continue
            rows.append(o)
            if o["identity_subspace_is_complete"]:
                print(f"[ident] {tap:6s} {task:18s} rank={o['identity_rank']:3d}/"
                      f"{o['feature_dim']:4d}  VACUOUS (complete subspace)")
                continue
            print(f"[ident] {tap:6s} {task:18s} rank={o['identity_rank']:3d}/"
                  f"{o['feature_dim']:4d}  overlap={o['overlap_sq']:.4f} "
                  f"chance={o['chance_sq']:.4f}  ratio={o['ratio_to_chance']:.3f}  "
                  f"n={o.get('n_scored', 0)}")

    # Per-tap macro over tasks. The depth trend is the claim: separation should DEEPEN.
    print("\n=== MACRO over tasks (the depth trend is the claim) ===")
    macro = {}
    for tap in taps:
        rs = [r["ratio_to_chance"] for r in rows
              if r["tap"] == tap and not r["identity_subspace_is_complete"]
              and r["ratio_to_chance"] == r["ratio_to_chance"]]
        if not rs:
            print(f"  {tap:6s} no admissible cells")
            continue
        macro[tap] = sum(rs) / len(rs)
        below = sum(1 for v in rs if v < 1.0)
        print(f"  {tap:6s} mean ratio_to_chance {macro[tap]:.3f}   "
              f"below 1 in {below}/{len(rs)} tasks")

    if len(macro) >= 2:
        ks = [t for t in taps if t in macro]
        print(f"\n[trend] {' -> '.join(f'{t} {macro[t]:.3f}' for t in ks)}")
        print(f"[trend] deepest-minus-shallowest {macro[ks[-1]] - macro[ks[0]]:+.3f} "
              "(NEGATIVE = separation grows with depth)")

    json.dump({"red_dir": a.red_dir, "lobes": lobes, "rows": rows, "macro": macro},
              open(a.out, "w"), indent=1)
    print(f"\n[out] {a.out}  {len(rows)} cells")


if __name__ == "__main__":
    main()
