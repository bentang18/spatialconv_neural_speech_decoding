"""How many lobes do all Lite subjects actually share?

The parcel axis is dead for cross-subject work: the board's own CS cells intersect to 3-7
DKT parcels and the all-subject intersection is empty. Lobes are the coarser axis that
should survive. This measures whether it does, and with how many electrodes behind each
cell -- a lobe held up by one electrode in one subject is not a shared coordinate.

Runs standalone on a Delta CPU node against the board cache, so PARCEL_LOBE_KEYS is
inlined rather than imported. test_viz_lobe_audit.py pins the copy to
anatomy.parcel_lobe_keys(); if that test fails, regenerate the literal, do not hand-edit.
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import Counter, defaultdict

import numpy as np
import torch

# index-aligned with the DKT parcel id, unknown (== len(vocabulary)) appended last
PARCEL_LOBE_KEYS: tuple[str, ...] = (
    'lh-cingulate', 'lh-frontal', 'lh-occipital', 'lh-temporal',
    'lh-temporal', 'lh-parietal', 'lh-temporal', 'lh-insula',
    'lh-cingulate', 'lh-occipital', 'lh-frontal', 'lh-occipital',
    'lh-frontal', 'lh-temporal', 'lh-frontal', 'lh-temporal',
    'lh-frontal', 'lh-frontal', 'lh-frontal', 'lh-occipital',
    'lh-parietal', 'lh-cingulate', 'lh-frontal', 'lh-parietal',
    'lh-cingulate', 'lh-frontal', 'lh-frontal', 'lh-parietal',
    'lh-temporal', 'lh-parietal', 'lh-temporal', 'rh-cingulate',
    'rh-frontal', 'rh-occipital', 'rh-temporal', 'rh-temporal',
    'rh-parietal', 'rh-temporal', 'rh-insula', 'rh-cingulate',
    'rh-occipital', 'rh-frontal', 'rh-occipital', 'rh-frontal',
    'rh-temporal', 'rh-frontal', 'rh-temporal', 'rh-frontal',
    'rh-frontal', 'rh-frontal', 'rh-occipital', 'rh-parietal',
    'rh-cingulate', 'rh-frontal', 'rh-parietal', 'rh-cingulate',
    'rh-frontal', 'rh-frontal', 'rh-parietal', 'rh-temporal',
    'rh-parietal', 'rh-temporal', 'lh-mtl', 'lh-mtl',
    'lh-subcortical', 'lh-subcortical', 'lh-subcortical', 'lh-subcortical',
    'rh-mtl', 'rh-mtl', 'rh-subcortical', 'rh-subcortical',
    'rh-subcortical', 'rh-subcortical', 'unknown',
)
UNKNOWN_LOBE = "unknown"


def _collapse_hemi(key: str) -> str:
    return key if key == UNKNOWN_LOBE else key.split("-", 1)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/projects/bhqk/htang13/v3_board_cache_board_cdlin_45k")
    ap.add_argument("--min-elec", type=int, default=2,
                    help="a lobe needs this many electrodes in a subject to count as present")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.cache, "enc_s*_t*.pt")))
    print(f"[cache] {args.cache}  n_files={len(paths)}")
    assert paths, "no encode shards found"

    # subject -> lobe key -> electrode count. Sessions of one subject share anatomy, so the
    # cross-subject axis is defined over subjects, not over sessions.
    per_subject: dict[int, Counter] = defaultdict(Counter)
    per_session: list[tuple[str, Counter]] = []

    for p in paths:
        rec = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        canon = np.asarray(rec["parcel_canon"], dtype=np.int64)
        assert canon.min() >= 0 and canon.max() < len(PARCEL_LOBE_KEYS), \
            f"parcel id out of range in {p}: [{canon.min()}, {canon.max()}]"
        counts = Counter(PARCEL_LOBE_KEYS[i] for i in canon)
        key = f"S{rec['subject_id']}T{rec['trial_id']}"
        per_session.append((key, counts))
        subj = int(rec["subject_id"])
        if not per_subject[subj]:
            per_subject[subj] = counts.copy()
        else:
            # trials of a subject should carry the same montage up to a dropped channel
            for lobe in set(counts) | set(per_subject[subj]):
                per_subject[subj][lobe] = max(per_subject[subj][lobe], counts[lobe])
        del rec

    subjects = sorted(per_subject)
    print(f"[subjects] {subjects}")

    for scheme, keyfn in (("hemi-split", lambda k: k), ("hemi-pooled", _collapse_hemi)):
        pooled = {s: Counter() for s in subjects}
        for s in subjects:
            for lobe, n in per_subject[s].items():
                pooled[s][keyfn(lobe)] += n
        lobes = sorted({lobe for s in subjects for lobe in pooled[s] if lobe != UNKNOWN_LOBE})

        print(f"\n=== {scheme}: electrodes per lobe per subject "
              f"(min-elec={args.min_elec} to count as present) ===")
        head = "lobe".ljust(16) + "".join(f"S{s}".rjust(6) for s in subjects) + "  n_subj"
        print(head)
        print("-" * len(head))
        shared, near = [], []
        for lobe in lobes:
            row = [pooled[s][lobe] for s in subjects]
            n_present = sum(1 for v in row if v >= args.min_elec)
            print(lobe.ljust(16) + "".join(str(v).rjust(6) for v in row) + f"  {n_present}/{len(subjects)}")
            if n_present == len(subjects):
                shared.append(lobe)
            elif n_present >= len(subjects) - 1:
                near.append(lobe)

        print(f"\n[{scheme}] |L*| (present in ALL {len(subjects)} subjects) = {len(shared)}  {shared}")
        print(f"[{scheme}] present in all but one           = {len(near)}  {near}")
        if shared:
            floor = min(pooled[s][lobe] for s in subjects for lobe in shared)
            print(f"[{scheme}] weakest shared cell = {floor} electrodes")

    print("\n=== per-session unknown-electrode count ===")
    for key, counts in per_session:
        n_unk = counts[UNKNOWN_LOBE]
        tot = sum(counts.values())
        print(f"  {key:<8} n_elec={tot:3d}  unknown={n_unk} ({100.0 * n_unk / tot:.1f}%)")


if __name__ == "__main__":
    main()
