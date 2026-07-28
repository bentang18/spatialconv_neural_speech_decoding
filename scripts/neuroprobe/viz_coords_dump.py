"""Dump each Lite session's electrode coordinates in the encode's canonical row order.

The per-electrode taps (``enc{t}_elec``) are stored one row per CANONICAL contact. To paint
those rows onto a brain, every row has to be matched to a physical electrode, and the only
thing the cache carries on that axis is ``parcel_canon`` -- the DKT tag per canonical
contact. So the alignment is not assumed here, it is TESTED: the parcel tags recomputed for
the Lite voltage order must equal ``parcel_canon`` exactly. If they do, the canonical order
IS the Lite voltage order and ``aligned_voltage_coords(..., electrode_set="lite")`` is
row-aligned to the features. If they do not, this refuses to write, because a brain figure
with silently permuted electrodes is worse than no figure.

Coordinates are NATIVE subject-space (L, I, P) from ``depth-wm.csv``. Not MNI -- there is no
common template here and none is wanted; each subject is rendered in their own space.

Login-node sized: a few CSV reads per subject, no cache is opened beyond its metadata.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch


def make_parcel_fn(bt_root: str, atlas: str = "dkt"):
    """The DKT hard tag per electrode, as a plain numpy function.

    Deliberately NOT ``dispatch_v3.make_bt_parcel_fn``: importing that module pulls in
    lr_schedule and so neuraltrain, which is absent from the Delta CPU environment this
    script runs in. The rule is reproduced rather than imported, and two things keep the
    copy from rotting -- a test pins it against ``make_bt_parcel_fn``'s output, and at
    runtime the ``[check]`` below compares the tags to the encode's own ``parcel_canon``,
    so a drifted rule refuses to write instead of mislabelling every row of a brain figure.
    """
    from speech_decoding.studies.braintreebank.anatomy import (
        aligned_voltage_support,
        atlas_spec,
    )

    lcol, plabels = atlas_spec(atlas)
    unknown_id = len(plabels)          # reserved id, distinct from every real parcel 0..K-1

    def parcel_fn(subject_id: int, trial_id: int, labels) -> np.ndarray:
        hs = aligned_voltage_support(
            bt_root, subject_id, trial_id=trial_id,
            parcel_labels=plabels, unmapped_policy="zero", label_column=lcol,
        )
        by_label = {
            lab: (int(hs.support[c].argmax()) if bool(hs.support[c].any()) else unknown_id)
            for c, lab in enumerate(hs.electrode_labels)
        }
        missing = [lab for lab in labels if lab not in by_label]
        if missing:
            raise KeyError(f"subject {subject_id} trial {trial_id}: cache labels absent "
                           f"from the voltage order {missing[:5]}")
        return np.asarray([by_label[lab] for lab in labels], dtype=np.int64)

    return parcel_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="encode cache dir (for parcel_canon)")
    ap.add_argument("--bt-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from speech_decoding.studies.braintreebank.anatomy import (
        aligned_voltage_coords,
        lite_voltage_order,
    )

    parcel_fn = make_parcel_fn(args.bt_root)
    out: dict = {}
    ok = True
    for path in sorted(glob.glob(os.path.join(args.cache, "enc_s*_t*.pt"))):
        rec = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        subj, trial = int(rec["subject_id"]), int(rec["trial_id"])
        canon = np.asarray(rec["parcel_canon"], dtype=np.int64)

        order = list(lite_voltage_order(args.bt_root, subj, trial))
        got = parcel_fn(subj, trial, order)
        same = len(got) == len(canon) and bool((got == canon).all())
        n_bad = int(len(canon)) if len(got) != len(canon) else int((got != canon).sum())
        print(f"[check] s{subj}_t{trial} rows={len(canon)} lite={len(order)} "
              f"mismatched_parcels={n_bad} -> {'OK' if same else 'VIOLATED'}", flush=True)
        ok &= same
        if not same:
            continue

        coords = aligned_voltage_coords(args.bt_root, subj, trial_id=trial,
                                        electrode_set="lite")
        assert coords.shape == (len(canon), 3), coords.shape
        key = f"s{subj}_t{trial}"
        out[f"{key}/coords"] = coords.astype(np.float32)
        out[f"{key}/labels"] = np.asarray(order, dtype=object).astype(str)
        out[f"{key}/parcel"] = canon

    if not ok:
        raise SystemExit("[check] VIOLATED -- canonical order is not the Lite voltage order; "
                         "refusing to write coordinates that would mislabel every row")
    np.savez_compressed(args.out, **out)
    print(f"[write] {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB, "
          f"{len(out) // 3} sessions)", flush=True)


if __name__ == "__main__":
    main()
