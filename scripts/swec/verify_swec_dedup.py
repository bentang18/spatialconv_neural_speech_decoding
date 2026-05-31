#!/usr/bin/env python
"""Re-prove the SWEC 68->50 dedup against the LIVE HuggingFace revision.

The 2026-05-19 audit found that 18 of the 68 HF folders are content-identical
re-exports of 18 others, so the true unique cohort is 50 subjects / 6672 h.
Before any download *skips* those 18 folders we re-verify the claim against the
current HF revision (HF could have re-uploaded; the audit memo is 9 days old and
carries an explicit "verify before asserting" guard).

Two levels of proof:

  1. Metadata (all 68 folders, over HTTP, no bulk download): read data/ieeg
     shape, channels/sampling_rate attrs, and the full data/seizures list. Group
     folders by (channels, sr, n_samples, seizure-list); two distinct patients
     cannot share an identical exact sample count AND seizure-timestamp list.
     Re-derive the duplicate map and assert it equals the committed manifest's.

  2. Content (spot-check pairs, decompresses a few 3-min chunks from the part
     files): assert the actual voltage arrays are bit-identical across pairs.

Pins the HF commit SHA so verification and download bind to the same bytes.
Exits non-zero on ANY mismatch; writes a fresh report under reports/.

Run on DCC (needs internet + h5py + hdf5plugin):
    .venv/bin/python scripts/swec/verify_swec_dedup.py
"""

from __future__ import annotations

import argparse
import json
import sys
import pathlib

import numpy as np

import _swec_common as C  # noqa: E402  (sibling import; run from repo root)


def _open_h5(fs, path: str):
    import h5py

    return h5py.File(fs.open(path, "rb"), "r")


def _read_total_metadata(fs, repo_path: str) -> dict:
    """Read shape/attrs/seizures from an IDxx_total.h5 (cheap range reads)."""
    with _open_h5(fs, repo_path) as h5:
        ieeg = h5["data/ieeg"]
        shape = tuple(int(x) for x in ieeg.shape)
        channels = int(h5.attrs["channels"])
        sr = int(round(float(np.asarray(h5.attrs["sampling_rate"]).item())))
        # channel axis = whichever dim equals the channels attr
        if shape[0] == channels:
            n_samples = shape[1]
            orient = "(C, T)"
        elif shape[1] == channels:
            n_samples = shape[0]
            orient = "(T, C)"
        else:
            raise ValueError(
                f"{repo_path}: neither ieeg axis {shape} matches channels={channels}"
            )
        seiz = h5["data/seizures"][:]
        onsets = [round(float(x), 2) for x in np.asarray(seiz["onsets"]).ravel()]
        offsets = [round(float(x), 2) for x in np.asarray(seiz["offsets"]).ravel()]
    return {
        "shape": shape,
        "orient": orient,
        "channels": channels,
        "sampling_rate_hz": sr,
        "n_samples": int(n_samples),
        "n_seizures": len(onsets),
        "seizures": sorted(zip(onsets, offsets)),
    }


def _signature(meta: dict) -> tuple:
    return (
        meta["channels"],
        meta["sampling_rate_hz"],
        meta["n_samples"],
        tuple(meta["seizures"]),
    )


def _content_spotcheck(fs, repo_prefix: str, a: str, b: str, n_chunks: int) -> dict:
    """Read a few 3-min chunks from the part files of folders a and b; assert
    bit-identical decompressed voltage. Needs hdf5plugin for LZ4HC."""
    import hdf5plugin  # noqa: F401  (registers the LZ4HC decoder)

    def part_path(folder: str) -> str:
        return f"{repo_prefix}/{folder}/{folder}_part_1.h5"

    with _open_h5(fs, part_path(a)) as ha, _open_h5(fs, part_path(b)) as hb:
        da, db = ha["data/ieeg"], hb["data/ieeg"]
        if da.shape != db.shape:
            return {"pair": [a, b], "ok": False, "reason": f"part shape {da.shape} != {db.shape}"}
        # chunked to 3 min; sample columns = whichever axis is longer
        t_axis = 1 if da.shape[1] >= da.shape[0] else 0
        sr = int(round(float(np.asarray(ha.attrs["sampling_rate"]).item())))
        chunk = sr * 180
        T = da.shape[t_axis]
        n = min(n_chunks, max(1, T // chunk))
        starts = np.linspace(0, max(0, T - chunk), n, dtype=int)
        for s in starts:
            sl = slice(s, s + chunk)
            if t_axis == 1:
                xa, xb = da[:, sl], db[:, sl]
            else:
                xa, xb = da[sl, :], db[sl, :]
            if not np.array_equal(xa, xb):
                return {
                    "pair": [a, b],
                    "ok": False,
                    "reason": f"voltage differs at sample {int(s)}",
                }
    return {"pair": [a, b], "ok": True, "n_chunks": int(n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", default=None, help="HF commit SHA (default: resolve current)")
    ap.add_argument(
        "--content-pairs",
        type=int,
        default=3,
        help="number of duplicate pairs to bit-compare at signal level (0 to skip)",
    )
    ap.add_argument("--chunks-per-pair", type=int, default=3)
    ap.add_argument(
        "--out",
        default=str(C.REPO_ROOT / "reports" / "swec_dataset_audit_2026_05_19" / "dedup_reverify.json"),
    )
    args = ap.parse_args()

    from huggingface_hub import HfApi, HfFileSystem

    rev = args.revision or HfApi().dataset_info(C.HF_REPO_ID).sha
    print(f"HF repo:     {C.HF_REPO_ID}")
    print(f"HF revision: {rev}")

    fs = HfFileSystem()
    repo_prefix = f"datasets/{C.HF_REPO_ID}@{rev}"

    # live folder list
    live = sorted(
        p.split("/")[-1]
        for p in fs.ls(repo_prefix, detail=False)
        if p.split("/")[-1].startswith("ID")
    )
    expected = [r.folder for r in C.load_manifest()]
    print(f"\nlive folders: {len(live)} | manifest folders: {len(expected)}")
    if live != sorted(expected):
        print("!! LIVE FOLDER SET DIFFERS FROM MANIFEST")
        print("   only-live:", sorted(set(live) - set(expected)))
        print("   only-manifest:", sorted(set(expected) - set(live)))
        return 2

    manifest = {r.folder: r for r in C.load_manifest()}
    metas: dict[str, dict] = {}
    drift: list[str] = []
    print("\nreading metadata for all folders...")
    for folder in live:
        m = _read_total_metadata(fs, f"{repo_prefix}/{folder}/{folder}_total.h5")
        metas[folder] = m
        row = manifest[folder]
        mism = []
        if m["channels"] != row.channels:
            mism.append(f"channels {m['channels']}!={row.channels}")
        if m["sampling_rate_hz"] != row.sampling_rate_hz:
            mism.append(f"sr {m['sampling_rate_hz']}!={row.sampling_rate_hz}")
        if m["n_samples"] != row.n_samples:
            mism.append(f"n_samples {m['n_samples']}!={row.n_samples}")
        if m["n_seizures"] != row.n_seizures:
            mism.append(f"n_seizures {m['n_seizures']}!={row.n_seizures}")
        flag = "  DRIFT: " + "; ".join(mism) if mism else ""
        if mism:
            drift.append(folder)
        print(
            f"  {folder}: {m['orient']} C={m['channels']} sr={m['sampling_rate_hz']} "
            f"T={m['n_samples']} sz={m['n_seizures']}{flag}"
        )

    # derive dedup purely from live signatures
    first_with_sig: dict[tuple, str] = {}
    derived_dupmap: dict[str, str] = {}
    for folder in live:  # ID-sorted, so the lower ID wins as "original"
        sig = _signature(metas[folder])
        if sig in first_with_sig:
            derived_dupmap[folder] = first_with_sig[sig]
        else:
            first_with_sig[sig] = folder

    derived_unique = [f for f in live if f not in derived_dupmap]
    manifest_dupmap = C.duplicate_map()
    manifest_unique = C.unique_folders()

    print(f"\nderived unique: {len(derived_unique)} | derived duplicates: {len(derived_dupmap)}")

    ok = True
    if drift:
        ok = False
        print(f"!! {len(drift)} folder(s) drifted from the committed manifest: {drift}")
    if derived_dupmap != manifest_dupmap:
        ok = False
        print("!! DERIVED DUPLICATE MAP != MANIFEST")
        only_d = {k: v for k, v in derived_dupmap.items() if manifest_dupmap.get(k) != v}
        only_m = {k: v for k, v in manifest_dupmap.items() if derived_dupmap.get(k) != v}
        print("   derived-only:", only_d)
        print("   manifest-only:", only_m)
    else:
        print("OK: live duplicate map exactly matches the committed manifest (18 pairs).")
    if sorted(derived_unique) != sorted(manifest_unique):
        ok = False
        print("!! derived unique set != manifest unique set")

    # content spot-check on the highest-signal-density pairs first
    content_results = []
    if args.content_pairs > 0 and derived_dupmap:
        pairs = sorted(
            derived_dupmap.items(),
            key=lambda kv: metas[kv[0]]["n_seizures"],
            reverse=True,
        )[: args.content_pairs]
        print(f"\ncontent spot-check on {len(pairs)} pair(s) (bit-identical voltage):")
        for dup, orig in pairs:
            res = _content_spotcheck(fs, repo_prefix, dup, orig, args.chunks_per_pair)
            content_results.append(res)
            status = "OK" if res["ok"] else "FAIL"
            print(f"  {dup} == {orig}: {status} {res.get('reason', '')}")
            if not res["ok"]:
                ok = False

    report = {
        "hf_repo": C.HF_REPO_ID,
        "hf_revision": rev,
        "n_live_folders": len(live),
        "n_unique": len(derived_unique),
        "n_duplicates": len(derived_dupmap),
        "duplicate_map": derived_dupmap,
        "unique_folders": derived_unique,
        "manifest_drift": drift,
        "content_spotcheck": content_results,
        "verdict": "PASS" if ok else "FAIL",
    }
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    print(f"\nVERDICT: {report['verdict']}")
    if ok:
        print(f"Download set = {len(derived_unique)} unique folders (skip {len(derived_dupmap)} dupes).")
        print(f"Pinned revision for fetch: {rev}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
