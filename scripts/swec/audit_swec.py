#!/usr/bin/env python
"""Audit downloaded SWEC HDF5 files for integrity + contract conformance.

For each subject folder under --dest, checks:

  1. Integrity   — completeness + decodability. On-disk file size must equal the
                   HF sibling size (HF Xet already content-verifies bytes during
                   download; size-match is the on-disk completeness proxy), and
                   boundary data/ieeg chunks must decompress without error.
                   NOTE: the embedded info/checksums are STALE vs the HF-hosted
                   parts (curators re-compressed before upload -> b2sum(part) !=
                   stored checksum), so they are NOT used as an integrity gate.
  2. Schema      — only {data/ieeg, data/seizures, info/*} + attrs
                   {patient, channels, sampling_rate}; assert NO anatomy fields
                   (coords / MNI / labels / channel names) -> SWEC is anatomy-blind.
  3. Orientation — which data/ieeg axis is channels; report (C, T) vs (T, C).
  4. Reconcile   — channels / sr / n_samples / n_seizures vs the committed
                   2026-05-19 manifest (exact match required).
  5. PSD sanity  — (--psd) Welch PSD on a mid-recording 30 s block averaged over
                   channels; report power at 100/120/140 Hz to confirm the
                   0.5-120 Hz hardware band-pass (-> trainable bins k0-k21).

Writes a verified on-disk manifest CSV. Exits non-zero on any failure.

Run on DCC (needs h5py + hdf5plugin; --psd needs scipy; size check needs net):
    .venv/bin/python scripts/swec/audit_swec.py --subjects ID19 --psd
    .venv/bin/python scripts/swec/audit_swec.py          # all unique, on disk
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

import _swec_common as C  # noqa: E402

_ANATOMY_HINTS = (
    "coord", "mni", "ras", "label", "anat", "montage", "location", "channel_name", "ch_name",
)


def _walk_keys(h5) -> list[str]:
    keys: list[str] = []
    h5.visit(lambda name: keys.append(name))
    return keys


def fetch_hf_sizes(subjects: list[str], revision: str) -> dict[str, int]:
    """{relpath: size} for the audited subjects' files, from the HF revision."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(
            C.HF_REPO_ID, repo_type=C.HF_REPO_TYPE, files_metadata=True, revision=revision
        )
        want = set(subjects)
        return {
            s.rfilename: int(s.size)
            for s in info.siblings
            if s.size and s.rfilename.split("/")[0] in want
        }
    except Exception as e:  # offline / API error -> skip size check gracefully
        print(f"WARN: could not fetch HF sizes ({e}); skipping size check\n")
        return {}


def audit_subject(folder: pathlib.Path, manifest: dict, hf_sizes: dict, do_psd: bool) -> dict:
    import h5py
    import hdf5plugin  # noqa: F401  (registers the LZ4HC decoder)

    sid = folder.name
    res: dict = {"folder": sid, "checks": {}, "ok": True}

    def fail(check: str, msg: str):
        res["checks"][check] = f"FAIL: {msg}"
        res["ok"] = False

    def ok(check: str, msg: str = ""):
        res["checks"][check] = f"OK {msg}".strip()

    total = folder / f"{sid}_total.h5"
    if not total.exists():
        fail("present", f"{total.name} missing")
        return res

    # --- metadata from total.h5 (reliable: shape/attrs/seizures/info-files) ---
    with h5py.File(total, "r") as h5:
        keys = _walk_keys(h5)
        anatomy = [k for k in keys if any(h in k.lower() for h in _ANATOMY_HINTS)]
        attr_anatomy = [a for a in h5.attrs if any(h in a.lower() for h in _ANATOMY_HINTS)]
        if anatomy or attr_anatomy:
            fail("anatomy_blind", f"unexpected anatomy fields: {anatomy + attr_anatomy}")
        else:
            ok("anatomy_blind", "no coords/labels/channel-names")

        data_keys = {k for k in keys if not k.startswith("info")}
        extra = data_keys - {"data", "data/ieeg", "data/seizures"}
        if extra:
            res["checks"]["schema_extra"] = f"NOTE: extra data keys {sorted(extra)}"
        else:
            ok("schema", "data/ieeg + data/seizures only")

        ieeg = h5["data/ieeg"]
        shape = tuple(int(x) for x in ieeg.shape)
        channels = int(h5.attrs["channels"])
        sr = int(round(float(np.asarray(h5.attrs["sampling_rate"]).item())))
        if shape[0] == channels:
            orient, n_samples = "(C, T)", shape[1]
        elif shape[1] == channels:
            orient, n_samples = "(T, C)", shape[0]
        else:
            fail("orientation", f"neither axis {shape} == channels {channels}")
            orient, n_samples = "?", max(shape)
        ok("orientation", orient)
        n_seizures = len(h5["data/seizures"][:])
        part_names = [f.decode() if isinstance(f, bytes) else str(f) for f in h5["info/files"][:]]
        # the VDS source filenames are KNOWN-CORRUPT for many subjects (they cite
        # another patient's parts). info/files is authoritative; the VDS is not.
        try:
            vsrc = {vs.file_name for vs in ieeg.virtual_sources()}
        except Exception:
            vsrc = set()
        res["checks"]["vds_sources"] = (
            "OK self-referential" if vsrc <= set(part_names)
            else f"NOTE: VDS cites foreign parts {sorted(vsrc)[:2]} -> read parts directly, NOT the VDS"
        )

    # --- integrity: completeness (size vs HF, summed part length) + decode ---
    # Read PART FILES DIRECTLY (the total.h5 VDS returns fill-zeros / foreign data).
    part_paths = [folder / pn for pn in part_names]
    missing = [pn for pn, pp in zip(part_names, part_paths) if not pp.exists()]
    size_bad = []
    if hf_sizes:
        for p in sorted(folder.glob("*.h5")):
            rel = f"{sid}/{p.name}"
            if rel not in hf_sizes:
                size_bad.append(f"{p.name} not on HF")
            elif p.stat().st_size != hf_sizes[rel]:
                size_bad.append(f"{p.name} {p.stat().st_size}!={hf_sizes[rel]}")

    chunk = sr * 180
    sum_T, chan_bad, decode_bad, part_c_axis0 = 0, [], [], []
    for pp in part_paths:
        if not pp.exists():
            continue
        with h5py.File(pp, "r") as hp:
            d = hp["data/ieeg"]
            ps = tuple(int(x) for x in d.shape)
            ca = 0 if ps[0] == channels else (1 if ps[1] == channels else None)
            if ca is None:
                chan_bad.append(f"{pp.name}{ps}!C={channels}")
                continue
            part_c_axis0.append(pp)
            T_i = ps[1 - ca]
            sum_T += T_i
            for s in sorted({0, max(0, T_i - chunk)}):
                try:
                    _ = d[:, s:s + chunk] if ca == 0 else d[s:s + chunk, :]
                except Exception as e:  # truncated/corrupt LZ4HC chunk
                    decode_bad.append(f"{pp.name}@{s}:{type(e).__name__}")

    len_bad = []
    if not missing and not chan_bad and sum_T != n_samples:
        len_bad.append(f"sum(parts)={sum_T}!=total={n_samples}")
    problems = (
        [f"{m} missing" for m in missing] + size_bad + chan_bad + decode_bad + len_bad
    )
    if problems:
        fail("integrity", "; ".join(problems))
    else:
        sz = "size==HF, " if hf_sizes else ""
        ok("integrity", f"{len(part_c_axis0)} part(s): {sz}sum-len==total, chunks decode")

    # --- reconcile vs committed manifest ---
    row = manifest.get(sid)
    if row is None:
        res["checks"]["reconcile"] = "NOTE: not in committed manifest"
    else:
        mism = []
        if channels != row.channels:
            mism.append(f"channels {channels}!={row.channels}")
        if sr != row.sampling_rate_hz:
            mism.append(f"sr {sr}!={row.sampling_rate_hz}")
        if n_samples != row.n_samples:
            mism.append(f"n_samples {n_samples}!={row.n_samples}")
        if n_seizures != row.n_seizures:
            mism.append(f"n_seizures {n_seizures}!={row.n_seizures}")
        if mism:
            fail("reconcile", "; ".join(mism))
        else:
            ok("reconcile", "matches 5/19 manifest")

    # --- PSD roll-off sanity (signal-bearing block from part_1, over channels) ---
    # Long-term SWEC recordings contain zero-filled gaps; scan windows for signal.
    if do_psd and orient != "?" and part_c_axis0:
        from scipy.signal import welch

        with h5py.File(part_c_axis0[0], "r") as hp:
            d = hp["data/ieeg"]
            ps = tuple(int(x) for x in d.shape)
            ca = 0 if ps[0] == channels else 1
            T_i = ps[1 - ca]
            seg = sr * 30
            block, chosen = None, None
            for s0 in np.linspace(0, max(0, T_i - seg), 16, dtype=int):
                b = d[:, s0:s0 + seg] if ca == 0 else np.asarray(d[s0:s0 + seg, :]).T
                b = np.asarray(b, dtype=np.float64)
                if np.std(b) > 0:
                    block, chosen = b, int(s0)
                    break
        if block is None:
            res["checks"]["psd_rolloff"] = "NOTE: all sampled windows flat/zero"
        else:
            f, pxx = welch(block, fs=sr, nperseg=min(sr * 2, block.shape[1]), axis=1)
            pmean = pxx.mean(axis=0)

            def at(hz):
                return float(pmean[np.argmin(np.abs(f - hz))])

            p100, p120, p140 = at(100), at(120), at(140)
            roll = (p140 / p100) if p100 > 0 else float("nan")
            note = f"@{round(chosen / sr)}s P100={p100:.2e} P120={p120:.2e} P140={p140:.2e} (P140/P100={roll:.3f})"
            # 4th-order Butterworth at 120 Hz: monotone roll-off past cutoff,
            # |H|^2 ~ 0.25 at f/fc=1.17 -> expect P140<P120<P100 and P140<~0.5*P100.
            if p100 > 0 and p140 < p120 < p100 and roll < 0.5:
                ok("psd_rolloff", f"band-limited ~120 Hz (Butterworth skirt); {note}")
            else:
                res["checks"]["psd_rolloff"] = f"NOTE: roll-off unclear; {note}"

    res["meta"] = {
        "channels": channels,
        "sampling_rate_hz": sr,
        "n_samples": int(n_samples),
        "hours": round(n_samples / sr / 3600, 2),
        "n_seizures": n_seizures,
        "orientation": orient,
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=C.DCC_DEST)
    ap.add_argument("--subjects", nargs="*", default=None, help="default: all unique on disk")
    ap.add_argument("--revision", default=C.PINNED_REVISION, help="HF revision for size check")
    ap.add_argument("--psd", action="store_true", help="run PSD roll-off sanity check")
    ap.add_argument("--no-size-check", action="store_true", help="skip the HF size lookup (offline)")
    ap.add_argument("--out", default=None, help="verified manifest CSV (default: <dest>/verified_manifest.csv)")
    args = ap.parse_args()

    dest = pathlib.Path(args.dest)
    manifest = {r.folder: r for r in C.load_manifest()}
    subjects = args.subjects or [s for s in C.unique_folders() if (dest / s).exists()]
    if not subjects:
        print(f"no subject folders found under {dest}")
        return 2

    hf_sizes = {} if args.no_size_check else fetch_hf_sizes(subjects, args.revision)

    print(f"auditing {len(subjects)} subject(s) under {dest}\n")
    results = []
    all_ok = True
    for sid in subjects:
        r = audit_subject(dest / sid, manifest, hf_sizes, args.psd)
        results.append(r)
        all_ok &= r["ok"]
        print(f"{sid}: {'PASS' if r['ok'] else 'FAIL'}")
        for check, val in r["checks"].items():
            print(f"    {check}: {val}")
        if "meta" in r:
            m = r["meta"]
            print(f"    meta: {m['orientation']} C={m['channels']} sr={m['sampling_rate_hz']} "
                  f"T={m['n_samples']} ({m['hours']} h) sz={m['n_seizures']}")
        print()

    out = pathlib.Path(args.out) if args.out else dest / "verified_manifest.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["folder", "channels", "sampling_rate_hz", "n_samples", "hours", "n_seizures", "orientation", "audit"])
        for r in results:
            m = r.get("meta", {})
            w.writerow([
                r["folder"], m.get("channels"), m.get("sampling_rate_hz"), m.get("n_samples"),
                m.get("hours"), m.get("n_seizures"), m.get("orientation"),
                "PASS" if r["ok"] else "FAIL",
            ])
    print(f"wrote {out}")
    print(f"\nVERDICT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
