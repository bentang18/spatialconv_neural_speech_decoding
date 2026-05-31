#!/usr/bin/env python
"""Download the SWEC-iEEG dataset to DCC, skipping the 18 re-export duplicates.

Downloads only the 50 unique folders (default) into /work/ht203/data/swec/ via
huggingface_hub.snapshot_download with allow_patterns. Resumable. Binds to a
pinned HF revision (use the SHA verify_swec_dedup.py printed) so we fetch exactly
the bytes we verified.

This script IS the documented regenerate spec for the /work cache (CLAUDE.md
storage-tiering rule): after a 75-day purge, one command re-fetches everything.

Run on DCC (needs internet + hf_xet):
    # smoke: smallest subject only
    .venv/bin/python scripts/swec/fetch_swec.py --revision <SHA> --subjects ID19
    # full unique cohort
    .venv/bin/python scripts/swec/fetch_swec.py --revision <SHA>
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import _swec_common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", default=C.PINNED_REVISION, help="HF commit SHA to pin")
    ap.add_argument("--dest", default=C.DCC_DEST)
    ap.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="folder IDs to fetch (default: the 50 unique). e.g. --subjects ID19",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    subjects = args.subjects or C.unique_folders()
    dupes = set(C.duplicate_map())
    bad = [s for s in subjects if s in dupes]
    if bad:
        print(f"!! refusing to fetch known duplicate folders: {bad}")
        print("   (these are content-identical re-exports; use their originals)")
        return 2

    allow = [f"{s}/*" for s in subjects]
    dest = pathlib.Path(args.dest)
    print(f"repo:      {C.HF_REPO_ID}")
    print(f"revision:  {args.revision or '(latest — pinning recommended)'}")
    print(f"dest:      {dest}")
    print(f"subjects:  {len(subjects)} folders -> {subjects if len(subjects) <= 12 else str(subjects[:12]) + ' ...'}")
    if args.dry_run:
        print("dry-run: allow_patterns =", allow)
        return 0

    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=C.HF_REPO_ID,
        repo_type=C.HF_REPO_TYPE,
        revision=args.revision,
        allow_patterns=allow,
        local_dir=str(dest),
        max_workers=8,
    )
    print(f"\ndownloaded to: {path}")
    for s in subjects:
        d = dest / s
        if d.exists():
            n = len(list(d.glob("*.h5")))
            sz = sum(f.stat().st_size for f in d.glob("*.h5")) / 1e9
            print(f"  {s}: {n} .h5 files, {sz:.1f} GB")
        else:
            print(f"  {s}: MISSING after download")
    return 0


if __name__ == "__main__":
    sys.exit(main())
