"""Do the 15 board tasks share one train/val/test partition per session?

WHY THIS DECIDES THE FT LOOP SHAPE. ``_loo_ridge_risk`` takes y as (n, K) and masks per column
(v3_ws_partialft_pilot.py:153-200: "the multi-task columns share one Gram, so K tasks cost one
extra solve, not K forwards"; ``msk`` is False where a task has no label and those rows are
dropped from the risk). So tasks with DIFFERENT finite-row sets can still share one fine-tune --
the only thing that must match is the PARTITION, i.e. which clip indices are train vs val vs test.

If all 15 share it, one FT per (session, fold) serves all 15 tasks and the board FT drops from
690 cell-runs to 46. If they do not, the groups are whatever this prints, and the loop keys on
the signature rather than on the task.

Prints the DECISION MAP before the data: signature counts are the null (15 distinct = no sharing
possible, 1 = full collapse), and the answer is read off the count.

READS THE SPLITS OFF THE EXISTING BOARD CACHE rather than rebuilding labels -- the encode already
stored ``ws_split``/``cs_split`` in every record (v3_probe_encode_r4.py:203-204). Rebuilding them
would need the braintreebank label stack (mne et al.), which the Delta CPU conda env does not
carry, and would re-derive something already on disk.

🔴 MEMORY: the board records are 50-65 GB each because they carry ``enc12_elec``. torch.load is
called with mmap=True so tensor STORAGES are never faulted in -- only the pickle metadata and the
small numpy index arrays are read. Never drop the mmap flag here.

CPU only -- splits and metadata, no model, no GPU.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os

import numpy as np
import torch


def _sig(a) -> str:
    """Order-insensitive fingerprint of a row-index set."""
    v = np.sort(np.asarray(a, dtype=np.int64))
    return hashlib.blake2b(v.tobytes(), digest_size=8).hexdigest()


def _partition_sig(sp: dict) -> str:
    """Fingerprint of one split dict, over whichever of train/val/test it carries."""
    return "/".join(f"{k}:{_sig(sp[k])}" for k in sorted(sp) if sp[k] is not None)


def _conflicts(splits: dict) -> tuple:
    """Do all tasks' splits RESTRICT one global row->role partition?

    Raw index sets differ across tasks because each split is taken over that task's FINITE rows
    (the finite counts below span ~2378-3500), so equality of index sets is the WRONG test and will
    say "no sharing" even when sharing is total. The right test is CONFLICT: a row assigned 'train'
    by one task and 'test' by another. Zero conflicts => one global partition exists, every task's
    split is that partition restricted to its finite rows, and ``_loo_ridge_risk``'s per-column
    mask makes a single multitask fine-tune exact.

    Returns (n_conflicts, n_rows_covered, per_role_counts).
    """
    role: dict = {}
    bad = 0
    for sp in splits.values():
        for r, idx in sp.items():
            if idx is None:
                continue
            for i in np.asarray(idx, dtype=np.int64).tolist():
                prev = role.setdefault(i, r)
                if prev != r:
                    bad += 1
    cnt: dict = {}
    for r in role.values():
        cnt[r] = cnt.get(r, 0) + 1
    return bad, len(role), cnt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="/projects/bhqk/htang13/v3_board_cache_board_r6_40k")
    p.add_argument("--limit", type=int, default=0, help="stop after N records (0 = all)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache_dir, "enc_s*_t*.pt")))
    if args.limit:
        files = files[: args.limit]
    print(f"[decision map] board tasks x {len(files)} cached sessions from {args.cache_dir}")
    print("  distinct ws partition signatures == 1   -> ONE FT per (session, fold) serves all 15")
    print("  distinct == n_tasks -> no sharing; FT stays per task")
    print("  anything between -> group by signature; that count IS the speedup divisor\n")

    for f in files:
        rec = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
        if not isinstance(rec, dict):
            print(f"  {os.path.basename(f)}: unexpected record type {type(rec)}")
            continue
        ws = rec.get("ws_split")
        cs = rec.get("cs_split")
        labels = rec.get("labels") or {}
        name = os.path.basename(f)
        if ws is None:
            print(f"  {name}: NO ws_split -- keys = {sorted(rec)[:12]}")
            continue
        ws_sig, cs_sig = {}, {}
        for t, folds in ws.items():
            ws_sig[t] = "|".join(f"{fd}:{_partition_sig(sp)}"
                                 for fd, sp in sorted(folds.items()))
        for t, sp in (cs or {}).items():
            cs_sig[t] = _partition_sig(sp)
        nws, ncs = len(set(ws_sig.values())), len(set(cs_sig.values()))
        fin = {t: int(np.isfinite(np.asarray(v, dtype=float)).sum())
               for t, v in labels.items()}
        rng = f"{min(fin.values())}-{max(fin.values())}" if fin else "n/a"
        print(f"  {name:44s} tasks={len(ws_sig):2d}  distinct ws partitions={nws:2d}  "
              f"cs={ncs:2d}  finite-label range {rng}")

        # THE DECIDING TEST — does one global row->role partition restrict to all 15 tasks?
        nfold = max(len(f) for f in ws.values())
        for fd in range(nfold):
            per_task = {t: f[fd] for t, f in ws.items() if fd in f}
            bad, cov, cnt = _conflicts(per_task)
            print(f"        ws fold {fd}: conflicts={bad:6d}  rows covered={cov:5d}  "
                  f"{ {k: v for k, v in sorted(cnt.items())} }  "
                  f"{'SHARED' if bad == 0 else 'NOT SHARED'}")
        if cs:
            bad, cov, cnt = _conflicts(cs)
            print(f"        cs      : conflicts={bad:6d}  rows covered={cov:5d}  "
                  f"{ {k: v for k, v in sorted(cnt.items())} }  "
                  f"{'SHARED' if bad == 0 else 'NOT SHARED'}")
        del rec


if __name__ == "__main__":
    main()
