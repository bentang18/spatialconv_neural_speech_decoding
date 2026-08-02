#!/usr/bin/env python
"""Slim enc0-only board cache — the delta the frontend ablation actually needs.

A board session record is 44-71 GB, and ~99% of it is the encoder taps (enc12_elec alone is
119 x 13312 per window). The `fm:` frontend ablation touches ONLY enc0/enc0_elec — 348 columns
per unit — so every arm would otherwise pay a 50 GB eager load to read 0.8 GB. This copies the
two enc0 taps plus the record's metadata into a slim record with the SAME payload schema, so
`v3_board_readout.py` consumes it unchanged (`--cache-dir <slim> --tags <out-tag>`).

Not an optimization for its own sake: it is what turns the ablation from a 120-200G/cell job
into a ~24G/cell one, which on a partition that runs ~133/136 allocated is the difference
between backfilling immediately and queueing behind whole-node requests. Delta CPU bills
MAX_TRES == mem_MB/2, so the slim cache also cuts the bill ~6x for the same arms.

ELEC_LABELS ARE EMBEDDED. CSession needs `elec_labels` for its cross-session identity intersect
and the encoder never writes it; the readout's sidecar path reads it from a pickle and validates
it against `enc12_elec`, which a slim record does not carry. Writing the labels INTO the record
removes the dependency entirely (`_load` skips the sidecar branch when the field is present) and
keeps the check: the label count is asserted against enc0_elec here instead.

The guards are the point. A slim cache that silently drops rows, reorders units, or carries a
band layout that does not describe its own payload would produce a complete, plausible ablation
grid answering a different question. So every invariant is asserted AND printed:
  * the band layout describes the payload  (sum_b T_b*F_b == stored width)
  * rows survive exactly                   (n_windows, per-tap row count)
  * units survive exactly                  (parcels vs present_parcels, elecs vs elec_labels)
  * the copy is bit-exact                  (fp16 in, fp16 out, checksum printed per tap)

Usage (Delta CPU, one array cell per session):
  python scripts/neuroprobe/fe_abl_slim_cache.py --cache-dir <full> --tag pbs50_cd45k \
      --out-dir <slim> --out-tag feabl --index $SLURM_ARRAY_TASK_ID \
      --sidecar /projects/bhqk/htang13/arm0_elec_labels_sidecar.pkl
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import torch

# The 12 Neuroprobe-Lite sessions, in the readout's own order so --index means the same thing
# in both jobs (v3_board_readout.py:LITE_SESSIONS).
LITE_SESSIONS = ((1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
                 (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1))
KEEP_TAPS = ("enc0", "enc0_elec")
# Copied verbatim; everything the readout reads off a record other than `feats`.
KEEP_KEYS = ("subject_id", "trial_id", "ckpt_tag", "present_parcels", "parcel_canon",
             "band_lengths", "band_fdims", "clip_starts", "labels", "ws_split", "cs_split",
             "n_windows")


def _checksum(t: torch.Tensor) -> float:
    """Order-sensitive fingerprint of a tap, printed so slim-vs-full parity is checkable."""
    x = t.to(torch.float64)
    return float((x * torch.arange(1, x.shape[-1] + 1, dtype=torch.float64)).sum().item())


def slim(rec: dict, elec_labels=None) -> dict:
    """Full board record → enc0-only record, with every invariant asserted and printed."""
    tl = [int(v) for v in np.asarray(rec["band_lengths"]).ravel()]
    fd = [int(v) for v in np.asarray(rec["band_fdims"]).ravel()]
    width = sum(t * f for t, f in zip(tl, fd))
    n_win = int(rec["n_windows"])
    out: dict = {k: rec[k] for k in KEEP_KEYS if k in rec}
    out["feats"] = {}
    for tap in KEEP_TAPS:
        if tap not in rec["feats"]:
            raise SystemExit(f"record has no {tap!r} — this is not an enc0-bearing board cache")
        raw = rec["feats"][tap]["raw"]
        if raw.shape[-1] != width:
            raise SystemExit(
                f"{tap}: stored width {raw.shape[-1]} != sum_b T_b*F_b = {width} from "
                f"band_lengths {tl} / band_fdims {fd} — the layout does not describe this cache")
        if raw.shape[0] != n_win:
            raise SystemExit(f"{tap}: {raw.shape[0]} rows != n_windows {n_win}")
        if raw.dtype != torch.float16:
            raise SystemExit(f"{tap}: dtype {raw.dtype}, expected float16 — copy would not be exact")
        dense = raw.clone().contiguous()          # materialize out of the mmap
        if not torch.equal(dense, raw):
            raise SystemExit(f"{tap}: copy is not bit-exact")
        out["feats"][tap] = {"raw": dense}
        print(f"[check] {tap}: {tuple(dense.shape)} {dense.dtype} "
              f"= {tl} frames x {fd} bins = {width}  checksum {_checksum(dense[:1]):.6e}",
              flush=True)
    n_parcels = out["feats"]["enc0"]["raw"].shape[1]
    if n_parcels != len(np.asarray(rec["present_parcels"]).ravel()):
        raise SystemExit(f"enc0 has {n_parcels} parcel columns but present_parcels lists "
                         f"{len(np.asarray(rec['present_parcels']).ravel())}")
    n_elec = out["feats"]["enc0_elec"]["raw"].shape[1]
    if elec_labels is not None:
        if len(elec_labels) != n_elec:
            raise SystemExit(f"sidecar labels ({len(elec_labels)}) != enc0_elec electrodes "
                             f"({n_elec}) — wrong session or a stale sidecar")
        out["elec_labels"] = elec_labels
    print(f"[check] units: {n_parcels} parcels, {n_elec} electrodes, {n_win} windows; "
          f"elec_labels {'EMBEDDED' if elec_labels is not None else 'ABSENT (csession will be empty)'}",
          flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-tag", required=True)
    p.add_argument("--index", type=int, required=True, help="index into LITE_SESSIONS")
    p.add_argument("--sidecar", default=None, help="arm0 elec-labels pickle, keyed 's{S}_t{T}'")
    a = p.parse_args()

    s, t = LITE_SESSIONS[a.index]
    src = f"{a.cache_dir}/enc_s{s}_t{t}_{a.tag}.pt"
    dst = f"{a.out_dir}/enc_s{s}_t{t}_{a.out_tag}.pt"
    print(f"[slim] s{s}_t{t}  {src} -> {dst}", flush=True)

    labels = None
    if a.sidecar:
        with open(a.sidecar, "rb") as fh:
            labels = pickle.load(fh).get(f"s{s}_t{t}")
        if labels is None:
            raise SystemExit(f"sidecar has no entry for s{s}_t{t}")

    rec = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    out = slim(rec, labels)
    torch.save(out, dst)
    print(f"[slim] wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
