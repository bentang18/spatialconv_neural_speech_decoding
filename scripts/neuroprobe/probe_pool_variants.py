"""Electrode-pooling variants for the CS/WS readout: convex attention pool vs the mean.

Replaces the electrode MEAN that ``v3_probe_encode_r4._pool_parcels`` bakes into the cache with a
convex softmax re-weighting computed in the readout from the unpooled ``enc12_elec`` tap. Time and
band granularity are untouched — electrodes are the ONLY axis pooled, because they are the only
axis that is both cross-subject non-aligned and subject-varying in cardinality (band-time bin 7 of
HGA means the same physical thing in every subject; electrode 7 does not).

    w_{c,band} ∝ exp(β · zscore_within_segment(‖x_{c,band}‖)),   Σ_segment w = 1

TWO VARIANTS, ONE OPERATOR. They differ only in the softmax's normalization DOMAIN:

  A (per-parcel)      segment = parcel_canon        → (n, |P|, k_full, 256), CS parcel intersection
  B (indiscriminate)  segment = all electrodes      → (n,   1, k_full, 256), NO atlas, NO intersection

Indexing w by (electrode, band, parcel) would add no degrees of freedom — ``parcel_canon`` is a
map, so contact c determines its parcel. The segment id IS the whole difference.

WEIGHTS ARE PER (ELECTRODE, BAND), shared across time within a band — three spatial filters, one
per band, which is the same premise that justifies three bands in the frontend at all (the contact
carrying the HGA speech response need not carry the 2-14 Hz envelope). Per-time-BIN weights are
deliberately NOT offered: they draw consecutive bins from a shifting spatial support, so the pooled
trace stops being a coherent time series — which is exactly the temporal granularity this design is
trying to protect.

β IS DIMENSIONLESS. ‖x‖ carries arbitrary scale that differs by band and subject, so raw
exp(β‖x‖) would mean something different in every segment (β=1 ≈ mean in one band, ≈ argmax in
another) and the β axis would be uninterpretable. The norms are z-scored WITHIN the segment first.
Consequences: β=0 is EXACTLY the arithmetic mean, and β is comparable across bands and subjects.

PARITY (``--mode assert``, run it first): variant A at β=0 must reproduce the cache's own ``enc12``
tap, because ``enc12_elec`` is documented as that tap's exact pre-mean input (encode line 306-308,
"the diff is the pooling and nothing else"). If it does not match, the segment map or the
(n, C, k_full, 256) reshape is wrong and every AUROC downstream is garbage. Same consistency check
task #11 pre-registered ("mean-only must reproduce ~0.6004").

Ridge, AUROC, standardization, splits, cohort and anchor are IMPORTED from v3_probe_readout_r4 —
never reimplemented — so a pooling delta is a pooling delta and not a protocol delta. λ stays at
that module's CONST_LAM_MULT.

SCOPE (Ben 2026-07-24): this is a side-by-side, NOT a change to the pretrain probe's reported
protocol. The probe's headline CS/WS numbers stay mean-pooled, so the LR pick reads off an
instrument that did not move.

Requires ``--elec-taps 12`` at encode. Variant A additionally requires ``parcel_canon`` in the
payload; without it A is SKIPPED LOUDLY and B still runs (B needs no anatomy at all).

Usage (CPU, NCSA Delta — absolute paths enforced):
  python scripts/neuroprobe/probe_pool_variants.py --mode assert \
      --cache-dir /projects/bhqk/htang13/v3_probe_cache_r6_10k_lr3e-3 --tag r6_10k_lr3e-3
  python scripts/neuroprobe/probe_pool_variants.py --mode all \
      --cache-dir /projects/bhqk/htang13/v3_probe_cache_r6_10k_lr3e-3 --tag r6_10k_lr3e-3 \
      --out /projects/bhqk/htang13/results_pool_variants_10k_lr3e-3.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v3_probe_readout_r4 import (  # noqa: E402
    CS_TEST_SUBJECTS,
    CS_TRAIN_ANCHOR,
    NORMS,
    PROBE_COHORT_7,
    PROBE_TASKS,
    _finite,
    _load,
    _ridge_test,
)

ENC_D_MODEL = 256
ELEC_TAP = "enc12_elec"
POOLED_TAP = "enc12"
# Variant B has no atlas, so its single segment gets a sentinel id. CS then intersects seg_atlas
# uniformly for BOTH variants — B's [-1] ∩ [-1] = [-1] — with no special case in _cs_cell_pooled.
GLOBAL_ATLAS_ID = -1
DEFAULT_BETAS = (0.0, 0.5, 1.0, 2.0, 4.0)
# fp16 relative precision is 2^-11 ≈ 4.9e-4; the stored enc12 is an fp16 round of an fp32 mean and
# our weighted sum accumulates in a different order, so demand agreement at the storage floor's
# scale rather than bitwise.
PARITY_RTOL = 2e-3
N_WS = len(PROBE_COHORT_7)
N_ARRAY = N_WS + len(CS_TEST_SUBJECTS)


def _band_slices(rec, k_full) -> list[tuple[int, int]]:
    """band_lengths → [(lo,hi)] over the k_full time axis. The three bands are decimated 8/2/1 by
    the stem, so they occupy DIFFERENT numbers of time bins and the boundaries are not uniform."""
    lens = [int(x) for x in rec["band_lengths"]]
    if sum(lens) != k_full:
        raise SystemExit(f"band_lengths {lens} sum to {sum(lens)}, but k_full={k_full}")
    out, lo = [], 0
    for n in lens:
        out.append((lo, lo + n))
        lo += n
    return out


def _segments(rec, variant, min_elecs=1, max_elecs=10**9):
    """→ (list of electrode-column arrays, seg_atlas ids). Segment order == present_parcels order
    for A, which is what makes the β=0 parity check against the stored enc12 meaningful."""
    n_contacts = rec["feats"][ELEC_TAP]["raw"].shape[1]
    if variant == "B":
        return [np.arange(n_contacts, dtype=np.int64)], np.asarray([GLOBAL_ATLAS_ID], dtype=np.int64)
    pc = rec.get("parcel_canon")
    if pc is None:
        raise KeyError("parcel_canon")
    pc = np.asarray(pc, dtype=np.int64)
    if pc.shape[0] != n_contacts:
        raise SystemExit(f"parcel_canon has {pc.shape[0]} contacts but {ELEC_TAP} has {n_contacts}")
    cols, atlas = [], []
    for p in np.asarray(rec["present_parcels"], dtype=np.int64):
        c = np.where(pc == p)[0]
        if c.size and min_elecs <= c.size <= max_elecs:
            cols.append(c.astype(np.int64))
            atlas.append(int(p))
    return cols, np.asarray(atlas, dtype=np.int64)


def _band_weights(blk, beta: float):
    """blk (m, c, k_band, d) → convex weights (m, c). β=0 returns EXACTLY 1/c (the mean)."""
    m, c = blk.shape[0], blk.shape[1]
    if c == 1 or beta == 0.0:
        return torch.full((m, c), 1.0 / c, dtype=blk.dtype)
    nrm = blk.reshape(m, c, -1).norm(dim=2)                       # (m, c) per-electrode band energy
    mu = nrm.mean(dim=1, keepdim=True)
    sd = nrm.std(dim=1, unbiased=False, keepdim=True)             # biased: c=2 must not blow up
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    return torch.softmax(beta * (nrm - mu) / sd, dim=1)


def _pool(rec, rows, seg_cols, band_slices, beta: float, chunk: int) -> np.ndarray:
    """(rows, S, F) fp32. Chunked over rows: the fp32 expansion of the elec tap is the memory
    blowup (the fp16 cache itself stays resident once), so it is materialized a chunk at a time."""
    elec = rec["feats"][ELEC_TAP]["raw"]                           # torch fp16 (n, C, F)
    rows = np.asarray(rows, dtype=np.int64)
    f = int(elec.shape[2])
    k = f // ENC_D_MODEL
    if k * ENC_D_MODEL != f:
        raise SystemExit(f"{ELEC_TAP} feature dim {f} is not a multiple of d={ENC_D_MODEL}")
    out = np.empty((rows.shape[0], len(seg_cols), f), dtype=np.float32)
    for i in range(0, rows.shape[0], chunk):
        idx = rows[i:i + chunk]
        m = idx.shape[0]
        x = elec[torch.from_numpy(idx)].to(torch.float32).reshape(m, -1, k, ENC_D_MODEL)
        buf = torch.empty((m, len(seg_cols), k, ENC_D_MODEL), dtype=torch.float32)
        for s, cols in enumerate(seg_cols):
            xs = x[:, torch.from_numpy(cols)]                      # (m, |cols|, k, d)
            for lo, hi in band_slices:
                blk = xs[:, :, lo:hi, :]
                w = _band_weights(blk, beta)
                buf[:, s, lo:hi, :] = (w[:, :, None, None] * blk).sum(dim=1)
        out[i:i + m] = buf.reshape(m, len(seg_cols), f).numpy()
    return out


def _feat_rows(z, rows, seg_idx=None) -> np.ndarray:
    x = z[np.asarray(rows, dtype=np.int64)]
    if seg_idx is not None:
        x = x[:, np.asarray(seg_idx, dtype=np.int64)]
    return x.reshape(x.shape[0], -1)


def _ws_session_pooled(z, rec, task, norm) -> float:
    """z (n, S, F) — pooled ONCE per (variant, β) and indexed per fold, never re-pooled."""
    y = np.asarray(rec["labels"][task], dtype=np.float64)
    folds = []
    for sp in rec["ws_split"][task].values():
        tr, te = _finite(y, sp["train"]), _finite(y, sp["test"])
        if len(tr) < 2 or len(te) < 2:
            folds.append(float("nan"))
            continue
        folds.append(_ridge_test(_feat_rows(z, tr), y[tr], _feat_rows(z, te), y[te], norm))
    return float(np.nanmean(folds)) if folds else float("nan")


def _cs_cell_pooled(z_a, a_atlas, anchor_rec, z_t, t_atlas, test_rec, task, norm) -> float:
    y_a = np.asarray(anchor_rec["labels"][task], dtype=np.float64)
    y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
    tr = _finite(y_a, np.arange(len(y_a)))
    te = _finite(y_t, test_rec["cs_split"][task]["test"])
    if len(tr) < 2 or len(te) < 2:
        return float("nan")
    common = np.intersect1d(a_atlas, t_atlas)                      # B: [-1] ∩ [-1] = [-1]
    if common.size == 0:
        return float("nan")
    a_idx = [int(np.where(a_atlas == c)[0][0]) for c in common]
    t_idx = [int(np.where(t_atlas == c)[0][0]) for c in common]
    return _ridge_test(_feat_rows(z_a, tr, a_idx), y_a[tr],
                       _feat_rows(z_t, te, t_idx), y_t[te], norm)


def _prep(rec, variant, min_e, max_e):
    """→ (seg_cols, seg_atlas, band_slices) or None when the variant is unavailable/empty."""
    f = int(rec["feats"][ELEC_TAP]["raw"].shape[2])
    bslices = _band_slices(rec, f // ENC_D_MODEL)
    try:
        seg_cols, seg_atlas = _segments(rec, variant, min_e, max_e)
    except KeyError:
        print(f"[skip] variant {variant}: cache has no 'parcel_canon' — re-encode with the "
              f"payload fix, or run variant B only (it needs no anatomy).", flush=True)
        return None
    if not seg_cols:
        print(f"[skip] variant {variant}: no segment survived the electrode-count filter "
              f"[{min_e},{max_e}]", flush=True)
        return None
    return seg_cols, seg_atlas, bslices


def _key(variant, beta, norm, task) -> str:
    return f"{variant}|b{beta:g}|{norm}|{task}"


def _parcel_sizes(rec) -> dict:
    """Always recorded: the β effect should concentrate in well-sampled parcels if the mechanism
    is the mean's 1/n dilution. Small parcels make the softmax high-variance."""
    pc = rec.get("parcel_canon")
    if pc is None:
        return {}
    pc = np.asarray(pc, dtype=np.int64)
    return {int(p): int(np.sum(pc == p)) for p in np.asarray(rec["present_parcels"], dtype=np.int64)}


# ── shards (mirrors v3_probe_readout_r4's 13-way array: 7 WS sessions + 6 CS test subjects) ──
def _ws_shard(cache_dir, tag, session, variants, betas, chunk, min_e, max_e) -> dict:
    rec = _load(cache_dir, session, tag)
    cells = {}
    for variant in variants:
        prep = _prep(rec, variant, min_e, max_e)
        if prep is None:
            continue
        seg_cols, _seg_atlas, bslices = prep
        rows = np.arange(int(rec["feats"][ELEC_TAP]["raw"].shape[0]))
        for beta in betas:
            z = _pool(rec, rows, seg_cols, bslices, beta, chunk)
            for norm in NORMS:
                for task in PROBE_TASKS:
                    cells[_key(variant, beta, norm, task)] = _ws_session_pooled(z, rec, task, norm)
            del z
            print(f"[ws S{session[0]}T{session[1]}] {variant} β={beta:g} done "
                  f"({len(seg_cols)} segments)", flush=True)
    return {"kind": "ws", "name": f"S{session[0]}T{session[1]}", "cells": cells,
            "parcel_sizes": _parcel_sizes(rec)}


def _cs_shard(cache_dir, tag, test_subject, variants, betas, chunk, min_e, max_e) -> dict:
    anchor_rec = _load(cache_dir, CS_TRAIN_ANCHOR, tag)
    ts = (test_subject, next(t for (s, t) in PROBE_COHORT_7 if s == test_subject))
    test_rec = _load(cache_dir, ts, tag)
    cells = {}
    for variant in variants:
        pa, pt = _prep(anchor_rec, variant, min_e, max_e), _prep(test_rec, variant, min_e, max_e)
        if pa is None or pt is None:
            continue
        a_cols, a_atlas, a_bs = pa
        t_cols, t_atlas, t_bs = pt
        a_rows = _finite(np.asarray(anchor_rec["labels"][PROBE_TASKS[0]], dtype=np.float64),
                         np.arange(int(anchor_rec["feats"][ELEC_TAP]["raw"].shape[0])))
        t_rows = np.arange(int(test_rec["feats"][ELEC_TAP]["raw"].shape[0]))
        for beta in betas:
            z_a = _pool(anchor_rec, np.arange(int(anchor_rec["feats"][ELEC_TAP]["raw"].shape[0])),
                        a_cols, a_bs, beta, chunk)
            z_t = _pool(test_rec, t_rows, t_cols, t_bs, beta, chunk)
            for norm in NORMS:
                for task in PROBE_TASKS:
                    cells[_key(variant, beta, norm, task)] = _cs_cell_pooled(
                        z_a, a_atlas, anchor_rec, z_t, t_atlas, test_rec, task, norm)
            del z_a, z_t
            print(f"[cs S{test_subject}] {variant} β={beta:g} done "
                  f"(|anchor∩test|={np.intersect1d(a_atlas, t_atlas).size}, "
                  f"anchor finite rows={a_rows.size})", flush=True)
    return {"kind": "cs", "test": f"S{test_subject}", "cells": cells}


# ── parity: variant A at β=0 must be the cache's own enc12 ────────────────────────────
def _parity(cache_dir, tag, session, chunk, n_rows) -> bool:
    rec = _load(cache_dir, session, tag)
    if ELEC_TAP not in rec["feats"]:
        raise SystemExit(f"cache has no '{ELEC_TAP}' — re-encode with --elec-taps 12")
    prep = _prep(rec, "A", 1, 10**9)
    if prep is None:
        raise SystemExit("parity needs variant A (parcel_canon); cache is missing it")
    seg_cols, seg_atlas, bslices = prep
    n = int(rec["feats"][ELEC_TAP]["raw"].shape[0])
    rows = np.arange(min(n_rows, n))
    got = _pool(rec, rows, seg_cols, bslices, 0.0, chunk)              # (r, |P|, F)
    ref = rec["feats"][POOLED_TAP]["raw"][torch.from_numpy(rows)].to(torch.float32).numpy()
    present = np.asarray(rec["present_parcels"], dtype=np.int64)
    if got.shape != ref.shape:
        print(f"[parity] FAIL shape {got.shape} vs stored enc12 {ref.shape}", flush=True)
        return False
    if not np.array_equal(seg_atlas, present):
        print(f"[parity] FAIL segment order != present_parcels order", flush=True)
        return False
    scale = float(np.std(ref)) or 1.0
    rel = float(np.max(np.abs(got - ref))) / scale
    ok = rel <= PARITY_RTOL
    print(f"[parity] variant A β=0 vs stored enc12 on {rows.size} rows: "
          f"max|Δ|/std(ref) = {rel:.2e} (tol {PARITY_RTOL:.0e}) → {'OK' if ok else 'VIOLATED'}",
          flush=True)
    if not ok:
        print("[parity] the segment map or the (n,C,k_full,256) reshape is wrong — every AUROC "
              "below it would be garbage. Do NOT read the results.", flush=True)
    return ok


# ── merge / report ───────────────────────────────────────────────────────────────────
def _empty(variants, betas) -> dict:
    return {_key(v, b, nm, t): {"ws_per_session": {}, "cs_per_test": {}}
            for v in variants for b in betas for nm in NORMS for t in PROBE_TASKS}


def _finalize(cells) -> dict:
    for c in cells.values():
        ws, cs = list(c["ws_per_session"].values()), list(c["cs_per_test"].values())
        c["ws_cohort"] = float(np.nanmean(ws)) if ws else float("nan")
        c["cs_mean"] = float(np.nanmean(cs)) if cs else float("nan")
    return cells


def _merge(shard_dir, variants, betas) -> tuple[dict, dict]:
    cells, sizes = _empty(variants, betas), {}
    for path in sorted(glob.glob(f"{shard_dir}/ws_*.json")):
        with open(path) as f:
            sh = json.load(f)
        sizes[sh["name"]] = sh.get("parcel_sizes", {})
        for key, val in sh["cells"].items():
            cells.setdefault(key, {"ws_per_session": {}, "cs_per_test": {}})
            cells[key]["ws_per_session"][sh["name"]] = val
    for path in sorted(glob.glob(f"{shard_dir}/cs_*.json")):
        with open(path) as f:
            sh = json.load(f)
        for key, val in sh["cells"].items():
            cells.setdefault(key, {"ws_per_session": {}, "cs_per_test": {}})
            cells[key]["cs_per_test"][sh["test"]] = val
    return _finalize(cells), sizes


def _report(cells, variants, betas) -> None:
    """β ladder per variant, and the delta vs β=0 — which IS the hypothesis test, since β=0 is
    exactly the current mean-pool baseline."""
    for direction, key in (("CS mean over test subjects", "cs_mean"), ("WS cohort mean", "ws_cohort")):
        print(f"\n=== pooling β ladder — {direction} ===", flush=True)
        for variant in variants:
            for norm in NORMS:
                for task in PROBE_TASKS:
                    base = cells.get(_key(variant, 0.0, norm, task), {}).get(key, float("nan"))
                    row = []
                    for b in betas:
                        v = cells.get(_key(variant, b, norm, task), {}).get(key, float("nan"))
                        row.append(f"β{b:g}:{v:.4f}({v - base:+.4f})")
                    print(f"  {variant} {norm:3s} {task:16s} " + "  ".join(row), flush=True)
    print("\n(Δ is vs β=0, which IS the current mean-pool baseline — so Δ is the whole result.)",
          flush=True)


def _require_abs(path, what) -> str:
    if not os.path.isabs(path):
        raise SystemExit(f"{what} must be an ABSOLUTE path (standing rule), got: {path}")
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", choices=("assert", "all", "array", "merge"), default="all")
    p.add_argument("--variants", default="A,B", help="A=per-parcel, B=indiscriminate")
    p.add_argument("--betas", default=",".join(f"{b:g}" for b in DEFAULT_BETAS))
    p.add_argument("--chunk", type=int, default=64, help="rows per fp32 pooling chunk")
    p.add_argument("--min-parcel-elecs", type=int, default=1,
                   help="variant A stratification: drop parcels with fewer electrodes")
    p.add_argument("--max-parcel-elecs", type=int, default=10**9)
    p.add_argument("--parity-rows", type=int, default=256)
    p.add_argument("--array-index", type=int)
    p.add_argument("--shard-dir")
    p.add_argument("--out")
    args = p.parse_args()

    _require_abs(args.cache_dir, "--cache-dir")
    for opt, val in (("--shard-dir", args.shard_dir), ("--out", args.out)):
        if val is not None:
            _require_abs(val, opt)
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    bad = [v for v in variants if v not in ("A", "B")]
    if bad:
        raise SystemExit(f"unknown --variants {bad}; known: ['A', 'B']")
    betas = tuple(float(b) for b in args.betas.split(",") if b.strip())
    if 0.0 not in betas:
        raise SystemExit("--betas must include 0 — it is the mean-pool baseline every Δ is "
                         "measured against, and the parity anchor")
    shard = dict(variants=variants, betas=betas, chunk=args.chunk,
                 min_e=args.min_parcel_elecs, max_e=args.max_parcel_elecs)

    if args.mode == "assert":
        ok = _parity(args.cache_dir, args.tag, CS_TRAIN_ANCHOR, args.chunk, args.parity_rows)
        raise SystemExit(0 if ok else 1)

    if args.mode == "array":
        if args.shard_dir is None or args.array_index is None:
            raise SystemExit("mode=array needs --shard-dir and --array-index")
        os.makedirs(args.shard_dir, exist_ok=True)
        n = args.array_index
        if n < N_WS:
            sh = _ws_shard(args.cache_dir, args.tag, PROBE_COHORT_7[n], **shard)
            path = f"{args.shard_dir}/ws_{sh['name']}_{args.tag}.json"
        elif n < N_ARRAY:
            sh = _cs_shard(args.cache_dir, args.tag, CS_TEST_SUBJECTS[n - N_WS], **shard)
            path = f"{args.shard_dir}/cs_{sh['test']}_{args.tag}.json"
        else:
            raise SystemExit(f"--array-index {n} out of range 0..{N_ARRAY - 1}")
        with open(path, "w") as f:
            json.dump(sh, f)
        print(f"[array {n}] {sh['kind']} -> {path}", flush=True)
        return

    if args.mode == "merge":
        if args.shard_dir is None or args.out is None:
            raise SystemExit("mode=merge needs --shard-dir and --out")
        cells, sizes = _merge(args.shard_dir, variants, betas)
    else:  # all — serial, for a single node
        if args.out is None:
            raise SystemExit("mode=all needs --out")
        cells, sizes = _empty(variants, betas), {}
        for session in PROBE_COHORT_7:
            sh = _ws_shard(args.cache_dir, args.tag, session, **shard)
            sizes[sh["name"]] = sh.get("parcel_sizes", {})
            for key, val in sh["cells"].items():
                cells.setdefault(key, {"ws_per_session": {}, "cs_per_test": {}})
                cells[key]["ws_per_session"][sh["name"]] = val
        for s in CS_TEST_SUBJECTS:
            sh = _cs_shard(args.cache_dir, args.tag, s, **shard)
            for key, val in sh["cells"].items():
                cells.setdefault(key, {"ws_per_session": {}, "cs_per_test": {}})
                cells[key]["cs_per_test"][sh["test"]] = val
        cells = _finalize(cells)

    payload = {"tag": args.tag, "cache_dir": args.cache_dir, "variants": list(variants),
               "betas": list(betas), "norms": list(NORMS),
               "parcel_elec_counts": sizes,
               "config": {"min_parcel_elecs": args.min_parcel_elecs,
                          "max_parcel_elecs": args.max_parcel_elecs, "chunk": args.chunk},
               "cells": cells}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    _report(cells, variants, betas)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
