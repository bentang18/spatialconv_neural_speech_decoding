"""r4 depth-ladder probe — Stage 2 readout (WS + CS), keep-time, parcel-pooled.

Consumes the pooled caches from ``v3_probe_encode_r4.py``: per session, per encoder tap
(enc0/enc3/enc6/enc12), a ``(n_win, |P|, F)`` fp16 feature where electrodes are ALREADY
pooled to parcels (MEAN over electrodes per band-slot) at encode. So both directions are cheap:

  - WS (within-session): flatten ALL present parcels → (n, |P|·F), row-subset both folds,
    dual ridge, balanced AUROC (avg of 2 folds). NB parcel-pooled (Ben 2026-07-15, OOM guard)
    — NOT electrode-resolved like r1's WS, so WS is not directly comparable to r1's WS number;
    the enc0→enc12 ORDERING within this run is what the depth ladder reads.
  - CS (cross-subject): fit the anchor (2,1) on all finite rows, transfer to each held-out
    test subject's cs_split test half, over the anchor∩test PARCEL intersection (same v2
    direction as r1). Parcel columns aligned by atlas id.

Perceiver taps (Ben 2026-07-16) ride the same caches: ``dec``/``lat`` (+ ``_rand`` fresh-init
twins). ``dec`` has the enc taps' parcel axis and takes the normal CS intersection; ``lat`` is
PARCEL-FREE (see PARCEL_FREE_TAPS) and is CS-scored WHOLE — no intersection, 6144 dims in every
subject. Read the perceiver block against the _rand twin, not against enc12: tap-vs-twin is a
matched contrast, tap-vs-enc12 crosses the teacher-vs-context stream caveat.

Ridge/metric primitives INLINED from the r1 readout — zero speech_decoding dep, runs on the
stock NCSA Delta pytorch module. λ held CONSTANT (lam_mult=1.0) across every cell. Sole
divergence from r1: the ridge GEMMs run fp32 (r1 was fp64 at a much narrower, electrode-
resolved feature; at r4's |P|·F=186k a fp64 z_anchor is 2.6 GB and the array went NUMA-bound).
The n×n solve stays fp64. Shards from the two dtypes must NOT be merged — rerun all 13.

Depth-ladder invariant (feedback-build-the-invariant-into-the-probe): enc0 is the input floor;
the print at the end shows CS(enc0) vs CS(enc3/6/12) so the M9 "no block earns anything"
ordering is read directly, per encoder.

Usage (CPU, NCSA Delta):
  python scripts/neuroprobe/v3_probe_readout_r4.py \
      --cache-dir /projects/bhqk/htang13/v3_probe_cache_r4_10k --tags r4_10k \
      --out results_v3_probe_r4.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

PROBE_TASKS = ("onset", "delta_volume", "word_index", "gpt2_surprisal")
CS_TRAIN_ANCHOR = (2, 1)
CS_TEST_SUBJECTS = (1, 3, 4, 6, 8, 9)
ENCODERS = ("enc0", "enc3", "enc6", "enc12")
# Write-only Perceiver taps (Ben 2026-07-16), each with its fresh-init control twin.
#   dec  (n,|P|,S·128)  — the per-(parcel,slot) query read, PRE-head. Parcel axis == the enc
#                         taps' `present_parcels`, so the CS intersection needs no special case.
#   lat  (n,1,S·M·128)  — the processed latent bank. PARCEL-FREE by construction (the 12 latents
#                         are shared learned params), so it is natively subject-independent and
#                         CS takes it WHOLE: no intersection, identical 6144 dims every session.
#                         It is also dec's information CEILING (dec = f(lat, parcel, slot)).
PERC_TAPS = ("dec", "lat", "dec_rand", "lat_rand")
PARCEL_FREE_TAPS = ("lat", "lat_rand")
ALL_TAPS = ENCODERS + PERC_TAPS
# Readout conditioning of the raw parcel-mean feature. std = per-feature z-score on TRAIN stats
# (the FM linear-probe convention; also removes the enc0-is-already-normalized asymmetry — enc0
# is robust-z'd pre-pool while taps are at network scale). raw = as-cached (r1/M9-comparable).
# std PRIMARY, raw the cross-check: they can disagree if signal is variance-concentrated (raw)
# vs spread across many dims (std) — that disagreement is itself diagnostic of collapse.
NORMS = ("std", "raw")
CONST_LAM_MULT = 1.0
PROBE_COHORT_7 = ((1, 0), (2, 1), (3, 2), (4, 2), (6, 0), (8, 0), (9, 0))


def dual_ridge_scores(z_train, y_train, z_test, lam=None, lam_mult=1.0):
    """G=ZZᵀ, α=(G+λI)⁻¹y, lam=lam_mult·tr(G)/N (from online_probe; GEMMs in fp32).

    The n²·D Gram dominates (~2.3 TFLOP/cell at |P|=14), so the GEMMs run fp32; the n×n
    solve is promoted back to fp64, where conditioning matters and cost is ~2 orders less.
    """
    z_train = np.asarray(z_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float64)
    z_test = np.asarray(z_test, dtype=np.float32)
    g = (z_train @ z_train.T).astype(np.float64)
    n = g.shape[0]
    if lam is None:
        lam = lam_mult * float(np.trace(g) / max(n, 1))
    alpha = np.linalg.solve(g + lam * np.eye(n), y_train)
    return (z_test @ z_train.T).astype(np.float64) @ alpha


def auroc(scores, labels):
    from sklearn.metrics import roc_auc_score

    y = (np.asarray(labels) > 0).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores)))


def _finite(y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    r = np.asarray(rows, dtype=np.int64)
    return r[np.isfinite(y[r])]


def _standardize(z_tr, z_te):
    """Per-feature z-score on TRAIN stats only, applied to both (never fit on test).
    Constant columns (σ=0) → σ:=1 so they map to 0 rather than blowing up."""
    mu = z_tr.mean(axis=0)
    sd = z_tr.std(axis=0)
    sd[sd == 0] = 1.0
    return (z_tr - mu) / sd, (z_te - mu) / sd


def _ridge_test(z_tr, y_tr, z_te, y_te, norm) -> float:
    if len(y_tr) < 2 or len(y_te) < 2:
        return float("nan")
    if norm == "std":
        z_tr, z_te = _standardize(z_tr, z_te)
    return auroc(dual_ridge_scores(z_tr, y_tr, z_te, lam_mult=CONST_LAM_MULT), y_te)


def _feat(rec, enc, rows) -> np.ndarray:
    """(n,|P|,F) fp16 raw cache → selected rows, all parcels, flat fp32 → (r, |P|·F)."""
    x = rec["feats"][enc]["raw"]                                # torch fp16 (n,|P|,F)
    x = x[np.asarray(rows, dtype=np.int64)].to(torch.float32).numpy()
    return x.reshape(x.shape[0], -1)


def _feat_parcels(rec, enc, rows, col_idx) -> np.ndarray:
    """Same, restricted to parcel columns `col_idx` (CS intersection), flat → (r, |cols|·F)."""
    x = rec["feats"][enc]["raw"]
    x = x[np.asarray(rows, dtype=np.int64)][:, np.asarray(col_idx, dtype=np.int64)]
    x = x.to(torch.float32).numpy()
    return x.reshape(x.shape[0], -1)


def _ws_session(rec, enc, task, norm) -> float:
    y = np.asarray(rec["labels"][task], dtype=np.float64)
    folds = []
    for sp in rec["ws_split"][task].values():
        tr, te = _finite(y, sp["train"]), _finite(y, sp["test"])
        if len(tr) < 2 or len(te) < 2:
            folds.append(float("nan"))
            continue
        folds.append(_ridge_test(_feat(rec, enc, tr), y[tr],
                                 _feat(rec, enc, te), y[te], norm))
    return float(np.nanmean(folds))


def _cs_cell(anchor_rec, test_rec, enc, task, norm) -> float:
    y_anchor = np.asarray(anchor_rec["labels"][task], dtype=np.float64)
    y_test = np.asarray(test_rec["labels"][task], dtype=np.float64)
    tr = _finite(y_anchor, np.arange(len(y_anchor)))
    te = _finite(y_test, test_rec["cs_split"][task]["test"])
    if len(tr) < 2 or len(te) < 2:
        return float("nan")

    if enc in PARCEL_FREE_TAPS:
        # No parcel axis to intersect — the latent bank is the SAME 12 shared learned slots in
        # every subject, so slot m at time s already means the same thing on both sides and the
        # feature is dimension-matched by construction. This is the cleanest CS transfer in the
        # suite: no intersection, no montage bridge, full 6144 dims.
        return _ridge_test(_feat(anchor_rec, enc, tr), y_anchor[tr],
                           _feat(test_rec, enc, te), y_test[te], norm)

    a_p = np.asarray(anchor_rec["present_parcels"], dtype=np.int64)
    t_p = np.asarray(test_rec["present_parcels"], dtype=np.int64)
    common = np.intersect1d(a_p, t_p)
    if common.size == 0:
        return float("nan")
    a_idx = [int(np.where(a_p == c)[0][0]) for c in common]     # aligned by atlas id
    t_idx = [int(np.where(t_p == c)[0][0]) for c in common]

    z_anchor = _feat_parcels(anchor_rec, enc, tr, a_idx)
    z_test = _feat_parcels(test_rec, enc, te, t_idx)
    return _ridge_test(z_anchor, y_anchor[tr], z_test, y_test[te], norm)


def _load(cache_dir, session, tag):
    s, t = session
    return torch.load(f"{cache_dir}/enc_s{s}_t{t}_{tag}.pt", map_location="cpu", weights_only=False)


# ── sharded units (one SLURM array task each; all cells are independent) ────────────
def _ws_shard(cache_dir, tag, session, taps=ALL_TAPS) -> dict:
    """All WS cells for ONE session → {key: auroc}. Loads that session's cache only."""
    rec = _load(cache_dir, session, tag)
    cells = {f"{tag}|{enc}|{v}|{task}": _ws_session(rec, enc, task, v)
             for enc in taps for v in NORMS for task in PROBE_TASKS}
    return {"kind": "ws", "name": f"S{session[0]}T{session[1]}", "cells": cells}


def _cs_shard(cache_dir, tag, test_subject, taps=ALL_TAPS) -> dict:
    """All CS cells for ONE held-out test subject → {key: auroc}. Loads anchor + that test."""
    anchor_rec = _load(cache_dir, CS_TRAIN_ANCHOR, tag)
    ts = (test_subject, next(t for (ss, t) in PROBE_COHORT_7 if ss == test_subject))
    test_rec = _load(cache_dir, ts, tag)
    cells = {f"{tag}|{enc}|{v}|{task}": _cs_cell(anchor_rec, test_rec, enc, task, v)
             for enc in taps for v in NORMS for task in PROBE_TASKS}
    return {"kind": "cs", "test": f"S{test_subject}", "cells": cells}


def _empty_cells(tags, taps=ALL_TAPS) -> dict:
    return {f"{tag}|{enc}|{v}|{t}": {"ws_per_session": {}, "cs_per_test": {}}
            for tag in tags for enc in taps for v in NORMS for t in PROBE_TASKS}


def _finalize(cells: dict) -> dict:
    for c in cells.values():
        ws, cs = list(c["ws_per_session"].values()), list(c["cs_per_test"].values())
        c["ws_cohort"] = float(np.nanmean(ws)) if ws else float("nan")
        c["cs_mean"] = float(np.nanmean(cs)) if cs else float("nan")
    return cells


def _merge_shards(tags, shard_dir, taps=ALL_TAPS) -> dict:
    cells = _empty_cells(tags, taps)
    for path in sorted(glob.glob(f"{shard_dir}/ws_*.json")):
        with open(path) as f:
            sh = json.load(f)
        for key, val in sh["cells"].items():
            cells[key]["ws_per_session"][sh["name"]] = val
    for path in sorted(glob.glob(f"{shard_dir}/cs_*.json")):
        with open(path) as f:
            sh = json.load(f)
        for key, val in sh["cells"].items():
            cells[key]["cs_per_test"][sh["test"]] = val
    return _finalize(cells)


def _compute_all(cache_dir, tags, taps=ALL_TAPS) -> dict:
    """Serial path (mode=all): every session + every test in one process."""
    cells = _empty_cells(tags, taps)
    for tag in tags:
        for session in PROBE_COHORT_7:
            sh = _ws_shard(cache_dir, tag, session, taps)
            for key, val in sh["cells"].items():
                cells[key]["ws_per_session"][sh["name"]] = val
            print(f"[{tag}] WS done {sh['name']}", flush=True)
        for s in CS_TEST_SUBJECTS:
            sh = _cs_shard(cache_dir, tag, s, taps)
            for key, val in sh["cells"].items():
                cells[key]["cs_per_test"][sh["test"]] = val
            print(f"[{tag}] CS done {sh['test']}", flush=True)
    return _finalize(cells)


def _print_ladder(tags, results, taps=ALL_TAPS) -> None:
    encs = [e for e in ENCODERS if e in taps]
    if encs:
        for direction, key in (("CS mean over test subjects", "cs_mean"),
                               ("WS cohort mean", "ws_cohort")):
            print(f"\n=== r4 depth ladder — {direction} (per encoder tap) ===", flush=True)
            for tag in tags:
                for v in NORMS:
                    for task in PROBE_TASKS:
                        row = [f"{e}:{results[f'{tag}|{e}|{v}|{task}'][key]:.4f}" for e in encs]
                        print(f"  {direction[:2]} {tag} {v:3s} {task:16s} " + "  ".join(row),
                              flush=True)
    _print_perceiver(tags, results, taps)


def _print_perceiver(tags, results, taps=ALL_TAPS) -> None:
    """Perceiver taps against their fresh-init twins — the contrast that survives the stream
    caveat (both sides carry it identically). lat is CS-scored WHOLE (no parcel intersection);
    dec is a deterministic function of lat, so lat is dec's ceiling and dec>lat means only that
    the parcel query made the same information more linearly accessible."""
    bases = [b for b in ("dec", "lat") if b in taps and f"{b}_rand" in taps]
    if not bases:
        return
    print("\n=== r4 perceiver taps — trained vs fresh-init control (Δ = trained − rand) ===",
          flush=True)
    for tag in tags:
        for v in NORMS:
            for base in bases:
                for task in PROBE_TASKS:
                    def val(suffix, key):
                        return results[f"{tag}|{base}{suffix}|{v}|{task}"][key]
                    cs, cs_r = val("", "cs_mean"), val("_rand", "cs_mean")
                    ws, ws_r = val("", "ws_cohort"), val("_rand", "ws_cohort")
                    print(f"  {tag} {v:3s} {base:3s} {task:16s} "
                          f"CS {cs:.4f} (rand {cs_r:.4f}, Δ{cs - cs_r:+.4f})   "
                          f"WS {ws:.4f} (rand {ws_r:.4f}, Δ{ws - ws_r:+.4f})", flush=True)


# array-index layout: 0..len(cohort)-1 → WS per session; then one per CS test subject
N_WS = len(PROBE_COHORT_7)
N_ARRAY = N_WS + len(CS_TEST_SUBJECTS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--tags", default="r4_10k")
    p.add_argument("--out", default=None, help="final merged JSON (mode all/merge)")
    p.add_argument("--mode", choices=("all", "array", "merge"), default="all")
    p.add_argument("--array-index", type=int, help="mode=array: 0..%d" % (N_ARRAY - 1))
    p.add_argument("--shard-dir", default=None, help="mode=array writes here, mode=merge reads here")
    p.add_argument("--taps", default=",".join(ALL_TAPS),
                   help="comma-separated subset of ALL_TAPS. COMPUTE ONLY THE DELTA: the enc "
                        "ladder is a re-encode-invariant function of the ckpt, so when a tag is "
                        "re-encoded purely to add Perceiver taps, pass --taps dec,lat,dec_rand,"
                        "lat_rand and read enc0/3/6/12 off the existing results JSON.")
    args = p.parse_args()
    tags = tuple(t.strip() for t in args.tags.split(","))
    taps = tuple(t.strip() for t in args.taps.split(","))
    bad = [t for t in taps if t not in ALL_TAPS]
    if bad:
        raise SystemExit(f"unknown --taps {bad}; known: {list(ALL_TAPS)}")

    if args.mode == "array":
        if args.shard_dir is None or args.array_index is None:
            raise SystemExit("mode=array needs --shard-dir and --array-index")
        tag = tags[0]
        os.makedirs(args.shard_dir, exist_ok=True)
        n = args.array_index
        if n < N_WS:
            sh = _ws_shard(args.cache_dir, tag, PROBE_COHORT_7[n], taps)
            path = f"{args.shard_dir}/ws_{sh['name']}_{tag}.json"
        elif n < N_ARRAY:
            sh = _cs_shard(args.cache_dir, tag, CS_TEST_SUBJECTS[n - N_WS], taps)
            path = f"{args.shard_dir}/cs_{sh['test']}_{tag}.json"
        else:
            raise SystemExit(f"--array-index {n} out of range 0..{N_ARRAY - 1}")
        with open(path, "w") as f:
            json.dump(sh, f)
        print(f"[array {n}] {sh['kind']} -> {path}", flush=True)
        return

    if args.mode == "merge":
        if args.shard_dir is None or args.out is None:
            raise SystemExit("mode=merge needs --shard-dir and --out")
        results = _merge_shards(tags, args.shard_dir, taps)
    else:  # all
        if args.out is None:
            raise SystemExit("mode=all needs --out")
        results = _compute_all(args.cache_dir, tags, taps)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    _print_ladder(tags, results, taps)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
