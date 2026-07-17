"""v3 board-comparable readout — the Neuroprobe-Lite leaderboard number (WS + CS).

Sibling of ``v3_probe_readout_r4.py``: SAME ridge/parcel-pool mechanism on the SAME pooled
caches, so the erosion diagnostic and the board claim are one probe differing ONLY in the eval
universe. Consumes ``v3_probe_encode_r4.py --sessions board --tasks board15 --electrode-set
lite`` (per session a ``(n_win, |P|, F)`` fp16 parcel-mean feature per tap, + 15 label vectors,
+ the board splits).

Four deliberate changes from the diagnostic, each forced by board-comparability:

  1. CS ANCHOR = subject-2 TRIAL-4 — upstream's own ``DS_DM_TRAIN_SUBJECT_ID/TRIAL_ID``
     (config.py:36), verified. The diagnostic used (2,1). (2,4) is NOT in our pretrain, so
     using upstream's exact anchor costs no leakage — it removes a divergence.
  2. CS TEST = the 10 Lite CrossSubject cells = every Lite (s,t) with s != 2. Subjects 7 & 10
     are in NO pretrain session of ours ⇒ a TRUE held-out-subject transfer.
  3. TASKS = the 15 leaderboard tasks (verified == upstream NEUROPROBE_TASKS).
  4. ELECTRODES = the Lite montage, applied at ENCODE (see the encode's --electrode-set lite).

λ / tap / norm are SELECTED ON THE VAL HALF and reported on the TEST half — upstream's own
design, not a divergence of ours. ``generate_splits_cross_subject`` (train_test_splits.py:65)
and ``generate_splits_within_session`` BOTH carve the eval set in half:
    test_size = len(test_dataset); val_size = test_size // 2
    val = range(0, val_size)   test = range(val_size, test_size)
The val half comes out of the TEST session, NOT the anchor — so selecting on it costs the fit
set ZERO rows. (An earlier draft of this file chose GCV-on-the-anchor to avoid "wasting"
training samples; that rationale was a misreading — there is nothing to waste.) The reported
test half is never fit on and never selected on.

``build_session_targets`` already emits exactly these halves (``ws_split[task][fold]`` →
train/val/test, ``cs_split[task]`` → val/test), so the splits are upstream-faithful by
construction and this file just has to USE the val key.

Ridge/metric primitives INLINED — zero speech_decoding dep, runs on the stock NCSA Delta
pytorch module. fp32 GEMM into a fp64 n×n eigendecomposition (a fp64 z_anchor at |P|·F is
multi-GB and sends the array NUMA-bound). ONE eigh per (task, cell, tap, norm) serves the whole
λ grid: the ridge smoother's eigenvalues are w/(w+λ), so every λ reuses the same (w, V, c) and
costs one O(n²) mat-vec instead of a fresh O(n³) solve.

Usage (CPU, NCSA Delta; --mem>=64G):
  python scripts/neuroprobe/v3_board_readout.py \
      --cache-dir /projects/bhqk/htang13/v3_board_cache --tags board_r4_20k \
      --mode all --out results_v3_board.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

# The 15 Neuroprobe leaderboard tasks — verified == upstream neuroprobe.config.NEUROPROBE_TASKS.
BOARD_TASKS = (
    "onset", "speech", "volume", "delta_volume", "pitch", "word_index",
    "word_gap", "gpt2_surprisal", "word_head_pos", "word_part_speech",
    "word_length", "global_flow", "local_flow", "frame_brightness", "face_num",
)
# Upstream DS_DM_TRAIN_SUBJECT_ID, DS_DM_TRAIN_TRIAL_ID (config.py:36) — verified.
CS_TRAIN_ANCHOR = (2, 4)
# Every Lite (s,t) with s != 2 — upstream generate_splits_cross_subject asserts test != anchor.
CS_TEST_CELLS = ((1, 1), (1, 2), (3, 0), (3, 1), (4, 0), (4, 1),
                 (7, 0), (7, 1), (10, 0), (10, 1))
# The 12 Lite sessions — verified == upstream NEUROPROBE_LITE_SUBJECT_TRIALS.
LITE_SESSIONS = ((1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
                 (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1))
ENCODERS = ("enc0", "enc3", "enc6", "enc12")          # parcel-mean (electrodes pooled at encode)
# Per-electrode taps (Ben 2026-07-16): WS keeps ALL electrodes by DEFAULT; the parcel-mean is
# the opt-in comparison, reported beside it so the diff is visible. CS cannot use these —
# electrode identity is not shared across subjects, which is what the parcel bridge exists for.
ELEC_TAPS = ("enc12_elec",)
WS_TAPS = ELEC_TAPS + ENCODERS
CS_TAPS = ENCODERS
NORMS = ("std", "raw")
# λ grid as multipliers of trace(G)/n (the diagnostic's pinned lam_mult=1.0 sits at the centre).
LAM_MULTS = tuple(np.logspace(-3.0, 3.0, 13))


def auroc(scores, labels) -> float:
    """Verbatim from online_probe.auroc. NaN if the eval half is single-class."""
    from sklearn.metrics import roc_auc_score

    y = (np.asarray(labels) > 0).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores)))


def _finite(y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    r = np.asarray(rows, dtype=np.int64)
    return r[np.isfinite(y[r])]


def _standardize(z_tr, others):
    """Per-feature z-score on TRAIN stats only (never fit on val/test). σ=0 → 1."""
    mu = z_tr.mean(axis=0)
    sd = z_tr.std(axis=0)
    sd[sd == 0] = 1.0
    return (z_tr - mu) / sd, [(z - mu) / sd for z in others]


def _feat(rec, enc, rows, col_idx=None) -> np.ndarray:
    """(n,|P|,F) fp16 cache → rows (and optionally parcel columns) → flat fp32 (r, ·)."""
    x = rec["feats"][enc]["raw"][np.asarray(rows, dtype=np.int64)]
    if col_idx is not None:
        x = x[:, np.asarray(col_idx, dtype=np.int64)]
    x = x.to(torch.float32).numpy()
    return x.reshape(x.shape[0], -1)


def _lam_grid(z_tr, y_tr, evals):
    """Fit ridge on (z_tr, y_tr); score every λ in LAM_MULTS on each eval set.

    Returns {eval_name: {lam_mult: auroc}}. One fp64 eigendecomposition of G=Z_trZ_trᵀ serves
    the whole grid — the ridge solution is α = V diag(1/(w+λ)) Vᵀ y, so sweeping λ reuses
    (w, V, c=Vᵀy) and costs one mat-vec each. GEMMs are fp32 (memory), G/solve are fp64.
    λ NEVER enters through an eval set: only through w, which is anchor-side only.
    """
    if len(y_tr) < 2:
        return {name: {m: float("nan") for m in LAM_MULTS} for name in evals}
    g = np.asarray(z_tr @ z_tr.T, dtype=np.float64)             # fp32 GEMM → fp64 Gram
    n = g.shape[0]
    w, V = np.linalg.eigh(g)                                    # G symmetric PSD ⇒ w >= 0
    c = V.T @ np.asarray(y_tr, dtype=np.float64)
    base = float(np.sum(w) / max(n, 1))                         # trace(G)/n — the λ scale
    kern = {name: np.asarray(z @ z_tr.T, dtype=np.float64) for name, (z, _) in evals.items()}
    out: dict = {name: {} for name in evals}
    for m in LAM_MULTS:
        alpha = V @ (c / (w + m * base))
        for name, (_, y) in evals.items():
            out[name][m] = (auroc(kern[name] @ alpha, y) if len(y) >= 2 else float("nan"))
    return out


def _select(grid) -> dict:
    """grid = {(enc,norm): {"val": {m: auroc}, "test": {m: auroc}}} → pick argmax VAL.

    Selection is over (tap, norm, λ) jointly, per cell, on that cell's own val half. Ties and
    all-NaN val (single-class val half) → NaN, reported as such rather than silently defaulting
    to a λ, so a degenerate cell cannot masquerade as a scored one.
    """
    best = None
    for (enc, norm), d in grid.items():
        for m, va in d["val"].items():
            if np.isnan(va):
                continue
            if best is None or va > best["val"]:
                best = {"val": va, "test": d["test"][m], "enc": enc, "norm": norm,
                        "lam_mult": float(m)}
    if best is None:
        return {"val": float("nan"), "test": float("nan"), "enc": None, "norm": None,
                "lam_mult": float("nan")}
    return best


def _parcel_cols(anchor_rec, test_rec):
    """Anchor∩test parcel columns, aligned BY ATLAS ID (not by position)."""
    a_p = np.asarray(anchor_rec["present_parcels"], dtype=np.int64)
    t_p = np.asarray(test_rec["present_parcels"], dtype=np.int64)
    common = np.intersect1d(a_p, t_p)
    if common.size == 0:
        return None, None, common
    a_idx = [int(np.where(a_p == c)[0][0]) for c in common]
    t_idx = [int(np.where(t_p == c)[0][0]) for c in common]
    return a_idx, t_idx, common


def _per_tap_and_joint(grid, taps) -> dict:
    """Split one (tap,norm)→λ-grid into a per-tap report AND the joint selection.

    per_tap selects (norm, λ) WITHIN a tap, so the depth ladder and the per-electrode-vs-
    parcel-mean contrast (Ben 2026-07-16) stay readable — selecting the tap jointly would fuse
    the two feature units into one number and hide exactly that diff. joint additionally
    selects the tap on val, which is the board-headline rule. Same grid, so per_tap is free.
    """
    return {
        "per_tap": {t: _select({k: v for k, v in grid.items() if k[0] == t})
                    for t in taps if any(k[0] == t for k in grid)},
        "joint": _select(grid),
    }


def _ws_cell(rec, task, taps) -> dict:
    """Within-session: board KFold(2). Per fold fit train, select on the val half, report the
    test half; average the two folds' test AUROCs."""
    y = np.asarray(rec["labels"][task], dtype=np.float64)
    folds = []
    for _fold, sp in sorted(rec["ws_split"][task].items()):
        tr, va, te = (_finite(y, sp["train"]), _finite(y, sp["val"]), _finite(y, sp["test"]))
        if len(tr) < 2 or len(te) < 2:
            continue
        grid = {}
        for enc in taps:
            if enc not in rec["feats"]:
                continue
            z_tr = _feat(rec, enc, tr)
            z_va, z_te = _feat(rec, enc, va), _feat(rec, enc, te)
            for norm in NORMS:
                a, (b, c) = ((z_tr, [z_va, z_te]) if norm == "raw"
                             else _standardize(z_tr, [z_va, z_te]))
                grid[(enc, norm)] = _lam_grid(a, y[tr], {"val": (b, y[va]), "test": (c, y[te])})
        if grid:
            folds.append(_per_tap_and_joint(grid, taps))
    if not folds:
        return {"test": float("nan"), "per_tap": {}}
    def _avg(pick):
        vals = [pick(f) for f in folds]
        vals = [v for v in vals if v is not None]
        return float(np.nanmean(vals)) if vals else float("nan")
    return {
        "test": _avg(lambda f: f["joint"]["test"]),
        "per_tap": {t: {"test": _avg(lambda f, t=t: f["per_tap"].get(t, {}).get("test"))}
                    for t in taps if any(t in f["per_tap"] for f in folds)},
        "sel": folds[0]["joint"],
    }


def _cs_cell(anchor_rec, test_rec, task, taps) -> dict:
    """Cross-subject: fit the anchor's finite rows, select on the test cell's val half, report
    its test half. Features are the anchor∩test parcel intersection (atlas-id aligned)."""
    y_a = np.asarray(anchor_rec["labels"][task], dtype=np.float64)
    y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
    tr = _finite(y_a, np.arange(len(y_a)))
    va = _finite(y_t, test_rec["cs_split"][task]["val"])
    te = _finite(y_t, test_rec["cs_split"][task]["test"])
    if len(tr) < 2 or len(te) < 2:
        return {"test": float("nan")}
    a_idx, t_idx, common = _parcel_cols(anchor_rec, test_rec)
    if common.size == 0:
        return {"test": float("nan")}
    grid = {}
    for enc in taps:
        if enc not in anchor_rec["feats"] or enc not in test_rec["feats"]:
            continue
        z_tr = _feat(anchor_rec, enc, tr, a_idx)
        z_va, z_te = _feat(test_rec, enc, va, t_idx), _feat(test_rec, enc, te, t_idx)
        for norm in NORMS:
            a, (b, c) = ((z_tr, [z_va, z_te]) if norm == "raw"
                         else _standardize(z_tr, [z_va, z_te]))
            grid[(enc, norm)] = _lam_grid(a, y_a[tr], {"val": (b, y_t[va]), "test": (c, y_t[te])})
    if not grid:
        return {"test": float("nan")}
    both = _per_tap_and_joint(grid, taps)
    out = dict(both["joint"])
    out["per_tap"] = {t: {"test": s["test"]} for t, s in both["per_tap"].items()}
    out["sel"] = both["joint"]
    out["n_parcels"] = int(common.size)
    return out


def _load(cache_dir, session, tag):
    s, t = session
    return torch.load(f"{cache_dir}/enc_s{s}_t{t}_{tag}.pt", map_location="cpu",
                      weights_only=False)


# ── sharded units (one SLURM array task each; all cells are independent) ────────────
def _ws_shard(cache_dir, tag, session, taps=WS_TAPS) -> dict:
    rec = _load(cache_dir, session, tag)
    cells = {f"{tag}|{task}": _ws_cell(rec, task, taps) for task in BOARD_TASKS}
    return {"kind": "ws", "name": f"S{session[0]}T{session[1]}", "cells": cells}


def _cs_shard(cache_dir, tag, cell, taps=CS_TAPS) -> dict:
    taps = tuple(t for t in taps if t not in ELEC_TAPS)   # CS is parcel-bridged by necessity
    anchor_rec = _load(cache_dir, CS_TRAIN_ANCHOR, tag)
    test_rec = _load(cache_dir, cell, tag)
    cells = {f"{tag}|{task}": _cs_cell(anchor_rec, test_rec, task, taps) for task in BOARD_TASKS}
    return {"kind": "cs", "name": f"S{cell[0]}T{cell[1]}", "cells": cells}


def _blank(tags) -> dict:
    return {f"{tag}|{t}": {"ws_per_session": {}, "cs_per_cell": {}, "sel": {},
                           "ws_tap": {}, "cs_tap": {}}
            for tag in tags for t in BOARD_TASKS}


def _absorb(res, sh) -> None:
    kind = sh["kind"]
    key = "ws_per_session" if kind == "ws" else "cs_per_cell"
    tapkey = "ws_tap" if kind == "ws" else "cs_tap"
    for k, val in sh["cells"].items():
        res[k][key][sh["name"]] = val["test"]
        res[k]["sel"][f"{kind}:{sh['name']}"] = {
            x: (val.get("sel") or {}).get(x) for x in ("enc", "norm", "lam_mult")
        } | {"n_parcels": val.get("n_parcels")}
        for tap, s in (val.get("per_tap") or {}).items():
            res[k][tapkey].setdefault(tap, {})[sh["name"]] = s["test"]


def _merge(tags, shard_dir) -> dict:
    res = _blank(tags)
    for kind in ("ws", "cs"):
        for path in sorted(glob.glob(f"{shard_dir}/{kind}_*.json")):
            with open(path) as f:
                _absorb(res, json.load(f))
    return _finalize(res)


def _finalize(res: dict) -> dict:
    for c in res.values():
        ws, cs = list(c["ws_per_session"].values()), list(c["cs_per_cell"].values())
        c["ws_cohort"] = float(np.nanmean(ws)) if ws else float("nan")
        c["cs_mean"] = float(np.nanmean(cs)) if cs else float("nan")
        for src, dst in (("ws_tap", "ws_tap_mean"), ("cs_tap", "cs_tap_mean")):
            c[dst] = {tap: float(np.nanmean(list(d.values()))) for tap, d in c[src].items() if d}
    return res


def _compute_all(cache_dir, tags, ws_taps=WS_TAPS, cs_taps=CS_TAPS) -> dict:
    res = _blank(tags)
    for tag in tags:
        for session in LITE_SESSIONS:
            sh = _ws_shard(cache_dir, tag, session, ws_taps)
            _absorb(res, sh)
            print(f"[{tag}] WS done {sh['name']}", flush=True)
        for cell in CS_TEST_CELLS:
            sh = _cs_shard(cache_dir, tag, cell, cs_taps)
            _absorb(res, sh)
            print(f"[{tag}] CS done {sh['name']}", flush=True)
    return _finalize(res)


def _report(tags, res) -> None:
    for tag in tags:
        print(f"\n=== board Neuroprobe-Lite (tag={tag}) — test-half AUROC ===", flush=True)
        print(f"  {'task':18s} {'CS (10 cells)':>14s} {'WS (12 sess)':>14s}", flush=True)
        cs_all, ws_all = [], []
        for t in BOARD_TASKS:
            c = res[f"{tag}|{t}"]
            cs_all.append(c["cs_mean"])
            ws_all.append(c["ws_cohort"])
            print(f"  {t:18s} {c['cs_mean']:14.4f} {c['ws_cohort']:14.4f}", flush=True)
        print(f"  {'MACRO(15)':18s} {float(np.nanmean(cs_all)):14.4f} "
              f"{float(np.nanmean(ws_all)):14.4f}", flush=True)

        # Per-tap ladder: the depth ordering AND (in WS) the per-electrode vs parcel-mean
        # feature-unit diff (Ben 2026-07-16) — the reason per_tap is reported at all.
        for direction, key in (("CS", "cs_tap_mean"), ("WS", "ws_tap_mean")):
            taps = sorted({tp for t in BOARD_TASKS for tp in res[f"{tag}|{t}"][key]})
            if not taps:
                continue
            print(f"\n=== {direction} per tap (macro over 15 tasks), tag={tag} ===", flush=True)
            for tp in taps:
                vals = [res[f"{tag}|{t}"][key].get(tp, float("nan")) for t in BOARD_TASKS]
                print(f"  {tp:12s} {float(np.nanmean(vals)):.4f}", flush=True)
            if direction == "WS" and {"enc12_elec", "enc12"} <= set(taps):
                e = float(np.nanmean([res[f"{tag}|{t}"][key].get("enc12_elec", np.nan)
                                      for t in BOARD_TASKS]))
                p = float(np.nanmean([res[f"{tag}|{t}"][key].get("enc12", np.nan)
                                      for t in BOARD_TASKS]))
                print(f"  [diff] enc12 per-electrode − parcel-mean = {e - p:+.4f} "
                      f"({e:.4f} vs {p:.4f})", flush=True)
    # The board's headline is the CS macro over the 15 tasks; the leaderboard reference point
    # is CS #1 CNN 0.578 > PopT 0.575 (reference-neuroprobe-cs-leaderboard-2026-05).
    print("\n[check] selection: every reported number is a TEST-half AUROC; λ/tap/norm were "
          "chosen on the VAL half only (upstream train_test_splits.py:65).", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--tags", default="board_r4_20k")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("all", "ws", "cs", "merge"), default="all")
    p.add_argument("--index", type=int, help="shard index (mode=ws|cs)")
    p.add_argument("--shard-dir")
    p.add_argument("--taps", default="",
                   help=f"comma-separated subset of {WS_TAPS} (default: all; CS drops "
                        f"{ELEC_TAPS} automatically)")
    args = p.parse_args()

    tags = tuple(t.strip() for t in args.tags.split(","))
    taps = tuple(t.strip() for t in args.taps.split(",") if t.strip()) or WS_TAPS
    bad = [t for t in taps if t not in WS_TAPS]
    if bad:
        raise SystemExit(f"unknown taps {bad}; choose from {WS_TAPS}")

    if args.mode in ("ws", "cs"):
        cells = LITE_SESSIONS if args.mode == "ws" else CS_TEST_CELLS
        cell = cells[args.index]
        fn = _ws_shard if args.mode == "ws" else _cs_shard
        sh = fn(args.cache_dir, tags[0], cell, taps)
        os.makedirs(args.shard_dir, exist_ok=True)
        out = f"{args.shard_dir}/{args.mode}_{sh['name']}.json"
        with open(out, "w") as f:
            json.dump(sh, f, indent=2)
        print(f"wrote {out}", flush=True)
        return

    res = _merge(tags, args.shard_dir) if args.mode == "merge" else _compute_all(
        args.cache_dir, tags, taps, tuple(t for t in taps if t not in ELEC_TAPS))
    _report(tags, res)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}\nMERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
