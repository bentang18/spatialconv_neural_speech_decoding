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

λ — and ONLY λ (Ben 2026-07-17) — is SELECTED ON THE VAL HALF; every number is reported on the
TEST half. Tap (enc0/3/6/12, per-electrode vs parcel-mean) and norm (std/raw/std_target) are
REPORTED AXES: the output is the full grid, one complete protocol per entry scored over every
cell, so nothing is discarded and no cell's protocol is picked by a max across taps or norms.
λ is the exception because it is not a result but a per-fit regularization parameter that must
be set somehow — and the val half is upstream's own design for exactly that, not a divergence
of ours. ``generate_splits_cross_subject`` (train_test_splits.py:65)
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
# norm is a REPORTED axis, not a selected one (Ben 2026-07-17): both columns are computed over
# every cell and both are printed. Measured on our own r4 20k probe (results_v3_probe_r4_20k.json,
# 16 paired tap×task cells): std beats raw 16/16 WS (meanΔ +0.0277) but only 9/16 CS (meanΔ
# +0.0026, median +0.0007 — a coin flip), and the whole CS edge is carried by enc0 (the already-
# normalized input floor, +0.0281): every real encoder tap mildly PREFERS raw in CS (enc3 −0.0046,
# enc6 −0.0091 @ 25% std-wins, enc12 −0.0041). Mechanism for the regime split: WS fits μ/σ on the
# same session it scores, so the scaler is valid; CS fits μ/σ on the ANCHOR and applies it to a
# DIFFERENT subject, where a scale mismatch makes z-scoring actively harmful. That is a RESULT,
# and val-selecting over it would bury it — and would make the headline a per-cell mixture of two
# protocols. CS additionally reports "std_target" (per-domain/AdaBN; see _standardize_per_domain).
# (Upstream's pip package ships no baseline readout, so nothing external forces a choice.)
LAM_MULTS = tuple(np.logspace(-4.0, 4.0, 25))


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
    """Per-feature z-score on TRAIN stats only (never fit on val/test). σ=0 → 1.

    This is the canonical frozen-FM linear probe: the MAE/MoCo-v3/DINO lineage puts a
    BatchNorm WITHOUT affine before the linear head, and BN at eval uses statistics
    accumulated over probe TRAINING — i.e. train-set stats, which in CS means the ANCHOR's.
    """
    mu = z_tr.mean(axis=0)
    sd = z_tr.std(axis=0)
    sd[sd == 0] = 1.0
    return (z_tr - mu) / sd, [(z - mu) / sd for z in others]


def _standardize_per_domain(z_tr, z_va, z_te):
    """CS ablation: each SUBJECT z-scored in its OWN frame (AdaBN-style target adaptation).

    Anchor uses anchor stats (identical to _standardize's train side, so G/eigh/α are unchanged);
    the target subject uses the target's OWN stats — fit on its VAL half ONLY and applied to both
    val and test, so no test-half statistic is ever touched. Motivation: plain std maps the target
    into the ANCHOR's coordinate frame ((z_t − μ_a)/σ_a), which is wrong exactly when subjects'
    feature scales differ — i.e. in the CS regime the probe is measuring. Per-subject target
    normalization is standard in cross-subject EEG/BCI transfer (AdaBN; the Euclidean-Alignment
    family), though NOT part of the canonical vision linear probe.

    REPORTED, NEVER SELECTED (Ben 2026-07-17): it is a third CS norm column on the same footing
    as std/raw, scored over every cell. It is a DIFFERENT CLAIM from the other two — "transfer
    GIVEN target statistics" rather than "frozen features transfer" — which is precisely why it
    must not be fused into them by an argmax: that would make the number target-adapted on some
    cells and not others.

    Rules check: SUBMIT.md constrains only (1) splits from train_test_splits.py and (2) no
    pretraining on eval data — nothing about normalization. Fitting an UNSUPERVISED scaler on the
    val half uses strictly less information than the λ/tap selection upstream already sanctions
    on the same half.
    """
    mu_a, sd_a = z_tr.mean(axis=0), z_tr.std(axis=0)
    sd_a[sd_a == 0] = 1.0
    mu_t, sd_t = z_va.mean(axis=0), z_va.std(axis=0)   # target stats: VAL half only
    sd_t[sd_t == 0] = 1.0
    return (z_tr - mu_a) / sd_a, [(z_va - mu_t) / sd_t, (z_te - mu_t) / sd_t]


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


def _select_lam(d) -> dict:
    """One (tap,norm)'s λ-grid {"val": {m: auroc}, "test": {m: auroc}} → pick argmax VAL.

    λ is the ONLY val-selected axis (Ben 2026-07-17). Tap and norm are REPORTED, not selected:
    every (tap, norm) keeps its own complete number over every cell, so the depth ladder
    (enc0/3/6/12), the per-electrode-vs-parcel-mean contrast, and the std/raw/std_target
    normalization contrast all survive intact instead of collapsing into one argmax. λ is the
    exception because it is not a result — it is a per-fit regularization parameter that has to
    be set somehow, and upstream's own val half exists for exactly that.

    All-NaN val (single-class val half) → NaN, reported as such rather than silently defaulting
    to a λ, so a degenerate cell cannot masquerade as a scored one.

    ``lam_pinned`` flags a selected λ sitting on a grid boundary: the optimum is then outside
    the grid and the reported AUROC is a truncation artifact, not a fit. It is plausible and
    wrong — exactly the failure that cannot be caught by eye — so it is carried to the report.
    """
    best = None
    for m, va in d["val"].items():
        if np.isnan(va):
            continue
        if best is None or va > best["val"]:
            best = {"val": va, "test": d["test"][m], "lam_mult": float(m),
                    "lam_pinned": bool(m in (LAM_MULTS[0], LAM_MULTS[-1]))}
    if best is None:
        return {"val": float("nan"), "test": float("nan"), "lam_mult": float("nan"),
                "lam_pinned": False}
    return best


def _cell_key(tap, norm) -> str:
    return f"{tap}|{norm}"


def _grid_cells(grid) -> dict:
    """{(tap,norm): λ-grid} → {"tap|norm": λ-selected result}. Nothing is dropped or fused."""
    return {_cell_key(t, nm): _select_lam(d) for (t, nm), d in grid.items()}


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


def _ws_cell(rec, task, taps) -> dict:
    """Within-session: board KFold(2). Per fold fit train, λ-select on the val half, report the
    test half; average the two folds' test AUROCs, per (tap, norm)."""
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
            folds.append(_grid_cells(grid))
    if not folds:
        return {"cells": {}}
    keys = sorted({k for f in folds for k in f})
    out = {}
    for k in keys:
        vals = [f[k]["test"] for f in folds if k in f]
        out[k] = {"test": float(np.nanmean(vals)) if vals else float("nan"),
                  "lam_pinned": bool(any(f[k]["lam_pinned"] for f in folds if k in f)),
                  "lam_mult": [f[k]["lam_mult"] for f in folds if k in f]}
    return {"cells": out}


def _cs_cell(anchor_rec, test_rec, task, taps) -> dict:
    """Cross-subject: fit the anchor's finite rows, λ-select on the test cell's val half, report
    its test half. Features are the anchor∩test parcel intersection (atlas-id aligned)."""
    y_a = np.asarray(anchor_rec["labels"][task], dtype=np.float64)
    y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
    tr = _finite(y_a, np.arange(len(y_a)))
    va = _finite(y_t, test_rec["cs_split"][task]["val"])
    te = _finite(y_t, test_rec["cs_split"][task]["test"])
    if len(tr) < 2 or len(te) < 2:
        return {"cells": {}}
    a_idx, t_idx, common = _parcel_cols(anchor_rec, test_rec)
    if common.size == 0:
        return {"cells": {}}
    grid: dict = {}
    for enc in taps:
        if enc not in anchor_rec["feats"] or enc not in test_rec["feats"]:
            continue
        z_tr = _feat(anchor_rec, enc, tr, a_idx)
        z_va, z_te = _feat(test_rec, enc, va, t_idx), _feat(test_rec, enc, te, t_idx)
        for norm in NORMS:
            a, (b, c) = ((z_tr, [z_va, z_te]) if norm == "raw"
                         else _standardize(z_tr, [z_va, z_te]))
            grid[(enc, norm)] = _lam_grid(a, y_a[tr], {"val": (b, y_t[va]), "test": (c, y_t[te])})
        # CS-only per-domain (AdaBN-style) normalization — a third REPORTED norm column, on the
        # same footing as std/raw. It is not a headline and not selected against them; it is the
        # protocol "target subject z-scored in its own frame", reported over every cell.
        a, (b, c) = _standardize_per_domain(z_tr, z_va, z_te)
        grid[(enc, "std_target")] = _lam_grid(
            a, y_a[tr], {"val": (b, y_t[va]), "test": (c, y_t[te])})
    if not grid:
        return {"cells": {}}
    return {"cells": _grid_cells(grid), "n_parcels": int(common.size)}


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
    return {f"{tag}|{t}": {"ws": {}, "cs": {}, "pinned": {}, "n_parcels": {}}
            for tag in tags for t in BOARD_TASKS}


def _absorb(res, sh) -> None:
    """Fold one shard in. res[task][kind]["tap|norm"][cell_name] = test AUROC — the full grid,
    every entry populated over every cell. No axis is collapsed at merge time either."""
    kind = sh["kind"]
    for k, val in sh["cells"].items():
        for gk, s in (val.get("cells") or {}).items():
            res[k][kind].setdefault(gk, {})[sh["name"]] = s["test"]
            if s.get("lam_pinned"):
                res[k]["pinned"].setdefault(f"{kind}:{gk}", []).append(sh["name"])
        if val.get("n_parcels") is not None:
            res[k]["n_parcels"][sh["name"]] = val["n_parcels"]


def _merge(tags, shard_dir) -> dict:
    res = _blank(tags)
    for kind in ("ws", "cs"):
        for path in sorted(glob.glob(f"{shard_dir}/{kind}_*.json")):
            with open(path) as f:
                _absorb(res, json.load(f))
    return _finalize(res)


def _finalize(res: dict) -> dict:
    """Cohort-mean each grid entry over its cells (12 WS sessions / 10 CS cells)."""
    for c in res.values():
        for kind in ("ws", "cs"):
            c[f"{kind}_mean"] = {gk: float(np.nanmean(list(d.values())))
                                 for gk, d in c[kind].items() if d}
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


def _macro(res, tag, kind, gk) -> float:
    """Macro over the 15 board tasks of one grid entry's cohort mean."""
    v = [res[f"{tag}|{t}"].get(f"{kind}_mean", {}).get(gk, np.nan) for t in BOARD_TASKS]
    return float(np.nanmean(v)) if not all(np.isnan(x) for x in v) else float("nan")


def _report(tags, res) -> None:
    """Print the GRID, not a headline (Ben 2026-07-17).

    Every (tap, norm) is a complete protocol scored over every cell: 12 WS sessions / 10 CS
    cells × 15 tasks, λ chosen on each cell's val half. Nothing is selected across taps or
    norms and nothing is discarded, so the depth ladder (enc0/3/6/12), the per-electrode vs
    parcel-mean feature-unit diff, and the std/raw/std_target normalization contrast are all
    readable side by side, and each column is one protocol describable in a single sentence.
    Which column is THE board number is a claim, and claims are Ben's to make — this file's
    job is to put every number on the table honestly.
    """
    for tag in tags:
        for kind, label in (("cs", "CS (anchor S2T4 → 10 cells)"), ("ws", "WS (12 sessions)")):
            gks = sorted({g for t in BOARD_TASKS for g in res[f"{tag}|{t}"].get(kind, {})},
                         key=lambda g: (g.split("|")[1], g.split("|")[0]))
            if not gks:
                continue
            print(f"\n=== {label} test-half AUROC — tag={tag} ===", flush=True)
            print(f"  {'task':18s}" + "".join(f"{g:>18s}" for g in gks), flush=True)
            for t in BOARD_TASKS:
                m = res[f"{tag}|{t}"].get(f"{kind}_mean", {})
                print(f"  {t:18s}" + "".join(f"{m.get(g, float('nan')):18.4f}" for g in gks),
                      flush=True)
            print(f"  {'MACRO(15)':18s}"
                  + "".join(f"{_macro(res, tag, kind, g):18.4f}" for g in gks), flush=True)

        # Contrasts the grid above already contains, stated as differences so the ordering is
        # not left to the reader's eye. These are READS of the table, never separate fits.
        for kind, taps in (("cs", ENCODERS), ("ws", WS_TAPS)):
            norms = sorted({g.split("|")[1] for t in BOARD_TASKS
                            for g in res[f"{tag}|{t}"].get(kind, {})})
            if not norms:
                continue
            print(f"\n=== {kind.upper()} contrasts (macro over 15 tasks), tag={tag} ===",
                  flush=True)
            for nm in norms:
                if nm == "std":
                    continue
                for tp in taps:
                    x, y = (_macro(res, tag, kind, f"{tp}|{nm}"),
                            _macro(res, tag, kind, f"{tp}|std"))
                    if np.isnan(x) or np.isnan(y):
                        continue
                    print(f"  {tp:12s} {nm:11s} {x:.4f}  vs std {y:.4f}  Δ {x - y:+.4f}",
                          flush=True)
            if kind == "ws":
                for nm in norms:
                    e, q = (_macro(res, tag, kind, f"enc12_elec|{nm}"),
                            _macro(res, tag, kind, f"enc12|{nm}"))
                    if not (np.isnan(e) or np.isnan(q)):
                        print(f"  [diff] enc12 per-electrode − parcel-mean ({nm}) = {e - q:+.4f}"
                              f"  ({e:.4f} vs {q:.4f})", flush=True)

    # λ pin check: a boundary argmax means the optimum lies OUTSIDE the grid, so the AUROC is a
    # truncation artifact indistinguishable by eye from a fit. Name it, count it, print it.
    pinned = {f"{k}  {gk}": cells for k, c in res.items()
              for gk, cells in c.get("pinned", {}).items() if cells}
    n_fits = sum(len(d) for c in res.values() for kind in ("ws", "cs")
                 for d in c.get(kind, {}).values())
    n_pin = sum(len(v) for v in pinned.values())
    if pinned:
        print(f"\n[check] λ grid: VIOLATED — {n_pin}/{n_fits} fits selected a λ on a grid "
              f"boundary ({LAM_MULTS[0]:.1e} or {LAM_MULTS[-1]:.1e}); their optimum is OUTSIDE "
              f"the grid and those AUROCs are truncation artifacts. Widen LAM_MULTS and re-run."
              f" First 10: {list(pinned)[:10]}", flush=True)
    else:
        print(f"\n[check] λ grid: OK — 0/{n_fits} fits pinned to a boundary; every selected λ "
              f"is interior to [{LAM_MULTS[0]:.1e}, {LAM_MULTS[-1]:.1e}].", flush=True)
    print("[check] selection: every number above is a TEST-half AUROC. λ is the ONLY axis "
          "chosen on the val half (upstream train_test_splits.py:65); tap and norm are "
          "reported in full, never selected.", flush=True)


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
