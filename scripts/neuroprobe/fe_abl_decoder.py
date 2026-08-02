#!/usr/bin/env python
"""Decoder-parity arm: OUR enc0 features through THEIR classifier.

The frontend ablation (`fm:` arms) varies the features and holds the decoder fixed. This holds
the features fixed and swaps the decoder, because the enc0-vs-`Linear_*` gap on the board is
confounded by BOTH: our .6744/.6660/.5872 was produced by a dual ridge whose λ is selected on a
val half, theirs by `LogisticRegression()` at the sklearn default C=1.0 with nothing selected
(eval_population.py:257). Without this arm, "our frontend is better" and "their decoder is
untuned" are the same number.

WHAT IS HELD FIXED. Everything except the estimator: the same cache, the same `_feat` path, the
same splits, the same train/test rows, the same fit-on-train standardization, the same AUROC.

TWO ARMS, because "ridge vs logistic" and "tuned vs untuned" are not the same question:
  * `logreg`    — C left at the sklearn default 1.0, nothing selected. Upstream's entry verbatim,
                  and the val half is UNUSED (their protocol has nothing to select).
  * `logregcv`  — C selected on the SAME val half by the SAME rule the ridge uses for λ (argmax
                  AUROC, ties keep the smallest C), over the same logspace(-4, 4, 25) span.
Only the second isolates the estimator FAMILY. If `logregcv` recovers the ridge, the decoder
effect is regularization STRENGTH, not ridge-vs-logistic — a much narrower claim, and the one
that squares with the C-tuned logistic in `probe_v14_frozen_logistic.py` tying the ridge.

WHY C=1.0 IS NOT A NEUTRAL DEFAULT. sklearn minimizes ½‖w‖² + C·Σᵢ loss — a SUM over samples, so
the penalty relative to the MEAN loss is 1/(C·n) and shrinkage falls as the train set grows. At
C=1.0 that is 1/13592 for a CS anchor fit (n > d = 5568): effectively unregularized. In WS the
same C=1.0 sits at 1/6800 with d = 41412 >> n, where the problem is underdetermined and the
penalty still binds. So the default is weakest exactly where the fit must survive a domain shift.

THE CLASSIFIER IS COPIED, NOT APPROXIMATED: `LogisticRegression(random_state=seed,
max_iter=10000, tol=1e-3)` — upstream verbatim, C=1.0 default, lbfgs, L2.

WHAT THIS ARM DOES NOT ANSWER. It is one-sided. The symmetric arm (their features through our
ridge) needs their frontend re-encoded on our cache, which is a Stage-2 rebake. So a result here
bounds how much of the gap the decoder can explain on OUR side; it cannot attribute the
remainder to the frontend by itself.

Shards are written in the readout's own schema, so `v3_board_readout.py --merge` consumes them.

Usage (Delta CPU, one array cell per session):
  python scripts/neuroprobe/fe_abl_decoder.py --cache-dir <slim> --tag feabl \
      --mode ws --index $SLURM_ARRAY_TASK_ID --shard-dir <dir> --taps enc0_elec --workers 4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from scripts.neuroprobe.v3_board_readout import (
    BOARD_TASKS,
    CS_TEST_CELLS,
    CS_TRAIN_ANCHOR,
    LITE_SESSIONS,
    _elec_cols,
    _feat,
    _finite,
    _have,
    _is_elec,
    _load,
    _map_tasks,
    _parcel_cols,
    _sibling,
    _standardize,
    auroc,
)

SEED = 42          # upstream's default seed (eval_population.py argparse default)
# Same span and resolution as the ridge's λ grid (v3_board_readout.py LAM_MULTS), so "tuned" means
# the same amount of searching for both estimators. C is INVERSE regularization, hence no reversal
# is needed for the comparison to be fair — only the span matters.
C_GRID = tuple(np.logspace(-4, 4, 25))
# MEASURED 2026-08-01: on the CS anchor fit (n=13592 > d=5568) this grid PINS AT ITS FLOOR —
# 78/150 fits chose C=1e-4 and 114/150 chose C<=1e-3, i.e. the optimum is at least 10^4x more
# shrinkage than the sklearn default and lies BELOW the floor. A pinned floor makes the tuned
# arm a LOWER BOUND on what tuning is worth, which understates exactly the effect being measured,
# so the span is a CLI knob and CS must be run wide enough to leave the floor interior.
C_SPAN_WIDE = (-10.0, 4.0, 57)


def _NAME(c_grid) -> str:
    """Column name carries the arm, so a tuned and an untuned run can never merge into one cell."""
    return "logreg" if c_grid is None else "logregcv"


def _fold_mean(rs) -> dict:
    """Average WS folds. C is reported as the per-fold list — a single mean C would hide a split
    that chose 1e-4 on one fold and 1e4 on the other."""
    out = {"test": float(np.nanmean([r["test"] for r in rs]))}
    if "C" in rs[0]:
        out["C"] = [r["C"] for r in rs]
        out["C_pinned"] = bool(any(r["C_pinned"] for r in rs))
    return out


def _fit_score(z_tr, y_tr, evals, c) -> list:
    """Upstream's classifier at one C, on already-standardized features."""
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y_tr)) < 2:
        return [float("nan")] * len(evals)
    clf = LogisticRegression(C=c, random_state=SEED, max_iter=10000, tol=1e-3)
    clf.fit(z_tr, y_tr)
    return [auroc(clf.predict_proba(z)[:, 1], y) for z, y in evals]


def _score(z_tr, y_tr, z_te, y_te, z_va=None, y_va=None, c_grid=None) -> dict:
    """Standardize on train (== StandardScaler fit-on-train), then fit.

    Two arms, and the difference between them is the whole point:
      * c_grid=None  → C=1.0, the sklearn default. This is upstream's board entry verbatim.
      * c_grid given → C chosen on the VAL half by argmax AUROC, ties keeping the SMALLEST C —
        the ridge's own λ rule, on the ridge's own selection set. Only then is "ridge vs logistic"
        a contrast between ESTIMATOR FAMILIES rather than between tuned and untuned.
    """
    if c_grid is None:
        a, (b,) = _standardize(z_tr, [z_te])
        return {"test": _fit_score(a, y_tr, [(b, y_te)], 1.0)[0]}
    grid = tuple(c_grid)
    a, (v, b) = _standardize(z_tr, [z_va, z_te])
    best_c, best_va, best_te = grid[0], -np.inf, float("nan")
    for c in grid:                                 # strict > ⇒ ties keep the smallest C
        va, te = _fit_score(a, y_tr, [(v, y_va), (b, y_te)], c)
        if not np.isnan(va) and va > best_va:
            best_c, best_va, best_te = c, va, te
    return {"test": best_te, "C": float(best_c),
            "C_pinned": bool(best_c in (grid[0], grid[-1]))}


def _ws_cell(rec, task, taps, c_grid=None) -> dict:
    """Board KFold(2), train half → test half, averaged over folds.

    The val half is untouched by the C=1.0 arm (upstream selects nothing) and IS the selection
    set for the tuned arm — the same half the ridge selects λ on."""
    y = np.asarray(rec["labels"][task], dtype=np.float64)
    folds: dict = {}
    for _fold, sp in sorted(rec["ws_split"][task].items()):
        tr, va, te = _finite(y, sp["train"]), _finite(y, sp["val"]), _finite(y, sp["test"])
        if len(tr) < 2 or len(te) < 2:
            continue
        for enc in taps:
            if not _have(rec, enc):
                continue
            z_tr, z_te = _feat(rec, enc, tr), _feat(rec, enc, te)
            z_va = _feat(rec, enc, va) if c_grid is not None else None
            r = _score(z_tr, y[tr], z_te, y[te], z_va, y[va], c_grid)
            folds.setdefault(f"{enc}|{_NAME(c_grid)}", []).append(r)
    return {"cells": {k: _fold_mean(v) for k, v in folds.items() if v}}


def _cs_cell(anchor_rec, test_rec, task, taps, c_grid=None) -> dict:
    y_a = np.asarray(anchor_rec["labels"][task], dtype=np.float64)
    y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
    tr = _finite(y_a, np.arange(len(y_a)))
    te = _finite(y_t, test_rec["cs_split"][task]["test"])
    if len(tr) < 2 or len(te) < 2:
        return {"cells": {}}
    a_idx, t_idx, common = _parcel_cols(anchor_rec, test_rec)
    if common.size == 0:
        return {"cells": {}}
    va = _finite(y_t, test_rec["cs_split"][task]["val"])
    out = {f"{enc}|{_NAME(c_grid)}": _score(
        _feat(anchor_rec, enc, tr, a_idx), y_a[tr],
        _feat(test_rec, enc, te, t_idx), y_t[te],
        _feat(test_rec, enc, va, t_idx) if c_grid is not None else None, y_t[va], c_grid)
        for enc in taps}
    return {"cells": out, "n_parcels": int(common.size)}


def _csession_cell(train_rec, test_rec, task, taps, c_grid=None) -> dict:
    y_a = np.asarray(train_rec["labels"][task], dtype=np.float64)
    y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
    tr = _finite(y_a, np.arange(len(y_a)))
    te = _finite(y_t, test_rec["cs_split"][task]["test"])
    if len(tr) < 2 or len(te) < 2:
        return {"cells": {}}
    va = _finite(y_t, test_rec["cs_split"][task]["val"])
    p_a, p_t, p_common = _parcel_cols(train_rec, test_rec)
    e_a, e_t, n_elec = _elec_cols(train_rec, test_rec)
    out = {}
    for enc in taps:
        col_a, col_t = (e_a, e_t) if _is_elec(enc) else (p_a, p_t)
        if col_a is None:
            continue
        out[f"{enc}|{_NAME(c_grid)}"] = _score(
            _feat(train_rec, enc, tr, col_a), y_a[tr],
            _feat(test_rec, enc, te, col_t), y_t[te],
            _feat(test_rec, enc, va, col_t) if c_grid is not None else None, y_t[va], c_grid)
    return {"cells": out, "n_parcels": int(p_common.size) if p_a is not None else 0,
            "n_elec": n_elec}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", required=True, choices=("ws", "cs", "csession"))
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--taps", required=True, help="comma-separated; enc0/enc0_elec only")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--c-grid", action="store_true",
                   help="select C on the val half (arm `logregcv`) instead of the C=1.0 default")
    p.add_argument("--c-span", default=None, metavar="LO,HI,N",
                   help="logspace(LO,HI,N) for --c-grid; widen when the floor pins "
                        f"(measured-wide default {C_SPAN_WIDE})")
    a = p.parse_args()
    c_grid = None
    if a.c_grid:
        lo, hi, n = ([float(v) for v in a.c_span.split(",")] if a.c_span else C_SPAN_WIDE)
        c_grid = tuple(np.logspace(lo, hi, int(n)))

    import sklearn

    taps = tuple(t for t in a.taps.split(",") if t)
    print(f"[decoder-parity] sklearn {sklearn.__version__}", flush=True)
    cells = LITE_SESSIONS if a.mode == "ws" else (
        CS_TEST_CELLS if a.mode == "cs" else LITE_SESSIONS)
    cell = cells[a.index]
    print(f"[decoder-parity] mode={a.mode} cell=S{cell[0]}T{cell[1]} taps={taps} "
          f"arm={_NAME(c_grid)} clf=LogisticRegression("
          f"C={f'val-selected over {len(c_grid)} pts in [{c_grid[0]:.2g},{c_grid[-1]:.2g}]' if c_grid else '1.0 (sklearn default)'}, "
          f"max_iter=10000, tol=1e-3, seed={SEED})", flush=True)

    if a.mode == "ws":
        rec = _load(a.cache_dir, cell, a.tag)
        out = _map_tasks(lambda t, tp: _ws_cell(rec, t, tp, c_grid), taps, a.workers)
    elif a.mode == "cs":
        anchor, test = _load(a.cache_dir, CS_TRAIN_ANCHOR, a.tag), _load(a.cache_dir, cell, a.tag)
        out = _map_tasks(lambda t, tp: _cs_cell(anchor, test, t, tp, c_grid), taps, a.workers)
    else:
        train, test = _load(a.cache_dir, _sibling(cell), a.tag), _load(a.cache_dir, cell, a.tag)
        out = _map_tasks(lambda t, tp: _csession_cell(train, test, t, tp, c_grid), taps, a.workers)

    sh = {"kind": a.mode, "name": f"S{cell[0]}T{cell[1]}",
          "cells": {f"{a.tag}|{k}": v for k, v in out.items()}}
    os.makedirs(a.shard_dir, exist_ok=True)
    path = f"{a.shard_dir}/{a.mode}_{sh['name']}.json"
    with open(path, "w") as fh:
        json.dump(sh, fh)
    for t in BOARD_TASKS[:3]:
        print(f"[decoder-parity] {t}: "
              f"{ {k: round(v['test'], 4) for k, v in (out[t].get('cells') or {}).items()} }",
              flush=True)
    print(f"[decoder-parity] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
