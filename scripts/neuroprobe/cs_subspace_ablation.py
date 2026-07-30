#!/usr/bin/env python
"""CAUSAL test: do subject-identity directions CARRY the task content, or merely sit beside it?

WHY THIS AND NOT MORE GEOMETRY. Every earlier identity analysis was DESCRIPTIVE and none changed a
decision (`feedback-run-the-causal-intervention-descriptive-geometry-is-not-informative-2026-07-29`).
Two interventions that DO exist both came back null, and both are weak by construction:
  * LEACE erasure of the session concept -- RANK-1, and its eraser IS the session mean shift, so a
    ~0 AUROC cost is a THEOREM (AUROC is invariant to a constant score offset).
  * `std_target` (AdaBN, per-domain mean AND scale removed) -- +.0027/+.0013/+.0021 at enc0/3/12,
    but 5/10 paired cells at every tap = CHANCE. Null.
Neither removes the identity SUBSPACE, and identity is still AUROC 1.000 at enc12 after both. This
script removes DIRECTIONS and re-measures the board CS number.

WHY TOP-PC IS THE IDENTITY INTERVENTION. `leace_toppc` established that at enc12 the top PC IS the
session offset, and `dir_between_frac` rises .0747 -> .9951 with depth. So at enc12, projecting out
the top PCs IS projecting out identity -- with the advantage that it is rank-sweepable, needs no
session labels, and is fit on TRAIN ONLY (the anchor), so it cannot leak the test subject.

PRE-REGISTERED, written before the run:
  null            arm `none` reproduces the board CS macro EXACTLY (asserted, not eyeballed).
                  A projection arm at delta 0 => those directions are not load-bearing.
  achievable      CS macro ~.59-.60 vs .5 chance, so there is ~.10 of range to lose. Not saturated.
  decision map    out_top HURTS monotonically in k, while out_rand ~ 0
                      => identity and content SHARE the high-variance directions
                      => invariance is the WRONG target; do not sell identity removal as a method
                  out_top ~ 0 too
                      => content lives in LOW-variance directions; the shared-directions reading
                         from task_identity_overlap is WRONG and must be retracted
                  out_top HELPS
                      => identity is an active nuisance; removal IS a method contribution
                  keep_top recovers most of baseline => the content really is in those directions
                      (the positive form of the same claim; guards against out_top hurting merely
                       because ANY rank-k removal hurts, which out_rand also controls)
  admissibility   out_rand is the isotropic rank-matched control. If out_rand hurts as much as
                  out_top, rank alone explains it and NOTHING here is about identity or variance.

Reuses v3_board_readout's own _feat/_standardize_inplace/_lam_grid/_select_lam, so the ridge, the
lambda grid and the val-half selection are bit-identical to the board. The projection is applied
AFTER the train-fitted std, because that is the space the ridge sees and the space
task_identity_overlap measured in.

ARM: default board_r6_40k -- the arm every other identity/LEACE analysis used. 45k is the board
HEADLINE arm; mixing them is the #1 recorded defect, so the tag is printed with every number.

Usage (Delta CPU, one shard per cell):
  python scripts/neuroprobe/cs_subspace_ablation.py --cache-dir /projects/bhqk/htang13/v3_board_cache \
      --tag board_r6_40k --cell 1,1 --out cs_sub_s1t1.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from scripts.neuroprobe.v3_board_readout import (
    BOARD_TASKS,
    CS_TEST_CELLS,
    CS_TRAIN_ANCHOR,
    _feat,
    _finite,
    _lam_grid,
    _load,
    _parcel_cols,
    _select_lam,
    _standardize_inplace,
)

SEED = 33
RANKS = (1, 4, 16, 32)
# enc0 and enc12 BRACKET the identity question (dir_between_frac .0747 -> .9951: the top PC IS the
# session offset at enc12 and is NOT at enc0), but two points cannot show a GRADIENT. enc3/enc6 are
# what make the result identity-specific rather than merely tap-dependent, and they match the taps
# task_identity_overlap was measured at, so the two can be read side by side.
TAPS = ("enc0", "enc3", "enc6", "enc12")


def _top_basis(z: np.ndarray, kmax: int) -> np.ndarray:
    """Top-kmax right singular vectors of z, via the n x n Gram (D >> n here).

    Returns (D, kmax) orthonormal, columns in DESCENDING variance order.
    """
    g = (z @ z.T).astype(np.float64)
    w, u = np.linalg.eigh(g)
    idx = np.argsort(w)[::-1][:kmax]
    w, u = w[idx], u[:, idx]
    keep = w > max(1e-10 * float(w[0]), 1e-30)
    w, u = w[keep], u[:, keep]
    v = z.T @ u / np.sqrt(w)
    q, _ = np.linalg.qr(v)          # re-orthonormalize against fp32 drift
    return q.astype(np.float32)


def _project(z: np.ndarray, q: np.ndarray, *, keep: bool) -> np.ndarray:
    """keep=False -> remove span(q).  keep=True -> retain ONLY span(q)."""
    c = z @ q
    return (c @ q.T).astype(np.float32) if keep else (z - c @ q.T).astype(np.float32)


def _arms(z_tr: np.ndarray, rng) -> dict:
    """{arm: (D,k) basis or None} — all fit on TRAIN ONLY."""
    d = z_tr.shape[1]
    kmax = min(max(RANKS), z_tr.shape[0] - 1, d)
    top = _top_basis(z_tr, kmax)
    out: dict = {"none": None}
    for k in RANKS:
        if k > top.shape[1]:
            continue
        out[f"out_top{k}"] = ("out", top[:, :k])
        out[f"keep_top{k}"] = ("keep", top[:, :k])
        g = rng.standard_normal((d, k)).astype(np.float32)
        out[f"out_rand{k}"] = ("out", np.linalg.qr(g)[0].astype(np.float32))
    return out


def run_cell(cache_dir, tag, cell, taps=TAPS, mmap=False) -> dict:
    anchor = _load(cache_dir, CS_TRAIN_ANCHOR, tag, mmap=mmap)
    test = _load(cache_dir, cell, tag, mmap=mmap)
    a_idx, t_idx, common = _parcel_cols(anchor, test)
    assert common.size, f"cell {cell}: empty parcel intersection"
    rng = np.random.default_rng(SEED)
    rows = []
    for tap in taps:
        if tap not in anchor["feats"] or tap not in test["feats"]:
            print(f"[skip] {tap} absent", flush=True)
            continue
        for task in BOARD_TASKS:
            y_a = np.asarray(anchor["labels"][task], dtype=np.float64)
            y_t = np.asarray(test["labels"][task], dtype=np.float64)
            tr = _finite(y_a, np.arange(len(y_a)))
            va = _finite(y_t, test["cs_split"][task]["val"])
            te = _finite(y_t, test["cs_split"][task]["test"])
            if len(tr) < 2 or len(te) < 2:
                print(f"[skip] {tap} {task}: too few rows", flush=True)
                continue
            z_tr = _feat(anchor, tap, tr, a_idx)
            z_va, z_te = _feat(test, tap, va, t_idx), _feat(test, tap, te, t_idx)
            # board order: train-fitted std FIRST (mutates), then the projection
            z_tr, (z_va, z_te) = _standardize_inplace(z_tr, [z_va, z_te])
            for arm, spec in _arms(z_tr, rng).items():
                if spec is None:
                    a, b, c = z_tr, z_va, z_te
                else:
                    op, q = spec
                    kp = op == "keep"
                    a = _project(z_tr, q, keep=kp)
                    b, c = _project(z_va, q, keep=kp), _project(z_te, q, keep=kp)
                sel = _select_lam(_lam_grid(a, y_a[tr], {"val": (b, y_t[va]),
                                                         "test": (c, y_t[te])}))
                rows.append({"tag": tag, "cell": f"s{cell[0]}_t{cell[1]}", "tap": tap,
                             "task": task, "arm": arm, "test": sel["test"],
                             "val": sel["val"], "lam_pinned": sel["lam_pinned"],
                             "n_parcels": int(common.size), "d": int(z_tr.shape[1]),
                             "n_train": int(len(tr))})
            print(f"[done] {tap} {task}  d={z_tr.shape[1]}  n_tr={len(tr)}", flush=True)
    return {"cell": list(cell), "tag": tag, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--tag", default="board_r6_40k")
    ap.add_argument("--cell", required=True, help="S,T e.g. 1,1  (must be a CS test cell)")
    ap.add_argument("--taps", default=",".join(TAPS),
                    help="comma list; use a subset to compute only the DELTA of an earlier run")
    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cell = tuple(int(v) for v in a.cell.split(","))
    assert cell in CS_TEST_CELLS, f"{cell} is not one of the 10 CS cells {CS_TEST_CELLS}"

    print((__doc__ or "").split("Usage")[0])
    print(f"[arm]  tag={a.tag}  cell={cell}  anchor={CS_TRAIN_ANCHOR}  taps={a.taps}  "
          f"ranks={RANKS}  seed={SEED}")
    print("[null] arm 'none' must reproduce the board CS cell exactly — asserted at merge, not here")
    print("[read] out_top hurts & out_rand ~0 => identity and content SHARE directions")
    print("[read] out_top ~0 too => content is in LOW-variance dirs => retract the shared reading")
    print("[read] out_rand hurts as much as out_top => rank alone explains it => nothing is shown\n")
    res = run_cell(a.cache_dir, a.tag, cell,
                   taps=tuple(t for t in a.taps.split(",") if t), mmap=a.mmap)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n[out] {a.out}  {len(res['rows'])} rows")


if __name__ == "__main__":
    main()
