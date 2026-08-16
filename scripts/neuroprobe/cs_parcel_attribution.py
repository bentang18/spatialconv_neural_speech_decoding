#!/usr/bin/env python
"""Does PRETRAINING tighten the link between physiology and decoding? enc0 vs enc12, per parcel.

THE QUESTION (Greg, via Ben 2026-08-12). The encoder is frozen and never sees a label, so it
cannot select anatomy for the task; the only thing that chooses where the signal is read from is
the linear ridge on top. That decoder is linear, so its anatomy is readable rather than inferred.
The interesting comparison is therefore not "which parcel does enc12 use" but "does enc12 read
from MORE physiologically appropriate anatomy than enc0 does".

  enc0   the spectrogram floor, same parcels, same splits, same protocol
  enc12  the pretrained tap

Both are fit here so the profiles are paired on the cell. A difference is the result; NO
difference is also a result, and the third outcome is the uncomfortable one. See DECISION MAP.

TWO READINGS, and only one of them travels across taps:

  LOPO         drop parcel p from the anchor-cap-test intersection, refit the whole lambda grid,
               re-select on val, re-score test. An AUROC drop, so it is in the SAME UNITS at both
               taps and is the instrument for the enc0-vs-enc12 comparison.
  WEIGHT MASS  ||beta_p|| at the selected lambda. Descriptive, and NOT comparable across taps:
               enc0 is 348 features per parcel against enc12's 52*384, fit through a different
               solver branch. Reported per tap, never differenced between them.

SIZE IS THE CONFOUND, and it is printed rather than hoped away. Dropping superior temporal from
S3 removes 21 of 27 intersection contacts, so a big AUROC drop is exactly what "I deleted most of
the input" also predicts. Every LOPO row carries the contacts removed, so the readout can ask
whether a parcel sits above the size trend instead of assuming it does.

GATE FIRST. The full-intersection refit must reproduce the published shard's <tap>|std test AUROC
and its selected lambda_mult, for EVERY tap and task, before any delta is computed. Fatal, not a
warning: a LOPO delta against an unverified baseline is not a measurement.

Attribution is exact because the cache is stored (n, |P|, F) and flattened row-major
(_pool_parcels in v3_probe_encode_r4), so beta reshapes to (|P|, F) with no ambiguity, and because
the encoder has no cross-shaft pathway (ENC_LAYOUT is 12 L1 blocks, towers.py) so a parcel's
features are a function of that parcel's contacts alone.

Usage:
  python scripts/neuroprobe/cs_parcel_attribution.py --cache <DIR> --tag <TAG> \
      --index 2 --shard-dir <published cs shards> --out <DIR>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.neuroprobe.v3_board_readout import (  # noqa: E402
    CS_TEST_CELLS,
    CS_TRAIN_ANCHOR,
    LAM_MULTS,
    UNKNOWN_PARCEL_ID,
    _feat,
    _finite,
    _linear_grams,
    _load,
    _select_lam,
    _standardize_inplace,
    auroc,
)

TAPS = ("enc0", "enc12")
NORM = "std"
GATE_TOL = 1e-3          # board/LEACE agree to 4.1e-4; the dual/primal fp32 gap is 2.7e-5

# ── PRE-REGISTERED, written before the first number was read ──────────────────────────────────
# The claim under test is "pretraining improves the physiology-to-decoding linkage". Stating the
# three outcomes in advance is what stops the third one from being retold as the first.
DECISION_MAP = """
DECISION MAP (pre-registered; STG = superior temporal, the a-priori speech parcel)
  H1  STG sensitivity HIGHER at enc12 than enc0, paired over cells
        => pretraining concentrates the readout on classical speech cortex. Greg's hypothesis.
  H0  no change
        => pretraining AMPLIFIES what was already there without RELOCATING it. Consistent with
           the coexistence frame and with a selective gain; NOT a failure, and NOT alignment.
  H2  STG sensitivity LOWER at enc12 while AUROC RISES
        => the added signal sits OUTSIDE classical speech cortex. Legitimate, possibly the most
           interesting outcome, and it must NOT be written up as improved physiological alignment.
  SIZE  if per-parcel drop tracks contacts-removed with no residual for STG, the anatomy claim is
        explained by parcel size and there is nothing anatomical to report at all. Checked FIRST.
"""


def _fit_grid(z_tr, y_tr, evals):
    """The board's ridge, with beta kept. Returns (grid, beta_at(lam_mult)).

    Branches on the SAME `d < n` test `_lam_grid` uses, so this fits whatever the board fit. That
    matters here: CS enc0 is d = |P| * 348 against n_train 2096-3500, the one tap in the whole
    board that lands in the primal, while CS enc12 is d = |P| * 52 * 384 and is always dual.
    Duplicated from `_lam_grid` rather than imported because that function returns AUROC only and
    discards the coefficients, and widening its contract would touch the code path that produces
    every published board number.
    """
    idx = {m: i for i, m in enumerate(LAM_MULTS)}
    if z_tr.shape[1] < z_tr.shape[0]:
        # PRIMAL — mirrors _lam_grid_primal. beta falls out directly, one column per lambda.
        a_mat = np.asarray(z_tr.T @ z_tr, dtype=np.float64)
        n = z_tr.shape[0]
        w, V = np.linalg.eigh(a_mat)
        c = V.T @ (z_tr.T @ np.asarray(y_tr, dtype=np.float64))
        base = float(np.trace(a_mat) / max(n, 1))               # == sum(eig(G))/n, the dual scale
        lam = np.asarray(LAM_MULTS, dtype=np.float64) * base
        beta = V @ (c[:, None] / (w[:, None] + lam[None, :]))   # (d, |LAM|)
        grid: dict = {}
        for name, (z, y) in evals.items():
            if len(y) < 2:
                grid[name] = {m: float("nan") for m in LAM_MULTS}
                continue
            s = np.asarray(z, dtype=np.float64) @ beta
            grid[name] = {m: auroc(s[:, i], y) for i, m in enumerate(LAM_MULTS)}
        return grid, (lambda m: beta[:, idx[m]]), "primal"

    # DUAL — mirrors _lam_grid. beta is recovered as Z_tr^T alpha.
    g, kern = _linear_grams(z_tr, evals)
    w, V = np.linalg.eigh(g)
    c = V.T @ np.asarray(y_tr, dtype=np.float64)
    base = float(np.sum(w) / max(g.shape[0], 1))
    grid = {name: {} for name in evals}
    for m in LAM_MULTS:
        alpha = V @ (c / (w + m * base))
        for name, (_, y) in evals.items():
            grid[name][m] = auroc(kern[name] @ alpha, y) if len(y) >= 2 else float("nan")
    zt = np.asarray(z_tr, dtype=np.float64)
    return grid, (lambda m: zt.T @ (V @ (c / (w + m * base)))), "dual"


def _parcel_norms(beta, n_parcels):
    if beta.size % n_parcels:
        raise SystemExit(f"d={beta.size} is not divisible by |P|={n_parcels}")
    return np.linalg.norm(beta.reshape(n_parcels, -1), axis=1)


def _cols(anchor_rec, test_rec):
    a_p = np.asarray(anchor_rec["present_parcels"], dtype=np.int64)
    t_p = np.asarray(test_rec["present_parcels"], dtype=np.int64)
    common = np.intersect1d(a_p, t_p)
    common = common[common != UNKNOWN_PARCEL_ID]
    a_idx = [int(np.where(a_p == p)[0][0]) for p in common]
    t_idx = [int(np.where(t_p == p)[0][0]) for p in common]
    return a_idx, t_idx, common


def _contacts(rec, parcels):
    """Contacts per parcel, from parcel_canon — the SIZE control for every LOPO drop."""
    pc = np.asarray(rec["parcel_canon"], dtype=np.int64)
    return {int(p): int(np.sum(pc == p)) for p in parcels}


def _fit(anchor_rec, test_rec, tap, tr, va, te, y_a, y_t, a_idx, t_idx, want_beta):
    z_tr = _feat(anchor_rec, tap, tr, a_idx)
    z_va = _feat(test_rec, tap, va, t_idx)
    z_te = _feat(test_rec, tap, te, t_idx)
    z_tr, (z_va, z_te) = _standardize_inplace(z_tr, [z_va, z_te])
    evals = {"val": (z_va, y_t[va]), "test": (z_te, y_t[te])}
    grid, beta_at, branch = _fit_grid(z_tr, y_a[tr], evals)
    sel = _select_lam(grid)
    beta = (_parcel_norms(beta_at(sel["lam_mult"]), len(a_idx))
            if want_beta and np.isfinite(sel["lam_mult"]) else None)
    return sel, beta, branch, z_tr.shape[1]


def run(cache, tag, cell, shard_path, out_dir, taps, mmap):
    print(DECISION_MAP, flush=True)
    pub = json.load(open(shard_path))["cells"]
    anchor_rec = _load(cache, CS_TRAIN_ANCHOR, tag, mmap=mmap)
    test_rec = _load(cache, cell, tag, mmap=mmap)
    a_idx, t_idx, common = _cols(anchor_rec, test_rec)
    n_a = _contacts(anchor_rec, common)
    n_t = _contacts(test_rec, common)
    print(f"[cell] S{cell[0]}T{cell[1]}  anchor S{CS_TRAIN_ANCHOR[0]}T{CS_TRAIN_ANCHOR[1]}  "
          f"|P|={len(common)}", flush=True)
    for p in common:
        print(f"[parcel]   atlas {int(p):3d}  anchor {n_a[int(p)]:3d} / test {n_t[int(p)]:3d} "
              f"contacts", flush=True)

    out = {"cell": f"S{cell[0]}T{cell[1]}", "tag": tag, "norm": NORM,
           "parcels": common.tolist(), "contacts_anchor": n_a, "contacts_test": n_t,
           "taps": {t: {} for t in taps}}

    for key, blk in pub.items():
        task = key.split("|", 1)[1]
        y_a = np.asarray(anchor_rec["labels"][task], dtype=np.float64)
        y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
        tr = _finite(y_a, np.arange(len(y_a)))
        va = _finite(y_t, test_rec["cs_split"][task]["val"])
        te = _finite(y_t, test_rec["cs_split"][task]["test"])
        if len(tr) < 2 or len(te) < 2:
            continue

        for tap in taps:
            ref = blk["cells"].get(f"{tap}|{NORM}")
            if ref is None:
                continue
            full, beta, branch, d = _fit(anchor_rec, test_rec, tap, tr, va, te,
                                         y_a, y_t, a_idx, t_idx, True)
            d_auc = abs(full["test"] - ref["test"])
            if not (d_auc <= GATE_TOL and full["lam_mult"] == ref["lam_mult"]):
                raise SystemExit(
                    f"GATE FAIL {tap} {task}: refit test {full['test']:.6f} vs published "
                    f"{ref['test']:.6f} (|d| {d_auc:.2e}, tol {GATE_TOL}); "
                    f"lam {full['lam_mult']:.6g} vs {ref['lam_mult']:.6g}")

            lopo = {}
            if len(common) >= 2:
                for j, p in enumerate(common):
                    keep = [k for k in range(len(common)) if k != j]
                    s, _, _, _ = _fit(anchor_rec, test_rec, tap, tr, va, te, y_a, y_t,
                                      [a_idx[k] for k in keep], [t_idx[k] for k in keep], False)
                    drop = full["test"] - s["test"]
                    lopo[int(p)] = {
                        "test": s["test"], "drop": drop,
                        # gain above chance is the house normalizer; it is what makes a drop at
                        # enc0 comparable to a drop at enc12 when the two sit at different AUROC.
                        "drop_norm": drop / (full["test"] - 0.5) if full["test"] > 0.5 else None,
                        "contacts_removed": n_t[int(p)], "lam_mult": s["lam_mult"],
                    }

            out["taps"][tap][task] = {
                "full_test": full["test"], "published_test": ref["test"], "branch": branch,
                "d": int(d), "lam_mult": full["lam_mult"],
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "beta_norm": {int(p): float(v) for p, v in zip(common, beta)}
                if beta is not None else None,
                "lopo": lopo,
            }
            worst = max(lopo.items(), key=lambda kv: kv[1]["drop"]) if lopo else None
            print(f"[ok] {tap:6s} {task:18s} test {full['test']:.4f} ({branch}, d={d}, "
                  f"gate |d| {d_auc:.1e})"
                  + (f"  worst LOPO atlas {worst[0]} -{worst[1]['drop']:.4f} "
                     f"({worst[1]['contacts_removed']} contacts)" if worst else ""),
                  flush=True)

    dst = pathlib.Path(out_dir) / f"parcelattr_cs_S{cell[0]}T{cell[1]}_{tag}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1)
    counts = ", ".join("{}:{}".format(t, len(out["taps"][t])) for t in taps)
    print(f"[write] {dst}  ({counts})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    # --index indexes CS_TEST_CELLS, the SAME list `--mode cs` shards over, so an array task here
    # and the published shard it gates against are the same cell by construction.
    ap.add_argument("--index", type=int, required=True,
                    help=f"0..{len(CS_TEST_CELLS) - 1}, indexes CS_TEST_CELLS")
    ap.add_argument("--shard-dir", required=True, help="published cs shards (the gate)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--taps", default=",".join(TAPS))
    ap.add_argument("--mmap", action="store_true")
    a = ap.parse_args()
    s, t = CS_TEST_CELLS[a.index]
    run(a.cache, a.tag, (s, t), f"{a.shard_dir}/cs_S{s}T{t}.json", a.out,
        tuple(a.taps.split(",")), a.mmap)


if __name__ == "__main__":
    main()
