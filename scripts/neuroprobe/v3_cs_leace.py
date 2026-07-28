"""CS identity-erasure arm: does removing subject identity change cross-subject decoding?

Mirrors ``v3_board_readout._cs_cell`` exactly -- same anchor (S2T4), same anchor-intersect-test
parcel columns, same finite-row masks, same ``cs_split`` val/test halves, same lambda grid -- and
adds erasure arms alongside the board's own norms so every number is comparable to the leaderboard
by construction, not by re-derivation.

THE ANCHOR IS ONE SUBJECT. ``CS_TRAIN_ANCHOR = (2, 4)``, so the CS training set has no subject
variance and a multi-class identity eraser cannot be fitted on it. The nuisance that actually
exists in this regime is the anchor-vs-test-subject shift, so the concept is BINARY per cell:
domain 0 = anchor rows, domain 1 = test rows. That makes the eraser rank 1 -- it spends exactly
one direction -- and makes this a domain-adaptation experiment, which is what CS transfer is.

Fitting the eraser needs test-subject FEATURES but never test LABELS, so it is transductive, not
leakage. The board already ships the weaker version of the same idea: ``std_target`` is AdaBN,
equalising each domain's per-feature mean and variance. The three arms are a ladder of increasing
domain correction, all scored identically:

    std          anchor statistics applied to the test subject -- no adaptation (the .6021 baseline)
    std_target   AdaBN: per-domain first and second moments, per feature
    leace        provably complete LINEAR domain erasure (arXiv 2306.03819)

Linear-only erasure is the right scope because the readout is ridge.

Checks printed BEFORE any score, per (cell, task, tap):
  * ``id_auc_before`` / ``id_auc_after`` -- held-out anchor-vs-test decodability. The eraser is
    refit on half the rows and scored on the other half, so this measures whether erasure
    GENERALISES rather than restating the fit-data identity.
  * ``residual_cov`` -- domain covariance left on the fit rows; ~0 unless truncated.
  * ``var_removed`` -- fraction of feature variance destroyed, against its ``(C-1)/rank`` floor.
    Near the floor means removal was nearly free; far above means identity rode high-variance
    directions.

A null result is only interpretable once ``id_auc_after`` is at chance. Unchanged CS with identity
still decodable means the erasure did not transfer, NOT that identity was inert.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import v3_board_readout as B  # noqa: E402

from speech_decoding.experiments.leace import fit_leace  # noqa: E402

ARMS = ("std", "std_target", "leace")

# MMAP IS MANDATORY HERE, not a tuning knob. A board session cache is 40-64 GB on disk (anchor
# s2_t4 is 47 GB), and this driver holds TWO of them open at once, so the eager path needs
# 87-111 GB before a single feature is gathered. Launched at 64G it OOM-killed all ten cells in
# the LOADER, before the first print (job 20540001, 07-27).
#
# Paging costs nothing we use: the CS arms gather only the parcel-POOLED taps, so the ~34 GB
# enc12_elec tensor in each file is never indexed and therefore never read. ``MMAP_DEFAULT["cs"]``
# is False in v3_board_readout because that default is written for the whole-file consumers; the
# board's own CS runs override it on the command line.
LOAD_MMAP = True


def _probe_auc(x_tr, y_tr, x_te, y_te) -> float:
    """rcond is load-bearing: erased features are rank-deficient and pinv's default tolerance
    inverts the erased direction into a ~1e14-norm weight vector, which reads as 'erasure
    failed' when it is numerical blow-up in the probe."""
    mu = x_tr.mean(0)
    w = np.linalg.pinv(x_tr - mu, rcond=1e-8) @ (y_tr - y_tr.mean())
    return B.auroc((x_te - mu) @ w, y_te - 0.5)


def _identity_checks(z_a, z_t, n_components, seed=33):
    """Held-out anchor-vs-test decodability, before and after an eraser fit on the other half."""
    rng = np.random.default_rng(seed)
    ia, it = rng.permutation(len(z_a)), rng.permutation(len(z_t))
    ha, ht = len(z_a) // 2, len(z_t) // 2
    fit = np.vstack([z_a[ia[:ha]], z_t[it[:ht]]])
    held = np.vstack([z_a[ia[ha:]], z_t[it[ht:]]])
    y_fit = np.r_[np.zeros(ha), np.ones(len(z_t[it[:ht]]))]
    y_held = np.r_[np.zeros(len(z_a) - ha), np.ones(len(z_t) - ht)]
    if min(y_fit.sum(), len(y_fit) - y_fit.sum(), y_held.sum(), len(y_held) - y_held.sum()) < 2:
        return float("nan"), float("nan")
    before = _probe_auc(fit, y_fit, held, y_held)
    er = fit_leace(fit, y_fit.astype(int), n_components=n_components)
    after = _probe_auc(er(fit), y_fit, er(held), y_held)
    return before, after


def _cs_cell_arms(anchor_rec, test_rec, task, taps, n_components) -> dict:
    y_a = np.asarray(anchor_rec["labels"][task], dtype=np.float64)
    y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
    tr = B._finite(y_a, np.arange(len(y_a)))
    va = B._finite(y_t, test_rec["cs_split"][task]["val"])
    te = B._finite(y_t, test_rec["cs_split"][task]["test"])
    if len(tr) < 2 or len(te) < 2:
        return {}
    a_idx, t_idx, common = B._parcel_cols(anchor_rec, test_rec)
    if common.size == 0:
        return {}

    grid, checks = {}, {}
    for enc in taps:
        if enc not in anchor_rec["feats"] or enc not in test_rec["feats"]:
            continue
        z_tr = B._feat(anchor_rec, enc, tr, a_idx)
        z_va = B._feat(test_rec, enc, va, t_idx)
        z_te = B._feat(test_rec, enc, te, t_idx)

        def evals(b, c):
            return {"val": (b, y_t[va]), "test": (c, y_t[te])}

        z_test_all = np.vstack([z_va, z_te])
        before, after = _identity_checks(z_tr, z_test_all, n_components)

        domain = np.r_[np.zeros(len(z_tr)), np.ones(len(z_test_all))].astype(int)
        eraser = fit_leace(np.vstack([z_tr, z_test_all]), domain, n_components=n_components)
        rank = min(len(domain) - 1, z_tr.shape[1])
        checks[enc] = {
            "id_auc_before": before,
            "id_auc_after": after,
            "residual_cov": eraser.residual_cov,
            "var_removed": eraser.var_removed,
            "var_removed_floor": 1.0 / rank,
            "n_parcels": int(common.size),
            "d": int(z_tr.shape[1]),
        }
        print(f"    [check] {enc:>6} id_auc {before:.3f} -> {after:.3f} | "
              f"residual_cov {eraser.residual_cov:.2e} | var_removed {eraser.var_removed:.5f} "
              f"(floor {1.0 / rank:.5f}) | {'OK' if abs(after - 0.5) < 0.10 else 'ERASURE DID NOT TRANSFER'}",
              flush=True)

        e_tr, e_va, e_te = eraser(z_tr), eraser(z_va), eraser(z_te)
        a, (b, c) = B._standardize_inplace(e_tr, [e_va, e_te])
        grid[(enc, "leace")] = B._lam_grid(a, y_a[tr], evals(b, c))

        a, (b, c) = B._standardize_per_domain(z_tr, z_va, z_te)
        grid[(enc, "std_target")] = B._lam_grid(a, y_a[tr], evals(b, c))

        a, (b, c) = B._standardize_inplace(z_tr, [z_va, z_te])
        grid[(enc, "std")] = B._lam_grid(a, y_a[tr], evals(b, c))

    if not grid:
        return {}
    return {"cells": B._grid_cells(grid), "checks": checks, "n_parcels": int(common.size)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--cell", required=True, help="test cell as 'S,T' (never the anchor)")
    p.add_argument("--taps", default="enc0,enc12", help="enc0 is the paired control")
    p.add_argument("--tasks", default="", help="default: every task in the cache")
    p.add_argument("--leace-components", type=int, default=None,
                   help="row-space truncation; None keeps erasure exact (watch residual_cov)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cell = tuple(int(v) for v in args.cell.split(","))
    assert cell != B.CS_TRAIN_ANCHOR, f"{cell} is the anchor; upstream asserts test != anchor"
    assert cell in B.LITE_SESSIONS, f"{cell} is not a Lite session"
    taps = tuple(t for t in args.taps.split(",") if t)

    anchor_rec = B._load(args.cache, B.CS_TRAIN_ANCHOR, args.tag, mmap=LOAD_MMAP)
    test_rec = B._load(args.cache, cell, args.tag, mmap=LOAD_MMAP)
    print(f"[check] loaded anchor+test mmap={LOAD_MMAP} "
          f"({'lazy -- elec taps never paged' if LOAD_MMAP else 'EAGER -- will OOM on 40-64 GB caches'})",
          flush=True)
    tasks = [t for t in (args.tasks.split(",") if args.tasks else sorted(test_rec["labels"]))
             if t in test_rec["labels"] and t in anchor_rec["labels"]]
    print(f"[run] anchor {B.CS_TRAIN_ANCHOR} -> test {cell} | {len(tasks)} tasks | taps {taps} | "
          f"arms {ARMS}", flush=True)

    out: dict = {}
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        print(f"  {task}", flush=True)
        res = _cs_cell_arms(anchor_rec, test_rec, task, taps, args.leace_components)
        if not res:
            continue
        out[task] = res

        # Scores go to the LOG as well as the JSON. enc12 is d=93184 and a cell takes hours, so a
        # walltime kill used to lose everything: the JSON was written once at the end and the
        # numbers appeared nowhere else (job 20540912 was on track to burn 80 cores for 2 h and
        # emit nothing). Now every task is durable twice over -- rewritten JSON plus a log line.
        for key in sorted(res["cells"]):
            print(f"    [score] {task} {key} test={res['cells'][key]['test']:.4f} "
                  f"lam={res['cells'][key]['lam_mult']}", flush=True)
        tmp = dest.with_suffix(".partial")
        tmp.write_text(json.dumps(out, indent=1, default=float))
        tmp.replace(dest)                      # atomic: a reader never sees a half-written file
        print(f"  [wrote] {len(out)}/{len(tasks)} tasks -> {dest}", flush=True)

    print(f"[done] {dest}  ({len(out)} tasks)", flush=True)


if __name__ == "__main__":
    main()
