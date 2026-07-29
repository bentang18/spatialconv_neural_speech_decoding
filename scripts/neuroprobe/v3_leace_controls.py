"""Does the LEACE null mean anything? Geometry of the erased direction, plus two control arms.

The published result is that erasing identity at enc12 destroys ~21% of the representation's
variance and moves cross-subject decoding by 7e-6. Two readings survive that number equally well:

  A. Pretraining RELOCATES identity into a high-variance subspace DISJOINT from content.
  B. The erased axis is the global common-mode / amplitude axis -- which differs across subjects
     for recording reasons (gain, impedance, referencing, montage) and which the decoder never
     used -- so deleting it was always going to be free, in any model.

Reading B is not exotic. The leading variance direction of an iEEG representation is routinely
common mode, ``_standardize_inplace`` already z-scores per feature with anchor statistics, and
``std_target`` (AdaBN) -- a strictly weaker version of the same correction -- was a WASH. Nothing
measured so far separates A from B, and the paper's contribution rests entirely on A.

Three things are needed and none require a new encode:

**Geometry.** Where does the erased direction actually sit? If it is essentially PC1, and PC1
carries about as much variance as the erasure removes, B is the parsimonious reading. If the
direction is spread over many PCs, or sits well down the spectrum, A survives. Also measured: its
overlap with the uniform (common-mode) vector, and with the between-domain mean shift that
per-feature standardisation already removes -- the direct test of "AdaBN should have caught this".

**A rank control (`leace_shuf`).** The eraser spends ONE direction out of ~7000. If content is
distributed and redundant, deleting any single direction costs nothing whatever it encodes, and
the null is a statement about ridge's robustness rather than about geometry. Shuffling the domain
labels erases an arbitrary rank-1 direction and measures that floor. Without it we cannot claim
the test HAD POWER to detect a cost -- the same gap `randmask` was built to close for masking.

**A variance control (`leace_toppc`).** Stronger, because it is matched on the quantity that makes
the headline sound impressive: it deletes PC1 outright, roughly as much variance as the identity
erasure, but chosen with no reference to identity. If deleting PC1 is also free, then "21% of
variance for 7e-6" says nothing about identity -- it says the top of this spectrum is inert.

All three erasers share ONE SVD (`fit_leace(..., svd=)`). The factorisation depends only on the
features, never on the concept, and at d=93184 it is the entire bill -- the ridge that follows is
~10% of it. So the controls cost ~13% more than the identity arm alone, not 3x.

Scored through the same ``B._lam_grid`` as the board, so every arm is comparable by construction.
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
import v3_cs_leace as L  # noqa: E402

from speech_decoding.experiments.leace import LeaceEraser, fit_leace  # noqa: E402

ARMS = ("std", "leace", "leace_shuf", "leace_toppc")


def _top_pc_eraser(mean, basis, s, k: int = 1) -> LeaceEraser:
    """Orthogonally delete the top ``k`` principal components.

    Built by hand rather than through ``fit_leace`` because there is no concept here -- the point
    is a direction chosen with NO reference to identity, matched only on variance. In basis
    coordinates that projector is a diagonal indicator, which makes ``r(x)`` the plain orthogonal
    removal of those components.
    """
    proj = np.zeros((s.size, s.size))
    proj[np.arange(k), np.arange(k)] = 1.0
    return LeaceEraser(
        mean=mean, basis=basis, proj=proj,
        var_removed=float((s[:k] ** 2).sum() / (s**2).sum()),
        residual_cov=0.0, sv=s, removed_dir=basis[:, :k],
    )


def _along(x: np.ndarray, mu: np.ndarray, d: np.ndarray, domain: np.ndarray) -> dict:
    """Split the variance a unit direction ``d`` carries into BETWEEN-domain and WITHIN-domain.

    This is the number the headline turns on. AUROC is computed from a linear score ``w . x`` and
    is invariant to a CONSTANT per-row offset, so any part of ``d``'s variance that is just the
    anchor-vs-test mean offset was never visible to the metric -- deleting it is free in any model,
    trained or random. Only the WITHIN-domain (trial-to-trial) part can move a score. So
    "erasing identity destroys 20.7% of the variance for 7e-6" only means what we want it to mean
    if a real share of that 20.7% is within-domain.

    The same split is reported for PC1, because if PC1 is ALSO mostly a domain offset then the
    whole top of this spectrum is a recording-frame artifact rather than representational content.
    """
    p = x @ d - float(mu @ d)                           # projected, centered, without copying x
    gm = np.array([p[domain == g].mean() for g in (0, 1)])
    nb = np.array([int((domain == g).sum()) for g in (0, 1)])
    total = float(p.var(ddof=1))
    between = float((nb * (gm - p.mean()) ** 2).sum() / (p.size - 1))
    return {"var": total, "between_frac": between / total if total > 0 else float("nan")}


def _geometry(er: LeaceEraser, x: np.ndarray, domain: np.ndarray) -> dict:
    """Locate the erased direction in the variance spectrum, and split what it carries.

    ``cos_domain_mean_shift`` is an ALGEBRAIC IDENTITY, not a measurement: for a binary concept the
    LEACE projector's range is ``span(Sigma_xz)`` and ``Sigma_xz`` is proportional to
    ``mu_anchor - mu_test``, so it is 1.0 exactly (verified to 1e-15 on synthetic data). It is kept
    as a self-check -- a value below 1 means something upstream is wrong -- and NEVER as evidence.
    Its real content is interpretive: what LEACE erases here IS the between-session mean-shift
    axis, which is why the between/within split below is the load-bearing diagnostic.
    """
    s, d = er.sv, er.removed_dir[:, 0]
    tot = float((s**2).sum())
    c = er.basis.T @ d                                  # the direction, expressed over PCs
    w = c**2 / max(float((c**2).sum()), 1e-300)         # its distribution over PCs
    uni = np.ones(d.size) / np.sqrt(d.size)             # common mode: every feature, equal weight
    mu = er.mean
    dir_v = _along(x, mu, d, domain)
    pc1_v = _along(x, mu, er.basis[:, 0], domain)
    dm = x[domain == 0].mean(0) - x[domain == 1].mean(0)
    dm_n = float(np.linalg.norm(dm))
    return {
        # If these two are close AND cos_pc1 ~ 1, the erasure is just "delete PC1".
        "pc1_var_frac": float(s[0] ** 2 / tot),
        "var_removed": er.var_removed,
        "var_along_dir": float((w * (s**2)).sum() / tot),
        "cos_pc1": float(abs(c[0])),
        # How many PCs the direction really occupies. ~1 => it IS a principal axis.
        "pc_participation": float(1.0 / max(float((w**2).sum()), 1e-300)),
        "wt_in_top10_pcs": float(w[:10].sum()),
        "pc_com": float((w * np.arange(w.size)).sum()),  # centre of mass in the spectrum
        # THE decisive split. between_frac ~ 1 => the erasure deleted an AUROC-invisible offset.
        "dir_between_frac": dir_v["between_frac"],
        "pc1_between_frac": pc1_v["between_frac"],
        # The "boring explanation" axis, and the identity check.
        "cos_common_mode": float(abs(d @ uni)),
        "cos_domain_mean_shift": float(abs(d @ (dm / dm_n))) if dm_n > 0 else float("nan"),
    }


def _cell_arms(anchor_rec, test_rec, task, taps, n_components) -> dict:
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
        evals = {"val": (None, y_t[va]), "test": (None, y_t[te])}

        z_test_all = np.vstack([z_va, z_te])
        stacked = np.vstack([z_tr, z_test_all])
        domain = np.r_[np.zeros(len(z_tr)), np.ones(len(z_test_all))].astype(int)

        # THE shared factorisation. Everything below is a different concept over the same features.
        xc = stacked.astype(np.float64)
        mean = xc.mean(0)
        xc -= mean
        svd = np.linalg.svd(xc, full_matrices=False)
        del xc

        rng = np.random.default_rng(33)
        shuf = domain.copy()
        rng.shuffle(shuf)

        erasers = {
            "leace": lambda: fit_leace(stacked, domain, n_components=n_components, svd=svd),
            "leace_shuf": lambda: fit_leace(stacked, shuf, n_components=n_components, svd=svd),
            "leace_toppc": lambda: _top_pc_eraser(mean, svd[2][: svd[1].size].T, svd[1]),
        }
        for name, make in erasers.items():
            er = make()
            if name == "leace":
                checks[enc] = {
                    "n_parcels": int(common.size), "d": int(z_tr.shape[1]),
                    "n_rows": int(stacked.shape[0]), "rank": int(svd[1].size),
                    **_geometry(er, stacked, domain),
                }
                g = checks[enc]
                assert g["cos_domain_mean_shift"] > 0.999, (
                    f"the erased direction must BE the between-domain mean shift for a binary "
                    f"concept, got cos={g['cos_domain_mean_shift']:.6f}")
                print(f"    [geom] {enc:>6} pc1_var {g['pc1_var_frac']:.4f} | erased_var "
                      f"{g['var_removed']:.4f} | cos_pc1 {g['cos_pc1']:.4f} | "
                      f"pc_participation {g['pc_participation']:7.1f} | top10 "
                      f"{g['wt_in_top10_pcs']:.4f} | cos_common {g['cos_common_mode']:.4f} | "
                      f"BETWEEN dir {g['dir_between_frac']:.4f} pc1 {g['pc1_between_frac']:.4f}",
                      flush=True)
            a, (b, c) = B._standardize_inplace(
                L._f32(er(z_tr)), [L._f32(er(z_va)), L._f32(er(z_te))])
            grid[(enc, name)] = B._lam_grid(
                a, y_a[tr], {"val": (b, evals["val"][1]), "test": (c, evals["test"][1])})
            checks[enc][f"var_removed_{name}"] = er.var_removed
            del er

        a, (b, c) = B._standardize_inplace(z_tr, [z_va, z_te])
        grid[(enc, "std")] = B._lam_grid(
            a, y_a[tr], {"val": (b, evals["val"][1]), "test": (c, evals["test"][1])})
        del svd

    if not grid:
        return {}
    return {"cells": B._grid_cells(grid), "checks": checks, "n_parcels": int(common.size)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--cell", required=True, help="test cell as 'S,T' (never the anchor)")
    p.add_argument("--taps", default="enc0,enc12")
    p.add_argument("--tasks", default="", help="default: every task in the cache")
    p.add_argument("--leace-components", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cell = tuple(int(v) for v in args.cell.split(","))
    assert cell != B.CS_TRAIN_ANCHOR, f"{cell} is the anchor"
    assert cell in B.LITE_SESSIONS, f"{cell} is not a Lite session"
    taps = tuple(t for t in args.taps.split(",") if t)

    anchor_rec = B._load(args.cache, B.CS_TRAIN_ANCHOR, args.tag, mmap=L.LOAD_MMAP)
    test_rec = B._load(args.cache, cell, args.tag, mmap=L.LOAD_MMAP)
    tasks = [t for t in (args.tasks.split(",") if args.tasks else sorted(test_rec["labels"]))
             if t in test_rec["labels"] and t in anchor_rec["labels"]]
    print(f"[run] anchor {B.CS_TRAIN_ANCHOR} -> test {cell} | {len(tasks)} tasks | taps {taps} | "
          f"arms {ARMS}", flush=True)

    out: dict = {}
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        print(f"  {task}", flush=True)
        res = _cell_arms(anchor_rec, test_rec, task, taps, args.leace_components)
        if not res:
            continue
        out[task] = res
        for key in sorted(res["cells"]):
            print(f"    [score] {task} {key} test={res['cells'][key]['test']:.4f}", flush=True)
        tmp = dest.with_suffix(".partial")
        tmp.write_text(json.dumps(out, indent=1, default=float))
        tmp.replace(dest)
        print(f"  [wrote] {len(out)}/{len(tasks)} tasks -> {dest}", flush=True)

    print(f"[done] {dest}  ({len(out)} tasks)", flush=True)
    B._phase_report(f"leace_controls {cell} taps={','.join(taps)}")


if __name__ == "__main__":
    main()
