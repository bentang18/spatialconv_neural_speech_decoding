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


def _subspace_alignment(y: np.ndarray, domain: np.ndarray, ks=(8, 32, 128)) -> dict:
    """Do the two sessions share a coordinate system, once their mean difference is removed?

    This is the POSITIVE question, and it is a different claim from anything LEACE measures.
    "Identity is filed in a disjoint subspace" says where the session DIFFERENCE lives. It says
    nothing about whether the remaining structure is shared -- and shared structure is what
    actually makes a ridge fit on subject A work on subject B. A model could file identity
    perfectly and still be useless cross-subject if each session's content lived in its own
    unrelated directions.

    Measured as the mean squared principal cosine between the top-k within-session principal
    subspaces of the two domains: ||Va^T Vt||_F^2 / k, in [0, 1]. Each domain is centered on ITS
    OWN mean first, so the between-session offset -- the thing LEACE erases and AUROC ignores --
    cannot contribute.

    ``y`` is the data already expressed in the shared row-space basis, (n, rank). That is exact,
    not an approximation: every row lies in the pooled affine span, and each domain mean is an
    average of rows, so per-domain centering cannot leave it. So this costs two SVDs of
    (n_domain, rank) instead of two of (n_domain, 93184) -- essentially free.

    Alignment alone does not pin down the SHAPE of the correspondence, and the shape is the
    interesting part. Ben's read of the trajectory figures is that same-task trajectories from
    different subjects trace the same path up to a SCALE change, with no rotation. In this
    language that is a much stronger claim than overlap: it says the test session's covariance is
    close to DIAGONAL in the anchor's own eigenbasis, i.e. the two sessions agree on the axes
    themselves and disagree only on how far they travel along each. An arbitrary orthogonal map
    would preserve the subspace and destroy the diagonal, so `diag_k` separates the two readings
    that eyeballing two or three principal components cannot.
    """
    rank = y.shape[1]
    rng = np.random.default_rng(17)
    perm = rng.permutation(len(domain))
    out: dict = {}

    def top(rows: np.ndarray, k: int):
        z = y[rows] - y[rows].mean(0)
        _, s, vt = np.linalg.svd(z, full_matrices=False)
        return vt[:k].T, s[:k]                                         # (rank, k), (k,)

    def diag_mass(c: np.ndarray, sb: np.ndarray) -> float:
        cov = (c * (sb**2)) @ c.T
        return float(np.square(np.diag(cov)).sum() / np.square(cov).sum())

    def stats(ra: np.ndarray, rb: np.ndarray, k: int):
        va, _ = top(ra, k)
        vb, sb = top(rb, k)
        c = va.T @ vb                                                  # (k, k) principal cosines
        align = float(np.square(c).sum() / k)
        # b's covariance expressed in a's eigenbasis; how much of it sits on the diagonal.
        # Compared against a RANDOM ROTATION of the same spectrum, never against an absolute
        # threshold: diagonal mass has a baseline set by the eigenvalue spread alone. With a steep
        # spectrum every diagonal entry converges to the mean eigenvalue and a fully rotated
        # covariance still keeps ~60% of its mass on the diagonal. "Same axes up to scale" is a
        # claim ABOUT the rotation, so the random rotation is the hypothesis it has to beat.
        rot = [diag_mass(np.linalg.qr(rng.normal(size=(k, k)))[0], sb) for _ in range(8)]
        return align, diag_mass(c, sb), float(np.mean(rot))

    # The reference is a CEILING, not a floor. Splitting the pooled rows at random gives two iid
    # samples of one distribution, so their subspaces agree as closely as finite n permits -- the
    # best any pair could do -- and cross-session alignment is reported as a FRACTION of that. The
    # floor is analytic (k/rank) and is kept only because it is near zero at these ranks, which
    # would otherwise make any nonzero overlap look impressive.
    real =[np.flatnonzero(domain == 0), np.flatnonzero(domain == 1)]
    fake = [perm[: len(real[0])], perm[len(real[0]):]]
    for k in ks:
        if k >= min(len(real[0]), len(real[1])) or k >= rank:
            continue
        a_ov, a_dg, a_rot = stats(real[0], real[1], k)
        c_ov, c_dg, _ = stats(fake[0], fake[1], k)
        out[f"align_k{k}"] = a_ov
        out[f"align_k{k}_ceil"] = c_ov
        out[f"align_k{k}_floor"] = k / rank
        out[f"align_k{k}_frac"] = a_ov / c_ov if c_ov > 0 else float("nan")
        out[f"diag_k{k}"] = a_dg
        out[f"diag_k{k}_rot"] = a_rot
        out[f"diag_k{k}_ceil"] = c_dg
    return out


def _task_direction(y: np.ndarray, rows: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Unit vector from the label-0 class mean to the label-1 class mean, within one session."""
    pos, neg = rows[lab[rows] > 0], rows[lab[rows] <= 0]
    if len(pos) < 2 or len(neg) < 2:
        return np.zeros(y.shape[1])
    d = y[pos].mean(0) - y[neg].mean(0)
    n = float(np.linalg.norm(d))
    return d / n if n > 0 else d


def _task_alignment(y: np.ndarray, domain: np.ndarray, lab: np.ndarray, reps=64) -> dict:
    """Do the two sessions encode the TASK along the same axis, and is that axis the session axis?

    ``_subspace_alignment`` asks whether the sessions share a coordinate system for their full
    single-trial covariance, and the answer at the deep tap is "less than at enc0". That does not
    settle the question the project actually cares about, because single-trial covariance is
    dominated by session-private variability -- electrode placement, impedance, drift -- and the
    shared, task-locked component can be a small slice of it. The trajectory figures average trials
    within a condition, which suppresses exactly that private part; this statistic does the same
    thing, in the same shared basis, at the resolution the binary tasks allow: one discriminative
    direction per session.

    Two numbers, and they answer different questions:

    ``task_cos``     -- cross-session agreement of the task axis. Its null is a LABEL SHUFFLE done
                        independently inside each session, which keeps every session-specific
                        property (spectrum, class balance, n) and destroys only the task.
    ``task_vs_sess`` -- overlap of the task axis with the between-session offset, per session. This
                        is the honest version of "separability": the offset is a real, large
                        direction, and whether the task rides on it or is orthogonal to it is a
                        measurable fact rather than an inference from an erasure that costs nothing.
    """
    rows = [np.flatnonzero(domain == 0), np.flatnonzero(domain == 1)]
    d_a, d_t = (_task_direction(y, r, lab) for r in rows)
    if not d_a.any() or not d_t.any():
        return {}

    sess = y[rows[1]].mean(0) - y[rows[0]].mean(0)
    sess = sess / max(float(np.linalg.norm(sess)), 1e-300)

    rng = np.random.default_rng(101)
    null = []
    for _ in range(reps):
        sh = lab.copy()
        for r in rows:                                   # shuffle WITHIN session, never across
            sh[r] = rng.permutation(sh[r])
        null.append(abs(float(_task_direction(y, rows[0], sh) @ _task_direction(y, rows[1], sh))))
    null = np.asarray(null)
    obs = abs(float(d_a @ d_t))
    return {
        "task_cos": obs,
        "task_cos_null": float(null.mean()),
        "task_cos_null_p95": float(np.quantile(null, 0.95)),
        "task_cos_p": float((null >= obs).mean()),
        "task_cos_frac": float(obs / null.mean()) if null.mean() > 0 else float("nan"),
        # Orthogonality of task to the session offset. Chance is ~1/sqrt(rank), so report both.
        "task_vs_sess_a": abs(float(d_a @ sess)),
        "task_vs_sess_t": abs(float(d_t @ sess)),
        "task_vs_sess_chance": float(1.0 / np.sqrt(y.shape[1])),
    }


def _geometry(er: LeaceEraser, x: np.ndarray, domain: np.ndarray) -> dict:
    """Locate the erased direction in the variance spectrum, and split what it carries.

    ``cos_domain_mean_shift`` is an ALGEBRAIC IDENTITY, not a measurement: for a binary concept the
    LEACE projector's range is ``span(Sigma_xz)`` and ``Sigma_xz`` is proportional to
    ``mu_anchor - mu_test``, so it is 1.0 exactly (verified to 1e-15 on synthetic data). It is kept
    as a self-check -- a value below 1 means something upstream is wrong -- and NEVER as evidence.
    Its real content is interpretive: what LEACE erases here IS the between-session mean-shift
    axis, which is why the between/within split below is the load-bearing diagnostic.

    ``cos_pc1`` and ``pc_participation`` have a NULL that must be printed beside them, because
    ANISOTROPY alone drives both toward "it IS a principal axis" with no group structure at all: a
    i^-3 spectrum gives cos_pc1 ~.67 / participation 2.6 from nothing. Dimensionality was never the
    mechanism -- white features at d/n=13 give .04 / 180. Under the null the group-mean difference
    is a draw from ``N(0, Sigma * 4/n)``, so its expected squared loading on PC_i goes as lambda_i:

        E[cos_pc1^2] = pc1_var_frac          and       E[pc_participation] = 1 / sum(f_i^2)

    So ``cos_pc1_excess`` (observed cos^2 over that null) is the number to read; a large
    ``cos_pc1`` beside a large ``pc1_var_frac`` is not evidence of anything. Simulated across decay
    0-3 and d/n 0.29-13.3, the null excess never exceeds ~1.5, while a rigid offset of 1 total sd
    reaches ~5. High d/n biases it DOWNWARD (~0.6, sample pc1 eigenvalue inflation), i.e. toward
    the safe side, so the deep taps' excess is if anything understated.

    Read per-cell values with care even so: under the null ``cos_pc1^2`` rides one chi-square-1
    variate, so a SINGLE cell near 1 is weak evidence and the agreement across cells is the signal.
    """
    s, d = er.sv, er.removed_dir[:, 0]
    tot = float((s**2).sum())
    f = s**2 / tot                                      # the spectrum, as fractions of variance
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
        # ...but ~1 is only meaningful against what the spectrum alone would give. See the docstring.
        "cos_pc1_null": float(np.sqrt(f[0])),
        "cos_pc1_excess": float(c[0] ** 2 / max(float(f[0]), 1e-300)),
        "pc_participation_null": float(1.0 / max(float((f**2).sum()), 1e-300)),
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
        checks[enc] = {"n_parcels": int(common.size), "d": int(z_tr.shape[1]),
                       "n_rows": int(stacked.shape[0]), "rank": int(svd[1].size)}

        # The positive question, asked of the same factorisation: with the session offset removed
        # by per-domain centering, do the two sessions USE the same axes? `u * s` is exactly the
        # stacked rows in the shared basis, so this is a change of coordinates, not a new fit.
        al = _subspace_alignment(svd[0] * svd[1], domain)
        checks[enc].update(al)
        for k in sorted({int(q.split("_k")[1]) for q in al if q.startswith("align_k")
                         and q.count("_") == 1}):
            print(f"    [algn] {enc:>6} k={k:<4} overlap {al[f'align_k{k}']:.4f} | ceiling "
                  f"{al[f'align_k{k}_ceil']:.4f} | frac {al[f'align_k{k}_frac']:.4f} | floor "
                  f"{al[f'align_k{k}_floor']:.5f} | diag {al[f'diag_k{k}']:.4f} vs ceil "
                  f"{al[f'diag_k{k}_ceil']:.4f} vs rot {al[f'diag_k{k}_rot']:.4f}", flush=True)

        # Same basis, but the TASK-locked component rather than the full single-trial covariance.
        labels = np.r_[y_a[tr], y_t[va], y_t[te]]
        ta = _task_alignment(svd[0] * svd[1], domain, labels)
        checks[enc].update(ta)
        if ta:
            print(f"    [task] {enc:>6} cross-session cos {ta['task_cos']:.4f} | null "
                  f"{ta['task_cos_null']:.4f} (p95 {ta['task_cos_null_p95']:.4f}) | frac "
                  f"{ta['task_cos_frac']:.2f} | p {ta['task_cos_p']:.3f} || vs session offset "
                  f"a {ta['task_vs_sess_a']:.4f} t {ta['task_vs_sess_t']:.4f} | chance "
                  f"{ta['task_vs_sess_chance']:.4f}", flush=True)
        for name, make in erasers.items():
            er = make()
            # Geometry for EVERY eraser, not just the real one. `leace_shuf` is the matched null
            # for these statistics and it is the only defensible one: dir_between_frac is measured
            # along a direction CHOSEN to maximise the domain gap, so it is biased upward, and the
            # bias grows with d/n -- the axis on which enc0 (d/n ~ 0.3) and enc12 (d/n ~ 13) differ
            # by ~45x. Measured on synthetic data with a matched d/n, the NULL dir_between_frac is
            # 0.15 at enc0's ratio and 0.89 at enc12's. Comparing raw enc0-vs-enc12 without this
            # null reads a dimensionality artifact as a fact about the model. The shuffled arm
            # supplies the null on the REAL features, at the real spectrum, for free -- the SVD is
            # already shared, so this costs one extra projection.
            geom = _geometry(er, stacked, domain)
            checks[enc].update({f"{k}_{name}": v for k, v in geom.items()})
            if name == "leace":
                checks[enc].update(geom)          # unprefixed keys stay the published contract
                assert geom["cos_domain_mean_shift"] > 0.999, (
                    f"the erased direction must BE the between-domain mean shift for a binary "
                    f"concept, got cos={geom['cos_domain_mean_shift']:.6f}")
            print(f"    [geom] {enc:>6} {name:<11} var {geom['var_removed']:.4f} | cos_pc1 "
                  f"{geom['cos_pc1']:.4f} vs null {geom['cos_pc1_null']:.4f} (x"
                  f"{geom['cos_pc1_excess']:.2f}) | pc_part {geom['pc_participation']:7.1f} vs "
                  f"null {geom['pc_participation_null']:7.1f} | BETWEEN "
                  f"{geom['dir_between_frac']:.4f} | effective_within "
                  f"{geom['var_removed'] * (1 - geom['dir_between_frac']):.5f}", flush=True)
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
