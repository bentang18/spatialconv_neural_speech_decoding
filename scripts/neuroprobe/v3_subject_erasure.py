"""Measurement C: erase between-subject directions, re-run the keep-S CS ridge (delta-only).

Estimate the top-m between-subject discriminant directions U (512-d, the per-parcel pooled
unit), project them out of every (parcel, frame) block, and re-run the PUBLISHED cross-subject
ridge on the held-out subject.  Reads the block cache written by v3_subject_nuisance.py --
encodes nothing, and refits no published number it could have read.

Estimation is LEAVE-EVAL-SUBJECT-OUT: for test subject s, U comes from the other six
(anchor + the five non-s test subjects).  s's own data never touches U.

The ridge is dual, so erasure is a Gram correction and the m-sweep is nearly free:

    X_e[:, block_pt] = B_pt (I - U Uᵀ)   =>   G_e = G - Y Yᵀ,  Y = [ B_pt U ]_pt

and because U's columns are ordered, the prefix-m Grams accumulate column by column -- one
pass gives every m.  **m=0 therefore reproduces the published AUROC exactly**; the run prints
that delta as its own correctness check.

Ranks are matched against two controls, because dropping rank from a ridge design can help on
its own and would otherwise be mistaken for a nuisance effect:
  random-m  -- a random orthonormal m-dim subspace
  pca-m     -- the top-m PRINCIPAL directions (highest-variance, subject-agnostic)
and against a leaky ORACLE ceiling (U estimated on anchor-vs-s, the subject being tested).

  python -u v3_subject_erasure.py --cell enc3:30k --out erase_enc3_30k.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from v3_subject_nuisance import ANCHOR, SESSIONS, TASKS, TEST_SUBJECTS, block_path

M_GRID = [0, 1, 2, 3, 5, 8, 12, 16, 24, 32]
CONST_LAM_MULT = 1.0  # parity with v3_probe_readout_keeptime
D = 512               # per-parcel pooled unit: [mean_e, std_e] over d=256


# ------------------------------------------------------- direction estimation

def _groups(blk: dict, par: dict, sessions, rng, cap: int) -> dict:
    """{(subject, parcel): (<=cap, 512) tokens}. Every (clip, frame) of a parcel is one sample of
    the 512-d unit, which is parcel-AGNOSTIC (the same 512 dims whatever parcel it pools)."""
    out = {}
    for sess in sessions:
        for j, p in enumerate(par[sess].tolist()):
            x = blk[sess][:, j].astype(np.float32).reshape(-1, D)   # (n*T, 512)
            if len(x) > cap:
                x = x[rng.choice(len(x), size=cap, replace=False)]
            out[(sess[0], int(p))] = x.astype(np.float64)
    return out


def _lda_dirs(groups: dict, k: int, shrink: float) -> np.ndarray:
    """Top-k whitened-LDA directions from the WITHIN-PARCEL between-subject scatter.

    ``groups`` maps (subject, parcel) -> tokens.  Scatter is accumulated per parcel over the
    subjects that HAVE that parcel, then summed.  Conditioning on parcel matters:

      * pooling tokens across parcels would score region COMPOSITION (only S4 has parcel 74) as
        subject identity, which is not what breaks anchor->test transfer;
      * what breaks transfer is that for a SHARED parcel the anchor's and the test subject's
        512-vectors have different moments (30 electrodes pooled vs 4), so anchor-fit ridge
        weights meet features that do not mean the same thing;
      * and there is no parcel common to all subjects anyway (S1/S2 sit in parcels 35-69, the
        rest in 0-30), so a common-parcel restriction leaves the estimation set EMPTY.

    Conditioning also lifts the rank<=n_subjects-1 ceiling, since scatter accrues across parcels.
    Whitened (not raw S_B): raw between-scatter is exactly the span of the mean differences, so
    erasing it zeroes the means and INLP would stall after one round."""
    parcels: dict[int, list] = {}
    for (subj, p), t in groups.items():
        parcels.setdefault(p, []).append(t)

    S_B = np.zeros((D, D))
    S_W = np.zeros((D, D))
    n_par = 0
    for p, ts in parcels.items():
        if len(ts) < 2:                                    # a parcel one subject owns says nothing
            continue
        n_par += 1
        mus = np.stack([t.mean(0) for t in ts])
        M = mus - mus.mean(0)
        S_B += M.T @ M / len(M)
        for t in ts:
            c = t - t.mean(0)
            S_W += c.T @ c / len(c)
    if n_par == 0:
        raise SystemExit("no parcel is shared by >=2 estimation subjects -- cannot estimate U")
    S_B /= n_par
    S_W /= sum(len(ts) for ts in parcels.values() if len(ts) >= 2)
    S_W += shrink * np.trace(S_W) / D * np.eye(D)          # ridge: S_W is rank-deficient after INLP
    L = np.linalg.cholesky(S_W)
    W = np.linalg.solve(L, np.linalg.solve(L, S_B).T).T    # S_W^-1 S_B, symmetrised below
    ev, V = np.linalg.eigh((W + W.T) / 2)
    return V[:, np.argsort(ev)[::-1][:k]]


def _orth_append(U: np.ndarray | None, V: np.ndarray) -> np.ndarray:
    """Gram-Schmidt V against U, append, re-orthonormalise."""
    A = V if U is None else np.concatenate([U, V], axis=1)
    Q, _ = np.linalg.qr(A)
    return Q


def _inlp(groups: dict, m_max: int, shrink: float) -> np.ndarray:
    """Iterative nullspace projection: fit directions, erase, refit on the residual.

    Each round is capped by the LDA rank, so reaching m_max REQUIRES the iteration -- it is not
    an optional refinement.  Directions are always refit on the residual of the ORIGINAL tokens
    after erasing everything found so far."""
    n_subj = len({s for s, _ in groups})
    U = None
    cur = {k: v.copy() for k, v in groups.items()}
    while U is None or U.shape[1] < m_max:
        have = 0 if U is None else U.shape[1]
        k = min(max(n_subj - 1, 1), m_max - have)
        V = _lda_dirs(cur, k, shrink)
        U = _orth_append(U, V)
        cur = {g: t - (t @ U) @ U.T for g, t in groups.items()}
    return U[:, :m_max]


def _pca_dirs(groups: dict, m_max: int) -> np.ndarray:
    """Top-m PRINCIPAL directions of the pooled tokens: a subject-AGNOSTIC, rank-matched control.

    Dropping rank from a ridge design can help on its own; without this, any lift from the
    subject-targeted erasure would be uninterpretable."""
    C = np.zeros((D, D))
    n = 0
    for t in groups.values():
        C += t.T @ t
        n += len(t)
    mu = sum(t.sum(0) for t in groups.values()) / n
    C = C / n - np.outer(mu, mu)
    _, V = np.linalg.eigh(C)
    return V[:, ::-1][:, :m_max]


def _eta2(groups: dict, U: np.ndarray, m: int) -> float:
    """WITHIN-PARCEL between-subject variance fraction after erasing U[:, :m] — ties C back to A.

    Conditioned on parcel for the same reason the directions are (see _lda_dirs): the marginal
    version would score region composition as subject identity."""
    cur = groups if m == 0 else {g: t - (t @ U[:, :m]) @ U[:, :m].T for g, t in groups.items()}
    parcels: dict[int, list] = {}
    for (_s, p), t in cur.items():
        parcels.setdefault(p, []).append(t)
    tr_B = tr_T = 0.0
    n_par = 0
    for ts in parcels.values():
        if len(ts) < 2:
            continue
        n_par += 1
        mus = np.stack([t.mean(0) for t in ts])
        mu = mus.mean(0)
        tr_B += float(((mus - mu) ** 2).sum() / len(mus))
        tr_T += float(sum(((t - mu) ** 2).sum() for t in ts) / sum(len(t) for t in ts))
    return tr_B / tr_T if tr_T > 0 else float("nan")


# ------------------------------------------------------------------ the ridge

def _auroc(scores, y) -> float:
    from sklearn.metrics import roc_auc_score
    yb = (np.asarray(y) > 0).astype(int)
    if yb.min() == yb.max():
        return float("nan")
    return float(roc_auc_score(yb, np.asarray(scores)))


def _base_grams(a_blk, a_keep, t_blk, t_keep):
    """G = Xtr Xtrᵀ and K = Xte Xtrᵀ on the UNERASED design.

    Accumulated parcel-by-parcel in float64, never materialising X.  Depends only on the rows,
    NOT on U -- so it is built once per task and reused by every erasure variant."""
    G = np.zeros((len(a_blk), len(a_blk)))
    K = np.zeros((len(t_blk), len(a_blk)))
    for p_a, p_t in zip(a_keep, t_keep):
        A = a_blk[:, p_a].astype(np.float64).reshape(len(a_blk), -1)   # (r, T*512)
        B = t_blk[:, p_t].astype(np.float64).reshape(len(t_blk), -1)
        G += A @ A.T
        K += B @ A.T
    return G, K


def _erase_sweep(a_blk, a_keep, t_blk, t_keep, U, m_grid, G0, K0):
    """{m: (G_m, K_m)} for every m, in ONE pass over U's columns.

    Column j contributes independently, so the prefix-m Grams are a running subtraction and the
    whole m-grid costs one sweep."""
    out = {}
    Gm, Km = G0.copy(), K0.copy()
    if 0 in m_grid:
        out[0] = (Gm.copy(), Km.copy())
    for j in range(max(m_grid)):
        u = U[:, j]
        Ya = np.concatenate([a_blk[:, p].astype(np.float64) @ u for p in a_keep], axis=1)  # (r,P*T)
        Yt = np.concatenate([t_blk[:, p].astype(np.float64) @ u for p in t_keep], axis=1)
        Gm -= Ya @ Ya.T
        Km -= Yt @ Ya.T
        if (j + 1) in m_grid:
            out[j + 1] = (Gm.copy(), Km.copy())
    return out


def _ridge(G, K, y_tr, y_te) -> float:
    n = len(G)
    lam = CONST_LAM_MULT * float(np.trace(G) / max(n, 1))   # recomputed on the ERASED Gram
    alpha = np.linalg.solve(G + lam * np.eye(n), np.asarray(y_tr, dtype=np.float64))
    return _auroc(K @ alpha, y_te)


# ------------------------------------------------------------------------ main

def run(tap: str, tag: str, variant: str, *, seed: int, tok_cap: int, shrink: float,
        m_grid: list[int], out_path: str) -> None:
    rng = np.random.default_rng(seed)
    z = {s: np.load(block_path(tap, tag, s, variant)) for s in SESSIONS}
    blk = {s: z[s]["blocks"] for s in SESSIONS}
    par = {s: z[s]["parcels"] for s in SESSIONS}
    a_sess = ANCHOR
    results = {"tap": tap, "step": tag, "variant": variant, "m_grid": m_grid, "subjects": {}}

    for s in TEST_SUBJECTS:
        t_sess = next(x for x in SESSIONS if x[0] == s)
        inter = np.array(sorted(set(par[a_sess].tolist()) & set(par[t_sess].tolist())), dtype=np.int64)
        a_keep = np.searchsorted(par[a_sess], inter)
        t_keep = np.searchsorted(par[t_sess], inter)

        # --- U: leave-eval-subject-out. s's data never touches U. No common-parcel restriction:
        # the six estimation subjects share NO parcel at all (S1/S2 in 35-69, the rest in 0-30),
        # and the 512-d unit is parcel-agnostic, so scatter is conditioned on parcel instead.
        est = [x for x in SESSIONS if x[0] != s]
        toks = _groups(blk, par, est, rng, tok_cap)

        m_max = max(m_grid)
        U = {
            "inlp": _inlp(toks, m_max, shrink),
            "pca": _pca_dirs(toks, m_max),
            "random": np.linalg.qr(rng.standard_normal((D, m_max)))[0],
        }
        # leaky ceiling: U from anchor-vs-s (the very subject being held out)
        U["oracle"] = _inlp(_groups(blk, par, [a_sess, t_sess], rng, tok_cap), m_max, shrink)

        sres = {"n_parcels_inter": int(len(inter)), "n_est_groups": len(toks),
                "eta2_after": {}, "auroc": {}, "std_half_energy": {}}
        # WHAT did we erase?  The 512-d unit is [mean_e | std_e] over the electrodes in a parcel.
        # std_e is IDENTICALLY 0 for a 1-electrode parcel and mean_e's noise scales as 1/sqrt(k),
        # and electrode counts differ wildly across subjects in the SAME parcel (anchor has 30 in
        # parcel 13, S9 has 4).  So a subject-invariant encoder would still leak subject identity
        # through this pooling.  Energy of U in the std half separates "we erased a neural subject
        # code" from "we erased electrode-count structure".
        for name, Um in U.items():
            e = (Um ** 2).sum(0)
            sres["std_half_energy"][name] = [float(v) for v in (Um[D // 2:] ** 2).sum(0) / e]

        # Rows are task-specific (finite-label mask + that task's cs_split), but the UNERASED
        # Gram depends only on the rows -- so build it once per task and reuse it for every
        # erasure variant, instead of once per (variant, task).
        rows_a = np.arange(len(blk[a_sess]))
        pre = {}
        for task in TASKS:
            ya, yt = z[a_sess][f"y_{task}"], z[t_sess][f"y_{task}"]
            tr = rows_a[np.isfinite(ya[rows_a])]                  # anchor: all finite rows
            te = np.asarray(z[t_sess][f"cs_{task}"])
            te = te[np.isfinite(yt[te])]                          # test: its cs_split test half
            A, B = blk[a_sess][tr], blk[t_sess][te]
            G0, K0 = _base_grams(A, a_keep, B, t_keep)
            pre[task] = (A, B, G0, K0, ya[tr], yt[te])

        for name, Um in U.items():
            sres["eta2_after"][name] = {str(m): _eta2(toks, Um, m) for m in m_grid}
            for task in TASKS:
                A, B, G0, K0, y_tr, y_te = pre[task]
                grams = _erase_sweep(A, a_keep, B, t_keep, Um, m_grid, G0, K0)
                for m in m_grid:
                    G, K = grams[m]
                    sres["auroc"].setdefault(name, {}).setdefault(task, {})[str(m)] = _ridge(
                        G, K, y_tr, y_te)
                del grams
            b0 = np.mean([sres["auroc"][name][t]["0"] for t in TASKS])
            best_m = max(m_grid, key=lambda m: np.mean([sres["auroc"][name][t][str(m)] for t in TASKS]))
            best = np.mean([sres["auroc"][name][t][str(best_m)] for t in TASKS])
            print(f"[{tap} {tag}] S{s} {name:7s} m=0 {b0:.4f} -> best m={best_m:<2d} {best:.4f} "
                  f"(delta {best - b0:+.4f})  eta2 {sres['eta2_after'][name]['0']:.4f} -> "
                  f"{sres['eta2_after'][name][str(best_m)]:.4f}", flush=True)
        del pre
        results["subjects"][f"S{s}"] = sres
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    print(f"wrote {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True, help="tap:tag, e.g. enc3:30k")
    p.add_argument("--variant", default="ln", choices=["ln", "raw"])
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--tok-cap", type=int, default=3000,
                   help="tokens per (subject, parcel) for estimating U. ~6 subj x ~18 parcels x "
                        "3000 = ~320k tokens pooled, ample for a 512x512 S_W")
    p.add_argument("--shrink", type=float, default=1e-2, help="S_W ridge (trace-relative)")
    p.add_argument("--m", type=int, action="append", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    tap, tag = a.cell.split(":")
    run(tap, tag, a.variant, seed=a.seed, tok_cap=a.tok_cap, shrink=a.shrink,
        m_grid=sorted(a.m or M_GRID), out_path=a.out)


if __name__ == "__main__":
    main()
