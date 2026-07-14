"""M10 — is there ANY global structure beyond the common mode?

Ben's nit (2026-07-15, and he is right): M7 is a PAIRWISE LINEAR variogram. Post-shaft-CAR
it reads ~0.000 between shafts in every band at every distance — but that only licenses
"a naive ADDITIVE LINEAR cross-shaft broadcast has nothing to add." It does NOT license
"no learned many-to-one function of the whole montage is informative about a local
contact." Long-range pixel-PAIR correlation in a natural image is also ~0, and global
context obviously matters there. M7 is structurally blind to:
  * many-to-one structure  (140 contacts each at r=0.03 can still be jointly predictive)
  * multiplicative structure (a gain/regime relationship has ZERO linear correlation)
  * anything defined on a summary (parcel/area) rather than on a contact pair

M10 measures exactly that space. It GATES the r4 L2 design:

  M10a  MANY-TO-ONE cross-shaft predictability, at the contact level, post-CAR.
        Predict contact c's HGA at time t from (i) its OWN shaft's other contacts at t
        [baseline] and (ii) + the top-K PCs of ALL OTHER shafts at t [full].
        Report dR2(K) = R2_full - R2_baseline, held out.
        Matched to what L2 actually implements: attention output is a weighted SUM of
        values, i.e. a low-rank many-to-one linear map at a single time slot. So this is
        the right FUNCTION CLASS, not a strawman.
        The K-sweep is the point: it says how many global latents buy how much local
        predictability => it SETS M for the Perceiver-L2, rather than us guessing 8.
        If dR2 ~ 0 at every K, the "L2 has nothing to move" claim survives WITHIN the
        class L2 implements, and L2's only legitimate job is common-mode removal.

  M10b  MULTIPLICATIVE GATING. Does the global state at t predict local HGA MAGNITUDE
        (gain), even where it does not predict local HGA (level)?
        Three steps, and step 1 is what makes it honest:
          1. ridge  g_t -> r_ct     (level)      -> R2_lin   [expect ~0, per M7]
          2. e_ct = r_ct - fitted   (kill the linear part FIRST — otherwise a linear
             g->r relationship induces a spurious g->|r| one and we would fool ourselves)
          3. ridge  g_t -> |e_ct|   (gain)       -> R2_gain
        R2_lin vs R2_gain IS the finding. Decides the READ mechanism:
          gain fires  => the global state sets the regime => GAIN-ONLY FiLM read
          gain null   => L2 has no read job at all => common-mode removal, no path back
        LIMITATION, stated up front: the v3 cache is |STFT| MAGNITUDE — there is no
        phase. True phase-amplitude coupling is therefore NOT measurable here. What is
        measurable is power-amplitude coupling. That is not a weakness of the probe for
        THIS decision: magnitude is the only thing the model can see either, so the probe
        is matched to the model's actual information.

  M10c  PARCEL-SUMMARY RELIABILITY (split-half) + inter-parcel predictability.
        Ben wants the Reve/WRITE-head target to be a per-parcel / brain-area summary
        rather than the common mode (rank-1, trivially the mean, saturates in a few
        hundred steps). Correct instinct — but a parcel-mean target is only WELL-POSED
        if the parcel mean is RELIABLE. Post-CAR within-parcel HGA correlation is ~0.03
        (M7's 0-5mm bin), so the parcel mean may be mostly independent noise averaged
        down. Split-half reliability is the CEILING on any head predicting it: you
        cannot predict more of a target than is reliable. If reliability ~ 0, the target
        is noise and the head would re-learn the conditional mean — the smear, one level
        up. This measures the ceiling BEFORE we build the head.

  M10d  EIGENSPECTRUM of the parcel-summary matrix (P x T).
        rank-1  => the common mode really is all there is => no harder target exists.
        rank ~k => there IS a reliable, LOW-DIM but MULTIDIMENSIONAL global state,
                   richer than the mean. THAT is the WRITE target (predict its top-k
                   components), and its rank sets k and M.

COMMON-MODE CONVENTION — two different objects, do not conflate them:
  * SHAFT-CAR (per-shaft mean removed): the right control for INTER-SHAFT questions
    (M10a/M10b), because it is the shaft mean that carries the inter-shaft correlation.
  * GLOBAL-mean removal (grand mean across ALL contacts): the right control for the
    PARCEL questions (M10c/M10d). Shaft-CAR would be an artifact there — a parcel is
    often close to a shaft segment, so shaft-CAR drives its parcel mean toward zero BY
    CONSTRUCTION and would fake a null.

RIGOR
  * Held out BY CLIP (train / val / test), never by random cell — the envelope is
    autocorrelated and a random split leaks the answer across the boundary.
  * Clips are independently sampled, so they can OVERLAP in the recording. Any test clip
    whose span touches a train/val clip's span is DROPPED, so test is genuinely unseen.
  * Ridge lambda swept on the VALIDATION fold only, then refit on train+val, scored once
    on test. No test-set peeking.
  * NULL = within-clip CIRCULAR SHIFT of the cross/global block. This preserves each
    predictor's autocorrelation and the target's, and destroys ONLY the cross-relation.
    A plain shuffle would destroy the autocorrelation and inflate significance — the
    classic error in this measurement.
  * Train AND test R2 both reported, so overfitting is visible rather than hidden.
  * Per-session AND pooled over the 13 pretrain montages.

Model-FREE (no checkpoint). CPU. DeltaAI/Delta login node:

  ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
  .venv/bin/python -m scripts.neuroprobe.probe_v3_global_structure \
      --band-root $ROOT \
      --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
      --bt-root /projects/bhqk/htang13/braintreebank \
      --out /projects/bhqk/htang13/probe_out_v3/field_stats/global_structure.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
from speech_decoding.models.v14_converged_v3.clip_sampler import sample_clip_start
from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
from scripts.neuroprobe.probe_v3_field_stats import (
    BAND_DIRS,
    FPS,
    V3_SESSIONS,
    WINSOR,
)

K_SWEEP = (1, 2, 4, 8, 16, 32)      # global-latent budget for M10a — this sweep SETS M
LAMBDAS = np.logspace(-2, 5, 15)    # ridge grid, chosen on the VALIDATION fold only
N_PERM = 20                         # circular-shift null draws
MIN_PARCEL_CONTACTS = 4             # a split-half needs >=2 per half


# ---------------------------------------------------------------------------
# data: clips + their start frames (we need the starts to prune train/test overlap)
# ---------------------------------------------------------------------------
def _read_clips(spec, n_clips: int, clip_frames: int, seed: int):
    """-> (bands, starts). bands: per band (n_clips, N, F, T), robust-z + winsor, guard-2
    spans excluded by the model's OWN sampler. starts: (n_clips,) frame offsets."""
    keep = spec.keep_idx.numpy()
    out: list[list[np.ndarray]] = [[] for _ in spec.band_paths]
    starts: list[int] = []
    mms = [np.load(p, mmap_mode="r") for p in spec.band_paths]
    for i in range(n_clips):
        g = torch.Generator().manual_seed(seed + i)
        t0 = sample_clip_start(
            n_frames=spec.n_frames, clip_frames=clip_frames,
            bad_spans_s=spec.bad_spans_s, fps=FPS, generator=g,
        )
        starts.append(int(t0))
        for b, (mm, norm) in enumerate(zip(mms, spec.band_norms)):
            clip = np.asarray(mm[keep, :, t0 : t0 + clip_frames], dtype=np.float32)
            out[b].append(norm.transform(torch.from_numpy(clip)).numpy())
    del mms
    return [np.stack(o, 0) for o in out], np.asarray(starts)


def _split_clips(starts: np.ndarray, clip_frames: int, frac=(0.65, 0.15)):
    """Train / val / test BY CLIP, then drop any test|val clip whose span OVERLAPS a
    train clip's span. Clips are sampled independently so overlap is possible, and an
    overlapping test clip would be partly-seen data — that is leakage, so it is removed
    rather than hoped away."""
    n = len(starts)
    n_tr = int(round(frac[0] * n))
    n_va = int(round(frac[1] * n))
    idx = np.arange(n)
    tr, va, te = idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]

    def _clean(cand, seen):
        lo_s, hi_s = starts[seen], starts[seen] + clip_frames
        ok = []
        for i in cand:
            a, b = starts[i], starts[i] + clip_frames
            if not np.any((a < hi_s) & (b > lo_s)):   # no span overlap with any `seen`
                ok.append(i)
        return np.asarray(ok, dtype=int)

    va = _clean(va, tr)
    te = _clean(te, np.concatenate([tr, va]) if len(va) else tr)
    return tr, va, te


def _design(env_by_band: list[np.ndarray], clips: np.ndarray, rows: np.ndarray | None = None):
    """(n_clips,N,T) per band  ->  (n_sel*T, n_rows*3) design over the selected contacts.
    Band-major per contact so a contact's 3 bands stay adjacent (cosmetic, but it keeps
    the PC loadings readable)."""
    sel = [e[clips] for e in env_by_band]                       # per band (n_sel, N, T)
    if rows is not None:
        sel = [e[:, rows] for e in sel]
    stk = np.stack(sel, axis=2)                                  # (n_sel, N, 3, T)
    n_sel, n_r, _, T = stk.shape
    return stk.transpose(0, 3, 1, 2).reshape(n_sel * T, n_r * 3)  # (n_sel*T, n_r*3)


def _target(env_hga: np.ndarray, clips: np.ndarray, c: int) -> np.ndarray:
    return env_hga[clips, c, :].reshape(-1)                      # (n_sel*T,)


# ---------------------------------------------------------------------------
# ridge with an honest lambda: swept on VAL, refit on train+val, scored ONCE on test
# ---------------------------------------------------------------------------
def _standardize(tr: np.ndarray, *others: np.ndarray):
    mu, sd = tr.mean(0, keepdims=True), tr.std(0, keepdims=True)
    sd = np.maximum(sd, 1e-8)
    return [(tr - mu) / sd] + [(o - mu) / sd for o in others]


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    G = X.T @ X
    G.flat[:: G.shape[0] + 1] += lam
    return np.linalg.solve(G, X.T @ y)


def _r2(y: np.ndarray, yh: np.ndarray, y_ref_mean: float) -> float:
    ss_res = float(((y - yh) ** 2).sum())
    ss_tot = float(((y - y_ref_mean) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _fit_eval(Xtr, ytr, Xva, yva, Xte, yte) -> tuple[float, float]:
    """-> (test R2, train R2). lambda picked on VAL, then refit on train+val."""
    Xtr, Xva, Xte = _standardize(Xtr, Xva, Xte)
    Xtr = np.column_stack([Xtr, np.ones(len(Xtr))])
    Xva = np.column_stack([Xva, np.ones(len(Xva))])
    Xte = np.column_stack([Xte, np.ones(len(Xte))])
    mu_tr = float(ytr.mean())

    best_lam, best = LAMBDAS[0], -np.inf
    for lam in LAMBDAS:
        w = _ridge_fit(Xtr, ytr, lam)
        s = _r2(yva, Xva @ w, mu_tr)
        if s > best:
            best, best_lam = s, lam

    Xfull = np.vstack([Xtr, Xva])
    yfull = np.concatenate([ytr, yva])
    w = _ridge_fit(Xfull, yfull, best_lam)
    return _r2(yte, Xte @ w, mu_tr), _r2(yfull, Xfull @ w, mu_tr)


def _pca_fit(Xtr: np.ndarray, k_max: int):
    """PCA fit on TRAIN ONLY (leakage guard), returned as (mean, components)."""
    mu = Xtr.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    return mu, Vt[:k_max]


def _circshift_clipwise(X: np.ndarray, n_clips: int, T: int, rng) -> np.ndarray:
    """Within-clip circular row shift. Preserves each column's autocorrelation AND the
    block's Gram matrix exactly (a shift is a within-clip row permutation); destroys only
    the cross-relation to the target. This is the null a plain shuffle cannot give."""
    Z = X.reshape(n_clips, T, X.shape[1]).copy()
    for i in range(n_clips):
        Z[i] = np.roll(Z[i], int(rng.integers(1, T)), axis=0)
    return Z.reshape(X.shape)


# ---------------------------------------------------------------------------
# M10a — many-to-one cross-shaft predictability (contact level)
#
# LEAK/DEGENERACY NOTE — the first cut of this probe was WRONG and it is worth stating
# why, because the same trap sank the M7 interpretation.
#   * v1 ran on SHAFT-CAR'd features. Shaft-CAR subtracts the per-shaft mean, so the
#     residuals of a shaft SUM TO ZERO: contact c is then an EXACT linear function of its
#     own shaft-mates. The within-shaft baseline hit R2 = 1.000 and dR2 was identically 0.
#     The probe measured nothing at all.
#   * The fix is NOT "demean globally instead": the grand mean contains c, so a
#     global-demeaned other-shaft block linearly encodes -(c + own_others), and combined
#     with the baseline it reconstructs c EXACTLY. That is a direct target leak.
# So: run on RAW features, and give the baseline the free global signal EXPLICITLY as the
# mean of contacts NOT on c's shaft (leak-free by construction — c is nowhere in it).
# dR2 then answers the actual question: does cross-shaft structure BEYOND the common mode
# add anything, given the shaft's own context and the common mode?
# ---------------------------------------------------------------------------
def m10a(env_raw, shaft_id, tr, va, te, T, targets, rng) -> dict:
    hga = env_raw[2]
    out = {f"K{k}": [] for k in K_SWEEP}
    out_null = {f"K{k}": [] for k in K_SWEEP}
    base_r2, full_train_r2 = [], []

    for c in targets:
        s = shaft_id[c]
        own = np.where((shaft_id == s) & (np.arange(len(shaft_id)) != c))[0]
        oth = np.where(shaft_id != s)[0]
        if len(own) < 1 or len(oth) < K_SWEEP[-1]:
            continue

        ytr, yva, yte = (_target(hga, ix, c) for ix in (tr, va, te))
        Wtr, Wva, Wte = (_design(env_raw, ix, own) for ix in (tr, va, te))   # own shaft
        Otr, Ova, Ote = (_design(env_raw, ix, oth) for ix in (tr, va, te))   # other shafts

        # The FREE global signal: the common mode, computed from OTHER shafts only, so c
        # cannot leak into it. This goes in the BASELINE — we are asking what a global
        # summary adds ON TOP of the common mode, not whether the common mode exists.
        cm_tr, cm_va, cm_te = (O.mean(1, keepdims=True) for O in (Otr, Ova, Ote))
        Btr = np.column_stack([Wtr, cm_tr])
        Bva = np.column_stack([Wva, cm_va])
        Bte = np.column_stack([Wte, cm_te])

        r2_base, _ = _fit_eval(Btr, ytr, Bva, yva, Bte, yte)
        base_r2.append(r2_base)

        # Residual cross-shaft structure: project the other-shaft common mode OUT, then
        # take the top-K PCs of what remains. Fit on TRAIN clips only.
        Rtr, Rva, Rte = (O - O.mean(1, keepdims=True) for O in (Otr, Ova, Ote))
        mu, comp = _pca_fit(Rtr, K_SWEEP[-1])
        Ptr, Pva, Pte = ((R - mu) @ comp.T for R in (Rtr, Rva, Rte))

        for k in K_SWEEP:
            r2_full, r2_tr = _fit_eval(
                np.column_stack([Btr, Ptr[:, :k]]), ytr,
                np.column_stack([Bva, Pva[:, :k]]), yva,
                np.column_stack([Bte, Pte[:, :k]]), yte,
            )
            out[f"K{k}"].append(r2_full - r2_base)
            if k == K_SWEEP[-1]:
                full_train_r2.append(r2_tr)

        # NULL: circular-shift ONLY the cross-shaft PC block. Baseline and target stay
        # aligned, so this isolates how much dR2 K USELESS predictors buy by chance.
        for _ in range(N_PERM):
            Ptr_n = _circshift_clipwise(Ptr, len(tr), T, rng)
            Pva_n = _circshift_clipwise(Pva, len(va), T, rng)
            Pte_n = _circshift_clipwise(Pte, len(te), T, rng)
            for k in K_SWEEP:
                r2_n, _ = _fit_eval(
                    np.column_stack([Btr, Ptr_n[:, :k]]), ytr,
                    np.column_stack([Bva, Pva_n[:, :k]]), yva,
                    np.column_stack([Bte, Pte_n[:, :k]]), yte,
                )
                out_null[f"K{k}"].append(r2_n - r2_base)

    return {
        "n_targets": len(base_r2),
        "R2_baseline_ownshaft_plus_commonmode": (
            round(float(np.mean(base_r2)), 5) if base_r2 else None),
        "R2_train_at_Kmax": round(float(np.mean(full_train_r2)), 5) if full_train_r2 else None,
        "dR2_by_K": {k: round(float(np.mean(v)), 5) for k, v in out.items() if v},
        "dR2_null_by_K": {k: round(float(np.mean(v)), 5) for k, v in out_null.items() if v},
        "dR2_null_p95_by_K": {
            k: round(float(np.percentile(v, 95)), 5) for k, v in out_null.items() if v
        },
    }


# ---------------------------------------------------------------------------
# M10b — multiplicative gating: does the global state predict local HGA MAGNITUDE?
#
# LEAK NOTE: v1 built the global state from ALL contacts, INCLUDING the target c. The
# "global state" therefore literally contained the thing it was predicting, which is why
# its level-R2 came back 0.06-0.25 instead of the ~0 the physics predicts. Here g is built
# from contacts NOT on c's shaft — c is nowhere in the predictors.
# ---------------------------------------------------------------------------
def m10b(env_raw, shaft_id, tr, va, te, T, targets, rng, k_global=8) -> dict:
    hga = env_raw[2]
    lin, gain, gain_null = [], [], []

    for c in targets:
        s = shaft_id[c]
        oth = np.where(shaft_id != s)[0]
        if len(oth) < k_global:
            continue

        ytr, yva, yte = (_target(hga, ix, c) for ix in (tr, va, te))
        Otr, Ova, Ote = (_design(env_raw, ix, oth) for ix in (tr, va, te))
        mu, comp = _pca_fit(Otr, k_global)                    # TRAIN-only PCA
        gtr, gva, gte = ((O - mu) @ comp.T for O in (Otr, Ova, Ote))

        # 1. LEVEL: does the global state predict local HGA linearly?
        r2_lin, _ = _fit_eval(gtr, ytr, gva, yva, gte, yte)
        lin.append(r2_lin)

        # 2. Kill the linear part FIRST. Without this, any linear g->y relationship would
        #    induce a spurious g->|y| one and we would report gating that is really level.
        gs_tr, gs_va, gs_te = _standardize(gtr, gva, gte)
        A = np.column_stack([gs_tr, np.ones(len(gs_tr))])
        w = _ridge_fit(A, ytr, 1.0)
        etr = np.abs(ytr - A @ w)
        eva = np.abs(yva - np.column_stack([gs_va, np.ones(len(gs_va))]) @ w)
        ete = np.abs(yte - np.column_stack([gs_te, np.ones(len(gs_te))]) @ w)

        # 3. GAIN: does g predict the MAGNITUDE of what's left?
        r2_gain, _ = _fit_eval(gtr, etr, gva, eva, gte, ete)
        gain.append(r2_gain)

        for _ in range(N_PERM):
            r2_n, _ = _fit_eval(
                _circshift_clipwise(gtr, len(tr), T, rng), etr,
                _circshift_clipwise(gva, len(va), T, rng), eva,
                _circshift_clipwise(gte, len(te), T, rng), ete,
            )
            gain_null.append(r2_n)

    return {
        "k_global": k_global,
        "n_targets": len(lin),
        "R2_level_linear": round(float(np.mean(lin)), 5) if lin else None,
        "R2_gain": round(float(np.mean(gain)), 5) if gain else None,
        "R2_gain_null": round(float(np.mean(gain_null)), 5) if gain_null else None,
        "R2_gain_null_p95": round(float(np.percentile(gain_null, 95)), 5) if gain_null else None,
        "note": "cache is |STFT| MAGNITUDE — no phase, so this is power-amplitude "
                "coupling, not true PAC. Matched to what the model can see.",
    }


# ---------------------------------------------------------------------------
# M10c / M10d — parcel summaries: is Ben's WRITE target reliable, and what rank is it?
# ---------------------------------------------------------------------------
def m10cd(env_raw, parcel_id, tr, va, te, T, rng, n_splits=20) -> dict:
    """M10c/M10d — is a PARCEL-STATE summary a well-posed WRITE target, and what rank is it?

    LEAK NOTE on the v1 of this: it built parcel summaries from GLOBAL-DEMEANED features.
    Demeaning by the grand mean makes the size-weighted parcel summaries sum to ~zero, so
    "predict parcel j from the other parcels" is partly solved by that constraint alone —
    the 0.66-0.90 R2 v1 reported was inflated by an identity, not earned. Here every
    regression runs on RAW summaries and the COMMON MODE (the other-parcel mean, which
    never contains parcel j) is given to the BASELINE. dR2 over that baseline is the real
    question: does the MULTIDIMENSIONAL parcel state carry anything the common mode does
    not? Only that increment justifies M > 1 latents.

    Split-half reliability stays on raw features and is reported both with and without the
    common mode projected out — it is the CEILING on any head predicting this target, and
    it is what tells us whether the parcel mean is signal or averaged-down noise.
    """
    parcels = [p for p in np.unique(parcel_id)
               if int((parcel_id == p).sum()) >= MIN_PARCEL_CONTACTS]
    if len(parcels) < 3:
        return {"n_parcels": len(parcels), "skipped": "fewer than 3 usable parcels"}

    hga = env_raw[2]
    all_clips = np.concatenate([tr, va, te])

    # --- M10c(i) split-half reliability = the CEILING ---------------------------------
    # For each parcel: split its contacts in half, correlate the two half-means over time.
    # "cm-removed" projects out the mean of contacts OUTSIDE this parcel (leak-free), so
    # it measures reliable parcel structure that is NOT just the common mode.
    rel_raw, rel_cm = [], []
    for p in parcels:
        idx = np.where(parcel_id == p)[0]
        out_idx = np.where(parcel_id != p)[0]
        cm = hga[all_clips][:, out_idx].mean(1).reshape(-1)       # (n*T,) — excludes parcel p
        rr, rc = [], []
        for _ in range(n_splits):
            perm = rng.permutation(idx)
            a, b = perm[: len(perm) // 2], perm[len(perm) // 2 :]
            h1 = hga[all_clips][:, a].mean(1).reshape(-1)
            h2 = hga[all_clips][:, b].mean(1).reshape(-1)
            if h1.std() < 1e-8 or h2.std() < 1e-8:
                continue
            rr.append(float(np.corrcoef(h1, h2)[0, 1]))
            # regress the common mode out of BOTH halves, then correlate the residuals
            A = np.column_stack([cm, np.ones(len(cm))])
            q1 = h1 - A @ np.linalg.lstsq(A, h1, rcond=None)[0]
            q2 = h2 - A @ np.linalg.lstsq(A, h2, rcond=None)[0]
            if q1.std() > 1e-8 and q2.std() > 1e-8:
                rc.append(float(np.corrcoef(q1, q2)[0, 1]))
        for acc, out in ((rr, rel_raw), (rc, rel_cm)):
            if acc:
                r = float(np.mean(acc))
                out.append(2 * r / (1 + r) if r > -1 else 0.0)    # Spearman-Brown -> full

    # --- M10c(ii) does the MULTIDIM parcel state beat the COMMON MODE alone? ----------
    def _psum(clips):
        cols = [np.stack([e[clips][:, parcel_id == p].mean(1) for p in parcels], 1)
                for e in env_raw]                                  # per band (n,P,T)
        return np.stack(cols, 2).transpose(0, 3, 1, 2).reshape(len(clips) * T, len(parcels) * 3)

    Str, Sva, Ste = (_psum(ix) for ix in (tr, va, te))
    Ytr, Yva, Yte = (
        np.stack([hga[ix][:, parcel_id == p].mean(1).reshape(-1) for p in parcels], 1)
        for ix in (tr, va, te)
    )
    d_inter, d_null, base_inter = [], [], []
    for j in range(len(parcels)):
        keep = [c for c in range(Str.shape[1]) if c // 3 != j]     # drop parcel j's OWN bands
        Otr, Ova, Ote = Str[:, keep], Sva[:, keep], Ste[:, keep]
        # BASELINE = the common mode only (mean of the other parcels — never contains j)
        r2_b, _ = _fit_eval(Otr.mean(1, keepdims=True), Ytr[:, j],
                            Ova.mean(1, keepdims=True), Yva[:, j],
                            Ote.mean(1, keepdims=True), Yte[:, j])
        # FULL = the whole multidimensional other-parcel state
        r2_f, _ = _fit_eval(Otr, Ytr[:, j], Ova, Yva[:, j], Ote, Yte[:, j])
        base_inter.append(r2_b)
        d_inter.append(r2_f - r2_b)
        for _ in range(max(N_PERM // 4, 3)):
            r2_n, _ = _fit_eval(
                _circshift_clipwise(Otr, len(tr), T, rng), Ytr[:, j],
                _circshift_clipwise(Ova, len(va), T, rng), Yva[:, j],
                _circshift_clipwise(Ote, len(te), T, rng), Yte[:, j],
            )
            d_null.append(r2_n - r2_b)

    # --- M10d eigenspectrum of the parcel-state matrix (P x T) ------------------------
    # Reported RAW and after removing PC1. PC1 IS the common mode — the cleanest possible
    # definition of it — so "rank after PC1 removal" is exactly "is there a global state
    # richer than the common mode, and how many dimensions does it have?" That number
    # SETS M. Grand-mean subtraction is deliberately NOT used (it is the leaky version).
    def _spec(deflate_pc1: bool):
        M = np.stack([hga[all_clips][:, parcel_id == p].mean(1).reshape(-1) for p in parcels], 0)
        M = M - M.mean(1, keepdims=True)
        M = M / np.maximum(M.std(1, keepdims=True), 1e-8)
        if deflate_pc1:
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            M = M - np.outer(U[:, 0] * S[0], Vt[0])                # strip the common mode
        e = np.linalg.svd(M, compute_uv=False) ** 2
        frac = e / max(e.sum(), 1e-12)
        return {
            "var_top1": round(float(frac[0]), 4),
            "var_top3": round(float(frac[:3].sum()), 4),
            "var_top8": round(float(frac[:8].sum()), 4),
            "participation_ratio": round(float((e.sum() ** 2) / max((e**2).sum(), 1e-12)), 2),
            "n_for_90pct": int(np.searchsorted(np.cumsum(frac), 0.90) + 1),
        }

    return {
        "n_parcels": len(parcels),
        "M10c_reliability_raw": round(float(np.mean(rel_raw)), 4) if rel_raw else None,
        "M10c_reliability_commonmode_removed": (
            round(float(np.mean(rel_cm)), 4) if rel_cm else None),
        "M10c_interparcel_R2_commonmode_baseline": (
            round(float(np.mean(base_inter)), 5) if base_inter else None),
        "M10c_interparcel_dR2_over_commonmode": (
            round(float(np.mean(d_inter)), 5) if d_inter else None),
        "M10c_interparcel_dR2_null_p95": (
            round(float(np.percentile(d_null, 95)), 5) if d_null else None),
        "M10d_spectrum_raw": _spec(False),
        "M10d_spectrum_pc1_removed": _spec(True),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--band-root", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", default=os.environ.get("ROOT_DIR_BRAINTREEBANK", ""))
    p.add_argument("--n-clips", type=int, default=128)
    p.add_argument("--clip-frames", type=int, default=96)
    p.add_argument("--n-targets", type=int, default=24, help="target contacts per session")
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--out")
    a = p.parse_args()

    specs = load_v3_sessions(
        sessions=V3_SESSIONS,
        band_cache_dirs=[os.path.join(a.band_root, b) for b in BAND_DIRS],
        span_dir=a.span_dir,
        parcel_fn=make_bt_parcel_fn(a.bt_root),
        lof_report_path=None,
        winsor=WINSOR,
    )
    print(f"M10 — {len(specs)} sessions | K sweep {K_SWEEP} | {N_PERM} circular-shift "
          f"null draws | held out BY CLIP with overlap pruning\n", flush=True)

    rows = []
    for spec in specs:
        sid, tid = spec.session_key
        sc = spec.setup.sidecar
        shaft_id = sc.shaft_id.numpy()
        parcel_id = spec.setup.parcel_id.cpu().numpy()

        bands, starts = _read_clips(spec, a.n_clips, a.clip_frames, a.seed)
        env_raw = [b.mean(2) for b in bands]                 # per band (n_clips, N, T)
        T = env_raw[0].shape[-1]

        tr, va, te = _split_clips(starts, a.clip_frames)
        if len(va) < 4 or len(te) < 4:
            print(f"[s{sid}t{tid}] SKIP — val/test too small after overlap pruning "
                  f"({len(tr)}/{len(va)}/{len(te)})", flush=True)
            continue

        rng = np.random.default_rng(a.seed + sid * 100 + tid)
        n_c = len(shaft_id)
        targets = rng.choice(n_c, size=min(a.n_targets, n_c), replace=False)

        rec = {
            "subject_id": sid, "trial_id": tid, "n_contacts": n_c,
            "n_shafts": int(sc.n_shafts),
            "clips_train_val_test": [len(tr), len(va), len(te)],
            "M10a": m10a(env_raw, shaft_id, tr, va, te, T, targets, rng),
            "M10b": m10b(env_raw, shaft_id, tr, va, te, T, targets, rng),
            "M10cd": m10cd(env_raw, parcel_id, tr, va, te, T, rng),
        }
        rows.append(rec)

        A, B, C = rec["M10a"], rec["M10b"], rec["M10cd"]
        print(f"[s{sid}t{tid}] N={n_c} clips {len(tr)}/{len(va)}/{len(te)}", flush=True)
        print(f"    M10a  base(own-shaft+common-mode) R2 "
              f"{A['R2_baseline_ownshaft_plus_commonmode']}  |  dR2 {A['dR2_by_K']}", flush=True)
        print(f"          null p95                       {A['dR2_null_p95_by_K']}", flush=True)
        print(f"    M10b  level R2 {B['R2_level_linear']}  gain R2 {B['R2_gain']}  "
              f"(gain null p95 {B['R2_gain_null_p95']})", flush=True)
        print(f"    M10cd parcels {C.get('n_parcels')}  "
              f"reliability {C.get('M10c_reliability_commonmode_removed')} "
              f"(raw {C.get('M10c_reliability_raw')})  "
              f"inter-parcel dR2-over-common-mode "
              f"{C.get('M10c_interparcel_dR2_over_commonmode')} "
              f"(null p95 {C.get('M10c_interparcel_dR2_null_p95')})", flush=True)
        print(f"          parcel-state spectrum, PC1(common mode) REMOVED: "
              f"{C.get('M10d_spectrum_pc1_removed')}", flush=True)

    # ---- pooled ----
    print("\n" + "=" * 78)
    print("M10 — POOLED over sessions (the numbers that gate the r4 L2 design)")
    print("=" * 78)
    pooled: dict = {}
    if rows:
        pooled["M10a_dR2_by_K"] = {
            f"K{k}": round(float(np.mean([r["M10a"]["dR2_by_K"][f"K{k}"] for r in rows
                                          if f"K{k}" in r["M10a"]["dR2_by_K"]])), 5)
            for k in K_SWEEP
        }
        pooled["M10a_dR2_null_p95_by_K"] = {
            f"K{k}": round(float(np.mean([r["M10a"]["dR2_null_p95_by_K"][f"K{k}"] for r in rows
                                          if f"K{k}" in r["M10a"]["dR2_null_p95_by_K"]])), 5)
            for k in K_SWEEP
        }
        pooled["M10a_baseline_R2"] = round(
            float(np.mean([r["M10a"]["R2_baseline_ownshaft_plus_commonmode"] for r in rows])), 5)
        for key, src in (("M10b_level_R2", "R2_level_linear"), ("M10b_gain_R2", "R2_gain"),
                         ("M10b_gain_null_p95", "R2_gain_null_p95")):
            pooled[key] = round(float(np.mean([r["M10b"][src] for r in rows])), 5)
        cd = [r["M10cd"] for r in rows if r["M10cd"].get("n_parcels", 0) >= 3]
        if cd:
            for key, src in (
                ("M10c_reliability_cm_removed", "M10c_reliability_commonmode_removed"),
                ("M10c_reliability_raw", "M10c_reliability_raw"),
                ("M10c_interparcel_commonmode_R2", "M10c_interparcel_R2_commonmode_baseline"),
                ("M10c_interparcel_dR2", "M10c_interparcel_dR2_over_commonmode"),
                ("M10c_interparcel_dR2_null_p95", "M10c_interparcel_dR2_null_p95"),
            ):
                pooled[key] = round(float(np.mean([c[src] for c in cd])), 5)
            for tag, sk in (("raw", "M10d_spectrum_raw"), ("pc1rm", "M10d_spectrum_pc1_removed")):
                for f in ("var_top1", "var_top8", "participation_ratio", "n_for_90pct"):
                    pooled[f"M10d_{tag}_{f}"] = round(
                        float(np.mean([c[sk][f] for c in cd])), 3)

        print("\nM10a — does a K-dim GLOBAL summary add to local-HGA prediction, ON TOP OF")
        print("       the contact's own shaft AND the common mode?")
        print(f"  baseline R2 (own shaft + common mode): {pooled['M10a_baseline_R2']}")
        for k in K_SWEEP:
            d = pooled["M10a_dR2_by_K"][f"K{k}"]
            n = pooled["M10a_dR2_null_p95_by_K"][f"K{k}"]
            print(f"  K={k:>2}: dR2 {d:+.5f}   (null p95 {n:+.5f})   -> "
                  f"{'REAL' if d > n else 'null'}")

        print("\nM10b — LEVEL vs GAIN (decides additive cross-attn vs GAIN-ONLY FiLM read)")
        print(f"  R2(global -> local HGA level) : {pooled['M10b_level_R2']:+.5f}")
        print(f"  R2(global -> local HGA |gain|): {pooled['M10b_gain_R2']:+.5f}  "
              f"(null p95 {pooled['M10b_gain_null_p95']:+.5f})")

        if cd:
            print("\nM10c — is a PARCEL-STATE summary a well-posed WRITE target?")
            print(f"  split-half reliability, common mode REMOVED: "
                  f"{pooled['M10c_reliability_cm_removed']}   <-- CEILING on any such head")
            print(f"  split-half reliability, raw                : "
                  f"{pooled['M10c_reliability_raw']}")
            print(f"  inter-parcel R2 from the COMMON MODE alone : "
                  f"{pooled['M10c_interparcel_commonmode_R2']}")
            print(f"  ...dR2 from the MULTIDIM parcel state      : "
                  f"{pooled['M10c_interparcel_dR2']:+.5f}  "
                  f"(null p95 {pooled['M10c_interparcel_dR2_null_p95']:+.5f})")
            print("\nM10d — parcel-state RANK. PC1 == the common mode; the rank of what is")
            print("       LEFT after stripping it is the dimension of the real global state,")
            print("       and that number SETS M.")
            print(f"  raw        : top1 {pooled['M10d_raw_var_top1']}  "
                  f"PR {pooled['M10d_raw_participation_ratio']}  "
                  f"n@90% {pooled['M10d_raw_n_for_90pct']}")
            print(f"  PC1 removed: top1 {pooled['M10d_pc1rm_var_top1']}  "
                  f"PR {pooled['M10d_pc1rm_participation_ratio']}  "
                  f"n@90% {pooled['M10d_pc1rm_n_for_90pct']}   <-- M")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"k_sweep": list(K_SWEEP), "n_perm": N_PERM,
                       "per_session": rows, "pooled": pooled}, fh, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
