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
    _shaft_car,
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
# M10a — many-to-one cross-shaft predictability (contact level, post-shaft-CAR)
# ---------------------------------------------------------------------------
def m10a(env_car, shaft_id, tr, va, te, T, targets, rng) -> dict:
    hga = env_car[2]
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
        Btr, Bva, Bte = (_design(env_car, ix, own) for ix in (tr, va, te))
        Otr, Ova, Ote = (_design(env_car, ix, oth) for ix in (tr, va, te))

        r2_base, _ = _fit_eval(Btr, ytr, Bva, yva, Bte, yte)
        base_r2.append(r2_base)

        mu, comp = _pca_fit(Otr, K_SWEEP[-1])            # PCA on TRAIN clips only
        Ptr, Pva, Pte = ((O - mu) @ comp.T for O in (Otr, Ova, Ote))

        for k in K_SWEEP:
            r2_full, r2_tr = _fit_eval(
                np.column_stack([Btr, Ptr[:, :k]]), ytr,
                np.column_stack([Bva, Pva[:, :k]]), yva,
                np.column_stack([Bte, Pte[:, :k]]), yte,
            )
            out[f"K{k}"].append(r2_full - r2_base)
            if k == K_SWEEP[-1]:
                full_train_r2.append(r2_tr)

        # NULL: circular-shift ONLY the cross-shaft block. The within-shaft baseline and
        # the target stay aligned, so this isolates "how much dR2 does adding K USELESS
        # global predictors buy by chance/overfit" — which is exactly the question.
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
        "R2_within_shaft_baseline": round(float(np.mean(base_r2)), 5) if base_r2 else None,
        "R2_train_at_Kmax": round(float(np.mean(full_train_r2)), 5) if full_train_r2 else None,
        "dR2_by_K": {k: round(float(np.mean(v)), 5) for k, v in out.items() if v},
        "dR2_null_by_K": {k: round(float(np.mean(v)), 5) for k, v in out_null.items() if v},
        "dR2_null_p95_by_K": {
            k: round(float(np.percentile(v, 95)), 5) for k, v in out_null.items() if v
        },
    }


# ---------------------------------------------------------------------------
# M10b — multiplicative gating: does the global state predict local HGA MAGNITUDE?
# ---------------------------------------------------------------------------
def m10b(env_raw, env_car, tr, va, te, T, targets, rng, k_global=8) -> dict:
    hga_car = env_car[2]
    Gtr, Gva, Gte = (_design(env_raw, ix) for ix in (tr, va, te))   # ALL contacts, raw
    mu, comp = _pca_fit(Gtr, k_global)
    gtr, gva, gte = ((G - mu) @ comp.T for G in (Gtr, Gva, Gte))    # the global state g_t

    lin, gain, gain_null = [], [], []
    for c in targets:
        ytr, yva, yte = (_target(hga_car, ix, c) for ix in (tr, va, te))

        # 1. LEVEL: does g predict r linearly?  (M7 says no — this confirms in-sample.)
        r2_lin, _ = _fit_eval(gtr, ytr, gva, yva, gte, yte)
        lin.append(r2_lin)

        # 2. Kill the linear part FIRST. Without this, any residual linear g->r would
        #    leak into |r| and we would report gating that is really just level.
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
def _global_demean(env: list[np.ndarray]) -> list[np.ndarray]:
    """Remove the GRAND mean across ALL contacts at each t, per band. This is the GLOBAL
    common mode. Deliberately NOT shaft-CAR: a parcel often ~ a shaft segment, so
    shaft-CAR would zero its parcel mean BY CONSTRUCTION and fake a null."""
    return [e - e.mean(axis=1, keepdims=True) for e in env]


def m10cd(env_raw, parcel_id, tr, va, te, T, rng, n_splits=20) -> dict:
    env_g = _global_demean(env_raw)
    parcels = [p for p in np.unique(parcel_id)
               if int((parcel_id == p).sum()) >= MIN_PARCEL_CONTACTS]
    if len(parcels) < 3:
        return {"n_parcels": len(parcels), "skipped": "fewer than 3 usable parcels"}

    hga_g, hga_raw = env_g[2], env_raw[2]
    all_clips = np.concatenate([tr, va, te])

    # --- M10c(i) split-half reliability = the CEILING on any parcel-summary head -------
    rel_g, rel_raw = [], []
    for p in parcels:
        idx = np.where(parcel_id == p)[0]
        rg, rr = [], []
        for _ in range(n_splits):
            perm = rng.permutation(idx)
            a, b = perm[: len(perm) // 2], perm[len(perm) // 2 :]
            for src, acc in ((hga_g, rg), (hga_raw, rr)):
                h1 = src[all_clips][:, a].mean(1).reshape(-1)
                h2 = src[all_clips][:, b].mean(1).reshape(-1)
                if h1.std() > 1e-8 and h2.std() > 1e-8:
                    acc.append(float(np.corrcoef(h1, h2)[0, 1]))
        for acc, out in ((rg, rel_g), (rr, rel_raw)):
            if acc:
                r = float(np.mean(acc))
                out.append(2 * r / (1 + r) if r > -1 else 0.0)   # Spearman-Brown -> full

    # --- M10c(ii) inter-parcel predictability, vs that ceiling ------------------------
    def _psum(env_list, clips):
        cols = [np.stack([e[clips][:, parcel_id == p].mean(1) for p in parcels], 1)
                for e in env_list]                                # per band (n,P,T)
        return np.stack(cols, 2).transpose(0, 3, 1, 2).reshape(len(clips) * T, len(parcels) * 3)

    Str, Sva, Ste = (_psum(env_g, ix) for ix in (tr, va, te))
    ytr_all, yva_all, yte_all = (
        np.stack([hga_g[ix][:, parcel_id == p].mean(1).reshape(-1) for p in parcels], 1)
        for ix in (tr, va, te)
    )
    inter, inter_null = [], []
    for j in range(len(parcels)):
        keep = [c for c in range(Str.shape[1]) if c // 3 != j]    # drop parcel j's OWN 3 bands
        r2, _ = _fit_eval(Str[:, keep], ytr_all[:, j], Sva[:, keep], yva_all[:, j],
                          Ste[:, keep], yte_all[:, j])
        inter.append(r2)
        for _ in range(max(N_PERM // 4, 3)):
            r2n, _ = _fit_eval(
                _circshift_clipwise(Str[:, keep], len(tr), T, rng), ytr_all[:, j],
                _circshift_clipwise(Sva[:, keep], len(va), T, rng), yva_all[:, j],
                _circshift_clipwise(Ste[:, keep], len(te), T, rng), yte_all[:, j],
            )
            inter_null.append(r2n)

    # --- M10d eigenspectrum of the parcel-summary matrix (P x T) ----------------------
    def _spec(src):
        M = np.stack([src[all_clips][:, parcel_id == p].mean(1).reshape(-1) for p in parcels], 0)
        M = M - M.mean(1, keepdims=True)
        sd = np.maximum(M.std(1, keepdims=True), 1e-8)
        sv = np.linalg.svd(M / sd, compute_uv=False)
        e = sv**2
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
        "M10c_reliability_hga_global_demeaned": round(float(np.mean(rel_g)), 4) if rel_g else None,
        "M10c_reliability_hga_raw": round(float(np.mean(rel_raw)), 4) if rel_raw else None,
        "M10c_interparcel_R2": round(float(np.mean(inter)), 5) if inter else None,
        "M10c_interparcel_R2_null_p95": (
            round(float(np.percentile(inter_null, 95)), 5) if inter_null else None
        ),
        "M10d_spectrum_global_demeaned": _spec(hga_g),
        "M10d_spectrum_raw": _spec(hga_raw),
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
        env_car = [np.stack([_shaft_car(e[i], shaft_id) for i in range(e.shape[0])], 0)
                   for e in env_raw]

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
            "M10a": m10a(env_car, shaft_id, tr, va, te, T, targets, rng),
            "M10b": m10b(env_raw, env_car, tr, va, te, T, targets, rng),
            "M10cd": m10cd(env_raw, parcel_id, tr, va, te, T, rng),
        }
        rows.append(rec)

        A, B, C = rec["M10a"], rec["M10b"], rec["M10cd"]
        print(f"[s{sid}t{tid}] N={n_c} clips {len(tr)}/{len(va)}/{len(te)}", flush=True)
        print(f"    M10a  within-shaft R2 {A['R2_within_shaft_baseline']}  |  "
              f"dR2 {A['dR2_by_K']}", flush=True)
        print(f"          null           {A['dR2_null_by_K']}", flush=True)
        print(f"    M10b  level R2 {B['R2_level_linear']}  gain R2 {B['R2_gain']}  "
              f"(gain null p95 {B['R2_gain_null_p95']})", flush=True)
        print(f"    M10cd parcels {C.get('n_parcels')}  "
              f"reliability {C.get('M10c_reliability_hga_global_demeaned')} "
              f"(raw {C.get('M10c_reliability_hga_raw')})  "
              f"inter-parcel R2 {C.get('M10c_interparcel_R2')}  "
              f"spectrum {C.get('M10d_spectrum_global_demeaned')}", flush=True)

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
        pooled["M10a_within_shaft_R2"] = round(
            float(np.mean([r["M10a"]["R2_within_shaft_baseline"] for r in rows])), 5)
        for key, src in (("M10b_level_R2", "R2_level_linear"), ("M10b_gain_R2", "R2_gain"),
                         ("M10b_gain_null_p95", "R2_gain_null_p95")):
            pooled[key] = round(float(np.mean([r["M10b"][src] for r in rows])), 5)
        cd = [r["M10cd"] for r in rows if r["M10cd"].get("n_parcels", 0) >= 3]
        if cd:
            pooled["M10c_parcel_reliability"] = round(
                float(np.mean([c["M10c_reliability_hga_global_demeaned"] for c in cd])), 4)
            pooled["M10c_parcel_reliability_raw"] = round(
                float(np.mean([c["M10c_reliability_hga_raw"] for c in cd])), 4)
            pooled["M10c_interparcel_R2"] = round(
                float(np.mean([c["M10c_interparcel_R2"] for c in cd])), 5)
            pooled["M10d_var_top1"] = round(
                float(np.mean([c["M10d_spectrum_global_demeaned"]["var_top1"] for c in cd])), 4)
            pooled["M10d_var_top8"] = round(
                float(np.mean([c["M10d_spectrum_global_demeaned"]["var_top8"] for c in cd])), 4)
            pooled["M10d_participation_ratio"] = round(
                float(np.mean([c["M10d_spectrum_global_demeaned"]["participation_ratio"]
                               for c in cd])), 2)

        print("\nM10a — does a K-dim GLOBAL summary add anything to the within-shaft "
              "prediction of local HGA?")
        print(f"  within-shaft baseline R2 : {pooled['M10a_within_shaft_R2']}")
        for k in K_SWEEP:
            d = pooled["M10a_dR2_by_K"][f"K{k}"]
            n = pooled["M10a_dR2_null_p95_by_K"][f"K{k}"]
            verdict = "REAL" if d > n else "null"
            print(f"  K={k:>2}: dR2 {d:+.5f}   (null p95 {n:+.5f})   -> {verdict}")

        print(f"\nM10b — LEVEL vs GAIN (decides additive-vs-FiLM read)")
        print(f"  R2(global -> local HGA level) : {pooled['M10b_level_R2']:+.5f}")
        print(f"  R2(global -> local HGA |gain|): {pooled['M10b_gain_R2']:+.5f}  "
              f"(null p95 {pooled['M10b_gain_null_p95']:+.5f})")

        if cd:
            print(f"\nM10c/d — is a PARCEL summary a well-posed WRITE target?")
            print(f"  split-half reliability (global-demeaned): "
                  f"{pooled['M10c_parcel_reliability']}   <-- CEILING on any such head")
            print(f"  split-half reliability (raw, w/ common mode): "
                  f"{pooled['M10c_parcel_reliability_raw']}")
            print(f"  inter-parcel held-out R2                 : {pooled['M10c_interparcel_R2']}")
            print(f"  parcel-state spectrum: top1 {pooled['M10d_var_top1']}  "
                  f"top8 {pooled['M10d_var_top8']}  "
                  f"participation ratio {pooled['M10d_participation_ratio']}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"k_sweep": list(K_SWEEP), "n_perm": N_PERM,
                       "per_session": rows, "pooled": pooled}, fh, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
