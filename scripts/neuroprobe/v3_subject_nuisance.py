"""Subject-nuisance diagnostic on the existing v3 keep-time tap caches (delta-only, no encode).

Measurement A — is subject identity in the embedding?  Fit a 7-way subject classifier on the
clip embeddings and report BOTH balanced accuracy and the between-subject variance fraction
tr(S_B)/tr(S_T) (accuracy saturates; the variance fraction does not).

Measurement B — does it track the CS damage?  Emit one row per (tap, step) cell and per
(tap, step, test-subject), so subject-decodability can be regressed against the CS AUROC
already sitting in the result JSONs.  Nothing here recomputes a published CS/WS number.

Stage ``blocks`` is the single pass over the 640 GB of keep-time caches.  It writes the
BLOCK CACHE: per-frame electrode->parcel [mean_e, std_e], i.e. exactly the 512-d unit the
keep-time CS ridge concatenates over (parcel, frame) --- but with T KEPT, so the same artifact
also feeds the erasure intervention (v3_subject_erasure.py).  Bit-faithful to
``v3_probe_readout_keeptime._rows_fp32`` (per-token LN over d) then ``_pool_rows``.

  python -u v3_subject_nuisance.py --stage blocks --cell enc3:30k
  python -u v3_subject_nuisance.py --stage ab --cell enc3:30k --out sn_enc3_30k.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

N_PARCELS = 75
SESSIONS = [(1, 0), (2, 1), (3, 2), (4, 2), (6, 0), (8, 0), (9, 0)]
ANCHOR = (2, 1)
TEST_SUBJECTS = [1, 3, 4, 6, 8, 9]
TASKS = ["onset", "delta_volume", "word_index", "gpt2_surprisal"]

CACHE_ROOT = "/projects/bhqk/htang13"
# (tap, tag) -> keep-time cache dir.  enc12@40k is absent: its anchor file is a truncated stub.
CELLS = {
    ("enc12", "10k"): f"{CACHE_ROOT}/v3_probe_cache_keeptime",
    ("enc12", "20k"): f"{CACHE_ROOT}/v3_probe_cache_keeptime",
    ("enc12", "30k"): f"{CACHE_ROOT}/v3_probe_cache_keeptime",
    ("enc3", "30k"): f"{CACHE_ROOT}/v3_probe_cache_keeptime_b3",
    ("enc3", "40k"): f"{CACHE_ROOT}/v3_probe_cache_keeptime_b3",
    ("enc6", "30k"): f"{CACHE_ROOT}/v3_probe_cache_keeptime_b6",
    ("enc6", "40k"): f"{CACHE_ROOT}/v3_probe_cache_keeptime_b6",
}
BLOCK_DIR = f"{CACHE_ROOT}/v3_block_cache"


def block_path(tap: str, tag: str, session, variant: str) -> str:
    s, t = session
    return f"{BLOCK_DIR}/blk_{tap}_{tag}_s{s}_t{t}_{variant}.npz"


# --------------------------------------------------------------- stage: blocks

def _blocks_for_session(src: str, variant: str, chunk: int):
    """(n,N,T,d) fp16 keep-time tap -> (n,P,T,2d) fp16 per-frame [mean_e, std_e] per parcel.

    LN is applied per (row, electrode, frame) token over d BEFORE the spatial pool, matching
    ``_rows_fp32`` -> ``_pool_rows``.  P = the parcels this session actually has; any pair's
    intersection is then a column subset, so one pass serves every CS cell."""
    rec = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    enc = rec["enc12_kt"]                                     # (n, N, T, d) fp16
    pid = np.asarray(rec["parcel_id"], dtype=np.int64)        # (N,)
    n, _N, T, d = enc.shape
    parcels = np.unique(pid)
    cols = [np.where(pid == int(p))[0] for p in parcels]

    out = np.empty((n, len(parcels), T, 2 * d), dtype=np.float16)
    for i in range(0, n, chunk):
        x = enc[i : i + chunk].float()                        # (r, N, T, d)
        if variant == "ln":
            x = F.layer_norm(x, (d,))
        for j, c in enumerate(cols):
            sub = x[:, c, :, :]                               # (r, |c|, T, d)
            # std over electrodes: unbiased=False so a 1-electrode parcel gives 0, not NaN
            pooled = torch.cat([sub.mean(1), sub.std(1, unbiased=False)], dim=-1)
            out[i : i + chunk, j] = pooled.numpy().astype(np.float16)
    meta: dict[str, np.ndarray] = {
        f"y_{t}": np.asarray(rec["labels"][t], dtype=np.float64)
        for t in TASKS if t in rec["labels"]}
    meta.update({f"cs_{t}": np.asarray(rec["cs_split"][t]["test"], dtype=np.int64)
                 for t in TASKS if t in rec.get("cs_split", {})})
    return out, parcels, meta


def stage_blocks(cells, variant: str, chunk: int) -> None:
    os.makedirs(BLOCK_DIR, exist_ok=True)
    for tap, tag in cells:
        for sess in SESSIONS:
            dst = block_path(tap, tag, sess, variant)
            if os.path.exists(dst):
                print(f"[skip] {dst}", flush=True)
                continue
            s, t = sess
            src = f"{CELLS[(tap, tag)]}/enc_s{s}_t{t}_{tag}.pt"
            blk, parcels, meta = _blocks_for_session(src, variant, chunk)
            tmp = dst[:-4] + ".tmp.npz"       # savez appends .npz unless the name already ends in it
            np.savez(tmp, blocks=blk, parcels=parcels, **meta)
            os.replace(tmp, dst)
            print(f"[blocks] {tap} {tag} s{s}t{t} {blk.shape} P={len(parcels)} -> {dst}",
                  flush=True)
            del blk


# ------------------------------------------------------------------- stage: ab

def _marginal_embed(blk: np.ndarray) -> np.ndarray:
    """Parcel-MARGINALIZED clip embedding: pool over parcels, then over T -> (n, 2048).

    NO DKT parcel is present in all seven subjects (the 7-way intersection is empty; the CS ridge
    runs on 3--7 shared parcels per anchor/test pair), so a 7-way classifier in the ridge's own
    per-parcel space does not exist -- there is no common support to fit it on.  Marginalizing the
    parcel axis gives a subject-comparable embedding at the cost of leaving the ridge's exact
    space.  This is the SECONDARY number; the pairwise anchor-vs-S contrast below is the primary
    one, because it is fit in exactly the space its CS cell uses."""
    x = blk.astype(np.float32)                                # (n, P, T, 512)
    x = np.concatenate([x.mean(1), x.std(1)], axis=-1)        # (n, T, 1024) pool parcels
    return np.concatenate([x.mean(1), x.std(1)], axis=-1)     # (n, 2048)   pool time


def _clip_embed(blk: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Block cache -> pooled clip embedding: collapse T with [mean_t, std_t].

    (n, P, T, 512)[:, keep] -> (n, |keep| * 1024).  This is the temporally-pooled version of
    the ridge's OWN features (the ridge instead flattens T), so it is the closest thing to
    'the clip embedding the CS readout sees'."""
    x = blk[:, keep].astype(np.float32)                       # (n, |keep|, T, 512)
    e = np.concatenate([x.mean(2), x.std(2)], axis=-1)        # (n, |keep|, 1024)
    return e.reshape(len(e), -1)


def _variance_fraction(X: np.ndarray, g: np.ndarray) -> dict:
    """tr(S_B)/tr(S_T) — share of embedding variance carried by the group means.

    Trace-only (never forms the D x D scatter).  Also returns the S_B eigen-spectrum via the
    small G x G Gram, which says how many directions the nuisance actually occupies."""
    mu = X.mean(0)
    tr_T = float(((X - mu) ** 2).sum() / len(X))
    M, w = [], []
    for k in np.unique(g):
        Xk = X[g == k]
        M.append(Xk.mean(0) - mu)
        w.append(len(Xk) / len(X))
    M, w = np.stack(M), np.asarray(w)
    tr_B = float((w[:, None] * M**2).sum())
    Mw = M * np.sqrt(w)[:, None]
    ev = np.clip(np.linalg.eigvalsh(Mw @ Mw.T)[::-1], 0, None)
    return {"eta2_between": tr_B / tr_T if tr_T > 0 else float("nan"),
            "tr_between": tr_B, "tr_total": tr_T,
            "sb_eig_frac": [float(v / ev.sum()) for v in ev] if ev.sum() > 0 else []}


def _classify(X: np.ndarray, g: np.ndarray, *, seed: int, n_pca: int = 200) -> dict:
    """Linear subject classifier: standardize -> PCA -> multinomial logistic, 5-fold on clips."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    n_pca = min(n_pca, X.shape[1], len(X) - 1)
    Z = PCA(n_components=n_pca, random_state=seed).fit_transform(
        StandardScaler().fit_transform(X.astype(np.float64)))
    accs, aucs = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Z, g):
        clf = LogisticRegression(max_iter=2000).fit(Z[tr], g[tr])
        accs.append(balanced_accuracy_score(g[te], clf.predict(Z[te])))
        if len(np.unique(g)) == 2:
            aucs.append(roc_auc_score(g[te], clf.predict_proba(Z[te])[:, 1]))
    out = {"bal_acc": float(np.mean(accs)), "bal_acc_sd": float(np.std(accs)),
           "chance": 1.0 / len(np.unique(g))}
    if aucs:
        out["auroc"] = float(np.mean(aucs))
    return out


def stage_ab(cells, variant: str, *, seed: int, max_clips: int, out_path: str) -> None:
    results = {}
    for tap, tag in cells:
        recs = {}
        for sess in SESSIONS:
            z = np.load(block_path(tap, tag, sess, variant))
            recs[sess] = {"blocks": z["blocks"], "parcels": z["parcels"]}
        rng = np.random.default_rng(seed)

        # --- 7-way. The parcel set shared by all seven subjects is EMPTY (verified: only
        # parcels 7/13/28 reach even 6 subjects), so this is fit on the parcel-marginalized
        # embedding, NOT the ridge's per-parcel space. See _marginal_embed.
        inter7 = set(recs[SESSIONS[0]]["parcels"].tolist())
        for r in recs.values():
            inter7 &= set(r["parcels"].tolist())
        inter7 = np.array(sorted(inter7), dtype=np.int64)

        n_bal = min(min(len(r["blocks"]) for r in recs.values()), max_clips)
        Xs, gs = [], []
        for sess, r in recs.items():
            idx = np.sort(rng.choice(len(r["blocks"]), size=n_bal, replace=False))
            Xs.append(_marginal_embed(r["blocks"][idx]))
            gs.append(np.full(n_bal, sess[0]))
        X7, g7 = np.concatenate(Xs), np.concatenate(gs)
        del Xs

        cell = {"tap": tap, "step": tag, "variant": variant,
                "n_parcels_inter7": int(len(inter7)), "sevenway_space": "parcel_marginalized",
                "n_clips_per_subject": int(n_bal), "dim7": int(X7.shape[1]),
                "sevenway": {**_variance_fraction(X7, g7), **_classify(X7, g7, seed=seed)},
                "pairwise": {}}
        sw = cell["sevenway"]
        print(f"[{tap} {tag}] 7-way (parcel-marginalized; inter7={len(inter7)} parcels) "
              f"dim={X7.shape[1]} n/subj={n_bal} | bal_acc={sw['bal_acc']:.4f} "
              f"(chance {1/7:.4f})  eta2_between={sw['eta2_between']:.4f}", flush=True)
        del X7

        # --- pairwise anchor-vs-test, each in the SAME parcel intersection its CS cell uses
        a = recs[ANCHOR]
        for s in TEST_SUBJECTS:
            sess = next(x for x in SESSIONS if x[0] == s)
            r = recs[sess]
            inter = np.array(sorted(set(a["parcels"].tolist()) & set(r["parcels"].tolist())),
                             dtype=np.int64)
            nb = min(len(a["blocks"]), len(r["blocks"]), max_clips)
            ia = np.sort(rng.choice(len(a["blocks"]), size=nb, replace=False))
            ib = np.sort(rng.choice(len(r["blocks"]), size=nb, replace=False))
            Xp = np.concatenate([
                _clip_embed(a["blocks"][ia], np.searchsorted(a["parcels"], inter)),
                _clip_embed(r["blocks"][ib], np.searchsorted(r["parcels"], inter))])
            gp = np.concatenate([np.zeros(nb, dtype=int), np.ones(nb, dtype=int)])
            cell["pairwise"][f"S{s}"] = {
                "n_parcels_inter": int(len(inter)), "dim": int(Xp.shape[1]),
                **_variance_fraction(Xp, gp), **_classify(Xp, gp, seed=seed)}
            pw = cell["pairwise"][f"S{s}"]
            print(f"    anchor-vs-S{s}: P_inter={len(inter):2d} "
                  f"auroc={pw['auroc']:.4f} eta2={pw['eta2_between']:.4f}", flush=True)
            del Xp
        results[f"{tap}|{tag}"] = cell
        del recs

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["blocks", "ab"], required=True)
    p.add_argument("--cell", action="append", default=None,
                   help="tap:tag, e.g. enc3:30k; repeatable. Default = all seven.")
    p.add_argument("--variant", default="ln", choices=["ln", "raw"])
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--max-clips", type=int, default=3000)
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--out", default=f"{CACHE_ROOT}/v3_subject_nuisance.json")
    a = p.parse_args()

    cells = [tuple(c.split(":")) for c in a.cell] if a.cell else list(CELLS)
    for c in cells:
        if c not in CELLS:
            raise SystemExit(f"unknown cell {c}; known: {sorted(CELLS)}")

    if a.stage == "blocks":
        stage_blocks(cells, a.variant, a.chunk)
    else:
        stage_ab(cells, a.variant, seed=a.seed, max_clips=a.max_clips, out_path=a.out)


if __name__ == "__main__":
    main()
