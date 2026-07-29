"""Per-electrode single-trial AUROC with a permutation null, one shard per session.

``viz_anatomy`` (measure #1) answers Greg's question at PARCEL resolution off trial-averaged
condition means. This is the companion at CONTACT resolution off single trials, and it adds
the one thing a condition mean cannot give: a calibrated null. Where #1's colour axis has a
true zero by construction, here "is this contact real?" is answered by shuffling the labels
and re-running the entire fit, so the null absorbs every quirk of the pipeline rather than
resting on a distributional assumption.

THE STATISTIC, and why each piece is there
------------------------------------------
For one contact and one time bin the feature is a C-vector (enc0: the HGA band's frequency
bins; enc12: 256 latent dims). AUROC needs a scalar, so the C axis has to collapse. Three
ways to do it, two of them wrong here:

  * mean over C -- meaningful for enc0 (it is power) and meaningless for enc12, whose dims
    carry arbitrary signs. Not comparable across taps, which is the whole point of a tap axis.
  * norm over C -- label-free and comparable, but discards direction, so a contact whose two
    conditions differ in PATTERN at equal energy reads as zero.
  * a class-difference projection -- keeps direction and is comparable across taps, but is
    fit with the labels, so it must be cross-validated or the AUROC is circular.

The third, cross-validated. The trials split into two interleaved halves (``viz_reduce``'s
convention: session drift is slow, so alternating trials splits it evenly between halves
where a random split would leave the answer seed-dependent). The projection direction is the
class-mean difference on one half; the AUROC is computed on the OTHER half's trials. Both
folds run and the two AUROCs average -- they score disjoint trials, so the average is over
independent estimates rather than a reuse of the same data.

WHAT MAKES THE NULL EXACT
-------------------------
The labels are permuted and the WHOLE procedure re-runs -- PC basis reuse aside, the
direction is refit on the permuted labels every time. So the null inherits the fitting bias,
the fold structure, the trial counts and the feature covariance; nothing has to be assumed
about them. A null built by permuting only at the scoring step would be anti-conservative,
because it would leave the direction fit on the true labels.

Permutation is BLOCKED in trial order by default (``--perm-block``). Free permutation
assumes trials are exchangeable, and for this task menu some are not: ``word_index`` and
``word_head_pos`` rise within a session by construction, so any contact with slow drift
would beat a free-permutation null on drift alone. Permuting inside contiguous blocks keeps
the slow trend in both the real and permuted data, which is the standard fix. A block that
happens to be single-class contributes no shuffling, which narrows the null -- conservative,
not anti-conservative, so the failure direction is the safe one.

The C axis is first reduced to ``--pc`` label-free principal components, pooled over trials
and time, per contact. This is a cost decision (enc12 is C=256, and the null re-runs the fit
a few hundred times) and it cannot leak, because no label is consulted to build the basis.
Retained variance is printed per tap and stored in the shard so the trade is auditable
rather than assumed.

DELIBERATELY NOT DONE
---------------------
No cluster-corrected inference over space. Contacts are not on a regular lattice and the
montages differ per subject, so a spatial cluster statistic would need a neighbourhood
definition this data does not support. Multiplicity over (contact x time) is handled by the
max-statistic instead, which needs no neighbourhood -- see ``maxstat_threshold`` for why it,
and not BH-FDR, is the primary correction at a few hundred permutations.

No causal claim. An AUROC map says where the information IS, not where it is USED; only
occlusion could say the latter. #1 carries the same caveat and for the same reason.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


# --------------------------------------------------------------------------------------
# AUROC
# --------------------------------------------------------------------------------------
def auroc_cols(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC of every column of ``scores`` (n, M) against binary ``y`` (n,).

    Mann-Whitney U on ranks. Columns with no variance are returned as exactly 0.5: a dead
    contact projects to an all-equal score, and leaving the rank order to fall out of the
    trial INDEX would then read the label's own drift as signal -- the precise artefact the
    blocked permutation exists to prevent, so it must not be reintroduced here. That guard is
    also why the sort need not be stable, which matters: the sort is the inner loop of the
    permutation null and an unstable sort is the faster one.
    """
    n = scores.shape[0]
    assert y.shape == (n,), (scores.shape, y.shape)
    n1 = int(y.sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return np.full(scores.shape[1], 0.5)
    order = np.argsort(scores, axis=0)
    ranks = np.empty(scores.shape, dtype=np.float64)
    rr = np.arange(1, n + 1, dtype=np.float64)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rr, scores.shape), axis=0)
    r1 = ranks[y == 1].sum(axis=0)
    au = (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    flat = scores.max(axis=0) <= scores.min(axis=0)
    au[flat] = 0.5
    return au


def _halves(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interleaved halves of trial POSITIONS, interleaved WITHIN each class.

    ``viz_reduce._halves`` interleaves each class's trial list separately, and that detail is
    load-bearing rather than incidental. Interleaving the pooled list instead would put every
    even-position trial in one fold, so a label that happens to alternate lands one class
    entirely in each fold and the direction cannot be fit at all. Splitting within class
    keeps the drift-halving property AND guarantees both classes in both folds.
    """
    h0, h1 = [], []
    for cls in (0, 1):
        ix = np.where(y == cls)[0]
        h0.append(ix[0::2])
        h1.append(ix[1::2])
    return np.sort(np.concatenate(h0)), np.sort(np.concatenate(h1))


def cv_auroc(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Two-fold cross-validated AUROC per (row, time) cell. ``Z`` is (n, P, T, K).

    Direction from the fit half's class-mean difference, AUROC on the held-out half, both
    folds averaged. Returns (P, T).
    """
    _, P, T, _ = Z.shape
    h0, h1 = _halves(y)
    out = np.zeros((P, T), dtype=np.float64)
    for fit, ev in ((h0, h1), (h1, h0)):
        yf, ye = y[fit], y[ev]
        if yf.sum() == 0 or yf.sum() == len(yf) or ye.sum() == 0 or ye.sum() == len(ye):
            out += 0.5
            continue
        w = Z[fit][yf == 1].mean(axis=0) - Z[fit][yf == 0].mean(axis=0)   # (P, T, K)
        s = np.einsum("nptk,ptk->npt", Z[ev], w, optimize=True)           # (n_ev, P, T)
        out += auroc_cols(s.reshape(len(ev), P * T), ye).reshape(P, T)
    return out / 2.0


def block_permute(y: np.ndarray, rng: np.random.Generator, block: int) -> np.ndarray:
    """Shuffle labels inside contiguous blocks of trial order (block<=0 => free shuffle)."""
    if block <= 0 or block >= len(y):
        return rng.permutation(y)
    out = y.copy()
    for i in range(0, len(y), block):
        j = min(i + block, len(y))
        out[i:j] = rng.permutation(out[i:j])
    return out


# --------------------------------------------------------------------------------------
# the label-free channel reduction
# --------------------------------------------------------------------------------------
def pca_basis(cov: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` eigenvectors of each (C, C) covariance in ``cov`` (P, C, C).

    Returns ``(basis (P, C, k), retained fraction (P,))``. Symmetric eigendecomposition, so
    ``eigh`` -- ordering is ascending, hence the reversed slice.
    """
    k = min(k, cov.shape[1])
    ev, vec = np.linalg.eigh(cov)
    ev = np.clip(ev[:, ::-1], 0.0, None)
    vec = vec[:, :, ::-1]
    tot = ev.sum(axis=1)
    frac = np.where(tot > 0, ev[:, :k].sum(axis=1) / np.maximum(tot, 1e-30), 1.0)
    return np.ascontiguousarray(vec[:, :, :k]), frac


# --------------------------------------------------------------------------------------
# one session
# --------------------------------------------------------------------------------------
def read_session(path: str, *, tap: str, band: str):
    """``(rec, feat, cols, T, C)`` for one tap, mmapped -- nothing large is materialised."""
    import torch

    from scripts.neuroprobe.viz_reduce import _band_slice

    rec = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if tap not in rec["feats"]:
        return rec, None, None, 0, 0
    feat = rec["feats"][tap]["raw"]
    band_lengths = tuple(int(x) for x in rec["band_lengths"])
    band_fdims = rec.get("band_fdims")
    cols, T, C = _band_slice(tap, band, band_lengths, band_fdims, int(feat.shape[2]))
    canon = np.asarray(rec["parcel_canon"], dtype=np.int64)
    n_rows = len(canon) if tap.endswith("_elec") else len(np.asarray(rec["present_parcels"]))
    assert int(feat.shape[1]) == n_rows, f"{tap}: {feat.shape[1]} rows != {n_rows}"
    return rec, feat, cols, T, C


def _chunks(feat, cols, sel: np.ndarray, T: int, C: int, chunk: int):
    """Yield ``(offset, (m, P, T, C) float32)`` blocks of the selected trials.

    A generator rather than one array on purpose: at enc12 the full selection is
    n x P x T x 256 floats -- ~26 GB for a big session -- and the only thing downstream wants
    is its projection into K<<C dims. Materialising the whole thing first would set the job's
    memory bill (Delta CPU bills on memory, not cores) at 8x what the work needs.

    Both layouts put one band in a contiguous column run, so this slices rather than
    fancy-indexes: the other bands' pages then never come off disk at all.
    """
    import torch

    contiguous = len(cols) > 0 and cols[-1] - cols[0] == len(cols) - 1
    P = int(feat.shape[1])
    for i0 in range(0, len(sel), chunk):
        ix = sel[i0:i0 + chunk]
        if contiguous:
            x = feat[ix, :, int(cols[0]):int(cols[-1]) + 1].to(torch.float32).numpy()
        else:
            x = feat[ix, :, :].to(torch.float32).numpy()[:, :, cols]
        yield i0, x.reshape(len(ix), P, T, C)


def project(feat, cols, sel, T, C, chunk, basis) -> np.ndarray:
    """(len(sel), P, T, K) float32, streamed so the (…, C) array never exists in full."""
    P, K = int(feat.shape[1]), basis.shape[2]
    Z = np.empty((len(sel), P, T, K), dtype=np.float32)
    b32 = basis.astype(np.float32)
    for i0, x in _chunks(feat, cols, sel, T, C, chunk):
        Z[i0:i0 + x.shape[0]] = np.einsum("nptc,pck->nptk", x, b32, optimize=True)
    return Z


def session_shard(path: str, *, taps, tasks, band: str, n_pc: int, n_perm: int,
                  perm_block: int, chunk: int, seed: int, store_null: bool = True,
                  cov_stride: int = 4, verbose: bool = True) -> dict:
    out: dict = {}
    meta_done = False
    for tap in taps:
        rec, feat, cols, T, C = read_session(path, tap=tap, band=band)
        if not meta_done:
            out["subject_id"] = np.int64(int(rec["subject_id"]))
            out["trial_id"] = np.int64(int(rec["trial_id"]))
            out["parcel_canon"] = np.asarray(rec["parcel_canon"], dtype=np.int64)
            out["n_windows"] = np.int64(int(rec["n_windows"]))
            out["perm_block"] = np.int64(perm_block)
            out["n_perm"] = np.int64(n_perm)
            meta_done = True
        if feat is None:
            print(f"[skip] {tap} not in record", flush=True)
            continue
        P = int(feat.shape[1])
        out[f"T/{tap}"] = np.int64(T)
        out[f"C/{tap}"] = np.int64(C)

        # label-free PC basis for this tap: one covariance per row, pooled over trials & time
        cov = np.zeros((P, C, C), dtype=np.float64)
        csum = np.zeros((P, C), dtype=np.float64)
        # Every ``cov_stride``-th window. A (256, 256) covariance from a whole session is
        # estimated from n_windows*T ~ 870k samples, which is four orders of magnitude more
        # than its 33k free parameters, so striding costs nothing measurable and this pass
        # otherwise reads the entire 40+ GB tap off Lustre.
        wins = np.arange(0, int(rec["n_windows"]), max(1, cov_stride))
        for _, x in _chunks(feat, cols, wins, T, C, chunk):
            xf = x.reshape(-1, P, C).transpose(1, 0, 2).astype(np.float64)   # (P, m, C)
            cov += xf.transpose(0, 2, 1) @ xf
            csum += xf.sum(axis=1)
            del x, xf
        m_tot = len(wins) * T
        assert m_tot > 20 * C, f"{tap}: {m_tot} samples for a {C}x{C} covariance is too few"
        mu = csum / m_tot
        cov = cov / m_tot - mu[:, :, None] * mu[:, None, :]
        basis, frac = pca_basis(cov, n_pc)
        K = basis.shape[2]
        out[f"pc_var/{tap}"] = frac.astype(np.float32)
        if verbose:
            print(f"[pc] {tap}: C={C} -> K={K}, retained var "
                  f"min={frac.min():.3f} med={np.median(frac):.3f}", flush=True)

        for task in tasks:
            y_all = np.asarray(rec["labels"][task], dtype=float)
            sel = np.where(np.isfinite(y_all) & np.isin(y_all, (0.0, 1.0)))[0]
            sel = np.sort(sel)
            y = y_all[sel].astype(np.int64)
            if len(sel) < 20 or y.sum() == 0 or y.sum() == len(y):
                print(f"[skip] {tap} {task}: n={len(sel)} n1={int(y.sum())}", flush=True)
                continue
            t0 = time.time()
            Z = project(feat, cols, sel, T, C, chunk, basis)                 # (n, P, T, K)
            # standardize per (row, time, pc) column. Label-free, so it cannot inflate the
            # AUROC; it is here because the projection direction is a difference of class
            # MEANS, which one high-variance PC would otherwise dominate outright.
            Z -= Z.mean(axis=0, keepdims=True)
            Z /= np.maximum(Z.std(axis=0, keepdims=True), 1e-8)
            obs = cv_auroc(Z, y)
            rng = np.random.default_rng(seed + abs(hash((tap, task))) % 100000)
            null = np.empty((n_perm, P, T), dtype=np.float32)
            for p in range(n_perm):
                null[p] = cv_auroc(Z, block_permute(y, rng, perm_block))
            del Z
            # The WHOLE null cube is stored, not a summary. It is ~3 MB per (tap, task) and it
            # is what lets the multiplicity correction be chosen -- and changed -- downstream
            # without re-running the array. A stored null_mean/null_sd pair would silently
            # commit the analysis to a Gaussian approximation of a rank statistic's null.
            out[f"auroc/{tap}/{task}"] = obs.astype(np.float32)
            # the per-permutation MAP MAXIMUM is all the FWER threshold needs and costs 2 kB,
            # so it is stored separately from the cube -- the aggregator then works even on a
            # shard written with --no-store-null
            out[f"null_max/{tap}/{task}"] = np.abs(null - 0.5).max(axis=(1, 2))
            if store_null:
                out[f"null/{tap}/{task}"] = null
            out[f"n/{task}"] = np.int64(len(sel))
            out[f"n1/{task}"] = np.int64(int(y.sum()))
            if verbose:
                thr = float(np.quantile(np.abs(null - 0.5).max(axis=(1, 2)), 0.95))
                print(f"[auroc] {tap} {task}: n={len(sel)} P={P} T={T} "
                      f"max={obs.max():.3f} fwer_thr={0.5 + thr:.3f} "
                      f"n_cells_fwer={(np.abs(obs - .5) > thr).sum()} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return out


# --------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------
def maxstat_threshold(null: np.ndarray, alpha: float = 0.05) -> float:
    """FWER-controlling threshold on ``|AUROC - .5|``.

    Takes either a null cube (n_perm, P, T) or the per-permutation maxima (n_perm,) already
    reduced out of one.

    Nichols & Holmes 2002. Each permutation contributes ONE number -- the largest deviation
    anywhere in the map -- and the (1-alpha) quantile of those maxima is the threshold. Any
    observed cell above it is significant with family-wise error alpha over the whole
    (contact x time) map.

    This, and not BH, is the primary correction here, for a resolution reason that is easy to
    miss: a permutation p-value cannot go below 1/(n_perm+1), so with a few hundred
    permutations and a few thousand cells, BH's smallest thresholds (q*k/m) sit BELOW the p
    floor and an isolated real contact can never be rejected no matter how strong it is. The
    max-statistic needs no per-cell p resolution at all, because the multiplicity is handled
    by the maximum rather than by counting.
    """
    if null.ndim == 1:
        maxima = null                       # already |AUROC-.5| maxima, not AUROC
    else:
        assert null.ndim == 3, null.shape
        maxima = np.abs(null - 0.5).max(axis=(1, 2))
    return float(np.quantile(maxima, 1.0 - alpha))


def pooled_p(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Per-cell p from the null POOLED over cells -- the finer-resolution, liberal companion.

    Trades an exchangeability-across-cells assumption (every cell shares the same null, which
    holds only approximately: cells share n, K and the fold structure but not their feature
    covariance) for a p floor of 1/(n_perm*P*T+1). Use for a graded map; use
    ``maxstat_threshold`` for any claim that a particular contact is real.
    """
    dev_o = np.abs(obs - 0.5)
    pool = np.sort(np.abs(null - 0.5).ravel())
    ge = len(pool) - np.searchsorted(pool, dev_o.ravel(), side="left")
    return ((1.0 + ge) / (1.0 + len(pool))).reshape(obs.shape)


def bh_fdr(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg mask over a flat p-vector."""
    flat = p.ravel()
    m = len(flat)
    order = np.argsort(flat)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = flat[order] <= thresh
    k = np.where(passed)[0]
    mask = np.zeros(m, dtype=bool)
    if len(k):
        mask[order[:k[-1] + 1]] = True
    return mask.reshape(p.shape)


def aggregate(shard_dir: str, *, taps, tasks, alpha: float = 0.05,
              inference: str = "maxstat") -> dict:
    """Per-session shards -> per (session, tap, task, DKT base) contact-level summaries.

    ``inference`` picks the multiplicity correction. ``maxstat`` is FWER over the whole
    (contact x time) map and is what any "this contact is real" claim should use. ``fdr`` is
    BH at q=alpha on pooled-null p-values -- more sensitive, but it buys that sensitivity with
    an exchangeability-across-cells assumption AND it controls a false-DISCOVERY rate, so a
    map drawn from it will contain expected false positives by design. It needs the stored
    null cube; the maxima alone are not enough.
    """
    from scripts.neuroprobe.viz_anatomy import dkt_tables

    base_of, lobe_of_base = dkt_tables()
    files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".npz"))
    assert files, f"no shards in {shard_dir}"
    A: dict = {}
    cov: dict = {}
    for fn in files:
        z = np.load(os.path.join(shard_dir, fn), allow_pickle=False)
        sub = int(z["subject_id"])
        sess = f"s{sub}_t{int(z['trial_id'])}"
        canon = z["parcel_canon"]
        bases = np.array([base_of.get(int(i), "unknown") for i in canon])
        cov[sess] = (sub, bases)
        for tap in taps:
            for task in tasks:
                key = f"auroc/{tap}/{task}"
                if key not in z.files:
                    continue
                au = z[key]
                peak = np.abs(au - 0.5).max(axis=1)
                if inference == "fdr":
                    cube = f"null/{tap}/{task}"
                    assert cube in z.files, (
                        f"{fn}: --inference fdr needs the full null cube, which this shard was "
                        "written without (--no-store-null). Use maxstat or re-run.")
                    thr = float("nan")
                    sig = bh_fdr(pooled_p(au, z[cube]), alpha).any(axis=1)
                else:
                    nkey = next((k for k in (f"null_max/{tap}/{task}", f"null/{tap}/{task}")
                                 if k in z.files), None)
                    assert nkey, f"{fn}: {tap}/{task} has a map but no null to judge it by"
                    thr = maxstat_threshold(z[nkey], alpha)
                    sig = peak > thr
                A[(sess, tap, task)] = {
                    "peak": peak, "sig": sig, "fwer_thr": thr,
                    "auroc_at_peak": au[np.arange(au.shape[0]), np.abs(au - .5).argmax(axis=1)],
                    "t_at_peak": np.abs(au - .5).argmax(axis=1),
                    "T": au.shape[1],
                }
    return {"A": A, "cov": cov, "lobe_of_base": lobe_of_base}


def region_table(agg: dict, tap: str, task: str) -> list[tuple[str, float, int, int, int]]:
    """``(base, mean peak |AUROC-.5| over contacts, n_sig, n_contacts, n_subjects)`` desc."""
    A, cov = agg["A"], agg["cov"]
    per: dict[str, list] = {}
    subs: dict[str, set] = {}
    for (sess, tp, tk), d in A.items():
        if tp != tap or tk != task:
            continue
        sub, bases = cov[sess]
        for b in np.unique(bases):
            m = bases == b
            per.setdefault(b, []).append((d["peak"][m], d["sig"][m]))
            subs.setdefault(b, set()).add(sub)
    rows = []
    for b, chunks in per.items():
        peak = np.concatenate([c[0] for c in chunks])
        sig = np.concatenate([c[1] for c in chunks])
        rows.append((b, float(peak.mean()), int(sig.sum()), int(len(peak)), len(subs[b])))
    rows.sort(key=lambda r: -r[1])
    return rows


def sig_fraction(agg: dict, tap: str, task: str, base: str) -> tuple[int, int, int]:
    """``(n_sig, n_contacts, n_sessions_with_any)`` for one region, pooled over sessions."""
    ns = nc = nsess = 0
    for (sess, tp, tk), d in agg["A"].items():
        if tp != tap or tk != task:
            continue
        m = agg["cov"][sess][1] == base
        if not m.any():
            continue
        ns += int(d["sig"][m].sum())
        nc += int(m.sum())
        nsess += int(d["sig"][m].any())
    return ns, nc, nsess


def report(agg: dict, *, taps, tasks, alpha: float = 0.05, top: int = 6) -> dict:
    """Print the contact-level answer to Greg's question, and the falsifier beside it."""
    from scripts.neuroprobe.viz_anatomy import EVENT, LEVEL, TARGET, VISUAL_TASKS

    A = agg["A"]
    sessions = sorted({s for (s, _, _) in A})
    print(f"\nsessions={len(sessions)}  alpha={alpha} (max-statistic FWER over contact x time)")
    out: dict = {}
    for tap in taps:
        thr = [d["fwer_thr"] for (_, tp, _), d in A.items() if tp == tap]
        if not thr:
            continue
        print(f"\n################ {tap}  (FWER threshold on |AUROC-.5|: "
              f"median {np.median(thr):.3f}, range {min(thr):.3f}-{max(thr):.3f})")
        for task in tasks:
            rows = region_table(agg, tap, task)
            if not rows:
                continue
            names = [r[0] for r in rows]
            rank = names.index(TARGET) + 1 if TARGET in names else -1
            ns, nc, nsess = sig_fraction(agg, tap, task, TARGET)
            out[(tap, task)] = {"st_rank": rank, "st_sig": ns, "st_n": nc, "st_sess": nsess}
            head = "  ".join(f"{b}({mp:.3f},{s}/{n})" for b, mp, s, n, _ in rows[:top])
            print(f"  {task:18s} STG rank {rank:>3}  sig {ns:>3}/{nc:<3} in {nsess} sess | {head}")

        # Greg's test and its falsifier, side by side, at contact resolution
        ev = [out[(tap, t)] for t in EVENT if (tap, t) in out]
        lv = [out[(tap, t)] for t in LEVEL if (tap, t) in out]
        vis = [out[(tap, t)] for t in VISUAL_TASKS if (tap, t) in out]
        def frac(rs):
            n = sum(r["st_n"] for r in rs)
            return (sum(r["st_sig"] for r in rs) / n) if n else float("nan")
        print(f"  --> STG top-3 in {sum(1 for r in ev if 0 < r['st_rank'] <= 3)}/{len(ev)} event"
              f" tasks, {sum(1 for r in lv if 0 < r['st_rank'] <= 3)}/{len(lv)} level,"
              f" {sum(1 for r in vis if 0 < r['st_rank'] <= 3)}/{len(vis)} visual")
        print(f"  --> STG contacts surviving FWER: event {frac(ev):.3f}, level {frac(lv):.3f},"
              f" visual {frac(vis):.3f}")
    return out


# --------------------------------------------------------------------------------------
# self-test: the whole pipeline on data with a known answer
# --------------------------------------------------------------------------------------
def self_test(seed: int = 0) -> None:
    """Run the real functions on synthetic data before this ever costs cluster time.

    Three properties, each of which would invalidate the figure if broken: the null is
    calibrated, a planted effect is found in the right cell, and the blocked permutation
    actually protects against a drift-coupled label.
    """
    rng = np.random.default_rng(seed)
    n, P, T, K = 240, 6, 8, 4
    y = np.tile([0, 1], n // 2)

    # 1. no effect anywhere -> AUROC straddles .5, and the max-stat threshold controls FWER:
    #    across independent realisations of the null, ~alpha of MAPS should have any cell
    #    exceed it. This is the property the whole significance claim rests on.
    Z = rng.normal(size=(n, P, T, K))
    au = cv_auroc(Z, y)
    assert abs(au.mean() - 0.5) < 0.02, au.mean()
    nulls = np.stack([cv_auroc(Z, block_permute(y, rng, 0)) for _ in range(400)])
    # the threshold must be checked on permutations it was NOT built from, or the exceedance
    # rate is 5% by definition of a quantile and the test proves nothing
    thr = maxstat_threshold(nulls[:200], 0.05)
    fwer = float((np.abs(nulls[200:] - 0.5).max(axis=(1, 2)) > thr).mean())
    assert 0.005 < fwer < 0.15, f"max-stat FWER is {fwer:.3f}, wanted ~0.05"
    # the per-cell p floor that makes BH unusable at this n_perm, stated as a number
    assert 1.0 / 201.0 > 0.05 / (P * T), "BH would be resolvable here -- revisit the choice"

    # 2. a planted effect in ONE contact and a window of bins is recovered there and only there
    Z2 = rng.normal(size=(n, P, T, K))
    Z2[y == 1, 2, 3:6, 0] += 1.2
    au2 = cv_auroc(Z2, y)
    hit = np.zeros((P, T), dtype=bool)
    hit[2, 3:6] = True
    n2 = np.stack([cv_auroc(Z2, block_permute(y, rng, 0)) for _ in range(199)])
    thr2 = maxstat_threshold(n2, 0.05)
    sig2 = np.abs(au2 - 0.5) > thr2
    # SENSITIVITY and LOCALISATION. Specificity is deliberately not asserted cell-by-cell
    # here: at alpha=.05 a single map has a 5% chance of one false positive BY DESIGN, so
    # such an assert would flake one run in twenty and teach nothing. Property 1 tests the
    # false-positive rate the only way it can be tested -- over many null maps.
    assert sig2[hit].all(), (au2[hit], thr2)
    # the strongest cell is inside the planted window -- WHICH of its three bins wins is a
    # coin flip, they are equal in expectation, so only membership can be asserted
    assert hit.ravel()[au2.argmax()], np.unravel_index(au2.argmax(), (P, T))
    assert sig2[~hit].sum() <= 2, (sig2[~hit].sum(), np.abs(au2[~hit] - .5).max(), thr2)

    # 3. a drift-coupled label beats a FREE permutation on drift alone; blocking removes it
    y_drift = (np.arange(n) >= n // 2).astype(np.int64)      # label IS the trial half
    drift = np.linspace(0, 3, n)[:, None, None, None]
    Zd = rng.normal(size=(n, P, T, K)) + drift               # every cell drifts, no label info
    au_d = cv_auroc(Zd, y_drift)
    free = np.stack([cv_auroc(Zd, block_permute(y_drift, rng, 0)) for _ in range(99)])
    blocked = np.stack([cv_auroc(Zd, block_permute(y_drift, rng, 24)) for _ in range(99)])
    p_free = (1 + (np.abs(free - .5) >= np.abs(au_d - .5)[None]).sum(0)) / 100.0
    p_blk = (1 + (np.abs(blocked - .5) >= np.abs(au_d - .5)[None]).sum(0)) / 100.0
    assert p_free.mean() < 0.1, f"free permutation should be fooled by drift: {p_free.mean():.3f}"
    assert p_blk.mean() > 3 * p_free.mean(), (p_free.mean(), p_blk.mean())

    # 4. the PC reduction keeps what it claims to keep
    C = 12
    W = rng.normal(size=(C, 3))
    X = rng.normal(size=(500, 3)) @ W.T + 0.01 * rng.normal(size=(500, C))
    covm = np.cov(X.T)[None]
    _, fr = pca_basis(covm, 3)
    assert fr[0] > 0.99, fr

    # 5. AUROC agrees with a brute-force reference
    s = rng.normal(size=(50, 3))
    yy = rng.integers(0, 2, 50)
    ref = np.array([
        np.mean([(a > b) + 0.5 * (a == b) for a in s[yy == 1, j] for b in s[yy == 0, j]])
        for j in range(3)])
    assert np.allclose(auroc_cols(s, yy), ref), (auroc_cols(s, yy), ref)

    # 6. the pooled null resolves p far below the per-cell floor, which is its only purpose
    pp = pooled_p(au, nulls[:200])
    assert pp.min() >= 1.0 / (1 + 200 * P * T)
    assert pp.min() < 1.0 / 201.0 or np.abs(au - .5).max() < np.abs(nulls[:200] - .5).max()

    print("[self-test] all 6 properties hold", flush=True)


# --------------------------------------------------------------------------------------
def main() -> None:
    # argv is scanned for --self-test before ANY project import, so the pre-flight check runs
    # in a bare numpy environment. It is the thing you want to be able to run when the module
    # env is wrong, which is exactly when a heavyweight import would stop you running it.
    if "--self-test" in sys.argv:
        seed = 33
        if "--seed" in sys.argv:
            seed = int(sys.argv[sys.argv.index("--seed") + 1])
        self_test(seed)
        return

    from scripts.neuroprobe.viz_anatomy import TASKS

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true",
                   help="verify the statistic on synthetic data and exit (needs no cache and no "
                        "project imports -- handled before argparse so it runs in a bare env)")
    p.add_argument("--cache", help="encode cache dir written by v3_probe_encode_r4")
    p.add_argument("--out-dir", default="results/elec_auroc")
    p.add_argument("--session-index", type=int, default=-1,
                   help="shard index into the sorted session list; -1 = all")
    p.add_argument("--taps", default="enc0_elec,enc12_elec")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--band", default="hga")
    p.add_argument("--pc", type=int, default=32, help="label-free PCs kept per contact")
    p.add_argument("--n-perm", type=int, default=500,
                   help="permutations. 500 because the max-stat threshold is a 95th percentile "
                        "and MEASURED cost is ~1.9 h per session per 200 perms over 15 tasks "
                        "x 2 taps, so the accuracy is affordable")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--no-store-null", action="store_true",
                   help="keep only the per-permutation maxima (2 kB) instead of the full null "
                        "cube (~8 MB per tap/task). The cube is what lets the multiplicity "
                        "correction be revisited without re-running the array.")
    p.add_argument("--perm-block", type=int, default=200,
                   help="trials per permutation block; 0 = free (anti-conservative under drift)")
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--cov-stride", type=int, default=4,
                   help="window stride for the label-free PC covariance pass")
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--aggregate", action="store_true", help="summarise shards in --out-dir")
    p.add_argument("--inference", choices=("maxstat", "fdr"), default="maxstat",
                   help="maxstat = FWER over the whole map (use for any per-contact claim); "
                        "fdr = BH on pooled-null p, more sensitive but expects false positives")
    a = p.parse_args()
    taps = tuple(t for t in a.taps.split(",") if t)
    tasks = tuple(t for t in a.tasks.split(",") if t)

    if a.aggregate:
        agg = aggregate(a.out_dir, taps=taps, tasks=tasks, alpha=a.alpha,
                        inference=a.inference)
        report(agg, taps=taps, tasks=tasks, alpha=a.alpha)
        return

    assert a.cache, "--cache is required unless --self-test or --aggregate"
    files = sorted(f for f in os.listdir(a.cache) if f.endswith(".pt"))
    assert files, f"no .pt records in {a.cache}"
    todo = files if a.session_index < 0 else [files[a.session_index]]
    os.makedirs(a.out_dir, exist_ok=True)
    for fn in todo:
        out = session_shard(os.path.join(a.cache, fn), taps=taps, tasks=tasks, band=a.band,
                            n_pc=a.pc, n_perm=a.n_perm, perm_block=a.perm_block,
                            chunk=a.chunk, seed=a.seed, store_null=not a.no_store_null,
                            cov_stride=a.cov_stride)
        dst = os.path.join(a.out_dir, f"elec_s{int(out['subject_id'])}"
                                     f"_t{int(out['trial_id'])}_{a.band}.npz")
        np.savez_compressed(dst, **out)
        print(f"[write] {dst}", flush=True)


if __name__ == "__main__":
    main()
