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
import zlib
from concurrent.futures import ThreadPoolExecutor

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


def auroc_rows(sT: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC of every ROW of ``sT`` (M, n) against binary ``y`` (n,). Row-major twin of
    ``auroc_cols``, verified bit-identical to it.

    Same Mann-Whitney U, but the rank sum is gathered rather than scattered: ``y[order]``
    picks class 1's sorted positions directly, so there is no (M, n) float64 rank array to
    allocate and no ``put_along_axis``. Ranks are summed in int64, which is exact here
    (n*n <= 1.1e7 << 2^53) rather than merely accurate. The flat-column guard matches
    ``auroc_cols`` and exists for the same reason: a dead contact's all-equal score must not
    fall back on the trial INDEX order, which would read the label's own drift as signal.
    """
    n = sT.shape[1]
    assert y.shape == (n,), (sT.shape, y.shape)
    n1 = int(y.sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return np.full(sT.shape[0], 0.5)
    order = np.argsort(sT, axis=-1)
    r1 = (y[order] * np.arange(1, n + 1, dtype=np.int64)).sum(axis=-1)
    au = (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    au[sT.max(axis=-1) <= sT.min(axis=-1)] = 0.5
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


def contact_major(Z: np.ndarray) -> np.ndarray:
    """(n, P, T, K) -> contiguous (P, T, n, K). Label-free, so it is done ONCE per (tap, task).

    ``_halves`` splits WITHIN each class, so the fold index sets change with every permuted
    label vector and the folds themselves cannot be hoisted out of the permutation loop. The
    layout can: putting the contact axis first makes each threaded contact chunk a contiguous
    view, and makes the score matrix fall out (P*T, n) row-major -- the axis ``argsort`` wants.
    The previous (n, P*T) form sorted down a strided axis.
    """
    return np.ascontiguousarray(Z.transpose(1, 2, 0, 3))


def cv_auroc_cm(Zc: np.ndarray, y: np.ndarray, pool=None, nthread: int = 1) -> np.ndarray:
    """Two-fold cross-validated AUROC per (row, time) cell, from ``contact_major`` features.

    Direction from the fit half's class-mean difference, AUROC on the held-out half, both
    folds averaged. Returns (P, T). Identical statistic to the (n, P, T, K) form; this is the
    layout and arithmetic rearranged so it can be threaded.

    Two changes carry the speedup. (1) The class-mean difference is a CONTRACTION over the
    full trial axis with weights ``+1/n1`` on the fit half's class 1, ``-1/n0`` on its class 0
    and **zero on the eval half** -- so no fold is ever materialised. The previous form
    fancy-index-copied ``Z[ev]`` once and ``Z[fit]`` twice (once per class mask) on every one
    of ``n_perm`` iterations, which at enc12 is several GB of pure memcpy per permutation.
    Both folds' weights go in one (2, n) matrix, so each permutation makes exactly two passes
    over the features rather than four. (2) ``np.argsort`` releases the GIL, so threading over
    contact chunks parallelises the inner loop with no process copies and no extra memory --
    the sbatch's "argsort does not thread" note is true only within a single numpy call.

    Agreement with the previous implementation is ~2e-6 in AUROC, not bit-exact: the
    contraction sums the trial axis in a different order in float32, which flips a handful of
    near-tie ranks out of millions of pairs. That is sound at any size because the observed
    statistic and every permutation use the SAME estimator, so the permutation test remains
    exact for the estimator actually computed. Changing the estimator BETWEEN them is what
    would not be sound.
    """
    P, T, n, _ = Zc.shape
    W = np.zeros((2, n), dtype=np.float32)
    folds, degenerate = [], 0.0
    for i, (fit, ev) in enumerate(((_halves(y)[0], _halves(y)[1]),
                                  (_halves(y)[1], _halves(y)[0]))):
        yf, ye = y[fit], y[ev]
        n1 = int(yf.sum())
        if n1 == 0 or n1 == len(yf) or ye.sum() == 0 or ye.sum() == len(ye):
            degenerate += 0.5
            continue
        W[i, fit] = np.where(yf == 1, 1.0 / n1, -1.0 / (len(yf) - n1))
        folds.append((i, ev, ye))
    if not folds:
        return np.full((P, T), 0.5)
    sl = [slice(c[0], c[-1] + 1)
          for c in np.array_split(np.arange(P), max(1, nthread)) if len(c)]

    def one(s: slice) -> np.ndarray:
        w = np.matmul(W, Zc[s])                                           # (p, T, 2, K)
        sc = np.matmul(Zc[s], np.ascontiguousarray(w.transpose(0, 1, 3, 2)))   # (p,T,n,2)
        acc = np.full((s.stop - s.start, T), degenerate)
        for i, ev, ye in folds:
            e = np.ascontiguousarray(sc[:, :, ev, i])                     # (p, T, n_ev)
            acc += auroc_rows(e.reshape(-1, len(ev)), ye).reshape(-1, T)
        return acc

    parts = list(pool.map(one, sl)) if pool is not None else [one(s) for s in sl]
    return np.concatenate(parts, axis=0) / 2.0


def cv_auroc(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Two-fold cross-validated AUROC per (row, time) cell. ``Z`` is (n, P, T, K).

    Single-shot wrapper: the shard path transposes once and reuses it across the whole
    permutation loop, so this exists for tests and one-off calls.
    """
    return cv_auroc_cm(contact_major(Z), y)


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
def read_session(path: str, *, tap: str, band: str, band_fdims_override=None):
    """``(rec, feat, cols, T, C)`` for one tap, mmapped -- nothing large is materialised."""
    import torch

    from scripts.neuroprobe.viz_reduce import _band_slice

    rec = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if tap not in rec["feats"]:
        return rec, None, None, 0, 0
    feat = rec["feats"][tap]["raw"]
    band_lengths = tuple(int(x) for x in rec["band_lengths"])
    band_fdims = rec.get("band_fdims")
    if band_fdims is None and band_fdims_override is not None:
        # The board cache stores no band_fdims, and enc0's per-band widths are not recoverable
        # from the flattened total: (7,6,7) and (3,5,8) both sum to 348 at the 1 s window, so
        # _band_slice's width check is necessary and NOT sufficient. Only ever pass a value READ
        # off a record the same frontend wrote -- here (7,6,7), read off the 2 s v3r6 viz cache.
        band_fdims = tuple(int(f) for f in band_fdims_override)
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


def n_free_cores() -> int:
    """Cores actually allocated to this process, not the node's core count.

    Slurm gives the job a cpuset, and ``os.cpu_count()`` reports the whole node (128 on Delta)
    which would oversubscribe the allocation eightfold. ``sched_getaffinity`` reports the
    cpuset. Linux-only, hence the fallback.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def session_shard(path: str, *, taps, tasks, band: str, n_pc: int, n_perm: int,
                  perm_block: int, chunk: int, seed: int, store_null: bool = True,
                  cov_stride: int = 4, verbose: bool = True, band_fdims_override=None,
                  nthread: int = 1, checkpoint=None) -> dict:
    out: dict = {}
    meta_done = False
    pool = ThreadPoolExecutor(nthread) if nthread > 1 else None
    for tap in taps:
        rec, feat, cols, T, C = read_session(path, tap=tap, band=band,
                                             band_fdims_override=band_fdims_override)
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
            # transpose once for the whole permutation run, then drop the original: both live
            # at once only for the duration of the copy, so the peak is 2x Z (~3.2 GB at enc12)
            Zc = contact_major(Z)
            del Z
            obs = cv_auroc_cm(Zc, y, pool, nthread)
            # crc32, NOT hash(): Python salts string hashes per process (PYTHONHASHSEED), so
            # hash() here would mean --seed does not actually pin the null -- every re-run and
            # every array task would draw a different one, and a permutation p-value that
            # changes when you re-run it is not a p-value. Each (tap, task) still gets its own
            # independent stream; SeedSequence does the mixing, so crc32 only has to be stable.
            rng = np.random.default_rng(np.random.SeedSequence(
                [seed, zlib.crc32(tap.encode()), zlib.crc32(task.encode())]))
            null = np.empty((n_perm, P, T), dtype=np.float32)
            for p in range(n_perm):
                null[p] = cv_auroc_cm(Zc, block_permute(y, rng, perm_block), pool, nthread)
            del Zc
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
                # --n-perm 0 is the map-only mode used to smoke-test the read path against a
                # real record; there is no threshold to quote then, and no inference either.
                nm = out[f"null_max/{tap}/{task}"]
                thr = float(np.quantile(nm, 0.95)) if len(nm) else float("nan")
                print(f"[auroc] {tap} {task}: n={len(sel)} P={P} T={T} "
                      f"max={obs.max():.3f} fwer_thr={0.5 + thr:.3f} "
                      f"n_cells_fwer={(np.abs(obs - .5) > thr).sum()} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            # Flush after every (tap, task), not once at the end of the session. A shard is
            # 30 units long, so end-only writing means nothing at all is readable for hours
            # and a walltime kill returns nothing. ``aggregate`` skips absent keys, so a
            # partially written shard is already a usable -- and honestly labelled -- result.
            if checkpoint is not None:
                checkpoint(out)
    if pool is not None:
        pool.shutdown()
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
    # ".tmp"/".partial" are in-flight checkpoint writes from a still-running array. Reading one
    # is a BadZipFile, so aggregating a live shard dir must skip them by NAME -- catching the
    # exception instead would silently drop a whole session's finished results on a race.
    files = sorted(f for f in os.listdir(shard_dir)
                   if f.endswith(".npz") and ".tmp" not in f and ".partial" not in f)
    assert files, f"no shards in {shard_dir}"
    A: dict = {}
    cov: dict = {}
    canon_of: dict = {}
    for fn in files:
        z = np.load(os.path.join(shard_dir, fn), allow_pickle=False)
        sub = int(z["subject_id"])
        sess = f"s{sub}_t{int(z['trial_id'])}"
        canon = z["parcel_canon"]
        bases = np.array([base_of.get(int(i), "unknown") for i in canon])
        # A parcel tap's map has one row per DISTINCT parcel, not one per contact, so it needs its
        # own label axis. np.unique(parcel_canon) reproduces the cache's own `present_parcels`
        # field exactly (verified on all 12 board records), which is why this is recoverable from
        # what the shard already stores instead of needing the cache or a re-run.
        pres = np.unique(canon)
        bases_p = np.array([base_of.get(int(i), "unknown") for i in pres])
        cov[sess] = (sub, bases, bases_p)
        # kept verbatim so the brain render can ASSERT its electrode axis against this one
        # rather than trust that two files written by different scripts agree on the order
        canon_of[sess] = canon
        for tap in taps:
            for task in tasks:
                key = f"auroc/{tap}/{task}"
                if key not in z.files:
                    continue
                au = z[key]
                # The unit is read off the tap name, so verify it against the map that arrived:
                # a parcel map silently labelled with the contact axis would misattribute every
                # row rather than fail, and both axes live in the same file.
                exp = len(bases) if tap.endswith("_elec") else len(bases_p)
                assert au.shape[0] == exp, (
                    f"{fn}: {tap}/{task} has {au.shape[0]} rows but its unit implies {exp} "
                    f"(contacts={len(bases)}, parcels={len(bases_p)})")
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
                    # the full (rows x time) map, not just its reduction: the movie needs every
                    # frame, and at 119 x 32 float32 the whole cube over 12 sessions x 2 taps x
                    # 15 tasks is ~5.5 MB, so there is nothing to save by dropping it
                    "au": au,
                    "peak": peak, "sig": sig, "fwer_thr": thr,
                    "auroc_at_peak": au[np.arange(au.shape[0]), np.abs(au - .5).argmax(axis=1)],
                    "t_at_peak": np.abs(au - .5).argmax(axis=1),
                    "T": au.shape[1],
                }
    return {"A": A, "cov": cov, "canon": canon_of, "lobe_of_base": lobe_of_base}


def row_bases(agg: dict, sess: str, tap: str) -> np.ndarray:
    """Row-axis DKT base labels for this tap's UNIT: contacts for ``*_elec``, parcels otherwise.

    The unit is a property of the tap, not of the run, and both axes are stored side by side. So
    it is derived here and never defaulted -- a parcel map indexed by the contact axis is a shape
    error when the counts differ and a silent misattribution when they happen to agree.

    Consequence for any number built on this: at a parcel tap the denominator counts
    (session x distinct parcel) units, NOT contacts, so parcel and contact values are on
    different scales and must never be pooled or compared as absolutes.
    """
    _, bases_elec, bases_parcel = agg["cov"][sess]
    return bases_elec if tap.endswith("_elec") else bases_parcel


def unit_of(tap: str) -> str:
    """``"contacts"`` or ``"parcels"`` -- for labelling axes so a figure cannot lie about its n."""
    return "contacts" if tap.endswith("_elec") else "parcels"


def region_table(agg: dict, tap: str, task: str,
                 sessions=None) -> list[tuple[str, float, int, int, int]]:
    """``(base, mean peak |AUROC-.5| over rows, n_sig, n_rows, n_subjects)`` desc.

    ``n_rows`` is in this tap's unit -- see :func:`row_bases`.
    """
    A, cov = agg["A"], agg["cov"]
    per: dict[str, list] = {}
    subs: dict[str, set] = {}
    for (sess, tp, tk), d in A.items():
        if tp != tap or tk != task or (sessions is not None and sess not in sessions):
            continue
        sub = cov[sess][0]
        bases = row_bases(agg, sess, tap)
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


def sig_fraction(agg: dict, tap: str, task: str, base: str,
                 sessions=None) -> tuple[int, int, int]:
    """``(n_sig, n_contacts, n_sessions_with_any)`` for one region, pooled over sessions."""
    ns = nc = nsess = 0
    for (sess, tp, tk), d in agg["A"].items():
        if tp != tap or tk != task or (sessions is not None and sess not in sessions):
            continue
        m = row_bases(agg, sess, tap) == base
        if not m.any():
            continue
        ns += int(d["sig"][m].sum())
        nc += int(m.sum())
        nsess += int(d["sig"][m].any())
    return ns, nc, nsess


def report(agg: dict, *, taps, tasks, alpha: float = 0.05, top: int = 6, only=None) -> dict:
    """Print the contact-level answer to Greg's question, and the falsifier beside it.

    ``only`` restricts every number to a session list -- pass ``matched_sessions(...)`` for any
    figure or claim that compares taps or families, and nothing for the per-task survey.
    """
    from scripts.neuroprobe.viz_anatomy import EVENT, TARGET, VISUAL_TASKS

    A = agg["A"]
    sessions = sorted({s for (s, _, _) in A} if only is None else set(only))
    print(f"\nsessions={len(sessions)}{' MATCHED' if only is not None else ''}  alpha={alpha}"
          f" (max-statistic FWER over contact x time)")
    out: dict = {}
    for tap in taps:
        thr = [d["fwer_thr"] for (s, tp, _), d in A.items() if tp == tap and s in set(sessions)]
        if not thr:
            continue
        print(f"\n################ {tap}  (FWER threshold on |AUROC-.5|: "
              f"median {np.median(thr):.3f}, range {min(thr):.3f}-{max(thr):.3f})")
        for task in tasks:
            rows = region_table(agg, tap, task, sessions=only)
            if not rows:
                continue
            names = [r[0] for r in rows]
            rank = names.index(TARGET) + 1 if TARGET in names else -1
            ns, nc, nsess = sig_fraction(agg, tap, task, TARGET, sessions=only)
            out[(tap, task)] = {"st_rank": rank, "st_sig": ns, "st_n": nc, "st_sess": nsess}
            head = "  ".join(f"{b}({mp:.3f},{s}/{n})" for b, mp, s, n, _ in rows[:top])
            print(f"  {task:18s} STG rank {rank:>3}  sig {ns:>3}/{nc:<3} in {nsess} sess | {head}")

        # Greg's test and its falsifier, side by side, at contact resolution
        ev = [out[(tap, t)] for t in EVENT if (tap, t) in out]
        al = [out[(tap, t)] for t in AUDIO_LEVEL if (tap, t) in out]
        vis = [out[(tap, t)] for t in VISUAL_TASKS if (tap, t) in out]
        def frac(rs):
            n = sum(r["st_n"] for r in rs)
            return (sum(r["st_sig"] for r in rs) / n) if n else float("nan")
        print(f"  --> STG top-3 in {sum(1 for r in ev if 0 < r['st_rank'] <= 3)}/{len(ev)} event"
              f" tasks, {sum(1 for r in al if 0 < r['st_rank'] <= 3)}/{len(al)} audio-level,"
              f" {sum(1 for r in vis if 0 < r['st_rank'] <= 3)}/{len(vis)} visual")
        print(f"  --> STG {unit_of(tap)} surviving FWER: event {frac(ev):.3f}, "
              f"audio-level {frac(al):.3f}, visual {frac(vis):.3f}"
              f"   [{len(sessions)} sessions{'' if only is None else ', MATCHED'}]")
    return out


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
# The 3-way DISJOINT partition of the 15 tasks. viz_anatomy's LEVEL is TASKS[9:], which
# CONTAINS all four VISUAL_TASKS, so an "event / level / visual" triple reports the visual
# tasks twice and its "level" number is really "everything that is not an event". Splitting
# level into its auditory remainder makes the visual control exclusive, which is the whole
# point of quoting it: it is the only column that is supposed to be near zero.
AUDIO_LEVEL = ("volume", "pitch")


def families() -> tuple[tuple[str, tuple[str, ...]], ...]:
    from scripts.neuroprobe.viz_anatomy import EVENT, TASKS, VISUAL_TASKS
    fam = (("event", EVENT), ("audio-level", AUDIO_LEVEL), ("visual", VISUAL_TASKS))
    flat = [t for _, ts in fam for t in ts]
    assert len(flat) == len(set(flat)) == len(TASKS), (
        f"families must partition TASKS exactly: {len(flat)} slots, "
        f"{len(set(flat))} distinct, {len(TASKS)} tasks")
    return fam


def matched_sessions(agg: dict, taps, tasks) -> list[str]:
    """Sessions holding EVERY (tap, task) cell being compared.

    Without this, a tap-to-tap or family-to-family comparison silently mixes cohorts: the
    visual tasks are present in far fewer sessions than the event ones (``face_num`` 3 of 12),
    so pooling a fraction over "whatever ran" compares 12 sessions against 3 and reads the
    cohort difference as an effect. Any number that spans taps or families must come from
    this list, and the figure must print how many sessions it kept.
    """
    have: dict[str, set] = {}
    for (sess, tp, tk) in agg["A"]:
        have.setdefault(sess, set()).add((tp, tk))
    need = {(tp, tk) for tp in taps for tk in tasks}
    return sorted(s for s, h in have.items() if need <= h)


def _fam_fraction(agg, tap, base, fam_tasks, sessions=None) -> tuple[int, int]:
    """``(n_sig, n_rows)`` for one region pooled over a family's tasks, in this tap's unit."""
    ns = nc = 0
    for t in fam_tasks:
        s, n, _ = sig_fraction(agg, tap, t, base, sessions=sessions)
        ns += s
        nc += n
    return ns, nc


def n_units_of(agg, base: str, tap: str, sessions=None) -> int:
    """Rows in one region pooled over sessions, in ``tap``'s unit. Task-independent, so it is the
    honest denominator to print on a region axis: .95 of 10 contacts and .56 of 202 look
    identical on a colour scale and mean very different things."""
    return sum(int((row_bases(agg, sess, tap) == base).sum()) for sess in agg["cov"]
               if sessions is None or sess in sessions)


def _region_rows(agg, tap, sessions=None, min_units: int = 0, top: int = 16) -> list[str]:
    """Regions with enough coverage to plot, ordered by their EVENT-family sig fraction.

    The coverage floor is in ``tap``'s UNIT, so it cannot be one constant. A region contributes
    one row per contact at an ``_elec`` tap (tens to hundreds) but at most one row per session per
    hemisphere at a parcel tap (<=24 here), so reusing the contact floor of 20 would reject nearly
    every region and quietly return a short list instead of failing. ``min_units=0`` picks the
    unit's own default; pass a number only to override both.
    """
    from scripts.neuroprobe.viz_anatomy import EVENT

    if not min_units:
        min_units = 20 if tap.endswith("_elec") else 4
    bases = {b for sess in agg["cov"]
             if sessions is None or sess in sessions
             for b in row_bases(agg, sess, tap)}
    scored = []
    for b in sorted(bases - {"unknown"}):
        ns, nc = _fam_fraction(agg, tap, b, EVENT, sessions)
        if nc >= min_units * len(EVENT):
            scored.append((ns / nc, b))
    scored.sort(reverse=True)
    assert scored, (
        f"{tap}: no region cleared the coverage floor of {min_units} "
        f"{unit_of(tap)} -- refusing to render an empty figure")
    return [b for _, b in scored[:top]]


def fig_calibration(agg: dict, taps, out: str, sessions=None) -> str:
    """Observed map maximum against that map's own permutation FWER threshold, one point per
    (session, tap, task). The panel that justifies every other panel.

    This is what measure #1 has no version of. ``d_cv`` has no null, so a small value there is
    indistinguishable from "few trials"; here each cell carries its own threshold drawn from 500
    label permutations of the SAME fitting procedure, and a point's height above the diagonal is
    the only thing that licenses calling any contact real. Two things should be legible: the
    thresholds cluster tightly (the null is a property of n and the fold structure, not of the
    task), and the visual tasks sit ON the line while the event tasks sit far above it.
    """
    import matplotlib.pyplot as plt

    fam = families()
    colour = {"event": "#d62728", "audio-level": "#1f77b4", "visual": "#2ca02c"}
    fig, axes = plt.subplots(1, len(taps), figsize=(5.4 * len(taps), 4.6), dpi=150, sharex=True,
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, tap in zip(axes, taps):
        for fname, ts in fam:
            xs, ys, n_above = [], [], 0
            for (sess, tp, tk), d in agg["A"].items():
                if tp != tap or tk not in ts or (sessions is not None and sess not in sessions):
                    continue
                if not np.isfinite(d["fwer_thr"]):
                    continue
                xs.append(d["fwer_thr"])
                ys.append(float(d["peak"].max()))
                n_above += int(ys[-1] > xs[-1])
            if xs:
                ax.scatter(xs, ys, s=22, c=colour[fname], ec="k", lw=.3, alpha=.85,
                           label=f"{fname} ({n_above}/{len(xs)} cells have a survivor)")
        lo = 0.0
        hi = max(.05, float(max(ax.get_xlim()[1], ax.get_ylim()[1])))
        ax.plot([lo, hi], [lo, hi], "k--", lw=.9)
        ax.text(hi, hi, " y = threshold ", fontsize=6, rotation=45, va="bottom", ha="right")
        ax.set_xlabel("permutation FWER threshold on |AUROC $-$ 0.5|")
        ax.set_title(tap, fontsize=10)
        ax.grid(lw=.3, alpha=.4)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("observed map maximum |AUROC $-$ 0.5|")
    fig.suptitle("MEASURE #2 CALIBRATION: every (session, task) cell against its OWN null. "
                 "Points on the dashed line have no significant contact;\nheight above it is "
                 "the effect. Thresholds cluster tightly, as a null that depends on n and the "
                 "fold structure -- not on the task -- should.\nNote the visual cells are NOT "
                 "empty: the dissociation is in the FRACTION of contacts (figE1/E2), not in "
                 "visual being undetectable.", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out, "figE0_calibration.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")
    return p


def fig_dissociation(agg: dict, tap: str, out: str, sessions=None) -> str:
    """Region x task fraction of contacts surviving the whole-map FWER threshold.

    This is the panel measure #1 could not produce. #1's ``d_cv`` has no null, so its region
    ordering is only an ordering; here every cell is a count of contacts that beat a
    calibrated max-statistic threshold, so the visual column being near zero is a RESULT and
    not just a small number. Colour is sequential, not diverging: a fraction has a floor at 0
    and no meaningful midpoint, and RdBu_r would invent one.
    """
    import matplotlib.pyplot as plt

    from scripts.neuroprobe.viz_anatomy import NEIGHBOURS, TARGET

    fam = families()
    cols = [t for _, ts in fam for t in ts]
    rows = _region_rows(agg, tap, sessions)
    assert rows, f"{tap}: no region has enough coverage to plot"
    M = np.full((len(rows), len(cols)), np.nan)
    N = np.zeros((len(rows), len(cols)), dtype=int)
    for i, b in enumerate(rows):
        for j, t in enumerate(cols):
            ns, nc, _ = sig_fraction(agg, tap, t, b, sessions=sessions)
            if nc:
                M[i, j] = ns / nc
                N[i, j] = nc

    fig, ax = plt.subplots(figsize=(13.5, 0.42 * len(rows) + 3.0), dpi=150)
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0.0,
                   vmax=float(np.nanmax(M)) if np.isfinite(M).any() else 1.0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=55, ha="right", fontsize=8)
    # identity marks are glyphs PREFIXED ONTO THE TICK LABEL, never colour -- on a magma scale a
    # coloured tick would read as a value, and a separate text at negative x lands on top of the
    # label instead of beside it. Same convention as viz_anatomy's region axes.
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([("▶ " if b == TARGET else "▷ " if b in NEIGHBOURS else "   ")
                        + f"{b} (n={n_units_of(agg, b, tap, sessions)})" for b in rows],
                       fontsize=8)
    edge = 0
    for name, ts in fam:
        edge += len(ts)
        if edge < len(cols):
            ax.axvline(edge - 0.5, color="w", lw=2.2)
        ax.text(edge - len(ts) / 2 - 0.5, -0.85, name, ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]) and M[i, j] > 0.005:
                ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=5.5, color="w" if M[i, j] < 0.6 * np.nanmax(M) else "k")
    unit = unit_of(tap)
    fig.colorbar(im, ax=ax, fraction=.018, pad=.01,
                 label=f"fraction of {unit} surviving FWER")
    ns = len(sessions) if sessions is not None else len({s for (s, _, _) in agg["A"]})
    ax.set_title(f"MEASURE #2 -- {tap}: single-trial per-{unit[:-1]} decodability, "
                 f"max-statistic FWER at $\\alpha$=0.05\n"
                 f"{ns} sessions{' (MATCHED)' if sessions is not None else ''}; "
                 f"▶ superiortemporal, ▷ transversetemporal / middletemporal. n = {unit} "
                 f"pooled over sessions.\nThe visual block is the CONTROL. It is dark at enc0; "
                 f"whether it stays dark with depth is the result, not an assumption.",
                 fontsize=9,
                 pad=38)   # room for the family labels, which sit just above the top row
    fig.tight_layout()
    p = os.path.join(out, f"figE1_dissociation_{tap}.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")
    return p


def fig_family(agg: dict, taps, out: str, regions=None) -> str:
    """Family sig-fraction per region, one bar group per tap, on MATCHED sessions only.

    The matching is the point. The event-vs-visual ratio is a comparison across families, and
    the tap-to-tap change is a comparison across taps, so both legs must stand on the same
    sessions or the number is a cohort artifact.
    """
    import matplotlib.pyplot as plt

    from scripts.neuroprobe.viz_anatomy import NEIGHBOURS, TARGET, TASKS

    if regions is None:
        regions = (NEIGHBOURS[1], TARGET, "middletemporal", "inferiorparietal")
    units = {unit_of(t) for t in taps}
    assert len(units) == 1, (
        f"figE2 puts one y axis on all of {list(taps)}, but they span units {sorted(units)}. "
        "A contact fraction and a parcel fraction have different denominators, so bars drawn "
        "side by side would invite a comparison that is not defined. Render them separately.")
    unit = units.pop()
    sess = matched_sessions(agg, taps, TASKS)
    assert sess, ("no session has every (tap, task) cell yet -- the array is still running. "
                  "Re-run when it finishes; an unmatched version of this figure is not worth "
                  "drawing.")
    fam = families()
    fig, axes = plt.subplots(1, len(regions), figsize=(3.5 * len(regions), 3.7), dpi=150,
                             sharey=True)
    axes = np.atleast_1d(axes)
    w = 0.8 / len(taps)
    colour = plt.get_cmap("viridis")(np.linspace(.25, .75, len(taps)))
    for ax, base in zip(axes, regions):
        for k, tap in enumerate(taps):
            fr, lab = [], []
            for name, ts in fam:
                ns, nc = _fam_fraction(agg, tap, base, ts, sess)
                fr.append(ns / nc if nc else np.nan)
                lab.append(f"{ns}/{nc}")
            x = np.arange(len(fam)) + (k - (len(taps) - 1) / 2) * w
            ax.bar(x, fr, width=w, color=colour[k], label=tap, ec="k", lw=.4)
            for xi, f, l in zip(x, fr, lab):
                if np.isfinite(f):
                    ax.text(xi, f + .012, l, ha="center", fontsize=5, rotation=90)
        ax.set_xticks(range(len(fam)))
        ax.set_xticklabels([n for n, _ in fam], fontsize=8)
        ax.set_title(base + ("  ◀" if base == TARGET else ""), fontsize=9)
        ax.grid(axis="y", lw=.3, alpha=.4)
        ax.set_ylim(0, 1.0)    # headroom for the rotated n_sig/n labels, and a fixed 0-1 scale
                               # so the four regions are read against each other, not rescaled
    axes[0].set_ylabel(f"fraction of {unit} surviving FWER")
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"MEASURE #2: the event-vs-visual dissociation, and what depth does to it "
                 f"({len(sess)} MATCHED sessions, all 15 tasks present in each; unit = {unit})",
                 fontsize=9)
    fig.tight_layout()
    p = os.path.join(out, "figE2_family.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")
    return p


def fig_brain_elec(agg: dict, tmpl: dict, tap: str, tasks, out: str, sessions=None) -> str:
    """Every contact on the BrainTreebank's own hemisphere renders, coloured by peak
    |AUROC-.5|, with FWER survivors marked.

    The electrode axis is ASSERTED against the shard's ``parcel_canon``, not assumed: the two
    files are written by different scripts, and a permuted axis would paint real effects onto
    the wrong dots -- which looks like a result instead of like a bug.
    """
    import matplotlib.pyplot as plt

    assert tap.endswith("_elec"), (
        f"{tap} is a {unit_of(tap)} tap: its map has one row per parcel, so there is no "
        "per-contact value to put on a contact. Only an _elec tap can be rendered as dots.")
    fig, axes = plt.subplots(len(tasks), 2, figsize=(12.5, 4.3 * len(tasks)), dpi=150,
                             squeeze=False)
    dev = [d["peak"] for (s, tp, tk), d in agg["A"].items()
           if tp == tap and tk in tasks and (sessions is None or s in sessions)]
    vmax = float(np.percentile(np.concatenate(dev), 99)) if dev else 0.1
    n_join = 0
    for r, task in enumerate(tasks):
        for c, sd in enumerate(("left", "right")):
            ax = axes[r][c]
            ax.imshow(tmpl["img"][sd])
            ax.axis("off")
            for (subj, trial), v in tmpl["pts"].items():
                sess = f"s{subj}_t{trial}"
                d = agg["A"].get((sess, tap, task))
                if d is None or (sessions is not None and sess not in sessions):
                    continue
                canon = agg["canon"][sess]
                assert len(canon) == len(v["pid"]) and np.array_equal(canon, v["pid"]), (
                    f"{sess}: the shard's electrode axis ({len(canon)}) disagrees with the "
                    f"template's ({len(v['pid'])}) -- refusing to render onto the wrong dots")
                m = v["found"] & (v["side"] == sd)
                if not m.any():
                    continue
                if r == 0:
                    n_join += int(m.sum())      # both panels, else this reports one hemisphere
                sig = m & d["sig"]
                q = m & ~d["sig"]
                ax.scatter(v["xy"][q, 0], v["xy"][q, 1], s=7, c=d["peak"][q], cmap="magma",
                           vmin=0, vmax=vmax, alpha=.45, lw=0)
                ax.scatter(v["xy"][sig, 0], v["xy"][sig, 1], s=30, c=d["peak"][sig],
                           cmap="magma", vmin=0, vmax=vmax, ec="k", lw=.7)
            if c == 0:
                ns = sum(int(d["sig"].sum()) for (s, tp, tk), d in agg["A"].items()
                         if tp == tap and tk == task
                         and (sessions is None or s in sessions))
                nc = sum(int(len(d["sig"])) for (s, tp, tk), d in agg["A"].items()
                         if tp == tap and tk == task
                         and (sessions is None or s in sessions))
                ax.set_title(f"{task}  --  {ns}/{nc} contacts survive FWER", fontsize=10,
                             loc="left")
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=.014, pad=.01,
                 label="peak |AUROC $-$ 0.5|")
    fig.suptitle(f"MEASURE #2 -- {tap}: per-contact single-trial decodability. Large "
                 f"black-edged dots clear the whole-map FWER threshold; small faded dots do "
                 f"not.\n{n_join} contacts projected. Colour is shared across rows, so the "
                 f"visual row is directly comparable to the speech rows.", fontsize=9)
    p = os.path.join(out, f"figE3_brain_{tap}.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")
    return p


def movie_brain_elec(agg: dict, tmpl: dict, tap: str, tasks, out: str, *,
                     win: tuple[float, float], sessions=None, fps: int = 4) -> str:
    """figE3 played over time: one frame per time bin, dots coloured by that bin's |AUROC-.5|.

    This is figE3's spatial map crossed with figE4's latency claim, and it needs no new compute --
    the shards already store the whole ``(contacts x time)`` AUROC map, and figE3 was only ever
    showing its max over time.

    Two things make the per-frame threshold legitimate rather than a multiplicity disaster. The
    FWER threshold is a MAX-statistic over the whole contact-by-time map, so a dot that clears it
    in any single frame is already corrected for having looked at every contact AND every bin --
    marking frames individually spends no extra alpha. And the colour scale is fixed across all
    frames from the whole cube, so brightness changes are the signal moving, not the scale
    rescaling under it.

    What it still is NOT: a propagation measurement. Each frame is an independent
    cross-validated fit at that bin, so this shows WHEN decodability is present, not signal
    travelling between sites, and at an encoder tap the receptive field spans the window so the
    frame index stops meaning time at all. Hence the enc0-only guard.
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    assert tap.endswith("_elec"), (
        f"{tap} is a {unit_of(tap)} tap -- no per-contact value exists to animate on dots.")
    assert tap.startswith("enc0"), (
        f"{tap}: an encoder time bin has already attended over the whole window, so a frame "
        "index there is not a time and the movie would invite a propagation reading that the "
        "measurement cannot support. Animate enc0 only; use the static figE3 for depth.")
    w0, w1 = float(win[0]), float(win[1])
    assert w1 > w0, f"--win must be start,end with end > start (got {win})"

    cells = [(s, d) for (s, tp, tk), d in agg["A"].items()
             if tp == tap and tk in tasks and (sessions is None or s in sessions)]
    assert cells, f"no {tap} cells for {list(tasks)}"
    Ts = {d["au"].shape[1] for _, d in cells}
    assert len(Ts) == 1, f"{tap}: mixed time axes {Ts} -- refusing one ms axis over them"
    T = Ts.pop()
    step = (w1 - w0) / T * 1000.0
    t_ms = w0 * 1000.0 + (np.arange(T) + 0.5) * step
    # fixed across every frame, from the full cube: a per-frame scale would make a quiet bin look
    # as bright as a loud one and turn "the scale rescaled" into "the brain lit up"
    vmax = float(np.percentile(np.abs(np.concatenate(
        [d["au"] for _, d in cells], axis=0) - 0.5), 99.5))

    fig, axes = plt.subplots(len(tasks), 2, figsize=(12.5, 4.3 * len(tasks)), dpi=110,
                             squeeze=False)
    art: list[tuple] = []
    for r, task in enumerate(tasks):
        for c, sd in enumerate(("left", "right")):
            ax = axes[r][c]
            ax.imshow(tmpl["img"][sd])
            ax.axis("off")
            for (subj, trial), v in tmpl["pts"].items():
                sess = f"s{subj}_t{trial}"
                d = agg["A"].get((sess, tap, task))
                if d is None or (sessions is not None and sess not in sessions):
                    continue
                canon = agg["canon"][sess]
                assert len(canon) == len(v["pid"]) and np.array_equal(canon, v["pid"]), (
                    f"{sess}: the shard's electrode axis ({len(canon)}) disagrees with the "
                    f"template's ({len(v['pid'])}) -- refusing to render onto the wrong dots")
                m = v["found"] & (v["side"] == sd)
                if not m.any():
                    continue
                # ONE artist per (task, side, session) holding every contact, restyled per frame.
                # Splitting survivors into their own artist would mean rebuilding artists whenever
                # the surviving SET changes, which is exactly what changes between frames.
                sc = ax.scatter(v["xy"][m, 0], v["xy"][m, 1], c=np.zeros(int(m.sum())),
                                cmap="magma", vmin=0, vmax=vmax, s=7, ec="k", lw=0)
                art.append((sc, np.abs(d["au"][m] - 0.5), float(d["fwer_thr"])))
            if c == 0:
                ax.set_title(task, fontsize=11, loc="left")

    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=.014, pad=.01,
                 label="|AUROC $-$ 0.5| in this time bin")
    sup = fig.suptitle("", fontsize=10)

    def draw(t: int):
        n_sig = 0
        for sc, cube, thr in art:
            val = cube[:, t]
            hit = val > thr
            n_sig += int(hit.sum())
            sc.set_array(val)
            sc.set_sizes(np.where(hit, 34.0, 7.0))
            sc.set_linewidths(np.where(hit, .7, 0.0))
            sc.set_alpha(None)
        sup.set_text(
            f"MEASURE #2 -- {tap}: single-trial decodability over time, {t_ms[t]:+.0f} ms "
            f"(bin {t + 1}/{T}, width {step:.1f} ms)\n"
            f"{n_sig} contacts clear the whole-map FWER threshold in THIS bin. Colour scale and "
            f"threshold are fixed across frames.\nEach frame is an independent fit -- this is "
            f"when decodability is present, NOT signal propagating.")
        return [a[0] for a in art] + [sup]

    ani = animation.FuncAnimation(fig, draw, frames=T, interval=1000 // max(fps, 1), blit=False)
    p = os.path.join(out, f"figE5_movie_{tap}.gif")
    ani.save(p, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[fig] {p}  ({T} frames, {step:.1f} ms/bin, vmax={vmax:.3f})")
    return p


def build_demo_elec(agg: dict, tmpl: dict, out: str, *, taps, tasks,
                    win: tuple[float, float], sessions=None) -> str:
    """The scrub demo, driven by measure #2 instead of measure #1.

    Reuses ``viz_anatomy``'s template wholesale -- same canvas, scrubber, hover and LUT. Three
    things differ from the ``d_cv`` demo and all three are substantive:

    * **Per CONTACT, not per parcel.** measure #1 had one value per (subject, parcel) and needed a
      group index to stay small; measure #2 is already per contact, and at 32 bins the whole cube
      is a few hundred kB, so every contact carries its own series (``g[i] = i``).
    * **Signed.** The value is ``AUROC - 0.5``, which is signed, so the diverging RdBu_r scale
      means what it looks like it means.
    * **The FWER layer.** Survivors of the whole-map max-statistic threshold get a heavy ring and
      everything else fades. Without it this would be measure #1 in different paint -- the
      calibrated null IS what measure #2 adds.

    ``win`` is REQUIRED for the same reason as in :func:`fig_peak_time`: the ms axis is the
    scrubber's entire content, and ``viz_anatomy``'s ``_t_ms`` describes the 2 s reduction.
    """
    import base64

    from scripts.neuroprobe.viz_anatomy import MARK, _img_png, render_demo_html

    w0, w1 = float(win[0]), float(win[1])
    assert w1 > w0, f"--win must be start,end with end > start (got {win})"
    assert tmpl["pts"], "no projected contacts"

    # one trial per subject: two trials of the same subject are the same montage, so pooling both
    # would double-weight that subject. Same rule as viz_anatomy._pooled_contacts -- but the
    # electrode INDEX is kept here, because the value lives per contact and must be looked up.
    pick: dict = {}
    for key in sorted(tmpl["pts"]):
        pick.setdefault(key[0], key)

    pts = []
    for subj, key in sorted(pick.items()):
        sess = f"s{key[0]}_t{key[1]}"
        if sessions is not None and sess not in sessions:
            continue
        v = tmpl["pts"][key]
        canon = agg["canon"].get(sess)
        if canon is None:
            continue
        assert len(canon) == len(v["pid"]) and np.array_equal(canon, v["pid"]), (
            f"{sess}: the shard's electrode axis ({len(canon)}) disagrees with the template's "
            f"({len(v['pid'])}) -- refusing to render onto the wrong dots")
        base_of_row = row_bases(agg, sess, taps[0])
        for i in np.where(v["found"])[0]:
            pts.append({"x": float(v["xy"][i, 0]), "y": float(v["xy"][i, 1]),
                        "side": v["side"][i], "subj": subj, "sess": sess, "ei": int(i),
                        "base": base_of_row[i]})
    assert pts, "no contact survived the session filter"

    series, thr, lims = {}, {}, {}
    for tap in taps:
        for task in tasks:
            k = f"{tap}|{task}"
            rows, trow, vals = [], [], []
            for q in pts:
                d = agg["A"].get((q["sess"], tap, task))
                if d is None:
                    rows.append(None)
                    trow.append(0)
                    continue
                a = d["au"][q["ei"]] - 0.5
                rows.append([int(round(1000 * x)) if np.isfinite(x) else None for x in a])
                trow.append(int(round(1000 * float(d["fwer_thr"]))))
                vals.append(np.abs(a))
            if not vals:
                continue
            series[k] = rows
            thr[k] = trow
            lims[k] = round(float(np.nanpercentile(np.concatenate(vals), 98)) or 0.05, 4)
    assert series, f"no (tap, task) in {list(taps)}x{list(tasks)} had data"

    import matplotlib as mpl
    lut = (255 * mpl.colormaps["RdBu_r"](np.linspace(0, 1, 128))[:, :3]).round().astype(int)
    labels = []
    for j, (base, (name, _)) in enumerate(MARK.items()):
        for si, sd in enumerate(("left", "right")):
            m = [q for q in pts if q["base"] == base and q["side"] == sd]
            if len(m) >= 2:
                labels.append({"s": si, "name": name, "dx": (-1) ** j * 0.13, "dy": j,
                               "x": round(sum(q["x"] for q in m) / len(m), 1),
                               "y": round(sum(q["y"] for q in m) / len(m), 1)})

    T = len(next(r for r in series[next(iter(series))] if r is not None))
    step = (w1 - w0) / T * 1000.0
    t_ms = w0 * 1000.0 + (np.arange(T) + 0.5) * step
    payload = {
        "img": {sd: base64.b64encode(_img_png(tmpl["img"][sd])).decode("ascii")
                for sd in ("left", "right")},
        "wh": {sd: [int(tmpl["img"][sd].shape[1]), int(tmpl["img"][sd].shape[0])]
               for sd in ("left", "right")},
        "x": [round(q["x"], 1) for q in pts],
        "y": [round(q["y"], 1) for q in pts],
        "side": [0 if q["side"] == "left" else 1 for q in pts],
        "g": list(range(len(pts))),      # one series per CONTACT, not per parcel group
        "lab": [f"S{q['subj']} · {q['base']}" for q in pts],
        "series": series, "thr": thr, "lim": lims,
        "t_ms": [round(float(v), 1) for v in t_ms],
        "lut": lut.tolist(), "labels": labels,
    }
    keys = [f"{tp}|{tk}" for tp in taps for tk in tasks if f"{tp}|{tk}" in series]
    ns = len({q["sess"] for q in pts})
    html = render_demo_html(
        payload, keys,
        title="Measure #2: single-trial decodability, per contact, over time",
        h1="Which contacts carry the label, and when? Single-trial, cross-validated, "
           "FWER-controlled",
        intro=f"Colour is <b>AUROC &minus; 0.5</b> from a <b>single-trial</b> cross-validated fit "
              f"at that contact and that time bin — no trial averaging anywhere. Contacts with a "
              f"<b>heavy black ring</b> beat the <b>whole-map max-statistic FWER threshold</b> "
              f"from 500 label permutations; the faded ones do not. That threshold is corrected "
              f"over every contact <i>and</i> every time bin jointly, so a ring in any single "
              f"frame is already multiplicity-controlled — scrubbing spends no extra alpha. "
              f"<b>Dot area and colour both encode the effect</b>, and <b>hover</b> gives the "
              f"subject, DKT parcel, value and verdict. {len(pts)} contacts, {ns} subjects, "
              f"{T} bins of {step:.1f} ms.",
        note="Colour is a <b>per-contact</b> value here, not a parcel average — neighbouring dots "
             "are independent measurements. Each frame is an <b>independent fit</b>, so this "
             "shows <i>when</i> the label is decodable, <b>not</b> signal propagating between "
             "sites; and these are 1 s-window labels, so sustained decodability is expected.",
        valname="AUROC&minus;.5")
    p = os.path.join(out, "demo_measure2.html")
    with open(p, "w") as f:
        f.write(html)
    print(f"[demo] {p}  ({os.path.getsize(p)/1e6:.1f} MB, {len(keys)} views, "
          f"{len(pts)} contacts, {T} bins, {step:.1f} ms/bin)")
    return p


def fig_peak_time(agg: dict, tap: str, out: str, *, win: tuple[float, float],
                  tasks=("onset", "speech"), sessions=None) -> str:
    """When each region's FWER-surviving contacts peak -- the honest version of the latency
    question.

    Restricted to ``enc0`` by construction of the caller: an enc12 time bin has seen the whole
    window through self-attention, so its argmax is not a latency. Even here this is a peak
    time, not an onset time, and a one-bin difference is not a propagation result -- the figure
    prints the bin width so that cannot be forgotten. The measure #1 version of this claim was
    retracted for exactly this reason.

    ``win`` is REQUIRED and is not defaulted to ``viz_anatomy``'s ``WIN_START_S/WIN_END_S``.
    Those constants describe the 2 s reduction; the electrode cache this measure reads may be
    the 1 s one, and the two give 62.5 vs 31.25 ms per bin and a different onset bin. Guessing
    would put a wrong millisecond axis on a latency figure, which is the one error this figure
    exists to avoid making. Pass the span read off the cache that produced these shards.
    """
    import matplotlib.pyplot as plt

    from scripts.neuroprobe.viz_anatomy import NEIGHBOURS, TARGET

    w0, w1 = float(win[0]), float(win[1])
    assert w1 > w0, f"--win must be start,end with end > start (got {win})"
    regions = (NEIGHBOURS[1], TARGET, "middletemporal")
    Ts = {d["T"] for (_, tp, _), d in agg["A"].items() if tp == tap}
    assert len(Ts) == 1, f"{tap}: mixed time axes {Ts} -- refusing to put one ms axis on them"
    T = Ts.pop()
    step = (w1 - w0) / T * 1000.0
    t_ms = w0 * 1000.0 + (np.arange(T) + 0.5) * step
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 3.6), dpi=150,
                            sharey=True)
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        for base in regions:
            v = []
            for (sess, tp, tk), d in agg["A"].items():
                if tp != tap or tk != task or (sessions is not None and sess not in sessions):
                    continue
                m = (row_bases(agg, sess, tap) == base) & d["sig"]
                v.extend(t_ms[d["t_at_peak"][m]])
            if len(v) < 5:
                continue
            v = np.sort(np.asarray(v))
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                    label=f"{base} (n={len(v)}, med {np.median(v):.0f} ms)")
            ax.axvline(np.median(v), lw=.7, ls=":", alpha=.6,
                       color=ax.lines[-1].get_color())
        ax.axvline(0, color="k", lw=.8)
        ax.set_xlabel("peak time (ms from onset)")
        ax.set_title(task, fontsize=10)
        ax.grid(lw=.3, alpha=.4)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel(f"cumulative fraction of FWER-surviving {unit_of(tap)}")
    fig.suptitle(f"MEASURE #2 -- {tap}: peak time of significant {unit_of(tap)}. Window "
                 f"[{w0}, {w1}] s, T={T} => bin width {step:.2f} ms -- a one-bin median gap "
                 f"is NOT a cascade.", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out, f"figE4_peak_time_{tap}.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")
    return p


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

    # 7. the row-major AUROC is bit-identical to the column-major one, including the flat-column
    #    guard. Not "close": these two share the significance threshold, so a drift between them
    #    would move the map without moving anything that reports on the map.
    s2 = rng.normal(size=(120, 40)).astype(np.float32)
    s2[:, 4] = 2.0
    y2 = (rng.random(120) < 0.4).astype(np.int64)
    assert np.array_equal(auroc_cols(s2, y2), auroc_rows(np.ascontiguousarray(s2.T), y2))

    # 8. threading changes nothing at all. The permutation loop runs on however many cores the
    #    cpuset has, so a thread-count-dependent statistic would make the null depend on the
    #    hardware -- exactly as disqualifying as the salted-hash seed bug this file already
    #    guards against. Chunk boundaries land differently for 1, 3 and 7 threads, and 7 does
    #    not divide P, so the ragged last chunk is covered too.
    Zc = contact_major(Z)
    base = cv_auroc_cm(Zc, y)
    for nt in (3, 7):
        with ThreadPoolExecutor(nt) as ex:
            assert np.array_equal(base, cv_auroc_cm(Zc, y, ex, nt)), nt
    # and it must hold under permuted labels too, where the fold split itself moves
    yp = block_permute(y, np.random.default_rng(5), 40)
    with ThreadPoolExecutor(4) as ex:
        assert np.array_equal(cv_auroc_cm(Zc, yp), cv_auroc_cm(Zc, yp, ex, 4))

    print("[self-test] all 8 properties hold", flush=True)


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
    p.add_argument("--band-fdims", default="",
                   help="per-band F_b for records that store no band_fdims, e.g. 7,6,7. Needed "
                        "only for an enc0 tap: the encoder taps share one d so their band offsets "
                        "follow from band_lengths alone. The width check cannot validate this "
                        "(several triples share a total), so pass ONLY a value read off a record "
                        "the same frontend wrote -- never one back-solved from the width")
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
    p.add_argument("--threads", type=int, default=0,
                   help="threads for the permutation loop; 0 = every core in the cpuset. "
                        "argsort releases the GIL, so this is a real ~5-10x")
    p.add_argument("--cov-stride", type=int, default=4,
                   help="window stride for the label-free PC covariance pass")
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--aggregate", action="store_true", help="summarise shards in --out-dir")
    p.add_argument("--figs", action="store_true",
                   help="with --aggregate, render the measure #2 figure set into --fig-dir")
    p.add_argument("--fig-dir",
                   default="results/showcase/3_where_the_signal_lives/singletrial_contacts",
                   help="contact-unit default; the parcel-unit depth run must pass "
                        "results/showcase/3_where_the_signal_lives/singletrial_parcels_by_depth -- the two "
                        "units have different denominators and must never share a folder")
    p.add_argument("--red-dir", default="",
                   help="viz reduction dir; needed only by the brain render, which joins the "
                        "electrode template on it. Omit to skip that one figure.")
    p.add_argument("--bt-root", default=".cache/braintreebank")
    p.add_argument("--win", default="",
                   help="window span as start,end in SECONDS, read off the cache that wrote "
                        "these shards (e.g. -0.5,1.5 for the 2 s reduction). Required by the "
                        "peak-time figure and deliberately not defaulted: the 1 s and 2 s "
                        "caches give different ms/bin and a different onset bin")
    p.add_argument("--brain-tasks", default="onset,speech,local_flow",
                   help="rows of the brain render; keep a visual control in the list or the "
                        "figure shows only the half of the result that is flattering")
    p.add_argument("--demo", action="store_true",
                   help="also write demo_measure2.html: the interactive scrub demo of "
                        "viz_anatomy, driven by measure #2 per contact with the FWER layer. "
                        "Needs --red-dir and --win. Any _elec tap, including encoder taps")
    p.add_argument("--demo-tasks", default="onset,speech,pitch,frame_brightness",
                   help="views offered in the demo's menu, in menu order")
    p.add_argument("--movie", action="store_true",
                   help="also render figE5: figE3 animated over time bins. Needs --red-dir and "
                        "--win, and applies only to an enc0 _elec tap -- see movie_brain_elec")
    p.add_argument("--movie-fps", type=int, default=4)
    p.add_argument("--inference", choices=("maxstat", "fdr"), default="maxstat",
                   help="maxstat = FWER over the whole map (use for any per-contact claim); "
                        "fdr = BH on pooled-null p, more sensitive but expects false positives")
    a = p.parse_args()
    taps = tuple(t for t in a.taps.split(",") if t)
    tasks = tuple(t for t in a.tasks.split(",") if t)
    fdims = tuple(int(f) for f in a.band_fdims.split(",") if f) or None
    assert fdims is None or len(fdims) == 3, "--band-fdims needs one F_b per band"

    if a.aggregate:
        agg = aggregate(a.out_dir, taps=taps, tasks=tasks, alpha=a.alpha,
                        inference=a.inference)
        report(agg, taps=taps, tasks=tasks, alpha=a.alpha)
        if not a.figs:
            return
        os.makedirs(a.fig_dir, exist_ok=True)
        # every cross-tap / cross-family number comes off the matched list; the per-tap survey
        # heatmap does not, and says so in its own title
        sess = matched_sessions(agg, taps, tasks)
        print(f"\n[fig] {len(sess)} of {len({s for (s, _, _) in agg['A']})} sessions have all "
              f"{len(taps)}x{len(tasks)} cells: {sess}")
        fig_calibration(agg, taps, a.fig_dir)
        for tap in taps:
            fig_dissociation(agg, tap, a.fig_dir)
        # the cross-tap and cross-family figures need the matched list; while the array is
        # still filling in enc12's visual tasks that list can be empty, and an unmatched
        # version of these is a cohort artifact rather than a weaker result
        if sess:
            report(agg, taps=taps, tasks=tasks, alpha=a.alpha, only=sess)
            fig_family(agg, taps, a.fig_dir)
        else:
            print("[fig] NO session has all cells yet -- skipping figE2 and the matched report. "
                  "figE1 above is per-tap and unaffected.")
        # enc0 only: an enc12 time bin has seen the whole window through self-attention, so
        # its argmax is not a latency and a ms axis on it would be a lie. And only when the
        # window span is supplied -- see fig_peak_time's docstring for why it is not defaulted.
        win = tuple(float(x) for x in a.win.split(",") if x)
        if len(win) == 2:
            for tap in taps:
                if tap.startswith("enc0"):
                    fig_peak_time(agg, tap, a.fig_dir, win=win, sessions=sess)
        else:
            print("[fig] --win not given: skipping the peak-time figure rather than guessing "
                  "the millisecond axis (the 1 s and 2 s caches differ by 2x per bin)")
        # the dot renders are contact-level by construction: a parcel tap has no per-contact
        # value to paint, so those taps are dropped here rather than at the assert inside
        elec_taps = tuple(t for t in taps if t.endswith("_elec"))
        if a.red_dir and elec_taps:
            from scripts.neuroprobe.viz_anatomy import (dkt_tables, invariant_projection,
                                                        load_template_coords)
            base_of, lobe_of_base = dkt_tables()
            tmpl = load_template_coords(a.red_dir, a.bt_root)
            invariant_projection(tmpl, base_of, lobe_of_base)
            bt = tuple(t for t in a.brain_tasks.split(",") if t)
            for tap in elec_taps:
                fig_brain_elec(agg, tmpl, tap, bt, a.fig_dir)
            if a.demo:
                # unlike the movie this is NOT enc0-only: the demo names the value AUROC-.5 per
                # bin and warns in its own text that a frame is not a latency, so an encoder tap
                # is legitimate to browse -- it is the ms-axis CLAIM that enc12 cannot support.
                if len(win) == 2:
                    build_demo_elec(agg, tmpl, a.fig_dir, taps=elec_taps,
                                    tasks=tuple(t for t in a.demo_tasks.split(",") if t),
                                    win=win, sessions=sess or None)
                else:
                    print("[demo] --win not given: skipping the demo rather than guessing the "
                          "millisecond axis")
            if a.movie:
                # enc0 only, and only with a window: the movie's whole content is a ms axis
                mv = tuple(t for t in elec_taps if t.startswith("enc0"))
                if len(win) == 2 and mv:
                    for tap in mv:
                        movie_brain_elec(agg, tmpl, tap, bt, a.fig_dir, win=win,
                                         fps=a.movie_fps)
                else:
                    print("[fig] --movie needs --win and an enc0 _elec tap: skipping figE5")
        elif a.red_dir:
            print(f"[fig] no _elec tap in {list(taps)}: skipping the brain render and movie "
                  "(a parcel map has no per-contact value to put on a dot)")
        else:
            print("[fig] --red-dir not given: skipping the brain render")
        return

    assert a.cache, "--cache is required unless --self-test or --aggregate"
    files = sorted(f for f in os.listdir(a.cache) if f.endswith(".pt"))
    assert files, f"no .pt records in {a.cache}"
    todo = files if a.session_index < 0 else [files[a.session_index]]
    os.makedirs(a.out_dir, exist_ok=True)
    nthread = a.threads if a.threads > 0 else n_free_cores()
    print(f"[cfg] {nthread} threads for the permutation loop "
          f"({n_free_cores()} cores in this cpuset)", flush=True)
    for fn in todo:
        def dst_of(o: dict) -> str:
            return os.path.join(a.out_dir, f"elec_s{int(o['subject_id'])}"
                                           f"_t{int(o['trial_id'])}_{a.band}.npz")

        def checkpoint(o: dict) -> None:
            # The null CUBES are excluded from checkpoints and written only in the final pass:
            # they are ~7.6 MB each against ~17 kB for the map plus the per-permutation maxima,
            # so including them would make every one of the 30 flushes rewrite the whole file.
            # maxstat inference needs only the maxima, so a checkpoint is fully usable; the
            # cubes buy the liberal --inference fdr companion, which can wait for the end.
            # The temp name must NOT end in .npz. np.savez_compressed appends .npz to any path
            # that lacks it, so a ".tmp" path would land at "<dst>.npz.tmp.npz" -- which ends in
            # .npz, so a concurrent aggregate() globs the half-written file and dies on a
            # BadZipFile. Writing through an open handle suppresses the suffix entirely.
            tmp = dst_of(o) + ".partial"
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **{k: v for k, v in o.items()
                                        if not k.startswith("null/")})
            os.replace(tmp, dst_of(o))    # atomic: aggregate never sees a half-written shard

        out = session_shard(os.path.join(a.cache, fn), taps=taps, tasks=tasks, band=a.band,
                            n_pc=a.pc, n_perm=a.n_perm, perm_block=a.perm_block,
                            chunk=a.chunk, seed=a.seed, store_null=not a.no_store_null,
                            cov_stride=a.cov_stride, band_fdims_override=fdims,
                            nthread=nthread, checkpoint=checkpoint)
        tmp = dst_of(out) + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, dst_of(out))
        print(f"[write] {dst_of(out)}", flush=True)


if __name__ == "__main__":
    main()
