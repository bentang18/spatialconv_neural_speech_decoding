"""R31 — LABEL-EFFICIENCY CURVE: does pretraining buy calibration data?

THE PITCH THIS TESTS. A foundation model's claim to a BCI audience is not "higher ceiling", it is
"same accuracy, less calibration time". So the deliverable is AUROC vs the NUMBER OF LABELLED
TRIALS in the target session, for the pretrained encoder (``enc12_elec``) against its own
zero-parameter depth-0 floor (``enc0_elec``) -- identical features, identical splits, identical
readout, the ONLY difference being whether the encoder was pretrained.

═══ TWO TIERS, AND ONLY ONE IS AT RISK ═════════════════════════════════════════════════════════
  TIER 1  "fewer labels for a target AUROC"   null: ratio == 1.0
          Follows from ANY positive offset plus curve concavity. Near-certain; mildly interesting.
  TIER 2  "the gap is WIDEST when labels are scarce"   null: d(gap)/d(log2 N) == 0
          🔮 PRIOR, STATED BEFORE THE RUN: our own gain law says TIER 2 FAILS. The measured law is
          AUC_d - .5 = k (AUC_0 - .5) with k ~ 1.17-1.28, i.e. the gap is PROPORTIONAL to enc0's
          headroom -- so it should be WIDEST AT LARGE N, the opposite of the FM story. A negative
          slope here contradicts the gain law and is the real result; a positive slope confirms it
          and means we may claim TIER 1 ONLY.
This is why the run is worth its compute: the decision (may we make the calibration-time pitch at
all?) genuinely turns on the sign, and the sign is not predictable from what we already know.

═══ WHAT IS AND IS NOT VARIED ══════════════════════════════════════════════════════════════════
The readout is the LOCKED frozen-ridge contract, untouched: std z-score on train stats, dual ridge,
one fp64 eigh, λ = argmax over the 25-point val grid. See reference-frozen-ridge-readout-contract.
Every primitive is IMPORTED from v3_board_readout rather than reimplemented, so the curve cannot
drift from the published board by an accident of restatement.

The ONE thing this file adds is a row subsample of the train half (and, in the honest column, of
the val half with it).

═══ ⚠️ THE SPLIT IS KFold(2), shuffle=False ════════════════════════════════════════════════════
neuroprobe/config.py:40 (N_FOLDS=2) and train_test_splits.py:221 (shuffle=False). So a fold's
train IS one contiguous half of the session and val/test are halves of the other
(train_test_splits.py:231-235). "N trials" therefore has two readings:

  RANDOM (this file, primary)     N trials drawn from anywhere in the calibration block.
  CONTIGUOUS PREFIX (--contiguous) the first N trials recorded. The honest BCI scenario, BUT with
      k=2 only ONE fold has train preceding test; the other decodes the past. Available as a flag,
      deliberately NOT the overnight number -- it is a different question with half its folds
      unphysical, and mixing them into one mean would hide that.

═══ TWO COLUMNS OFF ONE EIGENDECOMPOSITION ═════════════════════════════════════════════════════
``_lam_grid`` takes a free-form dict of named eval sets (v3_board_readout.py:681-720) and
``_linear_grams`` builds each name's kernel independently (:604-605), so a third eval set perturbs
NOTHING about the fit. That buys both readings of "labelled data" from one fit:

  col "both"      λ selected on the SUBSAMPLED val  -- subsample train AND val. The honest
                  calibration budget: val is labelled data the user also has to sit through.
  col "trainonly" λ selected on the FULL val        -- subsample train only. What most papers do
                  implicitly, reported so the difference is visible instead of assumed away.

TEST IS NEVER SUBSAMPLED. Both columns are scored on the full 875-row test half.

⚠️ ``grams=`` is deliberately NOT passed even though the val-subset kernel is a row slice of the
full-val kernel. Passing grams FORCES the dual branch (:689-691); letting ``_lam_grid`` pick its
own branch is what keeps the N=full anchor BIT-comparable to the published ``_ws_cell``. A free
GEMM is not worth turning the anchor check into an approximate one.

═══ INVARIANTS, ASSERTED AND PRINTED ═══════════════════════════════════════════════════════════
  1. ANCHOR. At N=full the curve must reproduce the published board cell, computed by calling the
     REAL ``_ws_cell`` on the same record. If it does not, nothing else in the run means anything.
  2. PAIRING. The draw is made ONCE per (task, fold, N, seed) and its digest is asserted identical
     for every tap -- enc0 and enc12 see the SAME N trials, which is what makes this a paired test.
  3. BALANCE. Every draw is stratified at the parent fold's class balance, >=1 per class.
  4. CENSUS. n_train / n_val / n_pos / n_neg printed per point, so a degenerate small-N cell is
     visible in the log rather than silently NaN.

Usage (one array task per WS session; merge afterwards):
  python v3_board_samplecurve.py --cache-dir DIR --tag TAG --index I --shard-dir D
  python v3_board_samplecurve.py --mode merge --shard-dir D --out curve.json
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

# Import the readout as the SAME module object the tests and the launcher see. Importing it twice
# under two names would give this file its own copies of NORMS/LAM_CV/RBF, so a flag set in one
# would silently not apply in the other -- the exact class of drift the file-only deploy rule
# exists to prevent. Package path first (pytest, `python -m`), bare path only as the fallback for
# `python scripts/neuroprobe/v3_board_samplecurve.py`.
try:
    from scripts.neuroprobe import v3_board_readout as R
except ImportError:                                                             # bare-script run
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import v3_board_readout as R                                    # type: ignore[no-redef]

BOARD_TASKS = R.BOARD_TASKS
ELEC_TAPS = R.ELEC_TAPS
LITE_SESSIONS = R.LITE_SESSIONS
CS_TEST_CELLS = R.CS_TEST_CELLS
CS_TRAIN_ANCHOR = R.CS_TRAIN_ANCHOR
_feat, _finite, _have = R._feat, R._finite, R._have
_lam_grid, _load, _select_lam = R._lam_grid, R._load, R._select_lam
_standardize_inplace, _ws_cell = R._standardize_inplace, R._ws_cell
_cs_cell, _parcel_cols = R._cs_cell, R._parcel_cols

# The two taps the additive-vs-multiplicative test needs, in each regime's OWN unit. CS is
# parcel-mean because electrode identity is not shared across subjects (v3_board_readout.py:113-114).
CURVE_TAPS = {"ws": ("enc0_elec", "enc12_elec"), "cs": ("enc0", "enc12")}
SHARD_PREFIX = {"ws": "wscurve", "cs": "cscurve"}

# Powers of two so "how many doublings of calibration data" reads straight off the x axis, and a
# log-linear fit is a straight line. 16 is ~16 s of calibration at 1 s trials; below that a
# stratified draw stops being meaningful (a class can reach 1 row).
N_GRID = (16, 32, 64, 128, 256, 512, 1024)

# CS trains on the WHOLE anchor session (v3_board_readout.py:1301, `_finite` over arange), not on a
# KFold half, so its parent is ~2x a WS train half and the grid gets one more doubling. Points above
# the parent are filtered out at runtime (`g < len(tr)`), so an unreachable entry costs nothing.
N_GRID_CS = N_GRID + (2048,)

# Seeds TAPER because that is where the variance is. At N=16 which 16 trials you drew dominates;
# at N=full there is no draw at all and one pass is exact, not an estimate. Flat 5 seeds costs ~8x
# a full board readout (eval kernels dominate, not the Gram); this schedule lands at ~3-4x.
SEEDS_FOR_N = {16: 5, 32: 5, 64: 5, 128: 5, 256: 3, 512: 3, 1024: 3, 2048: 3}
FULL = "full"           # the anchor point: no subsample, 1 pass, must equal the published cell

COLUMNS = ("both", "trainonly")


def _x_order(k):
    """x-axis sort key. The grid mixes ints with the `full` sentinel, so a bare sorted() would
    raise TypeError comparing str to int — always order through this."""
    return (1, 0) if k == FULL else (0, int(k))


def _digest(idx) -> str:
    """Stable across processes and runs -- unlike hash(), which is PYTHONHASHSEED-salted."""
    return hashlib.blake2b(np.asarray(idx, dtype=np.int64).tobytes(), digest_size=8).hexdigest()


def _rng(session, task, fold, n, seed):
    """Draw seed derived from the CELL, not from a global counter, so any point is reproducible in
    isolation and re-running one shard cannot shift another's draws."""
    key = f"{session[0]}:{session[1]}|{task}|{fold}|{n}|{seed}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))


def _strat_draw(y, rows, n, rng, contiguous=False):
    """n rows from `rows`, stratified at the parent's class balance, >=1 per class.

    Stratification is not a nicety: an unstratified draw of 16 from an unbalanced task can come
    back single-class, and `auroc` returns NaN on a single-class eval half -- which would silently
    delete the hardest points of the curve rather than reporting them.

    Returns rows in ASCENDING ORDER so the gather is a monotone index and the N=full draw is
    element-for-element the parent array.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if n >= rows.size:
        return rows
    pos = rows[y[rows] > 0]
    neg = rows[y[rows] <= 0]
    if pos.size == 0 or neg.size == 0:
        return np.sort(rows[:n] if contiguous else rng.choice(rows, size=n, replace=False))
    # Round to the parent balance, then clamp so BOTH classes survive and neither is over-drawn.
    n_pos = int(round(n * pos.size / rows.size))
    n_pos = max(1, min(n_pos, pos.size, n - 1))
    n_neg = n - n_pos
    if n_neg > neg.size:                      # rare: parent is extremely skewed toward positives
        n_neg = neg.size
        n_pos = min(n - n_neg, pos.size)
    if contiguous:
        take = np.concatenate([pos[:n_pos], neg[:n_neg]])
    else:
        take = np.concatenate([rng.choice(pos, size=n_pos, replace=False),
                               rng.choice(neg, size=n_neg, replace=False)])
    return np.sort(take)


def _fit_point(rec, tap, tr_s, va, va_pos, te, y, want_sub_val):
    """One (tap, N, seed) fit → {column: selected-λ result}.

    ⚠️ ORDER IS LOAD-BEARING, exactly as in ``_run_norms``: the val subset is sliced AFTER the
    in-place standardization, so it carries the same train-fitted μ/σ as everything else instead
    of being standardized twice or with its own statistics.
    """
    z_tr = _feat(rec, tap, tr_s)
    z_va, z_te = _feat(rec, tap, va), _feat(rec, tap, te)
    a, (b, c) = _standardize_inplace(z_tr, [z_va, z_te])
    evals = {"valfull": (b, y[va]), "test": (c, y[te])}
    if want_sub_val:
        # Positional slice of the ALREADY-standardized full-val block: no second gather, no second
        # standardize, and by construction the same rows the "both" column is entitled to.
        evals["val"] = (b[va_pos], y[va[va_pos]])
    g = _lam_grid(a, y[tr_s], evals)
    out = {"trainonly": _select_lam({"val": g["valfull"], "test": g["test"]})}
    # At N=full the subsample IS the parent, so the two columns are the same fit read twice --
    # aliasing rather than recomputing keeps that identity exact instead of merely near.
    out["both"] = (_select_lam({"val": g["val"], "test": g["test"]}) if want_sub_val
                   else dict(out["trainonly"]))
    return out


def _ws_curve_cell(rec, session, task, taps, contiguous=False):
    """One (session, task) → curve points over N x seed x tap x column, plus the census."""
    y = np.asarray(rec["labels"][task], dtype=np.float64)
    pts, census = [], []
    for fold, sp in sorted(rec["ws_split"][task].items()):
        tr, va, te = (_finite(y, sp["train"]), _finite(y, sp["val"]), _finite(y, sp["test"]))
        if len(tr) < 2 or len(te) < 2:
            continue
        # `full` LAST so the log reads bottom-up as "and here is the anchor".
        for n in [g for g in N_GRID if g < len(tr)] + [None]:
            is_full = n is None
            nn = len(tr) if is_full else int(n)
            for seed in range(1 if is_full else SEEDS_FOR_N[int(n)]):
                rng = _rng(session, task, fold, nn, seed)
                tr_s = _strat_draw(y, tr, nn, rng, contiguous)
                # Val is subsampled at the SAME FRACTION as train -- calibration data is one
                # budget, not two. Floor of 4 keeps a val half scoreable at all (auroc needs >=2
                # rows and both classes present).
                frac = len(tr_s) / len(tr)
                n_va = len(va) if is_full else max(4, int(round(frac * len(va))))
                va_s = _strat_draw(y, va, n_va, _rng(session, task, fold, nn, 1000 + seed),
                                   contiguous)
                # Positions of the val subset WITHIN `va` -- `_fit_point` slices the standardized
                # val block positionally, so it needs offsets, not row ids.
                #
                # NOT searchsorted: `va` is NOT guaranteed ascending. It is _finite(y, split["val"]),
                # which preserves whatever order the cached split stored, and the board readout never
                # noticed because it only ever uses va via _feat(rec, enc, va) and y[va] -- both
                # gathered with the SAME index, so both are order-agnostic. searchsorted silently
                # requires sorted input and returns garbage positions otherwise.
                # Order does not matter downstream either (features and labels are gathered with the
                # same va_pos), so the invariant is a SET equality, not an elementwise one.
                va_pos = np.flatnonzero(np.isin(va, va_s))
                assert va_pos.size == va_s.size and np.array_equal(
                    np.sort(va[va_pos]), np.sort(va_s)), "val subset is not a subset of val"
                d_tr, d_va = _digest(tr_s), _digest(va_s)
                want_sub = len(va_s) < len(va)
                for tap in taps:
                    if not _have(rec, tap):
                        continue
                    # INVARIANT 2: every tap fits the SAME rows. Structural here (the draw is made
                    # once, outside this loop) but asserted anyway -- the whole paired analysis is
                    # void if it ever stops being true, and a structural guarantee that is never
                    # checked is how it stops being true.
                    assert _digest(tr_s) == d_tr and _digest(va_s) == d_va, "draw drifted across taps"
                    res = _fit_point(rec, tap, tr_s, va, va_pos, te, y, want_sub)
                    for col in COLUMNS:
                        pts.append({"task": task, "fold": int(fold), "tap": tap, "col": col,
                                    "n": nn, "n_is_full": is_full, "seed": seed,
                                    "test": res[col]["test"],
                                    "lam_pinned": bool(res[col]["lam_pinned"])})
                yb = (y[tr_s] > 0)
                # `n_is_full` is NOT redundant with n: after the per-task label filter a train
                # half can land exactly on a grid value, and then "N=128" and "the anchor" are
                # indistinguishable by n alone.
                census.append({"task": task, "fold": int(fold), "n": nn, "seed": seed,
                               "n_is_full": is_full,
                               "n_train": len(tr_s), "n_val": len(va_s), "n_test": len(te),
                               "n_pos": int(yb.sum()), "n_neg": int((~yb).sum()),
                               "parent_bal": float((y[tr] > 0).mean()),
                               "draw_bal": float(yb.mean())})
    return pts, census


def _cs_fit_point(anchor_rec, test_rec, tap, tr_s, a_idx, va, te, t_idx, y_a, y_t):
    """One (tap, N, seed) cross-subject fit → the selected-λ result for the reported `std` norm.

    Mirrors the "std" branch of ``_run_norms`` (v3_board_readout.py:1238-1245) followed by
    ``_grid_cells``' ``_select_lam``: standardize on the ANCHOR's train stats, one λ grid, select on
    the test cell's val half. `grams=` is omitted for the same reason as the WS path — letting
    ``_lam_grid`` choose its own branch is what keeps the N=full anchor bit-comparable to ``_cs_cell``.
    """
    z_tr = _feat(anchor_rec, tap, tr_s, a_idx)
    z_va, z_te = _feat(test_rec, tap, va, t_idx), _feat(test_rec, tap, te, t_idx)
    a, (b, c) = _standardize_inplace(z_tr, [z_va, z_te])
    g = _lam_grid(a, y_a[tr_s], {"val": (b, y_t[va]), "test": (c, y_t[te])})
    return _select_lam({"val": g["val"], "test": g["test"]})


def _cs_curve_cell(anchor_rec, test_rec, task, taps, contiguous=False):
    """One (CS test cell, task) → curve points over N x seed x tap, plus the census.

    ═══ WHAT N MEANS HERE, AND WHY IT IS NOT THE WS QUESTION ════════════════════════════════════
    CS fits the ANCHOR session (2,4) and scores a held-out subject, so N is "labelled trials from
    ONE DONOR patient", not "calibration trials from the target". The target contributes no training
    rows at any N.

    ⚠️ VAL IS NEVER SUBSAMPLED, unlike the WS curve's "both" column. val here is the TEST CELL's
    ``cs_split`` val half (v3_board_readout.py:1302) — it belongs to the target subject and is the
    protocol's, not a calibration budget the donor controls. Subsampling it would answer a question
    the benchmark does not pose. Points are therefore emitted under col ``trainonly`` ONLY, which is
    what that column already means: subsample train, select λ on the full val.

    There are no folds: CS has one train set and one val/test pair, so `fold` is pinned to 0.
    """
    y_a = np.asarray(anchor_rec["labels"][task], dtype=np.float64)
    y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
    tr = _finite(y_a, np.arange(len(y_a)))
    va = _finite(y_t, test_rec["cs_split"][task]["val"])
    te = _finite(y_t, test_rec["cs_split"][task]["test"])
    pts, census = [], []
    if len(tr) < 2 or len(te) < 2:
        return pts, census
    a_idx, t_idx, common = _parcel_cols(anchor_rec, test_rec)
    if common.size == 0:
        return pts, census
    for n in [g for g in N_GRID_CS if g < len(tr)] + [None]:
        is_full = n is None
        nn = len(tr) if is_full else int(n)
        for seed in range(1 if is_full else SEEDS_FOR_N[int(n)]):
            # KEYED ON THE ANCHOR, NOT THE TEST CELL. The donor session is the SAME for all 10 cells,
            # so keying on the cell would silently average over 10x more donor subsamples than the
            # claim is about -- a different, lower-variance estimand. One draw, ten patients.
            rng = _rng(CS_TRAIN_ANCHOR, task, 0, nn, seed)
            tr_s = _strat_draw(y_a, tr, nn, rng, contiguous)
            d_tr = _digest(tr_s)
            for tap in taps:
                if not (_have(anchor_rec, tap) and _have(test_rec, tap)):
                    continue
                # INVARIANT 2, as in the WS path: every tap fits the SAME donor rows.
                assert _digest(tr_s) == d_tr, "draw drifted across taps"
                res = _cs_fit_point(anchor_rec, test_rec, tap, tr_s, a_idx, va, te, t_idx, y_a, y_t)
                pts.append({"task": task, "fold": 0, "tap": tap, "col": "trainonly",
                            "n": nn, "n_is_full": is_full, "seed": seed,
                            "test": res["test"], "lam_pinned": bool(res["lam_pinned"])})
            yb = (y_a[tr_s] > 0)
            census.append({"task": task, "fold": 0, "n": nn, "seed": seed, "n_is_full": is_full,
                           "n_train": len(tr_s), "n_val": len(va), "n_test": len(te),
                           "n_pos": int(yb.sum()), "n_neg": int((~yb).sum()),
                           "parent_bal": float((y_a[tr] > 0).mean()),
                           "draw_bal": float(yb.mean())})
    return pts, census


def _cs_anchor_check(anchor_rec, test_rec, task, taps, pts) -> list:
    """INVARIANT 1 for CS — the N=full point must BE the published ``_cs_cell``.

    Same licence the WS curve runs under: computed by calling the REAL cell function, never by
    restating it. Note ``_cs_cell`` returns ``_grid_cells`` UNFOLDED (no fold averaging), so the
    comparison is against a single value rather than a mean over folds.
    """
    pub = _cs_cell(anchor_rec, test_rec, task, taps)["cells"]
    rows = []
    for tap in taps:
        key = f"{tap}|std"
        if key not in pub:
            continue
        mine = [p["test"] for p in pts
                if p["task"] == task and p["tap"] == tap and p["n_is_full"]]
        if not mine:
            continue
        rows.append({"task": task, "tap": tap, "col": "trainonly",
                     "published": float(pub[key]["test"]),
                     "curve": float(np.nanmean(mine)),
                     "absdiff": float(abs(np.nanmean(mine) - pub[key]["test"]))})
    return rows


def _anchor_check(rec, task, taps, pts) -> list:
    """INVARIANT 1 — the N=full point must BE the published board cell.

    Computed by calling the REAL ``_ws_cell`` on this record, not by restating it. Any drift means
    the subsample harness perturbed the fit itself, and every other point on the curve inherits
    that drift, so this is the check that licenses the run.
    """
    pub = _ws_cell(rec, task, taps)["cells"]
    rows = []
    for tap in taps:
        key = f"{tap}|std"
        if key not in pub:
            continue
        for col in COLUMNS:
            mine = [p["test"] for p in pts
                    if p["task"] == task and p["tap"] == tap and p["col"] == col and p["n_is_full"]]
            if not mine:
                continue
            # _ws_cell averages the folds; the curve keeps them separate, so average to compare.
            rows.append({"task": task, "tap": tap, "col": col,
                         "published": float(pub[key]["test"]),
                         "curve": float(np.nanmean(mine)),
                         "absdiff": float(abs(np.nanmean(mine) - pub[key]["test"]))})
    return rows


ANCHOR_TOL = 1e-9
# Not a tolerance so much as a floor on "exact": the anchor path shares the eigendecomposition and
# the eval kernels with the published one (adding an eval NAME cannot change another name's
# arithmetic), so the honest expectation is 0.0 and anything above this is a real divergence.


def _anchor_verdict(rows) -> list:
    """Offending anchor rows, or [] if the anchor holds. Raises on an EMPTY row list.

    The empty case is the one that has to raise rather than return []: no rows means nothing was
    compared, and a guard that reports "no violations" because it checked nothing is worse than no
    guard at all. Every other bad state shows up as a row with a large absdiff."""
    if not rows:
        raise AssertionError(
            "🔴 ANCHOR VACUOUS: zero (tap, col) comparisons were made, so the N=full point was "
            "never checked against the published _ws_cell. Nothing here is licensed.")
    return [r for r in rows if r["absdiff"] >= ANCHOR_TOL]


def _shard(cache_dir, tag, index, regime, taps, contiguous=False) -> dict:
    """One array task: one WS session, or one CS test cell against the fixed anchor."""
    if regime == "cs":
        cell = CS_TEST_CELLS[index]
        anchor_rec = _load(cache_dir, CS_TRAIN_ANCHOR, tag, mmap=False)
        test_rec = _load(cache_dir, cell, tag, mmap=False)
        curve_fn = lambda task: _cs_curve_cell(anchor_rec, test_rec, task, taps, contiguous)
        check_fn = lambda task, p: _cs_anchor_check(anchor_rec, test_rec, task, taps, p)
        published = "_cs_cell"
    else:
        cell = LITE_SESSIONS[index]
        rec = _load(cache_dir, cell, tag, mmap=False)
        curve_fn = lambda task: _ws_curve_cell(rec, cell, task, taps, contiguous)
        check_fn = lambda task, p: _anchor_check(rec, task, taps, p)
        published = "_ws_cell"

    pts, census, anchor = [], [], []
    for i, task in enumerate(BOARD_TASKS):
        p, c = curve_fn(task)
        pts += p
        census += c
        a = check_fn(task, p)
        anchor += a
        # FAIL FAST, on the FIRST task. A broken anchor invalidates every point in every shard, so
        # discovering it at the end of a 12-way array costs ~20x what discovering it here does.
        bad = _anchor_verdict(a)
        if i == 0 and bad:
            raise AssertionError(
                f"🔴 ANCHOR FAILED on the first task ({task}): the N=full point does not reproduce "
                f"the published {published}. The subsample harness has perturbed the FIT, so no point "
                f"on this curve is on the board protocol. Worst: {max(bad, key=lambda r: r['absdiff'])}")
        print(f"  [{task}] {len(p)} points  anchor max|diff| "
              f"{max(r['absdiff'] for r in a):.2e}", flush=True)
    return {"kind": SHARD_PREFIX[regime], "regime": regime,
            "name": f"S{cell[0]}T{cell[1]}", "tag": tag,
            "contiguous": bool(contiguous), "points": pts, "census": census, "anchor": anchor}


# ── merge + decision ──────────────────────────────────────────────────────────────────────────

def _curve(pts, tap, col):
    """{N: macro AUROC} — mean over seeds, then folds, then the 15 tasks, then the cells.

    Aggregation order matters and this one matches the board's: a cell is a (session, task) and the
    macro is the mean over the 15 tasks of the cohort mean, so no task with more folds or more
    seeds can weigh more than another.
    """
    by = {}
    for p in pts:
        if p["tap"] != tap or p["col"] != col:
            continue
        by.setdefault(p["n_bucket"], {}).setdefault(p["task"], {}).setdefault(p["cell"], []).append(p["test"])
    out = {}
    for n, tasks in by.items():
        per_task = [np.nanmean([np.nanmean(v) for v in cells.values()]) for cells in tasks.values()]
        out[n] = float(np.nanmean(per_task))
    return {k: out[k] for k in sorted(out, key=_x_order)}


def _reach(curve, target):
    """Smallest N at which `curve` reaches `target`, log2-interpolated between bracketing points.

    Returns None when the curve never gets there -- reported as such, never silently clipped to the
    grid end, because "did not reach" and "reached at the last point" are different results.
    """
    ns = sorted(curve)
    for i, n in enumerate(ns):
        if curve[n] >= target:
            if i == 0:
                return float(n)
            n0, a0, a1 = ns[i - 1], curve[ns[i - 1]], curve[n]
            if a1 <= a0:
                return float(n)
            f = (target - a0) / (a1 - a0)
            return float(2.0 ** (np.log2(n0) + f * (np.log2(n) - np.log2(n0))))
    return None


def _merge(shard_dir, regime="ws") -> dict:
    pts, census, anchor, tags, contig = [], [], [], set(), set()
    paths = sorted(glob.glob(f"{shard_dir}/{SHARD_PREFIX[regime]}_*.json"))
    if not paths:
        raise SystemExit(f"🔴 no {SHARD_PREFIX[regime]}_*.json shards in {shard_dir} — "
                         f"nothing to merge for regime '{regime}'")
    for path in paths:
        with open(path) as f:
            sh = json.load(f)
        # A ws and a cs shard must NEVER land in the same merge: the taps differ, the train set
        # differs, and averaging them would produce a number of no regime at all.
        assert sh.get("regime", "ws") == regime, f"{path} is regime {sh.get('regime')}, want {regime}"
        tags.add(sh["tag"])
        contig.add(sh["contiguous"])
        for p in sh["points"]:
            # A cell is a (session, task); `n_bucket` folds `full` into a single x position so the
            # anchor is comparable across cells whose train halves differ in length after the
            # per-task label filter.
            p["cell"] = sh["name"]
            p["n_bucket"] = FULL if p["n_is_full"] else p["n"]
            pts.append(p)
        census += [dict(c, cell=sh["name"]) for c in sh["census"]]
        anchor += [dict(a, cell=sh["name"]) for a in sh["anchor"]]
    return {"points": pts, "census": census, "anchor": anchor, "regime": regime,
            "tags": sorted(tags), "contiguous": sorted(contig)}


def _subject(cell_name: str) -> str:
    """'S3T1' -> 'S3'. The bootstrap resamples SUBJECTS, not cells: two sessions of one patient are
    not two independent draws from the population the claim generalizes over."""
    return cell_name.split("T")[0]


def _addmult(pts, tap0, tap12, col, nboot=2000, seed=0) -> dict:
    """THE TEST. Regress (AUC_enc12 - .5) on (AUC_enc0 - .5) across the N sweep: y = a + k·x.

    ═══ THE DECISION MAP, AND WHAT EACH OUTCOME WOULD MEAN ═══════════════════════════════════════
      a > 0, k ≈ 1  →  ADDITIVE. Pretraining supplies a fixed increment that the readout cannot buy
                       with more labels. The curve is SHIFTED UP.
      a ≈ 0, k > 1  →  MULTIPLICATIVE. Pretraining is pure GAIN on whatever signal the frontend
                       already exposes — it amplifies, it does not add. Where the frontend has
                       nothing (x=0), pretraining gives nothing.
      both nonzero  →  BOTH mechanisms present; report both, claim neither alone.
      neither       →  UNDERPOWERED at this grid — no claim.

    NULLS, stated before the run: a = 0 and k = 1. Both are two-sided.

    Error bars come from a BOOTSTRAP OVER SUBJECTS (2000 resamples), not from OLS residual d.o.f.:
    the sweep's points share cells, so the N-grid points are NOT independent observations and an OLS
    CI over them would be far too tight. The x-axis is measured too (enc0 is an estimate), so `a` is
    EIV-biased toward zero if anything — which makes a nonzero `a` the conservative finding and a
    null `a` the one that has to be read carefully.
    """
    by = {}
    for p in pts:
        if p["col"] != col or p["tap"] not in (tap0, tap12) or p["n_is_full"]:
            continue
        by.setdefault(_subject(p["cell"]), {}).setdefault(p["n_bucket"], {}) \
          .setdefault(p["task"], {}).setdefault((p["cell"], p["tap"]), []).append(p["test"])

    subs = sorted(by)
    ns = sorted({n for s in subs for n in by[s]}, key=_x_order)

    def fit(draw):
        xs, ys = [], []
        for n in ns:
            # Aggregation order is _curve's EXACTLY -- seeds, then cells, then tasks -- so that
            # fit(subs) reproduces the printed curve rather than a second, differently-weighted
            # estimate of it. A subject drawn twice contributes its cells twice; that IS the
            # bootstrap weighting, and it is why the pooling happens here and not per subject.
            acc = {tap0: {}, tap12: {}}
            for s in draw:
                for task, cellmap in by[s].get(n, {}).items():
                    for (_cell, tap), vals in cellmap.items():
                        acc[tap].setdefault(task, []).append(np.nanmean(vals))
            if not acc[tap0] or not acc[tap12]:
                continue
            macro = {t: np.nanmean([np.nanmean(v) for v in acc[t].values()]) for t in acc}
            xs.append(macro[tap0] - .5)
            ys.append(macro[tap12] - .5)
        if len(xs) < 3:
            return None, None
        k, a = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
        return float(a), float(k)

    a_hat, k_hat = fit(subs)
    rng = np.random.default_rng(seed)
    boots = [fit(list(rng.choice(subs, size=len(subs), replace=True))) for _ in range(nboot)]
    A = np.asarray([b[0] for b in boots if b[0] is not None])
    K = np.asarray([b[1] for b in boots if b[1] is not None])
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"a": a_hat, "k": k_hat, "a_ci": ci(A), "k_ci": ci(K),
            "n_points": len(ns), "n_subjects": len(subs), "n_boot": len(A)}


def _report(m) -> None:
    regime = m.get("regime", "ws")
    tap0, tap12 = CURVE_TAPS[regime]
    grid = N_GRID_CS if regime == "cs" else N_GRID
    cols = COLUMNS if regime == "ws" else ("trainonly",)
    want_cells = 10 if regime == "cs" else 12
    pts, anchor = m["points"], m["anchor"]
    cells = sorted({p["cell"] for p in pts})
    tasks = sorted({p["task"] for p in pts})
    print("\n" + "=" * 96)
    print(f"R31 LABEL-EFFICIENCY CURVE [{regime.upper()}]   tags={m['tags']}  "
          f"contiguous={m['contiguous']}  taps={tap0} vs {tap12}")
    print(f"units: {len(cells)} cells x {len(tasks)} tasks = {len(cells) * len(tasks)} "
          f"(board convention is {want_cells * 15}; NEVER read below it)")
    if regime == "cs":
        print(f"N = labelled trials from the DONOR session S{CS_TRAIN_ANCHOR[0]}T{CS_TRAIN_ANCHOR[1]}. "
              f"The target subject contributes ZERO training rows at every N.")
    print("=" * 96)

    # ── INVARIANT 1 first: if the anchor is wrong, no number below it is worth reading ──────────
    print("\n=== INVARIANT 1 · N=full MUST reproduce the published board cell ===")
    if not anchor:
        print("  🔴 NO ANCHOR ROWS — cannot license this run")
    else:
        worst = max(anchor, key=lambda a: a["absdiff"])
        for tap in sorted({a["tap"] for a in anchor}):
            d = [a["absdiff"] for a in anchor if a["tap"] == tap]
            print(f"  {tap:<12} max |curve - published| = {max(d):.2e} over {len(d)} (task, col)")
        ok = worst["absdiff"] < 1e-9
        print(f"  worst: {worst['cell']} {worst['task']} {worst['tap']} {worst['col']} "
              f"published {worst['published']:.6f} vs curve {worst['curve']:.6f}")
        print(f"  VERDICT: {'✅ ANCHOR EXACT — curve is on the published protocol' if ok else '🔴 ANCHOR DRIFTED — DO NOT QUOTE THIS RUN'}")

    # ── INVARIANT 3/4 census: is any point degenerate? ──────────────────────────────────────────
    print("\n=== INVARIANT 3/4 · draw census ===")
    cs = m["census"]
    for n in sorted({c["n"] for c in cs if c["n"] <= max(grid)}):
        sub = [c for c in cs if c["n"] == n]
        mn_pos = min(c["n_pos"] for c in sub)
        mn_neg = min(c["n_neg"] for c in sub)
        bal = max(abs(c["draw_bal"] - c["parent_bal"]) for c in sub)
        print(f"  N={n:>5}  min n_pos {mn_pos:>4}  min n_neg {mn_neg:>4}  "
              f"max |draw_bal - parent_bal| {bal:.3f}  median n_val "
              f"{int(np.median([c['n_val'] for c in sub])):>4}")

    for col in cols:
        c12, c0 = _curve(pts, tap12, col), _curve(pts, tap0, col)
        if not c12 or not c0:
            continue
        if regime == "cs":
            label = ("subsample the DONOR's train rows; λ on the target cell's own val half — "
                     "val belongs to the protocol, not to a calibration budget")
        else:
            label = ("subsample train AND val — the honest calibration budget" if col == "both"
                     else "subsample train only, λ on the full val — the conventional reading")
        print(f"\n{'=' * 96}\nCOLUMN '{col}' · {label}\n{'=' * 96}")
        print(f"  {'N':>6}  {'enc0':>8}  {'enc12':>8}  {'gap':>8}")
        for n in sorted(c12, key=_x_order):
            if n not in c0:
                continue
            print(f"  {str(n):>6}  {c0[n]:8.4f}  {c12[n]:8.4f}  {c12[n] - c0[n]:+8.4f}")

        # ── TIER 1 ─────────────────────────────────────────────────────────────────────────────
        target = c0.get(FULL)
        n_full = int(np.median([p["n"] for p in pts if p["tap"] == tap0 and p["n_is_full"]]))
        print(f"\n  TIER 1 · labels to reach enc0's full-data AUROC ({target:.4f})   NULL: ratio = 1.0")
        got = _reach({k: v for k, v in c12.items() if k != FULL}, target)
        if got is None:
            print(f"    enc12 does NOT reach it within N<={max(grid)} ⇒ NO TIER-1 CLAIM at this grid")
        else:
            print(f"    enc0  needs ~{n_full} trials (its full train half)")
            print(f"    enc12 needs ~{got:.0f} trials  ⇒  RATIO {n_full / got:.2f}x  "
                  f"({n_full}s → {got:.0f}s of calibration at 1 s trials)")

        # ── TIER 2 ─────────────────────────────────────────────────────────────────────────────
        xs = [n for n in c12 if n != FULL and n in c0]
        if len(xs) >= 3:
            lg = np.log2(np.asarray(xs, dtype=np.float64))
            gap = np.asarray([c12[n] - c0[n] for n in xs], dtype=np.float64)
            slope = float(np.polyfit(lg, gap, 1)[0])
            print(f"\n  TIER 2 · d(gap)/d(log2 N) = {slope:+.5f} per doubling   NULL: 0.0")
            print(f"    gain law (k~1.17-1.28) PREDICTS a POSITIVE slope — gap widest at LARGE N.")
            if slope > 0:
                print("    ⇒ POSITIVE: gain law CONFIRMED. Pretraining does NOT preferentially help")
                print("      the scarce-label regime. 🚫 CLAIM TIER 1 ONLY — do NOT write 'especially")
                print("      valuable when labels are scarce'.")
            else:
                print("    ⇒ NEGATIVE: gap WIDENS as labels get scarce — this CONTRADICTS the gain")
                print("      law and IS the foundation-model result. Re-check the gain law before")
                print("      quoting both in the same paper.")

        # ── TIER 3 · ADDITIVE vs MULTIPLICATIVE ────────────────────────────────────────────────
        am = _addmult(pts, tap0, tap12, col)
        if am["a"] is not None:
            a, (alo, ahi) = am["a"], am["a_ci"]
            k, (klo, khi) = am["k"], am["k_ci"]
            print(f"\n  TIER 3 · (enc12-.5) = a + k·(enc0-.5) over the N sweep   NULLS: a=0, k=1")
            print(f"    {am['n_points']} N-points, {am['n_subjects']} subjects, "
                  f"{am['n_boot']} subject-bootstraps")
            print(f"    a = {a:+.4f}  95% CI [{alo:+.4f}, {ahi:+.4f}]   (null 0)")
            print(f"    k = {k:.4f}  95% CI [{klo:.4f}, {khi:.4f}]   (null 1)")
            a_sig, k_sig = not (alo <= 0 <= ahi), not (klo <= 1 <= khi)
            verdict = {(True, False): "ADDITIVE — a fixed increment labels cannot buy",
                       (False, True): "PURE MULTIPLICATIVE — gain on what the frontend exposes, "
                                      "nothing where it exposes nothing",
                       (True, True): "BOTH — report both terms, claim neither alone",
                       (False, False): "UNDERPOWERED at this grid — NO CLAIM"}[(a_sig, k_sig)]
            print(f"    ⇒ {verdict}")

        # Paired sign test at each N, over the 180 units, so the curve carries its own error bar.
        print(f"\n  paired enc12 - enc0 over units (a unit is one (session, task), seeds averaged):")
        for n in sorted(c12, key=_x_order):
            pair = {}
            for p in pts:
                if p["col"] != col or p["n_bucket"] != n:
                    continue
                pair.setdefault((p["cell"], p["task"], p["tap"]), []).append(p["test"])
            keys = {(c, t) for (c, t, _) in pair}
            d = [np.nanmean(pair[(c, t, tap12)]) - np.nanmean(pair[(c, t, tap0)])
                 for (c, t) in keys
                 if (c, t, tap12) in pair and (c, t, tap0) in pair]
            d = [x for x in d if np.isfinite(x)]
            if not d:
                continue
            wins = sum(1 for x in d if x > 0)
            print(f"    N={str(n):>6}  mean {np.mean(d):+.4f}  enc12 wins {wins}/{len(d)}")

    pin = [p for p in pts if p["lam_pinned"]]
    print(f"\n  λ LO-pinned on {len(pin)}/{len(pts)} fits ({100.0 * len(pin) / max(len(pts), 1):.1f}%) "
          f"— a LO pin is a real truncation (v3_board_readout.py:859); high here means the small-N "
          f"points want LESS shrinkage than the grid allows and the curve's left end is an artifact")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="ws", choices=("ws", "cs", "merge"))
    p.add_argument("--cache-dir")
    p.add_argument("--tag", default="pbs50_cd45k")
    p.add_argument("--index", type=int, default=0,
                   help="array task id: index into LITE_SESSIONS (ws) or CS_TEST_CELLS (cs)")
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--out", default="samplecurve.json")
    p.add_argument("--regime", default="ws", choices=("ws", "cs"),
                   help="merge mode only: which shard family to merge. A merge NEVER mixes the two.")
    p.add_argument("--taps", default="", help="default: the regime's own unit (see CURVE_TAPS)")
    p.add_argument("--contiguous", action="store_true",
                   help="draw the FIRST N trials instead of a random N. ⚠️ with KFold(2) only one "
                        "fold has train preceding test; the other decodes the past. A different "
                        "question, NOT the overnight number.")
    args = p.parse_args()

    if args.mode == "merge":
        m = _merge(args.shard_dir, args.regime)
        _report(m)
        with open(args.out, "w") as f:
            json.dump(m, f)
        print(f"\nwrote {args.out}\nMERGE_DONE", flush=True)
        return

    taps = (tuple(t.strip() for t in args.taps.split(",") if t.strip())
            or CURVE_TAPS[args.mode])
    cell = CS_TEST_CELLS[args.index] if args.mode == "cs" else LITE_SESSIONS[args.index]
    grid = N_GRID_CS if args.mode == "cs" else N_GRID
    t0 = time.perf_counter()
    print(f"[R31/{args.mode}] S{cell[0]}T{cell[1]} taps={taps} contiguous={args.contiguous} "
          f"N_GRID={grid} seeds={SEEDS_FOR_N}", flush=True)
    if args.mode == "cs":
        print(f"  train anchor = S{CS_TRAIN_ANCHOR[0]}T{CS_TRAIN_ANCHOR[1]} (subsampled); "
              f"val/test = this cell's cs_split (NEVER subsampled)", flush=True)
    sh = _shard(args.cache_dir, args.tag, args.index, args.mode, taps, args.contiguous)
    os.makedirs(args.shard_dir, exist_ok=True)
    out = f"{args.shard_dir}/{SHARD_PREFIX[args.mode]}_{sh['name']}.json"
    with open(out, "w") as f:
        json.dump(sh, f)
    bad = [a for a in sh["anchor"] if a["absdiff"] >= 1e-9]
    print(f"[anchor] {len(sh['anchor'])} rows, {len(bad)} with |diff| >= 1e-9 "
          f"{'✅' if not bad else '🔴 ' + str(bad[:3])}", flush=True)
    print(f"wrote {out}  ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
