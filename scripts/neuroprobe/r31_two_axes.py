"""R31 · are the two fitted laws ONE law?

This project has two regressions with the same shape and different axes, and the verdict memo
currently ducks the relationship between them by saying they are "different axes, neither confirms
nor refutes the other". That is true and it is also a dodge, because together they make a joint
prediction that either holds or does not.

  ACROSS N      (R31)        macro gap = a + k*(headroom),  measured k ~ 1  ->  the gap is a CONSTANT
                             in the label budget: more labels do not buy more pretraining benefit.
  ACROSS TASKS  (gain law)   AUC_d - .5 = k_d * (AUC_0 - .5), k > 1, THROUGH THE ORIGIN  ->  the
                             benefit is a fixed MULTIPLE of what the frontend already exposes.

Both cannot be "the" shape of the benefit unless they are describing different arguments of the same
function. The reconciliation, if there is one, is:

    pretraining multiplies each TASK's headroom by a fixed factor, and that factor does not depend
    on how many labels the readout was fitted with.

which predicts, per task t:            gap_t = (k - 1) * headroom_t          -- a line through 0
and refutes the alternative:           gap_t = a                             -- a flat line

So this file decomposes the across-N constant BY TASK and asks which of those two it is.

⭐ THE DECISION THIS CHANGES. If it is through the origin, the memo's §2 can say the two laws are one
law and the contribution stance can state the mechanism in one sentence. If it is flat, then the
across-task gain law is an intercept rather than a multiplier and the mechanism memo's headline is
wrong. Those are different papers, so this is worth running.

⚠️ WHY THE GAP IS ESTIMATED AS A MEAN OVER N, NOT AS A PER-TASK INTERCEPT. Fitting `a_t` per task by
regressing across N is ill-conditioned: within ONE task, enc0 barely moves over the sweep, so the
intercept is an extrapolation off a nearly-vertical stack of points. The mean gap is the right
estimator precisely BECAUSE the macro across-N fit already established k ~ 1 -- the gap is constant
in N, so its mean over N is its constant. That is a dependency on the §2 result, stated rather than
hidden: if §2 had come out multiplicative this estimator would be wrong.

⚠️ THE VISUAL TASKS CAN MANUFACTURE THE ANSWER. They sit near chance AND gain little, so they are
points near the origin, and points near the origin are exactly what pins a line to the origin. The
15-task fit is therefore NOT the evidence; the 11-task fit, with all four visual tasks dropped, is.
Both are printed, and the decision is read off the 11.

🚫 This does not compare numerically to the ledger's per-task k table. Those k are no-intercept
slopes fitted across CELLS at full data; this K is a slope across TASKS with an intercept, over
subsampled budgets. They are different estimators and forcing them into agreement would be inventing
a number. The claim tested here is SHAPE -- through the origin or flat.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.neuroprobe.v3_board_samplecurve import _panel, CURVE_TAPS

SRC = pathlib.Path("results/showcase/2_what_pretraining_does")
# The ledger's per-task gain law, transcribed from `paper_figs_r6.py:705` (itself transcribed from
# `project-what-transfers-is-change-coded-not-speech-2026-07-29`). Fitted across CELLS at FULL data
# in CS -- so it is independent of the N sweep, which is what makes it usable as a check here and
# NOT as a calibration. Never recomputed into agreement.
LEDGER_K = {
    "word_length": 1.557, "word_head_pos": 1.330, "word_gap": 1.320, "gpt2_surprisal": 1.274,
    "word_index": 1.259, "onset": 1.252, "speech": 1.229, "delta_volume": 1.146,
    "word_part_speech": 1.116, "volume": 1.068, "local_flow": 1.032, "global_flow": 0.932,
    "face_num": 0.827, "frame_brightness": 0.746, "pitch": 0.642,
}
CURVES = {"ws": SRC / "samplecurve_pbs50_cd45k.json",
          "csession": SRC / "samplecurve_csession_pbs50_cd45k.json",
          "cs": SRC / "samplecurve_cs_pbs50_cd45k.json"}
COL = "trainonly"
NBOOT = 4000
# Transcribed from `paper_figs_r6.py:648-654`, itself the ledger's set. Not redefined here --
# the event/level cut comes from the LABEL DEFINITIONS, so this analysis cannot quietly invent a
# grouping that flatters it.
EVENT = ("onset", "speech", "delta_volume", "word_index", "word_head_pos",
         "word_length", "gpt2_surprisal", "word_gap", "word_part_speech")
LEVEL = ("volume", "pitch", "local_flow", "global_flow", "face_num", "frame_brightness")
VISUAL = ("local_flow", "global_flow", "face_num", "frame_brightness")


def per_task_panel(pts, tap0, tap12, col=COL):
    """The `_panel` tensor plus the task names it is indexed by.

    `_panel` does not return the task axis labels, and this analysis is entirely about that axis.
    Rather than change a function four other callers depend on, the order is reconstructed under the
    SAME filter `_panel` applies internally -- and then asserted against the tensor's own width, so a
    drift in either place fails here instead of silently mislabelling every task.
    """
    tot, cnt, subs, ns = _panel(pts, tap0, tap12, col)
    tasks = sorted({p["task"] for p in pts
                    if p["col"] == col and p["tap"] in (tap0, tap12) and not p["n_is_full"]})
    assert len(tasks) == tot.shape[3], f"task axis {tot.shape[3]} != reconstructed {len(tasks)}"
    return tot, cnt, subs, ns, tasks


def _taskfit(tot, cnt, keep, xmode="meanN"):
    """(A, K, headroom[], gap[]) for a subject draw, over the tasks selected by `keep`.

    `xmode` exists to bound a known bias in K, not to pick a favourable answer -- both are printed.
      meanN  headroom averaged over the same N points as the gap. Self-consistent, but the sweep
             includes budgets where enc0 has NOT saturated, so x is deflated and K comes out HIGH.
      topN   headroom at the largest budget on the grid, where enc0 is nearest its full value. This
             is legitimate ONLY because §2 established the gap is constant in N, so pairing a
             top-N headroom with a sweep-mean gap is not mixing two different quantities.
    The SHAPE verdict is read off A, which a proportional deflation of x leaves essentially alone;
    the K magnitude is the part that moves, which is why it is never quoted against the ledger's k.
    """
    def f(draw):
        w = np.bincount(np.asarray(draw, dtype=np.int64), minlength=tot.shape[0]).astype(float)
        t = np.tensordot(w, tot, axes=(0, 0))               # [N, 2, task]
        c = np.tensordot(w, cnt, axes=(0, 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            per = np.where(c > 0, t / np.where(c > 0, c, 1), np.nan)
        with warnings.catch_warnings():                     # an all-NaN task is a real "no data"
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(per, axis=0)                     # [2, task]  averaged over the N sweep
        x = (per[-1, 0] if xmode == "topN" else m[0])[keep] - .5     # task headroom
        y = m[1][keep] - m[0][keep]                         # task gap  (the across-N constant)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            return None
        K, A = np.polyfit(x[ok], y[ok], 1)
        return float(A), float(K), x, y
    return f


def two_axes(pts, tap0, tap12, tasks_drop=(), xmode="meanN", nboot=NBOOT, seed=0):
    tot, cnt, subs, ns, tasks = per_task_panel(pts, tap0, tap12)
    keep = np.array([t not in tasks_drop for t in tasks])
    assert keep.sum() >= 3, "need at least 3 tasks to fit a line across tasks"
    fit = _taskfit(tot, cnt, keep, xmode)
    base = fit(np.arange(len(subs)))
    assert base is not None, "the point estimate itself did not fit"
    A0, K0, x0, y0 = base

    rng = np.random.default_rng(seed)
    As, Ks = [], []
    for _ in range(nboot):
        r = fit(rng.integers(0, len(subs), len(subs)))
        if r is None:
            continue
        As.append(r[0]); Ks.append(r[1])
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"A": A0, "A_ci": q(As), "K": K0, "K_ci": q(Ks), "n_boot": len(As), "xmode": xmode,
            "n_subjects": len(subs), "n_points": len(ns),
            "tasks": [t for t, k in zip(tasks, keep) if k],
            "headroom": x0, "gap": y0}


def ledger_check(r, nperm=10000, seed=0) -> dict:
    """Does the across-N per-task gap follow the ledger's per-task k, fitted on the OTHER axis?

    The comparison is `gap_t` against `(k_t - 1) * headroom_t`, NOT against `k_t` directly, and the
    reason is the trap `paper_figs_r6._k` documents: a per-task ratio divides by an enc0 that can sit
    .003 above chance and explodes. Multiplying instead of dividing keeps every task finite.

    Reported as a RANK correlation with a label-permutation null, because the two axes are not
    expected to agree in scale -- one is fitted at full data across cells, the other over subsampled
    budgets across N. Agreement in ORDER is the falsifiable part; agreement in magnitude is not
    claimed and would not mean anything if it appeared. (It does not appear: the measured gaps run
    roughly 2x the predicted ones, which is the estimator difference, not a finding.)

    🔴 THIS IS NOT AN INDEPENDENT REPLICATION AND MUST NOT BE WRITTEN AS ONE. Both fits are the same
    model family scored on the same Neuroprobe CS test sets. What agreement shows is that the
    across-N decomposition RECOVERS the across-cell structure -- two estimators, one experiment.
    Two independent experiments agreeing would be a different and much stronger claim.
    """
    pred = np.array([(LEDGER_K[t] - 1.0) * h for t, h in zip(r["tasks"], r["headroom"])])
    obs = np.asarray(r["gap"], dtype=float)
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    rho = lambda a, b: float(np.corrcoef(rank(a), rank(b))[0, 1])
    r0 = rho(pred, obs)
    rng = np.random.default_rng(seed)
    null = np.array([rho(pred, rng.permutation(obs)) for _ in range(nperm)])
    return {"rho": r0, "p": float((np.abs(null) >= abs(r0)).mean()), "n": len(obs),
            "null_hi": float(np.percentile(np.abs(null), 95)), "nperm": nperm}


def selectivity_check(r) -> dict:
    """Is the scatter around the fitted line STRUCTURED by the event/level cut?

    THIS EXISTS TO STOP THIS ANALYSIS FROM ERASING A HARDENED FINDING. Fitting ONE slope across all
    tasks says "the benefit is a fixed multiple of headroom" and implicitly treats every deviation
    as noise. But the ledger's per-task result is a clean event-over-level split, which means the
    deviations are not noise -- they ARE the change-coded selectivity. If the level tasks sit
    systematically below the line, then K is a SUMMARY of a family-dependent multiplier, not a law,
    and it must be written that way.

    Tested on the RESIDUAL `gap - (A + K*headroom)`, never on a per-task ratio: dividing by an enc0
    that sits near chance is the blow-up `paper_figs_r6._k` documents.

    NULL, exact and combinatorial, no bootstrap needed: under random assignment of families to
    tasks, the probability that all `nl` level tasks land on the `nl` most-negative residuals is
    1 / C(n, nl). With 11 non-visual tasks and 2 level tasks that is 1/55 = .018, so the test can
    fire at all -- state it before reading the answer.
    """
    x, y = np.asarray(r["headroom"], float), np.asarray(r["gap"], float)
    resid = y - (r["A"] + r["K"] * x)
    order = np.argsort(resid)                              # most negative first
    lvl = {i for i, t in enumerate(r["tasks"]) if t in LEVEL}
    evt = {i for i, t in enumerate(r["tasks"]) if t in EVENT}
    n, nl = len(r["tasks"]), len(lvl)
    if nl == 0 or nl == n or lvl | evt != set(range(n)):
        return {"testable": False, "n_level": nl, "n_event": len(evt), "n": n}
    ranks = sorted(((int(np.where(order == i)[0][0]), r["tasks"][i]) for i in lvl))
    return {"testable": True, "n": n, "n_level": nl,
            "all_bottom": lvl == set(order[:nl].tolist()),
            "p_exact": 1.0 / math.comb(n, nl),
            "level_ranks": {t: rk for rk, t in ranks},
            "resid": resid, "order": order}


def verdict(r) -> str:
    """NULLS STATED BEFORE THE FIT: A = 0 (through the origin), K = 0 (flat)."""
    a_sig = not (r["A_ci"][0] <= 0 <= r["A_ci"][1])
    k_sig = not (r["K_ci"][0] <= 0 <= r["K_ci"][1])
    return {
        (False, True): "THROUGH THE ORIGIN ⇒ the per-task constant SCALES with headroom ⇒ the "
                       "across-N and across-task laws ARE ONE LAW",
        (True, False): "FLAT ⇒ every task gets the SAME constant ⇒ 🔴 CONTRADICTS the across-task "
                       "gain law, which is a multiplier",
        (True, True): "BOTH an offset and a slope ⇒ the multiplier is real but does not account "
                      "for all of it — 🚫 do not state either law alone",
        (False, False): "UNDERPOWERED — neither term clears its null; 🚫 claim nothing",
    }[(a_sig, k_sig)]


def main() -> None:
    print("=" * 96)
    print(f"R31 · TWO AXES, ONE LAW?   column '{COL}' · {NBOOT} subject-bootstraps")
    print("  model   gap_task = A + K · headroom_task     (fitted ACROSS TASKS, gap averaged over N)")
    print("  NULLS   A = 0  (a line through the origin — a pure multiplier across tasks)")
    print("          K = 0  (a flat line — the same constant for every task)")
    print("  RANGE   headroom spans near-chance tasks to ~.25 above chance, so both nulls are")
    print("          distinguishable in principle; the CIs below say whether they are in practice.")
    print("=" * 96)

    for regime, src in CURVES.items():
        if not src.exists():
            print(f"\n[skip] {regime}: {src.name} not on disk yet")
            continue
        m = json.load(open(src))
        worst = max(m["anchor"], key=lambda a: a["absdiff"])
        assert worst["absdiff"] < 1e-9, f"{regime} ANCHOR DRIFTED ({worst}) — refusing to report"
        tap0, tap12 = CURVE_TAPS[regime]
        pts = m["points"]

        print(f"\n{'─' * 96}\n{regime.upper()}   taps {tap0}/{tap12}   tag {m['tags'][0]}")
        for label, drop in (("15 tasks (submission macro)", ()),
                            ("11 tasks (visual dropped — THE EVIDENCE)", VISUAL)):
            rs = [two_axes(pts, tap0, tap12, tasks_drop=drop, xmode=xm)
                  for xm in ("meanN", "topN")]
            r = rs[0]
            print(f"\n  {label}   {len(r['tasks'])} tasks, {r['n_subjects']} subjects, "
                  f"{r['n_points']} N-points, {r['n_boot']} boots")
            for rr in rs:
                print(f"    [{rr['xmode']:>5}]  A = {rr['A']:+.5f} "
                      f"[{rr['A_ci'][0]:+.5f}, {rr['A_ci'][1]:+.5f}]   "
                      f"K = {rr['K']:+.5f} [{rr['K_ci'][0]:+.5f}, {rr['K_ci'][1]:+.5f}]   "
                      f"⇒ {verdict(rr).split(' ⇒ ')[0]}")
            print(f"    NULLS A = 0, K = 0.  🚫 slope+1 ({1 + rs[0]['K']:.3f} meanN / "
                  f"{1 + rs[1]['K']:.3f} topN) is NOT the ledger's per-task k — different "
                  f"estimator, different axis.")
            print(f"    ⇒ {verdict(r)}")
            if verdict(rs[0]) != verdict(rs[1]):
                print("    🔴 THE TWO HEADROOM DEFINITIONS DISAGREE ON THE VERDICT — the shape is "
                      "not robust to the x definition; claim nothing until it is.")
            if drop:
                # The ledger's k was fitted in CS at the CS taps. Running it against the elec taps
                # would be comparing a table to data it was never fitted on, so it is CS-only.
                if regime == "cs":
                    lc = ledger_check(r)
                    print(f"    LEDGER CHECK  gap_t vs (k_t−1)·headroom_t, k from the ledger's "
                          f"across-CELLS fit, n={lc['n']}")
                    print(f"      Spearman ρ = {lc['rho']:+.3f}   perm-null |ρ| 95th pct "
                          f"{lc['null_hi']:.3f}   p = {lc['p']:.4f}  ({lc['nperm']} perms)")
                    print(f"      ⇒ {'ORDER AGREES ACROSS THE TWO AXES' if lc['p'] < .05 else
                                     'no order agreement at this n — 🚫 do not claim the axes align'}")
                    print("      🔴 same model, same test sets — two ESTIMATORS agreeing, NOT two "
                          "independent experiments. Magnitudes differ ~2x and are not claimed.")
                sel = selectivity_check(r)
                if sel["testable"]:
                    print(f"    SELECTIVITY  are the {sel['n_level']} LEVEL tasks the "
                          f"{sel['n_level']} most-NEGATIVE residuals off the fitted line?")
                    print(f"      exact null P = 1/C({sel['n']},{sel['n_level']}) = "
                          f"{sel['p_exact']:.4f}   level ranks (0 = most negative): "
                          f"{sel['level_ranks']}")
                    print(f"      ⇒ {'YES — the scatter is STRUCTURED by the event/level cut ⇒ K is a '
                                     'SUMMARY of a family-dependent multiplier, NOT a law'
                                     if sel['all_bottom'] else
                                     'NO — level tasks are not the bottom; 🚫 do not claim the '
                                     'across-N axis reproduces the event/level split'}")
                order = np.argsort(-r["headroom"])
                resid = sel.get("resid")
                print("    per task (sorted by headroom, meanN):")
                for i in order:
                    t = r["tasks"][i]
                    fam = "EVENT" if t in EVENT else "level"
                    res = f"   resid {resid[i]:+.4f}" if resid is not None else ""
                    print(f"      {t:>18} {fam}  headroom {r['headroom'][i]:+.4f}"
                          f"   gap {r['gap'][i]:+.4f}{res}   ledger k {LEDGER_K[t]:.3f}")


if __name__ == "__main__":
    main()
