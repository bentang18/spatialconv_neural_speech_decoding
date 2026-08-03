"""R31 · WHY the same-sized gap buys 2.8x label efficiency within-session and 22.9x cross-subject.

═══ THE PUZZLE THIS FILE EXISTS TO RESOLVE ══════════════════════════════════════════════════════
The two regimes report almost the SAME gap at full data — WS +0.0209, CS +0.0222 — and the same
SHAPE for it: additive (k CI covers 1, a CI excludes 0) with a flat gap-vs-N slope in both. Yet the
label-equivalence ratio differs by ~8x (2.78x vs 22.87x). Read carelessly that looks like two
findings, or like the ratio being unstable. It is neither, and the resolution is one line of
algebra:

    if  enc0(log2 N) ~ c + s*log2 N   and   enc12 = enc0 + a   (additive, which is what we measured)
    then enc12 reaches enc0's full-data score  a/s  DOUBLINGS early,  so   RATIO = 2^(a/s).

The ratio is therefore NOT an independent fact about pretraining. It is the additive gap divided by
the EXCHANGE RATE of the regime — how much AUROC one doubling of labels buys there. Same numerator,
different denominator.

So the claim to test is: s_CS < s_WS, i.e. donor labels are worth LESS per doubling than target
labels, by enough to explain the ratio gap. That is a prediction with a number attached, and it can
fail: if s were equal across regimes, the 8x ratio difference would have to come from somewhere else
and the additive reading would be in trouble.

═══ NULLS, STATED BEFORE THE NUMBERS ════════════════════════════════════════════════════════════
  H1  s_WS - s_CS = 0     (labels are worth the same in both regimes)      two-sided
  H2  predicted 2^(a/s) reproduces the measured label-equivalence ratio.   The null here is not a
      p-value but a RECONSTRUCTION check: if the algebra above is the right account, the predicted
      ratio lands on the measured one. If it does not, the linear-in-log2N model is wrong and the
      whole framing goes with it. Printed side by side so it cannot be quietly skipped.

Both bootstrapped over SUBJECTS, the unit the claim generalizes over, on the same panel _addmult
uses so this file cannot disagree with TIER 3 for bookkeeping reasons.

═══ WHAT THIS FILE MAY NOT BE USED TO SAY ═══════════════════════════════════════════════════════
🚫 The WS and CS taps DIFFER (per-electrode vs parcel-mean) because electrode identity does not
   survive a subject change. So s_WS vs s_CS confounds "the train data moved to another brain" with
   "the readout unit changed". This file measures the contrast; it does not attribute it. The
   csession curve is the tap-matched control that does, and until it lands the attribution is open.
🚫 Nothing here licenses "pretraining is especially valuable when labels are scarce". TIER 2 is FLAT
   in both regimes: the gap is the same size at every budget. The ratio is large cross-subject
   because labels are WEAK there, not because the gap grows.
"""
import json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.neuroprobe.v3_board_samplecurve import _panel, _curve, _reach, CURVE_TAPS, FULL

SRC = pathlib.Path("results/showcase/2_what_pretraining_does")
COL = "trainonly"
NBOOT = 4000


def _macro(tot, cnt, draw):
    """Subject-weighted macro curve [N, 2] for a bootstrap draw carrying multiplicity."""
    import warnings
    w = np.bincount(np.asarray(draw, dtype=np.int64), minlength=tot.shape[0]).astype(float)
    t = np.tensordot(w, tot, axes=(0, 0))
    c = np.tensordot(w, cnt, axes=(0, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        per_task = np.where(c > 0, t / np.where(c > 0, c, 1), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(per_task, axis=2)


def _cross(lg, y, tgt):
    """log2 N at which increasing curve `y` reaches `tgt`, linearly interpolated in log2 N."""
    if tgt <= y[0] or tgt > y[-1]:
        return np.nan
    j = min(max(int(np.searchsorted(y, tgt)), 1), len(y) - 1)
    lo, hi = y[j - 1], y[j]
    if hi <= lo:
        return np.nan
    return float(lg[j - 1] + (tgt - lo) / (hi - lo) * (lg[j] - lg[j - 1]))


def _shift_doublings(lg, y0, a):
    """How many doublings of labels the additive gap `a` is worth, by INVERTING the measured enc0
    curve rather than a straight-line fit to it.

    The additive model says enc12(x) = enc0(x) + a. So enc12 hits enc0's top score at whatever x
    satisfies enc0(x) = enc0(top) - a, and the answer is the distance from that x to the top. The
    earlier version divided a by a GLOBAL linear slope, which is only the same thing when the curve
    is a straight line in log2 N. It is not quite: WS flattens from +0.0180 per doubling over the
    whole grid to +0.0163 over the top three points, so the global fit understated the doublings and
    the reconstruction fired a false 🔴. Inverting the curve makes no linearity assumption and is
    the actual content of "additive".
    """
    x = _cross(lg, y0, float(y0[-1]) - a)
    return float(lg[-1] - x) if np.isfinite(x) else np.nan


def exchange(pts, tap0, tap12, nboot=NBOOT, seed=0):
    """s = d(enc0)/d(log2 N), a = mean gap, and the ratio those two PREDICT."""
    tot, cnt, subs, ns = _panel(pts, tap0, tap12, COL)
    lg = np.log2(np.asarray([float(n) for n in ns], dtype=np.float64))

    def one(draw):
        m = _macro(tot, cnt, draw)
        ok = np.isfinite(m).all(axis=1)
        if ok.sum() < 3:
            return None
        s = float(np.polyfit(lg[ok], m[ok, 0], 1)[0])
        a = float(np.mean(m[ok, 1] - m[ok, 0]))
        # PREDICTED vs MEASURED must be read against the SAME target, so both use enc0 at the top
        # GRID point. (The headline TIER 1 ratio instead targets enc0 at FULL, whose x is the train
        # half size and is not on the panel at all — a different, coarser basis. Mixing the two is
        # how the first version of this check produced a false 🔴.)
        tgt = float(m[ok, 0][-1])
        pred = _shift_doublings(lg[ok], m[ok, 0], a)
        meas = lg[ok][-1] - _cross(lg[ok], m[ok, 1], tgt)
        return s, a, pred, float(meas)

    hat = one(np.arange(len(subs)))
    rng = np.random.default_rng(seed)
    B = [r for r in (one(rng.integers(0, len(subs), len(subs))) for _ in range(nboot)) if r]
    arr = np.asarray(B)
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    fin = lambda j: arr[np.isfinite(arr[:, j]), j]
    return {"s": hat[0], "a": hat[1], "pred": hat[2], "meas_grid": hat[3], "n_subjects": len(subs),
            "s_ci": ci(arr[:, 0]), "a_ci": ci(arr[:, 1]),
            "pred_ci": ci(fin(2)), "meas_grid_ci": ci(fin(3)),
            "tot": tot, "cnt": cnt, "subs": subs, "lg": lg}


def measured_ratio(m, regime):
    """The TIER 1 label-equivalence ratio, recomputed EXACTLY as _report prints it.

    enc0's cost is not interpolated: its target IS its own full-data score, so it reaches it only at
    full, and the honest x for that is the median train-half size. enc12's cost is log2-interpolated
    between the two grid points that bracket the target, with FULL stripped first — FULL has no
    numeric x and sorting it against ints is a TypeError, which is how this was caught.
    """
    tap0, tap12 = CURVE_TAPS[regime]
    pts = m["points"]
    c0, c12 = _curve(pts, tap0, COL), _curve(pts, tap12, COL)
    target = c0[FULL]
    n_full = int(np.median([p["n"] for p in pts if p["tap"] == tap0 and p["n_is_full"]]))
    return n_full, _reach({k: v for k, v in c12.items() if k != FULL}, target)


def main() -> None:
    out = {}
    for regime, fn in (("ws", "samplecurve_pbs50_cd45k.json"),
                       ("cs", "samplecurve_cs_pbs50_cd45k.json"),
                       ("csession", "samplecurve_csession_pbs50_cd45k.json")):
        p = SRC / fn
        if not p.exists():
            print(f"[skip] {regime}: {p.name} not on disk yet")
            continue
        m = json.load(open(p))
        tap0, tap12 = CURVE_TAPS[regime]
        e = exchange(m["points"], tap0, tap12)
        n0, n12 = measured_ratio(m, regime)
        e["meas"] = (n0 / n12) if (n0 and n12) else np.nan
        e["n0"], e["n12"] = n0, n12
        out[regime] = e

    print("\n" + "=" * 94)
    print("R31 · THE EXCHANGE RATE — why one additive gap buys different amounts of label efficiency")
    print("=" * 94)
    print("  s  = d(enc0)/d(log2 N)   how much AUROC ONE DOUBLING of labels buys in this regime")
    print("  a  = mean enc12 - enc0   the additive gap (TIER 3 says k~1, so a is the whole story)")
    print("  RECONSTRUCTION: shift the MEASURED enc0 curve up by a and ask where it crosses enc0's")
    print("  top score. If pretraining really is a constant offset, that is exactly where enc12")
    print("  crosses. Both sides use the top GRID point as target so they are the same question.")

    for regime, e in out.items():
        print(f"\n{regime.upper()}  ({e['n_subjects']} subjects, taps {'/'.join(CURVE_TAPS[regime])})")
        print(f"  s        = {e['s']:+.5f} AUROC per doubling   95% CI "
              f"[{e['s_ci'][0]:+.5f}, {e['s_ci'][1]:+.5f}]")
        print(f"  a        = {e['a']:+.5f}                      95% CI "
              f"[{e['a_ci'][0]:+.5f}, {e['a_ci'][1]:+.5f}]")
        print(f"  predicted {e['pred']:.2f} doublings  95% CI [{e['pred_ci'][0]:.2f}, "
              f"{e['pred_ci'][1]:.2f}]   (enc0 curve shifted up by a)")
        print(f"  measured  {e['meas_grid']:.2f} doublings  95% CI [{e['meas_grid_ci'][0]:.2f}, "
              f"{e['meas_grid_ci'][1]:.2f}]   (where enc12 actually crosses)")
        agree = e["pred_ci"][0] <= e["meas_grid"] <= e["pred_ci"][1]
        print(f"  ⇒ RECONSTRUCTION {'✅ HOLDS' if agree else '🔴 FAILS'} — a constant offset "
              f"{'explains' if agree else 'does NOT explain'} the label saving")
        print(f"  ── headline TIER 1 (target = enc0 at FULL, coarser basis): {e['meas']:.1f}x  "
              f"(enc0 {e['n0']} -> enc12 {e['n12']:.0f})")

    # ── H1 · is a label worth less across the boundary? paired over shared subjects ──────────────
    for other in ("cs", "csession"):
        if "ws" not in out or other not in out:
            continue
        A, B = out["ws"], out[other]
        shared = sorted(set(A["subs"]) & set(B["subs"]))
        if len(shared) < 3:
            print(f"\n[skip] ws vs {other}: only {len(shared)} shared subjects")
            continue
        ia = [A["subs"].index(s) for s in shared]
        ib = [B["subs"].index(s) for s in shared]

        def sl(e, idx, draw):
            m = _macro(e["tot"], e["cnt"], [idx[j] for j in draw])
            ok = np.isfinite(m).all(axis=1)
            return float(np.polyfit(e["lg"][ok], m[ok, 0], 1)[0]) if ok.sum() >= 3 else np.nan

        base = np.arange(len(shared))
        d_hat = sl(A, ia, base) - sl(B, ib, base)
        rng = np.random.default_rng(1)
        D = np.asarray([sl(A, ia, d) - sl(B, ib, d)
                        for d in (rng.integers(0, len(shared), len(shared)) for _ in range(NBOOT))])
        D = D[np.isfinite(D)]
        lo, hi = np.percentile(D, 2.5), np.percentile(D, 97.5)
        print(f"\n{'-' * 94}")
        print(f"H1 · PAIRED over the {len(shared)} subjects ws and {other} share "
              f"({', '.join(shared)}) · {len(D)} bootstraps")
        print(f"  s_ws - s_{other} = {d_hat:+.5f} AUROC per doubling   95% CI "
              f"[{lo:+.5f}, {hi:+.5f}]   NULL: 0")
        if lo > 0:
            print(f"  ⇒ A LABEL IS WORTH LESS in {other}. The same additive gap therefore buys more")
            print(f"    doublings there — which is the whole of the ratio difference, not a second effect.")
        elif hi < 0:
            print(f"  ⇒ a label is worth MORE in {other} — the opposite of the account above.")
        else:
            print(f"  ⇒ NULL — no detectable difference in what a label buys. The ratio gap then does")
            print(f"    NOT reduce to the exchange rate and needs another explanation.")

    print()


if __name__ == "__main__":
    main()
