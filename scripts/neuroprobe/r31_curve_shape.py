"""R31 · is the label curve actually log-linear, or is it saturating?

WHY THIS IS A DECISION AND NOT A CURIOSITY. The verdict's §3 rests on an exchange rate `s` --
"AUROC one doubling of labels buys" -- and quotes it as a CONSTANT (s_ws +.01801, s_cs +.00776).
That is only meaningful if the curve is log-linear over the sweep. If the curve saturates instead,
`s` is a range-dependent artefact of where the grid happens to stop, contrasting s_ws against s_cs
compares two different functional forms, and "you cannot buy cross-subject decoding with labels"
would need restating (as a stronger claim -- an asymptote is worse than a slow slope -- but a
different one).

THE TEST. On x = log2(N), fit

    linear      y = a + s·x
    quadratic   y = a + s·x + q·x²

and ask whether the curvature `q` clears zero. `q` is the whole test: q = 0 means log-linear is
adequate and `s` is a constant; q < 0 means the curve bends over and `s` is not.

NULL: q = 0, from a bootstrap over SUBJECTS (not cells -- cells within a subject are not
independent, and the subject is the generalisation unit this project reports). A subject
resample recomputes the macro curve from scratch and refits, so the CI carries the cohort's own
variability, not the residual scatter of an already-averaged curve.

DECISION MAP, fixed before the numbers are read:
  q CI excludes 0 and q < 0   -> SATURATING. `s` is range-dependent; §3 needs restating.
  q CI includes 0             -> log-linear ADEQUATE. `s` stands as a constant over this range.
  q CI excludes 0 and q > 0   -> accelerating. Would be very strange; treat as a bug, not a result.

A third model is fitted for scale only, never for the verdict: a saturating approach to an
asymptote, y = A − c·2^(−α·x). Its rmse says whether saturation would fit BETTER, which is a
different question from whether the linear fit is INADEQUATE. Only `q` decides.

⚠️ FULL is excluded from every fit. Its N differs per cell, so it is not a grid point and putting
it on the x axis would mix a per-cell number into a cohort curve.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.neuroprobe.fig_r31_ws_vs_cs import COL, SPEC, load  # noqa: F401  (load gates the file)
from scripts.neuroprobe.v3_board_samplecurve import CURVE_TAPS, FULL

import json

ORDER = ("ws", "csession", "cs")
NBOOT = 4000
SEED = 0


def _table(pts, tap):
    """(N, task, cell) -> the seed/fold mean, collapsed ONCE. The bootstrap then only re-averages
    this table over cells, so 4000 resamples cost 4000 small means instead of 4000 full scans.
    The aggregation order is still the board's: seeds+folds, then tasks, then cells."""
    by = {}
    for p in pts:
        if p["tap"] != tap or p["col"] != COL or p["n_is_full"]:
            continue
        by.setdefault(p["n_bucket"], {}).setdefault(p["task"], {}).setdefault(
            p["cell"], []).append(p["test"])
    ns = sorted(by)
    tasks = sorted({t for v in by.values() for t in v})
    cells = sorted({c for v in by.values() for tv in v.values() for c in tv})
    ci = {c: i for i, c in enumerate(cells)}
    # NaN where a (n, task, cell) was never run; nanmean below then skips it exactly as the
    # unvectorised version's missing dict key would have.
    A = np.full((len(ns), len(tasks), len(cells)), np.nan)
    for i, n in enumerate(ns):
        for j, t in enumerate(tasks):
            for c, v in by[n].get(t, {}).items():
                A[i, j, ci[c]] = np.nanmean(v)
    return ns, cells, A


def _macro(A, idx):
    """{N: macro} for a cell index list -- mean over tasks of the cohort mean over those cells."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.nanmean(A[:, :, idx], axis=2), axis=1)


def _fits(x, y):
    """linear, quadratic, and the saturating reference. Returns (s, q, rmse_lin, rmse_quad, rmse_sat)."""
    lin = np.polyfit(x, y, 1)
    quad = np.polyfit(x, y, 2)
    r_lin = float(np.sqrt(np.mean((y - np.polyval(lin, x)) ** 2)))
    r_quad = float(np.sqrt(np.mean((y - np.polyval(quad, x)) ** 2)))

    # y = A - c*2^(-alpha*x), scanned over alpha because it is the only nonlinear parameter; A and
    # c are then linear-in-parameters given alpha. Grid, not an optimiser: 3 numbers, 8 points.
    best = (np.inf, None)
    for alpha in np.linspace(0.02, 3.0, 300):
        b = 2.0 ** (-alpha * x)
        M = np.stack([np.ones_like(x), b], 1)
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        r = float(np.sqrt(np.mean((y - M @ coef) ** 2)))
        if r < best[0]:
            best = (r, alpha)
    return float(lin[0]), float(quad[0]), r_lin, r_quad, best[0], best[1]


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"NULL: q = 0 (log-linear adequate).  {NBOOT} bootstraps over SUBJECTS.  column '{COL}'.")
    print("DECISION: CI excludes 0 and q<0 -> SATURATING, `s` is range-dependent."
          "   CI includes 0 -> log-linear adequate.\n")
    hdr = (f"{'rung':10s} {'tap':6s} {'pts':>4s} {'subj':>5s} {'s/doubling':>11s} "
           f"{'q':>10s} {'q 95% CI':>22s} {'rmse lin':>9s} {'rmse quad':>10s} "
           f"{'rmse sat':>9s}  verdict")
    print(hdr)
    print("-" * len(hdr))

    for r in ORDER:
        m = json.load(open(SPEC[r]["src"]))
        load(SPEC[r]["src"], r, SPEC[r]["units"])          # the anchor + unit gate, unchanged
        pts = m["points"]
        cells = sorted({p["cell"] for p in pts})
        subj = sorted({c.split("T")[0] for c in cells})
        by_subj = {s: [c for c in cells if c.split("T")[0] == s] for s in subj}

        for tap in CURVE_TAPS[r]:
            ns, tcells, A = _table(pts, tap)
            ti = {c: i for i, c in enumerate(tcells)}
            x = np.log2(np.array(ns, float))
            y = _macro(A, list(range(len(tcells))))
            s, q, r_lin, r_quad, r_sat, alpha = _fits(x, y)

            idx_by_subj = {s_: [ti[c] for c in cc if c in ti] for s_, cc in by_subj.items()}
            qs = []
            for _ in range(NBOOT):
                pick = rng.choice(subj, size=len(subj), replace=True)
                idx = [i for s_ in pick for i in idx_by_subj[s_]]
                qs.append(np.polyfit(x, _macro(A, idx), 2)[0])
            lo, hi = np.percentile(qs, [2.5, 97.5])
            verdict = ("SATURATING" if hi < 0 else "accelerating?? CHECK" if lo > 0
                       else "log-linear adequate")
            print(f"{r:10s} {tap:6s} {len(ns):4d} {len(subj):5d} {s:+11.5f} "
                  f"{q:+10.5f} [{lo:+.5f},{hi:+.5f}] {r_lin:9.5f} {r_quad:10.5f} "
                  f"{r_sat:9.5f}  {verdict}")

    print("\nRANGE for scale: the grid spans "
          f"{np.log2(3500 / 16):.1f} doublings; a curve that saturates only above the top grid "
          "point is indistinguishable from log-linear here BY CONSTRUCTION, and this test says\n"
          "nothing about N beyond the sweep.")


if __name__ == "__main__":
    main()
