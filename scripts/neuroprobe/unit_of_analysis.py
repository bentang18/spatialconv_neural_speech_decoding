"""SETTLE THE UNIT OF ANALYSIS. Three places in this repo test three different units, and until
now nothing said which one the paper uses.

    r31_reach_ci.py                 resamples SUBJECTS
    fig_r30_geometry_ablation.py    Wilcoxon signed-rank over CELLS      (session means)
    factorial_2x2.py                sign test over UNITS                 (task x session)

Those are not three opinions about the same test. They are three different populations being
generalized over, and they will disagree by construction: pooling 15 tasks per session as
independent draws multiplies n by 15 without adding 15 independent measurements, because every task
in a session is scored from the SAME neural data through the SAME readout.

═══ THE POINT OF THIS SCRIPT ════════════════════════════════════════════════════════════════════
It does not pick a winner by looking at which p is prettiest. It prints, for one contrast, every
(unit x test) cell TOGETHER WITH THE SMALLEST P THAT UNIT COULD EVER PRODUCE. That floor is the
part that actually settles the question, because a unit whose floor is above .05 cannot return a
significant result no matter what the data say, and quoting a significant p from a finer unit is
then a statement about tasks, not about patients.

For an exact paired nonparametric test on n pairs there are 2^n equally likely sign assignments
under the null and exactly one is maximally extreme, so BOTH the sign test and the exact Wilcoxon
signed-rank bottom out at 2/2^n:

    n =  5  ->  .0625     n =  6  ->  .0313     n = 10  ->  .0020     n = 12  ->  .0005

⇒ at 5 cross-subject patients NO exact paired test can reach p < .05. That is a property of the
cohort, not of the effect, and it is the single most important fact on this page.

═══ THE RULE THIS SCRIPT EXISTS TO FIX IN PLACE, BEFORE THE IID NUMBERS LAND ════════════════════
1. The unit is the one the CLAIM generalizes over. A claim about patients is tested over SUBJECTS.
2. If the subject-level floor exceeds .05, the honest report is the EFFECT and the DIRECTION COUNT
   at subject level, and the statement "n is too small for significance at this unit" -- never a
   significant p borrowed from a finer unit presented as if it were the same claim.
3. A finer unit may be reported as a SECONDARY, explicitly labelled as a within-session/per-task
   statement.
4. Whatever is chosen is applied UNIFORMLY. Choosing per contrast after seeing the p is the defect
   this file exists to prevent.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.neuroprobe.fig_r30_geometry_ablation import (  # noqa: E402
    ARMS, ARMS_VITS384, REGIMES, load, load_shards, series)
from scripts.neuroprobe.v3_board_samplecurve import _subject  # noqa: E402

UNITS = ("subject", "cell", "task x cell")


def floor_p(n):
    """Smallest attainable two-sided p for an exact paired test on n nonzero pairs."""
    return min(1.0, 2.0 / 2 ** n) if n > 0 else float("nan")


def sign_p(d, eps=1e-12):
    nz = np.asarray([x for x in d if abs(x) > eps], float)
    n = len(nz)
    if n == 0:
        return float("nan"), 0, 0
    npos = int((nz > 0).sum())
    k = min(npos, n - npos)
    from math import comb
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return p, npos, n


def wilcoxon_p(d, eps=1e-12):
    nz = np.asarray([x for x in d if abs(x) > eps], float)
    if len(nz) < 1:
        return float("nan"), 0, 0
    return float(wilcoxon(nz).pvalue), int((nz > 0).sum()), len(nz)  # type: ignore[attr-defined]


def aggregate(gain, unit):
    """gain is a Series indexed (cell, task). Returns the per-unit deltas."""
    if unit == "task x cell":
        return gain.values
    if unit == "cell":
        return gain.groupby(level=0).mean().values
    if unit == "subject":
        return gain.groupby(gain.index.get_level_values(0).map(_subject)).mean().values
    raise ValueError(unit)


def report(name, gain):
    """The verdict comes from the SIGN TEST alone.

    Taking min(sign, wilcoxon) would be a two-test multiple comparison dressed up as one number, so
    one test is designated and the other is printed as a DIAGNOSTIC. The sign test is the primary
    because it assumes only P(+) = P(-), whereas the exact Wilcoxon signed-rank additionally
    assumes the differences are symmetric about the null median -- an assumption AUROC gains break,
    since a few high-headroom tasks carry large gains while most sit near zero. Where the two
    disagree sharply the asymmetry is real and the signed-rank p is the untrustworthy one, so the
    disagreement is flagged rather than averaged away.
    """
    print(f"\n  {name}")
    print(f"    {'unit':<12} {'n':>4} {'mean':>9} {'dir':>8}  {'SIGN':>8} {'(wilcox)':>9}"
          f"  {'floor':>8}   verdict")
    verdicts = {}
    for unit in UNITS:
        d = aggregate(gain, unit)
        ps, npos, n = sign_p(d)
        pw, _, _ = wilcoxon_p(d)
        fl = floor_p(n)
        capable = fl <= .05
        sig = capable and ps < .05
        verdicts[unit] = None if not capable else sig
        note = "CANNOT REACH .05" if not capable else ("significant" if sig else "not significant")
        if capable and (ps < .05) != (pw < .05):
            note += "   ⚠ tests disagree (skew)"
        print(f"    {unit:<12} {n:>4} {np.mean(d):>+9.4f} {npos:>4}/{n:<3} "
              f"{ps:>8.4f} {pw:>9.4f} {fl:>8.4f}   {note}")
    return verdicts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="r6", choices=("r6", "vits384"))
    a = ap.parse_args()

    # r6 arms live in the d=256 ledger; the vits384 arms are read straight from shards, exactly as
    # the figure does. Same split, so the two paths cannot disagree about provenance.
    arms = ARMS if a.arms == "r6" else ARMS_VITS384
    L = load() if a.arms == "r6" else load_shards(arms)
    print("=" * 100)
    print(f"UNIT OF ANALYSIS -- gain over the zero-parameter floor (enc12 - enc0), arms={a.arms}")
    print("=" * 100)

    flips = []
    for regime, t0, t12, _title in REGIMES:
        floor = series(L, arms[0][2], regime, t0)
        print(f"\n{'=' * 100}\n[{regime.upper()}]")
        for arm, label, artifacts in arms:
            e12 = series(L, artifacts, regime, t12)
            if e12 is None or floor is None:
                print(f"\n  {arm} ({label}): no shard at this tap -- skipped")
                continue
            idx = floor.index.intersection(e12.index)
            v = report(f"{arm} ({label})", e12[idx] - floor[idx])
            got = {k: x for k, x in v.items() if x is not None}
            if len(set(got.values())) > 1:
                flips.append((regime, arm, got))

    print("\n" + "=" * 100)
    print("WHERE THE UNIT CHANGES THE ANSWER")
    print("=" * 100)
    if not flips:
        print("  none -- every contrast that CAN be tested agrees across the units that can test it.")
    for regime, arm, got in flips:
        print(f"  {regime:<9} {arm:<10} " +
              "  ".join(f"{u}={'sig' if s else 'ns'}" for u, s in got.items()))
    print("\n  Units marked CANNOT REACH .05 are excluded above: they carry no verdict to flip.")
    print("=" * 100)
    print("""
THE SETTLED RULE (fixed 2026-08-16, applies to every contrast in the paper)

  PRIMARY   sign test over SUBJECTS. The claim is about patients, so patients are the unit, and
            the sign test is preferred over signed-rank because AUROC gains are skewed and
            signed-rank additionally assumes symmetry about the null median.
  FLOOR     n=5 subjects (cs) floors at .0625, so CROSS-SUBJECT CANNOT REACH p<.05 AT ALL.
            n=6 (ws/csession) floors at .0312, so only a UNANIMOUS 6/6 clears it.
  WHEN THE FLOOR BLOCKS IT   report the effect size and the direction count and say the cohort is
            the limit. Never substitute a finer unit's p for the same sentence.
  SECONDARY cell (session) level, explicitly labelled as such.
  BANNED    task x cell as a significance claim. It is the ONLY unit that ever manufactures
            significance the coarser units do not support, and every flip found here runs that one
            direction. 15 tasks in a session are one recording read 15 ways, not 15 draws.
""")


if __name__ == "__main__":
    main()
