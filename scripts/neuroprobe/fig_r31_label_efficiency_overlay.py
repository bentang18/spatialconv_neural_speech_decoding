"""R31 · all three rungs on ONE pair of axes.

The 3x2 ladder figure keeps the rungs in separate rows on purpose. This is the overlay Ben asked
for: same two measurements, one axes each, so the three rungs can be read against each other
without the eye travelling between panels.

🚨 PANEL A OVERLAYS THREE AXES THAT ARE NOT THE SAME AXIS. N is target-session calibration trials
   for ws, other-session trials from the SAME patient for csession, and DONOR-session trials for
   cs. Drawing them on one x makes them LOOK like one budget. They are not, and no sentence may
   compare an ws N against a cs N as though they were the same resource. The x label and the
   suptitle both say so. `fig_r31_ws_vs_cs.py:9` is the original statement of this rule.

✅ PANEL B IS LICENSED TO OVERLAY. The gain-law coordinate (gap vs headroom) is a ratio of AUROCs
   in every regime, carries no trial-count units, and is the coordinate the law is stated in --
   `fig_r31_ws_vs_cs.py:19` says exactly this. The three regimes ARE comparable there, and the
   overlay is the whole point: three flat lines at nearly the same height, and not one of the three
   clouds points at the origin.

⚠️ COLUMN IS `trainonly` (the only column all three regimes share) -- so ws reads ~2.8x here where
the claims table's D1 reads 2.69x on `both`. Same measurement, different column, NOT a correction.

Gates, tap map and colours are IMPORTED, never restated.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.neuroprobe.fig_r31_ws_vs_cs import ACC, OUT, SPEC, _style, load
from scripts.neuroprobe.v3_board_samplecurve import FULL, _reach

ORDER = ("ws", "csession", "cs")


def _pts(d):
    """The N grid with FULL kept last, and the two curves on it."""
    ns = [n for n in sorted(d["c0"], key=lambda v: (v == FULL, v)) if n != FULL]
    return ns, ns + [FULL]


def overlay_saving(ax, got):
    """A · six curves. enc12 solid, its own enc0 dashed in the same colour, one colour per rung.
    Each rung's target line is that rung's OWN enc0 at full data -- there is no shared target."""
    rows = []
    for r in ORDER:
        s, d = SPEC[r], got[r]
        ns, _ = _pts(d)
        x = np.array(ns, float)
        target, n_full = d["c0"][FULL], d["n_full"]
        reach = _reach({k: v for k, v in d["c12"].items() if k != FULL}, target)

        ax.plot(x, [d["c0"][n] for n in ns], "--o", color=s["colour"], lw=1.1, ms=2.6,
                alpha=.45, mfc="white", mew=.9)
        ax.plot(x, [d["c12"][n] for n in ns], "-o", color=s["colour"], lw=1.6, ms=3.4,
                label=s["short"])
        ax.plot([n_full], [d["c12"][FULL]], "o", color=s["colour"], ms=5.2, mfc="white",
                mew=1.4, zorder=6)
        ax.plot([n_full], [target], "o", color=s["colour"], ms=4.4, mfc="white", mew=1.0,
                alpha=.55, zorder=6)
        ax.plot([reach, n_full], [target, target], color=s["colour"], ls=":", lw=.9, alpha=.7)
        rows.append((r, s, target, n_full, reach))

    ax.set_xscale("log", base=2)
    ax.set_xlabel("labelled trials N (log$_2$)  —  ⚠ A DIFFERENT SOURCE PER RUNG, not one budget")
    ax.set_ylabel("board macro AUROC")
    ax.set_title("A · the curves   (solid = enc12 pretrained, dashed = enc0 frontend)",
                 loc="left", fontsize=8)

    # The savings go in a block, not in three arrows: the ws and csession target lines are .008
    # apart and their arrows would sit on top of each other.
    txt = "enc12 trials to match its own enc0 at full data\n" + "\n".join(
        f"  {s['short']:<24s}{reach:>5.0f} / {n_full:<5d}{n_full / reach:>6.1f}×"
        for _, s, _, n_full, reach in rows)
    ax.text(.015, .975, txt, transform=ax.transAxes, va="top", ha="left", fontsize=6.4,
            family="monospace", color=ACC,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ACC, lw=.7, alpha=.9))
    ax.legend(loc="lower right", fontsize=6.8, title="rung", title_fontsize=6.8)
    return rows


def overlay_falsifier(ax, got):
    """B · the licensed overlay. Both models fitted per rung to the SAME points; the additive line
    is drawn solid because it is the one the data pick, the origin line dotted because it is the
    one they reject. Verdict is printed from the fit, never asserted."""
    gx_max = max(v - .5 for d in got.values() for v in d["c0"].values()) * 1.06
    out = []
    for r in ORDER:
        s, d = SPEC[r], got[r]
        _, nb = _pts(d)
        xs = np.array([d["c0"][n] - .5 for n in nb])
        ys = np.array([d["c12"][n] - .5 for n in nb])
        gap = ys - xs
        k = float(xs @ ys / (xs @ xs))
        a = float(gap.mean())
        r_add = float(np.sqrt(np.mean((gap - a) ** 2)))
        r_mul = float(np.sqrt(np.mean((gap - (k - 1) * xs) ** 2)))

        # Each fit is drawn only over the headroom its OWN regime actually spans. Extrapolating
        # the cs origin-line out to ws's headroom would draw a prediction on data cs never saw.
        gx = np.linspace(0, xs.max() * 1.06, 100)
        ax.plot(gx, (k - 1) * gx, ":", color=s["colour"], lw=1.1, alpha=.55)
        ax.plot([0, xs.max() * 1.06], [a, a], color=s["colour"], lw=1.4, ls="--")
        ax.plot(xs, gap, "o", color=s["colour"], ms=4.2, zorder=5,
                label=f"{s['short']}   $a$={a:+.4f}  $k$={k:.3f}")
        out.append(dict(r=r, a=a, k=k, r_add=r_add, r_mul=r_mul,
                        lo=min(0.0, float(gap.min())),
                        hi=max(float(gap.max()), (k - 1) * xs.max() * 1.06)))

    ax.axhline(0, color="#ccc", lw=.8, zorder=0)
    ax.set_xlim(0, gx_max)
    ax.set_ylim(min(o["lo"] for o in out) * 1.15, max(o["hi"] for o in out) * 1.22)
    ax.set_xlabel(r"enc0 headroom  (AUROC$_0$ $-$ 0.5)   — a unitless axis, shared legitimately")
    ax.set_ylabel(r"enc12 $-$ enc0  (AUROC)")
    win = all(o["r_add"] < o["r_mul"] for o in out)
    ax.set_title("B · dashed = additive fit (flat), dotted = multiplicative fit (through origin)"
                 + ("   → flat wins on all three" if win else "   → MIXED"),
                 loc="left", fontsize=8)
    ax.legend(loc="lower right", fontsize=6.4)
    return out


def main() -> None:
    got = {r: load(SPEC[r]["src"], r, SPEC[r]["units"]) for r in ORDER}

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    rows = overlay_saving(axes[0], got)
    out = overlay_falsifier(axes[1], got)

    fig.suptitle(
        "R31 · all three rungs on one axes · pbs50_cd45k · column 'trainonly' (ws reads 2.8× here "
        "vs 2.69× on 'both' — same measurement, different column)\n"
        "180 / 180 / 150 units · anchors exact (max |curve−published| = 0e+00) · "
        "!! in A the three x axes are DIFFERENT QUANTITIES sharing one line; in B the axis is "
        "unitless and the overlay is exact",
        fontsize=7.2, y=1.02, color="#444")
    fig.tight_layout(w_pad=2.4)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_r31_label_efficiency_overlay.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"{'rung':10s} {'target':>8s} {'n_full':>7s} {'reach':>7s} {'saving':>8s} "
          f"{'a':>9s} {'k':>7s} {'rmse add':>9s} {'rmse mul':>9s}  verdict")
    for (r, _s, target, n_full, reach), o in zip(rows, out):
        assert o["r"] == r
        print(f"{r:10s} {target:8.4f} {n_full:7d} {reach:7.0f} {n_full / reach:7.2f}x "
              f"{o['a']:+9.4f} {o['k']:7.3f} {o['r_add']:9.5f} {o['r_mul']:9.5f}  "
              f"{'ADDITIVE' if o['r_add'] < o['r_mul'] else 'MULTIPLICATIVE'}")
    print(f"\n  -> {OUT}/fig_r31_label_efficiency_overlay.pdf|.png")


if __name__ == "__main__":
    main()
