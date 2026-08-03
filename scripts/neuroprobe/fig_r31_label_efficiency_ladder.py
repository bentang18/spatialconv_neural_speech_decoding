"""R31 · the label-efficiency figure, drawn for all three rungs of the ladder.

Same two panels as `fig_r31_samplecurve.py`, once per regime:
  A  the saving   -- how many labelled trials enc12 needs for what enc0 needs at full data
  B  the falsifier -- the gap plotted in the gain law's own coordinate, with BOTH models fitted to
                      the SAME points so the reader sees which one the data pick

⚠️ COLUMN IS `trainonly`, NOT `both`. Only ws HAS a `both` column; cs and csession were run
`trainonly` only, and `trainonly` is therefore the only column all three regimes share. That is why
the ws row here reads ~2.8x and the claims-table headline (D1) reads 2.69x -- SAME measurement,
different column, NOT a correction. 🚫 Never quote a number off this figure next to D1's 2.69x.

⚠️ THE AXIS RANGES ARE SHARED, THE AXES ARE NOT THE SAME AXIS. Every row is drawn on identical
x and y limits so a saving in one row is the same number of centimetres as a saving in another --
that is the only reason they are shared. N still MEANS a different thing per row (target-session
trials / other-session trials, same patient / donor-session trials), which is why each row keeps
its own x label and why the rows stay separate panels. See `fig_r31_ws_vs_cs.py:9`.

⚠️ ws still tops out at 1750 labelled trials (half a session) where csession and cs reach 3500, so
the ws curve simply ends earlier on the shared axis. The cleanly comparable pair is csession-vs-cs.
The suptitle says so; do not crop it off.

The gates (anchor exact, unit count) and the tap map are IMPORTED from `fig_r31_ws_vs_cs`, not
restated -- a second copy of a gate is a gate that can silently drift from the first.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.neuroprobe.fig_r31_ws_vs_cs import ACC, GREY, OUT, SPEC, _style, load
from scripts.neuroprobe.v3_board_samplecurve import FULL, _reach

ORDER = ("ws", "csession", "cs")


def panel_saving(ax, d, colour, short, xlabel):
    """A · the curves and the saving arrow. Target is enc0 AT FULL DATA, same as everywhere else."""
    ns = [n for n in sorted(d["c0"], key=lambda v: (v == FULL, v)) if n != FULL]
    target, n_full = d["c0"][FULL], d["n_full"]
    reach = _reach({k: v for k, v in d["c12"].items() if k != FULL}, target)
    x = np.array(ns, float)

    ax.plot(x, [d["c0"][n] for n in ns], "-o", color=GREY, lw=1.5, ms=3.2,
            label="enc0 (|STFT| frontend)")
    ax.plot(x, [d["c12"][n] for n in ns], "-o", color=colour, lw=1.5, ms=3.2,
            label="enc12 (pretrained)")
    ax.plot([n_full], [target], "o", color=GREY, ms=5, mfc="white", mew=1.3, zorder=5)
    ax.plot([n_full], [d["c12"][FULL]], "o", color=colour, ms=5, mfc="white", mew=1.3, zorder=5)
    ax.axhline(target, color=GREY, ls=":", lw=0.9)

    if reach:
        ax.annotate("", xy=(reach, target), xytext=(n_full, target),
                    arrowprops=dict(arrowstyle="<->", color=ACC, lw=1.3, shrinkA=0, shrinkB=0))
        ax.annotate(f"{n_full / reach:.1f}× fewer labels", (np.sqrt(reach * n_full), target),
                    textcoords="offset points", xytext=(0, -7), ha="center", va="top",
                    color=ACC, fontsize=7.6, fontweight="bold")
    ax.annotate(f"enc0, full data ({target:.4f})", (ns[0], target), textcoords="offset points",
                xytext=(0, 3), ha="left", va="bottom", color=GREY, fontsize=6.2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("board macro AUROC")
    ax.set_title(f"{short}  ·  {reach:.0f} vs {n_full} trials", loc="left", fontsize=8)
    ax.legend(loc="lower right", fontsize=6.6)
    return dict(ns=ns, target=target, n_full=n_full, reach=reach)


def panel_falsifier(ax, d, colour, short):
    """B · additive vs multiplicative, both fitted to the same points. Lower rmse wins, and the
    winner is printed rather than asserted -- the point of the panel is that the data choose."""
    ns = [n for n in sorted(d["c0"], key=lambda v: (v == FULL, v)) if n != FULL]
    nb = ns + [FULL]
    lab = [str(n) for n in ns] + [f"full ({d['n_full']})"]
    # alternate above/below; the `full` label drops clear of the last N label when that one is
    # also below, otherwise the two overprint (csession/cs both land there).
    off = [(0, 6.0) if i % 2 == 0 else (0, -7.5) for i in range(len(ns))]
    off += [(0, -16.0) if len(ns) % 2 == 0 else (0, -8.0)]
    xs = np.array([d["c0"][n] - .5 for n in nb])
    ys = np.array([d["c12"][n] - .5 for n in nb])
    gap = ys - xs

    k_mult = float(xs @ ys / (xs @ xs))          # multiplicative, through the origin
    a_add = float(gap.mean())                    # additive, a horizontal line
    r_add = float(np.sqrt(np.mean((gap - a_add) ** 2)))
    r_mul = float(np.sqrt(np.mean((gap - (k_mult - 1) * xs) ** 2)))

    # Fit drawn only over the headroom THIS regime spans -- extrapolating cs's origin-line out to
    # ws's headroom would draw a prediction on data cs never saw, and it stretches the shared y.
    gx = np.linspace(0, xs.max() * 1.06, 100)
    ax.plot(gx, (k_mult - 1) * gx, "-", color=GREY, lw=1.4,
            label=f"multiplicative  $k$={k_mult:.3f}\n(through origin)")
    ax.axhline(a_add, color=ACC, lw=1.4, ls="--", label=f"additive  $a$={a_add:+.4f}")
    ax.plot(xs, gap, "o", color=colour, ms=4.4, zorder=5, label=f"measured (N = {ns[0]} … full)")
    for xi, gi, t, o in zip(xs, gap, lab, off):
        ax.annotate(t, (xi, gi), textcoords="offset points", xytext=o, ha="center",
                    va="bottom" if o[1] > 0 else "top", fontsize=5.5, color=colour)

    ax.axhline(0, color="#ccc", lw=.8, zorder=0)
    ax.set_xlabel(r"enc0 headroom  (AUROC$_0$ $-$ 0.5)")
    ax.set_ylabel(r"enc12 $-$ enc0  (AUROC)")
    win = "flat" if r_add < r_mul else "proportional"
    ax.set_title(f"{short}  ·  gap is {win}   rmse {r_add:.4f} vs {r_mul:.4f}",
                 loc="left", fontsize=8)
    ax.legend(loc="lower right", fontsize=6.2)
    return dict(a=a_add, k=k_mult, r_add=r_add, r_mul=r_mul,
                lo=min(0.0, float(gap.min())),
                hi=max(float(gap.max()), (k_mult - 1) * xs.max() * 1.06))


def main() -> None:
    got = {r: load(SPEC[r]["src"], r, SPEC[r]["units"]) for r in ORDER}

    _style()
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 8.6))

    # SHARED RANGES, computed before anything is drawn so no row is scaled to itself. A saving
    # spanning half a panel must span half a panel in every row or the rows cannot be eyeballed
    # against each other -- that, and only that, is what sharing the limits buys here.
    gx_max = max(v - .5 for d in got.values() for v in d["c0"].values()) * 1.06
    lo_a = min(v for d in got.values() for v in d["c0"].values())
    hi_a = max(v for d in got.values() for v in d["c12"].values())
    pad = (hi_a - lo_a) * .07
    x_lo = min(n for d in got.values() for n in d["c0"] if n != FULL) / 1.45
    x_hi = max(d["n_full"] for d in got.values()) * 1.45

    rows = []
    for i, r in enumerate(ORDER):
        s, d = SPEC[r], got[r]
        a = panel_saving(axes[i][0], d, s["colour"], f"{'ABC'[i]}{i + 1} · {s['short']}",
                         s["xlabel"])
        b = panel_falsifier(axes[i][1], d, s["colour"], f"{'ABC'[i]}{i + 1} · {s['short']}")
        rows.append((r, s, d, a, b))

    y_lo = min(b["lo"] for *_, b in rows) * 1.15
    y_hi = max(b["hi"] for *_, b in rows) * 1.30
    for i in range(3):
        axes[i][0].set_xlim(x_lo, x_hi)
        axes[i][0].set_ylim(lo_a - pad, hi_a + pad)
        axes[i][1].set_xlim(0, gx_max)
        axes[i][1].set_ylim(y_lo, y_hi)

    fig.suptitle(
        "R31 · label efficiency across the three rungs · pbs50_cd45k · column 'trainonly' "
        "(the only column all three regimes share — ws reads 2.8× here vs 2.69× on 'both')\n"
        "180 / 180 / 150 units · anchors exact (max |curve−published| = 0e+00) · rows share x and y "
        "RANGES so the savings compare by eye, but N means a different thing per row (see x labels)\n"
        "⚠ ws tops out at 1750 trials, csession and cs at 3500 — csession-vs-cs is the "
        "cleanly comparable pair",
        fontsize=7.0, y=1.005, color="#444")
    fig.tight_layout(h_pad=2.2)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_r31_label_efficiency_ladder.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"{'rung':10s} {'units':>6s} {'anchor':>8s} {'target':>8s} {'n_full':>7s} "
          f"{'reach':>7s} {'saving':>8s} {'a':>9s} {'k':>7s} {'rmse add':>9s} {'rmse mul':>9s}  verdict")
    for r, s, d, a, b in rows:
        print(f"{r:10s} {d['units']:6d} {d['anchor']:8.0e} {a['target']:8.4f} {a['n_full']:7d} "
              f"{a['reach']:7.0f} {a['n_full'] / a['reach']:7.2f}x {b['a']:+9.4f} {b['k']:7.3f} "
              f"{b['r_add']:9.5f} {b['r_mul']:9.5f}  "
              f"{'ADDITIVE' if b['r_add'] < b['r_mul'] else 'MULTIPLICATIVE'}")

    # The saving is TARGET-DEPENDENT and it is SMALLEST at small N. Printed for every rung because
    # that is the line of evidence that keeps "especially valuable when labels are scarce" dead.
    for r, s, d, a, b in rows:
        row = []
        for n in a["ns"][3:]:
            hit = _reach({k: v for k, v in d["c12"].items() if k != FULL}, d["c0"][n])
            if hit:
                row.append(f"{n}->{hit:.0f}={n / hit:.2f}x")
        print(f"\nSAVING vs target (enc0 at N) · {r}:  " + "  ".join(row)
              + f"  full={a['n_full'] / a['reach']:.2f}x")
    print(f"\n  -> {OUT}/fig_r31_label_efficiency_ladder.pdf|.png")


if __name__ == "__main__":
    main()
