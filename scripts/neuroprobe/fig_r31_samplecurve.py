"""R31 showcase figure — label efficiency, and the additive-vs-multiplicative test.

The aggregation is IMPORTED from v3_board_samplecurve, never reimplemented: `_curve` and `_reach`
are the same functions that produced the merged report, so the figure cannot silently disagree with
the number we quote. If they change, this figure changes with them.

Column 'both' (train AND val subsampled) is the primary panel on purpose -- it is the honest
calibration budget: a site that only has N labelled trials cannot tune lambda on a full val set it
does not have. It is also the CONSERVATIVE of the two (2.69x vs 2.78x).

Panel B is the falsifier, plotted in the gain-law's own coordinate. Two models are drawn against
the SAME points, each fit to those points, so the reader sees which one the data pick:
  multiplicative  gap = (k-1) * x     -- a line through the ORIGIN
  additive        gap = a             -- a horizontal line
"""
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.neuroprobe.v3_board_samplecurve import _curve, _reach, FULL

SRC = pathlib.Path("results/showcase/2_what_pretraining_does/samplecurve_pbs50_cd45k.json")
OUT = pathlib.Path("results/showcase/2_what_pretraining_does")
COL = "both"
INK, GREY, ACC = "#1f4e79", "#8a8f98", "#d98324"


def _style() -> None:                      # lifted from paper_figs_r6._style, same paper look
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "legend.frameon": False, "axes.titlesize": 8.5, "axes.labelsize": 8,
    })


def main() -> None:
    m = json.load(open(SRC))
    pts = m["points"]

    # ── the anchor GATES the figure. A drifted anchor means the curve is not on the board
    # protocol, and a pretty plot of an off-protocol curve is worse than no plot. ─────────────
    worst = max(m["anchor"], key=lambda a: a["absdiff"])
    assert worst["absdiff"] < 1e-9, f"ANCHOR DRIFTED ({worst}) — refusing to draw"
    cells = sorted({p["cell"] for p in pts}); tasks = sorted({p["task"] for p in pts})
    units = len(cells) * len(tasks)
    assert units == 180, f"expected the 180-unit board convention, got {units}"

    c12, c0 = _curve(pts, "enc12_elec", COL), _curve(pts, "enc0_elec", COL)
    ns = [n for n in sorted(c0, key=lambda v: (v == FULL, v)) if n != FULL]
    n_full = int(np.median([p["n"] for p in pts if p["tap"] == "enc0_elec" and p["n_is_full"]]))
    target = c0[FULL]
    reach = _reach({k: v for k, v in c12.items() if k != FULL}, target)

    _style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.1, 2.85))

    # ── A · label efficiency ────────────────────────────────────────────────────────────────
    x = np.array(ns, float)
    axA.plot(x, [c0[n] for n in ns], "-o", color=GREY, lw=1.5, ms=3.4, label="enc0 (|STFT| frontend)")
    axA.plot(x, [c12[n] for n in ns], "-o", color=INK, lw=1.5, ms=3.4, label="enc12 (pretrained)")
    axA.plot([n_full], [c0[FULL]], "o", color=GREY, ms=5, mfc="white", mew=1.3, zorder=5)
    axA.plot([n_full], [c12[FULL]], "o", color=INK, ms=5, mfc="white", mew=1.3, zorder=5)
    axA.axhline(target, color=GREY, ls=":", lw=0.9)
    axA.annotate("", xy=(reach, target), xytext=(n_full, target),
                 arrowprops=dict(arrowstyle="<->", color=ACC, lw=1.3, shrinkA=0, shrinkB=0))
    axA.text(np.sqrt(reach * n_full), target - 0.0125, f"{n_full/reach:.1f}× fewer labels",
             ha="center", va="top", color=ACC, fontsize=8, fontweight="bold")
    axA.annotate(f"enc0, full data ({target:.4f})", (ns[0], target), textcoords="offset points",
                 xytext=(0, 3), ha="left", va="bottom", color=GREY, fontsize=6.6)
    axA.set_xscale("log", base=2)
    axA.set_xlabel("labelled trials used to fit the readout  (N, log$_2$)")
    axA.set_ylabel("board macro AUROC")
    axA.set_title(f"A · enc12 needs ~{reach:.0f} trials for what enc0 needs ~{n_full}", loc="left")
    axA.legend(loc="lower right", fontsize=7)

    # ── B · the falsifier, in the gain law's coordinate ──────────────────────────────────────
    nb = ns + [FULL]                             # panel B plots FULL too — it is a measured point
    lab = [str(n) for n in ns] + [f"full ({n_full})"]
    off = [(0, 6.5)] * len(ns) + [(0, -8)]       # 'full' goes BELOW — it collides with 1024 above
    xs = np.array([c0[n] - .5 for n in nb]); ys = np.array([c12[n] - .5 for n in nb])
    gap = ys - xs
    k_mult = float(xs @ ys / (xs @ xs))          # multiplicative fit, through the origin
    a_add = float(gap.mean())                    # additive fit
    gx = np.linspace(0, xs.max() * 1.06, 100)
    axB.plot(gx, (k_mult - 1) * gx, "-", color=GREY, lw=1.4,
             label=f"multiplicative  $k$={k_mult:.3f}\n(through origin)")
    axB.axhline(a_add, color=ACC, lw=1.4, ls="--", label=f"additive  $a$={a_add:+.4f}")
    axB.plot(xs, gap, "o", color=INK, ms=4.6, zorder=5, label=f"measured (N = {ns[0]} … full)")
    for xi, gi, t, o in zip(xs, gap, lab, off):
        axB.annotate(t, (xi, gi), textcoords="offset points", xytext=o, ha="center",
                     va="bottom" if o[1] > 0 else "top", fontsize=5.8, color=INK)
    r_add = float(np.sqrt(np.mean((gap - a_add) ** 2)))
    r_mul = float(np.sqrt(np.mean((gap - (k_mult - 1) * xs) ** 2)))
    axB.set_xlim(0, xs.max() * 1.06); axB.set_ylim(0, max(gap.max(), (k_mult-1)*xs.max()) * 1.32)
    axB.set_xlabel(r"enc0 headroom  (AUROC$_0$ $-$ 0.5)")
    axB.set_ylabel(r"enc12 $-$ enc0  (AUROC)")
    axB.set_title(f"B · gap is flat, not proportional   rmse {r_add:.4f} vs {r_mul:.4f}", loc="left")
    axB.legend(loc="upper left", fontsize=6.6)

    fig.suptitle(f"R31 · within-session label efficiency · {m['tags'][0]} · "
                 f"{len(cells)} sessions × {len(tasks)} tasks = {units} units · "
                 f"anchor exact (max |curve−published| = {worst['absdiff']:.0e})",
                 fontsize=7.2, y=1.02, color="#444")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_r31_label_efficiency.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"units {units} | anchor {worst['absdiff']:.0e} | target {target:.4f} | "
          f"enc0 {n_full} -> enc12 {reach:.0f} = {n_full/reach:.2f}x")
    print(f"PANEL B  additive a={a_add:+.4f} rmse {r_add:.5f} | multiplicative k={k_mult:.3f} "
          f"rmse {r_mul:.5f}  ->  {'ADDITIVE wins' if r_add < r_mul else 'MULTIPLICATIVE wins'}")

    # ── the saving is TARGET-DEPENDENT. Printed so nobody quotes the headline as a constant,
    # and so the "scarce labels" sentence stays dead: the saving is SMALLEST at small N. ───────
    row = []
    for n in ns[3:]:
        r = _reach({k: v for k, v in c12.items() if k != FULL}, c0[n])
        if r:
            row.append(f"{n}->{r:.0f}={n/r:.2f}x")
    print("SAVING vs target (enc0 at N):  " + "  ".join(row) + f"  full={n_full/reach:.2f}x")
    assert n_full / reach == max([n_full / reach] + [n / _reach(
        {k: v for k, v in c12.items() if k != FULL}, c0[n]) for n in ns[3:]]), \
        "the full-data target no longer gives the LARGEST saving — recheck the scarce-label ban"
    print(f"  -> {OUT}/fig_r31_label_efficiency.pdf|.png")


if __name__ == "__main__":
    main()
