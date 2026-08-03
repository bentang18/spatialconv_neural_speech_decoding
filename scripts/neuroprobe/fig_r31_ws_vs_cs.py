"""R31 · what pretraining affords WITHIN a patient vs ACROSS patients.

The curves answer different questions and the figure is built so they cannot be read as one:

  WS        N = calibration trials from the TARGET session   "how much of THIS patient do I need?"
  CSESSION  N = trials from the same patient, ANOTHER day    "how much of this BRAIN do I need?"
  CS        N = labelled trials from the DONOR session S2T4  "how much of SOMEONE ELSE do I need?"

⚠️ The x axes are therefore NOT the same axis and never share one. The curve panels are separate for
that reason alone.

⚠️ THE LADDER IS NOT ONE-VARIABLE END TO END. ws and csession share the SAME tap (`enc*_elec`); cs
reads `enc*` (parcel-mean) because electrode identity does not survive a subject change. So
ws → csession is a clean one-variable step — only the train data moved — and anything that survives
it is about the DATA. csession → cs moves the tap AND the brain at once, and every contrast that
crosses it is confounded on both. Read the ladder in that order and the confound stays visible.

The law panel is the one place the regimes ARE comparable: the gain-law coordinate (gap vs headroom)
is a ratio of AUROCs in both, carries no trial-count units, and is exactly the coordinate the gain
law is stated in. Two models are fit to each regime's points and drawn against them:

  multiplicative  gap = (k-1)·x   — a line through the ORIGIN. Pretraining AMPLIFIES what the
                                    frontend already exposes, and gives nothing where it exposes
                                    nothing.
  additive        gap = a         — a horizontal line. Pretraining supplies a fixed increment that
                                    the readout cannot buy with more labels.

⚠️ ALL regimes are read off the `trainonly` column. WS also has a `both` column (train AND val
subsampled) and it is the more conservative, more honest calibration budget — but the off-target
regimes have no such column by construction (their val belongs to the target and to the protocol),
so quoting WS-`both` against their `trainonly` would contrast two different experiments. The WS
`both` numbers are printed separately, never plotted here.

Nothing in this file decides the verdict: `_addmult` returns it with its nulls and a bootstrap over
SUBJECTS, and the title prints whatever comes back.
"""
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.neuroprobe.v3_board_samplecurve import (
    _addmult, _curve, _panel, _panel_fit, _reach, CURVE_TAPS, FULL)

OUT = pathlib.Path("results/showcase/2_what_pretraining_does")
COL = "trainonly"                      # the ONLY column every regime has
GREY, ACC = "#8a8f98", "#d98324"
NBOOT = 4000

# The ladder, in order of distance from the target recording. `units` is asserted at load: a merge
# that timed out yields FEWER cells, and a partial cell is a different experiment wearing the same
# name -- so the count is a gate, not a label.
SPEC = {
    "ws":       dict(src=OUT / "samplecurve_pbs50_cd45k.json", units=180, colour="#1f4e79",
                     name="within-session", short="within a patient",
                     xlabel="target-session calibration trials  (N, log$_2$)"),
    "csession": dict(src=OUT / "samplecurve_csession_pbs50_cd45k.json", units=180, colour="#2f6f4e",
                     name="cross-session", short="same patient, another day",
                     xlabel="OTHER-SESSION labelled trials, same patient  (N, log$_2$)"),
    "cs":       dict(src=OUT / "samplecurve_cs_pbs50_cd45k.json", units=150, colour="#7b2d26",
                     name="cross-subject", short="across patients",
                     xlabel="DONOR-session labelled trials  (N, log$_2$)"),
}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "legend.frameon": False, "axes.titlesize": 8.5, "axes.labelsize": 8,
    })


def load(src, regime, want_units):
    m = json.load(open(src))
    pts = m["points"]
    # The anchor GATES the panel. A drifted anchor means the curve is not on the board protocol,
    # and a pretty plot of an off-protocol curve is worse than no plot at all.
    worst = max(m["anchor"], key=lambda a: a["absdiff"])
    assert worst["absdiff"] < 1e-9, f"{regime} ANCHOR DRIFTED ({worst}) — refusing to draw"
    cells = sorted({p["cell"] for p in pts}); tasks = sorted({p["task"] for p in pts})
    units = len(cells) * len(tasks)
    assert units == want_units, f"{regime}: expected {want_units} units, got {units}"
    tap0, tap12 = CURVE_TAPS[regime]
    c0, c12 = _curve(pts, tap0, COL), _curve(pts, tap12, COL)
    n_full = int(np.median([p["n"] for p in pts if p["tap"] == tap0 and p["n_is_full"]]))
    am = _addmult(pts, tap0, tap12, COL, nboot=NBOOT)
    return dict(m=m, pts=pts, c0=c0, c12=c12, n_full=n_full, am=am,
                cells=cells, tasks=tasks, units=units, anchor=worst["absdiff"])


def verdict(am):
    a_sig = not (am["a_ci"][0] <= 0 <= am["a_ci"][1])
    k_sig = not (am["k_ci"][0] <= 1 <= am["k_ci"][1])
    return {(True, False): "ADDITIVE", (False, True): "MULTIPLICATIVE",
            (True, True): "BOTH", (False, False): "UNDERPOWERED"}[(a_sig, k_sig)]


def paired_contrast(A, B, ra, rb, nboot=NBOOT, seed=0):
    """Is the LAW different between regimes? Tested PAIRED on the subjects both regimes share.

    Two separate CIs overlapping is not a test of a difference, and with only 5 CS subjects the
    marginal CIs are wide enough that reading the dissociation off them would be reading noise.
    Resampling the SHARED subjects once and refitting both regimes on that same draw cancels the
    subject-level noise the two regimes have in common, which is the whole point of pairing.

    ⚠️ WHAT THIS DOES AND DOES NOT LICENSE — AND IT DEPENDS ON WHICH PAIR. ws vs csession share the
    tap (`enc*_elec`), so their contrast is one-variable and the magnitude of the difference means
    what it says. Any pair involving cs does NOT: cs is scored parcel-mean with different feature
    dimensionality, so the ABSOLUTE size of `a` is not commensurable and a difference in magnitude
    alone would mean little there. The claim a cs-crossing contrast supports is about SHAPE --
    whether the gap is a constant offset or scales with the headroom -- which is why both the
    intercept and the slope are reported, and why the caller prints which kind of pair it asked for.
    """
    wt, wc, wsubs, _ = _panel(A["pts"], *CURVE_TAPS[ra], COL)
    ct, cc, csubs, _ = _panel(B["pts"], *CURVE_TAPS[rb], COL)
    shared = sorted(set(wsubs) & set(csubs))
    # Restrict BOTH panels to the shared subjects, in the same order, so one index vector addresses
    # the same patient in each regime. That identity is what makes the resample paired.
    wi = [wsubs.index(s) for s in shared]
    ci_ = [csubs.index(s) for s in shared]
    fw = _panel_fit(wt[wi], wc[wi])
    fc = _panel_fit(ct[ci_], cc[ci_])

    rng = np.random.default_rng(seed)
    da, dk = [], []
    for _ in range(nboot):
        # ONE draw, both regimes — that is what makes it paired and cancels the shared subject noise.
        draw = rng.integers(0, len(shared), len(shared))
        aw, kw = fw(draw)
        ac, kc = fc(draw)
        if aw is None or ac is None:
            continue
        da.append(aw - ac); dk.append(kc - kw)
    base = np.arange(len(shared))
    aw0, kw0 = fw(base)
    ac0, kc0 = fc(base)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"shared": shared, "ra": ra, "rb": rb, "a_a": aw0, "a_b": ac0, "k_a": kw0, "k_b": kc0,
            "d_a": aw0 - ac0, "d_a_ci": q(da), "d_k": kc0 - kw0, "d_k_ci": q(dk),
            "same_tap": CURVE_TAPS[ra] == CURVE_TAPS[rb], "n_boot": len(da)}


def panel_curve(ax, d, title, xlabel, colour):
    ns = [n for n in sorted(d["c0"], key=lambda v: (v == FULL, v)) if n != FULL]
    target = d["c0"][FULL]
    reach = _reach({k: v for k, v in d["c12"].items() if k != FULL}, target)
    x = np.array(ns, float)
    ax.plot(x, [d["c0"][n] for n in ns], "-o", color=GREY, lw=1.5, ms=3.4,
            label="enc0 (|STFT| frontend, 0 params)")
    ax.plot(x, [d["c12"][n] for n in ns], "-o", color=colour, lw=1.5, ms=3.4,
            label="enc12 (pretrained)")
    ax.plot([d["n_full"]], [target], "o", color=GREY, ms=5, mfc="white", mew=1.3, zorder=5)
    ax.plot([d["n_full"]], [d["c12"][FULL]], "o", color=colour, ms=5, mfc="white", mew=1.3, zorder=5)
    ax.axhline(target, color=GREY, ls=":", lw=0.9)
    if reach is not None and reach < d["n_full"]:
        ax.annotate("", xy=(reach, target), xytext=(d["n_full"], target),
                    arrowprops=dict(arrowstyle="<->", color=ACC, lw=1.3, shrinkA=0, shrinkB=0))
        ax.text(np.sqrt(reach * d["n_full"]), target - 0.010,
                f"{d['n_full']/reach:.1f}× fewer", ha="center", va="top",
                color=ACC, fontsize=7.4, fontweight="bold")
        sub = f"{d['n_full']/reach:.1f}× fewer labels"
    else:
        sub = "never reaches it"
    ax.annotate(f"enc0, full data ({target:.4f})", (ns[0], target), textcoords="offset points",
                xytext=(0, 3), ha="left", va="bottom", color=GREY, fontsize=6.4)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("board macro AUROC")
    ax.set_title(f"{title}   {sub}", loc="left")
    ax.legend(loc="lower right", fontsize=6.6)
    return reach


def panel_law(ax, loaded):
    """Every regime in the gain law's own coordinate, each with the two competing fits."""
    xmax = 0.0
    for r, d in loaded.items():
        colour, name = SPEC[r]["colour"], SPEC[r]["name"]
        nb = [n for n in sorted(d["c0"], key=lambda v: (v == FULL, v))]
        xs = np.array([d["c0"][n] - .5 for n in nb]); ys = np.array([d["c12"][n] - .5 for n in nb])
        gap = ys - xs
        a, k = d["am"]["a"], d["am"]["k"]
        gx = np.linspace(0, xs.max() * 1.06, 60)
        ax.plot(gx, a + (k - 1) * gx, "-", color=colour, lw=1.3, alpha=.85)
        ax.plot(xs, gap, "o", color=colour, ms=4.4, zorder=5,
                label=f"{name}: $a$={a:+.4f} [{d['am']['a_ci'][0]:+.4f},{d['am']['a_ci'][1]:+.4f}], "
                      f"$k$={k:.3f} [{d['am']['k_ci'][0]:.2f},{d['am']['k_ci'][1]:.2f}]\n"
                      f"      ⇒ {verdict(d['am'])}")
        d["_xs"], d["_gap"] = xs, gap
        xmax = max(xmax, float(xs.max()))
    ax.axhline(0, color="#ccc", lw=.8, zorder=0)
    ax.set_xlim(0, xmax * 1.06)
    ax.set_xlabel(r"enc0 headroom  (AUROC$_0$ $-$ 0.5)")
    ax.set_ylabel(r"enc12 $-$ enc0  (AUROC)")
    ax.set_title(f"{chr(65 + len(loaded))} · the falsifier: intercept vs slope", loc="left")
    ax.legend(loc="upper left", fontsize=5.6)


def main() -> None:
    # csession is the tap-matched control and may still be on the queue. Draw what is on disk and
    # SAY which rungs are missing -- a silently 2-panel figure reads as a finished 2-panel figure.
    loaded, missing = {}, []
    for r, spec in SPEC.items():
        if spec["src"].exists():
            loaded[r] = load(spec["src"], r, spec["units"])
        else:
            missing.append(r)
    assert "ws" in loaded and "cs" in loaded, f"need ws and cs on disk, have {sorted(loaded)}"

    _style()
    ncol = len(loaded) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(3.55 * ncol, 2.95))
    for i, (r, d) in enumerate(loaded.items()):
        panel_curve(axes[i], d, f"{chr(65 + i)} · {SPEC[r]['short']}",
                    SPEC[r]["xlabel"], SPEC[r]["colour"])
    panel_law(axes[-1], loaded)

    units = ", ".join(f"{r.upper()} {len(d['cells'])}×{len(d['tasks'])}={d['units']} units"
                      for r, d in loaded.items())
    fig.suptitle(
        f"R31 · {loaded['ws']['m']['tags'][0]} · column '{COL}' · {units} · "
        f"anchors exact (max |curve−published| "
        f"{max(d['anchor'] for d in loaded.values()):.0e}) · {NBOOT} subject-bootstraps"
        + (f" · ⚠️ NOT ON DISK: {', '.join(missing)}" if missing else ""),
        fontsize=6.8, y=1.03, color="#444")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_r31_ws_vs_cs.{ext}", bbox_inches="tight")
    plt.close(fig)

    # ── the verdict, printed with its nulls so the figure is never the only record ─────────────
    print("=" * 92)
    print(f"R31 label ladder · column '{COL}' · NULLS: a = 0, k = 1")
    if missing:
        print(f"⚠️ NOT ON DISK, rungs skipped: {', '.join(missing)}")
    print("=" * 92)
    for r, d in loaded.items():
        name, am = SPEC[r]["name"], d["am"]
        target = d["c0"][FULL]
        reach = _reach({k: v for k, v in d["c12"].items() if k != FULL}, target)
        print(f"\n{name}  ({d['units']} units, {am['n_subjects']} subjects, "
              f"{am['n_points']} N-points)")
        print(f"  enc0 full {target:.4f}  enc12 full {d['c12'][FULL]:.4f}  "
              f"gap {d['c12'][FULL] - target:+.4f}")
        print(f"  labels to match enc0-full: enc0 {d['n_full']}  ->  enc12 "
              f"{'never' if reach is None else f'{reach:.0f}'}"
              + ("" if reach is None else f"  = {d['n_full']/reach:.2f}x"))
        print(f"  a = {am['a']:+.5f}  95% CI [{am['a_ci'][0]:+.5f}, {am['a_ci'][1]:+.5f}]")
        print(f"  k = {am['k']:.4f}  95% CI [{am['k_ci'][0]:.4f}, {am['k_ci'][1]:.4f}]")
        print(f"  ⇒ {verdict(am)}")

    # ── the DISSOCIATION itself, paired. Two overlapping CIs are not a test of a difference. ────
    # Every ADJACENT rung of the ladder, so the tap-matched step is reported next to the confounded
    # one and nobody has to remember which is which.
    rungs = list(loaded)
    pairs = list(zip(rungs, rungs[1:]))
    if len(rungs) > 2:                      # ...plus end-to-end, the number already in the ledger
        pairs.append((rungs[0], rungs[-1]))
    for ra, rb in pairs:
        pc = paired_contrast(loaded[ra], loaded[rb], ra, rb)
        A, B = ra.upper(), rb.upper()
        print("\n" + "-" * 92)
        print(f"PAIRED CONTRAST {A} vs {B} over the {len(pc['shared'])} subjects they share "
              f"({', '.join(pc['shared'])}) · {pc['n_boot']} bootstraps")
        if pc["same_tap"]:
            print(f"  ✅ TAP-MATCHED ({CURVE_TAPS[ra][0]}/{CURVE_TAPS[ra][1]} both sides) — the only "
                  "thing that moved is the TRAIN DATA, so the magnitude means what it says.")
        else:
            print(f"  ⚠️ TAPS DIFFER ({'/'.join(CURVE_TAPS[ra])} vs {'/'.join(CURVE_TAPS[rb])}) — the "
                  "brain AND the readout unit moved together.")
            print("     Claim SHAPE only (constant offset vs scaling with headroom), never the "
                  "absolute size of a.")
        print(f"  a:  {A} {pc['a_a']:+.5f}   {B} {pc['a_b']:+.5f}")
        print(f"      a_{A} - a_{B} = {pc['d_a']:+.5f}  95% CI [{pc['d_a_ci'][0]:+.5f}, "
              f"{pc['d_a_ci'][1]:+.5f}]   NULL: 0")
        print(f"  k:  {A} {pc['k_a']:.4f}   {B} {pc['k_b']:.4f}")
        print(f"      k_{B} - k_{A} = {pc['d_k']:+.5f}  95% CI [{pc['d_k_ci'][0]:+.5f}, "
              f"{pc['d_k_ci'][1]:+.5f}]   NULL: 0")
        a_sig = not (pc["d_a_ci"][0] <= 0 <= pc["d_a_ci"][1])
        k_sig = not (pc["d_k_ci"][0] <= 0 <= pc["d_k_ci"][1])
        if a_sig or k_sig:
            print(f"  ⇒ THE LAW DIFFERS ACROSS THIS STEP "
                  f"({'intercept' if a_sig else ''}{' and ' if a_sig and k_sig else ''}"
                  f"{'slope' if k_sig else ''} separate from 0)")
        else:
            print("  ⇒ NO MEASURED DIFFERENCE IN LAW across this step at this power — 🚫 do NOT "
                  "write the dissociation")
    print(f"\n  -> {OUT}/fig_r31_ws_vs_cs.pdf|.png")


if __name__ == "__main__":
    main()
