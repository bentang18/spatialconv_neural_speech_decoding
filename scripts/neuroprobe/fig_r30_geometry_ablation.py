"""R30: what each correspondence prior is worth, drawn as gain over the zero-parameter floor.

Four arms at the 45k board point -- both priors, no shaft RoPE, no parcel embedding, neither --
each as enc12 minus enc0, one panel per regime.

WHY enc12 - enc0 AND NOT arm-minus-baseline:
    enc0 never touches the tower (`v3_probe_encode_r4.py:477`, "enc0 never reads weights"), so the
    floor is the SAME array for every arm.  Verified here: max|delta| across arms is 0 in ws and
    csession and 4e-4 in cs, against a macro identical to 4 decimal places.  That makes the two
    tables ALGEBRAICALLY THE SAME OBJECT, offset by a constant -- the depth gain of the baseline.
    They are one result shown once, not two results.  Never quote both p-value sets as two findings.

Provenance discipline:
  ours   -> recomputed from the SHARDS (never a merged JSON), std / ridge / test, 15 tasks
  taps   -> ws/csession scored per-electrode (enc*_elec), cs on parcel means (enc*)
  arms   -> one shard directory per arm; the ledger tags several artifacts `pbs50_cd45k`
            (board_ft, ens_ws, ens_cs), so the artifact path is filtered, not the arm tag

Null: Wilcoxon signed-rank on the per-cell mean gain against zero, 12 cells in ws and csession,
10 in cs.  n=10 puts the smallest attainable p at ~2e-3, so an unstarred bar is "does not clear
its null at this n", not "is zero".
"""
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # NeurIPS rejects Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

LEDGER = pathlib.Path("results/r6_era/RESULTS_LEDGER.csv")
OUT = pathlib.Path("results/showcase/3_correspondence")

ARMS = (
    ("baseline", "both priors",
     ["results/r6_era/board/shards_pbs50_cd45k"]),
    ("nosrope", "no shaft RoPE",
     ["results/r6_era/board/shards_board_nosrope_trap45k"]),
    ("noparcel", "no parcel embed",
     ["results/r6_era/board/shards_board_noparcel_trap45k",
      "results/r6_era/board/shards_board_noparcel_trap45k_enc0"]),
    ("both_off", "neither",
     ["results/r6_era/board/shards_board_noparcel_nosrope_trap45k"]),
)
REGIMES = (("ws", "enc0_elec", "enc12_elec", "Within session"),
           ("csession", "enc0_elec", "enc12_elec", "Cross session\nnew session, same patient"),
           ("cs", "enc0", "enc12", "Cross subject\nnew patient, zero labels"))

OURS, ABLATED, FLOOR_RED, INK = "#1f4e79", "#9aa7b4", "#c1440e", "#333333"


def load():
    L = pd.read_csv(LEDGER)
    return L[(L.norm == "std") & (L.decoder == "ridge") & (L.split == "test")]


def series(L, artifacts, regime, tap):
    a = L[L.artifact.isin(artifacts) & (L.regime == regime) & (L.tap == tap)]
    if a.empty:
        return None
    s = a.set_index(["cell", "task"]).value
    assert not s.index.duplicated().any(), f"duplicate rows for {artifacts}/{regime}/{tap}"
    return s


def collect(L):
    rows = []
    for regime, t0, t12, _ in REGIMES:
        floor = series(L, ARMS[0][2], regime, t0)
        for arm, label, artifacts in ARMS:
            own = series(L, artifacts, regime, t0)
            if own is None:
                drift = None                      # noparcel ws/csession carry no enc0 shard
            else:
                m = floor.index.intersection(own.index)
                drift = float(np.abs(own[m].values - floor[m].values).max())
                assert drift < 2e-3, f"{arm}/{regime} enc0 drifts {drift:.1e} from the floor"
            e12 = series(L, artifacts, regime, t12)
            idx = floor.index.intersection(e12.index)
            gain = e12[idx] - floor[idx]
            per_cell = gain.groupby(level=0).mean()
            rows.append(dict(regime=regime, arm=arm, label=label,
                             enc0=float(floor[idx].mean()), enc12=float(e12[idx].mean()),
                             gain=float(gain.mean()), ncell=len(per_cell),
                             npos=int((per_cell > 0).sum()),
                             p=float(wilcoxon(per_cell.values).pvalue),
                             n=len(idx), enc0_drift=drift))
    return pd.DataFrame(rows)


def render(R):
    OUT.mkdir(parents=True, exist_ok=True)
    lo, hi = R.gain.min(), R.gain.max()
    pad = (hi - lo) * .30
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.9), sharey=True)
    for ax, (regime, _, _, title) in zip(axes, REGIMES):
        d = R[R.regime == regime].set_index("arm").loc[[a for a, _, _ in ARMS]]
        x = np.arange(len(d))
        ax.bar(x, d.gain, .66, zorder=3,
               color=[OURS] + [ABLATED] * 3,
               edgecolor="none")
        ax.axhline(0, color=FLOOR_RED, lw=2.0, zorder=4)
        for i, (g, p) in enumerate(zip(d.gain, d.p)):
            up = g >= 0
            ax.text(i, g + (pad * .10 if up else -pad * .10), f"{g:+.4f}",
                    ha="center", va="bottom" if up else "top",
                    fontsize=9.0, fontweight="bold",
                    color=OURS if i == 0 else "#5b6773")
            if p >= .05:
                ax.text(i, g + (pad * .34 if up else -pad * .34), "n.s.",
                        ha="center", va="bottom" if up else "top",
                        fontsize=8.2, style="italic", color="#8a5a3b")
        ax.set_xticks(x)
        ax.set_xticklabels(d.label, fontsize=8.8, rotation=18, ha="right")
        ax.set_title(title, fontsize=9.6, color=INK, pad=8)
        ax.set_ylim(lo - pad, hi + pad)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#eee", lw=.8, zorder=0)
        ax.tick_params(axis="y", length=0, labelsize=8.6)
    axes[0].set_ylabel("gain over the zero-parameter floor\n(AUROC, enc12 − enc0)",
                       fontsize=9.2, color=INK)
    axes[0].text(-.36, 0, "floor", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="center", fontsize=8.6, color=FLOOR_RED, fontweight="bold")
    fig.text(.5, -.055,
             "45k board checkpoint  ·  15 Neuroprobe Lite tasks  ·  frozen encoder, ridge readout  ·  "
             "recomputed from shards  ·  n.s. = does not clear Wilcoxon signed-rank over cells at p<.05",
             ha="center", fontsize=7.6, color="#666")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_r30_geometry_ablation.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def main():
    R = collect(load())
    render(R)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False,
                      formatters={"enc0": "{:.4f}".format, "enc12": "{:.4f}".format,
                                  "gain": "{:+.4f}".format, "p": "{:.2g}".format}))
    print("\ninteraction  (both priors + neither) − (no RoPE + no parcel):")
    for regime, _, _, _ in REGIMES:
        g = R[R.regime == regime].set_index("arm").gain
        print(f"  {regime:<9}{g['baseline'] + g['both_off'] - g['nosrope'] - g['noparcel']:+.4f}")
    print(f"\nwrote {OUT}/fig_r30_geometry_ablation.{{png,pdf}}")


if __name__ == "__main__":
    main()
