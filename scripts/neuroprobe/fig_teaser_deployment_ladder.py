"""TEASER: the three deployment conditions a BCI engineer actually faces.

One model, the two speech tasks, the three regimes the benchmark itself defines.  Drawn as
three LABELLED GROUPS with no connecting line: these are distinct protocols, not points on a
shared axis, so nothing here implies that labels or sessions move you along a continuum.  That
distinction is the whole reason the calibration-curve version was rejected.

Deliberately NOT in this figure:
  - the zero-parameter floor  -> it is a results-section table; as ink it would dominate
  - seen vs unseen subjects   -> unseen mean k 1.108 vs seen 1.371, so at a squint it reads
                                 "unseen is worse".  Text claim, never a lead figure.
"""
import json, pathlib, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM = "pbs50_cd45k"
SH = pathlib.Path("results/r6_era/board/shards_pbs50_cd45k")
OUT = pathlib.Path("results/showcase/1_beats_the_board")
TASKS = ("onset", "speech")
# tap differs by regime -- ws/csession are scored per-electrode, cs on parcel means.
REGIMES = (("ws",       "enc12_elec", "same session"),
           ("csession", "enc12_elec", "new session, same patient\nno retraining"),
           ("cs",       "enc12",      "NEW PATIENT\nzero labels"))
BAR = {"onset": "#1f4e79", "speech": "#4b86b4"}
HERO = "#c1440e"


def macro(kind: str, tap: str, task: str) -> tuple[float, int]:
    vals = {}
    for f in sorted(SH.glob(f"{kind}_*.json")):
        d = json.loads(f.read_text())
        v = d["cells"].get(f"{ARM}|{task}", {}).get("cells", {}).get(f"{tap}|std")
        if v is not None:
            vals[d["name"]] = v["test"]
    return statistics.fmean(vals.values()), len(vals)


tab, ncells = {}, {}
for kind, tap, _ in REGIMES:
    for t in TASKS:
        m, n = macro(kind, tap, t)
        tab[(kind, t)] = m; ncells[kind] = n
assert ncells["ws"] == ncells["csession"] == 12, f"[check] VIOLATED ws/csession cells {ncells}"
assert ncells["cs"] == 10, f"[check] VIOLATED cs cells {ncells['cs']}"
print(f"[check] OK cells ws/csession {ncells['ws']}, cs {ncells['cs']}, arm {ARM}, from shards")
for kind, tap, lbl in REGIMES:
    print(f"  {kind:9s} {tap:11s} " + "  ".join(f"{t} {tab[(kind,t)]:.4f}" for t in TASKS))

fig, ax = plt.subplots(figsize=(8.6, 5.0))
W, GAP = .34, .26
for gi, (kind, _, lbl) in enumerate(REGIMES):
    hero = kind == "cs"
    for ti, t in enumerate(TASKS):
        x = gi * (2 * W + GAP) + ti * W
        v = tab[(kind, t)]
        ax.bar(x, v - .5, W * .92, bottom=.5, color=HERO if hero else BAR[t],
               alpha=1.0 if hero else .95, zorder=3,
               edgecolor="white", linewidth=1.2)
        ax.text(x, v + .012, f"{v:.3f}", ha="center", va="bottom", zorder=4,
                fontsize=13 if hero else 11.5, fontweight="bold" if hero else "normal",
                color=HERO if hero else "#333")
        ax.text(x, .512, t, ha="center", va="bottom", rotation=90, fontsize=9.5,
                color="white", zorder=4)
    ax.text(gi * (2 * W + GAP) + W / 2, .462, lbl, ha="center", va="top",
            fontsize=11, fontweight="bold" if hero else "normal",
            color=HERO if hero else "#333")

RIGHT = 2 * (2 * W + GAP) + 2 * W - .06
ax.axhline(.5, color="#444", lw=1.1)
ax.set_ylim(.40, 1.0); ax.set_xlim(-.34, RIGHT)
ax.set_yticks([.5, .6, .7, .8, .9, 1.0])
ax.set_yticklabels(["0.5\nchance", "0.6", "0.7", "0.8", "0.9", "1.0"])
ax.set_ylabel("AUROC", fontsize=11)
ax.set_xticks([])
ax.set_title("One model, three deployment conditions", fontsize=15, pad=14)
for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e6e6e6", lw=.8, zorder=0)
fig.text(.5, -.02, f"speech onset and speech-vs-silence · Neuroprobe Lite · "
         f"{ncells['ws']} sessions (within/cross-session), {ncells['cs']} held-out cells (cross-subject) · "
         f"shipped checkpoint {ARM}",
         ha="center", fontsize=8, color="#666")
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_teaser_deployment_ladder.{ext}", bbox_inches="tight", dpi=190)
print(f"  -> {OUT}/fig_teaser_deployment_ladder.pdf|.png")
