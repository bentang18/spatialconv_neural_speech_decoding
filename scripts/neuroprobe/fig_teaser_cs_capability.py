"""Teaser: cross-subject capability with ZERO labels from the target patient.

Two facts in one glance -- the LEVEL on speech tasks, and that the gain is speech-SELECTIVE
rather than uniform.  Deliberately different from `fig5cs_per_task_ladder`:

  fig5 (mechanism)  4-checkpoint mean, merged JSONs, ranked by k  -> onset lands 6th
  this  (capability) the SHIPPED checkpoint only, from SHARDS,    -> ranked by what it decodes

A capability claim has to quote the checkpoint that was submitted, so no checkpoint averaging
here.  k is the ledger's estimator (no-intercept slope over cells), NOT a ratio of macro means.
All 15 tasks are drawn: selectivity is the claim, so filtering to the speech tasks would be
assuming the conclusion.
"""
import json, pathlib, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM = "pbs50_cd45k"
SH = pathlib.Path("results/r6_era/board/shards_pbs50_cd45k")
OUT = pathlib.Path("results/showcase/1_beats_the_board")
EVENT = ("onset", "speech", "delta_volume", "word_index", "word_head_pos",
         "word_length", "gpt2_surprisal", "word_gap", "word_part_speech")
FAM_COLOR = {"event": "#1f4e79", "level": "#d98324"}
FLOOR_C = "#9aa5b1"
PRETTY = {"delta_volume": "delta volume", "word_index": "word index", "gpt2_surprisal": "gpt2 surprisal",
          "word_head_pos": "word head pos", "word_length": "word length", "word_gap": "word gap",
          "word_part_speech": "word part speech", "local_flow": "local flow", "global_flow": "global flow",
          "face_num": "face num", "frame_brightness": "frame brightness"}


def load() -> dict[str, dict[str, dict[str, float]]]:
    """task -> tap -> cell -> AUROC, from the SHARDS (never a merged JSON)."""
    out: dict = {}
    for f in sorted(SH.glob("cs_*.json")):
        d = json.loads(f.read_text()); cell = d["name"]
        for key, blk in d["cells"].items():
            arm, task = key.split("|")
            if arm != ARM:
                continue
            for tapnorm, v in blk["cells"].items():
                tap, norm = tapnorm.split("|")
                if norm != "std" or tap not in ("enc0", "enc12"):
                    continue
                out.setdefault(task, {}).setdefault(tap, {})[cell] = v["test"]
    return out


def k_slope(cells: dict[str, dict[str, float]], tap: str) -> float:
    """Ledger estimator: no-intercept slope of (tap-.5) on (enc0-.5) over cells."""
    xs = [cells["enc0"][c] - .5 for c in sorted(cells["enc0"])]
    ys = [cells[tap][c] - .5 for c in sorted(cells["enc0"])]
    return sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)


rows = load()
assert len(rows) == 15, f"[check] VIOLATED need all 15 Lite tasks, got {len(rows)}"
for t, c in rows.items():
    assert set(c) == {"enc0", "enc12"}, f"[check] VIOLATED {t} taps {set(c)}"
    n = {len(c["enc0"]), len(c["enc12"])}
    assert n == {10}, f"[check] VIOLATED {t} cell counts {n}, want 10/10 (partial cells lie)"
    assert set(c["enc0"]) == set(c["enc12"]), f"[check] VIOLATED {t} cell sets differ across taps"

tab = {}
for t, c in rows.items():
    k0, k12 = k_slope(c, "enc0"), k_slope(c, "enc12")
    assert abs(k0 - 1.0) < 1e-9, f"[check] VIOLATED k(enc0) != 1 for {t}"
    tab[t] = {"floor": statistics.fmean(c["enc0"].values()),
              "full": statistics.fmean(c["enc12"].values()),
              "k": k12, "fam": "event" if t in EVENT else "level"}

order = sorted(tab, key=lambda t: -tab[t]["full"])
print(f"[check] OK 15 tasks x 10/10 cells, arm {ARM}, from shards, k(enc0)=1 exactly")
print(f"{'task':20s} {'floor':>7s} {'model':>7s} {'delta':>8s} {'k':>6s}  family")
for t in order:
    r = tab[t]
    print(f"{t:20s} {r['floor']:7.4f} {r['full']:7.4f} {r['full']-r['floor']:+8.4f} {r['k']:6.3f}  {r['fam']}")
ev = sorted(tab[t]["k"] for t in tab if tab[t]["fam"] == "event")
lv = sorted(tab[t]["k"] for t in tab if tab[t]["fam"] == "level")
print(f"[check] event k >= {ev[0]:.3f}  vs  level k <= {lv[-1]:.3f}  "
      f"-> {'CLEAN 9/9 SPLIT' if ev[0] > lv[-1] else 'SPLIT DOES NOT HOLD'}")

fig, ax = plt.subplots(figsize=(9.2, 6.4))
y = range(len(order))
LOSS_C = "#a03030"
for i, t in enumerate(order):
    r = tab[t]
    # A loss must not be drawn as a rightward segment on top of the floor -- it reads as a
    # gain. Bar runs to min(floor, full); the delta segment then points the way it actually
    # went, red when the encoder LOSES to the floor.
    base, tip = min(r["floor"], r["full"]), max(r["floor"], r["full"])
    gain = r["full"] >= r["floor"]
    ax.plot([.5, base], [i, i], lw=7, color=FLOOR_C, solid_capstyle="butt", zorder=1)
    ax.plot([base, tip], [i, i], lw=7, color=FAM_COLOR[r["fam"]] if gain else LOSS_C,
            solid_capstyle="butt", zorder=2, alpha=1.0 if gain else .85)
    ax.plot([r["floor"]], [i], "|", color="white", ms=10, mew=1.7, zorder=4)
    ax.text(tip + .006, i, f"k {r['k']:.2f}", va="center", fontsize=8.5,
            color=FAM_COLOR[r["fam"]] if gain else LOSS_C)
ax.set_yticks(list(y)); ax.set_yticklabels([PRETTY.get(t, t) for t in order], fontsize=9.5)
ax.invert_yaxis()
ax.axvline(.5, color="#444", lw=.9, ls=":")
ax.set_xlim(.5, .90); ax.set_xlabel("cross-subject AUROC  ·  zero labels from the target patient")
ax.set_title("Decoding a brain the model has never seen\n"
             "grey = zero-parameter floor · colour = what pretraining adds", fontsize=12, pad=12)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
h = [plt.Line2D([], [], lw=7, color=FLOOR_C), plt.Line2D([], [], lw=7, color=FAM_COLOR["event"]),
     plt.Line2D([], [], lw=7, color=FAM_COLOR["level"]), plt.Line2D([], [], lw=7, color=LOSS_C),
     plt.Line2D([], [], color="#555", lw=0, marker="|", ms=10, mew=1.7)]
ax.legend(h, ["zero-parameter floor", "gain · event / change-coded (9)",
              "gain · sustained level (6)", "LOSS to the floor", "floor level"],
          loc="lower right", frameon=False, fontsize=8.5)
fig.text(.5, .005, f"all 15 Lite tasks · 10/10 held-out cells · shipped checkpoint {ARM} · "
         f"k = no-intercept slope of (AUROC−.5) on (enc0−.5) over cells, so k(enc0)=1",
         ha="center", fontsize=7.6, color="#666")
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_teaser_cs_capability.{ext}", bbox_inches="tight", dpi=190)
print(f"  -> {OUT}/fig_teaser_cs_capability.pdf|.png")
