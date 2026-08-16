"""TEASER: the zero-parameter floor drawn across the whole leaderboard, one row per regime.

Renders TWO versions of the same figure:

  all15  the submission macro -- every Lite task, the number that goes on the board
  snr5   the same figure over the tasks that carry signal, because 10 near-chance tasks
         dilute every model's delta toward zero and the 15-task macro understates all of them

TASK-SELECTION RULE (snr5), fixed before looking at our own numbers:
    keep a task iff the BEST PUBLISHED entry on it reaches 0.70 AUROC within-session.
The rule never reads our floor or our model, so it cannot be tuned in our favour, and the script
asserts three independent robustness checks that the chosen set is not an artifact of the cut:
  (a) the cut is insensitive -- any threshold in the .647-.765 gap gives the identical set
  (b) the same five are the top five by published performance in CROSS-SUBJECT as well
  (c) using the MEDIAN published entry instead of the best gives the identical set
The set keeps `volume`, a sustained-level task we barely beat the floor on, so it also does not
quietly select for our own change-coded mechanism.

Provenance discipline:
  board  -> the VENDORED upstream leaderboard JSONs; `Overall` is exactly the unweighted mean of
            the 15 per-task means, so a subset macro is recomputed the same way for EVERY entry
  ours   -> recomputed from the SHARDS (never a merged JSON), shipped ckpt
  taps   -> ws/csession are scored per-electrode (enc*_elec), cs on parcel means (enc*)

No error bars: the board publishes a `sem` whose definition we cannot verify against ours, and
plotting two differently-defined intervals side by side would be worse than plotting none.
"""
import argparse, json, pathlib, statistics, collections, re
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # NeurIPS rejects Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

_ap = argparse.ArgumentParser()
_ap.add_argument("--arm", default="pbs50_cd45k",
                 help="arm tag carried INSIDE the shard cell keys; guards against arm mixing")
_ap.add_argument("--shards", default="results/r6_era/board/shards_pbs50_cd45k")
_ap.add_argument("--suffix", default="",
                 help="appended to every output filename so a new checkpoint never clobbers the old figure")
_args = _ap.parse_args()

ARM = _args.arm
SH = pathlib.Path(_args.shards)
SUFFIX = _args.suffix
BOARD = pathlib.Path("../neuroprobe/analyses/figures")
OUT = pathlib.Path("results/showcase/1_beats_the_board")
REGIMES = (("ws",       "WithinSession", "enc0_elec", "enc12_elec", "Within session"),
           ("csession", "CrossSession",  "enc0_elec", "enc12_elec", "Cross session  ·  new session, same patient"),
           ("cs",       "CrossSubject",  "enc0",      "enc12",      "Cross subject  ·  NEW PATIENT, zero labels"))

SNR_CUT = 0.70
FLOOR_RED, OURS, BELOW, ABOVE = "#c1440e", "#1f4e79", "#d8dce1", "#8fa3b8"

# One bar per METHOD FAMILY, its best-scoring variant in that regime.  Only GLIS-GNN and the RNN
# are left out, and they sit at the bottom of every board.  This never hides a strong published
# entry -- DIVER-1 tops ws, the CNN tops csession and cs, MLP is third in csession and cs -- and
# the "below the floor" count printed on each panel is still computed over the FULL board, not
# over the drawn subset.  `Linear (raw voltage)` is deliberately NOT a bar: it is a losing variant
# of a family already drawn, and the raw -> STFT -> Lap+STFT ladder is the frontend ablation
# figure, not this one.
FAMILIES = (("DIVER-1", lambda k: "DIVER-1" in k),
            ("PopT", lambda k: "opulation" in k),
            ("CNN", lambda k: k.startswith("CNN_")),
            ("MLP", lambda k: k.startswith("MLP_")),
            ("Linear", lambda k: k.startswith("Linear (")),
            ("BrainBERT", lambda k: k.startswith("BrainBERT")))


def short(name: str) -> str:
    n = name
    n = n.replace("Laplacian_rereferencing_spectrogram", "Lap+STFT")
    n = n.replace("_Lap+STFT", " Lap+STFT")
    n = re.sub(r"\(frozen; off-the-shelf; per-window STFT z-scoring\)", "off-shelf", n)
    n = re.sub(r"\(untrained; frozen; off-the-shelf; per-window STFT z-scoring\)", "UNTRAINED", n)
    n = n.replace("(frozen, global z-scoring)", "").replace("(global z-scoring)", "")
    n = n.replace("(off-the-shelf; per-window STFT z-scoring)", "off-shelf")
    n = n.replace("Population Transformer", "PopT").replace("PopulationTransformer", "PopT")
    n = n.replace("(Laplacian re-referencing + spectrogram)", "Lap+STFT")
    n = n.replace("(spectrogram)", "STFT").replace("(raw voltage)", "raw")
    n = n.replace("GLIS-GNN (ST-GCN, Functional Graph)", "GLIS-GNN")
    n = n.replace("DIVER-1_0.1s_tiny_frozen", "DIVER-1 frozen").replace("DIVER-1_0.1s_tiny", "DIVER-1")
    n = n.replace("RNN (gru)", "RNN").replace("_", " ")
    return re.sub(r"\s+", " ", n).strip()


def board_json(bfile: str) -> dict:
    return json.loads((BOARD / f"neuroprobe_eval_leaderboard_{bfile}.json").read_text())


def board_macro(bd: dict, tasks: tuple[str, ...]) -> dict[str, float]:
    """Subset macro for every published entry, averaged exactly the way `Overall` is."""
    out = {}
    for model in bd["Overall"]:
        per = [bd[t][model]["mean"] for t in tasks if model in bd[t]]
        if len(per) == len(tasks):
            out[model] = statistics.fmean(per)
    return out


def ours(kind: str, tap: str, tasks: tuple[str, ...]) -> float:
    per = collections.defaultdict(dict)
    for f in sorted(SH.glob(f"{kind}_*.json")):
        d = json.loads(f.read_text())
        for key, blk in d["cells"].items():
            arm, task = key.split("|")
            if arm != ARM:
                continue
            v = blk["cells"].get(f"{tap}|std")
            if v:
                per[d["name"]][task] = v["test"]
    n = {len(t) for t in per.values()}
    assert n == {15}, f"[check] VIOLATED {kind}/{tap} task counts {n}, want 15 everywhere"
    return statistics.fmean(statistics.fmean(t[k] for k in tasks) for t in per.values())


def best_per_family(macro: dict[str, float]) -> list[tuple[str, float]]:
    """Best variant of each family. A family absent from a regime (DIVER-1 outside ws) is skipped."""
    out = []
    for _, match in FAMILIES:
        hit = [(k, v) for k, v in macro.items() if match(k)]
        if hit:
            out.append(max(hit, key=lambda kv: kv[1]))
    return out


# ---- select the SNR set, and prove the selection is not an artifact of the cut ----------------
BD = {k: board_json(bf) for k, bf, *_ in REGIMES}
ALL15 = tuple(k for k in BD["ws"] if k != "Overall")
assert len(ALL15) == 15, f"[check] VIOLATED board exposes {len(ALL15)} task keys, want 15"
bestpub = {t: max(v["mean"] for v in BD["ws"][t].values()) for t in ALL15}
medpub = {t: statistics.median(v["mean"] for v in BD["ws"][t].values()) for t in ALL15}
SNR = tuple(t for t in ALL15 if bestpub[t] >= SNR_CUT)

kept, drop = sorted((bestpub[t] for t in SNR)), sorted((bestpub[t] for t in ALL15 if t not in SNR))
print(f"[rule] keep a task iff best published within-session entry >= {SNR_CUT}")
print(f"[rule] -> {len(SNR)} tasks: {', '.join(SNR)}")
print(f"[rule] (a) insensitive: any cut in [{drop[-1]:.3f}, {kept[0]:.3f}] gives this same set "
      f"(gap {kept[0]-drop[-1]:.3f})")
cs_top = {t for t in sorted(ALL15, key=lambda t: -max(v["mean"] for v in BD["cs"][t].values()))[:len(SNR)]}
print(f"[rule] (b) same set is the top-{len(SNR)} by published CROSS-SUBJECT score: {cs_top == set(SNR)}")
med_set = set(sorted(ALL15, key=lambda t: -medpub[t])[:len(SNR)])
print(f"[rule] (c) same set using the MEDIAN published entry instead of the best: {med_set == set(SNR)}")
assert cs_top == set(SNR) and med_set == set(SNR), "[check] VIOLATED SNR set is not robust"


def render(tasks: tuple[str, ...], slug: str, headline: str, subtitle: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 11.0))
    print(f"\n=== {slug}  ({len(tasks)} tasks)")
    for ax, (kind, _, t0, t12, title) in zip(axes, REGIMES):
        macro = board_macro(BD[kind], tasks)
        floor, full = ours(kind, t0, tasks), ours(kind, t12, tasks)
        entries = sorted(((short(k), v) for k, v in best_per_family(macro)), key=lambda kv: -kv[1])
        # counted over EVERY published entry, not the drawn subset
        below, n_board = sum(1 for v in macro.values() if v < floor), len(macro)
        print(f"{kind:9s} floor {floor:.4f}  full {full:.4f}  gap {full-floor:+.4f}  "
              f"| {below}/{n_board} published entries BELOW the floor "
              f"| drawing: " + ", ".join(n for n, _ in entries))

        labels = [n for n, _ in entries] + ["OUR FLOOR\n(0 params)", "OUR MODEL"]
        vals = [v for _, v in entries] + [floor, full]
        colors = [(BELOW if v < floor else ABOVE) for _, v in entries] + [FLOOR_RED, OURS]
        order = sorted(range(len(vals)), key=lambda i: vals[i])   # ascending: the eye climbs to ours
        x = range(len(order))
        ax.bar(x, [vals[i] - .5 for i in order], .74, bottom=.5,
               color=[colors[i] for i in order], zorder=3,
               edgecolor=["#333" if i >= len(entries) else "none" for i in order], linewidth=1.1)
        ax.axhline(floor, color=FLOOR_RED, lw=2.0, ls="--", zorder=4)
        ax.set_xticks(list(x))
        ax.set_xticklabels([labels[i] for i in order], rotation=0, ha="center", fontsize=8.5)
        for tick, i in zip(ax.get_xticklabels(), order):
            if i >= len(entries):
                tick.set_fontweight("bold")
                tick.set_color(FLOOR_RED if i == len(entries) else OURS)

        hi = max(vals)
        ax.set_ylim(.5, hi + (hi - .5) * .30)
        ax.set_ylabel("AUROC", fontsize=10)
        ax.set_title(title, fontsize=13, loc="left", pad=8, fontweight="bold")
        ax.text(.012, .95, f"{below} of {n_board} published entries\nfall below a "
                f"ZERO-PARAMETER baseline", transform=ax.transAxes, ha="left", va="top",
                fontsize=10.5, color=FLOOR_RED, fontweight="bold")
        # the only gap in the panel that pretraining bought.  Ours is the tallest bar, so the arrow
        # goes in the empty margin to its RIGHT -- inside the panel it would land on another bar.
        xm = order.index(len(entries) + 1)
        ax.set_xlim(-.75, len(order) - 1 + 1.5)
        ax.annotate("", xy=(xm + .52, full), xytext=(xm + .52, floor), zorder=6,
                    arrowprops=dict(arrowstyle="<->", color=OURS, lw=2.0, shrinkA=0, shrinkB=0))
        ax.text(xm + .64, (floor + full) / 2, f"+{full-floor:.4f}",
                ha="left", va="center", fontsize=11, fontweight="bold", color=OURS, zorder=6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#eee", lw=.8, zorder=0)

    fig.suptitle(headline, fontsize=14.5, y=.995)
    fig.text(.5, .002, subtitle + "\nbest variant of each published method family shown (GLIS-GNN and RNN "
             f"omitted, bottom of every board); counts are over ALL entries · ours from shards, {ARM} · "
             "floor = 3-band |STFT| + deltas + regularised ridge, no encoder",
             ha="center", fontsize=7.8, color="#666")
    fig.tight_layout(rect=(0.0, .012, 1.0, .985))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_teaser_floor_splits_board{slug}{SUFFIX}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  -> {OUT}/fig_teaser_floor_splits_board{slug}{SUFFIX}.pdf|.png")


def render_delta(tasks: tuple[str, ...], slug: str, headline: str, subtitle: str) -> None:
    """Same numbers, re-anchored on the floor.

    Absolute AUROC compresses the claim: at ~.82 on an axis rooted at chance, a +.025 gap is a
    sliver.  The claim is not "our AUROC is .85", it is "everything published sits below a
    zero-parameter baseline and we sit above it" -- so plot distance FROM that baseline and the
    sentence becomes the geometry.  Zero here is a real, stated reference, not a cropped axis.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 10.2))
    print(f"\n=== {slug}  ({len(tasks)} tasks)")
    for ax, (kind, _, t0, t12, title) in zip(axes, REGIMES):
        macro = board_macro(BD[kind], tasks)
        floor, full = ours(kind, t0, tasks), ours(kind, t12, tasks)
        entries = sorted(((short(k), v) for k, v in best_per_family(macro)), key=lambda kv: -kv[1])
        below, n_board = sum(1 for v in macro.values() if v < floor), len(macro)
        print(f"{kind:9s} floor {floor:.4f}  full {full:.4f}  gap {full-floor:+.4f}  "
              f"| {below}/{n_board} below")

        rows = sorted([(n, v) for n, v in entries] + [("OUR MODEL", full)], key=lambda kv: kv[1])
        y = range(len(rows))
        for i, (n, v) in enumerate(rows):
            d, mine = v - floor, n == "OUR MODEL"
            c = OURS if mine else (ABOVE if d > 0 else BELOW)
            ax.barh(i, d, .68, color=c, zorder=3,
                    edgecolor="#123" if mine else "none", linewidth=1.2)
            ax.text(d + (.0016 if d > 0 else -.0016), i, f"{d:+.4f}",
                    ha="left" if d > 0 else "right", va="center", zorder=4,
                    fontsize=11 if mine else 8.6, fontweight="bold" if mine else "normal",
                    color=OURS if mine else "#6b7480")

        # asymmetric limits: a symmetric axis leaves half the panel empty whenever the worst
        # entry is further from the floor than we are, which is most of the time
        lo, hi = min(r[1] - floor for r in rows), max(r[1] - floor for r in rows)
        L, R = lo * 1.62, hi * 1.75
        ax.set_xlim(L, R)
        ax.axvspan(0, R, color=OURS, alpha=.035, zorder=0)
        ax.axvline(0, color=FLOOR_RED, lw=2.4, zorder=5)
        ax.set_yticks(list(y))
        ax.set_yticklabels([n for n, _ in rows], fontsize=9.6)
        for tick, (n, _) in zip(ax.get_yticklabels(), rows):
            if n == "OUR MODEL":
                tick.set_fontweight("bold"); tick.set_color(OURS); tick.set_fontsize(10.6)
        ax.set_ylim(-.7, len(rows) - .28)
        ax.set_title(f"{title}", fontsize=12.5, loc="left", pad=6, fontweight="bold")
        ax.text(0, len(rows) - .48, f"  zero-parameter floor  ({floor:.3f} AUROC)",
                ha="left", va="center", fontsize=9.2, color=FLOOR_RED, fontweight="bold")
        ax.text(L, len(rows) - .48, f"{below} of {n_board} published entries land here  ",
                ha="left", va="center", fontsize=9.2, color="#6b7480")
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="both", length=0)
        ax.set_xticks([])
        ax.grid(axis="x", color="#f0f0f0", lw=.8, zorder=0)

    axes[-1].set_xlabel("AUROC relative to the zero-parameter floor", fontsize=10, labelpad=8)
    fig.suptitle(headline, fontsize=14.5, y=.997)
    fig.text(.5, .002, subtitle + "\nbest variant of each published method family shown (GLIS-GNN and RNN "
             f"omitted, bottom of every board); counts are over ALL entries · ours from shards, {ARM} · "
             "floor = 3-band |STFT| + deltas + regularised ridge, no encoder",
             ha="center", fontsize=7.8, color="#666")
    fig.tight_layout(rect=(0.0, .014, 1.0, .982))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_teaser_floor_splits_board{slug}{SUFFIX}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  -> {OUT}/fig_teaser_floor_splits_board{slug}{SUFFIX}.pdf|.png")


ALL_SUB = "Neuroprobe Lite, all 15 tasks — the submission macro ·"
SNR_SUB = (f"Neuroprobe Lite, the {len(SNR)} tasks with signal ({', '.join(SNR)}) — kept iff the "
           f"best published within-session entry reaches {SNR_CUT:.2f}, a rule that never reads "
           "our own scores ·")
HEAD = "Most published intracranial foundation models do not beat a spectrogram"
# 🚫 THE SNR-5 VARIANTS ARE NOT RENDERED (Ben 2026-08-10: "we only report all15 — we shall
# never confuse a reader"). The selection rule above still RUNS, because its three robustness
# checks are cheap and a silently-broken rule is worse than an unused one, but a second macro
# over a different task set is a second headline number, and two headline numbers is how a
# reader ends up quoting the flattering one. The published macro is over 15 tasks. Only 15.
render(ALL15, "", HEAD, ALL_SUB)
render_delta(ALL15, "_delta", HEAD, ALL_SUB)
