"""Collect/aggregate Whisper ceiling-probe results into a markdown summary.

Reads `ceiling_results.csv` (output of run_probe_ceiling.py), aggregates per
(task, split, layer) across (subject, trial) pairs, and writes a markdown table
to --out-md.

Headline numbers reported:
    - Per-task best layer (within / cross-session / cross-subject)
    - Concat-all-layers vs best single layer (does merging help?)
    - Late-fusion vs concat-all (per-layer ensemble vs single LogReg)
    - Comparison to Neuroprobe SOTA gates: CSubject 0.578 AUROC, CSession 0.667
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics


NEUROPROBE_GATE_CSESSION = 0.667  # multi-class -- approx
NEUROPROBE_GATE_CSUBJECT_BIN = 0.578


def aggregate(rows: list[dict]) -> dict:
    """Group by (task, split, layer) and compute mean/stdev/n."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        try:
            auroc = float(row["auroc"])
        except (ValueError, TypeError):
            continue
        if auroc != auroc:  # NaN
            continue
        key = (row["task"], row["split"], row["layer"])
        groups[key].append(auroc)
    out = {}
    for key, vals in groups.items():
        if not vals:
            continue
        out[key] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out


def best_layer_per_task_split(agg: dict) -> dict:
    """For each (task, split), find the layer with highest mean AUROC."""
    best: dict[tuple, dict] = {}
    for (task, split, layer), stats in agg.items():
        key = (task, split)
        if key not in best or stats["mean"] > best[key]["mean"]:
            best[key] = {"layer": layer, **stats}
    return best


def markdown_report(agg: dict, csv_path: Path) -> str:
    best = best_layer_per_task_split(agg)
    tasks = sorted({t for (t, _, _) in agg})
    splits = ["within_trial", "cross_session", "cross_movie"]

    lines = []
    lines.append(f"# Whisper-large-v3 teacher ceiling — {csv_path.name}")
    lines.append("")
    lines.append(f"Source CSV: `{csv_path}`")
    lines.append("")

    # Headline table: best layer per (task, split)
    lines.append("## Best layer per task × split")
    lines.append("")
    lines.append("| task | within_trial | cross_session | cross_movie |")
    lines.append("|---|---|---|---|")
    for task in tasks:
        row = [task]
        for split in splits:
            r = best.get((task, split))
            if r is None:
                row.append("—")
            else:
                row.append(f"{r['mean']:.3f} @ {r['layer']} (n={r['n']})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Layer-merge head-to-head
    lines.append("## Layer-merge sister vs best single layer (within-trial)")
    lines.append("")
    lines.append("| task | best_single | concat_all | mean_all | concat_top3 | late_fusion |")
    lines.append("|---|---|---|---|---|---|")
    for task in tasks:
        single = best.get((task, "within_trial"))
        single_str = f"{single['mean']:.3f} @ {single['layer']}" if single else "—"
        def pick(layer_label: str) -> str:
            keys = [k for k in agg if k[0] == task and k[1] == "within_trial"
                    and (k[2] == layer_label or k[2].startswith(layer_label))]
            if not keys:
                return "—"
            best_match = max(keys, key=lambda k: agg[k]["mean"])
            return f"{agg[best_match]['mean']:.3f}"
        lines.append(
            f"| {task} | {single_str} | {pick('concat_all')} | "
            f"{pick('mean_all')} | {pick('concat_top3')} | {pick('late_fusion')} |"
        )
    lines.append("")

    # Teacher generalization across movies — Whisper features come from WAV,
    # so "cross-subject" is mechanically cross-movie. The two columns below
    # are both cross-movie at the feature level: cross-session = per-subject
    # A->B (1 train movie, 1 test movie); cross-movie LOSO = pool all-but-one
    # of the 9 unique movies. Brain-side submission gates apply to the v14
    # student, not this teacher-ceiling table.
    lines.append("## Teacher generalization across movies")
    lines.append("")
    lines.append("| task | best cross-movie LOSO AUROC | best cross-session AUROC |")
    lines.append("|---|---|---|")
    for task in tasks:
        cs = best.get((task, "cross_movie"))
        css = best.get((task, "cross_session"))
        cs_str = f"{cs['mean']:.3f} @ {cs['layer']}" if cs else "—"
        css_str = f"{css['mean']:.3f} @ {css['layer']}" if css else "—"
        lines.append(f"| {task} | {cs_str} | {css_str} |")
    lines.append("")

    # Sanity stats
    lines.append("## Per-task n")
    lines.append("")
    lines.append("| task | within_trial n | cross_session n | cross_movie LOSO n |")
    lines.append("|---|---|---|---|")
    for task in tasks:
        ns = []
        for split in splits:
            ns_for_split = [agg[k]["n"] for k in agg if k[0] == task and k[1] == split]
            ns.append(max(ns_for_split) if ns_for_split else 0)
        lines.append(f"| {task} | {ns[0]} | {ns[1]} | {ns[2]} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    with open(args.probe_csv) as fh:
        rows = list(csv.DictReader(fh))
    print(f"[collect] {len(rows)} rows loaded")
    agg = aggregate(rows)
    print(f"[collect] {len(agg)} groups")
    md = markdown_report(agg, args.probe_csv)
    args.out_md.write_text(md)
    print(f"[collect] wrote {args.out_md}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-csv", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    return p.parse_args()


if __name__ == "__main__":
    main()
