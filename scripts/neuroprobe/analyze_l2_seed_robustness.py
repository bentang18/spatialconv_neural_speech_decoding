"""L.2 seed-robustness analyzer.

Compares L.2 winner (R4xI2 = shaft_laplacian × stft_abs) across seeds 42, 43,
44 to characterize seed-noise on the linear baseline pipeline. Reads
`metrics.json` from each seed sweep root and emits `seed_robustness.md` +
`seed_robustness.json` to a chosen output dir.

Empirical note (2026-05-10): linear baseline + sklearn lbfgs is seed-invariant
at the current row count (n_rows=15=n_tasks per session, no `rng.choice` cap
firing). Single-seed reporting is therefore sufficient for L.* freeze numbers.
This script confirms that property and surfaces any session where it breaks.

Usage:
    .venv/bin/python scripts/neuroprobe/analyze_l2_seed_robustness.py \\
        --seed-root seed42=reports/neuroprobe_stage0_l2_neuralset_parity_2026_05_06 \\
        --seed-root seed43=reports/neuroprobe_stage0_l2_seed43_2026_05_09 \\
        --seed-root seed44=reports/neuroprobe_stage0_l2_seed44_2026_05_09 \\
        --out-dir reports/neuroprobe_stage0_l2_seed_robustness_2026_05_10
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


SESSION_RE = re.compile(r"sub(\d+)_trial(\d+)")


def session_id(p: Path) -> str:
    m = SESSION_RE.search(str(p))
    return f"({m.group(1)},{m.group(2)})" if m else str(p)


def load_seed(seed_root: Path) -> dict[str, float]:
    out = {}
    for metrics_path in sorted(seed_root.glob("L.2.R4xI2/sub*/metrics.json")):
        with open(metrics_path) as f:
            m = json.load(f)
        out[session_id(metrics_path.parent)] = m["mean_test_roc_auc"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-root", action="append", required=True,
                   help="name=path entries for each seed sweep root")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    seeds: dict[str, dict[str, float]] = {}
    for entry in args.seed_root:
        name, _, path = entry.partition("=")
        seeds[name] = load_seed(Path(path))

    sessions = sorted({s for d in seeds.values() for s in d})

    rows = []
    per_session_sds = []
    for s in sessions:
        vals = {name: d.get(s) for name, d in seeds.items()}
        present = [v for v in vals.values() if v is not None]
        sd = statistics.stdev(present) if len(present) > 1 else None
        if sd is not None:
            per_session_sds.append(sd)
        rows.append({"session": s, "by_seed": vals, "sd_across_seeds": sd})

    seed_means = {name: statistics.mean(d.values()) if d else None for name, d in seeds.items()}

    summary = {
        "seed_means": seed_means,
        "max_per_session_sd": max(per_session_sds) if per_session_sds else None,
        "median_per_session_sd": statistics.median(per_session_sds) if per_session_sds else None,
    }

    if summary["max_per_session_sd"] is not None and summary["max_per_session_sd"] < 1e-6:
        verdict = "seed-invariant: linear baseline output is byte-identical across seeds at current row count. Single-seed reporting is sufficient."
    elif summary["max_per_session_sd"] is not None and summary["max_per_session_sd"] < 0.005:
        verdict = "seed-stable: per-session sd < 0.005 noise band. Single-seed adequate; multi-seed confirms."
    else:
        verdict = f"seed-jittery: max per-session sd = {summary['max_per_session_sd']:.4f}. Report multi-seed mean ± sd."

    md = []
    md.append("# L.2 seed-robustness — R4xI2 across seeds")
    md.append("")
    md.append("Cell: L.2 winner (R4xI2 = shaft_laplacian × stft_abs) + N1 (train_set_fixed).")
    md.append("")
    md.append("## Aggregate")
    md.append("")
    md.append("| seed | n | mean |")
    md.append("|---|---|---|")
    for name, d in seeds.items():
        if d:
            md.append(f"| {name} | {len(d)} | {statistics.mean(d.values()):.4f} |")
        else:
            md.append(f"| {name} | 0 | — |")
    md.append("")
    md.append("## Per-session")
    md.append("")
    seed_names = list(seeds.keys())
    md.append("| session | " + " | ".join(seed_names) + " | sd across seeds |")
    md.append("|---|" + "|".join("---" for _ in seed_names) + "|---|")
    for r in rows:
        cells = [f"{r['by_seed'][n]:.4f}" if r["by_seed"][n] is not None else "—" for n in seed_names]
        sd_cell = f"{r['sd_across_seeds']:.4f}" if r["sd_across_seeds"] is not None else "—"
        md.append(f"| {r['session']} | " + " | ".join(cells) + f" | {sd_cell} |")
    md.append("")
    md.append("## Verdict")
    md.append("")
    md.append(verdict)
    md.append("")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_md = args.out_dir / "seed_robustness.md"
    out_json = args.out_dir / "seed_robustness.json"
    out_md.write_text("\n".join(md))
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "rows": rows, "verdict": verdict}, f, indent=2)
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    print()
    print(verdict)


if __name__ == "__main__":
    main()
