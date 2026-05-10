"""L.4 anchor-robustness analyzer.

Compares L.2 winner (R4xI2 = shaft_laplacian × stft_abs) at the leaderboard
anchor [0, 1] s vs an in-band lead anchor [-0.375, +0.625] s. Reads
`metrics.json` from each cell × session and emits `anchor_robustness.md` +
`anchor_robustness.json` to the sweep root.

In-band region per `docs/neuroprobe/stage_1.md` Temporal Scale section:
starts between about -0.375 s and +0.125 s are robustness territory; starts
outside that band are leakage controls. A1 inside this band; A0 is the
official Neuroprobe anchor.

Decision rule:
- |A1 - A0| within seed-noise (~0.005) → anchor-robust.
- A1 dominates A0 by > 0.005 → leakage hint (pre-stim signal beating onset).
- A0 dominates A1 by > 0.005 → onset-locked, no jitter slack.

Usage:
    .venv/bin/python scripts/neuroprobe/analyze_l4_anchor_robustness.py \\
        --sweep-root reports/neuroprobe_stage0_l4_anchor_2026_05_09/
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


def load_cell(cell_dir: Path) -> dict[str, float]:
    out = {}
    for metrics_path in sorted(cell_dir.glob("sub*/metrics.json")):
        with open(metrics_path) as f:
            m = json.load(f)
        out[session_id(metrics_path.parent)] = m["mean_test_roc_auc"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-root", type=Path, required=True)
    args = p.parse_args()

    a0_dir = args.sweep_root / "L.4.A0_baseline_0_1s"
    a1_dir = args.sweep_root / "L.4.A1_lead_neg375_pos625"
    assert a0_dir.is_dir() and a1_dir.is_dir(), "missing L.4 cell dirs"

    a0 = load_cell(a0_dir)
    a1 = load_cell(a1_dir)
    sessions = sorted(set(a0) | set(a1))

    rows = []
    deltas = []
    for s in sessions:
        v0 = a0.get(s)
        v1 = a1.get(s)
        d = (v1 - v0) if (v0 is not None and v1 is not None) else None
        if d is not None:
            deltas.append(d)
        rows.append({"session": s, "a0": v0, "a1": v1, "delta": d})

    a0_vals = [v for v in a0.values()]
    a1_vals = [v for v in a1.values()]
    summary = {
        "a0_mean": statistics.mean(a0_vals),
        "a0_sd": statistics.stdev(a0_vals),
        "a0_n": len(a0_vals),
        "a1_mean": statistics.mean(a1_vals),
        "a1_sd": statistics.stdev(a1_vals),
        "a1_n": len(a1_vals),
        "delta_mean": statistics.mean(deltas),
        "delta_sd": statistics.stdev(deltas),
        "n_a1_better": sum(1 for d in deltas if d > 0),
        "n_a0_better": sum(1 for d in deltas if d < 0),
        "n_tie_band_0_005": sum(1 for d in deltas if abs(d) <= 0.005),
    }

    if abs(summary["delta_mean"]) <= 0.005:
        verdict = "anchor-robust (Δ within ±0.005 noise band)"
    elif summary["delta_mean"] > 0.005:
        verdict = "leakage hint: in-band lead beats onset by >0.005 — investigate pre-stim signal"
    else:
        verdict = "onset-locked: leaderboard anchor dominates in-band lead by >0.005"

    md = []
    md.append("# L.4 anchor-robustness — A0 vs A1 (in-band lead)")
    md.append("")
    md.append("Cell: L.2 winner (R4xI2 = shaft_laplacian × stft_abs) + N1 (train_set_fixed).")
    md.append("")
    md.append(f"- A0: anchor [0, 1] s — Neuroprobe leaderboard default")
    md.append(f"- A1: anchor [-0.375, +0.625] s — in-band lead per docs/neuroprobe/stage_1.md")
    md.append("")
    md.append(f"## Aggregate")
    md.append("")
    md.append(f"| cell | n | mean | sd |")
    md.append(f"|---|---|---|---|")
    md.append(f"| A0 | {summary['a0_n']} | {summary['a0_mean']:.4f} | {summary['a0_sd']:.4f} |")
    md.append(f"| A1 | {summary['a1_n']} | {summary['a1_mean']:.4f} | {summary['a1_sd']:.4f} |")
    md.append(f"| Δ (A1 − A0) | {len(deltas)} | {summary['delta_mean']:+.4f} | {summary['delta_sd']:.4f} |")
    md.append("")
    md.append(f"A1 better in {summary['n_a1_better']}/{len(deltas)} sessions; A0 better in {summary['n_a0_better']}/{len(deltas)}; |Δ| ≤ 0.005 in {summary['n_tie_band_0_005']}/{len(deltas)}.")
    md.append("")
    md.append("## Per-session")
    md.append("")
    md.append("| session | A0 | A1 | A1 − A0 |")
    md.append("|---|---|---|---|")
    for r in rows:
        a0_s = f"{r['a0']:.4f}" if r["a0"] is not None else "—"
        a1_s = f"{r['a1']:.4f}" if r["a1"] is not None else "—"
        d_s = f"{r['delta']:+.4f}" if r["delta"] is not None else "—"
        md.append(f"| {r['session']} | {a0_s} | {a1_s} | {d_s} |")
    md.append("")
    md.append(f"## Verdict")
    md.append("")
    md.append(verdict)
    md.append("")

    out_md = args.sweep_root / "anchor_robustness.md"
    out_json = args.sweep_root / "anchor_robustness.json"
    out_md.write_text("\n".join(md))
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "rows": rows, "verdict": verdict}, f, indent=2)
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    print()
    print(verdict)


if __name__ == "__main__":
    main()
