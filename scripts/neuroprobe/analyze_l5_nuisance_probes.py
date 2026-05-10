"""L.5.P2 within-subject session-id nuisance-probe analyzer.

Reads `metrics.json` from a single L.5 run dir (output of
`run_l5_nuisance_probes.py`) and renders a kill-criterion verdict.

Per `docs/neuroprobe/stage_0.md` L.5 spec — kill the L.2 winner view if any
subject's held-out P2 macro AUROC > 0.95. P2 is run within-subject
(session-id-from-features); P1 (subject-id) was dropped because it is
trivially decodable from per-subject feature width.

Usage:
    .venv/bin/python scripts/neuroprobe/analyze_l5_nuisance_probes.py \\
        --run-dir reports/neuroprobe_stage0_l5_nuisance_probes_2026_05_10/L.5.P2/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


KILL_AUROC = 0.95


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="L.5 run directory containing metrics.json.")
    args = p.parse_args()

    metrics_path = args.run_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"No metrics.json at {metrics_path}.")
    with open(metrics_path) as f:
        m = json.load(f)

    csv_path = args.run_dir / "nuisance_probe_metrics.csv"
    csv_text = csv_path.read_text() if csv_path.exists() else ""

    per_subject = m["p2_per_subject"]
    macro_auroc = float(m["p2_macro_auroc"])
    macro_balacc = float(m["p2_macro_balacc"])
    max_auroc = float(m["p2_max_auroc"])
    min_auroc = float(m["p2_min_auroc"])
    kill_any = bool(m["p2_kill_any"])
    kill_subjects = list(m["p2_kill_subjects"])
    n_subjects = int(m["n_subjects_probed"])
    threshold = float(m.get("kill_auroc_threshold", KILL_AUROC))

    if kill_any:
        verdict = (
            f"KILL — {len(kill_subjects)} subject(s) exceed AUROC > {threshold:.2f} "
            f"({kill_subjects}). View leaks within-subject session identity. "
            "Do not freeze for v14."
        )
    elif max_auroc < 0.65:
        verdict = (
            f"clean — max per-subject AUROC = {max_auroc:.4f} (well below "
            f"{threshold:.2f}). View carries minimal session-drift signal."
        )
    else:
        verdict = (
            f"caution — max per-subject AUROC = {max_auroc:.4f}, below kill "
            f"threshold ({threshold:.2f}) but above 0.65. Some session drift "
            "remains; OK to proceed but flag for v14 ablation."
        )

    md: list[str] = []
    md.append("# L.5.P2 — within-subject session-id nuisance-probe analysis")
    md.append("")
    md.append(f"Run dir: `{args.run_dir}`.")
    md.append("")
    md.append(
        f"P2 = within-subject session-id from L.2 winner features. P1 "
        "(subject-id) was dropped because it is trivially decodable from "
        "per-subject feature width and therefore not informative about the "
        "view's content."
    )
    md.append("")
    md.append(f"Kill criterion: any subject's held-out macro AUROC > {threshold:.2f}.")
    md.append("")

    md.append("## Headline")
    md.append("")
    md.append(f"- **macro AUROC across {n_subjects} subjects: {macro_auroc:.4f}**")
    md.append(f"- macro balanced accuracy: {macro_balacc:.4f}")
    md.append(f"- max per-subject AUROC: {max_auroc:.4f}")
    md.append(f"- min per-subject AUROC: {min_auroc:.4f}")
    md.append(f"- kill subjects: {kill_subjects or 'none'}")
    md.append("")

    md.append("## Per-subject")
    md.append("")
    md.append("| subject | n_classes | n_train | n_test | balacc | chance | AUROC | kill |")
    md.append("|---|---|---|---|---|---|---|---|")
    for s in per_subject:
        md.append(
            f"| {s['subject_id']} | {s['n_classes']} | {s['n_train']} | "
            f"{s['n_test']} | {s['balacc']:.4f} | {s['chance']:.4f} | "
            f"{s['auroc']:.4f} | {'YES' if s['kill'] else 'no'} |"
        )
    md.append("")

    md.append("## Verdict")
    md.append("")
    md.append(verdict)
    md.append("")
    md.append("## Interpretation guide")
    md.append("")
    md.append(
        f"- Per-subject AUROC > {threshold:.2f}: feature distribution drifts "
        "across that subject's sessions enough that a linear classifier can "
        "exploit it. v14 inherits the leak unless its preprocessing collapses "
        "the drift.\n"
        "- Per-subject AUROC near 0.5: features are session-invariant within "
        "subject; cross-session linear wins must come from stimulus-locked "
        "structure.\n"
        f"- 0.65 < AUROC ≤ {threshold:.2f}: residual drift present. "
        "Acceptable for Stage-0 freeze but document for v14 ablation."
    )

    if csv_text:
        md.append("")
        md.append("## Per-subject probe diagnostics CSV")
        md.append("")
        md.append("```")
        md.append(csv_text.rstrip("\n"))
        md.append("```")

    out_md = args.run_dir / "nuisance_analysis.md"
    out_json = args.run_dir / "nuisance_analysis.json"
    out_md.write_text("\n".join(md) + "\n")
    out_json.write_text(json.dumps({
        "macro_auroc": macro_auroc,
        "macro_balacc": macro_balacc,
        "max_auroc": max_auroc,
        "min_auroc": min_auroc,
        "kill_any": kill_any,
        "kill_subjects": kill_subjects,
        "n_subjects_probed": n_subjects,
        "kill_threshold": threshold,
        "verdict": verdict,
        "per_subject": per_subject,
    }, indent=2) + "\n")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    print()
    print(verdict)


if __name__ == "__main__":
    main()
