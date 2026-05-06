"""Submit Neuroprobe Stage 0 L.4 anchor-window sanity sweep on DCC.

Tag-along to L.2: a single-cell sanity row testing whether shifting the
feature window from [0, 1]s to [-0.375, +0.625]s materially changes test
AUROC at the L.2 winner cell. Motivation: D.0/D.0a use [0, 1]s anchored at
word onset, but speech-evoked responses lead the onset (cf. Duraivel 2023:
~600 ms before-after delay). If [-0.375, +0.625] beats [0, 1] at linear
scope, anchor offset is load-bearing for v14 readout design.

Defaults to the L.2 winner cell (passed via --ref-kind / --view-kind). One
sbatch per (anchor-window × session) — 2 anchor cells × 12 BT Lite sessions
= 24 jobs (or 12 if --baseline-anchor-only is set, just to confirm parity).

Usage on DCC (after L.2 winner identified):
    .venv/bin/python scripts/neuroprobe/submit_l4_anchor_sanity.py \\
        --ref-kind shaft_laplacian --view-kind hg_envelope \\
        --normalization train_set_fixed
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from speech_decoding.studies.braintreebank.labels import NEUROPROBE_TASKS
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS


# Two anchor windows: D.0 baseline and the leading-window sanity row.
ANCHORS: tuple[tuple[str, float, float], ...] = (
    ("L.4.A0_baseline_0_1s", 0.0, 1.0),
    ("L.4.A1_lead_neg375_pos625", 0.375, 0.625),
)


@dataclass(frozen=True)
class AnchorJob:
    cell_id: str
    start_before: float
    end_after: float
    subject_id: int
    trial_id: int
    ref_kind: str
    view_kind: str
    normalization: str

    @property
    def name(self) -> str:
        return (
            f"l4anc_{self.cell_id.lower().replace('.', '_').replace('-', '_neg')}"
            f"_sub{self.subject_id}_trial{self.trial_id}"
        )


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_l4_anchor_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    sessions = _parse_sessions(args.sessions) or list(BT_LITE_SESSIONS)
    anchors = ANCHORS
    if args.baseline_only:
        anchors = (ANCHORS[0],)
    if args.lead_only:
        anchors = (ANCHORS[1],)

    rows: list[dict[str, str]] = []
    for cell_id, start_before, end_after in anchors:
        for sub_id, trial_id in sessions:
            job = AnchorJob(
                cell_id=cell_id,
                start_before=start_before,
                end_after=end_after,
                subject_id=sub_id,
                trial_id=trial_id,
                ref_kind=args.ref_kind,
                view_kind=args.view_kind,
                normalization=args.normalization,
            )
            report_dir = out_root / cell_id / f"sub{sub_id}_trial{trial_id}"
            report_dir.mkdir(parents=True, exist_ok=True)
            sbatch_path = report_dir / "submit.sbatch"
            command = build_command(job, report_dir, args)
            sbatch_path.write_text(build_sbatch(job, report_dir, command, args))
            job_id = ""
            if args.dry_run:
                print(f"[dry-run] {sbatch_path}")
            else:
                result = subprocess.run(
                    ["sbatch", str(sbatch_path)],
                    check=True, text=True, capture_output=True,
                )
                print(result.stdout.strip())
                job_id = result.stdout.strip().split()[-1]
            rows.append({
                "job_id": job_id,
                "cell_id": cell_id,
                "anchor_start_before": str(start_before),
                "anchor_end_after": str(end_after),
                "subject_id": str(sub_id),
                "trial_id": str(trial_id),
                "ref_kind": args.ref_kind,
                "view_kind": args.view_kind,
                "normalization": args.normalization,
                "report_dir": str(report_dir),
                "sbatch_path": str(sbatch_path),
            })

    write_manifest(out_root / "launch_manifest.csv", rows)
    print(f"Wrote {out_root / 'launch_manifest.csv'} ({len(rows)} jobs)")


def build_command(job: AnchorJob, report_dir: Path, args: argparse.Namespace) -> str:
    cmd = [
        ".venv/bin/python",
        "scripts/neuroprobe/run_stage0_linear_baseline.py",
        "--bt-root", str(args.bt_root),
        "--neuroprobe-repo", str(args.neuroprobe_repo),
        "--out-dir", str(report_dir),
        "--subject-id", str(job.subject_id),
        "--trial-id", str(job.trial_id),
        "--task", ",".join(NEUROPROBE_TASKS),
        "--split-type", "CrossSession",
        "--binary-tasks", "false",
        "--backend", "neuralset",
        "--ref-kind", job.ref_kind,
        "--view-kind", job.view_kind,
        "--normalization", job.normalization,
        "--anchor-start-before", str(job.start_before),
        "--anchor-end-after", str(job.end_after),
        "--seed", str(args.seed),
    ]
    return "ROOT_DIR_BRAINTREEBANK=" + shlex.quote(str(args.bt_root)) + " " + shlex.join(cmd)


def build_sbatch(
    job: AnchorJob, report_dir: Path, command: str, args: argparse.Namespace
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job.name[:48]}",
        f"#SBATCH -p {args.partition}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines.extend([
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH -t {args.time}",
        f"#SBATCH -o {report_dir.resolve()}/slurm-%j.out",
        f"#SBATCH -e {report_dir.resolve()}/slurm-%j.err",
        "",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        "export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        "export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        "export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        "export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        f"cd {args.repo_root}",
        command,
        "",
    ])
    return "\n".join(lines)


def _parse_sessions(spec: str) -> list[tuple[int, int]] | None:
    if not spec:
        return None
    out: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        sub_str, trial_str = chunk.strip().split(":")
        out.append((int(sub_str), int(trial_str)))
    return out


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--bt-root", type=Path, default=Path("/work/ht203/data/braintreebank"))
    parser.add_argument(
        "--neuroprobe-repo", type=Path,
        default=Path("/work/ht203/repo/neuroprobe_upstream"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    parser.add_argument(
        "--ref-kind", required=True,
        choices=("raw", "bipolar", "shaft_laplacian", "global_car", "shaft_car"),
        help="L.2 winner reference kind",
    )
    parser.add_argument(
        "--view-kind", required=True,
        choices=("raw_voltage", "stft_abs", "hg_envelope"),
        help="L.2 winner view kind",
    )
    parser.add_argument(
        "--normalization", required=True,
        help="L.1 winner normalization (e.g. train_set_fixed)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sessions", default="",
        help="Comma-separated subject:trial pairs; default = all 12 BT Lite sessions.",
    )
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run the [0,1]s baseline anchor (parity check).")
    parser.add_argument("--lead-only", action="store_true",
                        help="Only run the [-0.375, +0.625]s leading anchor.")
    parser.add_argument("--partition", default="common,scavenger,coganlab-gpu")
    parser.add_argument("--account", default="coganlab")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", default="24G")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
