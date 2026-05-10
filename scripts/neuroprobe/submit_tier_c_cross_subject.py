"""Submit Neuroprobe Stage 0 Tier-C CrossSubject parity sweep on DCC.

Tag-along to L.1 + L.2: tests whether the L.1 + L.2 winners survive the
CrossSubject distribution shift (where train and eval subjects differ).
This is the key generalization test for the v14 atlas-anchored thesis: if
neither winner beats the upstream baseline at CrossSubject, the L-sweep
freeze is overfit to within-subject distribution.

Default cells (3 per session):
- C.0_baseline: upstream N1 × R4xI2 (sanity / reference)
- C.1_l1_winner: L.1 winner × R4xI2 (does normalization choice transfer?)
- C.2_l1_l2_winner: L.1 winner × L.2 winner ref+view (joint generalization)

Subject 2 is the upstream Neuroprobe DS_DM_TRAIN_SUBJECT_ID; the wrapper
rejects subject_id=2 for CrossSubject. This submitter skips (2,*) sessions.

Usage on DCC (after L.1 + L.2 winners identified):
    .venv/bin/python scripts/neuroprobe/submit_tier_c_cross_subject.py \\
        --l1-winner-norm train_set_robust_mad \\
        --l2-winner-ref shaft_laplacian \\
        --l2-winner-view hg_envelope
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


# Subject 2 is the upstream DS_DM_TRAIN_SUBJECT_ID — wrapper rejects it.
TRAIN_SUBJECT_ID = 2


@dataclass(frozen=True)
class CrossSubjectCell:
    cell_id: str
    ref_kind: str
    view_kind: str
    normalization: str
    label: str


def build_cells(
    l1_winner_norm: str,
    l2_winner_ref: str,
    l2_winner_view: str,
    baseline_norm: str = "train_set_fixed",
    baseline_ref: str = "shaft_laplacian",
    baseline_view: str = "stft_abs",
    skip_baseline: bool = False,
    only_baseline: bool = False,
) -> list[CrossSubjectCell]:
    cells: list[CrossSubjectCell] = []
    if not skip_baseline:
        cells.append(CrossSubjectCell(
            cell_id="C.0_baseline",
            ref_kind=baseline_ref, view_kind=baseline_view,
            normalization=baseline_norm,
            label=f"upstream baseline ({baseline_norm}, {baseline_ref}, {baseline_view})",
        ))
    if only_baseline:
        return cells
    cells.append(CrossSubjectCell(
        cell_id="C.1_l1_winner",
        ref_kind=baseline_ref, view_kind=baseline_view,
        normalization=l1_winner_norm,
        label=f"L.1 winner only ({l1_winner_norm}, baseline ref/view)",
    ))
    cells.append(CrossSubjectCell(
        cell_id="C.2_l1_l2_winner",
        ref_kind=l2_winner_ref, view_kind=l2_winner_view,
        normalization=l1_winner_norm,
        label=f"L.1+L.2 winner ({l1_winner_norm}, {l2_winner_ref}, {l2_winner_view})",
    ))
    return cells


@dataclass(frozen=True)
class CrossSubjectJob:
    cell: CrossSubjectCell
    subject_id: int
    trial_id: int

    @property
    def name(self) -> str:
        return (
            f"tierc_{self.cell.cell_id.lower().replace('.', '_')}_"
            f"sub{self.subject_id}_trial{self.trial_id}"
        )


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_tier_c_cross_subject_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    sessions = _parse_sessions(args.sessions) or [
        (s, t) for (s, t) in BT_LITE_SESSIONS if s != TRAIN_SUBJECT_ID
    ]
    cells = build_cells(
        l1_winner_norm=args.l1_winner_norm,
        l2_winner_ref=args.l2_winner_ref,
        l2_winner_view=args.l2_winner_view,
        baseline_norm=args.baseline_norm,
        baseline_ref=args.baseline_ref,
        baseline_view=args.baseline_view,
        skip_baseline=args.skip_baseline,
        only_baseline=args.only_baseline,
    )

    rows: list[dict[str, str]] = []
    for cell in cells:
        for sub_id, trial_id in sessions:
            if sub_id == TRAIN_SUBJECT_ID:
                continue
            job = CrossSubjectJob(cell=cell, subject_id=sub_id, trial_id=trial_id)
            report_dir = out_root / cell.cell_id / f"sub{sub_id}_trial{trial_id}"
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
                "cell_id": cell.cell_id,
                "label": cell.label,
                "ref_kind": cell.ref_kind,
                "view_kind": cell.view_kind,
                "normalization": cell.normalization,
                "split_type": "CrossSubject",
                "subject_id": str(sub_id),
                "trial_id": str(trial_id),
                "report_dir": str(report_dir),
                "sbatch_path": str(sbatch_path),
            })

    write_manifest(out_root / "launch_manifest.csv", rows)
    print(f"Wrote {out_root / 'launch_manifest.csv'} ({len(rows)} jobs)")


def build_command(
    job: CrossSubjectJob, report_dir: Path, args: argparse.Namespace
) -> str:
    cmd = [
        ".venv/bin/python",
        "scripts/neuroprobe/run_stage0_linear_baseline.py",
        "--bt-root", str(args.bt_root),
        "--neuroprobe-repo", str(args.neuroprobe_repo),
        "--out-dir", str(report_dir),
        "--subject-id", str(job.subject_id),
        "--trial-id", str(job.trial_id),
        "--task", ",".join(NEUROPROBE_TASKS),
        "--split-type", "CrossSubject",
        "--binary-tasks", "false",
        "--backend", "neuralset",
        "--ref-kind", job.cell.ref_kind,
        "--view-kind", job.cell.view_kind,
        "--normalization", job.cell.normalization,
        "--seed", str(args.seed),
    ]
    return "ROOT_DIR_BRAINTREEBANK=" + shlex.quote(str(args.bt_root)) + " " + shlex.join(cmd)


def build_sbatch(
    job: CrossSubjectJob, report_dir: Path, command: str, args: argparse.Namespace
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
        "--l1-winner-norm", required=True,
        help="L.1 winner normalization recipe.",
    )
    parser.add_argument(
        "--l2-winner-ref", required=True,
        choices=("raw", "bipolar", "shaft_laplacian", "global_car", "shaft_car"),
    )
    parser.add_argument(
        "--l2-winner-view", required=True,
        choices=("raw_voltage", "stft_abs", "hg_envelope"),
    )
    parser.add_argument("--baseline-norm", default="train_set_fixed")
    parser.add_argument("--baseline-ref", default="shaft_laplacian")
    parser.add_argument("--baseline-view", default="stft_abs")
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip the C.0 baseline cell (use if upstream parity already established).",
    )
    parser.add_argument(
        "--only-baseline", action="store_true",
        help="Dispatch only the C.0 baseline cell (skip C.1 / C.2). "
             "Useful for backfilling C.0 into a sweep dispatched with --skip-baseline.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sessions", default="",
        help="Comma-separated subject:trial pairs; default = all BT Lite sessions except subject 2.",
    )
    parser.add_argument("--partition", default="common,scavenger,coganlab-gpu")
    parser.add_argument("--account", default="coganlab")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", default="24G")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
