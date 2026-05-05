"""Submit Neuroprobe Stage 0 L.1 normalization sweep on DCC.

Dispatches one sbatch job per (cell × session). Each job runs all 15 Neuroprobe
tasks on one BT Lite session under one normalization recipe. Default sweep covers
all 6 cells × 12 Lite sessions = 72 jobs on CrossSession multiclass (D.0b
protocol). CrossSubject binary (D.0a protocol) is dispatched only when
`--include-cross-subject` is set, since L.1's headline is the multiclass gap.

Each job's working directory is a per-cell-per-session report folder; the
`run_stage0_linear_baseline.py` wrapper writes diagnostics.csv, metrics.json,
signal_qc.png, and an ExperimentLogger sidecar inside it.
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


TRAIN_SUBJECT_ID = 2

NORMALIZATION_CELLS: tuple[tuple[str, str], ...] = (
    ("L.1.N0", "per_window_z"),
    ("L.1.N1", "train_set_fixed"),
    ("L.1.N2", "per_session_fixed"),
    ("L.1.N3", "train_set_scale_only"),
    ("L.1.N4", "none"),
    ("L.1.N5", "per_session_robust_mad"),
)


@dataclass(frozen=True)
class SweepJob:
    cell: str
    normalization: str
    split_type: str
    binary_tasks: bool
    subject_id: int
    trial_id: int
    tasks: tuple[str, ...]

    @property
    def eval_mode(self) -> str:
        return "binary" if self.binary_tasks else "multiclass"

    @property
    def name(self) -> str:
        return (
            f"l1_{self.normalization}_{self.eval_mode}_"
            f"sub{self.subject_id}_trial{self.trial_id}"
        )


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_l1_normalization_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(
        cells=args.cells.split(",") if args.cells else None,
        sessions=_parse_sessions(args.sessions),
        include_cross_subject=args.include_cross_subject,
    )

    rows: list[dict[str, str]] = []
    for job in jobs:
        report_dir = out_root / job.cell / f"sub{job.subject_id}_trial{job.trial_id}"
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
                check=True,
                text=True,
                capture_output=True,
            )
            print(result.stdout.strip())
            job_id = result.stdout.strip().split()[-1]
        rows.append(
            {
                "job_id": job_id,
                "cell": job.cell,
                "normalization": job.normalization,
                "split_type": job.split_type,
                "eval_mode": job.eval_mode,
                "subject_id": str(job.subject_id),
                "trial_id": str(job.trial_id),
                "tasks": ",".join(job.tasks),
                "report_dir": str(report_dir),
                "sbatch_path": str(sbatch_path),
            }
        )

    manifest_path = out_root / "launch_manifest.csv"
    write_manifest(manifest_path, rows)
    print(f"Wrote {manifest_path} ({len(rows)} jobs)")


def build_jobs(
    *,
    cells: list[str] | None,
    sessions: list[tuple[int, int]] | None,
    include_cross_subject: bool,
) -> list[SweepJob]:
    cell_filter = set(cells) if cells else None
    session_set = sessions if sessions is not None else list(BT_LITE_SESSIONS)
    jobs: list[SweepJob] = []
    for cell_id, normalization in NORMALIZATION_CELLS:
        if cell_filter and cell_id not in cell_filter and normalization not in cell_filter:
            continue
        for sub_id, trial_id in session_set:
            jobs.append(
                SweepJob(
                    cell=cell_id,
                    normalization=normalization,
                    split_type="CrossSession",
                    binary_tasks=False,
                    subject_id=sub_id,
                    trial_id=trial_id,
                    tasks=NEUROPROBE_TASKS,
                )
            )
        if include_cross_subject:
            for sub_id, trial_id in session_set:
                if sub_id == TRAIN_SUBJECT_ID:
                    continue
                jobs.append(
                    SweepJob(
                        cell=cell_id,
                        normalization=normalization,
                        split_type="CrossSubject",
                        binary_tasks=True,
                        subject_id=sub_id,
                        trial_id=trial_id,
                        tasks=NEUROPROBE_TASKS,
                    )
                )
    return jobs


def build_command(job: SweepJob, report_dir: Path, args: argparse.Namespace) -> str:
    cmd = [
        ".venv/bin/python",
        "scripts/neuroprobe/run_stage0_linear_baseline.py",
        "--bt-root",
        str(args.bt_root),
        "--neuroprobe-repo",
        str(args.neuroprobe_repo),
        "--out-dir",
        str(report_dir),
        "--subject-id",
        str(job.subject_id),
        "--trial-id",
        str(job.trial_id),
        "--task",
        ",".join(job.tasks),
        "--split-type",
        job.split_type,
        "--binary-tasks",
        "true" if job.binary_tasks else "false",
        "--preprocess-type",
        args.preprocess_type,
        "--normalization",
        job.normalization,
        "--seed",
        str(args.seed),
    ]
    return "ROOT_DIR_BRAINTREEBANK=" + shlex.quote(str(args.bt_root)) + " " + shlex.join(cmd)


def build_sbatch(
    job: SweepJob,
    report_dir: Path,
    command: str,
    args: argparse.Namespace,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job.name[:48]}",
        f"#SBATCH -p {args.partition}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH -t {args.time}",
        f"#SBATCH -o {report_dir.resolve()}/slurm-%j.out",
        f"#SBATCH -e {report_dir.resolve()}/slurm-%j.err",
    ]
    if args.account:
        lines.insert(3, f"#SBATCH --account={args.account}")
    lines.extend(
        [
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
        ]
    )
    return "\n".join(lines)


def _parse_sessions(spec: str | None) -> list[tuple[int, int]] | None:
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
    parser.add_argument(
        "--bt-root",
        type=Path,
        default=Path("/work/ht203/data/braintreebank"),
    )
    parser.add_argument(
        "--neuroprobe-repo",
        type=Path,
        default=Path("/work/ht203/repo/neuroprobe_upstream"),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/work/ht203/repo/speech"),
        help="Working directory for the launched job (project root on DCC).",
    )
    parser.add_argument("--preprocess-type", default="laplacian-stft_abs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cells",
        default="",
        help="Comma-separated cell IDs (L.1.N0..N5) or recipe names; default = all 6.",
    )
    parser.add_argument(
        "--sessions",
        default="",
        help="Comma-separated subject:trial pairs (e.g. '1:1,2:4,10:1'); default = all 12 BT Lite sessions.",
    )
    parser.add_argument(
        "--include-cross-subject",
        action="store_true",
        help="Also dispatch CrossSubject binary jobs (D.0a-style); off by default.",
    )
    parser.add_argument("--partition", default="coganlab-gpu")
    parser.add_argument("--account", default="coganlab")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", default="160G")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
