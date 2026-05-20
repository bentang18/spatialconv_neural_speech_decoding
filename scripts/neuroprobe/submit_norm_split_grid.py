"""Submit the v14 normalization-split linear sweep on DCC.

Settles one open v14 question: does Nv14 — robust z per (electrode, freq-bin,
session) — help or hurt by fusing two jobs (per-recording scale removal AND
1/f equalization)? No existing N0-N8 cell tests the per-freq-bin axis (N8
pools over freq). One linear sweep decides it.

Two normalization cells, both per-session robust (median + 1.4826*MAD,
transductive), on log_stft × shaft_car:
  A — per_session_robust_freqbin: stat per (electrode, freq-bin) — Nv14 proxy.
  B — per_session_robust_pooled:  stat per electrode pooled over time+freq —
      pooled-z split (Job-1 only), preserves each recording's spectral shape.

Run on BOTH protocols (multiclass): CrossSession (12 BT_LITE sessions) and
CrossSubject (10 sessions, subject 2 excluded as DS_DM train subject).
2 cells × (12 + 10) sessions × 15 tasks = 44 jobs.

Cell C (pooled-z + fixed corpus-shared per-bin gain) is intentionally NOT
included: a fixed diagonal affine applied identically to train and test is
absorbed by a linear classifier's weights, so it is degenerate with B under a
linear probe. The learnable per-bin affine it proxies is meaningful only in
the v14 model, where it is added regardless of this sweep — see
memory/project_v14_normalization_split_experiment_2026_05_19.md.
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

NORMS: tuple[tuple[str, str], ...] = (
    ("A_freqbin", "per_session_robust_freqbin"),
    ("B_pooled",  "per_session_robust_pooled"),
)

PROTOCOLS: tuple[str, ...] = ("CrossSession", "CrossSubject")


@dataclass(frozen=True)
class SweepJob:
    cell_id: str
    normalization: str
    split_type: str
    subject_id: int
    trial_id: int

    @property
    def name(self) -> str:
        return f"ns_{self.cell_id.split('.')[-1][:20]}_s{self.subject_id}t{self.trial_id}"


def build_jobs() -> list[SweepJob]:
    jobs: list[SweepJob] = []
    for norm_tag, normalization in NORMS:
        for protocol in PROTOCOLS:
            tag = "csession" if protocol == "CrossSession" else "csubject"
            cell_id = f"L.NSPLIT.{norm_tag}_{tag}"
            for sub_id, trial_id in BT_LITE_SESSIONS:
                if protocol == "CrossSubject" and sub_id == TRAIN_SUBJECT_ID:
                    continue
                jobs.append(
                    SweepJob(
                        cell_id=cell_id,
                        normalization=normalization,
                        split_type=protocol,
                        subject_id=sub_id,
                        trial_id=trial_id,
                    )
                )
    return jobs


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_normsplit_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for job in build_jobs():
        report_dir = out_root / job.cell_id / f"sub{job.subject_id}_trial{job.trial_id}"
        report_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(report_dir, job, args)
        sbatch_path = report_dir / "submit.sbatch"
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
            "cell_id": job.cell_id,
            "ref_kind": "shaft_car",
            "view_kind": "log_stft",
            "normalization": job.normalization,
            "split_type": job.split_type,
            "subject_id": str(job.subject_id),
            "trial_id": str(job.trial_id),
            "report_dir": str(report_dir),
            "sbatch_path": str(sbatch_path),
        })

    write_manifest(out_root / "launch_manifest.csv", rows)
    print(f"Wrote {out_root / 'launch_manifest.csv'} ({len(rows)} jobs)")


def build_command(report_dir: Path, job: SweepJob, args: argparse.Namespace) -> str:
    cmd = [
        ".venv/bin/python",
        "scripts/neuroprobe/run_stage0_linear_baseline.py",
        "--bt-root", str(args.bt_root),
        "--neuroprobe-repo", str(args.neuroprobe_repo),
        "--out-dir", str(report_dir),
        "--subject-id", str(job.subject_id),
        "--trial-id", str(job.trial_id),
        "--task", ",".join(NEUROPROBE_TASKS),
        "--split-type", job.split_type,
        "--binary-tasks", "false",
        "--backend", "neuralset",
        "--ref-kind", "shaft_car",
        "--view-kind", "log_stft",
        "--normalization", job.normalization,
        "--cell-id", job.cell_id,
        "--seed", str(args.seed),
    ]
    return ("ROOT_DIR_BRAINTREEBANK=" + shlex.quote(str(args.bt_root))
            + " " + shlex.join(cmd))


def build_sbatch(
    job: SweepJob, report_dir: Path, command: str, args: argparse.Namespace,
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


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--bt-root", type=Path, default=Path("/work/ht203/data/braintreebank"))
    p.add_argument(
        "--neuroprobe-repo", type=Path,
        default=Path("/work/ht203/repo/neuroprobe_upstream"),
    )
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--partition", default="scavenger-gpu,common,scavenger")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--mem", default="64G")
    p.add_argument("--time", default="12:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
