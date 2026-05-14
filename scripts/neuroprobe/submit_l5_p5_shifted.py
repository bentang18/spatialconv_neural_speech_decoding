"""Submit Stage-0 L.5.P5 shifted (post-trial) window leakage probe on DCC.

Loads [0, +6]s of post-onset context and slices the [+5, +6]s sub-window for
features. This places the window 5 seconds after onset — well after the
production response is over (~450ms duration) and well into the next trial's
ITI/quiescence. Flag if AUROC > chance + 0.05.

If a window 5 s after onset still decodes the current trial's label, either
(a) the trial duration estimate is wrong, (b) some block-level confound
extends across trials, or (c) the label is leaking into something
non-neural in the recording.

12 BT Lite sessions × 1 seed = 12 jobs.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from speech_decoding.studies.braintreebank.labels import NEUROPROBE_TASKS
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_l5_p5_shifted_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    sessions = list(BT_LITE_SESSIONS)
    cell_id = "L.5.P5"

    rows: list[dict[str, str]] = []
    for sub_id, trial_id in sessions:
        report_dir = out_root / cell_id / f"sub{sub_id}_trial{trial_id}"
        report_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(report_dir, sub_id, trial_id, args, cell_id)
        sbatch_path = report_dir / "submit.sbatch"
        sbatch_path.write_text(build_sbatch(
            f"l5p5_sub{sub_id}_trial{trial_id}",
            report_dir, command, args,
        ))
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
            "subject_id": str(sub_id),
            "trial_id": str(trial_id),
            "report_dir": str(report_dir),
            "sbatch_path": str(sbatch_path),
        })
    _write_manifest(out_root / "launch_manifest.csv", rows)
    print(f"Wrote {out_root / 'launch_manifest.csv'} ({len(rows)} jobs)")


def build_command(
    report_dir: Path, sub_id: int, trial_id: int,
    args: argparse.Namespace, cell_id: str,
) -> str:
    cmd = [
        ".venv/bin/python",
        "scripts/neuroprobe/run_stage0_linear_baseline.py",
        "--bt-root", str(args.bt_root),
        "--neuroprobe-repo", str(args.neuroprobe_repo),
        "--out-dir", str(report_dir),
        "--subject-id", str(sub_id),
        "--trial-id", str(trial_id),
        "--task", ",".join(NEUROPROBE_TASKS),
        "--split-type", "CrossSession",
        "--binary-tasks", "false",
        "--backend", "neuralset",
        "--ref-kind", "shaft_laplacian",
        "--view-kind", "stft_abs",
        "--normalization", "train_set_fixed",
        "--anchor-start-before", "0.0",
        "--anchor-end-after", "6.0",
        "--feature-bin-start", "5.0",
        "--feature-bin-end", "6.0",
        "--cell-id", cell_id,
        "--seed", str(args.seed),
    ]
    return ("ROOT_DIR_BRAINTREEBANK=" + shlex.quote(str(args.bt_root))
            + " " + shlex.join(cmd))


def build_sbatch(
    job_name: str, report_dir: Path, command: str, args: argparse.Namespace,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name[:48]}",
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


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
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
    p.add_argument("--partition", default="common,scavenger,coganlab-gpu")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--mem", default="48G")
    p.add_argument("--time", default="12:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
