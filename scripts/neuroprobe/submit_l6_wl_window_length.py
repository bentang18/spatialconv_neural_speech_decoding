"""Submit Neuroprobe Stage 0 L.6.WL window-length sweep on DCC.

Tests robustness of the L.4 frozen anchor to window length, holding the start
of the feature window fixed at word onset (anchor_start_before=0.0) and
sweeping anchor_end_after ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0} s. The L.4 D.0
baseline is length=1.0 (= L.6.WL.1p00).

Spec note: this redefines L.6.WL from the original docs/neuroprobe/stage_0.md
sketch (which described sub-windowing). The simpler length-only sweep here
isolates the "how much temporal context" question from the orthogonal
"sub-window concat vs mean-pool" design choice.

L.2 winner config (R4×I2×N1 = shaft_laplacian × stft_abs × train_set_fixed),
seed=42 only (sweep over length, not seed). 6 lengths × 12 BT Lite sessions
= 72 jobs.
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


# (cell_id, anchor_end_after_seconds)
LENGTHS: tuple[tuple[str, float], ...] = (
    ("L.6.WL.0p25", 0.25),
    ("L.6.WL.0p50", 0.50),
    ("L.6.WL.0p75", 0.75),
    ("L.6.WL.1p00", 1.00),
    ("L.6.WL.1p50", 1.50),
    ("L.6.WL.2p00", 2.00),
)


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_l6_wl_window_length_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    sessions = list(BT_LITE_SESSIONS)

    rows: list[dict[str, str]] = []
    for cell_id, end_after in LENGTHS:
        for sub_id, trial_id in sessions:
            report_dir = out_root / cell_id / f"sub{sub_id}_trial{trial_id}"
            report_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(report_dir, sub_id, trial_id, end_after, args, cell_id)
            sbatch_path = report_dir / "submit.sbatch"
            sbatch_path.write_text(build_sbatch(
                f"l6wl_{cell_id.split('.')[-1]}_sub{sub_id}_trial{trial_id}",
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
                "anchor_end_after_s": str(end_after),
                "subject_id": str(sub_id),
                "trial_id": str(trial_id),
                "report_dir": str(report_dir),
                "sbatch_path": str(sbatch_path),
            })

    write_manifest(out_root / "launch_manifest.csv", rows)
    print(f"Wrote {out_root / 'launch_manifest.csv'} ({len(rows)} jobs)")


def build_command(
    report_dir: Path, sub_id: int, trial_id: int, end_after: float,
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
        "--anchor-end-after", str(end_after),
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
    p.add_argument("--partition", default="common,scavenger,coganlab-gpu")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--mem", default="32G")
    p.add_argument("--time", default="12:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
