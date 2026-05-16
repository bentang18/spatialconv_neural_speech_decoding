"""Submit Neuroprobe Stage 0 L.1 normalization sweep at CrossSubject on DCC.

Re-runs the L.1 normalization candidates at CrossSubject after the N1-vs-N8
reversal (CSession N1 wins by 5.8 pp / CSubject N8 wins by ~1 pp) showed
the L.1 freeze is CSession-conditional. Mechanism + lens:
[[project_per_channel_scale_within_vs_cross_protocol_2026_05_15]].

Lens prediction:
- N0 (per_window_z) + N8 (per_channel z): DESTROY per-channel scale → win
  or tie at CSubject (already confirmed for N8)
- N1/N2/N3/N5/N6/N7: PRESERVE per-channel scale → lose ground at CSubject

Holds reference = shaft_laplacian (R4 winner) and view = stft_abs (I2 winner)
constant. N1 (baseline) and N8 already have CSubject data from prior runs;
this sweep fills the missing 7 cells.

L.1.* × R4 × I2, seed=42, 7 cells × 10 BT_LITE sessions (subject 2
excluded as upstream DS_DM_TRAIN_SUBJECT_ID) = 70 jobs at 64G.
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


TRAIN_SUBJECT_ID = 2

# (cell_id, normalization)  --  N1 and N8 already have CSubject data, omitted.
NORM_CELLS: tuple[tuple[str, str], ...] = (
    ("L.1.N0_per_window_z",            "per_window_z"),
    ("L.1.N2_per_session_fixed",       "per_session_fixed"),
    ("L.1.N3_train_set_scale_only",    "train_set_scale_only"),
    ("L.1.N4_none",                    "none"),
    ("L.1.N5_per_session_robust_mad",  "per_session_robust_mad"),
    ("L.1.N6_train_set_robust_mad",    "train_set_robust_mad"),
    ("L.1.N7_per_session_robust_scale","per_session_robust_scale"),
)


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_l1_normalization_cross_subject_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    sessions = [(s, t) for (s, t) in BT_LITE_SESSIONS if s != TRAIN_SUBJECT_ID]

    rows: list[dict[str, str]] = []
    for cell_id, normalization in NORM_CELLS:
        for sub_id, trial_id in sessions:
            report_dir = out_root / cell_id / f"sub{sub_id}_trial{trial_id}"
            report_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(report_dir, sub_id, trial_id, normalization, args, cell_id)
            sbatch_path = report_dir / "submit.sbatch"
            sbatch_path.write_text(build_sbatch(
                f"l1cs_{cell_id.split('.')[-1][:12]}_s{sub_id}_t{trial_id}",
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
                "normalization": normalization,
                "ref_kind": "shaft_laplacian",
                "view_kind": "stft_abs",
                "split_type": "CrossSubject",
                "subject_id": str(sub_id),
                "trial_id": str(trial_id),
                "report_dir": str(report_dir),
                "sbatch_path": str(sbatch_path),
            })

    write_manifest(out_root / "launch_manifest.csv", rows)
    print(f"Wrote {out_root / 'launch_manifest.csv'} ({len(rows)} jobs)")


def build_command(
    report_dir: Path, sub_id: int, trial_id: int, normalization: str,
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
        "--split-type", "CrossSubject",
        "--binary-tasks", "false",
        "--backend", "neuralset",
        "--ref-kind", "shaft_laplacian",
        "--view-kind", "stft_abs",
        "--normalization", normalization,
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
    p.add_argument("--partition", default="scavenger-gpu,common,scavenger")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--mem", default="64G")
    p.add_argument("--time", default="12:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
