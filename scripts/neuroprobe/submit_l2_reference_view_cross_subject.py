"""Submit Neuroprobe Stage 0 L.2 reference × view sweep at CrossSubject on DCC.

Re-runs the L.2 reference and view alternatives at CrossSubject after
the N1-vs-N8 reversal showed L.2 freeze (R4 × I2) is CSession-conditional.
Mechanism + lens:
[[project_per_channel_scale_within_vs_cross_protocol_2026_05_15]].

Lens predictions:
- Reference axis (5 alts vs R4 winner): scale-destroying refs (bipolar,
  shaft_laplacian R4) likely robust cross-subject (C.3 already showed
  bipolar GENERALIZES); scale-preserving refs (raw, global_car, shaft_car,
  median) likely lose ground at CSubject.
- View axis (3 alts vs I2 winner): amplitude-sensitive views likely lose
  ground; instantaneous_phase is amplitude-invariant — the sleeper that
  might survive cross-subject best.

Holds normalization = train_set_fixed (N1 winner) constant. The R4 × I2
baseline at CSubject is already covered by L.1's N1 cell. Band views
(B30_70 / B70_90 / B90_120 / B120_150 / B150_300 / B300_500) are already
in flight via submit_l6_nr_band_sweep_cross_subject.py. C.4 already covers
hg_envelope (HG HURTS at CSubject).

Cell list (8 cells):
  Reference axis (vary ref, hold view=stft_abs):
    L.2.R0_raw_x_I2
    L.2.R1_bipolar_x_I2
    L.2.R3_global_car_x_I2
    L.2.R5_shaft_car_x_I2
    L.2.R6_median_x_I2
  View axis (vary view, hold ref=shaft_laplacian):
    L.2.R4_x_I0_raw_voltage
    L.2.R4_x_log_stft
    L.2.R4_x_instantaneous_phase

8 cells × 10 BT_LITE sessions (subject 2 excluded as upstream
DS_DM_TRAIN_SUBJECT_ID) = 80 jobs at 64G.
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


@dataclass(frozen=True)
class L2Cell:
    cell_id: str
    ref_kind: str
    view_kind: str


CELLS: tuple[L2Cell, ...] = (
    # Reference axis — vary ref, hold view=stft_abs (I2 winner)
    L2Cell("L.2.R0_raw_x_I2",              "raw",            "stft_abs"),
    L2Cell("L.2.R1_bipolar_x_I2",          "bipolar",        "stft_abs"),
    L2Cell("L.2.R3_global_car_x_I2",       "global_car",     "stft_abs"),
    L2Cell("L.2.R5_shaft_car_x_I2",        "shaft_car",      "stft_abs"),
    L2Cell("L.2.R6_median_x_I2",           "median",         "stft_abs"),
    # View axis — vary view, hold ref=shaft_laplacian (R4 winner)
    L2Cell("L.2.R4_x_I0_raw_voltage",      "shaft_laplacian","raw_voltage"),
    L2Cell("L.2.R4_x_log_stft",            "shaft_laplacian","log_stft"),
    L2Cell("L.2.R4_x_instantaneous_phase", "shaft_laplacian","instantaneous_phase"),
)


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_l2_reference_view_cross_subject_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    sessions = [(s, t) for (s, t) in BT_LITE_SESSIONS if s != TRAIN_SUBJECT_ID]

    rows: list[dict[str, str]] = []
    for cell in CELLS:
        for sub_id, trial_id in sessions:
            report_dir = out_root / cell.cell_id / f"sub{sub_id}_trial{trial_id}"
            report_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(report_dir, sub_id, trial_id, cell, args)
            sbatch_path = report_dir / "submit.sbatch"
            sbatch_path.write_text(build_sbatch(
                f"l2cs_{cell.cell_id.split('.')[-1][:14]}_s{sub_id}t{trial_id}",
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
                "cell_id": cell.cell_id,
                "ref_kind": cell.ref_kind,
                "view_kind": cell.view_kind,
                "normalization": "train_set_fixed",
                "split_type": "CrossSubject",
                "subject_id": str(sub_id),
                "trial_id": str(trial_id),
                "report_dir": str(report_dir),
                "sbatch_path": str(sbatch_path),
            })

    write_manifest(out_root / "launch_manifest.csv", rows)
    print(f"Wrote {out_root / 'launch_manifest.csv'} ({len(rows)} jobs)")


def build_command(
    report_dir: Path, sub_id: int, trial_id: int, cell: L2Cell,
    args: argparse.Namespace,
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
        "--ref-kind", cell.ref_kind,
        "--view-kind", cell.view_kind,
        "--normalization", "train_set_fixed",
        "--cell-id", cell.cell_id,
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
