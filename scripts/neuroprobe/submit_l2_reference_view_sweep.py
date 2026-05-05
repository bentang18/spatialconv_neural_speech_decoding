"""Submit Neuroprobe Stage 0 L.2 reference × input-view sweep on DCC.

L.2 covers the 12-cell hand-picked subset documented in `docs/neuroprobe/stage_0.md`.
Each cell maps a (reference, input-view) pair to an upstream
`--preprocess-type` string consumed by `preprocess_data` in
`run_stage0_linear_baseline.py`. The normalization recipe is taken from L.1's
winner (default `train_set_fixed` until L.1 lands).

Cells R1 (global CAR) and R2 (within-shaft CAR) require
`CARIeegExtractor`/`ShaftCARIeegExtractor` in `src/speech_decoding/extractors/reference.py`
(L.0e). Until those land, those cells are skipped with a clear notice instead
of failing silently.

Usage on DCC after L.1 lands:
    .venv/bin/python scripts/neuroprobe/submit_l2_reference_view_sweep.py \\
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


TRAIN_SUBJECT_ID = 2


@dataclass(frozen=True)
class L2Cell:
    cell_id: str
    reference: str
    input_view: str
    preprocess_type: str
    requires_extractor: str | None
    note: str


L2_CELLS: tuple[L2Cell, ...] = (
    L2Cell("L.2.R0xI0", "raw_monopolar", "raw_voltage", "none", None,
           "floor / v14-aligned ceiling"),
    L2Cell("L.2.R0xI2", "raw_monopolar", "stft_log_power", "stft_abs", None,
           "spectral without re-reference"),
    L2Cell("L.2.R1xI3", "global_car", "hg_envelope", "remove_line_noise-stft_abs",
           "CARIeegExtractor",
           "conventional CAR+HG control (gated on L.0e)"),
    L2Cell("L.2.R2xI3", "within_shaft_car", "hg_envelope", "remove_line_noise-stft_abs",
           "ShaftCARIeegExtractor",
           "physics-matched sEEG candidate (gated on L.0e)"),
    L2Cell("L.2.R2xI4", "within_shaft_car", "multi_band_log_power",
           "remove_line_noise-stft_abs", "ShaftCARIeegExtractor",
           "richer view on shaft-CAR (gated on L.0e)"),
    L2Cell("L.2.R3xI2", "bipolar", "stft_log_power", "laplacian-stft_abs", None,
           "bipolar spectral (upstream Laplacian = shaft-aware bipolar)"),
    L2Cell("L.2.R3xI3", "bipolar", "hg_envelope", "laplacian", None,
           "bipolar HG conventional"),
    L2Cell("L.2.R3xI4", "bipolar", "multi_band_log_power", "laplacian-stft_abs", None,
           "bipolar wide"),
    L2Cell("L.2.R4xI2", "shaft_laplacian", "stft_log_power", "laplacian-stft_abs", None,
           "D.0 default ('Lap+spec')"),
    L2Cell("L.2.R4xI3", "shaft_laplacian", "hg_envelope", "laplacian", None,
           "biologically privileged"),
    L2Cell("L.2.R4xI4", "shaft_laplacian", "multi_band_log_power",
           "laplacian-stft_abs", None,
           "rich shaft-Lap view"),
    L2Cell("L.2.R4xI5", "shaft_laplacian", "wavelet", "laplacian-stft_abs", None,
           "scale-localized shaft-Lap"),
)


@dataclass(frozen=True)
class SweepJob:
    cell: L2Cell
    subject_id: int
    trial_id: int
    tasks: tuple[str, ...]
    normalization: str

    @property
    def name(self) -> str:
        return (
            f"l2_{self.cell.cell_id.lower().replace('.', '_')}_"
            f"sub{self.subject_id}_trial{self.trial_id}"
        )


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_l2_reference_view_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    sessions = _parse_sessions(args.sessions) or list(BT_LITE_SESSIONS)
    cells = _filter_cells(args.cells, args.skip_gated)

    rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for cell in L2_CELLS:
        if cell.cell_id not in {c.cell_id for c in cells}:
            skipped.append(
                {
                    "cell": cell.cell_id,
                    "reason": "gated on L.0e (extractor missing)" if cell.requires_extractor and args.skip_gated else "filtered out by --cells",
                }
            )
            continue
        for sub_id, trial_id in sessions:
            job = SweepJob(
                cell=cell,
                subject_id=sub_id,
                trial_id=trial_id,
                tasks=NEUROPROBE_TASKS,
                normalization=args.normalization,
            )
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
                    check=True,
                    text=True,
                    capture_output=True,
                )
                print(result.stdout.strip())
                job_id = result.stdout.strip().split()[-1]
            rows.append(
                {
                    "job_id": job_id,
                    "cell": cell.cell_id,
                    "reference": cell.reference,
                    "input_view": cell.input_view,
                    "preprocess_type": cell.preprocess_type,
                    "subject_id": str(sub_id),
                    "trial_id": str(trial_id),
                    "normalization": args.normalization,
                    "tasks": ",".join(NEUROPROBE_TASKS),
                    "report_dir": str(report_dir),
                    "sbatch_path": str(sbatch_path),
                }
            )

    write_manifest(out_root / "launch_manifest.csv", rows)
    if skipped:
        write_manifest(out_root / "skipped_cells.csv", skipped)
    print(f"Wrote {out_root / 'launch_manifest.csv'} ({len(rows)} jobs, {len(skipped)} cells skipped)")


def _filter_cells(spec: str, skip_gated: bool) -> list[L2Cell]:
    cells: list[L2Cell] = list(L2_CELLS)
    if spec:
        wanted = set(spec.split(","))
        cells = [c for c in cells if c.cell_id in wanted]
    if skip_gated:
        cells = [c for c in cells if c.requires_extractor is None]
    return cells


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
        "CrossSession",
        "--binary-tasks",
        "false",
        "--preprocess-type",
        job.cell.preprocess_type,
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
        "--neuroprobe-repo",
        type=Path,
        default=Path("/work/ht203/repo/neuroprobe_upstream"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    parser.add_argument(
        "--normalization",
        default="train_set_fixed",
        help="Normalization recipe; default = upstream/L.1 N1. Set to L.1 winner once L.1 lands.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cells",
        default="",
        help="Comma-separated cell IDs (e.g. 'L.2.R0xI0,L.2.R4xI2'); default = all 12.",
    )
    parser.add_argument(
        "--sessions",
        default="",
        help="Comma-separated subject:trial pairs; default = all 12 BT Lite sessions.",
    )
    parser.add_argument(
        "--skip-gated",
        action="store_true",
        default=True,
        help="Skip cells that require an extractor not yet implemented (R1/R2). Default true until L.0e lands.",
    )
    parser.add_argument(
        "--include-gated",
        dest="skip_gated",
        action="store_false",
        help="Override --skip-gated to attempt R1/R2 cells (will fail until L.0e ships).",
    )
    parser.add_argument("--partition", default="coganlab-gpu")
    parser.add_argument("--account", default="coganlab")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", default="48G")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
