"""Submit Neuroprobe Stage 0 L.3 filtering sweep on DCC.

Runs on the L.2 winner (R4xI2 = shaft_laplacian × stft_abs) + L.1 winner
(N1 = train_set_fixed). Tests whether more aggressive front-end cleaning
helps the linear baseline.

Cells dispatched (per `docs/neuroprobe/stage_0.md` L.3 spec):
  L.3.F0  none (status quo, parity to L.2 winner)
  L.3.F1  60 Hz notch + 120 + 180 Hz harmonics
  L.3.F2  F1 + 0.5 Hz Butterworth HPF
  L.3.F3  F1 + 1.0 Hz Butterworth HPF

Filters apply to the FULL session voltage before trial windowing — see
`scripts/neuroprobe/preprocess_views.py:apply_temporal_filter_inplace`.

L.3.E0 (BT Lite mask only) is identical to L.3.F0 — no separate dispatch.
L.3.E1 (extra robust per-channel exclusions: flatline + amplitude-outlier +
clipping) deferred to a follow-up: requires session-statistic-driven channel
mask + cache reslicing, complexity not worth bundling with filtering MVP.

Usage:
    .venv/bin/python scripts/neuroprobe/submit_l3_filtering_sweep.py \\
        --include-f0  # parity row, optional
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


@dataclass(frozen=True)
class FilterCell:
    cell_id: str
    notch_freqs: str
    hpf_hz: float
    description: str


CELLS: tuple[FilterCell, ...] = (
    FilterCell("L.3.F0", "",            0.0, "no filter (parity to L.2 winner)"),
    FilterCell("L.3.F1", "60,120,180",  0.0, "60 Hz + harmonics notch"),
    FilterCell("L.3.F2", "60,120,180",  0.5, "F1 + 0.5 Hz HPF"),
    FilterCell("L.3.F3", "60,120,180",  1.0, "F1 + 1.0 Hz HPF"),
)


@dataclass(frozen=True)
class FilterJob:
    cell: FilterCell
    subject_id: int
    trial_id: int
    ref_kind: str
    view_kind: str
    normalization: str

    @property
    def name(self) -> str:
        return (
            f"l3flt_{self.cell.cell_id.lower().replace('.', '_')}"
            f"_sub{self.subject_id}_trial{self.trial_id}"
        )


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/neuroprobe_stage0_l3_filtering_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    sessions = _parse_sessions(args.sessions) or list(BT_LITE_SESSIONS)
    cells = [c for c in CELLS if (args.include_f0 or c.cell_id != "L.3.F0")]
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = [c for c in cells if c.cell_id in wanted]
    if not cells:
        raise SystemExit("No L.3 cells selected after filtering.")

    rows: list[dict[str, str]] = []
    for cell in cells:
        for sub_id, trial_id in sessions:
            job = FilterJob(
                cell=cell,
                subject_id=sub_id, trial_id=trial_id,
                ref_kind=args.ref_kind, view_kind=args.view_kind,
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
                    check=True, text=True, capture_output=True,
                )
                print(result.stdout.strip())
                job_id = result.stdout.strip().split()[-1]
            rows.append({
                "job_id": job_id,
                "cell_id": cell.cell_id,
                "notch_freqs": cell.notch_freqs,
                "hpf_hz": str(cell.hpf_hz),
                "description": cell.description,
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


def build_command(job: FilterJob, report_dir: Path, args: argparse.Namespace) -> str:
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
        "--cell-id", job.cell.cell_id,
        "--notch-freqs", job.cell.notch_freqs,
        "--hpf-hz", str(job.cell.hpf_hz),
        "--seed", str(args.seed),
    ]
    return "ROOT_DIR_BRAINTREEBANK=" + shlex.quote(str(args.bt_root)) + " " + shlex.join(cmd)


def build_sbatch(
    job: FilterJob, report_dir: Path, command: str, args: argparse.Namespace
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--bt-root", type=Path, default=Path("/work/ht203/data/braintreebank"))
    p.add_argument(
        "--neuroprobe-repo", type=Path,
        default=Path("/work/ht203/repo/neuroprobe_upstream"),
    )
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument("--ref-kind", default="shaft_laplacian", help="L.2 winner reference")
    p.add_argument("--view-kind", default="stft_abs", help="L.2 winner view")
    p.add_argument("--normalization", default="train_set_fixed", help="L.1 winner")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sessions", default="",
                   help="Comma-separated subject:trial pairs; default = all 12 BT Lite sessions.")
    p.add_argument("--cells", default="",
                   help="Comma-separated cell IDs to dispatch (default = all). "
                        "E.g. 'L.3.F1,L.3.F2'.")
    p.add_argument("--include-f0", action="store_true",
                   help="Include L.3.F0 parity cell. Default skip — L.3.F0 byte-equivalent "
                        "to L.2 winner R4xI2 already on disk.")
    p.add_argument("--partition", default="common,scavenger,coganlab-gpu")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--mem", default="48G")
    p.add_argument("--time", default="12:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
