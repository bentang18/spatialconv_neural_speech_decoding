"""Submit the FE filterbank sweep on DCC (CPU `common` partition).

Settles v14's log-octave (constant-Q) filterbank vs iMINDBench raw-STFT-bins,
and the {f0, cap, octave_step, half_bw} knobs, with a cheap logistic probe.
Spec: `reports/fe_filterbank_sweep_build_2026_06_03.md`.

One sbatch job per (cell × eval-type × session). Each job runs all 15 Neuroprobe
binary tasks under one (reference, view, filterbank) config via
`run_stage0_linear_baseline.py --backend neuralset`. The collector reads both the
rank-4 headline {speech, onset, volume, pitch} and the all-15 robustness mean.

Grid (build doc §4): G gate (Laplacian + raw bins, harness check, run once),
0 control (shaft-CAR + raw bins, Δ-baseline), 1 default (constant-Q matched),
2 drop-delta, 3 trim-octave, 4 finer-⅓, 5 finer-¼, 6 smoother-0.75, 7 smoother-1.0.
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
from speech_decoding.views import multi_stft_dead_bin_count


# Cross-subject trains on sub 2 / trial 4 (upstream DS_DM_TRAIN) — never a test cell.
TRAIN_SUBJECT_ID = 2

# Frozen multi-STFT grid sampling rate (build doc §3). Used only to count dead bins.
SAMPLING_RATE = 2048

# Headline ranking set (build doc §5). Reported alongside the all-15 mean; both
# are computed by the collector from the same per-task rows.
RANK4_TASKS: tuple[str, ...] = ("speech", "onset", "volume", "pitch")

THIRD = 1.0 / 3.0


@dataclass(frozen=True)
class Cell:
    cell_id: str
    ref_kind: str
    view_kind: str
    f0_hz: float
    cap_hz: float
    octave_step: float
    half_bw: float
    role: str


# build doc §4 — coarse first pass, one knob off the default per cell.
CELLS: tuple[Cell, ...] = (
    Cell("G", "shaft_laplacian", "raw_multi_stft", 2.0, 256.0, 0.5, 0.5,
         "iMINDBench-exact gate (Laplacian + raw bins); reproduce ~0.685, run once"),
    Cell("0", "shaft_car", "raw_multi_stft", 2.0, 256.0, 0.5, 0.5,
         "control: shaft-CAR raw bins, Δ baseline"),
    Cell("1", "shaft_car", "multi_stft", 2.0, 256.0, 0.5, 0.5,
         "default: constant-Q matched (15 bins)"),
    Cell("2", "shaft_car", "multi_stft", 4.0, 256.0, 0.5, 0.5,
         "drop-delta: f0=4 (13 bins)"),
    Cell("3", "shaft_car", "multi_stft", 2.0, 128.0, 0.5, 0.5,
         "trim-octave: cap=128 (13 bins)"),
    Cell("4", "shaft_car", "multi_stft", 2.0, 256.0, THIRD, THIRD,
         "finer-⅓: step=⅓ matched (22 bins)"),
    Cell("5", "shaft_car", "multi_stft", 2.0, 256.0, 0.25, 0.25,
         "finer-¼: step=¼ matched (29 bins)"),
    Cell("6", "shaft_car", "multi_stft", 2.0, 256.0, 0.5, 0.75,
         "smoother-0.75: wider triangle at ½ spacing (15 bins)"),
    Cell("7", "shaft_car", "multi_stft", 2.0, 256.0, 0.5, 1.0,
         "smoother-1.0: max averaging at ½ spacing (15 bins)"),
)
CELLS_BY_ID = {c.cell_id: c for c in CELLS}


@dataclass(frozen=True)
class SweepJob:
    cell: Cell
    split_type: str  # WithinSession | CrossSubject
    subject_id: int
    trial_id: int
    tasks: tuple[str, ...]

    @property
    def eval_tag(self) -> str:
        return "within" if self.split_type == "WithinSession" else "cross"

    @property
    def name(self) -> str:
        return f"fe_{self.cell.cell_id}_{self.eval_tag}_sub{self.subject_id}_t{self.trial_id}"


def build_jobs(
    *,
    cell_ids: list[str] | None,
    sessions: list[tuple[int, int]] | None,
    eval_types: list[str],
) -> list[SweepJob]:
    cells = [CELLS_BY_ID[c] for c in cell_ids] if cell_ids else list(CELLS)
    session_set = sessions if sessions is not None else list(BT_LITE_SESSIONS)
    jobs: list[SweepJob] = []
    for cell in cells:
        if "within" in eval_types:
            for sub_id, trial_id in session_set:
                jobs.append(SweepJob(cell, "WithinSession", sub_id, trial_id, NEUROPROBE_TASKS))
        # Cell G is the iMINDBench-exact harness gate (§2): reproduce ~0.685 once,
        # within-session only. It is not a Δ-ranking cell, so it skips the cross fan-out.
        if "cross" in eval_types and cell.cell_id != "G":
            for sub_id, trial_id in session_set:
                if sub_id == TRAIN_SUBJECT_ID:
                    continue  # sub 2 is the fixed train subject, never a test cell
                jobs.append(SweepJob(cell, "CrossSubject", sub_id, trial_id, NEUROPROBE_TASKS))
    return jobs


def build_command(job: SweepJob, report_dir: Path, args: argparse.Namespace) -> str:
    cmd = [
        ".venv/bin/python",
        "scripts/neuroprobe/run_stage0_linear_baseline.py",
        "--bt-root", str(args.bt_root),
        "--neuroprobe-repo", str(args.neuroprobe_repo),
        "--out-dir", str(report_dir),
        "--subject-id", str(job.subject_id),
        "--trial-id", str(job.trial_id),
        "--task", ",".join(job.tasks),
        "--split-type", job.split_type,
        "--binary-tasks", "true",
        "--backend", "neuralset",
        "--ref-kind", job.cell.ref_kind,
        "--view-kind", job.cell.view_kind,
        "--fbank-f0-hz", str(job.cell.f0_hz),
        "--fbank-cap-hz", str(job.cell.cap_hz),
        "--fbank-octave-step", repr(job.cell.octave_step),
        "--fbank-half-bw", repr(job.cell.half_bw),
        "--normalization", args.normalization,
        "--notch-freqs", args.notch_freqs,  # build doc §3: per-corpus notch frozen across all cells (BT = US = 60 Hz)
        "--seed", str(args.seed),
    ]
    return "ROOT_DIR_BRAINTREEBANK=" + shlex.quote(str(args.bt_root)) + " " + shlex.join(cmd)


def build_sbatch(job: SweepJob, report_dir: Path, command: str, args: argparse.Namespace) -> str:
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
    lines.extend([
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


def main() -> None:
    args = _parse_args()
    out_root = args.out_root or Path(
        f"reports/fe_filterbank_sweep_{datetime.now():%Y_%m_%d}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(
        cell_ids=args.cells.split(",") if args.cells else None,
        sessions=_parse_sessions(args.sessions),
        eval_types=[e.strip() for e in args.eval_types.split(",") if e.strip()],
    )

    # Dead-bin count per filterbank cell (FE-MULTISTFT-1): constant-Q triangles
    # that fall between rfft bins become constant-zero features and confound the
    # resolution axis (cells 4/5). Reported into the manifest so the analyst can
    # read the confound; does NOT change the filterbank.
    dead_by_cell: dict[str, int] = {}
    for cell in {j.cell.cell_id: j.cell for j in jobs}.values():
        if cell.view_kind == "multi_stft":
            dead_by_cell[cell.cell_id] = multi_stft_dead_bin_count(
                SAMPLING_RATE,
                {"f0_hz": cell.f0_hz, "cap_hz": cell.cap_hz,
                 "octave_step": cell.octave_step, "half_bw_octaves": cell.half_bw},
            )
    flagged = {cid: n for cid, n in dead_by_cell.items() if n > 0}
    if flagged:
        print("WARNING: dead constant-Q bins (FE-MULTISTFT-1, confounds resolution axis): "
              + ", ".join(f"cell {cid}={n}" for cid, n in sorted(flagged.items())))

    rows: list[dict[str, str]] = []
    for job in jobs:
        report_dir = out_root / job.cell.cell_id / f"{job.eval_tag}_sub{job.subject_id}_t{job.trial_id}"
        report_dir.mkdir(parents=True, exist_ok=True)
        sbatch_path = report_dir / "submit.sbatch"
        command = build_command(job, report_dir, args)
        sbatch_path.write_text(build_sbatch(job, report_dir, command, args))
        job_id = ""
        if args.dry_run:
            print(f"[dry-run] {sbatch_path}")
        else:
            result = subprocess.run(
                ["sbatch", str(sbatch_path)], check=True, text=True, capture_output=True,
            )
            print(result.stdout.strip())
            job_id = result.stdout.strip().split()[-1]
        rows.append({
            "job_id": job_id,
            "cell": job.cell.cell_id,
            "ref_kind": job.cell.ref_kind,
            "view_kind": job.cell.view_kind,
            "f0_hz": str(job.cell.f0_hz),
            "cap_hz": str(job.cell.cap_hz),
            "octave_step": repr(job.cell.octave_step),
            "half_bw": repr(job.cell.half_bw),
            "dead_bins": str(dead_by_cell.get(job.cell.cell_id, 0)),
            "eval": job.eval_tag,
            "split_type": job.split_type,
            "subject_id": str(job.subject_id),
            "trial_id": str(job.trial_id),
            "tasks": ",".join(job.tasks),
            "report_dir": str(report_dir),
            "sbatch_path": str(sbatch_path),
        })

    manifest_path = out_root / "launch_manifest.csv"
    write_manifest(manifest_path, rows)
    print(f"Wrote {manifest_path} ({len(rows)} jobs)")


def _parse_args() -> argparse.Namespace:
    import os

    env_root = os.environ.get("ROOT_DIR_BRAINTREEBANK")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bt-root", type=Path, default=Path(env_root) if env_root else None, required=env_root is None)
    p.add_argument("--neuroprobe-repo", type=Path, default=Path("/work/ht203/repo/neuroprobe_upstream"))
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--cells", default=None, help="Comma list of cell ids (default: all). e.g. 0,1")
    p.add_argument("--sessions", default=None, help="Comma list sub:trial (default: all lite). e.g. 1:1,3:0")
    p.add_argument("--eval-types", default="within,cross", help="Comma list of {within,cross}.")
    p.add_argument("--normalization", default="per_session_robust_mad_cf",
                   help="Held-constant normalization recipe. Default = per-(C,F) robust median+MAD "
                        "POOLED OVER TIME (build doc §3 'per-(C,F) session robust-z' + v14 Nv14): one "
                        "stat per (channel,freq) over trials+time, so the within-window temporal "
                        "envelope is preserved. Frozen across all cells.")
    p.add_argument("--notch-freqs", default="60,120,180",
                   help="Held-constant mains notch (Hz, comma list). Default = v14 canonical "
                        "recipe 60/120/180 (BT = US). Frozen across all cells (build doc §3); "
                        "never affects ranking. Pass '' to disable.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--partition", default="common")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpus-per-task", type=int, default=8,
                   help="BLAS threads for the LogReg gradient (the bottleneck on the "
                        "~166k-feature raw-bins cells). Free on `common`.")
    p.add_argument("--mem", default="128G",
                   help="Sized for the cross cells: the L.3 canonical filter upcasts the FULL "
                        "session voltage to float64 and sosfiltfilt holds input+output at once. "
                        "Train sub 2 / trial 4 is 164 ch x 27.5M samp = 36 GB float64 -> ~72 GB "
                        "transient + the persistent test cache. 32G OOMs. `common` mem is free.")
    p.add_argument("--time", default="08:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
