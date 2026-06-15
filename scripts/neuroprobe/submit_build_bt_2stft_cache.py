"""DCC submitter: build the v14 2STFT front-end spec cache, one SLURM-array task
per pretrain session (#165).

Unlike the Whisper teacher cache (one GPU job looping movies internally), the
front-end |STFT| spec cache is CPU-bound and embarrassingly parallel per session,
so this submits a SLURM ARRAY on the ``common`` (CPU) partition: array task ``i``
runs ``dispatch_v14.py --cache-only --cache-session-index i`` which builds the
whole-movie |STFT| memmap for the i-th session of the resolved SSL corpus
(``_SESSIONS_BY_MODE['pretrain']`` = 13 sessions, indices 0..12). For the 2STFT
front-end each task writes BOTH band caches (``band_low`` + ``band_high``
spec_cache_dir subdirs).

``--cache-only`` drives the SAME study.run -> segmenter.apply -> dataset.prepare
path the training run uses and exits BEFORE the trainer (no GPU), so the
materialized cache is byte-identical to what the run memmap-slices back. Pass the
SAME front-end / atlas / exclusion flags as the run so the extractor uid (hence
the spec-cache key) matches; the run must point ``--spec-cache-dir`` at the same
directory.

Usage (sync + sbatch from laptop, via the dcc dispatch wrapper):

    scripts/dcc/dispatch scripts/neuroprobe/submit_build_bt_2stft_cache.py \\
        --spec-cache-dir /hpc/group/coganlab/ht203/cache_neuroai/v14_2stft_spec_cache

Single-session smoke (first-light):

    scripts/dcc/dispatch scripts/neuroprobe/submit_build_bt_2stft_cache.py \\
        --array-range 0-0 \\
        --spec-cache-dir /hpc/group/coganlab/ht203/cache_neuroai/v14_2stft_smoke

Preflight without submitting (writes the .sbatch, prints it):

    scripts/dcc/dispatch scripts/neuroprobe/submit_build_bt_2stft_cache.py \\
        --spec-cache-dir /hpc/.../v14_2stft_spec_cache --dry-run
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir or Path(
        f"reports/bt_2stft_cache_build_{datetime.now():%Y_%m_%d}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    sbatch_path = log_dir / "submit_2stft_cache.sbatch"
    sbatch_path.write_text(build_sbatch(log_dir=log_dir, args=args))

    print(f"[submit] spec_cache_dir = {args.spec_cache_dir}")
    print(f"[submit] array          = {args.array_range} (one task per session)")
    print(f"[submit] partition       = {args.cpu_partition}")
    print(f"[submit] frontend/atlas  = {args.frontend} / {args.atlas} "
          f"(exclude_single_elec={args.exclude_single_electrode_parcels})")
    print(f"[submit] sbatch          = {sbatch_path}")

    if args.dry_run:
        print("[dry-run] sbatch contents:\n" + sbatch_path.read_text())
        return

    res = subprocess.run(
        ["sbatch", str(sbatch_path)],
        check=True, text=True, capture_output=True,
    )
    print("[sbatch 2stft-cache]", res.stdout.strip())


def build_sbatch(log_dir: Path, args: argparse.Namespace) -> str:
    lines = [
        "#!/bin/bash",
        "#SBATCH -J bt_2stft_cache",
        f"#SBATCH -p {args.cpu_partition}",
        f"#SBATCH --array={args.array_range}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines.extend([
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH -t {args.time}",
        f"#SBATCH -o {log_dir.resolve()}/cache-%A_%a.out",
        f"#SBATCH -e {log_dir.resolve()}/cache-%A_%a.err",
        "",
        "set -euo pipefail",
        f"cd {args.repo_root}",
        # dispatch_v14 -> neuroprobe.config hard-requires ROOT_DIR_BRAINTREEBANK.
        # Export it in the script so the job is hermetic regardless of the
        # submitting shell (a bare non-interactive SSH submission won't have it,
        # and sbatch --export=ALL would then drop it).
        f"export ROOT_DIR_BRAINTREEBANK={shlex.quote(str(args.root_dir_braintreebank))}",
    ])
    if args.extractor_cache_folder is not None:
        # The RAW-waveform exca cache (electrode_tokens) is keyed by the extractor
        # uid ONLY — it is BLIND to the bad-electrode set / loader code version. A
        # static-exclusion change (anatomy._BT_V14_EXTRA_BAD_ELECTRODES) does NOT
        # invalidate it, so a spec rebuild that reuses a stale electrode_tokens dir
        # silently re-STFTs pre-drop waveforms (the 2026-06-15 no-op-rebuild bug).
        # Pointing this at a FRESH dir forces raw _get_data through bt_load_raw so
        # the static drop actually lands. MUST be fresh whenever the bad-electrode
        # set changed; the run then reads BOTH this and --spec-cache-dir.
        lines.append(
            f"export EXCA_EXTRACTOR_CACHE_FOLDER="
            f"{shlex.quote(str(args.extractor_cache_folder))}"
        )
    if args.pythonpath:
        # Isolated-worktree runs reuse the MAIN clone's .venv binary but must
        # import the worktree's src, so PYTHONPATH wins over the editable install.
        lines.append(f"export PYTHONPATH={shlex.quote(str(args.pythonpath))}")
    lines.append("")
    # --cache-only returns 0 before the trainer; --num-workers 0 keeps the build
    # single-process (dataset.prepare() runs in the main process regardless).
    # The front-end / atlas / exclusion flags MUST match the run so the extractor
    # uid (and therefore the spec-cache key) is identical.
    cmd = [
        f"{shlex.quote(args.python)} -m speech_decoding.experiments.dispatch_v14 \\",
        "    --phase 1 --mode full \\",
        f"    --frontend {shlex.quote(args.frontend)} --pool mean \\",
    ]
    if args.mean_pool_std:
        cmd.append("    --mean-pool-std \\")
    cmd.append(f"    --atlas {shlex.quote(args.atlas)} \\")
    if args.exclude_single_electrode_parcels:
        cmd.append("    --exclude-single-electrode-parcels \\")
    cmd.extend([
        f"    --clip-len {args.clip_len} \\",
        f"    --spec-cache-dir {shlex.quote(str(args.spec_cache_dir))} \\",
        "    --num-workers 0 \\",
        "    --cache-only --cache-session-index ${SLURM_ARRAY_TASK_ID}",
    ])
    lines.extend(cmd)
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--spec-cache-dir", type=Path, required=True,
        help="Spec-cache root (persistent /hpc tier). The training run MUST pass "
             "the SAME --spec-cache-dir so it reads these caches back.",
    )
    p.add_argument(
        "--array-range", default="0-12",
        help="SLURM array range = session indices in _SESSIONS_BY_MODE['pretrain'] "
             "(13 sessions → 0-12). Use 0-0 for a single-session smoke.",
    )
    p.add_argument(
        "--log-dir", type=Path, default=None,
        help="Where the .sbatch + slurm logs land (default reports/bt_2stft_cache_build_<date>).",
    )
    p.add_argument(
        "--extractor-cache-folder", type=Path, default=None,
        help="Exported as EXCA_EXTRACTOR_CACHE_FOLDER — the RAW-waveform exca cache "
             "(electrode_tokens). It is keyed by the extractor uid ONLY and is BLIND "
             "to the bad-electrode set, so it MUST point at a FRESH dir whenever the "
             "static exclusion changed (else the spec rebuild reuses stale pre-drop "
             "waveforms — the 2026-06-15 no-op-rebuild bug). The run reads both this "
             "and --spec-cache-dir. Default None = inherit the env (only safe when the "
             "electrode set is unchanged).",
    )
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument(
        "--root-dir-braintreebank", default="/work/ht203/data/braintreebank",
        help="Exported as ROOT_DIR_BRAINTREEBANK in the job (neuroprobe.config requires it).",
    )
    p.add_argument(
        "--python", default=".venv/bin/python",
        help="Interpreter for the build (relative to repo-root, or absolute).",
    )
    p.add_argument(
        "--pythonpath", default=None,
        help="If set, exported as PYTHONPATH so a worktree's src shadows the "
             "editable install (e.g. /work/ht203/repo/speech_bt/src).",
    )
    # Front-end / atlas / exclusion — MUST match the training run's flags.
    p.add_argument("--frontend", default="2stft", choices=("raw", "2stft"))
    p.add_argument("--atlas", default="dkt", choices=("dk", "dkt"))
    p.add_argument("--mean-pool-std", dest="mean_pool_std", action="store_true",
                   default=True)
    p.add_argument("--no-mean-pool-std", dest="mean_pool_std", action="store_false")
    p.add_argument("--exclude-single-electrode-parcels",
                   dest="exclude_single_electrode_parcels", action="store_true",
                   default=True)
    p.add_argument("--no-exclude-single-electrode-parcels",
                   dest="exclude_single_electrode_parcels", action="store_false")
    p.add_argument("--clip-len", type=float, default=5.0)
    # SLURM resources (CPU node; no GPU).
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpu-partition", default="common",
                   help="CPU partition for the array (no coganlab CPU partition exists).")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="64G")
    p.add_argument("--time", default="02:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
