"""DCC submitter: GUARD-2 (bad-window) precompute for the Cogan D-cohort — one SLURM
-array task per manifest run. Cogan sibling of ``submit_precompute_guard1_static_cogan.py``.

Unlike guard-1 (which reads RAW voltage), guard-2 for Cogan reads the already-baked
3-band 32 Hz spec cache: array task ``i`` runs
``precompute_bad_windows_cogan.py --manifest <csv> --spec-cache-dir <dir>`` (via
``SLURM_ARRAY_TASK_ID``), sliding the shared hot/cat/abs/flat detector over the i-th
run's cached |STFT| magnitudes on a 1 s grid and merging bad windows to spans.

Because it mmap-reads the cache (no full-montage float64, no STFT re-derive), it is
cheap: ~2 CPUs / a few GB / a couple minutes per run. It therefore runs AFTER the
3-band cache bake (``--cache-only --cache-band {v3slow,v3mid,hga}``) has populated
``<spec-cache-dir>/band_{v3slow,v3mid,hga}`` for the corpus.

Reads only the synced code + the baked spec cache + the run manifest (for session
enumeration) — all on ``/work``, never ``/hpc``. Uses the probe_bench venv by absolute
path (the repo/speech clone has no ``.venv`` of its own).

Output dir holds tiny ``cogan{sid}_t{tid}.json`` bad-window spans; the v3 loader reads
them via ``cache_index.index_bad_windows`` so the clip sampler avoids bad spans.

Usage (run ON DCC after ``git pull`` in the clone, once the cache is baked):

    /work/ht203/probe_bench/.venv/bin/python \\
      scripts/neuroprobe/submit_precompute_bad_windows_cogan.py \\
        --spec-cache-dir /work/ht203/cogan_v3/spec_cache \\
        --out-dir /work/ht203/cogan_v3/guard2/bad_windows

Single-run smoke (first-light, one task):

    ... submit_precompute_bad_windows_cogan.py --array-range 0-0 \\
        --spec-cache-dir /work/ht203/cogan_v3/spec_cache \\
        --out-dir /work/ht203/cogan_v3/guard2/bad_windows_smoke

Preflight without submitting (writes + prints the .sbatch):

    ... submit_precompute_bad_windows_cogan.py --dry-run \\
        --spec-cache-dir /work/ht203/cogan_v3/spec_cache \\
        --out-dir /work/ht203/cogan_v3/guard2/bad_windows
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

# 938 runs in cogan_manifest.csv (build_cogan_manifest.py, DCC-verified 2026-07-13)
# -> array indices 0-937.
_DEFAULT_ARRAY = "0-937"
_DEFAULT_MANIFEST = "/work/ht203/cogan_v3/manifest/cogan_manifest.work.csv"


def build_sbatch(log_dir: Path, args: argparse.Namespace) -> str:
    lines = [
        "#!/bin/bash",
        "#SBATCH -J cogan_guard2_badwin",
        f"#SBATCH -p {args.cpu_partition}",
        f"#SBATCH --array={args.array_range}",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    if args.dependency:
        lines.append(f"#SBATCH --dependency={args.dependency}")
    lines.extend([
        "#SBATCH --requeue",  # scavenger is preemptible -> requeue on preemption
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH -t {args.time}",
        f"#SBATCH -o {log_dir.resolve()}/g2c-%A_%a.out",
        f"#SBATCH -e {log_dir.resolve()}/g2c-%A_%a.err",
        "",
        "set -euo pipefail",
        f"cd {args.repo_root}",
        "",
        # scripts/ is NOT a package -> run by PATH, not -m. The slider reads
        # SLURM_ARRAY_TASK_ID to pick its manifest row (session enumeration).
        f"{shlex.quote(args.python)} -u scripts/neuroprobe/precompute_bad_windows_cogan.py \\",
        f"    --manifest {shlex.quote(str(args.manifest))} \\",
        f"    --spec-cache-dir {shlex.quote(str(args.spec_cache_dir))} \\",
        f"    --detect-window {args.detect_window} \\",
        f"    --out-dir {shlex.quote(str(args.out_dir))}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir or Path(
        f"reports/guard2_badwin_cogan_{datetime.now():%Y_%m_%d}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    sbatch_path = log_dir / "submit_guard2_badwin_cogan.sbatch"
    sbatch_path.write_text(build_sbatch(log_dir=log_dir, args=args))

    print(f"[submit] manifest       = {args.manifest}")
    print(f"[submit] spec_cache_dir = {args.spec_cache_dir}")
    print(f"[submit] out_dir        = {args.out_dir}")
    print(f"[submit] detect_window  = {args.detect_window}s")
    print(f"[submit] array          = {args.array_range} (one task per manifest run)")
    print(f"[submit] partition      = {args.cpu_partition}")
    print(f"[submit] sbatch         = {sbatch_path}")

    if args.dry_run:
        print("[dry-run] sbatch contents:\n" + sbatch_path.read_text())
        return

    res = subprocess.run(
        ["sbatch", str(sbatch_path)],
        check=True, text=True, capture_output=True,
    )
    print("[sbatch cogan-guard2-badwin]", res.stdout.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=_DEFAULT_MANIFEST,
                   help="cogan_manifest.csv (row order = array order; used for "
                        "session enumeration only, NOT for paths — the cache is read).")
    p.add_argument("--spec-cache-dir", required=True,
                   help="Baked 3-band cache root holding band_{v3slow,v3mid,hga}.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Bad-window output dir. One cogan{sid}_t{tid}.json per task.")
    p.add_argument("--detect-window", type=float, default=1.0,
                   help="Detection tiling window in seconds (default 1.0 — matches BT).")
    p.add_argument("--array-range", default=_DEFAULT_ARRAY,
                   help="SLURM array range = manifest row indices (938 runs -> 0-937). "
                        "Use 0-0 for a single-run smoke; append %%N to throttle concurrency.")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Where the .sbatch + slurm logs land "
                        "(default reports/guard2_badwin_cogan_<date>).")
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument("--python", default="/work/ht203/repo/speech/.venv/bin/python",
                   help="Interpreter. Imports cache_index, which pulls in "
                        "speech_decoding.models.__init__ -> neuraltrain (the probe_bench "
                        "venv LACKS it). The clone's own .venv has the full stack.")
    p.add_argument("--dependency", default=None,
                   help="SLURM --dependency string (e.g. afterok:<band1>:<band2>:<band3>) "
                        "for the overnight chain — run guard-2 only after the cache bake.")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpu-partition", default="common,scavenger",
                   help="CPU partitions for the array (SLURM picks whichever has room).")
    p.add_argument("--cpus", type=int, default=2)
    # mmap-reads 3 band .npy + small per-frame/z-std arrays; even the longest Cogan
    # runs sit well under 32G (no full-montage float64, no STFT re-derive).
    p.add_argument("--mem", default="32G")
    p.add_argument("--time", default="00:30:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
