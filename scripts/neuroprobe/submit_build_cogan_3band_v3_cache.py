#!/usr/bin/env python
"""Submit the Cogan D-cohort v14_converged_v3 3-band (SLOW+MID+HGA) magnitude
spec-cache build to DCC — MASSIVELY PARALLEL across common+scavenger (cache-only,
CPU). Cogan sibling of ``submit_build_bt_3band_v3_cache.py``.

Mechanism (byte-identical front-end to BT's v3 bake, drift-guarded by
``test_dispatch_cogan_cache_seam``): each of the 3 v3 bands rides the single-grid
``electrode_tokens`` slot via ``dispatch_v14 --study cogan --cache-band
{v3slow,v3mid,hga}`` (``--cache-only`` exits before the trainer, so no v3 model is
needed). The band view is constructed exactly as the v3 run reads it (STFT_{V3_SLOW/
V3_MID/2BAND_HGA} + the shared common_fe_kwargs + ``band_<name>`` spec-cache subdir),
so a future v3 ``--session`` run HITs this cache. The ONLY difference vs the BT bake
is the corpus source: a :class:`DCohortStudy` (938 localized runs) instead of BT's
13 SSL sessions.

  SLOW  N=1024 hop=64  2-14 Hz  |STFT| -> 7 bins   [cache-band v3slow]
  MID   N=256  hop=64  16-56 Hz |STFT| -> 6 bins   [cache-band v3mid]
  HGA   N=128  hop=64  64-160Hz |STFT| -> 7 bins   [cache-band hga]

Voltage->feature preprocessing is BYTE-IDENTICAL to the BT v3 bake (shaft-CAR ->
notch 60/120/.. -> 0.5 Hz HPF -> STFT -> |mag| -> per-(elec,bin,session) robust-z);
only the corpus differs. The guard-1 STATIC contact drop bakes in via
``COGAN_GUARD1_STATIC`` (the collector's map): the drop enters the exca cache uid
(``DCohortStudy.iter_timelines`` folds ``cogan_extra_bad`` into each timeline), so
the memmap depends on the drop and ``cogan_load_raw`` drops exactly those contacts
PRE-CAR. If the map is absent the bake still runs with NO static drop (warn-once) --
so the array is safe to launch before the map lands, but for the REAL bake wire it
AFTER the guard-1 collector (``--dependency afterok:<collector>``, map exported).

PARALLELISM: one independent CPU array task per (band x session) -> 3 x 938 = 2814
tasks. Default partition ``common,scavenger`` so SLURM lands each task wherever a
node is free; ``--requeue`` re-runs any preempted task (each task independent +
idempotent). One sbatch is submitted PER BAND (3 arrays of 0-937).

READS ONLY ``/work`` (synced code + staged EDFs + the run manifest) -- never
``/hpc``. WRITES ONLY under ``--spec-cache-dir`` / ``--extractor-cache-folder`` on
``/work``. Uses the probe_bench venv by absolute path (the repo/speech clone has no
``.venv``).

Usage (run ON DCC after ``git pull`` in the clone):

    /work/ht203/probe_bench/.venv/bin/python \\
      scripts/neuroprobe/submit_build_cogan_3band_v3_cache.py \\
        --spec-cache-dir /work/ht203/cogan_v3/spec_cache \\
        --extractor-cache-folder /work/ht203/cogan_v3/extractors_3band_v3 \\
        --guard1-static /work/ht203/cogan_v3/guard1/cogan_guard1_static.json

Single-band single-session smoke first (validates the seam end-to-end on DCC):

    ... --bands hga --array-range 0-0 \\
        --spec-cache-dir /work/ht203/cogan_v3/spec_cache_smoke \\
        --extractor-cache-folder /work/ht203/cogan_v3/extractors_smoke

Overnight chain (afterok the guard-1 collector job):

    ... --dependency afterok:<collector_jobid>   # prints the 3 band array job IDs
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

# 938 runs in cogan_manifest.csv (build_cogan_manifest.py, DCC-verified 2026-07-13)
# -> array indices 0-937. Same row order as the guard-1/guard-2 arrays (all three
# drive off manifest row order via --cache-session-index / SLURM_ARRAY_TASK_ID).
_DEFAULT_ARRAY = "0-937"
_DEFAULT_MANIFEST = "/work/ht203/cogan_v3/manifest/cogan_manifest.work.csv"
# v3 3-band set (uniform hop=64). Mirrors _COGAN_CACHE_BANDS in dispatch_v14 (itself
# drift-guarded against BT's band_const map by test_dispatch_cogan_cache_seam).
_BANDS = ("v3slow", "v3mid", "hga")


def build_sbatch(log_dir: Path, band: str, args: argparse.Namespace) -> str:
    array_range = args.array_range or _DEFAULT_ARRAY
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J cogan3band_{band}",
        f"#SBATCH -p {args.cpu_partition}",
        f"#SBATCH --array={array_range}",
        "#SBATCH --requeue",  # scavenger preemption -> re-run the (band,session) task
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    if args.dependency:
        lines.append(f"#SBATCH --dependency={args.dependency}")
    lines.extend([
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH -t {args.time}",
        f"#SBATCH -o {log_dir.resolve()}/cogan-cache-{band}-%A_%a.out",
        f"#SBATCH -e {log_dir.resolve()}/cogan-cache-{band}-%A_%a.err",
        "",
        "set -euo pipefail",
        f"cd {args.repo_root}",
        # Fresh, Cogan-specific extractor cache so the STATIC drop lands (not a stale
        # pre-drop reuse). NO ROOT_DIR_BRAINTREEBANK -- the --study cogan path never
        # calls bt_load_raw (dispatch_v14:1212 reads it only inside that BT loader).
        f"export EXCA_EXTRACTOR_CACHE_FOLDER={shlex.quote(str(args.extractor_cache_folder))}",
    ])
    if args.guard1_static:
        # Guard-1 static-bad map (COGAN_GUARD1_STATIC). Folded into each timeline's
        # extra_bad by DCohortStudy.iter_timelines -> enters the cache uid -> baked
        # pre-CAR. Absent/unset -> empty map -> no static drop (warn-once).
        lines.append(
            f"export COGAN_GUARD1_STATIC={shlex.quote(str(args.guard1_static))}"
        )
    if args.pythonpath:
        lines.append(f"export PYTHONPATH={shlex.quote(str(args.pythonpath))}")
    lines.append("")

    py = shlex.quote(args.python)
    lines.extend([
        # scripts/ is NOT a package; dispatch is a module. --study cogan routes to
        # build_cogan_cache_experiment; --cache-session-index picks the manifest row.
        f"{py} -m speech_decoding.experiments.dispatch_v14 \\",
        "    --phase 1 --mode full --electrode-set all --pool mean \\",
        "    --study cogan \\",
        f"    --cogan-manifest {shlex.quote(str(args.manifest))} \\",
        f"    --cogan-notch-hz {args.notch_hz} \\",
        f"    --atlas {shlex.quote(args.atlas)} \\",
        f"    --clip-len {args.clip_len} \\",
        f"    --spec-cache-dir {shlex.quote(str(args.spec_cache_dir))} \\",
        "    --num-workers 0 \\",
        f"    --cache-only --cache-band {band} \\",
        "    --cache-session-index ${SLURM_ARRAY_TASK_ID}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir or Path(
        f"reports/cogan_3band_v3_cache_build_{datetime.now():%Y_%m_%d}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    bad = [b for b in bands if b not in _BANDS]
    if bad:
        raise SystemExit(f"unknown band(s) {bad}; choose from {_BANDS}")

    print(f"[submit] manifest       = {args.manifest}")
    print(f"[submit] spec_cache_dir = {args.spec_cache_dir}")
    print(f"[submit] extractor_dir  = {args.extractor_cache_folder}")
    print(f"[submit] guard1_static  = {args.guard1_static or '(unset -> no static drop)'}")
    print(f"[submit] bands          = {bands}")
    print(f"[submit] array          = {args.array_range or _DEFAULT_ARRAY}")
    print(f"[submit] dependency     = {args.dependency or '(none)'}")

    job_ids = []
    for band in bands:
        sbatch_text = build_sbatch(log_dir, band, args)
        sbatch_path = log_dir / f"cogan_cache_{band}.sbatch"
        sbatch_path.write_text(sbatch_text)
        if args.dry_run:
            print(f"\n===== {sbatch_path} =====")
            print(sbatch_text)
            continue
        res = subprocess.run(
            ["sbatch", str(sbatch_path)],
            capture_output=True, text=True, check=True,
        )
        out = res.stdout.strip()
        print(f"[sbatch cogan-3band {band}] {out}")
        # "Submitted batch job 12345" -> 12345 (for afterok wiring of guard-2)
        job_ids.append(out.split()[-1])

    if args.dry_run:
        print("\n(dry-run: no jobs submitted)")
    else:
        print(f"\nband array job ids: {':'.join(job_ids)}")
        print("wire guard-2 with:  submit_precompute_bad_windows_cogan.py "
              f"--dependency afterok:{':'.join(job_ids)} ...")
        print(f"logs + sbatch: {log_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec-cache-dir", type=Path, required=True,
                   help="Spec-cache root. Each band lands in <root>/band_<name> "
                        "(band_v3slow, band_v3mid, band_hga). The v3 run MUST pass the "
                        "SAME --spec-cache-dir to read these. FRESH, Cogan-specific dir.")
    p.add_argument("--extractor-cache-folder", type=Path, required=True,
                   help="EXCA_EXTRACTOR_CACHE_FOLDER (electrode_tokens raw-waveform "
                        "cache). FRESH so the STATIC drop lands (not a stale pre-drop reuse).")
    p.add_argument("--guard1-static", type=Path, default=None,
                   help="Guard-1 static-bad map (COGAN_GUARD1_STATIC). The collector "
                        "output cogan_guard1_static.json. Unset -> NO static drop "
                        "(warn-once); set it for the real bake (afterok the collector).")
    p.add_argument("--manifest", default=_DEFAULT_MANIFEST,
                   help="cogan_manifest.work.csv (row order = array order; the study "
                        "reads edf/channels paths from it). MUST be the /work-repointed copy.")
    p.add_argument("--bands", default="v3slow,v3mid,hga",
                   help="Comma list of v3 bands to build (default all three).")
    p.add_argument("--array-range", default=None,
                   help="Override the SLURM array range (default 0-937 = 938 runs). "
                        "Use 0-0 for a single-session smoke.")
    p.add_argument("--dependency", default=None,
                   help="SLURM --dependency string (e.g. afterok:<collector_jobid>) "
                        "for the overnight chain. Applied to every band array.")
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--repo-root", type=Path, default=Path("/work/ht203/repo/speech"))
    p.add_argument("--python", default=".venv/bin/python",
                   help="Interpreter for the bake. MUST have the full training stack "
                        "(neuraltrain + neuralset) -- dispatch_v14 imports neuraltrain, "
                        "which the probe_bench venv LACKS. Default .venv/bin/python is the "
                        "clone's own venv (sbatch cd's to --repo-root first), matching the "
                        "BT v3 bake.")
    p.add_argument("--pythonpath", default=None)
    p.add_argument("--notch-hz", type=float, default=60.0,
                   help="Mains notch (Duke = 60 Hz, same as BT). Passed as --cogan-notch-hz.")
    p.add_argument("--atlas", default="dkt", choices=("dk", "dkt"),
                   help="Cache-irrelevant to the per-band spec memmap; default dkt.")
    p.add_argument("--clip-len", type=float, default=4.0,
                   help="Cache-irrelevant (whole-movie spec memmap is clip-independent); "
                        "kept for build parity (v3 = 4 s clips).")
    p.add_argument("--account", default="coganlab")
    p.add_argument("--cpu-partition", default="common,scavenger",
                   help="Comma list -> SLURM lands each task wherever a node is free.")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="192G",
                   help="Per-task memory (matches BT v3 bake: uniform hop=64 slow/mid "
                        "memmaps are large; the longest Cogan runs need headroom).")
    p.add_argument("--time", default="04:00:00")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
