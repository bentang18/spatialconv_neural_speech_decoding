#!/bin/bash
#SBATCH --job-name=v3_samplecurve_cs
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%A_%a.out
#
#   sbatch scripts/neuroprobe/v3_board_samplecurve_cs.sbatch
#   # then, after the array drains -- --array=0 OVERRIDES the header, or the merge runs 10x and
#   # ten processes race to write the same JSON:
#   sbatch --array=0 --mem=16G --time=00:30:00 --dependency=afterany:<ARRAYID> \
#          scripts/neuroprobe/v3_board_samplecurve_cs.sbatch merge
#
# R31/CS — the CROSS-SUBJECT label-efficiency curve. Same machinery as the WS curve, a DIFFERENT
# question, and the difference is the whole point of running it:
#
#   WS  N = calibration trials from the TARGET session.       "how much of this patient do I need?"
#   CS  N = labelled trials from the DONOR session S2T4.      "how much of SOMEONE ELSE do I need?"
#
# The target subject contributes ZERO training rows at every N. What is being swept is the size of
# the labelled set on the OTHER side of the subject boundary.
#
# ═══ WHY 10 ARRAY TASKS ═════════════════════════════════════════════════════════════════════════
# One per CS test cell (CS_TEST_CELLS — every Lite cell whose subject is not the anchor's), the same
# sharding the board's CS readout uses. Full --array up front, never a serial loop.
#
# ═══ THE TAPS ARE PARCEL-MEAN, NOT PER-ELECTRODE ════════════════════════════════════════════════
# enc0/enc12, not enc0_elec/enc12_elec. Electrode identity is not shared across subjects, so a CS
# curve on the electrode taps would be measuring nothing. The script picks the regime's own unit
# from CURVE_TAPS; --taps is left unset ON PURPOSE so the default cannot be inherited from the WS
# launcher by copy-paste.
#
# ═══ MEMORY: 150G, AND WHY NOT THE BOARD'S 176G ═════════════════════════════════════════════════
# board_readout_all3.sh:49 runs CS at 176G with workers=4 and MEASURED MaxRSS 107G (20548325). That
# 107G is dominated by the eager load of TWO records (anchor + test); the per-worker private cost is
# small because parcel taps are ~40x lighter than electrode taps, which is what bought workers=4
# there. This script has NO workers — _map_tasks is not used, the 15 tasks run serially — so the
# private term is ~1/4 of that budget and 150G clears the two-record load with real headroom. 150G
# also matches the WS curve launcher, keeping this one a minimal delta off a proven file.
# Cores are FREE below 0.512 x mem_GB = 76, so 32 bills identically to 8.
#
# ═══ WALLTIME ══════════════════════════════════════════════════════════════════════════════════
# 12 h, and it is NOT padding — the bill is on actual wall, so the only cost of a generous limit is
# backfill position. CS is cheaper than WS per fit in the GEMM that forms the gram (~40x lighter d)
# but MORE expensive in the eigendecomposition: CS fits the WHOLE anchor session where WS fits a
# KFold half, so n is ~2x and the n^3 term is ~8x. Those pull opposite ways and the net is not known
# in advance, so the limit is set where a timeout cannot silently truncate the array.
set -uo pipefail

MODE="${1:-shard}"

# ═══ DELTA CPU ENVIRONMENT — LIFTED VERBATIM FROM v3_board_samplecurve.sbatch:55-58 ══════════════
# Do NOT substitute the dtai incantation. `python/miniforge3_pytorch/2.10.0` is the DeltaAI
# (aarch64) module and does not exist on Delta CPU (x86), and $TREE/.venv/bin/python is a SYMLINK
# INTO that dtai module path — it resolves on dtai and dangles here. `ls -l` on a dangling symlink
# still prints a line, so check the TARGET (`python -c` or `ls -lL`), never the link.
module load pytorch-conda/2.8
TREE=/projects/bhqk/htang13/speech_board
export PYTHONPATH="$TREE:$TREE/src:${PYTHONPATH:-}"
cd "$TREE"

CACHE=/projects/bhqk/htang13/v3_board_cache_board_pbs50_cd45k
TAG=pbs50_cd45k
# SEPARATE shard dir from the WS curve. The merge globs by prefix and asserts on the shard's own
# `regime` field, so co-locating them would be safe — but a separate dir means a stray re-run of the
# WS launcher cannot touch this run's inputs at all, which is cheaper than being clever.
SHARDS=/projects/bhqk/htang13/samplecurve_cs_shards_${TAG}
OUT=/projects/bhqk/htang13/samplecurve_cs_${TAG}.json

# Give BLAS every core: no workers, so nothing else is competing for them.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

mkdir -p "$SHARDS"

if [[ "$MODE" == "merge" ]]; then
  echo "[R31/cs merge] shards=$SHARDS"
  numactl --interleave=all python -u scripts/neuroprobe/v3_board_samplecurve.py \
    --mode merge --regime cs --shard-dir "$SHARDS" --out "$OUT"
  exit $?
fi

IDX="${SLURM_ARRAY_TASK_ID:?this script must run as an array (--array=0-9)}"
echo "[R31/cs shard] idx=$IDX cache=$CACHE tag=$TAG"
echo "[contract]  CS only | taps enc0,enc12 (parcel-mean) | donor S2T4 subsampled"
echo "[contract]  target val/test NEVER subsampled — they are the protocol's, not a calibration budget"
echo "[anchor]    N=full MUST reproduce the published _cs_cell — read the [anchor] line below"
numactl --interleave=all python -u scripts/neuroprobe/v3_board_samplecurve.py \
  --mode cs --cache-dir "$CACHE" --tag "$TAG" --index "$IDX" --shard-dir "$SHARDS"
