#!/bin/bash
#SBATCH --job-name=v3_samplecurve
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=10:00:00
#SBATCH --array=0-11
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%A_%a.out
#
#   sbatch scripts/neuroprobe/v3_board_samplecurve.sbatch
#   # then, after the array drains -- --array=0 OVERRIDES the header, or the merge runs 12x and
#   # twelve processes race to write the same JSON:
#   sbatch --array=0 --mem=16G --time=00:30:00 --dependency=afterany:<ARRAYID> \
#          scripts/neuroprobe/v3_board_samplecurve.sbatch merge
#
# R31 — LABEL-EFFICIENCY CURVE off the LIVE board cache. ZERO GPU: the encode already wrote both
# taps (v3_board_encode_r6.sbatch:20-28 writes enc0_elec AND enc12_elec), so this is a refit sweep
# on frozen features we have already paid for. It must not contend with the pbs50 geometry
# ablations on dtai — hence Delta CPU, a different machine and a different account.
#
# ═══ WHY 12 ARRAY TASKS ══════════════════════════════════════════════════════════════════════════
# One per WS session (LITE_SESSIONS), the same sharding the board readout uses. Full --array up
# front, never a serial loop: the 12 cells are independent and a serial pass would take 12x the
# wall for the same core-hours.
#
# ═══ MEMORY IS THE BILL, NOT CORES ══════════════════════════════════════════════════════════════
# Delta CPU bills (mem_MB/2) x wall, so cores below 0.512 x mem_GB are FREE: at 150G that is 76,
# and 32 sits well under it. 150G (not the board readout's 120G) because the curve holds ONE extra
# array the plain readout does not — the standardized val SUBSET — on top of train+val+test. Peak
# is at the N=full point and is ~27 GB for enc12_elec single-worker; the margin is for the fp32
# gather transients, not guesswork.
#
# WORKERS=1 DELIBERATELY. _map_tasks forks per task and each child holds its own fp32 slices; at
# enc12_elec (d=1.58e6) three workers is ~66 GB of private design matrices before the standardize
# temp. This script's per-task cost is already ~3-4x a plain readout, so the safe choice is one
# task at a time with all 32 cores going to BLAS.
#
# ═══ WALLTIME IS A BACKFILL KNOB ════════════════════════════════════════════════════════════════
# A plain eager WS shard runs ~85 min (20298989_0, 07-27). The sweep's eval kernels dominate and
# the tapered seed schedule lands it at ~3-4x ⇒ ~5 h. 10 h is ~2x that, chosen so a slow node
# cannot silently truncate the array, not as padding — an OOM'd or timed-out job bills IN FULL.
set -uo pipefail

MODE="${1:-shard}"

module load python/miniforge3_pytorch/2.10.0
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE:$TREE/src:${PYTHONPATH:-}"
cd "$TREE"

CACHE=/projects/bhqk/htang13/v3_board_cache_board_pbs50_cd45k
TAG=pbs50_cd45k
SHARDS=/projects/bhqk/htang13/samplecurve_shards_${TAG}
OUT=/projects/bhqk/htang13/samplecurve_${TAG}.json

# Give BLAS every core: workers=1, so nothing else is competing for them.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

mkdir -p "$SHARDS"

if [[ "$MODE" == "merge" ]]; then
  echo "[R31 merge] shards=$SHARDS"
  "$VENV" -u scripts/neuroprobe/v3_board_samplecurve.py \
    --mode merge --shard-dir "$SHARDS" --out "$OUT"
  exit $?
fi

IDX="${SLURM_ARRAY_TASK_ID:?this script must run as an array (--array=0-11)}"
echo "[R31 shard] idx=$IDX cache=$CACHE tag=$TAG"
echo "[contract]  WS only | taps enc0_elec,enc12_elec | RANDOM stratified draw | test NEVER subsampled"
echo "[anchor]    N=full MUST reproduce the published board cell — read the [anchor] line below"
"$VENV" -u scripts/neuroprobe/v3_board_samplecurve.py \
  --cache-dir "$CACHE" --tag "$TAG" --index "$IDX" --shard-dir "$SHARDS"
