#!/bin/bash
#SBATCH --job-name=r6_probe_readout
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%A_%a.out
# r6 pretrain-probe readout, Delta CPU. STANDARD grid: std-only (PROBE_NORMS default),
# enc0/3/6/12 (r6's 3-band |STFT| frontend HAS enc0, unlike r5). Sharded: --array=0-12
# (7 WS PROBE_COHORT_7 + 6 CS_TEST_SUBJECTS) then a dependent merge.
#   array:  sbatch --array=0-12 --mem=128G v3_probe_readout_r6.sbatch array <CACHE> <TAG> <SHARD>
#   merge:  sbatch --dependency=afterany:<AID> --mem=32G v3_probe_readout_r6.sbatch merge <CACHE> <TAG> <SHARD> <OUT>
# 64 cores (not r5's 8): the ridge is BLAS-bound and this matches the proven r4 readout. mem is
# passed at launch (array 128G / merge 32G) not baked, because a CS shard loads TWO sessions AND
# r6 carries the big enc0 |STFT| tap (r5 omitted enc0) -- 48G OOM-killed CS shards (job 20456873).
set -euo pipefail
MODE="${1:?array|merge}"; CACHE="${2:?CACHE_DIR}"; TAG="${3:?TAG}"; SHARD="${4:?SHARD_DIR}"
cd /projects/bhqk/htang13/speech
module load pytorch-conda/2.8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PROBE_NORMS=std
TAPS="${TAPS:-enc0,enc3,enc6,enc12}"
if [ "$MODE" = "array" ]; then
  numactl --interleave=all python -u scripts/neuroprobe/v3_probe_readout_r4.py \
    --cache-dir "$CACHE" --tags "$TAG" --mode array \
    --array-index "$SLURM_ARRAY_TASK_ID" --shard-dir "$SHARD" --taps "$TAPS"
else
  OUT="${5:?OUT_JSON}"
  numactl --interleave=all python -u scripts/neuroprobe/v3_probe_readout_r4.py \
    --cache-dir "$CACHE" --tags "$TAG" --mode merge \
    --shard-dir "$SHARD" --taps "$TAPS" --out "$OUT"
fi
