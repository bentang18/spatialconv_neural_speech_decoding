#!/bin/bash
#SBATCH --job-name=r6_probe_readout
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%A_%a.out
# r6 pretrain-probe readout, Delta CPU. STANDARD grid: std-only (PROBE_NORMS default),
# enc0/3/6/12 (r6's 3-band |STFT| frontend HAS enc0, unlike r5). Sharded: --array=0-12
# (7 WS PROBE_COHORT_7 + 6 CS_TEST_SUBJECTS) then a dependent merge.
#   array:  sbatch --array=0-12 --mem=64G v3_probe_readout_r6.sbatch array <CACHE> <TAG> <SHARD>
#   merge:  sbatch --dependency=afterany:<AID> --mem=32G v3_probe_readout_r6.sbatch merge <CACHE> <TAG> <SHARD> <OUT>
# 32 cores (not r5's 8): the ridge is BLAS-bound, and 32 is the most that is still FREE at 64G.
# ⚠️ CORES AND MEM ARE COUPLED HERE -- Delta CPU bills MAX_TRES, so billed = max(mem_MB/2,
# cores/0.512·1000/1000...) and in practice cores under 0.512 x mem_GB cost NOTHING while cores
# above it START SETTING THE BILL. Break-even is 33 cores at 64G, and 65 at 128G. This said 64
# while mem was 128G, where it was free; when mem dropped to 64G it silently became a 2x
# overbill for anyone following the usage line above (which passes --mem but NOT --cpus-per-task).
# So: NEVER change one of these without recomputing the other. 32 is also the value the 30k
# pretrain probes actually ran at, where elements finished in ~90 s -- it is proven, not merely
# affordable. mem is passed at launch (array 64G / merge 32G) not baked, because a CS shard loads
# TWO sessions AND r6 carries the big enc0 |STFT| tap (r5 omitted enc0) -- 48G OOM-killed CS
# shards (job 20456873).
# ⚠️ SIZE mem PER CACHE, and measure it on a CS index (7-12), never a WS one (0-6). WS loads one
# session, CS loads two, so a WS MaxRSS understates the array's requirement by ~2x. The 30k
# pretrain-probe cache measures 18.5-19.9G on WS (job 20580085) and runs at 64G; the board-scale
# cache needed more than 48G on CS. Same script, 2.6x apart -- so do not carry a number over.
#
# --time=00:20:00 IS FOR BACKFILL, and it is sized, not guessed. Over 834 completed elements the
# max elapsed is 8:52 (merges are ~15s), so 20 min is 2.25x the observed worst case. It was 1h,
# which is 7x the max and made every shard wait for a full-size slot instead of dropping into a
# gap -- pure latency for no benefit, since Delta bills ACTUAL wall x mem/2 and a shorter request
# costs exactly the same. Do NOT cut this to the 15 min the encode uses: the cells that OOM'd at
# 48G were killed at 8:59 on MEMORY, so at 128G they run to completion and land nearer the top of
# the range. A timeout is the expensive failure here -- 20566474 TIMED OUT 10/10 at --time=01:30
# and billed 494 h for zero output, because a killed job bills in full.
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
