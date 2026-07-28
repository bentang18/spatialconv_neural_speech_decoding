#!/bin/bash
#SBATCH --job-name=viz_reduce
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=/projects/bhqk/htang13/viz/reduce_%A_%a.out
set -euo pipefail
CACHE="${1:?need CACHE}"
OUT="${2:?need OUT}"
TAPS="${3:-enc0,enc3,enc6,enc12}"
# 4th arg: per-band frequency bins, for shards written before band_fdims was in the payload
# (three of the 2 s shards predate the scp). 7,6,7 is READ off a shard the same frontend
# wrote -- it is NOT back-solvable from the enc0 width.
FDIMS="${4:-}"
# 5th arg: task subset. The default six are what the lobe figures need. The per-electrode
# reduction narrows this on purpose -- its rows are contacts, not parcels, so each stored
# array is ~6x larger and only one task is needed to paint a brain.
TASKS="${5:-}"
module load pytorch-conda/2.8
numactl --interleave=all python -u /projects/bhqk/htang13/viz/viz_reduce.py \
  --cache "$CACHE" --out-dir "$OUT" --session-index "$SLURM_ARRAY_TASK_ID" \
  --taps "$TAPS" --band hga --chunk 256 \
  ${FDIMS:+--band-fdims "$FDIMS"} \
  ${TASKS:+--tasks "$TASKS"}
