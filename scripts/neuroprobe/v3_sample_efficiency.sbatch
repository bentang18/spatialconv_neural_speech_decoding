#!/bin/bash
#SBATCH --job-name=v3_sampeff
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/v3_sampeff_%A_%a.out
# Downstream SAMPLE EFFICIENCY: CS AUROC vs number of labelled train rows, enc0 vs enc12.
# One array task per CS test cell -- ALL TEN, never a subset. Partial cells lie: the decoder
# ablation read .6279 at 4 cells and .5991 at 10.
#
#   usage: sbatch --array=0-9 --mem=<G> v3_sample_efficiency.sbatch <CACHE> <TAG> <OUTDIR> [TAPS]
#
# MEMORY: this opens TWO session caches (anchor + test), each 40-64 GB, so it inherits the board
# readout's loader profile, NOT the eraser's. mmap keeps the unused elec taps off the heap. 128G
# is the number the r6 board CS shards use; 48G OOM-killed them. Do not shave.
#
# WALLTIME: the ridge is DUAL, so cost scales with n_train^2 -- the small fractions are nearly
# free and the total is only ~2-3x a single board shard, dominated by frac=1.0 and 0.5. 12h is
# slack, not a measurement; short jobs earn backfill, so trim once one cell has a real timing.
set -euo pipefail
CACHE="${1:?CACHE_DIR}"; TAG="${2:?TAG}"; OUTDIR="${3:?OUT_DIR}"; TAPS="${4:-enc0,enc12}"

cd /projects/bhqk/htang13/speech
module load pytorch-conda/2.8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# BILLING: Delta CPU bills MAX_TRES (CPU=1000, Mem=512G) on a 128-core/257617M node, so
# billing == mem_MB/2 and cores up to ~0.512*mem_GB are ALREADY PAID FOR. Cores beyond that flip
# the bill to the CPU term and genuinely cost more. 32 @ 128G sits under the 65-core break-even
# and stays small enough to backfill onto a partly-used node.
CPT="${SLURM_CPUS_PER_TASK:-8}"
BE=$(( ${SLURM_MEM_PER_NODE:-0} / 2000 ))
echo "[check] alloc cpus=$CPT mem=${SLURM_MEM_PER_NODE:-?}M omp=$OMP_NUM_THREADS; break-even cpus=$BE"
if [ "$BE" -gt 0 ] && [ "$CPT" -lt "$BE" ]; then
  echo "[check] UNDER-REQUESTED: --cpus-per-task=$BE bills the SAME as $CPT (billing is memory-bound)"
elif [ "$BE" -gt 0 ] && [ "$CPT" -gt "$BE" ]; then
  echo "[check] OVER-REQUESTED: $CPT cpus bills $(( CPT * 1000 )) vs $(( BE * 1000 )) at break-even -- this COSTS more"
fi

# Every Lite session whose subject is not the anchor's (S2), ordered as in v3_board_readout.
CELLS=(1,1 1,2 3,0 3,1 4,0 4,1 7,0 7,1 10,0 10,1)
CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"
mkdir -p "$OUTDIR"

numactl --interleave=all python -u scripts/neuroprobe/v3_sample_efficiency.py \
  --cache "$CACHE" --tag "$TAG" --cell "$CELL" --taps "$TAPS" \
  --out "$OUTDIR/sampeff_S${CELL%,*}T${CELL#*,}.json"
