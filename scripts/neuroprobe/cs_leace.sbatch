#!/bin/bash
#SBATCH --job-name=cs_leace
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/projects/bhqk/htang13/cs_leace_%A_%a.out
# CS identity-erasure arm: std (leaderboard baseline) vs std_target (AdaBN) vs leace, per cell.
# One array task per CS test cell -- ALL TEN, never a subset: partial cells lie (.6279 at 4 cells
# became .5991 at 10).
#
#   usage: sbatch --array=0-9 --mem=<G> cs_leace.sbatch <CACHE> <TAG> <OUTDIR> [TAPS] [COMPONENTS]
#
# MEMORY: 64G OOM-KILLED ALL TEN CELLS (job 20540001, 07-27) and the cause was NOT the eraser --
# it was the LOADER. Session caches are 40-64 GB each and this driver opens two, so the eager path
# needs 87-111 GB before any feature is gathered. Fixed by LOAD_MMAP in v3_cs_leace.py; the elec
# taps are then never paged because the CS arms never index them.
# What remains is the eraser's dense fp64 SVD over the stacked anchor+test rows: enc12 is
# d=93184, so each (n, d) fp64 array is ~2.6 GB at n=3500 and the fit holds several at once
# (xc, vt, basis, plus LAPACK workspace and pinv's own SVD in the identity check).
# 128G leaves real headroom on that; do NOT shave it back toward 64G -- that number is disproven.
#
# COMPONENTS truncates the eraser's row-space basis. Leave it EMPTY (exact erasure) unless the SVD
# will not fit -- truncation makes erasure approximate and shows up as a nonzero residual_cov in
# the per-cell [check] line. Read that line before trusting any null result: unchanged CS with
# identity still decodable means the erasure did not transfer, not that identity was inert.
set -euo pipefail
CACHE="${1:?CACHE_DIR}"; TAG="${2:?TAG}"; OUTDIR="${3:?OUT_DIR}"
TAPS="${4:-enc0,enc12}"; COMPONENTS="${5:-}"
# Deployed to its OWN tree, not the live board tree: idprobe2 and the board arrays import
# v3_board_readout, and this ships the post-R8 copy the parity test was run against.
cd /projects/bhqk/htang13/speech_leace
module load pytorch-conda/2.8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# BILLING: MAX_TRES with Mem=512G on a 128-core/257617M node ⇒ billing == mem_MB/2, so cores up
# to ~0.512·mem_GB are already paid for and cores beyond it flip the bill to the CPU term.
BE=$(( ${SLURM_MEM_PER_NODE:-0} / 2000 ))
echo "[check] alloc cpus=${SLURM_CPUS_PER_TASK:-8} mem=${SLURM_MEM_PER_NODE:-?}M; break-even cpus=$BE"
if [ "$BE" -gt 0 ] && [ "${SLURM_CPUS_PER_TASK:-8}" -lt "$BE" ]; then
  echo "[check] UNDER-REQUESTED: --cpus-per-task=$BE bills the SAME as ${SLURM_CPUS_PER_TASK:-8}"
fi

# Every Lite session whose subject is not the anchor's (S2), ordered as in v3_board_readout.
CELLS=(1,1 1,2 3,0 3,1 4,0 4,1 7,0 7,1 10,0 10,1)
CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"
mkdir -p "$OUTDIR"

EXTRA=()
if [[ -n "$COMPONENTS" ]]; then EXTRA=(--leace-components "$COMPONENTS"); fi

numactl --interleave=all python -u scripts/neuroprobe/v3_cs_leace.py \
  --cache "$CACHE" --tag "$TAG" --cell "$CELL" --taps "$TAPS" \
  --out "$OUTDIR/leace_S${CELL%,*}T${CELL#*,}.json" "${EXTRA[@]}"
