#!/bin/bash
#SBATCH --job-name=leace_ctrl
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/projects/bhqk/htang13/leace_ctrl_%A_%a.out
# Does the LEACE null have any content? Geometry of the erased direction + two matched controls
# (rank-matched shuffled-domain, variance-matched top-PC), scored through the board's own ridge.
#
#   usage: sbatch --array=0-9 --mem=64G leace_controls.sbatch <CACHE> <TAG> <OUTDIR> [TAPS] [TASKS]
#
# ALL TEN CELLS, never a subset -- partial cells lie (.6279 at 4 cells became .5991 at 10).
#
# TASKS is the cost knob and the reason this is affordable. The GEOMETRY answer (is the erased
# direction just PC1 / just a between-domain offset?) does not depend on the task at all: the
# eraser is fit on the domain label, and the task only enters the ridge that follows. So stage 1
# runs ONE task per cell and still gets every geometry number over all ten cells. Only the two
# CONTROL ARMS need a task menu, and only if stage 1 leaves the reading open.
#
# MEMORY: 64G, same as cs_leace, measured peak there 30.7 G. This driver adds two more erasers but
# fits/scores/frees them ONE AT A TIME, and they share the single SVD (fit_leace(svd=)), so the
# resident set is one eraser's basis (0.75 G at d=93184) above the cs_leace shape, not three.
# CPUS 32 = the break-even 0.512*64; the check below prints it so an under-request cannot pass
# silently. Bill is mem_MB/2 * wall, so those cores are already paid for.
set -euo pipefail
CACHE="${1:?CACHE_DIR}"; TAG="${2:?TAG}"; OUTDIR="${3:?OUT_DIR}"
TAPS="${4:-enc0,enc12}"; TASKS="${5:-}"
# Its OWN tree. The enc3 array 20565621 is LIVE out of /projects/bhqk/htang13/speech_leace and
# redeploying there mid-array would silently change a running experiment.
cd /projects/bhqk/htang13/speech_leace_ctrl
module load pytorch-conda/2.8
# OpenBLAS reads OPENBLAS_NUM_THREADS in preference to OMP_NUM_THREADS; setting only OMP leaves it
# single-threaded-by-default and wastes cores that are already paid for.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
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
if [[ -n "$TASKS" ]]; then EXTRA=(--tasks "$TASKS"); fi

numactl --interleave=all python -u scripts/neuroprobe/v3_leace_controls.py \
  --cache "$CACHE" --tag "$TAG" --cell "$CELL" --taps "$TAPS" \
  --out "$OUTDIR/ctrl_S${CELL%,*}T${CELL#*,}.json" "${EXTRA[@]}"
