#!/bin/bash
#SBATCH --job-name=board_partialft
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%A_%a.out
#
#   sbatch --array=0-11 --mem=113560M  v3_board_partialft.sbatch <CKPT> ws       <CACHE> <TAG> <OUTDIR>
#   sbatch --array=0-11 --mem=227120M  v3_board_partialft.sbatch <CKPT> csession <CACHE> <TAG> <OUTDIR>
#   sbatch --array=0-9  --mem=227120M  v3_board_partialft.sbatch <CKPT> cs       <CACHE> <TAG> <OUTDIR>
#
# 🔴 MEMORY IS THE REASON THE THREE LINES DIFFER, AND IT COSTS REAL FAIRSHARE.
# ws trains and evaluates inside ONE session, so one tap-11 cache is resident: 60-90 GiB
# (measured 60.4 GiB at n_win=9164; the largest Lite session is 13,592 windows). That fits
# --mem=113560M, which is one GPU's DefMemPerGPU and bills MAX_TRES 1000.
# cs and csession train on a DIFFERENT record than they evaluate (anchor S2T4 / the sibling
# trial), so BOTH caches are resident at once: ~131-161 GiB. That needs 227120M, which BILLS
# 2000 -- double. There is no way around it: the train rows are read every epoch and the val rows
# are read every epoch, so neither cache can be dropped. Storing the tap in bf16 would halve it
# but carries ~4e-3 relative error and would FAIL the parity-L0 gate (threshold 1e-3), which is
# the one thing standing between a block-split bug and a published number. Not worth it.
#
# ARRAY INDEX = CELL. ws/csession are the 12 Lite sessions, cs is the 10 CS test cells
# (v3_board_readout.py:98-118). Never fold cells together -- the board test is PAIRED OVER CELLS.
#
# Each task runs all 15 board tasks off ONE tap-11 encode per session; only block 12 reruns per
# task. The 15 tasks do NOT share a train/val/test partition (audit/board_split_signature.py:
# 2,100-6,400 conflicting rows per session/fold), so they cannot share a fine-tune -- but they do
# share the 12-block forward, which is what this shape buys.
set -uo pipefail
CKPT=$1; REGIME=$2; CACHE=$3; TAG=$4; OUTDIR=$5
module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
mkdir -p "$OUTDIR"
[ -f "$CKPT" ] || { echo "[FATAL] ckpt missing: $CKPT"; exit 1; }
[ -d "$CACHE" ] || { echo "[FATAL] board cache missing: $CACHE"; exit 1; }
echo "[board-ft] regime=$REGIME cell_index=$SLURM_ARRAY_TASK_ID ckpt=$CKPT tag=$TAG mem=${SLURM_MEM_PER_NODE}M"
srun "$VENV" -u scripts/neuroprobe/v3_board_partialft.py \
  --ckpt "$CKPT" \
  --regime "$REGIME" \
  --cell-index "$SLURM_ARRAY_TASK_ID" \
  --board-cache-dir "$CACHE" \
  --board-tag "$TAG" \
  --band-cache-dir $ROOT/band_v3slow \
  --band-cache-dir $ROOT/band_v3mid \
  --band-cache-dir $ROOT/band_hga \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --out "$OUTDIR/ft_${REGIME}_${SLURM_ARRAY_TASK_ID}.json" \
  "${@:6}"
