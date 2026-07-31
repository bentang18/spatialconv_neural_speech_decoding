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
#   sbatch --array=0-11 --mem=113560M  v3_board_partialft.sbatch <CKPT> ws       <CACHE> <TAG> <OUTDIR> --x11-dtype fp16
#   sbatch --array=0-11 --mem=113560M  v3_board_partialft.sbatch <CKPT> csession <CACHE> <TAG> <OUTDIR> --x11-dtype fp16
#   sbatch --array=0-9  --mem=113560M  v3_board_partialft.sbatch <CKPT> cs       <CACHE> <TAG> <OUTDIR> --x11-dtype fp16
#
# 🔴 EVERY REGIME BILLS 1000. 113560M IS ONE GPU'S DefMemPerGPU; 227120M BILLS DOUBLE FOR THE
# SAME ONE GPU, so staying under the share is worth real fairshare on ~34 array tasks.
# 🔴 fp16 ON ALL THREE REGIMES — fp32 ws DOES NOT FIT, MEASURED. The earlier claim here ("80.2 GiB
# in fp32, MaxRSS 93.8 GiB, it fits") was inferred from a --parity-only run, which never builds
# the training-side slices; the full ws run OOM-KILLED at this same 113560M (job 2790770, dead in
# 1m40s during setup). fp16 ws is the configuration that actually completed (job 2790998).
# cs and csession train on a DIFFERENT record than they evaluate (anchor S2T4 / the sibling
# trial), so BOTH caches are resident -- MEASURED 40.1 + 32.3 GiB at fp16 (jobs 2791036/2791037),
# which fits the share; fp32 would be ~160 GiB and force 227120M.
# fp16 does NOT weaken the parity gate: parity-L0 keeps CERT_ROWS=256 windows in fp32 no matter
# what --x11-dtype is, so the STRUCTURAL question (is the block split right?) is still certified
# at full precision. fp16's effect is reported separately as [parity-L0s], measured 1.910e-3 --
# storage noise, not a bug, on features the board pipeline already stores fp16 anyway.
# MEASURED GATES: elec unit (ws/csession) is BIT-EXACT, rel=0.000e+00. parcel unit (cs) is
# rel=5.984e-04 -- NOT zero, and expected: the cached parcel tap is the electrode mean taken over
# values the encoder had already rounded to fp16, so the pool and the rounding commute in one path
# and not the other. It is fp16-storage-sized, an order under the 1e-3 gate, and NOT a split bug.
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
# 🔴 LITE, NOT PRETRAIN. The board's Lite sessions include HELD-OUT subjects (1, 3, 4, 7, 10)
# that are deliberately absent from the pretrain band cache — pointing at
# v14_3band_v3_spec_pretrain fails with "expected exactly one cache entry for subject 1 trial 1,
# found 0". The WS pilot never hit this because PROBE_COHORT_7 is entirely inside the pretrain
# set. Read straight off the board encode launcher (v3_board_encode_r6.sbatch:42), which is what
# produced the caches this harness must be parity-checked against.
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_lite
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
