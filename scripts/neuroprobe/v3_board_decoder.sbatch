#!/bin/bash
#SBATCH --job-name=board_dec
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=113560M
#SBATCH --time=04:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/board_dec_%A_%a.out
# CS decoder sweep: {mlp, naive-CNN, per-band-CNN} x {enc0, enc12}, one array cell per CS test
# session (CS_TEST_CELLS, 10 cells). GPU is correct here and NOT a violation of the "CPU probes go
# to Delta CPU" rule -- that rule is about the ridge readout, which is a linear solve. These arms
# are real decoder fwd/bwd (up to 100 epochs x ~13k anchor rows x 90 fits per cell), which is
# exactly what a GPU is for.
#
# mem=113560M is the ghx4 DefMemPerGPU fair share: going above it bills MAX_TRES at the whole-node
# rate. The enc12 anchor design matrix is ~11 GB fp32 and is standardized IN-PLACE to stay under it.
#
#   usage: sbatch --array=0-9 v3_board_decoder.sbatch <CACHE_DIR> <TAG> <OUT_DIR> [ARMS] [TAPS]
set -euo pipefail
CACHE="${1:?CACHE_DIR}"; TAG="${2:?TAG}"; OUT="${3:?OUT_DIR}"
ARMS="${4:-mlp,naive,perband}"; TAPS="${5:-enc0,enc12}"
module load python/miniforge3_pytorch/2.10.0
TREE=/projects/bhqk/htang13/speech
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"
echo "[board_dec] cell=$SLURM_ARRAY_TASK_ID tag=$TAG arms=$ARMS taps=$TAPS"
"$TREE/.venv/bin/python" -u scripts/neuroprobe/v3_board_decoder.py \
  --cache-dir "$CACHE" --tag "$TAG" --index "$SLURM_ARRAY_TASK_ID" \
  --out-dir "$OUT" --arms "$ARMS" --taps "$TAPS"
