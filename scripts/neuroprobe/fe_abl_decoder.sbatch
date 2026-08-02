#!/bin/bash
#SBATCH --job-name=fe_dec
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/bhqk/htang13/fe_dec_%A_%a.out
# Decoder-parity arm: OUR enc0 features through UPSTREAM's LogisticRegression(C=1.0).
# Same slim cache, same splits, same standardization as the locked readout — only the estimator
# differs, which is the only way to tell "our frontend is better" apart from "their decoder is
# untuned". See scripts/neuroprobe/fe_abl_decoder.py.
#
# Anything after WORKERS is forwarded verbatim to the script (e.g. --c-grid for the tuned arm).
#   usage: sbatch --array=0-N fe_abl_decoder.sbatch <CACHE> <TAG> <SHARD> <MODE> <TAPS> <WORKERS> [ARGS...]
set -euo pipefail
CACHE="${1:?CACHE}"; TAG="${2:?TAG}"; SHARD="${3:?SHARD_DIR}"
MODE="${4:?ws|cs|csession}"; TAPS="${5:?TAPS}"; WK="${6:-4}"
cd /projects/bhqk/htang13/speech_board
module load pytorch-conda/2.8
# This script imports the readout as `scripts.neuroprobe.v3_board_readout` (a PEP 420 namespace
# package), so the REPO ROOT must be on sys.path. Running a file directly only puts that file's
# own directory there, which is why the readout's own launcher never needed this.
export PYTHONPATH="/projects/bhqk/htang13/speech_board:${PYTHONPATH:-}"
CPT="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=$(( CPT / WK ))
echo "[check] cpus=$CPT mem=${SLURM_MEM_PER_NODE:-?}M workers=$WK omp=$OMP_NUM_THREADS break-even=$(( ${SLURM_MEM_PER_NODE:-0} / 2000 ))"
numactl --interleave=all python -u scripts/neuroprobe/fe_abl_decoder.py \
  --cache-dir "$CACHE" --tag "$TAG" --mode "$MODE" --index "$SLURM_ARRAY_TASK_ID" \
  --shard-dir "$SHARD" --taps "$TAPS" --workers "$WK" "${@:7}"
