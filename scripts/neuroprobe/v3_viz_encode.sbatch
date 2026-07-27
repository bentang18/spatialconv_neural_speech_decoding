#!/bin/bash
#SBATCH --job-name=viz_encode
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=113560M
#SBATCH --time=00:40:00
#SBATCH --array=0-11
#SBATCH --output=/projects/bhqk/htang13/viz/viz_encode_%A_%a.out
#
# 2 s visualization encode. Same 12 Neuroprobe-Lite sessions and same forward code as
# v3_board_encode_r6.sbatch, with two deliberate departures:
#
#   --clip-dur 2.0 --clip-offset -0.5   window is -0.5 -> +1.5 s around the clip start
#                                       instead of the leaderboard's [start, start+1 s].
#
# BREAKS BOARD PARITY ON PURPOSE. Nothing this writes may be reported as a leaderboard
# number. 2.0 s is not arbitrary: it is the SSL training clip length, so the longer window
# is in-distribution rather than an extrapolation, and the 0.5 s of pre-onset baseline is
# what makes a trajectory readable (it has to start somewhere before the event).
#
# NO --elec-taps. The 1 s board cache is 571 GB with them; at 2 s it would be ~1.1 TB
# against 2.2 TB of headroom on /projects/bhqk, while Ben's pretrains are writing ckpts to
# the same allocation. Every lobe-level figure runs off the parcel-mean taps plus the
# per-parcel electrode counts in parcel_canon, so the elec taps buy only the brain render
# and are a separate, smaller job when that lands.
#
# One array task per session (feedback-always-parallel-shard): caps memory at a single
# session's peak, and the short walltime earns backfill against the deep ghx4 queue.
set -euo pipefail
CKPT="${1:?usage: sbatch v3_viz_encode.sbatch <CKPT> <TAG> <OUTDIR>}"
TAG="${2:?need TAG}"
OUTDIR="${3:?need OUTDIR}"
cd /projects/bhqk/htang13/speech
module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/projects/bhqk/htang13/speech/src
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_lite
srun /projects/bhqk/htang13/speech/.venv/bin/python -u -m scripts.neuroprobe.v3_probe_encode_r4 \
  --ckpt "$CKPT" --tag "$TAG" --out-dir "$OUTDIR" \
  --band-cache-dir $ROOT/band_v3slow \
  --band-cache-dir $ROOT/band_v3mid \
  --band-cache-dir $ROOT/band_hga \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --batch-size 64 \
  --frontend v3r6 \
  --sessions board --tasks board15 --electrode-set lite \
  --session-index "$SLURM_ARRAY_TASK_ID" \
  --clip-dur 2.0 --clip-offset -0.5
