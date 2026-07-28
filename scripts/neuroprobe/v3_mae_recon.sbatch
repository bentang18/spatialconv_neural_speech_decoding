#!/bin/bash
#SBATCH --job-name=mae_recon
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=113560M
#SBATCH --time=00:20:00
#SBATCH --output=/projects/bhqk/htang13/viz/mae_recon_%j.out
#
# One session's masked reconstruction, dumped for the paper strip. A single forward at the
# SSL clip length (2 s) -- no sweep, no array, so the short walltime earns backfill against
# the ghx4 queue. mem stays at DefMemPerGPU: ghx4 bills MAX_TRES, so asking for more than
# 113560M overcharges without buying anything.
#
# This overwrites NOTHING in the model tree. v3_mae_recon.py is a new standalone script that
# wraps the objective's _mae_output locally; the training path is untouched, which is the
# point -- a debug flag inside objective.py would be a permanent hazard for a one-off figure.
set -euo pipefail
CKPT="${1:?usage: sbatch v3_mae_recon.sbatch <CKPT> <SUBJECT> <TRIAL> <OUT>}"
SUBJ="${2:?need SUBJECT}"
TRIAL="${3:?need TRIAL}"
OUT="${4:?need OUT}"
cd /projects/bhqk/htang13/speech
module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/projects/bhqk/htang13/speech/src
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_lite
# no srun, unlike the encode launchers. This is one process on one GPU with no ranks to
# wire together, and srun died twice on two different nodes (gh141, gh024) with "Error
# configuring interconnect" before Python ever started. sbatch already exports
# CUDA_VISIBLE_DEVICES for --gpus-per-node=1, so the step launcher is pure overhead here --
# dropping it removes the dependency rather than retrying against it.
/projects/bhqk/htang13/speech/.venv/bin/python -u -m scripts.neuroprobe.v3_mae_recon \
  --ckpt "$CKPT" \
  --band-cache-dir $ROOT/band_v3slow \
  --band-cache-dir $ROOT/band_v3mid \
  --band-cache-dir $ROOT/band_hga \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --subject "$SUBJ" --trial "$TRIAL" \
  --n-clips 8 --clip-dur 2.0 --seed 33 \
  --out "$OUT"
