#!/bin/bash
#SBATCH --job-name=elec_encode
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=125000M
#SBATCH --time=00:30:00
#SBATCH --array=0-11
#SBATCH --output=/projects/bhqk/htang13/viz/elec_encode_%A_%a.out
#
# Per-ELECTRODE encode for measure #2 (viz_elec_auroc.py). ONLY NEEDED IF an --elec-taps 0,12
# cache over the BOARD session/task set does not already exist -- check before submitting; the
# existing v3_probe_encode_r6_elec.sbatch wrote --elec-taps 12 over the cohort7/probe4
# DEFAULTS, which is the wrong session set and the wrong task set for this figure.
#
# THE WINDOW IS 1 s (board parity), NOT the 2 s viz window, and that is a choice with a reason:
#
#   * disk. v3_viz_encode.sbatch refuses elec taps at 2 s because the cache would be ~1.1 TB
#     against 2.2 TB of headroom on /projects/bhqk while pretrains write ckpts there. At 1 s it
#     is ~571 GB (MEASURED for the board cache with the elec taps).
#   * #2 does not need the pre-onset baseline that 2 s buys. Measure #1 subtracts the first 16
#     frames because a condition-mean map has no other way to remove a baseline offset; here
#     significance comes from a label permutation, which absorbs any baseline offset for free.
#   * 1 s IS the leaderboard window, so these AUROCs are quotable next to board numbers.
#
# --elec-taps 0,12 gets both ends of the depth axis on the canonical contact axis. Tap 0 costs
# almost nothing extra (~348 columns vs enc12_elec's k_full*256) and it is the pre-pretraining
# anatomical map -- the half that actually answers "is the signal in STG in the DATA".
#
# --mem is ABOVE ghx4's DefMemPerGPU (113560M = billing 1000) on purpose. MEASURED peak with
# enc12_elec alone was 108.4 GiB (job 2718998 MaxRSS), leaving ~2 GiB of headroom; adding
# enc0_elec eats into that. 125000M bills 122.07*9.01726 = ~1101, a 10% premium, which is far
# cheaper than an OOM -- an OOM'd job bills in full and returns nothing.
#
# Sharded one session per array task (feedback-always-parallel-shard): caps memory at a single
# session's peak and the short walltime earns backfill against the deep ghx4 queue.
set -euo pipefail
CKPT="${1:?usage: sbatch viz_elec_encode.sbatch <CKPT> <TAG> <OUTDIR>}"
TAG="${2:?need TAG}"
OUTDIR="${3:?need OUTDIR}"   # put this on /work/nvme, NOT /projects -- ~571 GB
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
  --elec-taps 0,12
