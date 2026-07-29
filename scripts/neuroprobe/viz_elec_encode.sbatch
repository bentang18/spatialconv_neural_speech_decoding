#!/bin/bash
#SBATCH --job-name=elec_encode
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=113560M
#SBATCH --time=00:30:00
#SBATCH --array=0-11
#SBATCH --output=/projects/bhqk/htang13/viz/elec_encode_%A_%a.out
#
# Per-ELECTRODE encode for measure #2 (viz_elec_auroc.py). ONLY NEEDED IF an --elec-taps 0,12
# cache over the BOARD session/task set does not already exist -- check before submitting; the
# existing v3_probe_encode_r6_elec.sbatch wrote --elec-taps 12 over the cohort7/probe4
# DEFAULTS, which is the wrong session set and the wrong task set for this figure.
#
# THE WINDOW IS 1 s (board parity), NOT the 2 s viz window. Sized from the ACTUAL 2 s cache
# (/projects/bhqk/htang13/v3_viz_cache_cdlin_45k, read 2026-07-29: 12 sessions, 353 GB,
# n_windows 9063-13592, 100-120 contacts, band_lengths (8,32,64), band_fdims (7,6,7), fp16):
#
#   * DISK. At 1 s band_lengths are (4,16,32) so k_full=52, enc0 is 348 columns and enc12 is
#     52*256=13312. Over the 12 sessions' real n and contact counts that is enc12_elec 389 GB +
#     enc0_elec 10 GB + parcel taps 171 GB = ~570 GB, which reproduces the 571 GB figure the viz
#     sbatch quotes. The 2 s version would be ~1.14 TB. /work/nvme has 4.1 TB free so 2 s WOULD
#     fit -- disk is no longer the binding constraint, RAM is (below).
#   * RAM. The write is held in memory before torch.save. At 1 s the largest session is
#     enc12_elec 43.0 GB + enc0_elec 1.1 GB + parcel taps 17.5 GB = ~62 GB, comfortably inside
#     ghx4's DefMemPerGPU (113560M, billing exactly 1000). At 2 s it is ~124 GB, which does NOT
#     fit and would need ~250G -- a 2.25x memory bill and much worse backfill, to buy an axis
#     detail measure #1 already covers.
#   * #2 does not need the pre-onset baseline that 2 s buys. Measure #1 subtracts the first 16
#     frames because a condition-mean map has no other way to remove a baseline offset; here
#     significance comes from a label permutation, which absorbs any baseline offset for free.
#   * 1 s IS the leaderboard window, so these AUROCs are quotable next to board numbers.
#
# To match #1's axis exactly instead, add --clip-dur 2.0 --clip-offset -0.5 and RAISE --mem to
# 250G. Do not raise the window without raising the memory.
#
# --elec-taps 0,12 gets both ends of the depth axis on the canonical contact axis. Tap 0 adds
# only ~2.5% to the write (348 columns vs 13312) and it is the pre-pretraining anatomical map --
# the half that actually answers "is the signal in STG in the DATA at all".
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
