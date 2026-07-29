#!/bin/bash
#SBATCH --job-name=elec_auroc
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=/projects/bhqk/htang13/viz/elec_auroc_%A_%a.out
#
# Measure #2: per-electrode single-trial AUROC + max-statistic permutation FWER, one array task
# per session. CPU only -- there is no forward pass here, only a readout of the encode cache.
#
# THIS IS THE WHOLE JOB. viz_elec_encode.sbatch is NOT needed: the elec taps already exist in
# /projects/bhqk/htang13/v3_board_cache_board_cdlin_45k (read 2026-07-29: 12 records, 571 GB,
# band_lengths (4,16,32) = the 1 s board window, feats enc0/enc0_elec/enc3/enc6/enc12/enc12_elec,
# all 15 board tasks). So measure #2 costs one CPU array and no GPU time at all.
#
# --band-fdims IS REQUIRED for the enc0 tap on that cache: its records store band_fdims = None,
# and enc0's per-band widths are not recoverable from the flattened total. 7,6,7 is READ off the
# 2 s viz cache written by the same v3r6 frontend, and 4*7 + 16*6 + 32*7 = 348 = the stored enc0
# width on all 12 records (checked). The width check alone would NOT be enough -- several triples
# total 348 -- which is why the value comes from a sibling record, not from arithmetic.
#
# WALLTIME AND MEMORY ARE BOTH MEASURED, not guessed, and they are sized against the two ways a
# Delta CPU job wastes fairshare:
#
#   TIME. MEASURED on the real record (s10_t0, 2026-07-29), not extrapolated from n_windows:
#   n_windows is 9063-13592 but the board15 labels are binary median splits defined on only
#   ~2378-3500 windows, so n_task ~ 3300. Timing enc0_elec/onset at n_perm 0 vs 20 gives
#   (18-5)/20 = 0.65 s per permutation at K=7. cv_auroc is NOT K-free -- timed at the same
#   shapes it costs 0.839 s at K=7 and 1.504 s at K=32 (a fixed rank-sort cost plus a per-K
#   term) -- so enc12_elec scales to 0.65 x 1.504/0.839 = ~1.17 s, which independently
#   reproduces the earlier 1.14 s bench at K=32. Per session:
#       enc0  0.65 s x 500 x 15 tasks = 1.4 h
#       enc12 1.17 s x 500 x 15 tasks = 2.4 h    + ~10 min of loads/cov/projection = ~4.0 h
#   --time=12:00:00 is a ~3x margin, deliberately generous because requested walltime does NOT
#   affect the bill -- only actual wall time does -- while a TIMEOUT bills in full and returns
#   nothing (job 20566474 burned 494 h that way at --time=01:30). At the 2 s window T doubles
#   and so does this: ~8 h, which is why the margin is not trimmed.
#
#   The two taps are NOT matched in K, and that is inherent rather than a choice: enc0's HGA
#   band has only F_b=7 columns so --pc 32 is a no-op there and keeps 100% of its variance,
#   while enc12 has C=256 and keeps 32 PCs. Each contact is scored on its own features, so this
#   does not bias WHERE the signal is; it does mean the taps' AUROC ceilings are not identical,
#   so read the maps' anatomy against each other, not their absolute heights.
#
#   MEMORY. This is what the bill is: billed_h = (mem_MB/2)/1000 per wall-hour, so 16G costs
#   8.19 h per wall-hour and cores under 0.512*mem_GB (= 8 here) are FREE. 8 cpus is therefore
#   free throughput. Peak RSS at the real shapes: the (n, P, T, K) projection 1.7 GiB + one
#   (chunk, P, T, C) block 1.0 GiB + per-contact covariances 62 MB + the null cube 15 MB = ~4 GiB.
#   16G is ~4x that. Expected total bill at ~4.0 h actual: 4.0 x 12 x 8.19 = ~393 h against a
#   28,455 h balance. Do NOT trim this to 8G -- an OOM bills in full and returns nothing.
#
# cpu-preempt + --requeue is the right partition for an array this shape (dcc-array-flood /
# throughput-per-fairshare), but only once the runtime is confirmed on one session -- a requeue
# restarts a shard from scratch, and at ~5 h that is expensive. Submit shard 0 alone first.
#
# RUN THE PRE-FLIGHT ON THE LOGIN NODE FIRST -- it needs no cache and takes ~30 s:
#   python -m scripts.neuroprobe.viz_elec_auroc --self-test
# It verifies max-stat FWER calibration, planted-effect recovery and the drift guard on
# synthetic data. If it fails, nothing below is worth submitting.
set -euo pipefail
CACHE="${1:?usage: sbatch viz_elec_auroc.sbatch <CACHE_DIR> <OUT_DIR> [TAPS] [N_PERM]}"
OUT="${2:?need OUT_DIR}"
TAPS="${3:-enc0_elec,enc12_elec}"
NPERM="${4:-500}"
FDIMS="${5:-7,6,7}"
cd /projects/bhqk/htang13/speech
module load pytorch-conda/2.8
# numpy's einsum threads; argsort does not, so this helps the projection and not the inner loop
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONPATH=/projects/bhqk/htang13/speech/src
# pytorch-conda's `python`, NOT .venv/bin/python: .venv on /projects is the DeltaAI aarch64
# build and has no x86 interpreter at all, so .venv/bin/python here fails before Python starts.
# This matches viz_reduce.sbatch and leace_controls.sbatch, and the shard path was RUN under it
# against a real board record. Aggregation is the one step that needs more: region_table calls
# dkt_tables -> the braintreebank study package -> mne, which this module lacks. Run it as
#   /projects/bhqk/htang13/venv_probe_x86/bin/python -m scripts.neuroprobe.viz_elec_auroc \
#     --aggregate --out-dir <OUT> --taps ... --tasks ...
numactl --interleave=all python -u -m scripts.neuroprobe.viz_elec_auroc \
  --cache "$CACHE" --out-dir "$OUT" \
  --session-index "$SLURM_ARRAY_TASK_ID" \
  --taps "$TAPS" --band hga --n-perm "$NPERM" --perm-block 200 --pc 32 --cov-stride 4 \
  --band-fdims "$FDIMS" --seed 33
