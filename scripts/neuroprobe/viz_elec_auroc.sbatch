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
# WALLTIME AND MEMORY ARE BOTH MEASURED, not guessed, and they are sized against the two ways a
# Delta CPU job wastes fairshare:
#
#   TIME. cv_auroc BENCHMARKED at n=3000, P=128, T=32, K=32: 1.14 s per permutation. The real
#   shapes (READ off the cache 2026-07-29) are kinder than n_windows suggests: n_windows is
#   9063-13592 but the board15 labels are binary median splits defined on only ~2378-3500 of
#   them, so n_task ~ 3500. At P~119, T=32 (1 s), K=32 that is ~1.25 s per permutation, so
#   500 perms x 15 tasks x 2 taps = ~5.2 h. --time=12:00:00 is a ~2.3x margin, deliberately
#   generous because requested walltime does NOT affect the bill -- only actual wall time does --
#   while a TIMEOUT bills in full and returns nothing (job 20566474 burned 494 h that way at
#   --time=01:30). At the 2 s window T doubles and so does this: ~10 h.
#
#   MEMORY. This is what the bill is: billed_h = (mem_MB/2)/1000 per wall-hour, so 16G costs
#   8.19 h per wall-hour and cores under 0.512*mem_GB (= 8 here) are FREE. 8 cpus is therefore
#   free throughput. Peak RSS at the real shapes: the (n, P, T, K) projection 1.7 GiB + one
#   (chunk, P, T, C) block 1.0 GiB + per-contact covariances 62 MB + the null cube 15 MB = ~4 GiB.
#   16G is ~4x that. Expected total bill at ~5.2 h actual: 5.2 x 12 x 8.19 = ~511 h against a
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
cd /projects/bhqk/htang13/speech
module load pytorch-conda/2.8
# numpy's einsum threads; argsort does not, so this helps the projection and not the inner loop
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONPATH=/projects/bhqk/htang13/speech/src
numactl --interleave=all .venv/bin/python -u -m scripts.neuroprobe.viz_elec_auroc \
  --cache "$CACHE" --out-dir "$OUT" \
  --session-index "$SLURM_ARRAY_TASK_ID" \
  --taps "$TAPS" --band hga --n-perm "$NPERM" --perm-block 200 --pc 32 --cov-stride 4
