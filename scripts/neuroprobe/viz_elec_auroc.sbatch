#!/bin/bash
#SBATCH --job-name=elec_auroc
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
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
#   ~2378-3500 windows, so n_task ~ 3300. The FIRST launch of this array (20580098/20580318)
#   ran the permutation loop single-threaded and reported 531-586 s per (tap, task), i.e.
#   ~1.06 s/perm at K=7 -- so ~6.2 h/session, and it sat at 13% CPU efficiency (AveCPU
#   01:01:01 at 58:34 wall, 1.04 of 8 allocated cores).
#
#   cv_auroc_cm now threads the permutation loop over contact chunks (np.argsort releases the
#   GIL) and no longer re-materialises the folds inside it. Benchmarked at n=3300, P=119,
#   T=32 on 8 threads: 5.2x at enc0 (K=7), 9.9x at enc12 (K=32). Per session:
#       enc0  ~0.42 h + enc12 ~0.40 h + ~10 min of loads/cov/projection = ~1.0 h
#   --time=06:00:00 is a ~6x margin, deliberately generous because requested walltime does NOT
#   affect the bill -- only actual wall time does -- while a TIMEOUT bills in full and returns
#   nothing (job 20566474 burned 494 h that way at --time=01:30). At the 2 s window T doubles
#   and so does this, which is why the margin is not trimmed.
#
#   The two taps are NOT matched in K, and that is inherent rather than a choice: enc0's HGA
#   band has only F_b=7 columns so --pc 32 is a no-op there and keeps 100% of its variance,
#   while enc12 has C=256 and keeps 32 PCs. Each contact is scored on its own features, so this
#   does not bias WHERE the signal is; it does mean the taps' AUROC ceilings are not identical,
#   so read the maps' anatomy against each other, not their absolute heights.
#
#   MEMORY. This is what the bill is: billed_h = (mem_MB/2)/1000 per wall-hour, so 16G costs
#   8.19 h per wall-hour and cores under 0.512*mem_GB (= 8 here) are FREE. 8 cpus is therefore
#   free throughput -- which the first launch paid for and did not use. Peak RSS at the real
#   shapes: the (n, P, T, K) projection 1.7 GiB, briefly doubled by the contact_major transpose
#   (both live only for the duration of the copy), + one (chunk, P, T, C) block 1.0 GiB +
#   per-contact covariances 62 MB + the null cube 15 MB = ~6 GiB. MEASURED MaxRSS on the
#   single-threaded launch was 2.0 GiB. 16G holds the transposed peak with room; do NOT trim it
#   to 8G -- an OOM bills in full and returns nothing, and 8G would also halve the FREE core
#   budget (0.512*mem_GB) and with it the speedup. Expected total bill at ~1.0 h actual:
#   1.0 x 12 x 8.19 = ~98 h against a ~28,039 h balance.
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
# Which deployed tree to run. Defaults to the standard clone; override to run a staged copy
# without touching a tree an in-flight array is executing from.
ROOT="${ROOT:-/projects/bhqk/htang13/speech}"
cd "$ROOT"
module load pytorch-conda/2.8
# ONE BLAS thread ON PURPOSE. The permutation loop is now parallelised in Python over contact
# chunks, so leaving BLAS at 8 would put 8 threads inside each of 8 workers and contend for an
# 8-core cpuset. The per-chunk matmuls are small (K <= 32) so BLAS threading buys little there
# anyway; the one step that loses is the projection einsum, which is ~20 s per (tap, task)
# against a permutation loop that this trade shrinks by 5-10x.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT/src"
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
  --band-fdims "$FDIMS" --seed 33 --threads "${SLURM_CPUS_PER_TASK:-8}"
# --threads from SLURM_CPUS_PER_TASK, NOT the --threads 0 affinity probe: on a Delta login node
# sched_getaffinity reports all 128 cores, and if a job ever inherited that it would spawn 128
# threads onto an 8-core cpuset. cpus-per-task is the number that is actually allocated.
