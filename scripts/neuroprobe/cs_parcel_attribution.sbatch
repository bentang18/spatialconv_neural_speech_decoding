#!/bin/bash
#SBATCH --job-name=cs_parcattr
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bhqk/htang13/cs_parcattr_%A_%a.out
# Per-cell CS parcel attribution: ridge weight mass + leave-one-parcel-out, at enc12|std.
#
#   usage: sbatch --array=0-9 --mem=64G cs_parcel_attribution.sbatch <CACHE> <TAG> <SHARDS> <OUT> [MMAP]
#
# <SHARDS> is the PUBLISHED cs shard directory for this tag. Every task gates its full-intersection
# refit against cs_S{s}T{t}.json before any LOPO delta is computed, and a mismatch is fatal -- a
# delta against an unverified baseline is not a measurement.
#
# 🔴 --mmap IS THE DEFAULT HERE, AND IT IS A MEMORY DECISION, NOT A SPEED ONE. Measured from the
# zip directory of the cd55k cache (2026-08-12), per record:
#     enc12_elec  47.7 GiB (anchor) / 59.9 GiB (s10_t1)   <- CS NEVER GATHERS THIS
#     enc12       5.61 / 9.48 GiB                          <- what CS actually reads
#     enc0        0.10 / 0.13 GiB
# Eager load pays for the whole 71 + 99 GiB; CS touches ~15 GiB of it. That is why the board's cs
# shards needed 200G, and why an earlier 176G request here would have OOM-KILLED s1_t1, s7_t0 and
# s10_t1, whose anchor+test exceed it. Under --mmap a tap never gathered is never paged in, so the
# resident set is the parcel taps (~15 GiB worst case) plus fp32 working slices (n_train 3500 x d
# 139,776 fp32 ~ 2 GB) -- call it ~20 GiB, requested at 64G for headroom.
#
# The usual objection to --mmap is that it turns the load into scattered Lustre reads, which cost
# WS ~78 min. That argument does not transfer: cs is FIT-bound, not load-bound, and this job runs
# ~8x the fits of a normal cs shard, so any load penalty amortizes over far more compute. The
# pages also stay in cache across the LOPO loop, which re-gathers the same rows repeatedly.
#
# COST. Each cell fits 15 tasks x 2 taps x (1 full + |P| leave-one-out), |P| = 3-7 measured.
# Billing is memory-bound at 0.512*mem_GB core-equivalents, so 64G bills ~33 against 176G's ~90.
#
# WORKERS: none. The 15 tasks run serially inside one process because the LOPO loop already keeps
# BLAS busy and the cs design matrix is small enough that forking buys little against the risk of
# multiplying peak RSS on a 938G cache.
set -euo pipefail
CACHE="${1:?CACHE_DIR}"; TAG="${2:?TAG}"; SHARDS="${3:?PUBLISHED_CS_SHARD_DIR}"; OUT="${4:?OUT_DIR}"
MMAP="${5:---mmap}"
cd /projects/bhqk/htang13/speech_board
module load pytorch-conda/2.8
CPT="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="$CPT"
BE=$(( ${SLURM_MEM_PER_NODE:-0} / 2000 ))
echo "[check] alloc cpus=$CPT mem=${SLURM_MEM_PER_NODE:-?}M omp=$OMP_NUM_THREADS; break-even cpus=$BE"

# The cell comes from CS_TEST_CELLS inside the script, the SAME list `--mode cs` shards over, so
# the array index, the cache record, and the shard used as the gate can never disagree.
numactl --interleave=all python -u scripts/neuroprobe/cs_parcel_attribution.py \
  --cache "$CACHE" --tag "$TAG" --index "$SLURM_ARRAY_TASK_ID" \
  --shard-dir "$SHARDS" --out "$OUT" $MMAP
