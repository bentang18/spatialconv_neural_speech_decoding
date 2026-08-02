#!/bin/bash
#SBATCH --job-name=fe_abl_slim
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bhqk/htang13/fe_abl_slim_%A_%a.out
# One array cell per Lite session: copy enc0 + enc0_elec (+ embedded elec_labels) out of the
# 44-71 GB board record into a ~0.8 GB slim record the `fm:` ablation can load cheaply.
#
# WHY 48G. The read is mmap'd, so only the enc0 regions fault in (~0.8 GB) — but page cache from
# an mmap counts against the cgroup, and an under-request that OOMs still BILLS IN FULL. 48G is
# sized to hold the working copy plus slack, not the 50 GB source.
# --cpus-per-task=24 is FREE: Delta CPU bills MAX_TRES == mem_MB/2, and the CPU term only bites
# above ~0.512*mem_GB cores (24 at 48G), so 24 bills identically to the sbatch default of 8.
#
#   usage: sbatch --array=0-11 fe_abl_slim.sbatch <FULL_CACHE> <TAG> <SLIM_DIR> <OUT_TAG>
set -euo pipefail
SRC="${1:?FULL_CACHE}"; TAG="${2:?TAG}"; DST="${3:?SLIM_DIR}"; OUT="${4:?OUT_TAG}"
cd /projects/bhqk/htang13/speech_board
module load pytorch-conda/2.8
mkdir -p "$DST"
numactl --interleave=all python -u scripts/neuroprobe/fe_abl_slim_cache.py \
  --cache-dir "$SRC" --tag "$TAG" --out-dir "$DST" --out-tag "$OUT" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --sidecar /projects/bhqk/htang13/arm0_elec_labels_sidecar.pkl
