#!/bin/bash
#SBATCH --job-name=pool_variants
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/pool_variants_%A_%a.out
#
# Electrode-pooling variants (A per-parcel / B indiscriminate) for the pretrain-probe cohort.
# CPU-ONLY on Delta CPU — never a dtai GPU slot.
#
#   sbatch pool_variants_readout.sbatch assert <CACHE> <TAG>
#   sbatch --array=0-12 pool_variants_readout.sbatch array <CACHE> <TAG> <SHARDS>
#   sbatch --dependency=afterany:$AID pool_variants_readout.sbatch merge <CACHE> <TAG> <SHARDS> <OUT>
#
# Preamble/cpu/mem copied from the working v3_probe_readout_r4.sbatch on this host — same
# dependency surface (numpy/torch/sklearn only; probe_pool_variants imports the ridge and splits
# from v3_probe_readout_r4 and nothing from speech_decoding). mem is 128G not 96G because a CS
# shard holds TWO sessions' enc12_elec resident at once and that tap is unpooled.
set -euo pipefail
module load pytorch-conda/2.8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

MODE="$1"
CACHE="$2"
TAG="$3"
SHARDS="${4:-}"
OUT="${5:-}"
BETAS="${BETAS:-0,0.5,1,2,4}"
VARIANTS="${VARIANTS:-A,B}"

cd /projects/bhqk/htang13/speech
SCRIPT=scripts/neuroprobe/probe_pool_variants.py

case "$MODE" in
  assert)
    python "$SCRIPT" --mode assert --cache-dir "$CACHE" --tag "$TAG"
    ;;
  array)
    python "$SCRIPT" --mode array --cache-dir "$CACHE" --tag "$TAG" \
      --shard-dir "$SHARDS" --array-index "$SLURM_ARRAY_TASK_ID" \
      --variants "$VARIANTS" --betas "$BETAS"
    ;;
  merge)
    python "$SCRIPT" --mode merge --cache-dir "$CACHE" --tag "$TAG" \
      --shard-dir "$SHARDS" --out "$OUT" --variants "$VARIANTS" --betas "$BETAS"
    ;;
  *)
    echo "unknown mode '$MODE' (assert|array|merge)" >&2
    exit 2
    ;;
esac
echo "POOL_VARIANTS_DONE mode=$MODE rc=$?"
