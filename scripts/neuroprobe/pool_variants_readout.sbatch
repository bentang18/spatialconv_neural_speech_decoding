#!/bin/bash
#SBATCH --job-name=pool_variants
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/pool_variants_%A_%a.out
#SBATCH --error=/projects/bhqk/htang13/logs/pool_variants_%A_%a.out
#
# Electrode-pooling variants (A per-parcel / B indiscriminate) for the pretrain-probe cohort.
# CPU-ONLY — Delta CPU, never a dtai GPU slot.
#
#   sbatch pool_variants_readout.sbatch assert <CACHE> <TAG>
#   sbatch --array=0-12 --mem=128G pool_variants_readout.sbatch array <CACHE> <TAG> <SHARDS>
#   sbatch --dependency=afterany:$AID pool_variants_readout.sbatch merge <CACHE> <TAG> <SHARDS> <OUT>
#
# --cpus-per-task=8 is ONE NUMA node of the 8x16-core layout; numactl --interleave=all keeps the
# allocation off node0 (which starves under the default policy).
# --mem: the CS shards hold TWO sessions' enc12_elec resident at once and the tap is large, so
# request generously on the array (128G) — the assert and merge steps are cheap.
set -euo pipefail

# Match the module/venv preamble of the sibling r*_probe_readout.sbatch on this host — this script
# has the same dependency surface (numpy / torch / sklearn only, zero speech_decoding import).
# <<< paste the same `module load` / venv activate line the sibling readout sbatch uses >>>

MODE="$1"
CACHE="$2"
TAG="$3"
SHARDS="${4:-}"
OUT="${5:-}"
SCRIPT="${SCRIPT:-/projects/bhqk/htang13/probe_pool_variants.py}"
BETAS="${BETAS:-0,0.5,1,2,4}"
VARIANTS="${VARIANTS:-A,B}"

case "$MODE" in
  assert)
    numactl --interleave=all python "$SCRIPT" --mode assert \
      --cache-dir "$CACHE" --tag "$TAG"
    ;;
  array)
    numactl --interleave=all python "$SCRIPT" --mode array \
      --cache-dir "$CACHE" --tag "$TAG" --shard-dir "$SHARDS" \
      --array-index "$SLURM_ARRAY_TASK_ID" --variants "$VARIANTS" --betas "$BETAS"
    ;;
  merge)
    numactl --interleave=all python "$SCRIPT" --mode merge \
      --cache-dir "$CACHE" --tag "$TAG" --shard-dir "$SHARDS" --out "$OUT" \
      --variants "$VARIANTS" --betas "$BETAS"
    ;;
  *)
    echo "unknown mode '$MODE' (assert|array|merge)" >&2
    exit 2
    ;;
esac
