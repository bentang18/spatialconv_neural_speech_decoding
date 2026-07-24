#!/bin/bash
# One-line morning launch for the electrode-pooling variants (task #41), one r6 LR arm.
#
#   ./launch_pool_variants.sh r6_10k_lr3e-3
#
# Runs, in order, with Slurm doing the waiting:
#   1. assert  — variant A at beta=0 must reproduce the cache's own enc12 tap. If this fails
#                the segment map or the reshape is wrong and NO downstream AUROC is readable,
#                so the array is chained --dependency=afterok on it and simply never starts.
#   2. array   — 13 shards (7 WS sessions + 6 CS test subjects), the standing parallel-shard rule.
#   3. merge   — afterany on the array, writes the JSON and prints the beta ladder.
#
# Absolute paths throughout (standing rule). Delta CPU only — this never touches a dtai GPU slot.
set -euo pipefail

TAG="${1:?usage: launch_pool_variants.sh <TAG>   e.g. r6_10k_lr3e-3}"
ROOT="${ROOT:-/projects/bhqk/htang13}"
CACHE="${CACHE:-$ROOT/v3_probe_cache_$TAG}"
SHARDS="${SHARDS:-$ROOT/v3_pool_shards_$TAG}"
OUT="${OUT:-$ROOT/results_pool_variants_$TAG.json}"
SB="$ROOT/pool_variants_readout.sbatch"

for p in "$CACHE" "$SHARDS" "$OUT"; do
  [[ "$p" = /* ]] || { echo "not an absolute path: $p" >&2; exit 2; }
done
[[ -d "$CACHE" ]] || { echo "cache dir does not exist: $CACHE" >&2; exit 2; }
mkdir -p "$SHARDS" "$ROOT/logs"

ASSERT=$(sbatch --parsable --mem=32G "$SB" assert "$CACHE" "$TAG")
ARRAY=$(sbatch --parsable --dependency=afterok:"$ASSERT" --array=0-12 --mem=128G \
        "$SB" array "$CACHE" "$TAG" "$SHARDS")
MERGE=$(sbatch --parsable --dependency=afterany:"$ARRAY" --mem=32G \
        "$SB" merge "$CACHE" "$TAG" "$SHARDS" "$OUT")

echo "tag     $TAG"
echo "cache   $CACHE"
echo "assert  $ASSERT   (array is afterok on this — a parity failure stops everything)"
echo "array   $ARRAY    (0-12)"
echo "merge   $MERGE -> $OUT"
echo
echo "watch:  squeue -j $ASSERT,$ARRAY,$MERGE"
echo "parity: tail -5 $ROOT/logs/pool_variants_${ASSERT}_*.out"
echo "result: grep -A40 'pooling β ladder' $ROOT/logs/pool_variants_${MERGE}_*.out"
