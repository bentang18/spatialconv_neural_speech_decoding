#!/bin/bash
# 3-regime readout for the 64 Hz HGA question, two arms, plus merges.
#
#   usage: enc0_hga64_readout.sh <CACHE_DIR> <TAG> <SHARD_ROOT>
#
# NOT board_readout_all3.sh. That driver hardcodes 120G/200G/176G, sized for a 44-71 GB board
# record whose enc12_elec flattens to ~1.58M features, and it forwards no TAPS argument. These
# records are enc0-only (~1.3 GB) and every arm here NEEDS a tap override, because the regime
# defaults are enc12_elec/enc12 and this cache carries neither.
#
# THE TWO ARMS ARE THE EXPERIMENT:
#   enc0              -- HGA at its native 64 Hz (band_lengths 4,16,64)
#   fm:hga64t32:enc0  -- the SAME bake subsampled back to 32 Hz, i.e. same 31.25 ms window, same
#                        4 bins, same everything except rate.
# The published 32 Hz enc0 differs from both in window AND bins, so it cannot isolate rate on its
# own. Arm 2 is what turns "64 Hz is better" into "the RATE is what is better" -- without it a
# win is equally explainable by the narrower window. Note arm 2 is a subsampling control and is
# NOT the kernel-2/stride-2 conv escape hatch: a learned stride-2 conv can encode within-pair
# change, which this deliberately discards.
#
# MEMORY. The ridge is dual, so the feature count does not enter the Gram; the hog is the design
# matrix. enc0_elec here is 120 x 380 = 45,600 features, so Z at n_train~10.9k is ~2.0 GB fp32
# and ~4 GB with the standardized copy alongside, plus a 1.3 GB eager record. 32G is that with
# headroom, against the 120-200G the full-record driver needs. CS taps are parcel-mean (23 x 380)
# and ~40x lighter, which is what buys workers=4.
# --cpus-per-task is set to each regime's break-even (mem_GB/2): under it the cores are free,
# over it billing flips to the CPU term and costs more.
#
# SHARD DIRS ARE PER ARM. Shard files are named {mode}_{cell}.json with no tap in the name, so
# two arms sharing a shard dir silently overwrite each other and the merge reads whichever landed
# last -- a complete, plausible grid answering the wrong question.
set -euo pipefail
CACHE="${1:?CACHE_DIR}"; TAG="${2:?TAG}"; ROOT="${3:?SHARD_ROOT}"
TREE=/projects/bhqk/htang13/speech_board
S=/projects/bhqk/htang13/board_readout_lean.sbatch
[ -f "$S" ] || { echo "[FATAL] lean sbatch missing: $S"; exit 1; }
[ -d "$CACHE" ] || { echo "[FATAL] cache dir missing: $CACHE"; exit 1; }
N=$(ls "$CACHE"/enc_s*_t*.pt 2>/dev/null | wc -l)
[ "$N" -eq 12 ] || { echo "[FATAL] expected 12 session records in $CACHE, found $N"; exit 1; }

submit_arm () {
  local ARM="$1" SLUG="$2" SHARD="$ROOT/$SLUG"
  mkdir -p "$SHARD"
  # ws/csession run on the ELECTRODE unit, cs on the PARCEL unit -- the same per-regime unit the
  # published enc0 numbers were computed on. Changing the unit here would make the comparison a
  # unit contrast rather than a rate contrast.
  local WS CSN CS MRG
  WS=$( sbatch --parsable --array=0-11 --mem=32G --cpus-per-task=16 "$S" "$CACHE" "$TAG" "$SHARD" ws       1 --no-mmap "${ARM}_elec")
  CSN=$(sbatch --parsable --array=0-11 --mem=32G --cpus-per-task=16 "$S" "$CACHE" "$TAG" "$SHARD" csession 1 --no-mmap "${ARM}_elec")
  CS=$( sbatch --parsable --array=0-9  --mem=24G --cpus-per-task=12 "$S" "$CACHE" "$TAG" "$SHARD" cs       4 --no-mmap "${ARM}")
  # afterANY, never afterok: under afterok one OOM or timeout leaves the merge in
  # DependencyNeverSatisfied and it silently never runs. afterany fires once every cell is
  # terminal, a dead cell just yields a partial grid, and the merge unions by cell name so
  # backfilling is idempotent.
  MRG=$(sbatch --parsable --dependency=afterany:"$WS":"$CSN":"$CS" \
    --account=bhqk-delta-cpu --partition=cpu --nodes=1 --ntasks-per-node=1 \
    --cpus-per-task=2 --mem=16G --time=00:15:00 --job-name="mrg_$SLUG" \
    --output=/projects/bhqk/htang13/merge_"$SLUG"_%j.out \
    --wrap="cd $TREE && module load pytorch-conda/2.8 && python -u scripts/neuroprobe/v3_board_readout.py --cache-dir $CACHE --tags $TAG --mode merge --shard-dir $SHARD --out $SHARD/MERGED_$SLUG.json")
  echo "[arm] $SLUG taps=$ARM ws=$WS csession=$CSN cs=$CS merge=$MRG -> $SHARD/MERGED_$SLUG.json"
}

submit_arm "enc0"             "hga64_native64"
submit_arm "fm:hga64t32:enc0" "hga64_sub32"
