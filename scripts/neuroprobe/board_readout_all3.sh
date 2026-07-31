#!/bin/bash
# Submit the whole 3-regime Neuroprobe-Lite board readout + auto-merge in one call.
#
#   usage: board_readout_all3.sh <CACHE_DIR> <TAG> <SHARD_DIR>
#
# WHY THIS EXISTS: the per-regime --mem and --workers are a CONTRACT, not preferences, and every
# value below was paid for in dead shards. Re-deriving them from the runbook prose is what keeps
# producing empty grids, so they live in code with their evidence attached.
#
#   ws        120G  workers=1 : two full 12-cell arrays completed here (20298989, 20317060),
#                               15-22 min/cell.
#   csession  200G  workers=1 : loads TWO records (cell + sibling trial). 180G OOM-killed 8 of 12
#                               (20298990); 176G completed but peaked at MaxRSS 181G (20548324),
#                               i.e. over its own request -- do not shave. 190G/210G completed.
#   cs        176G  workers=4 : MEASURED MaxRSS 107G (20548325). The runbook's "CS ~40G" is the
#                               --mmap figure and OOMs under the eager default: CS only GATHERS the
#                               parcel taps, but an eager torch.load still faults in the whole
#                               record including the 34 GB enc12_elec it never touches. Parcel taps
#                               are ~40x lighter than electrode taps, which is what buys workers=4.
#
# EAGER ON ALL THREE. MMAP_DEFAULT is False for every mode (v3_board_readout.py:601): mmap does not
# make loading free, it DEFERS it into ~180 scattered per-(task,fold,split) gathers at ~24 MB/s
# against ~86 MB/s for one sequential read -- measured 78 min, 73% of a WS shard.
#
# --cpus-per-task=16 is FREE. Delta CPU bills MAX_TRES == mem_MB/2, and the CPU term only bites
# above ~0.512*mem_GB cores (61 at 120G), so 16 bills identically to the sbatch default of 8.
# It is not raised to break-even because a 61-core request schedules far worse on a partition that
# runs ~133/136 allocated, and queue wait dominates a 20-minute shard.
#
# MERGE DEPENDS ON afterANY, NEVER afterok. Under afterok a single OOM/timeout leaves the merge
# in DependencyNeverSatisfied and it silently never runs (this stranded merge 20297526 on 07-20).
# afterany fires once every cell is terminal; a dead cell just yields a partial grid, and _merge
# unions by cell name so backfilling and re-running is idempotent.
set -euo pipefail
CACHE="${1:?CACHE_DIR}"; TAG="${2:?TAG}"; SHARD="${3:?SHARD_DIR}"
TREE=/projects/bhqk/htang13/speech_board
cd "$TREE"
[ -d "$CACHE" ] || { echo "[FATAL] cache dir missing: $CACHE"; exit 1; }
N=$(ls "$CACHE"/enc_s*_t*.pt 2>/dev/null | wc -l)
[ "$N" -eq 12 ] || { echo "[FATAL] expected 12 session records in $CACHE, found $N"; exit 1; }
mkdir -p "$SHARD"
S=scripts/neuroprobe/board_readout_lean.sbatch
WS=$( sbatch --parsable --array=0-11 --mem=120G --cpus-per-task=16 "$S" "$CACHE" "$TAG" "$SHARD" ws       1)
CSN=$(sbatch --parsable --array=0-11 --mem=200G --cpus-per-task=16 "$S" "$CACHE" "$TAG" "$SHARD" csession 1)
CS=$( sbatch --parsable --array=0-9  --mem=176G --cpus-per-task=16 "$S" "$CACHE" "$TAG" "$SHARD" cs       4)
MRG=$(sbatch --parsable --dependency=afterany:"$WS":"$CSN":"$CS" \
  --account=bhqk-delta-cpu --partition=cpu --nodes=1 --ntasks-per-node=1 \
  --cpus-per-task=2 --mem=16G --time=00:15:00 --job-name=board_merge \
  --output=/projects/bhqk/htang13/board_merge_%j.out \
  --wrap="cd $TREE && module load pytorch-conda/2.8 && python -u scripts/neuroprobe/v3_board_readout.py --cache-dir $CACHE --tags $TAG --mode merge --shard-dir $SHARD --out $SHARD/MERGED_$TAG.json")
echo "[all3] tag=$TAG ws=$WS csession=$CSN cs=$CS merge=$MRG"
echo "[all3] shards -> $SHARD    merged -> $SHARD/MERGED_$TAG.json"
