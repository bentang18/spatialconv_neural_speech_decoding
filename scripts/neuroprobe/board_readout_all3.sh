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
#
# 4th arg = ENC12_ONLY. Set it for an ABLATION arm that only needs the enc12 row in all three
# regimes: ws/csession fit enc12_elec ALONE (not 2 taps) and cs fits enc12 ALONE (not 5). Wall is
# close to linear in tap count and the Delta CPU bill is mem_MB/2 x WALL, so this is a ~2x saving
# on ws/csession and ~5x on cs. It is strictly less work, with no estimate involved.
#
# 🔴 --mem DELIBERATELY UNCHANGED, and the reason is a correction worth keeping. The matching
# encode drops enc0_elec, and I first sized the memory down on the theory that this shrinks the
# record by ~37%. IT DOES NOT. Tap WIDTHS are not uniform: the encoder taps are d_model-wide
# (13312) while enc0 is the raw |STFT| width (348), so per window a record is
#   enc0 16x348 + 4 parcel taps 16x13312 + enc12_elec 119x13312 = 2,483,076 halves
# and enc0_elec would add only 119x348 = 41,412, i.e. **1.7% of a record, not 37%**. Dropping it
# is still right (it never reads weights, so it is arm-identical by construction) but it buys
# disk and encode time, NOT readout memory. The eager load still faults in the whole ~52-66G
# record, so the proven 120G/200G/176G floor stands exactly as measured.
# ⚠️ Restricting taps does not lower the peak either: workers=1 keeps ONE design matrix alive, so
# peak = record + one Z regardless of how many taps are fit in sequence.
set -euo pipefail
CACHE="${1:?CACHE_DIR}"; TAG="${2:?TAG}"; SHARD="${3:?SHARD_DIR}"; ENC12="${4:-}"
TREE=/projects/bhqk/htang13/speech_board
cd "$TREE"
[ -d "$CACHE" ] || { echo "[FATAL] cache dir missing: $CACHE"; exit 1; }
N=$(ls "$CACHE"/enc_s*_t*.pt 2>/dev/null | wc -l)
[ "$N" -eq 12 ] || { echo "[FATAL] expected 12 session records in $CACHE, found $N"; exit 1; }
mkdir -p "$SHARD"
# The lean sbatch lives NEXT TO the caches, not in the tree: it is deployed to
# /projects/bhqk/htang13/ and the git copy under scripts/neuroprobe/ is the reconciled record, not
# what Slurm reads. Referencing it relative to $TREE fails with "Unable to open file".
S=/projects/bhqk/htang13/board_readout_lean.sbatch
[ -f "$S" ] || { echo "[FATAL] lean sbatch missing: $S"; exit 1; }
M_WS=120G; M_CSN=200G; M_CS=176G     # measured floor; see the enc0_elec width note above
if [ -n "$ENC12" ]; then
  T_ELEC=enc12_elec; T_PARC=enc12
  echo "[all3] ENC12-ONLY: taps ws/csession=$T_ELEC cs=$T_PARC | mem UNCHANGED $M_WS/$M_CSN/$M_CS"
else
  T_ELEC=""; T_PARC=""
fi
WS=$( sbatch --parsable --array=0-11 --mem=$M_WS  --cpus-per-task=16 "$S" "$CACHE" "$TAG" "$SHARD" ws       1 --no-mmap "$T_ELEC")
CSN=$(sbatch --parsable --array=0-11 --mem=$M_CSN --cpus-per-task=16 "$S" "$CACHE" "$TAG" "$SHARD" csession 1 --no-mmap "$T_ELEC")
CS=$( sbatch --parsable --array=0-9  --mem=$M_CS  --cpus-per-task=16 "$S" "$CACHE" "$TAG" "$SHARD" cs       4 --no-mmap "$T_PARC")
MRG=$(sbatch --parsable --dependency=afterany:"$WS":"$CSN":"$CS" \
  --account=bhqk-delta-cpu --partition=cpu --nodes=1 --ntasks-per-node=1 \
  --cpus-per-task=2 --mem=16G --time=00:15:00 --job-name=board_merge \
  --output=/projects/bhqk/htang13/board_merge_%j.out \
  --wrap="cd $TREE && module load pytorch-conda/2.8 && python -u scripts/neuroprobe/v3_board_readout.py --cache-dir $CACHE --tags $TAG --mode merge --shard-dir $SHARD --out $SHARD/MERGED_$TAG.json")
echo "[all3] tag=$TAG ws=$WS csession=$CSN cs=$CS merge=$MRG"
echo "[all3] shards -> $SHARD    merged -> $SHARD/MERGED_$TAG.json"
