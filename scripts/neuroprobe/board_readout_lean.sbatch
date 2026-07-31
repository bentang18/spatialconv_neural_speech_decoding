#!/bin/bash
#SBATCH --job-name=board_ro
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bhqk/htang13/board_ro_%A_%a.out
# Lean per-regime Neuroprobe-Lite board readout shard. Submitted 3x (ws / csession / cs), each
# with its own --array, --mem, and WORKERS arg via the sbatch CLI, so each regime bills only the
# memory it needs and BACKFILLS into the free RAM on busy nodes (a 240G whole-node request could
# not schedule while the cpu partition was 133/136 allocated).
#
# WHY THE MEMORY FLOOR: the ridge is DUAL — Gram = Z_tr @ Z_tr.T is (n_train x n_train), so the
# feature count (enc12_elec flattens to ~1.58M = 119 elec x 13312) does NOT enter the Gram. The
# hog is the DESIGN MATRIX Z itself: n_train x 1.58M fp32 ~ 11-22GB, and a norm's standardized
# copy coexists with the raw one during the std fit (~2x). WORKERS=1 keeps exactly one Z alive,
# so peak ~ one-fit floor; workers>1 multiplies it. CS taps are parcel-mean (~40x smaller) so CS
# runs workers=4. numactl --interleave=all avoids NUMA node-0 starvation.
#
#   usage: sbatch --array=<0-N> --mem=<G> board_readout_lean.sbatch <CACHE> <TAG> <SHARD> <MODE> <WORKERS> [MMAP] [TAPS]
# TAPS (optional, comma-separated) overrides the per-regime default unit. Empty = per-regime
# default (WS/CSession electrode, CS parcel). Passing the parcel taps to ws/csession is the
# UNIT CONTROL: it puts all three regimes on the same feature unit so a regime effect can be
# told apart from a spatial-averaging effect. Shard files are named {mode}_{cell}.json with no
# tap in the name, so a tap-overridden run MUST use its own --shard-dir or it silently
# overwrites the default-unit shards.
# MMAP defaults to --no-mmap (EAGER load). The script's own MMAP_DEFAULT is False for every mode:
# eager sequential load runs ~86 MB/s and leaves gathers as fast RAM indexing, whereas --mmap DEFERS
# the read into ~180 per-(task,fold,split) scattered gathers at ~24 MB/s, which measured 78 min (73%
# of a WS shard) of Lustre re-paging. Eager needs mem >= file(~40-61G) + fp32 working slices(~44G),
# so WS/CSession run ~110-176G here, not the 48G that --mmap allowed.
#
# CSession needs elec_labels for its cross-session identity intersect (_elec_cols); the encoder
# never writes that field, so every CSession cell silently returns empty ({}) without this sidecar.
# Labels are a pure function of (channels, drop-set), independent of model weights, so the arm0
# sidecar is valid for any arm on the same 12 Lite sessions — verified 07-26: per-session electrode
# count matches exactly between the sidecar and the r6_board_40k cache for all 12 sessions.
set -euo pipefail
CACHE="${1:?CACHE_DIR}"; TAG="${2:?TAG}"; SHARD="${3:?SHARD_DIR}"
MODE="${4:?MODE ws|cs|csession}"; WK="${5:?WORKERS}"; MMAP="${6:---no-mmap}"; TAPS="${7:-}"
cd /projects/bhqk/htang13/speech_board
module load pytorch-conda/2.8
CPT="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=$(( CPT / WK ))
# BILLING: Delta CPU is MAX_TRES with TRESBillingWeights=CPU=1000,Mem=512G, and the node is
# 128 cores / 257617M ⇒ billing == mem_MB/2 and the CPU term only bites above ~0.512·mem_GB
# cores. Under that line the extra cores are FREE (we already paid for them with the memory);
# above it billing flips to the CPU term and costs MORE. Print both so a submit that leaves
# parallelism on the table -- or overshoots into a higher bill -- says so instead of going quiet.
BE=$(( ${SLURM_MEM_PER_NODE:-0} / 2000 ))
echo "[check] alloc cpus=$CPT mem=${SLURM_MEM_PER_NODE:-?}M workers=$WK omp=$OMP_NUM_THREADS; break-even cpus=$BE"
if [ "$BE" -gt 0 ] && [ "$CPT" -lt "$BE" ]; then
  echo "[check] UNDER-REQUESTED: --cpus-per-task=$BE bills the SAME as $CPT (billing is memory-bound)"
elif [ "$BE" -gt 0 ] && [ "$CPT" -gt "$BE" ]; then
  echo "[check] OVER-REQUESTED: $CPT cpus bills $(( CPT * 1000 )) vs $(( BE * 1000 )) at break-even -- this COSTS more"
fi
SIDECAR_ARGS=()
if [[ "$MODE" == "csession" ]]; then
  SIDECAR_ARGS=(--elec-labels-sidecar /projects/bhqk/htang13/arm0_elec_labels_sidecar.pkl)
fi
TAPS_ARGS=()
if [[ -n "$TAPS" ]]; then
  TAPS_ARGS=(--taps "$TAPS")
  echo "[check] TAP OVERRIDE: $TAPS (regime default NOT used) -> shard dir must be tap-specific"
fi
# Anything after TAPS is forwarded verbatim to the readout (e.g. --dump-lam-grid, --lam-rule).
# TAPS is positional and may be empty, so pass "" to skip it and still reach these.
numactl --interleave=all python -u scripts/neuroprobe/v3_board_readout.py \
  --cache-dir "$CACHE" --tags "$TAG" --mode "$MODE" --index "$SLURM_ARRAY_TASK_ID" \
  --shard-dir "$SHARD" --no-std-target --workers "$WK" "$MMAP" "${SIDECAR_ARGS[@]}" \
  "${TAPS_ARGS[@]}" "${@:8}" --out /dev/null
