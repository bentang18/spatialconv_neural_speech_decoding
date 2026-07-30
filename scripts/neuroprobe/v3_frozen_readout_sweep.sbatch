#!/bin/bash
#SBATCH --job-name=frozen_readout
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%A_%a.out
#
# Price the READOUT FAMILY at FROZEN features — the missing R0 term. NO GPU: this reads the
# EXISTING encode cache (built for the FT pilot's L0 parity gate), so it spends no encoder forward.
#
# WHY DELTA CPU AND NOT dtai. The whole computation is a 1279 x 212,992 standardize, one 1279^2
# eigendecomposition and a few hundred full-batch matmuls per cell. There is no fwd/bwd through the
# encoder at all, so a GPU would idle — and dtai fairshare is at 0.05, where a ghx4 slot costs
# hours of queue. 32 cores sits under 0.512 x 64 GB, so the cores are free against the memory bill.
#
# COST MODEL. The fit is DUAL (representer theorem), so per cell the only large operation is
# building the 1279 x 212,992 standardize and its 1279^2 Gram: ~0.7 TFLOP, a few seconds on 32
# cores. Every subsequent fit runs on that Gram at 1.6 MFLOP a step, so |lams| x |lrs| = 18 fits x
# 2000 steps is seconds, not hours. Cost is now dominated by loading the cache, not by fitting.
# Sharded one session per array task, so wall is flat in cohort size.
#
#   smoke (one session, one task, seconds):
#     sbatch --array=0 --time=00:10:00 v3_frozen_readout_sweep.sbatch smoke
#   full (7 sessions in parallel, then a merge report):
#     AID=$(sbatch --parsable --array=0-6 v3_frozen_readout_sweep.sbatch shard)
#     sbatch --dependency=afterany:$AID --cpus-per-task=4 --mem=8G --time=00:10:00 \
#            v3_frozen_readout_sweep.sbatch merge
set -euo pipefail

MODE="${1:?usage: sbatch v3_frozen_readout_sweep.sbatch <smoke|shard|merge>}"
BASE=/projects/bhqk/htang13
CACHE=$BASE/v3_probe_cache_pbs50_20k
TAG=pbs50_20k
SHARDS=$BASE/frozen_readout_$TAG

mkdir -p "$SHARDS"
cd $BASE/speech
# DELTA CPU, x86 -- `pytorch-conda/2.8` and the module's own `python`, NOT `.venv/bin/python`:
# the venv is aarch64 and exists only on dtai. numactl interleaving matches the readout sbatch,
# which is the runbook this borrows wholesale rather than re-deriving.
module load pytorch-conda/2.8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export ROOT_DIR_BRAINTREEBANK=$BASE/braintreebank

# A bash ARRAY, not a string: the merge glob must reach python UNEXPANDED. An earlier draft used
# an unquoted $SEL, which the shell would have expanded into one argv entry per shard file --
# argparse takes only one --merge value, so the merge would have died on its own success.
case "$MODE" in
  smoke) SEL=(--session-index 0 --tasks onset --steps 2000 --lams 1e-3,1e-1 --lrs 0.1
              --out "$SHARDS/smoke_probe.json") ;;
  shard) SEL=(--session-index "$SLURM_ARRAY_TASK_ID"
              --out "$SHARDS/s$SLURM_ARRAY_TASK_ID.json") ;;
  # 🚨 s[0-9]*, NOT s* -- `s*.json` also matches `smoke.json`, and the smoke shard is a
  # DELIBERATELY underfit 60-step run on cells the real shards also cover. Pooling it silently
  # duplicated S1T0 onset f0/f1 with their underfit values and reported 58 cells for a 56-cell
  # design. Arm mixing is the #1 defect in this project; the glob is where it got in.
  merge) SEL=(--merge "$SHARDS/s[0-9]*.json") ;;
  *) echo "FATAL: mode must be smoke|shard|merge, got '$MODE'" >&2; exit 1 ;;
esac

echo "[frozen-readout] mode=$MODE task=${SLURM_ARRAY_TASK_ID:-none} cache=$CACHE tag=$TAG"
numactl --interleave=all python -u scripts/neuroprobe/v3_frozen_readout_sweep.py \
  --cache-dir "$CACHE" --tag "$TAG" --threads "${SLURM_CPUS_PER_TASK:-8}" "${SEL[@]}"
