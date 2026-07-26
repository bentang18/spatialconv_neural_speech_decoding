#!/usr/bin/env bash
# One command, N arms, IN PARALLEL: pretrain-probe depth ladder for r6 checkpoints.
#
#   enc0/enc3/enc6/enc12  x  std  x  {WS, CS}  x  4 tasks
#
# Usage:
#   scripts/neuroprobe/probe_ckpt_r6.sh <CKPT> <TAG> [<CKPT> <TAG> ...]
# Example (the cooldown A/B — two arms, one command):
#   scripts/neuroprobe/probe_ckpt_r6.sh \
#     /projects/bhqk/htang13/v3_ckpt_v3_r6_cd5k_1sqrt_from40k/ladder-step=45000.ckpt cd1sqrt_45k \
#     /projects/bhqk/htang13/v3_ckpt_v3_r6_cd5k_linear_from40k/ladder-step=45000.ckpt cdlin_45k
#
# EVERY ARM RUNS CONCURRENTLY. Arms are separate jobs on the same fairshare — serializing
# them buys nothing and costs a full encode+readout of wall-clock per arm. Do not "queue
# them one at a time to be polite to the GPU"; that constraint does not exist.
#
# Why two clusters: ENCODE needs the GPU forward + the aarch64 venv => `ssh dtai`. READOUT is
# a model-free CPU ridge => `ssh delta`. Shared Lustre, SEPARATE schedulers, so there is no
# cross-cluster SLURM dependency — this driver polls the shared cache dir between the stages.
#
# Differences from probe_ckpt.sh (which is the r4 driver, left alone):
#   - r6 sbatch pair (v3_probe_encode_r6.sbatch / v3_probe_readout_r6.sbatch), std-only.
#   - mem passed at LAUNCH: array 128G / merge 32G. A CS shard loads two sessions and r6
#     carries the big enc0 |STFT| tap; 48G OOM-killed CS shards before.
#   - merge is `afterany`, NOT afterok: one OOM/timeout cell would leave an afterok merge
#     stuck in DependencyNeverSatisfied forever. The merge tolerates missing shards.
#   - walltime/mem are baked into the r6 encode sbatch (113560M = DefMemPerGPU so ghx4
#     MAX_TRES billing stays at 1000; 15 min earns backfill). Do not override them here.
#
# Idempotent per TAG: the encode skips sessions whose cache exists, so a re-run is cheap —
# but that also means REUSING A TAG SILENTLY NO-OPS. New ckpt => new tag.
set -uo pipefail

(( $# >= 2 && $# % 2 == 0 )) || {
  echo "usage: probe_ckpt_r6.sh <CKPT> <TAG> [<CKPT> <TAG> ...]" >&2; exit 2; }

BASE=/projects/bhqk/htang13
NSESS=7    # PROBE_COHORT_7
NARR=13    # 7 WS sessions + 6 CS test subjects
SSH="ssh -o ConnectTimeout=15"   # macOS has no GNU `timeout`; never leave a bare ssh to hang
RUNLOG=$(mktemp -d "${TMPDIR:-/tmp}/probe_r6.XXXXXX")

run_arm() {
  local CKPT=$1 TAG=$2 LOG=$RUNLOG/$2.log
  local CACHE=$BASE/v3_probe_cache_$TAG SHARD=$BASE/v3_probe_shards_$TAG
  local RESULT=$BASE/results_v3_probe_$TAG.json
  echo "[$TAG] encode <- $CKPT" | tee -a "$LOG"

  local AID
  AID=$($SSH dtai "sbatch --parsable $BASE/v3_probe_encode_r6.sbatch '$CKPT' '$TAG' '$CACHE'") || {
    echo "[$TAG] !! encode submit failed" | tee -a "$LOG"; return 1; }
  echo "[$TAG] encode job $AID -> $CACHE" | tee -a "$LOG"

  local n st
  while :; do
    n=$($SSH dtai "ls $CACHE/*.pt 2>/dev/null | wc -l" | tr -d ' ')
    [ "${n:-0}" -ge "$NSESS" ] && { echo "[$TAG] caches $n/$NSESS" | tee -a "$LOG"; break; }
    st=$($SSH dtai "squeue -j $AID -h -o %t 2>/dev/null" | tr -d ' ')
    [ -z "$st" ] && { echo "[$TAG] !! encode $AID ended at $n/$NSESS — see $BASE/probe_encode_r6_$AID.out" | tee -a "$LOG"; return 1; }
    sleep 25
  done

  $SSH delta "mkdir -p $SHARD && rm -f $SHARD/*.json" >/dev/null 2>&1
  local BID MID
  BID=$($SSH delta "sbatch --parsable --array=0-$((NARR-1)) --mem=128G $BASE/v3_probe_readout_r6.sbatch array '$CACHE' '$TAG' '$SHARD'")
  MID=$($SSH delta "sbatch --parsable --dependency=afterany:$BID --mem=32G $BASE/v3_probe_readout_r6.sbatch merge '$CACHE' '$TAG' '$SHARD' '$RESULT'")
  echo "[$TAG] readout array=$BID merge=$MID (afterany)" | tee -a "$LOG"

  while :; do
    [ "$($SSH delta "test -s $RESULT && echo 1 || echo 0")" = "1" ] && break
    [ -z "$($SSH delta "squeue -j $MID -h -o %t 2>/dev/null" | tr -d ' ')" ] && {
      echo "[$TAG] !! merge $MID exited with no result json" | tee -a "$LOG"; return 1; }
    sleep 20
  done
  echo "[$TAG] merged -> $RESULT" | tee -a "$LOG"
  $SSH delta "sed -n '/=== r4 depth ladder/,\$p' $BASE/logs/r6_probe_readout_${MID}_*.out" >> "$LOG" 2>/dev/null
  return 0
}

declare -a TAGS=()
while (( $# )); do
  run_arm "$1" "$2" &
  TAGS+=("$2")
  shift 2
done
wait

for T in "${TAGS[@]}"; do
  echo; echo "########## $T ##########"
  cat "$RUNLOG/$T.log" 2>/dev/null
done
echo; echo "logs: $RUNLOG"
