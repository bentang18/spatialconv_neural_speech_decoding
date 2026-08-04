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

  # space_rope is DERIVED from the ckpt path, not asked for. It is the only tower setting that
  # cannot be read off the ckpt — L1RoPE registers idx_freq persistent=False, so a nospacerope
  # ckpt is key- and value-identical to a normal one and loads into a space-rope-ON shell with
  # NO error (v3_probe_encode_r4.py:142-147). deep_sup and parcel_embed are inferred there and a
  # wrong inference fails the strict load; this one just yields a plausible, wrong AUROC.
  # Deriving it removes the human step, which is precisely the step that gets forgotten in a
  # multi-arm invocation where the flag applies to only one arm. The sbatch re-checks the
  # derivation against the ckpt name in BOTH directions and refuses to run on a mismatch, so a
  # renamed ckpt dir is a hard failure, never a silent one.
  # 🔴 BOTH SPELLINGS. The R30 geometry arms are named `nosrope` by the training launcher
  # (v3_r6_pbs50_ablation_trap45k.sbatch:86,102 -> v3_ckpt_v3_r6_pbspace111_sf50_nosrope_trap45k),
  # while this pattern only ever matched `nospacerope`. A `nosrope` ckpt therefore derived an EMPTY
  # EXTRA, and the sbatch guard -- which reads the SAME pattern -- agreed with it, so both sides
  # were wrong together and the guard passed. That is the precise silent-wrong AUROC the guard
  # exists to prevent. Any new name for this ablation MUST be added here and in the sbatch.
  local EXTRA=""
  case "$CKPT" in *nospacerope*|*nosrope*) EXTRA="--no-space-rope" ;; esac
  echo "[$TAG] encode <- $CKPT  extra='${EXTRA:-<none>}'" | tee -a "$LOG"

  local AID
  AID=$($SSH dtai "sbatch --parsable $BASE/v3_probe_encode_r6.sbatch '$CKPT' '$TAG' '$CACHE' '$EXTRA'") || {
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
  # --mem=64G MEASURED, not guessed (was 128G, which was 6.7x over). On the 30k pretrain-probe
  # cache the WS shards peak at 18.5-19.9G (job 20580085_0/1/2, MaxRSS via sacct). 64G is 3.2x
  # that -- deliberately NOT sized to ~24G, because indices 0-6 are WS (ONE session each) while
  # 7-12 are CS and _cs_shard loads TWO (anchor + test), so the measured half is the LIGHT half.
  # On the larger board-scale cache those same CS indices OOM-killed at 48G (job 20456873), which
  # is the whole reason 128G was picked; 64G keeps ~1.8x over a two-session estimate here.
  # Two reasons this is worth getting right, and only one is the bill: billing == mem_MB/2, so
  # 128G->64G halves it, but the bigger win is BACKFILL -- a 128G ask was the largest request in
  # the queue and waited for a full-size slot, while these elements run in ~90 s.
  # --cpus-per-task=32 is FREE at 64G: Delta CPU bills MAX_TRES (CPU=1000, Mem=512G) on a
  # 128-core/257617M node, so billing == mem_MB/2 == 32768 and the CPU term stays under it.
  # Break-even is 33 cores at 64G (it was 65 at 128G), so 32 is still free but now only just --
  # do NOT raise cores without raising mem, or the CPU term starts setting the bill.
  BID=$($SSH delta "sbatch --parsable --array=0-$((NARR-1)) --mem=64G --cpus-per-task=32 $BASE/v3_probe_readout_r6.sbatch array '$CACHE' '$TAG' '$SHARD'")
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
