#!/usr/bin/env bash
# One-command depth-ladder probe of an r4 (v14_converged_v3 Design B) checkpoint.
#
#   enc0/enc3/enc6/enc12  x  {std, raw}  x  {WS, CS}  x  4 tasks
#
# Encode runs on DeltaAI GPU (`ssh dtai`, aarch64 venv, needs the teacher forward);
# readout runs on Delta CPU (`ssh delta`, 13-task array @ 8 cores each — the ZZ^T GEMM
# is bandwidth-bound and peaks at 8 threads). The two clusters share Lustre but have
# separate schedulers, so this driver submits encode, polls the shared cache dir, then
# submits the readout array + a dependent merge. Run it from a shell that has both the
# `dtai` and `delta` ssh aliases (i.e. Ben's laptop).
#
# Usage:
#   scripts/neuroprobe/probe_ckpt.sh <CKPT_PATH_ON_DTAI> <TAG>
# Example:
#   scripts/neuroprobe/probe_ckpt.sh /projects/bhqk/htang13/v3_ckpt_r4/ladder-step=25000.ckpt r4_25k
#
# Idempotent: encode skips sessions whose cache already exists, so re-running a TAG is cheap.
# Long-running (~10-15 min); nohup it if you want to detach.
set -euo pipefail

CKPT="${1:?usage: probe_ckpt.sh <CKPT_ON_DTAI> <TAG>}"
TAG="${2:?usage: probe_ckpt.sh <CKPT_ON_DTAI> <TAG>}"

BASE=/projects/bhqk/htang13
CACHE=$BASE/v3_probe_cache_${TAG}
SHARD=$BASE/v3_probe_shards_${TAG}
RESULT=$BASE/results_v3_probe_${TAG}.json
NSESS=7    # PROBE_COHORT_7
NARR=13    # 7 WS sessions + 6 CS test subjects

echo "== [1/3] ENCODE on dtai (ckpt=$CKPT tag=$TAG) =="
# --time override: the sbatch defaults to 3 h, which is ~6x the real job and too fat to
# backfill — ghx4 sits at 40 nodes drain$ and a 3 h ask queued ~2 h behind Priority. The
# r4_10k encode wrote all 7 caches in a 4-min window (03:00:39→03:04:36); 30 min is 6x
# that with room for model load + spec-cache reads. Safe to run tight: the encode SKIPS
# sessions whose cache exists, so a walltime kill costs one requeue, not the work.
AID=$(ssh dtai "sbatch --parsable --time=00:30:00 $BASE/v3_probe_encode_r4.sbatch '$CKPT' '$TAG' '$CACHE'")
echo "   encode job $AID  ->  $CACHE"
while :; do
  n=$(ssh dtai "ls $CACHE/*.pt 2>/dev/null | wc -l" | tr -d ' ')
  [ "$n" -ge "$NSESS" ] && { echo "   caches complete ($n/$NSESS)"; break; }
  st=$(ssh dtai "squeue -j $AID -h -o %t 2>/dev/null" | tr -d ' ')
  echo "   caches=$n/$NSESS encode=[${st:-done}]"
  if [ -z "$st" ]; then
    echo "!! encode job $AID ended with only $n/$NSESS caches — see $BASE/probe_encode_${AID}.out" >&2
    exit 1
  fi
  sleep 30
done

echo "== [2/3] READOUT on delta (13-task array @ 8 cores + merge) =="
ssh delta "mkdir -p $SHARD && rm -f $SHARD/*.json"
# --time override: the DCC readout_array.sbatch defaults to 30 min, which the biggest
# sessions' fp64 CS n×n solve blows past (4 shards TIMEOUT'd at 30:24 on the r4_10k run,
# breaking the afterok merge). 2 h gives ~4× headroom; readout is CPU-cheap so idle-cost
# is bounded by the actual solve, not the ceiling.
BID=$(ssh delta "sbatch --parsable --time=02:00:00 $BASE/readout_array.sbatch '$CACHE' '$TAG' '$SHARD'")
MID=$(ssh delta "sbatch --parsable --dependency=afterok:$BID $BASE/readout_merge.sbatch '$SHARD' '$TAG' '$RESULT' '$CACHE'")
echo "   array $BID, merge $MID (afterok:$BID)"
MOUT=$BASE/probe_merge_${MID}.out
while :; do
  if ssh delta "grep -q MERGE_DONE $MOUT 2>/dev/null"; then break; fi
  ns=$(ssh delta "ls $SHARD/ws_*.json $SHARD/cs_*.json 2>/dev/null | wc -l" | tr -d ' ')
  bst=$(ssh delta "squeue -j $BID -h -o %t 2>/dev/null | sort -u | tr '\n' ' '")
  mst=$(ssh delta "squeue -j $MID -h -o %t 2>/dev/null" | tr -d ' ')
  echo "   shards=$ns/$NARR array=[${bst:-done}] merge=[${mst:-pending}]"
  if [ -z "$bst" ] && [ "$ns" -lt "$NARR" ] && [ -z "$mst" ]; then
    echo "!! array $BID finished with $ns/$NARR shards; merge dependency failed — see $BASE/probe_arr_${BID}_*.out" >&2
    exit 1
  fi
  sleep 20
done

echo "== [3/3] DEPTH LADDER (tag=$TAG) =="
ssh delta "sed -n '/=== r4 depth ladder/,\$p' $MOUT"
echo "   full json: $RESULT (on delta/dtai shared Lustre)"
