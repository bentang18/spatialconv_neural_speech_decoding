#!/bin/bash
#SBATCH --job-name=r6_cb_memprobe
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=113560M
#SBATCH --time=00:40:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
#
# WHAT: measure the LARGEST token-exact contact-budget that fits at v=0.50 (--mask-space-frac 0.0),
# so the maskspace0 race runs at the real ceiling instead of at an extrapolation.
#
# WHY THIS EXISTS. The fitted model GiB ~ 39.0 + 136.8*v (at cb11000) says v=0.50/cb11000 needs
# 107 GiB on a 95 GiB card, which is why cb was cut to 5500. But the LIVE ms0 job reads only
# 54.6 GiB at cb5500/v=0.50, where that model predicts 73.2 -- it OVER-predicts by ~25%. Worse,
# the two memory points I have do not fit its functional form at all: solving A + B*v*(cb/11000)
# through (cb11000, v=.375, 90.3) and (cb5500, v=.50, 54.6) gives a NEGATIVE intercept, which is
# physically impossible for "weights + Adam states". So the model cannot answer "does cb11000 fit"
# -- its honest range for that cell is ~70-109 GiB, i.e. straddling the card. Measure instead.
#
# WRITES NOTHING, TOUCHES NO ARM. --ckpt-ladder-every 0 and no --ckpt-dir => the ModelCheckpoint
# callback is never constructed (the guard at dispatch_v3.py:427 is `ckpt_ladder_every > 0 and
# ckpt_dir`). No --wandb-project => default None => the WandbLogger branch at :441 never fires and
# `logger` stays False. So this cannot interleave a ladder or appear in the intuition project.
#
# EXACT PRODUCTION PATH, --compile INCLUDED. Inductor's peak is not the eager peak and the decision
# is about the production config. Compile is also the dominant wall cost here -- but two of the four
# rungs (5500, 11000) should hit a WARM TORCHINDUCTOR_CACHE_DIR, since the live ms0 job and the
# cb11000 keeper already compiled those exact shapes into the same shared cache.
#
# RUNG 1 IS A CONTROL, NOT A DATA POINT. cb5500 is KNOWN to fit -- job 2764373 is running it right
# now and nvidia-smi reads 54.6 GiB. If this harness does not reproduce ~54.6 GiB there, then it is
# not measuring the production footprint and its verdict on cb11000 must be DISCARDED, not acted on.
# This is the whole reason the control costs a rung: without it, a "FITS" is unfalsifiable.
#
# TOKEN-EXACT ONLY. Every rung satisfies cb x accum == 66000 contacts per OPTIMIZER step, asserted
# per rung, so each is a drop-in for the others with the same gradient up to fp accumulation order.
# An "OFAT" that changes contacts/step is not an OFAT.
#
# WHY A SHORT RUN IS INFORMATIVE HERE (it would not be for a padded loader). shaft_batch.py:138-147
# packs OVERFILL-AND-TRIM: it draws shafts past the budget then trims the last so Sum(n_keep) ==
# contact_budget EXACTLY, "pinned to ONE COMPILED SHAPE with ZERO padding". So the contact count is
# deterministic every micro-batch and the dominant activation term does not vary run-to-run. The
# ONE residual is shaft COMPOSITION: attention is within-shaft, so cost goes as Sum(n_s^2) at fixed
# Sum(n_s), and a pack of few long shafts is heavier than many short ones. That is why STEPS is 20
# and not 2 -- 120-240 micro-batches is enough to sample an unlucky composition. It is still not
# 240k of them, which is why the decision rule below is OCCUPANCY, not a bare FITS/FAILED.
set -uo pipefail
module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bhqk/htang13/torchinductor_cache_r6
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
LOGD=/projects/bhqk/htang13/logs
mkdir -p "$LOGD"

CARD_MIB=97871          # ghx4 GH200, ~95.6 GiB
BUDGET_S=$((33 * 60))   # do not START a rung after this; walltime is 40 min
STEPS=20                # see the composition note above

RESULTS=""

probe_one() {
  local CB=$1 ACC=$2
  if [ $((CB * ACC)) -ne 66000 ]; then
    echo "  SKIP cb=$CB accum=$ACC -- NOT token-exact ($((CB*ACC)) != 66000)"; return
  fi
  local RLOG="$LOGD/memprobe_${SLURM_JOB_ID:-local}_cb${CB}.log"
  echo "=============================================================="
  echo "[rung] cb=$CB accum=$ACC  tokens/step=$((CB*ACC))  steps=$STEPS  log=$RLOG"

  # Peak via nvidia-smi polling. There is NO memory in the v3 heartbeat (dispatch_v3.py:148-154
  # prints step/sec/elapsed only; max_memory_allocated lives in the v14_joint/v14_converged
  # modules, not this path), and editing the shared tree is out -- live dtai jobs import it. This
  # is also the metric that matters: it counts the CUDA context and fragmentation, which
  # torch.cuda.max_memory_allocated() excludes but which is what actually OOMs. With
  # expandable_segments the allocator does not return memory to the driver, so memory.used is
  # effectively a running max and a 1 s poll cannot miss the peak by much.
  local MAXF; MAXF=$(mktemp); echo 0 > "$MAXF"
  ( while :; do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      if [ -n "${u:-}" ] 2>/dev/null; then
        m=$(cat "$MAXF" 2>/dev/null || echo 0)
        if [ "$u" -gt "$m" ] 2>/dev/null; then echo "$u" > "$MAXF"; fi
      fi
      sleep 1
    done ) & local POLL=$!

  local t0=$SECONDS
  "$VENV" -u -m speech_decoding.experiments.dispatch_v3 \
    --bt-root /projects/bhqk/htang13/braintreebank \
    --band-cache-dir $ROOT/band_v3slow \
    --band-cache-dir $ROOT/band_v3mid \
    --band-cache-dir $ROOT/band_hga \
    --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
    --session 1:0 --session 2:1 --session 2:2 --session 2:3 --session 2:5 \
    --session 2:6 --session 3:2 --session 4:2 --session 6:0 --session 6:1 \
    --session 6:4 --session 8:0 --session 9:0 \
    --frontend v3r6 --objective mae --mask-space-frac 0.0 \
    --batch-unit shaft --contact-budget "$CB" --shaft-alpha 0.5 \
    --lr 6e-3 --weight-decay 0.04 --grad-clip 3.0 --adam-beta2 0.95 --seed 33 \
    --clip-len 2.0 \
    --session-z-winsor-slow 15 --session-z-winsor-mid 15 --session-z-winsor-hga 20 \
    --batch-size 32 --accumulate-grad-batches "$ACC" \
    --accelerator gpu --devices 1 --precision bf16-mixed \
    --num-workers 12 --monitor-every-n-steps 1 --log-every-n-steps 1 \
    --compile \
    --ckpt-ladder-every 0 \
    --ssl-max-steps "$STEPS" > "$RLOG" 2>&1
  local rc=$?
  local dt=$((SECONDS - t0))
  kill "$POLL" 2>/dev/null; wait "$POLL" 2>/dev/null
  local peak; peak=$(cat "$MAXF" 2>/dev/null || echo 0); rm -f "$MAXF"

  local oom=""
  grep -qiE "out of memory|OutOfMemoryError" "$RLOG" && oom=" (CUDA OOM)"

  local verdict pct gib hb
  gib=$(awk "BEGIN{printf \"%.1f\", $peak/1024}")
  pct=$(awk "BEGIN{printf \"%.0f\", 100*$peak/$CARD_MIB}")
  if [ "$rc" -eq 0 ] && [ -z "$oom" ]; then verdict="FITS"; else verdict="FAILED$oom"; fi

  # Throughput, for free -- and derived from the elapsed= DELTA, deliberately NOT from
  # sec_per_step. That field's semantics are ambiguous across trees: the callback measures _sec
  # across `step != _last_step` transitions where step is trainer.global_step, i.e. PER OPTIMIZER
  # STEP after task #38's fix, but the incumbent race launcher's own header still multiplies it by
  # accum ("~0.362 s x accum 12"), i.e. assumes PER MICRO-BATCH. I do not know which version is
  # deployed at /projects/bhqk/htang13/speech, and guessing wrong is a 6-12x error.
  #   (elapsed_last - elapsed_first) / (step_last - step_first)
  # is immune to that: elapsed is wall seconds since the first batch and step is the optimizer
  # step, so the slope is seconds per OPTIMIZER step under either reading. Warmup/compile lives in
  # the first heartbeats, so the window starts at step > 5.
  hb=$(awk '
    match($0, /step=[0-9]+/)      { s = substr($0, RSTART+5, RLENGTH-5) + 0 }
    match($0, /elapsed=[0-9]+s/)  { e = substr($0, RSTART+8, RLENGTH-9) + 0
                                    if (s > 5) { if (s0 == 0) { s0 = s; e0 = e } ; s1 = s; e1 = e } }
    END { if (s1 > s0) printf "%.2f", (e1 - e0) / (s1 - s0); else printf "n/a" }
  ' "$RLOG" 2>/dev/null)

  echo "[rung] cb=$CB accum=$ACC -> $verdict  peak=${gib} GiB (${pct}% of card)  ~${hb} s/opt-step  rc=$rc  wall=${dt}s"
  [ "$verdict" != "FITS" ] && { echo "    ---- tail of $RLOG ----"; tail -15 "$RLOG" | sed "s/^/    | /"; }
  LAST_VERDICT="$verdict"
  RESULTS="${RESULTS}${CB}|${ACC}|${verdict}|${gib}|${pct}|${hb}|${dt}
"
}

# RUNGS ARE AN ARGUMENT: a comma-separated cb list, accum DERIVED as 66000/cb.
#   sbatch v3_r6_cb_memprobe.sbatch 8250,6600
# Default is the first sweep. Pass a list to bisect: the first sweep found cb5500 FITS at 57% and
# cb11000 OOM at 99%, so the max lives strictly between them, and the only token-exact divisors of
# 66000 in that gap are 6000 (accum 11), 6600 (10) and 8250 (8). Ordered DESCENDING when bisecting
# so the most valuable rung runs first -- an OOM rung dies in ~84 s while a fitting rung costs
# ~270 s, so a descending order also fails cheap and succeeds last.
RUNGS="${1:-5500,11000,13200,16500}"
read -ra LADDER <<< "${RUNGS//,/ }"
LAST_VERDICT=""
for i in "${!LADDER[@]}"; do
  CB=${LADDER[$i]}
  case "$CB" in ''|*[!0-9]*) echo "  SKIP cb='$CB' -- not a positive integer"; continue ;; esac
  if [ "$((66000 % CB))" -ne 0 ]; then
    echo "  SKIP cb=$CB -- 66000 % $CB = $((66000 % CB)) != 0, no integer accum is token-exact"
    continue
  fi
  set -- "$CB" "$((66000 / CB))"
  if [ "$SECONDS" -gt "$BUDGET_S" ]; then
    echo "[budget] ${SECONDS}s > ${BUDGET_S}s -- NOT starting cb=$1. A timeout bills in FULL and"
    echo "         returns nothing, while an unstarted rung is merely an unknown. Re-run for it."
    continue
  fi
  probe_one "$1" "$2"

  # EARLY EXIT ON A DESCENDING BISECT. Peak memory is monotone in cb at fixed v (the activation
  # term scales with cb, nothing else changes), so once a rung FITS every SMALLER rung must fit
  # too and running them proves nothing. Only break when the rungs still ahead are all smaller --
  # on the default ASCENDING ladder the first FITS is cb5500 and breaking there would abandon the
  # whole sweep, which is the bug this condition exists to prevent.
  if [ "$LAST_VERDICT" = "FITS" ] && [ "$((i + 1))" -lt "${#LADDER[@]}" ]; then
    desc=1
    for j in $(seq $((i + 1)) $((${#LADDER[@]} - 1))); do
      [ "${LADDER[$j]}" -lt "$CB" ] 2>/dev/null || { desc=0; break; }
    done
    if [ "$desc" = "1" ]; then
      echo "[early-exit] cb=$CB FITS and every remaining rung (${LADDER[*]:$((i+1))}) is SMALLER."
      echo "             Peak memory is monotone in cb at fixed v, so those all fit by implication."
      echo "             cb=$CB IS THE MAX of this ladder. Not burning GPU to re-prove monotonicity."
      break
    fi
  fi
done

echo
echo "=============================================================="
echo "SUMMARY  (card = $CARD_MIB MiB = 95.6 GiB; every rung token-exact at 66000 contacts/step)"
printf "  %-7s %-6s %-18s %-9s %-6s %-13s %s\n" cb accum verdict peak_GiB pct s_per_optstep wall_s
printf "%s" "$RESULTS" | while IFS='|' read -r cb acc v g p h d; do
  [ -z "$cb" ] && continue
  printf "  %-7s %-6s %-18s %-9s %-6s %-13s %s\n" "$cb" "$acc" "$v" "$g" "$p" "$h" "$d"
done
cat <<'EOF'

HOW TO READ THIS, IN ORDER:

1. CONTROL FIRST. cb5500 must say FITS at ~54.6 GiB -- that is what nvidia-smi reads on the live
   ms0 job 2764373. If it does not, this harness is not measuring the production footprint and
   NOTHING below it means anything. Discard the run, do not act on it.

2. THEN OCCUPANCY, NOT "FITS". A rung that fits at 20 steps has sampled ~120-240 micro-batches; a
   40k run sees ~240k. Contact count is exact every batch (overfill-and-trim), so the risk is not
   drift -- it is an unlucky shaft composition, since within-shaft attention costs Sum(n_s^2) at
   fixed Sum(n_s). Read the peak as a floor on the true worst case:
     <= 85%  -> safe, take it.
     86-93%  -> probably fine, and it is the regime maskspace25 already survived for 40k at 95%.
     >= 94%  -> the same tightness we were already nervous about. An OOM at hour 20 costs the arm.
   The incumbent cb5500 is still running as the free fallback either way, so a tight rung is a
   calculated risk, not a bet-the-arm one.

3. THEN THE THROUGHPUT COLUMN, sceptically. It is a median over ~17 heartbeats and the extrapolated
   target is ~3.07 s/step at cb11000. Above cb11000 the gain flattens hard -- the memo's own table
   puts 8250->11000 at ~5% against 5500->8250 at +36% -- so if 13200/16500 clear, they are a bonus,
   NOT a reason to re-race twice. Re-racing costs a queue wait and discards banked steps.

4. WHAT TO DO. If cb11000 clears at a sane occupancy, the ms0 race (2773232, currently cb8250)
   should be rebuilt at the largest sane rung. That means superseding a job Ben already authorised,
   so it is his call, not an automatic follow-through.
EOF
