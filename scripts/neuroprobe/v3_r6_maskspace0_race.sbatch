#!/bin/bash
#SBATCH --job-name=v3_r6_ms0_race
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
#
# RACE maskspace0 at a MEASURED occupancy. This is v3_r6_maskspace0_lr6e-3_40k_cb5500.sbatch
# VERBATIM with exactly three deltas: contact-budget/accum taken from $1, and a --resume-ckpt off
# the running arm's ladder. Everything else is byte-copied, deliberately.
#
#   usage: sbatch v3_r6_maskspace0_race.sbatch <CONTACT_BUDGET>
#          accum is DERIVED as 66000/CB and both are asserted below.
#
# WHY cb IS AN ARGUMENT AND NOT A LITERAL. The first cut of this file hardcoded 8250, chosen by
# solving the fitted model GiB ~ 39.0 + 136.8*v for "the occupancy maskspace25 already survived".
# That model is now known to be WRONG in this regime -- job 2773794 measured cb5500/v=0.50 at
# 54.6 GiB where the model predicts 73.2, a 34% over-prediction, confirmed against the live ms0
# job reading the identical 54.6. So the right cb is whatever 2773794 MEASURED as the largest
# token-exact rung at a sane occupancy, and baking a solved-from-a-broken-model literal into the
# filename is exactly the mistake that cost this arm 6k steps the first time. Pass the measured
# value; do not re-derive it.
#
# TOKEN-EXACT, so this is NOT a config change to the experiment: CB x accum == 66000 contacts per
# OPTIMIZER step, same as the incumbent's 5500 x 12. Micro-batches are equal-sized within each
# config, so mean-of-means == global mean and the gradient is the same up to fp accumulation
# order. The assert below is the thing that makes the switch legitimate mid-run; if it fails, the
# arm is no longer comparable and the job MUST NOT start.
#
# HANDOFF, NOT A PERMANENT RACE (Ben 07-29). Two arms on one question is duplication, not
# insurance. The incumbent 2764373 stays up ONLY until this job is running and past its memory
# peak; then it is cancelled by hand. That ordering matters and must not be inverted: this job
# picks its --resume-ckpt at START, so every hour the incumbent keeps laddering while this one
# pends is banked progress this one inherits for free. Cancel the incumbent early and you throw
# those steps away.
#
# SEPARATE CKPT_DIR IS LOAD-BEARING. Both jobs write ladder-step=<N>.ckpt every 1000 steps. If
# they shared a directory they would interleave ladders from two different throughput regimes
# and a later probe would silently read a MIXED ladder -- the cross-ckpt slip that already cost
# us one wrong csession number. So this writes its own dir and reads the incumbent's read-only.
#
# EVERY CONFIG FLAG IS RE-PASSED ON PURPOSE. Nothing about the masking config lives in the
# state_dict, and dispatch_v3.py:546 documents the precedent: L1RoPE.idx_freq is
# persistent=False, so a resume that drops a flag silently restores the DEFAULT. Drop
# --mask-space-frac 0.0 here and this arm silently becomes the 0.50 baseline and looks like a
# null result.
#
# LR IS CONSTANT AFTER WARMUP -- do not repeat the mistake the incumbent's header makes. Its
# "SCHEDULE PARITY NOTE" claims no --stable-steps means LR decays over total_steps, so 30k and
# 40k arms would sit at different LRs at a shared step. That is FALSE: --min-lr-ratio defaults
# to 1.0 (dispatch_v3.py:551, "constant LR (min_lr_ratio=1.0, V-JEPA-2 style)" at :448), and
# _multiplier returns min_lr_ratio + (1-min_lr_ratio)*shaped == 1.0 at every post-warmup step
# (lr_schedule.py:88). This is WSD with the D done separately -- the run IS the stable phase and
# cooldown is a forked run (cd5k_*_from40k, which could not exist if the main run already
# decayed). So --ssl-max-steps sets RUN LENGTH ONLY, every arm is schedule-matched at every
# shared step regardless of its total, and maskspace25/70 at 30k ARE valid comparators here.
# --ssl-max-steps stays 40000 because that is this arm's target length, not for LR reasons.
set -uo pipefail
module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bhqk/htang13/torchinductor_cache_r6
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WANDB_API_KEY="$(cat /projects/bhqk/htang13/.wandb_key)"
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"

CB="${1:?usage: sbatch v3_r6_maskspace0_race.sbatch <CONTACT_BUDGET>  (accum derived as 66000/CB)}"
case "$CB" in ''|*[!0-9]*) echo "FATAL: CB '$CB' is not a positive integer." >&2; exit 1 ;; esac
if [ "$((66000 % CB))" -ne 0 ]; then
  echo "FATAL: 66000 % $CB = $((66000 % CB)) != 0, so no integer accum is token-exact." >&2
  echo "  Divisors of 66000 at or above the useful range: 8250 11000 13200 16500 22000 33000." >&2
  echo "  A cb that does not divide 66000 makes this arm incomparable to the 40k wave." >&2
  exit 1
fi
ACCUM=$((66000 / CB))
TOKENS=$((CB * ACCUM))
if [ "$TOKENS" -ne 66000 ] || [ "$ACCUM" -lt 1 ]; then
  echo "FATAL: cb($CB) x accum($ACCUM) = $TOKENS != 66000 contacts/step." >&2
  echo "  The whole basis for switching cb mid-run is that the optimizer step is TOKEN-EXACT." >&2
  echo "  Off-budget => this arm is not comparable to the 40k wave. Refusing to start." >&2
  exit 1
fi

RUN_NAME=v3_r6_maskspace0_lr6e-3_40k_cb${CB}
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
SRC_CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_v3_r6_maskspace0_lr6e-3_40k
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain

if [ "$CKPT_DIR" = "$SRC_CKPT_DIR" ]; then
  echo "FATAL: race ckpt dir == incumbent ckpt dir; would interleave two ladders." >&2; exit 1
fi

# Resume from the LATEST ladder ckpt that has settled. Picked at job START, not at submit, so a
# long queue wait is not wasted -- the incumbent keeps laddering while this pends and we resume
# from wherever it got to. The -mmin +2 filter skips a ckpt the incumbent may still be writing;
# torch.save is not atomic and a torn ckpt would fail the load (loudly, but hours later).
RESUME=""
RESUME_STEP=""
for f in "$SRC_CKPT_DIR"/ladder-step=*.ckpt; do
  [ -f "$f" ] || continue
  [ -n "$(find "$f" -maxdepth 0 -mmin +2 2>/dev/null)" ] || continue
  s=${f##*step=}; s=${s%.ckpt}
  case "$s" in ''|*[!0-9]*) continue ;; esac
  if [ -z "$RESUME_STEP" ] || [ "$s" -gt "$RESUME_STEP" ]; then RESUME_STEP=$s; RESUME=$f; fi
done
if [ -z "$RESUME" ]; then
  echo "FATAL: no settled ladder-step=*.ckpt in $SRC_CKPT_DIR" >&2; exit 1
fi
if [ "$RESUME_STEP" -ge 40000 ]; then
  echo "FATAL: resume step $RESUME_STEP >= 40000; the arm is already done, nothing to race." >&2
  exit 1
fi

echo "[ms0 race] cb=$CB accum=$ACCUM tokens/step=$TOKENS (== incumbent 5500x12, TOKEN-EXACT)"
echo "[ms0 race] mem for this cb was MEASURED by job 2773794, not predicted -- see its SUMMARY."
echo "[ms0 race] (the old 39.0+136.8v model over-predicted cb5500/v=0.50 by 34%: 73.2 vs 54.6 actual)"
echo "[ms0 race] resume <- $RESUME (step $RESUME_STEP), $((40000 - RESUME_STEP)) steps left to 40000"
echo "[ms0 race] ckpt-dir  -> $CKPT_DIR  (SEPARATE from incumbent, no ladder interleave)"
echo "[ms0 race] delta vs r6 baseline is STILL --mask-space-frac 0.0, re-passed below"

srun "$VENV" -u -m speech_decoding.experiments.dispatch_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --band-cache-dir $ROOT/band_v3slow \
  --band-cache-dir $ROOT/band_v3mid \
  --band-cache-dir $ROOT/band_hga \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --session 1:0 --session 2:1 --session 2:2 --session 2:3 --session 2:5 \
  --session 2:6 --session 3:2 --session 4:2 --session 6:0 --session 6:1 \
  --session 6:4 --session 8:0 --session 9:0 \
  --frontend v3r6 --objective mae --mask-space-frac 0.0 \
  --batch-unit shaft --contact-budget $CB --shaft-alpha 0.5 \
  --lr 6e-3 --weight-decay 0.04 --grad-clip 3.0 --adam-beta2 0.95 --seed 33 \
  --clip-len 2.0 \
  --session-z-winsor-slow 15 --session-z-winsor-mid 15 --session-z-winsor-hga 20 \
  --batch-size 32 --accumulate-grad-batches $ACCUM \
  --accelerator gpu --devices 1 --precision bf16-mixed \
  --num-workers 12 --monitor-every-n-steps 10 --log-every-n-steps 10 \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 1000 \
  --resume-ckpt "$RESUME" \
  --ssl-max-steps 40000
