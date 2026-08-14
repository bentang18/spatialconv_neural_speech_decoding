#!/bin/bash
#SBATCH --job-name=r6_vits384_tube_cd55k
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=113560M
#SBATCH --time=19:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
#
#   JOB 2 OF 2.
#   sbatch --dependency=afterany:<tube40k jobid> scripts/neuroprobe/v3_r6_vits384_tube_cd55k.sbatch vits384
#
# 40000 -> 55000 UNDER THE TIME-TUBE CONTRACT: flat to 50000, then LINEAR 6e-3 -> 0.
# This job absorbs BOTH legs the old chain ran separately (ext50k 40k->50k, then cd55k
# 50k->55k), which is what keeps the whole contract at TWO jobs instead of three.
#
# ═══ WHY THIS IS ONE JOB AND NOT TWO ═════════════════════════════════════════════════════
# 15,000 x 4.005 s/step = 60,075 s = 16.7 h against a 19 h request. The old split existed
# because its flat leg was already fragmented, not because 15k needs two walls.
#
# ═══ WHY 50000 IS THE BRANCH POINT ═══════════════════════════════════════════════════════
# UNCHANGED from the canonical, and pre-specified BOARD-BLIND -- the cooldown value comes from
# the 4-task pretrain probe plateauing, never from a board read. The uncooled ladder read out
# to 60k (WS enc12|std, n=28): .76866@20k .77247@35k .77386@40k .77568@45k .77532@50k
# .77512@55k .77560@60k. Four rungs from 45k sit inside .00056; nothing after 45k pays. Ben's
# call is to cool from 50000: tied with the 45k peak, with 10k more training behind it.
# 🚨 CS on that probe is UNREADABLE (per-cell SD .0197) and played NO part in the choice.
# ⚠️ Carried over from a measurement made under the PER-CONTACT draw. It is the same decision
# PROCEDURE ("no further improvement over four rungs"), applied to a schedule fixed in advance;
# it is NOT a claim that this contract's probe plateaus at the same step. Do not re-tune it
# from a board read.
#
# ═══ THE SCHEDULE ════════════════════════════════════════════════════════════════════════
# --stable-steps 50000 --decay-shape linear --min-lr-ratio 0.0 --ssl-max-steps 55000.
# _WarmupCosineLR._multiplier is a pure function of global_step and load_state_dict
# (lr_schedule.py:96-106) restores ONLY progress, so resuming at 40000 gives:
#   40000..50000  multiplier 1.0   (past warmup, below stable_steps) -- PEAK 6e-3
#   50000..55000  linear 1.0 -> 0.0
# i.e. bit-identical to one uninterrupted 0->55k run. The parent is written under
# --min-lr-ratio 1.0 so step 40000 is at peak. 🚫 NEVER branch a cooldown off a decay-era ckpt.
#
# ═══ 🔴 THE `_tube_` GUARD ═══════════════════════════════════════════════════════════════
# --band-time-unit leaves NOTHING in the state_dict, so an old per-contact 40k ckpt is
# indistinguishable from a tubed one by inspection. The resume path MUST contain `_tube_`;
# the check below is a hard gate, and it is the only mechanical bar to cooling the wrong
# contract. dispatch_v3 prints a [mask] fingerprint at startup -- compare it against the
# parent's log line before trusting the output.
# ⚠ ext != cd: what ships is the 55k COOLED ckpt. Nothing below 55000 boards.
set -uo pipefail

ARM="${1:?usage: sbatch v3_r6_vits384_tube_cd55k.sbatch <vits384|vits384_noparcel|vits384_nosrope|vits384_noparcel_nosrope>}"
case "$ARM" in
  vits384)                  ABL="--enc-width 384" ;;
  vits384_noparcel)         ABL="--enc-width 384 --no-parcel-embed" ;;
  vits384_nosrope)          ABL="--enc-width 384 --no-space-rope" ;;
  vits384_noparcel_nosrope) ABL="--enc-width 384 --no-parcel-embed --no-space-rope" ;;
  *) echo "[FATAL] unknown arm '$ARM'"; exit 1 ;;
esac

# 🔴 SPACE-ROPE CROSS-CHECK, BOTH DIRECTIONS.
case "$ARM"   in *nosrope*|*nospacerope*) NAME_SAYS_OFF=1 ;; *) NAME_SAYS_OFF=0 ;; esac
case " $ABL " in *" --no-space-rope "*)   FLAG_SAYS_OFF=1 ;; *) FLAG_SAYS_OFF=0 ;; esac
if [ "$NAME_SAYS_OFF" != "$FLAG_SAYS_OFF" ]; then
  echo "[FATAL] space_rope mismatch -- name says off=$NAME_SAYS_OFF, flags say off=$FLAG_SAYS_OFF (ABL='$ABL')"; exit 1
fi
[ -n "$ABL" ] || { echo "[FATAL] ABL empty for arm '$ARM'"; exit 1; }

CB=8250
ACC=8
if [ "$((CB * ACC))" -ne 66000 ]; then
  echo "[FATAL] cb=$CB x accum=$ACC = $((CB * ACC)) != 66000 contacts/opt-step"; exit 1
fi

# 🔴 ipe MUST MATCH JOB 1 EXACTLY. --batch-unit shaft feeds an IterableDataset, which Lightning
# does NOT fast-forward on resume: __iter__ restarts at pack 0 every time, and the pack sequence
# is a pure function of (seed, EPOCH, worker). The only thing that makes this job draw clips job
# 1 has not already trained on is a DIFFERENT epoch number, and current_epoch is restored from
# the ckpt as global_step/ipe. A different cps or num-workers here shifts ipe, lands the resume
# on a different epoch than the parent ended on, and silently re-serves seen packs.
#
# 🔴 --shaft-alpha 0.0 MUST ALSO MATCH JOB 1. Full rationale lives in the job-1 header; the
# short form is that the sampler's pool is (session, shaft) PAIRS, so alpha tempers RECORDING
# SESSION COUNT rather than electrode count, and alpha=0 is the only value whose per-subject
# share (exactly 1/7) does not depend on that census at all.
CPS=616
NSESS=13
NW=32
IPE=$(( ((CPS * NSESS / NW) * NW) / ACC ))
if [ "$IPE" -ne 1000 ]; then
  echo "[FATAL] iterations-per-epoch is $IPE, not 1000 -- MUST match job 1 (cps=$CPS workers=$NW accum=$ACC)"; exit 1
fi
if [ $((40000 % IPE)) -ne 0 ] || [ $((55000 % IPE)) -ne 0 ]; then
  echo "[FATAL] 40000/55000 are not epoch boundaries at ipe=$IPE"; exit 1
fi

module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# MATCHES JOB 1 so this starts on an already-warm inductor cache (~20 min saved).
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bhqk/htang13/torchinductor_cache_r6_tube_${ARM}
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WANDB_API_KEY="$(cat /projects/bhqk/htang13/.wandb_key)"
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"

PARENT=v3_r6_tube_pbspace111_sf50_${ARM}_40k
RUN_NAME=v3_r6_tube_pbspace111_sf50_${ARM}_cd55k
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
RESUME=/projects/bhqk/htang13/v3_ckpt_${PARENT}/ladder-step=40000.ckpt
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain

# 🔴 CONTRACT GUARD: refuse to cool anything that is not a tube run. An old-draw 40k ckpt is
# key- and value-identical in every way this script could otherwise inspect.
case "$RESUME" in
  *_tube_*) : ;;
  *) echo "[FATAL] resume path lacks '_tube_': $RESUME -- that is the OLD per-contact draw"; exit 1 ;;
esac
# Doubles as the real dependency gate: submitted afterANY, so if the parent died before writing
# 40000 this FATALs in seconds instead of cooling a stale ckpt or training from zero.
[ -f "$RESUME" ] || { echo "[FATAL] resume ckpt missing: $RESUME -- parent $PARENT did not reach 40000"; exit 1; }
[ -e "$CKPT_DIR" ] && { echo "[FATAL] $CKPT_DIR already exists -- refusing to mix ladders"; exit 1; }

echo "[r6 vits384 TIME-TUBE COOLDOWN] arm=$ARM flags='$ABL' run=$RUN_NAME"
echo "[contract] --band-time-unit shaft. Parent verified to carry '_tube_' in its path."
echo "[branch]   resume=$RESUME  (EXACTLY step 40000, PEAK LR -- parent ran --min-lr-ratio 1.0 throughout)"
echo "[schedule] JOB 2 OF 2: CONSTANT 6e-3 to 50000, then LINEAR 6e-3->0 over 50000..55000, --min-lr-ratio 0.0, ladder=500"
echo "[epoch]    clips-per-session=$CPS -> ipe=$IPE, IDENTICAL to job 1. Resume at 40000 = epoch 40, a boundary, so this job draws packs job 1 never reached."
echo "[matched]  cb=$CB accum=$ACC (=$((CB*ACC)) contacts/opt-step) sf50 per_band_space=ON widths=1,1,1 lr=6e-3 seed=33 workers=$NW [FILE-ONLY scp deploy]"
echo "[budget]   15000 steps x ~4.005 s/step = ~16.7 h against a 19 h wall"
echo "[read]     the 55k COOLED ckpt is what boards. 🚫 never board a rung below 55000."

srun "$VENV" -u -m speech_decoding.experiments.dispatch_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --band-cache-dir $ROOT/band_v3slow \
  --band-cache-dir $ROOT/band_v3mid \
  --band-cache-dir $ROOT/band_hga \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --session 1:0 --session 2:1 --session 2:2 --session 2:3 --session 2:5 \
  --session 2:6 --session 3:2 --session 4:2 --session 6:0 --session 6:1 \
  --session 6:4 --session 8:0 --session 9:0 \
  --frontend v3r6 --objective mae \
  --batch-unit shaft --contact-budget $CB --shaft-alpha 0.0 \
  --lr 6e-3 --weight-decay 0.04 --grad-clip 3.0 --adam-beta2 0.95 --seed 33 \
  --min-lr-ratio 0.0 --stable-steps 50000 --decay-shape linear \
  --clip-len 2.0 \
  --session-z-winsor-slow 15 --session-z-winsor-mid 15 --session-z-winsor-hga 20 \
  --batch-size 32 --accumulate-grad-batches $ACC \
  --accelerator gpu --devices 1 --precision bf16-mixed \
  --clips-per-session $CPS \
  --num-workers $NW --monitor-every-n-steps 10 --log-every-n-steps 10 \
  --mask-space-frac 0.50 --per-band-space --space-block-w-bands 1,1,1 \
  --band-time-unit shaft \
  $ABL \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --resume-ckpt $RESUME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 500 \
  --ssl-max-steps 55000
