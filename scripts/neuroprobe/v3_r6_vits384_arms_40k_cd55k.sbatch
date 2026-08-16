#!/bin/bash
#SBATCH --job-name=r6_vits384_arm_cd55k
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=113560M
#SBATCH --time=20:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
#
#   sbatch scripts/neuroprobe/v3_r6_vits384_arms_40k_cd55k.sbatch vits384_noparcel
#
# EACH GEOMETRY ABLATION ARM 40k -> 55k IN ONE JOB: FLAT TO 50k, THEN COOL. (Ben 2026-08-08)
#
# ── WHY ONE JOB AND NOT TWO ──────────────────────────────────────────────────────────────
# The LR schedule is ARGS-SIDE, and _WarmupCosineLR.load_state_dict (lr_schedule.py:97-107)
# restores ONLY last_epoch/_step_count -- the SHAPE is rebuilt from this file's flags every
# time. _multiplier (lr_schedule.py:83-91) keys off the GLOBAL step. So resuming at 40000 with
#   --min-lr-ratio 0.0 --stable-steps 50000 --decay-shape linear --ssl-max-steps 55000
# gives mult=1.0 for 40000..49999 (step < stable_steps) and linear 1->0 over 50000..55000, in
# a single invocation. warmup_steps=5000 is long past 40000, so there is NO re-warmup.
# This replaces the two-job chain (v3_r6_vits384_arms_ext50k.sbatch + _cd55k.sbatch, jobs
# 2906651-2906656, cancelled): same schedule, one dependency fewer, one torch.compile fewer.
#
# ── WHY 50k IS THE BRANCH POINT ──────────────────────────────────────────────────────────
# Chosen FROM the measured plateau, never pre-committed. Uncooled baseline, 4-task pretrain
# probe, WS enc12|std, n=28:
#   .76866@20k  .77247@35k  .77386@40k  .77568@45k  .77532@50k  .77512@55k  .77560@60k
# Four rungs from 45k on sit inside .00056; 60k-45k = -.00008 (14/28, sign p=1.00). Ben cools
# from 50k: tied with the 45k peak, with 10k more training behind it. 🚨 CS on that probe is
# UNREADABLE (per-cell SD .0197) -- the 50k CS bump did not survive to 60k and played no part.
#
# ── PARITY WITH THE BASELINE, WHICH IS THE WHOLE POINT ───────────────────────────────────
# The baseline reaches 55k as 40k -> 50k flat (2898613, --min-lr-ratio 1.0) then 50k -> 55k
# cooled (2906650, this same cooldown block). An arm covers the identical 15000 steps under
# the identical LR curve; the only differences are cosmetic and listed here so nobody has to
# rediscover them:
#   (a) the baseline has a resume boundary at 40000 AND 50000, an arm only at 40000;
#   (b) the baseline's flat segment laddered every 1000, an arm ladders every 500 throughout
#       (one job = one ladder rate; 500 is chosen because it is the CANONICAL COOLDOWN
#       resolution, matching v3_r6_pbs50_cooldown.sbatch and pbs50_cd45k, and the cooldown is
#       the segment that gets read).
# Neither touches the trajectory: the LR curve depends only on global_step, and saving a ckpt
# is deterministic. 🔴 EVERY ARM MUST TRAVEL THE SAME SCHEDULE OR THE 2x2 IS NOT A 2x2.
#
# ── BRANCH POINT IS UNCOOLED, BY CONSTRUCTION ────────────────────────────────────────────
# Resume is EXACTLY step 40000 from the arm's own uncooled *_40k dir, at PEAK LR.
# 🚫 NEVER resume from a *_cd45k_resume40k dir (the 2894634/35/36 cancellations left partial
#    cooldowns behind, the same hazard as cd45k's 41k/42k/43k). Decay-era weights are not a
#    constant-LR branch point and would silently produce an off-schedule arm.
# ⚠ ext != cd. In THIS run's dir, rungs 40500..50000 are UNCOOLED and 50500..55000 are
#   DECAY-ERA; only 55000 is the terminal cooled ckpt, and only it boards.
#
# ── NO-TRACE FLAGS ───────────────────────────────────────────────────────────────────────
# --no-space-rope leaves NOTHING in the state_dict: key- and value-identical either way. A
# forgotten flag runs the WRONG arm with no error, so the name<->flag cross-check below is a
# hard gate, not a courtesy. --no-parcel-embed and --enc-width are loud (they fail to load).
#
# TIMING: 15000 steps x ~4.0 s/step ~= 16.7 h (the baseline's 10000-step segment took 11:03:50)
# plus ~20 min startup/compile. 20 h wall. Walltime is the BACKFILL knob -- do not inflate it.
# NEVER self-kill.
set -uo pipefail

ARM="${1:?usage: sbatch v3_r6_vits384_arms_40k_cd55k.sbatch <vits384_noparcel|vits384_nosrope|vits384_noparcel_nosrope>}"
case "$ARM" in
  vits384_noparcel)         ABL="--enc-width 384 --no-parcel-embed" ;;
  vits384_nosrope)          ABL="--enc-width 384 --no-space-rope" ;;
  vits384_noparcel_nosrope) ABL="--enc-width 384 --no-parcel-embed --no-space-rope" ;;
  vits384)
    echo "[FATAL] the BASELINE already has its 40k->50k segment (v3_r6_pbspace111_sf50_vits384_ext50k,"
    echo "        job 2898613) and its cooldown (2906650). Re-running it here would fork the ladder."; exit 1 ;;
  *) echo "[FATAL] unknown arm '$ARM'"; exit 1 ;;
esac

# 🔴 SPACE-ROPE CROSS-CHECK, BOTH DIRECTIONS. The one flag no state_dict check can catch.
case "$ARM"   in *nosrope*|*nospacerope*) NAME_SAYS_OFF=1 ;; *) NAME_SAYS_OFF=0 ;; esac
case " $ABL " in *" --no-space-rope "*)   FLAG_SAYS_OFF=1 ;; *) FLAG_SAYS_OFF=0 ;; esac
if [ "$NAME_SAYS_OFF" != "$FLAG_SAYS_OFF" ]; then
  echo "[FATAL] space_rope mismatch -- name says off=$NAME_SAYS_OFF, flags say off=$FLAG_SAYS_OFF (ABL='$ABL')"; exit 1
fi
# An empty ABL would silently run the BASELINE under an ablation's name.
[ -n "$ABL" ] || { echo "[FATAL] ABL empty for arm '$ARM'"; exit 1; }

CB=8250
ACC=8
# Token identity is an ASSERT, not a comment: a wrong pair silently changes the optimizer step.
if [ "$((CB * ACC))" -ne 66000 ]; then
  echo "[FATAL] cb=$CB x accum=$ACC = $((CB * ACC)) != 66000 contacts/opt-step"; exit 1
fi

module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bhqk/htang13/torchinductor_cache_r6_${ARM}
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WANDB_API_KEY="$(cat /projects/bhqk/htang13/.wandb_key)"
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"

PARENT=v3_r6_pbspace111_sf50_${ARM}_40k
RUN_NAME=v3_r6_pbspace111_sf50_${ARM}_ext50k_cd55k
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
RESUME=/projects/bhqk/htang13/v3_ckpt_${PARENT}/ladder-step=40000.ckpt
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain

# --resume-ckpt defaults to None: a missing/typo'd path does NOT error, it trains from scratch
# and burns the whole wall. Fail loudly instead.
[ -f "$RESUME" ] || { echo "[FATAL] resume ckpt missing: $RESUME"; exit 1; }
[ -e "$CKPT_DIR" ] && { echo "[FATAL] $CKPT_DIR already exists -- refusing to mix ladders"; exit 1; }

echo "[r6 vits384 ARM 40k->55k, ONE JOB] arm=$ARM flags='$ABL' run=$RUN_NAME"
echo "[branch]   resume=$RESUME  (EXACTLY step 40000, PEAK LR, uncooled parent -- NOT a *_cd45k_resume40k dir)"
echo "[schedule] FLAT 6e-3 over 40000..49999, then LINEAR 6e-3->0 over 50000..55000 (--min-lr-ratio 0.0 --stable-steps 50000), ladder=500"
echo "[matched]  cb=$CB accum=$ACC (=$((CB*ACC)) contacts/opt-step) sf50 per_band_space=ON widths=1,1,1 lr=6e-3 seed=33 workers=32 [FILE-ONLY scp deploy]"
echo "[parity]   same 15000 steps and same LR curve as baseline 2898613 + 2906650"
echo "[read]     ⚠ 40500..50000 are UNCOOLED, 50500..55000 are DECAY-ERA. ONLY 55000 boards. ext != cd."
echo "[budget]   15000 steps x ~4.0 s/step = ~16.7 h against a 20 h wall"

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
  --batch-unit shaft --contact-budget $CB --shaft-alpha 0.5 \
  --lr 6e-3 --weight-decay 0.04 --grad-clip 3.0 --adam-beta2 0.95 --seed 33 \
  --min-lr-ratio 0.0 --stable-steps 50000 --decay-shape linear \
  --clip-len 2.0 \
  --session-z-winsor-slow 15 --session-z-winsor-mid 15 --session-z-winsor-hga 20 \
  --batch-size 32 --accumulate-grad-batches $ACC \
  --accelerator gpu --devices 1 --precision bf16-mixed \
  --num-workers 32 --monitor-every-n-steps 10 --log-every-n-steps 10 \
  --mask-space-frac 0.50 --per-band-space --space-block-w-bands 1,1,1 \
  $ABL \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --resume-ckpt $RESUME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 500 \
  --ssl-max-steps 55000
