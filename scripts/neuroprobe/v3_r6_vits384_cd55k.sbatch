#!/bin/bash
#SBATCH --job-name=r6_vits384_cd55k
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=113560M
#SBATCH --time=07:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
#
#   sbatch scripts/neuroprobe/v3_r6_vits384_cd55k.sbatch vits384
#   sbatch --dependency=afterany:<ext50k jobid> scripts/neuroprobe/v3_r6_vits384_cd55k.sbatch vits384_noparcel
#
# COOL EVERY vits384 GEOMETRY ARM 50k -> 55k. LINEAR 6e-3 -> 0. (Ben 2026-08-08)
#
# ── WHY 50k IS THE BRANCH POINT ──────────────────────────────────────────────────────────
# Chosen FROM the measured plateau, never pre-committed. The uncooled baseline ladder read
# out to 60k on the 4-task probe (WS enc12|std, n=28):
#   .76866@20k  .77247@35k  .77386@40k  .77568@45k  .77532@50k  .77512@55k  .77560@60k
# Four rungs from 45k on sit inside .00056; 60k-45k = -.00008 (14/28, sign p=1.00). Nothing
# after 45k pays. Ben's call is to cool from 50k: statistically tied with the 45k peak, with
# 10k more training behind it. 🚨 CS on that probe is UNREADABLE (per-cell SD .0197) -- the
# 50k CS bump did NOT survive to 60k, so it played no part in this choice.
#
# ── THE SCHEDULE, AND WHY IT IS THE SAME FOR ALL FOUR ARMS ───────────────────────────────
# stable to 50000, then LINEAR 6e-3 -> 0 over 50000..55000, --min-lr-ratio 0.0, ladder 500.
# Identical shape to the canonical cooldown (v3_r6_pbs50_cooldown.sbatch) and to the cancelled
# 40k->45k resumes -- only the anchor moves, 40000 -> 50000.
# 🔴 A 2x2 IS ONLY A 2x2 IF EVERY CELL TRAVELS THE SAME SCHEDULE. The three 40k->45k ablation
#    cooldowns (2894634/35/36) were cancelled for this reason; each arm now goes 40k -> 50k
#    flat (v3_r6_vits384_arms_ext50k.sbatch) and then through this file.
#
# ── BRANCH POINT IS UNCOOLED, BY CONSTRUCTION ────────────────────────────────────────────
# Parent is always ${ARM}_ext50k, whose every ckpt was written under --min-lr-ratio 1.0, so
# step 50000 sits at PEAK LR. 🚫 NEVER branch a cooldown off a decay-era ckpt (cd45k's
# 41k/42k/43k, or anything the cancelled resumes left behind).
# ⚠ ext != cd: the ext rungs are NOT board-comparable. What ships is the 55k COOLED ckpt.
#
# ── NO-TRACE FLAGS ───────────────────────────────────────────────────────────────────────
# --no-space-rope is key- and value-identical in the state_dict, so a forgotten flag cools the
# WRONG arm with no error. The name<->flag cross-check below is a hard gate.
#
# TIMING: 5000 steps x ~4.0 s/step ~= 5.6 h. 7 h wall, matching the cooldown launcher this
# replaces. Walltime is the BACKFILL knob -- do not inflate it. NEVER self-kill.
set -uo pipefail

ARM="${1:?usage: sbatch v3_r6_vits384_cd55k.sbatch <vits384|vits384_noparcel|vits384_nosrope|vits384_noparcel_nosrope>}"
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

PARENT=v3_r6_pbspace111_sf50_${ARM}_ext50k
RUN_NAME=v3_r6_pbspace111_sf50_${ARM}_cd55k
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
RESUME=/projects/bhqk/htang13/v3_ckpt_${PARENT}/ladder-step=50000.ckpt
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain

# This doubles as the real dependency gate: submitted afterANY, so if the parent died before
# writing 50000 this FATALs in seconds instead of cooling a stale ckpt or training from zero.
[ -f "$RESUME" ] || { echo "[FATAL] resume ckpt missing: $RESUME -- parent $PARENT did not reach 50000"; exit 1; }
[ -e "$CKPT_DIR" ] && { echo "[FATAL] $CKPT_DIR already exists -- refusing to mix ladders"; exit 1; }

echo "[r6 vits384 COOLDOWN] arm=$ARM flags='$ABL' run=$RUN_NAME"
echo "[branch]   resume=$RESUME  (EXACTLY step 50000, PEAK LR -- parent dir is 100% uncooled)"
echo "[schedule] stable=50000 -> LINEAR 6e-3->0 over 50000..55000, --min-lr-ratio 0.0, ladder=500"
echo "[matched]  cb=$CB accum=$ACC (=$((CB*ACC)) contacts/opt-step) sf50 per_band_space=ON widths=1,1,1 lr=6e-3 seed=33 workers=32 [FILE-ONLY scp deploy]"
echo "[read]     the 55k COOLED ckpt is what boards. 🚫 never board an ext rung -- ext != cd."
echo "[budget]   5000 steps x ~4.0 s/step = ~5.6 h against a 7 h wall"

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
