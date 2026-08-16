#!/bin/bash
#SBATCH --job-name=r6_vits384_ext50k
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=113560M
#SBATCH --time=14:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
#
#   sbatch scripts/neuroprobe/v3_r6_vits384_ext50k.sbatch
#
# EXTEND vits384 FROM 40k TO 50k AT CONSTANT LR. NO COOLDOWN. (Ben 2026-08-07)
#
# ── THE QUESTION ─────────────────────────────────────────────────────────────────────────
# Two live explanations for why d=384 buys almost nothing over d=256 on the board:
#   (a) data-limited  -- 100 opt-steps per corpus pass, so 40k = ~400 epochs; extra capacity
#                        has nothing new to fit.
#   (b) undertrained  -- the pretrain probe shows vits384 STILL CLIMBING into 40k
#                        (enc12 .7340@20k -> .7382@35k -> .7403@40k, last 5k faster per step)
#                        while d=256 is terminal at 40k. A data-starved model plateaus
#                        EARLIER, not later.
# These predict OPPOSITE things past 40k, and past-40k was unmeasured. This run measures it.
#
# ── WHY NO COOLDOWN IS THE RIGHT CHOICE HERE, NOT A SHORTCUT ─────────────────────────────
# The question is the SHAPE of the trajectory, not the quality of a shippable checkpoint. A
# cooldown would confound "did it keep learning" with "did the decay bank it". Holding LR flat
# keeps every ladder rung on ONE schedule, so 40k / 45k / 50k are mutually comparable:
#   parent ckpts <=40000  : uncooled (trapezoid is FLAT until 40000)
#   this run 40500..50000 : uncooled (min_lr_ratio 1.0)
# 🔴 THE OUTPUT IS THEREFORE NOT BOARD-COMPARABLE TO pbs50_cd45k OR board_vits384_cd45k.
#    ext != cd. Read this ladder against the UNCOOLED parent rungs only, on the 4-task probe.
#    Deciding to ship any of it means a SEPARATE cooldown run from the chosen branch point.
#
# ── CONSTANT LR IS ARGS-SIDE, VERIFIED IN CODE ───────────────────────────────────────────
# lr_schedule.py:91  mult = min_lr_ratio + (1 - min_lr_ratio) * shaped  => 1.0 when ratio=1.0.
# lr_schedule.py:96  load_state_dict restores ONLY last_epoch/_step_count and deliberately does
#   NOT let the ckpt clobber total_steps/min_lr_ratio/stable_steps/decay_shape. So the schedule
#   shape comes from THESE args, not from the parent's trapezoid. --stable-steps/--decay-shape
#   are omitted on purpose: at ratio 1.0 they cannot move the multiplier, and passing them
#   would imply a decay that does not exist.
#
# ── THE BRANCH POINT ─────────────────────────────────────────────────────────────────────
# Parent v3_r6_pbspace111_sf50_vits384_cd45k is the ORIGINAL 45k trapezoid (job 2873602) that
# died ~43,150 against the 48 h wall. Its schedule is flat over 0..40000, and at exactly step
# 40000 the multiplier is still 1.0 (progress = (40000-40000)/5000 = 0), so step-40000 is at
# PEAK LR and is a legitimate constant-LR branch point.
# 🗑 That dir also holds 41000/42000/43000 -- those are PARTIAL-COOLDOWN ckpts. NEVER branch
#    from them. This script hardcodes 40000 and asserts it exists.
#
# ── NO-TRACE FLAG ────────────────────────────────────────────────────────────────────────
# --enc-width 384 is LOUD (it changes the state_dict, so a wrong value fails to load) unlike
# --mask-time-frac / --no-space-rope / --parcel-perm-seed. It is still passed explicitly
# because the resume path builds the model from args before loading weights.

set -uo pipefail

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
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bhqk/htang13/torchinductor_cache_r6_vits384
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WANDB_API_KEY="$(cat /projects/bhqk/htang13/.wandb_key)"
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"

PARENT=v3_r6_pbspace111_sf50_vits384_cd45k
RUN_NAME=v3_r6_pbspace111_sf50_vits384_ext50k
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
RESUME=/projects/bhqk/htang13/v3_ckpt_${PARENT}/ladder-step=40000.ckpt
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain

# Fail LOUDLY rather than starting fresh: --resume-ckpt defaults to None, so a missing/typo'd
# path does NOT error -- it silently trains from scratch and burns the wall.
[ -f "$RESUME" ] || { echo "[FATAL] resume ckpt missing: $RESUME"; exit 1; }
# Never write into the parent dir. A fresh dir is what keeps the uncooled ladder separable
# from the cooled cd45k ladder for the rest of this program.
[ -e "$CKPT_DIR" ] && { echo "[FATAL] $CKPT_DIR already exists -- refusing to mix ladders"; exit 1; }

echo "[r6 vits384 EXTENSION] run=$RUN_NAME"
echo "[branch]   resume=$RESUME  (EXACTLY step 40000, PEAK LR, uncooled -- 41k/42k/43k are cooldown-era, NOT branch points)"
echo "[schedule] CONSTANT 6e-3 over 40000..50000, --min-lr-ratio 1.0, NO decay, ladder=1000"
echo "[matched]  cb=$CB accum=$ACC (=$((CB*ACC)) contacts/opt-step) sf50 per_band_space=ON widths=1,1,1 lr=6e-3 seed=33 workers=32 [FILE-ONLY scp deploy]"
echo "[read]     vs the UNCOOLED parent rungs (20k/35k/40k) on the 4-task probe. NOT vs cd45k -- ext != cd."
echo "[budget]   10000 steps x ~4.003 s/step = ~11.1 h against a 14 h wall"

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
  --min-lr-ratio 1.0 \
  --clip-len 2.0 \
  --session-z-winsor-slow 15 --session-z-winsor-mid 15 --session-z-winsor-hga 20 \
  --batch-size 32 --accumulate-grad-batches $ACC \
  --accelerator gpu --devices 1 --precision bf16-mixed \
  --num-workers 32 --monitor-every-n-steps 10 --log-every-n-steps 10 \
  --mask-space-frac 0.50 --per-band-space --space-block-w-bands 1,1,1 \
  --enc-width 384 \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --resume-ckpt $RESUME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 1000 \
  --ssl-max-steps 50000
