#!/bin/bash
#SBATCH --job-name=r6_vits384_ext60k
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
#   sbatch --dependency=afterany:2898613 scripts/neuroprobe/v3_r6_vits384_ext60k.sbatch
#
# EXTEND vits384 FROM 50k TO 60k AT CONSTANT LR. NO COOLDOWN. (Ben 2026-08-08)
#
# ── WHY THIS RUN EXISTS ──────────────────────────────────────────────────────────────────
# The ext45k pretrain probe SETTLED the question the ext50k run was launched to ask: ViT-S is
# UNDERTRAINED at 40k, not data-limited. enc12|std WS per-cell OLS slope on step across
# 20k/35k/40k/45k = +0.000275/1k, 23/28 cells positive, sign p=0.0009, t=4.64 -- and the
# 40k->45k segment (+0.000363/1k) is the FASTEST of the three measured. Nothing has bent in
# 25k steps, so the top of the curve is still not located.
#   WS-only  ladder: .7687 / .7725 / .7739 / .7757   (28 cells)
#   POOLED   ladder: .7340 / .7382 / .7403 / .7413   (52 cells)  <-- do not mix the two
#
# ── THE COOLDOWN POLICY THIS RUN IMPLEMENTS (Ben) ────────────────────────────────────────
# "We choose our cooldown from when downstream performance plateaus." So: keep extending at
# FLAT LR and probe the ladder, then branch a SEPARATE cooldown from whatever rung the probe
# says stopped paying. Cooling early would bank a number but destroy the comparability of
# every rung after it -- a cooled ckpt and an uncooled ckpt are not on the same schedule.
# 🔴 THIS LADDER IS THEREFORE UNCOOLED AND NOT BOARD-COMPARABLE. ext != cd. Read 50500..60000
#    against the uncooled rungs only, on the 4-task probe. Shipping ANY rung means a separate
#    cooldown run branched from it.
#
# ── PROBE POWER, SO THE NEXT READ IS NOT WASTED ──────────────────────────────────────────
# WS per-cell paired SD ~= 0.0060 (sem 0.0011) on n=28. A 5k gap is t~1.1 (p~.26) and CANNOT
# resolve; a 10k gap is t~2.1. So probe 60k against 50k (10k), not 55k against 50k, and read
# the multi-rung SLOPE rather than any single pairwise contrast.
# 🚨 CS IS UNREADABLE ON THIS PROBE: per-cell SD 0.0197, 3.3x WS, on 24 cells. Even the big
#    20k->40k jump only reaches t=1.89. A CS null here is UNINFORMATIVE, never negative evidence.
#
# ── THE BRANCH POINT ─────────────────────────────────────────────────────────────────────
# Parent is the ext50k run (job 2898613), which is flat 6e-3 over ALL of 40000..50000, so step
# 50000 is at PEAK LR and is a legitimate constant-LR branch point.
# ✅ Unlike the cd45k parent, this dir has NO partial-cooldown rungs to trip over -- every
#    ckpt in it was written under min_lr_ratio 1.0.
#
# ── DEPENDENCY IS afterANY, ON PURPOSE ───────────────────────────────────────────────────
# The real precondition is "ladder-step=50000.ckpt exists", not "the parent exited 0". With
# afterok, a parent that writes 50000 and then trips on some teardown path would block this
# job forever and lose the night. With afterany the guard below is what gates us: if 50000 is
# missing this FATALs in seconds and costs nothing.
#
# ── NO-TRACE FLAG ────────────────────────────────────────────────────────────────────────
# --enc-width 384 is LOUD (wrong value fails to load) unlike --mask-time-frac / --no-space-rope
# / --parcel-perm-seed. Still passed explicitly: the resume path builds the model from args
# before loading weights.

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

PARENT=v3_r6_pbspace111_sf50_vits384_ext50k
RUN_NAME=v3_r6_pbspace111_sf50_vits384_ext60k
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
RESUME=/projects/bhqk/htang13/v3_ckpt_${PARENT}/ladder-step=50000.ckpt
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain

# Fail LOUDLY rather than starting fresh: --resume-ckpt defaults to None, so a missing/typo'd
# path does NOT error -- it silently trains from scratch and burns the wall. This is also the
# real dependency gate (see afterany note above).
[ -f "$RESUME" ] || { echo "[FATAL] resume ckpt missing: $RESUME -- parent 2898613 did not reach 50000"; exit 1; }
# Never write into the parent dir. A fresh dir keeps each extension segment separable.
[ -e "$CKPT_DIR" ] && { echo "[FATAL] $CKPT_DIR already exists -- refusing to mix ladders"; exit 1; }

echo "[r6 vits384 EXTENSION 2] run=$RUN_NAME"
echo "[branch]   resume=$RESUME  (EXACTLY step 50000, PEAK LR, uncooled -- parent dir is 100% uncooled)"
echo "[schedule] CONSTANT 6e-3 over 50000..60000, --min-lr-ratio 1.0, NO decay, ladder=1000"
echo "[matched]  cb=$CB accum=$ACC (=$((CB*ACC)) contacts/opt-step) sf50 per_band_space=ON widths=1,1,1 lr=6e-3 seed=33 workers=32 [FILE-ONLY scp deploy]"
echo "[read]     vs the UNCOOLED rungs only (WS .7687@20k .7725@35k .7739@40k .7757@45k). NOT vs cd45k -- ext != cd."
echo "[policy]   cooldown is chosen LATER, from whichever rung the probe says stopped paying."
echo "[budget]   10000 steps x ~4.07 s/step = ~11.3 h against a 14 h wall"

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
  --ssl-max-steps 60000
