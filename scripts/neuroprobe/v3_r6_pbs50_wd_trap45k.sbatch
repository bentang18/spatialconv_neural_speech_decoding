#!/bin/bash
#SBATCH --job-name=v3_r6_pbs50_wd_trap45k
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=113560M
#SBATCH --time=48:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
#
# ⚡ LOADER SIZING (08-03): 72 CPUs / 32 workers, was 16 / 12. Pre-08-03 runs were DATALOADER-BOUND --
# GPU util averaged 43% with 37% of polls at exactly 0%, while sstat AveCPU showed the 12 workers
# eating ~11 of the 16 allocated cores, i.e. saturated at their configured parallelism. ghx4 has
# TRESBillingWeights CPU=13.888888 and MAX_TRES, so 72 CPUs bills exactly 1000.0 -- the same as the
# 1 GPU and as DefMemPerGPU=113560M. The cores are FREE up to 72; 73 would cost. --mem pinned here so
# the default can never drift and raise the bill.
# 🔴 --num-workers CHANGES THE DATA ORDER (shaft_dataset.py:77 wid/nw, :121 seed, :125 packs//streams)
# => 32 is NOT seed-matched to 12. Never straddle this line inside one arm-vs-arm comparison.
#
#   sbatch scripts/neuroprobe/v3_r6_pbs50_wd_trap45k.sbatch wd0p4
#
# WEIGHT-DECAY OFAT OFF THE LIVE CANONICAL BOARD ARM (pbs50), FROM SCRATCH, PAPER-GRADE.
# Ben 2026-08-01, after reading Kim/Kotha/Liang/Hashimoto, "Pre-training under Infinite Compute"
# (arXiv 2509.14786): "queue up a baseline OFAT - only change is 0.4 WD instead of 0.04".
#
# ═══ THE ONE CHANGE ══════════════════════════════════════════════════════════════════════════════
#   --weight-decay 0.04  ->  --weight-decay 0.4      (10x)
# NOTHING ELSE MOVES. This file is the geometry-ablation launcher
# (v3_r6_pbs50_ablation_trap45k.sbatch, md5 f386c0c9a151567ab0bc063106cad736) with the arm
# whitelist reduced to one entry, ABL emptied, and that single flag changed. VERIFY BY DIFF
# against that file before submitting -- do not trust this comment.
#
# ═══ WHY 0.4, AND WHAT WOULD MAKE IT WRONG ═══════════════════════════════════════════════════════
# The paper fixes a 200M-token corpus and asks what to do with unlimited compute. Under a standard
# recipe, parameter scaling TURNS OVER (their 1.4B is worse than their 600M); what unlocks it is
# far stronger regularization -- tuned wd 0.8 @150M -> 3.2 @1.4B against a standard-practice 0.1.
# Our regime rhymes: fixed corpus, compute available, and the r6 arm SATURATES ~30k.
#
# 🔴 THE 30x DOES NOT TRANSFER AS A MULTIPLIER, and this launcher does not claim it does:
#   - their optimum RISES with parameter count and their SMALLEST model (150M) is ~15x ours
#     (12xL1 d256 encoder + 6xL1 d128 predictor, towers.py:49-53 => order 10M params);
#   - decoupled-AdamW regularization strength is the lr x wd PRODUCT, and their LR is not recorded
#     here, so no honest ratio against our 6e-3 x 0.04 exists;
#   - their high wd is compensating 8-16 EPOCHS over a fixed corpus. Our passes-per-45k is UNKNOWN.
#     If we are near one pass, high wd is predicted to HURT and this arm is close to known-answer.
# 0.4 is chosen as the smallest jump that is MEASURABLE, not as their optimum: arm-vs-arm sd is
# ~.01, so anything under ~5x is unreadable at one seed. The defensible statement is only that
# 0.04 sits BELOW standard practice (0.1), not that 0.4 is right.
#
# ═══ HOW THIS GETS ADJUDICATED -- READ BEFORE QUOTING ANY NUMBER ═════════════════════════════════
# 🔴 COMPARATOR IS THE CANONICAL pbs50_cd45k ARM, *NOT* the three geometry ablations. Those remove
# components; this one changes an optimizer constant off the same unablated baseline.
# 🔴 PRETRAIN PROBE FIRST. The board test half is the reported metric; picking this arm by looking
# at it is selection on test, which is exactly the winner's curse that took the FT pilot from
# +.0320 to +.0035. Promote to a board eval ONLY if the probe clears, and report the probe result
# either way. A wd arm that "wins" only on the board is not a result.
# 🔴 wd 0.04 is part of the v3 LOCKED contract. This arm is an OFAT ON that contract, authorized by
# Ben 2026-08-01. It does NOT change the contract; a keep decision would be a separate call.
#
# ═══ ONE JOB, NOT TWO ════════════════════════════════════════════════════════════════════════════
# ONE srun, NO --resume-ckpt, trapezoid evaluated from step 0 to 45000 inside one process:
#     5k LINEAR warmup 0->6e-3   (--warmup-steps default 5000, dispatch_v3.py:550)
#     FLAT at 6e-3 to step 40000 (--stable-steps 40000)
#     LINEAR 6e-3 -> 0 to 45000  (--decay-shape linear --min-lr-ratio 0.0 --ssl-max-steps 45000)
# This is the same merged schedule the geometry arms use, which was verified to reproduce the
# canonical two-stage pbs50 -> cooldown LR trajectory to max |diff| = 0.000e+00 over all 45000
# steps. So this arm is step-comparable to pbs50_cd45k at every shared step.
#
# ═══ MATCHED TO CANONICAL ════════════════════════════════════════════════════════════════════════
# 13 sessions, v3r6/mae, shaft batching cb 11000 alpha 0.5, lr 6e-3 clip 3.0 beta2 0.95 seed 33,
# clip-len 2.0, winsor 15/15/20, bs 32 accum 6, bf16-mixed, mask block --mask-space-frac 0.50
# --per-band-space --space-block-w-bands 1,1,1. Geometry UNABLATED (parcel embed ON, space RoPE ON).
#
# TIMING. 45000 x ~3.467 s/step ~= 43.3 h, plus ~20 min for a COLD inductor cache (the r6_pbspace
# cache was deleted in the 07-31 storage sweep). ~43.7 h inside a 48 h wall = ~4 h margin.
# Ladder every 1000 per the OFAT rule. NOTE: a ladder ckpt taken mid-cooldown is NOT a legitimate
# branch point -- a timeout needs a requeue, not a rescue from step 44000. Watch s/step in hour 1.
set -uo pipefail

ARM="${1:?usage: sbatch v3_r6_pbs50_wd_trap45k.sbatch wd0p4}"
# Explicit whitelist, same guard as the geometry launcher: a typo'd arm must FAIL rather than
# silently fall through to a run that would look valid in wandb and burn 43 h.
case "$ARM" in
  wd0p4) ABL="" ; WD="0.4" ;;
  *) echo "[FATAL] unknown arm '$ARM' (expected wd0p4)"; exit 1 ;;
esac

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
RUN_NAME=v3_r6_pbspace111_sf50_${ARM}_trap45k
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
echo "[r6 pbs50 WEIGHT-DECAY OFAT] arm=$ARM wd=$WD (canonical 0.04) run=$RUN_NAME"
echo "[schedule] ONE JOB, no resume: warmup 5000 LINEAR -> flat 6e-3 -> stable 40000 -> LINEAR to 0 at 45000; ladder 1000"
echo "[matched]  space_frac=0.50 per_band_space=ON widths=1,1,1 lr=6e-3 seed=33 cb=11000 accum=6 geometry=UNABLATED [FILE-ONLY scp deploy]"
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
  --batch-unit shaft --contact-budget 11000 --shaft-alpha 0.5 \
  --lr 6e-3 --weight-decay $WD --grad-clip 3.0 --adam-beta2 0.95 --seed 33 \
  --min-lr-ratio 0.0 --stable-steps 40000 --decay-shape linear \
  --clip-len 2.0 \
  --session-z-winsor-slow 15 --session-z-winsor-mid 15 --session-z-winsor-hga 20 \
  --batch-size 32 --accumulate-grad-batches 6 \
  --accelerator gpu --devices 1 --precision bf16-mixed \
  --num-workers 32 --monitor-every-n-steps 10 --log-every-n-steps 10 \
  --mask-space-frac 0.50 --per-band-space --space-block-w-bands 1,1,1 \
  $ABL \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 1000 \
  --ssl-max-steps 45000
