#!/bin/bash
#SBATCH --job-name=v3_r6_pbspace111_sf50_cd5k_linear_from40k
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=113560M
#SBATCH --time=06:00:00
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
#   sbatch scripts/neuroprobe/v3_r6_pbs50_cooldown.sbatch
#
# WSD COOLDOWN off the pbs50 arm's 40k ckpt -- the candidate submission checkpoint.
#
# PROVENANCE: byte-identical to the CANONICAL cooldown that produced board_cdlin_45k
# (/projects/bhqk/htang13/v3_r6_cd5k_linear_from40k.sbatch, job 2728166) except for the three
# lines that MUST change (RUN_NAME, RESUME, TORCHINDUCTOR_CACHE_DIR) and the mask block below.
# Same stable/decay/target/ladder: LR flat at 6e-3 through step 40000 (--stable-steps 40000), then
# LINEAR 6e-3 -> 0 (--min-lr-ratio 0.0 --decay-shape linear) over 40000->45000. 5k/45k = 11%,
# inside the WSD 10-20% band. Ladder every 500 to read the curve.
#
# 🔴 THE ONE THING THAT WOULD SILENTLY RUIN THIS RUN -- the mask flags MUST be re-passed.
# --per-band-space is action="store_true" (default False) and --space-block-w-bands defaults to
# None (dispatch_v3.py:621,628); --mask-space-frac is applied from args, not restored from the
# ckpt (:338-362). A cooldown that omits them RESTORES THE SHARED SPATIAL TUBE at the default
# fraction and the last 5k steps would train a DIFFERENT arm than the 40k they branch from --
# with no error and no log line saying so. This is the same class of trap dispatch_v3.py:546
# already documents for L1RoPE.idx_freq (persistent=False ⇒ absent from the state_dict).
# The three values below are read straight off the parent launcher
# (scripts/neuroprobe/v3_r6_perbandspace.sbatch) and the parent run name
# v3_r6_pbspace111_sf50_lr6e-3_40k, which encodes widths 111 and sf50 verbatim.
#
# WIDTHS ARE 1,1,1 -- NOT 4,4,4. Ben 07-29 ("I think we swap to 0.25 with width 1") overrode the
# R20-derived width 4. Both live pbs arms ran 1,1,1 and the run name says so. Do not "fix" this
# against the R20 memo, which records the derivation that was then overridden.
#
# TIMING: 5000 x ~3.47 s/step ~= 4.8 h + warm-cache compile; 6 h walltime for margin. Reusing the
# parent's TORCHINDUCTOR_CACHE_DIR (..._r6_pbspace) keeps the compile warm -- a cold inductor cache
# costs ~20 min on this arch.
set -uo pipefail
module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bhqk/htang13/torchinductor_cache_r6_pbspace
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WANDB_API_KEY="$(cat /projects/bhqk/htang13/.wandb_key)"
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"
RUN_NAME=v3_r6_pbspace111_sf50_cd5k_linear_from40k
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
RESUME=/projects/bhqk/htang13/v3_ckpt_v3_r6_pbspace111_sf50_lr6e-3_40k/ladder-step=40000.ckpt
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
# Fail LOUDLY rather than starting a fresh run from step 0: --resume-ckpt defaults to None, so a
# missing/typo'd path does NOT error -- it silently trains from scratch and burns the walltime.
[ -f "$RESUME" ] || { echo "[FATAL] resume ckpt missing: $RESUME"; exit 1; }
echo "[r6 pbs50 WSD COOLDOWN] run=$RUN_NAME resume=$RESUME stable=40000 -> decay LINEAR 6e-3->0 over 5k (max=45000) ladder=500 space_frac=0.50 per_band_space=ON widths=1,1,1 [FILE-ONLY scp deploy]"
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
  --lr 6e-3 --weight-decay 0.04 --grad-clip 3.0 --adam-beta2 0.95 --seed 33 \
  --min-lr-ratio 0.0 --stable-steps 40000 --decay-shape linear \
  --clip-len 2.0 \
  --session-z-winsor-slow 15 --session-z-winsor-mid 15 --session-z-winsor-hga 20 \
  --batch-size 32 --accumulate-grad-batches 6 \
  --accelerator gpu --devices 1 --precision bf16-mixed \
  --num-workers 32 --monitor-every-n-steps 10 --log-every-n-steps 10 \
  --mask-space-frac 0.50 --per-band-space --space-block-w-bands 1,1,1 \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --resume-ckpt $RESUME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 500 \
  --ssl-max-steps 45000
