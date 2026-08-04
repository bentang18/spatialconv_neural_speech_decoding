#!/bin/bash
#SBATCH --job-name=r6_ofat40k
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
#   sbatch scripts/neuroprobe/v3_r6_pbs50_ofat40k.sbatch <vits384|notimemask>
#
# TWO OFAT ARMS OFF THE pbs50 BASELINE (Ben 2026-08-04). ONE launcher file with an arm switch,
# for the reason the trap45k launcher is one file: an OFAT is only readable if every
# non-ablated flag is byte-identical, and separate files drift.
#
# ── ARM vits384 — ViT-Small WIDTH ────────────────────────────────────────────────────────
# Ben: "Try ViT small OFAT - change d=384, heads = 6 - and keep predictor width at 1/2 encoder
# width yeah? heads the same yeah?"  Yes to both, and it is not a new sizing policy: the shipped
# config (enc 256/4, pred 128/4) ALREADY satisfies heads = d/64, pred_d = enc_d/2 and
# pred_heads = enc_heads. `--enc-width 384` evaluates that same policy at the next rung of the
# DeiT ladder, so head_dim stays 64 (encoder) and 32 (predictor):
#
#   arm      | enc d/heads | pred d/heads | head_dim enc/pred | params
#   baseline |   256 / 4   |   128 / 4    |      64 / 32      | 10.98M
#   vits384  |   384 / 6   |   192 / 6    |      64 / 32      | 24.65M  (2.25x = 1.5^2)
#
# DEPTH IS NOT TOUCHED — 12 L1 encoder blocks and 6 L1 predictor blocks either way, so the
# contrast is width alone. `width_spec` (towers.py) derives heads and the predictor from the one
# number and REJECTS an off-ladder width at construction; `width_spec(256)` reproduces the four
# locked constants exactly, so the default is byte-identical to every prior run (test_width_ofat).
#
# ⚠️ THE TEST IS ASYMMETRIC AND THAT IS NOT FIXABLE HERE. At a fixed 40k steps a 2.25x model is
# less converged per parameter than the baseline, so a null is "384 does not pay off BY 40k",
# never "width does not help". Read it that way or not at all.
#
# ── ARM notimemask — NO TEMPORAL MASK ────────────────────────────────────────────────────
# Ben: "just like how we did a no spatial mask arm - do you think we should do a no temporal mask
# arm? probably don't need full 45k steps - I just want to see early signal at ~20k steps."
#
# This is NOT merely an ablation, it is the missing half of a MATCHED-FRACTION PAIR. pbs00 masks
# 0.50 of tokens with the SPACE axis off; this arm masks 0.50 with the TIME axis off. Measured on
# the real sampler (test_masking_r6.py::test_no_time_mask_and_no_space_mask_mask_the_same_fraction):
#
#   arm         | space | time | masked fraction
#   baseline    |  .50  | .50  |     0.7500
#   pbs00       |  .00  | .50  |     0.5000
#   notimemask  |  .50  | .00  |     0.5000
#
# So notimemask vs pbs00 holds HOW MUCH is masked fixed and moves only WHICH AXIS does it. Both
# comparators are already on disk (`pbs00_20k`, `pbs50_20k` in results/r6_era/RESULTS_LEDGER.csv),
# so the read costs no extra GPU.
# 🔴 CARRY pbs00's OWN CAVEAT FORWARD: pbs00 (job 2764373) came from the hard-mask OFAT launcher at
# cb 5500 x accum 12 rather than 11000 x 6 (token-exact, seed-level). notimemask-vs-baseline is the
# clean single-knob contrast; notimemask-vs-pbs00 inherits that caveat.
# ⚠️ EXPECT IT SLOWER. Visible tokens go 25% -> 50%, so the encoder sequence roughly doubles and
# within-shaft attention is quadratic in it. Read at 20k as Ben asked; LR is CONSTANT after warmup
# so the 20k ckpt here and the baseline's 20k ckpt are schedule-identical and directly comparable.
#
# ── WHY --num-workers 12 AND NOT 32 ──────────────────────────────────────────────────────
# 🔴 --num-workers CHANGES THE DATA ORDER (shaft_dataset.py:77 wid/nw, :121 seed, :125
# packs//streams). The 08-03 throughput fix (b3ced77) moved the pretrain launchers 16c/12w ->
# 72c/32w, but EVERY pbs50 baseline these two arms are read against — pbs50_20k, pbs50_40k — was
# produced BEFORE that commit, at 12 workers. Launching at 32 would straddle the line the launchers
# themselves warn about, and the resulting nuisance is the same order (~.01, R28) as the effects
# being measured. So: 12 workers, matched.
# The 72 CPUs are kept anyway — cpus-per-task does NOT enter the data order, and on ghx4
# (1000 x max(cpus/72, mem/113560, gpus)) 72 cores bill exactly the same as 16 on one GPU. The 12
# workers simply stop being core-starved, which is a free speedup at zero cost to comparability.
#
# ── SCHEDULE ─────────────────────────────────────────────────────────────────────────────
# 40k, ONE JOB, FROM SCRATCH, NO cooldown, ladder every 1000. LR is CONSTANT after the 5000-step
# warmup (--min-lr-ratio default 1.0, dispatch_v3.py:574), so --ssl-max-steps sets RUN LENGTH ONLY:
# every ladder ckpt is a legitimate read point and a walltime cut costs at most the last 1000 steps.
# Baseline reference at 12 workers: R30's arms did 45k in ~46 h (3.70 s/step). Both arms here are
# heavier than that baseline, so 40k may not land inside the 48 h wall — that is ACCEPTED, not a
# failure. Read what the ladder holds. NEVER wire this to self-kill and never resume across arms.
set -uo pipefail
ARM="${1:?usage: sbatch v3_r6_pbs50_ofat40k.sbatch <vits384|notimemask>}"
# Explicit whitelist: a typo'd arm must FAIL, never fall through to an unablated run that would
# look like a valid OFAT in wandb and burn 40 h.
case "$ARM" in
  vits384)     ABL="--enc-width 384" ;;
  notimemask)  ABL="--mask-time-frac 0.0" ;;
  *) echo "[FATAL] unknown arm '$ARM' (expected vits384 | notimemask)"; exit 1 ;;
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
RUN_NAME=v3_r6_pbspace111_sf50_${ARM}_40k
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
echo "[r6 pbs50 OFAT 40k] arm=$ARM flags='$ABL' run=$RUN_NAME"
echo "[schedule] ONE JOB, from scratch, no resume: warmup 5000 LINEAR -> CONSTANT 6e-3 to 40000; ladder 1000"
echo "[matched]  space_frac=0.50 per_band_space=ON widths=1,1,1 lr=6e-3 seed=33 cb=11000 accum=6 workers=12 [FILE-ONLY scp deploy]"
echo "[read]     vits384 -> vs pbs50_40k @40k; notimemask -> vs pbs50_20k AND pbs00_20k @20k (matched masked fraction 0.50)"
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
  --clip-len 2.0 \
  --session-z-winsor-slow 15 --session-z-winsor-mid 15 --session-z-winsor-hga 20 \
  --batch-size 32 --accumulate-grad-batches 6 \
  --accelerator gpu --devices 1 --precision bf16-mixed \
  --num-workers 12 --monitor-every-n-steps 10 --log-every-n-steps 10 \
  --mask-space-frac 0.50 --per-band-space --space-block-w-bands 1,1,1 \
  $ABL \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 1000 \
  --ssl-max-steps 40000
