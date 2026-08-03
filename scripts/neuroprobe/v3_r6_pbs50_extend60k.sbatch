#!/bin/bash
#SBATCH --job-name=v3_r6_pbs50_extend60k
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
#
#   sbatch scripts/neuroprobe/v3_r6_pbs50_extend60k.sbatch
#
# IS 40k THE RIGHT STOP UNDER PER-BAND SPATIAL MASKING? Resume the canonical pbs50 stable phase
# 40000 -> 60000 at CONSTANT LR, then pretrain-probe the ladder against the 40k point.
# Ben 2026-08-01: "perhaps 40k is not the right stopping point - 40k was [chosen] for previous runs
# before we did pbs, but pbs gives far far more spatial mask diversity so perhaps we should train
# for even longer. I THINK THERE IS PERFORMANCE LEFT TO SQUEEZE."
#
# ═══ WHY THIS IS NOT A KNOWN-ANSWER RUN ══════════════════════════════════════════════════════════
# The "saturates ~30k" rule (MEMORY.md, ledger D1) was measured on the PRE-pbs 6e-3 keeper:
# 4-task CS .6652@10k .6719@20k .6724@30k .6747@40k .6728@50k. Its 40k->50k step is -.0019 against
# a documented 0.005 noise floor => that old 50k point is a WASH, not a decline.
# The pbs-era ladder (R22, family pbs_probe in results/r6_era/RESULTS_LEDGER.csv, enc12/ridge/std,
# coverage verified identical across arms) says something different:
#   CS (24 units): 20k .6925 | 30k .6906 | 35k .6882 | 40k .6973  <- MAX, paired vs 30k +.0067 18+/6-
#   WS (28 units): 20k .7667 | 30k .7690 | 35k .7678 | 40k .7724  <- MAX, paired vs 30k +.0034 21+/7-
# 40k is the TERMINAL point AND the MAXIMUM in both regimes. That is the signature of an early stop.
# ⚠️ HONEST BOUND: total 20k->40k movement is only ~+.005 and the ladder DIPS at 30-35k, so this
# trajectory is noisy at the .005 scale and 40k-as-max is partly luck. Both regimes turning up
# together at the last point is the reason to spend ~20 h, not proof of anything.
#
# ═══ SCHEDULE: CONSTANT LR, NOT A COOLDOWN ═══════════════════════════════════════════════════════
# --min-lr-ratio defaults to 1.0 and --stable-steps to 0 (dispatch_v3.py:551,:554) => LR is FLAT at
# 6e-3 for the whole continuation, and --ssl-max-steps is a pure STOP-POINT (dispatch_v3.py:448-452).
# The resumed global_step is 40000 > warmup 5000, so the run starts at peak LR with no re-warmup.
# 🔴 DELIBERATELY NOT A COOLDOWN. The question is "is the stable phase still climbing", which must be
# read on the SAME LR regime at both ends. Comparing an uncooled 60k against the cooled pbs50_cd45k
# would confound with the cooldown effect, which is REAL on all three regimes (board p=.041).
# ✅ Because LR is CONSTANT, EVERY ladder ckpt is a legitimate branch point -- unlike the trap45k
# arms, a timeout here costs nothing and is not a rescue, and a short read is still usable.
# TIMING: 20000 x ~3.467 s/step ~= 19.3 h + ~20 min cold inductor cache = ~19.6 h in a 24 h wall.
#
# WHY 60k AND NOT 50k (Ben 2026-08-01, "why don't we do extend -> 60k instead and scope for 24
# hours"): the read is a SHAPE against a .005 noise floor, so POINT COUNT is the binding resource --
# ladder-every-1000 gives 19 points (42k..60k) instead of 5, which is the difference between fitting
# a trend and eyeballing two numbers. Cost is +9.7 h for +100 passes of headroom.
#
# ═══ BRANCH POINT ════════════════════════════════════════════════════════════════════════════════
# /projects/bhqk/htang13/v3_ckpt_v3_r6_pbspace111_sf50_lr6e-3_40k/ladder-step=40000.ckpt
# This is the UNCOOLED end-of-stable-phase ckpt of the canonical pbs50 arm -- the same ckpt the
# cd5k_linear_from40k cooldown branched from, and the same one that produced the `pbs50_40k` probe
# rows above. So the 40k comparator is already in the ledger at MATCHED coverage. Writes go to a NEW
# ckpt dir; the source dir is not touched.
#
# ═══ THE RESUME REPLAY -- CHECKED IN CODE, NOT MATERIAL ══════════════════════════════════════════
# The PACK STREAM is epoch-keyed: ShaftPackDataset.__iter__ seeds ONE generator g = f(seed, epoch,
# stream) (shaft_dataset.py:119-124) and advances it sequentially, so a resume restarts g and
# REPLAYS packs from pack 0 -- same shafts, same t0.
# ✅ THAT IS FINE HERE, because the MASK is step-keyed and resume-safe BY DESIGN: _step_generator
# seeds f(seed, global_step) (v14_converged_v3_module.py:135-143, docstring "resume-stable (same
# step => same mask) and fresh each step"), so steps 40000-60000 draw masks that have NEVER EXISTED.
# For an MAE the mask IS the training problem; the window only supplies content already seen ~400x.
# Pack k replayed at step 40000+k = same shafts, DIFFERENT visible set and DIFFERENT reconstruction
# target = a new training problem. Mask space is combinatorial in ~11000 contacts (50% per band,
# independent, x the 50% time mask) => patterns never repeat.
# 🔢 REPETITION REGIME: corpus = 13.19 M contact-seconds (1792 contacts, 27.9 h wall); one pack =
# 11000 contacts x 2 s = 22.0 k contact-seconds; 240000 packs @40k steps => ~400 PASSES. 60k => 600.
# 🚨 The PL EPOCH COUNTER IS A TRAP: packs_per_epoch = clips_per_session x n_sessions = 520000
# (datamodule.py:83), so the 40k ckpt reads epoch=0 / epoch_progress.completed=0. That is NOT "under
# one pass" -- every pack is an independent random draw, so the epoch boundary means nothing here.
# 🔴 THE ONE PLACE REPLAY STILL KILLS YOU IS ARM-vs-ARM: the deep-sup resume was rejected
# (reference_run_config_ledger.md:220) because one arm replaying against a fresh comparator is a
# confound REGARDLESS of mask freshness. This is a SINGLE TRAJECTORY -- no arm to bias.
# ⚠️ Still not bit-equivalent to a fresh 60k run -- say so in one sentence when reporting.
#
# ⚠️ L1RoPE.idx_freq is persistent=False (pe.py:93-94) => NOT in the state_dict => any geometry flag
# omitted on resume SILENTLY reverts. This arm is geometry-UNABLATED (parcel embed ON, space RoPE ON
# at base_index=8), which is the default, so no ablation flag is passed -- but the MASK flags
# (--mask-space-frac / --per-band-space / --space-block-w-bands) are ALSO not in the state_dict and
# MUST be re-passed verbatim. They are, below, byte-identical to the canonical arm.
#
# ═══ HOW THIS GETS ADJUDICATED ═══════════════════════════════════════════════════════════════════
# 🔴 PRETRAIN PROBE ONLY, comparator = the existing `pbs50_40k` rows (CS .6973 / WS .7724), same
# script, same 4 tasks, same cells. Probe the ladder at 45k/50k/55k/60k (more rungs available if the
# shape is ambiguous) -- a SHAPE, not one point, because the .005 noise floor makes any single step
# unreadable. 400 -> 600 passes is the span being read.
# 🔴 NOTHING GOES NEAR THE BOARD unless the probe shape is clearly still rising. The board test half
# is the reported metric; picking a ship-step by looking at it is selection on test.
# ⚠️ A 4-task probe is a SCREEN. It cannot settle a board question in either direction, and the
# probe/board ship-point are known to DISAGREE (ledger D3: board plateaus ~20k, probe peaks ~40k).
set -uo pipefail

module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bhqk/htang13/torchinductor_cache_r6_ext60k
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WANDB_API_KEY="$(cat /projects/bhqk/htang13/.wandb_key)"
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"
RUN_NAME=v3_r6_pbspace111_sf50_extend60k
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
RESUME=/projects/bhqk/htang13/v3_ckpt_v3_r6_pbspace111_sf50_lr6e-3_40k/ladder-step=40000.ckpt

# Fail loudly rather than silently starting a fresh 60k run from step 0 (which would look valid in
# wandb for ~20 h and answer nothing).
if [ ! -f "$RESUME" ]; then echo "[FATAL] branch ckpt missing: $RESUME"; exit 1; fi

echo "[r6 pbs50 EXTEND] resume=$RESUME -> 60000, CONSTANT lr 6e-3 (no cooldown), ladder 1000"
echo "[matched]  space_frac=0.50 per_band_space=ON widths=1,1,1 wd=0.04 lr=6e-3 seed=33 cb=11000 accum=6 geometry=UNABLATED"
echo "[compare]  vs ledger arm_tag pbs50_40k (enc12 ridge/std): CS .6973 (24u) / WS .7724 (28u) [FILE-ONLY scp deploy]"
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
  --compile \
  --resume-ckpt "$RESUME" \
  --wandb-project intuition --run-name $RUN_NAME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 1000 \
  --ssl-max-steps 60000
