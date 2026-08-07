#!/bin/bash
#SBATCH --job-name=r6_vits384_geom
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=113560M
#SBATCH --time=47:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
#
#   sbatch scripts/neuroprobe/v3_r6_vits384_geom40k.sbatch <noparcel|nosrope|noparcel_nosrope>
#   then, per arm:
#   sbatch scripts/neuroprobe/v3_r6_pbs50_ofat_resume40k.sbatch vits384_<same arm>
#
# THE R30 GEOMETRY 2x2, RE-RUN AT ViT-Small WIDTH (Ben 2026-08-06). The d=256 grid
# (v3_r6_pbs50_ablation_trap45k.sbatch, R30) found both geometry inputs load-bearing on all
# three regimes and SUB-ADDITIVE together. This file runs the other three cells of that same
# 2x2 at --enc-width 384. The fourth cell -- vits384 with geometry intact -- is ALREADY ON
# DISK (job 2873602 to step 40000, cooled down by job 2892192), which is why this launcher
# has three arms and not four.
#
# ═══ 🔴 WHY THIS IS 40k-CONSTANT AND NOT A 45k TRAPEZOID ═════════════════════════════════
# IT DOES NOT FIT. This is arithmetic against a HARD partition limit, not a preference:
# `scontrol show partition ghx4` gives MaxTime=2-00:00:00, so 48 h is the ceiling and there is
# no walltime to buy.
#
#   MEASURED on job 2873602 (the vits384 arm, same width, same cb/accum, same 32 workers):
#     ladder ckpt mtimes 28000 -> 40000 = 47,820 s over 12,000 steps = 3.985 s/step
#     all-in, incl. cold inductor + ckpt writes: 172,800 s produced ~43,150 steps = 4.005 s/step
#   45,000 x 4.005 = 180,225 s = 50.06 h  vs a 48.0 h wall  =>  2.06 h / 4.1% SHORT.
#   Put the other way: 45k inside 48 h needs <= 3.84 s/step. ViT-Small runs 4.005.
#
# THE LEVERS WERE PRICED AND NONE OF THEM CLOSES 4.1%:
#   • accum 8 -> 6 (cb 11000) is the big one (-25% micro-batches) and it OOMs. MEASURED, job
#     2873440, dead at 00:02:28: it wanted ~96.7 GiB of a 95.08 GiB card. 8250 x 8 is already
#     the cheapest TOKEN-EXACT rung that fits (66000 = cb x accum; 7 does not divide 66000, so
#     there is no rung between 6 and 8).
#   • flash-varlen (V3_FLASH_VARLEN=1, commit 835637e) buys ~3.4%, NOT enough, and it is the
#     WRONG 3.4%. The measured saving is a FIXED ~16.8 ms per micro-batch (L1 towers 313.28 ->
#     296.5 ms; the win is 18 graph breaks -> 0, which is per-BLOCK and shape-independent).
#     Against ViT-Small's 0.495 s micro-batch that is 3.4% => 48.4 h. Still over the wall.
#     🔴 AND IT WOULD CONFOUND THE GRID. flash and njt are different kernels. The fourth cell
#     is njt-trained and finished; running these three on flash puts a kernel change inside
#     EVERY delta in the 2x2, on a grid whose headline is a sub-additive interaction of order
#     .005-.05. The recorded decision (835637e, and the roofline memo) is LAND IT OFF, FLIP IT
#     NEXT PROGRAM. Nothing here changes that.
#   • --monitor-every-n-steps is NOT a lever. The tap is already gated to the last accum
#     micro-batch (v14_converged_v3_module.py:212), so at monitor_every=10 / accum 8 it costs
#     ~0.3% amortised. The "+0.47 s/opt-step" in that comment is the cost the gate ALREADY
#     removed. 🚫 do not re-quote it as available headroom.
#   • A warm inductor cache saves ~0.3 h, and each arm compiles a different graph anyway.
#
# ⇒ SPLIT AT 40000, WHICH COSTS NOTHING SCIENTIFICALLY. The 45k trapezoid is FLAT until 40000
# and is a bit-exact prefix of a 40k constant-LR run over steps 0-39999 (min_lr_ratio defaults
# to 1.0, dispatch_v3.py:574 => constant after warmup). The trap45k header verified the merge
# of the two-stage pair against the single-schedule canonical at max |diff| = 0.000e+00 over
# all 45000 steps. So arm = this job + the resume cooldown IS the 45k trapezoid, exactly as
# the notimemask arm (job 2873441) already ran it.
# 40,000 x 4.005 = 160,200 s = 44.5 h; --time=47:00:00 leaves 2.5 h. Walltime is the BACKFILL
# knob, so this is deliberately NOT the 48 h maximum -- and the 7 h cooldown that follows
# drops into any hole, which one 48 h job never does. NEVER wire this to self-kill.
#
# ═══ WHAT EACH ARM REMOVES ═══════════════════════════════════════════════════════════════
# Lifted verbatim from v3_r6_pbs50_ablation_trap45k.sbatch so the two widths ablate the SAME
# thing. Re-read that header for the full argument; the mechanics:
#   noparcel  --no-parcel-embed : the DKT parcel-identity embed is NOT CONSTRUCTED (pe.py:189-190
#     early-returns, self.embed = None, forward returns zeros, and every call site ADDS it, so
#     zeros == never added). No dead parameter, no weight decay on it, honest param count.
#   nosrope   --no-space-rope   : L1RoPE's CONTACT-INDEX frequencies are zeroed (pe.py:90-91) =>
#     identity rotation on the index pairs. The head_dim split, the pair convention and the
#     ENTIRE TIME half stay bit-identical. Removes within-shaft contact ORDER, not rotary.
#   noparcel_nosrope : both => the model keeps NO explicit spatial position at all. This is the
#     cell that makes the 2x2 an interaction rather than two one-way ablations.
#
# ⚠️ WORDING: self-attention without positional input is permutation-EQUIVARIANT, not
# invariant, and it does not "learn order". Do not write either phrase off these runs.
#
# ⚠️ THE WIDTH ASYMMETRY IS NOT FIXABLE HERE, and it is inherited by every cell. At a fixed
# 40k+5k a 2.25x model is less converged per parameter, so any null in this grid reads "does
# not pay off BY 45k at this width", never "geometry does not matter at ViT-Small".
#
# ═══ MATCHED ═════════════════════════════════════════════════════════════════════════════
# To the vits384 cell, flag for flag: cb 8250 x accum 8 (TOKEN-EXACT at 66000 contacts/opt-step),
# --enc-width 384, sf50 + per-band-space + widths 1,1,1, lr 6e-3, wd 0.04, clip 3.0, beta2 0.95,
# seed 33, clip-len 2.0, winsor 15/15/20, bf16-mixed, workers 32, ladder 1000, 13 sessions.
# 🔴 cb is 8250 and NOT the d=256 grid's 11000 -- ViT-Small does not fit at 11000 (above). So
# vits384-grid vs d256-grid carries a cb/accum difference; every cell WITHIN this grid is
# matched, which is what the 2x2 reads. 🚫 Do not diff a vits384 cell against a d256 cell.
# ✅ WORKERS: 32 everywhere in this family, so no data-order boundary is crossed.
set -uo pipefail
ARM="${1:?usage: sbatch v3_r6_vits384_geom40k.sbatch <noparcel|nosrope|noparcel_nosrope>}"
# Explicit whitelist. A typo must FAIL, never fall through to an unablated run that would look
# like a valid cell in wandb and burn 44 h.
case "$ARM" in
  noparcel)          ABL="--no-parcel-embed" ;;
  nosrope)           ABL="--no-space-rope" ;;
  noparcel_nosrope)  ABL="--no-parcel-embed --no-space-rope" ;;
  *) echo "[FATAL] unknown arm '$ARM' (expected noparcel | nosrope | noparcel_nosrope)"; exit 1 ;;
esac
# The ablation flag is the whole point of the run; assert rather than trust the case block.
[ -n "$ABL" ] || { echo "[FATAL] ABL empty for arm '$ARM' -- this would run the vits384 CELL"; exit 1; }

WIDTH=384
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
# NAMED TO MATCH THE RESUME. v3_r6_pbs50_ofat_resume40k.sbatch builds its cache dir as
# torchinductor_cache_r6_${ARM} with ARM=vits384_<this arm>, which is this exact string, so the
# 5k cooldown starts on a cache this job already warmed (~20 min saved, and it is the only
# throughput lever in this family that costs nothing). Rename one and you must rename both.
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bhqk/htang13/torchinductor_cache_r6_vits384_${ARM}
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WANDB_API_KEY="$(cat /projects/bhqk/htang13/.wandb_key)"
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"
# 🔴 THE NAME CARRIES A GUARD. v3_probe_encode_r6.sbatch and probe_ckpt_r6.sh both derive
# --no-space-rope from the ckpt PATH matching *nosrope*, because L1RoPE.idx_freq is
# persistent=False and a nospacerope ckpt is key- AND value-identical to a normal one. Keep
# `nosrope` and `noparcel` verbatim in this run name or the probe encode goes silently wrong.
RUN_NAME=v3_r6_pbspace111_sf50_vits384_${ARM}_40k
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
echo "[r6 vits384 GEOMETRY 40k] arm=$ARM flags='--enc-width $WIDTH $ABL' run=$RUN_NAME"
echo "[schedule] ONE JOB, from scratch, no resume, ladder 1000, max=40000: warmup 5000 LINEAR -> CONSTANT 6e-3 (min_lr_ratio default 1.0)"
echo "[schedule] this is the BIT-EXACT PREFIX of the 45k trapezoid; finish with: sbatch v3_r6_pbs50_ofat_resume40k.sbatch vits384_$ARM"
echo "[matched]  cb=$CB accum=$ACC (=$((CB*ACC)) contacts/opt-step) sf50 per_band_space=ON widths=1,1,1 lr=6e-3 seed=33 workers=32 [FILE-ONLY scp deploy]"
echo "[wall]     40000 x ~4.005 s/step ~= 44.5 h inside a 47 h request; 45k does NOT fit a 48 h wall at this width (see header)"
echo "[read]     vs the vits384 cell (v3_ckpt_v3_r6_pbspace111_sf50_vits384_cd45k_resume40k) @45k -- NOT vs a d=256 arm"
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
  --clip-len 2.0 \
  --session-z-winsor-slow 15 --session-z-winsor-mid 15 --session-z-winsor-hga 20 \
  --batch-size 32 --accumulate-grad-batches $ACC \
  --accelerator gpu --devices 1 --precision bf16-mixed \
  --num-workers 32 --monitor-every-n-steps 10 --log-every-n-steps 10 \
  --mask-space-frac 0.50 --per-band-space --space-block-w-bands 1,1,1 \
  --enc-width $WIDTH \
  $ABL \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 1000 \
  --ssl-max-steps 40000
