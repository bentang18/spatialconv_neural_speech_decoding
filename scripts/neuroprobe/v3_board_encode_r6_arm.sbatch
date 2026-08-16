#!/bin/bash
#SBATCH --job-name=board_encode_arm
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=01:30:00
#SBATCH --output=/projects/bhqk/htang13/board_encode_arm_%x_%j.out
#
#   sbatch --dependency=afterok:<train jobid> scripts/neuroprobe/v3_board_encode_r6_arm.sbatch vits384_nosrope
#
# ARM-AWARE board encode for the vits384 geometry 2x2 at the 55k COOLED ckpt. (Ben 2026-08-09)
#
# ── WHY THIS EXISTS AND v3_board_encode_r6.sbatch DOES NOT SUFFICE ───────────────────────
# 🪤 --no-space-rope is a NO-TRACE flag. L1RoPE registers idx_freq persistent=False, so a
# no-space-rope ckpt is key- AND value-identical to a normal one: the state_dict check CANNOT
# catch a mismatch (v3_probe_encode_r4.py:521-527). The stock launcher hardcodes its encode
# invocation with no way to pass it, so encoding vits384_nosrope / vits384_noparcel_nosrope
# through it would SILENTLY apply contact-index rotation the trained model never saw, and the
# result would look completely normal. Hence an arm-keyed launcher with a hard cross-check.
#   INFERABLE off the ckpt (self-verifying, wrong value cannot load): d_model, parcel_embed,
#   deep_sup — so --enc-width and --no-parcel-embed are NOT passed here, by design.
#   NOT INFERABLE (must be passed by hand): --no-space-rope, --parcel-perm-seed.
#   ⭐ None of these arms use --parcel-perm-seed (that is R35), so it is deliberately absent.
#
# ── PARENT ───────────────────────────────────────────────────────────────────────────────
# Each arm's terminal COOLED ckpt from the merged 40k->55k job (2906757/58/59):
#   v3_ckpt_v3_r6_pbspace111_sf50_${ARM}_ext50k_cd55k/ladder-step=55000.ckpt
# ⚠ ext != cd. ONLY step 55000 is cooled; 40500..50000 in that dir are UNCOOLED and must never
#   be boarded. The BASELINE is excluded here -- its dir is ..._vits384_cd55k (no ext50k) and it
#   was already encoded as job 2911364.
#
# ── WHAT WE READ OFF THIS ────────────────────────────────────────────────────────────────
# Paper scope (Ben 2026-08-09): the arms report **enc12 only**. The depth ladder enc0/3/6/9/12 is
# CSubject-only and is the BASELINE's result. enc0 comes out anyway and is worth checking: it is
# ARM-INDEPENDENT, so it MUST tie across all four caches -- a drift means the encodes are not
# comparable and no enc12 number from them is readable.
#
# TIMING: the d=384 cd45k board encode took 28:54 (job 2894019). 1:30 wall. Cache is ~938 G/arm.
set -euo pipefail

ARM="${1:?usage: sbatch v3_board_encode_r6_arm.sbatch <vits384_noparcel|vits384_nosrope|vits384_noparcel_nosrope>}"
case "$ARM" in
  vits384_noparcel)         SROPE_ARGS=() ;;
  vits384_nosrope)          SROPE_ARGS=(--no-space-rope) ;;
  vits384_noparcel_nosrope) SROPE_ARGS=(--no-space-rope) ;;
  vits384)
    echo "[FATAL] the BASELINE lives in ..._vits384_cd55k (not _ext50k_cd55k) and was encoded as"
    echo "        job 2911364 via v3_board_encode_r6.sbatch. Re-encoding would fork the tag."; exit 1 ;;
  *) echo "[FATAL] unknown arm '$ARM'"; exit 1 ;;
esac

# 🔴 SPACE-ROPE CROSS-CHECK, BOTH DIRECTIONS. The one flag no state_dict check can catch.
case "$ARM" in *nosrope*|*nospacerope*) NAME_SAYS_OFF=1 ;; *) NAME_SAYS_OFF=0 ;; esac
case " ${SROPE_ARGS[*]:-} " in *" --no-space-rope "*) FLAG_SAYS_OFF=1 ;; *) FLAG_SAYS_OFF=0 ;; esac
if [ "$NAME_SAYS_OFF" != "$FLAG_SAYS_OFF" ]; then
  echo "[FATAL] space_rope mismatch -- name says off=$NAME_SAYS_OFF, flags say off=$FLAG_SAYS_OFF"; exit 1
fi

CKPT=/projects/bhqk/htang13/v3_ckpt_v3_r6_pbspace111_sf50_${ARM}_ext50k_cd55k/ladder-step=55000.ckpt
TAG=board_${ARM}_cd55k
OUTDIR=/work/nvme/bhqk/htang13/v3_board_cache_${TAG}

[ -f "$CKPT" ] || { echo "[FATAL] cooled ckpt missing: $CKPT -- did the 40k->55k job reach 55000?"; exit 1; }
[ -e "$OUTDIR" ] && { echo "[FATAL] $OUTDIR already exists -- refusing to mix caches"; exit 1; }

echo "[board encode ARM] arm=$ARM space_rope_off=$NAME_SAYS_OFF tag=$TAG"
echo "[ckpt]    $CKPT (TERMINAL COOLED rung -- ext rungs in that dir are NOT boardable)"
echo "[flags]   passed by hand: ${SROPE_ARGS[*]:-<none>}   | inferred off ckpt: d_model, parcel_embed, deep_sup"
echo "[read]    arms report enc12; enc0 is arm-independent and MUST tie across all four caches"

cd /projects/bhqk/htang13/speech
module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/projects/bhqk/htang13/speech/src
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_lite
srun /projects/bhqk/htang13/speech/.venv/bin/python -u -m scripts.neuroprobe.v3_probe_encode_r4 \
  --ckpt "$CKPT" --tag "$TAG" --out-dir "$OUTDIR" \
  --band-cache-dir $ROOT/band_v3slow \
  --band-cache-dir $ROOT/band_v3mid \
  --band-cache-dir $ROOT/band_hga \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --batch-size 64 \
  --frontend v3r6 \
  ${SROPE_ARGS[@]+"${SROPE_ARGS[@]}"} \
  --sessions board --tasks board15 --electrode-set lite --elec-taps 0,12
