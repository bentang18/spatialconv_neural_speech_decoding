#!/bin/bash
#SBATCH --job-name=r6_ofat_cd
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
#   sbatch scripts/neuroprobe/v3_r6_pbs50_ofat_resume40k.sbatch <vits384|notimemask>
#   sbatch scripts/neuroprobe/v3_r6_pbs50_ofat_resume40k.sbatch <vits384_noparcel|vits384_nosrope|vits384_noparcel_nosrope>
#
# FINISHES EVERY accum-8 ARM THAT DOES NOT FIT 45k IN ONE 48 h JOB.
#
# TWO FAMILIES, ONE FILE, because they share the whole contract that matters: branch at step
# 40000, cb 8250 x accum 8, LINEAR 6e-3 -> 0 over 40000..45000, and a flag that must be
# re-passed or the run is silently the wrong arm.
#   R36 OFAT      : vits384, notimemask                    (parents below, launched 08-04)
#   vits384 2x2   : vits384_{noparcel,nosrope,noparcel_nosrope}
#                   parents from v3_r6_vits384_geom40k.sbatch, which runs 40k CONSTANT on
#                   purpose -- 45k does not fit a 48 h wall at ViT-Small (4.005 s/step x 45000
#                   = 50.06 h vs a HARD ghx4 MaxTime of 2-00:00:00). That launcher's header
#                   carries the full arithmetic and prices every lever that was rejected.
#
# WHY THIS EXISTS. Both accum-8 arms reach step 40000 but neither reaches 45000, so each needs
# the 5k cooldown run as its own job. They get there by DIFFERENT routes, because they were
# launched from different revisions of v3_r6_pbs50_ofat45k.sbatch -- verified 2026-08-04 by
# reading the live job headers, not inferred from the launcher:
#
#   arm        | job     | parent run name       | parent schedule        | s/step | outcome
#   vits384    | 2873602 | ..._vits384_cd45k     | 45k trapezoid          | 4.003  | 50.0 h > 48 h wall, dies ~43,150
#   notimemask | 2873441 | ..._notimemask_40k    | 40k CONSTANT 6e-3      | 3.866  | 43.0 h < 48 h wall, completes 40k
#   cosine     | 2873604 | ..._cosine_45k        | 45k cosine             | 2.817  | 35.2 h, lands 45,000 ✓
#  (parcelperm | 2872940 | ..._parcelperm_trap45k| 45k trapezoid          | 2.828  | 35.3 h, lands 45,000 ✓ -- R35)
#
# 🔴 THE PARENT CKPT DIR THEREFORE DIFFERS BY ARM and is NOT derivable from one suffix. That is
# why PARENT is set per-arm in the case block below rather than built from a shared template.
# cosine and parcelperm need nothing from this file; passing `cosine` is a FATAL below, not a
# no-op, because running it would fork a second copy of an arm that already completed.
#
# ⚠️ BOTH ROUTES LAND ON THE SAME BRANCH POINT. notimemask's 40k constant-LR run IS the flat
# prefix; vits384's trapezoid is flat over the same range. Neither arm's step-40000 ckpt has
# seen any decay, so one cooldown schedule finishes both.
#
# ── BRANCH FROM EXACTLY STEP 40000, NOT FROM WHERE THE PARENT DIED ───────────────────────
# The trapezoid is FLAT until 40000 and is a bit-exact prefix of a 40k constant-LR run over
# steps 0-39999 (verified against lr_schedule.py 2026-08-04, multiplier identical at every
# step), so the step-40000 ladder ckpt IS a legitimate branch point. Ckpts taken DURING the
# decay are NOT -- that is what the trap45k header's warning is about.
# 🗑 vits384: DISCARD the 41k-43k ckpts. Ben 2026-08-04 chose to let 2873602 ride rather than
# restart ("keep it running I think - we checkpoint every 1k so no problemo"); only the ~3.6 h
# partial cooldown is lost. This job regenerates 40000->45000 cleanly.
# ✅ notimemask has NOTHING to discard -- its parent is constant-LR and never decays at all.
#
# ── 🔴 THE ONE THING THAT WOULD SILENTLY RUIN THIS RUN ───────────────────────────────────
# The ablated flag MUST be re-passed. It is applied from args and is NOT restored from the
# ckpt, exactly like --per-band-space / --mask-space-frac in v3_r6_pbs50_cooldown.sbatch and
# --parcel-perm-seed in R35. THE ARMS FAIL DIFFERENTLY:
#   • --enc-width 384   fails LOUD   -- a forgotten flag builds d=256 and the load raises
#                                       "size mismatch" (test_wrong_width_shell_cannot_
#                                       silently_load_a_ckpt).
#   • --no-parcel-embed fails LOUD   -- the embed is a real parameter, so its presence or
#                                       absence changes the state_dict KEYS and the strict
#                                       load rejects the mismatch.
#   • --mask-time-frac 0.0 fails SILENT -- it leaves NO trace in the state_dict. A notimemask
#                                       cooldown that forgets it restores the default temporal
#                                       mask and trains the BASELINE arm for its last 5k steps,
#                                       with no error and no log line saying so.
#   • --no-space-rope   fails SILENT -- L1RoPE registers idx_freq persistent=False (pe.py:93-94)
#                                       so a nosrope ckpt is key- AND value-identical to a
#                                       normal one. A cooldown that forgets it restores
#                                       base_index=8 and cools down the WRONG arm
#                                       (dispatch_v3.py:546-547 documents exactly this).
# That asymmetry is why this is ONE file with a whitelist rather than five copy-and-edit files:
# the loud failure needs no protection, the silent one does, and separate files drift. Same
# argument the ofat45k launcher makes in its own header.
#
# ── WHAT IS MATCHED ──────────────────────────────────────────────────────────────────────
# Everything non-ablated is byte-identical to the parent launcher: cb 8250 x accum 8 (token-
# exact at 66000 contacts/opt-step), sf50 + per-band-space + widths 1,1,1, lr 6e-3, wd 0.04,
# clip 3.0, beta2 0.95, seed 33, workers 32. The decay is the canonical one the published
# baseline ran: LINEAR 6e-3 -> 0 over 40000->45000, --min-lr-ratio 0.0.
# LADDER 500, not 1000 -- matches the canonical cooldown (v3_r6_pbs50_cooldown.sbatch:93) so
# the cooldown segment is read at the same resolution as pbs50_cd45k.
#
# TORCHINDUCTOR_CACHE_DIR reuses the PARENT ARM's cache (torchinductor_cache_r6_${ARM}), which
# is already warm from the 40k+ this arm just ran -- a cold cache costs ~20 min on this arch.
#
# TIMING: 5000 steps x ~4.0 s/step ~= 5.6 h warm. 7 h walltime for margin; walltime is the
# BACKFILL knob, so do not inflate it further. NEVER wire this to self-kill.
set -uo pipefail
ARM="${1:?usage: sbatch v3_r6_pbs50_ofat_resume40k.sbatch <vits384|notimemask|vits384_noparcel|vits384_nosrope|vits384_noparcel_nosrope>}"
# Explicit whitelist. A typo must FAIL, never fall through to an unablated run that would look
# like a valid OFAT in wandb and burn 6 h.
# PARENT is read from the LIVE job headers (2026-08-04), not built from a shared suffix: the two
# arms were launched from different launcher revisions and their ckpt dirs genuinely differ.
# The vits384_* parents ARE derivable from the arm (one launcher, one revision, one naming
# rule) -- but they are written out anyway, in the same column as the two that are not, so a
# reader never has to know which half of this table is a template and which is a fact.
case "$ARM" in
  vits384)     ABL="--enc-width 384"      ; PARENT=v3_r6_pbspace111_sf50_vits384_cd45k    ;;
  notimemask)  ABL="--mask-time-frac 0.0" ; PARENT=v3_r6_pbspace111_sf50_notimemask_40k   ;;
  vits384_noparcel)
    ABL="--enc-width 384 --no-parcel-embed"
    PARENT=v3_r6_pbspace111_sf50_vits384_noparcel_40k ;;
  vits384_nosrope)
    ABL="--enc-width 384 --no-space-rope"
    PARENT=v3_r6_pbspace111_sf50_vits384_nosrope_40k ;;
  vits384_noparcel_nosrope)
    ABL="--enc-width 384 --no-parcel-embed --no-space-rope"
    PARENT=v3_r6_pbspace111_sf50_vits384_noparcel_nosrope_40k ;;
  cosine)      echo "[FATAL] cosine finishes 45k in ONE job (35.2 h < 48 h wall) and needs no resume"; exit 1 ;;
  *) echo "[FATAL] unknown arm '$ARM' (expected vits384 | notimemask | vits384_noparcel | vits384_nosrope | vits384_noparcel_nosrope)"; exit 1 ;;
esac
# 🔴 SPACE-ROPE CROSS-CHECK, both directions. This is the ONE flag no state_dict check can
# catch, so the arm NAME and the flag must agree or the job refuses to run -- the same guard
# v3_probe_encode_r6.sbatch applies at encode time, for the same reason.
case "$ARM"    in *nosrope*|*nospacerope*) NAME_SAYS_OFF=1 ;; *) NAME_SAYS_OFF=0 ;; esac
case " $ABL "  in *" --no-space-rope "*)   FLAG_SAYS_OFF=1 ;; *) FLAG_SAYS_OFF=0 ;; esac
if [ "$NAME_SAYS_OFF" != "$FLAG_SAYS_OFF" ]; then
  echo "[FATAL] space_rope mismatch -- arm name says off=$NAME_SAYS_OFF, flags say off=$FLAG_SAYS_OFF (ABL='$ABL')"; exit 1
fi
# The ablation flag is the whole point of the run. An empty ABL here means a refactor dropped
# it, which for notimemask is the SILENT failure above -- assert rather than trust the case.
[ -n "$ABL" ] || { echo "[FATAL] ABL empty for arm '$ARM' -- this would run the BASELINE"; exit 1; }

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
RUN_NAME=v3_r6_pbspace111_sf50_${ARM}_cd45k_resume40k
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
RESUME=/projects/bhqk/htang13/v3_ckpt_${PARENT}/ladder-step=40000.ckpt
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
# Fail LOUDLY rather than starting a fresh run from step 0: --resume-ckpt defaults to None, so
# a missing/typo'd path does NOT error -- it silently trains from scratch and burns the wall.
[ -f "$RESUME" ] || { echo "[FATAL] resume ckpt missing: $RESUME"; exit 1; }
echo "[r6 OFAT cooldown resume] arm=$ARM flags='$ABL' run=$RUN_NAME"
echo "[branch]   resume=$RESUME  (EXACTLY step 40000 -- the flat prefix; decay-era ckpts are NOT branch points)"
echo "[schedule] stable=40000 -> LINEAR 6e-3->0 over 40000..45000, --min-lr-ratio 0.0, ladder=500"
echo "[matched]  cb=$CB accum=$ACC (=$((CB*ACC)) contacts/opt-step) sf50 per_band_space=ON widths=1,1,1 lr=6e-3 seed=33 workers=32 [FILE-ONLY scp deploy]"
echo "[read]     vs pbs50_cd45k @45k (the CANONICAL baseline, not pbs50_40k)"
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
  --min-lr-ratio 0.0 --stable-steps 40000 --decay-shape linear \
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
  --ssl-max-steps 45000
