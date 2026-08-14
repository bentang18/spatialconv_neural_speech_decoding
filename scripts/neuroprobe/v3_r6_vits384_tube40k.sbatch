#!/bin/bash
#SBATCH --job-name=r6_vits384_tube40k
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
#   JOB 1 OF 2.
#   sbatch scripts/neuroprobe/v3_r6_vits384_tube40k.sbatch vits384
#   then, when it reaches 40000:
#   sbatch --dependency=afterany:<this jobid> scripts/neuroprobe/v3_r6_vits384_tube_cd55k.sbatch vits384
#
# THE TIME-TUBE CONTRACT, FROM SCRATCH (Ben 2026-08-14). r6 drew each band's width-4 time
# blocks PER CONTACT, so a masked (contact, t) kept visible same-shaft neighbours at the SAME t
# to copy from. With ENC_LAYOUT = ("L1",)*12 there is no L2 anywhere, so L1 is the only place
# context comes from -- and that shortcut let the pretext be solved without modelling time.
# masking.py now draws the blocks PER SHAFT and shares them across the shaft's contacts.
#
# ═══ 🔴 WHY THIS IS A FRESH RUN AND NOT A RESUME ═════════════════════════════════════════
# EVERY vits384 ckpt on disk was trained under the per-contact draw. The cd55k chain resumes
# ${ARM}_ext50k @50000, which traces back through ext50k -> 40k, all old contract. Cooling that
# would give 50k steps of one mask contract and 5k of another -- arm mixing, in the one run
# that ships. The contract changes the INPUT from step 0, so it starts from step 0.
#
# ═══ 🔴 WHY THE SPLIT IS 40k + 15k AND NOT THREE JOBS ════════════════════════════════════
# `scontrol show partition ghx4` gives MaxTime=2-00:00:00. MEASURED at this width on job
# 2873602: 4.005 s/step all-in (cold inductor and ckpt writes included).
#   this job   40,000 x 4.005 = 160,200 s = 44.5 h   inside a 47 h request, 2.5 h spare
#   job 2      15,000 x 4.005 =  60,075 s = 16.7 h   inside a 19 h request
#   total      55,000 x 4.005 = 220,275 s = 61.2 h   the canonical GPU-hour figure, unchanged
# 45k in ONE job needs <= 3.84 s/step and ViT-Small runs 4.005, so 40k is the largest ladder
# rung that fits. The old chain needed THREE jobs (40k, ext50k, cd55k) only because its flat
# leg was already split; folding 40k->50k into the cooldown job costs 16.7 h of a 19 h wall
# and removes one queue wait. There is no 3rd job.
#
# ═══ THE SCHEDULE, AND WHY THE SPLIT IS EXACT ════════════════════════════════════════════
# _WarmupCosineLR._multiplier is a pure function of global_step, and load_state_dict
# (lr_schedule.py:96-106) restores ONLY progress -- it deliberately does NOT clobber
# total_steps/stable_steps/min_lr_ratio from the parent ckpt, which is what makes branching a
# flat run into a cooldown legal. So this 2-job split is BIT-IDENTICAL in LR to one 0->55k run:
#   here      warmup 5000 LINEAR -> CONSTANT 6e-3 to 40000   (--min-lr-ratio 1.0, the default)
#   job 2     CONSTANT 6e-3 to 50000, then LINEAR 6e-3 -> 0 over 50000..55000
# Step 40000 therefore sits at PEAK LR, which job 2 requires. 🚫 NEVER branch a cooldown off a
# decay-era ckpt.
#
# ═══ NO-TRACE FLAGS ══════════════════════════════════════════════════════════════════════
# --band-time-unit leaves NOTHING in the state_dict (there is no save_hyperparameters anywhere,
# dispatch_v3.py:574), exactly like --no-space-rope and --parcel-perm-seed. It is passed
# EXPLICITLY below even though "shaft" is now the default, so this file is the record of which
# contract ran. dispatch_v3 also prints a [mask] fingerprint line at startup -- grep it in the
# sbatch log to prove the arm after the fact.
# 🚨 The default was flipped to "shaft" deliberately: a forgotten flag now runs the NEW
# contract. --band-time-unit contact exists ONLY to reproduce a pre-2026-08-14 ckpt.
#
# ═══ 🔴 ITERATIONS PER EPOCH = 1000, AND WHY IT IS NOT 40000 ═════════════════════════════
# Under --batch-unit shaft the loader is an IterableDataset with batch_size=None
# (datamodule.py:126), so --batch-size is IGNORED and ONE PACK IS ONE MICRO-BATCH. The old
# --clips-per-session 40000 therefore gave 40000 x 13 = 520,000 packs = 65,000 opt-steps per
# epoch, and the ENTIRE 0->55k run was 0.846 of ONE epoch: set_epoch fired exactly once, at 0.
# Consequence, measured not assumed: an IterableDataset is NOT fast-forwarded on resume, so
# job 2 re-entered __iter__ with epoch still 0 and the SAME stream seed and replayed packs
# 0..119,999 -- 15,000 steps the parent had already trained on -- while packs 320,000..519,999
# were never reached.
# 616 x 13 = 8008 packs, floored to 250 per worker x 32 workers = 8000 packs = EXACTLY 1000
# opt-steps per epoch. 40000 and 55000 both land on epoch boundaries, and the ladder cadence
# matches. This is V-JEPA's convention: a continuous random-clip sampler has no natural pass,
# so iterations-per-epoch is DECLARED and the epoch is a scheduling and seeding block.
# 📏 For the paper the honest data figure is separate and unchanged: one pass over the 3,956
# contact-hour corpus is 107.9 opt-steps, so 55,000 steps is ~510 passes.
# 🚨 ipe DEPENDS ON --num-workers (the floor division above). Change one, recompute both.
#
# ═══ 🔴 --shaft-alpha 0.0: THE SUBJECT AXIS IS UNIFORM (Ben 2026-08-14) ═══════════════════
# The sampler picks a SUBJECT with P ∝ n_shafts(subject)^alpha, then a shaft uniformly inside
# it (shaft_batch.py:110-134). The pool is (session, shaft) PAIRS with no dedup across a
# subject's trials (shaft_batch.py:84-97), so a subject's shafts enter once PER SESSION and
# alpha ends up tempering RECORDING-SESSION COUNT, not electrode count. Implants here are
# nearly uniform (12-17 shafts, 1.42x spread), so the old alpha=0.5 was correcting a variable
# that is already flat while leaving the session imbalance half-uncorrected:
#   subject           S1    S2    S3    S4    S6    S8    S9     max/min
#   sessions           1     5     1     1     3     1     1
#   pool refs         13    85    12    15    36    13    12
#   alpha=1         7.0% 45.7%  6.5%  8.1% 19.4%  7.0%  6.5%       7.08x
#   alpha=0.5      10.8% 27.7% 10.4% 11.7% 18.1% 10.8% 10.4%       2.66x
#   alpha=0 (THIS) 14.3% 14.3% 14.3% 14.3% 14.3% 14.3% 14.3%       1.00x
# 🚫 alpha=1 is NOT "shaft-uniform across patients" -- it hands subject 2 nearly half of every
# training clip. The subject is the foundation-model axis and cross-subject is the eval axis,
# so the mixture is balanced there.
# ⭐ WHY IT COSTS NO COVERAGE: the run is ~510 passes over the corpus, so under alpha=0
# subject 2's shafts still get ~130 passes each and single-session subjects ~840. Nothing is
# starved; we are in the repetition regime, not the coverage regime. Within-subject draws stay
# uniform over all 85 refs, so all five of subject 2's sessions still appear.
# ⚠️ NO MEASUREMENT BEHIND THIS. alpha has never been ablated (the 07-21 freeze memo built the
# endpoints in precisely so it could be, and it never was). This run therefore changes the mask
# contract AND the sampler at once: if the board moves, it is NOT attributable to either alone.
set -uo pipefail

ARM="${1:?usage: sbatch v3_r6_vits384_tube40k.sbatch <vits384|vits384_noparcel|vits384_nosrope|vits384_noparcel_nosrope>}"
case "$ARM" in
  vits384)                  ABL="--enc-width 384" ;;
  vits384_noparcel)         ABL="--enc-width 384 --no-parcel-embed" ;;
  vits384_nosrope)          ABL="--enc-width 384 --no-space-rope" ;;
  vits384_noparcel_nosrope) ABL="--enc-width 384 --no-parcel-embed --no-space-rope" ;;
  *) echo "[FATAL] unknown arm '$ARM'"; exit 1 ;;
esac

# 🔴 SPACE-ROPE CROSS-CHECK, BOTH DIRECTIONS. L1RoPE.idx_freq is persistent=False, so a
# nosrope ckpt is key- AND value-identical to a normal one; the name is the only carrier.
case "$ARM"   in *nosrope*|*nospacerope*) NAME_SAYS_OFF=1 ;; *) NAME_SAYS_OFF=0 ;; esac
case " $ABL " in *" --no-space-rope "*)   FLAG_SAYS_OFF=1 ;; *) FLAG_SAYS_OFF=0 ;; esac
if [ "$NAME_SAYS_OFF" != "$FLAG_SAYS_OFF" ]; then
  echo "[FATAL] space_rope mismatch -- name says off=$NAME_SAYS_OFF, flags say off=$FLAG_SAYS_OFF (ABL='$ABL')"; exit 1
fi
[ -n "$ABL" ] || { echo "[FATAL] ABL empty for arm '$ARM'"; exit 1; }

CB=8250
ACC=8
# Token identity is an ASSERT, not a comment: a wrong pair silently changes the optimizer step.
if [ "$((CB * ACC))" -ne 66000 ]; then
  echo "[FATAL] cb=$CB x accum=$ACC = $((CB * ACC)) != 66000 contacts/opt-step"; exit 1
fi

# ipe ASSERT. packs = CPS x sessions; the dataset floors to packs/NW per worker
# (shaft_dataset.py:125), so the emitted count is (packs // NW) x NW and ipe = that / accum.
CPS=616
NSESS=13
NW=32
IPE=$(( ((CPS * NSESS / NW) * NW) / ACC ))
if [ "$IPE" -ne 1000 ]; then
  echo "[FATAL] iterations-per-epoch is $IPE, not 1000 (cps=$CPS sessions=$NSESS workers=$NW accum=$ACC)"; exit 1
fi
if [ $((40000 % IPE)) -ne 0 ]; then
  echo "[FATAL] 40000 is not an epoch boundary at ipe=$IPE -- the resume would replay a partial epoch"; exit 1
fi

module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NAMED TO MATCH JOB 2 so the cooldown starts on a cache this job already warmed (~20 min).
# Rename one and you must rename both.
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bhqk/htang13/torchinductor_cache_r6_tube_${ARM}
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WANDB_API_KEY="$(cat /projects/bhqk/htang13/.wandb_key)"
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"

# 🔴 THE NAME CARRIES TWO GUARDS.
# (1) v3_probe_encode_r6.sbatch and probe_ckpt_r6.sh derive --no-space-rope from the ckpt PATH
#     matching *nosrope*, so keep `nosrope`/`noparcel` verbatim or the probe encode goes wrong.
# (2) `_tube_` is what separates this contract from every ckpt already on disk. Job 2 REFUSES
#     to resume a path without it, which is the only mechanical bar to cooling an old-draw ckpt.
RUN_NAME=v3_r6_tube_pbspace111_sf50_${ARM}_40k
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain

[ -e "$CKPT_DIR" ] && { echo "[FATAL] $CKPT_DIR already exists -- refusing to mix ladders"; exit 1; }

echo "[r6 vits384 TIME-TUBE 40k] arm=$ARM flags='$ABL' run=$RUN_NAME"
echo "[contract] --band-time-unit shaft: width-4 time blocks drawn PER SHAFT, shared by its contacts. FRESH from step 0 -- every ckpt on disk is the old per-contact draw."
echo "[schedule] JOB 1 OF 2, no resume, ladder 1000, max=40000: warmup 5000 LINEAR -> CONSTANT 6e-3 (--min-lr-ratio 1.0 default). Step 40000 is at PEAK LR, as job 2 requires."
echo "[next]     sbatch --dependency=afterany:\$SLURM_JOB_ID scripts/neuroprobe/v3_r6_vits384_tube_cd55k.sbatch $ARM"
echo "[epoch]    clips-per-session=$CPS -> ipe=$IPE opt-steps. 40000 = epoch 40 EXACTLY, so job 2 resumes on a FRESH epoch seed. 🚫 --batch-size is IGNORED under --batch-unit shaft."
echo "[matched]  cb=$CB accum=$ACC (=$((CB*ACC)) contacts/opt-step) sf50 per_band_space=ON widths=1,1,1 lr=6e-3 seed=33 workers=$NW [FILE-ONLY scp deploy]"
echo "[wall]     40000 x ~4.005 s/step ~= 44.5 h inside a 47 h request; 45k does NOT fit a 48 h wall at this width (see header)"
echo "[read]     🚫 NOTHING here boards. What ships is the 55k COOLED ckpt from job 2."

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
  --batch-unit shaft --contact-budget $CB --shaft-alpha 0.0 \
  --lr 6e-3 --weight-decay 0.04 --grad-clip 3.0 --adam-beta2 0.95 --seed 33 \
  --clip-len 2.0 \
  --session-z-winsor-slow 15 --session-z-winsor-mid 15 --session-z-winsor-hga 20 \
  --batch-size 32 --accumulate-grad-batches $ACC \
  --accelerator gpu --devices 1 --precision bf16-mixed \
  --clips-per-session $CPS \
  --num-workers $NW --monitor-every-n-steps 10 --log-every-n-steps 10 \
  --mask-space-frac 0.50 --per-band-space --space-block-w-bands 1,1,1 \
  --band-time-unit shaft \
  $ABL \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 1000 \
  --ssl-max-steps 40000
