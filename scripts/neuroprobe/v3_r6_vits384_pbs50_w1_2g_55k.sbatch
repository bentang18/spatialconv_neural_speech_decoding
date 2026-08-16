#!/bin/bash
#SBATCH --job-name=r6_vits384_pbs50_w1_2g_55k
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=72
#SBATCH --mem=227120M
#SBATCH --time=47:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
#
#   THE WHOLE 0 -> 55000 RUN IN ONE JOB, 2 GPUs. One per 2x2 geometry arm:
#   for A in vits384 vits384_noparcel vits384_nosrope vits384_noparcel_nosrope; do
#     sbatch scripts/neuroprobe/v3_r6_vits384_tube_2g_55k.sbatch $A
#   done
#
# CELL D OF THE 2x2 (Ben 2026-08-15, launched after the 10k read). pbs50 SPACE TIER x WIDTH 1.
# The 2x2 is {space tier on/off} x {block width 4/1}; A=2952079 B=2952162 C=2952187 are the other
# three. At 10k, w1 beat w4 on BOTH regimes (WS +.0037 t+2.69, CS +.0052 t+0.81) and pbs50 was the
# CS-preferred tier, so D is the CANDIDATE OPTIMUM, not a filler cell.
# PREDICTION, measured not guessed: population time-axis copy-R2 moves -.0234 with width under
# nospace but only -.0152 under pbs50 (half the contacts are fully tubed and have NO time-copy
# path at all), so D-C should be about 2/3 of B-A, ~+.0024 WS, near the one-seed noise floor.
#
# ═══ 🔴 WHY THIS RUN EXISTS: SIX THINGS MOVED, NOT ONE ═══════════════════════════════════
# The canonical ckpt v3_r6_pbspace111_sf50_vits384_cd55k and the mask arms 2952079 (space 0 /
# time .75 / w4) and 2952162 (same, w1) differ in SIX ways, only one of which is the mask:
#
#                        canonical cd55k              2952079 / 2952162
#   mask                 sf .50 / tf .50 / w4         sf 0 / tf .75 / w4 or w1   <- the question
#   shaft-alpha          0.5                          0.0        (NEVER ablated)
#   clips-per-session    default -> ipe 65,000        616 -> ipe 1000
#   band embed init      0.02                         1e-6 (BAND_EMB_INIT_STD, stem + predictor)
#   epoch seed           pre-fix (replayed epoch 0)   on_train_start fix
#   provenance           cooldown RESUME from 50k     fresh from 0, 1 GPU -> 2 GPU
#
# All bundled by commit a7132e0. The band init is explicitly NO-TRACE ("leaves nothing in a
# converged state_dict") and the epoch-seed bug bit RESUMED runs, which the canonical was.
# ⇒ WITHOUT THIS RUN A MASK RESULT IN EITHER DIRECTION IS UNATTRIBUTABLE. That is the #1 defect
# (arm mixing), and it applies to a WIN exactly as much as to a LOSS.
#
# THIS RUN DOES THREE JOBS AT ONCE:
#   1. Makes 2952079 and 2952162 single-knob deltas on the mask, after the fact.
#   2. Tests the alpha/epoch/init bundle ON ITS OWN against v3_r6_pbspace111_sf50_vits384_cd55k.
#   3. Gives a FRESH-FROM-0 canonical, retiring the resume the epoch-seed bug affected.
#
# ⚠️ It is NOT a clean single-knob test of any ONE chassis change against the canonical -- alpha,
# ipe, band init and the resume seam all move together here. It bounds the BUNDLE, not its parts.
#
# ═══ 🔴 WHY THIS IS A FRESH RUN AND NOT A RESUME ═════════════════════════════════════════
# The whole point is to hold the chassis fixed against 2952079/2952162, and those are fresh from
# 0. Cooling an old ckpt would reintroduce exactly the resume seam this run exists to remove.
#
# ═══ 🔴 WHY 2 GPUs, AND WHY THE CONTRACT IS UNCHANGED ════════════════════════════════════
# On ONE GPU the run does not fit: 55,000 x 4.005 s/step = 61.2 h against a 48 h MaxTime, which
# is the only reason the 1-GPU version is a 40k + 15k chain at all. Two ranks fold it into one
# job and DELETE THE RESUME SEAM, which is the single most dangerous part of this pipeline.
# The optimizer contract is preserved EXACTLY, not approximately:
#                       1 GPU (tube40k + tube_cd55k)      2 GPU (this file)
#   workers per rank                          32                        16
#   TOTAL streams                             32                        32   <- identical
#   accum per rank                             8                         4
#   contacts / opt-step               8250x8 = 66,000        8250x4x2 = 66,000
#   iterations per epoch                    1000                      1000
# shaft_dataset.py:73-82 builds the stream id as `rank * nw + wid` over `world * nw` streams, so
# world=2/nw=16 yields the SAME 32 stream ids, the SAME seeds, and the SAME pack multiset per
# epoch as world=1/nw=32. What differs is only which packs land in which opt-step group.
# ⚠ NOT bit-identical to the 1-GPU pair, and nothing here claims it is. The BATCH, the ipe and
# the per-epoch pack multiset are identical; the grouping of packs into opt-steps is not.
# 🚫 --same-session-ranks stays OFF. It forces every rank onto one session and would destroy
# the subject mixture that --shaft-alpha 0.0 exists to enforce.
# 🚫 --ddp-static-graph stays OFF. static_graph killed r3 at the first backward
# (project-v3-r3-ddp-static-graph-crash-2026-07-14). find_unused=False is safe: v3 has no
# dormant requires_grad params (dispatch_v3.py:517-533).
# 🚨 DDP SCALING IS MODELLED, NOT PROFILED. 1.5x gives 40.8 h and 1.8x gives 34.0 h, both inside
# the 47 h wall, so the decision is robust to the model being wrong; the WALL is not a bound on
# correctness anyway, because every ladder rung is an epoch boundary (see below).
#
# ═══ 🔴 --cpus-per-task=72 IS LOAD-BEARING. 36 DOES NOT LAUNCH. ══════════════════════════
# MEASURED 2026-08-14, jobs 2950160-63 vs probes 2950166/67. At --cpus-per-task=36 all four
# arms died in 3 s, before python, with:
#     srun: error: task 0 launch failed: Error configuring interconnect
# A ghx4 node is 288 CPUs / 488000M / 4x GH200 laid out as gpu:...:4(S:0-3) -- ONE GPU PER
# SOCKET, 72 CPUs per socket. At 36 CPUs/task both ranks pack onto a single socket and contend
# for that socket's one cxi NIC, and the switch/hpe_slingshot plugin refuses the launch. At 72
# each rank owns a socket, its GPU and its NIC, and srun launches clean (probe 2950166).
# 🚫 --network=single_node_vni is NOT the fix and MAKES IT WORSE: SwitchParameters carries
# single_node_vni=user, but probe 2950167 requested one and reproduced the exact failure.
# Do not add a --network flag. Single-node DDP needs NVLink, not Slingshot.
# 💸 THE CPU AND MEMORY ARE FREE. Billing is 1000 x max(cpus/72, mem_MB/113560, gpus), so at 2
# GPUs the gpus term already pins it to 2000/hr: 144 CPUs and 227,120M cost EXACTLY what 72 and
# 113,560M cost. The old request was one GPU's share of the node while being billed for two.
# ⭐ This also retires the host-memory risk. Each rank now has its own 113,560M rather than two
# ranks splitting one rank's old budget while both mmap the same band .npy files. An OOM here
# would still be the LOADER, not the model, and would still appear in the first seconds.
#
# ═══ THE SCHEDULE ════════════════════════════════════════════════════════════════════════
# --stable-steps 50000 --decay-shape linear --min-lr-ratio 0.0 --ssl-max-steps 55000:
#   0..5000       warmup, LINEAR 0 -> 6e-3
#   5000..50000   CONSTANT 6e-3
#   50000..55000  LINEAR 6e-3 -> 0
# Identical in LR to the 40k + 15k chain, which was built to reproduce exactly this curve.
#
# ═══ WHY 50000 IS THE BRANCH POINT ═══════════════════════════════════════════════════════
# UNCHANGED from the canonical, and pre-specified BOARD-BLIND: the cooldown step comes from the
# 4-task pretrain probe plateauing, never from a board read. The uncooled ladder read out to 60k
# (WS enc12|std, n=28): .76866@20k .77247@35k .77386@40k .77568@45k .77532@50k .77512@55k
# .77560@60k. Four rungs from 45k sit inside .00056.
# 🚨 CS on that probe is UNREADABLE (per-cell SD .0197) and played NO part in the choice.
# ⚠️ Carried over from a measurement made under the PER-CONTACT draw. It is the same decision
# PROCEDURE ("no further improvement over four rungs") applied to a schedule fixed in advance.
# It is NOT a claim that this contract's probe plateaus at the same step. Do not re-tune it off
# a board read.
#
# ═══ 🔴 ITERATIONS PER EPOCH = 1000 ══════════════════════════════════════════════════════
# Under --batch-unit shaft the loader is an IterableDataset with batch_size=None
# (datamodule.py:126), so --batch-size is IGNORED and ONE PACK IS ONE MICRO-BATCH. The old
# --clips-per-session 40000 gave 65,000 opt-steps per epoch, so an entire 55k run was 0.846 of
# ONE epoch and set_epoch fired exactly once, at 0. That is what made the old resume replay
# packs it had already trained on: an IterableDataset is NOT fast-forwarded on resume, __iter__
# restarts at pack 0, and the pack sequence is a pure function of (seed, EPOCH, worker).
# 616 x 13 = 8008 packs, floored to 250 per stream x 32 streams = 8000 packs, and at accum 4 on
# 2 ranks that is EXACTLY 1000 opt-steps per epoch.
# ⭐ CONSEQUENCE THAT MATTERS HERE: ladder rungs land every 1000 steps, so EVERY rung is an
# epoch boundary. If this job hits the wall, resuming from any rung draws fresh packs.
# 📏 The paper's data figure is separate and unchanged: one pass over the 3,956 contact-hour
# corpus is 107.9 opt-steps, so 55,000 steps is ~510 passes.
# 🚨 ipe DEPENDS ON world AND --num-workers. Change either, recompute both.
#
# ═══ 🔴 --shaft-alpha 0.0: THE SUBJECT AXIS IS UNIFORM (Ben 2026-08-14) ═══════════════════
# The sampler picks a SUBJECT with P ∝ n_shafts(subject)^alpha, then a shaft uniformly inside it
# (shaft_batch.py:110-134). The pool is (session, shaft) PAIRS with no dedup across a subject's
# trials (shaft_batch.py:84-97), so a subject's shafts enter once PER SESSION and alpha tempers
# RECORDING-SESSION COUNT, not electrode count. Implants here are nearly uniform (12-17 shafts):
#   subject           S1    S2    S3    S4    S6    S8    S9     max/min
#   sessions           1     5     1     1     3     1     1
#   pool refs         13    85    12    15    36    13    12
#   alpha=1         7.0% 45.7%  6.5%  8.1% 19.4%  7.0%  6.5%       7.08x
#   alpha=0.5      10.8% 27.7% 10.4% 11.7% 18.1% 10.8% 10.4%       2.66x
#   alpha=0 (THIS) 14.3% 14.3% 14.3% 14.3% 14.3% 14.3% 14.3%       1.00x
# 🚫 alpha=1 is NOT "shaft-uniform across patients", it hands subject 2 nearly half of every
# training clip. The subject is the foundation-model axis and cross-subject is the eval axis.
# ⭐ THE DECIDING PROPERTY: alpha=0 is the ONLY value whose per-subject share does not depend on
# the pool census at all. At alpha=0 every subject is exactly 1/7 by construction. At any
# alpha>0 the shares are a function of `refs`, and that census is soft: the localization CSVs
# give 1078 contacts while the loaded corpus is 939 over 13 sessions. The session multiplicity
# (1,5,1,1,3,1,1) is exact and dominates; the shafts-per-implant term is not.
# ⚠️ RESIDUALS, stated because they are real: contact MASS is still 1.55x spread (shafts differ
# in length and packs fill to a CONTACT budget, so uniform shaft draws are not uniform gradient
# mass), and S2's individual (session, shaft) refs become the RAREST in the pool, 7.08x below
# S9's. At ~510 corpus passes that is repetition, not coverage, so nothing starves.
# ⚠️ NO MEASUREMENT BEHIND THIS. alpha has never been ablated. This run changes the mask
# contract AND the sampler at once: if the board moves, it is NOT attributable to either alone.
#
# ═══ NO-TRACE FLAGS ══════════════════════════════════════════════════════════════════════
# --band-time-unit, --no-space-rope and --shaft-alpha leave NOTHING in the state_dict (there is
# no save_hyperparameters anywhere, dispatch_v3.py:574). --band-time-unit is passed EXPLICITLY
# below even though "shaft" is now the default, so this file is the record of which contract
# ran. dispatch_v3 prints a [mask] fingerprint at startup: grep it in the log to prove the arm.
set -uo pipefail

ARM="${1:?usage: sbatch v3_r6_vits384_tube_2g_55k.sbatch <vits384|vits384_noparcel|vits384_nosrope|vits384_noparcel_nosrope>}"
case "$ARM" in
  vits384)                  ABL="--enc-width 384" ;;
  vits384_noparcel)         ABL="--enc-width 384 --no-parcel-embed" ;;
  vits384_nosrope)          ABL="--enc-width 384 --no-space-rope" ;;
  vits384_noparcel_nosrope) ABL="--enc-width 384 --no-parcel-embed --no-space-rope" ;;
  *) echo "[FATAL] unknown arm '$ARM'"; exit 1 ;;
esac

# 🔴 SPACE-ROPE CROSS-CHECK, BOTH DIRECTIONS. L1RoPE.idx_freq is persistent=False, so a nosrope
# ckpt is key- AND value-identical to a normal one; the name is the only carrier.
case "$ARM"   in *nosrope*|*nospacerope*) NAME_SAYS_OFF=1 ;; *) NAME_SAYS_OFF=0 ;; esac
case " $ABL " in *" --no-space-rope "*)   FLAG_SAYS_OFF=1 ;; *) FLAG_SAYS_OFF=0 ;; esac
if [ "$NAME_SAYS_OFF" != "$FLAG_SAYS_OFF" ]; then
  echo "[FATAL] space_rope mismatch -- name says off=$NAME_SAYS_OFF, flags say off=$FLAG_SAYS_OFF (ABL='$ABL')"; exit 1
fi
# 🔴 SAME CROSS-CHECK FOR THE PARCEL EMBED. --no-parcel-embed DOES change the state_dict (the
# embed table disappears), but by then the run is spent; catch it here.
case "$ARM"   in *noparcel*) NAME_NOPARCEL=1 ;; *) NAME_NOPARCEL=0 ;; esac
case " $ABL " in *" --no-parcel-embed "*) FLAG_NOPARCEL=1 ;; *) FLAG_NOPARCEL=0 ;; esac
if [ "$NAME_NOPARCEL" != "$FLAG_NOPARCEL" ]; then
  echo "[FATAL] parcel_embed mismatch -- name says off=$NAME_NOPARCEL, flags say off=$FLAG_NOPARCEL (ABL='$ABL')"; exit 1
fi
[ -n "$ABL" ] || { echo "[FATAL] ABL empty for arm '$ARM'"; exit 1; }

CB=8250
ACC=4
WORLD=2
# 🔴 THE MASK CONTRACT, AS VARIABLES. All three leave NOTHING in the state_dict, so a wrong or
# forgotten value runs a different arm with no error. Each is cross-checked against RUN_NAME below
# and printed in the fingerprint, so the log is the record.
SF=0.50   # space fraction, per band, tubed across time -- the pbs50 tier
TF=0.50   # time fraction, per band, on that band's own grid
BW=1      # 🔴 THE ONLY KNOB vs 2952187. width 1 => _cover_rank degenerates to independent uniform
          # per-token selection = RANDOM masking on the time axis.
UNIT=contact  # 🚨 the DEFAULT flipped to "shaft" in a7132e0. Omitting this runs the TUBE.
if [ "$UNIT" != "contact" ]; then
  echo "[FATAL] band_time_unit=$UNIT -- this control must reproduce the pbs50 PER-CONTACT draw"; exit 1
fi
# Token identity is an ASSERT, not a comment: a wrong triple silently changes the optimizer step.
if [ "$((CB * ACC * WORLD))" -ne 66000 ]; then
  echo "[FATAL] cb=$CB x accum=$ACC x world=$WORLD = $((CB * ACC * WORLD)) != 66000 contacts/opt-step"; exit 1
fi
if [ "$WORLD" -ne "$SLURM_NTASKS" ]; then
  echo "[FATAL] WORLD=$WORLD but SLURM_NTASKS=$SLURM_NTASKS -- --ntasks-per-node and --devices disagree"; exit 1
fi
# 🔴 ONE SOCKET PER RANK OR srun WILL NOT LAUNCH. See the header: 36 reproduces
# "Error configuring interconnect" in 3 s. This asserts it instead of rediscovering it.
if [ "${SLURM_CPUS_PER_TASK:-0}" -lt 72 ]; then
  echo "[FATAL] cpus-per-task=${SLURM_CPUS_PER_TASK:-unset}, need 72 (one full socket per rank) or srun fails configuring the interconnect"; exit 1
fi

# ipe ASSERT. packs = CPS x sessions; the dataset floors to packs/STREAMS per worker
# (shaft_dataset.py:125) where STREAMS = world x nw, so each RANK emits (packs // STREAMS) x NW
# micro-batches and ipe = that / accum.
CPS=616
NSESS=13
NW=16
STREAMS=$(( WORLD * NW ))
IPE=$(( ((CPS * NSESS / STREAMS) * NW) / ACC ))
if [ "$IPE" -ne 1000 ]; then
  echo "[FATAL] iterations-per-epoch is $IPE, not 1000 (cps=$CPS sessions=$NSESS world=$WORLD workers=$NW accum=$ACC)"; exit 1
fi
if [ "$STREAMS" -ne 32 ]; then
  echo "[FATAL] streams=$STREAMS, not 32 -- the pack multiset no longer matches the 1-GPU contract"; exit 1
fi
if [ $((55000 % IPE)) -ne 0 ]; then
  echo "[FATAL] 55000 is not an epoch boundary at ipe=$IPE"; exit 1
fi

module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# SHARED WITH v3_r6_vits384_tube_2g_smoke.sbatch so a prior smoke run warms this (~20 min).
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bhqk/htang13/torchinductor_cache_r6_pbs50w1_2g_${ARM}
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WANDB_API_KEY="$(cat /projects/bhqk/htang13/.wandb_key)"
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"

# 🔴 THE NAME CARRIES A GUARD: v3_probe_encode_r6.sbatch and probe_ckpt_r6.sh derive
# --no-space-rope from the ckpt PATH matching *nosrope*, so keep `nosrope`/`noparcel` verbatim
# or the probe encode runs the wrong arm. `_tube_` separates this contract from every ckpt
# already on disk.
RUN_NAME=v3_r6_pbs50_sf50_t50_w1_${ARM}_2g_cd55k
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain

[ -e "$CKPT_DIR" ] && { echo "[FATAL] $CKPT_DIR already exists -- refusing to mix ladders"; exit 1; }
# 🔴 NAME/FLAG CROSS-CHECK for all three silent mask flags, both directions.
case "$RUN_NAME" in *_sf50_t50_w1_*) : ;; *) echo "[FATAL] RUN_NAME '$RUN_NAME' does not carry the sf50_t50_w1 tag"; exit 1 ;; esac
if [ "$SF" != "0.50" ] || [ "$TF" != "0.50" ] || [ "$BW" -ne 1 ]; then
  echo "[FATAL] mask mismatch -- name says sf50/t50/w1, flags say SF=$SF TF=$TF BW=$BW"; exit 1
fi
# 🔴 THIS FILE IS THE WIDTH-1 CELL. Refuse to be edited into the w4 arm by accident.
if [ "$BW" -ne 1 ]; then
  echo "[FATAL] this file is cell D (width 1); BW=$BW. Use v3_r6_vits384_pbs50_2g_55k.sbatch for width 4"; exit 1
fi

echo "[r6 vits384 pbs50 WIDTH-1 CELL D 2-GPU 0->55k] arm=$ARM flags='$ABL' run=$RUN_NAME"
echo "[contract] pbs50 MASK REPRODUCED: space=$SF per-band tubed across time, time=$TF, band_block_w=$BW, --band-time-unit $UNIT."
echo "[contract] visible = (1-$SF)x(1-$TF) = 0.25 => 75% masked. SINGLE-KNOB delta vs v3_r6_pbs50_sf50_t50_w4_${ARM}_2g_cd55k (job 2952187). FRESH from step 0."
echo "[ddp]      world=$WORLD, accum=$ACC/rank, workers=$NW/rank -> streams=$STREAMS. Batch/ipe/pack-multiset IDENTICAL to the 1-GPU pair; opt-step GROUPING is not."
echo "[slurm]    cpus-per-task=$SLURM_CPUS_PER_TASK (ONE SOCKET PER RANK -- 36 fails 'Error configuring interconnect' in 3 s), mem=227120M, billing still 2000/hr"
echo "[batch]    cb=$CB x accum=$ACC x world=$WORLD = $((CB*ACC*WORLD)) contacts/opt-step"
echo "[epoch]    cps=$CPS -> ipe=$IPE. Ladder every 1000 = EVERY rung is an epoch boundary, so a wall-clock resume draws fresh packs."
echo "[schedule] ONE JOB: warmup 5000 -> CONSTANT 6e-3 to 50000 -> LINEAR 6e-3->0 over 50000..55000 (--min-lr-ratio 0.0)"
echo "[sampler]  --shaft-alpha 0.0: uniform on the SUBJECT axis, 1/7 each BY CONSTRUCTION (census-independent)"
echo "[matched]  sf$SF time=$TF w$BW unit=$UNIT lr=6e-3 wd=0.04 clip=3.0 beta2=0.95 seed=33 [FILE-ONLY scp deploy]"
echo "[wall]     55000 steps. 1-GPU rate is 4.005 s/step (61.2 h); this needs >=1.5x to fit 47 h. SCALING IS MODELLED, NOT PROFILED."
echo "[read]     what ships is the 55000 COOLED ckpt. 🚫 never board a rung below 55000."

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
  --min-lr-ratio 0.0 --stable-steps 50000 --decay-shape linear \
  --clip-len 2.0 \
  --session-z-winsor-slow 15 --session-z-winsor-mid 15 --session-z-winsor-hga 20 \
  --batch-size 32 --accumulate-grad-batches $ACC \
  --accelerator gpu --devices $WORLD --precision bf16-mixed \
  --clips-per-session $CPS \
  --num-workers $NW --monitor-every-n-steps 10 --log-every-n-steps 10 \
  --mask-space-frac $SF --per-band-space --space-block-w-bands 1,1,1 \
  --mask-time-frac $TF --band-block-w $BW \
  --band-time-unit $UNIT \
  $ABL \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 1000 \
  --ssl-max-steps 55000
