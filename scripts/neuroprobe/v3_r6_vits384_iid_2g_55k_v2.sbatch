#!/bin/bash
#SBATCH --job-name=r6_vits384_iid_2g_v2
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
#   sbatch scripts/neuroprobe/v3_r6_vits384_iid_2g_55k.sbatch vits384
#
# ARM E -- NO MASK STRUCTURE AT ALL. Ben 2026-08-15: "we must do this run: just random drop 75% of
# tokens - no rule at all. e.g. shuffle all encoder tokens, then drop the first 75%."
#
# ONE uniform permutation over EVERY (contact, band, token) cell of the clip; mask exactly
# round(0.75 x total). No space tier, no time tier, no block width, no per-contact budget, no
# per-band budget. This is MAE's own recipe (argsort uniform noise, keep the first 25%) applied to
# the full r6 token grid, and it is the control the factorized space (x) time mask has never had.
#
# 🔴 WHY THE GLOBAL COUNT IS EXACT AND NOT BERNOULLI. The encoder gathers visible tokens into a
# fixed (B, M_vis, d) buffer (objective.py:190), so M_vis must be a compile-time constant. A
# per-cell Bernoulli draw has the same marginal and would break the static shapes. A single
# permutation fixes the total while leaving every position free, which is exactly why Ben's
# formulation is the one that runs.
#
# ═══ 🎯 THE NULL, STATED BEFORE THE MEASUREMENT ══════════════════════════════════════════
# The ONLY thing this changes relative to v3_r6_nospace_t75_w1_<ARM>_2g_cd55k (job 2952162) is the
# per-(contact, band) masked fraction, which the per-contact draw snaps to exactly
# round(frac x length). MEASURED from the real sampler (test_masking_r6.py, 2 s clip, 20 contacts,
# 64 rows), snapped sd -> iid sd of the per-contact masked fraction:
#     band   tokens   snapped sd   iid sd   contacts fully masked
#     SLOW        8      0.00000   0.1505                    9.6%
#     MID        32      0.00000   0.0772                     0.0%
#     HGA        64      0.00000   0.0536                     0.0%
# So the freedom is real but concentrated where the grid is shortest, and HGA -- which carries 62%
# of the loss weight by token count -- moves by only ~5 points of sd. SLOW carries 8%.
# ⚖ THE HONEST PREDICTION IS A NULL OR A SMALL LOSS, and both outcomes are publishable:
#   NULL -> the exact-count constraint was never load-bearing, and the mask contract collapses to
#           ONE sentence with no structural prior of any kind: mask 75% of tokens at random.
#   LOSS -> stratification is doing real work, which is a positive finding about this signal and
#           says the field-standard i.i.d. draw is not the right default for sEEG.
#   GAIN -> the exact-count constraint was an artificial regularity the model was exploiting.
# 🚫 It is NOT a rate sweep. --iid-mask-frac exists to pin 0.75 for the rate match, not to tune.
#
# ═══ WHY THIS ARM AND NOT THE OTHER CELL ═════════════════════════════════════════════════
# The two-tier arms already answered the questions they were built for at 10k (WS enc12, paired per
# cell, n=28): the space tier is +.00777 (t 5.07) and the block width is +.00371 (t 2.69), both in
# the direction of LESS structure. R42 explains the width half mechanically -- HGA r(lag) is at its
# .058 floor by lag 2, so a width-4 block pays full conditioning cost to defeat a copy route that
# is not there. This arm is the endpoint of that trend rather than another interior cell.
# 🚨 CS DISAGREES WITH WS ON THE SPACE TIER (-.00882, t -1.74) and is UNDERPOWERED at n=24 (needs
# ~.010-.012 to resolve). Nothing here is settled on CS, and this arm does not settle it either.
#
# ═══ 🔴 WHY THIS IS A FRESH RUN AND NOT A RESUME ═════════════════════════════════════════
# The contract changes the INPUT from step 0. Cooling an old-draw ckpt would give 50k steps of one
# mask contract and 5k of another, which is arm mixing in the one run that ships.
#
# ═══ 🔴 WHY 2 GPUs, AND WHY THE CONTRACT IS UNCHANGED ════════════════════════════════════
# On ONE GPU the run does not fit: 55,000 x 4.005 s/step = 61.2 h against a 48 h MaxTime. Two ranks
# fold it into one job and DELETE THE RESUME SEAM. The optimizer contract is preserved EXACTLY:
#                       1 GPU (tube40k + tube_cd55k)      2 GPU (this file)
#   workers per rank                          32                        16
#   TOTAL streams                             32                        32   <- identical
#   accum per rank                             8                         4
#   contacts / opt-step               8250x8 = 66,000        8250x4x2 = 66,000
#   iterations per epoch                    1000                      1000
# shaft_dataset.py:73-82 builds the stream id as `rank * nw + wid` over `world * nw` streams, so
# world=2/nw=16 yields the SAME 32 stream ids, the SAME seeds, and the SAME pack multiset per
# epoch as world=1/nw=32. What differs is only which packs land in which opt-step group.
# ⚠ NOT bit-identical to the 1-GPU pair, and nothing here claims it is.
# 🚫 --same-session-ranks stays OFF. It forces every rank onto one session and would destroy
# the subject mixture that --shaft-alpha 0.0 exists to enforce.
# 🚫 --ddp-static-graph stays OFF. static_graph killed r3 at the first backward
# (project-v3-r3-ddp-static-graph-crash-2026-07-14).
#
# ═══ 🔴 --cpus-per-task=72 IS LOAD-BEARING. 36 DOES NOT LAUNCH. ══════════════════════════
# MEASURED 2026-08-14, jobs 2950160-63 vs probes 2950166/67. At --cpus-per-task=36 all four arms
# died in 3 s, before python, with "srun: error: task 0 launch failed: Error configuring
# interconnect". A ghx4 node is 288 CPUs / 488000M / 4x GH200 as gpu:...:4(S:0-3) -- ONE GPU PER
# SOCKET, 72 CPUs per socket. At 36 both ranks pack onto one socket and contend for that socket's
# single cxi NIC. At 72 each rank owns a socket, its GPU and its NIC.
# 🚫 --network=single_node_vni is NOT the fix and MAKES IT WORSE (probe 2950167).
# 💸 THE CPU AND MEMORY ARE FREE. Billing is 1000 x max(cpus/72, mem_MB/113560, gpus), so at 2 GPUs
# the gpus term pins it to 2000/hr: 144 CPUs and 227,120M cost EXACTLY what 72 and 113,560M cost.
#
# ═══ THE SCHEDULE ════════════════════════════════════════════════════════════════════════
# --stable-steps 50000 --decay-shape linear --min-lr-ratio 0.0 --ssl-max-steps 55000:
#   0..5000       warmup, LINEAR 0 -> 6e-3
#   5000..50000   CONSTANT 6e-3
#   50000..55000  LINEAR 6e-3 -> 0
# ⚠️ The 50000 branch point is carried over from a pretrain-probe plateau measured under the
# PER-CONTACT draw, and is pre-specified BOARD-BLIND. It is the same decision PROCEDURE ("no
# further improvement over four rungs") applied to a schedule fixed in advance, NOT a claim that
# this contract's probe plateaus at the same step. Do not re-tune it off a board read.
#
# ═══ 🔴 ITERATIONS PER EPOCH = 1000 ══════════════════════════════════════════════════════
# 616 x 13 = 8008 packs, floored to 250 per stream x 32 streams = 8000 packs, and at accum 4 on 2
# ranks that is EXACTLY 1000 opt-steps per epoch. Ladder rungs land every 1000, so EVERY rung is an
# epoch boundary and a wall-clock resume draws fresh packs.
# 🚨 ipe DEPENDS ON world AND --num-workers. Change either, recompute both.
#
# ═══ 🔴 --shaft-alpha 0.0: THE SUBJECT AXIS IS UNIFORM (Ben 2026-08-14) ═══════════════════
# At alpha=0 every subject is exactly 1/7 BY CONSTRUCTION, the only value whose per-subject share
# does not depend on the pool census at all. Matches every arm in this 2x2 plus canon.
# ⚠️ NO MEASUREMENT BEHIND alpha ITSELF. It has never been ablated, and it is held FIXED across
# every arm here precisely so it cannot confound the mask comparison.
#
# ═══ 🪤 NO-TRACE FLAGS -- THE #1 DEFECT ══════════════════════════════════════════════════
# --mask-iid is a store_true that leaves NOTHING in the state_dict (no save_hyperparameters
# anywhere, dispatch_v3.py:574). A forgotten flag runs the sf0/w4 TWO-TIER arm with no error and no
# way to tell afterwards. Three guards below: the flag is built into MASKFLAGS and asserted, the
# RUN_NAME must carry the iid75 tag, and --mask-space-frac must be 0.0 (the sampler itself raises
# otherwise, since the tiers compose multiplicatively and a live space tier would silently push the
# realised rate past 0.75 and break the rate match).
# dispatch_v3 prints a [mask] fingerprint at startup: grep it in the log to prove the arm.
set -uo pipefail

ARM="${1:?usage: sbatch v3_r6_vits384_iid_2g_55k.sbatch <vits384|vits384_noparcel|vits384_nosrope|vits384_noparcel_nosrope>}"
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
case "$ARM"   in *noparcel*) NAME_NOPARCEL=1 ;; *) NAME_NOPARCEL=0 ;; esac
case " $ABL " in *" --no-parcel-embed "*) FLAG_NOPARCEL=1 ;; *) FLAG_NOPARCEL=0 ;; esac
if [ "$NAME_NOPARCEL" != "$FLAG_NOPARCEL" ]; then
  echo "[FATAL] parcel_embed mismatch -- name says off=$NAME_NOPARCEL, flags say off=$FLAG_NOPARCEL (ABL='$ABL')"; exit 1
fi
[ -n "$ABL" ] || { echo "[FATAL] ABL empty for arm '$ARM'"; exit 1; }

CB=8250
ACC=4
WORLD=2

# 🔴 THE MASK CONTRACT, AS ONE ASSERTED STRING. Nothing here reaches the state_dict, so this
# variable and the RUN_NAME are the only record that arm E ran and not the two-tier default.
IIDF=0.75
SPACEF=0.0
MASKFLAGS="--mask-iid --iid-mask-frac $IIDF --mask-space-frac $SPACEF"
case " $MASKFLAGS " in *" --mask-iid "*) : ;; *) echo "[FATAL] MASKFLAGS lost --mask-iid: '$MASKFLAGS'"; exit 1 ;; esac
if [ "$IIDF" != "0.75" ]; then
  echo "[FATAL] iid_mask_frac=$IIDF, not 0.75 -- this arm is rate-matched to every two-tier arm, it is not a rate sweep"; exit 1
fi
if [ "$SPACEF" != "0.0" ]; then
  echo "[FATAL] mask_space_frac=$SPACEF -- the i.i.d. arm has NO space tier; the tiers compose multiplicatively and would mask past $IIDF"; exit 1
fi
# 🚫 These three belong to the two-tier sampler and are UNREAD under --mask-iid. Passing them would
# read as though they were active. They are deliberately absent from the srun line below.
case " $MASKFLAGS " in
  *" --band-block-w "*|*" --band-time-unit "*|*" --mask-time-frac "*|*" --per-band-space "*)
    echo "[FATAL] MASKFLAGS carries a two-tier flag that --mask-iid does not read: '$MASKFLAGS'"; exit 1 ;;
esac

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
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bhqk/htang13/torchinductor_cache_r6_iid_2g_${ARM}
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WANDB_API_KEY="$(cat /projects/bhqk/htang13/.wandb_key)"
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"

# 🔴 THE NAME CARRIES A GUARD: v3_probe_encode_r6.sbatch and probe_ckpt_r6.sh derive
# --no-space-rope from the ckpt PATH matching *nosrope*, so keep `nosrope`/`noparcel` verbatim
# or the probe encode runs the wrong arm.
# ═══ 🔴 v2 -- THE v1 RUN (job 2955569) WAS INVALID FROM STEP 1 ══════════════════════════
# v1 drew ONE permutation over the whole clip. That holds M_vis per clip (objective.py:190) so it
# compiled, launched and trained 3078 steps -- then every one of the 247 tensors went NaN in a
# SINGLE step from a pristine state (loss .9109, grad_l2 .1318, clip_ratio .044 of clip 3.0,
# rankme 276.5, input tripwire nonfinite_frac 0 with absmax pinned at the winsor bounds 20/15/15
# both before AND after). Not a blowup, not the data, not the node (gh099 ECC 0/0).
# The cause: build_visible_pack takes the per-unit VISIBLE COUNTS from CLIP 0 and applies them to
# the whole batch (pack_r4.py:212), because towers.forward_flat_pack documents cu_seqlens as
# "clip-shared (per-shaft visible count is a per-session constant)". A global permutation breaks
# that -- measured over 256 clips it violated the invariant in 235 of them and understated
# max_seqlen as 169 against a true 210, so attention was mis-grouped on most clips from step 1 and
# the varlen kernel eventually ran off the end of a block.
# 🚫 v1's ladder is therefore NOT a resume point at ANY rung, including the finite step-3000 one.
# This is a FRESH run from step 0. v1's ckpts are kept (never delete) but are scientifically void.
#
# ═══ THE RULE, AS BEN STATED IT (2026-08-16) ═══════════════════════════════════════════
#   "drop 75% random per sensor unit - whether that is shaft rn, or ECoG in the future -
#    per full joint attention unit"
# The count is exact per JOINT-ATTENTION UNIT and free in every other respect: which contacts,
# which bands, which timepoints, and the per-(contact, band) split. cu_seqlens segments the
# sequence by exactly that unit, so this IS the packing contract stated positively, and any future
# geometry whose attention unit is not a shaft inherits the correct rule with no code change.
# 📏 IT COSTS NOTHING ON THE AXIS THE ARM EXISTS TO FREE. Same montage, same rate, 256 clips,
# per-(contact,band) masked-fraction sd -- v1 global draw vs this per-unit draw:
#     SLOW .2179 -> .2181 (fully-masked contacts 31.4% -> 31.7%)
#     MID  .1076 -> .1063
#     HGA  .0766 -> .0744
# The per-contact two-tier draw snaps ALL of those to exactly 0.0000.
RUN_NAME=v3_r6_iid75_${ARM}_2g_cd55k_v2
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain

[ -e "$CKPT_DIR" ] && { echo "[FATAL] $CKPT_DIR already exists -- refusing to mix ladders"; exit 1; }
# 🔴 NAME/FLAG CROSS-CHECK for the silent --mask-iid flag. The name is the only thing that survives
# into the ckpt path, so the probe pipeline reads the arm off it.
case "$RUN_NAME" in *_iid75_*) : ;; *) echo "[FATAL] RUN_NAME '$RUN_NAME' does not carry the iid75 tag"; exit 1 ;; esac
case "$RUN_NAME" in *_w1_*|*_w4_*|*_t75_*|*_sf50_*) echo "[FATAL] RUN_NAME '$RUN_NAME' carries a TWO-TIER tag; this arm has no width and no tiers"; exit 1 ;; esac

echo "[r6 vits384 ARM E -- i.i.d. MASK, NO STRUCTURE, 2-GPU 0->55k] arm=$ARM flags='$ABL' run=$RUN_NAME"
echo "[contract] uniform draw over EVERY (contact, band, token) cell; exactly round($IIDF x cells) masked PER JOINT-ATTENTION UNIT."
echo "[contract] NO space tier, NO time tier, NO block width, NO per-contact budget, NO per-band budget."
echo "[contract] v2: the count is pinned PER UNIT, not globally. v1 (2955569) pinned it globally and NaN'd at step 3079"
echo "[contract] because build_visible_pack copies per-unit cu_seqlens from CLIP 0 (pack_r4.py:212). FRESH from step 0."
echo "[contract] mask flags: $MASKFLAGS   (--band-block-w / --band-time-unit / --mask-time-frac are UNREAD here and NOT passed)"
echo "[contract] rate-matched to every two-tier arm: (1-0.50)x(1-0.50) = 0.25 visible = 75% masked. FRESH from step 0."
echo "[why]      the global count is EXACT because the encoder gathers visible tokens into a fixed (B, M_vis, d) buffer (objective.py:190)."
echo "[why]      a per-cell Bernoulli draw has the same marginal and would break the static shapes."
echo "[E null]   the ONLY delta vs v3_r6_nospace_t75_w1_${ARM}_2g_cd55k is the per-(contact,band) masked fraction,"
echo "[E null]   which the per-contact draw snaps to exactly round(frac x length). MEASURED sd, snapped -> iid:"
echo "[E null]     SLOW (8 tok) 0.00000 -> 0.1505, 9.6% of contacts fully masked"
echo "[E null]     MID (32 tok) 0.00000 -> 0.0772, 0.0%"
echo "[E null]     HGA (64 tok) 0.00000 -> 0.0536, 0.0%   <- HGA is 62% of loss weight and barely moves"
echo "[E null]   HONEST PREDICTION: NULL or a SMALL LOSS. A NULL is the informative outcome -- it collapses"
echo "[E null]   the mask contract to one sentence with no structural prior of any kind."
echo "[ddp]      world=$WORLD, accum=$ACC/rank, workers=$NW/rank -> streams=$STREAMS. Batch/ipe/pack-multiset IDENTICAL to the 1-GPU pair; opt-step GROUPING is not."
echo "[slurm]    cpus-per-task=$SLURM_CPUS_PER_TASK (ONE SOCKET PER RANK -- 36 fails 'Error configuring interconnect' in 3 s), mem=227120M, billing still 2000/hr"
echo "[batch]    cb=$CB x accum=$ACC x world=$WORLD = $((CB*ACC*WORLD)) contacts/opt-step"
echo "[epoch]    cps=$CPS -> ipe=$IPE. Ladder every 1000 = EVERY rung is an epoch boundary, so a wall-clock resume draws fresh packs."
echo "[schedule] ONE JOB: warmup 5000 -> CONSTANT 6e-3 to 50000 -> LINEAR 6e-3->0 over 50000..55000 (--min-lr-ratio 0.0)"
echo "[sampler]  --shaft-alpha 0.0: uniform on the SUBJECT axis, 1/7 each BY CONSTRUCTION (census-independent), HELD FIXED across every arm"
echo "[matched]  iid $IIDF lr=6e-3 wd=0.04 clip=3.0 beta2=0.95 seed=33 [FILE-ONLY scp deploy]"
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
  $MASKFLAGS \
  $ABL \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 1000 \
  --ssl-max-steps 55000
