#!/bin/bash
#SBATCH --job-name=v3_r6_pbs50_abl_trap45k
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
#   sbatch scripts/neuroprobe/v3_r6_pbs50_ablation_trap45k.sbatch noparcel
#   sbatch scripts/neuroprobe/v3_r6_pbs50_ablation_trap45k.sbatch nosrope
#   sbatch scripts/neuroprobe/v3_r6_pbs50_ablation_trap45k.sbatch noparcel_nosrope
#
# GEOMETRY ABLATIONS OFF THE LIVE CANONICAL BOARD ARM (pbs50), FROM SCRATCH, PAPER-GRADE.
# Ben 2026-08-01: "these numbers will genuinely go into the paper, so we are on a proper pass."
# Supersedes the 20k pretrain-probe reads (#18 space-RoPE, #19 noparcel), which were taken off a
# different, pre-pbs50 baseline and at a step where the arm had not cooled down.
#
# ONE FILE, THREE ARMS -- deliberately. An OFAT is only readable if every non-ablated flag is
# byte-identical across arms; three copy-pasted launchers drift silently. The arm selects exactly
# one line (ABL below) and nothing else.
#
# ═══ WHAT EACH ARM REMOVES ═══════════════════════════════════════════════════════════════════════
# noparcel  --no-parcel-embed : the DKT parcel-identity embed is NOT CONSTRUCTED. pe.py:189-190
#   early-returns before the nn.Embedding is built, leaving self.embed = None, and forward returns
#   zeros; every call site ADDS it to the residual (towers.py:167/198/281) so zeros == never added.
#   Genuinely ripped out: no dead parameter, no weight decay on it, honest ckpt param count.
# nosrope   --no-space-rope   : L1RoPE's CONTACT-INDEX frequencies are zeroed (pe.py:90-91), so
#   ang_idx = 0 ⇒ cos 1 / sin 0 ⇒ identity rotation on the index pairs. The head_dim split
#   (pairs = head_dim//4 per axis), the pair convention and the ENTIRE TIME half are bit-identical
#   to canonical -- this removes within-shaft contact ORDER, not the rotary mechanism.
# noparcel_nosrope : both. Since the parcel embed is PARCEL-level and band_type_emb is BAND-level,
#   index-RoPE is the only remaining per-token carrier of within-shaft contact identity -- so this
#   arm is the joint control: the model keeps NO explicit spatial position at all.
#
# ⚠️ WORDING: self-attention without positional input is permutation-EQUIVARIANT, not invariant,
# and it does not "learn order". Do not write either phrase off these runs.
#
# ═══ ONE JOB, NOT TWO -- Ben: "make sure it doesn't start and stop again" ════════════════════════
# It does not. There is ONE srun, NO --resume-ckpt, and the trapezoid is a single schedule
# evaluated from step 0 to 45000 inside one process:
#     5k LINEAR warmup 0->6e-3   (--warmup-steps default 5000, dispatch_v3.py:550)
#     FLAT at 6e-3 to step 40000 (--stable-steps 40000)
#     LINEAR 6e-3 -> 0 to 45000  (--decay-shape linear --min-lr-ratio 0.0 --ssl-max-steps 45000)
# VERIFIED, not assumed: replaying _WarmupCosineLR for the merged config against the canonical
# TWO-STAGE pair (v3_r6_perbandspace.sbatch 40k, then v3_r6_pbs50_cooldown.sbatch resumed at 40000)
# gives max |merged - canonical| = 0.000e+00 over ALL 45000 steps. The merged job trains the
# identical LR trajectory the canonical pbs50_cd45k arm did, so these ablations are comparable to it.
#
# Merging also REMOVES a live trap rather than adding one: L1RoPE.idx_freq is persistent=False
# (pe.py:93-94) ⇒ absent from the state_dict ⇒ a cooldown that forgot --no-space-rope would
# SILENTLY restore base_index=8 and cool down the WRONG arm (dispatch_v3.py:546-547 documents this).
# With no resume there is no second command that can forget a flag.
#
# ═══ MATCHED TO CANONICAL ════════════════════════════════════════════════════════════════════════
# Every other flag is lifted verbatim from the pbs50 pair: 13 sessions, v3r6/mae, shaft batching
# cb 11000 alpha 0.5, lr 6e-3 wd 0.04 clip 3.0 beta2 0.95 seed 33, clip-len 2.0, winsor 15/15/20,
# bs 32 accum 6, bf16-mixed, and the mask block --mask-space-frac 0.50 --per-band-space
# --space-block-w-bands 1,1,1 (widths are 1,1,1 -- NOT 4,4,4; see the parent launcher's header).
#
# TIMING -- THE ONE REAL RISK. 45000 x ~3.467 s/step ~= 43.3 h, plus ~20 min for a COLD inductor
# cache (the r6_pbspace cache was deleted in the 07-31 storage sweep, and each arm compiles a
# different graph anyway, so each gets its own dir). ~43.7 h inside a 48 h wall = ~4 h margin.
# Ladder every 1000 per the OFAT rule. NOTE: unlike a constant-LR run, a ladder ckpt taken mid-
# cooldown is NOT a legitimate branch point -- if a job times out before 45000 the arm needs a
# requeue, not a rescue from step 44000. Watch the s/step in the first hour.
set -uo pipefail

ARM="${1:?usage: sbatch v3_r6_pbs50_ablation_trap45k.sbatch <noparcel|nosrope|noparcel_nosrope>}"
# Explicit whitelist: a typo'd arm must FAIL, never silently fall through to an unablated run that
# would look like a valid ablation in wandb and burn 43 h.
case "$ARM" in
  noparcel)          ABL="--no-parcel-embed" ;;
  nosrope)           ABL="--no-space-rope" ;;
  noparcel_nosrope)  ABL="--no-parcel-embed --no-space-rope" ;;
  *) echo "[FATAL] unknown arm '$ARM' (expected noparcel | nosrope | noparcel_nosrope)"; exit 1 ;;
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
echo "[r6 pbs50 GEOMETRY ABLATION] arm=$ARM flags='$ABL' run=$RUN_NAME"
echo "[schedule] ONE JOB, no resume: warmup 5000 LINEAR -> flat 6e-3 -> stable 40000 -> LINEAR to 0 at 45000; ladder 1000"
echo "[matched]  space_frac=0.50 per_band_space=ON widths=1,1,1 lr=6e-3 seed=33 cb=11000 accum=6 [FILE-ONLY scp deploy]"
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
  $ABL \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 1000 \
  --ssl-max-steps 45000
