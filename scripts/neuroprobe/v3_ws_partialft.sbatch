#!/bin/bash
#SBATCH --job-name=ws_partialft
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=113560M
#SBATCH --time=04:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/ws_partialft_%A_%a.out
#
# WS partial fine-tuning pilot — unfreeze the last encoder block, keep the ridge readout fixed,
# and ask whether the frozen-probe column we have always quoted is leaving a MAE-style gap.
#
# --mem=113560M is one GPU's DefMemPerGPU so ghx4 MAX_TRES billing stays at 1000 (same rule as
# v3_probe_encode_r6.sbatch). The x11 cache lives on the GPU (~25 GB of 96), not in host RAM.
#
# COST MODEL (why 2h is generous, not sloppy): the script caches tap-11 ONCE per session with a
# full 12-block forward (~40 s, matching the measured probe encode), then re-runs ONLY block 12
# per epoch — ~1.5 s train fwd+bwd + ~3.3 s eval fwd + ~0.7 s ridge/eigh = ~6 s/epoch. With
# early stopping that is 2-4 min/cell. Stage A shards 3 ways by lr (8 cells each, 20-35 min);
# stage B shards 7 ways by session (8 cells each, ~35 min). SLURM bills actual usage, so the
# walltime margin is free; it only affects backfill priority.
#
# STAGE A  (lr sweep on ONE session, val-selected):
#   sbatch --array=0-2 v3_ws_partialft.sbatch a
# STAGE A-wd (the wd leg; a + awd 0.5 + awd 3.0 tile a 3x3 lr x wd grid):
#   sbatch --array=0-2 v3_ws_partialft.sbatch awd 0.5
# STAGE B  (full 7x4x2 grid at the stage-A winner):
#   sbatch --array=0-6 v3_ws_partialft.sbatch b <LR>
# STAGE C  (the corrected head -- matched bn preprocessing + multitask, sharded by question):
#   sbatch --array=0-2 v3_ws_partialft.sbatch c
set -euo pipefail

STAGE="${1:?usage: sbatch v3_ws_partialft.sbatch <a|awd|b|c> [LR|WD]}"
BASE=/projects/bhqk/htang13
CKPT=$BASE/v3_ckpt_v3_r6_pbspace111_sf50_lr6e-3_40k/ladder-step=20000.ckpt
TAG=pbs50_20k

# The arm is pbs50 (job 2764203) — the adoption candidate — so the FT delta lands on the number
# we are actually choosing, and pairs against the pbs50/pbs25 double dissociation read at the
# SAME step. Its frozen probe AND its encode cache already exist, which is what makes the
# feature-level parity gate (L0) possible at all.
BASELINE=$BASE/results_v3_probe_$TAG.json
PARITY_CACHE=$BASE/v3_probe_cache_$TAG

case "$STAGE" in
  a) LRS=(1e-4 3e-4 1e-3)
     # Both arms sweep here: for `head` the lr is the logistic head's own step size, so it needs
     # its own winner. Selecting one lr on block12 and imposing it on head would weaken cell B.
     SEL="--stage a --session-index 0 --lrs ${LRS[$SLURM_ARRAY_TASK_ID]} --arms block12"
     OUT=$BASE/ws_partialft_stageA_lr${LRS[$SLURM_ARRAY_TASK_ID]}.json ;;
  awd) WD="${2:?stage awd needs a weight decay}"
     # The WD leg of stage A (Ben 07-30: "add a weight decay arm as well"). Same one session and
     # same lr ladder as stage a, at a different wd, so a and awd together tile a 3x3 lr x wd grid
     # with no duplicated work. WHY IT IS A REAL KNOB HERE and not a copy of MAE's 0.05: the head
     # is ~213k params against ~3.6k training windows, ~60x overparameterized, where MAE's ViT-L
     # head faced 1.28M images. MAE itself brackets the range rather than settling it — 0.05 for
     # end-to-end FT, 0 for linear probing ("regularization is in general harmful for linear
     # probing", mae.tex:616), 0.3 for supervised-from-scratch ViT-L (mae.tex:701). No verified
     # source for wd 3.0 was found, so it is included as an ARM to be measured, not as a citation.
     LRS=(1e-4 3e-4 1e-3)
     SEL="--stage a --session-index 0 --lrs ${LRS[$SLURM_ARRAY_TASK_ID]} --wd $WD --arms block12"
     OUT=$BASE/ws_partialft_stageA_wd${WD}_lr${LRS[$SLURM_ARRAY_TASK_ID]}.json ;;
  c) # STAGE C -- THE CORRECTED-HEAD STAGE. Sharded by QUESTION, not by lr: stage A showed that
     # sweeping lr cannot rescue a head whose PREPROCESSING does not match the ridge's, so each
     # task here changes exactly ONE thing and carries its own lr ladder.
     #   0  head_bn      : encoder FROZEN, bn head. This is R0 measured in the PRIMAL -- the same
     #                     term v3_frozen_readout_sweep.py fits in the DUAL on the same features,
     #                     so the two experiments cross-check on real data. Free invariant: no
     #                     encoder param moves, so cell C MUST equal cell A; any C != A is a
     #                     feature-path bug, not a result. Its lr ladder is 10x higher because a
     #                     zero-init linear on standardized features wants a bigger step than a
     #                     pretrained block does.
     #   1  block12_bn   : the incumbent FT arm with MATCHED preprocessing. Against task 0 it
     #                     isolates "was d(D-A) = -.0060 a LayerNorm artifact?", because task 0
     #                     supplies the readout-cost offset that -.0060 confounds.
     #   2  block12_bn_mt: adds the multi-task target block -- 4 label columns on the SAME train
     #                     rows, which is VideoMAE v2's post-pre-training lever (arXiv 2303.16727)
     #                     in its leak-free form. Against task 1 it isolates "was the FT starved of
     #                     supervision?" -- our block 12 is ~790k params against ~1.3k rows.
     ARMS=(head block12 block12)
     CLRS=(1e-3,3e-3,1e-2 1e-4,3e-4,1e-3 1e-4,3e-4,1e-3)
     MTF=("" "" "--multitask")
     NAMES=(head_bn block12_bn block12_bn_mt)
     I=$SLURM_ARRAY_TASK_ID
     SEL="--stage a --session-index 0 --lrs ${CLRS[$I]} --arms ${ARMS[$I]} --head-norm bn ${MTF[$I]}"
     OUT=$BASE/ws_partialft_stageC_${NAMES[$I]}.json ;;
  b) LR="${2:?stage b needs the stage-A winning LR}"
     # `head` (encoder frozen, logistic head only) is the missing 2x2 cell: it separates readout
     # family from weights, so "FT beats the board" cannot be a logistic-vs-ridge artifact. It is
     # pointless in stage A (there is no encoder lr to sweep) and ~doubles stage B to ~6 GPU-h.
     SEL="--stage b --session-index $SLURM_ARRAY_TASK_ID --lrs $LR --arms block12"
     OUT=$BASE/ws_partialft_stageB_s$SLURM_ARRAY_TASK_ID.json ;;
  *) echo "FATAL: stage must be 'a', 'awd', 'b' or 'c', got '$STAGE'" >&2; exit 1 ;;
esac

# pbs50 is an r6 arm => MAE => there is NO EMA teacher; the tower is objective.online.*
# (the r6 encode sbatch passes --online for the same reason). A wrong choice here fails the
# strict state_dict load rather than mis-loading silently, but the default belongs on the arm.
echo "[ws-partialft] stage=$STAGE task=${SLURM_ARRAY_TASK_ID:-0} ckpt=$CKPT"
echo "[ws-partialft] baseline=$BASELINE parity_cache=$PARITY_CACHE out=$OUT"

cd /projects/bhqk/htang13/speech
module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain

srun .venv/bin/python -u -m scripts.neuroprobe.v3_ws_partialft_pilot \
  --ckpt "$CKPT" \
  --baseline-json "$BASELINE" --baseline-prefix "$TAG" \
  --parity-cache-dir "$PARITY_CACHE" \
  --band-cache-dir $ROOT/band_v3slow \
  --band-cache-dir $ROOT/band_v3mid \
  --band-cache-dir $ROOT/band_hga \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --tower online \
  --out "$OUT" \
  $SEL
