#!/bin/bash
# Stage-1 close-out: 7-LH pooled-joint run of the new Stage-1 default.
#
# per_cell + partial_conv masking + factorized_2d_frozen pe2d @ d=32, depth=3,
# pool=(4,8). Extends default validation from 4-core to the full Phase-1 LH
# cohort (S14/S16/S23/S26/S33/S39/S62). Saves checkpoints for Stage-2
# warm-start / SSL init.
#
# Closes H1.2 at the originally-defined patient scope. 15 jobs.
#
# Task-ID encoding:
#     task_id = fold * 3 + seed
#
#SBATCH --job-name=v14pp_pooled_7lh_default
#SBATCH --partition=coganlab-gpu
#SBATCH --array=0-14
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/ht203/logs/v14core/%x_%A_%a.out
#SBATCH --error=/work/ht203/logs/v14core/%x_%A_%a.err

set -euo pipefail

PATIENTS="S14,S16,S23,S26,S33,S39,S62"
DEPTH=3
FOLDS=(0 1 2 3 4)
SEEDS=(0 1 2)

task_id=${SLURM_ARRAY_TASK_ID}
seed_idx=$(( task_id % 3 ))
fold=$(( task_id / 3 ))
seed=${SEEDS[$seed_idx]}

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
OUT_DIR=/work/ht203/results/v14_pooled/7lh_default_partialconv_pe2dfrozen_d32_depth${DEPTH}

export LD_LIBRARY_PATH=/work/ht203/miniconda3/envs/speech/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} pooled-7lh-default patients=${PATIENTS} fold=${fold} seed=${seed} depth=${DEPTH} arm=per_cell+partial_conv+pe2d_frozen"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode pooled \
    --patients "${PATIENTS}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --backbone-depth "${DEPTH}" \
    --masking-mode partial_conv \
    --spatial-pe-mode factorized_2d_frozen \
    --save-checkpoint \
    --out-dir "${OUT_DIR}"
