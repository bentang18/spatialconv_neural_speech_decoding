#!/bin/bash
# Stage-2 baseline: 7-LH pooled-joint run of the T3.1 default.
#
# per_cell + partial_conv + factorized_2d (learned) pe2d + hierarchical_atlas
# readout @ d=32, depth=3, pool=(4,8). Fills the 7-LH row of the Stage-2
# scoreboard (cohort-growth reference at N=7 before lex expansion).
#
# Supersedes v14_pooled_7lh_default_ckpt_dcc.sh (which used pe2d_frozen + D1
# readout — the pre-T3.1 default). Save ckpts for SSL warm-start at Stage 2.
#
# 15 jobs (5 folds × 3 seeds).
#
# Task-ID encoding:
#     task_id = fold * 3 + seed
#
#SBATCH --job-name=v14pp_pooled_7lh_t31
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
PYTHON=/work/ht203/repo/speech/.venv/bin/python
OUT_DIR=/work/ht203/results/v14_pooled/7lh_t31_partialconv_pe2d_hieratlas_d32_depth${DEPTH}

export LD_LIBRARY_PATH=/work/ht203/repo/speech/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} pooled-7lh-t31 patients=${PATIENTS} fold=${fold} seed=${seed} depth=${DEPTH} arm=per_cell+partial_conv+pe2d+hierarchical_atlas"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode pooled \
    --patients "${PATIENTS}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --backbone-depth "${DEPTH}" \
    --masking-mode partial_conv \
    --spatial-pe-mode factorized_2d \
    --readout-mode hierarchical_atlas \
    --save-checkpoint \
    --out-dir "${OUT_DIR}"
