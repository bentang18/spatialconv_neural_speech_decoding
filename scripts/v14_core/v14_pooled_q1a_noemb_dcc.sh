#!/bin/bash
# DCC SLURM array for v14 pooled Q1a ablation: NO parcel embedding.
#
# Identical to Q1a (4 core, d=32, depth=3, k=3, pool=(4,8)) but passes
# --no-parcel-embedding, disabling the `support @ P_emb` atlas anchor.
#
# Purpose: test whether the soft parcel embedding actually helps the
# pooled model, or whether cross-patient gain is purely from data
# amplification + round-robin interleave.
#
# Task-ID encoding:
#     task_id = fold * 3 + seed
#
#SBATCH --job-name=v14pp_pooled_q1a_noemb
#SBATCH --partition=coganlab-gpu
#SBATCH --array=0-14
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/work/ht203/logs/v14core/%x_%A_%a.out
#SBATCH --error=/work/ht203/logs/v14core/%x_%A_%a.err

set -euo pipefail

PATIENTS="S14,S26,S33,S62"
DEPTH=3
FOLDS=(0 1 2 3 4)
SEEDS=(0 1 2)

task_id=${SLURM_ARRAY_TASK_ID}
seed_idx=$(( task_id % 3 ))
fold=$(( task_id / 3 ))
seed=${SEEDS[$seed_idx]}

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
OUT_DIR=/work/ht203/results/v14_pooled/Q1a_4core_d32_depth${DEPTH}_noemb

export LD_LIBRARY_PATH=/work/ht203/miniconda3/envs/speech/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} pooled-noemb patients=${PATIENTS} fold=${fold} seed=${seed} depth=${DEPTH}"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode pooled \
    --patients "${PATIENTS}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --backbone-depth "${DEPTH}" \
    --no-parcel-embedding \
    --out-dir "${OUT_DIR}"
