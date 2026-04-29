#!/bin/bash
# DCC SLURM array for v14 pooled Q1c: 3 core patients (S14, S26, S33) — drop S62.
#
# Purpose: isolate whether S62 is the Q1a contaminant. Q1a population PER was
# 0.794 with S62 at 0.810; if Q1c pop PER drops, S62 is hurting the 4-core pool.
#
# Config: best P8 baseline — d=32, depth=3, k=3, pool=(4,8). 15 jobs total.
#
# Task-ID encoding:
#     task_id = fold * 3 + seed
#
#SBATCH --job-name=v14pp_pooled_q1c
#SBATCH --partition=coganlab-gpu
#SBATCH --array=0-14
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/work/ht203/logs/v14core/%x_%A_%a.out
#SBATCH --error=/work/ht203/logs/v14core/%x_%A_%a.err

set -euo pipefail

PATIENTS="S14,S26,S33"
DEPTH=3
FOLDS=(0 1 2 3 4)
SEEDS=(0 1 2)

task_id=${SLURM_ARRAY_TASK_ID}
seed_idx=$(( task_id % 3 ))
fold=$(( task_id / 3 ))
seed=${SEEDS[$seed_idx]}

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/repo/speech/.venv/bin/python
OUT_DIR=/work/ht203/results/v14_pooled/Q1c_3core_d32_depth${DEPTH}

export LD_LIBRARY_PATH=/work/ht203/repo/speech/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} pooled patients=${PATIENTS} fold=${fold} seed=${seed} depth=${DEPTH}"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode pooled \
    --patients "${PATIENTS}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --backbone-depth "${DEPTH}" \
    --out-dir "${OUT_DIR}"
