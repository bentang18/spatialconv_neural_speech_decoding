#!/bin/bash
# Pooled Q1a + flat + label smoothing 0.1 + mixup α=0.2.
#
# Best spatial-mixing recipe (flat front-end) + two biggest training-side
# tricks from the 2026-03-30 autoresearch sweep (−4.1pp and −2.9pp on S14).
# No eval-side changes (k-NN / TTA) — keeps the model vanilla.
#
# 15 jobs total (5 folds × 3 seeds).
#
# Task-ID encoding: task_id = fold * 3 + seed
#
#SBATCH --job-name=v14pp_q1a_flat_ls_mix
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
SEEDS=(0 1 2)

task_id=${SLURM_ARRAY_TASK_ID}
seed_idx=$(( task_id % 3 ))
fold=$(( task_id / 3 ))
seed=${SEEDS[$seed_idx]}

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/repo/speech/.venv/bin/python
OUT_DIR=/work/ht203/results/v14_pooled/Q1a_4core_d32_depth${DEPTH}_flat_ls01_mix02

export LD_LIBRARY_PATH=/work/ht203/repo/speech/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} pooled-flat-ls01-mix02 patients=${PATIENTS} fold=${fold} seed=${seed}"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode pooled \
    --patients "${PATIENTS}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --backbone-depth "${DEPTH}" \
    --temporal-frontend flat \
    --label-smoothing 0.1 \
    --mixup-alpha 0.2 \
    --out-dir "${OUT_DIR}"
