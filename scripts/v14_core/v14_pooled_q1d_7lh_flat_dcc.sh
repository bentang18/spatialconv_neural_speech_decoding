#!/bin/bash
# Pooled Q1d (7 LH) with the flat Conv1d front-end.
#
# Phase-1 LH cohort: S14, S16, S23, S26, S33, S39, S62. Tests whether
# extending from 4 → 7 patients recovers cross-patient transfer. At 4
# patients, pooled flat (0.791) ≈ per-subject flat mean (0.784) — no
# transfer signal. If 7 patients still don't beat per-subject, the
# bottleneck is architectural, not data.
#
# 15 jobs total (5 folds × 3 seeds).
#
# Task-ID encoding: task_id = fold * 3 + seed
#
#SBATCH --job-name=v14pp_pooled_q1d_flat
#SBATCH --partition=coganlab-gpu
#SBATCH --array=0-14
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/work/ht203/logs/v14core/%x_%A_%a.out
#SBATCH --error=/work/ht203/logs/v14core/%x_%A_%a.err

set -euo pipefail

PATIENTS="S14,S16,S23,S26,S33,S39,S62"
DEPTH=3
SEEDS=(0 1 2)

task_id=${SLURM_ARRAY_TASK_ID}
seed_idx=$(( task_id % 3 ))
fold=$(( task_id / 3 ))
seed=${SEEDS[$seed_idx]}

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
OUT_DIR=/work/ht203/results/v14_pooled/Q1d_7lh_d32_depth${DEPTH}_flat

export LD_LIBRARY_PATH=/work/ht203/miniconda3/envs/speech/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} pooled-7lh-flat patients=${PATIENTS} fold=${fold} seed=${seed}"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode pooled \
    --patients "${PATIENTS}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --backbone-depth "${DEPTH}" \
    --temporal-frontend flat \
    --out-dir "${OUT_DIR}"
