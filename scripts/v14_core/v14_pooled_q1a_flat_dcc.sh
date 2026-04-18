#!/bin/bash
# Pooled Q1a (4 core) with the baseline flat Conv1d front-end.
#
# Identical to Q1a but --temporal-frontend flat — swaps per-cell
# Conv1d(8→32) for Conv1d(256→32, k=30, s=10). Parcel embedding is
# auto-disabled in flat mode (no cell dim). The single best recipe
# we've tried in one experiment: baseline spatial-mixing prior +
# cross-patient data amplification + round-robin interleave.
#
# 15 jobs total (5 folds × 3 seeds).
#
# Task-ID encoding: task_id = fold * 3 + seed
#
#SBATCH --job-name=v14pp_pooled_q1a_flat
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
PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
OUT_DIR=/work/ht203/results/v14_pooled/Q1a_4core_d32_depth${DEPTH}_flat

export LD_LIBRARY_PATH=/work/ht203/miniconda3/envs/speech/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} pooled-flat patients=${PATIENTS} fold=${fold} seed=${seed}"
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
