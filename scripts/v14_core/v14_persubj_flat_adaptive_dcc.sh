#!/bin/bash
# Per-subject v14 with flat Conv1d front-end AND AdaptiveAvgPool2d.
#
# Runs S14 (control, 8×16 — pools must agree with masked_mean) and
# S62 (12×22 — overlapping-bin rule differs) at d=32, depth=3, k=3,
# pool=(4,8). Isolates the pool-geometry confound from the flat
# front-end win we already saw.
#
# 2 patients × 5 folds × 3 seeds = 30 jobs.
#
# Task-ID encoding:
#     task_id = pt_idx * 15 + fold * 3 + seed
#   pt_idx ∈ {0: S14, 1: S62}
#
#SBATCH --job-name=v14pp_flat_adaptive
#SBATCH --partition=coganlab-gpu
#SBATCH --array=0-29
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/work/ht203/logs/v14core/%x_%A_%a.out
#SBATCH --error=/work/ht203/logs/v14core/%x_%A_%a.err

set -euo pipefail

PATIENTS=(S14 S62)
DEPTH=3
SEEDS=(0 1 2)

task_id=${SLURM_ARRAY_TASK_ID}
pt_idx=$(( task_id / 15 ))
rem=$(( task_id % 15 ))
fold=$(( rem / 3 ))
seed_idx=$(( rem % 3 ))
patient=${PATIENTS[$pt_idx]}
seed=${SEEDS[$seed_idx]}

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/repo/speech/.venv/bin/python
OUT_DIR=/work/ht203/results/v14_per_phoneme/${patient}/flat_adaptive_d32_depth${DEPTH}

export LD_LIBRARY_PATH=/work/ht203/repo/speech/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} patient=${patient} fold=${fold} seed=${seed} frontend=flat pool=adaptive_avg"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode per-phoneme \
    --patient "${patient}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --backbone-depth "${DEPTH}" \
    --temporal-frontend flat \
    --pool-method adaptive_avg \
    --out-dir "${OUT_DIR}"
