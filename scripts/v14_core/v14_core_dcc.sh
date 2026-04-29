#!/bin/bash
# DCC SLURM array for v14-core First-Run Protocol (Task E1).
#
# 4 patients × 5 folds × 3 seeds × 2 grid-mixer depths = 120 jobs.
# Task-ID encoding (base-first, depth-last):
#     task_id = ((p_idx * 5 + fold) * 3 + seed_idx) * 2 + depth_idx
#
# Decoded back to CLI args below. One GPU, 32 GB RAM, 8 h wall per job.
#
#SBATCH --job-name=v14core
#SBATCH --partition=coganlab-gpu
#SBATCH --array=0-119
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/work/ht203/logs/v14core/%x_%A_%a.out
#SBATCH --error=/work/ht203/logs/v14core/%x_%A_%a.err

set -euo pipefail

PATIENTS=(S14 S26 S33 S62)
FOLDS=(0 1 2 3 4)
SEEDS=(0 1 2)
DEPTHS=(1 2)

task_id=${SLURM_ARRAY_TASK_ID}
depth_idx=$(( task_id % 2 ))
seed_idx=$(( (task_id / 2) % 3 ))
fold=$(( (task_id / 6) % 5 ))
p_idx=$(( task_id / 30 ))

patient=${PATIENTS[$p_idx]}
seed=${SEEDS[$seed_idx]}
depth=${DEPTHS[$depth_idx]}

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/repo/speech/.venv/bin/python
OUT_DIR=/work/ht203/results/v14core/${patient}/depth${depth}

export LD_LIBRARY_PATH=/work/ht203/repo/speech/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} patient=${patient} fold=${fold} seed=${seed} depth=${depth}"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --patient "${patient}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --grid-mixer-depth "${depth}" \
    --out-dir "${OUT_DIR}"
