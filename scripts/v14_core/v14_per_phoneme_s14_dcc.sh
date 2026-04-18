#!/bin/bash
# DCC SLURM array for the v14 per-phoneme S14 sweep (plan P8).
#
# 1 patient × 5 folds × 3 seeds × 2 backbone depths = 30 jobs.
# Task-ID encoding (base-first, depth-last):
#     task_id = (fold * 3 + seed_idx) * 2 + depth_idx
#
#SBATCH --job-name=v14pp_s14
#SBATCH --partition=coganlab-gpu
#SBATCH --array=0-29
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/work/ht203/logs/v14core/%x_%A_%a.out
#SBATCH --error=/work/ht203/logs/v14core/%x_%A_%a.err

set -euo pipefail

FOLDS=(0 1 2 3 4)
SEEDS=(0 1 2)
DEPTHS=(1 3)  # ablation ladder rung 1 per plan P8

task_id=${SLURM_ARRAY_TASK_ID}
depth_idx=$(( task_id % 2 ))
seed_idx=$(( (task_id / 2) % 3 ))
fold=$(( task_id / 6 ))

patient=S14
seed=${SEEDS[$seed_idx]}
depth=${DEPTHS[$depth_idx]}

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
OUT_DIR=/work/ht203/results/v14_per_phoneme/${patient}/depth${depth}

export LD_LIBRARY_PATH=/work/ht203/miniconda3/envs/speech/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} patient=${patient} fold=${fold} seed=${seed} depth=${depth}"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode per-phoneme \
    --patient "${patient}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --backbone-depth "${depth}" \
    --out-dir "${OUT_DIR}"
