#!/bin/bash
# DCC SLURM array: per-subject v14 with the baseline FLAT Conv1d front-end.
#
# Runs on S14 (control), S26, S33, S62 at d=32, depth=3, k=3, pool=(4,8).
# Swaps per-cell Conv1d(8→32) for flat Conv1d(256→32, k=30, s=10) on the
# flattened (B, 8·n_cells, T) tensor — the baseline 2026-04-04 front-end.
#
# Purpose: test hypothesis (1) from the baseline-gap diagnosis — that the
# 32× temporal-front-end capacity collapse is what costs S26/S33/S62.
# Predicted: meaningful drop on S26/S33, modest on S62, S14 ~unchanged.
#
# 4 patients × 5 folds × 3 seeds = 60 jobs.
#
# Task-ID encoding:
#     task_id = patient_idx * 15 + fold * 3 + seed
#   patient_idx ∈ {0:S14, 1:S26, 2:S33, 3:S62}
#
#SBATCH --job-name=v14pp_flatconv
#SBATCH --partition=coganlab-gpu
#SBATCH --array=0-59
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/work/ht203/logs/v14core/%x_%A_%a.out
#SBATCH --error=/work/ht203/logs/v14core/%x_%A_%a.err

set -euo pipefail

PATIENTS=(S14 S26 S33 S62)
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
PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
OUT_DIR=/work/ht203/results/v14_per_phoneme/${patient}/flatconv_d32_depth${DEPTH}

export LD_LIBRARY_PATH=/work/ht203/miniconda3/envs/speech/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} patient=${patient} fold=${fold} seed=${seed} depth=${DEPTH} frontend=flat"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode per-phoneme \
    --patient "${patient}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --backbone-depth "${DEPTH}" \
    --temporal-frontend flat \
    --out-dir "${OUT_DIR}"
