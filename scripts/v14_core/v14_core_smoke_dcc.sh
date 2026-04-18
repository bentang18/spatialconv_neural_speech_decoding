#!/bin/bash
# Smoke-test ONE job (S14 fold=0 seed=0 depth=1, 5 epochs) before launching
# the full 120-job array. Validates GPU + data loading + training + eval.
#
#SBATCH --job-name=v14smoke
#SBATCH --partition=coganlab-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ht203/logs/v14core/smoke_%A.out
#SBATCH --error=/work/ht203/logs/v14core/smoke_%A.err

set -euo pipefail

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
OUT_DIR=/work/ht203/results/v14core/smoke

export LD_LIBRARY_PATH=/work/ht203/miniconda3/envs/speech/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] smoke start"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --patient S14 \
    --fold 0 \
    --seed 0 \
    --grid-mixer-depth 1 \
    --out-dir "${OUT_DIR}" \
    --smoke

echo "[$(date -Is)] smoke done"
