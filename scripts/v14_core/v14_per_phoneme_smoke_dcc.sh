#!/bin/bash
# Smoke-test ONE per-phoneme job (S14 fold=0 seed=0 depth=3) before launching
# the full per-phoneme array. Validates the baseline-aligned pipeline on DCC.
#
#SBATCH --job-name=v14pp_smoke
#SBATCH --partition=coganlab-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ht203/logs/v14core/pp_smoke_%A.out
#SBATCH --error=/work/ht203/logs/v14core/pp_smoke_%A.err

set -euo pipefail

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
OUT_DIR=/work/ht203/results/v14core/per_phoneme_smoke

export LD_LIBRARY_PATH=/work/ht203/miniconda3/envs/speech/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] per-phoneme smoke start"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode per-phoneme \
    --patient S14 \
    --fold 0 \
    --seed 0 \
    --backbone-depth 3 \
    --out-dir "${OUT_DIR}" \
    --smoke

echo "[$(date -Is)] per-phoneme smoke done"
