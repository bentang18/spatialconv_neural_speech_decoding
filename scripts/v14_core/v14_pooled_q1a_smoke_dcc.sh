#!/bin/bash
# DCC smoke for v14 pooled Q1a: 2 patients × fold 0 × seed 0 × depth 3, --smoke.
#
# Purpose: pipeline validation — verify pooled dataset build, round-robin
# interleave, pooled CV, per-patient test eval all work on real DCC GPU before
# the 15-job full launch.
#
#SBATCH --job-name=v14pp_pooled_smoke
#SBATCH --partition=coganlab-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ht203/logs/v14core/%x_%j.out
#SBATCH --error=/work/ht203/logs/v14core/%x_%j.err

set -euo pipefail

PATIENTS="S14,S26"
DEPTH=3

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
OUT_DIR=/work/ht203/results/v14_pooled/smoke

export LD_LIBRARY_PATH=/work/ht203/miniconda3/envs/speech/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] pooled smoke patients=${PATIENTS} depth=${DEPTH}"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode pooled \
    --patients "${PATIENTS}" \
    --fold 0 \
    --seed 0 \
    --backbone-depth "${DEPTH}" \
    --out-dir "${OUT_DIR}" \
    --smoke
