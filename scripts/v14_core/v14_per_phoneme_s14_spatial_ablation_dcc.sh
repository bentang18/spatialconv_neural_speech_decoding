#!/bin/bash
# DCC SLURM array for the S14 spatial-front-end ablation over P8 baseline
# (d=32, depth=3, k=3, pool=(4,8)).
#
# 4 configs × 5 folds × 3 seeds = 60 jobs. Baseline cell is NOT re-run here.
#
# Configs (cfg_idx):
#   0 → k=1, pool=(4,8)    (no spatial context, baseline pool)
#   1 → k=5, pool=(4,8)    (larger receptive field, baseline pool)
#   2 → k=3, pool=(2,4)    (baseline kernel, coarser pool — 8 cells)
#   3 → k=3, pool=(8,16)   (baseline kernel, no pool — 128 cells on 8×16)
#
# Task-ID encoding (base-first, cfg-last):
#     task_id = (fold * 3 + seed_idx) * 4 + cfg_idx
#
#SBATCH --job-name=v14pp_s14_spatial
#SBATCH --partition=coganlab-gpu
#SBATCH --array=0-59
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/work/ht203/logs/v14core/%x_%A_%a.out
#SBATCH --error=/work/ht203/logs/v14core/%x_%A_%a.err

set -euo pipefail

FOLDS=(0 1 2 3 4)
SEEDS=(0 1 2)
CFG_KERNEL=(1 5 3 3)
CFG_POOL_H=(4 4 2 8)
CFG_POOL_W=(8 8 4 16)
CFG_TAG=(k1_pool4x8 k5_pool4x8 k3_pool2x4 k3_pool8x16)

task_id=${SLURM_ARRAY_TASK_ID}
cfg_idx=$(( task_id % 4 ))
seed_idx=$(( (task_id / 4) % 3 ))
fold=$(( task_id / 12 ))

seed=${SEEDS[$seed_idx]}
kernel=${CFG_KERNEL[$cfg_idx]}
pool_h=${CFG_POOL_H[$cfg_idx]}
pool_w=${CFG_POOL_W[$cfg_idx]}
tag=${CFG_TAG[$cfg_idx]}
patient=S14

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
OUT_DIR=/work/ht203/results/v14_per_phoneme/${patient}/${tag}

export LD_LIBRARY_PATH=/work/ht203/miniconda3/envs/speech/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} patient=${patient} fold=${fold} seed=${seed} k=${kernel} pool=${pool_h}x${pool_w}"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode per-phoneme \
    --patient "${patient}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --backbone-depth 3 \
    --d-model 32 \
    --conv2d-kernel "${kernel}" \
    --pool-h "${pool_h}" \
    --pool-w "${pool_w}" \
    --out-dir "${OUT_DIR}"
