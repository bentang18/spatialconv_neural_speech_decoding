#!/bin/bash
# DCC SLURM array for the S14 (width, depth) ablation over P8's (d=32, depth=3).
#
# 3 configs × 5 folds × 3 seeds = 45 jobs. Baseline cell (d=32, depth=3) is
# already in hand from P8 and is NOT re-run here.
#
# Configs (cfg_idx):
#   0 → d_model=32, depth=5  (depth-only ablation)
#   1 → d_model=64, depth=3  (width-only ablation)
#   2 → d_model=64, depth=5  (combined)
#
# Task-ID encoding (base-first, cfg-last):
#     task_id = (fold * 3 + seed_idx) * 3 + cfg_idx
#
#SBATCH --job-name=v14pp_s14_ablation
#SBATCH --partition=coganlab-gpu
#SBATCH --array=0-44
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/work/ht203/logs/v14core/%x_%A_%a.out
#SBATCH --error=/work/ht203/logs/v14core/%x_%A_%a.err

set -euo pipefail

FOLDS=(0 1 2 3 4)
SEEDS=(0 1 2)
CFG_DMODEL=(32 64 64)
CFG_DEPTH=(5 3 5)

task_id=${SLURM_ARRAY_TASK_ID}
cfg_idx=$(( task_id % 3 ))
seed_idx=$(( (task_id / 3) % 3 ))
fold=$(( task_id / 9 ))

seed=${SEEDS[$seed_idx]}
d_model=${CFG_DMODEL[$cfg_idx]}
depth=${CFG_DEPTH[$cfg_idx]}
patient=S14

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/repo/speech/.venv/bin/python
OUT_DIR=/work/ht203/results/v14_per_phoneme/${patient}/d${d_model}_depth${depth}

export LD_LIBRARY_PATH=/work/ht203/repo/speech/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} patient=${patient} fold=${fold} seed=${seed} d=${d_model} depth=${depth}"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode per-phoneme \
    --patient "${patient}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --backbone-depth "${depth}" \
    --d-model "${d_model}" \
    --out-dir "${OUT_DIR}"
