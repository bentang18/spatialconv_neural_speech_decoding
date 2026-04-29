#!/bin/bash
# Pooled Q1a (4 core) at d=16 — capacity-floor ablation.
#
# Canonical per_cell + parcel embedding, depth 3, just width dropped from
# 32 to 16. Tests whether d=32 is over-parameterized at Phase-1 scale. At
# d=16 the backbone has a single head × 16 — head-dim stays 16 (standard).
#
# Total params drop from ~47k to ~22k.
#
# 15 jobs total (5 folds × 3 seeds). Task-ID encoding: task_id = fold * 3 + seed
#
#SBATCH --job-name=v14pp_q1a_d16
#SBATCH --partition=coganlab-gpu
#SBATCH --array=0-14
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
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
PYTHON=/work/ht203/repo/speech/.venv/bin/python
OUT_DIR=/work/ht203/results/v14_pooled/Q1a_4core_d16_depth${DEPTH}

export LD_LIBRARY_PATH=/work/ht203/repo/speech/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${OUT_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} pooled-d16 patients=${PATIENTS} fold=${fold} seed=${seed}"
nvidia-smi || true

cd "${REPO}"
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode pooled \
    --patients "${PATIENTS}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --backbone-depth "${DEPTH}" \
    --d-model 16 \
    --out-dir "${OUT_DIR}"
