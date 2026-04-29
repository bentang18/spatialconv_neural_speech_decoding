#!/bin/bash
# LOPO smoke: held-out S26, fold 0, seed 0.
# One job that exercises the full pretrain → finetune pipeline end-to-end
# before launching the 60-job array.
#
#SBATCH --job-name=v14pp_lopo_smoke
#SBATCH --partition=coganlab-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/work/ht203/logs/v14core/%x_%j.out
#SBATCH --error=/work/ht203/logs/v14core/%x_%j.err

set -euo pipefail

HELD_OUT=S26
PRETRAIN_PATIENTS="S14,S33,S62"
DEPTH=3
FOLD=0
SEED=0

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/repo/speech/.venv/bin/python
ROOT=/work/ht203/results/v14_lopo/smoke_flat_d32_depth${DEPTH}
PRETRAIN_DIR=${ROOT}/pretrain_heldout_${HELD_OUT}
FINETUNE_DIR=${ROOT}/finetune_${HELD_OUT}

export LD_LIBRARY_PATH=/work/ht203/repo/speech/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${PRETRAIN_DIR}" "${FINETUNE_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] SMOKE held_out=${HELD_OUT} pretrain=${PRETRAIN_PATIENTS} fold=${FOLD} seed=${SEED}"
nvidia-smi || true

cd "${REPO}"

SORTED_SLUG=$(echo "${PRETRAIN_PATIENTS}" | tr ',' '\n' | sort | paste -sd '_' -)
PRETRAIN_TAG=pooled_${SORTED_SLUG}_fold${FOLD}_seed${SEED}_depth${DEPTH}
CKPT=${PRETRAIN_DIR}/${PRETRAIN_TAG}.ckpt.pt

echo "[$(date -Is)] === stage 1: pretrain ==="
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode pooled \
    --patients "${PRETRAIN_PATIENTS}" \
    --fold "${FOLD}" \
    --seed "${SEED}" \
    --backbone-depth "${DEPTH}" \
    --temporal-frontend flat \
    --save-checkpoint \
    --out-dir "${PRETRAIN_DIR}"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: expected checkpoint ${CKPT} not found after pretrain" >&2
    ls -la "${PRETRAIN_DIR}" >&2
    exit 1
fi

echo "[$(date -Is)] === stage 2: finetune on ${HELD_OUT} ==="
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode per-phoneme \
    --patient "${HELD_OUT}" \
    --fold "${FOLD}" \
    --seed "${SEED}" \
    --backbone-depth "${DEPTH}" \
    --temporal-frontend flat \
    --init-from "${CKPT}" \
    --out-dir "${FINETUNE_DIR}"

echo "[$(date -Is)] === done ==="
