#!/bin/bash
# Parameterized LOPO pretrain → per-patient finetune sbatch for ablations.
#
# Invoke via sbatch --export=ALL,ABLATION_NAME=<tag>,EXTRA_FLAGS="<flags>" ...
# ABLATION_NAME is used for the results ROOT and job-name suffix.
# EXTRA_FLAGS is passed through to BOTH pretrain and finetune stages so the
# arch is identical across the two-stage run.
#
# 60 jobs: 4 held-out × 5 folds × 3 seeds.
# Task-ID: task_id = pt_idx * 15 + fold * 3 + seed
#   pt_idx ∈ {0: S14, 1: S26, 2: S33, 3: S62}
#
#SBATCH --job-name=v14pp_lopo_abl
#SBATCH --partition=coganlab-gpu
#SBATCH --array=0-59
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/work/ht203/logs/v14core/%x_%A_%a.out
#SBATCH --error=/work/ht203/logs/v14core/%x_%A_%a.err

set -euo pipefail

: "${ABLATION_NAME:?ABLATION_NAME must be set (sbatch --export)}"
: "${EXTRA_FLAGS:=}"

CORE=(S14 S26 S33 S62)
SEEDS=(0 1 2)

task_id=${SLURM_ARRAY_TASK_ID}
pt_idx=$(( task_id / 15 ))
rem=$(( task_id % 15 ))
fold=$(( rem / 3 ))
seed_idx=$(( rem % 3 ))
held_out=${CORE[$pt_idx]}
seed=${SEEDS[$seed_idx]}

PRETRAIN_PATIENTS=""
for p in "${CORE[@]}"; do
    if [[ "$p" != "$held_out" ]]; then
        PRETRAIN_PATIENTS+="${PRETRAIN_PATIENTS:+,}${p}"
    fi
done

REPO=/work/ht203/repo/speech
PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
ROOT=/work/ht203/results/v14_lopo/${ABLATION_NAME}
PRETRAIN_DIR=${ROOT}/pretrain_heldout_${held_out}
FINETUNE_DIR=${ROOT}/finetune_${held_out}

export LD_LIBRARY_PATH=/work/ht203/miniconda3/envs/speech/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${PRETRAIN_DIR}" "${FINETUNE_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} LOPO held_out=${held_out} pretrain=${PRETRAIN_PATIENTS} fold=${fold} seed=${seed} ablation=${ABLATION_NAME}"
echo "[$(date -Is)] extra_flags: ${EXTRA_FLAGS}"
nvidia-smi || true

cd "${REPO}"

echo "[$(date -Is)] === stage 1: pretrain on ${PRETRAIN_PATIENTS} ==="
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode pooled \
    --patients "${PRETRAIN_PATIENTS}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --save-checkpoint \
    --out-dir "${PRETRAIN_DIR}" \
    ${EXTRA_FLAGS}

CKPT=$(ls -1t ${PRETRAIN_DIR}/pooled_*_fold${fold}_seed${seed}_*.ckpt.pt 2>/dev/null | head -1)
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
    echo "ERROR: no checkpoint matching pooled_*_fold${fold}_seed${seed}_*.ckpt.pt in ${PRETRAIN_DIR}" >&2
    ls -la "${PRETRAIN_DIR}" >&2
    exit 1
fi
echo "[$(date -Is)] using checkpoint: ${CKPT}"

echo "[$(date -Is)] === stage 2: finetune on ${held_out} ==="
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode per-phoneme \
    --patient "${held_out}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --init-from "${CKPT}" \
    --out-dir "${FINETUNE_DIR}" \
    ${EXTRA_FLAGS}

echo "[$(date -Is)] === done ==="
