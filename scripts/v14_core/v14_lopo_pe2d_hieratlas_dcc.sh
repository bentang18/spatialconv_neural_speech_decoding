#!/bin/bash
# LOPO pretrain → per-patient finetune for T3.1 composition:
# per_cell + partial_conv masking + factorized_2d pe2d + hierarchical_atlas readout.
#
# Closes the one untested gap in the top-pooled family. Pooled-joint
# 4-core is 0.765 ± 0.042 (v14_ablation_log.csv row 160), tied with T3.4
# (hierarchical) at 0.770 within noise and 0.049 pp better than the
# previous Stage-1 default. T3.4 LOPO = 0.787 ± 0.002 (4-core mean).
# Hypothesis: atlas-anchored query (pooled_support @ P_emb) either matches
# or improves T3.4 LOPO — if it does, T3.1 becomes the Stage-1 default
# over T3.4 by trading a 32-cell free cell_query for an anatomy-indexed
# projection of the parcel embedding.
#
# 60 jobs total (4 held-out × 5 folds × 3 seeds).
#
#SBATCH --job-name=v14pp_lopo_pe2d_hieratlas
#SBATCH --partition=coganlab-gpu
#SBATCH --array=0-59
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/work/ht203/logs/v14core/%x_%A_%a.out
#SBATCH --error=/work/ht203/logs/v14core/%x_%A_%a.err

set -euo pipefail

CORE=(S14 S26 S33 S62)
DEPTH=3
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
PYTHON=/work/ht203/repo/speech/.venv/bin/python
ROOT=/work/ht203/results/v14_lopo/per_cell_partialconv_pe2d_hieratlas_d32_depth${DEPTH}
PRETRAIN_DIR=${ROOT}/pretrain_heldout_${held_out}
FINETUNE_DIR=${ROOT}/finetune_${held_out}

export LD_LIBRARY_PATH=/work/ht203/repo/speech/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}

mkdir -p "${PRETRAIN_DIR}" "${FINETUNE_DIR}"
mkdir -p /work/ht203/logs/v14core

echo "[$(date -Is)] task=${task_id} LOPO held_out=${held_out} pretrain=${PRETRAIN_PATIENTS} fold=${fold} seed=${seed} arm=per_cell+partial_conv+pe2d+hierarchical_atlas"
nvidia-smi || true

cd "${REPO}"

SORTED_SLUG=$(echo "${PRETRAIN_PATIENTS}" | tr ',' '\n' | sort | paste -sd '_' -)
PRETRAIN_TAG=pooled_${SORTED_SLUG}_fold${fold}_seed${seed}_depth${DEPTH}
CKPT=${PRETRAIN_DIR}/${PRETRAIN_TAG}.ckpt.pt

echo "[$(date -Is)] === stage 1: pretrain (per_cell + partial_conv + pe2d + hierarchical_atlas) ==="
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode pooled \
    --patients "${PRETRAIN_PATIENTS}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --backbone-depth "${DEPTH}" \
    --masking-mode partial_conv \
    --spatial-pe-mode factorized_2d \
    --readout-mode hierarchical_atlas \
    --save-checkpoint \
    --out-dir "${PRETRAIN_DIR}"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: expected checkpoint ${CKPT} not found after pretrain" >&2
    ls -la "${PRETRAIN_DIR}" >&2
    exit 1
fi

echo "[$(date -Is)] === stage 2: finetune on ${held_out} (per_cell + partial_conv + pe2d + hierarchical_atlas) ==="
${PYTHON} scripts/v14_core/train_v14_core.py \
    --mode per-phoneme \
    --patient "${held_out}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --backbone-depth "${DEPTH}" \
    --masking-mode partial_conv \
    --spatial-pe-mode factorized_2d \
    --readout-mode hierarchical_atlas \
    --init-from "${CKPT}" \
    --out-dir "${FINETUNE_DIR}"

echo "[$(date -Is)] === done ==="
