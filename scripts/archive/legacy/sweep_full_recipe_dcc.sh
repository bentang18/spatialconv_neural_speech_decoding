#!/bin/bash
#SBATCH --job-name=sweep_full
#SBATCH --partition=coganlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/ht203/logs/sweep_full_%j.out
#SBATCH --error=/work/ht203/logs/sweep_full_%j.err

# Full-recipe sweep: per-phoneme vs full-trial with k-NN, TTA, mixup
# 5 conditions (some 3-seed, some 1-seed) × 5 folds × ~20-40s/fold
# Estimated ~45 min on GPU

set -e

cd /work/ht203/repo/speech

PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
export DEVICE=cuda
export PYTHONUNBUFFERED=1

echo "=== Starting full-recipe sweep at $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

$PYTHON scripts/sweep_full_recipe.py \
    --paths configs/paths.yaml \
    --device cuda

echo ""
echo "=== Done at $(date) ==="
