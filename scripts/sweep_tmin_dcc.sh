#!/bin/bash
#SBATCH --job-name=sweep_tmin
#SBATCH --partition=coganlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ht203/logs/sweep_tmin_%j.out
#SBATCH --error=/work/ht203/logs/sweep_tmin_%j.err

# Quick sweep: tmin × head_type × per-phoneme on S14, per-patient
# 6 conditions × 5 folds × ~40s/fold = ~20 min total on GPU

set -e

cd /work/ht203/repo/speech

PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
export DEVICE=cuda
export PYTHONUNBUFFERED=1

echo "=== Starting sweep at $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

$PYTHON scripts/sweep_tmin_perpos.py \
    --paths configs/paths.yaml \
    --device cuda

echo ""
echo "=== Done at $(date) ==="
