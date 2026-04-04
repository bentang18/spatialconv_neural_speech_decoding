#!/bin/bash
#SBATCH --job-name=sweep_pad
#SBATCH --partition=coganlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ht203/logs/sweep_pad_%j.out
#SBATCH --error=/work/ht203/logs/sweep_pad_%j.err

# Padding grid sweep: 10 tmin/tmax conditions × 5 folds × ~8s/fold = ~7 min

set -e
cd /work/ht203/repo/speech

PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
export DEVICE=cuda
export PYTHONUNBUFFERED=1

echo "=== Padding grid sweep at $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PYTHON scripts/sweep_padding_grid.py --paths configs/paths.yaml --device cuda
echo "=== Done at $(date) ==="
