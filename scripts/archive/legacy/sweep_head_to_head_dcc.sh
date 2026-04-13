#!/bin/bash
#SBATCH --job-name=sweep_h2h
#SBATCH --partition=coganlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/ht203/logs/sweep_h2h_%j.out
#SBATCH --error=/work/ht203/logs/sweep_h2h_%j.err

# Head-to-head: learned attention vs per-phoneme MFA, fair comparison
# 5 conditions × 3 seeds × 5 folds × ~10-15s/fold = ~30-45 min

set -e
cd /work/ht203/repo/speech

PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
export DEVICE=cuda
export PYTHONUNBUFFERED=1

echo "=== Head-to-head sweep at $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PYTHON scripts/sweep_head_to_head.py --paths configs/paths.yaml --device cuda
echo "=== Done at $(date) ==="
