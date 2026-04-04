#!/bin/bash
#SBATCH --job-name=sweep_mpt
#SBATCH --partition=coganlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/work/ht203/logs/sweep_mpt_%j.out
#SBATCH --error=/work/ht203/logs/sweep_mpt_%j.err

# Multi-patient sweep: per-phoneme vs full-trial on all 11 PS patients
# 11 patients × 2 conditions × 5 folds × ~8s/fold = ~15 min

set -e
cd /work/ht203/repo/speech

PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
export DEVICE=cuda
export PYTHONUNBUFFERED=1

echo "=== Multi-patient sweep at $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PYTHON scripts/sweep_multipatient.py --paths configs/paths.yaml --device cuda
echo "=== Done at $(date) ==="
