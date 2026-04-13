#!/bin/bash
#SBATCH --partition=coganlab-gpu
#SBATCH --account=coganlab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/ht203/logs/exp_%x_%j.out
#SBATCH --error=/work/ht203/logs/exp_%x_%j.err

# Usage: sbatch --job-name=exp18 run_experiment.sh exp18_domain_adversarial.py

source /work/ht203/miniconda3/etc/profile.d/conda.sh
conda activate /work/ht203/miniconda3/envs/speech
cd /work/ht203/repo/speech
export DEVICE=cuda

SCRIPT="${1:-train.py}"
echo "=== Running experiment: $SCRIPT ==="
echo "=== Start time: $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

python "scripts/autoresearch_lopo/experiments/$SCRIPT" 2>&1

echo "=== End time: $(date) ==="
echo "=== Extracting results ==="
grep "^val_per:" /dev/stdin 2>/dev/null || true
