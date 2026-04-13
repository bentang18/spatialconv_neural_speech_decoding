#!/bin/bash
#SBATCH --job-name=data_diag
#SBATCH --output=scripts/diagnostic_data_quality.log
#SBATCH --error=scripts/diagnostic_data_quality.log
#SBATCH --partition=coganlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4

export PYTHONUNBUFFERED=1
cd /work/ht203/repo/speech

/work/ht203/miniconda3/envs/speech/bin/python scripts/diagnostic_data_quality.py \
    --bids-root /work/ht203/data/BIDS
