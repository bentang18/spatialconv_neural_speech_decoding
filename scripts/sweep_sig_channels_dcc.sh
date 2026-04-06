#!/bin/bash
#SBATCH --job-name=sig_ch
#SBATCH --output=scripts/sweep_sig_channels.log
#SBATCH --error=scripts/sweep_sig_channels.log
#SBATCH --partition=coganlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4

export PYTHONUNBUFFERED=1
cd /work/ht203/repo/speech

/work/ht203/miniconda3/envs/speech/bin/python scripts/sweep_sig_channels.py \
    --bids-root /work/ht203/data/BIDS \
    --target S14 \
    --seeds 3
