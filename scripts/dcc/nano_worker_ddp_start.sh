#!/bin/bash
# Start the warm **4-GPU DDP** nano worker as a persistent sbatch job holding 4
# GPUs on coganlab-gpu. Submit once at the start of a 4-GPU iteration session:
#
#   SPEECH_REPO=/work/ht203/repo/speech_bt sbatch scripts/dcc/nano_worker_ddp_start.sh
#
# then fire runs with ``nano_ddp_submit``. Stop with ``scancel <jobid>`` or
# ``touch $NANO_DDP_WORKER_DIR/queue/__stop__``.
#
# Resources mirror the verified cold 4-GPU run (job 47920242: 64 CPU / 128G /
# gpu:4 / 1 node) so host RAM never OOMs the dataset prepare spike.
#SBATCH --job-name=nano_ddp
#SBATCH --partition=coganlab-gpu
#SBATCH --account=coganlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/work/ht203/nano_ddp_worker_%j.log
set -euo pipefail

cd "${SPEECH_REPO:-/work/ht203/repo/speech_bt}"

export NANO_DDP_WORKER_DIR="${NANO_DDP_WORKER_DIR:-$HOME/.nano_worker_ddp}"
export WANDB_API_KEY="$(cat /hpc/home/ht203/.wandb_key)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank
export EXCA_CACHE_FOLDER=/hpc/group/coganlab/ht203/cache_neuroai
export EXCA_EXTRACTOR_CACHE_FOLDER=/work/ht203/cache/v14_extractors

echo "nano_worker_ddp_start: repo=$(pwd) worker_dir=$NANO_DDP_WORKER_DIR ntasks=$SLURM_NTASKS"
# srun inherits --ntasks-per-node=4 → 4 ranks, one per GPU. Each imports once and
# runs dispatch_v14.main(argv) together under Lightning DDP (exca local).
srun .venv/bin/python -u scripts/dcc/nano_worker_ddp.py
