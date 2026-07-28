#!/bin/bash
#SBATCH --job-name=board_merge
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/bhqk/htang13/board_merge_%j.out
# Merge the 22 board shards -> the leaderboard table. Prints MERGE_DONE for the driver poll.
set -euo pipefail
SHARD="${1:?need SHARD_DIR}"
TAG="${2:?need TAG}"
OUT="${3:?need OUT}"
cd /projects/bhqk/htang13/speech_board
module load pytorch-conda/2.8
python -u scripts/neuroprobe/v3_board_readout.py \
  --cache-dir /dev/null --tags "$TAG" --mode merge --shard-dir "$SHARD" --out "$OUT"
