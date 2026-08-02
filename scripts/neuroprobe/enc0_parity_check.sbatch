#!/bin/bash
#SBATCH --job-name=enc0_parity
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bhqk/htang13/enc0_parity_check_%j.out
#
# The gate: does the --band-rates code path still reproduce the PUBLISHED enc0?
#
# Delta CPU, not DeltaAI: this only needs torch + numpy to load two record dicts and subtract
# them, and it imports nothing from speech_decoding -- so the missing project stack that forced
# the ENCODE onto DeltaAI is irrelevant here, and a GPU share would be wasted on tensor loads.
#
# REFERENCE IS THE SLIM fe_abl CACHE, not the full board record it came from. Pointing at
# v3_board_cache_board_pbs50_cd45k OOM-killed job 20807953 at 24G: that record is 44-71 GB and
# ~99% encoder taps, and an eager load faults all of it in to read 0.8 GB of enc0. The slim copy
# is a valid reference because fe_abl_slim_cache.py asserts the enc0/enc0_elec copy is BIT-EXACT
# (fe_abl_slim_cache.py:79-81) and carries the layout metadata verbatim -- so it is the same
# tensors, and enc0 is weightless anyway, hence identical across ckpt-tagged records.
#
# 48G with mmap on both sides, matching fe_abl_slim.sbatch's sizing and for its reason: the read
# is mmap'd so only the enc0 regions fault in, but mmap page cache still counts against the
# cgroup, and an under-request that OOMs BILLS IN FULL. 8 cores is under 0.512*48, i.e. free.
set -euo pipefail
TREE=/projects/bhqk/htang13/speech_enc0
cd "$TREE"
module load pytorch-conda/2.8
export PYTHONPATH="$TREE:$TREE/src"

numactl --interleave=all python -u scripts/neuroprobe/enc0_parity_check.py \
  --published-dir /projects/bhqk/htang13/v3_board_cache_fe_abl \
  --published-tag feabl \
  --new-dir /projects/bhqk/htang13/enc0_parity \
  --new-tag enc0_parity
