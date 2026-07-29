#!/bin/bash
#SBATCH --job-name=shaft_band_corr
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%A_%a.out
# R20 -- along-shaft per-band correlation r(d), one array task per V14_PRETRAIN_SESSIONS entry.
#   sbatch --array=0-12 v3_shaft_band_corr.sbatch
# No model, no GPU: this reads band caches and correlates contacts. Delta CPU, never dtai GPU.
# Delta CPU bills on MEMORY (MAX_TRES == mem_MB/2), so cores below 0.512*mem_GB = 16.4 are FREE
# at 32G -- 16 cores costs exactly what 1 would. 20 min walltime to earn backfill.
set -euo pipefail
SESSIONS=("1 0" "2 1" "2 2" "2 3" "2 5" "2 6" "3 2" "4 2" "6 0" "6 1" "6 4" "8 0" "9 0")
IDX="${SLURM_ARRAY_TASK_ID:?needs --array=0-12}"
read -r SUBJ TRIAL <<< "${SESSIONS[$IDX]}"
OUTDIR="${OUTDIR:-/projects/bhqk/htang13/shaft_band_corr}"
mkdir -p "$OUTDIR"
cd /projects/bhqk/htang13/speech
module load pytorch-conda/2.8
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# -m puts CWD on sys.path (so `scripts.neuroprobe` resolves) but NOT src/, and the conda module's
# python has no editable install -- without this, importing speech_decoding dies at module load.
export PYTHONPATH=/projects/bhqk/htang13/speech/src:${PYTHONPATH:-}
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_lite
numactl --interleave=all python -u -m scripts.neuroprobe.v3_shaft_band_corr \
  --band-cache-dir $ROOT/band_v3slow \
  --band-cache-dir $ROOT/band_v3mid \
  --band-cache-dir $ROOT/band_hga \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --subject "$SUBJ" --trial "$TRIAL" \
  --electrode-set pretrain \
  --n-clips 64 --clip-dur 2.0 --max-offset 6 \
  --out "$OUTDIR/shaft_band_corr_s${SUBJ}t${TRIAL}.json"
