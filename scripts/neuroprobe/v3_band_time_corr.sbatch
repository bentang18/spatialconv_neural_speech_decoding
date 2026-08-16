#!/bin/bash
#SBATCH --job-name=band_time_corr
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=113560M
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
# R42 -- within-contact per-band TIME autocorrelation r(lag) over all 13 V14_PRETRAIN_SESSIONS.
#   sbatch scripts/neuroprobe/v3_band_time_corr.sbatch
#
# The TIME twin of R20 (v3_shaft_band_corr.sbatch). Same sessions, same clips, same _window_bands
# path, so r(lag) here is directly comparable to R20's r(d) across contacts. Zero GPU, but dtai and
# NOT Delta CPU: speech_decoding.models.__init__ eagerly imports the legacy zoo (mlp -> neuraltrain),
# so any speech_decoding.models.* import needs the aarch64 project venv. Delta's pytorch-conda has no
# neuraltrain. One job looping 13 sessions, not a 13-wide array -- holding 13 GPUs to use none is
# worse. No srun: one process, one node, nothing to rank-wire.
set -uo pipefail
OUTDIR="${OUTDIR:-/projects/bhqk/htang13/band_time_corr}"
mkdir -p "$OUTDIR"
TREE=/projects/bhqk/htang13/speech
cd "$TREE"
module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
SESSIONS=("1 0" "2 1" "2 2" "2 3" "2 5" "2 6" "3 2" "4 2" "6 0" "6 1" "6 4" "8 0" "9 0")
fail=0
for S in "${SESSIONS[@]}"; do
  read -r SUBJ TRIAL <<< "$S"
  echo "===== s${SUBJ}t${TRIAL} ====="
  "$TREE/.venv/bin/python" -u -m scripts.neuroprobe.v3_band_time_corr \
    --band-cache-dir $ROOT/band_v3slow \
    --band-cache-dir $ROOT/band_v3mid \
    --band-cache-dir $ROOT/band_hga \
    --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
    --bt-root /projects/bhqk/htang13/braintreebank \
    --subject "$SUBJ" --trial "$TRIAL" \
    --electrode-set pretrain \
    --n-clips 64 --clip-dur 2.0 --max-lag 8 \
    --out "$OUTDIR/band_time_corr_s${SUBJ}t${TRIAL}.json" || { echo "FAILED s${SUBJ}t${TRIAL}"; fail=1; }
done
echo "[done] R42 sweep finished, fail=$fail"
exit $fail
