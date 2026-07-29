#!/bin/bash
#SBATCH --job-name=shaft_band_corr
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=113560M
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
# R20 -- along-shaft per-band correlation r(d) over all 13 V14_PRETRAIN_SESSIONS.
#   sbatch scripts/neuroprobe/v3_shaft_band_corr.sbatch
#
# WHY dtai AND NOT DELTA CPU, against the standing "CPU-only probe -> Delta CPU" rule. This job
# uses ZERO GPU -- it correlates cached |STFT| bins. It ran on Delta first (jobs 20565238,
# 20565281) and died at import both times: speech_decoding/models/__init__.py eagerly imports the
# legacy model zoo (mlp -> neuraltrain), so ANY `speech_decoding.models.*` import needs the project
# venv, and the venv is aarch64 = dtai-only. Delta's pytorch-conda module has no neuraltrain. The
# probe/readout scripts that DO run on Delta import no speech_decoding at all, which is exactly why
# they work. Fix is one job, not a 13-wide array: 13 sequential in-process runs on ONE node beats
# holding 13 GPUs to use none of them. ~2 min/session => well inside the 1 h walltime, which also
# earns backfill. mem at DefMemPerGPU because ghx4 bills MAX_TRES.
#
# No srun: one process, one node, nothing to rank-wire (see v3_mae_recon.sbatch -- srun died with
# "Error configuring interconnect" on two nodes before Python started).
set -uo pipefail
OUTDIR="${OUTDIR:-/projects/bhqk/htang13/shaft_band_corr}"
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
  "$TREE/.venv/bin/python" -u -m scripts.neuroprobe.v3_shaft_band_corr \
    --band-cache-dir $ROOT/band_v3slow \
    --band-cache-dir $ROOT/band_v3mid \
    --band-cache-dir $ROOT/band_hga \
    --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
    --bt-root /projects/bhqk/htang13/braintreebank \
    --subject "$SUBJ" --trial "$TRIAL" \
    --electrode-set pretrain \
    --n-clips 64 --clip-dur 2.0 --max-offset 6 \
    --out "$OUTDIR/shaft_band_corr_s${SUBJ}t${TRIAL}.json" || { echo "FAILED s${SUBJ}t${TRIAL}"; fail=1; }
done
echo "[done] R20 sweep finished, fail=$fail"
exit $fail
