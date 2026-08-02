#!/bin/bash
#SBATCH --job-name=enc0_hga64
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=113560M
#SBATCH --time=01:30:00
#SBATCH --output=/projects/bhqk/htang13/enc0_hga64_dtai_%j.out
#
# The 64 Hz HGA enc0 arm -- on DeltaAI, not Delta CPU.
#
# enc0 needs no GPU (--enc0-only: no ckpt, no model), so Delta CPU was the right instinct and the
# wrong environment: its pytorch-conda module carries none of the project stack. The encode's
# import closure reaches mne AND neuralset; the readout reaches neither, which is why the readout
# runs there happily and the encode cannot. Chasing that chain would mean rebuilding an
# environment that already exists here.
#
# NOT AN ARRAY, deliberately. On Delta CPU sharding is free (cores under 0.512*mem_GB cost
# nothing), so the standing rule is to shard everything. Here the billing unit is a GPU share, so
# a 12-cell array would spend 12 GPU-shares on work that never launches a kernel. One job walks
# all 12 sessions instead (--session-index omitted = all).
#
# Code comes from speech_enc0, deps from the LIVE venv. /projects is shared with Delta CPU, so the
# tree deployed there is already here. PYTHONPATH puts speech_enc0 ahead of the venv, and
# `import speech_decoding` was VERIFIED to resolve into speech_enc0 -- so this cannot pick up the
# live training tree's copy, and it writes nothing into it.
#
# usage: sbatch enc0_hga64_dtai.sbatch <parity|fine> <OUTDIR>
set -euo pipefail
ARM="${1:?usage: sbatch enc0_hga64_dtai.sbatch <parity|fine> <OUTDIR>}"
OUTDIR="${2:?need OUTDIR}"
TREE=/projects/bhqk/htang13/speech_enc0
VENV=/projects/bhqk/htang13/speech/.venv/bin/python
cd "$TREE"
module load python/miniforge3_pytorch/2.10.0
export PYTHONPATH="$TREE:$TREE/src"
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai

LITE=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_lite
FINE=/work/nvme/bhqk/htang13/cache_neuroai/v14_finehga_spec_eval

case "$ARM" in
  parity) HGA_DIR="$LITE/band_hga";   RATES=() ;;
  fine)   HGA_DIR="$FINE/band_v3hga"; RATES=(--band-rates 1/1,1/1,2/1) ;;
  *) echo "unknown arm '$ARM' (want parity|fine)" >&2; exit 2 ;;
esac

echo "[check] arm=$ARM hga_cache=$HGA_DIR rates=${RATES[*]:-uniform}"
"$VENV" -c 'import speech_decoding as s; print("[check] speech_decoding ->", s.__file__)'

srun "$VENV" -u -m scripts.neuroprobe.v3_probe_encode_r4 \
  --tag "enc0_$ARM" --out-dir "$OUTDIR" \
  --band-cache-dir "$LITE/band_v3slow" \
  --band-cache-dir "$LITE/band_v3mid" \
  --band-cache-dir "$HGA_DIR" \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --frontend v3r6 \
  --enc0-only --elec-taps 0 \
  --sessions board --tasks board15 --electrode-set lite \
  "${RATES[@]}"
