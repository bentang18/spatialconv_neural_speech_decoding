#!/bin/bash
#SBATCH --job-name=board_encode_r6
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bhqk/htang13/board_encode_r6_%j.out
#
# Neuroprobe-Lite leaderboard-parity encode for an r6 ckpt. Runs from the LIVE tree (not the old
# speech_board copy): the live v3_probe_encode_r4.py already carries the board loader as a
# CLI-injected keep_labels_fn (--sessions board --tasks board15 --electrode-set lite, lines
# 41-159), AND it has the v3r6 frontend the old r4-era speech_board tree lacks. Each Slurm job is
# its own process on its own node, so this cannot touch the running pretrains' session_loader.
#
# Cache = v14_3band_v3_spec_lite (the 12 Lite sessions), same band_hop=64 uid as pretrain.
# --elec-taps 0,12 writes enc0_elec (per-electrode |STFT| frontend) + enc12_elec (per-electrode
# depth-12), on top of the always-written parcel-mean enc0/3/6/9/12. Tap 0 does NOT route through
# GPU_TAPS=(3,6,9,12)/the teacher forward -- v3_probe_encode_r4.py's _enc0_elec computes it directly
# off the frontend bands (same canonical-contact axis as enc12_elec, verified by re-pooling to
# parcels and diffing against enc0's parcel-mean). Fixes the e74fb27 regression that dropped the
# enc0 depth-0 floor from board WS/CSession by naming a tap the encoder could not yet produce --
# see project-board-ws-lost-enc0-floor-e74fb27-2026-07-26 memo. Board readout's
# WS_TAPS/CSESSION_TAPS = ELEC_TAPS = ("enc0_elec","enc12_elec") already expected this; no readout
# change needed. CSubject consumes enc0/3/6/12 parcel-mean, unaffected either way.
# --mem=160G is MEASURED, not guessed: job 2756113 (12 sessions, 6 taps) peaked at 122.3 GiB
# MaxRSS on the srun step. The old 300G dated from when _encode_taps cat-ed a list of per-batch
# tensors and so held the buffer AND its concatenation alive at once -- that arm measured 230.2
# GiB for the identical work. e714b47 preallocates instead, which halved the peak. The enc9 tap
# adds ~9% to the payload (parcel taps are 16x13312 vs enc12_elec's 119x13312), so budget ~134
# GiB; 160G is that plus headroom. Sessions are encoded ONE AT A TIME in this single job (not
# sharded -- board encode is a GPU forward pass on dtai; only READOUT shards on Delta CPU, see
# probe_ckpt_r6.sh workflow doc). Walltime 30m: 2756113 ran 19:44, and short jobs earn backfill.
set -euo pipefail
CKPT="${1:?usage: sbatch v3_board_encode_r6.sbatch <CKPT> <TAG> <OUTDIR>}"
TAG="${2:?need TAG}"
OUTDIR="${3:?need OUTDIR}"
cd /projects/bhqk/htang13/speech
module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/projects/bhqk/htang13/speech/src
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_lite
srun /projects/bhqk/htang13/speech/.venv/bin/python -u -m scripts.neuroprobe.v3_probe_encode_r4 \
  --ckpt "$CKPT" --tag "$TAG" --out-dir "$OUTDIR" \
  --band-cache-dir $ROOT/band_v3slow \
  --band-cache-dir $ROOT/band_v3mid \
  --band-cache-dir $ROOT/band_hga \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --batch-size 64 \
  --frontend v3r6 \
  --sessions board --tasks board15 --electrode-set lite --elec-taps 0,12
