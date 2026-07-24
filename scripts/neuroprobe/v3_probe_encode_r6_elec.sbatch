#!/bin/bash
#SBATCH --job-name=probe_encode_elec
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:15:00
#SBATCH --output=/projects/bhqk/htang13/probe_encode_elec_%j.out
#
# v3_probe_encode_r4.sbatch + --frontend v3r6 + --elec-taps 12.
# --frontend v3r6 is NOT cosmetic: it implies --online. r6 is MAE, which has no EMA teacher, so
# the r4 default (--frontend v3r4 -> objective.teacher.model.*) finds no keys and dies at load
# (jobs 2718966-68, "wrong ckpt layout"). The DATA path is r4's verbatim either way.
# Same forward, same taps, same caches; enc12 is ADDITIONALLY written unpooled on the canonical
# contact axis as enc12_elec. Needed by:
#   - probe_subspace_surplus.py (#40): the native d=256 token space. Without it the probe falls
#     back to the parcel-POOLED cache, where contact averaging attenuates within-parcel
#     directions, so the measured spectrum is only a lower bound.
#   - probe_pool_variants.py (#41): the unpooled input the readout re-pools itself.
# Extra cost is storage + write time only. MEASURED: the r4 encode runs ~7 min (sacct, 8 jobs,
# 05:47-07:07), so 15 min is a ~2x margin for the added elec write -- keep it SHORT so these
# backfill instead of queueing behind the 48h training arms on Priority.
set -euo pipefail
CKPT="${1:?usage: sbatch v3_probe_encode_r6_elec.sbatch <CKPT> <TAG> <OUTDIR>}"
TAG="${2:?need TAG}"
OUTDIR="${3:?need OUTDIR}"
cd /projects/bhqk/htang13/speech
module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
srun .venv/bin/python -u -m scripts.neuroprobe.v3_probe_encode_r4 \
  --ckpt "$CKPT" --tag "$TAG" --out-dir "$OUTDIR" \
  --band-cache-dir $ROOT/band_v3slow \
  --band-cache-dir $ROOT/band_v3mid \
  --band-cache-dir $ROOT/band_hga \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --batch-size 64 \
  --frontend v3r6 \
  --elec-taps 12
