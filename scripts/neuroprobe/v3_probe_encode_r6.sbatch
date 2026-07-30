#!/bin/bash
#SBATCH --job-name=probe_encode_r6
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=113560M
#SBATCH --time=00:15:00
#SBATCH --output=/projects/bhqk/htang13/probe_encode_r6_%j.out
#
# PLAIN r6 pretrain-probe encode = v3_probe_encode_r4 + --frontend v3r6, NO --elec-taps.
# This is the standard grid encode: writes only the parcel-pooled enc0/3/6/12 taps that the
# 13-shard v3_probe_readout_r4 reads. It deliberately omits the unpooled enc12_elec tap (that
# is the #40/#41 add-on and costs ~200 GB/encode); the plain cache is a few GB/session.
# --frontend v3r6 implies --online (r6 is MAE, no EMA teacher; the r4 default looks for
# objective.teacher.model.* and dies at load). DATA path is r4's verbatim. --mem 113560M =
# one GPU's DefMemPerGPU so MAX_TRES billing stays at 1000. 15 min = ~2x the measured ~7 min.
#
# 4th arg = EXTRA encode flags, for ablation arms whose setting CANNOT be read off the ckpt.
# Only ONE flag is in that class: --no-space-rope. deep_sup and parcel_embed ARE inferred from
# ckpt keys (v3_probe_encode_r4.py:136/141) and a wrong inference fails the strict load, but
# L1RoPE registers idx_freq persistent=False, so a --no-space-rope ckpt is key- AND
# value-identical to a normal one (:142-147). Omit the flag and the encode rotates by contact
# index the trained model never saw, the state_dict check PASSES, and the number is silently
# wrong. Hence the guard below: the flag is cross-checked against the ckpt NAME in both
# directions, so a convention miss is a hard failure instead of a plausible AUROC.
set -euo pipefail
CKPT="${1:?usage: sbatch v3_probe_encode_r6.sbatch <CKPT> <TAG> <OUTDIR> [EXTRA_FLAGS]}"
TAG="${2:?need TAG}"
OUTDIR="${3:?need OUTDIR}"
EXTRA="${4:-}"

case "$CKPT" in *nospacerope*) NAME_SAYS_OFF=1 ;; *) NAME_SAYS_OFF=0 ;; esac
case " $EXTRA " in *" --no-space-rope "*) FLAG_SAYS_OFF=1 ;; *) FLAG_SAYS_OFF=0 ;; esac
if [ "$NAME_SAYS_OFF" != "$FLAG_SAYS_OFF" ]; then
  echo "FATAL: space_rope mismatch — ckpt name says off=$NAME_SAYS_OFF, flags say off=$FLAG_SAYS_OFF" >&2
  echo "  ckpt : $CKPT" >&2
  echo "  extra: '$EXTRA'" >&2
  echo "  This is the ONE flag no state_dict check can catch. Refusing to write a cache that" >&2
  echo "  would be silently wrong. Pass --no-space-rope for a nospacerope ckpt, or omit it." >&2
  exit 1
fi
echo "[encode] space_rope guard OK: ckpt-name off=$NAME_SAYS_OFF, extra flags: '${EXTRA:-<none>}'"
EXTRA_ARGS=()
[ -n "$EXTRA" ] && read -ra EXTRA_ARGS <<< "$EXTRA"
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
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
