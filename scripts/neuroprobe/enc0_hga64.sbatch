#!/bin/bash
#SBATCH --job-name=enc0_hga64
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bhqk/htang13/enc0_hga64_%A_%a.out
#
# "Does a 64 Hz high-gamma stream beat the current 32 Hz one at enc0?" (Ben 2026-08-02)
#
# enc0 is a pure function of the CACHED BANDS -- no ckpt, no model, no GPU (--enc0-only) -- so
# this runs on Delta CPU, not the GPU partition the board encode uses. 32 cores sits under
# 0.512*64 GB, so the cores are free and the bill is (mem/2)*wall.
#
# The HGA cache already existed: /work/nvme/.../v14_finehga_spec_eval/band_v3hga, baked
# 2026-07-22 for the r5 2-stream arm at band_nperseg=64 / band_hop=32 => a 31.25 ms window at
# 64 Hz with 4 bins over 64-160 Hz. SLOW/MID come from v14_3band_v3_spec_lite (band_hop=64,
# 32 Hz) -- the SAME leaves the published enc0 used, so slow/mid are untouched. NOTE the finehga
# dir carries its OWN band_v3slow/band_v3mid at the pre-uniform-hop rates (512/128); using those
# would silently change two more axes, which is why they are not referenced here.
#
# --band-rates 1/1,1/1,2/1 DECLARES that third cache is at 2x the 32 Hz clip clock. It is checked
# against every cache's real band_hop (assert_band_rates_match_cache); a wrong declaration is a
# compressed, time-shifted slice at the RIGHT shape, not an error.
#
# ARM (arg 1):
#   parity -- uniform 32 Hz caches, NO --band-rates: must reproduce the published enc0. Until it
#             does, no number from the `fine` arm means anything.
#   fine   -- slow/mid @32 Hz + HGA @64 Hz. enc0 goes 348 -> 380 cols/unit: 4x7 | 16x6 | 64x4.
#
# usage: sbatch --array=0-11 enc0_hga64.sbatch <parity|fine> <OUTDIR>
set -euo pipefail
ARM="${1:?usage: sbatch --array=0-11 enc0_hga64.sbatch <parity|fine> <OUTDIR>}"
OUTDIR="${2:?need OUTDIR}"
TREE=/projects/bhqk/htang13/speech_enc0
cd "$TREE"
module load pytorch-conda/2.8
# The script is run as `-m scripts.neuroprobe...` (PEP 420 namespace pkg), so the TREE ROOT must
# be importable, and src/ carries speech_decoding.
#
# USER_SITE is on PYTHONPATH deliberately. The encode's import closure reaches `mne` (via
# braintreebank/study.py), which is installed in ~/.local but INVISIBLE here: the pytorch-conda
# env sets ENABLE_USER_SITE=False, so the interpreter prints a user-site path it never adds. The
# readout never imports mne, which is why only the encode trips on it. Adding the path is
# deliberately preferred over `pip install --user`: the user-site dir is shared with every job on
# this module, and a pip run there could drag numpy/scipy under the LOCKED readout.
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
export PYTHONPATH="$TREE:$TREE/src:$USER_SITE:${PYTHONPATH:-}"
python -c 'import mne; print("[check] mne", mne.__version__)'
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai

LITE=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_lite
FINE=/work/nvme/bhqk/htang13/cache_neuroai/v14_finehga_spec_eval

case "$ARM" in
  parity) HGA_DIR="$LITE/band_hga";      RATES=() ;;
  fine)   HGA_DIR="$FINE/band_v3hga";    RATES=(--band-rates 1/1,1/1,2/1) ;;
  *) echo "unknown arm '$ARM' (want parity|fine)" >&2; exit 2 ;;
esac

echo "[check] arm=$ARM hga_cache=$HGA_DIR rates=${RATES[*]:-uniform} idx=${SLURM_ARRAY_TASK_ID:-0}"
echo "[check] mem=${SLURM_MEM_PER_NODE:-?}M cpus=${SLURM_CPUS_PER_TASK:-?} free-core-ceiling=$(( ${SLURM_MEM_PER_NODE:-0} * 512 / 1000000 ))"

numactl --interleave=all python -u -m scripts.neuroprobe.v3_probe_encode_r4 \
  --tag "enc0_$ARM" --out-dir "$OUTDIR" \
  --band-cache-dir "$LITE/band_v3slow" \
  --band-cache-dir "$LITE/band_v3mid" \
  --band-cache-dir "$HGA_DIR" \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --frontend v3r6 \
  --enc0-only --elec-taps 0 \
  --sessions board --tasks board15 --electrode-set lite \
  --session-index "${SLURM_ARRAY_TASK_ID:-0}" \
  "${RATES[@]}"
