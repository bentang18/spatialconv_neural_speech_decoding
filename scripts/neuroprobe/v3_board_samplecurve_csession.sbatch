#!/bin/bash
#SBATCH --job-name=v3_samplecurve_csn
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%A_%a.out
#
#   sbatch scripts/neuroprobe/v3_board_samplecurve_csession.sbatch
#   sbatch --array=0 --mem=16G --time=00:30:00 --dependency=afterany:<ARRAYID> \
#          scripts/neuroprobe/v3_board_samplecurve_csession.sbatch merge
#
# R31/CSession — the CROSS-SESSION label-efficiency curve. This is the CONTROL that gives the
# within-session-vs-cross-subject contrast its meaning.
#
# ═══ WHY IT IS WORTH ITS COMPUTE ════════════════════════════════════════════════════════════════
# The claim under test is that pretraining behaves differently ACROSS a subject boundary than it
# does within one. With only ws and cs measured, the obvious objection is that cs is simply HARDER
# — more distribution shift, lower operating point — and that nothing about SUBJECTS is doing the
# work. csession splits those apart: same patient, different session, so the recording changes but
# the brain does not.
#
#   ws        train = the target session itself     N = calibration trials FROM the target
#   csession  train = the target's OTHER session    N = same patient, another day     <-- HERE
#   cs        train = a different patient           N = someone else entirely
#
# If csession sits with ws, the boundary is the SUBJECT. If it sits with cs, the boundary is merely
# "train and test are different recordings" and the subject framing is wrong. Either way the answer
# changes what we may write, which is what makes it a measurement rather than a nice-to-have.
#
# ═══ AND IT IS THE TAP-MATCHED CONTROL — THIS IS THE REAL REASON ════════════════════════════════
# ws and cs do NOT share a tap: ws reads enc*_elec (per-electrode), cs reads enc* (parcel-mean),
# because electrode identity does not survive a subject change. So the measured ws-vs-cs contrast
# confounds TWO changes at once — the train data moved to another brain, AND the readout unit
# changed. Any claim of the form "the intercept is larger cross-subject" is, as it stands, partly a
# statement about parcel-mean pooling.
# csession breaks that confound: it runs the SAME enc*_elec taps as ws while moving the train data
# off the target recording. ws -> csession is therefore a clean one-variable step. Whatever survives
# it is about the DATA, not about the pooling.
#
# ═══ THE SIDECAR IS NOT OPTIONAL ════════════════════════════════════════════════════════════════
# csession intersects electrodes by IDENTITY (_elec_cols), and the encoder never wrote elec_labels,
# so without --elec-labels-sidecar every cell returns {} and the job exits 0 having measured
# NOTHING (board_readout_lean.sbatch:36-39). The script now refuses to start in csession mode
# without it, and the anchor check would raise ANCHOR VACUOUS even if it did. Two guards, because a
# clean-looking zero is the single most expensive failure mode here.
# Labels are a pure function of (channels, drop-set) and independent of model weights, so the arm0
# sidecar is valid for any arm on the same 12 Lite sessions (verified 07-26, per-session electrode
# counts match exactly).
#
# ═══ 12 ARRAY TASKS ═════════════════════════════════════════════════════════════════════════════
# CSESSION_CELLS == LITE_SESSIONS: every Lite session is a cross-session test cell, trained on its
# sibling trial. Same sharding the board's csession readout uses.
#
# ═══ MEMORY: 200G, THE BOARD'S OWN CSESSION NUMBER ══════════════════════════════════════════════
# board_readout_all3.sh:48 runs csession at 200G, and its header (:12-14) attaches the evidence:
# csession loads TWO records with the HEAVY per-electrode taps (unlike cs, which is parcel-mean and
# ~40x lighter). 180G OOM-killed 8 of 12 cells (20298990); 176G completed but peaked at MaxRSS 181G
# (20548324) — over its own request. 190G and 210G completed. So 200G is a measured floor plus a
# real margin, not a guess, and this curve holds MORE than the board readout does (it keeps the
# subsample bookkeeping alive across the N grid), so shaving it is the wrong direction.
# Cores are FREE below 0.512 x mem_GB = 102, so 32 bills identically to 8.
#
# ═══ WALLTIME: 12 h ════════════════════════════════════════════════════════════════════════════
# The reference point is MEASURED: the WS curve (20715662) ran 2:01-2:25 per cell on the same heavy
# enc12_elec tap. csession moves that in two opposite directions and the net is not knowable up
# front, which is exactly why the limit is set where a timeout cannot truncate the array:
#   CHEAPER  no folds and only the `trainonly` column, so far fewer fits than WS.
#   DEARER   it fits the WHOLE sibling session where WS fits a KFold half, so n is ~2x => the O(n^2 d)
#            gram is ~4x and the O(n^3) eigendecomposition ~8x PER FIT, and the grid runs one rung
#            further (N_GRID_CS tops out at 2048, which a full sibling session can actually reach).
# The bill is on ACTUAL wall, so a generous limit costs only backfill position; a timeout costs the
# night. 12 h is ~5x the measured WS cell and matches the CS launcher already proven on this tree.
set -uo pipefail

MODE="${1:-shard}"

module load pytorch-conda/2.8
TREE=/projects/bhqk/htang13/speech_board
export PYTHONPATH="$TREE:$TREE/src:${PYTHONPATH:-}"
cd "$TREE"

CACHE=/projects/bhqk/htang13/v3_board_cache_board_pbs50_cd45k
TAG=pbs50_cd45k
SIDECAR=/projects/bhqk/htang13/arm0_elec_labels_sidecar.pkl
SHARDS=/projects/bhqk/htang13/samplecurve_csession_shards_${TAG}
OUT=/projects/bhqk/htang13/samplecurve_csession_${TAG}.json

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

mkdir -p "$SHARDS"

if [[ "$MODE" == "merge" ]]; then
  echo "[R31/csession merge] shards=$SHARDS"
  numactl --interleave=all python -u scripts/neuroprobe/v3_board_samplecurve.py \
    --mode merge --regime csession --shard-dir "$SHARDS" --out "$OUT"
  exit $?
fi

IDX="${SLURM_ARRAY_TASK_ID:?this script must run as an array (--array=0-11)}"
echo "[R31/csession shard] idx=$IDX cache=$CACHE tag=$TAG"
echo "[contract]  CSession | taps enc0_elec,enc12_elec | SIBLING session subsampled"
echo "[contract]  target val/test NEVER subsampled — they are the protocol's"
echo "[anchor]    N=full MUST reproduce the published _csession_cell — read the [anchor] line below"
numactl --interleave=all python -u scripts/neuroprobe/v3_board_samplecurve.py \
  --mode csession --cache-dir "$CACHE" --tag "$TAG" --index "$IDX" --shard-dir "$SHARDS" \
  --elec-labels-sidecar "$SIDECAR"
