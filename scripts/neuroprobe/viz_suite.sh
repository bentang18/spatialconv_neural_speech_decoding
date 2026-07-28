#!/usr/bin/env bash
# Regenerate the whole cross-subject visualization suite from one reduction directory.
#
#   scripts/neuroprobe/viz_suite.sh <RED_DIR> <OUT_DIR> <OFFSET_S> [RED_ELEC_DIR] [COORDS_NPZ]
#
# Four scripts, one window. They were run by hand for the 1 s pass and the arguments that
# have to agree between them -- the taps, the task list, the seconds of the first frame --
# only agreed because I retyped them correctly each time. OFFSET_S is the one that would
# fail silently: it is only a time-axis label, so a wrong value mislabels every panel and
# nothing crashes. Passing it once, here, is the point of the file.
#
# The brain render is optional because it needs a PER-ELECTRODE reduction, which is a
# separate (and much smaller) set of taps than the pooled one the other three read.
set -euo pipefail
RED="${1:?usage: viz_suite.sh <RED_DIR> <OUT_DIR> <OFFSET_S> [RED_ELEC_DIR] [COORDS_NPZ]}"
OUT="${2:?need OUT_DIR}"
OFF="${3:?need OFFSET_S, e.g. 0 for a 0-1s window or -0.5 for a 2s window centred on onset}"
RED_ELEC="${4:-}"
COORDS="${5:-}"
# Board results for the SAME checkpoint, so the page can show decoding accuracy next to the
# geometry. Optional; an env var rather than a positional so it cannot be confused with the
# two optional brain-render paths.
BOARD_JSON="${BOARD_JSON:-}"

# DERIVED, never passed in. The baseline is the pre-stimulus part of the window, so it is a
# function of OFFSET and the frame rate and nothing else. Typing it separately is how the
# time axis got mislabelled the first time; one input, one derivation.
HZ=32
NPRE=$(python3 -c "import sys; o=float(sys.argv[1]); print(max(0, round(-o*$HZ)))" "$OFF")
echo "### origin: n_pre=$NPRE frames (offset $OFF s at $HZ Hz)"

PY=.venv/bin/python
TAPS=enc0,enc3,enc6,enc12
# All 15 Neuroprobe tasks, in `braintreebank/labels.py` order. The reduction must have been
# built with the same list -- `viz_reduce.py --tasks` defaults to a 6-task subset, and a task
# missing from the shards is dropped from the page rather than shown empty.
TASKS=onset,speech,volume,delta_volume,pitch,word_index,word_gap,gpt2_surprisal,word_head_pos,word_part_speech,word_length,global_flow,local_flow,frame_brightness,face_num
mkdir -p "$OUT"

echo "### figures  (red=$RED out=$OUT taps=$TAPS)"
$PY -m scripts.neuroprobe.viz_figures --red-dir "$RED" --out-dir "$OUT" \
  --task onset --taps "$TAPS" --tasks-quant "$TASKS" \
  --n-pre "$NPRE" --hz "$HZ" --offset "$OFF"

echo "### videos"
$PY -m scripts.neuroprobe.viz_video --red-dir "$RED" --out-dir "$OUT" \
  --taps enc12 --tasks "$TASKS" --animate onset,speech,frame_brightness --n-pre "$NPRE"

echo "### demo page"
$PY -m scripts.neuroprobe.viz_demo --red-dir "$RED" --out "$OUT/demo.html" \
  --taps "$TAPS" --tasks "$TASKS" --offset "$OFF" --n-pre "$NPRE" \
  ${BOARD_JSON:+--board-json "$BOARD_JSON"}

if [[ -n "$RED_ELEC" && -n "$COORDS" ]]; then
  echo "### brain render"
  $PY -m scripts.neuroprobe.viz_brain --red-dir "$RED_ELEC" --coords "$COORDS" \
    --out-dir "$OUT" --tap enc12_elec --task onset --offset "$OFF" --video --fps 12
else
  echo "### brain render SKIPPED (no per-electrode reduction given)"
fi

echo "### done -> $OUT"
ls -1 "$OUT" | wc -l | xargs echo "files:"
