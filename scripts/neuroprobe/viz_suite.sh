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

PY=.venv/bin/python
TAPS=enc0,enc3,enc6,enc12
TASKS=onset,speech,delta_volume,word_index,word_part_speech,frame_brightness
mkdir -p "$OUT"

echo "### figures  (red=$RED out=$OUT taps=$TAPS)"
$PY -m scripts.neuroprobe.viz_figures --red-dir "$RED" --out-dir "$OUT" \
  --task onset --taps "$TAPS" --tasks-quant "$TASKS"

echo "### videos"
$PY -m scripts.neuroprobe.viz_video --red-dir "$RED" --out-dir "$OUT" \
  --taps enc12 --tasks "$TASKS" --animate onset,speech,frame_brightness

echo "### demo page"
$PY -m scripts.neuroprobe.viz_demo --red-dir "$RED" --out "$OUT/demo.html" \
  --taps "$TAPS" --tasks "$TASKS" --offset "$OFF"

if [[ -n "$RED_ELEC" && -n "$COORDS" ]]; then
  echo "### brain render"
  $PY -m scripts.neuroprobe.viz_brain --red-dir "$RED_ELEC" --coords "$COORDS" \
    --out-dir "$OUT" --tap enc12_elec --task onset --offset "$OFF" --video --fps 12
else
  echo "### brain render SKIPPED (no per-electrode reduction given)"
fi

echo "### done -> $OUT"
ls -1 "$OUT" | wc -l | xargs echo "files:"
