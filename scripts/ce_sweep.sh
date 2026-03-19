#!/usr/bin/env bash
# CE architecture sweep on S14: tests GRU capacity, temporal stride, and time crop
set -euo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1
PYTHON=".venv/bin/python"
SCRIPT="scripts/train_per_patient.py"
DEVICE="mps"

echo "CE SWEEP: Run 1 — CE H=128 (vs H=32 baseline PER=0.700)"
$PYTHON $SCRIPT --config configs/per_patient_ce_h128.yaml --patients S14 --device $DEVICE

echo "CE SWEEP: Run 2 — CE H=128 + time crop [0, 1.0s] (removes 500ms pre-stimulus)"
$PYTHON $SCRIPT --config configs/per_patient_ce_h128.yaml --patients S14 --device $DEVICE --tmin 0.0 --tmax 1.0

echo "CE SWEEP: Run 3 — CE H=128 + stride=10 (20Hz, matches Spalding/Duraivel)"
$PYTHON $SCRIPT --config configs/per_patient_ce_h128_s10.yaml --patients S14 --device $DEVICE

echo "CE SWEEP COMPLETE"
