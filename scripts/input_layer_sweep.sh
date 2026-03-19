#!/usr/bin/env bash
# Input layer sweep on S14 with CE loss, stride=10 (20Hz)
# Tests: baseline stride change, less aggressive pooling, more conv channels
set -euo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1
PYTHON=".venv/bin/python"
SCRIPT="scripts/train_per_patient.py"
DEVICE="mps"

echo "INPUT SWEEP: Run A — CE + stride=10 baseline (pool 2×4, C=8, d_shared=64)"
$PYTHON $SCRIPT --config configs/per_patient_ce_s10.yaml --patients S14 --device $DEVICE

echo "INPUT SWEEP: Run B — CE + stride=10 + pool(4,8) → d_shared=256"
$PYTHON $SCRIPT --config configs/per_patient_ce_s10_pool48.yaml --patients S14 --device $DEVICE

echo "INPUT SWEEP: Run C — CE + stride=10 + C=16 → d_shared=128"
$PYTHON $SCRIPT --config configs/per_patient_ce_s10_c16.yaml --patients S14 --device $DEVICE

echo "INPUT SWEEP COMPLETE"
