#!/usr/bin/env bash
# Sweep per-patient CE variants on S14
# ISOLATED experiments first (change one thing), then COMPOUND
set -euo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1
PYTHON=".venv/bin/python"
SCRIPT="scripts/train_per_patient.py"
DEVICE="${1:-mps}"

echo "============================================================"
echo "ISOLATED EXPERIMENTS (one change from CE baseline, PER=0.700)"
echo "============================================================"

echo ""
echo "=== 1. 2-layer Conv2d at d=64 (+584 params, total ~42K) ==="
$PYTHON $SCRIPT --config configs/per_patient_ce_2layer_d64.yaml --patients S14 --device "$DEVICE"

echo ""
echo "=== 2. C=16 at d=128 (+10K params, total ~52K) ==="
$PYTHON $SCRIPT --config configs/per_patient_ce_c16_d128.yaml --patients S14 --device "$DEVICE"

echo ""
echo "=== 3. Position attention (replaces mean pooling, +1.5K params) ==="
$PYTHON $SCRIPT --config configs/per_patient_ce_attn.yaml --patients S14 --device "$DEVICE"

echo ""
echo "============================================================"
echo "COMPOUND EXPERIMENTS (multiple changes, ~290-480K params)"
echo "============================================================"

echo ""
echo "=== 4. Balanced: d=256, H=64, stride=10 (~289K) ==="
$PYTHON $SCRIPT --config configs/per_patient_ce_balanced.yaml --patients S14 --device "$DEVICE"
