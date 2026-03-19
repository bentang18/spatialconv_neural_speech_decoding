#!/usr/bin/env bash
# Hyperparameter sweep on S14: Runs 5-8
# Each run modifies per_patient.yaml, trains, then restores for the next.
# Run 4 (blank_bias=0.0, original augmentation) is the baseline for comparison.
set -euo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$(dirname "$0")/.."
source .venv/bin/activate

CONFIG=configs/per_patient.yaml
DEVICE=mps
PATIENTS="S14"

# Helper: run training and label it
run_experiment() {
    local label="$1"
    echo ""
    echo "============================================================"
    echo "SWEEP: $label"
    echo "============================================================"
    python scripts/train_per_patient.py --config "$CONFIG" --patients $PATIENTS --device $DEVICE
}

# --- Run 5: Augmentation tuning ---
# time_shift 30→15, amp_scale 0.3→0.15, noise 0.05→0.10
# (Already staged in per_patient.yaml by previous edit)
run_experiment "Run 5: aug tune (time_shift=15, amp_scale=0.15, noise=0.10)"

# --- Run 6: Weight decay 1e-3 (was 1e-4) ---
# Reset augmentation is already set from Run 5, keep those changes
sed -i '' 's/weight_decay: 1.0e-4/weight_decay: 1.0e-3/' "$CONFIG"
run_experiment "Run 6: weight_decay=1e-3"
# Restore
sed -i '' 's/weight_decay: 1.0e-3/weight_decay: 1.0e-4/' "$CONFIG"

# --- Run 7: LR 5e-4 (was 1e-3) ---
sed -i '' 's/lr: 1.0e-3/lr: 5.0e-4/' "$CONFIG"
run_experiment "Run 7: lr=5e-4"
# Restore
sed -i '' 's/lr: 5.0e-4/lr: 1.0e-3/' "$CONFIG"

# --- Run 8: GRU dropout 0.1 (was 0.3) ---
sed -i '' 's/gru_dropout: 0.3/gru_dropout: 0.1/' "$CONFIG"
run_experiment "Run 8: gru_dropout=0.1"
# Restore
sed -i '' 's/gru_dropout: 0.1/gru_dropout: 0.3/' "$CONFIG"

echo ""
echo "============================================================"
echo "SWEEP COMPLETE — all 4 runs finished"
echo "============================================================"
