#!/usr/bin/env bash
# Fast sweep: 1 seed (42) only, per-patient CE variants on S14
set -euo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python"
SC="scripts/train_per_patient.py"
D="${1:-mps}"

# Override seeds to [42] only by patching at runtime via a temp config approach
# Simpler: just use --seeds flag if supported, otherwise we modify configs
# The train script doesn't have --seeds flag, so let's add a quick one-liner

run() {
  local name="$1" config="$2"
  echo ""
  echo "=== $name ==="
  # Create temp config with seeds: [42]
  local tmp="/tmp/sweep_${name}.yaml"
  sed 's/seeds: \[42, 137, 256\]/seeds: [42]/' "$config" > "$tmp"
  $PY $SC --config "$tmp" --patients S14 --device "$D"
}

echo "SWEEP: 1 seed, S14, per-patient CE variants"
echo "Baseline reference: CE d=64 H=32 → PER=0.700 (mean 3 seeds)"
echo "============================================================"

run "2layer_d64"  "configs/per_patient_ce_2layer_d64.yaml"
run "c16_d128"    "configs/per_patient_ce_c16_d128.yaml"
run "ce_attn"     "configs/per_patient_ce_attn.yaml"
run "balanced"    "configs/per_patient_ce_balanced.yaml"

echo ""
echo "============================================================"
echo "SWEEP COMPLETE"
