#!/bin/bash
#SBATCH --job-name=viz_coords
#SBATCH --account=bhqk-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/bhqk/htang13/viz/coords_%j.out
set -euo pipefail
# A batch job rather than a login-node run: it is only a few CSV reads per subject, but it
# opens 12 encode shards to read parcel_canon, and the login node is capped at 60s/1core/4GB.
CACHE="${1:?need CACHE}"
BT_ROOT="${2:?need BT_ROOT}"
OUT="${3:?need OUT}"
module load pytorch-conda/2.8
numactl --interleave=all python -u /projects/bhqk/htang13/viz/viz_coords_dump.py \
  --cache "$CACHE" --bt-root "$BT_ROOT" --out "$OUT"
