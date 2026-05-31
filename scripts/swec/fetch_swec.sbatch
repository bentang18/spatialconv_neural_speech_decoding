#!/bin/bash
# Full SWEC-iEEG pull + audit as a Slurm job on a CPU compute node.
#
# WHY Slurm and not the login node: hf_xet spawns many parallel workers; the DCC
# login-node cgroup/watchdog SIGKILLs it within seconds. Compute nodes have
# outbound internet (verified 2026-05-30, HTTP/2 200 to huggingface.co) and run
# in an independent cgroup, so the multi-hour pull survives SSH disconnect.
#
# Submit (from the laptop, after scripts/dcc/sync):
#   ssh ht203@dcc-login.oit.duke.edu \
#     'cd /work/ht203/repo/speech && sbatch scripts/swec/fetch_swec.sbatch'
# Resumable: if it hits the time limit, just resubmit — snapshot_download skips
# completed files.
#
#SBATCH --job-name=swec_fetch
#SBATCH --account=coganlab
#SBATCH --partition=common
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/work/ht203/data/swec/swec_fetch_%j.log

set -euo pipefail
cd /work/ht203/repo/speech

echo "[$(date)] host=$(hostname) — SWEC full fetch (50 unique folders)"
# HF token is read from ~/.cache/huggingface/token by huggingface_hub (not env).
.venv/bin/python -u scripts/swec/fetch_swec.py

echo "[$(date)] fetch complete — auditing all 50 on-disk folders"
.venv/bin/python -u scripts/swec/audit_swec.py --psd

echo "[$(date)] DONE — see verified_manifest.csv"
