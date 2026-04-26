#!/bin/bash
# A5 — DCC sync recipe for Stage-3 sEEG D-cohort (2026-04-24)
#
# Run FROM LAPTOP (rsyncs over SSH to ht203@dcc-login.oit.duke.edu).
# Every command is prefaced with --dry-run. Remove it once you've
# confirmed the target sizes are what you expect. All paths here
# assume the laptop's Box mount + the local repo's data/ dir.
#
# ORDERING: 1 before 2; 2 before training. 3 is optional safety net.

set -euo pipefail

BOX_COGANLAB="/Users/bentang/Library/CloudStorage/Box-Box/CoganLab"
BOX_RECON="/Users/bentang/Library/CloudStorage/Box-Box/ECoG_Recon"
DCC_USER="ht203"
DCC_HOST="dcc-login.oit.duke.edu"
DCC_WORK="/work/ht203/data"

# ─────────────────────────────────────────────────────────────────────
# 1. Push locally-built D-patient caches (Box-only artifacts) to DCC
#    These are the Stage-3 cohort artifacts we built from Box + the
#    pre-baked 3mm BNA CSVs. DCC has NO copy yet.
# ─────────────────────────────────────────────────────────────────────

# Support cache (122 Tier-1 CSVs, ~500 KB total)
rsync -av --dry-run \
  ./data/atlas/support_cache_v2c_snap_dcohort/ \
  "${DCC_USER}@${DCC_HOST}:${DCC_WORK}/atlas/support_cache_v2c_snap_dcohort/"

# RAS coord cache (128 CSVs, ~1.5 MB)
rsync -av --dry-run \
  ./data/dcohort_coords/ \
  "${DCC_USER}@${DCC_HOST}:${DCC_WORK}/dcohort_coords/"

# Manifest (once B3 writes it)
# rsync -av --dry-run \
#   ./data/dcohort_manifest.csv \
#   "${DCC_USER}@${DCC_HOST}:${DCC_WORK}/dcohort_manifest.csv"

# ─────────────────────────────────────────────────────────────────────
# 2. Stage-3 training data — Box → DCC for the 57 D-patients missing
#    from /datacommons BIDS roots. Nothing needs rsyncing for
#    PS or SentenceRep (DCC has all 50/34 already).
# ─────────────────────────────────────────────────────────────────────

# LexicalDecRepDelay: 31 D-patients Box-only
# See reports/dcc_sync_plan_2026_04_24/missing_lexdelay.txt
while read -r pt; do
  rsync -av --dry-run \
    "${BOX_COGANLAB}/BIDS-1.0_LexicalDecRepDelay/BIDS/sub-${pt}/" \
    "${DCC_USER}@${DCC_HOST}:/datacommons/coganlab/BIDS-1.0_LexicalDecRepDelay/BIDS/sub-${pt}/" \
    || echo "  failed: sub-${pt}"
done < ./reports/dcc_sync_plan_2026_04_24/missing_lexdelay.txt

# LexicalDecRepNoDelay: 26 D-patients, entire task absent from DCC
# See reports/dcc_sync_plan_2026_04_24/missing_lexnodelay.txt
# Need to create a new BIDS root on DCC first — do this ONCE:
#   ssh ht203@dcc-login 'mkdir -p /hpc/group/coganlab/BIDS-1.0_LexicalDecRepNoDelay/BIDS'
while read -r pt; do
  rsync -av --dry-run \
    "${BOX_COGANLAB}/BIDS-1.0_LexicalDecRepNoDelay/BIDS/sub-${pt}/" \
    "${DCC_USER}@${DCC_HOST}:/hpc/group/coganlab/BIDS-1.0_LexicalDecRepNoDelay/BIDS/sub-${pt}/" \
    || echo "  failed: sub-${pt}"
done < ./reports/dcc_sync_plan_2026_04_24/missing_lexnodelay.txt

# ─────────────────────────────────────────────────────────────────────
# 3. (Optional) Newer recons Box-only → DCC
#    18 D-patients are on Box but not on /datacommons/ECoG_Recon_Full
#    OR /hpc/group/coganlab/ECoG_Recon. See missing_recons.txt.
#    Only 7 of these (D116–D125 range) overlap the 85 cohort set.
#    Likely un-needed for Stage-3 training (support caches already
#    built locally), but rsync keeps DCC authoritative for future
#    pipelines.
# ─────────────────────────────────────────────────────────────────────
while read -r pt; do
  rsync -av --dry-run \
    "${BOX_RECON}/${pt}/" \
    "${DCC_USER}@${DCC_HOST}:/hpc/group/coganlab/ECoG_Recon/${pt}/" \
    || echo "  failed: ${pt}"
done < ./reports/dcc_sync_plan_2026_04_24/missing_recons.txt

echo
echo "All commands are --dry-run. Remove the flag after validating sizes."
