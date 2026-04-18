# Spatial Parity Handoff (2026-04-16) — SUPERSEDED

**This document's central thesis is refuted. See
`docs/session_cras_fix_2026-04-16.md` for the current state.**

## Status

This note was written during the first half of 2026-04-16, when
fsaverage bakes and cvs_avg35 volumetric lookups were failing to agree
catastrophically (argmax agreement at 0–3%). The hypothesis in this
document — that the divergence reflects a semantic gap between "2D
surface lookup" and "3D volumetric lookup" — turned out to be wrong.

The actual cause, found later the same day:
`src/speech_decoding/v14/coordinates.py` v1 (and its MATLAB source
`sub2AvgBrainClinical.m`) terminated at a vertex lookup on
`cvs_avg35_inMNI152/surf/lh.pial-outer-smoothed`. That surface has
`cras = (0, 0, 0)` stored in its FreeSurfer header while actually
living in subject tkrRAS. Sibling `lh.pial` has `cras = (-1, -17, 19)`.
Every cached `_MNI152.csv` was offset from true MNI by `+(-1, -17, 19)
mm` for every electrode on every patient, so the "volumetric oracle"
this document compared against was sampling BNA PM ~22 mm off the
correct cortical point on every patient.

Once the cras translation was added (`coordinates.py` v2), the cvs_avg35
oracle and the fsaverage bake agree on relative Tier-1 ranking
(spearman >= 0.6 on all 7 patients, >= 0.9 on 4/7) and disagree on
argmax only where the two operators legitimately differ (2D geodesic
smoothing vs 3D Euclidean + 8 mm dilation; see
`data/atlas/fsaverage_parity/decision_memo.md`).

## What's still true from this document

- The fsaverage bake is anatomically correct. The 2–4-step audit in the
  "what was tested" section below still holds.
- `data/atlas/fsaverage_bake_fast2/` is the active 246-frame bake.
- `scripts/build_fsaverage_parcel_frames.py` / `_spatial_parity.py`
  still exist and run.

## What's no longer true

- The gate `min_argmax_agreement = 0.85` against the volumetric oracle
  is not a correctness gate; see the decision memo.
- "Volumetric vs surface lookup is load-bearing" is not the
  divergence. The coord frame was.
- Tier-1 list `A45r / A12/47l / A38l / TE1.0/1.2` (from the buggy cache)
  is stale. The re-derived fsaverage Tier-1 15 is in
  `src/speech_decoding/v14/token_spec.py`.

## Where to read next

- `docs/session_cras_fix_2026-04-16.md` — full session log of the
  coord-frame fix, cache regeneration, and Tier-1 re-derivation.
- `data/atlas/fsaverage_parity/decision_memo.md` — revised parity
  interpretation and current gate.
- `docs/implementation_tasks.md` — `#4`, `#5`, `#35`, `#36` updated
  reflecting the closure.
- `src/speech_decoding/v14/token_spec.py` — canonical new Tier-1 list.

---

## Original document (preserved for historical context only)

### Goal

Test whether we can replace the old runtime volumetric Brainnetome
lookup with a cleaner surface-baked atlas pipeline **without materially
changing the spatial labeling regime** used by the current oracle.

The parity gates used throughout were:

- Tier-1 symmetric difference `<= 2`
- per-patient argmax agreement `>= 0.85`
- per-patient Tier-1 token-support Spearman `>= 0.9`

These gates assumed the volumetric oracle was ground truth. That
assumption did not survive the cras-fix investigation.

### Baseline Oracle (as understood at the time)

- code: `src/speech_decoding/v14/coordinates.py` v1
- cached outputs: `data/mni_coords/*_MNI152.csv` (pre-cras-fix;
  archived under `data/mni_coords/archive_pre_cras_fix/`)

Semantics at the time the document was written:

1. patient native electrode
2. -> patient `pial-outer-smoothed`
3. -> patient `sphere-outer-mni.reg`
4. -> `cvs_avg35_inMNI152` `sphere-outer.reg`
5. -> `cvs_avg35_inMNI152` `pial-outer-smoothed`
6. runtime query of dilated volumetric Brainnetome PM support at the
   "MNI" coord.

The coord in step 5 was never actually in MNI — the cras translation
was missing. Post-fix, the same path produces true MNI coords.
