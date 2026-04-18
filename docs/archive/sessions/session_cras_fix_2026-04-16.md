# Session notes — coord-frame fix + Tier-1 re-derivation (2026-04-16)

Supersedes the central thesis of `docs/spatial_parity_handoff_2026-04-16.md`.
That handoff correctly identified the symptom (catastrophic argmax=0 parity
between volumetric oracle and surface bakes) but mis-identified the cause.
The real cause: the oracle's cached coords were not in MNI.

## Root cause

`sub2AvgBrainClinical.m` and its Python port in `coordinates.py` v1
terminated at a vertex lookup in
`cvs_avg35_inMNI152/surf/<hem>.pial-outer-smoothed`. That surface has
`cras = (0, 0, 0)` stored in its FreeSurfer header, while the sibling
`lh.pial` has `cras = (-1, -17, 19)`. Both surfaces are in the same
subject tkrRAS; only `lh.pial` has the cras metadata to translate to
scanner RAS. Without adding cras, cached coords were offset from true
MNI by +(-1, -17, 19). Sampling BNA_PM (in MNI) at these coords landed
~22 mm off every electrode, and flipped whole sub-gyri.

Triangulation across S14/S26/S33/S62 confirmed: surface-baked (frame-
agnostic vertex lookup) argmax matches volumetric sampling only when
cras is added to the cached coords; without cras, volumetric argmax lives
17-19 mm anterior/inferior of the correct parcel.

## Fix (on our side)

`src/speech_decoding/v14/coordinates.py` v2 (algorithm version
`"sub2AvgBrainClinical.m python port v2 (2026-04-16, cras-corrected)"`):

- Reads cras from `<avg>/surf/lh.pial` at runtime (not hardcoded).
- Asserts `lh.pial` cras == `rh.pial` cras (or raises).
- Adds cras to the step-4 output, so output is in scanner RAS (MNI).
- Sidecar metadata now records `avg_cras_mm`.
- Generalizes: for fsaverage (cras=0) it's a no-op; for cvs_avg35
  it's the correct translation. If `DEFAULT_AVG_SUBJECT` is ever
  changed, the fix automatically adapts.

Tests updated (`tests/v14/test_coordinates.py`):
- Tightened MNI bounding box (a missing cras shift would fail it).
- Added S14 landmark assertion: centroid must sit in left ventral SMC
  corridor.
- MATLAB oracle test now adds cras to Zac's fixture before comparing.

All 84 v14 tests pass post-fix.

## What moved

`data/mni_coords/*_MNI152.csv` for S14, S16, S23, S26, S33, S39, S62 —
regenerated with v2. Per-electrode delta vs archived v1 is exactly
`(-1, -17, 19)` mm for every electrode on every patient.

Old caches archived under `data/mni_coords/archive_pre_cras_fix/` with
a README that gives the inverse-shift formula (`mni = archive + (-1,
-17, 19)`).

Old ranking archived at `data/atlas/parcel_support_pre_cras_fix.csv`.
New ranking at `data/atlas/parcel_support.csv`.

## Tier-1 diff

argmax_wins ≥ 10 rule still yields 15 parcels on the corrected cache
(cardinality robust to the fix).

| Status | ROI | Region | Old argmx | New argmx |
|---|---|---|---|---|
| kept | 53 | PrG A4hf (face/head M1) | 168 | 247 |
| kept | 155 | PoG A1/2/3ulhf (face/head S1) | 28 | 193 |
| kept | 63 | PrG A6cvl (ventral premotor) | 24 | 176 |
| kept | 17 | MFG IFJ | 27 | 147 |
| kept | 33 | IFG A45c (Broca caudal) | 223 | 89 |
| **new** | 139 | IPL A40rd (supramarginal) | 0 | 84 |
| kept | 39 | IFG A44v (Broca ventral) | 123 | 66 |
| kept | 29 | IFG A44d (Broca dorsal) | 34 | 64 |
| kept | 31 | IFG IFS | 181 | 55 |
| kept | 61 | PrG A4tl (tongue M1) | 44 | 52 |
| **new** | 159 | PoG A2 (dorsal sensory) | 0 | 31 |
| kept | 73 | STG TE1.0/1.2 (Heschl) | 12 | 23 |
| **new** | 23 | MFG A8vl | 0 | 20 |
| kept | 21 | MFG A9/46v | 100 | 17 |
| kept | 157 | PoG A1/2/3tonIa (tongue S1) | 29 | 15 |
| **dropped** | 35 | IFG A45r (Broca rostral) | 199 | **0** |
| **dropped** | 51 | OrG A12/47l (lateral orbital) | 48 | **0** |
| **dropped** | 77 | STG A38l (temporal pole) | 40 | **0** |

Direction of the shift: dropped parcels are anterior-ventral-orbital;
added parcels are posterior-dorsal-parietal. Matches the inverse of
+17 mm anterior / -19 mm ventral shift.

Dominant speech-motor anchors unchanged (A4hf, A44v, A44d, A45c, IFJ,
A6cvl, A1/2/3, TE1.0/1.2).

## Two surviving paths for electrode-to-parcel assignment

### Path I — pure fsaverage, snap-to-pial

Patient pial → patient sphere.reg → fsaverage sphere.reg → fsaverage
pial nearest vertex → fsaverage_bake_fast2 lookup.

- One pial vertex per electrode. Integer lookup.
- PSF baked into the atlas (2D geodesic FWHM 3.5 mm).
- No query-time hyperparameters. Standard-FreeSurfer target.
- Code already exists: `src/speech_decoding/v14/fsaverage_projection.py`.

### Path II — cvs_avg35 + weighted kernel bridge

Patient → cvs_avg35 dural envelope (corrected coords v2) → Gaussian-
weighted set of cvs_avg35 pial vertices within radius R → cvsavg35_
bake2 lookup (weighted sum).

- Soft assignment. R and σ are hyperparameters.
- Models query-time PSF (can be patient/impedance specific later).
- Cons: cvs_avg35 is a non-standard, stripped reconstruction (137k
  vertices, no lh.white, so atlas smoothing used Python adjacency
  fallback). More machinery.

Decision criterion: Path I is strictly simpler with comparable
correctness. Path II is only superior if we foresee needing
per-electrode or per-patient PSF adjustment at query time.

## Outstanding blockers / contract items to re-audit

The following items in `docs/implementation_tasks.md` were all derived
from or justified against the buggy cache. Status of each should be
re-examined before the Phase-1 loader is written:

- `#3` Token mask rule (argmax in Tier-1): needs to target the new 15.
- `#4` Tier-1 list: frozen on old ranking; diff shown above.
- `#5` Support statistic (Gaussian σ=1.5 mm over dilated PM): the σ
  is still physically motivated, but the dilated PM choice is
  intertwined with `#35`.
- `#35` PM dilation (8 mm): introduced to patch "coverage gaps." Many
  of those gaps may have been shift artifacts. Need to test whether
  raw `BNA_PM_4D.nii.gz` now gives adequate coverage.
- `#36` fsaverage migration: the step-3/4 audits say the bake is
  correct; `cvs_avg35_bake2` also correct. Once parity scripts re-run
  on corrected cache, this may pass cleanly.
- `#34` Phoneme audit: independent of spatial side; S14 pilot PASS
  stands.

## Next steps (in order of impact)

1. **Decide Path I or II.** Single biggest decision blocking the loader.
2. **Unfreeze + re-freeze `token_spec.py`** if the new 15 is accepted.
3. **Re-run parity scripts** on the corrected cache — confirm fsaverage
   passes cleanly, confirm `#36` is unblocked.
4. **Test raw `BNA_PM_4D.nii.gz` (no dilation).** If coverage is
   adequate, retire `#35` and the dilated PM volume.
5. **Rewrite `docs/spatial_parity_handoff_2026-04-16.md`** with a
   pointer to this doc.
6. **Send Zac message** (draft in session history).

## Files touched this session

Edited:
- `src/speech_decoding/v14/coordinates.py` — v2 cras fix
- `tests/v14/test_coordinates.py` — tightened bounds, landmark check,
  oracle test frame-adjusted

Written (artifacts):
- `data/mni_coords/{S14,S16,S23,S26,S33,S39,S62}_MNI152.csv` — v2
- `data/mni_coords/_projection_meta.json` — v2
- `data/mni_coords/archive_pre_cras_fix/` — archived v1 caches + README
- `data/atlas/parcel_support.csv` — new ranking
- `data/atlas/parcel_support_pre_cras_fix.csv` — archived old ranking
- `docs/session_cras_fix_2026-04-16.md` — this file

Not touched (awaiting decision):
- `src/speech_decoding/v14/token_spec.py` (frozen; 3-in-3-out diff
  above)
- `docs/implementation_tasks.md` (#3, #4, #5, #35, #36 all affected)
- `docs/spatial_parity_handoff_2026-04-16.md` (thesis refuted)
- Parity scripts (not re-run)
- `scripts/dilate_pm.py` and `BNA_PM_dilated_8mm.nii.gz` (may be
  retirable)
