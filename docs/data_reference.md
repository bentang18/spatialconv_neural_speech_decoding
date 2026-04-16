# Data Reference

Per-patient tables and corpus sizes extracted from `CLAUDE.md` to keep the main context file light. Load this doc when touching the data loader, channel-bookkeeping code, parcel layout, or SSL plans.

## Array Layouts

Ground truth: `data/recording_details/uecog_recording_details.xlsx` (`Duke Subjects` sheet, `Electrode (Mapping)` column), plus the per-patient files in `data/channel_maps/`. The old "infer the grid from the TSV" heuristic is retired.

| Array | Map | Shape | Populated | Pitch | Patients |
|---|---|---|---|---|---|
| 128 Strip | Map 4 | 8×16 | 128 / 128 | 1.33 mm | S14, S16, S22 (RH), S23, S26 |
| 256 Grid | Map 3 | 46×24 | 256 / 1104 (I/cross) | 1.72 mm | S32, S33, S39, S58, S62 |
| 256 Hybrid Strip | Map 8 | TBD (macro + micro) | TBD | TBD | S57 |

Notes:
- **Phase-1 core patients** (S14, S26, S33, S62) use Map 3 or Map 4. S32 and S57 are excluded from Phase 1, so Map 8 and the S57 micro wiring are deferred with the patient.
- **S39 is 256 Grid Map 3.** The stray `S39_channelMap.mat` (8×16, byte-identical to the 128 Strip template) must not be loaded. The authoritative S39 map is `S39_channelMapAll.mat`.
- **S58** uses Map 3. Local `S58_channelMap.mat` is a compact (12, 24) crop of the central rows of the full (46, 24) Map 3, with values re-encoded as zero-indexed amp channels 0..255. The verifier must prove the row-slice alignment (blocker #12).
- **S62 `Duke Subjects` row is stale** — it says `Electrode Used? No`, but the recording is real and the `Speech task` sheet gives the right numbers.
- Map 4 is 0-indexed in the spreadsheet; the local `*_channelMap.mat` is 1-indexed. Same layout, +1 offset.
- All local `*_channelMapAll.mat` files (S32, S33, S39, S62) are byte-identical. Map 3 is a single generic layout, not per-patient.

The 1-to-1 amp-channel → physical-electrode → RAS bridge is a discussion item, not a written function (blocker #12).

## Significant Channels

`.fif` files contain ALL channels (not filtered). `sigChannel.mat` files identify task-responsive channels via permutation cluster test (upstream). Available for 9/11 patients (missing S32, S57).

| Patient | Sig ch | Total | % sig |
|---------|--------|-------|-------|
| S14 | 111 | 128 | 87% |
| S16 | 65 | 128 | 51% |
| S22 | 74 | 128 | 58% |
| S23 | 63 | 128 | 49% |
| S26 | 111 | 128 | 87% |
| S32 | ? | 256 | ? |
| S33 | 149 | 256 | 58% |
| S39 | 144 | 256 | 56% |
| S57 | ? | 256 | ? |
| S58 | 171 | 256 | 67% |
| S62 | 201 | 256 | 78% |

Sig-channel filtering did not improve S14 (85% sig) in the Conv2d baseline. The v14 channel-inclusion policy is still unresolved — treat `all non-artifact` vs `sig-only` as a blocker.

## Artifact Channels (electronic, not brain signal)

Some channels show extreme activations (>10 std in >5% of trials) — electronic artifacts from mic feedback / amp saturation, confirmed by Zac. **Exclude entirely** (clipping leaves confounded signal). The legacy `detect_artifact_channels()` zeroed in place to preserve `(H,W)` — wrong for v14, since zeroed rows inflate parcel-support denominators. v14 must drop channels from the signal tensor and the coordinate tensor together. Discussion item.

| Patient | Chronic artifact ch | Max value (std) |
|---------|-------------------|-----------------|
| S14 | 0 | 43 |
| S26 | 4 | 15 |
| S39 | **20** | **627** |
| S57 | **15** | 83 |
| S58 | **37** | 149 |

S39/S57/S58 are the worst. S14/S16/S23/S32 are clean (0 chronic).

## Inter-Patient Spatial Mismatch

Arrays are placed by surgeon, not standardized. Key fact: **no shared channel-index space across patients**, only partial anatomical overlap in where arrays land.

An older MNI overlap analysis predates the ACPC→MNI re-flag — those numbers are no longer trustworthy. v14 solves this not by electrode matching but by mapping each patient into shared Brainnetome parcel/subparcel space (after coordinates are verified).

## Brainnetome Core Parcels (provisional 16-parcel candidate list)

Top 16 LH ROIs by patient reachability, from a systematic check of all 123 LH Brainnetome ROIs + speech-relevant candidates (2026-04-06). Top 15 have ≥4 patients; #16 (A2) has 3.

- Motor (3): A6cvl (ventral PMC, 9pts), A4tl (tongue M1, 7pts), A4hf (face M1, 6pts)
- Sensory (3): A1/2/3tonIa (tongue S1, 8pts), A1/2/3ulhf (face S1, 5pts), A2 (proprioceptive, 3pts)
- Broca's (6): A44d (8pts), A45c (8pts), A44v (7pts), A45i (7pts), A45r (5pts), A44op (4pts)
- Auditory (2): STGpp (planum polare, 4pts), STGa (anterior STG, 4pts)
- Insula (1): INSa (articulatory planning, 4pts)
- Executive (1): MFG (dorsolateral PFC, 6pts)

Lives as `DEFAULT_BASE_PARCELS` in `src/speech_decoding/v14/token_spec.py`. **Provisional** — pre-dates Phase 1, must be re-discussed before it locks. Same for the 2-token splits in `DEFAULT_SPLIT_COUNTS` (`A6cvl`, `A4hf`, `A1/2/3ulhf`, `A2`, `A1/2/3tonIa`) that give `N_tok = 21` — an open blocker in `docs/implementation_tasks.md`.

Older centroid-VE logic (reachability thresholds, distance-to-ROI routing, 25/15 mm thresholds) is quarantined under `archive/legacy/data/atlas.py`. v14 uses volumetric Brainnetome PM membership, not centroid routing.

## Raw Continuous Recordings (for SSL)

456 min across 29 patients (13 PS + 17 Lexical, zero patient overlap). Raw 2kHz EDF files in BIDS: `sub-{id}/ieeg/sub-{id}_task-{task}_acq-01_run-01_ieeg.edf`. Need HGA extraction (CAR → 70-150Hz filterbank → Hilbert → 200Hz) to match existing `productionZscore` features. PS: ~199 min, Lexical: ~257 min. S14 longest at 31 min.
