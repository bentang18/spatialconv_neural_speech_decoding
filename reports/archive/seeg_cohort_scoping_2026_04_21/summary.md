# sEEG D-cohort Stage-3 scoping audit (2026-04-21)

Self-audit while lex uECoG data-unblock is pending. Goal: what we can answer about the Cogan lab sEEG D-cohort ourselves (Box + DCC, read-only) without waiting on Zac or Nanlin.

## Key findings

**All 4 speech tasks have complete BIDS + full FreeSurfer recons on Box.** Unlike lex uECoG, recons are not the Stage-3 blocker.

- `Box/ECoG_Recon/D<N>` (no zero-pad; BIDS `sub-D0023` → recon `D23`).
- 129 total D-recons on Box; 85 unique D-patients appear in ≥1 of the 4 speech-task BIDS roots.
- RAS file format (`D<N>_elec_locations_RAS_brainshifted.txt`) is identical to S-patient uECoG recons. `fsaverage_projection.py` transfers unchanged.

**Task identity: identical to our uECoG PS.** `BIDS-1.4_Phoneme_sequencing` events.tsv uses the exact same 52 CVC/VCV tokens (`uga.wav`, `ugae.wav`, `ipu.wav`, `ava.wav`, `aega.wav`, `avae.wav`, `ukae.wav`, ...) — same 9-phoneme label map, no remap needed.

**BNA atlas lookups pre-baked per D-recon.** `D<N>_elec_location_radius_{1,2,3,5,7,10}mm_aparc.BN_atlas+aseg.mgz.csv`. Patient-space BNA parcellation via FreeSurfer, probabilistic (sphere voxel-count proportions). Same parcel-naming convention as our Tier-1 (`A4hf_L`, `A1/2/3ulhf_R`, ...). Different implementation from our fsaverage-vertex support — would need parity check on uECoG S-patients before loader adoption, but sound as a coverage oracle for scoping.

**Derivatives are trial-level multi-band, not phoneme-level.** PS sEEG has `epoch(CAR)/sub-D*/epoch(band)(power)/*_desc-{baseline,perception,production}_{theta,alpha,beta,gamma,highgamma,low}.fif`. No `epoch(phonemeLevel)(CAR)` — Stage-3 would need to epoch to phoneme from production.fif using MFA alignments.

## Projectable cohort × Tier-1 coverage (3mm radius argmax)

| Task | Projectable | LH Tier-1 ≥10 | RH Tier-1 ≥10 | Either ≥10 | Zero Tier-1 |
|---|---:|---:|---:|---:|---:|
| Phoneme_sequencing | 50 | **17** | 9 | 25 | 3 |
| LexicalDecRepDelay | 47 | 13 | 11 | 24 | 4 |
| LexicalDecRepNoDelay | 24 | 7 | 6 | 11 | 3 |
| SentenceRep | 34 | 7 | 2 | 9 | 4 |

"Projectable" = has `lh.pial` + `lh.sphere.reg` + an `elec_locations_RAS*` file on Box.

**PS ∩ LexDelay = 27 patients** (can cross-task train on subset); PS ∪ all 4 tasks = 85 unique.

**3 PS D-patients with zero Tier-1 coverage** (non-speech depth placements): D66, D71, D100. Filter from supervised Stage-3 cohort; retain for continuous-corpus SSL.

## Top PS sEEG contributors

**LH-dominant (Stage-3 direct extension of our 7-LH uECoG cohort):**

| Rank | D-id | BIDS | N_elec | LH Tier-1 |
|---|---|---|---:|---:|
| 1 | D40 | D0040 | 188 | 42 |
| 2 | D73 | D0073 | 200 | 39 |
| 3 | D96 | D0096 | 234 | 25 |
| 4 | D79 | D0079 | 256 | 23 |
| 5 | D93 | D0093 | 210 | 22 |

**RH-dominant:**

| Rank | D-id | BIDS | N_elec | RH Tier-1 |
|---|---|---|---:|---:|
| 1 | D75 | D0075 | 215 | 28 |
| 2 | D54 | D0054 | 200 | 19 |
| 3 | D23 | D0023 | 121 | 16 |
| 4 | D45 | D0045 | 202 | 16 |
| 5 | D84 | D0084 | 230 | 16 |

**Hemisphere distribution across PS ≥5 Tier-1 patients:** 22 LH-only, 13 RH-only, 6 bilateral. sEEG implantations are one-sided; extending Tier-1 to RH roughly doubles the usable cohort.

## Implications for Stage 3

1. **Direct LH extension path**: 17 PS sEEG LH D-patients + 7 uECoG LH S-patients = **24 LH patients on the same 9-phoneme task**, ~3.4× our current Stage-2 cohort ceiling. No label remap. The cross-sensor transfer test is a first-class Stage-3 hypothesis.

2. **RH extension is mechanical**: Brainnetome is bilateral; the 15-parcel Tier-1 has natural RH twins. `P_emb` expands from 15 → 30. Unlocks 9 additional PS RH D-patients (≥10 Tier-1 RH).

3. **Patchy coverage shape**: sEEG patients average 10-42 Tier-1 electrodes from scattered depth contacts vs uECoG's 100-200 densely tiled. Soft parcel embedding handles this (absent parcels contribute zero), but per-parcel support is sparser and noisier — cross-sensor LOPO is genuinely harder than within-uECoG LOPO.

4. **Cross-task extension is available for free**: 27-patient PS ∩ LexDelay overlap lets us cross-train on identical 9-phoneme labels (PS) + 28-ARPABET (lex) on the same patients. Strong test of our Stage-2 28-ARPABET joint head.

5. **Zero-coverage D-patients (3/50 PS)** are SSL-only; filter from supervised sets.

## Open questions (still self-answerable; not done)

- **Sig-channel fraction per D-patient**: speech-vs-baseline HG t-tests on `epoch(CAR)/*desc-production_highgamma.fif` vs `*desc-baseline_highgamma.fif`. Or hunt for pre-computed stats in `derivatives/stats/` / `statistics/`.
- **Parity check: 3mm patient-space BNA CSV vs our fsaverage-vertex support** on uECoG S-patients. If they agree, we can skip re-running `fsaverage_projection.py` on every D-patient and use the pre-baked CSVs.
- **MFA alignment availability for sEEG PS**: same production-phoneme alignments we use for uECoG PS, or do we need to re-run MFA on the sEEG audio files?

## Open questions (need Nanlin)

- Whether Nanlin's downstream pipeline writes derivatives elsewhere (outside BIDS root, on a lab server).
- Which of these 50 PS D-patients she considers usable (her version of Zac's "good/decent" tiering for uECoG lex).
- sEEG preprocessing conventions: impedance exclusion, CAR scope, z-score recipe.

## Artifacts

- `coverage.csv` — per-patient (task, D-id, N_elec, LH_tier1, RH_tier1, status) for all 155 BIDS D-entries.
- `scripts/audit_seeg_cohort_coverage.py` — re-runnable audit source.
