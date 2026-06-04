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

The 1-to-1 amp-channel → physical-electrode → RAS bridge is frozen per blocker `#12` (see `implementation_tasks.md`). A 1-to-1 verifier is on the implementation checklist; the bridge itself is agreed.

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

Sig-channel filtering did not improve S14 (85% sig) in the Conv2d baseline. The v14 channel-inclusion policy is frozen per `#11`: **all non-artifact channels**, with sig-channel masks reserved as ablation-only metadata.

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

## Brainnetome Tier-1 Parcels (frozen, `N_tok = 15`)

Selection rule frozen 2026-04-14 per `implementation_tasks.md` `#4`: `argmax_wins >= 10` (a parcel is Tier-1 iff at least 10 Phase-1 LH electrodes have it as their globally dominant BNA assignment). Uniform `k_parcel = 1` — no 2-token splits in Phase 1. Canonical list with BNA indices is `DEFAULT_BASE_PARCELS` in `src/speech_decoding/v14/token_spec.py`.

Re-derived 2026-04-16 late on the fsaverage bake (`data/atlas/fsaverage_parity/fsaverage_cohort_ranking.csv`) over 1280 Phase-1 LH electrodes (S14, S16, S23, S26, S33, S39, S62), ordered by `argmax_wins` descending then `token_support` descending:

| Rank | BNA idx | Parcel | argmax_wins | token_support | Patients | Notes |
|---|---|---|---|---|---|---|
| 1 | 53 | A4hf | 307 | 178.18 | 7 | face/head M1 |
| 2 | 155 | A1/2/3ulhf | 246 | 169.47 | 6 | face/head S1 |
| 3 | 157 | A1/2/3tonIa | 151 | 103.42 | 6 | tongue S1 |
| 4 | 61 | A4tl | 110 | 90.00 | 7 | tongue M1 |
| 5 | 17 | IFJ | 89 | 49.16 | 4 | inferior frontal junction |
| 6 | 29 | A44d | 65 | 45.71 | 5 | Broca dorsal |
| 7 | 33 | A45c | 60 | 55.19 | 5 | Broca caudal |
| 8 | 63 | A6cvl | 59 | 67.74 | 7 | ventral premotor |
| 9 | 21 | A9/46v | 38 | 24.37 | 4 | ventral 9/46 |
| 10 | 139 | A40rd | 33 | 26.69 | 5 | supramarginal rostrodorsal (PFt) |
| 11 | 31 | IFS | 31 | 42.75 | 5 | inferior frontal sulcus |
| 12 | 159 | A2 | 26 | 56.95 | 6 | area 2 (dorsal S1) |
| 13 | 145 | A40rv | 24 | 24.80 | 5 | supramarginal rostroventral (PFop) |
| 14 | 39 | A44v | 18 | 24.32 | 5 | Broca ventral |
| 15 | 79 | A22r | 17 | 8.17 | 3 | rostral area 22 |

Cutoff is robust: the `argmax_wins ∈ [5, 9]` band is empty on current data, so any threshold in `[5, 10]` gives the same 15-parcel list. 13 of 15 parcels overlap with the cras-corrected cvs_avg35 derivation (87%), confirming cross-registration stability. The motor/somatosensory speech effectors (A4hf, A1/2/3ulhf, A1/2/3tonIa, A4tl) dominate, consistent with intra-op sensorimotor coverage; classical Broca subdivisions (A44d, A45c, A44v) and rostral STG (A22r) are present.

Older centroid-VE logic (reachability thresholds, distance-to-ROI routing, 25/15 mm thresholds) is quarantined under `archive/legacy/data/atlas.py`. v14 uses baked fsaverage surface BNA vectors per `#36`, not centroid routing.

### Per-electrode BNA support radius — canonical = 3 mm (first-principles)

`ECoG_Recon/D<N>/elec_recon/` ships pre-baked BNA CSVs at 1/2/3/5/7/10 mm. We use **3 mm** for both uECoG and D-cohort sEEG, derived from three independent constraints that converge on this value:

1. **HG sensitivity volume of a depth contact.** DIXI Microdeep contact (0.8 mm × 2 mm). Dubey & Ray 2019 (J Neurosci) measured 50%-power half-width ~1–2 mm for HG on depth contacts; Lachaux 2003/2012 estimated effective HG sampling radius ~3 mm. → r_phys ≈ **2–3 mm**.
2. **Localization uncertainty in patient-space recon.** FreeSurfer cortical surface sub-mm + post-implant brain-shift correction 1–2 mm + atlas-snap quantization 0.5–1 mm. → r_loc ≈ **1–2 mm**.
3. **Inter-contact spacing constraint.** DIXI standard 3.5 mm (verified 3.3–3.5 mm median in the 2026-04-24 A6 sensor-geometry audit). r > 3.5 mm starts attributing parcels to a contact that physically belong to its neighbor. → soft ceiling **r < 3.5 mm**.

Convolution of (1) and (2): √(r_phys² + r_loc²) ≈ √(2.5² + 1.5²) ≈ 2.9 mm. Combined with the 3.5 mm ceiling from (3), the goldilocks pick is **3 mm**.

**Empirical confirmation**: 2026-04-21 BNA parity audit at 3 mm gave 74% Tier-1 argmax coverage on D-cohort — argmax is meaningful and not noise-dominated. Smaller radii (1 mm) produce many all-zero supports (sphere lands in WM/CSF); larger (5+ mm) saturate adjacent contacts to redundant supports.

**Encoded in**:
- `scripts/v14_core/build_dpatient_support_cache.py` (D-cohort builder)
- `scripts/bna_parity_dpatients.py` (parity audit)
- `scripts/audit_seeg_cohort_coverage.py` (cohort coverage)
- `scripts/build_euclidean_support_cache.py` (uECoG builder)

Re-derivation justifies the convention without needing Nanlin's input — the radius is fully determined by HG biophysics + recon-pipeline noise + DIXI geometry, all independently measurable.

## Corpus Scale (Phase 1.5 data audit, 2026-04-18)

Numbers below were audited from local BIDS on 2026-04-18 (from each `_ieeg.json`'s `RecordingDuration` and events.tsv). Supersedes the prior "~24 h" and "456 min / 29 patients" figures, which referenced a different cross-cohort estimate that included sEEG.

| Corpus | Hours | Patients | Per-patient range | Source |
|---|---|---|---|---|
| PS intra-op uECoG | **2.83 h** (170 min) | 11 | 3.7–31 min (median ~15 min) | `ps_bids_root/sub-<pt>/ieeg/..._ieeg.json` |
| Phase-1 LH cohort | **1.89 h** (114 min) | 7 (S14, S16, S23, S26, S33, S39, S62) | same | same |
| Lexical intra-op uECoG (raw) | **3.96 h** (238 min) | 16 (all 16 readable; S78/S81 use older BIDS filename format) | 6.5–33 min (median ~11 min) | `lex_bids_root/sub-<pt>/ieeg/..._ieeg.json` |
| Lexical, fsaverage-projectable **today** | **0.74 h** (44 min) | 3 (S76, S78, S81) | 10.6–13.2 min | Blocked by missing FreeSurfer recons (see below). |
| **Combined raw (zero patient overlap)** | **6.79 h** | **27 unique** | — | — |
| **Combined fsaverage-projectable today** | **3.57 h** | **14 (11 PS + 3 lex)** | — | Lexical corpus gated by recon availability. |

PS per-patient durations:

```
S14  1861 s (31.0 min)   S32   945 s (15.8 min)
S16  1440 s (24.0 min)   S33   220 s ( 3.7 min)  [outlier low]
S22   804 s (13.4 min)   S39   730 s (12.2 min)
S23   810 s (13.5 min)   S57   632 s (10.5 min)
S26   710 s (11.8 min)   S58  1021 s (17.0 min)
                         S62  1028 s (17.1 min)
```

**Continuous / epoched ratio ≈ 1.04× median** (range 0.85× S33 → 2.43× S14). The response-locked 2.5 s trial windows already cover nearly all recording — "continuous SSL beyond epoched" is marginal for most patients. SSL at this scale is a representation-structure bet, not a data-expansion play; the real expansion lever for Phase 1.5 is the lexical corpus below.

## sEEG D-cohort Corpus (4-speech-task subset; 2026-04-24, re-derived 2026-06-03)

Audited via `scripts/seeg_corpus_audit.py` (JSON-first: `RecordingDuration` read from each `*_ieeg.json` sidecar; all 674 EDFs resolved from sidecar, zero EDF-header fallbacks, zero failures). Source: `reports/seeg_corpus_audit_2026_04_24/{per_run,per_patient,summary}.{csv,md}`.

| Task | Hours | D-patients | BIDS root (Box) |
|---|---:|---:|---|
| Phoneme sequencing (PS)       | **40.73 h** | 50 | `BIDS-1.4_Phoneme_sequencing/BIDS/sub-D*` |
| Lexical decision + delay      | **65.87 h** | 52 | `BIDS-1.0_LexicalDecRepDelay/BIDS/sub-D*` |
| Lexical decision, no delay    | **31.68 h** | 26 | `BIDS-1.0_LexicalDecRepNoDelay/BIDS/sub-D*` |
| Sentence repetition           | **42.31 h** | 34 | `BIDS-1.4_SentenceRep/BIDS/sub-D*` |
| **Grand total (4-task union)** | **180.59 h** | **87 unique** | — |

**Ratio to uECoG corpus**: 180.59 h / 6.79 h ≈ **26.6× more raw data**, **87 / 27 ≈ 3.2× more patients**.

> **Re-derived 2026-06-03** (`memory/project_d_cohort_data_inventory_2026_06_03.md`): the 87 / 180.59 h above is the **4-speech-task subset**, NOT the ceiling. Across all **14 BIDS paradigms** the union is **113 D-subjects / 384.7 h**; **134 distinct patients have FS recons** (max D146). Corrected facts: native sample rate is **mixed, predominantly 2048 Hz** (574 runs / 147 h; also 2000/1024/1000 — not a flat 2000); **recorded** EDF channels max **251** (the manifest's "366" = recon-localized contacts, not recorded); **DK is directly in the recons** (`aparc+aseg`, plus Destrieux `a2009s` + BNA at radii 1–10 mm), so no BNA→DK derivation needed; `iEEGReference` is `n/a` in every sidecar. Each task run is continuous-within-block (tileable for SSL); whole-session raw (`Natus/`) exists for only ~8 subjects (D108–D116, unconverted `.erd`).

Top-10 by total hours: D22 (6.09 h, PS+SR), D24 (5.75 h, all 4), D71 (5.05 h, all 4), D29 (4.96 h, all 4), D57 (4.62 h, all 4), D23 (4.56 h, PS+Lex+SR), D79 (4.05 h, PS+LexDelay), D59 (4.00 h, PS+LexDelay+SR), D53 (3.98 h, all 4), D28 (3.91 h, all 4). Five of the top-10 have coverage across all four speech tasks.

DCC presence (`reports/dcc_sync_plan_2026_04_24/`): PS 50/50 ✓, SentenceRep 34/34 ✓, LexDelay 21/52 (31 Box-only), LexNoDelay 0/26 (entire task Box-only). Of the 180.59 h, 122.24 h (PS+SR+partial-LexDelay) is DCC-accessible today; 58.35 h needs rsync from Box first. Per-patient join with support cache / sig-channel / z-score verdicts in `data/dcohort_manifest.csv` (B3).

## Box Mount Audit Pattern (lesson from A3, 2026-04-24)

The macOS Box mount (`/Users/bentang/Library/CloudStorage/Box-Box/`) is **latency-bound, not bandwidth-bound**. Every file open hits a remote stat; concurrent opens are partially serialized inside the Finder integration. Patterns that look fast on a local FS hang indefinitely on Box.

**A3's first version**: sequential `mne.io.read_raw_edf(path, preload=False)` over 674 BIDS `*_ieeg.edf` files (≥1 GB each). `preload=False` still reads enough header bytes for MNE to populate `n_times` and `info`, and Python's `Path.glob()` serializes those opens. Result: 0% CPU, no output for 22+ minutes, never completed.

**The fix that worked**: read the BIDS JSON sidecar instead. Every `*_ieeg.edf` has a co-located `*_ieeg.json` that already carries `SamplingFrequency`, `RecordingDuration`, and `SEEGChannelCount` (~250 bytes vs. 1 GB). Combined with a 16-thread `ThreadPoolExecutor`, all 674 runs probed in **1m54s, zero EDF-header fallbacks, zero failures**.

### Audit pattern: prefer sidecar metadata, parallelize aggressively, skip Finder

When auditing anything on the Box mount, in this order:

1. **Read BIDS sidecars (`*.json`, `*.tsv`) first.** They carry duration, sampling frequency, channel counts, event timing — most of what an audit needs. Open them via plain `Path.read_text()` + `json.loads`, not via MNE/pandas helpers that may also touch the data file.
2. **If you must touch the data file, parallelize.** Box latency per request is 0.5–2 s; throughput dies under serialization but scales linearly with `ThreadPoolExecutor(max_workers=16)`. This is I/O-bound work, GIL is irrelevant.
3. **Don't use `mne.io.read_raw_edf` to count samples.** Use the JSON `RecordingDuration` × `SamplingFrequency` instead. Even with `preload=False`, MNE opens the EDF.
4. **Don't use shell `find` over Box for anything but path enumeration.** `find -name '*_ieeg.edf'` works for lists but `find -size` triggers stats per file and stalls.
5. **Warm the cache before timing.** A first `ls` on a cold dir can take 10× longer than the second. If a script seems hung, check `lsof -p <pid> | grep Box-Box` to see whether it's progressing on different files (still working) or stuck on the same file (truly hung).
6. **All training reads happen on DCC, never via Box mount.** Box is for recon files, sidecar audits, and one-off coord/atlas extraction — not for `.fif` loads in a training loop.

### When the sidecar is missing

Fall back to the data-file header read, but in parallel and as a tagged source so the report shows it. `scripts/seeg_corpus_audit.py:_process_one_edf` is the reference: returns `source: "json" | "edf" | "error"` per row, and the summary surfaces fallback counts. If `source=edf` shows up in any future audit, that's a flag that the BIDS dataset is missing sidecars for those runs — fix upstream rather than perpetuating the slow path.

## Lexical Corpus

**Path**: `configs/paths.yaml` → `lex_bids_root` → `/Users/bentang/Documents/Code/speech/BIDS_1.0_Lexical_µECoG/BIDS_1.0_Lexical_µECoG/BIDS`.

**Cohort (16 patients, zero PS overlap)**: S41 S45 S47 S51 S53 S55 S56 S63 S67 S71 S73 S74 S75 S76 S78 S81.

Per-patient durations:
```
S41   651 s (10.8 min)   S53  2005 s (33.4 min)   S73   656 s (10.9 min)
S45  1040 s (17.3 min)   S55   872 s (14.5 min)   S74   391 s ( 6.5 min)
S47   951 s (15.9 min)   S56   774 s (12.9 min)   S75   702 s (11.7 min)
S51   660 s (11.0 min)   S63  1866 s (31.1 min)   S76   652 s (10.9 min)
                         S67   470 s ( 7.8 min)   S78   633 s (10.6 min)
                         S71  1149 s (19.2 min)   S81   795 s (13.2 min)
```

**Task structure (most patients)**: stimulus + response event pairs; 8 trial_types = `{stimulus, response} × {word, nonword} × {low, high frequency}`. ~280 events/patient, ~140 trials. Trial counts vary: S47 is 211 (outlier high), S67 and S74 are ~70 (short session), others 120–144.

**Inventory**: 72 unique items, ~4 reps each.
- Words (examples): `hazel siren bagel merit minus`
- Nonwords (examples, phonotactically legal CVC-like): `vagul minel lomic gapel nomel`
- MFA-alignable: `value` column in `events.tsv` carries the spoken string.

**Events schema variants (audited 2026-04-18)**:
- Standard (13/16 patients): `subject trial onset duration value trial_type sample`; filename `sub-<pt>_task-lexical_acq-01_run-01_events.tsv`.
- **S71** has BOTH tsv files, neither in the canonical merged shape (empirically re-verified 2026-04-20):
  - `sub-S71_task-lexical_acq-01_run-01_events.tsv`: 144 rows, 5 cols. `trial_type` holds the WORD ITSELF (`comet`, `tanic`, ...), not `stimulus`/`response`. `duration=0`. Onsets precede the other file's onsets by median 1.23 s (p5 1.01 s, p95 1.79 s) — consistent with stim→resp delay, so these rows are stim events but unlabeled as such.
  - `sub-S71_task-lexical_events.tsv`: 142 rows, 7 cols. `trial_type='response'` only. 71 trials × **2 words per trial** (= 142 responses).
  - **Task variant**: 2 words per trial suggests this session ran the Lexical Decision Repeat **Delay** protocol, different from the 1-word-per-trial structure every other lexical patient has.
  - **Canonical shape**: S78 (same older filename convention, `.fif` exists) has a single `_events.tsv` with merged `{stimulus, response} × trial` rows and explicit `trial_type` column. Regen for S71 likely requires labeling the word-keyed file as `stimulus`, concatenating with the response file, and possibly adapting MFA for 2-words-per-trial audio. Whether this alone recovers the `.fif` is unverified — could also be task variant, audio quality, or an unrelated upstream issue. Confirm with Zac.
- **S78, S81** use filename `sub-<pt>_task-lexical_events.tsv` (no `acq-01_run-01` entity); `trial_type` is simple `stimulus|response` without the `/word/low` suffix. **Derivatives `.fif` filenames are identical across all 16 patients** — the naming delta is in raw ieeg only, so our v14 loader needs no change.

**Phantom `sub-S52/` in `derivatives/events/`** (audited 2026-04-18): `sub-S52/phoneme/` and `sub-S52/word/` contain 340 real MFA phoneme events + 85 word events (`galef`, etc.). But no raw EDF / channels.tsv / electrodes.tsv exists anywhere in the BIDS root for S52. S52 is unusable until Zac provides the raw data (or confirms exclusion).

**Stimulus → response delay distribution (events.tsv, 14/16 patients)**:

| Pt | n | median (s) | p5 | p95 | Pt | n | median (s) | p5 | p95 |
|---|---|---|---|---|---|---|---|---|---|
| S41 | 131 | 1.243 | 0.873 | 1.973 | S67 | 70 | 1.276 | 0.811 | 2.382 |
| S45 | 139 | 1.164 | 0.966 | 1.558 | **S73** | 144 | **0.937** | 0.769 | 1.123 |
| S47 | 211 | 1.340 | 0.666 | 2.398 | S74 | 72 | 1.359 | 1.158 | 1.657 |
| S51 | 144 | 1.238 | 1.030 | 1.560 | S75 | 143 | 1.281 | 1.065 | 2.268 |
| S53 | 144 | 1.087 | 0.877 | 1.683 | S76 | 144 | 1.112 | 0.890 | 1.606 |
| S55 | 142 | 1.485 | 0.960 | 2.431 | S78 | 140 | 1.083 | 0.848 | 1.835 |
| S56 | 143 | 1.147 | 0.824 | 2.088 | S81 | 120 | 1.119 | 0.823 | 1.680 |
| S63 | 142 | 1.354 | 1.001 | 1.811 | S71 | — | broken | — | — |

Population mean ≈ 1.25 s (std ≈ 0.15 s across patients), consistent with PS cohort's 1.1 ± 0.3 s (Duraivel 2023). S73 is the fastest responder (0.937 s median, 0.769 s p5).

**Stim → phoneme-epoch overlap (Q12, 2026-04-18)**: the existing phoneme-level `.fif` carries `[-1.0, 1.495) s` around each phoneme onset (500 samples). For S73 (tightest case), stim duration is ~0.62 s; `stim_offset → response_onset` is 0.31 s median, 0.18 s p5. At the 500 ms pre-phoneme baseline window baked into the `.fif`, **93.8 % of first-phoneme epochs baseline-overlap with the stimulus presentation** for S73; 42.8 % across all phonemes. The v14 per-phoneme path uses a much tighter `[-0.15, 0.5)` s window, which leaves ≥29 ms margin for the median first-phoneme case on S73. Phase-1.5 should document this and keep tmin ≤ 0.15 s for first-phoneme training to minimize auditory-stim contamination; longer pre-onset windows require per-patient gating on `stim_offset → resp_onset`.

**Phoneme-level clock**: phoneme events in `.fif` are stored in the 2 kHz raw-sample clock (same as `events.tsv` `sample` column). Phoneme-0 of each trial locks to the response onset at raw-sample resolution across all patients (clock ratio 1.000, verified on 9 patients).

**Phoneme inventory — 28 ARPABET phonemes** (not 9 PS phonemes):
```
AA AE AH AO AY B D EH EY F G HH IH IY JH K L M N OW P R S T UH V W Z
```
- **PS ∩ Lexical = 8 phonemes**: AA AE B G IY K P V. PS-only: UW. Lexical-only: 20 phonemes (AH AO AY D EH EY F HH IH JH L M N OW R S T UH W Z).
- **S78 has 27 phonemes** (missing AE, UH; has UW — probably word-list coverage artifact; Monday Q).

**Preprocessing state**: same derivative tree as PS (`desc-{baseline,production,productionMeanSub,productionZscore}_highgamma.fif`). Phoneme-level `.fif` exists for **15/16 patients**: S41 S45 S47 S51 S53 S55 S56 S63 S67 S73 S74 S75 S76 S78 S81. S71 has no derivatives (see events-schema note below).

**FreeSurfer recon availability (2026-04-18, MAJOR Phase-1.5 blocker)**: the v14 fsaverage projection requires per-patient `lh.pial` + `lh.sphere.reg`. Checked on Box (`ECoG_Recon/`, `CoganLab/ECoG_Recon_Full/`, `CoganLab/ECoG_Task_Data/Intra Op Recon/`) and DCC (`/work/ht203/data/ECoG_Recon/`, `/hpc/group/coganlab/ECoG_Recon/`, `/hpc/group/coganlab/Data/ECoG_Recon_Full/`).

| Patient | FS recon on Box | FS recon on DCC | elec_recon/RAS | Notes |
|---|---|---|---|---|
| S41, S45, S47, S51, S53, S55, S56, S63, S67, S71, S73, S74, S75 | **no** | **no** | **no** | 13/16 lexical blocked |
| S76, S78, S81 | yes | no | yes | projectable |

The DCC `BIDS-1.0_LexicalDecRepDelay` tree is a different cohort (D-series). None of the 13 missing recons are in any location I can reach. **Raw MRI DICOMs** exist on Box at `CoganLab/MRI_IntraOp_Pro00072892/S{41,51,53,55,76,78,81}` for 7/16 patients (folders confirmed, contents time out over the Box mount — likely un-cached DICOMs). In principle we could `recon-all` 4 of these (S41/S51/S53/S55), but the S76/S78/S81 recons are already done in `ECoG_Recon/`, and 9/16 patients (S45, S47, S56, S63, S67, S71, S73, S74, S75) have **neither recon nor raw MRI** on my mount. Until Zac provides them, the fsaverage-projectable lexical corpus is **3 patients / 0.74 h**, not 16 / 3.96 h.

**Additional derivatives (lexical-only, not present in PS)**:
- `derivatives/decoding/sub-<pt>/` — Zac's pre-computed decoders, 33 files per patient across three families (audited 2026-04-18):
  - **`decode(production)(cca)(alignP1)(crossPatientTask)/`** — CCA-aligned cross-patient decoding, `tw[pre1,post1]` = 2 s window around production onset. Two files per patient:
    - PS-cohort aligned: `pts=[S14,S16,S22,S23,S26,S33,S39,S58,S62]` (9 pts, Spalding set + S26 − S57 − S36 + S58).
    - Within-lexical aligned: other 12 lexical patients.
    - Output shapes: `scores[10]` (10 CV folds), `y_preds[10, n_items]` where `n_items ≈ 130` (72 items × ~2 reps).
  - **`decode(production)(patientSpecific)/`** — per-patient, per-phoneme-position `_p1.._p5`, `tw[pre0.5,post0.5]` = 1 s around production, `scores[50]` (50 seeds), MeanSub × allChannel × sigChannel variants.
  - **`decode(production)(patientSpecific)(seq2seq)/`** — seq2seq variant.

  **Baseline accuracies (mean of scores, chance 1/28 = 0.036)** — sampled 3 patients:

  | Pt | PS-aligned CCA p1 | Lex-aligned CCA p1 | Patient-specific p1 / p2 / p3 / p4 |
  |---|---|---|---|
  | S41 | 0.130 | 0.100 | 0.046 / 0.076 / 0.076 / 0.194 |
  | **S73** | **0.351** | 0.325 | 0.248 / 0.224 / 0.208 / 0.242 |
  | S76 | 0.103 | 0.117 | 0.037 / 0.098 / 0.078 / 0.214 |

  **S73 is the lexical transfer anchor** — every head (patient-specific and cross-patient) is well above chance, and PS-aligned CCA (0.351) beats its own patient-specific (0.248). S41 and S76 are near-chance on patient-specific and should not be used as lone cross-patient evaluation anchors.
- `derivatives/phonemeLevel/sub-<pt>/ieeg/..._desc-phonemeLevel_ieeg.edf` — 651 MB CAR'd, resampled, continuous EDF (phoneme-level annotated) per patient — useful for SSL on continuous data.

**Implications for Phase 1.5**:
- Nonword branch is a direct structural extension of the PS task (CVC-style unfamiliar phoneme sequences) — cleanest cross-cohort generalization test for the PS architecture.
- Word branch extends the distribution to real English words.
- Phoneme inventory: ARPABET-28 (or 27 for S78). Joint PS+lexical training requires either a single 28-head output layer or a masked-softmax variant that emits PS's 9 vs lexical's 28. Filtering to the 8-phoneme intersection discards the majority of the signal.
- Zac's pre-computed cross-patient decoders (PS→lexical, within-lexical) establish that Duke already treats these cohorts as cross-task transferable at the feature level — Phase 1.5 extends this, doesn't invent it.

## Z-Score Recipe

**Empirically reverse-engineered 2026-04-18** from `baseline_highgamma.fif` + `productionMeanSub_highgamma.fif` + `productionZscore_highgamma.fif` across 7 patients (3 PS + 4 lexical). Recipe matches productionZscore with reconstruction correlation 1.0000 everywhere:

```
X_c = baseline_highgamma.mean(axis=(trials, time))    # per-channel pooled mean
Y_c = baseline_highgamma.std(axis=(trials, time))     # per-channel pooled std, ddof=1
productionZscore[trial, c, t] = (production[trial, c, t] - X_c) / Y_c
```

- **Aggregation**: per-channel, **pooled across ALL baseline trials AND all samples within the baseline window**. NOT per-epoch/per-trial (earlier writeup was wrong on this).
- **Baseline window**: pre-AUDITORY onset, 500 ms immediately preceding auditory stimulus (perception-locked), 100 samples at 200 Hz. `baseline_highgamma.fif` carries `tmin=-0.500, tmax=-0.005` with `n_epochs == n_trials`.
- **Statistic**: mean / std (ddof=1). `IEEG_Pipelines/ieeg/calc/scaling.py:rescale(mode='zscore')` applies the scaling; the pooling happens in the caller before `rescale` is invoked.

### Derivative files in `derivatives/epoch(CAR)/sub-<pt>/epoch(band)(power)/`

| File suffix | Content |
|---|---|
| `desc-baseline_highgamma.fif` | Pre-auditory baseline-window HGA, 500 ms, 100 samples, one per trial |
| `desc-perception_highgamma.fif` | Perception-locked HGA epochs |
| `desc-production_highgamma.fif` | Production-locked HGA epochs |
| `desc-productionMeanSub_highgamma.fif` | `production - X_c` (mean-subtracted only) |
| `desc-productionZscore_highgamma.fif` | `(production - X_c) / Y_c` — **active Phase-1 input** |

### Local data state (2026-04-18)
- **PS**: `baseline` + `productionMeanSub` + `productionZscore` have real data. `production_highgamma.fif` is all-NaN stub (reconstructible as `productionMeanSub + X_c`). `perception_highgamma.fif` is empty stub.
- **Lexical**: ALL five files have real data for 15/16 patients (only S71 is absent from `epoch(phonemeLevel)(CAR)` entirely).
- EDF raw 2 kHz broadband is present for all patients. `ieeg` Python package installs cleanly via `uv pip install ieeg`.

### Recording-level median/MAD (Phase 1.5 SSL candidate) ≡ recipe A up to per-channel affine

Tested on 7 patients (S14, S26, S62 phoneme-level PS; S41, S47, S55, S73 lexical):

| Patient | Cohort | ρ(z_A, z_B) median | scale_B/A p5..p95 | \|locΔ\|_p95 (z-A units) |
|---|---|---|---|---|
| S41 | lexical | **1.0000** | 0.82..1.24 | 0.45 |
| S47 | lexical | **1.0000** | 0.73..1.06 | 0.19 |
| S55 | lexical | **1.0000** | 0.87..1.04 | 0.23 |
| S73 | lexical | **1.0000** | 0.82..1.74 | 1.09 |
| S14 | PS | **1.0000** | 0.59..1.59 | 0.77 |
| S26 | PS | **1.0000** | 0.62..1.52 | 0.87 |
| S62 | PS | **1.0000** | 0.78..1.38 | 0.98 |

The two recipes differ only by a per-channel affine `(loc_c, scale_c)`. Class-separability η² (between / within-class variance ratio on phoneme classes in [0.05, 0.30) s post-onset) is **identical to 4 decimal places** across recipes on all patients — the affine is invariant to variance-ratio metrics.

**Implication**: a Conv1d / Conv2d / per-channel-linear first layer absorbs the per-channel affine into its weights. Phase 1.5 SSL can adopt recording-level median/MAD (or any per-channel affine normalization) without retraining Phase 1. Don't claim recording-level is a bit-exact drop-in — it is *affine-exact* per channel, not identical. Scripts: `scripts/zscore_recipe_comparison_{lexical,ps}.py`; reports under `reports/zscore_comparison_2026_04_18/` (gitignored).
