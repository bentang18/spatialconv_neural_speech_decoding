# Tier-2 D-cohort sEEG Integration into Neuroprobe Hillclimb

*Drafted 2026-04-25. Sibling to `plan.md` (Tier-2 of the pretraining corpus). Live until Tier 2 either submits or is abandoned.*

> **⚠️ SUPERSEDED 2026-06-03 — do not use for current integration.** This doc's core proposal (recipe (a): run the Cogan **HG @ 200 Hz** envelope recipe on raw, **BNA** atlas routing, "Tier" framing) predates the NeuroAI reset. v14's active front-end is **Multi-STFT on raw voltage** (not HG envelope) and the active atlas is **DK** (`aparc+aseg`, present directly in the recons — not BNA, which is now a gated P1 ablation). Several data facts here are also stale: native rate is **mixed, mostly 2048 Hz** (not a flat 2000 Hz EDF); the 180.59 h / 87 D-pts is the **4-speech-task subset** of a 113-pt / 384.7 h / 14-task inventory. Current state: D-cohort is BrainTreebank-shaped, loaders template off `studies/braintreebank/`. Canonical inventory: `memory/project_d_cohort_data_inventory_2026_06_03.md`; build plan: `docs/neuroprobe/b36_implementation_plan.md`. Kept for the pipeline-diff reasoning only.

This doc plans how to fold our 180.59 h Cogan sEEG D-cohort into the Neuroprobe hillclimb after Tier 1 (BrainTreebank-only) clears Linear (0.539). The hard part is not "more data" — it's making the two corpora live in the same input distribution. The sensors are identical (depth sEEG, ~3 mm DIXI inter-contact). The preprocessing pipelines are not.

The Tier-2 entry condition is unchanged from `plan.md`: only attempt this after Tier-1 SSL clears the Linear (Lap+spec) baseline of 0.539 cross-subject.

## TL;DR

- BrainTreebank ships **raw 2048 Hz voltage** in h5; all preprocessing happens at load time (eval_utils.py preprocessing chain).
- Cogan sEEG ships **z-scored 200 Hz high-gamma envelope** in `derivatives/.../desc-{baseline,production,perception}_highgamma.fif` — preprocessing is upstream, baked in.
- Two ways to align: (a) run our Cogan recipe on BT raw at load time, or (b) skip both and use BT-style raw + STFT for everything. We propose **(a)**: it preserves the v14 architectural commitment (HG @ 200 Hz, atlas-anchored parcel embeddings) and matches Stage-3's sEEG loader plan from the 2026-04-24 audit.
- Self-resolvable: most of the alignment work. Nanlin-blocked: 3 of the 5 items from the Stage-3 audit that also gate Tier 2.

## Pipeline diff — every difference, source-of-truth verified

All entries are anchored to a code path so we can rerun the diff if either side changes.

| Axis | Cogan sEEG D-cohort | BrainTreebank (Neuroprobe) | Notes |
|---|---|---|---|
| **Raw sample rate** | 2000 Hz EDF (per A3 sidecars: `RecordingDuration` × `SamplingFrequency`) | 2048 Hz h5 (`neuroprobe/config.py:23 SAMPLING_RATE`) | 2.4% offset; matters for filterbank passband edges and post-decimation alignment |
| **Storage form** | HG envelope, 200 Hz, z-scored (sEEG `production_highgamma.fif` direct; uECoG via `productionMeanSub` + `productionZscore`) | Raw voltage in microvolts (`f['data']['electrode_<i>']`) | Different surfaces; reconciliation has to choose one or compute the other on the fly |
| **Reference scheme** | CAR by default; Box has parallel `derivatives/statistics/sub-D*/{CAR,WM,M1,STG,HIPP,LING}/...` (Nanlin-blocked which is canonical) | None at storage; Laplacian re-reference at eval (`eval_utils.py:laplacian_rereference_neural_data` — subtracts mean of adjacent-numbered same-stem electrodes, e.g. `T1b2` ← `T1b1+T1b3`). Linear baseline winner uses Laplacian. | The two are **incompatible**. Cogan = global mean across kept channels; BT-Lap = local 2-neighbor average. Either re-reference Cogan as Laplacian or apply CAR to BT raw — see "Contract" below |
| **Line-noise notch** | Built into Cogan filter chain (60 Hz at the bandpass step is below the 70–150 Hz HG window, so notch is implicit) | Optional via `--preprocess.type remove_line_noise`: scipy `iirnotch` at 60/120/180/240/300 Hz, Q=30, applied bidirectionally with `filtfilt` (`eval_utils.py:140`) | Both end up clean above 70 Hz, so for HG-only output this difference is benign |
| **Bandpass + envelope** | 70–150 Hz Gaussian filterbank, 8 bands → Hilbert envelope per band → sum (`coganlab/IEEG_Pipelines`) | None at storage; STFT magnitude in baseline (`nperseg=512, poverlap=0.75, hann, 0–150 Hz`) at eval | This is the biggest divergence. STFT magnitude ≠ Hilbert sum-of-bands. Recreating Cogan HG from BT raw is straightforward; the inverse (recovering raw from Cogan HG) is impossible |
| **Decimation** | 2000 → 200 Hz post-Hilbert | None at storage; `--preprocess.type downsample_200` available via `scipy.signal.resample_poly` | Use BT's resample_poly to land at 200 Hz after our Hilbert step |
| **Z-score** | Per-channel mean/std pooled across all pre-auditory baseline trials × samples (500 ms perception-locked, 100 samples @ 200 Hz). Verified 2026-04-18 reconstruction ρ=1.0000 across 7 PS+lex pts | None at storage; sklearn `StandardScaler` over training-set features (post-flatten) at eval | BT has **no event-locked baseline window** in continuous movie data. Needs substitute — recording-level median/MAD is verified equivalent to Cogan recipe up to per-channel affine (ρ=1.0000), so use that on BT raw |
| **Bad-channel exclusion** | Impedance log10 > 6 + sig-channel mask (PS only); A1 audit found 40 ok / 1 low / 2 likely_bad on 43 D-pts with stats | Hard-coded `corrupted_elec.json` per subject + missing-coordinate exclusions in `_get_corrupted_electrodes` (`braintreebank_subject.py:69-73`) + `DC*`/`TRIG*` name filter | Both deterministic; no semantic conflict. Apply each side's filter independently before re-referencing |
| **Trial windowing** | Event-locked epochs in the `.fif` (`tmin=-1.0, tmax=1.495` s for phoneme-level lex; PS uses production-locked); response-onset clock | Word-onset locked, [0, 1] s for leaderboard (`bins_start_before_word_onset_seconds=0`, `bins_end_after_word_onset_seconds=1` ⇒ 2048 samples) | Window length compatible (1 s = 200 samples @ 200 Hz). **Semantic mismatch**: Cogan PS = production onset; BT = perception (word onset). For Tier-2 SSL pretrain we use continuous chunks anyway, so the event-lock difference disappears — only matters if we ever fine-tune jointly |
| **Coordinate frame** | Native patient space; pre-baked 3mm probabilistic BNA CSVs at `ECoG_Recon/<D>/elec_recon/D<N>_elec_location_radius_3mm_aparc.BN_atlas+aseg.mgz.csv` (parity-verified 2026-04-21) | `coordinates_type="cortical"` → `localization/elec_coords_full.csv` (XYZ, "STANDARDIZED BRAIN ATLAS CORTICAL PROJECTION", space unverified); `lpi` available; `mni` raises `NotImplementedError` | Stage-0 Block A (`docs/neuroprobe/stage_0.md`) is supposed to verify whether `cortical` is fsaverage. If yes, our existing v2c bake covers BT directly. If no, BT FreeSurfer recons need an analogous bake |
| **Atlas attached at runtime** | BNA probabilistic (Tier-1 15 LH or 30 bilateral); per-electrode soft support vector via `support_cache.py` | DK regions per electrode (`localization/sub_{id}/depth-wm.csv:DesikanKilliany`); cross-subject baseline mean-pools within DK region | Both side runs through our v14 BNA `P_emb` once we own the loader; DK on the BT side is only used by the Linear baseline we're trying to beat |
| **Continuous corpus available** | 180.59 h / 87 D-pts across 4 speech tasks (PS 40.73 h, LexDelay 65.87 h, LexNoDelay 31.68 h, SentenceRep 42.31 h) — A3 audit 2026-04-24 | ~25 h of legal pretrain (14 full + 5 partial sessions), ~33 h if we count nuanced shared-electrode subjects | D-cohort is 7× the legal BT pretrain. Tier 2 actually moves the needle if Tier 1 was data-starved |
| **DCC accessibility** | 122.24 h on DCC today; 58.35 h Box-only (LexDelay 31 D-pts missing + entire LexNoDelay) per A5 sync plan | All 12 Lite sessions + legal pretrain ~90 GB, downloadable via `braintreebank_download_extract.py --lite`. No Box dependency | BT lives wherever we put it on DCC; D-cohort needs the rsync recipe we already staged at `reports/dcc_sync_plan_2026_04_24/rsync_commands.sh` (Nanlin-blocked on direction) |

### What's the same

- **Sensor type**: depth sEEG, contact diameter ~0.8 mm, DIXI standard inter-contact ~3.3–3.5 mm (median in our A6 audit; BT uses similar Behnke-Fried/Ad-Tech depth probes per Wang 2024).
- **Band of interest**: high gamma 70–150 Hz. Both pipelines preserve it; only the envelope-extraction method differs.
- **Trial-window length on the eval side**: 1 s (= 2048 samples at 2048 Hz, = 200 samples at 200 Hz).
- **Atlas we attach**: BNA. DK on the BT side is the baseline's anchor, not ours.
- **Linear baseline mechanism cross-subject**: region-averaging via DK `combine_regions()` — our v14 BNA-`P_emb` thesis applies identically to both corpora.

## Reconciliation contract — pick a target representation, both sides land there

**Target**: `(N_e, T=200) float32`, 70–150 Hz Hilbert HG envelope, 200 Hz, per-channel z-scored, paired with a Tier-1 BNA support vector `(N_e, 15-or-30) float32`, an `electrode_active_mask (N_e,) bool`, and an `electrode_coordinates (N_e, 3) float32` in fsaverage RAS.

This is exactly what v14's `phoneme_dataset.py` already emits for uECoG, minus the grid-scatter (sEEG depth probes have no 2D grid — Stage-3 architecture decision pending in the prep memo). For Tier-2 SSL pretrain on continuous chunks the grid-scatter is irrelevant; for fine-tune on Neuroprobe Cross-Subject we use the per-electrode-token path (B-1 mode, scoped in `plan.md` Open Question #3).

**Cogan side** (D-cohort): already in this representation. Loader reads `desc-production_highgamma.fif` directly (sEEG ships z-scored, verified in A4). For SSL pretrain on continuous (non-event-locked) data we'd need to either MFA-epoch from production (Nanlin-blocked) or run the Cogan filterbank+Hilbert+z-score chain on raw EDFs ourselves.

**BrainTreebank side**: not in this representation; we have to compute it at load time. The chain:

```
raw 2048 Hz voltage
  → optional 60 Hz comb notch (eval_utils.remove_line_noise — port verbatim)
  → CAR re-reference (mean across kept BT-Lite electrodes; matches Cogan default; differs from BT-Lap baseline)
  → 70–150 Hz Gaussian filterbank, 8 bands, Hilbert envelope, sum
  → resample_poly 2048 → 200 Hz
  → recording-level median/MAD z-score per channel
```

The whole chain is straightforward Python. Existing parts we can reuse:
- `coganlab/IEEG_Pipelines` filterbank — port the Gaussian filter spec (centers + widths) into `src/speech_decoding/neuroprobe/preprocess.py`. Verify on a single channel against an MNE-equivalent path.
- Recording-level median/MAD z-score: drop-in. The 2026-04-18 audit already proved median/MAD ≡ Cogan baseline-pooled mean/std up to per-channel affine, ρ=1.0000.

**The one open architectural call**: CAR vs Laplacian on the BT side. CAR keeps Cogan parity; Laplacian matches the winning baseline. Stage-1 cold-start ablation. This is the same fork as `plan.md` "Tactical note" — staying HG-only is cleaner; we can run both and pick by AUROC.

## Self-resolvable work order

Sequenced; each step verifiable on disk before moving on. Targets after Tier 1 lands.

1. **Lock the BT preprocessing chain**: `src/speech_decoding/neuroprobe/preprocess.py`. Port `eval_utils.preprocess_data` shape, but emit our HG-200-Hz target instead of STFT magnitude. Output dtype + shape match `phoneme_dataset.py` per-electrode contract. Verify on `btbank1_0` (legal pretrain): n_samples × n_electrodes consistent, z-scored marginals (per-channel std 0.55–0.95 like our D40/D73/D96 verdict).
2. **BT coordinate space verification**: Stage-0 Block A is already meant to do this (`docs/neuroprobe/stage_0.md`). For Tier 2, the answer is the same — if `cortical` is fsaverage, BT and D-cohort share the same BNA bake; if not, we need a separate BT BNA bake. Do once for both programs.
3. **D-cohort continuous loader** (Tier-2-specific): `src/speech_decoding/neuroprobe/dcohort_loader.py`. Two modes:
    - `from_fif`: reads task-locked `desc-production_highgamma.fif` per D-pt. For SSL pretrain we'd need stitched continuous chunks, which means concatenating epochs — semantically dirty. Avoid for SSL.
    - `from_raw_edf`: reads raw EDFs (~180.59 h corpus), runs the same chain as BT. This is the right SSL path. The chain is already shared with BT, so once the Cogan-recipe-on-raw works for BT it works for D-cohort by configuration alone.
4. **Cohort cocktail manifest**: `data/neuroprobe/tier2_corpus_manifest.csv`. Joins `data/dcohort_manifest.csv` (122 ready D-pts) with the BT legal-pretrain table. Columns: `cohort, patient, n_electrodes_used, hours_raw, source_file, ras_csv, support_csv, status`. Idempotent rebuild from `prepare_new_dpatient.py` analog plus a BT enumerator.
5. **Bilateral Tier-1 BNA expansion**: `docs/strategy/stage_3_rh_expansion.md` is the design stub. Tier 2 needs LH+RH (BT subjects are bilateral; D-cohort is bilateral). Execute the 6-step change order from that doc on a Tier-2-only branch — `token_spec.DEFAULT_BASE_PARCELS` 15 → ~30, regenerate caches with the new `TIER1_COLUMNS`, update `phoneme_dataset.py:N_TIER1_PARCELS`. Hold this branch off main until Tier 2 commits to a submission.
6. **Stage-3 RH expansion docs-only stub already covers cache regen**: support cache rebuild for D-cohort via `scripts/v14_core/build_dpatient_support_cache.py` with bilateral parcel list. BT side will need a parallel support cache builder that reads `localization/elec_coords_full.csv` + projects to fsaverage + does the BNA bake lookup. New script: `scripts/v14_core/build_btbank_support_cache.py`.
7. **DCC stage**: rsync 58.35 h Box-only D-cohort data per `reports/dcc_sync_plan_2026_04_24/rsync_commands.sh` — but **only after Nanlin confirms direction** (item #5 in her blocker list; she may have a canonical location at `/hpc/group/coganlab/nanlinshi/` we shouldn't overwrite).
8. **Pretrain run**: extend `scripts/neuroprobe/stage2_ssl_pretrain.sh` to take a `--corpus tier1|tier2` flag. Tier-2 invocation reads the joined manifest, samples patient-balanced batches from BT + D-cohort. SSL objective is unchanged from `plan.md` Experiment #5 (parcel-space reconstruction at masked time steps).
9. **Fine-tune + eval**: unchanged from Stage 2 in `plan.md`. The fine-tune set is still S2/trial-4 (Cross-Subject train); Tier-2 only changes what the backbone was pretrained on.
10. **Submission gate**: same as `plan.md` — ≥ 0.56 to submit, ≥ 0.58 stretch, < Tier-1's number = abandon Tier 2 and submit Tier 1 instead. Don't dilute with both.

Verification artifacts at each step:
- (1) marginal-stats JSON per BT subject; per-channel std in [0.5, 1.5] for ≥95% of channels
- (3) random-window spot check: load 1 s from D40 raw EDF + Cogan FIF at the same trial timestamp, correlate; should match up to z-score affine (ρ ≥ 0.99)
- (4) idempotence: rerun manifest builder, byte-identical
- (8) loss-curve sanity check: Tier-2 SSL loss strictly below Tier-1's at matched compute

## Nanlin asks (two) and our calls (the rest)

Trimmed 2026-04-25. Most of the original blocker list is actually our call to make — kept here only what genuinely depends on Nanlin's pipeline knowledge.

**Ask Nanlin:**

1. **Laplacian / bipolar reference variant?** Box ships CAR / WM / M1 / STG / HIPP / LING stats — all global or anatomical-region references. BrainTreebank's winning baseline (0.539 cross-subject) uses a Laplacian re-reference: subtract the mean of adjacent same-stem depth contacts (e.g. `T1b2` ← `(T1b1 + T1b3) / 2`). Ask whether the Cogan pipeline has a Laplacian/bipolar option we missed, or which of the six existing references is closest to local-bipolar style. If no clear match, default to CAR. **Direct Tier-2 priority** — choosing Laplacian on the BT side and CAR on the Cogan side is one of the architectural forks in the alignment contract above.

2. **MFA / TextGrid / production-WAV location for D-cohort.** `SCRIPTS_USAGE.md` at the BIDS root references `D_Data/Phoneme_Sequencing/`, but that path isn't visible at `/datacommons/coganlab/D_Data/` on DCC. If they don't exist for D-cohort, we stay continuous-corpus / SSL-only on the D-cohort side — that's already the Tier-2 plan, so this is **not a blocker**, just due diligence in case they exist.

**Our calls** (don't bother him with these):

- **DCC sync direction.** Sync Box → `/work/ht203/data/` per the staged rsync recipe at `reports/dcc_sync_plan_2026_04_24/rsync_commands.sh`. No need to write into his tree under `/hpc/group/coganlab/nanlinshi/`.
- **Z-score recipe exact form.** A4 confirmed `production_highgamma.fif` is directly z-scored and consistent within-patient. uECoG audit showed mean/std ≡ recording-level median/MAD up to per-channel affine (ρ=1.0000) — recipe identity is academic for model-input purposes.
- **Authoritative usability tiering.** A1 sig-fraction proxy is good enough for SSL pretrain (false positives nearly free; we don't fine-tune on D-cohort in Tier 2). Defer his authoritative call until/unless we ever fine-tune on D-cohort.
- **Patient-space atlas convention.** Radius (3 mm), model-input form (full weighted Tier-1 support across `support_cache.py:TIER1_COLUMNS`), probability normalization (raw `[0, 100]`), and Tier-1 selection rule (argmax_wins ≥ N pooled across cohort) are all design calls we own. Decide empirically by ablation if any becomes load-bearing.

### One-shot Nanlin email

```
Subject: Stage-3 sEEG prep — two quick questions

1. Preprocessing reference / Laplacian-style variant. Box has parallel
   stats directories under derivatives/statistics/sub-D*/ for CAR / WM /
   M1 / STG / HIPP / LING — all global or anatomical-region references.
   The BrainTreebank cross-subject baselines use a Laplacian re-reference
   (subtract mean of adjacent same-stem depth contacts, e.g. T1b2 ←
   (T1b1 + T1b3) / 2), which is the winning recipe on the Neuroprobe
   leaderboard. Does the Cogan pipeline have a Laplacian or bipolar
   option I missed, or is the closest match in your tree one of the
   existing six? I assumed CAR for parity with the uECoG convention,
   but if any of WM / M1 / STG / HIPP / LING is closer to a local-bipolar
   style I'd rather use that.

2. MFA / TextGrid / production-WAV location for D-cohort. SCRIPTS_USAGE.md
   at the BIDS root references D_Data/Phoneme_Sequencing/, but that path
   isn't visible at /datacommons/coganlab/D_Data/ on DCC. If there's a
   canonical location for D-cohort phoneme-level alignments I should be
   reading from, point me at it. Totally fine if they don't exist —
   we'd just stay continuous-corpus / SSL-only on the D-cohort side.
```

## Risk + sequencing

**Don't start Tier 2 work until Tier 1 is over the bar.** The whole point of the Neuroprobe pivot was to validate the v14 thesis on a public benchmark. If Tier 1 lands above 0.539 on BT-only pretrain, Tier 2 is a real lift attempt with controlled scope. If Tier 1 lands below 0.539, the architectural prior alone isn't enough and adding D-cohort sEEG is unlikely to fix it — better to post-mortem and abandon than burn another 2–3 weeks.

Concrete cadence:
- Tier 1 finishes (~Stage 2 of `plan.md`, 2–3 weeks).
- If ≥ 0.539: open Tier 2 as next sprint, send the Nanlin email *the same day*. Items 1–6 of self-resolvable work fit in ~1 week; rsync (item 7) blocks on her reply; pretrain + fine-tune + submit (items 8–10) is another ~2 weeks.
- If < 0.539: don't start Tier 2. Submit Tier 1 if ≥ 0.539, otherwise post-mortem.

**Branch hygiene**: bilateral Tier-1 BNA (item 5) cascades to support cache headers, dataset shapes, and embedding tables. PS Stage-2 caches stay LH-15 — keep Tier 2 changes on a `tier2_bilateral` branch and don't merge into main until we're committed to submitting (or until PS Stage 2 resumes with a coordinated bilateral migration). The 6-step change order in `docs/strategy/stage_3_rh_expansion.md` is the canonical sequence.

**What this plan does not commit to**:
- Modality-mixed pretrain (uECoG + sEEG). That's Tier-3-equivalent — not in scope until BT-only and BT+D-cohort sEEG are both characterized.
- Per-electrode-token vs pseudo-grid loader for D-cohort fine-tune. Tier 2 fine-tunes on BT, not D-cohort, so the Stage-3 architecture decision is deferred.
- Cogan recipe parity with Nanlin's exact reference + z-score. We default to CAR + median/MAD until #1 and #2 from her email come back, then re-run the BT chain if she diverges.

## Reference docs

- `docs/neuroprobe/plan.md` — parent hillclimb plan, defines Tier sequencing
- `docs/neuroprobe/stage_0.md` — coord-space verification (shared between Tier 1 and Tier 2)
- `docs/references/data_reference.md` — Cogan preprocessing recipe + sEEG D-cohort corpus + Box mount audit pattern
- `docs/references/neuroprobe_benchmark.md` — BrainTreebank dataset + benchmark mechanics
- `docs/strategy/stage_3_rh_expansion.md` — bilateral parcel migration (executes here for Tier 2)
- `docs/tactics.md § backlog` — Nanlin-blocked list (canonical)
- `reports/dcc_sync_plan_2026_04_24/rsync_commands.sh` — staged Box → DCC recipe
- `data/dcohort_manifest.csv` — 128 D-pts, 122 ready, joins all Stage-3 audits
- `~/.claude/.../memory/project_seeg_stage3_prep_inflight_2026_04_24.md` — full Stage-3 prep findings
