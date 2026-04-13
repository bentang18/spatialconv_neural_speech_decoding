# v14 Implementation Task List

Current goal:

- **implement and verify `v14` on intra-op `uECoG` only**
- **start with `v14-core`, no learned per-patient calibration**
- **supervised training only in Phase 1**

Lightweight task/blocker list. Not a design document.

## Working Principle

**Every blocker below is a discussion item, not an engineering ticket.** v14 is slow, methodical, and precise — every logic step from raw voltages to phoneme decode is discussed, agreed, and understood before code is written. No handwavy implementation. No pre-committed numeric defaults. No legacy reuse. Rewriting from scratch is the default. See `CLAUDE.md` "Working Principle: Discuss Before Code".

Resolving a blocker means: logic agreed, contract (inputs, outputs, shapes, units) agreed, trade-offs and precedent written down, fallback path noted separately. "A likely default exists in the docs" is **not** resolution.

## Gating Rule

- [ ] **Do not begin Phase 1 implementation until each blocker has been explicitly discussed and frozen**
  - The goal is to write the interfaces once, not code against moving assumptions.
  - A blocker is not considered resolved just because a likely default exists in the docs.
  - Before implementation starts, each blocker should have:
    - an explicit decision
    - a written default
    - any important fallback / ablation path noted separately
    - an explicit agreement recorded (not "assumed to be fine")

## Blockers

- [ ] **#1 ACPC → MNI transform pipeline is wrong**
  - The current coordinate mapping should not be trusted as-is.
  - This is the top blocker because every later spatial step depends on it:
    - Brainnetome membership
    - parcel support
    - parcel-frame coordinates
    - learned calibration parameters
  - Before serious model implementation, verify the full coordinate chain and correct the transformation logic.
  - Open follow-ups:
    - confirm the exact native coordinate convention in the electrode files
    - confirm what `talairach.xfm` is actually doing in the current pipeline
    - ask Zac for the **MATLAB transform path he uses** and treat that as the trusted reference implementation
    - ask Zac for the **Python transform path/version** separately, because that version may be wrong
    - determine the correct ACPC → MNI procedure by reconciling the MATLAB and Python paths
    - decide whether current affine transforms are usable at all or whether nonlinear normalization is mandatory
  - Final verification requirement:
    - visualize the array in **ACPC subject coordinates**
    - visualize the transformed array in **MNI**
    - verify they land in essentially the same anatomical location after the transform, matching the visual check Zac showed locally

- [ ] **#2 Decide the exact temporal layer implementation**
  - This is now the next design blocker after coordinates.
  - The first pass should use a shared temporal layer, but the exact form is still not locked:
    - simple Conv1d patch embed / downsampler
    - small dilated temporal CNN tokenizer
    - another minimal shared temporal encoder
  - This decision matters because it sets:
    - token rate
    - local temporal receptive field
    - interface shape into the parcel summarizer
  - For the first pass, the goal is not the final best temporal front-end.
  - The goal is to choose the **simplest plausible shared temporal layer** that is expressive enough to validate the rest of `v14`.
  - Once chosen, the output interface should be written down explicitly:
    - tensor shape
    - token rate
    - receptive field

- [ ] **#3 Define the unsupported-vs-weak support threshold**
  - This is now an explicit blocker because it determines the semantics of the atlas-token interface.
  - The implementation needs a clear rule for:
    - when a parcel/subparcel is truly unsupported and should be masked out
    - when a parcel/subparcel is weakly observed but should remain active with low support
  - This decision affects:
    - `token_mask`
    - `token_support`
    - active-token counts per patient
    - one-parcel / low-coverage behavior
    - whether the model is masking missing parcels correctly rather than dropping weak evidence
  - Before locking the Phase 1 interface, write down:
    - the support statistic
    - the threshold rule
    - the expected behavior in edge cases

- [ ] **#4 Decide the default parcel split map**
  - The first implementation already assumes a fixed atlas/subparcel token interface, but the exact default split set still needs to be frozen for Phase 1.
  - This is not just an ablation detail because it determines:
    - `N_tok`
    - the local summarizer query counts `K_l`
    - the token-level connectivity bias expansion
    - low-coverage behavior for important motor/sensory parcels
  - Current strong candidates for default 2-token splits are:
    - `A6cvl`
    - `A4hf`
    - `A1/2/3ulhf`
    - `A2`
    - `A1/2/3tonIa`
  - `A4tl` remains the first additional split candidate rather than a locked default.
  - Before locking the Phase 1 interface, write down:
    - the exact default split set
    - the resulting `N_tok`
    - which parcels remain explicit extension candidates

- [ ] **#5 Define the exact parcel support statistic**
  - The model now depends on parcel support explicitly, but the exact support statistic is not yet frozen.
  - This is a blocker because it determines:
    - `token_mask`
    - `token_support`
    - the unsupported-vs-weak threshold itself
    - low-coverage behavior across patients
  - Before locking Phase 1, write down:
    - the exact support formula
    - whether it is based on summed memberships, effective electrode count, or another normalized quantity
    - whether support is static per patient / parcel or varies over time

- [ ] **#6 Lock the temporal-layer output contract**
  - Choosing the temporal layer architecture is not enough; the exact interface it emits must also be frozen.
  - This is a blocker because downstream modules depend on:
    - token rate
    - time axis semantics
    - receptive field
    - tensor layout handed to the local summarizer
  - Before locking Phase 1, write down:
    - patch length / stride or equivalent temporal resolution
    - output tensor shape
    - whether outputs are overlapping patch tokens or a continuous feature stream

- [ ] **#7 Decide how `token_support` enters the model**
  - The docs already assume `token_support` exists, but not how it is actually used.
  - This is a blocker because it changes whether low-support parcels are:
    - merely logged
    - appended to token features
    - used as attention bias / gating
    - used in loss weighting only
  - Before locking Phase 1, write down:
    - where `token_support` is injected
    - whether it affects backbone attention, decoder reads, or only diagnostics

- [ ] **#8 Lock token-level connectivity expansion details**
  - The design already assumes Brainnetome SC/FC bias initialization, but the exact expansion from parcel-level connectivity to token-level attention bias is still not frozen.
  - This is a blocker because it determines:
    - the initial inter-token graph structure
    - how split siblings inherit parent connectivity
    - the meaning and scale of the graph-attention prior
  - Before locking Phase 1, write down:
    - SC vs FC vs combined initialization
    - normalization / scaling of the connectivity matrix
    - sibling-token bias rule
    - fallback random / no-bias ablation path

- [ ] **#9 Lock the supervised training contract**
  - The Phase 1 model is supervised-only, but the exact supervised training contract still needs to be frozen.
  - This is a blocker because it determines:
    - the exact `v14-core` input window
    - target semantics
    - decoder training behavior
    - how evaluation is matched to training
  - Before locking Phase 1, write down:
    - exact response-locked or full-trial supervised window
    - target format for the 3-phoneme sequence
    - decoder training rule
    - train-time vs eval-time decoding behavior
    - loss definition

- [ ] **#10 Lock the parcel-frame construction contract**
  - The local summarizer depends on canonical parcel coordinates, but the exact construction is still not fully written down.
  - This is a blocker because it determines:
    - what “within-parcel geometry” actually means
    - reproducibility of the local summarizer inputs
    - whether parcel-frame features are computed consistently across patients
  - Before locking Phase 1, write down:
    - how parcel centroids are defined
    - how parcel axes / rotations are defined
    - how parcel axis scales are defined
    - whether these are computed once offline and cached
    - how canonical parcel coordinates are attached to local electrode features

- [ ] **#11 Lock the channel inclusion policy**
  - It is still unresolved whether Phase 1 should use:
    - all non-artifact channels
    - significant channels only
    - or another filtering rule
  - This is a blocker because it changes:
    - parcel support statistics
    - active-token counts
    - coverage comparisons across patients
    - the meaning of weak support
  - Before locking Phase 1, write down:
    - the default channel inclusion rule
    - what happens when sig-channel files are missing
    - whether channel filtering is part of the model interface or only an ablation

- [ ] **#12 Lock the amp → physical-electrode → ACPC bridge per patient and verify it 1-to-1**
  - Ground truth: `data/recording_details/uecog_recording_details.xlsx` plus `data/channel_maps/*.mat`. The TSV grid heuristic and the legacy "256-ch direct mapping" shortcut are both untrusted.
  - Zac's concern: BIDS `electrodes.tsv` positions are probably right, but the electrode *names* in that TSV may not line up with `.fif` channel names. Bridge through the per-patient map, not through TSV names.
  - Per-patient rule (Phase 1 core: S14, S26, S33, S62):
    - **128 Strip, Map 4** — S14, S16, S22, S23, S26. Shape (8, 16), pitch 1.33 mm. Local `*_channelMap.mat` = Map 4 with a +1 shift (Map 4 is 0-indexed, `.mat` is 1-indexed). Bridge: `fif ch N → (r,c) with chanMap[r,c]==N → phys = r*16 + c + 1 → RAS`.
    - **256 Grid, Map 3** — S33, S39, S62. Shape (46, 24), pitch 1.72 mm, I/cross, 256 populated out of 1104. All `*_channelMapAll.mat` files are byte-identical. Bridge: `amp ch → (r,c) in chanMapAll → phys → RAS`. Do not reuse the legacy `fif N → RAS N` shortcut.
    - **S58 is also 256 Grid Map 3.** Local `S58_channelMap.mat` is a compact (12, 24) crop: same NaN pattern as Map 3 central rows, values are zero-indexed amp channels 0..255. Not a mis-saved file. The verifier must prove the crop aligns with a contiguous 12-row slice of the full (46, 24) Map 3.
  - Excluded from Phase 1: S32 (no HG response), S57 (hybrid strip, 52/256 sig). Map 8 and the S57 micro wiring are deferred with the patient.
  - Stray file to delete or quarantine: `S39_channelMap.mat` is byte-identical to the 128-strip template but S39 is 256 Grid Map 3. The authoritative S39 map is `S39_channelMapAll.mat`.
  - Verification contract: for every PS patient and every `epochs.ch_names` entry, return exactly one physical electrode and exactly one RAS coordinate. Fail loudly on ambiguity, missing entries, or duplicate bridges. This is a 1-to-1 test, not a set-overlap test. The legacy `scripts/archive/legacy/diagnostic_channel_mapping.py` does overlap only and is not sufficient.
  - Before locking Phase 1, write down:
    - the mapping file to load for each patient (Map 3 or Map 4)
    - the exact amp → phys → RAS bridge per layout, including the S58 crop alignment
    - the 1-to-1 verifier contract: inputs, assertions, failure modes, report format

- [ ] **#13 Replace the current baseline loader semantics with an explicit `v14-core` channel interface**
  - The current `load_patient_data()` / `load_per_position_data()` behavior should not be reused blindly for `v14-core`.
  - Today those loaders zero artifact channels in place rather than removing them from the active channel set, and they rely on the old grid-construction path.
  - That is dangerous because it preserves channels in the tensor while mutating their values, which changes support semantics and can silently leak baseline assumptions into the new implementation.
  - Before locking Phase 1, write down:
    - whether artifact channels are removed or retained with masks
    - how channel inclusion interacts with parcel support
    - whether `v14-core` gets a new loader instead of adapting the old Conv2d loader

- [ ] **#14 Audit legacy baseline modules before reusing any of them in `v14-core`**
  - Mechanically enforced as of 2026-04-13: the entire pre-v14 tree is quarantined under `src/speech_decoding/archive/legacy/` and the pytest CI guard `tests/v14/test_no_legacy_imports.py` fails any import of a legacy path from a v14 module. The following files now live under the quarantine and are not importable from active code:
    - `archive/legacy/data/atlas.py` (centroid VE, distance thresholds)
    - `archive/legacy/data/coordinates.py` (includes the untrusted ACPC→MNI helper)
    - `archive/legacy/data/bids_dataset.py` (grid reshape loader)
    - `archive/legacy/data/grid.py`
    - `archive/legacy/data/augmentation.py`
    - `archive/legacy/data/collate.py`
    - `archive/legacy/data/sig_channels.py`
    - `archive/legacy/data/audio_features.py`
    - `archive/legacy/models/*` (Conv2d read-in, BiGRU backbone, heads, assembler)
    - `archive/legacy/training/*` (per-patient trainer, LOPO, CTC utils, MFA/phonological aux)
    - `archive/legacy/pretraining/*` (NCA-JEPA + BYOL/DINO/VICReg/LeWM + generators)
    - `archive/legacy/evaluation/metrics.py` and `content_collapse.py`
  - The discussion-first rule applies per module. Before any piece of v14-core is written, decide for each component (loader, coordinates, temporal layer, model, training loop, metrics) whether to:
    - leave it permanently in the quarantine and re-derive the need fresh, or
    - re-examine a specific helper and write a fresh v14-native version.
  - Copy-paste from the quarantine is forbidden. Re-derivation is the only sanctioned path.

- [ ] **#15 Justify or replace every numeric default in `src/speech_decoding/v14/config.py`**
  - The config dataclasses currently in `v14/config.py` were written before the 2026-04-13 discussion-first rule was locked.
  - The `AtlasConfig` and `PatientCalibrationConfig` defaults match the Phase-1 contract (real PM volume, all `learn_*=False`, leash constants match the documented bounds). These can stay as "contract-in-code" but their leash values (`max_translation_mm=15.0`, `max_rotation_rad=0.15`, `max_parcel_offset_mm=15.0`, `min_temperature=0.3`, `max_temperature=3.0`) are inherited from the pre-Phase-1 v14 draft and each value needs an explicit written justification before it becomes locked.
  - The `TemporalTokenizerConfig`, `LocalSummarizerConfig`, `BackboneConfig`, and `DecoderConfig` defaults are **pre-committed magic numbers that have not been discussed**:
    - `TemporalTokenizerConfig`: `d_model=64`, `patch_ms=250`, `stride_ms=50`, `sample_rate_hz=200`, `hidden_channels=(16, 32, 32)`
    - `LocalSummarizerConfig`: `d_model=64`, `point_mlp_hidden=64`, `parcel_embedding_dim=16`, `support_feature_dim=1`
    - `BackboneConfig`: `d_model=64`, `num_blocks=2`, `num_heads=4`, `ffn_hidden=256`, `dropout=0.2`
    - `DecoderConfig`: `d_model=64`, `num_queries=3`, `vocab_size=9`, `ar_embedding_dim=64`
  - Under the discussion-first rule, none of these numbers may enter a training run until:
    - each has a written justification tied back to data scale, token rate, or precedent from a named paper
    - the agreement is recorded
    - the trade-offs and fallback values are noted separately
  - `num_queries=3` and `vocab_size=9` are tied to 3-phoneme targets and the 9-phoneme PS set, so they are less speculative, but the exact label→index contract is itself a separate discussion (see #16).
  - The file currently carries a module-level banner listing every provisional field. The banner must stay until this blocker is resolved.

- [ ] **#16 Lock the v14 phoneme label → integer index contract**
  - The old `speech_decoding.data.phoneme_map` used a 1-indexed mapping with `0` reserved for the CTC blank. v14 does not use CTC, so that convention does not transfer.
  - As of 2026-04-13, the CTC and articulatory-matrix helpers have been split out of `src/speech_decoding/data/phoneme_map.py` and archived under `src/speech_decoding/archive/legacy/data/phoneme_map_ctc_articulatory.py`. The active `phoneme_map.py` module now only provides the label space itself (`ARPA_PHONEMES`, `PS2ARPA`, `ARPA2PS`, `_ALL_ARPABET`, `normalize_label`, `filter_to_ps_phonemes`).
  - Before the v14 decoder is wired up, the label→index contract must be discussed and locked:
    - 0-indexed `[0, 9)` vs 1-indexed `[1, 10)` with 0 reserved for something specific
    - whether there is a reserved padding / start / end index at all
    - canonical ordering of the 9 PS phonemes (the current `ARPA_PHONEMES` list ordering is inherited from the pre-v14 era — if we keep it, it should be an explicit agreement, not an assumption)
    - the exact integer assigned to each phoneme
    - how this contract surfaces in the dataset, the loss, and the metric
  - Until this is locked, do not write decoder or training code that assumes any particular index convention.
  - **Related discussion items from the 2026-04-13 phoneme-loading rigor audit: blockers #17–#25 below. #16 (this blocker) covers ordering and indexing; the rigor audit covers identity, ground truth, and loader assertions.**

#### Phoneme / label-space rigor audit (2026-04-13)

Blockers #17–#25 come from a first-pass audit of the PS label space and the upstream assumptions the pre-v14 loader silently inherited. These are **discussion items**, not fixes. No one should touch `phoneme_map.py`, write a v14 data loader, or report any phoneme-level metric until these are resolved — any v14 metric depends on the label space being correct.

- [ ] **#17 Confirm or fix PS vowel → ARPABET mapping (`ae`, `u`)**
  - The inherited mapping is `PS2ARPA["ae"] = "EH"` and `PS2ARPA["u"] = "UH"`. Under standard ARPABET, `EH = /ɛ/` (as in "bet") and `UH = /ʊ/` (as in "book"). The PS notation suggests `ae = /æ/` (as in "cat") → should be `AE`, and `u = /u/` (as in "boot") → should be `UW`. If that is right, the inherited mapping has silently misidentified two of the four PS vowels for the entire pre-v14 era.
  - Evidence for the concern (not proof): the archived articulatory-matrix file labels its `EH` row as "EH (/æ/): vowel, mid, front", which is internally inconsistent — /æ/ is *low-front*, /ɛ/ is *mid-front*. The row values are defensible for /ɛ/, but the ligature label `ae` and the IPA comment `/ae/` point at /æ/.
  - This must be resolved against the **actual phonetic content of the 52 PS tokens** (see #21) and against Zac's / Duraivel's original task design, not against any code already in the repo.
  - Decision needed:
    - the true ARPABET identity of each of the 9 PS symbols (`a`, `ae`, `i`, `u`, `b`, `p`, `v`, `g`, `k`)
    - whether `ARPA_PHONEMES` needs to be updated (likely: `EH → AE`, `UH → UW`)
    - retroactive note: every pre-v14 metric that reported "EH accuracy" or "UH accuracy" was reporting under the wrong phoneme identity if the concern is correct
  - Until this is locked, **no phoneme-level metric is trustworthy** and no new phoneme-loading code should be written.

- [ ] **#18 Add a load-time assertion on the BIDS event_id mapping**
  - `CLAUDE.md` quotes the mapping `{'a':1, 'ae':2, 'b':3, 'g':4, 'i':5, 'k':6, 'p':7, 'u':8, 'v':9}` but nothing in the codebase asserts that the .fif `event_id` dictionary for each patient actually contains these exact keys with these exact integer values.
  - Decision needed:
    - whether the v14 loader should hard-assert this mapping on every load and crash on divergence
    - or whether it should accept any bijection and build the label → ARPA map dynamically
    - what the expected upstream-producer guarantees actually are (ask Zac)
  - Output: either a frozen expected mapping in `data/` as JSON or a loader invariant that fails loudly on divergence. Not both — pick one.

- [ ] **#19 Verify (or remove) the "3 consecutive events = 1 trial" ordering assumption**
  - The archived `load_patient_data` did `all_data[0::n_phons]` and reconstructed the 3-phoneme label from events `[i*3], [i*3+1], [i*3+2]`. If a single phoneme-position epoch was ever dropped upstream (failed audio, MFA miss, bad trial), every subsequent trial silently misaligns by one. There is no consistency check — no "assert every triplet begins with a position-0 marker."
  - Decision needed:
    - whether the upstream .fif actually provides position markers (position-0 / position-1 / position-2 tags per epoch), or whether position is implicit in order only
    - whether v14 can switch to **positive position markers** instead of relying on stride-3 slicing
    - the loader invariant that must hold (e.g., every consecutive triplet has distinct positions `{0, 1, 2}`)
  - Do not reuse the old stride-3 slicing pattern in v14 without proving this invariant from the actual data.

- [ ] **#20 Decide the v14 trust posture for MFA alignment**
  - An archived memory (`project_data_quality_2026_03_30.md`) previously flagged "per-phoneme alignment unreliable ~50%". The best recent baseline (2026-04-04) used per-phoneme MFA epochs anyway and reached PER 0.734 on S14, so the alignment was "good enough for Conv2d" — but that is not the same as "good enough for the v14 decoder target."
  - Decision needed, pick exactly one:
    - **(a) Trust MFA end-to-end** — use per-phoneme epochs, don't re-verify. Requires an explicit statement of why the Conv2d-era reliability carries over.
    - **(b) Onset-locked full-trial only** — use one epoch per 3-phoneme trial, locked to response onset. Avoids the alignment question entirely.
    - **(c) Re-verify against audio** — before using per-phoneme MFA, spot-check N trials against the recorded audio and quantify alignment error.
  - Whichever option is chosen, the v14 loader contract and the decoder's time-axis semantics depend on it.

- [ ] **#21 Commit the canonical 52-token enumeration for the PS task**
  - `CLAUDE.md` says "52 CVC/VCV tokens, 3 phonemes each" but the literal list of 52 tokens is not in the repo anywhere. Consequences:
    - we cannot programmatically verify that every patient saw the same tokens
    - grouped-by-token CV currently groups by the 3-phoneme triple, which collapses any two distinct tokens sharing the same triple
    - we cannot sanity-check that audio and labels agree at the token level
    - phonotactic / position-dependent analyses are impossible to reproduce
  - Action:
    - ask Zac for the canonical 52-token list (preferably with both the PS notation and the IPA / ARPABET decomposition)
    - commit it under `data/` (or wherever agreed) as a JSON / CSV with fields: `token_id`, `ps_notation`, `arpabet_sequence`, `ipa`
    - add a load-time assertion that every trial in every patient decodes to one of the 52 committed tokens
  - This is a prerequisite for #17 (vowel identity can be verified against the canonical list) and for #25 (ground-truth fixture tests).

- [ ] **#22 Assert cross-task / cross-patient event_id consistency**
  - There is no test that the `epochs.event_id` dictionary is identical across patients within the PS task, or between the PS and Lexical tasks. The archived loader just trusted `epochs.event_id` to contain what it needed.
  - Decision needed:
    - the set of patients to include (PS only for Phase 1, but the assertion should also cover Lexical for any cross-task pooling later)
    - whether the assertion lives in the loader or in a standalone `tests/v14/test_event_id_invariants.py`
    - the failure mode: crash on divergence, warn, or fall back to per-patient mapping
  - Tied to #18; resolve together.

- [ ] **#23 Lock the `normalize_label` input contract**
  - Current behavior: `normalize_label` accepts lowercase PS (`a`, `ae`, ...), uppercase ARPABET with stress digits (`AA1`, `EH0`, ...), and canonical uppercase ARPABET (`AA`, `B`, ...). It raises `ValueError` on lowercase-ARPABET (`"aa"`) or empty string or anything else.
  - The three accepted input forms are not documented anywhere except inside the function. The contract is untested — a single upstream producer that starts emitting lowercase-ARPA or mixed-case labels would silently break.
  - Decision needed:
    - the explicit list of input forms v14 will accept, written down
    - whether v14 should *uppercase-first* before calling `normalize_label`, making the contract simpler
    - whether unknown labels should continue to raise or should be silently rejected via a `strict=False` flag
  - The resolution should be reflected in the module docstring and in tests that exercise every accepted form and every rejected form.

- [ ] **#24 Decide the `filter_to_ps_phonemes` fail-silent policy**
  - Current behavior: catches `ValueError` from `normalize_label` and returns `False` for unknown labels. This is intentional for cross-task filtering (the Lexical task has labels not in the PS set; those should be filtered out, not crash the loader).
  - Risk: a typo or data corruption silently becomes `False` instead of raising. Within-task use of this function could hide bugs.
  - Decision needed:
    - whether `filter_to_ps_phonemes` stays fail-silent (cross-task only) or is split into a strict and a non-strict variant
    - whether within-task callers must use a separate strict path
    - tests that cover both silent-reject and strict-reject paths
  - Tied to #23; resolve together.

- [ ] **#25 Add ground-truth phoneme fixture tests against the 52-token list**
  - Current `tests/test_phoneme_map.py` tests the internal consistency of the mapping (`normalize_label("a") == "AA"`), but never checks that `"AA"` corresponds to the /ɑ/ sound actually uttered in the audio. For a label space that is the foundation of every downstream metric, the contract should be exercised against the real token set.
  - Dependencies: #21 must be done first (need the canonical 52-token list). #17 should be done first (need the correct vowel identities).
  - Action:
    - after #17 and #21, write a test fixture that loads the 52-token list
    - assert that `normalize_label` round-trips every element of every token's ARPABET decomposition
    - assert that `filter_to_ps_phonemes` returns `True` for every phoneme in the PS vowel/consonant set and `False` for every non-PS ARPABET symbol
    - optionally: assert that `ARPA_PHONEMES` equals exactly the set of phonemes appearing across the 52 tokens
  - These tests are the first real rigor baseline for `phoneme_map.py`.

- [ ] Real probabilistic Brainnetome maps wired into the implementation path and verified
  - local PM file exists at `/Users/bentang/Documents/Code/speech/data/atlas/BNA_PM_4D.nii.gz`
  - verify ROI/channel indexing and orientation against the MPM label map
  - use the PM file as the active membership source for Phase 1
  - do **not** allow silent fallback to the old smoothed-MPM proxy
  - use the MPM file only for ROI indexing and sanity checks

- [ ] Core-patient coordinate sanity checks complete

## Phase 1: `v14-core` on uECoG Only

This phase intentionally excludes learned per-patient calibration.
This phase is also **supervised only**.

- [ ] Fix scope to **fixed atlas mapping first**
  - fixed translation / rotation from the corrected ACPC → MNI pipeline
  - no learned gain / offset
  - no learned rigid `Δ/ω`
  - no learned parcel offsets `δ_l`
  - no learned parcel temperatures `τ_l`
  - no additional fixed gain / impedance normalization from channel statistics for now
  - use the best available corrected coordinates and fixed atlas membership only

- [ ] Freeze scope to core `uECoG` patients: `S14`, `S26`, `S33`, `S62`
- [ ] Verify high-gamma feature loading path
- [ ] Verify artifact-channel exclusion path
- [ ] Verify sig-channel / all-channel policy for `v14`
- [ ] Verify that the old `load_patient_data()` semantics are not being reused blindly in `v14-core`
- [ ] Verify that legacy baseline modules are not silently defining the new `v14-core` contract

## Data / Spatial Interface

- [ ] Implement corrected coordinate loader for `v14`
- [ ] Write a 1-to-1 channel-map verifier that, for every Phase-1 PS patient, asserts `fif.ch_names → chanMap*/Map-N → phys_elec → RAS` is a complete injection. Report any channel that fails. Discussion-first; no code until the bridge contract is agreed.
- [ ] Confirm the S58 `_channelMap.mat` (12, 24) is a contiguous row-slice of the (46, 24) Map 3, then document the row offset. Expected outcome from the verifier, not a separate investigation.
- [ ] Delete (or flag and quarantine) the stray `S39_channelMap.mat` — byte-identical to the 128-strip template, but S39 is 256 Grid Map 3.
- [ ] Replace the TSV-based grid heuristic in `grid.py` / baseline loading with the exact mapping path where needed
- [ ] Request Zac's trusted MATLAB ACPC → MNI transform steps
- [ ] Request Zac's Python ACPC → MNI transform steps and compare against MATLAB
- [ ] Reproduce the MATLAB-vs-Python coordinate comparison locally
- [ ] Lock translation / rotation to the verified ACPC → MNI output for `v14-core`
- [ ] Implement soft Brainnetome membership lookup
- [ ] Implement parcel support / confidence summary
- [ ] Verify expected parcels are reached for core patients
- [ ] Add ACPC vs MNI array visualization check
- [ ] Visualize parcel support per patient
- [ ] Define threshold for unsupported vs weakly supported parcels
- [ ] Verify voltage/channel index → physical electrode → coordinate mapping end to end on representative 128-ch and 256-ch patients

## Shared Model Core

- [ ] Implement shared temporal layer
  - first-pass version only
  - fix output shape and token rate clearly
  - document the exact temporal interface it hands to the parcel summarizer
- [ ] Implement parcel-frame coordinate construction
- [ ] Implement within-parcel point encoder
- [ ] Implement Perceiver-style local summarizer
- [ ] Emit fixed `N_tok` token tensor plus `token_mask` and `token_support`
- [ ] Implement inter-region graph attention
- [ ] Implement Brainnetome SC/FC attention-bias initialization
  - expand parent-parcel SC/FC to token-level bias
  - decide normalization / scaling of the bias matrix
  - optionally add small sibling-token attraction bias within split parcels
- [ ] Implement temporal attention blocks
- [ ] Implement AR cross-attention decoder
- [ ] Verify the backbone works when only one parcel / token is active
  - region attention should degrade gracefully
  - temporal modeling should still function
- [ ] Ensure unsupported parcels are masked, not hallucinated
  - zero-fill inactive token values for storage only
  - prevent masked tokens from contributing through attention
  - prevent masked tokens from entering decoder reads
- [ ] Ensure weakly supported parcels remain active and carry low support rather than being dropped

## Training / Verification

- [ ] Build minimal end-to-end model assembly
- [ ] Integrate grouped-by-token CV-compatible batching
- [ ] Use supervised loss only for the first pass
- [ ] Add tiny-subset overfit test
- [ ] Add shape / token-count sanity tests
- [ ] Add missing-support masking test
- [ ] Add one-active-parcel test
- [ ] Add zero-filled masked-token leakage test
- [ ] Add atlas-bias initialization sanity test
  - verify token-level bias shape matches `N_tok`
  - verify split siblings inherit parent connectivity correctly
  - verify random / no-bias fallback is easy to run as an ablation

## Phase 1.5: SSL on Full Continuous uECoG

This is the next step after supervised `v14-core` correctness.

- [ ] Treat the exact SSL pretraining scope as a blocker only when Phase 1.5 begins
- [ ] Lock the sequencing explicitly:
  - first supervised `v14-core` on response-locked `uECoG`
  - then SSL on the full continuous `uECoG` corpus
- [ ] Do **not** default to response-locked SSL
  - keep response-locked SSL as an ablation only if needed later
- [ ] Verify the available full continuous `uECoG` corpus and patient coverage
- [ ] Define the continuous-data loading path for SSL
- [ ] Decide the exact SSL windowing / chunking interface
  - chunk length
  - overlap / stride
  - patient mixing policy
- [ ] Decide whether the first SSL pass pretrains:
  - temporal tokenizer only
  - tokenizer + local summarizer + backbone
  - full shared stack except decoder
- [ ] Define the default SSL objective for this stage
  - current default expectation: JEPA-style latent prediction
- [ ] Define the handoff from full-corpus SSL back to supervised response-locked fine-tuning

## Phase 2: Learned Per-Patient Calibration

Only start this after `v14-core` is working end-to-end.

- [ ] Implement gain / offset calibration
- [ ] Implement rigid `Δ/ω` correction with explicit bounds
- [ ] Implement parcel offsets `δ_l`
- [ ] Implement parcel temperatures `τ_l`
- [ ] Add freeze / unfreeze controls for staged optimization
- [ ] Decide whether the first learned calibration step should be gain/offset only or full `Δ/ω`
- [ ] Decide whether any fixed gain / impedance normalization from baseline-only channel statistics is warranted
- [ ] Verify learned calibration improves rather than destabilizes the fixed-atlas baseline

## Deferred Until After uECoG Correctness

- [ ] Extended `uECoG`
- [ ] `sEEG`
- [ ] External datasets
- [ ] Functional pooling variants
- [ ] Broad ablations
