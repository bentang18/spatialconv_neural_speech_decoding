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

- [x] **#5 Parcel support statistic — DECIDED 2026-04-13**
  - **Decision**: PM-weighted sum of non-artifact contacts inside the parcel.
    - Formula: `support[parcel] = Σ_i PM(parcel | x_i)` over non-artifact contacts `i`, using the real Brainnetome PM volume (`data/atlas/BNA_PM_4D.nii.gz`).
    - Purely geometric. Does **not** depend on task-responsiveness or the sig-channel files. Channel inclusion is a separate decision (#11).
    - Static per `(patient, parcel)`. Does not vary over time.
    - Consistent with Phase 1's commitment to the real PM volume over hard MPM labels.
  - **Rationale**:
    - Raw count throws away the PM uncertainty we loaded the real volume to preserve.
    - Task-responsiveness couples calibration to the supervised task, which breaks Phase 1.5 SSL — you cannot compute support for an SSL window without re-running the significance test.
  - **Open sub-questions** (tracked separately):
    - unsupported-vs-weak threshold → #3
    - how `token_support` enters the model → #7
  - **Phase 2 / sEEG note**: a flat PM-weighted sum over-rewards redundant shaft-internal sampling. When sEEG joins, the formula needs a diversity weighting — downweight contacts close to each other in MNI (or close along a shaft axis). Safe to defer: the Phase 1 formula is replaceable without breaking the `token_mask` / `token_support` interface.

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

- [x] **#7 How `token_support` enters the model — DECIDED 2026-04-13**
  - **Decision**: concat-to-token. The per-parcel scalar `token_support` is concatenated onto the token feature and passed through a linear projection back to `d`, so every token that enters inter-region attention carries its own support signal as part of its content. Attention-bias stays on the ablation list but is **not** active in the Phase 1 baseline.
  - **Rationale**: concat-to-token and attention-bias answer different questions, so they are not redundant. Concat puts support inside the token's representation — the model learns what low support means for content. Attention-bias externally discounts how much *other* tokens attend to a low-support token and does not touch its content.
  - A low-support token can still be the only evidence in its parcel (e.g. one contact in Broca's for some patient). The right behavior is "let other tokens attend fully, but mark it as low-confidence internally". That is concat-to-token without attention-bias. Attention-bias would wrongly suppress the only evidence available.
  - Conversely, a high-support token may be task-irrelevant, and attention-bias cannot express that (support is high, so no discount fires).
  - **Phase 1 discipline**: only one mechanism active in the baseline so ablations stay readable. Concat-to-token is the default; attention-bias is a named ablation comparison.
  - **Implementation note**: the concat dimension is `d + 1` before the linear projection, so the point-summarizer output `d` is preserved for the backbone. `token_mask` is still a separate binary signal and gates attention structurally — concat-to-token handles graded support inside the active set, `token_mask` handles hard absence.

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

- [x] **#10 Parcel-frame construction contract — DECIDED 2026-04-13**

  ### Origin
  - PM-weighted centroid over the Brainnetome PM volume (`data/atlas/BNA_PM_4D.nii.gz`).
    - `origin[parcel] = Σ_v PM(parcel | v) * position_v / Σ_v PM(parcel | v)`, summed over MNI voxels `v`.
    - Volumetric, not surface-projected. Keeps the origin consistent across uECoG (contacts on pia) and Phase 2 sEEG (contacts distributed through the cortical slab).

  ### Rotation method — fsaverage-based cortical-normal axes
  - **Space mapping**: use `nilearn.surface.vol_to_surf()` to project the BNA PM volume onto the fsaverage pial mesh. nilearn handles the MNI152 ↔ fsaverage (MNI305) space mapping internally — we do **not** build our own warp.
    - Resolution: `fsaverage` (full, 163 842 vertices per hemisphere). Fs6/5 is too coarse for small parcels like `A44op`. Cache is built once offline, so resolution does not matter for runtime.
    - Interpolation: `interpolation="linear"`, `kind="ball"` with a small radius (needs empirical tuning at build time). **Must be verified**: for a probabilistic atlas, we need "PM value near this vertex", not "line-integral through the cortical slab" — the nilearn defaults are tuned for fMRI BOLD, not atlas membership, and the choice must be checked against a flat-parcel smoke test.
    - Separate LH and RH projections. fsaverage has distinct `pial_left` and `pial_right` meshes and vertex sets. A left Brainnetome parcel must only receive left-hemisphere vertices, and vice versa. Mixing sides silently invalidates the axes.
  - **z-axis (cortical normal)**: first eigenvector of the **PM-weighted second-moment tensor** of the per-vertex pial normals:
    ```
    M_parcel = Σ_v PM(parcel | v) * n_v n_v^T / Σ_v PM(parcel | v)
    z_axis[parcel] = dominant eigenvector of M_parcel
    ```
    where `n_v` is the unit pial normal at fsaverage vertex `v`, and `v` ranges over the hemisphere-matched fsaverage vertices inside the parcel.
    - **Second-moment tensor, not mean-subtracted covariance.** The distinction is load-bearing — see rigor note below.
    - `n_v` from per-vertex normals computed from the fsaverage triangle mesh (cross products of adjacent triangle edges, area-weighted, normalized). Standard mesh-normal computation.
  - **Tangent axes (x, y)**: project the in-parcel vertex positions onto the plane perpendicular to `z_axis`, then run 2D PCA on the PM-weighted projected positions.
    - Primary tangent direction = first 2D PC (roughly "gyrus long axis" for curved parcels, "principal cortical extent" for flat ones).
    - Secondary tangent direction = second 2D PC, with sign fixed by `y = z × x` to force a right-handed frame.

  ### Rigor: why second-moment tensor, not mean
  - **Mean normal** fails on curved and sulcal parcels:
    - Flat gyral parcel (e.g. `A4hf` on the precentral crown): mean ≈ outward normal. Works.
    - Curved gyral parcel (e.g. `A44` wrapping over IFG): normals fan along an arc; mean points into the middle of the arc, which is roughly outward but underweights the curvature.
    - Sulcal / bimodal parcel (e.g. `INSa` in the insular pocket, `STGa` wrapping into STS): normals come from two opposing walls; mean ≈ 0 and the z-axis becomes undefined.
  - **Second-moment tensor** `M = E[n n^T]` handles all three cases correctly:
    - Concentrated distribution: `M ≈ v v^T` for the mean direction `v`, so the dominant eigenvector recovers the mean. Matches the simple case exactly.
    - Arc distribution: dominant eigenvector ≈ center of arc (same as mean), because the arc is spread around the mean direction.
    - Bimodal distribution (`+v` and `-v` clusters): `M = v v^T` (both contributions are equal because `n n^T` is sign-invariant). Dominant eigenvector = the bank-normal axis, which is the correct cortical-normal direction for a sulcal parcel.
  - Equivalently: `M` is a dyadic orientation tensor on unit vectors, and this is the standard formulation in diffusion-MRI orientation analysis for exactly this reason. Do **not** mean-subtract before the eigendecomposition — that would compute the covariance of residuals, which picks up noise in the concentrated case and tangent-plane variation in the arc case. We want the dominant *direction*, not the dominant *variation*.
  - The dominant eigenvalue `λ_1 ∈ [1/3, 1]` is a diagnostic for how well-concentrated the parcel's normals are: `λ_1 → 1` means tightly concentrated (flat gyrus), `λ_1 → 1/2` means bimodal (sulcal wall), `λ_1 → 1/3` means isotropic (should not happen for any real parcel — fail loudly if it does).

  ### Per-axis scale normalization
  - Feed `(x/σ_x, y/σ_y, z/σ_z)` to the shared point encoder, where `σ_*` are the standard deviations of the PM-weighted in-parcel positions projected along each parcel-frame axis.
  - Append `(log σ_x, log σ_y, log σ_z)` as three scalar features on the parcel's token (once per parcel, not per electrode). The point encoder sees parcel-size-invariant positions; parcel size lives on the token as a three-number side channel.
  - Std-based, not extent-based — robust to boundary outlier voxels.
  - Log-scaled token scalars because parcel sizes span more than an order of magnitude.

  ### Sign determinism
  - `z` (cortical normal): eigenvectors have ±1 ambiguity. Pin by `sign(z · (origin − brain_centroid)) > 0`, where `brain_centroid` is the mean of the fsaverage mesh vertices (approximately the MNI origin). This forces positive `z` = outward from the brain for every parcel on the cortical surface. For any parcel where the dot product is within numerical noise of zero (`< 1e-3`), fail loudly — the parcel is near the brain centre and the rule is ambiguous.
  - `x` (primary tangent): pin by "positive `x` points toward the anterior-most in-parcel fsaverage vertex, projected onto the tangent plane". Deterministic as long as there is a single unambiguous anterior extreme — flag otherwise.
  - `y` (secondary tangent): `y = z × x` (right-handed frame). No independent sign choice.

  ### Offline caching
  - Build once, cache to `data/atlas/parcel_frames.npz`.
  - Keys:
    - `parcel_ids`: `(N_parcel,) int` — BNA parcel indices covered by the cache
    - `hemisphere`: `(N_parcel,) str` — `"L"` or `"R"` per parcel
    - `origins`: `(N_parcel, 3) float` — PM-weighted centroids in MNI
    - `rotations`: `(N_parcel, 3, 3) float` — stacked `[x_axis; y_axis; z_axis]` per parcel
    - `sigmas`: `(N_parcel, 3) float` — axis standard deviations in mm
    - `log_sigmas`: `(N_parcel, 3) float` — pre-computed token-level scalars
    - `concentration`: `(N_parcel,) float` — dominant eigenvalue `λ_1` of the second-moment tensor (diagnostic for gyral / sulcal / isotropic)
    - `n_vertices`: `(N_parcel,) int` — number of fsaverage vertices that contributed (coverage diagnostic)
    - `build_metadata`: dict — fsaverage version, nilearn version, `vol_to_surf` configuration, PM threshold, build timestamp
  - Cache must be bit-deterministic given the same inputs. Running the builder twice produces byte-identical output — verified by content hash in CI.

  ### Phase 2 upgrade path (tracked separately in `## Phase 2`)
  - Refine from **per-parcel mean cortical normal** to **per-voxel local cortical normal**. Phase 1 stores one rotation per parcel; Phase 2 stores one rotation per PM voxel (or per contact, looked up at load time). The Phase 1 cache is a valid fallback for Phase 2, not a wasted artifact.
  - Per-voxel is strictly better only for sEEG in curved parcels (where contacts at the crown vs at the fundus should get different local normals). uECoG is invariant to this refinement because all contacts are on the pial surface.

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
- [ ] **Build and verify `data/atlas/parcel_frames.npz`** per the `#10` contract
  - Dependencies: fsaverage pial mesh (`nilearn.datasets.fetch_surf_fsaverage(mesh="fsaverage")`), BNA PM volume at `data/atlas/BNA_PM_4D.nii.gz`.
  - Builder steps (per hemisphere, then merged):
    1. Load fsaverage pial mesh and compute per-vertex unit normals from the triangle mesh (area-weighted average of adjacent face normals, normalized).
    2. Project the BNA PM volume onto the fsaverage mesh via `nilearn.surface.vol_to_surf()` using `interpolation="linear"`, `kind="ball"`. **Sub-task**: empirically pick the ball radius against a flat-parcel smoke test and lock it in `build_metadata`.
    3. For each BNA parcel on this hemisphere, compute the PM-weighted second-moment tensor of the per-vertex normals, its dominant eigenvector (= z-axis), and the dominant eigenvalue (= concentration diagnostic).
    4. Project in-parcel PM-weighted vertex positions onto the plane ⊥ z-axis; run 2D PCA to get x and y tangent axes; set `y = z × x` to force right-handed.
    5. Apply sign-determinism rules: `sign(z · (origin − brain_centroid)) > 0`, `x` points toward the anterior-most in-parcel vertex, `y = z × x`.
    6. Compute PM-weighted origin (volumetric, from `BNA_PM_4D.nii.gz` directly — not surface-projected).
    7. Compute per-axis sigmas from PM-weighted in-parcel positions in the parcel frame; pre-compute `log_sigmas`.
    8. Pack and save as `.npz` with the keys listed in `#10`.
  - Verification checklist (all must pass before the cache is committed):
    - [ ] Smoke test: flat gyral parcel `A4hf` — dominant eigenvalue `λ_1 > 0.9`, z-axis points approximately along `(origin − brain_centroid) / ||·||` (within a few degrees). Visualize as an arrow on fsaverage; manually confirm it looks "up" from the precentral gyrus.
    - [ ] Smoke test: curved gyral parcel `A44d` — z-axis points outward from the IFG ridge; `λ_1` lower than `A4hf`; tangent x-axis roughly along the gyrus long axis.
    - [ ] Smoke test: sulcal / pocket parcel `INSa` — mean normal is small (would be near-zero with the mean method), but second-moment z-axis is well-defined and `λ_1 ∈ [0.5, 0.9]`.
    - [ ] Numerical invariants: `trace(M) ≈ 1` for every parcel (because normals are unit vectors), fail loudly otherwise; `λ_1 ∈ [1/3, 1]`, fail loudly otherwise; rotation matrices are orthonormal to float32 precision; determinant `det(R) = +1` for every parcel (right-handed frame).
    - [ ] Hemisphere purity: every left BNA parcel receives only left-fsaverage vertices and vice versa. Assert at build time, fail loudly on any crossover.
    - [ ] Coverage floor: every parcel in `DEFAULT_BASE_PARCELS` has at least 50 fsaverage vertices contributing (after PM weighting). Below that, the second-moment statistic is too noisy.
    - [ ] Sign-rule coverage: every parcel has `|z · (origin − brain_centroid)| > 1e-3`. Any parcel where this fails is flagged and handled individually (probably a parcel near the brain centre where the outward-normal rule is ambiguous).
    - [ ] Determinism: running the builder twice produces a byte-identical `.npz`. Compared by SHA-256 in CI.
    - [ ] BNA vs fsaverage space sanity: project a known Brainnetome label map (MPM) onto the fsaverage surface and visually confirm that parcel boundaries land where expected on the mesh. This is the end-to-end check that `nilearn.surface.vol_to_surf` is doing what we think it is.
  - No `v14` code reads `parcel_frames.npz` until the verification checklist is green.
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

## Phase 2: Learned Per-Patient Calibration + Modality Join

Only start this after `v14-core` is working end-to-end.

- [ ] Implement gain / offset calibration
- [ ] Implement rigid `Δ/ω` correction with explicit bounds
- [ ] Implement parcel offsets `δ_l`
- [ ] Implement parcel temperatures `τ_l`
- [ ] Add freeze / unfreeze controls for staged optimization
- [ ] Decide whether the first learned calibration step should be gain/offset only or full `Δ/ω`
- [ ] Decide whether any fixed gain / impedance normalization from baseline-only channel statistics is warranted
- [ ] Verify learned calibration improves rather than destabilizes the fixed-atlas baseline

### sEEG modality join

- [ ] **Wire a modality embedding into the shared point encoder** — DECIDED 2026-04-13 (design commitment; implementation in Phase 2 when sEEG actually joins)
  - **Decision**: the shared point encoder takes `[x, y, z, m_embed]`, where `m_embed ∈ R^{d_m}` is a learned per-modality vector. Concat, not FiLM, unless concat under-performs in ablation. `modality_id ∈ {uECoG, sEEG}` for Phase 2; more modalities (external ECoG, MEG, ...) slot in the same way.
  - **Rationale**: a shared MLP from `(x,y,z) → d` trained on mixed data specializes toward whichever modality dominates the batch. A modality embedding lets the same weights carry two regimes — "on uECoG, ignore z because it's constant; on sEEG, use z because it's informative" — without forking into two networks.
  - **Do not make the latent queries modality-specific.** Keeping queries shared forces the summarizer to learn a modality-agnostic notion of "what to extract from a parcel". Per-modality queries effectively splits into two summarizers sharing a backbone and undoes the unification.
  - **Phase 1 implication**: reserve the input slot in the point encoder config now. The alternative is re-training the first layer at the sEEG join, because you cannot safely zero-pad a channel the encoder was never trained with. For Phase 1, `d_m = 0` (no-op) is acceptable only if the point encoder is explicitly marked as "modality-aware, currently one-hot uECoG".
  - **Does not fix** (separate Phase 2 items, each needs its own discussion):
    - support statistic — needs diversity weighting for sEEG (Phase 1 PM-weighted sum over-rewards shaft-internal redundancy; see #5)
    - parcel-frame axes — Phase 1 stores one per-parcel mean cortical normal; sEEG in curved parcels wants per-voxel local cortical normals so that contacts at the gyral crown vs sulcal fundus get different local rotations (see #10)
    - default split map — `DEFAULT_SPLIT_COUNTS` was picked from uECoG coverage; sEEG coverage geometry is different
- [ ] **Re-derive `DEFAULT_SPLIT_COUNTS` from sEEG coverage** when sEEG joins. The elongated-parcel splits in `token_spec.py` come from uECoG reachability across 11 PS patients; the right split set for sEEG is probably different.
- [ ] **Upgrade the support statistic with diversity weighting** when sEEG joins — downweight contacts close to each other in MNI (or close along a shaft axis) so four adjacent sEEG contacts on one shaft count as roughly one probe instead of four. Tracked in #5.
- [ ] **Refine parcel-frame axes from per-parcel mean cortical normal to per-voxel local cortical normals** when sEEG joins. Phase 1's `parcel_frames.npz` stores one `(origin, R)` per parcel, built from the PM-weighted second-moment tensor of fsaverage pial normals. That is correct for uECoG (all contacts on pia; constant cortical depth) and correct-enough for sEEG in flat parcels. For sEEG in curved parcels, a contact at the gyral crown vs a contact at the sulcal fundus should get *different* local rotations — the per-voxel upgrade stores one rotation per PM voxel (or computes one per contact at load time), which is a superset of the Phase 1 cache. Same `.npz` format, extra keys. The Phase 1 cache is a valid fallback. Tracked in #10.

## Deferred Until After uECoG Correctness

- [ ] Extended `uECoG`
- [ ] `sEEG`
- [ ] External datasets
- [ ] Functional pooling variants
- [ ] Broad ablations
