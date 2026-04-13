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

- [ ] **#4 Re-derive the base parcel set and default split map under the Phase 1 contract**
  - The current `DEFAULT_BASE_PARCELS` (16 parcels) and split candidates (`A6cvl`, `A4hf`, `A1/2/3ulhf`, `A2`, `A1/2/3tonIa`, with `A4tl` as the extension candidate) in `src/speech_decoding/v14/token_spec.py` were computed under the **quarantined v12 / centroid-VE pipeline**: node routing with 25/15 mm reachability thresholds, over all 11 PS patients including S22, S58 (RH), S32 (no HG), S57 (hybrid strip). None of that methodology survives into Phase 1 — the new contract is volumetric Brainnetome PM membership (`#5` DECIDED), the Phase 1 patient set excludes RH (`#30` DECIDED) plus S32 and S57, and reachability thresholds are gone. The provisional ranking is untrustworthy until recomputed under the Phase 1 contract.
  - This is not just an ablation detail — the chosen set determines:
    - `N_tok` (the shared spatial interface)
    - the local summarizer query counts `K_l` (`#26`)
    - the token-level connectivity bias expansion (`#8`)
    - low-coverage behavior for important motor/sensory parcels
  - **Action: recompute parcel × electrode coverage from scratch under the Phase 1 contract.**
    - **Membership rule**: volumetric Brainnetome PM from `data/atlas/BNA_PM_4D.nii.gz`. No centroid routing. No reachability thresholds.
    - **Coverage metric**: PM-weighted sum per `#5` — `support[parcel, patient] = Σ_i PM(parcel | x_i)` over non-artifact contacts `i` for that patient.
    - **Effective Phase 1 patient set**: `S14, S16, S23, S26, S33, S39, S62` (7 patients) — all 11 minus S22/S58 (RH, `#30`), S32 (no HG response), S57 (hybrid strip). Core four are `S14, S26, S33, S62`.
    - **Upstream dependencies**: `#1` (ACPC→MNI transform re-verified) and `#12` (amp → physical-electrode → ACPC bridge verified per patient). Without both, the coverage numbers are garbage-in-garbage-out.
    - **Outputs**:
      - a per-parcel × per-patient coverage matrix over *all* BNA LH parcels (not a pre-filtered shortlist) so the ranking is reproducible and auditable
      - a "patients reached" count per parcel under a declared threshold (e.g. `support ≥ 0.5` = "at least half an effective contact inside the parcel"), plus the raw distribution so the threshold is tuned against data, not pre-committed
      - a ranked list by reachability, ties broken by mean coverage across reached patients
      - the final Phase 1 base parcel set, chosen from the ranking with an explicit coverage-vs-anatomical-relevance tradeoff
    - **Artifacts**: a `reports/phase1_parcel_coverage_2026_04_13.md` report plus the per-patient × per-parcel coverage matrix cached under `data/atlas/` as `.npz` or CSV.
  - Discussion items to resolve *after* the re-derived ranking exists:
    - **Base parcel count**: 16 is a historical artifact of the old ranking. The right number for Phase 1 could be 12, 16, 20, or whatever the coverage distribution suggests. Default expectation: the smallest set where every core patient (S14, S26, S33, S62) has at least 8 reached parcels, so each contributes meaningfully to inter-region attention. Revisit after seeing the numbers.
    - **Default split map**: re-derived from the same coverage matrix. A parcel gets a 2-token split only if (a) it is in the base set *and* (b) it is elongated enough in cortical-normal parcel-frame space that a single-token summary loses real structure. Elongation metric comes from the parcel-frame sigmas in `parcel_frames.npz` (`#10`) — ratio `max(σ_x, σ_y) / min(σ_x, σ_y)` above a data-driven threshold. Threshold is chosen from the distribution, not guessed.
    - **Anatomical inclusion overrides**: the mechanical ranking may omit speech-critical parcels that happen to be under-covered in the Phase 1 set (e.g. STGa, INSa). Decide whether an anatomical override is acceptable and, if so, the rule — e.g. "include any parcel with ≥ 2 core patients reached *and* a named speech role from the literature" — or keep it purely mechanical. Default expectation: pure mechanical for the first frozen list; anatomical overrides are an ablation.
    - **Right-hemisphere future-proofing**: when Phase 2 opens RH patients, does the base set expand to include RH counterparts under a hemisphere-agnostic parcel index, or is a fresh RH-side ranking derived independently? Tied to the `#30` Phase 2 re-opens note.
    - **`token_spec.py` provisional flag**: `DEFAULT_BASE_PARCELS` and `DEFAULT_SPLIT_COUNTS` should stay explicitly marked as `# PROVISIONAL — pre-Phase 1, pending #4 re-derivation` until this blocker closes. Any v14 code path that reads them should assert the provisional flag is cleared before use.
  - No `N_tok`, no `K_l`, no token-level connectivity bias initialization, and no v14 summarizer code lands until the re-derivation has run and its outputs are reviewed.

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

- [ ] **#26 Lock the within-parcel cross-attention contract (electrode → parcel tokens)**
  - This is the one place where per-patient electrode space becomes shared atlas-token space. Every downstream decision depends on its exact input shape, output shape, and how the support statistic enters. It must be frozen before any summarizer code is written. Discussion-first.
  - Dependencies: `#2` / `#6` (temporal front-end output contract — sets the per-electrode feature dim `d` and token rate `T`), `#5` (support statistic, DECIDED), `#7` (how `token_support` enters the model, DECIDED), `#10` (parcel-frame construction, DECIDED), `#11` (channel inclusion policy).
  - Discussion items to resolve:
    - **Per-electrode input feature**: exact composition of each within-parcel token passed to cross-attention. Candidate: `[h_i(t) ∈ R^d, parcel_frame_coord_i ∈ R^3, log_sigma_parcel ∈ R^3]` with a shared point encoder `R^{d+3+3} → R^d`. Alternatives: `[h_i(t), coord_i]` only (size lives on the parcel token, per `#10`), or `[h_i(t), coord_i, modality_embed]` (reserved slot from Phase 2 sEEG join — decide if the slot is live in Phase 1).
    - **Time-axis handling**: does cross-attention run (a) once per time step `t` with queries attending over `N_p` electrodes at that `t`, (b) once over the full `(N_p × T)` flattened set, or (c) per-electrode temporal pooling first, then cross-attention over `(N_p, d)`? Option (a) keeps `T` downstream; option (b) is the most expressive but scales as `N_p · T`; option (c) collapses time early and breaks the `(N_tok × d × T)` backbone contract. Default expectation: (a).
    - **Latent query design**: how many queries per parcel (`k = 1` for single-token parcels, `k = 2` for split parcels per `#4`)? Are queries (i) a single shared set reused across all parcels, (ii) one shared set per split-count `k`, or (iii) parcel-specific? Default expectation: (ii) — one shared `k=1` query and one shared `k=2` query pair, reused across parcels. Parcel identity enters only through parcel-frame coordinates and `log_sigma`, not through query identity.
    - **Query count vs `N_tok`**: confirm that `Σ_parcel k_parcel = N_tok = 21` under the current `DEFAULT_SPLIT_COUNTS`. This is the invariant that ties the summarizer to the backbone.
    - **Attention mechanics**: number of heads, head dim, pre-norm vs post-norm, residual path (there is no natural residual because queries and keys live in different spaces — default: no residual, norm on query and key separately before dot product). Dropout policy.
    - **How `token_support` enters here vs later**: per `#7`, the decision was concat-to-token at the *parcel-token* level (after summarization), not at the electrode level. Confirm this means:
      - no per-electrode support weighting inside the cross-attention softmax
      - no query-level support injection
      - `token_support[parcel]` is concatenated onto the summarizer output `(k, d) → (k, d+1) → linear → (k, d)` before the backbone sees it
      - alternative considered and rejected: weighting the attention softmax by `PM(parcel | x_i)` per electrode. Rejected because it double-counts with the PM-weighted parcel assignment already done at membership time, and it suppresses lone-electrode parcels (the exact case `#7`'s concat-to-token is meant to handle).
    - **Electrode-side PM handling**: each electrode `i` has `PM(parcel | x_i) ∈ [0, 1]` from the BNA PM volume. Decide: (a) hard assignment — each electrode goes to its argmax parcel only; (b) soft assignment — each electrode enters every parcel's summarizer with weight `PM(parcel | x_i)`; (c) top-k assignment with `k=2` or `k=3`. Default expectation: (b) soft, because hard assignment throws away exactly the PM information the atlas provides. Implementation: the key/value tensor for parcel `p` is the list of electrodes with `PM(p | x_i) > ε`, with the PM value concatenated onto the input feature (or used as a pre-softmax bias — decide which).
    - **Output shape into the backbone**: the summarizer must emit exactly `(N_tok, d, T)` with `N_tok = 21` tokens in a canonical order, plus `token_mask ∈ {0,1}^{N_tok}` and `token_support ∈ R^{N_tok}`. Fix the canonical parcel ordering now — alphabetical by Brainnetome ID, with split-parcel sub-tokens adjacent and in a deterministic order.
    - **Masking and absent parcels**: if a parcel has zero contributing electrodes for a patient, what does the summarizer emit? Default: zero vector `(k, d)` plus `token_mask = 0`. Decide whether the zero is structurally produced (sum over empty set) or defensively inserted (assertion + fallback). These have different failure modes for hot paths.
    - **Parameter count budget**: rough target for the summarizer parameters. The point encoder (~`(d+3+3) × d × 2` layers), the latent queries (`k × d` each), and the cross-attention projections (`4 × d^2` per head). Confirm this is small relative to the backbone — the summarizer should not dominate the shared-parameter count, since it is applied independently per parcel.
  - Artifacts produced once this is frozen:
    - exact tensor signature for `LocalSummarizerConfig` and the forward contract, written into `src/speech_decoding/v14/local_summarizer.py` docstring
    - a shape/contract test under `tests/v14/` that constructs a dummy `(N_p, d, T)` + `(N_p, 3)` input for a fixed parcel and asserts the output is `(k, d, T)` with the expected `k` for both single-token and split-token parcels
    - a one-active-parcel test and a zero-electrode parcel test (ties into the masking item above)
  - No code in `src/speech_decoding/v14/local_summarizer.py` beyond the current stub until every bullet above is resolved.

- [ ] **#27 Lock the inter-region attention backbone contract**
  - This is the shared dynamics stack that operates on parcel tokens `(N_tok × d × T)` between the summarizer (`#26`) and the AR decoder. Every block count, every attention-axis decision, and every normalization choice is an explicit discussion item. Nothing is pre-committed. Discussion-first.
  - Dependencies: `#26` (input shape / token semantics coming in), `#8` (token-level SC/FC connectivity expansion for attention bias), `#9` (decoder contract — sets what the backbone must hand off), `#6` (temporal front-end output `T` and time-axis semantics).
  - Discussion items to resolve:
    - **Attention factorization**: joint spatiotemporal attention over `(N_tok · T)` tokens vs factored `(spatial over N_tok) → (temporal over T)` alternating blocks vs factored `(temporal) → (spatial)` vs a single-block with axial attention. Joint is the most expressive but scales as `(N_tok · T)^2` memory, which for `N_tok = 21` and e.g. `T ≈ 64` is `1344^2 ≈ 1.8M` entries per head per sample — not ruinous but not free. Factored is `N_tok^2 + T^2` which is two orders of magnitude smaller but can miss diagonal space-time structure. Default expectation: factored alternating spatial-then-temporal, because `N_tok` is small and spatial connectivity is informative (SC/FC bias) while temporal attention needs its own positional structure.
    - **Block count `B`**: how many (spatial, temporal) block pairs. Candidates: `B = 1`, `B = 2`, `B = 3`, `B = 4`. Default expectation: `B = 2` as the first frozen choice, with `B ∈ {1, 3}` as the first ablation sweep. Pre-committing to a specific `B` requires a stated reason (parameter budget, overfit risk on ~11 min of data per patient, etc.).
    - **Per-block structure**: pre-norm vs post-norm; FFN presence, width, and activation; residual connections on both attention and FFN (or only attention); dropout on attention weights, on residuals, on FFN; layer scale / init scale. Default expectation: pre-norm, `GELU` FFN at `4d` width, residual on both, dropout `0.1` on attention and FFN, no layer-scale. Every one of these is still open.
    - **Heads and head dim**: number of heads for spatial attention, number for temporal attention. These do not have to match. Default expectation: `d / head_dim` with `head_dim = 32` or `64`, same for both axes until an ablation says otherwise.
    - **Spatial attention bias from Brainnetome SC/FC**: resolved jointly with `#8`. Key sub-items that belong in `#27`, not `#8`: how the bias enters the softmax (additive logit vs gating), whether it's learnable (scalar gain per bias matrix) or fully fixed, and whether it's shared across heads or per-head. Default expectation: additive logit, learnable scalar gain initialized at `1.0`, shared across heads.
    - **Temporal attention positional structure**: time carries ordering, so spatial softmax is permutation-invariant but temporal softmax is not. Decide the positional encoding: (a) absolute sinusoidal PE added to token features, (b) relative position bias in the temporal softmax (T5-style), (c) rotary position embedding on temporal `q` and `k`, (d) learned absolute PE. Default expectation: (c) RoPE on temporal attention only — rotary is parameter-free, handles variable `T` cleanly, and doesn't add capacity to the spatial path where it would be meaningless. Explicitly rejecting Fourier PE on electrode positions (that was a stale v12 marker and the design doc already says "no Fourier PE").
    - **Masking and `token_mask` propagation**: the spatial softmax must mask out tokens where `token_mask[j] = 0` (absent parcel for this patient). The temporal softmax inside an absent-parcel row does not need to run at all — decide whether we skip those rows (efficient but branchy) or run them and zero their outputs (simple but wastes compute). Default expectation: run and zero, because `N_tok` is small enough and branchy code is harder to verify than a zero mask.
    - **`token_support` role inside the backbone**: per `#7`, `token_support` is concatenated onto each token feature at entry (before block 1) via `[d] ⊕ [1] → linear → [d]`. Decide whether this concat happens once at entry or re-injected at each block. Default expectation: once at entry; re-injection is an ablation. Attention-bias from support (the rejected-in-`#7` variant) stays off the baseline.
    - **Readout / handoff to decoder**: what does the backbone hand to the AR decoder in `#9`? Candidates: (a) the full `(N_tok, d, T)` tensor, decoder attends freely over `N_tok · T` keys; (b) a pooled `(d, T)` tensor via token-axis mean with mask; (c) a pooled `(N_tok, d)` tensor via time-axis mean; (d) the full tensor plus a pooled summary token. Default expectation: (a) full tensor, because the decoder is explicitly cross-attention and `N_tok · T` is small. `#27` must state this so `#9` can use it.
    - **Parameter count budget**: rough target for the backbone alone. With `d = 256`, factored attention at `B = 2`, `4d` FFN, `head_dim = 64`, the backbone is ~1.2M–2M parameters. Confirm this is the right scale relative to the 11 min / patient data regime. If not, the block count, width, or FFN expansion ratio must shrink before coding.
    - **Initialization**: attention projections (`xavier_uniform` vs `kaiming_uniform` vs small-scale `normal`), FFN init, bias init. Default expectation: PyTorch `MultiheadAttention` defaults are fine; only revisit if training is unstable.
    - **Gradient checkpointing**: whether to checkpoint spatial and temporal blocks. Not a correctness question, but it changes the max viable `B` and `T`. Decide after the memory envelope on DCC is measured — put off until the first training run.
  - Artifacts produced once this is frozen:
    - exact tensor signature for `BackboneConfig` and the forward contract, written into `src/speech_decoding/v14/backbone.py` docstring
    - a shape / contract test under `tests/v14/` that constructs a dummy `(B_batch, N_tok, d, T)` input plus `token_mask` and asserts the output is `(B_batch, N_tok, d, T)` with masked-token outputs zero-invariant (attention weights into masked tokens are zero, output slices at masked rows are zero)
    - a factored-vs-joint toggle that is exercised by unit test before any training run
    - a block-count sweep harness (even if just a config override) so `B ∈ {1, 2, 3}` can be run without editing code
  - No code in `src/speech_decoding/v14/backbone.py` beyond the current stub until every bullet above is resolved.

- [ ] **#28 Lock the AR cross-attention decoder contract**
  - The rough shape is known — "3 AR-conditioned decode queries attend over `N_tok · T`" from the design doc — but every fine detail that actually governs training and eval is still open. Discussion-first.
  - Dependencies: `#9` (supervised training contract — the loss and teacher-forcing policy live here), `#16` (phoneme label → integer index contract), `#17` (PS vowel → ARPABET fix — blocks any phoneme-level target), `#26` / `#27` (defines what the decoder cross-attends over), `#21` (canonical 52-token enumeration — end-of-sequence semantics).
  - Discussion items to resolve:
    - **Target length and target units**: PS tokens are CVC or VCV triples, so the decode target is always 3 phonemes. Decide whether the decoder is (a) fixed-length 3-step with no `<eos>`, (b) variable-length with `<eos>` terminator, or (c) fixed-length 3-step plus a parallel whole-token classification head over the 52 PS tokens as auxiliary supervision. Default expectation: (a) fixed-length 3, no `<eos>`, because the task guarantees length 3 and a learned terminator would only add noise on this data regime. (c) is attractive but belongs in an ablation, not the baseline.
    - **Query count and initialization**: three queries, one per phoneme slot. Decide whether the three queries are (i) independently learned vectors (`q_0, q_1, q_2 ∈ R^d`) with no shared structure, (ii) a single learned query `q` plus a learned slot-position embedding `p_i` added to it, or (iii) all three tied to a single learned query with the AR conditioning doing all the slot disambiguation. Default expectation: (ii) — shared query plus slot embedding. It's the smallest capacity choice that still distinguishes the three slots cleanly, and it doesn't force the first phoneme's query to do double duty as a no-context prior.
    - **Autoregressive conditioning mechanism**: what does "AR-conditioned" actually mean for query `q_i` when `i > 0`? Candidates: (a) add the embedding of the predicted phoneme `y_{i-1}` to `q_i` before cross-attention; (b) prepend a causal self-attention layer where `q_i` attends to `q_0, ..., q_{i-1}` plus their phoneme embeddings; (c) concatenate `y_{i-1}` embedding to `q_i` and linear-project back to `d`. Default expectation: (b) causal self-attention with phoneme embeddings, because it is the minimal change that lets the decoder carry soft uncertainty from slot `i-1` to slot `i` rather than committing to an argmax. (a) commits to a hard previous-token. (c) loses multi-step context.
    - **Causal self-attention depth**: if (b) is chosen, how many self-attention layers run before cross-attention? Candidates: 1 or 2. Default expectation: 1, because the history length is 2 (at most `y_0, y_1` when decoding slot 2) and deeper self-attention over length 2 buys nothing.
    - **Cross-attention keys / values**: cross-attention reads from the backbone output, which is `(N_tok, d, T)` per `#27` default. Decide whether keys and values are flattened to `(N_tok · T, d)` with no additional positional encoding, or whether a summary positional tag (parcel id, time index) is injected into keys before cross-attention. Default expectation: flat `(N_tok · T, d)` with no extra positional tag — RoPE on the backbone's temporal axis and SC/FC bias on the backbone's spatial axis already bake in the structure, and the decoder side just needs content.
    - **Cross-attention heads and layers**: how many cross-attention layers per decoder step, and how many heads each. Default expectation: 1 cross-attention layer, heads matching backbone (`head_dim = 64`). Multiple cross-attention layers on top of 1 self-attention layer is a classic transformer-decoder-block pattern but is overkill for length-3 targets on this data regime.
    - **`token_mask` propagation into cross-attention**: absent parcels must be masked out of the decoder's cross-attention softmax the same way they're masked in the backbone. Confirm the mask comes in as a single `(N_tok,)` vector broadcast across time, not a per-`t` mask. This should just work if `#27`'s mask propagation is correct, but it needs a unit test at the decoder boundary.
    - **Output head**: linear `d → |V|` where `|V|` is the PS phoneme vocabulary size from `#16`. Decide whether the output head is (i) a single shared linear over all three slots or (ii) three separate linears. Default expectation: (i) shared — three separate heads would triple parameters for no reason and break slot-position invariance of the output projection.
    - **Loss**: cross-entropy per slot, summed over the three slots, averaged over the batch. Decide whether to (a) use plain CE, (b) add label smoothing, (c) use focal CE (γ=2) which helped in the per-patient baseline. Default expectation: plain CE for the first frozen version, because label smoothing and focal CE are both wins on the old baseline but they're separate discussion items and should be turned on one at a time. Tied to `#9`.
    - **Teacher forcing vs scheduled sampling vs free running**: at train time, when computing query `q_i`, is the previous token `y_{i-1}` fed from the ground-truth label (teacher forcing), from the model's own argmax prediction (free running), or mixed? Default expectation: pure teacher forcing at train time. Scheduled sampling is a discussion item for later; the ~11 min / patient regime is too small to risk a second source of training noise. Tied to `#9`.
    - **Eval decoding**: greedy argmax vs beam search. For length-3 targets over a 9-phoneme vocabulary, the search space is 729 — fully enumerable. Default expectation: greedy at eval because beam adds nothing when length is fixed at 3; the length-3 exhaustive enumeration is an ablation worth noting but not the baseline. The grouped-by-token CV metric reads per-slot argmax accuracy (PER) directly.
    - **Train-time vs eval-time decoding relationship**: confirm that train-time teacher-forcing and eval-time greedy produce the same query sequence structure (same three slots, same slot embeddings, same cross-attention). The only difference is the source of `y_{i-1}`. Tied to `#9`.
    - **Position-of-slot metadata**: whether the slot index `i ∈ {0, 1, 2}` is exposed to the output head (beyond the query). Default expectation: no — the slot embedding enters the query, which already differentiates the three positions inside the decoder. Additional slot conditioning on the output head is an ablation.
    - **Parameter count budget**: shared query `d` + three slot embeddings `3 · d` + one causal self-attention layer + one cross-attention layer + output head `d × |V|`. Rough target: under 1M parameters. Confirm against the summarizer (`#26`) and backbone (`#27`) so the three components together are balanced.
    - **Failure modes to name now**:
      - decoder collapses to mode prediction (always predicts the most common phoneme triple). Mitigation is loss weighting or label smoothing — discussion item for `#9`, not decided here.
      - decoder ignores cross-attention because the slot embedding alone is enough to reach chance. Mitigation is a cross-attention drop-ablation (zero the cross-attention output and verify the PER goes to floor).
      - decoder leaks ground-truth through teacher forcing and eval-time PER is misleading. Mitigation is the free-running sanity check at eval — confirm it matches the teacher-forced eval within noise.
  - Artifacts produced once this is frozen:
    - exact tensor signature for `DecoderConfig` and the forward contract, written into `src/speech_decoding/v14/decoder.py` docstring
    - a shape / contract test under `tests/v14/` that constructs a dummy `(B_batch, N_tok, d, T)` + `token_mask` input and asserts the decoder output is `(B_batch, 3, |V|)`
    - a teacher-forcing / free-running equivalence test: running the decoder with ground-truth previous tokens should produce the same shape as running it auto-regressively with its own predictions, and both should respect `token_mask`
    - a cross-attention-zeroed ablation harness so the cross-attention-ignored failure mode above can be ruled out on the first training run
  - No code in `src/speech_decoding/v14/decoder.py` beyond the current stub until every bullet above is resolved.

- [x] **#29 Input window / epoching contract — DECIDED 2026-04-13**
  - **Decision**: response-onset-locked full-trial epoch. Not MFA per-phoneme. Window `tmin = -0.5 s`, `tmax = 1.0 s`. One sample = one trial = three phonemes as the decoder target.
  - **Rationale**: the `#28` AR cross-attention decoder is designed around a single continuous `N_tok · T` key set covering the full trial. Per-phoneme MFA epochs were optimal for the old Conv2d per-phoneme baseline but break the 3-slot decoder's cross-attention contract. Passing the whole trial is what the v14 decoder was designed for. The `-0.5 s` pre-onset window catches late stimulus-listening and motor-planning activity (auditory stimulus ends ~600 ms before response onset, per `CLAUDE.md`); the `1.0 s` post-onset tail covers the ~450 ms utterance plus a post-articulation buffer.
  - **Open sub-item**: `tmin` may shorten after looking at the data more carefully. `-0.5 s` is the frozen default for the first loader build, but the possibility of a tighter `tmin` (e.g. `-0.3 s` or `-0.2 s`) is left explicitly open as a one-conversation revisit — not a blocker. If it changes, `#29` gets amended in place; downstream contracts (`#28`, `#31`) do not depend on the exact `tmin` value.
  - **Tie-ins**:
    - `#9` supervised training contract: the "input window" sub-item is now closed; the remaining sub-items (target semantics detail, decoder training behavior, train/eval decoding parity) stay open inside `#9` / `#28`.
    - `#20` MFA trust posture: no longer gates v14 input alignment (input is response-locked, not MFA-aligned). MFA remains relevant only if it's used to order phoneme *targets* within a trial.
    - `#28` AR decoder: confirms "three slots = three phonemes of one trial" as the target structure. Already compatible.
    - Sample count per patient = non-artifact trial count directly (~46–178 per patient), *not* multiplied by slot count. This matters for `#31` batching and the optimizer-schedule short-discussion item.

- [x] **#30 Phase 1 right-hemisphere exclusion — DECIDED 2026-04-13**
  - **Decision**: Phase 1 excludes right-hemisphere patients (S22, S58) from training and evaluation. Core set is unchanged (S14, S26, S33, S62 — all LH). Phase 1 extended set is LH-only: S16, S23, S39. S22 and S58 are deferred to Phase 2 alongside the sEEG modality join.
  - **Rationale**: right-hemisphere routing to RH Brainnetome parcels is a real design decision with downstream consequences for `N_tok` semantics, hemisphere-agnostic vs per-hemisphere parcel index, and whether the backbone ever sees mixed-hemisphere batches. Rather than answer that now, leave RH patients out of the first correctness pass. The quarantined `mirror_to_left()` shortcut is wrong for volumetric membership and stays quarantined.
  - **`#10` implication**: `parcel_frames.npz` still builds both hemispheres by construction (the builder runs per-hemisphere and the RH cache is free). RH data is simply never loaded in Phase 1.
  - **Phase 2 re-opens**: decide whether RH patients use a shared LH∪RH parcel index, a disjoint RH parcel set, or a hemisphere flag on the modality embedding (tied to Phase 2 sEEG join). Added to the Phase 2 section below.
  - **CLAUDE.md implication**: the "Extended" patient set listed in `CLAUDE.md` currently includes `S22, S58`. The Phase 1 effective extended set is `S16, S23, S39`. Update `CLAUDE.md` to reflect this when the next substantive edit lands; not urgent on its own.

- [ ] **#31 Patient-mixing and batching policy**
  - This is the training-loop structure question: how samples from different patients compose inside a batch, and whether the first supervised `v14-core` run is per-patient or joint across core patients. Discussion-first.
  - Dependencies: `#13` (v14 loader semantics), `#29` (one-sample-per-trial epoching), `#30` (LH-only Phase 1 patient set), `#11` (channel inclusion policy — sets `N_ch` per patient).
  - Discussion items to resolve:
    - **First-run structure: per-patient or joint-across-patients?** The design doc says "shared dynamics" (joint), but "first overfit sanity" points per-patient. Candidates: (a) per-patient baseline first (four separate models, one per core patient), then joint as the second experiment; (b) joint across all four core patients from day one; (c) joint with per-patient fine-tuning as an ablation. Default expectation: **(a) per-patient first**, because the first correctness pass needs to beat PER 0.734 ± 0.007 on S14 specifically, and mixing in other patients makes that comparison noisier. The shared-dynamics claim is the *second* experiment.
    - **Variable `N_ch` across patients in a joint batch**: core four have 128, 128, 256, 256 contacts. Candidates: (i) pad to `max(N_ch)` with electrode-level mask; (ii) per-patient micro-batches inside a gradient-accumulation step; (iii) grouped-by-patient sampler so every batch is one patient at a time; (iv) jagged/variable-length tensors. Default expectation: **(iii) grouped-by-patient sampler** — simplest, no padding, aligns with per-patient-first. (i) is the right answer if and when joint batches are needed.
    - **`token_mask` heterogeneity within a batch**: even under grouped-by-patient, `token_mask` is per-patient-constant. Decide whether the loader emits `token_mask` per-sample or per-patient-at-load-time. Default expectation: **per-sample** `(B, N_tok)` — downstream code (summarizer, backbone, decoder) expects a per-sample mask tensor regardless. Redundant but simple.
    - **Sampler shuffling and epoch semantics**: (a) patients round-robin, trials shuffled within a patient; (b) patient-order shuffled each epoch, trials shuffled within a patient; (c) all trials of patient 1 before any of patient 2. Default expectation: **(b)**. (c) concentrates updates per patient and undoes joint training in the joint-run phase.
    - **Batch-size semantics**: is "batch size = B" measured in trials or patients? Under grouped-by-patient, it's trials per patient per batch. Decide fixed `trials_per_batch` (variable batch count per patient, patient with 178 trials gets more batches than patient with 46) vs one batch per patient per epoch (variable size). Default expectation: **fixed trials per batch**, variable batch count per patient.
    - **Per-patient trial count imbalance**: core patients span ~46–178 trials (~4× range). Loss weighting: each trial equal vs each patient equal. Default expectation: **each trial equal** (plain averaging). Patient-weighted loss is an ablation.
    - **CV fold ↔ batching interaction**: grouped-by-token CV splits the 52 tokens into train and test groups *per patient*. Confirm CV splitting happens at the trial-selection layer (per patient) and batching happens on the CV-filtered trial list, not before. Easy to get wrong silently.
    - **Joint-batch future compatibility**: even if the first run is per-patient, the sampler interface must let us swap "grouped by patient" for "mixed patients" without rewriting the loader. Confirm the interface supports both via a sampler strategy flag.
    - **Gradient accumulation policy**: if the DCC memory envelope forces `trials_per_batch` below the statistical floor, decide gradient accumulation to a fixed effective batch size vs smaller batches with more frequent steps. Default expectation: **accumulation to a fixed effective batch size**, because the effective-batch invariant matters for the lr schedule. Tied to the optimizer-schedule short-discussion item.
  - Artifacts produced once this is frozen:
    - exact signature for the v14 `DataLoader` / sampler contract, written into the loader module docstring
    - a shape/contract test under `tests/v14/` verifying: per-sample `token_mask` emission; per-patient batches have constant `N_ch`; patient-order shuffling deterministic under a seed; CV-filtered trial counts sum correctly
    - a per-patient-run harness that can produce the S14 run under baseline-comparison conditions
  - No code in the v14 data loader or training loop beyond the current stubs until every bullet above is resolved.

- [x] **#32 Input normalization at the v14 boundary — DECIDED 2026-04-13**
  - **Decision**: no additional normalization beyond upstream `productionZscore_highgamma`. The loader hands the model upstream-z-scored HGA features directly.
  - **Rejected alternatives** (named so they stay rejected):
    - per-sample re-normalization — throws away session-level calibration
    - per-batch re-normalization — leaks distribution info across trials in a batch
    - per-patient re-normalization at load time — redundant with upstream z-score
  - **Implication**: if the point encoder or temporal front-end (`#2` / `#6`) benefits from a `LayerNorm` at its input, that is an in-model architectural choice, not a data-side normalization. Discuss inside `#2` / `#6` if needed.

- [ ] **#34 End-to-end phoneme loading and trial-timing audit**
  - The pre-v14 loader path (`bids_dataset.py`, `load_patient_data`, `load_per_position_data`) was sloppy and never end-to-end verified. It trusted the upstream `.fif` annotations and the `phoneme_map.py` hardcoded dict without ever asserting either against the actual BIDS events or the audio. Every step from `.fif → (signal window, 3-phoneme target)` has to be re-verified from scratch before a v14 loader is written. This is the **operational audit** that the `#17`–`#25` discussion items assume but never execute.
  - Dependencies: `#17` (PS vowel ARPABET fix — load-bearing for any symbol-level check), `#21` (canonical 52-token list — the ground-truth target), `#29` (response-onset window is already frozen, so "what is t=0" must be answered for every patient).
  - Discussion items to resolve, each requiring *looking at actual data*, not reading upstream docstrings:
    - **What is `t=0` (response onset) for every Phase 1 patient?** Candidates: (a) mic onset from the audio stream; (b) first-phoneme onset from MFA; (c) button-press or stimulus-end marker; (d) an upstream-computed "response onset" annotation baked into the `.fif` file by the Cogan Lab pipeline. The `.fif` path is `...desc-productionZscore_highgamma.fif` — confirm which convention the annotations use and that it is identical across patients. A different `t=0` convention on different patients silently shifts the window by hundreds of ms and corrupts the comparison against the `0.734 ± 0.007` baseline.
    - **Trial identification in BIDS events**: read each patient's `events.tsv` and `.fif` annotations, enumerate trials, and confirm each trial has exactly three phoneme events. Report any patient where the count is not 3 per trial. The "3 consecutive events = 1 trial" assumption (`#19`) is the thing being verified here — this blocker is the execution, not the discussion.
    - **Phoneme ordering within a trial**: for each trial, read the three event symbols in order and compare against the canonical 52-token list (`#21`). Every trial should map to exactly one of the 52 tokens. Report any trial whose symbol triple does not appear in the canonical list. This catches: (i) out-of-order phoneme events; (ii) silent dropped phonemes upstream; (iii) unexpected symbol variants; (iv) mislabeled trials.
    - **`event_id` → ARPABET symbol verification on actual data**: open each patient's `.fif` and read the `event_id` dictionary from the annotations. Compare against the hardcoded mapping `{'a':1, 'ae':2, 'b':3, 'g':4, 'i':5, 'k':6, 'p':7, 'u':8, 'v':9}` in `CLAUDE.md`. Assert equality per patient. Report any mismatch. This is `#18` executed.
    - **Cross-task / cross-patient `event_id` consistency** (`#22` executed): compare the `event_id` dict across every PS patient and across PS vs Lexical. Flag any patient whose `event_id` diverges. The loader must refuse to operate on a divergent patient until the discrepancy is explained.
    - **MFA boundary spot-check against audio**: for a handful of trials per patient (5–10 is enough), read the MFA phoneme boundaries and the audio waveform side-by-side. Confirm the boundaries land where a human listener would put them. This is `#20` executed — the outcome is a trust rating for MFA per patient, not a per-trial correction.
    - **Response-onset alignment spot-check**: for the same trials, confirm that `t=0` corresponds to what the chosen convention claims it is. If the convention is "mic onset," run a waveform-energy check at `t=0` on the audio. If the convention is "MFA first-phoneme onset," confirm against the MFA boundaries. Report the measured offset distribution per patient.
    - **Does `#29`'s `[-0.5, 1.0] s` window actually land where we think it does?** Take a representative trial per patient, plot the response-aligned window on top of the audio envelope, and confirm: the utterance ends well before `tmax = 1.0 s`; the pre-onset tail at `tmin = -0.5 s` sits inside the post-stimulus motor-planning period; no adjacent trial bleeds into the window. If the answer is "no" for any patient, either that patient is excluded or `tmin` / `tmax` is revisited (`#29` re-opens).
    - **Trial counts per patient vs reported totals**: the `CLAUDE.md` number is 46–178 trials per patient. Count the actual trials in each `.fif` after loading and compare. Report every discrepancy. This catches: (i) upstream drops we never noticed; (ii) off-by-one indexing; (iii) cross-task contamination where Lexical trials leaked into the PS file.
    - **Bad-trial exclusion upstream**: find out whether the upstream `productionZscore_highgamma` pipeline has already dropped any trials (artifacts, alignment failures, audio problems). Read the BIDS derivative provenance if it exists; otherwise ask Zac. The v14 loader should know exactly which trials are *present* and which are *silently absent*, and should not introduce a *second* exclusion path on top of the first.
    - **Silence / rest epoch contamination**: confirm that no silence or rest epochs are sitting in the PS `.fif` file alongside real trials. If they are, decide how the loader identifies and rejects them.
    - **Per-patient audio availability for spot checks**: confirm the raw audio files are actually accessible on DCC or locally for the spot-check steps above. If the audio path is broken for any patient, that patient cannot complete the audit and has to be flagged.
    - **Are `event_id` integers used for anything beyond label lookup?** The legacy loader sometimes used the integer as the class index directly. Confirm whether `event_id = 1` for `"a"` is a stable contract or an accident of file creation order, and bake the answer into `#16` (phoneme label → integer index contract).
    - **MNE vs BIDS drift**: confirm that reading the `.fif` via `mne.read_epochs` yields the same annotations as reading `events.tsv` directly. Report any drift. The legacy code read from `mne` only; a second, independent read from `events.tsv` is the cheapest cross-check we can do.
  - Artifacts produced once this audit completes:
    - a single `reports/phoneme_audit_2026_04_13.md` that lists, per Phase 1 patient, the verdict on each of the items above (✓ clean, ⚠ discrepancy with a note, ✗ blocker)
    - per-patient summary: trial count, phoneme inventory, `t=0` convention, MFA trust rating, exclusion-path delta vs upstream
    - one or more minimal-repro scripts under `scripts/v14_audit/` (fresh, not reused from legacy) that can be re-run after any upstream pipeline change
    - explicit "closed" annotations on `#18`, `#19`, `#20`, `#22`, and any other `#17`–`#25` items whose operational question is resolved
    - a go/no-go decision per patient for Phase 1 inclusion (any patient with unresolved audit findings is deferred until resolved)
  - No v14 data loader is written until the audit report exists and every Phase 1 core patient (S14, S26, S33, S62) has a ✓ or a discussed ⚠ with an explicit workaround. The extended set (S16, S23, S39) can audit in parallel.
  - **Rule of thumb**: if any item above would have caused a wrong-but-silent training run in the legacy code, it is a real blocker. Do not bypass any item because "it probably works."

- [x] **#33 PER metric exact definition — DECIDED 2026-04-13**
  - **Decision**: slot-averaged PER. `PER = 1 − (correct slots / total slots)`, computed over all three slots of every trial, averaged over all trials in the held-out fold. Matches the old per-phoneme baseline `0.734 ± 0.007` so the comparison against v14 is apples-to-apples.
  - **Reporting**: per-patient PER and a population mean. 3-seed aggregation: mean ± std across seeds. Population mean is reported after per-patient numbers, not instead of them.
  - **Alternative considered and rejected**: trial-level "any slot wrong" accuracy. Harder to compare against the baseline and discards information about where errors land.
  - **Diagnostic only (not headline)**: per-slot PER (slot 0 / 1 / 2 separately) is reported for the first training run to check for slot-position bias, then dropped from headline reporting unless a bias shows up.

## Short Discussion Items (freeze before the first training run)

These are lighter-weight than full blockers. Each should take one conversation turn and produce a one-paragraph commitment, not a multi-page contract. They are listed here so they don't get forgotten in the rush to the first run.

- [ ] **Grouped-by-token CV protocol for v14.** Fold count; seed count (expectation: 3 to match the baseline); train/val split inside each fold for early stopping; metric aggregation across folds and seeds; reporting form (mean ± std). `evaluation/grouped_cv.py` handles the raw grouping but not the v14 protocol around it.
  - **Per-patient token coverage concern**: with 46–178 trials per patient and 52 PS tokens, not every patient has every token. When grouped-by-token CV splits the 52 tokens into train/test folds, a fold can end up with very few (or zero) test trials for a given patient, making that patient's per-patient PER noisy or undefined for that fold. Decide: (a) drop any patient-fold combination with fewer than `N_min` test trials from that patient's fold average, (b) require every test fold to contain at least one trial from every patient (constrains the token assignment), (c) report per-patient PER only over patient-folds that meet a coverage floor and raise the floor if too many folds are dropped, or (d) aggregate per-patient PER over *all trials across folds* rather than averaging per-fold first. Default expectation: (d) — "micro-average" per patient (pool all held-out predictions for a patient across all folds, then compute PER once) — because it sidesteps the small-fold-variance problem and matches how the baseline `0.734 ± 0.007` was reported. Discuss and confirm before the first run.
- [ ] **Optimizer / training schedule defaults.** Base lr, warmup, scheduler (cosine vs none vs step), weight decay, gradient clipping norm, batch size (tied to `#31`), epoch count, early-stop criterion, mixed-precision policy on DCC RTX 5000 Ada (32 GB). Not "what's optimal" but "what's the frozen first-run default, and why."
- [ ] **Augmentation policy.** The old per-phoneme baseline used `mixup α=0.2 + label smoothing 0.1 + focal CE γ=2`. Phase 1 baseline should decide whether any of those carry over as defaults or whether they're all off and added back as ablations one at a time. Expectation: **plain CE + no augmentation** for the first run, but write it down so recipe elements don't creep back in by habit.
- [ ] **Reproducibility / seed protocol.** Seed handling across dataloader, model init, CV splits, and any augmentation. Locked enough that a 3-seed report is well-defined. Short item — probably one sentence of commitment.

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
