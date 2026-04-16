# v14 Implementation Task List

Current goal:

- **implement and verify `v14` on intra-op `uECoG` only**
- **start with `v14-core`, no learned per-patient calibration**
- **supervised training only in Phase 1**

Lightweight task/blocker list. Not a design document. **Single source of truth for blockers and decisions.** Other docs point here; do not restate decisions elsewhere.

## Status (2026-04-14)

**Open blockers (8)**: #10 #12 #13 #14 #15 #25 #34 #36

**Decided blockers (28)**: #1 #2 #3 #4 #5 #6 #7 #8 #9 #11 #16 #17 #18 #19 #20 #21 #22 #23 #24 #26 #27 #28 #29 #30 #31 #32 #33 #35 (new: atlas gap-filling via PM dilation, **marked for deletion pending #36**) — full rationale lives in the `## Decisions` section at the bottom. The open blocker list above intentionally excludes them so a scan shows only unresolved items.

**2026-04-15 reopen note**: `#10` (parcel-frame construction) was previously listed as decided 2026-04-13, but on review the contract was never actually agreed — the write-up below is a draft, not a frozen decision. Every sub-choice (nilearn `vol_to_surf` configuration, second-moment-tensor vs alternatives, sign-determinism rules, cache schema) is open again and must go through the same discussion-first treatment as every other blocker before any builder code runs.

**2026-04-14 late-session architectural pivot (new, proposed, not yet executed):** `#36` proposes retiring the entire `cvs_avg35_inMNI152` path (`sub2AvgBrainClinical.m` port + 8 mm PM dilation from `#35` + envelope/pial fudge) in favor of a pure fsaverage pipeline where both the patient side (stock `lh.sphere.reg`) and the atlas side (`mri_vol2surf` onto ICBM152_fs, then `mri_surf2surf` to fsaverage) meet via explicit surface-based sphere registration. The aggregator for the atlas-side `mri_vol2surf` call is **locked as `--projfrac-avg 0 1 0.1`** (mean of 11 samples across the white→pial column) — this sub-decision is frozen even though the overall migration is still pending. If `#36` is executed, `#35` is deleted and the `## Decisions` section retires the dilation + cvs_avg35 entries.

**2026-04-14 session cleared the entire coord/parcel spatial chain:**
- `#1` ACPC→MNI pipeline — verified end-to-end (S14 oracle + S26 anatomy cross-check + 1280/1280 coverage on raw coords after dilation). No drift. No gate on S26.
- `#4` Parcel set and `N_tok` — 15 LH Tier-1 parcels selected by the argmax-centric rule `argmax_wins >= 10` on raw coords + 8 mm dilated PM + σ=1.5 mm Gaussian smoothing across 1280 Phase-1 LH electrodes. Revised twice on 2026-04-14: first from the original "top-12 by eff_N" to `eff_N >= 10 AND argmax_wins >= 1` after the leakage-sink diagnostic, then to the current argmax-only rule after visual inspection found 12 primary auditory cortex (TE1.0/1.2) electrodes excluded by the `eff_N >= 10` clause despite real dominance.
- `#3` Unsupported-vs-weak token_mask rule — revised to `argmax_p support(e, p) ∈ Tier1` (with explicit all-zero handling as `token_mask = False`) for structural consistency with the new `#4` argmax-centric rule. Old `max_p support >= 10%` rule is gone. Every Phase-1 LH electrode currently meets the new rule; exists to catch genuine outliers and future sEEG contacts in white matter.
- `#35` (new) Atlas gap-filling — nearest-neighbor PM dilation with `d_max = 8 mm` via `scripts/dilate_pm.py`. Replaces the previous plan of an electrode-side pial snap.

**Critical path to Phase 1 implementation** (longest chain of open blockers — these must resolve in this order before `v14-core` can be written):

- **Spatial chain**: ~~`#1`~~ + `#12` (amp → physical electrode → RAS bridge) → ~~`#4`~~ → ~~`#26`~~ — `#1`, `#4`, and `#26` now closed; only `#12` per-patient verification remains on the spatial side.
- **Temporal chain (parallel)**: ~~`#2`~~ + ~~`#6`~~ → `#26`
- **Shared dynamics**: ~~`#26`~~ → ~~`#27`~~ → ~~`#28`~~
- **Label / target chain (parallel)**: ~~`#17`~~ + ~~`#21`~~ + ~~`#16`~~ → `#28`
- **Training contract (parallel)**: `#9` → `#27` / `#28`
- **Independent audit prerequisite**: `#34` (end-to-end phoneme loading and trial-timing audit) gates the v14 data loader regardless of the spatial chain — resolve in parallel with `#12`.

Every open blocker still needs the discussion-first treatment defined below before any code against it is written.

## Working Principle

**Every blocker below is a discussion item, not an engineering ticket.** v14 is slow, methodical, and precise — every logic step from raw voltages to phoneme decode is discussed, agreed, and understood before code is written. No handwavy implementation. No pre-committed numeric defaults. No legacy reuse. Rewriting from scratch is the default. See `CLAUDE.md` "Working Principle: Discuss Before Code".

Resolving a blocker means: logic agreed, contract (inputs, outputs, shapes, units) agreed, trade-offs and precedent written down, fallback path noted separately. "A likely default exists in the docs" is **not** resolution.

**Scalability lens (2026-04-14)**: when two choices are equally correct for Phase 1, prefer the one that preserves a clean path to cross-task use, external datasets, and broader scaling. Do not add speculative infrastructure early, but do avoid Phase-1-only conventions when a standard, reusable contract works just as well.

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

- [x] **#1 ACPC → MNI transform pipeline — DECIDED 2026-04-14.** Closed. Full rationale moved to the `## Decisions` section at the bottom. Short summary: the Python port of `sub2AvgBrainClinical.m` in `src/speech_decoding/v14/coordinates.py` matches Zac's S14 MATLAB oracle within 1.5 mm (max 1.39, median 0.68; enforced by `tests/v14/test_coordinates.py`), produces anatomically plausible coords for all 7 Phase-1 LH patients, and the "S26 drifts 25 mm" diagnosis from earlier in the 2026-04-14 session was wrong — S26's ACPC→MNI delta (26.1 mm) is in-family with S14's oracle-verified delta (32.8 mm). The remaining post-trust PM-coverage concern is subsumed by `#35` (atlas gap-filling).

- [x] **#2 Decide the exact temporal layer implementation — DECIDED 2026-04-14**
  - **Decision**: Phase 1 uses the simplest shared per-electrode temporal tokenizer: a single `Conv1d` patch projection applied independently to each electrode trace.
  - **Frozen parameters**:
    - sample rate: `200 Hz`
    - kernel size: `30` samples = `150 ms`
    - stride: `10` samples = `50 ms`
  - **Interpretation**: each output token is an overlapping electrode-local temporal patch embedding. The tokenizer is intentionally local; longer-range temporal structure is deferred to the backbone.
  - **Why this choice**:
    - it is materially simpler than the older dilated-CNN tokenizer draft
    - `~150 ms` is on the right scale for phoneme-local HGA structure without collapsing the whole 3-phoneme utterance too early
    - `50 ms` stride preserves within-trial timing better than the coarser streaming decoders in Stavisky / Chang while staying computationally light
    - it keeps the Phase-1 front-end minimal enough that failures are interpretable elsewhere in the pipeline
  - **Not in the baseline**: no multi-stage temporal CNN, no dilations, no parcel-specific temporal stem, no learned temporal attention inside the tokenizer itself

- [x] **#3 Unsupported-vs-weak token_mask rule — DECIDED 2026-04-14, revised same day.** Closed. Full rationale in the `## Decisions` section. Short summary: `token_mask[e] = True` iff `argmax_p support(e, p) ∈ Tier1`, with the all-zero case `support(e, :) ≡ 0` mapped to `token_mask = False` explicitly. `argmax_p` is over all 246 parcels. An electrode is valid iff its anatomical home (its globally dominant parcel) is a parcel we model. This replaces the earlier `max_p support >= 10%` rule to make `#3` structurally consistent with the revised `#4` — both sides now answer the same "which parcel does this electrode live in" question via the same argmax computation. On the frozen pipeline all 1280 Phase-1 LH electrodes pass (after `#4`'s revision added TE1.0/1.2 to Tier-1), so the rule is a no-op on current data; it exists to catch genuine outliers and future sEEG contacts in white matter.

- [x] **#4 Base parcel set and default split map — DECIDED 2026-04-14, revised twice same day.** Closed. Full rationale in the `## Decisions` section. Short summary: `N_tok = 15` LH parcels, selected by the argmax-centric rule `argmax_wins(p) >= 10` on the frozen spatial pipeline (raw MNI + 8 mm PM dilation + σ=1.5 mm Gaussian). Default split map is uniform `k_parcel = 1` — no 2-token splits in Phase 1 (the 2-token variant is a named ablation). The 15 parcels: A45c (33), A45r (35), IFS (31), A4hf (53), A44v (39), A9/46v (21), A12/47l (51), A4tl (61), A38l (77), A44d (29), A1/2/3tonIa (157), A1/2/3ulhf (155), IFJ (17), A6cvl (63), TE1.0/1.2 (73).

- [x] **#6 Lock the temporal-layer output contract — DECIDED 2026-04-14**
  - **Input**: response-onset-locked full-trial signal window from `#29`, shape `signal ∈ R^{B × N_elec × 301}` at `200 Hz`.
  - **Output**: `h ∈ R^{B × N_elec × d × T_tok}` with
    - `T_tok = floor((301 - 30) / 10) + 1 = 28`
    - frozen Phase-1 shape: `(B, N_elec, d, 28)`
  - **Time semantics**:
    - token rate: `20 Hz` (`50 ms` step)
    - receptive field per token: `150 ms`
    - adjacent token overlap: `100 ms`
    - outputs are **overlapping patch tokens**, not a continuous frame stream
  - **Downstream contract**: the parcel summarizer in `#26` consumes the full time axis and should treat `T_tok = 28` as a sequence of overlapping temporal patches, not as independent non-overlapping bins.

- [x] **#8 Lock token-level connectivity expansion details — DECIDED 2026-04-14**
  - **Decision**: use combined SC + FC as the spatial-attention prior, not SC-only and not FC-only.
  - **How it enters the backbone**:
    - both SC and FC are normalized parcel-to-parcel bias matrices
    - both enter spatial attention as additive logit biases
    - baseline form:
      - `score(i,j) = q_i · k_j / sqrt(d_h) + gamma_sc * B_sc(i,j) + gamma_fc * B_fc(i,j)`
    - `gamma_sc` and `gamma_fc` are learnable scalar gains
    - gains are shared across heads
  - **Why this is the right contract**:
    - SC is the anatomical prior
    - FC is the dynamical prior
    - keeping them separate but additive preserves interpretability and lets training up/down-weight each source
    - additive logit bias makes them a **soft prior**, not a hard constraint or gate
  - **Soft-prior meaning**:
    - learned `q · k` content still drives attention
    - SC/FC only nudge the logits toward anatomically / functionally plausible parcel interactions
    - if either prior is unhelpful, training can drive its scalar gain toward zero
  - **Normalization / scaling**:
    - SC and FC must be normalized to comparable scale before entering attention
    - scaling stays outside the matrices themselves; the learned gains are the only trainable strength parameters in the baseline
  - **Sibling-token rule**:
    - deferred as bookkeeping because Phase 1 `#4` froze `k_parcel = 1`
    - if split parcels are reintroduced later, sibling tokens inherit the same parent SC/FC rows and columns in the initial expansion, with any sibling-specialization learned downstream
  - **Fallback ablations**:
    - no-bias spatial attention
    - SC-only
    - FC-only

- [x] **#9 Supervised training contract — DECIDED 2026-04-14**
  - **Input window**: response-onset-locked full-trial input per `#29`: one sample = one trial, `tmin = -0.5 s`, `tmax = 1.0 s`.
  - **Target format**: fixed-length 3-slot phoneme sequence per trial. No padding symbol, no `<bos>`, no `<eos>`, and no auxiliary 52-token classification head in the baseline.
  - **Decoder training rule**: teacher forcing during training.
  - **Train-time vs eval-time behavior**:
    - train time: teacher-forced autoregressive conditioning
    - eval time: free-running autoregressive decoding only; no teacher-forced eval is allowed
  - **Baseline eval decode**: exhaustive sequence scoring over the fixed 3-slot, 9-class output space. For Phase 1 this is cheap (`9^3 = 729` candidate sequences per trial) and removes search-approximation noise from the baseline reported metric.
  - **Comparison decodes**: greedy autoregressive decoding is logged as a comparison path; beam search is unnecessary in the baseline because exhaustive search dominates it on a fixed 729-sequence space.
  - **Loss**: plain cross-entropy per slot, aggregated across the 3 slots and averaged across trials. No label smoothing, focal loss, mixup, or auxiliary losses in the baseline.
  - **Why**: this is the simplest faithful supervised baseline. It keeps the training rule stable, keeps evaluation honest, and avoids conflating architecture effects with recipe tricks or search approximations.

- [ ] **#10 Parcel-frame construction contract — REOPENED 2026-04-15**

  **Status**: the write-up below is a discussion draft carried over from the earlier "DECIDED 2026-04-13" framing. On 2026-04-15 review it was flagged that no actual agreement ever happened — the contract needs to go through discussion-first treatment again. Treat every sub-section as open. Do **not** build `data/atlas/parcel_frames.npz` against this draft.

  Sub-items that must be explicitly agreed before this blocker closes:
  - origin definition (volumetric PM-weighted centroid vs surface-projected centroid vs contact-weighted mean)
  - rotation method (second-moment-tensor of pial normals vs per-vertex curvature tensor vs a per-voxel local-normal alternative)
  - whether fsaverage is the right surface target at all, or whether `#36`'s proposed fsaverage bake should be used directly as the vertex source (couples this blocker to `#36` execution)
  - `nilearn.surface.vol_to_surf()` configuration for the BNA → fsaverage projection (interpolation, `kind`, radius) — whether this is the right tool at all, or whether the `mri_vol2surf --projfrac-avg 0 1 0.1` path locked under `#36` should be used instead for consistency
  - tangent-axis definition and sign determinism
  - per-axis scale normalization scheme
  - cache schema and which fields are required vs diagnostic
  - Phase 1 vs Phase 2 upgrade boundary (per-parcel mean normal vs per-voxel local normal)

  ### Origin (DRAFT)
  - PM-weighted centroid over the Brainnetome PM volume (`data/atlas/BNA_PM_4D.nii.gz`).
    - `origin[parcel] = Σ_v PM(parcel | v) * position_v / Σ_v PM(parcel | v)`, summed over MNI voxels `v`.
    - Volumetric, not surface-projected. Keeps the origin consistent across uECoG (contacts on pia) and Phase 2 sEEG (contacts distributed through the cortical slab).

  ### Rotation method — fsaverage-based cortical-normal axes (DRAFT)
  - **Space mapping**: use `nilearn.surface.vol_to_surf()` to project the BNA PM volume onto the fsaverage pial mesh. nilearn handles the MNI152 ↔ fsaverage (MNI305) space mapping internally — we do **not** build our own warp.
    - Resolution: `fsaverage` (full, 163 842 vertices per hemisphere). Fs6/5 is too coarse for small parcels like `A44op`. Cache is built once offline, so resolution does not matter for runtime.
    - Interpolation: `interpolation="linear"`, `kind="ball"` with a small radius (needs empirical tuning at build time). **Must be verified**: for a probabilistic atlas, we need "PM value near this vertex", not "line-integral through the cortical slab" — the nilearn defaults are tuned for fMRI BOLD, not atlas membership, and the choice must be checked against a flat-parcel smoke test.
    - Separate LH and RH projections. fsaverage has distinct `pial_left` and `pial_right` meshes and vertex sets. A left Brainnetome parcel must only receive left-hemisphere vertices, and vice versa. Mixing sides silently invalidates the axes.
    - **Open**: if `#36` executes, the `mri_vol2surf --projfrac-avg 0 1 0.1` path onto ICBM152_fs + `mri_surf2surf` to fsaverage is already baking BNA onto fsaverage vertices. Re-deciding `#10` against *that* source (instead of `nilearn.surface.vol_to_surf`) removes one tooling dependency and keeps surface-side sampling consistent with the spatial pipeline.
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

  ### Rigor note on second-moment tensor vs mean (DRAFT)
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

  ### Per-axis scale normalization (DRAFT)
  - Feed `(x/σ_x, y/σ_y, z/σ_z)` to the shared point encoder, where `σ_*` are the standard deviations of the PM-weighted in-parcel positions projected along each parcel-frame axis.
  - Append `(log σ_x, log σ_y, log σ_z)` as three scalar features on the parcel's token (once per parcel, not per electrode). The point encoder sees parcel-size-invariant positions; parcel size lives on the token as a three-number side channel.
  - Std-based, not extent-based — robust to boundary outlier voxels.
  - Log-scaled token scalars because parcel sizes span more than an order of magnitude.

  ### Sign determinism (DRAFT)
  - `z` (cortical normal): eigenvectors have ±1 ambiguity. Pin by `sign(z · (origin − brain_centroid)) > 0`, where `brain_centroid` is the mean of the fsaverage mesh vertices (approximately the MNI origin). This forces positive `z` = outward from the brain for every parcel on the cortical surface. For any parcel where the dot product is within numerical noise of zero (`< 1e-3`), fail loudly — the parcel is near the brain centre and the rule is ambiguous.
  - `x` (primary tangent): pin by "positive `x` points toward the anterior-most in-parcel fsaverage vertex, projected onto the tangent plane". Deterministic as long as there is a single unambiguous anterior extreme — flag otherwise.
  - `y` (secondary tangent): `y = z × x` (right-handed frame). No independent sign choice.

  ### Offline caching (DRAFT)
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

  ### Phase 2 upgrade path (DRAFT, tracked separately in `## Phase 2`)
  - Refine from **per-parcel mean cortical normal** to **per-voxel local cortical normal**. Phase 1 stores one rotation per parcel; Phase 2 stores one rotation per PM voxel (or per contact, looked up at load time). If Phase 1's rotation scheme changes as part of reopening this blocker, the Phase 2 upgrade path changes with it.
  - Per-voxel is strictly better only for sEEG in curved parcels (where contacts at the crown vs at the fundus should get different local normals). uECoG is invariant to this refinement because all contacts are on the pial surface.

- [x] **#11 Channel inclusion policy — DECIDED 2026-04-14**
  - **Decision**: Phase 1 uses **all non-artifact channels** by default.
  - **Sig-channel masks**: if present, they are auxiliary metadata only and may be used in ablations, but they do not define the baseline loader contract.
  - **Missing sig-channel files**: no behavior change. A missing sig mask never alters which channels are loaded in the baseline path.
  - **Why**: sig-only is an older task-dependent filter, not the right common interface for a scalable atlas-token model. All-non-artifact is cleaner across patients, more reproducible, and more compatible with future external datasets where sig masks may not exist or may be defined differently.
  - **Consequence**: channel filtering is part of the data-interface contract, not something hidden inside the model. The baseline interface is anatomical validity plus artifact exclusion, not prior response-significance.

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
  - **MNI coordinates are keyed by electrode name, not by physical position** — DECIDED 2026-04-13:
    - The v14 MNI coordinate cache (`data/mni_coords/<pt>_MNI152.csv`, produced by the Python port of `sub2AvgBrainClinical.m`) is a row-per-electrode file with columns `name, x, y, z, hemisphere`. The `name` column is the **concatenated string from `.electrodeNames`** (e.g., `SMC16`, not `SMC 16`).
    - **Reason**: physical-electrode-index keying (`r*16+c+1`) does not generalize across array layouts. 256-ch Map 3 patients (S33, S62) have 256 populated electrodes in a 46×24 = 1104-position grid, so the row index in `.LEPTO` is not any simple function of `(r,c)`. Name keying is invariant to array layout, population density, and reconstruction session.
    - **Consequence for the bridge**: the blocker #12 contract becomes `fif ch_name → amp_ch → chanMap[r,c] → physical electrode index → physical electrode name → MNI coord by name`. The last step is a dict lookup into the cached CSV. The bridge's real job is therefore to produce a *physical electrode name* (from `.electrodeNames` at the `phys_idx - 1` row, since `.electrodeNames` is ordered row-for-row with `.LEPTO` and the local `<pt>_RAS_brainshifted.txt`).
    - **Ordering invariant to verify per patient**: `.electrodeNames` row `i` ↔ `.LEPTO` row `i` ↔ `<pt>_elec_locations_RAS_brainshifted.txt` row `i`. Verified for S14 (2026-04-13): all three files have 128 rows (post-header) with identical per-row coordinates and names. Must be re-verified for every Phase-1 patient before the bridge is trusted — this is a 1-to-1 check, not a set-overlap check.
    - **Format normalization gotcha**: the local `<pt>_RAS.txt` files split the name into two space-separated tokens (`SMC 16`), but `.electrodeNames` concatenates (`SMC16`). The canonical form is **concatenated** (match `.electrodeNames`). Any downstream code that has to join against the local `_RAS.txt` must strip whitespace from the two-token name before comparing.
    - **Name ≠ row index**: electrode names like `SMC16`, `SMC32`, `SMC48` are **grid labels**, not sequential 1..128 indices. The bridge must never infer row position from the integer suffix in the name.

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

- [x] **#16 v14 phoneme label → integer index contract — DECIDED 2026-04-14**
  - **Decision**: use plain `0`-indexed class ids with no reserved blank, padding, start, or end symbol. Phase 1 targets are fixed-length 3-phoneme sequences, so extra control symbols are not part of the baseline contract.
  - **Canonical ordering**: alphabetical ARPABET over the 9 PS phonemes, chosen for cross-task and external-dataset scalability rather than PS-specific or historical ordering:
    - `AA = 0`
    - `AE = 1`
    - `B = 2`
    - `G = 3`
    - `IY = 4`
    - `K = 5`
    - `P = 6`
    - `UW = 7`
    - `V = 8`
  - **Why**: alphabetical ARPABET is deterministic, standard, and reusable. It avoids baking task-specific conventions into class ids and gives a cleaner path to cross-task pooling and later expansion to larger ARPABET inventories.
  - **Contract boundary**:
    - raw upstream labels remain PS symbols
    - canonical phonetic labels are the standard-ARPABET strings from `#17`
    - decoder / loss / metrics use the integer ids above
    - upstream `event_id` integer order is metadata only and does **not** define decoder-vocabulary order
  - **Related discussion items from the 2026-04-13 phoneme-loading rigor audit: blockers #18–#25 below. #16 covers ordering and indexing; the rigor audit covers identity, ground truth, and loader assertions.**

#### Phoneme / label-space rigor audit (2026-04-13)

Blockers #17–#25 come from a first-pass audit of the PS label space and the upstream assumptions the pre-v14 loader silently inherited. These are **discussion items**, not fixes. No one should touch `phoneme_map.py`, write a v14 data loader, or report any phoneme-level metric until these are resolved — any v14 metric depends on the label space being correct.

- [x] **#17 PS vowel → ARPABET mapping (`ae`, `u`) — DECIDED 2026-04-14**
  - **Decision**: standard ARPABET is the canonical downstream phonetic label space. The 9 PS symbols map to:
    - `a -> AA`
    - `ae -> AE`
    - `i -> IY`
    - `u -> UW`
    - `b -> B`
    - `p -> P`
    - `v -> V`
    - `g -> G`
    - `k -> K`
  - **Why**: this retires the inherited project-specific `ae -> EH`, `u -> UH` convention in favor of standard phonetic mapping, which is cleaner for cross-task work, external datasets, and any future comparison against standard ARPABET corpora.
  - **Authority rule**: the raw PS symbols remain the trusted upstream labels. Standard ARPABET is the explicit remapped canonical label space. Existing derivative ARPABET annotations that use `EH` / `UH` are legacy artifacts and are **not** authoritative if they disagree with the PS-symbol mapping above.
  - **Implication**: any historical pre-v14 result that reported `EH` or `UH` for the PS task should be read as a naming artifact from the old mapping, not as the canonical phonetic identity going forward.

- [x] **#18 Load-time assertion on the BIDS event_id mapping — DECIDED 2026-04-14**
  - **Decision**: v14 uses one frozen PS event-id mapping and hard-asserts it on load. Divergence is a hard failure for the PS task; no dynamic per-patient remapping in Phase 1.
  - **Frozen mapping**: `{'a':1, 'ae':2, 'b':3, 'g':4, 'i':5, 'k':6, 'p':7, 'u':8, 'v':9}`.
  - **Why**: this is the simplest and most reproducible contract. It avoids silently baking patient-specific label hacks into the loader and scales better to future cross-task and external-dataset adapters.
  - **Implementation note**: the assertion lives in the v14 loader path rather than in a separate frozen JSON file. The actual operational audit still belongs to `#34`.

- [x] **#19 Trial ordering assumption — DECIDED 2026-04-14**
  - **Decision**: v14 does **not** trust stride-3 slicing as a contract. Trials must be reconstructed from explicit trial markers plus phoneme-position markers in the event tables (`trial`, `phoneme_idx` or equivalent positive position metadata).
  - **Loader invariant**: order-only grouping is not allowed. Every trial must be assembled from explicit per-event metadata that identifies both the parent trial and the within-trial phoneme slot.
  - **Why**: this is safer on the current data and generalizes better to datasets where missing or dropped events would silently corrupt an order-only contract.
  - **Operational audit**: `#34` still has to prove the current BIDS derivatives satisfy this invariant.

- [x] **#20 v14 trust posture for MFA alignment — DECIDED 2026-04-14.**
  - **Decision**: Phase 1 does not use MFA as a training-time dependency at all. Inputs stay response-onset-locked full-trial only: one sample = one trial = three phonemes, with `tmin = -0.5 s`, `tmax = 1.0 s` per `#29`.
  - **What MFA is not used for**: not for epoch boundaries, not for per-phoneme training samples, not for segmenting the neural input, and not as a required loader dependency.
  - **What MFA may still be used for**: audit only under `#34` if needed to cross-check timing or target-order assumptions against audio. That is a verification path, not part of the Phase 1 model contract.
  - **Rationale**: the v14 decoder is a 3-slot trial-level sequence model over one continuous response-locked window. The Conv2d per-phoneme MFA baseline solved a different problem and does not justify carrying MFA segmentation into the v14 contract.

- [x] **#21 Canonical 52-token enumeration for the PS task — DECIDED 2026-04-14**
  - **Decision**: the canonical manifest now lives at `data/ps_tokens.csv` with fields `token_id`, `ps_notation`, `ps_sequence`, `arpabet_sequence`, `ipa`, and `structure`.
  - **Source of truth**: the file was extracted from the local PhonemeSequence BIDS production-event derivatives and verified to contain the same 52-token set for all 11 PS patients (`S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62`).
  - **Canonicalization rule**: `token_id` is deterministic and non-semantic: alphabetical by `ps_notation`, numbered `0..51`. The raw PS token string remains the trusted upstream identity; the ARPABET and IPA columns are explicit derived annotations.
  - **Why**: this is sufficient for Phase 1 correctness and is cleaner than leaving the token list implicit in derivative event files. It also gives a stable contract for grouped-by-token CV, load-time assertions, and future cross-task / external-dataset adapters.
  - **Remaining work moved downstream**: loader assertions against this manifest are still part of `#34` / `#25`. This blocker only covers committing the canonical token inventory itself.

- [x] **#22 Cross-task / cross-patient event_id consistency — DECIDED 2026-04-14**
  - **Decision**: add a standalone invariant test for event-id consistency.
  - **Failure mode**:
    - PS-patient divergence is a hard failure.
    - Lexical divergence is warn-only for now, until cross-task pooling is active.
  - **Why**: Phase 1 needs a strict PS contract, but Lexical should not block unrelated progress before cross-task loading is turned on.
  - **Placement**: this belongs in a standalone test module rather than being implicit loader behavior.

- [x] **#23 `normalize_label` input contract — DECIDED 2026-04-14**
  - **Accepted inputs only**:
    - lowercase PS symbols (`a`, `ae`, `i`, `u`, `b`, `p`, `v`, `g`, `k`)
    - canonical uppercase ARPABET (`AA`, `AE`, `IY`, `UW`, `B`, `P`, `V`, `G`, `K`, and the broader English ARPABET set where relevant)
    - ARPABET with trailing stress digits (`AA1`, `AE0`, `IY1`, etc.)
  - **Rejected inputs**: lowercase ARPABET, mixed-case variants, empty strings, and any unknown symbol.
  - **Behavior**: unknown labels continue to raise `ValueError`. v14 does **not** silently uppercase-first before normalization.
  - **Why**: a strict contract is better here. It keeps upstream format boundaries explicit and avoids silently accepting malformed labels.

- [x] **#24 `filter_to_ps_phonemes` fail-silent policy — DECIDED 2026-04-14**
  - **Decision**: split strict and non-strict behavior conceptually.
  - **Cross-task use**: fail-silent filtering is allowed when the caller is intentionally screening a broader label inventory down to the PS set.
  - **Within-task use**: PS-task code must use a strict path and must not silently treat unknown labels as `False`.
  - **Why**: cross-task filtering needs permissive reject behavior, but within-task loading should never hide typos, corruption, or contract drift.
  - **Implementation note**: whether this becomes two functions or a strictness flag is an implementation detail, not an open contract question.

- [ ] **#25 Add ground-truth phoneme fixture tests against the 52-token list**
  - Current `tests/test_phoneme_map.py` tests the internal consistency of the mapping (`normalize_label("a") == "AA"`), but never checks that `"AA"` corresponds to the /ɑ/ sound actually uttered in the audio. For a label space that is the foundation of every downstream metric, the contract should be exercised against the real token set.
  - Dependencies: #21 must be done first (need the canonical 52-token list). #17 should be done first (need the correct vowel identities).
  - Action:
    - after #17 and #21, write a test fixture that loads the 52-token list
    - assert that `normalize_label` round-trips every element of every token's ARPABET decomposition
    - assert that `filter_to_ps_phonemes` returns `True` for every phoneme in the PS vowel/consonant set and `False` for every non-PS ARPABET symbol
    - optionally: assert that `ARPA_PHONEMES` equals exactly the set of phonemes appearing across the 52 tokens
  - These tests are the first real rigor baseline for `phoneme_map.py`.

- [x] **#26 Lock the within-parcel cross-attention contract (electrode → parcel tokens) — DECIDED 2026-04-14**
  - **Role**: this is the module that converts patient-specific electrode sets into the shared atlas-token interface consumed by the backbone.
  - **Input / output contract**:
    - input from `#6`: `h ∈ R^{B × N_elec × d × T}` with `T = 28`
    - output to the backbone: `z ∈ R^{B × N_tok × d × T}`
    - Phase-1 `#4` froze `k_parcel = 1`, so one token is emitted per parcel per time step
  - **Membership rule**:
    - parcel `p` only sees electrodes whose global anatomical home is `p`
    - concretely, electrode `e` enters parcel `p` iff `argmax_q support(e, q) = p`
    - no cross-parcel duplication of one electrode into multiple parcel summarizers
  - **Why membership is hard-argmax**:
    - keeps token semantics sharp across patients and external datasets
    - stays structurally consistent with the argmax-centric `#3` / `#4` decisions
    - avoids double-counting one electrode's evidence across neighboring parcel tokens
  - **Time-axis handling**:
    - summarization runs independently at each time step `t`
    - no flattening over `(N_elec × T)`
    - no early temporal pooling inside the summarizer
    - the backbone remains the first module that models longer-range temporal structure
  - **Baseline summarizer mechanism**:
    - shared point encoder over per-electrode inputs
    - then single-latent Perceiver-style cross-attention per parcel per time step
    - summarizer weights are shared across parcels in the baseline
  - **Perceiver cross-attention settings**:
    - one latent query per parcel-time slice in Phase 1 because `#4` froze `k_parcel = 1`
    - clean standard cross-attention block: pre-norm, attention, FFN, residual around FFN
    - head dimension `= 32`, with head count derived from `d`
    - dropout `= 0.1`
    - no latent self-attention stack inside the summarizer
    - no layer-scale, gating, or other custom stabilizing extras in the baseline
  - **Why keep this standard**:
    - the summarizer's job is narrow: convert a variable-size parcel-local electrode set into one token at one time step
    - it should not become a second backbone or absorb longer-range modeling that belongs in `#27`
  - **Why the summarizer is shared across parcels**:
    - the operation "summarize a variable-size set of parcel-local electrodes into one token" is common across parcels
    - sharing is statistically cleaner at this data scale than training parcel-specific summarizers on small per-parcel electrode subsets
    - parcel identity should be explicit in the input contract, not hidden in separate weights
  - **Parcel identity input**:
    - each electrode item includes a learned parcel embedding for the parcel currently being summarized
    - this lets a shared summarizer behave differently for different parcels without requiring parcel-specific weights
  - **Per-electrode input feature composition**:
    - temporal feature `h_e(t) ∈ R^d`
    - parcel-frame coordinate `coord_e,p ∈ R^3` from `#10`
    - parcel-size side channel `log_sigma_p ∈ R^3` from `#10`
    - scalar support feature `s_e,p`
    - parcel identity embedding `emb(p)`
  - **How support enters the summarizer**:
    - support enters as a scalar input feature on each electrode item
    - support also enters the cross-attention logits as a soft additive bias
    - baseline form:
      - `a_e = (q · k_e) / sqrt(d_h) + alpha * log(s_e,p + eps)`
      - `w_e = softmax_e(a_e)`
    - `alpha` is learnable and initialized small positive
    - `eps` exists only for numerical safety
  - **Why this is the right math / ML contract**:
    - the feature path lets the model learn nonlinear interactions among support, coordinates, and neural content
    - the additive logit bias encodes the prior that higher-support electrodes should be easier to trust for that parcel
    - additive logit bias is the right place for a confidence prior because it changes selection probability without directly changing feature magnitude
    - the baseline does **not** multiply values by support; that would entangle anatomical confidence with neural amplitude and can suppress informative boundary electrodes too aggressively
  - **Relation to `#7` token_support**:
    - this per-electrode support path is local to the summarizer
    - the parcel-level `token_support` path from `#7` still applies after summarization at the token level
    - `token_mask[p]` and `token_support[p]` are intentionally different signals:
      - `token_mask[p] = 1` iff parcel `p` has at least one electrode whose `argmax_q support(e, q) = p`
      - `token_support[p] = Σ_e support(e, p) / 100`, the full-support parcel rollup from `#5`, not an argmax-only sum
    - these are not duplicates: one is electrode-local trust within a parcel, one is parcel-level support seen by the backbone, and one is hard parcel presence/absence
  - **Masking and absent parcels**:
    - if a parcel has zero contributing electrodes for a patient, the summarizer emits a zero token and `token_mask = 0`
    - zero-filling is a storage convenience only; downstream attention must still respect the mask
    - correctness invariant: masked parcels are excluded by attention masks first, and only then zero-filled. Zero values are never relied on as the mask itself.
  - **Artifacts now unblocked**:
    - `src/speech_decoding/v14/local_summarizer.py` can use this as its exact forward contract
    - `tests/v14/` can add shape and mask behavior tests against this contract

- [x] **#27 Lock the inter-region attention backbone contract — DECIDED 2026-04-14**
  - This is the shared dynamics stack that operates on parcel tokens `(N_tok × d × T)` between the summarizer (`#26`) and the AR decoder. Every block count, every attention-axis decision, and every normalization choice is an explicit discussion item. Nothing is pre-committed. Discussion-first.
  - Dependencies: `#26` (input shape / token semantics coming in), `#8` (token-level SC/FC connectivity expansion for attention bias), `#9` (decoder contract — sets what the backbone must hand off), `#6` (temporal front-end output `T` and time-axis semantics).
  - Discussion items to resolve:
    - **Attention factorization — DECIDED 2026-04-14**: use factored alternating `(spatial over N_tok) → (temporal over T)` blocks, not joint spatiotemporal attention and not temporal-first. Why: `N_tok` is small, the spatial axis has explicit anatomical structure and will eventually carry the SC/FC bias from `#8`, and the temporal axis has an ordered local-token structure from `#6`. Factoring the two keeps those inductive biases clean, scales better to longer windows and external datasets, and avoids mixing space-time interactions earlier than necessary.
    - **Block count `B` — DECIDED 2026-04-14**: baseline uses `B = 2` factored block pairs. This is the clean middle ground at the current data scale: deeper than a one-pass spatial→temporal update, but still modest enough that the backbone does not dominate capacity or hide failures elsewhere in the pipeline. First named ablations are `B = 1` and `B = 3`.
    - **Per-block structure — DECIDED 2026-04-14**: use the clean standard transformer block. Each spatial and temporal sub-block is pre-norm, with residual connections around both attention and FFN. FFN width is `4d`, activation is `GELU`, dropout is `0.1` on attention and FFN, and the baseline does not use layer-scale or other stabilizing extras.
    - **Heads and head dim — DECIDED 2026-04-14**: keep the baseline simple. Use `head_dim = 32`, with the same head configuration on spatial and temporal attention. The actual head count is derived as `d / 32` once `d` is finalized under `#15`.
    - **Spatial attention bias from Brainnetome SC/FC**: resolved jointly with `#8`. Key sub-items that belong in `#27`, not `#8`: how the bias enters the softmax (additive logit vs gating), whether it's learnable (scalar gain per bias matrix) or fully fixed, and whether it's shared across heads or per-head. Default expectation: additive logit, learnable scalar gain initialized at `1.0`, shared across heads.
    - **Temporal attention positional structure — DECIDED 2026-04-14**: use RoPE on temporal attention only. Time is ordered and needs positional structure; the spatial parcel axis does not have a natural sequence geometry and therefore gets no positional encoding. RoPE is the clean default here because it is lightweight, parameter-free, and handles variable `T` without introducing extra learned capacity. Fourier PE on electrode or parcel coordinates remains explicitly rejected.
    - **Masking and `token_mask` propagation — DECIDED 2026-04-14**: the spatial softmax masks out tokens where `token_mask[j] = 0` (absent parcel for this patient). Temporal blocks run on the same tensor shape for simplicity, but masked parcel rows are explicitly zeroed after each block. This does not change semantics because masks, not zeros, carry the exclusion rule; zeroing is only a post-mask storage invariant.
    - **`token_support` role inside the backbone — DECIDED 2026-04-14**: per `#7`, `token_support[p]` is concatenated onto each parcel token once at backbone entry (before block 1) via `[d] ⊕ [1] → linear → [d]`. It is not re-injected at every block in the baseline. `token_support[p]` is the full-support parcel rollup from `#5`, not an argmax-only count, so the backbone sees parcel strength separately from hard parcel presence (`token_mask`).
    - **Readout / handoff to decoder — DECIDED 2026-04-14**: the backbone hands the decoder the full `(N_tok, d, T)` tensor. The decoder cross-attends over the flattened `(N_tok · T, d)` key/value set; there is no early pooling over tokens or time in the baseline. `token_mask` is a parcel-axis mask broadcast across time at the decoder boundary.
    - **Parameter / init / memory note**: no extra blocker here beyond keeping the backbone modest relative to `d` from `#15`. Use standard PyTorch initialization defaults in the baseline. Gradient checkpointing is explicitly *not* part of the correctness contract and can be revisited only if DCC memory requires it.
  - Artifacts produced once this is frozen:
    - exact tensor signature for `BackboneConfig` and the forward contract, written into `src/speech_decoding/v14/backbone.py` docstring
    - a shape / contract test under `tests/v14/` that constructs a dummy `(B_batch, N_tok, d, T)` input plus `token_mask` and asserts the output is `(B_batch, N_tok, d, T)` with masked-token outputs zero-invariant (attention weights into masked tokens are zero, output slices at masked rows are zero)
    - a factored-vs-joint toggle that is exercised by unit test before any training run
    - a block-count sweep harness (even if just a config override) so `B ∈ {1, 2, 3}` can be run without editing code
  - No code in `src/speech_decoding/v14/backbone.py` beyond the current stub until every bullet above is resolved.

- [x] **#28 Lock the AR cross-attention decoder contract — DECIDED 2026-04-14**
  - **Target structure**:
    - fixed-length 3-slot phoneme decode
    - no `<eos>`
    - no auxiliary 52-token head in the baseline
    - this matches `#9` exactly
  - **Overall decoder shape**:
    - one small autoregressive decoder block
    - causal self-attention over the 3 slot queries
    - then cross-attention to the backbone memory
    - then a shared linear vocab head
    - no extra decoder depth and no fancy extras in the baseline
  - **Query / slot initialization**:
    - one shared learned base query `q`
    - plus one learned slot embedding per phoneme position `p_0, p_1, p_2`
    - slot query `i` starts as `q + p_i`
  - **Autoregressive conditioning**:
    - previous-phoneme embeddings are added into the slot-query stream
    - the decoder then uses a single causal self-attention layer over the 3 slot queries
    - train time uses teacher-forced previous-token embeddings
    - eval time uses the decoder's own previous predictions
    - same slot/query structure at train and eval; only the source of previous-token embeddings changes
  - **Why this is the right simple baseline**:
    - one decoder block is enough for a 3-token output space
    - causal self-attention gives a real autoregressive mechanism without making the decoder deep
    - self-attention before cross-attention lets each slot condition on earlier slots before reading the neural memory
  - **Cross-attention memory**:
    - backbone output is the full `(N_tok, d, T)` tensor from `#27`
    - keys and values are flattened to `(N_tok · T, d)`
    - no extra parcel/time positional tag is added at the decoder boundary
    - the backbone already carries the temporal and spatial structure
  - **Cross-attention settings**:
    - one cross-attention layer
    - head dimension matches the backbone (`head_dim = 32`)
    - no extra cross-attention stack in the baseline
  - **Mask propagation**:
    - `token_mask` enters decoder cross-attention as a parcel-axis mask broadcast across time
    - absent parcels are fully excluded from the decoder softmax
  - **Output head**:
    - one shared linear head `d → |V|`
    - no slot-specific output heads
    - slot position is already represented in the query stream via the slot embeddings
  - **Loss / train / eval**:
    - plain slot-wise CE, teacher forcing in train, exhaustive decode at eval are already frozen in `#9`
    - this decoder contract is intentionally aligned to those decisions and adds no extra recipe behavior
  - **Artifacts now unblocked**:
    - `src/speech_decoding/v14/decoder.py` can use this as its exact forward contract
    - `tests/v14/` can add shape, masking, and teacher-forcing vs free-running contract tests against it
    - a cross-attention-zeroed ablation harness so the cross-attention-ignored failure mode above can be ruled out on the first training run
  - No code in `src/speech_decoding/v14/decoder.py` beyond the current stub until every bullet above is resolved.

- [x] **#31 Patient-mixing and batching policy — DECIDED 2026-04-14**
  - **Dataset unit**: one sample = one trial. CV splitting happens on the trial list first; batching happens only after CV-filtering.
  - **Baseline experimental schedule**: the first supervised correctness pass is **per-patient**, starting with S14, then the other core patients under the same contract. Joint-across-patient training is a later experiment, not the first baseline.
  - **Baseline batch composition**: grouped-by-patient batches with fixed `trials_per_batch`. Under the baseline sampler, every batch contains trials from exactly one patient.
  - **Per-sample metadata**: `patient_id` and `token_mask` are emitted per sample even when redundant within a grouped-by-patient batch. This keeps dataset semantics stable across future sampler changes.
  - **Loss weighting**: baseline loss is trial-weighted (plain averaging over trials), not patient-weighted.
  - **Shuffling**: patient order is shuffled each epoch and trials are shuffled within patient.
  - **Gradient accumulation**: if memory requires small physical batches, accumulate gradients to a fixed effective batch size rather than changing optimizer behavior fold-to-fold.
  - **Future-proofing requirement**: the sampler interface must support a later switch from `grouped_by_patient` to `mixed_patients` via a strategy flag, without changing dataset semantics or rewriting the loader.
  - **Why**: this keeps the first correctness pass interpretable against the existing per-patient baseline while still building infrastructure that can scale cleanly to joint training later.

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

- [ ] **#36 Retire cvs_avg35, migrate to fsaverage as the base brain — PROPOSED 2026-04-14 (architectural pivot, not yet executed)**

  This blocker proposes replacing the entire cvs_avg35_inMNI152 surface-projection path with a pure fsaverage pipeline, retiring `sub2AvgBrainClinical.m` port, `dilate_pm.py` (`#35`), and the envelope-vs-pial fudge in one move. The motivation is that cvs_avg35 is a niche ECoG-community choice (inherited from iELVis circa 2017) that does not scale well to new atlases and forces us to paper over a 5-8 mm ribbon gap with dilation. fsaverage is the FreeSurfer / neuroimaging-ecosystem standard and slots into every downstream tool we might eventually want (nilearn, pycortex, neuromaps, templateflow).

  This entry enumerates the sub-decisions. Some are locked (marked **LOCKED**); the overall migration is still **proposed** and requires an execution decision.

  ### Pipeline shape (both sides surface-to-surface via stock `sphere.reg`)

  - **Meeting point**: `fsaverage/lh.pial` (FreeSurfer's standard surface template, built from 40 subjects via surface-based registration on the sphere; in MNI305 space with a well-characterized ~1 mm affine to MNI152).
  - **Patient side**: subject native-RAS electrodes → nearest vertex on patient `lh.pial-outer-smoothed` (envelope; same as today) → follow through patient stock `lh.sphere.reg` → fsaverage pial vertex. No custom surface walk — replaces Zac's `sub2AvgBrainClinical.m` four-step projection with a single stock FreeSurfer registration lookup. Patient-side accuracy: ~1-3 mm, dominated by the sphere.reg residual (FreeSurfer `mris_register` is the field's most widely-used cortical registration; arguably-more-accurate alternatives like MSM-all from HCP are not a fit for T1-only ECoG data).
  - **Atlas side (two one-time steps)**:
    1. `mri_vol2surf --projfrac-avg 0 1 0.1` samples `BNA_PM_4D.nii.gz` along ICBM152_fs's cortical column (white→pial) at 11 evenly-spaced fractions per vertex and averages them → per-vertex BNA probability vector on ICBM152_fs's own pial. No warping — ICBM152_fs was recon'd from ICBM152 so its pial is in the same space as BNA's volume.
    2. `mri_surf2surf` transfers those vectors from `ICBM152_fs/sphere.reg` to `fsaverage/sphere.reg`, explicit surface-based gyral alignment on the sphere. Result: `fsaverage/lh.BNA.gii` with per-vertex 246-dim BNA probability vectors.
  - **Electrode-time path**: electrode → nearest fsaverage pial vertex (via patient sphere.reg) → read baked 246-vector → argmax = parcel, max = support → `(parcel_label, support, features)` token. No volumetric sampling at query time. No dilation. No envelope-vs-pial fudge.

  ### LOCKED sub-decisions (frozen even before the overall migration is executed)

  - **LOCKED: BNA projection aggregator is `mri_vol2surf --projfrac-avg 0 1 0.1` (mean of 11 samples across the white→pial column)** — DECIDED 2026-04-14.
    - **What**: for each ICBM152_fs pial vertex `V`, walk the line from `V_white` to `V_pial` at fractions `[0.0, 0.1, 0.2, ..., 1.0]`, trilinear-sample BNA_PM_4D at each of the 11 points, average the 11 246-vectors.
    - **Why mean (not max, mode, or midpoint-only)**: BNA is a continuous probability atlas — mean is the natural aggregation over a probability distribution. Mean also matches the physics of what a surface ECoG electrode records: the electrode integrates high-gamma across the full cortical thickness directly beneath it (layer I through layer VI), so a column-averaged atlas value is the right "what parcel is this column in" question.
    - **Why `projfrac-avg` over alternatives**:
      - `--projfrac 0.5` (single midpoint sample) — fast but point-samples instead of integrating, loses depth-wise variation. Wrong for our question.
      - `--projfrac-max` — right for "is this parcel present *anywhere* in the depth?" but wrong for probability atlases where we want the expected distribution.
      - `--projfrac-mode` — right for discrete categorical atlases (Desikan, Destrieux), but "mode of 11 continuous 246-vectors" isn't well-defined.
      - `--projfrac-avg 0 1 0.1` (mean) — right for probability atlases. Matches the physics of surface electrode integration.
    - **Information cost, explicit**: this is a dimensionality reduction — a 3D ribbon (~150k cortical voxels per hemisphere at 1.25 mm grid) collapses to a 2D sheet (~150k fsaverage vertices) with one 246-vector per vertex after column averaging. Depth-within-column variation is lost. For BNA specifically the loss is small because BNA's parcels are anatomical cortical areas, not laminar layers, so a given cortical column is in one parcel from layer I through layer VI with at most a soft gradient at parcel borders. This would be the wrong aggregator for a laminar-sensitive atlas (von Economo layers, cortical myelin content by layer); it is the right aggregator for BNA.
    - **Phase 2 sEEG note**: when sEEG joins, the aggregator may be revisited. For an sEEG contact at a specific depth fraction inside the ribbon, `--projfrac <fraction>` at the contact's actual depth is closer to the sEEG physics than `--projfrac-avg`. Phase 2 upgrade, not a Phase 1 concern.
    - **Where this lives in code**: a one-time `scripts/bake_bna_on_fsaverage.py` script that runs `mri_vol2surf --projfrac-avg 0 1 0.1` (per parcel channel, looped over the 246 channels of BNA_PM_4D) followed by `mri_surf2surf` to transfer to fsaverage. Output: `data/atlas/lh.BNA.gii` + `data/atlas/rh.BNA.gii`.

  - **LOCKED: physical PSF smoothing moves from 3D volumetric convolution to 2D geodesic surface smoothing** — DECIDED 2026-04-14.
    - The Gaussian PSF is still physically needed (microECoG records over ~1-3 mm FWHM cortical region, per Trumpis 2020 / Chiang 2020 / Viventi 2011 — same rationale as `#5`). In the cvs_avg35 path we expressed it as 3D volumetric convolution `G ∗ PM_p`. In the fsaverage path it is 2D geodesic smoothing on fsaverage's cortical mesh: `mri_surf2surf --fwhm 3.5 --hemi lh` applied to the baked `lh.BNA.gii`, producing `lh.BNA_smoothed.gii`.
    - **Why the surface version is strictly better for ECoG**: a 3D Gaussian centered on a contact at one sulcal bank puts ~15% of its weight on the opposite bank at thin (≤2 mm) sulci. The electrode cannot physically see the opposite bank. A 2D geodesic smoothing walks along the cortical manifold, so it cannot cross sulcal walls — matches the actual physics of surface recording.
    - **σ unchanged**: 1.5 mm ⇒ FWHM 3.5 mm, same as `#5`'s rationale.
    - **sEEG note**: for Phase 2 depth electrodes, the volumetric Gaussian comes back because depth contacts sample multiple cortical sheets through the ribbon volume. Phase 2 gets a dual path (surface for ribbon contacts, volumetric for the Gaussian if revisited).

  - **LOCKED: sEEG ribbon-only rule carries over unchanged**. The Phase 2 sEEG inclusion rule already tracked under `## Phase 2 → sEEG modality join` (ribbon check → FreeSurfer column walk → drop non-ribbon contacts, no fallback) is unchanged by `#36`. On fsaverage, "ribbon contacts go through the surface pipeline" means the same stock `mri_vol2surf --projfrac-avg` routing, and "non-ribbon contacts drop" means the same `token_mask = False`.

  ### Open sub-decisions (must resolve before execution)

  - **How to obtain `ICBM152_fs`**: the FreeSurfer recon of the ICBM152 2009c template. Options:
    1. Download from TemplateFlow, OSF, or the FreeSurfer contrib assets — community has almost certainly published this. Preferred if available. Resolution: check TemplateFlow's `MNI152NLin2009cAsym_fs` asset and neuromaps's bundled templates.
    2. Build it once via `recon-all` on the ICBM152 2009c T1. ~8 hours of CPU, overnight, one-time. Acceptable fallback.
    3. Use fsaverage directly and run `neuromaps.mni152_to_fsaverage` — simpler but has an additional ~1-2 mm gyral residual because it skips ICBM152_fs's intermediate surface alignment. Rejected as the primary path (would leave 1-2 mm of avoidable atlas-side residual) but acceptable as an A/B comparison baseline.

  - **Patient surface target**: the projection target for each patient is fsaverage's `lh.pial-outer-smoothed` (envelope), not `lh.pial`. This matches the current cvs_avg35 convention (electrodes sit on the dura ≈ envelope, not on the ribbon) and keeps the sphere.reg residual interpretable. Check: does fsaverage ship `lh.pial-outer-smoothed`? If yes, use it. If no (or if it is generated on the fly by FreeSurfer's envelope algorithm), run the envelope generation once for fsaverage as a one-time precompute.

  - **Handling vertices outside ICBM152's ribbon after projection**: a small minority of fsaverage vertices will land outside BNA's native ribbon even after surface-to-surface alignment (because ICBM152's 40-subject average ribbon is itself slightly softer than any single recon). Options: (a) leave these vertices with a zero 246-vector, and rely on `token_mask` to drop electrodes whose nearest vertex is zero; (b) apply a small-radius (1-2 mm) geodesic dilation on the baked surface to propagate into the remaining holes. Decide before freezing the bake.

  - **Phase 1 Tier-1 parcel list stability check**: after baking BNA onto fsaverage, re-run `scripts/rank_parcels_by_support.py` against the new baked labels and compute argmax_wins per parcel. Expected: the Tier-1 list of 15 parcels is unchanged or shifts by at most 1-2 parcels. If the list shifts by more, that signals a genuine alignment difference between the two pipelines and needs investigation before the migration is accepted. The `argmax_wins >= 10` rule itself is unchanged.

  - **External MNI-only datasets (Flinker, Chang) entry point**: the fsaverage pipeline adds a shorter path for external data that only ships MNI coordinates. The path is `MNI152 → MNI305 (canonical affine) → nearest fsaverage pial vertex → read baked BNA label`. Per-electrode accuracy: ~5-10 mm, dominated by whatever accuracy the external lab's own MNI projection had. Still good enough for parcel-level tokens at the BNA Tier-1 scale (15-40 mm parcel diameter). Adapter code is ~50 lines per dataset. This entry point does NOT exist in the cvs_avg35 path and is a scalability win for Phase 3 external data integration.

  ### What dies on execution

  - `data/mni_coords/*_MNI152.csv` cache semantics (replaced by fsaverage-vertex-indexed electrode cache)
  - `src/speech_decoding/v14/coordinates.py` (sub2AvgBrainClinical.m port) — replaced by a stock `sphere.reg` lookup script
  - `scripts/dilate_pm.py` — deleted; `#35` is closed by deletion rather than by patching
  - `data/atlas/BNA_PM_dilated_8mm.nii.gz` — deleted; replaced by `data/atlas/lh.BNA.gii` + `rh.BNA.gii`
  - `cvs_avg35_inMNI152` references in `scripts/matlab/plot_phase1_parcels.m` — either retired with a nilearn/pycortex fsaverage visualization or the MATLAB script is retargeted to `fsaverage`
  - Envelope-vs-pial fudge in `scripts/query_atlas_at_electrodes.py`

  ### What survives on execution

  - Phase-1 model architecture: unchanged. Still consumes `(parcel_label, support, features)` tokens.
  - Tier-1 parcel list (15 parcels, `argmax_wins ≥ 10` rule): to be re-verified on the baked labels; expected ≤ 2 parcels shift.
  - `token_mask` rule from `#3` (`argmax ∈ Tier1`): unchanged.
  - Support statistic semantics from `#5`: `support(e, p) = smoothed BNA probability at the electrode's fsaverage vertex`. The implementation changes from volumetric Gaussian convolution + trilinear sampling to surface geodesic smoothing + vertex lookup, but the semantics (unitless BNA percentage ∈ [0, 100]) are identical.
  - Parcel-frame construction from `#10`: `#10` is REOPENED as of 2026-04-15 and is no longer a frozen contract. The draft uses fsaverage-based second-moment tensors, which would be directly compatible with `#36`'s fsaverage bake, but the sub-choices (nilearn `vol_to_surf` vs `mri_vol2surf`, rotation method, caching schema) are open again and must be re-agreed alongside or after `#36` executes.
  - Phase 2 sEEG ribbon-only rule: unchanged.

  ### Accuracy budget comparison

  | source of error | cvs_avg35 (current) | fsaverage surface-to-surface (proposed) |
  |---|---|---|
  | patient → template | ~1–3 mm (Zac's custom surface walk) | ~1–3 mm (stock `sphere.reg`) |
  | atlas ↔ template ribbon gap | **5–8 mm at lateral crowns**, papered with 8 mm dilation | **0 mm** (eliminated — labels live on pial vertices) |
  | atlas-side `sphere.reg` residual | — | ~0.5–1 mm |
  | ribbon-sampling averaging at parcel borders | — | ~0.5–1 mm |
  | ICBM152_fs pial softness at gyral crowns | — | sub-mm |
  | **per-electrode total (worst case)** | **~3–8 mm** | **~2–4 mm** |

  Net gain: ~1-4 mm, most of it from eliminating the ribbon gap entirely. Remaining residuals are all standard sphere-registration physics, not template-mismatch hacks.

  ### Execution cost

  - ~1 hr: check TemplateFlow for `ICBM152_fs`; download or queue a `recon-all` overnight job
  - ~2 hr: write `coordinates_v2.py` around stock FreeSurfer `sphere.reg`
  - ~1 hr: write `scripts/bake_bna_on_fsaverage.py` (the 246-channel `mri_vol2surf` + `mri_surf2surf` loop)
  - ~5 min: `mri_surf2surf --fwhm 3.5` for the Gaussian PSF smoothing
  - ~1 hr: re-run `rank_parcels_by_support.py` against baked output, diff Tier-1
  - ~1 hr: update or replace MATLAB viz with nilearn fsaverage render
  - ~30 min: delete `dilate_pm.py`, cvs_avg35 references, update blocker docs (close `#35` by deletion, update `#3` / `#4` / `#5` to reference the fsaverage pipeline)

  **Total: one working day** (plus overnight `recon-all` if ICBM152_fs is not downloadable).

  ### Gate for execution

  Two things before flipping the switch:
  1. **Confirm `ICBM152_fs` is obtainable** (downloadable from TemplateFlow, or we accept the overnight `recon-all` cost).
  2. **Agree that the Phase 1 Tier-1 list is expected to survive re-ranking on the baked fsaverage labels.** Run the baking script, re-rank, diff. If ≤ 2 parcels shift, proceed. If > 2 parcels shift, pause and investigate before committing.

  Until both gates clear, `#35` (the 8 mm dilation) remains the active Phase-1 spatial pipeline — do not prematurely delete the dilation artifacts or coordinate CSVs.

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
  - **Blocked**: `#10` was REOPENED 2026-04-15 and is no longer a frozen contract. Do not run the builder against the current draft. The builder steps below stay checked in as a reference implementation target, but every sub-choice is subject to revision when `#10` actually closes.
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
- [ ] **Surface-mesh calibration (option opened by the v14 coordinate path, not yet committed)**
  - Context: Phase 1 coordinates come from Zac's `sub2AvgBrainClinical.m`, which projects every subdural electrode to a **vertex on the `cvs_avg35_inMNI152` outer-smoothed pial mesh** via spherical registration. Electrodes are therefore not free-floating points in MNI — they are indexed nodes on a shared cortical mesh.
  - This opens learned-calibration options that were not available under an affine-only transform:
    - **Geodesic distance bias** between electrodes (along the mesh) instead of or alongside Euclidean MNI distance, for any inter-region attention bias that uses pairwise distance.
    - **Surface-based `Δ/ω` corrections** — per-patient rigid calibration parametrized in the tangent plane of the local cortical surface rather than in free 3-space, which respects the fact that the surgeon's drift is along the cortex, not through it.
    - **Parcel-frame encoders defined on the pial mesh**, where the local frame at each electrode uses the true per-vertex surface normal and principal curvature directions, not a single per-parcel average normal (ties into the per-voxel normal upgrade tracked in `#10`).
  - **Out of scope for Phase 1.** Flagged here so we do not lose the option when Phase 2 calibration design begins. Phase 1 uses the MNI coordinates as plain 3D points for PM membership, parcel-frame construction, and inter-region bias — the mesh structure is a Phase 2 upgrade, not a Phase 1 requirement.

### sEEG modality join

- [ ] **sEEG contact inclusion rule — ribbon-only, no fallback** — DECIDED 2026-04-14 (design commitment; implementation in Phase 2 when sEEG actually joins)
  - **Decision**: every sEEG contact passes through a ribbon check *first*. If the contact voxel is inside the subject's cortical ribbon (between `lh.white`/`rh.white` and `lh.pial`/`rh.pial`, as per FreeSurfer's `ribbon.mgz` or `aparc+aseg.mgz`), it goes through the standard surface pipeline. If it is *not* in the ribbon, `token_mask = False` and the contact is dropped with no fallback path. No volumetric BNA sampling for sEEG, no allocortex atlas, no hippocampal-subfields integration, no thalamic/basal-ganglia tokens. One in-path, one out-path.
  - **Ribbon → pial-vertex projection uses FreeSurfer's cortical column walk**, not straight-line Euclidean kNN. An sEEG contact 8 mm deep in a sulcal fold must attribute to *its own* cortical bank, not to a Euclidean-nearest pial vertex that may sit on the opposite bank or a different gyrus. Mechanically: `mri_vol2surf --projfrac-avg` (or a Python port) walks from white → pial through the contact's column to find the correct pial vertex. From there, standard `sphere.reg` → fsaverage vertex → baked BNA label. Same surface pipeline as ECoG from step 2 onward; the only substitution is the column walk at step 1.
  - **Rationale for the hard-drop rule**: (a) every non-ribbon case was going to be excluded anyway — white-matter transit contacts don't carry clean parcel-specific signal, CSF contacts are degenerate, allocortex structures (hippocampus, amygdala, entorhinal) are not in the Phase-1 Tier-1 parcel layout and wouldn't produce tokens even if we tried to label them. (b) Consolidating these under one exclusion rule keeps the token space homogeneous: every sEEG token came from the same machinery as every ECoG token (fsaverage pial vertex → BNA label), so the model's embedding layer sees a uniform distribution regardless of modality. (c) It eliminates a whole sub-pipeline (volumetric sampler, allocortex atlas, second label space) that earns no measurable Phase-1 benefit — thalamus and basal ganglia are not standard clinical sEEG targets and mesial-temporal allocortex is covered by a different research question if/when it comes up.
  - **Fraction-dropped budget**: for a typical clinical sEEG implant, ribbon-residence is high because contacts are intentionally placed in cortex. The excluded fraction is mostly transit contacts between targets, which we would mask out regardless. No reason to expect ≥10% token loss from the rule.
  - **What gets added to the codebase**:
    - ribbon mask loader (~10 lines, stock FreeSurfer `ribbon.mgz` or `aparc+aseg.mgz`),
    - column-walk wrapper around `mri_vol2surf --projfrac-avg` (~30 lines; or a pure-Python port),
    - glue into the existing `(parcel_label, support, features)` token path (nothing new — sEEG produces the same tuples as ECoG).
  - **What does NOT get added**: volumetric BNA sampler for sEEG, hippocampal-subfields-T1 integration, separate allocortex atlas, thalamic/basal-ganglia parcels, any second label space. If a later research question requires mesial-temporal or deep-gray tokens, it is a *new* Phase-3+ scope decision and reopens this blocker; it is not a silent extension.
  - **Relationship to the fsaverage base decision** (tracked under the ECoG side): this rule assumes the Phase-1 ECoG pipeline has already been moved off cvs_avg35 and onto fsaverage pial + neuromaps-baked BNA vertex labels. On fsaverage, "surface pipeline" means stock `sphere.reg` from every patient's recon-all output + pre-computed per-vertex BNA probability vectors + geodesic Gaussian smoothing on the mesh. If Phase 1 stays on cvs_avg35, the sEEG rule above still applies conceptually (ribbon-only, column walk, drop otherwise) but the atlas lookup step uses the dilated BNA volume with the same Phase-1 sampling machinery. Either way, the inclusion rule is unchanged.
  - **Does not fix** (still open and tracked separately):
    - support statistic diversity weighting for shaft-internal redundancy (see #5)
    - per-voxel local cortical normals for parcel-frame rotation in curved parcels (see #10)
    - `DEFAULT_SPLIT_COUNTS` re-derivation from sEEG coverage (see below)
    - modality embedding for the point encoder (see below)

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

## Decisions

Frozen Phase-1 decisions, moved out of the open blocker list so a scan shows only unresolved items. Each entry keeps its full rationale because the reasoning is load-bearing when later blockers cite the decision. Numbering is stable — these are still referenced by their `#N` from elsewhere in the doc.

- [x] **#1 ACPC → MNI transform pipeline — DECIDED 2026-04-14**
  - **Decision**: trust `src/speech_decoding/v14/coordinates.py`, the Python port of Zac's `sub2AvgBrainClinical.m`, as the Phase-1 spatial projection. All 7 Phase-1 LH patients (S14, S16, S23, S26, S33, S39, S62) produce anatomically plausible MNI coordinates on `cvs_avg35_inMNI152`'s pial-outer-smoothed envelope. Output is cached per-patient at `data/mni_coords/<pt>_MNI152.csv` by `scripts/preprocess_mni_coordinates.py`, keyed by electrode *name* (see `#12`).
  - **Oracle verification (S14)**: median residual 0.68 mm, mean 0.70 mm, max 1.39 mm, p95 1.17 mm against Zac's MATLAB reference `tests/v14/fixtures/S14_mni152_avgCoords.csv`. Enforced by `tests/v14/test_coordinates.py::test_s14_projection_end_to_end` with tolerances max ≤ 1.5 mm / median ≤ 1.0 mm / per-axis mean ≤ 0.5 mm. Sub-1.5 mm is smaller than any Brainnetome parcel boundary, so parcel assignment is stable under this residual (which is itself file-version drift from an earlier reconstruction snapshot, not an algorithm bug — detailed diagnosis lives in the test docstring).
  - **Algorithm-level correctness proof**: `scipy.spatial.cKDTree.query(k=1)` matches MATLAB brute-force `argmin(sum((v-p)^2, axis=1))` bit-for-bit at every snap step for S14 (0/128 mismatches on both subject-pial and average-sphere snaps). `mne.read_surface` returns vertex coordinates with no silent cras offset (metadata cras is `[0,0,0]` and marked invalid). The port is numerically equivalent to the MATLAB computation given the same input files.
  - **Cross-patient sanity (non-oracle)**: the 2026-04-14 S26 drift scare was walked back in-session. S26's per-electrode ACPC→MNI L2 delta (26.1 mm, mean direction anterior-ventral) is in-family with S14's oracle-verified delta (32.8 mm, same direction). The ACPC→MNI transform produces displacements of this magnitude for every patient because ACPC is subject-native and cvs_avg35 differs from each subject's folded cortex by tens of mm; S14 only looks "small" because its grid happens to sit over dorsal frontal territory where BNA coverage was never an issue. Zac confirmed his published "S2" rendering is our S26 and it lands in the same anatomical region our pipeline produces.
  - **Post-trust PM-coverage concern is subsumed by `#35`**: the original blocker's sub-item "once MNI is trusted, sample BNA PM at every electrode and confirm nonzero" was motivated by the outer-smoothed surface potentially sitting a few mm outside the cortical ribbon. `#35` (nearest-neighbor PM dilation with `d_max = 8 mm`) handles this by extending the ribbon outward rather than by moving electrodes. On the frozen pipeline, 1280/1280 Phase-1 LH electrodes hit `max_p support(e,p) ≥ 10%`.
  - **Phase 2 upgrade path**: if publication requires exact MNI coordinates with perfect reproducibility, re-run Zac's MATLAB against the current Box reconstruction files and archive the specific file snapshot alongside the cached CSVs. Not needed for Phase 1 training correctness.

- [x] **#3 Unsupported-vs-weak token_mask rule — DECIDED 2026-04-14, revised same day**
  - **Final rule (revised)**:

    ```
    token_mask[e] = True   iff   argmax_p support(e, p) ∈ Tier1
                                 AND max_p support(e, p) > 0
    ```

    where `support(e, p) = (G * PM_p)(x_e)` is computed on the frozen spatial pipeline (raw MNI + 8 mm dilated PM + σ=1.5 mm Gaussian) and `argmax_p` runs over all 246 BNA parcels. The second clause `max_p support > 0` handles the degenerate case `support(e, :) ≡ 0` — an electrode outside every dilated ribbon voxel (e.g. a future sEEG contact in white matter beyond the 8 mm propagation radius, or a poorly-projected electrode in anatomically absent cortex) — where `argmax` is mathematically ill-defined. Such electrodes get `token_mask = False` explicitly. No epsilon or magic threshold.
  - **Rule evolution**:
    1. **Original (2026-04-14 morning)**: `max_p support(e, p) >= 10` over all 246 parcels. A quality-threshold rule. Defensible but with an ad-hoc `10%` number whose justification was "robust to `d_max ∈ [5, 10] mm`" rather than a principled physical meaning.
    2. **Final (2026-04-14 evening)**: `argmax_p support(e, p) ∈ Tier1`. Revised after `#4` switched from eff_N-centric to argmax-centric selection — it became structurally inconsistent for `#3` to still use a `max_support` threshold while `#4` was asking the dual question via `argmax_wins`. The two rules now answer the same "which parcel does this electrode live in" question from opposite sides: `#4` asks "how many electrodes argmax to this parcel" (parcel → electrode count); `#3` asks "does this electrode's argmax land in a parcel we model" (electrode → parcel membership).
  - **Why argmax-in-Tier-1 beats max-support >= 10**:
    - **Direct consistency with `#4`**. An electrode counts toward `argmax_wins(p)` in Tier-1 selection iff `p` is its argmax; the mask now uses exactly the same relationship. These are the same principle applied from both sides.
    - **Pathological case the old rule missed**. An electrode whose true anatomical home is a non-Tier-1 parcel at, say, `15%` support (and whose second-best is some Tier-1 parcel at `8%`) passed the old rule (`max >= 10%`) and would then be force-fed into a Tier-1 parcel it doesn't really belong to. Under the new rule it's correctly masked out — the model only sees electrodes whose anatomical home is a modelable parcel. On current Phase-1 data this case doesn't occur (all 1280 electrodes have argmax in Tier-1 after TE1.0/1.2 was added by `#4`'s revision), but the rule should express the principle, not just handle today's data.
    - **No magic number**. The old `10%` went away entirely from the data-quality gate. The new rule has zero free parameters on the primary clause; the secondary `> 0` clause is a hard structural check for the all-zero edge case, not a tunable threshold.
    - **No coupling decision between `#3` and the smoothing parameters**. Under the old rule, changing `σ` or `d_max` could shift electrodes across the `10%` boundary. Under the new rule, changes to smoothing parameters can only flip an electrode's argmax assignment (still a Tier-1 parcel → mask stays True) or push an electrode's support to uniform zero (all-zero → False). No continuous interaction with a threshold.
  - **Frozen-pipeline verification**: all 1280 Phase-1 LH electrodes pass the new rule (S14 128/128, S16 128/128, S23 128/128, S26 128/128, S33 256/256, S39 256/256, S62 256/256). This is consistent with the diagnostic from the `#4` revision: before TE1.0/1.2 was added to Tier-1 there were 12 electrodes whose argmax (parcel 73) was outside Tier-1; after the `#4` revision those 12 are covered and the mask is a no-op. The old `max_p >= 10%` rule also passes all 1280 on the frozen data, so the two rules agree on current cohort but diverge on the pathological cases described above.
  - **Relationship to Tier-1 selection**:
    - `#4`: "a parcel is Tier-1 iff ≥10 electrodes argmax to it"
    - `#3`: "an electrode is valid iff it argmaxes to a Tier-1 parcel"
    - These are dual. The only electrodes that can fail `#3` are those whose dominant parcel got rejected by `#4`'s `argmax_wins >= 10` clause — i.e., parcels with `1 <= argmax_wins <= 9`. The `argmax_wins ∈ [5, 9]` band is currently empty, and `argmax_wins ∈ [1, 4]` is non-empty only in the leakage-sink tail where each parcel has a tiny handful of dominant electrodes. The new `#3` rule drops those stray electrodes rather than force-feeding them into whichever Tier-1 parcel is their runner-up.
  - **Degenerate input handling**: the `max_p support > 0` secondary clause is a structural guard, not a threshold. It catches two cases that the primary `argmax ∈ Tier1` clause would silently mishandle:
    1. All-zero `support` vector — `numpy.argmax([0,0,...])` returns 0 (first index), which would then be checked against Tier-1 and happen to fail by coincidence. We want it to fail *by intent*, not by accident, so we check the zero case first.
    2. Floating-point noise at parcel-boundary voxels far from any electrode — if every parcel has a numerical value below ~1e-6 but technically nonzero, the argmax is meaningless. The `> 0` guard only catches the exact-zero case; noise at `~1e-6` still flows through. This is intentional: our Gaussian kernel has finite support and the PM values are `float32`, so exact zeros only arise when the electrode sits completely outside the dilated ribbon — which is exactly the "catch genuine outliers" case the mask is for.
  - **Phase 2 note**: when sEEG joins, the rule applies unchanged but expect more `token_mask = False` electrodes — white-matter shaft-internal contacts will genuinely be outside every cortical parcel's support and will fail the `argmax ∈ Tier1` clause (or the all-zero secondary clause if they're beyond the 8 mm dilation radius). That is correct behavior, not a tuning issue.
  - **Implementation contract**: the v14 loader computes the support matrix `(N_elec, 246)` once per patient, takes `argmax` along the parcel axis, tests membership against the Tier-1 index set from `token_spec.DEFAULT_BASE_PARCELS`, and checks the `max_p > 0` clause. Emits `token_mask ∈ {0, 1}^{N_elec}`. No threshold parameter lives anywhere in the v14 config.

- [x] **#4 Base parcel set `N_tok = 15` — DECIDED 2026-04-14, revised twice same day**
  - **Selection rule (final, revised 2026-04-14 after visual inspection)**: a parcel `p` becomes a Tier-1 token iff

    ```
    argmax_wins(p) >= 10
    ```

    where `argmax_wins(p)` is the number of Phase-1 LH electrodes for which `p` is the dominant parcel across all 246 BNA ROIs, computed on the frozen spatial pipeline (raw MNI coords + 8 mm dilated PM + σ=1.5 mm Gaussian smoothing).
  - **Rule evolution (three iterations, all same day)**:
    1. **Original — "top-12 by effective_N"**: a quantity rule dressed up as a quality rule. Replaced because `N_tok = 12` was arbitrary and because top-N cuts can't catch Gaussian-leakage failures (see next).
    2. **"`eff_N >= 10 AND argmax_wins >= 1`"**: caught the leakage-sink failure — parcels that accumulate `eff_N` only because the σ=1.5 mm Gaussian spreads probability into them from their real-cortex neighbors but have `argmax_wins = 0` (A4t M1-trunk, A44op, A4ul M1-upper-limb, A8vl, A6cdl). Replaced because the `eff_N >= 10` clause was itself arbitrary and excluded primary auditory cortex TE1.0/1.2 (`eff_N = 7.14`, `argmax_wins = 12`) despite 12 real dominant electrodes across 3 patients — discovered by visual inspection of `plot_phase1_parcels.m` showing posterior electrodes sitting on gray (unassigned) cortex.
    3. **"`argmax_wins >= 10`" (final)**: directly counts "how many electrodes have this parcel as their anatomical home." No proxies. The cutoff of 10 matches the 10% token-mask threshold from `#3` for numerological consistency, but the choice is robust — on current data there are no parcels with `argmax_wins ∈ [5, 9]`, so any cutoff in `[5, 12]` gives the same 15-parcel list.
  - **Why argmax-centric beats eff_N-centric**: `eff_N(p) = Σ_e support(e, p) / 100` is a *total* — it counts every electrode's fractional contribution, including weak runners-up the Gaussian has spread probability into. `argmax_wins(p)` is a *mode* — it counts only electrodes for which `p` is the best explanation. When these two disagree, the mode is closer to the cortical-anatomy question we actually care about ("how many electrodes live here"), and the total is closer to the sampling-density question ("how much energy does the PSF put into this parcel overall"). For Tier-1 selection we want the former, so we should measure it directly rather than use the latter as a proxy.

  - **The 15 parcels**, ordered by `argmax_wins` high → low (eff_N as tiebreak):

    | rk | idx | name | argmx | eff_N | pts | BNA region |
    |----|-----|------|-------|-------|-----|------------|
    |  1 |  33 | A45c        | 223 | 202.43 | 6 | IFG_L_6_3  caudal area 45 |
    |  2 |  35 | A45r        | 199 | 179.17 | 6 | IFG_L_6_4  rostral area 45 |
    |  3 |  31 | IFS         | 181 | 188.61 | 6 | IFG_L_6_2  inferior frontal sulcus |
    |  4 |  53 | A4hf        | 168 | 121.19 | 5 | PrG_L_6_1  area 4 head/face |
    |  5 |  39 | A44v        | 123 | 112.55 | 6 | IFG_L_6_6  ventral area 44 |
    |  6 |  21 | A9/46v      | 100 |  77.02 | 4 | MFG_L_7_4  ventral 9/46 |
    |  7 |  51 | A12/47l     |  48 |  32.08 | 2 | OrG_L_6_6  lateral 12/47 |
    |  8 |  61 | A4tl        |  44 |  52.15 | 5 | PrG_L_6_5  area 4 tongue/larynx |
    |  9 |  77 | A38l        |  40 |  15.60 | 3 | STG_L_6_5  lateral area 38 (anterior temporal pole) |
    | 10 |  29 | A44d        |  34 |  48.13 | 6 | IFG_L_6_1  dorsal area 44 |
    | 11 | 157 | A1/2/3tonIa |  29 |  26.93 | 2 | PoG_L_4_2  1/2/3 tongue/larynx |
    | 12 | 155 | A1/2/3ulhf  |  28 |  30.10 | 2 | PoG_L_4_1  1/2/3 upper limb/head/face |
    | 13 |  17 | IFJ         |  27 |  23.65 | 4 | MFG_L_7_2  inferior frontal junction |
    | 14 |  63 | A6cvl       |  24 |  31.94 | 5 | PrG_L_6_6  caudal ventrolateral 6 |
    | 15 |  73 | TE1.0/1.2   |  12 |   7.14 | 3 | STG_L_6_3  primary auditory (Heschl's) |

  - **Rejected by the rule** (all parcels in the ranking with `argmax_wins < 10`) — the full tail is either leakage sinks (`argmax_wins = 0`) or has so few dominant electrodes that a dedicated token would be underfit. The `[5, 9]` band is empty on current data, so the cutoff of 10 has comfortable margin. Leakage-sink examples: A4t (59, M1 trunk), A44op (37), A4ul (57, M1 upper limb), A8vl (23), A6cdl (55) — all have `eff_N` between 8 and 23 but `argmax_wins = 0`.
  - **TE1.0/1.2 (primary auditory cortex, parcel 73) — why it finally made it**: rank 15 under the new rule. 12 dominant electrodes across S33 (5), S62 (6), S26 (1). Mean support 35.2%, max 46.0%. Anatomically task-critical — it's primary auditory cortex on Heschl's gyrus, where the stimulus non-word lands, and the PS task is built around listening before repeating. Excluded by the earlier `eff_N >= 10` clause on a technicality (total `eff_N = 7.14` because the parcel is small and only 12 electrodes hit it), included now that we measure anatomical home directly. This is the reason the rule was revised.
  - **A38l anatomy note (unchanged from previous iteration)**: A38l (lateral anterior temporal pole, area 38) is anatomically surprising for a non-word repetition task — usually associated with semantic/lexical processing, not the motor-articulatory core. Per-patient breakdown: S26 contributes 11.22 of 15.60 eff_N (72%), S33 and S62 contribute the rest. `max_pm = 41.5%` is the second-lowest in the Tier-1 list (after TE1.0/1.2 at 46.0%), meaning no electrode sits squarely inside A38l; the 40 argmax wins are all cases where A38l is the best of several weak options. Real risk that S26's A38l support is 8 mm dilation propagating probability onto electrodes ~6 mm above the temporal pole. Still **flagged for visual verification** — if `plot_phase1_parcels.m` shows S26's grid reaching the lateral anterior temporal pole, keep it; if clearly dilation smearing, drop it and `N_tok` becomes 14.
  - **Default split map `k_parcel = 1` uniform**: no 2-token splits in Phase 1. `N_tok = 15`. The 2-token variant stays on the ablation list and requires `parcel_frames.npz` (`#10`).
  - **`N_tok = 15 ≤ 32`** budget cap clears comfortably. Token-level connectivity bias expansion (`#8`) builds its default bias matrix over these 15 parcels; sibling-token expansion is a no-op because every parcel has `k=1`.
  - **Ranking methodology**: `scripts/rank_parcels_by_support.py` with default flags (`--pm-path data/atlas/BNA_PM_dilated_8mm.nii.gz`, `--sigma-mm 1.5`). Full ranking at `data/atlas/parcel_support.csv`. The `argmax_wins` column is now the primary selection column.
  - **Aggregations dropped**: STGa, STGpp, INSa, MFG aggregates from earlier v12/v14 drafts are *not* in the Tier-1 set. Outside A38l (rank 9) and TE1.0/1.2 (rank 15), the ranking shows zero meaningful temporal and zero insular coverage across 1280 electrodes.
  - **Core-patient intersection check**: S14, S26, S33, S62 all contribute to multiple Tier-1 parcels. S14's grid is dorsal (A4hf, A4tl, A6cvl); S26's is ventral-anterior (A45r, A12/47l, A44v, A38l); S33 and S62 cover IFS/A45c/A44v as well as A1/2/3tonIa/ulhf and TE1.0/1.2.
  - **`token_spec.py` status**: `DEFAULT_BASE_PARCELS` in `src/speech_decoding/v14/token_spec.py` now holds the 15-parcel list above, ordered by `argmax_wins` descending. `PROVISIONAL_TOKEN_SPEC = False`. Tests under `tests/v14/test_token_spec.py` enforce the count and the uniform `k=1` split map.

- [x] **#35 Atlas gap-filling via nearest-neighbor PM dilation — DECIDED 2026-04-14** (new blocker, decided immediately)
  - **Problem**: `cvs_avg35_inMNI152/surf/lh.pial-outer-smoothed` (the surface `sub2AvgBrainClinical.m` projects electrodes onto) sits 2-5 mm outside BNA's MNI152 cortical ribbon at lateral gyral crowns, because cvs_avg35 was nonlinearly warped into MNI152 *after* a different FreeSurfer CVS-35 reconstruction — it is not a recon of the ICBM152 template itself. Raw sampling of `data/atlas/BNA_PM_4D.nii.gz` at electrode MNI coords therefore drops ~half of S26's 128-electrode grid into all-zero voxels, even though the grid is anatomically legitimate (classic STG+parsopercularis speech grid per subject-native FreeSurfer aparc+aseg). Other patients (S14, S16, S23, S33, S39, S62) happen to sit over territory where the mismatch is small.
  - **Rejected alternative 1 — electrode snap to `cvs_avg35 lh.pial`**: moves electrodes 0-4 mm inward. Was deployed briefly; didn't fix the problem because the pial itself extends beyond BNA's ribbon at lateral crowns. Also introduces a per-electrode mesh dependency that breaks the symmetry with Phase-2 sEEG (sEEG contacts have no natural pial anchor).
  - **Rejected alternative 2 — replace `cvs_avg35_inMNI152` with a FreeSurfer recon of the ICBM152 template**: would reduce the ribbon mismatch from ~5 mm to ~1-2 mm but costs ~1 day of FreeSurfer compute to recon the template and regenerate `{lh,rh}.sphere-outer-mni.reg` for every subject, plus re-verifying S14 against the oracle with a new sphere target. Not worth the work when `#35` fixes the same problem in 10 lines of preprocessing.
  - **Rejected alternative 3 — raise σ on the Gaussian PSF smoothing**: σ=2.5 mm isotropic would partially help but can't bridge 5-8 mm gaps (kernel mass falls off as `exp(-8²/(2·2.5²)) ≈ 0.6%`). Larger σ bleeds across sulcal banks. Gaussian is for PSF modeling (which has its own physically-correct σ from `#5`), not for ribbon-offset correction.
  - **Decision**: nearest-neighbor PM dilation. `scripts/dilate_pm.py` runs `scipy.ndimage.distance_transform_edt(~any_pm, return_indices=True)` to compute, for every voxel outside BNA's ribbon, its distance and the index of its nearest in-ribbon voxel. For every such voxel within `d_max = 8 mm`, the full 246-parcel PM vector of the nearest ribbon voxel is copied in. Voxels beyond `d_max` stay zero; electrodes there get `token_mask = False` at load time per `#3`. Output: `data/atlas/BNA_PM_dilated_8mm.nii.gz` (float32, same shape + affine as `BNA_PM_4D.nii.gz`). A safer-default `BNA_PM_dilated_5mm.nii.gz` is also cached but is not the production default — 8 mm is required to recover S26's far-lateral pocket (the 5 mm version leaves ~16 S26 electrodes below threshold).
  - **Why this is safe**: dilation runs once at atlas-load time and produces a cacheable artifact. Electrodes never move. The downstream Gaussian-PSF smoothing and trilinear sampling code is byte-identical to the pre-dilation pipeline; only the input volume changes. The physics of `support(e, p) = (G * PM_p)(x_e)` is unchanged because nearest-neighbor-dilated voxels enter the convolution identically to native ribbon voxels. Visualizable by pointing `scripts/matlab/plot_phase1_parcels.m` at the dilated file.
  - **Sharp edge — cross-sulcal propagation**: `distance_transform_edt` is straight-line Euclidean, not geodesic along cortex. At thin sulci (STS, IFS, parts of CS are 3-5 mm wide) the nearest-in-ribbon voxel for a point inside the sulcus could be on the opposite bank rather than the same gyrus. `d_max = 8 mm` is at the edge of where this matters; `d_max = 5 mm` would be safer but leaves S26 partially uncovered. The safeguard is visual inspection — re-rendering `plot_phase1_parcels.m` against the dilated volume and confirming no parcel colors visibly cross sulci. If visual inspection shows wrong-side propagation in a region we care about, fall back to `d_max = 5 mm` and accept that S26 contributes fewer electrodes to A12/47l (which is the region most at risk).
  - **Alternative-3 composition**: the dilation composes with the σ=1.5 mm Gaussian PSF smoothing from `#5` without interaction — the forward physics `support(e, p) = (G * PM_p)(x_e)` treats dilated and native voxels identically. A ribbon voxel that "owns" 20 dilated voxels via NN propagation adds its PM vector into the Gaussian-integrated support of electrodes near those 20 voxels, which is exactly the correct behavior if we believe the dilated voxels are anatomically cortex that BNA simply didn't label at that location.
  - **Rebuild**: re-run `.venv/bin/python scripts/dilate_pm.py --d-max 8` any time `data/atlas/BNA_PM_4D.nii.gz` changes. Deterministic, ~5 seconds.
  - **Phase 2 sEEG note**: dilation is symmetric — sEEG contacts sample the same dilated volume at their MNI coordinates. No surface anchoring, no snap step, no cortical-column logic needed for Phase 1. A future physics-aware refinement (geodesic-along-cortex dilation, or per-contact column-depth kernels for sEEG) is an upgrade path, not a required Phase-1 fix.

- [x] **#5 Parcel support statistic — DECIDED 2026-04-13, amended 2026-04-14**
  - **Per-electrode formula (amended 2026-04-14)**: `support(e, p) = (G ∗ PM_p)(x_e)`, where `G` is a 3D isotropic Gaussian with σ = 1.5 mm (FWHM ≈ 3.5 mm) and `x_e` is the electrode's MNI coordinate. Physically, this is the forward PSF model for intracranial microECoG: each cortical voxel emits, and the electrode's response is a Gaussian-weighted integral of nearby emissions. By the convolution theorem it equals `∫ G(y − x_e) · PM_p(y) dy`, so the implementation pre-convolves the 4D PM volume once at load time (via `scipy.ndimage.gaussian_filter` with a separable kernel on the 3 spatial axes; the parcel axis is untouched) and then trilinear-samples the convolved volume at each electrode's raw MNI position.
  - **σ rationale**: Trumpis 2020 Cogan-lab 1 mm uECoG high-gamma PSF FWHM ≈ 2 mm; Chiang 2020 FWHM ≈ 1.5 mm; Viventi 2011 (500 µm) FWHM ≈ 0.5 mm. σ=1.5 mm ⇒ FWHM ≈ 3.5 mm, slightly wider than pure physics to absorb 1-3 mm of snap/projection uncertainty without crossing into macro-ECoG territory (FWHM 5-10 mm). Tunable in `rank_parcels_by_support.py --sigma-mm`; pass 0 for pure trilinear A/B comparison.
  - **Patient/parcel rollup**: `effective_N(parcel p) = Σ_e support(e, p) / 100` over all Phase-1 non-artifact electrodes of a patient. This is the rollup direction used for the parcel ranking (`#4`). Both slicing directions — per-electrode parcel membership and per-parcel effective_N — are marginals of the same `(N_electrodes × N_parcels)` matrix `M[e, p] = support(e, p)`. Same implementation, different axis of sum.
  - **PM volume used**: `data/atlas/BNA_PM_dilated_8mm.nii.gz` (nearest-neighbor dilation of `BNA_PM_4D.nii.gz`; see `#35`). The dilation fills cvs_avg35-pial-vs-MNI152-ribbon gaps at lateral gyral crowns and composes linearly with the Gaussian convolution.
  - **Units**: `support(e, p) ∈ [0, 100]` (BNA PM percentage). `effective_N` sums these divided by 100, so `effective_N ≈ electrode-count` when every contributing electrode has peak support near 100%. This matches `#3`'s `10%` threshold definition.
  - **Purely geometric**: does not depend on task-responsiveness or sig-channel selection. Channel inclusion is `#11`. Static per `(electrode, parcel)` pair; does not vary over time.
  - **Rejected alternatives (named so they stay rejected)**:
    - raw-count membership (electrode in single MPM parcel): throws away PM uncertainty, loses sub-voxel information, badly behaved at parcel boundaries.
    - task-responsiveness-weighted sum: couples atlas calibration to the supervised task, breaks Phase-1.5 SSL (cannot compute support for an SSL window without running the significance test).
    - per-electrode K-nearest parcels: arbitrary K, no physical interpretation.
  - **Phase 2 / sEEG note**: a flat PM-weighted sum over-rewards redundant shaft-internal sampling. When sEEG joins, the formula needs a diversity weighting — downweight contacts close to each other in MNI (or close along a shaft axis). Safe to defer: the Phase 1 formula is replaceable without breaking the `token_mask` / `token_support` interface.

- [x] **#7 How `token_support` enters the model — DECIDED 2026-04-13**
  - **Decision**: concat-to-token. The per-parcel scalar `token_support` is concatenated onto the token feature and passed through a linear projection back to `d`, so every token that enters inter-region attention carries its own support signal as part of its content. Attention-bias stays on the ablation list but is **not** active in the Phase 1 baseline.
  - **Rationale**: concat-to-token and attention-bias answer different questions, so they are not redundant. Concat puts support inside the token's representation — the model learns what low support means for content. Attention-bias externally discounts how much *other* tokens attend to a low-support token and does not touch its content.
  - A low-support token can still be the only evidence in its parcel (e.g. one contact in Broca's for some patient). The right behavior is "let other tokens attend fully, but mark it as low-confidence internally". That is concat-to-token without attention-bias. Attention-bias would wrongly suppress the only evidence available.
  - Conversely, a high-support token may be task-irrelevant, and attention-bias cannot express that (support is high, so no discount fires).
  - **Phase 1 discipline**: only one mechanism active in the baseline so ablations stay readable. Concat-to-token is the default; attention-bias is a named ablation comparison.
  - **Implementation note**: the concat dimension is `d + 1` before the linear projection, so the point-summarizer output `d` is preserved for the backbone. `token_mask` is still a separate binary signal and gates attention structurally — concat-to-token handles graded support inside the active set, `token_mask` handles hard absence.

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

- [x] **#32 Input normalization at the v14 boundary — DECIDED 2026-04-13**
  - **Decision**: no additional normalization beyond upstream `productionZscore_highgamma`. The loader hands the model upstream-z-scored HGA features directly.
  - **Rejected alternatives** (named so they stay rejected):
    - per-sample re-normalization — throws away session-level calibration
    - per-batch re-normalization — leaks distribution info across trials in a batch
    - per-patient re-normalization at load time — redundant with upstream z-score
  - **Implication**: if the point encoder or temporal front-end (`#2` / `#6`) benefits from a `LayerNorm` at its input, that is an in-model architectural choice, not a data-side normalization. Discuss inside `#2` / `#6` if needed.

- [x] **#33 PER metric exact definition — DECIDED 2026-04-13**
  - **Decision**: slot-averaged PER. `PER = 1 − (correct slots / total slots)`, computed over all three slots of every trial, averaged over all trials in the held-out fold. Matches the old per-phoneme baseline `0.734 ± 0.007` so the comparison against v14 is apples-to-apples.
  - **Reporting**: per-patient PER and a population mean. 3-seed aggregation: mean ± std across seeds. Population mean is reported after per-patient numbers, not instead of them.
  - **Alternative considered and rejected**: trial-level "any slot wrong" accuracy. Harder to compare against the baseline and discards information about where errors land.
  - **Diagnostic only (not headline)**: per-slot PER (slot 0 / 1 / 2 separately) is reported for the first training run to check for slot-position bias, then dropped from headline reporting unless a bias shows up.
