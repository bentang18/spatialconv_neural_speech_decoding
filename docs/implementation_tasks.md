# v14 Implementation Tasks

Current goal:

- implement and verify `v14-core` on intra-op `uECoG` only
- start with a fixed atlas, no learned per-patient calibration
- Phase 1 is supervised only

This file is the source of truth for:

- open blockers
- frozen Phase 1 contracts
- the small set of implementation tasks that remain once blockers are closed

It is not a design notebook. Long rationale belongs in code comments, tests, or separate notes only when needed.
The full historical version is preserved at `docs/archive/implementation_tasks_archived.md`.

## Status (2026-04-15)

Open blockers (1): `#34`

Decided blockers (35): `#1 #2 #3 #4 #5 #6 #7 #8 #9 #10 #11 #12 #13 #14 #15 #16 #17 #18 #19 #20 #21 #22 #23 #24 #25 #26 #27 #28 #29 #30 #31 #32 #33 #35 #36`

Critical path to coding:

- loader/audit side: `#34`
- implementation readiness: `#12 #13 #14 #15 #25` are now frozen and can be executed

## Working Rule

- discuss logic first
- freeze the contract
- then write code
- when two choices are equally good for Phase 1, prefer the one that scales better to cross-task use and external datasets
- do not reuse legacy code directly

## Open Blockers

- [x] `#12` Amp -> physical electrode -> coordinate bridge
  - The stable part of the bridge terminates at a physical electrode name.
  - Under the pre-fsaverage cache, that name looks up `data/mni_coords/<pt>_MNI152.csv`.
  - Under the accepted fsaverage pivot (`#36`), the same name key will attach to the patient-side surface projection / fsaverage-vertex cache.
  - Do not bridge through `electrodes.tsv` names.
  - Do not reuse the old 256-ch direct `fif N -> RAS N` shortcut.
  - Name-keyed coordinates are canonical:
    - lookup key is the concatenated name from `<pt>.electrodeNames`
    - name is canonicalized to the concatenated form (`SMC16`, not `SMC 16`)
    - the numeric suffix in the name is not a row index
  - Layout rules:
    - 128-strip patients (`S14 S16 S22 S23 S26`) use Map 4 from local `*_channelMap.mat`
    - Map 4 is the spreadsheet layout plus a local `+1` shift
    - bridge: `fif amp channel -> (r,c) with chanMap[r,c] == amp -> phys_idx = r*16 + c + 1 -> physical name -> MNI coord by name`
    - 256-grid patients (`S33 S39 S62`) use Map 3 from `*_channelMapAll.mat`
    - bridge: `fif amp channel -> (r,c) in Map 3 -> physical electrode at that grid location -> physical name -> MNI coord by name`
    - `S58` also uses Map 3, but local `S58_channelMap.mat` is a 12x24 crop with zero-indexed amp values `0..255`
    - `S58` must first resolve the crop to a contiguous row-slice of full Map 3, then use the same Map 3 bridge
    - stray `S39_channelMap.mat` is non-authoritative and must never be loaded
  - Ordering invariant to verify per patient:
    - `.electrodeNames` row `i`
    - `.LEPTO` row `i`
    - local `_RAS_brainshifted.txt` row `i`
    - cached `*_MNI152.csv` entry with the same canonicalized name
  - Verifier contract:
    - for every retained `.fif` channel, return exactly one physical name and exactly one cached coordinate
    - fail loudly on ambiguity, missing entries, duplicates, crop mismatch, or row-order mismatch

- [x] `#13` Explicit `v14-core` loader interface
  - Write a fresh `v14-core` loader.
  - Artifact channels are removed from the active channel axis, not zeroed in place.
  - Baseline emitted sample fields:
    - `signal[N_ch, T]`
    - `coords[N_ch, 3]`
    - `patient_id`
    - `label`
    - `token_mask[N_tok]`
    - `token_support[N_tok]`
  - Channel inclusion follows `#11`: all non-artifact channels, no hidden sig-channel filtering.

- [x] `#14` Legacy audit
  - Legacy code remains quarantined under `src/speech_decoding/archive/legacy/`.
  - Active `v14` code is written fresh.
  - Legacy code may be consulted as historical reference only.
  - No direct reuse or copy-paste from quarantine.

- [x] `#15` Clean `src/speech_decoding/v14/config.py`
  - `config.py` should mirror frozen Phase 1 contracts only.
  - Remove stale speculative defaults.
  - Frozen width budget:
    - baseline `d_model = 64`
    - first width ablation `d_model = 128`
    - `head_dim = 32`
  - Frozen module defaults implied by the contracts above:
    - tokenizer: `200 Hz`, `150 ms` kernel, `50 ms` stride
    - backbone: `B = 2`, `4d` FFN, dropout `0.1`
    - decoder: fixed 3-slot AR decoder, one block, shared vocab head
  - Phase 2 learned-calibration leash constants must not act like live Phase 1 defaults.

- [x] `#25` Ground-truth phoneme fixture tests
  - Add fixture tests against `data/ps_tokens.csv`.
  - Minimum coverage:
    - `normalize_label` round-trip on canonical phonemes
    - `filter_to_ps_phonemes` in-set vs out-of-set behavior
    - optional exact inventory check against the manifest

- [ ] `#34` End-to-end phoneme loading and trial-timing audit
  - Re-verify the whole `.fif -> (signal window, 3-phoneme target)` path from actual data.
  - Core gate patients:
    - `S14`
    - `S26`
    - `S33`
    - `S62`
  - Extended audit in parallel:
    - `S16`
    - `S23`
    - `S39`
  - Must-pass checks:
    - `.fif` `event_id` matches the frozen PS mapping exactly
    - trials are reconstructed from explicit metadata, not stride-3 assumptions
    - every trial has exactly 3 phoneme events
    - every 3-phoneme sequence maps to exactly one entry in `data/ps_tokens.csv`
    - trial counts agree across `.fif` and BIDS event tables after explicit upstream exclusions
    - `t=0` in the epochs really means response onset
    - the frozen `[-0.5, 1.0] s` window lands where expected on real trials
    - no silent contamination from non-task / rest / malformed epochs
  - Spot-check only:
    - audio sanity checks on a small handful of trials per patient
    - MFA boundary comparison as audit-only cross-check, not a training dependency
    - PS vs Lexical `event_id` comparison as warn-only metadata
  - Deliverables:
    - one short markdown audit report
    - one rerunnable structural audit script
    - one small timing / plotting sanity-check script
    - per patient verdict: `pass`, `warn with workaround`, or `fail`
  - Closure rule:
    - `#34` closes only when the report and scripts exist
    - v14 loader remains blocked until every core patient is either `pass` or `warn with workaround`

## Frozen Phase 1 Contracts

### Spatial

- [x] `#1` ACPC -> MNI pipeline
  - Current trusted pre-fsaverage projection oracle is the Python port of Zac's `sub2AvgBrainClinical.m`.
  - Coordinates are cached at `data/mni_coords/<pt>_MNI152.csv`.
  - Lookup key is physical electrode name.
  - This is the trusted reference path to compare against while the fsaverage pipeline is executed.

- [x] `#3` Electrode validity / mask rule
  - Electrode is valid iff its parcel argmax lands in Tier-1 and `max_p support(e,p) > 0`.
  - No free threshold in the baseline rule.

- [x] `#4` Tier-1 parcel set
  - `N_tok = 15`
  - Selection rule: `argmax_wins >= 10`
  - Uniform `k_parcel = 1` in Phase 1
  - Tier-1 parcels:
    - `A45c`
    - `A45r`
    - `IFS`
    - `A4hf`
    - `A44v`
    - `A9/46v`
    - `A12/47l`
    - `A4tl`
    - `A38l`
    - `A44d`
    - `A1/2/3tonIa`
    - `A1/2/3ulhf`
    - `IFJ`
    - `A6cvl`
    - `TE1.0/1.2`

- [x] `#5` Support statistic
  - Per-electrode:
    - `support(e,p) = (G * PM_p)(x_e)`
  - `G` is Gaussian with `sigma = 1.5 mm`
  - Units are BNA PM percentage in `[0, 100]`
  - Parcel rollup:
    - `token_support[p] = sum_e support(e,p) / 100`

- [x] `#10` Parcel-frame construction
  - Build parcel frames directly from the baked fsaverage atlas from `#36`.
  - Do not create a second volumetric or `nilearn vol_to_surf` frame path.
  - Frame geometry uses the raw baked parcel weights before geodesic PSF smoothing.
  - The frame lives on the fsaverage pial surface and is a 2D local tangent chart, not a 3D volumetric frame.
  - Origin:
    - PM-weighted parcel centroid on fsaverage
    - snap to the nearest in-parcel fsaverage vertex
  - `z` axis:
    - compute the PM-weighted mean of in-parcel pial vertex normals
    - flip it if needed to agree with the origin-vertex pial normal
    - normalize it
    - if the weighted-mean norm is numerically too small, fall back to the origin-vertex pial normal directly
  - `u` axis:
    - project the global anterior direction into the tangent plane orthogonal to `z`
    - if that projection is numerically too small, fall back in order to projected superior, then projected left-right
    - use the same fallback order for every parcel
  - `v` axis:
    - `v = normalize(z × u)`
    - this is the only sign rule for `v`
  - Electrode coordinate:
    - project the electrode to fsaverage pial
    - subtract the parcel origin
    - express the signed local offsets in the `(u, v)` basis
    - no third coordinate in the baseline chart
  - Per-axis scale:
    - compute PM-weighted `sigma_u, sigma_v` from in-parcel fsaverage vertices in the same chart
    - feed `(u / sigma_u, v / sigma_v)` to the point encoder
    - append `(log sigma_u, log sigma_v)` once per parcel token as side metadata
  - Cache:
    - `data/atlas/parcel_frames.npz`
    - store parcel id/name/hemisphere, origin vertex index, origin xyz, basis matrix `[u, v, z]`, `sigma_u`, `sigma_v`, and build metadata identifying the atlas bake
  - QC requirement before training:
    - the frame builder must emit visual QC figures on fsaverage for every Tier-1 parcel
    - each figure must show the parcel support map, origin, `u` axis, `v` axis, and local normal `z`
    - inspect all 15 Tier-1 parcels once before trusting `parcel_frames.npz`
    - fail the build on any obviously flipped, degenerate, or anatomically nonsensical axis set

- [x] `#35` Gap-filling for the current spatial base
  - Historical stopgap on the old `cvs_avg35` path was nearest-neighbor PM dilation to `8 mm`.
  - It is now superseded by the accepted fsaverage pivot in `#36` and should be retired, not carried forward.

- [x] `#36` Retire `cvs_avg35`, move to fsaverage base brain
  - Decision: pivot to a pure fsaverage surface pipeline.
  - The target Phase-1 spatial base is:
    - patient side projected from `pial-outer-smoothed` onto fsaverage by stock surface registration
    - atlas side baked onto fsaverage once from the cortical ribbon, not from an envelope surface
    - query-time operation on the 2D pial surface, not 3D volumetric sampling
  - Atlas-side requirement that must be explicitly confirmed during execution:
    - the bake path must use a real nonlinear sulcal/gyral surface alignment, not an affine or nearest-space shortcut
    - intended path is `ICBM152_fs/sphere.reg -> fsaverage/sphere.reg` via `mri_surf2surf`
    - this confirmation is an execution check, not a reopened design debate
  - Surface roles are intentionally different:
    - patient side uses `pial-outer-smoothed` because electrodes sit on the dural envelope and Zac found that this gives the most consistent patient-to-template projection
    - atlas side uses the true cortical ribbon (`white -> pial`) because `mri_vol2surf --projfrac-avg 0 1 0.1` is explicitly a cortical-column average
    - `ICBM152_fs` is not the source atlas itself; it is the FreeSurfer reconstruction of the ICBM152 template that puts white/pial surfaces in the same space as the BNA PM volume before the nonlinear surface-to-surface transfer to fsaverage
  - Locked sub-decisions:
    - atlas column aggregation is `mri_vol2surf --projfrac-avg 0 1 0.1`
    - physical PSF smoothing moves from 3D volumetric smoothing to 2D geodesic surface smoothing on the fsaverage mesh
  - Consequences:
    - retire the current `cvs_avg35` dilation bridge
    - close `#10` on top of the fsaverage base, not the old volumetric path
    - treat the old MNI cache as a reference / migration oracle, not as the final Phase-1 spatial representation
  - Current execution status:
    - local FreeSurfer tooling and patient surfaces are present
    - the clean atlas-side bake is currently blocked only by the missing `ICBM152_fs` subject
    - DCC FreeSurfer availability is not yet confirmed in a user-ready path and should be checked with Zac before deciding where to run the one-time `recon-all`

### Input / Loader / Labels

- [x] `#11` Channel inclusion
  - Use all non-artifact channels
  - Sig-channel masks are ablation-only metadata

- [x] `#16` Label -> index contract
  - `0`-indexed
  - Alphabetical ARPABET order
  - `AA=0, AE=1, B=2, G=3, IY=4, K=5, P=6, UW=7, V=8`

- [x] `#17` PS -> standard ARPABET mapping
  - `a -> AA`
  - `ae -> AE`
  - `i -> IY`
  - `u -> UW`
  - `b/p/v/g/k -> B/P/V/G/K`
  - Standard ARPABET is the canonical downstream label space.

- [x] `#18` `event_id` assertion
  - Hard-assert:
    - `a:1, ae:2, b:3, g:4, i:5, k:6, p:7, u:8, v:9`

- [x] `#19` Trial reconstruction
  - No stride-3 assumption
  - Trials must be reconstructed from explicit metadata

- [x] `#20` MFA posture
  - No MFA in the Phase 1 training path
  - Inputs remain response-onset-locked full trials

- [x] `#21` Canonical 52-token list
  - Source of truth is `data/ps_tokens.csv`

- [x] `#22` Cross-patient / cross-task `event_id`
  - Hard fail for PS divergence
  - Warn-only for Lexical until cross-task pooling matters

- [x] `#23` `normalize_label`
  - Accept lowercase PS labels
  - Accept canonical uppercase ARPABET
  - Accept stressed ARPABET
  - Reject everything else

- [x] `#24` `filter_to_ps_phonemes`
  - Fail-silent only for cross-task screening
  - Within-task PS loading must be strict

- [x] `#29` Epoching
  - One sample = one full trial
  - Response-onset locked
  - `tmin = -0.5 s`
  - `tmax = 1.0 s`

- [x] `#30` Phase 1 hemisphere scope
  - Phase 1 excludes RH patients
  - Core set: `S14 S26 S33 S62`
  - Extended LH set: `S16 S23 S39`

- [x] `#31` Batching / patient mixing
  - Baseline run is per-patient first
  - Grouped-by-patient batches
  - One sample = one trial
  - `patient_id` and `token_mask` emitted per sample
  - Trial-weighted loss
  - Sampler must later support mixed-patient mode

- [x] `#32` Input normalization
  - No extra normalization beyond upstream `productionZscore_highgamma`

- [x] `#33` Metric
  - PER is slot-averaged phoneme error rate
  - Report per-patient and population mean

## First-Run Protocol

These are frozen first-run defaults for Phase 1. They are not major architecture blockers, but they should be stable before the first benchmark run.

### Evaluation

- grouped-by-token CV
- `5` outer folds
- outer folds fixed once and reused across all v14 comparisons
- `3` seeds: `42, 137, 256`
- within each training fold, make a grouped validation split from the training tokens only
- validation split default: `20%` of the training-token groups
- use the same train/val token partition across seeds for a given outer fold
- report:
  - per-patient PER mean ± std across seeds
  - population mean after per-patient numbers

### Training

- optimizer: `AdamW`
- base learning rate: `1e-3`
- weight decay: `1e-4`
- scheduler: cosine decay
- warmup: `20` epochs linear warmup
- gradient clipping: `1.0`
- baseline recipe: no augmentation, no label smoothing, no focal loss, no mixup
- mixed precision: enabled by default on DCC unless numerically unstable
- early stopping: enabled on validation PER

### Batch / Epoch Defaults

- grouped-by-patient batches from `#31`
- `trials_per_batch = 8`
- default effective batch size target: `32` trials via gradient accumulation if needed
- max epochs: `300`
- patience: `10` validation checks
- validation check cadence: every `5` epochs

### Model Core

- [x] `#2` Temporal tokenizer
  - Shared per-electrode `Conv1d` patch tokenizer
  - `200 Hz` input
  - kernel `150 ms` = `30` samples
  - stride `50 ms` = `10` samples

- [x] `#6` Temporal output contract
  - Input shape: `(B, N_elec, 301)`
  - Output shape: `(B, N_elec, d, 28)`
  - Overlapping patch tokens
  - Token rate `20 Hz`

- [x] `#7` `token_support` in the model
  - `token_support[p]` is concatenated onto the parcel token once at backbone entry
  - It is not an attention bias in the baseline

- [x] `#8` SC/FC prior
  - Use combined SC + FC
  - Enter spatial attention as additive logit biases
  - One learnable scalar gain per matrix
  - Soft prior only

- [x] `#9` Supervised training contract
  - Fixed 3-slot phoneme target
  - Teacher forcing in train
  - No teacher-forced eval
  - Exhaustive eval decode over `9^3 = 729` sequences
  - Plain slot-wise CE

- [x] `#26` Within-parcel summarizer
  - Argmax-only parcel membership
  - Shared summarizer across parcels
  - Per-time-step summarization
  - Input features per electrode item:
    - temporal feature
    - parcel-frame coordinate
    - `log_sigma`
    - scalar support
    - parcel embedding
  - Support enters two ways:
    - scalar input feature
    - additive attention-logit bias
  - Standard Perceiver cross-attention defaults:
    - one latent query per parcel-time slice
    - pre-norm
    - `head_dim = 32`
    - dropout `0.1`
    - no latent self-attention stack
  - `token_mask[p]` and `token_support[p]` are intentionally different:
    - `token_mask[p] = 1` iff parcel `p` has at least one argmax electrode
    - `token_support[p] = sum_e support(e,p) / 100`

- [x] `#27` Backbone
  - Factored spatial-then-temporal attention
  - `B = 2` block pairs
  - Standard transformer block:
    - pre-norm
    - residual around attention and FFN
    - FFN width `4d`
    - `GELU`
    - dropout `0.1`
  - `head_dim = 32`
  - RoPE on temporal attention only
  - Full `(N_tok, d, T)` tensor handed to decoder
  - Masking rule:
    - spatial attention masks absent parcels
    - masked rows are zeroed after each block
    - zeros are storage-only, never the mask itself

- [x] `#28` Decoder
  - One AR decoder block
  - Fixed 3-slot output
  - Shared learned base query plus one slot embedding per position
  - Previous-token embeddings enter the query stream
  - One causal self-attention layer over the 3 slot queries
  - Then one cross-attention layer to flattened backbone memory `(N_tok * T, d)`
  - Shared linear vocab head
  - No auxiliary token head

## Immediate Implementation Checklist

These are not new design blockers. They are the small practical tasks to finish once the remaining blockers close.

- [ ] write the `#34` audit report and scripts
- [ ] build `parcel_frames.npz` with visual QC figures for all 15 Tier-1 parcels
- [ ] implement the `#12` bridge verifier and row-order checks
- [ ] write the new `v14-core` loader for `#13`
- [ ] update `v14/config.py` for `#15`
- [ ] add `#25` phoneme fixture tests
- [ ] implement tokenizer, summarizer, backbone, decoder from the frozen contracts above
- [ ] add shape and masking tests for each module
- [ ] add a tiny-subset overfit test

## Deferred

- Phase 1.5 SSL on full continuous `uECoG`
- learned per-patient calibration
- `sEEG`
- external datasets
  - fallback for uncertain external-dataset MNI quality:
    - freeze the shared neural network weights first
    - then fit a per-patient rigid-body transform on electrode coordinates only
    - allowed transform is rotation + translation only, not free affine warping
    - use this only when the external dataset's native MNI projection is plausible but not trusted enough to use raw
    - treat it as a lightweight calibration adapter, not as a change to the shared model
    - evaluate it as an explicit external-dataset adaptation step, not as part of the base Phase 1 contract
- broad ablations
