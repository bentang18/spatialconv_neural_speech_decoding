# v14 Implementation Task List

This is a working task list for the first implementation phase.

Current goal:

- **implement and verify `v14` on intra-op `uECoG` only**
- **start with `v14-core` without learned per-patient calibration**
- **use supervised training only in Phase 1**

This file is intentionally lightweight. It is a place to track tasks and blockers, not a final design document.

## Gating Rule

- [ ] **Do not begin Phase 1 implementation until each blocker has been explicitly discussed and frozen**
  - The goal is to write the interfaces once, not code against moving assumptions.
  - A blocker is not considered resolved just because a likely default exists in the docs.
  - Before implementation starts, each blocker should have:
    - an explicit decision
    - a written default
    - any important fallback / ablation path noted separately

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

## Data / Spatial Interface

- [ ] Implement corrected coordinate loader for `v14`
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
