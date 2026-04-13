# v14 Implementation Start

This document narrows the implementation scope to the first executable target:

**verify the `v14-core` architecture on the existing intra-op `uECoG` data before expanding to learned calibration, `sEEG`, external datasets, or SSL.**

Implementation should not begin until the blocker list has been explicitly discussed and frozen. The point is to lock the interface contracts first and avoid coding against moving assumptions.

## Top Blocker

- **ACPC → MNI transform pipeline is currently not trusted**

Recent discussion with Zac indicates the current ACPC-to-MNI handling was wrong. This must be treated as the top blocker before serious spatial implementation work, because all atlas-driven parts of `v14` depend on it.

- **Temporal layer design is still the next architectural blocker**

Once coordinates are trustworthy, the next unresolved implementation decision is the exact shared temporal front-end. The first pass should choose the simplest plausible temporal layer that is expressive enough to validate the rest of `v14-core`.

- **Unsupported-vs-weak parcel support threshold is also still unresolved**

The first pass already assumes a fixed `N_tok` layout plus `token_mask` and `token_support`, but the threshold separating:

- truly unsupported parcels that should be masked out
- from weakly observed parcels that should remain active

is still an open implementation decision. That threshold affects the meaning of the token interface itself and should be treated as a real blocker rather than a small cleanup item.

- **Default parcel split map is not fully frozen**

The implementation already assumes a multi-token atlas interface for selected parcels, but the exact default split set still needs to be locked for Phase 1. This affects the token count, the local summarizer query layout, and the token-level graph-attention bias. The current likely default is the 21-token interface with 2-token splits for `A6cvl`, `A4hf`, `A1/2/3ulhf`, `A2`, and `A1/2/3tonIa`, with `A4tl` left as the first extension candidate.

- **Exact parcel support statistic is not yet defined**

The current plan assumes explicit parcel support, but the exact statistic has not been frozen. That matters because the support statistic underlies both `token_mask` and `token_support`, and therefore changes the effective active-token set for low-coverage patients.

- **Temporal-layer output contract is not yet locked**

Even once the temporal layer class is chosen, the exact interface it emits still needs to be frozen: token rate, receptive field, time-axis semantics, and output shape into the local summarizer. This should be treated as a blocker because it affects every downstream module boundary.

- **How `token_support` enters the model is still unresolved**

The docs already assume `token_support` exists, but not whether it is only diagnostic, appended to token features, used in attention biasing, or used in loss weighting. That changes the actual semantics of low-support parcels and should be fixed before Phase 1 implementation hardens.

- **Token-level connectivity expansion from Brainnetome SC/FC is not yet fully specified**

The design assumes atlas connectivity initializes inter-region graph attention, but the exact rule for expanding parent-parcel connectivity to token-level bias is still open. That includes SC vs FC, normalization, sibling-token handling, and the no-bias fallback.

- **Supervised training contract is not yet fully frozen**

Phase 1 is supervised-only, but the exact training contract still needs to be written down cleanly: the exact supervised input window, the target semantics, the decoder training rule, and the train-time versus eval-time decoding behavior. This should be locked before implementation spreads across trainer, decoder, and evaluation code.

- **Parcel-frame construction contract is not yet fully specified**

The local summarizer depends on canonical parcel coordinates, but the exact definition of parcel centroids, parcel axes, axis scales, and caching/precomputation is still not fully written down. This should be treated as a blocker because it defines the local geometric input itself.

- **Channel inclusion policy is still unresolved**

It is still not fully decided whether `v14-core` should use all non-artifact channels, sig-only channels, or another filtering rule. That decision changes parcel support, active-token counts, and the semantics of weak coverage, so it should be frozen before Phase 1 implementation hardens.

- **No stale Brainnetome proxy fallback should be allowed in Phase 1**

The real PM file is now available locally and should be the only active membership source for Phase 1. The old smoothed-MPM proxy is useful only as historical context. The MPM label map may still be used for ROI indexing and sanity checks, but the implementation should not silently fall back to pseudo-probabilities if the PM path is wrong.

## In Scope

- intra-op `uECoG` only
- full-trial `3`-phoneme decoding
- existing high-gamma features
- fixed Brainnetome volumetric membership from the best available corrected coordinates
- supervised training only for the first pass
- current `v14-core` interface:
  - fixed spatial mapping for the first pass
  - fixed translation / rotation from the corrected ACPC→MNI pipeline
  - no additional gain / impedance correction beyond the existing HGA preprocessing
  - shared temporal layer
  - within-parcel Perceiver-style summarizer
  - relational-temporal backbone
  - AR cross-attention decoder

## Out of Scope for the First Pass

- `sEEG`
- external chronic ECoG datasets
- SSL / JEPA pretraining during Phase 1
- audio contrastive losses
- functional pooling beyond the current atlas-first design
- broad ablation sweeps

Those matter later. They should not slow the first end-to-end implementation.

Also out of scope for the first pass:

- learned per-patient gain / offset
- learned rigid `Δ/ω`
- learned parcel offsets `δ_l`
- learned parcel temperatures `τ_l`
- additional fixed gain / impedance normalization from channel statistics
- SSL / JEPA objectives of any kind

## First Goal

The first goal is **implementation correctness**, not final performance.

That means:

- the data reach the right parcels
- the token interface is assembled correctly
- missing / weakly supported parcels are handled sensibly
- the model can overfit a tiny `uECoG` slice without shape or logic bugs

## Patient Scope

Start narrow:

- core patients: `S14`, `S26`, `S33`, `S62`

Only expand to the extended set after:

- coordinate handling is verified
- atlas coverage looks sane
- the end-to-end model runs cleanly

## Implementation Order

1. **Data and atlas interface**
   - load `uECoG` trials
   - load artifact-filtered channels
   - build corrected electrode coordinates
   - treat translation / rotation as fixed by the verified ACPC→MNI pipeline
   - compute soft Brainnetome membership
   - compute parcel support / confidence

2. **Shared temporal layer**
   - per-electrode temporal patches
   - first-pass shared temporal encoder
   - output `(N_elec x d x T_tok)`

3. **Within-parcel summarizer**
   - canonical parcel-frame coordinates
   - local point encoder
   - fixed latent queries
   - emit default atlas/subparcel token set
   - emit explicit token support alongside token values

4. **Relational-temporal backbone**
   - inter-region attention with atlas bias
   - temporal attention
   - hard masking for unsupported parcels
   - support-aware handling for weakly observed parcels

5. **Decoder**
   - `3` learned queries
   - additive AR conditioning
   - beam search over valid `CVC/VCV` outputs

6. **Minimal training loop integration**
   - grouped-by-token CV compatible
   - per-patient batching support
   - supervised loss only
   - tiny-batch overfit test

## Correctness Checks Before Real Training

These checks matter more than early leaderboard numbers.

- **Coordinate sanity**
  - left/right mirroring correct
  - array geometry consistent with known placements

- **Atlas sanity**
  - expected speech parcels receive support
  - support maps look plausible for the core patients

- **Token sanity**
  - default token count is correct
  - split parcels emit the right number of sub-tokens
  - unsupported parcels are masked rather than hallucinated
  - weakly supported parcels remain active but carry low support

- **Model sanity**
  - forward pass works on heterogeneous channel counts
  - gradients reach temporal layer, summarizer, backbone, and decoder
  - tiny training subset can overfit
  - model behaves sensibly when only one parcel / token is active
  - zero-filled inactive tokens cannot leak through as fake observations

## Practical Training Order

Do not begin with every optional degree of freedom active.

Recommended order:

1. run with fixed atlas membership only
   - fixed translation / rotation from the corrected ACPC→MNI pipeline
   - no extra gain / impedance normalization
   - supervised training only
   - fixed `N_tok` layout with hard mask for unsupported parcels
2. verify end-to-end gradients and overfit behavior
3. lock down the shared temporal layer shape and token rate
4. only then compare against simpler local-pooling ablations
5. add learned calibration only after the fixed-atlas baseline is stable

This keeps implementation bugs separate from coordinate bugs and overfitting caused by too much early flexibility.

## Code Boundary

The new implementation boundary should live under:

- `src/speech_decoding/v14/`

Suggested split:

- `config.py`
- `token_spec.py`
- `calibration.py`
- `tokenizer.py`
- `local_summarizer.py`
- `backbone.py`
- `decoder.py`
- `model.py`

Legacy baseline modules can remain where they are. They should not be repurposed into the new architecture.

## Success Criteria for Phase 1

Phase 1 is successful if all of the following are true:

- `uECoG` core-patient data flow is stable
- atlas/subparcel token construction is verified
- the model passes shape/support sanity checks
- a tiny subset can overfit
- the implementation is clean enough to support later ablations

For the first pass, the token interface should be implemented as:

- fixed `N_tok`
- hard `token_mask` for unsupported parcels
- scalar `token_support` for active but weakly observed parcels
- zero-filled inactive token values for storage convenience only

Conceptually, unsupported parcels do not exist for that patient. The zero-filled token is only an implementation placeholder; all attention and decoder reads must respect the mask.

Only after that should the project expand to:

- learned per-patient calibration
- SSL / pretraining on the full continuous `uECoG` corpus
- extended `uECoG` supervised coverage
- `sEEG`
- external datasets

## Next Step After `v14-core`

Once supervised `v14-core` is verified end-to-end, the next step should be:

- **SSL on the full continuous `uECoG` recordings**

Not:

- SSL on only the response-locked epochs

The reasoning is simple:

- response-locked windows are the right unit for the first supervised correctness pass
- they are **not** the right main corpus for SSL, because they throw away most of the available `uECoG` time
- full continuous `uECoG` gives more temporal diversity, more realistic context, and more total data for masked prediction / JEPA

So the sequencing should be:

1. supervised `v14-core` on response-locked `uECoG`
2. SSL on full continuous `uECoG`
3. supervised fine-tuning / reuse of the pretrained temporal-dynamical stack

Response-locked SSL can remain an ablation later if needed, but it should not be the default next step.
