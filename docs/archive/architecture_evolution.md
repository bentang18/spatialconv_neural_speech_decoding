# Architecture Evolution: v1 to v14

This document reconstructs how the cross-patient architecture evolved from the early supervised pipeline to the current `v14` design.

There is no single literal `v1.tex` on disk. The earliest stage is reconstructed from:

- `docs/archive/historical_supervised/architecture_design.md`
- `docs/archive/theoretical_framing.md`
- `docs/archive/neural_field_perceiver_versions/neural_field_perceiver.tex`

The useful distinction is not just version number. It is **what problem we thought we were solving**, **what the shared representation was**, and **what survived into the next version**.

## Executive Summary

The architecture moved through four main phases:

1. **Early supervised / shared-backbone phase**  
   Per-patient read-in + shared temporal model. Good transfer signal, but no explicit cross-patient spatial correspondence.

2. **Neural-field / reprojection phase (`v2`-`v11`)**  
   Treat each patient as a sparse view of a shared cortical field. Use coordinate-aware virtual electrodes, cross-attention, and an observation decoder with reprojection loss.

3. **Atlas-grounded VE phase (`v12`)**  
   Keep the common-space idea, but replace mostly learned latent geometry with **Brainnetome-anchored virtual electrodes** and explicit per-patient calibration.

4. **Atlas-calibrated token-space phase (`v14`)**  
   Stop treating learned virtual electrodes as the primary interface. Use **soft volumetric Brainnetome membership** plus a **within-parcel Perceiver-style summarizer** to form the shared atlas/subparcel token space directly.

The deepest conceptual shift was:

- from **learning where the common latent points should be**
- to **treating the atlas/parcellation itself as the common interface**

## The Through-Line

Several ideas persisted through almost every version:

- the problem is fundamentally **cross-patient spatial mismatch**
- the architecture should separate **patient-specific calibration** from **shared dynamics**
- channel identity is not meaningful across patients
- the shared model should operate on a **fixed-size common representation**
- the decoder should eventually read a **3-phoneme sequence**, not only easier binary tasks

What changed was the answer to:

- what the common representation should be
- how much geometry should be learned versus imposed by anatomy
- whether the model should reconstruct observations as part of training

## Version-by-Version Arc

### `v1` Era: Early Supervised Cross-Patient Model

Representative docs:

- `docs/archive/historical_supervised/architecture_design.md`
- `docs/archive/theoretical_framing.md`

Core idea:

- keep a **per-patient spatial input layer**
- learn a **shared temporal backbone**
- decode phonemes from the shared representation

Typical structure:

- patient-specific Conv2d or linear read-in
- shared BiGRU / temporal model
- CTC or CE decoding head

Why it existed:

- it matched the strongest empirical prior from the literature:
  - per-patient input layers
  - shared temporal dynamics
- it was the minimal scalable extension of the Spalding baseline

Main limitation that triggered the next phase:

- the shared backbone still had **no explicit notion of where one patient's electrodes sat relative to another's**
- spatial alignment was implicit and therefore weakly constrained

What survived:

- the **per-patient / shared** decomposition
- the belief that the real bottleneck was not just decoder choice, but **spatial transfer**

### Proto-NFP: Theoretical Framing and First Neural-Field Draft

Representative docs:

- `docs/archive/theoretical_framing.md`
- `docs/archive/neural_field_perceiver_versions/neural_field_perceiver.tex`

Core new idea:

- formalize the data as samples from a **shared cortical field**
- import the analogy to **multi-view 3D reconstruction**
- define cross-patient transfer as estimating a shared field from multiple partial views

New ingredients:

- explicit cortical field formulation `a(p, t)`
- low-dimensional latent articulatory state `z(t)`
- early **virtual electrode** concept
- **observation decoder / reprojection loss**
- heavy use of the camera / NeRF / SfM analogy

What changed:

- the architecture was no longer just “read-in + backbone”
- it became a model of **where activity lives in space**

Main weakness:

- too many new ideas entered at once:
  - virtual electrodes
  - coordinate encoding
  - reprojection
  - observation decoder
  - patient embeddings
  - multi-scale tokens
- this made failure hard to interpret

What survived:

- the idea that electrodes are **samples of an underlying shared spatial object**
- the idea that the common representation should be **smaller and more structured than raw channels**

### `v2`-`v5`: Full Neural Field Perceiver

Representative docs:

- `docs/archive/neural_field_perceiver_versions/neural_field_perceiver_v2.tex`
- `..._v3.tex`
- `..._v4.tex`
- `..._v5.tex`

Core idea:

- instantiate the full neural-field story
- use **Perceiver cross-attention** from learned virtual electrodes to variable patient electrode sets
- supervise with **phoneme loss + reprojection-style losses**

Architecture identity:

- electrode tokens with coordinate encodings
- learned virtual electrodes in MNI space
- distance-biased spatial cross-attention
- temporal self-attention
- observation decoder for reprojection
- shared phoneme decoder

What these versions were trying to prove:

- that a **shared spatial field model** could be learned from sparse, misaligned intra-op arrays
- that reprojection would regularize the shared representation in a physically meaningful way

Why this phase mattered:

- it created the first complete end-to-end statement of the problem
- it made clear that the cross-patient issue is not just “domain shift”; it is **sensor placement mismatch**

Why this phase stalled:

- the reprojection story was elegant, but increasingly looked **overcommitted to the 3D-vision analogy**
- the supervision was thin
- identifiability was weak at this data scale
- too much of the architecture depended on the model learning latent geometry rather than starting from a stronger anatomical prior

What survived:

- Perceiver-style bottlenecks
- explicit geometry-aware calibration
- fixed-size common-space tokens

### Beta Branch: Minimum Viable Coordinate Test

Representative docs:

- `docs/archive/neural_field_perceiver_versions/neural_field_perceiver_beta.md`
- `..._beta.tex`

Purpose:

- test the smallest defensible hypothesis before committing to the full neural-field machinery

Minimum viable claim:

- coarse anatomical coordinates help LOPO decoding when added to the best local spatial model

What was deliberately removed:

- full neural-field decoder
- cross-patient reprojection
- full latent geometry claims
- large Perceiver stack as the first test

Why it mattered:

- it separated the question
  - “do coordinates help at all?”
  from
  - “is the full neural-field architecture right?”

What survived:

- the discipline of asking what the **minimum testable spatial claim** actually is

### `v6`-`v11`: Mature Reprojection-Supervised Coordinate Model

Representative docs:

- `docs/archive/neural_field_perceiver_versions/neural_field_perceiver_v6.tex`
- `..._v7.tex`
- `..._v8.tex`
- `..._v9.tex`
- `..._v10.tex`
- `..._v11.tex`

Core idea:

- keep the neural-field / reprojection story, but harden it into a more concrete architecture

Stable ingredients through this era:

- corrected MNI coordinates
- Fourier positional encoding
- 16 virtual electrodes
- distance-biased spatial cross-attention
- temporal self-attention
- observation decoder
- reprojection and sometimes cross-patient reconstruction losses
- learned query readouts for phoneme positions

What improved versus earlier NFP:

- the architecture became cleaner and more concrete
- the decoder moved closer to the eventual 3-position query readout
- the model size and ablation program became more realistic

What still felt wrong by the end:

- the **observation decoder / reprojection loss** remained central despite weak evidence that it was the right bottleneck
- the architecture still leaned on **learned virtual electrode geometry**
- Fourier PE and distance bias were doing a lot of work, but the interpretation was drifting away from a true calibration model

What survived:

- the common-space bottleneck
- the importance of bounded per-patient geometric correction
- the move toward a small, explicit phoneme-query decoder

### `v12`: Atlas-Grounded Virtual Electrodes

Representative docs:

- `docs/archive/neural_field_perceiver_versions/neural_field_perceiver_v12.tex`
- `docs/archive/v12_era/v12_overview_for_greg.md`

This was the first version that really looked like the current project.

Core shift:

- virtual electrodes stopped being mostly free latent points
- they became **Brainnetome-anchored virtual electrodes**

Architecture:

- per-patient diagonal gain / offset normalization
- learned rigid correction `Δ/ω`
- per-VE offsets around atlas centroids
- 16 Brainnetome virtual electrodes
- VE cross-attention with distance bias
- temporal self-attention
- autoregressive phoneme decoder

Why `v12` was a major step:

- it replaced a generic learned latent spatial scaffold with a **real atlas prior**
- it made the per-patient/shared boundary much sharper
- it introduced the foundation-model framing more explicitly

Core claim of `v12`:

- each patient's electrodes are mapped into a **common functional space** defined by speech-relevant atlas positions

Why `v12` still was not the endpoint:

- it still treated the common interface as **virtual electrode queries**
- the default electrode-to-region operation was still **cross-attention**
- atlas regions were acting more like anchor points for learned aggregation than as the representation itself

What survived:

- Brainnetome as the right prior
- explicit per-patient calibration
- small shared model
- sequence decoder
- the idea that raw channels should be converted into a fixed anatomical interface before shared decoding

### `v13`: Transitional Pivot

There is no formal `v13` design doc in the archive, but the conceptual transition between `v12` and `v14` is clear in the surrounding notes and current docs.

Main shift:

- stop asking “where should learned virtual electrodes look?”
- start asking “what if the atlas/parcellation is already the right common interface?”

This is where two big ideas emerged:

- **the atlas is the calibration**
- **the model should separate spatial calibration from shared dynamics explicitly**

This transitional step set up the actual `v14` rewrite.

### Early `v14`: Parcellation Replaces VE Cross-Attention

Representative docs:

- `docs/neural_field_perceiver_v14.tex`
- `docs/current_direction.md`

First major `v14` shift:

- replace distance-based VE cross-attention with **soft volumetric Brainnetome parcellation**

New core decomposition:

1. **Spatial calibration**  
   per-patient, physics-constrained, atlas-guided
2. **Shared dynamics**  
   shared model over patient-independent regional tokens

Key differences from `v12`:

- no multi-view reconstruction framing as the default interpretation
- no reprojection loss as the architectural center
- no Fourier PE as a default spatial mechanism
- no learned virtual electrodes as the common-space interface

Early `v14` default:

- soft atlas membership
- one token per parcel or low-rank summary
- initially, **mean + spatial gradient pooling**
- then shared temporal / relational processing

Why this was the deepest conceptual cleanup:

- it made the atlas itself the spatial prior, rather than just an initialization for learned query locations
- it reframed the upstream problem as **calibration**, not latent scene reconstruction

### Current `v14`: Within-Parcel Perceiver Summarizer

Representative docs:

- `docs/neural_field_perceiver_v14.tex`
- `docs/current_direction.md`

Current default:

- **soft volumetric Brainnetome calibration**
- **canonical parcel-frame local point encoding**
- **within-parcel Perceiver-style summarizer**
- **atlas/subparcel token space**
- **small relational-temporal transformer**
- **3-query AR phoneme decoder**

This is the current answer to the strongest remaining objection to early parcellation:

- a single pooled parcel token can be too coarse when important dynamics live inside one elongated parcel

What changed relative to the first `v14` draft:

- `mean + gradient` moved from default to linear ablation
- within-parcel structure became a first-class part of the default model
- the post-parcellation stage is now described explicitly as **relational / graph-like**, not Euclidean
- the default temporal front-end absorbed the one BaRISTA component worth importing directly: a richer temporal tokenizer

Current core claim:

- electrodes are **observations**
- parcel/subparcel tokens are the **shared representation**
- local geometry matters only insofar as it helps estimate those shared tokens

## What Was Discarded Along the Way

These ideas were important stepping stones, but are no longer central to the active design:

- the **camera / multi-view reconstruction** framing as the main story
- **observation decoder + reprojection loss** as the default supervision anchor
- learned **virtual electrodes** as the primary common-space interface
- **Fourier PE** as the default spatial mechanism
- distance-biased electrode-to-VE cross-attention as the main calibration step
- the hope that a single parcel mean or first spatial moment was rich enough as the default interface

## What Survived Into `v14`

These were the durable memories of the earlier versions:

- the problem is still **shared dynamics + patient-specific spatial calibration**
- the common space must be **small and anatomically meaningful**
- per-patient calibration should be **low-parameter and bounded**
- the decoder should read the **whole spatiotemporal state**, not rigid fixed windows
- local spatial detail matters, but it should be preserved **only until the common representation is estimated**

## The Cleanest One-Line Summary

The architecture evolved from:

- **per-patient read-in + shared temporal model**
- to **learned neural-field / virtual-electrode reconstruction**
- to **atlas-grounded VE common space**
- to **soft atlas-calibrated parcel/subparcel tokens with local within-parcel Perceiver summarization**

That last step is the point where the project stopped trying to learn the common spatial scaffold implicitly and started treating anatomy as the scaffold itself.
