# Central Questions

These are the main unresolved architectural questions for the project. They should stay visible even if the implementation details change.

## 1. What should the shared representation be?

This is still the most important representational tension:

- **electrode tokens with anatomical positional bias**
- versus **atlas/subparcel tokens as the shared representation**

Representative examples:

- `SwinTW`, `PopT`: keep electrodes as tokens and inject anatomy through MNI / ROI bias
- `BaRISTA`: keep channel tokens, add parcel embeddings
- `MIBRAIN`, `Neuro-MoBRE`, current `v14`: move toward region-level common-space representations

The real question is no longer whether anatomy matters.

That is already clear.

The real question is:

- **should anatomy act as bias on electrode tokens, or should it define the shared tokens themselves?**

This remains the key ablation:

1. **Electrode-space baseline**
   - electrode tokens
   - MNI / ROI positional bias
   - no atlas-token collapse

2. **Atlas-token model**
   - soft atlas calibration
   - within-parcel summarizer
   - shared atlas/subparcel dynamics

## 2. At what level should we pool?

Even if atlas/subparcel tokens are the right shared object, the pooling question is not fully settled.

Important possibilities:

- **no early pooling**: keep electrode tokens and let the backbone learn integration
- **parcel embeddings on channel tokens**: BaRISTA-style
- **hard or soft parcel pooling**
- **within-parcel summarization before collapse**: current `v14`
- **functional pooling** rather than purely anatomical pooling

This is the key tension:

- pooling too early may destroy local information
- pooling too late may leave too much patient-specific sensor identity in the shared model

The current working belief is:

- **atlas/subparcel state is probably the right shared object**
- but **important within-parcel detail must be preserved long enough to estimate that state well**

That is why current `v14` uses:

- soft volumetric Brainnetome calibration
- within-parcel Perceiver-style local summarization
- atlas/subparcel token dynamics after collapse

Still unresolved:

- Should pooling remain strictly anatomical?
- Should there be a partly **functional pooling** layer or functional grouping on top of anatomy?
- Are the current selective subparcel splits enough, or should more of the local field be preserved?

## 3. Should we actually learn the per-patient calibration parameters?

The current model includes small per-patient parameters for:

- gain / impedance normalization
- translation / rotation correction
- parcel offsets / temperatures

This remains an active question:

- **Should these be learned at all, or should we mostly trust initialization / atlas alignment to avoid overfitting?**

The real tradeoff is:

- if learned, they can absorb real anatomical and recording variation
- but with this data regime, they can also become patient codes and overfit

So the unresolved sub-question is:

- **how much of the per-patient calibration should be trainable versus frozen or only weakly adapted?**

This includes:

- should gain/offset always be learned?
- should `Δ/ω` be learned, or mostly fixed after registration?
- should parcel offsets / temperatures be learned in full, or only enabled after stronger evidence?

## 4. How much of the spatial topology and spatiotemporal dynamics is really shared across patients?

This is the deepest scientific question behind all of the modeling choices.

The project assumes that at least some of the following are shared:

- parcel/subparcel-level speech-relevant cortical states
- inter-region topology
- temporal dynamics of phoneme production

But the exact level of sharing is still unknown.

Open possibilities:

- only very coarse region-level topology is shared
- finer within-parcel geometry is also partly shared
- temporal dynamics are more shared than spatial topology
- some patients share coverage but not functional substructure

This question controls everything else:

- if very little spatial structure is shared, electrode-space models may be safer
- if parcel/subparcel structure is genuinely shared, atlas-token models should win
- if only temporal dynamics are strongly shared, SSL may help the backbone more than calibration

## Current Working Position

The best current bet is still:

- **atlas/subparcel tokens are the stronger scientific claim**
- **electrode-space models with positional bias are the right baseline**
- **within-parcel structure matters enough that simple mean pooling is too coarse**

So the project is currently making this synthesis bet:

- electrodes are observations
- atlas/subparcel state is the shared object
- local detail should survive only long enough to estimate that shared object well

That is the current `v14` position, but these questions remain open until the matched ablations are run.
