# Archived Literature Findings

Historical literature-driven findings and broader research synthesis that informed earlier architecture planning. This file is archived context, not the active implementation contract.

If anything here conflicts with:

- `docs/neural_field_perceiver_v14.tex`
- `docs/current_direction.md`
- `docs/implementation_tasks.md`

then the active implementation docs win.

## Historical Literature Findings

- **Field consensus**: per-patient input → shared backbone (GRU) → CTC/CE. Used by Willett, Metzger, Singh, Boccato, Levin, BIT.
- **Transfer**: Singh — freeze shared backbone, fine-tune per-patient layers. Levin — 30% source replay prevents forgetting.
- **SSL**: Cross-subject benefit is the main value. Supervised cross-subject pretraining can fail without per-patient layers. Temporal masking tends to matter more than spatial masking for speech-like tasks.
- **Per-patient layers**: Still the decisive factor for cross-patient transfer in the literature.
- **Coordinate PE**: Mixed evidence. Some work finds little benefit once strong spatial attention is present; other work finds a meaningful effect.
- **Factored vs combined attention**: Both have literature support depending on topology and task.
- **Data regime**: Small epoched corpora reward strong architectural priors more than bigger models.
- **HGA ≠ spikes**: Do not pool Utah-array spikes with field-potential recordings.
- **Transformer vs GRU**: Transformer backbones have strong recent evidence, but many successful speech BCIs still rely on CTC/GRU-style pipelines.
- **MIBRAIN / Neuro-MoBRE / H2DiLR**: Strong support for region-level cross-patient representations, but mostly with hard regional assignments and heavier per-patient machinery.
- **Population Transformer**: Shows coordinate-aware self-attention can work without per-patient layers, but on easier tasks.
- **Brant family**: Scale alone does not solve cross-patient speech without explicit spatial identity.
- **BrainBERT**: Content-aware reconstruction helps avoid collapse on sparse neural signals.
- **BarISTA**: Parcel-level spatial encoding beats channel-level encoding and strongly validates parcel-scale modeling.
- **Charmander / POYO family**: Perceiver bottlenecks and moderate-capacity models are often enough; raw model scale gives limited gain.
- **NDT3**: Sensor variability remains a central blocker even at large scale.
- **NoMAD / KL alignment**: Distribution matching is a plausible later auxiliary idea, not part of the active Phase 1 contract.
- **Functional embeddings**: Interesting complementary idea, but not the active Phase 1 direction.

## Historical Data-Scaling Context

- **Flinker/Chen lab (NYU)**: estimated 50+h chronic ECoG speech corpus.
- **Chang lab (UCSF)**: estimated 20+h chronic high-density ECoG speech corpus.
- **Bouchard/Chang public Figshare**: small public chronic ECoG speech set.
- **CoganLab uECoG**: ~7.6h across 29 patients once continuous HGA is extracted.
- **CoganLab sEEG speech**: ~16.7h across 25 patients.
- **Epilepsy sEEG / public hospital datasets**: potentially much larger generic field-potential corpora.

These remain useful for later scaling plans, but they are not part of the active Phase 1 implementation contract.
