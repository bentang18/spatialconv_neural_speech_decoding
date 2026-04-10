# Paper Framing — Intracranial Field Potential Foundation Model

Updated: 2026-04-06 (pivot from architecture paper to foundation model paper)

## Core Narrative

Intracranial field potential recordings (uECoG, sEEG, macro-ECoG) measure the same biophysical signal — postsynaptic potentials / high-gamma activity (70-150Hz) — from electrodes at known MNI positions. Despite this, every cross-patient model in the field is trained from scratch on small labeled datasets, because electrode placement varies across patients and no common pretraining corpus exists.

We build the first foundation model for intracranial field potentials. The insight: a simple, modality-agnostic architecture (self-attention + MNI coordinates) treats all recordings identically — just electrodes with positions and time series. This enables SSL pretraining across modalities (sEEG + uECoG) and datasets, then transfer to downstream tasks. We evaluate on cross-patient speech decoding from intra-operative μECoG.

## The Decomposition (from v1, still valid)

Cross-patient neural activity decomposes as:

$$a^{(i)}(p, t) = f(T_i(p), t; y) + r^{(i)}(p, t)$$

- $f$ — shared neural dynamics (universal across patients)
- $T_i$ — patient-specific measurement transform (gain, registration error)
- $r^{(i)}$ — patient residual (pathology, noise)

The architecture maps onto this:

| Theory | Architecture | What it handles |
|---|---|---|
| Shared field $f(p, t; y)$ | Shared transformer backbone (spatial + temporal self-attn) | Universal temporal dynamics |
| Spatial identity in $f$ | Fourier PE on MNI coordinates | Brain position → function mapping |
| Measurement transform $T_i$ (gain) | Per-patient diagonal $s^{(i)} \odot x + b^{(i)}$ | Impedance, tissue, SNR |
| Registration correction in $T_i$ | Learned Δ/ω (6 params) on MNI coordinates | Systematic registration error |
| Residual $r^{(i)}$ | Whatever the model can't capture | Ideally small |

SSL pretraining teaches $f$ — the shared temporal dynamics — from unlabeled continuous data across patients and modalities. Per-patient layers absorb $T_i$ with minimal parameters (134/patient).

## What's Novel (paper contributions)

1. **First intracranial field potential foundation model.** BIT (Zhang 2026) built a foundation model for Utah array spikes (367h, 2 patients). We do the same for field potentials (HGA) from surface and depth electrodes. These are fundamentally different signals — HGA reflects postsynaptic activation (inputs to the measured cortex), spikes reflect action potentials (outputs). Different biophysics, different neural information, different clinical modalities. Nobody has built a foundation model for the field potential side.

2. **Cross-modality SSL pretraining.** The architecture is modality-agnostic: sEEG depth electrodes and uECoG surface grids are both "electrodes with MNI coordinates." Same model, same weights, same SSL objective. Prior work either treats each modality separately or requires modality-specific components (Conv2d for grids, sparse attention for sEEG). Our self-attention + coordinate approach handles both without modification.

3. **Scaling analysis for intracranial SSL.** How much pretraining data does a field potential foundation model need? BIT showed 20-30h is the productive range for spikes. We characterize scaling for HGA across modalities: does sEEG pretraining help uECoG downstream? Does more data always help, or is there a plateau?

4. **Cross-patient speech decoding as downstream evaluation.** The foundation model is evaluated on phoneme sequence decoding from intra-operative μECoG — a clinically relevant task (speech neuroprosthetics) with clean evaluation metrics (PER) and an established baseline (0.734 per-patient, 0.762 LOPO).

## What's Established (cite, don't claim)

- Self-attention + MNI PE for variable electrode layouts (seegnificant, Mentzelopoulos 2024)
- Factored spatial → temporal attention (seegnificant: +0.06 R², 5.5× faster)
- Per-patient input layers as decisive for cross-patient transfer (BIT, Willett, Singh, seegnificant)
- Temporal masking for SSL (BIT: temporal > spatial for speech)
- Teacher-forced AR decoder for sequence generation (standard)
- LP-FT adaptation protocol (Kumar et al. 2022)

## Why HGA ≠ Spikes (can't pool with BIT/Nason/Willett)

| Property | HGA (field potentials) | Spikes (Utah arrays) |
|---|---|---|
| Biophysics | Postsynaptic potentials | Action potentials |
| What it measures | Inputs arriving at cortex | Outputs leaving cortex |
| Spatial scale | ~1-2mm (ECoG), ~3-5mm (sEEG) | ~100μm (single unit) |
| Frequency | 70-150Hz | 250-5000Hz |
| Clinical modalities | uECoG, macro-ECoG, sEEG | Intracortical arrays (Utah, Neuropixels) |
| Existing foundation models | **None** | BIT (367h), NDT3 (2000h+) |

This gap is the paper's raison d'être.

## The Multi-View Insight (analysis, not architecture)

Each patient's electrode array is a noisy spatial sample of a shared cortical field. SSL pretraining recovers the shared field implicitly through temporal prediction — if the backbone learns good temporal representations from multiple patients' data, it's because those dynamics are shared.

**Post-hoc validation:** After training, extract backbone representations for spatially overlapping patients (S26 ↔ S33, 1.3mm nearest-neighbor distance). Can Patient A's latent predict Patient B's electrode activity? If yes, the shared field decomposition holds empirically.

## The Latent Manifold (analysis, not architecture)

Speech motor cortex activity during phoneme production may live on a ~5-8 dimensional manifold (articulatory degrees of freedom). The model doesn't enforce this — the representation is N×T'×d. 

**Post-hoc:** PCA/UMAP on encoder representations across patients and time bins. If effective dimensionality is ~5-8 and clusters align with articulatory features, the manifold hypothesis is validated. Either outcome is a finding.

## Paper Structure (sketch)

1. **Introduction**: The gap — no foundation model for intracranial field potentials. BIT did spikes; we do HGA. Modality-agnostic architecture enables cross-modality SSL.

2. **Architecture**: Brief (~1 page). Self-attention + Fourier PE (cite seegnificant). Per-patient diagonal normalization. Δ/ω coordinate correction. AR decoder for speech. The contribution is NOT the architecture.

3. **Data and Pretraining**: The core contribution.
   - Pretraining corpus: sEEG + uECoG, harmonization, HGA extraction
   - Temporal masking SSL recipe (BIT-adapted)
   - Scaling curves: performance vs pretraining hours, modality mix

4. **Experiments**:
   - Pretrained vs from-scratch on speech decoding (4 core patients, then population)
   - Cross-modality: does sEEG pretraining help uECoG downstream?
   - Scaling: 1h → 5h → 10h → 24h pretraining
   - Ablation: engineered pipeline (Conv2d + VEs + distance bias) vs default self-attention
   - Per-patient capacity: diagonal vs full affine, with/without output heads

5. **Analysis**:
   - What the model learned: attention maps, spatial structure, temporal dynamics
   - Cross-patient representation overlap (multi-view validation)
   - Manifold dimensionality
   - Learned Δ/ω magnitudes vs known registration error

6. **Discussion**: First HGA foundation model, scaling laws, limitations (N patients, intra-op, coordinate noise), future (epilepsy monitoring data, multi-site consortia).
