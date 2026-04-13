# Javadzadeh et al. 2025 - FunctionalMap: Learned Functional Embeddings for SEEG

## Citation
Javadzadeh, A., et al. (2025). FunctionalMap: Learning Functional Electrode Representations from Stereotactic EEG Data. *Submitted to ICLR 2026*.

## Setup
- **Recording modality**: Stereotactic EEG (sEEG), 2048 Hz sampling
- **Subjects**: 20 epilepsy patients, basal ganglia and thalamus recordings
- **Data scale**: Not precisely stated; standard clinical sEEG recordings (~minutes to hours per patient)
- **Tasks**: Downstream evaluation via masked-region reconstruction (reconstruct held-out region's LFP from other regions)
- **Brain regions**: Basal ganglia and thalamus -- subcortical, NOT cortical speech areas

## Architecture
- **Electrode embedding model**: Siamese CNN encoder. Two channels from same/different regions processed by shared CNN backbone
- **CNN encoder**: 4 convolutional layers → global average pool → linear projection → L2 normalization → 32-dim embedding. Scale: 117K-694K params depending on temporal input length
- **Contrastive training**: Siamese architecture with two objectives:
  1. **MSC (Multi-Session Contrastive)**: Channels from same brain region across different patients are positive pairs. Teaches anatomically-grounded functional similarity
  2. **PSC (Patient-Session Contrastive)**: Channels from same region within a single patient are positive pairs. Learns within-patient functional similarity
  - MSC consistently outperforms PSC -- cross-patient functional consistency > within-patient co-activation
- **Downstream transformer**: Standard encoder for masked-region reconstruction. 4-layer, d=128, 4 heads. ~1.37M params. Takes functional embeddings as positional encoding for each electrode
- **Region labels**: Assigned by clinical expert (neurosurgeon), NOT automated atlas parcellation. This is a limitation for scalability

## SSL/Pretraining
- **Contrastive SSL for embeddings**: The CNN encoder is trained purely on the contrastive objective (MSC or PSC) -- no behavioral labels. Learns electrode identity from functional similarity of LFP signals
- **Downstream task**: Masked-region reconstruction -- mask all electrodes in one brain region, reconstruct from remaining regions. Evaluates how well the functional embeddings capture spatial relationships
- **Not temporal masking**: The masking is spatial (entire region held out), not temporal

## Cross-Patient Handling
- **Zero per-patient parameters**: The functional embeddings are computed by the shared CNN encoder from each electrode's raw LFP. No per-patient layers, no per-electrode learned embeddings
- **Functional > MNI coordinates**: Functional embeddings outperform MNI coordinate-based positional encoding for masked-region reconstruction (p<0.001). The learned representations capture functional relationships that Euclidean MNI distance misses (e.g., thalamocortical connectivity not captured by proximity)
- **Generalization**: Functional embeddings transfer to held-out patients because functionally similar electrodes (same region, similar LFP patterns) get similar embeddings regardless of patient identity
- **Region labels from clinical expert**: Not atlas-based -- requires manual annotation. Limits scalability to new datasets without expert labeling

## I/O Features
- **Input**: Raw sEEG LFP at 2048 Hz, windowed into segments for CNN embedding
- **Output**: 32-dim L2-normalized functional embedding per electrode (embedding model). Reconstructed LFP for masked regions (downstream transformer)
- **Spatial**: Functional embeddings replace coordinates as positional encoding
- **Temporal**: Standard temporal patching for the downstream transformer

## Key Results
| Embedding Type | Masked-Region Reconstruction |
|---|---|
| FunctionalMap (MSC) | **Best** (p<0.001 vs MNI) |
| FunctionalMap (PSC) | Second |
| MNI coordinates | Worse than functional |
| Random embeddings | Worst |

Key findings:
- **Functional embeddings > MNI coordinates**: Learned functional similarity captures spatial relationships that Euclidean distance in MNI space misses. Particularly important for subcortical structures where proximity does not imply functional similarity
- **MSC > PSC**: Cross-patient contrastive training produces better embeddings than within-patient. Functional similarity across patients is a stronger training signal
- **32 dimensions sufficient**: Low-dimensional embeddings capture enough spatial/functional information for downstream reconstruction
- **Zero per-patient params work** for this reconstruction task (but task is relatively easy)

## v12 Comparison

**Functional embeddings are an alternative to coordinate-based spatial encoding.** FunctionalMap's core claim -- that learned functional similarity outperforms MNI coordinates for electrode identity -- challenges v12's Fourier PE on MNI coordinates. However, the context matters significantly.

**Why this finding may NOT transfer to v12:**
1. **Subcortical vs cortical**: Basal ganglia/thalamus have complex connectivity where Euclidean distance is a poor proxy for functional similarity (thalamic nuclei relay to distant cortex). Cortical speech areas (v12's domain) have more local functional organization -- MNI distance IS a reasonable proxy for functional similarity on the cortical surface
2. **sEEG vs uECOG**: sEEG electrodes span 3D brain volume along depth probes. uECOG is on the cortical surface. For surface arrays, 2D geodesic distance (approximated by MNI Euclidean on the surface) captures functional organization well
3. **Distance bias already addresses this**: v12's distance-biased cross-attention creates ~25mm soft receptive fields, which is a soft version of functional locality. The atlas VE positions (Brainnetome) encode population-average functional anatomy, not just coordinates
4. **Task difficulty**: Masked-region reconstruction is easier than 9-class phoneme decoding. Findings may not transfer to harder discriminative tasks

**Key differences:**
- **FunctionalMap learns embeddings from LFP patterns**: The CNN encoder maps each electrode's activity into a 32-dim space. This requires sufficient data per electrode to learn stable embeddings. v12's coordinates are available immediately with no data requirement
- **Expert region labels**: FunctionalMap requires clinical expert annotation of brain regions for contrastive pairs. v12 uses atlas parcellation (Brainnetome), which is automated and scalable
- **Zero per-patient params**: Works for reconstruction but likely insufficient for phoneme decoding (per BIT, NDT3, seegnificant findings)
- **Siamese architecture**: The contrastive training is specific to learning electrode embeddings. v12's architecture (VE cross-attention + distance bias) jointly learns spatial relationships during the main task

**What to import:**
- **Contrastive functional similarity as regularizer**: During v12 SSL, add an auxiliary loss encouraging electrodes in the same Brainnetome parcel (across patients) to produce similar VE cross-attention patterns. This is a soft version of MSC that doesn't require expert labels
- **Functional embedding diagnostic**: After v12 training, extract the effective electrode representations (from the cross-attention layer) and check if functionally similar electrodes cluster. This validates that VE cross-attention is learning meaningful spatial structure
- **MSC > PSC insight**: Cross-patient consistency is a stronger training signal than within-patient co-activation. Supports v12's cross-patient VE consistency regularizer for overlapping pairs (S26-S33, S22-S62)

**Common mistakes:**
- Do NOT abandon MNI coordinates for functional embeddings. The finding is specific to subcortical sEEG where Euclidean distance fails. For cortical uECOG, coordinates are a strong prior
- Do NOT require clinical expert region labels. v12 uses automated atlas parcellation (Brainnetome), which scales without manual annotation
- Do NOT assume zero per-patient params suffice. FunctionalMap's reconstruction task is far easier than phoneme decoding; the per-patient findings from BIT/NDT3/seegnificant still apply
- Do NOT conflate "functional > MNI" with "learned > fixed." The comparison is confounded -- functional embeddings have 32 learned dims vs 3 fixed MNI coords. v12's Fourier PE expands 3 MNI coords to 6F dimensions, partially closing this gap
