# H2DiLR: Towards Homogeneous Lexical Tone Decoding from Heterogeneous Intracranial Recordings

## Citation
Wu, D., Li, Z., Liang, M., Gao, C., Si, J., Zhu, J., & He, B. (2025). Towards Homogeneous Lexical Tone Decoding from Heterogeneous Intracranial Recordings. *ICLR 2025*. Same group as MIBRAIN and Neuro-MoBRE.

---

## Setup

- **Modality:** Stereoelectroencephalography (sEEG), depth electrodes for epilepsy monitoring
- **Patients:** 4 epilepsy patients; variable electrode counts and placement across brain regions
- **Task:** 4-class Mandarin lexical tone classification (Tone 1: flat, Tone 2: rising, Tone 3: dipping, Tone 4: falling). 1,221 trials per subject
- **Signal:** Raw broadband sEEG (preprocessing details consistent with MIBRAIN/Neuro-MoBRE pipeline)
- **Data split:** Standard train/test splits for evaluation; VQ encoders trained on all subjects' data without labels (self-supervised), then downstream classifier trained with labels

---

## Architecture

**Two-stage architecture with VQ disentanglement:**

### Stage 1: H2D (Heterogeneous-to-Homogeneous Disentanglement)

```
Per-subject sEEG (variable channels x time)
  |
-> Per-subject VQ encoder (5-layer ConvNet, ~1.55M params/subject)
     Each subject has a completely separate encoder
     Encodes neural activity into discrete codebook tokens
  |
-> Codebook assignment with shared/private split:
     Shared codebook: K_S=128 codes, D=256 dimensions (shared across all subjects)
     Per-subject private codebooks: K_P=32 codes, D=256 each
     |
     For each encoded token z:
       Compute nearest-neighbor in shared codebook -> z_shared
       Compute nearest-neighbor in private codebook -> z_private
       Routing: top nu=0.5 fraction (by shared codebook distance) -> assigned to shared
                remaining (1-nu)=0.5 -> assigned to private
     |
     Shared codes capture cross-subject patterns (tone representations)
     Private codes capture subject-specific patterns (electrode config, noise)
  |
-> Reconstruction decoder (per-subject, mirrors encoder)
     Reconstruct original sEEG from shared + private codes
     Loss: MSE reconstruction + VQ commitment loss
```

### Stage 2: ND (Neural Decoding from homogeneous tokens)

```
Frozen per-subject VQ encoders + frozen codebooks
  |
-> Only shared codebook tokens used (private discarded)
  |
-> Transformer decoder (4 blocks, d=128, 8 heads, ~0.958M params)
     Processes shared tokens from all subjects in a common space
  |
-> Linear classification head -> 4-class tone prediction
```

- **Total params:** ~7.2M (4 x 1.55M encoders + 0.958M decoder + codebooks)
- **Per-patient params:** ~1.55M (entire VQ encoder + private codebook per subject). This is massive -- 11,500x v12's 134/patient
- **Shared params:** ~0.958M (transformer decoder) + shared codebook (128 x 256 = 32K)

---

## SSL / Pretraining

**VQ autoencoding reconstruction (Stage 1):**
- Self-supervised: no task labels used during Stage 1
- Each subject's encoder trained to reconstruct its own sEEG through the VQ bottleneck
- Loss: MSE(reconstruction, original) + beta * commitment_loss(z, codebook_entries)
- Commitment loss prevents codebook entries from drifting too far from encoder outputs
- Training: 1,000 epochs per subject
- The shared/private split is the key innovation: it forces the shared codebook to capture only cross-subject-invariant structure, while private codebooks absorb subject-specific noise

**No joint multi-subject pretraining in Stage 1.** Each subject is trained independently with access to both shared and private codebooks. Cross-subject alignment emerges from the shared codebook being updated by all subjects' gradients simultaneously

---

## Cross-Patient Handling

**Mechanism:** Per-subject VQ encoders absorb ALL heterogeneity; shared codebook provides alignment.

- Each subject has a completely separate encoder (~1.55M params) that maps their unique electrode configuration into discrete codes
- The shared codebook (K_S=128, D=256) acts as a Rosetta Stone: all subjects' encoders must route their most informative tokens to the same 128 shared codes
- At decoding time (Stage 2), ONLY shared codes are used -- private codes discarded. This strips subject-specific information entirely
- The nu=0.5 routing threshold determines the shared/private split. nu=0.5 is optimal for tone decoding; nu=0.25 maximizes subject identity in private codes (useful for subject classification ablation)

**For a new subject:** Would require training an entirely new VQ encoder (~1.55M params) while keeping the shared codebook and decoder frozen. The paper does not explicitly evaluate new-subject transfer

**UMAP visualization:** Shared codes cluster by tone class across subjects. Private codes cluster by subject identity. This confirms the disentanglement is working as intended

---

## I/O Features

| Feature | Detail |
|---------|--------|
| Input representation | Raw broadband sEEG (per-subject variable channels) |
| Input spatial info | NONE -- no coordinates, no region labels, no atlas |
| Temporal resolution | ConvNet temporal processing (5 layers) |
| Output | 4-class tone classification |
| Sequence modeling | Transformer decoder (4 blocks), but single-class output |
| Vocabulary | 4 Mandarin tones |

---

## Key Results

### 4-class tone classification accuracy

| Method | Accuracy | Delta vs best baseline |
|--------|----------|----------------------|
| H2DiLR | **43.67%** | +6.7pp |
| Best pretrained baseline | 36.97% | -- |
| Single-subject (no cross-patient) | ~35% | -- |
| Chance | 25.0% | -- |

### Scaling with subject count

| Subjects (m) | Accuracy trend |
|---------------|---------------|
| m=1 | Baseline single-subject |
| m=2 | Improvement |
| m=3 | Further improvement |
| m=4 | **43.67%** (consistent monotonic gain) |

Unlike MIBRAIN where m<6 can hurt, H2DiLR shows consistent improvement from m=2 onward. The per-subject VQ encoders prevent negative transfer by isolating heterogeneity

### Ablation highlights

| Component | Effect |
|-----------|--------|
| Shared codebook only (no private) | Performance drops -- private codes needed to absorb noise |
| Private codebook only (no shared) | Near chance -- no cross-subject transfer |
| nu=0.5 (shared/private split) | Optimal for tone decoding |
| nu=0.25 | Best for subject classification (more info in private) |
| K_S=128, K_P=32 | Optimal codebook sizes |
| VQ commitment loss | Essential for codebook utilization |

### UMAP analysis
- Shared codes: clear tone-class clustering across all 4 subjects
- Private codes: clear subject-identity clustering, no tone structure
- Confirms disentanglement: shared captures task, private captures subject

---

## v12 Comparison

### What H2DiLR validates for v12
1. **Shared neural representations for speech features exist across patients.** The shared codebook successfully captures tone representations that generalize across 4 subjects with different electrode placements. This is the fundamental assumption underlying v12's VE common space
2. **Explicit disentanglement of shared vs subject-specific information helps.** The shared/private codebook split outperforms using all codes. v12's architecture implicitly achieves this: diagonal normalization captures subject-specific gain/offset, backbone captures shared representations
3. **Consistent scaling with subject count.** m=2->4 monotonically improves, with no initial dip. Per-subject encoders prevent negative transfer. v12's per-patient diagonal normalization serves the same stabilizing role

### Why H2DiLR's approach is wrong for v12's regime
1. **Massively over-parameterized per patient.** 1.55M params per subject with 1,221 trials each = ~1,270 params/trial. v12 has 134 per-patient params with 46-178 trials = 0.75-2.9 params/trial. H2DiLR's per-patient budget is 450-2000x larger relative to data
2. **No spatial information at all.** No coordinates, no region labels, no atlas. All spatial alignment is implicit through the VQ bottleneck. This only works because the per-subject encoders are large enough to learn arbitrary mappings. v12 exploits MNI coordinates explicitly
3. **4-class tone is much simpler than 9-phoneme sequences.** Mandarin tones are suprasegmental prosodic features (F0 contour), not articulatory phoneme identities. The spatial resolution requirements are different
4. **VQ discretization may lose information.** Quantizing continuous neural representations to 128 shared codes creates an information bottleneck that may be too aggressive for fine-grained phoneme distinctions. v12's continuous VE representations preserve more nuance

### Conceptual insight worth borrowing
The shared/private codebook split is an elegant framework for thinking about what cross-patient models need to learn. v12's architecture can be interpreted through this lens:
- **Shared codes** = VE representations (common functional space)
- **Private codes** = per-patient diagonal + delta/omega (measurement-specific calibration)
- **Routing** = distance-biased cross-attention (determines which electrode information enters the shared space)

The difference is that v12 achieves this with 134 per-patient params and continuous coordinates, while H2DiLR needs 1.55M per-patient params and no coordinates

---

## Regime Comparison Table

| Dimension | H2DiLR | v12 regime | Implication |
|-----------|--------|-----------|-------------|
| Data volume | 1,221 trials/subject (4 subjects) | 46-178 trials/patient (11 patients) | H2DiLR has 7-26x more trials/patient; can afford massive per-patient encoders |
| Per-patient params | 1.55M (~1,270/trial) | 134 (~0.75-2.9/trial) | H2DiLR is 450-2000x more per-patient params relative to data |
| Task complexity | 4-class tone (prosodic) | 9-class phoneme sequence (articulatory) | Tone = suprasegmental F0 contour; phonemes = somatotopic articulator control |
| Spatial information | None | MNI coords + Fourier PE + distance bias + 16 VEs | v12 compensates for fewer per-patient params with explicit spatial prior |
| Cross-patient mechanism | VQ shared codebook (128 discrete codes) | VE common space (16 continuous representations) | Both create bottleneck forcing shared structure; v12 is continuous |
| Disentanglement | Explicit shared/private codebook split | Implicit (diagonal = private, backbone = shared) | H2DiLR's explicit split is cleaner conceptually; v12's is lighter |
| Subject scaling | Monotonic from m=2 | Unknown (untested) | Per-subject encoders prevent negative transfer; v12's light per-patient layers should also prevent it |
| New subject transfer | Requires training new 1.55M encoder | LP-FT: fit 134 params | v12's new-patient cost is 11,500x lower |

### Key takeaway for v12
H2DiLR provides the cleanest conceptual demonstration that shared neural representations for speech features can be disentangled from subject-specific measurement noise via a discrete bottleneck. The shared/private codebook framework is a useful mental model for v12's architecture. But H2DiLR's approach (1.55M per-patient params, no coordinates, VQ discretization) is designed for a regime with 7-26x more data per patient than ours. v12's design -- continuous coordinates, atlas-grounded VEs, 134 per-patient params -- is the appropriate adaptation of the same principle to our data-starved regime. The frequency-domain pretraining target (shared with Neuro-MoBRE from the same group) is worth testing.
