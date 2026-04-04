# Cross-Patient Speech Decoding: Theoretical Framing

*Version 2. Problem framing only — architecture follows as a separate document.*

This document formalizes the cross-patient speech decoding problem, organizes evidence into explicit epistemic categories, and identifies the structural bottlenecks that any solution must address. Claims are categorized as **externally established** (published literature), **internally observed** (our experiments), or **working hypotheses** (plausible but unverified).

---

## 1. The Physical Problem

Each patient's micro-electrocorticography (uECOG) array is a **sparse spatial sample** of neural activity over the cortical surface during speech production. The array is placed subdurally over left ventral sensorimotor cortex (vSMC) during surgery, and its position in standardized brain space (MNI-152) is determined post-hoc via electrode reconstruction (BioImage Suite for DBS patients, Brainlab for tumor patients).

### 1.1. The cortical surface, not free 3D space

The cortex is a folded two-dimensional sheet, not a volume. Let $\mathcal{S}$ denote the cortical surface embedded in $\mathbb{R}^3$. The high-gamma analytic amplitude (HGA, 70-150 Hz envelope) is a function on this surface:

$$a: \mathcal{S} \times \mathbb{R} \to \mathbb{R}$$

For patient $i$ with $N_i$ electrodes at cortical surface positions $\{\mathbf{p}_j^{(i)}\}_{j=1}^{N_i} \subset \mathcal{S}$, we observe:

$$d_j^{(i)}(t) = a(\mathbf{p}_j^{(i)},\ t) + \epsilon_j^{(i)}(t)$$

where $\epsilon$ captures measurement noise, electrode impedance variation, and sub-electrode-scale neural activity not resolved by the ~2 mm contact spacing.

In practice, we approximate cortical surface positions with 3D MNI coordinates: $\mathbf{p}_j^{(i)} \approx (x_j^{(i)}, y_j^{(i)}, z_j^{(i)})_{\text{MNI}}$. This approximation has two consequences:

1. **Euclidean distance in MNI space does not equal geodesic distance on the cortical surface.** Two electrodes 3 mm apart in MNI may sit on opposite banks of a sulcus, making them functionally distant. For uECOG arrays on exposed cortex, the local patch under the array is approximately planar, so within-array Euclidean distances are reasonable. Cross-patient comparisons at larger scales are more affected by folding geometry.

2. **Surface-based coordinates (e.g., projection to fsaverage) would better respect cortical geometry** than volumetric MNI coordinates. Whether surface reconstructions exist for our patients is an open question — this depends on the BioImage Suite / Brainlab outputs available.

**The decoding problem**: Given observations $\{d_j^{(i)}(t)\}$ at approximately known surface positions, infer the phoneme sequence $\mathbf{y} = (y_1, y_2, y_3)$ that produced the spatiotemporal pattern.

**The cross-patient problem**: Train on patients $\{1, \ldots, K\}$ and decode for a held-out patient $K+1$ whose electrodes sample the cortical surface at **different positions**. The observations are not aligned: electrode $j$ on patient $i$ and electrode $j$ on patient $i'$ correspond to different cortical locations.

### 1.2. What makes this hard

| Factor | Quantity | Implication |
|--------|----------|-------------|
| Electrodes per patient | 63-201 | Sparse sampling of the cortical field |
| Electrode spacing | ~2 mm | Resolves individual gyral columns |
| Array offset across patients | 15-25 mm in MNI | Different cortical regions sampled |
| Array rotation | Variable | Channel indices meaningless across patients |
| Inter-subject correspondence uncertainty | Several mm | Coordinate-based alignment is approximate (see Section 3.2) |
| Cortical folding variability | Patient-specific | Sulcal geometry differs; MNI normalization is imperfect |
| Tumor mass effect (subset) | Variable | Macroscopic warping of cortical anatomy in S32, S33, S39, S57, S58, S62 |
| Trials per patient | 46-178 | Small-data regime for supervised learning |
| Total utterance per patient | ~1 min | Insufficient for single-patient SSL |
| Phoneme vocabulary | 9 classes x 3 positions | 27-way classification at each trial |


## 2. The Shared Dynamics Hypothesis

Cross-patient decoding is only possible if the cortical field $a(\mathbf{p}, t)$ shares structure across patients. We decompose:

$$a^{(i)}(\mathbf{p}, t) = f\bigl(T_i(\mathbf{p}),\ t;\ \mathbf{y}\bigr) + r^{(i)}(\mathbf{p}, t)$$

where:

- $f(\cdot, t; \mathbf{y})$ is the **shared neural field** — the spatiotemporal pattern that phoneme sequence $\mathbf{y}$ produces in canonical cortical space.

- $T_i: \mathcal{S}_i \to \mathcal{S}_{\text{canonical}}$ is a **patient-specific spatial transform** mapping each patient's cortical surface to a canonical reference. MNI normalization is our estimate of $T_i$, but it is imperfect.

- $r^{(i)}(\mathbf{p}, t)$ is a **patient-specific residual** capturing individual neural organization, signal characteristics, and functional plasticity.

The viability of cross-patient decoding depends on the ratio $\|f\| / \|r^{(i)}\|$ at the spatial resolution accessible to our measurements and alignment procedure.

### 2.1. Evidence for shared dynamics

We organize evidence into three explicit categories.

#### Externally established (published literature)

**Per-patient input layers + shared temporal backbone is the optimal transfer architecture.** Singh et al. (2025) demonstrate group training with N=25 sEEG patients: per-patient Conv1D + shared BiLSTM (2x64), frozen shared backbone + fine-tuned per-patient layers yields PER 0.49 vs 0.57 single-subject (p<0.001). Willett et al. (2023) use per-day affine input layer (256->256) + shared 5-layer GRU (512 hidden) for day-to-day BCI transfer. Both confirm the decomposition: shared temporal dynamics + patient/session-specific spatial mapping.

*Caveats: Singh's data regime is ~60 min/patient (vs our ~1 min); Conv1D hyperparameters are unreported. Willett is N=1 with ~40 min/day. The architectural pattern transfers; the hyperparameters do not.*

**Articulatory somatotopy in vSMC is reproducible within individuals.** Metzger et al. (2023) show somatotopic organization of speech articulators in one patient with chronic ECoG (253 channels): labial, front tongue, back tongue, and vocalic representations are spatially organized (p<0.0001), preserved even after 18 years of anarthria, suggesting the organization is intrinsic to motor cortex circuitry.

*Caveat: N=1 chronic implant. Cross-patient spatial consistency of the somatotopic map is inferred from anatomical uniformity but not directly measured in any paper in our corpus.*

**Articulatory features are a robust encoding basis in vSMC.** Wu et al. (2025) reconstruct speaker-invariant articulatory features from HD-ECoG over vSMC (PCC 0.75-0.80, N=3), including one intra-operative patient (PCC=0.76). The claim that articulatory features are "more robustly encoded than acoustic features" and "can be learned faster with limited neural data" is attributed by Wu to Chartier (2018), Conant (2018), and Anumanchipalli (2019).

*Caveat: Wu's reconstruction is word-level scalar loadings, not continuous trajectories. The robustness claim is borrowed from cited literature, not newly demonstrated in Wu.*

**MNI coordinates enable cross-patient ECoG transfer.** Chen et al. (2025, SwinTW) use MNI coordinates and ROI embeddings as positional information for cross-patient ECoG speech decoding across 43 ECoG + 9 sEEG subjects, reporting mean PCC 0.765 on unseen participants in cross-validation. This demonstrates that anatomical coordinates carry sufficient spatial information for subject-agnostic decoding in electrode-based neural recordings.

*Caveat: Standard ECoG has ~10 mm spacing, comparable to inter-subject correspondence uncertainty. Our uECOG spacing (~2 mm) is finer than this uncertainty, meaning coordinates may encode global array position but not resolve within-array relationships.*

**Motor cortex neural manifold structure is preserved across individuals.** Safaie et al. (2023, *Nature Neuroscience*) show that motor cortex population dynamics during reaching are geometrically similar across different monkeys — the neural manifold structure is preserved across individuals performing similar behavior.

*Caveat: This concerns limb motor cortex in non-human primates, not speech motor cortex in humans. The extension to speech is plausible but not directly tested.*

**Neural manifold dynamics are stable despite neural turnover.** Gallego et al. (2020, *Nature Neuroscience*) establish that low-dimensional manifold dynamics are preserved across long time periods within the same animals, even as the individual neurons contributing to the manifold change. This is evidence for **temporal stability of latent dynamics**, distinct from the cross-individual claim above.

*Caveat: This is within-individual stability, not cross-individual similarity. It supports the claim that the manifold is a robust property of the circuit, but does not directly show cross-patient transfer.*

**Consistent articulatory somatotopy across participants.** Bouchard et al. (2013, *Nature*) used high-density ECoG to demonstrate reproducible articulatory maps in vSMC across 3 participants. Anumanchipalli et al. (2019, *Nature*) reconstructed continuous articulatory trajectories from motor cortex.

*Caveat: Neither paper is summarized in our corpus. Bouchard's N=3 provides limited evidence for population-level consistency.*

#### Internally observed (our experiments)

**Cross-patient supervised transfer works at this data scale.** Our LOPO pipeline (per-patient SpatialConv + shared BiGRU backbone, 9 source patients) achieves PER 0.762 on held-out S14, compared to per-patient PER 0.737 (grouped-by-token CV). The shared backbone, trained on other patients' data, produces representations nearly as good as patient-specific training. This is direct evidence that decodable shared structure exists in $f$.

**Articulatory bottleneck head improves LOPO decoding.** The articulatory decomposition head (projecting to a 15-dimensional phonological feature space via a fixed linguistic matrix) outperforms a flat CE head for cross-patient decoding. This is consistent with the hypothesis that the shared field $f$ is organized along articulatory feature dimensions.

**55 experiments on S14 converge to PER 0.750-0.780.** Across supervised training modifications, SSL objectives, domain adaptation methods (DANN, CORAL, CCA), self-training, and knowledge distillation, all approaches converge to a narrow PER band. (See Section 4.4 for interpretation and caveats.)

**SSL features are not random but are far weaker than supervised.** JEPA features + LP-FT achieve PER 0.797 (vs 0.854 with basic evaluation), demonstrating partial structure. Supervised features with the same evaluation recipe achieve 0.737.

#### Working hypotheses (plausible but unverified)

**The ~0.76 PER convergence on S14 reflects a spatial alignment bottleneck.** This is one of several possible explanations (see Section 4.4). Alternatives include target-data scarcity, measurement noise from small validation folds, and S14-specific idiosyncrasy. The spatial alignment hypothesis is motivated by the architectural observation that per-patient spatial encoders cannot share spatial knowledge, but it has not been isolated via ablation.

**MNI coordinate-based spatial alignment will improve decoding.** This follows from the information-theoretic observation that MNI coordinates are currently unused information relevant to the decoding problem (Section 5). Whether the practical gain is large enough to matter depends on registration accuracy, the model's ability to exploit coordinates, and whether spatial alignment is actually the binding constraint.

**Cross-task data pooling (Lexical/pseudoword patients) is a high-value, low-risk path to more data.** The pseudoword repetition task (Duraivel 2025) uses the same 52 CVC/VCV tokens with the same 9 phonemes, overlapping patients (3 shared with Spalding), and compatible preprocessing. This nearly doubles training data with minimal methodological risk.


## 3. The Neural Manifold Connection

### 3.1. Two views of the same phenomenon

The cortical field model (Section 2) and the neural manifold hypothesis describe the same system at different levels:

- **The field view** (spatial): Activity $a(\mathbf{p}, t)$ is a function on the cortical surface evolving over time. Each electrode samples this field at a known position.

- **The manifold view** (population): The vector of all electrode activities $\mathbf{d}(t) = [d_1(t), \ldots, d_N(t)]$ lies on a low-dimensional manifold $\mathcal{M} \subset \mathbb{R}^N$. Motor behavior corresponds to trajectories on $\mathcal{M}$.

The bridge is a **latent state variable**. Let $\mathbf{z}(t) \in \mathbb{R}^k$ be the low-dimensional articulatory state — the configuration of the vocal tract at time $t$. Each electrode's activity is determined by two things: **where it sits** on the cortical surface and **what the articulatory state is**:

$$d_j^{(i)}(t) = g\bigl(\mathbf{p}_j^{(i)},\ \mathbf{z}(t)\bigr) + \epsilon_j^{(i)}(t)$$

where $g: \mathcal{S} \times \mathbb{R}^k \to \mathbb{R}$ is a **shared observation function** mapping (cortical position, articulatory state) to neural activity. This is the somatotopic map made explicit: electrodes over lip motor cortex respond to lip-state variables; electrodes over tongue motor cortex respond to tongue-state variables.

The dynamics on the manifold are determined by the phoneme sequence:

$$\dot{\mathbf{z}}(t) = h(\mathbf{z}(t);\ \mathbf{y})$$

Different phoneme sequences $\mathbf{y}$ produce different trajectories through articulatory state space.

### 3.2. Why this formulation matters

The observation function $g(\mathbf{p}, \mathbf{z})$ is the central learning target. It says: **the same function generates neural activity for all patients.** What differs is the set of positions $\{\mathbf{p}_j^{(i)}\}$ at which each patient samples $g$.

Without knowing electrode positions, the model must learn a separate observation function per patient — this is what per-patient SpatialConv does. With positions (MNI coordinates as an approximation of $\mathbf{p}$), the observation function $g$ can be parameterized as a shared function of brain coordinates, and cross-patient alignment is built into the architecture rather than learned implicitly.

### 3.3. The articulatory manifold has known structure

For speech, the manifold $\mathcal{M}$ is not abstract — it has physical meaning corresponding to vocal tract configuration. The articulatory degrees of freedom:

| Articulator | Approx. DoF | Controls |
|------------|-------------|----------|
| Tongue (tip, body, dorsum) | ~6 | Place of articulation, vowel quality |
| Jaw | ~2 | Aperture, protrusion |
| Lips | ~2 | Rounding, closure |
| Larynx | ~2 | Voicing, pitch |
| Velum | ~1 | Nasal/oral |
| **Total** | **~13-15** | |

For our 9 phonemes specifically, the effective dimensionality is lower. The contrasts that distinguish our phoneme inventory:

| Contrast | Phonemes distinguished | Approx. dimensions |
|----------|----------------------|-------------------|
| Place of articulation | labial (b,p,v) vs velar (g,k) | ~2 |
| Manner | stop vs fricative vs vowel | ~2 |
| Voicing | voiced (b,g,v) vs voiceless (k,p) | ~1 |
| Vowel height | low (a,ae) vs high (i,u) | ~1 |
| Vowel backness | front (ae,i) vs back (a,u) | ~1 |

**Effective manifold dimension for our task: ~5-8.** This is far below the electrode count (63-201), consistent with massive oversampling of a low-dimensional manifold. The cross-patient decoding problem reduces to: given sparse samples of the observation function $g(\mathbf{p}, \mathbf{z})$ at known positions $\mathbf{p}$, infer the latent trajectory $\mathbf{z}(t)$ and classify the phoneme sequence.

### 3.4. Cross-patient transfer as manifold alignment

Each patient's electrode array gives a different **projection** of the shared manifold:

$$\mathbf{d}^{(i)}(t) \approx W^{(i)} \mathbf{z}(t) + \boldsymbol{\epsilon}^{(i)}$$

where $W^{(i)} \in \mathbb{R}^{N_i \times k}$ is determined by which cortical positions the electrodes sample. Different patients have different $W^{(i)}$ because they have different electrode positions — different projections of the same underlying manifold.

Existing cross-patient alignment methods (CCA, Procrustes, CORAL) attempt to align these projections post-hoc. All failed in our experiments (Section 4). MNI coordinates offer an alternative: **aligning the projections a priori** by providing the geometric relationship between electrode positions.

The key question is whether the inter-subject correspondence uncertainty (several mm) is small enough relative to the spatial scale of the somatotopic map (~15-20 mm for the articulatory zones) that MNI coordinates provide useful alignment. Chen et al. (2025) suggest yes for standard ECoG at ~10 mm spacing. Whether this holds for uECOG at ~2 mm spacing is an open empirical question.


## 4. Decomposing Cross-Patient Variation

### 4.1. The spatial transform $T_i$

The primary source of cross-patient variation is **geometric**: different electrode positions sample different regions of the shared field. The transform $T_i$ mapping patient anatomy to canonical space decomposes:

$$T_i(\mathbf{p}) = A_i \cdot \mathbf{p} + \mathbf{b}_i + \delta_i(\mathbf{p})$$

where:

- $A_i \in \mathbb{R}^{3 \times 3}$ is a **rigid or affine transform** (rotation, scaling) reflecting gross anatomical differences. MNI normalization estimates and corrects this.

- $\mathbf{b}_i \in \mathbb{R}^3$ is a **translation** (array placement offset). Our arrays span 15-25 mm in MNI space.

- $\delta_i(\mathbf{p})$ is a **nonlinear residual** capturing sulcal geometry, local cortical folding, and — critically for tumor patients — **macroscopic warping from mass effect**. Tumor displacement is a geometric mapping error, not signal noise: it physically moves the somatotopic map relative to skull-based coordinates. This is the most severe form of $\delta_i$ in our cohort and violates the affine assumption of $A_i$.

After MNI normalization, we observe electrodes at positions $\hat{\mathbf{p}}_j^{(i)}$ with residual error $\sim\|\delta_i\|$.

### 4.2. Inter-subject correspondence uncertainty

**The "~5 mm" figure is not a stable physical constant.** The actual uncertainty has multiple sources that should be separated:

1. **Native-space electrode localization**: Validated pipelines can achieve sub-millimeter accuracy in some settings — this step is relatively precise.

2. **Brain shift correction**: Intra-operative brain shift (from craniotomy, CSF drainage, gravity) displaces the cortex relative to pre-operative imaging. Correction quality varies.

3. **Warping to common atlas space**: The nonlinear registration from patient anatomy to MNI-152 is the dominant source of inter-subject correspondence uncertainty. Quality depends on the algorithm, template, and — for tumor patients — whether the lesion distorts the warp.

The net result is that inter-subject correspondence has **several millimeters of uncertainty** for healthy anatomy, potentially more near tumors. For our purposes:

- At the scale of articulatory somatotopy (~15-20 mm zones): correspondence is reliable
- At the scale of electrode spacing (~2 mm): correspondence is unreliable
- In between (~5-10 mm): partially informative, accuracy varies by patient and region

### 4.3. Resolution mismatch and dual spatial encoding

uECOG electrode spacing (~2 mm) is finer than inter-subject correspondence uncertainty. This means:

| Scale | Resolution | Best encoded by | What it captures |
|-------|-----------|-----------------|------------------|
| Cross-patient | ~10-15 mm | MNI coordinates | Articulatory somatotopy (which cortical zone) |
| Intra-patient global | ~5-10 mm | MNI (approximate) | Array position on cortex |
| Intra-patient local | ~2 mm | Grid topology / local connectivity | Fine-grained spatial patterns |

This implies any coordinate-aware architecture should use **dual spatial encoding**: MNI coordinates for cross-patient alignment at the somatotopic scale, and local spatial structure (grid adjacency or learned) for within-patient fine-grained patterns.

### 4.4. The patient-specific residual $r^{(i)}$

After accounting for shared dynamics and spatial transform, the residual captures:

1. **Fine-grained neural organization**: Within-articulator spatial patterns that vary across individuals (e.g., the exact spatial relationship between tongue-tip and tongue-body representations).

2. **Signal characteristics**: Electrode impedance, contact quality, and SNR. Z-score normalization addresses amplitude scaling but not higher-order distributional differences.

3. **Functional plasticity**: Individual speech development, accent, and articulatory habits produce patient-specific neural tuning within the shared somatotopic framework.

4. **Pathology-specific effects**: Parkinson's motor cortex excitability changes (all non-tumor patients are Parkinson's/DBS); tumor-adjacent cortical reorganization (distinct from the geometric displacement captured in $\delta_i$).


## 5. What Our Experiments Reveal

### 5.1. The evaluation recipe dominates at this data scale

Across experiments on S14, training improvements contributed ~7pp and evaluation improvements contributed ~6.5pp:

| Component | Category | PER improvement |
|-----------|----------|----------------|
| Label smoothing 0.1 | Training | -4.1pp |
| Mixup alpha=0.2 | Training | -1.7pp |
| Per-position heads + dropout | Training | -1.2pp |
| Weighted k-NN (k=10, cosine) | Evaluation | -2.1pp |
| TTA (16 augmented copies) | Evaluation | -1.4pp |
| Articulatory + CE dual head | Evaluation | -1.1pp |
| Best-of per fold | Evaluation | -1.8pp |

This suggests that the backbone already extracts most of the signal available at this data scale, and the bottleneck is extracting predictions from the representation.

### 5.2. Multi-scale temporal structure is validated

Multi-scale temporal encoding (stride 3+5+10) was the single best architectural improvement (PER 0.766 -> 0.750). The three scales correspond to articulatory dynamics at different timescales:

| Scale | Temporal resolution | Articulatory correlate |
|-------|-------------------|----------------------|
| Stride 3 (~67 Hz) | ~15 ms | Fast transitions (stop release, tongue flap) |
| Stride 5 (~40 Hz) | ~25 ms | Syllable dynamics (CV transition) |
| Stride 10 (~20 Hz) | ~50 ms | Phoneme-level patterns (steady-state) |

### 5.3. SSL features carry partial structure

All SSL methods produce near-chance PER (0.85-0.91) under basic evaluation. However, JEPA features + LP-FT achieve PER 0.797, and weighted k-NN improves them by 2.9pp. SSL features are not random — they capture some spatial and temporal structure — but their feature geometry is far weaker than supervised features for phoneme discrimination.

TTA hurts JEPA features (-0.2pp), confirming that SSL features are not augmentation-invariant.

### 5.4. The ~0.76 PER convergence on S14

55 experiments across all paradigms converge to PER 0.750-0.780 on S14. This convergence has multiple contributing causes, and **attributing it to any single bottleneck would be premature**:

1. **S14 is no longer a held-out test patient.** With 55+ architectures evaluated against S14, it functions as a validation set. The convergence may partly reflect S14-specific characteristics — the noise floor of $r^{(14)}$ — rather than a universal information ceiling. Testing on additional patients is necessary to determine whether the wall is fundamental.

2. **Measurement noise from small validation folds.** ~30 samples per fold implies +-0.05 PER noise. Fold difficulty varies (fold 5 ~0.85, fold 4 ~0.70). Differences <2pp are within noise.

3. **Target-data scarcity.** S14 has 153 trials (~120 per training fold). Stage 2 adaptation contributes only ~1.6pp. The wall may reflect insufficient target data rather than insufficient cross-patient alignment.

4. **Spatial alignment may be one contributing factor.** Per-patient spatial encoders cannot share spatial knowledge. The shared backbone learns only temporal, not spatial, correspondence. But this has not been isolated as THE bottleneck.

Breaking through requires testing multiple hypotheses: **(a)** population-level evaluation on all patients (reducing measurement noise and S14 overfitting), **(b)** cross-task data pooling (addressing data scarcity directly), and **(c)** MNI coordinate-based spatial alignment (addressing spatial correspondence). These are complementary, not competing.


## 6. The Case for MNI Coordinates

### 6.1. Currently unused information

The current LOPO pipeline processes spatial information through **per-patient Conv2d** operating on grid topology:

$$d^{(i)} \in \mathbb{R}^{H_i \times W_i \times T} \xrightarrow{\text{Conv2d}_i} \mathbb{R}^{d \times T} \xrightarrow{\text{shared backbone}} \mathbb{R}^{d' \times T'}$$

Each patient's Conv2d is independent. The model has **zero information** about the spatial relationship between patient $i$'s electrodes and patient $i'$'s electrodes. The shared backbone implicitly assumes that its input features are homologous across the channel dimension — that feature dimension $k$ from patient $i$'s SpatialConv and feature dimension $k$ from patient $i'$'s SpatialConv carry corresponding information. This assumption is anatomically false: independent per-patient Conv2d layers produce feature spaces with no guaranteed alignment.

MNI coordinates are available for all patients but unused. They provide an explicit spatial correspondence between electrodes across patients.

### 6.2. What coordinates enable (in principle)

The observation function formulation from Section 3 makes the value explicit: if $d_j^{(i)}(t) = g(\mathbf{p}_j^{(i)}, \mathbf{z}(t)) + \epsilon$, then knowing $\mathbf{p}_j^{(i)}$ allows the model to learn $g$ as a shared function rather than learning independent mappings per patient.

Concretely, MNI coordinates enable:

1. **Identifying which articulatory zone each electrode samples.** The somatotopic map operates at ~15-20 mm scale, well above the inter-subject correspondence uncertainty.

2. **Shared spatial representations across patients.** Electrodes at similar MNI coordinates across patients can share representations, pooling spatial information across the cohort.

3. **Contextualizing variable electrode counts.** A patient with 201 electrodes and one with 63 have different cortical coverage. Coordinates tell the model which regions are covered and which are absent.

### 6.3. What coordinates do NOT guarantee

The availability of MNI coordinates does not automatically improve decoding. Coordinates help only if:

- The inter-subject correspondence uncertainty is small enough relative to the somatotopic scale that the coordinates provide useful alignment (plausible for ~15-20 mm articulatory zones, uncertain within ~5 mm neighborhoods)
- The model architecture can exploit the coordinate information (a design question)
- Spatial alignment is actually a binding constraint on performance (unverified — see Section 5.4)
- The coordinates are accurate enough for the specific patients (tumor patients may have larger errors)

**MNI coordinates are a necessary but not sufficient ingredient for coordinate-based cross-patient transfer.** Their value must be demonstrated empirically.


## 7. Available Data and Experimental Paths

### 7.1. Currently used

| Data source | Volume | Usage |
|------------|--------|-------|
| PS task trial-level HGA | 9 patients, ~1315 trials, ~22 min | Supervised LOPO training |
| S14 trial-level HGA | 153 trials, ~2.5 min | Target patient evaluation |
| Grid topology (from TSV) | Per-patient grid shapes | SpatialConv input structure |

### 7.2. Available but unused

| Data source | Volume | Potential usage | Risk level |
|------------|--------|-----------------|------------|
| **Cross-task pooling** (Duraivel pseudoword) | 3 overlapping patients, 156-208 trials each, same 9 phonemes | Nearly doubles labeled data for shared patients | **Low** — same tokens, same phonemes, same preprocessing, validated by Duraivel |
| MNI coordinates | All patients | Cross-patient spatial alignment | Medium — exist but need sourcing; architectural change required |
| Continuous recordings | ~8 min/patient x 21 patients | Temporal SSL | Medium — dominated by silence/non-speech; requires aggressive curation (VAD or energy thresholding) to avoid training on OR ambient dynamics |
| Lexical task patients | 13 patients, ~156-208 trials each | Spatial SSL, additional source patients (28 phonemes) | Medium — different phoneme set complicates supervised use |
| Audio recordings | 4 variants/patient | Auxiliary supervision | Medium — alignment with neural data needs verification |

### 7.3. Cross-task pooling deserves priority

The pseudoword repetition data (Duraivel 2025) is the lowest-risk, highest-confidence path to more training data:

- Same 52 CVC/VCV tokens, same 9 phonemes
- 3 patients overlap with our Spalding cohort (Duraivel S1=Spalding S8, S2=S5, S3=S1/S2)
- Same uECOG recording modality, same preprocessing pipeline, same IRB
- 156-208 additional trials per overlapping patient
- Validated by Duraivel's published analysis of the task structure

This should be pursued as a **central experimental branch**, not a side opportunity. It directly addresses the target-data scarcity bottleneck (Section 5.4) without requiring any architectural innovation.

### 7.4. The continuous data caveat

The ~168 minutes of continuous recordings are not ~168 minutes of useful SSL signal. Intra-operative recordings are dominated by:

- Silence between trials
- Operating room ambient noise
- Auditory perception (hearing the stimulus)
- Non-speech motor activity
- Anesthesia-related baseline fluctuations

Naive application of masked reconstruction or temporal prediction to this data would spend most capacity learning the dynamics of "operating room at rest." Utilizing continuous data for SSL requires an **aggressive curation pipeline** — voice activity detection, energy thresholding, or task-epoch selection — to extract the subset of recordings with speech-relevant neural activity.


## 8. Summary

### Externally established

- Per-patient spatial layer + shared temporal backbone is the optimal transfer architecture (Singh 2025, Willett 2023)
- Articulatory somatotopy exists in vSMC and is intrinsic to motor cortex circuitry (Metzger 2023, Bouchard 2013)
- MNI/atlas coordinates enable cross-patient ECoG transfer (Chen 2025)
- Motor cortex manifold dynamics are preserved across individuals (Safaie 2023) and stable over time (Gallego 2020)
- Articulatory features are robustly encoded in vSMC (Wu 2025, citing Chartier 2018, Anumanchipalli 2019)

### Internally observed

- Cross-patient LOPO transfer works: PER 0.762 vs per-patient 0.737 (S14, grouped CV)
- Articulatory bottleneck head improves cross-patient decoding
- Multi-scale temporal encoding (stride 3+5+10) is the best validated architectural improvement
- Evaluation recipe contributes ~6.5pp, rivaling training improvements (~7pp)
- All SSL methods fail at current data scale (~22 min) but features carry partial structure
- 55 experiments on S14 converge to PER 0.750-0.780 across all paradigms
- All domain adaptation methods (DANN, CORAL, CCA) provide zero benefit

### Working hypotheses

- The ~0.76 convergence on S14 reflects (in part) a spatial alignment bottleneck — but this has not been isolated from target-data scarcity, measurement noise, or S14-specific effects
- MNI coordinate-based spatial alignment will improve decoding — but the practical gain depends on registration accuracy, architectural design, and whether spatial alignment is binding
- The neural field $a(\mathbf{p}, t)$ decomposes into shared dynamics + patient-specific spatial transform + residual — a productive formulation, but the relative magnitudes are not yet measured
- Cross-task pooling is a high-value, low-risk path to more data — supported by task compatibility evidence (Duraivel 2025) but not yet tested in our pipeline
- Continuous recordings provide meaningful SSL signal after curation — the 9x data increase is an upper bound; the usable fraction is unknown

### The central question (restated with appropriate uncertainty)

Our current results are consistent with the hypothesis that explicit spatial correspondence is one remaining bottleneck, but they do not yet isolate that bottleneck from target-data scarcity, fold noise, and coverage mismatch. The highest-confidence experimental paths are: **(a)** population-level evaluation on all patients, **(b)** cross-task data pooling, and **(c)** MNI coordinate-based spatial alignment. These address complementary aspects of the problem and should be pursued in parallel where feasible.
