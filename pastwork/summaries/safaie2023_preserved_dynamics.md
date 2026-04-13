# Safaie et al. 2023 - Preserved Neural Dynamics Across Animals

## Citation
Safaie, M., Chang, J.C., Park, J., Miller, L.E., Ames, K.C., Lara, A.H., & Bhatt, D.K. (2023). Preserved neural dynamics across animals performing similar behaviour. *Nature*, 623, 765-771.

## Setup
- **Recording modality**: Intracortical -- 96-channel Utah arrays (monkeys) + Neuropixels (mice). Motor cortex in both species
- **Species**: 3 macaque monkeys + 4 mice
- **Tasks**: Arm reaching (monkeys, center-out, 8 directions) and forelimb reaching (mice, 4 directions). Kinematically similar motor tasks across species
- **Data scale**: ~100-400 trials per animal per condition. Standard single-session motor recordings
- **Analysis focus**: NOT a decoding model paper -- this is a neuroscience discovery paper. Uses PCA + CCA to reveal shared latent dynamics, then validates with cross-animal LSTM decoding

## Architecture
- **NOT a neural network architecture paper.** Uses established analysis methods:
  1. **PCA**: Reduce single-animal neural activity (N neurons x T timepoints) to low-dimensional latent trajectories (10-20 PCs)
  2. **CCA**: Align latent spaces across animals by finding maximally correlated dimensions. Linear alignment only -- no nonlinear warping
  3. **LSTM decoder**: Standard LSTM trained on one animal's aligned latent dynamics, tested on another animal's aligned dynamics. ~50K params. Used to validate that shared dynamics are decodable, not as the main contribution
- **Alignment pipeline**: Record → z-score → PCA (per animal) → CCA (cross-animal) → aligned latent space → cross-animal LSTM decoding

## SSL/Pretraining
- **N/A**: No SSL or pretraining. This is an analysis paper, not a representation learning paper. PCA + CCA are fit analytically (closed-form), not trained

## Cross-Patient Handling
- **PCA + CCA alignment**: Linear alignment between animals' latent spaces. CCA finds the rotation/projection that maximizes correlation between two animals' neural trajectories during similar behaviors
- **Behavioral similarity REQUIRED**: Alignment only works when animals perform kinematically similar tasks. Different behaviors → different dynamics → CCA fails to find shared structure. This is a crucial caveat
- **Linear alignment sufficient**: Nonlinear methods (e.g., KCCA) provide negligible improvement over linear CCA. The shared dynamics are linearly related across animals
- **~60 neurons minimum**: Below ~60 simultaneously recorded neurons, the PCA latent space is too noisy for reliable CCA alignment. Above 60, alignment quality plateaus. This sets a minimum electrode count for cross-animal transfer
- **Cross-animal LSTM R^2 ≈ 0.86**: After CCA alignment, an LSTM trained on animal A's aligned dynamics decodes animal B's kinematics at R^2 ≈ 0.86. Without alignment: R^2 ≈ 0.02 (chance). With alignment: 0.86. This is the core result

## I/O Features
- **Input**: Spike counts from Utah arrays / Neuropixels, binned at 10-50ms
- **Output**: Kinematic variables (hand/forelimb velocity, position)
- **Dimensionality**: 10-20 PCA dimensions capture >80% of neural variance. CCA alignment operates in this reduced space
- **Species**: Cross-species (monkey ↔ mouse) as well as within-species (monkey ↔ monkey)

## Key Results
| Configuration | Decoding R^2 | Notes |
|---|---|---|
| Cross-animal LSTM (CCA-aligned) | **0.86** | Core result |
| Cross-animal LSTM (unaligned) | 0.02 | Chance level |
| Within-animal LSTM | 0.92 | Upper bound |
| Cross-species (monkey → mouse, aligned) | 0.78 | Remarkable cross-species transfer |
| RNN simulation (unconstrained) | Variable dynamics | Dynamics diverge |
| RNN simulation (connectivity-constrained) | Shared dynamics | Constraint needed |

Key findings:
- **Motor cortex dynamics are conserved across animals**: The same low-dimensional trajectories appear in motor cortex during similar behaviors, regardless of the specific neurons recorded or even the species. This is a fundamental neuroscience finding
- **Linear alignment is sufficient**: CCA (linear) works as well as nonlinear methods. The relationship between animals' latent dynamics is a simple rotation, not a complex warping
- **Behavioral similarity is required**: Animals must perform similar motor actions for dynamics to align. Different behaviors produce different dynamics even in the same brain area. Alignment is behavior-specific, not anatomy-specific
- **~60 neurons minimum for reliable alignment**: Below this, PCA latent spaces are too noisy. v12 has 63-201 significant channels -- right at or above this threshold
- **RNN simulation insight**: Computational models of motor cortex produce shared dynamics ONLY when connectivity constraints are imposed. Without constraints, each network finds its own solution. This implies that shared dynamics arise from conserved circuit architecture, not training on similar data
- **Conservation spans species**: Monkey-mouse alignment (R^2=0.78) works despite >75 million years of evolutionary divergence. Within-species alignment is even stronger

## v12 Comparison

**This is the theoretical foundation for v12's cross-patient premise.** Safaie 2023 provides the strongest neuroscience evidence that neural dynamics are conserved across individuals performing similar tasks. If motor cortex dynamics are shared even across species (monkey ↔ mouse), then cross-patient speech motor dynamics in humans should be shared -- the variation from electrode placement is trivial compared to cross-species variation.

**Key implications for v12:**
1. **Shared dynamics exist -- the barrier is alignment, not biology**: Cross-animal R^2=0.02 (unaligned) → 0.86 (CCA-aligned) proves that shared dynamics are there but hidden by different neural bases (different neurons, different coordinate systems). v12's VE cross-attention + per-patient diagonal normalization is an alignment mechanism -- it maps each patient's electrodes into the same atlas-grounded space. If the alignment works, shared dynamics should be recoverable
2. **Linear alignment suffices**: CCA (linear rotation) captures most of the shared structure. v12's per-patient diagonal normalization (scale + bias) is even simpler than a full rotation. This is encouraging -- suggests that simple per-patient layers can handle the cross-patient alignment for surface recordings where variation is primarily gain/impedance
3. **~60 neurons minimum**: v12's patients have 63-201 significant channels. All are at or above the threshold for reliable cross-animal alignment. Even S23 (63 sig channels) should have enough spatial coverage
4. **Behavioral similarity required**: Animals must perform the same task for dynamics to align. All v12 patients perform non-word repetition (same 52 CVC/VCV tokens, same 9 phonemes). This condition is satisfied by design

**Key differences:**
- **Motor reaching vs speech production**: Safaie studies arm reaching (simple, 8 directions, well-studied). Speech production is far more complex (9 phonemes x 3 positions, temporal sequencing, articulatory coordination). Shared dynamics may be lower-dimensional or noisier for speech
- **Spikes vs HGA**: Safaie uses spike trains (action potentials). v12 uses HGA (postsynaptic field potentials). Different biophysics. HGA reflects population-level input activity averaged over ~1mm, which may actually be BETTER for revealing shared dynamics (averaging over individual neuron variability)
- **Cortical surface vs intracortical**: Utah arrays (Safaie) record from a fixed cortical depth. uECOG (v12) records from the surface. Surface recordings may have more spread/blurring, but also capture broader cortical patches
- **PCA+CCA vs learned alignment**: Safaie uses closed-form analytical alignment. v12 learns alignment end-to-end via VE cross-attention. v12's approach is more flexible but requires training data; Safaie's is data-efficient but limited to linear

**What to import:**
- **CCA alignment as diagnostic**: After v12 training, extract per-patient VE representations and check if CCA alignment reveals shared dynamics across patients. High CCA correlation = v12's VE space is successfully recovering shared structure. Low CCA correlation = alignment is failing
- **~60 neuron threshold**: When evaluating per-patient results, flag patients below 63 significant channels (S23 at 63 is borderline). These patients may have insufficient spatial coverage for cross-patient alignment
- **The RNN constraint insight**: Shared dynamics arise from conserved circuit architecture, not from training on similar data. This supports v12's atlas prior -- providing anatomical constraints (Brainnetome positions) encourages the model to learn solutions consistent with conserved speech circuit architecture
- **Linear sufficiency as design principle**: If linear CCA captures most shared structure, v12's per-patient diagonal (even simpler than linear rotation) should be sufficient for cross-patient alignment. Full affine or nonlinear per-patient layers are unnecessary overhead

**Common mistakes:**
- Do NOT claim speech dynamics are as well-conserved as motor reaching dynamics. Safaie shows conservation for simple reaching. Speech production has not been tested at this level of cross-subject dynamics analysis. v12 must demonstrate this empirically
- Do NOT equate CCA alignment with v12's learned alignment. CCA is analytical, requires trial-averaged condition means, and operates on latent trajectories. v12 learns end-to-end from single trials. The alignment mechanisms are fundamentally different
- Do NOT assume R^2=0.86 cross-animal transfer translates to any specific PER for v12. The tasks, modalities, and evaluation metrics are incomparable
- Do NOT overlook the behavioral similarity requirement. Safaie explicitly shows alignment fails when animals perform different tasks. If patients exhibit different speech strategies (e.g., compensatory articulation), cross-patient dynamics may diverge
