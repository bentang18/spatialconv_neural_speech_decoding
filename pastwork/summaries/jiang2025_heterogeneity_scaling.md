# Jiang et al. 2025 - Data Heterogeneity Limits Scaling of Neural Data Transformers

## Citation
Jiang, Y., Azabou, M., Ye, J., Dyer, E.L., & Pandarinath, C. (2025). Data Heterogeneity Limits Scaling of Neural Data Transformers. *Preprint*.

## Setup
- **Recording modality**: Neuropixels extracellular electrophysiology (spikes/LFP). Two datasets: Reaching-Sorting (RS, same brain region across sessions) and BrainWide Map (BWM, different brain regions)
- **Species**: Mice
- **Data scale**: RS: 130 sessions, BWM: 459 sessions. Both from International Brain Laboratory (IBL)
- **Tasks**: RS: reaching + object sorting (motor). BWM: stimulus detection (perceptual). Multiple downstream targets per dataset
- **Key variable**: Data heterogeneity -- RS has consistent region coverage; BWM has massive cross-session variability in recorded regions

## Architecture
- **Base model**: NDT-style Transformer encoder. 6-layer, d=512, 8 attention heads
- **Scale**: ~12M shared parameters + ~1.2M per-session "stitcher" parameters
- **Per-session stitchers**: Linear projection layers that map each session's unique neuron set to the shared embedding space. Analogous to per-patient read-in layers in BIT/v12
- **Temporal**: 20ms binned spike counts as input tokens
- **Spatial**: Per-neuron learned embeddings within each session (no cross-session spatial encoding)
- **Three pretraining objectives tested**:
  1. **Forward prediction**: Predict next timestep neural activity from history (autoregressive)
  2. **Masked prediction**: Reconstruct masked neural tokens (BERT-style)
  3. **Co-smoothing**: Reconstruct held-out neurons from observed subset (spatial completion)

## SSL/Pretraining
- **All three objectives are self-supervised** (no behavioral labels during pretraining)
- **Forward prediction** is most robust to data heterogeneity -- temporal dynamics are shared across brain regions even when spatial structure varies
- **Masked prediction** is second-best, reasonably robust
- **Co-smoothing** scales well on homogeneous data (RS) but FAILS to scale on heterogeneous data (BWM). Cross-region co-variation patterns don't generalize
- **Scaling curves**: Plot downstream performance vs number of pretraining sessions (1, 5, 10, 20, 40, 80, 130 for RS; up to 459 for BWM)

## Cross-Patient Handling
- **Per-session stitchers**: ~1.2M parameters per session (linear projections). This is expensive -- 10x more per-session params than v12's 134. But required because each session records different neurons
- **Session selection vs random**: On BWM, 5 carefully selected sessions (those with neurons in the target brain region) outperform 40 randomly selected sessions. 8x more data-efficient when selection accounts for relevance
- **Heterogeneity taxonomy**: The paper distinguishes three levels:
  1. Same region, same task (RS) -- scaling works for all objectives
  2. Different regions, same task (BWM) -- forward prediction scales, co-smoothing doesn't
  3. Different regions, different tasks -- not tested but expected to be hardest
- **Rankings are target-specific**: The best pretraining sessions for one downstream target differ from those for another. No universal "good" pretraining data

## I/O Features
- **Input**: Binned spike counts (20ms bins) per neuron per session
- **Output**: Behavioral variables (reaction times, movement kinematics, stimulus identity) via linear probes on pretrained representations
- **Evaluation**: Linear probe R^2 / accuracy on frozen pretrained representations (standard SSL evaluation protocol)

## Key Results
| Setting | Best Objective | Scaling Behavior |
|---|---|---|
| RS (homogeneous) | All three scale | Log-linear improvement to 130 sessions |
| BWM forward pred | Forward prediction | Modest scaling to 459 sessions |
| BWM masked pred | Masked prediction | Modest scaling |
| BWM co-smoothing | Co-smoothing | **FLAT or declining** beyond ~20 sessions |
| BWM 5 selected | Forward prediction | **Matches or beats 40 random** |

Key findings:
- **Data heterogeneity breaks naive scaling**: Simply adding more sessions does NOT guarantee improvement when sessions come from different brain regions. Co-smoothing actually degrades with more heterogeneous data
- **Forward prediction (temporal) is most robust**: Temporal dynamics generalize across brain regions better than spatial co-variation patterns. Validates temporal masking (BIT, v12) over spatial masking for heterogeneous data
- **Session selection >> random accumulation**: 5 relevant sessions > 40 random sessions (8x efficiency). Quality over quantity when data is heterogeneous
- **Rankings are target-specific**: No universal pretraining recipe -- the best data depends on what you want to decode downstream
- **Per-session stitchers are essential**: Without them, cross-session transfer fails entirely (consistent with NDT3, BIT findings)

## v12 Comparison

**Directly relevant to v12's multi-patient pretraining strategy.** v12 will pretrain on heterogeneous data (different patients, different array placements, different cortical coverage). Jiang 2025 provides the first systematic study of how this heterogeneity affects scaling.

**Key implications for v12:**
1. **Temporal masking is the right SSL objective**: Forward prediction / temporal masking scales with heterogeneous data; co-smoothing (spatial) does not. v12's plan for temporal span masking following BIT is validated by this finding. Spatial masking (BarISTA, Charmander) may not scale when combining patients with different cortical coverage
2. **Patient selection matters more than patient count**: Adding patients randomly may not help v12. Patients with overlapping cortical coverage (S26-S33, S22-S62) are likely more valuable than adding distant-coverage patients. This suggests a curriculum: start with high-overlap pairs, then gradually add patients
3. **5 selected > 40 random (8x efficiency)**: With v12's 11 patients, careful selection of the 4 core patients (S14, S26, S33, S62) is likely better than using all 11 with heterogeneous quality. This supports the core patient strategy in CLAUDE.md
4. **Per-session parameters are non-negotiable**: ~1.2M/session in their work, 134/patient in v12. Both papers converge on the necessity of per-session alignment layers

**Key differences:**
- **Modality**: Neuropixels spikes vs uECOG HGA. Different signal-to-noise characteristics
- **Scale**: 130-459 sessions vs 11 patients. v12 is far below the scaling regime where heterogeneity effects become pronounced
- **Spatial encoding**: Per-neuron learned embeddings (no coordinates). v12 uses MNI coordinates + atlas VEs, which should provide better cross-patient spatial alignment than learned embeddings
- **Heterogeneity source**: Different brain regions (BWM) vs different cortical positions over similar speech regions (v12). v12's heterogeneity is milder -- all patients are recording speech-motor cortex, just from different angles. This suggests v12's scaling behavior will be closer to RS (homogeneous) than BWM (heterogeneous)

**What to import:**
- **Session/patient ranking by target relevance**: Before multi-patient pretraining, rank patients by coverage of the target cortical regions (using VE reachability from atlas.py). Train on high-relevance patients first
- **Forward prediction as diagnostic**: Use forward prediction performance as a health check during v12 SSL -- if it doesn't improve with more patients, heterogeneity is the bottleneck
- **Curriculum by relevance**: Start pretraining on the 4 core patients (high quality, good coverage overlap), then gradually add extended patients. Monitor for degradation

**Common mistakes:**
- Do NOT assume more patients = better pretraining. Heterogeneity can cause scaling to plateau or even degrade (co-smoothing result)
- Do NOT use spatial masking SSL with heterogeneous patient data. Temporal masking is robust to spatial heterogeneity; spatial masking is not
- Do NOT treat all pretraining data as equally valuable. Patient selection by cortical coverage overlap is 8x more efficient than random inclusion
- Do NOT extrapolate their ~1.2M/session stitcher cost to v12. Their stitchers must handle arbitrary neuron identities; v12's 134-param diagonal normalization handles impedance/gain (much simpler variation)
