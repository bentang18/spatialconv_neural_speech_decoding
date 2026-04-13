# v12 Architecture — Paper Comparisons

Systematic comparison of papers against the Neural Field Perceiver v12 design.
Focus on 4 axes: architecture, SSL, anatomical variation, and I/O features.

**v12 reference architecture:**
```
electrode HGA + MNI coord
→ Conv1d(1→d, k=10, s=10)           shared temporal binning
→ s^(i) ⊙ x + b^(i)                per-patient diagonal (128 params/pt)
→ + Fourier PE(MNI + Δ/ω)           spatial identity
→ VE cross-attention (L=16 atlas, distance bias)  N_i → 16 common space
→ [VE self-attn (16×16) → Temporal self-attn] × B
→ AR decoder (beam search, 52 CVC/VCV tokens)
```

---

## 4 Comparison Axes

### A. Architecture
How is the model structured? What's shared vs per-patient? What are the key inductive biases? What's the backbone (CNN/RNN/Transformer)? How do brain regions or electrodes interact? How many params total and per-patient?

**v12:** Shared Conv1d temporal binning → per-patient diagonal (134 params) → Fourier PE on MNI → VE cross-attention (16 atlas positions, distance bias) → factored [VE self-attn → temporal self-attn] × B → AR decoder. ~171K total.

### B. SSL Objective
What's the self-supervised pretext task? What's masked / predicted / reconstructed? Temporal vs spatial masking? What loss? How much data? Does it actually help vs training from scratch?

**v12:** Temporal span masking (BIT-style: T_patch=5, max_span=15, ratio=0.5). MSE reconstruction. Per-patient layers active during SSL. 3-stage: sEEG SSL (~1000 min) → uECoG SSL (456 min) → supervised fine-tune.

### C. Cross-Patient Anatomical Variation
How do they handle the fact that every patient has electrodes in different places? What's the common space? Hard parcellation vs soft attention vs learned alignment? Do they use coordinates? What atlas? How do they handle missing coverage? What about within-vs-between patient variation?

**v12:** 16 Brainnetome sub-gyral atlas positions as VEs. Soft distance-biased cross-attention creates ~25mm receptive fields. Learned Δ/ω corrects registration error (6 params/pt). Deformable atlas ablation (A_deform, +48/pt). Electrode coordinates: ACPC → talairach.xfm → MNI → Δ/ω → Fourier PE(F=3).

### D. Input / Output Features
What goes in? Raw broadband, HGA, spikes, spectrograms? What comes out? Phoneme classes, phoneme sequences (CTC), speech parameters, continuous speech? What temporal resolution?

**v12 input:** HGA (70-150Hz, 200Hz, z-scored). Engineered from raw 2kHz via CAR → Gaussian filterbank → Hilbert → sum → z-score.
**v12 output:** 52 CVC/VCV tokens via AR decoder (teacher-forced training, constrained beam search inference). Per-phoneme epochs also supported (9-class CE for baselines).

### E. Lessons
What can we learn? What validates v12? What challenges our assumptions? Concrete ideas to import? Cautionary tales — what went wrong, what's over-engineered, what's under-reported?

---

## Paper Comparison Template

For each paper, fill in:
- **One-line summary** of the approach
- **Score card** across 4 axes (what they do, convergent/divergent with v12)
- **Validates** — what confirms v12 choices
- **Challenges** — what questions our assumptions
- **Import** — concrete ideas to bring into v12
- **v12 advantages** — where we're better
- **Key numbers** — results that matter for comparison

---

## 1. MIBRAIN (Wu, Di et al. 2025)

**Citation:** Wu et al., "Towards Unified Neural Decoding with Brain Functional Network Modeling," arXiv:2506.12055, 2025.
**Full summary:** `pastwork/summaries/MIBRAIN_2025.md`
**Setup:** 11 epilepsy patients, sEEG (depth electrodes, 30-88 ch/pt), 23 Mandarin consonants, audible + silent, 1-2 week monitoring.

### A. Architecture

Per-subject per-REGION 1D Conv encoders (2 blocks: k=7/s=4, k=5/s=2, d=32) — a **separate conv bank for every region of every subject**. Learnable prototype tokens (d=32) substitute for uncovered regions. Complete region sequence → shared group-wise temporal CNN (4 blocks k=3/s=2, D=4d=128, per-region independent). Region Attention Encoder: 2 MHSA blocks (FFN ratio 4) with ToMe (Bipartite Soft Matching) that dynamically groups co-active regions (N^c=21 → N^m<21). Per-subject MLP prediction heads. Total temporal downsample: 128×.

~2-5K params/pt (per-region conv banks + decoder). Shared: temporal CNN, region attention, prototype tokens.

**vs v12:** Same high-level flow (per-patient read-in → shared backbone → cross-region attention → per-patient head). But much heavier per-patient parameterization. Their per-region conv banks can't share temporal features across regions. Our shared Conv1d + 134-param diagonal is more elegant and scalable.

### B. SSL Objective

**Masked region autoencoding (SPATIAL masking).** Random subset of region tokens replaced by prototype tokens. Full sequence → temporal CNN → per-subject lightweight reconstructors → reconstruct original raw broadband sEEG. MSE loss. Masking ratio r ~ Uniform (bounds unspecified). 1000 epochs, AdamW lr=1e-5, wd=0.005, batch=16×N_subjects. Reconstructors discarded after pretraining.

The signal: "given what IFG and STG look like during /ba/, what should PreCen look like?" Directly trains prototypes to encode generic neural patterns per region by forcing cross-region imputation.

**vs v12:** We mask temporally (BIT-style spans), they mask spatially (regions). **Complementary pretext tasks** — temporal masking teaches dynamics, spatial masking teaches inter-region structure and trains prototypes. We should try both (new ablation A_ve_mask). **Cautionary:** No ablation of SSL vs training from scratch — can't determine how much the MAE actually helps vs the architecture alone.

### C. Cross-Patient Anatomical Variation

**Hard FreeSurfer parcellation.** T1 MRI → FreeSurfer → pial surface → each sEEG contact assigned to exactly one of 21 gyral regions (SPL, MFG, SFG, CG, IPL, PostCen, IFG, PreCen, PHG, Amyg, Hipp, STG, MTG, Pcun, FuG, INS, ITG, LG, Angular, Tha, SMG). Each patient covers a subset.

**NO MNI coordinates. NO PE. NO distance bias.** Alignment happens entirely through: (1) same region label → same position in token sequence, (2) prototype tokens bridge missing regions.

Unseen subjects: cosine similarity of trial-averaged responses → channel match → reorder/zero-pad → run through each existing encoder → majority vote. Acknowledged as limitation.

**vs v12:** Hard works for sEEG (contacts sit INSIDE structures — unambiguous). Soft is better for uECOG (surface electrodes straddle boundaries). Their success without coordinates is provocative — but regions are cm-scale (gyrus-level), coarser than the 15-25mm variation our distance bias addresses. v12 is the strict generalization: set α_h → ∞ and cross-attention becomes hard nearest-VE assignment.

### D. Input / Output Features

**Input:** Raw broadband sEEG at 512 Hz. Low-pass 200Hz + notch 50/100Hz, then downsampled. Model learns features end-to-end. Channel selection: PSD (4-150Hz band) articulation vs rest, paired t-test, BH-FDR P<0.05. Visual cortex excluded.

**Output:** 23-class CE (21 consonant initials + 2 simple finals). Per-subject MLP heads. No sequence decoding.

**vs v12:** We use engineered HGA at 200Hz — domain knowledge baked in, appropriate for ~1 min/pt. Their end-to-end approach needs more capacity. Our 52-token AR decoder is a harder decoding task. Their acoustic contamination check (Fig. S2: neural-audio correlation vs shuffled null) is worth replicating for our mic-feedback-prone patients.

### E. Lessons

**Validates v12:**
- Independent convergence on atlas region mapping — strongest possible evidence for VE approach
- Per-subject layers critical (even heavier than ours)
- SSL → supervised works at N=11
- Multi-sub > single-sub: +8%/+7% audible, +5%/+5% silent
- Imputed regions decode above chance — validates VE content vectors
- Generic pretrained models (Brant, BrainBERT) fail without cross-patient mechanism
- Factored processing (region-independent temporal → cross-region attention) = our design

**Challenges v12:**
- **Scaling initially hurts (CAUTIONARY).** 1-3 subjects degrade performance. Need ≥6. Don't bail at N<6.
- **Coordinates may not be needed** at gyrus-level. Distance bias adds complexity — verify via A_no_dist.
- **Spatial region masking works at N=11.** We only planned temporal masking.

**Import:**
1. VE masking SSL (HIGH) — new ablation A_ve_mask
2. Grad-CAM VE contribution analysis (HIGH for paper) — their Eq. 6
3. Audio contamination check (MEDIUM) — their Fig. S2
4. Functional collaboration visualization (MEDIUM for paper) — attention weights → connectivity

**Cautionary tales:**
- No SSL-vs-scratch ablation. Can't isolate MAE contribution from architecture. We must include this.
- Per-subject per-region encoders are a design dead end — don't scale to 50+ patients.
- Unseen-subject majority voting is hacky. Our LP-FT/TTO is strictly better.
- Scaling experiment on only 2 patients in enrollment order. Low statistical power.

### Key Numbers

| Metric | Value |
|--------|-------|
| Multi-sub gain (audible) | +8.08% (P10), +6.83% (P11) over single-sub |
| Multi-sub gain (silent) | +5.10% (P10), +4.97% (P11) |
| Scaling threshold | ≥6 subjects for P<0.05 improvement |
| LOO zero-shot | Above chance, all pts |
| Feature dim | d=32/region, D=128 after temporal CNN |
| Temporal downsample | 128× total |
| SSL | 1000 epochs, lr=1e-5, batch=16×N |
| Supervised | 200 epochs, lr=3e-4 |

---

## 2. Population Transformer (Chau et al. 2025)

**Citation:** Chau et al., "Population Transformer: Learning Population-Level Representations of Neural Activity," ICLR 2025.
**Full summary:** `pastwork/summaries/chau2025_population_transformer.md`
**Setup:** 10 sEEG subjects (Brain Treebank), 1688 electrodes (~167/subject), ~55.5h total, movie watching. 19 pretrain / 7 downstream.

### A. Architecture

**Modular two-stage: frozen temporal encoder → trainable spatial transformer.**

Stage 1: Per-channel temporal embedding via frozen pretrained encoder (BrainBERT d=768, or TOTEM d=64, Chronos d=512, TS2Vec d=320). NOT jointly trained.

Stage 2: Population Transformer. Input: temporal embedding + 3D sinusoidal PE (Left/Posterior/Inferior + ensemble membership). [CLS] token prepended. 6-layer Transformer encoder (d=512, 8 heads, dropout=0.1). Standard Vaswani self-attention over ALL channels regardless of distance. ~20M params total. **ZERO per-patient parameters.** Only per-patient info is 3D coordinates via PE.

**vs v12:** Both use 3D coordinates + attention for cross-patient spatial alignment. But PopT uses unconstrained self-attention (all channels attend to all others); v12 uses atlas-grounded cross-attention with distance bias (electrodes → 16 VE positions). PopT has no per-patient normalization; v12 has per-patient diagonal (134/pt). PopT operates on single time snapshots; v12 processes temporal sequences.

### B. SSL Objective

**Two discriminative objectives (NOT reconstructive):**

1. **Ensemble-wise:** Sample two disjoint channel subsets S_A, S_B. Draw activity at times t, t'. Predict if |t-t'| < 500ms (consecutive). BCE on [CLS]. Varying |S| trains robustness to different ensemble sizes.

2. **Channel-wise:** All channels at same time t. 10% randomly swapped with activity from random time t'. Predict which channels were swapped. BCE per token.

Total loss: L = L_ensemble + L_channel, equally weighted. LAMB lr=5e-4, batch=256, 500K steps, 2 days on 1 Titan RTX.

**Key ablation:** Reconstruction-only loss drops to 0.56 (Pitch) vs 0.74 (full). **Discriminative >> reconstructive** for this setting. BUT: BIT uses reconstructive successfully for speech — the difference may be that BIT reconstructs temporal sequences (dynamics), while PopT would reconstruct low-effective-dimension embeddings.

**vs v12:** We plan reconstructive temporal masking (BIT-style MSE). PopT's discriminative objectives are complementary — channel-swapping detection and temporal-proximity prediction could be added as auxiliary SSL losses for v12.

### C. Cross-Patient Anatomical Variation

**Self-attention + 3D PE only.** Each electrode's Left/Posterior/Inferior coordinate → sinusoidal encoding → summed with temporal embedding. No atlas, no VEs, no distance bias, no common space projection. Transformer self-attention must learn spatial relationships from scratch during pretraining.

Variable channel counts (72-205/subject) handled naturally by variable-length transformer input. Ensemble size varied during training for robustness.

**PE is critical:** Removing PE is the most damaging ablation (0.93→0.83 speech, 0.74→0.62 pitch). Bigger effect than removing either loss component. **This contradicts seegnificant (PE ΔR²=-0.02, NS)** — but seegnificant measured PE contribution ON TOP of spatial self-attention, while PopT removes PE entirely.

**Coordinate jitter (σ=5mm) didn't help** (Table 8). Challenges v12's planned coordinate jitter augmentation (2-5mm).

**vs v12:** PopT proves self-attention + PE works without atlas mapping. But: (1) they have ~167 electrodes/subject covering whole brain — much more spatial coverage than our focal arrays (63-201 channels over one cortical patch), (2) their tasks are binary detection, not phoneme decoding, (3) N=10 with ~167 electrodes might be enough for self-attention to learn population anatomy; our N=10 with focal coverage may not be.

### D. Input / Output Features

**Input:** Frozen temporal embeddings (d=64-768 depending on encoder) from broadband sEEG + 3D PE. NOT HGA, NOT raw signal to PopT. BrainBERT processes ~2s windows → single embedding per channel.

**Output:** Binary classification via [CLS] + linear (BCE). Tasks: pitch high/low, volume loud/soft, sentence onset, speech/non-speech.

**Temporal resolution: LOW.** Single time snapshot per forward pass — no temporal sequence modeling. PopT literally cannot do phoneme sequence decoding as designed.

**vs v12:** Our 20Hz temporal resolution (stride=10 from 200Hz HGA) with factored temporal self-attention is essential for phoneme decoding. PopT's detection tasks only require "did something happen in this window?" not "what sequence of phonemes occurred?"

### E. Lessons

**Validates v12:**
- 3D coordinates + attention works for variable electrodes across subjects
- **PE is critical** (most damaging ablation) — challenges seegnificant's null finding
- SSL pretraining dramatically helps (+0.20-0.23 AUC, 5× data efficiency)
- Modular factored approach (temporal + spatial) works with 25× fewer params than Brant
- Hold-one-out shows cross-subject generalization

**Challenges v12:**
- **No atlas/VE needed?** Self-attention + PE alone works. Evidence for A_self_attn ablation.
- **Coordinate jitter doesn't help** (σ=5mm). Our planned 2-5mm jitter may be wasteful.
- **No per-patient params and it works.** Per-patient diagonal may be insurance rather than necessity for spatial alignment. (But PopT doesn't normalize per-channel signals — BrainBERT handles that.)
- **Discriminative > reconstructive SSL** in their setting. Our temporal masking MSE may underperform.

**Import:**
1. **Channel-swap detection as auxiliary SSL loss** (MEDIUM) — swap 10% of VE representations, predict which are swapped. Teaches spatial context sensitivity.
2. **Temporal-proximity discrimination** (MEDIUM) — predict if two VE snapshots are from consecutive times. Complementary to masking.
3. **Vary ensemble size during training** (LOW — we already plan electrode subset sampling 70-100%)

**Cautionary tales:**
- Tasks are binary detection, not sequence decoding — much easier
- No temporal modeling within PopT — can't do our task
- ~167 electrodes/subject covering whole brain ≠ our ~100 focal surface electrodes
- Coordinate jitter negative result may not transfer (different coordinate system, different model)

### Key Numbers

| Metric | Value |
|--------|-------|
| Pitch ROC-AUC | 0.74 ± 0.03 |
| Speech/Non-speech ROC-AUC | 0.93 ± 0.02 |
| SSL gain vs random init | +0.20-0.23 AUC |
| Data efficiency | 5× (150 samples = 1000 baseline) |
| PE removal impact | 0.93 → 0.83 (speech), 0.74 → 0.62 (pitch) |
| Coord jitter | No improvement |
| Pretrain | LAMB lr=5e-4, batch=256, 500K steps |
| Params | ~20M (PopT alone) |
| Per-patient params | 0 |

---

## 3. Neuro-MoBRE (Wu, Di et al. 2025)

**Citation:** Wu et al., "Neuro-MoBRE: Exploring Multi-Subject Multi-Task Intracranial Decoding via Explicit Heterogeneity Resolving," arXiv:2508.04128, 2025.
**Full summary:** `pastwork/summaries/wu2025_neuro_mobre.md`
**Setup:** 11 epilepsy patients (same cohort as MIBRAIN), sEEG, 5 tasks (23-class initial, 11-class final, 4-class tone, seizure detect/predict). 1221 trials/subject for language.

### A. Architecture

**Decoder-only Transformer with Mixture of Brain Regional Experts (BrMoE).**

Brain-Regional-Temporal Tokenizer: 5 1D conv layers (filters {8,16,16,32,64}, kernels {15,7,5,3,3}, strides {7,4,3,2,2}) per channel independently → d=64 tokens. Additive embeddings: conv output + learnable temporal PE + learnable **region embedding** (one per FreeSurfer region).

BrMoE blocks (×4): Standard pre-norm MSA + residual → MoE FFN + residual. **21 expert FFNs** (= number of brain regions). Router aggregates temporal tokens per channel (sum), softmax dispatch TopK=2. CLS tokens → shared FFN; neural tokens → routed regional expert FFNs. Auxiliary load-balancing loss.

Task-Disentangled Information Aggregation (TIA): Per-task learnable CLS tokens, J=4× wider (256-dim), split into J standard d-dim tokens before MSA, merged after. Lightweight task-specific heads.

Hyperparams: 4 blocks, d=64, MLP=128, 8 heads. ~2-5M total. **ZERO per-patient params.** Scaling: Transformer-16-512 collapses from insufficient data.

**vs v12:** Both map electrodes to brain region representations. Neuro-MoBRE uses categorical region embeddings + MoE routing; v12 uses continuous MNI coordinates + distance-biased cross-attention. MoE creates shared processing pathways; VEs create shared spatial representations. v12 is more principled for multi-view reconstruction framing.

### B. SSL Objective

**3-stage:**

1. **Per-subject RMAE pretraining:** Train m independent subject-specific models. Region-structured masking (mask all tokens from same region, r=0.2). Predict frequency-domain representation (DFT magnitude + phase) → IDFT → MSE in time domain. 800 epochs, AdamW lr=5e-5, wd=0.05.

2. **Co-upcycling initialization:** Merge m pretrained models via ties-merging (Yadav 2023): prune 50% lowest-magnitude params, consensus sign → average matching params. Shared FFN and MoE FFNs randomly initialized (encourages expert specialization).

3. **Multi-task supervised fine-tuning:** 200 epochs, lr=5e-4, wd=0.1, batch=4/subject/task.

**vs v12:** Region-structured masking is their spatial equivalent of our temporal span masking. Frequency-domain target (DFT→IDFT) is a novel prediction space — more informative than raw MSE for oscillatory signals. Co-upcycling is interesting for v12's multi-stage training.

### C. Cross-Patient Anatomical Variation

**Discrete region embeddings + MoE routing.** Each sEEG electrode labeled with brain region (FreeSurfer). Learnable region embedding per region (categorical, no coordinates). MoE routing dispatches electrode tokens to specialized expert FFNs (TopK=2). **No MNI coordinates, no distance bias, no continuous spatial encoding.**

Expert count ablation: 4→8→16→21 experts, 21 (=region count) is optimal. Validates "one computational unit per brain region."

**vs v12:** Neuro-MoBRE uses discrete categorical regions where v12 uses continuous coordinates + atlas positions. v12's continuous approach is the strict generalization. Both confirm: the fundamental unit of cross-patient alignment is the brain region.

### D. Input / Output Features

**Input:** Raw broadband sEEG at 512Hz. Lowpass 200Hz, notch 50Hz + harmonics, per-channel z-score. Variable channel counts.

**Output:** 5 tasks — 23-class initial (CE), 11-class final (CE), 4-class tone (CE), seizure detection (BCE), seizure prediction (BCE). Per-task CLS + linear heads.

**vs v12:** We use HGA 200Hz (engineered), they use raw broadband (learned). Their multi-task CLS design is elegant but we're single-task. Their 23-class consonant task is closer to our 9-class phoneme than MIBRAIN's same 23-class.

### E. Lessons

**Validates v12:**
- Region-based common space works (3rd confirmation: MIBRAIN, H2DiLR, Neuro-MoBRE)
- **Small models correct for this data regime** — Transformer-16-512 collapses. Our ~171K with B=1 is well-calibrated.
- Expert count = region count is optimal — validates 16 VEs = 16 atlas ROIs
- Region-structured masking for SSL works
- Zero per-patient params still above chance in zero-shot LOSO (9/11 subjects)

**Challenges v12:**
- **No coordinates used at all** — 3rd paper in Wu group succeeding without. Our distance bias may add unnecessary complexity for sEEG.
- **No per-patient params and it works** (modestly). Combined with PopT, this is a pattern.
- **MoE routing as alternative to VE cross-attention** — both achieve "similar regions get similar processing"

**Import:**
1. **Frequency-domain SSL target** (MEDIUM) — DFT magnitude+phase → IDFT → MSE. More informative than raw temporal MSE for oscillatory signals. Unclear if useful for HGA (already envelope).
2. **Co-upcycling initialization** (LOW) — ties-merging per-subject pretrained models. Relevant if v12's per-patient diagonals are pretrained independently.
3. **Multi-task CLS tokens** (LOW for now) — TIA pattern useful if v12 ever becomes multi-task

### Key Numbers

| Metric | Value |
|--------|-------|
| Initial decoding (23-class) | 28.26% (chance 4.3%) |
| Final decoding (11-class) | 32.15% |
| Tone decoding (4-class) | 43.41% |
| Seizure detection | 89.57% acc |
| Avg improvement over baselines | +17.98% |
| Zero-shot LOSO | Above chance 9/11 subjects |
| Best baseline | H2DiLR 15.61% (initial) |
| Optimal experts | 21 (= region count) |
| Scaling collapse | Transformer-16-512 → 15.97% |

---

## 4. H2DiLR (Wu, Di et al. 2025)

**Citation:** Wu et al., "Towards Homogeneous Lexical Tone Decoding from Heterogeneous Intracranial Recordings," ICLR 2025.
**Full summary:** `pastwork/summaries/wu2025_h2dilr.md`
**Setup:** 4 epilepsy patients, sEEG, 4-class Mandarin tone, 407 syllables × 3 reps = 1221 trials/subject.

### A. Architecture

**Two-stage: VQ autoencoding → frozen decoding.**

Stage 1 (H2D): Per-subject VQ encoder (~1.55M/subject, 5-layer ConvNet, variable channels → d=256) → **dual codebook**: shared C^S (K=128, D=256) + per-subject private C^P_i (K=32/subject). Top ν tokens (closest to shared codebook) quantized via shared; rest via private. ν=0.5 optimal for task performance.

Stage 2 (ND): Freeze encoders + codebooks → lightweight Transformer decoder (4 blocks, d=128, 8 heads, ~0.958M) → FC output → 4-class tone.

Total: ~7.2M. Per-patient: **~1.55M** (entire encoder + private codebook). Orders of magnitude heavier than v12's 134/pt.

**vs v12:** Fundamentally different philosophy. H2DiLR absorbs ALL per-patient heterogeneity into heavy per-patient encoders, then uses VQ codebook routing for shared/private disentanglement. v12 uses lightweight per-patient normalization + coordinate-based spatial mapping. H2DiLR is only feasible with ~1221 trials/subject; our 46-178 trials would massively overfit 1.55M per-patient params.

### B. SSL Objective

VQ autoencoding reconstruction (MSE + commitment loss + codebook EMA). 1000 epochs, AdamW lr=5e-4, wd=0.01, batch=32. No labels needed. The H2D stage IS the self-supervised component.

**vs v12:** Completely different pretext task. VQ forces discrete codebook learning; our temporal masking forces continuous temporal prediction. VQ codebook provides interpretable structure (shared codes cluster by tone class in UMAP), but introduces codebook collapse risk and information bottleneck.

### C. Cross-Patient Anatomical Variation

**Per-subject encoders absorb everything.** No coordinates, no atlas, no PE, no spatial information at all. Each subject's ~1.55M encoder projects from variable electrode configurations into a common 256-dim latent space. The shared codebook captures homogeneous patterns; private codebook captures heterogeneous patterns.

ν partition factor controls the split: ν=0.5 optimal for task, ν=0.25 for subject classification (85.5% — the private codes ARE subject-discriminative). ν=1.0 degenerates to no private codes, drops to 40.37%.

**vs v12:** Brute-force approach. Works when you have enough trials per patient to train 1.55M-param encoders (1221 trials). Impossible in our regime (46-178 trials). v12's spatial modeling + lightweight diagonal IS the way to do this with limited data.

### D. Input / Output Features

**Input:** Raw broadband sEEG at 1000Hz (padded to 1000 samples = 1s). Variable channels/subject.

**Output:** 4-class tone (CE). Per-subject encoders + shared decoder.

### E. Lessons

**Validates v12:**
- Homogeneous neural representations DO exist across subjects for speech (shared codebook clusters by tone)
- Per-patient layers are essential (even heavier than ours — 1.55M!)
- More subjects = better (m=2→4 improves consistently)
- Two-stage (SSL then supervised) outperforms end-to-end

**Challenges v12:**
- **No coordinates and it works** (4th paper). Per-subject encoders can substitute for spatial modeling.
- **VQ discretization captures interpretable structure.** v12's continuous VE representations may be harder to interpret.

**Import:**
1. **Shared/private representation analysis** (MEDIUM for paper) — measure what fraction of VE representations cluster by patient vs phoneme. The ν diagnostic.
2. **VQ codebook as auxiliary analysis** (LOW) — post-hoc VQ on VE representations for interpretability

### Key Numbers

| Metric | Value |
|--------|-------|
| Tone accuracy (4-class) | 43.67% ± 1.78 (chance 25%) |
| Best pretrained baseline | NeuroBERT 36.94% |
| H2DiLR gain | +6.7pp over best baseline |
| Scaled-up variant | 45.95% (larger encoder + decoder) |
| Per-patient params | ~1.55M (encoder) + 8,192 (private codebook) |
| Subject classification (ν=0.25) | 85.5% |

---

## 5. Chinese uECOG BCI (2025)

**Citation:** 2025 (Science Advances). 256-ch flexible high-density ECoG, Huashan Hospital Shanghai.
**Full summary:** `pastwork/summaries/` (from 5eab8ea...pdf)
**Setup:** N=1 patient (43yo female, epilepsy), 256-ch flexible ECoG (1.3mm contacts, 3mm pitch, Neuroxess), vSMC/STG/pars opercularis, 394 Mandarin syllables × 30-60 reps, 9h over 11 days.

### A. Architecture

**Dual-stream stacked biLSTM.**

Syllable stream: All-channel HGA (1000ms, -300 to +700ms) → 4-layer stacked biLSTM (250→500→200 hidden) → FC → 394-class syllable.
Tone stream: Same architecture → 4-class tone with focal loss.

Each "stacked LSTM" = two sub-blocks of 2-layer biLSTM + LayerNorm + Dropout. Final prediction: temporal mean pooling across timesteps.

Compared: CNN-LSTM (~62%), ViT (~62%), **stacked LSTM (71.2%)** — LSTM wins.

Sentence pipeline: onset detection → syllable+tone → character dictionary lookup → 3-gram LM + beam search.

**vs v12:** Single-patient, no cross-patient mechanism. Architecture is simple stacked biLSTM — nothing shared across patients. But validates that (1) HGA from dense uECOG is sufficient for 394-class decoding, (2) biLSTM remains competitive vs transformers in small-data intracranial regime, (3) temporal mean pooling works for single-class prediction.

### B. SSL Objective

**None.** Purely supervised. 10-fold CV. Adam, ReduceLROnPlateau, mixup augmentation. 30-60 reps per syllable over 11 days is enough data without pretraining. Scaling: accuracy rises steeply from 5→20 reps, plateaus at 20+.

**vs v12:** They can afford pure supervision with ~1221 samples per syllable class. We have ~3-5 reps per phoneme — SSL is essential for us.

### C. Cross-Patient Anatomical Variation

**N=1. No cross-patient handling.** Discussion explicitly proposes v12-like future work: "construct a foundational model trained on data aggregated from multiple participants. Individual electrode locations could be mapped onto a standardized brain atlas... By incorporating anatomical information as parameters within the decoding algorithm."

They are describing v12. We are building what they propose as future work.

### D. Input / Output Features

**Input:** HGA 70-170Hz (slightly wider than our 70-150Hz). 15kHz raw → reject bad channels → CAR → downsample 400Hz → 50Hz notch → 70-170Hz Gaussian filter → Hilbert envelope → z-score. All 256 channels, no channel selection. Real-time pipeline: causal Butterworth, 50ms sliding window, fixed z-score params.

**Output:** Hierarchical: syllable (394) + tone (4) → tonal syllable → character (via dictionary) → sentence (via 3-gram LM + beam search). LM gives +11.6pp CAR (61.5% → 73.1%).

**vs v12:** Nearly identical HGA pipeline (70-170 vs 70-150Hz, CAR, Hilbert, z-score). Confirms our preprocessing is standard. Their 256-ch with no channel selection matches our approach. The LM boost (+11.6pp) is much larger than our constrained beam search over 52 tokens — we should consider larger LMs if vocabulary scales.

### E. Lessons

**Validates v12:**
- **HGA from dense uECOG = 71.2% on 394 classes.** Our signal IS sufficient for high-accuracy decoding.
- **vSMC electrodes dominate** (gradient-based saliency + fMRI validation). Our atlas positions (A6cvl, A4tl, A4hf) are the right ROIs.
- **All channels, no selection.** Model learns to weight informative channels — validates v12's include-all approach.
- **biLSTM > ViT** at this scale. Consistent with our GRU-based backbone choice.
- **Focal loss + mixup used.** Confirms our training recipe.
- **Foundation model direction explicitly proposed as next step.** We ARE the next step.

**Challenges v12:**
- **N=1 with 9h of data gets 71.2% on 394 classes.** How much will cross-patient pooling degrade single-patient performance? (MIBRAIN: initially hurts at N<6)
- **No acoustic intermediate representation.** Direct HGA→syllable works at N=1. Articulatory decomposition may be unnecessary overhead.
- **Their scaling curve (5→20 reps)** implies we are in the steep part (~3-5 reps). SSL must compensate for lack of repetitions.

**Import:**
1. **Real-time HGA pipeline** (MEDIUM for future) — causal Butterworth + 50ms window for online BCI
2. **fMRI-saliency validation** (MEDIUM for paper) — compare gradient-based electrode contributions to fMRI motor localizer

### Key Numbers

| Metric | Value |
|--------|-------|
| Syllable accuracy (394 classes) | 71.2% (chance 0.25%) |
| Tone accuracy (4 classes) | 69.1% (chance 25%) |
| Real-time CAR (neural only) | 61.5% |
| Real-time CAR (+ 3-gram LM) | 73.1% |
| Communication speed | 49.7 CPM |
| Rep scaling | 5 reps: ~20%, 20 reps: plateau ~71% |
| Array | 256-ch flexible ECoG, 1.3mm contacts, 3mm pitch |
| Training data | ~9h over 11 days |

---

## 6. BrainBERT (Wang et al. 2023)

**Citation:** Wang et al., "BrainBERT: Self-Supervised Representation Learning for Intracranial Recordings," ICLR 2023.
**Full summary:** `pastwork/summaries/wang2023_brainbert.md`
**Setup:** 10 sEEG subjects (Brain Treebank), 1688 electrodes (1249 post-Laplacian), ~43.7h, 4551 electrode-hours. 19 pretrain / 7 downstream.

### A. Architecture

**Per-electrode Transformer.** Raw voltage → STFT or superlet spectrogram (40 freq bins × time) → 6-layer Transformer encoder (d=768, 12 heads, ~28M params). Processes ONE electrode at a time — no multi-electrode model. [CLS] or averaged center embeddings → linear head for downstream binary classification.

**vs v12:** BrainBERT sidesteps the cross-patient spatial problem entirely by operating per-electrode. Useful as a temporal feature extractor (used by PopT), but cannot model inter-electrode interactions.

### B. SSL Objective

**Masked spectrogram autoencoding.** Mask time columns + frequency rows of spectrogram (p_mask=0.05, spans 1-5 steps, p_identity=0.10, p_replace=0.10). **Content-aware L1 loss:** base L1 on all masked positions + α × L1 on only masked positions where |Y| > γ. Rationale: 68% of z-scored spectrogram is near-zero; without content-aware term, model collapses to predicting zeros.

LAMB lr=1e-4, batch=256, 500K steps.

**vs v12:** Content-aware loss is directly importable. Our HGA after z-scoring will have similar sparsity during low-activity periods. We should add a content-aware MSE variant: upweight reconstruction of high-activation time bins. The identity/replace masking trick (prevent mask token shortcut) is also worth importing.

### C. Cross-Patient Anatomical Variation

**None architecturally.** All electrodes from all subjects pooled during SSL (no subject/location info). Generalization: hold-one-subject-out shows negligible degradation — the temporal dynamics of field potentials are consistent enough across brain regions and subjects for per-electrode SSL to transfer.

**vs v12:** BrainBERT works cross-subject because it avoids the spatial problem. v12 must solve the spatial problem to get multi-electrode decoding.

### D. Input / Output Features

**Input:** Raw sEEG → STFT or superlet spectrogram (0-200Hz, 40 bins, 5s windows). Laplacian re-referencing (not CAR). Z-scored per frequency bin.

**Output:** Per-electrode binary classification (ROC-AUC). Tasks: pitch, volume, sentence onset, speech/non-speech.

### E. Lessons

**Validates v12:**
- SSL on intracranial field potentials generalizes cross-subject (+0.23 AUC over random init)
- 5× data efficiency from pretraining — highly relevant for our data-limited regime
- Content-aware loss prevents collapse on sparse signals
- ~43.7h is sufficient for SSL. Our ~24h (sEEG+uECoG) is plausible.

**Challenges v12:**
- Per-electrode processing may be more robust than spatial modeling for transfer (avoids alignment errors)
- STFT beats superlet for fine-tuning (simpler representations fine-tune better)

**Import:**
1. **Content-aware reconstruction loss** (HIGH) — upweight reconstruction of high-activation bins during SSL
2. **Identity/replace masking trick** (MEDIUM) — p_identity=0.10, p_replace=0.10 alongside zeroed masks
3. **Intrinsic dimensionality diagnostic** (LOW) — measure ID of VE representations by brain region

### Key Numbers

| Metric | Value |
|--------|-------|
| Task avg ROC-AUC (STFT, fine-tuned) | 0.83 |
| SSL gain vs random init | +0.23 AUC |
| Data efficiency | 5× (150 = 1000 baseline) |
| Content-aware gain | +0.01-0.05 AUC/task |
| Hold-one-out degradation | Negligible |
| Pretrain | LAMB lr=1e-4, batch=256, 500K steps |
| Params | ~28M |

---

## 7. Brant (Zhang et al. 2023)

**Citation:** Zhang et al., "Brant: Foundation Model for Intracranial Neural Signal," NeurIPS 2023.
**Full summary:** `pastwork/summaries/zhang2023_brant.md`
**Setup:** 9 sEEG subjects (Zhejiang University), 2528h (1.01 TB), 505.69M params. 4 scale variants (69M→506M).

### A. Architecture

**Factored Transformer: temporal → spatial (sequential).**

Temporal encoder: 12-layer Transformer (d=2048, FFN=3072, 16 heads). Per-channel temporal patches → temporal representations.
Spatial encoder: 5-layer Transformer (d=2048, FFN=3072, 16 heads). All channels at each timestep.

Frequency encoding: PSD in 8 bands (theta→gamma5, 4-128Hz) → softmax-weighted band embeddings added to input. Learnable temporal PE. **NO spatial PE — channels are an anonymous bag.**

**vs v12:** Brant has the RIGHT factored attention pattern (temporal then spatial, validated by seegnificant +0.06 R²). But lacks spatial identity — channels have no coordinates, no region labels, no ordering semantics. This is why it fails for cross-patient speech.

### B. SSL Objective

MAE: 40% patches masked uniformly (random across time+space), MSE reconstruction of raw waveform. Adam, 750K updates, cyclic LR 3e-6→1e-5, 4×A100 80GB, 2.8 days.

**vs v12:** Same masked autoencoding family, but random masking (not temporal span). BIT showed temporal span > random for speech.

### C. Cross-Patient Anatomical Variation

**FAILS.** No spatial identity (channels are anonymous), no per-patient layers, no coordinates, no atlas. Works for spatially-agnostic tasks (seizure: 91.17%) but fails for speech (confirmed by MIBRAIN testing Brant as baseline — near chance).

**The anti-pattern.** Proves raw scale (505M params, 2528h) is insufficient without spatial identity mechanism. v12 exists because Brant doesn't work for speech.

### D. Input / Output Features

**Input:** Raw broadband sEEG at 250Hz. NOT HGA. Patch-based (6s windows at 250Hz).

**Output:** Signal forecasting, imputation, seizure detection (MLP on representations).

### E. Lessons

**Validates v12:**
- **Scale alone is insufficient** for cross-patient speech. Brant (505M, 2528h) AND BrainWave (~100M, 40,907h) both lack spatial identity → both fail/untested for speech. Architectural mechanism (VE cross-attention, per-patient layers) is necessary.
- Factored temporal→spatial attention IS correct pattern
- 505M params → still needs mechanism. Small-but-well-designed (171K) > large-but-generic (505M)

**Why Brant fails (5 reasons):**
1. No spatial identity — can't associate electrodes across patients
2. No per-patient layers — impedance/gain variation unmodeled
3. Raw broadband vs HGA — different biophysics
4. Random masking vs temporal span — doesn't learn speech dynamics
5. Enormous model for wrong problem — overfits to sEEG forecasting, irrelevant for phonemes

### Key Numbers

| Metric | Value |
|--------|-------|
| Params | 505.69M (largest: Brant) |
| Pre-training | 2528h, 9 subjects, 1.01 TB |
| Seizure detection | 91.17% acc |
| Cross-patient speech | Near chance (MIBRAIN baseline) |
| Training | 4×A100, 2.8 days, 750K steps |

---

## 7b. BrainWave (Yuan et al. 2025) — Brant successor

**Citation:** Yuan et al., "BrainWave: A Brain Signal Foundation Model for Clinical Applications," arXiv:2402.10251v7, 2025.
**Setup:** ~16,000 individuals, 40,907h (13.79 TB), mixed iEEG (5231h, 91 subjects) + scalp EEG (35,676h, 15,906 subjects). "First FM for both invasive and non-invasive."

### A. Architecture

RoBERTa backbone: D=768, 10 layers, 16 heads, ~90-110M params. Per-channel temporal encoding (independent) + Scale Alignment Layer (handles variable sampling rates 100-4096Hz via spectrogram normalization) + **Channel Attention** (bidirectional self-attention across channels at each timestep). Channels interact but with **NO spatial identity** — no coordinates, no channel indices, no anatomical labels. Anonymous bag of channels, same as Brant.

**ZERO per-patient params.** Fully shared model.

### B. SSL Objective

Masked spectrogram reconstruction (MAE-style). Random patch masking on time-frequency spectrograms. Raw broadband input. AdamW lr=1e-5, batch=2.56M patches, 16,600 steps, 4×A100, 100h training.

### C. Cross-Patient Anatomical Variation

**Still fails.** No coordinates, no atlas, no per-patient layers. Channel attention adds inter-channel interaction but without spatial identity — channels are anonymous. Cross-subject evaluation only on standardized EEG montages (fixed 10-20 system). Never tested on variable-electrode intracranial speech.

### D. Input / Output Features

Raw broadband → spectrograms. NOT HGA. Downstream: disease classification only (AD, seizure, depression, ADHD, sleep). **No speech, no motor, no BCI tasks.**

### E. Lessons

**Fixes ZERO of Brant's four weaknesses:**
1. No spatial identity → STILL no spatial identity
2. No per-patient layers → STILL none
3. Raw broadband only → STILL raw broadband spectrograms
4. Random masking → STILL random patch masking

**What's useful:** Joint EEG+iEEG pretraining helps (validates v12's mixed sEEG+uECoG SSL). Scale alignment for variable sampling rates is elegant engineering. 8-shot BrainWave nearly matches full-label competitors on disease classification.

**For v12 paper:** BrainWave demonstrates that scaling to 40,907h + 16K subjects + ~100M params is insufficient for cross-patient intracranial decoding without spatial identity mechanism. The Brant family's entire research direction (temporal pattern recognition at scale) is orthogonal to v12's (spatially-grounded cross-patient alignment).

---

## 7c. BrantX (KDD 2024) — Cross-modality alignment

**Citation:** BrantX, KDD 2024. Built on Brant-2 (1B params, ~4TB).
**Setup:** Cross-MODALITY alignment (EEG → EOG/ECG/EMG), NOT cross-patient. 267 subjects, 3 public sleep datasets with simultaneous EEG+EXG.

### Summary

**Completely orthogonal to v12.** BrantX adds two-level contrastive alignment (patch + sequence InfoNCE) between EEG and peripheral signals. All evaluation on standardized scalp EEG (fixed 10-20 montage). No intracranial data. No cross-patient mechanism. No spatial identity. Brant-2 backbone (1B params) used as frozen/fine-tuned encoder.

**Relevance to v12: NONE.** BrantX solves cross-modality transfer (brain→body signals), not cross-patient spatial alignment. Confirms the Brant team's research direction diverges from our problem entirely.

### Brant Family Evolution

```
Brant (NeurIPS 2023) → BrainWave (2025) → BrantX (KDD 2024) → Brant-2 (2024)
     |                       |                    |                    |
  505M, 2528h iEEG     ~100M, 40,907h         1B, ~4TB            1B, ~4TB
  MAE, sEEG only       MAE+channel attn       Cross-modality      Mixed EEG+iEEG
  No spatial ID         No spatial ID          No spatial ID        No spatial ID
  No per-patient        No per-patient         No per-patient       No per-patient
  Fails speech          Never tested speech    Not applicable       Never tested speech
```

**Bottom line:** The entire Brant family, despite scaling from 505M/2528h to 1B/40,907h, has never addressed the two mechanisms v12 is built around: (1) atlas-grounded spatial identity for variable electrode placements, and (2) per-patient normalization. Their success is on disease classification with standardized montages — a fundamentally different problem from cross-patient intracranial speech decoding.

---

## 8. Evanson "Minutes to Days" (2025)

**Citation:** Evanson et al., "From Minutes to Days: Scaling Intracranial Speech Decoding with Supervised Pretraining," Imaging Neuroscience, 2025.
**Full summary:** `pastwork/summaries/evanson2025_minutes_to_days.md`
**Setup:** 3 sEEG subjects (epilepsy monitoring, Rothschild Hospital Paris), 141-230 channels, 83-108h pretrain, 43-250 min task data.

### A. Architecture

Defossez et al. 2023 ("brainmagick") ConvNet: linear input projection → conv blocks with skip connections → Bahdanau attention → d-dim output vector. CLIP-style contrastive loss aligns brain embedding with wav2vec 2.0 audio embedding. Per-subject models (removed subject-specific and spatial attention modules).

### B. SSL Objective

**Supervised contrastive pretraining** (NOT SSL). Brain recordings aligned to ambient room audio (hospital camera microphone) via CLIP loss. 83-108h per patient from clinical monitoring (daytime only, 6:00-23:00). Audiobook task excluded from pretrain.

**Key findings:**
- Performance scales **log-linearly** with pretrain data — no plateau at 100h
- **Supervised pretrain >> SSL pretrain** (beats PopT, BrainBERT fine-tuning)
- Zero-shot fails badly (rank ~0.50) — fine-tuning essential
- Night data (23:00-6:00) provides no benefit

### C. Cross-Patient Anatomical Variation

**Single-patient only.** Explicitly acknowledges cross-patient as unsolved: "will require solving the heterogeneity in electrode implantation across patients, for example with a subject-embedding layer."

### D. Input / Output Features

**Input:** Broadband sEEG, 0.05-50Hz bandpass, 40Hz. Also tested gamma bipolar (70-120Hz Hilbert) — **improved 2/3 subjects.**
**Output:** CLIP retrieval (continuous wav2vec 2.0 embedding, not discrete classes). 3s windows.

### E. Lessons

**Validates v12:**
- Massive pretraining data helps (log-linear scaling, no plateau at 100h) — our 456 min is modest but in the right direction
- Fine-tuning essential after pretrain (no zero-shot) — matches our 3-stage plan
- Gamma/HGA improves over broadband for 2/3 subjects — validates HGA extraction
- Cross-day drift is enormous (r=0.95 decodable) — per-patient normalization essential
- Cross-patient problem explicitly identified as unsolved bottleneck

**Challenges v12:**
- **Supervised pretrain > SSL pretrain.** We lack paired audio for continuous data, so our temporal masking SSL may be suboptimal vs what's achievable with supervision.
- **Log-linear with no plateau at 100h** — our 7.6h uECoG may be too little

### Key Numbers

| Metric | Value |
|--------|-------|
| Pretrain data | 83-108h per subject |
| Scaling | Log-linear, no plateau at 100h |
| Supervised > SSL | Yes (Figure 15) |
| Gamma vs broadband | Improved 2/3 subjects |
| Zero-shot | Near chance (rank ~0.50) |

---

## 9. Neuro-BERT (Wu, Di et al. 2024)

**Citation:** Wu et al., "Neuro-BERT: Rethinking Masked Autoencoding for Self-Supervised Neurological Pretraining," arXiv:2204.12440, 2024.
**Full summary:** `pastwork/summaries/wu2024_neuro_bert.md`
**Setup:** Scalp EEG + EMG. SleepEDF (500 subjects), Epilepsy (500), Ninapro (10 subjects, 16 EMG ch). NOT intracranial.

### A. Architecture

ViT-style: 4 MSA blocks, d=128, FFN=512, ~0.798M. 1D conv patch embedding. Learnable [M] token for masked positions. Linear prediction head (discarded after SSL).

### B. SSL Objective

**Fourier Inversion Prediction (FIP).** Predict DFT magnitude + phase of masked patches → IDFT → MSE loss. Key: robust across mask ratios 10-60% (unlike spatiotemporal MSE which degrades at >10%). FIP as plug-in improves other MAE methods (+0.5-1pp). Linear decoder suffices.

**vs v12:** FIP not directly useful for HGA (already a frequency-domain feature). But the fragility of spatiotemporal MSE at high mask ratios is a caution for our planned 50% temporal masking.

### C. Cross-Patient Anatomical Variation

None. Single-dataset, mixed-subject evaluation. Fixed channel topology across subjects.

### E. Lessons

**For v12:**
- Precursor to MIBRAIN's SSL. Establishes Di Wu's masked autoencoding + biologically-motivated targets.
- **Spatiotemporal MSE fragile at high mask ratios** — verify our 50% ratio on HGA doesn't degrade
- Linear decoder matches MLP for SSL reconstruction head — keep reconstruction head simple
- Masked autoencoding > contrastive for fine-tuning transfer (but fine-tuning is essential)

---

## 10. Feng "Brain-to-Sentence" (2025)

**Citation:** Feng et al., "Acoustic Inspired Brain-to-Sentence Decoder for Logosyllabic Language," Cyborg and Bionic Systems, 2025.
**Full summary:** (from cbsystems.0257.pdf)
**Setup:** 4 epilepsy patients, sEEG (12-13 depth electrodes, 63-66 selected channels), Mandarin reading. Same group as MIBRAIN (Westlake/Zhejiang).

### A. Architecture

3-phase: (1) CNN backbone for sEEG features; (2) Three parallel decoders — initial prediction (articulatory POA/MOA/aspiration/devoice via learnable prototypes + adjacency matrix), tone prediction (neural-audio regularization with F0), final cluster prediction (NAR with formants F1-F3, k-means 11 groups); (3) 5-gram LM + Chinese-LLaMA-7B rescoring for sentence generation.

### B. SSL Objective

None. Per-subject supervised models entirely.

### C. Cross-Patient

N=4, per-patient only. No cross-patient transfer.

### D. Input / Output Features

**Input:** sEEG at 400Hz (downsampled from 2kHz). Per-subject channel selection.
**Output:** Hierarchical: initials (articulatory) + tones (F0) + final clusters (formants) → 5-gram + LLM → sentences. Best: 71% character accuracy.

### E. Lessons

**For v12:**
- **Neural-audio regularization** (NAR): using acoustic features as auxiliary training targets. We have synced audio — could use as auxiliary loss during supervised fine-tuning.
- **Subcortical contributions** (thalamus ~50% accuracy for some patients) — relevant for sEEG SSL. Brainnetome atlas in v12 is cortical-only; subcortical VEs may be needed.
- **Articulatory decomposition** via learnable prototypes — more flexible than our fixed 9×15 matrix, but v12 already found flat > articulatory for single-phoneme.

---

## 11. BarISTA (Oganesian et al. 2025)

**Citation:** Oganesian et al., "BarISTA: Brain Scale Informed Spatiotemporal Representation of Human Intracranial Neural Activity," NeurIPS 2025.
**Full summary:** `pastwork/summaries/oganesian2025_barista.md`
**Setup:** 10 sEEG subjects (Brain Treebank), 2048Hz, 29.2h pretraining. Code: github.com/ShanechiLab/BaRISTA

### A. Architecture

Channel-wise temporal patching (250ms at 2048Hz) → 5-layer dilated CNN temporal encoder → linear to d=64 → + **learnable spatial embedding** (3 scales: channel LPI coords → lookup, Destrieux parcel → lookup, DK lobe → lookup) → interleaved flat token sequence → 12-layer Transformer (d=64, 4 heads, RoPE temporal only). Combined space-time attention (NOT factored). ~1M params. **ZERO per-patient params.**

**vs v12:** Both map electrodes to brain region representations. BarISTA uses discrete learnable embedding lookup; v12 uses continuous distance-biased cross-attention to atlas positions. BarISTA processes all channels in a flat sequence; v12 compresses to 16 VEs. BarISTA's combined attention contradicts our factored design — but v12 operates on only 16 VEs after cross-attention, so the factored cost is minimal.

### B. SSL Objective

**JEPA-style masked latent token reconstruction.** Spatially-guided masking (~30%): select random spatial categories (parcels/lobes), mask ALL tokens from those categories across ALL timesteps. EMA target tokenizer (momentum 0.996). MSE in **latent space** (not observation space). Spatial-only masking — no temporal masking (acknowledged as limitation).

AdamW lr=1e-3, 70 epochs, <4h on 4×RTX 6000 Ada.

**vs v12:** We plan temporal masking with MSE in observation space. BarISTA shows spatial masking works AND latent targets work. Combined spatial+temporal masking (which neither BarISTA nor BIT does alone) could be v12's unique SSL contribution.

### C. Cross-Patient Anatomical Variation

**THE CENTRAL FINDING: Parcel-level encoding >> channel-level by +8-10pp AUC.**

Three spatial scales tested:
- **Channel-level (LPI):** Each channel's (x,y,z) discretized to integers → 3 embedding tables summed. Two nearby channels get DIFFERENT embeddings (no spatial proximity).
- **Parcel-level (Destrieux):** 1 embedding per cortical parcel. Channels in same parcel share embedding. 19-47 unique parcels/subject.
- **Lobe-level (DK):** 1 embedding per lobe. 4-9 unique lobes/subject.

Results: Parcel encode + channel mask = 0.862/0.869 vs channel encode + channel mask = 0.778/0.764. **+8.4pp/+10.5pp from coarser spatial encoding.** Lobe-level also beats channel-level but slightly worse than parcel.

Hold-out subject: 0.841/0.852 vs 0.862/0.869 when included. Only ~2pp degradation — cross-subject transfer works without per-patient layers for binary detection tasks.

**vs v12:** This is the strongest external validation of v12's VE approach. v12's 16 Brainnetome sub-gyral ROIs operate at a granularity between BarISTA's parcels and lobes. BarISTA proves that grouping electrodes by brain region IS the mechanism that enables cross-patient iEEG models. v12 extends this with soft distance-biased assignment (vs hard parcel boundaries) and per-patient normalization.

### D. Input / Output Features

**Input:** Raw broadband sEEG at 2048Hz. Laplacian re-referencing. Z-scored per 3s segment.
**Output:** Binary classification (ROC-AUC): sentence onset, speech/non-speech, volume, optical flow.

### E. Lessons

**Validates v12:**
- **Atlas-level > channel-level (+8-10pp)** — strongest possible validation of VE approach
- SSL pretraining helps substantially (+8-25pp over random init)
- Small models work (1M beats PopT's 20M and Brant's 505M)
- Cross-subject transfer feasible without per-patient layers (~2pp degradation)
- Spatial masking effective for SSL

**Challenges v12:**
- **Combined attention > factored** (0.836/0.847 vs 0.828/0.825) — contradicts seegnificant
- **No per-patient params needed** for binary detection
- **Discrete embedding lookup suffices** — v12's continuous Fourier PE + distance bias may be over-engineered for parcel-level grouping
- **No VE bottleneck needed** — flat sequence with parcel embeddings works

**Import:**
1. **JEPA-style latent reconstruction target** (HIGH) — reconstruct in latent space with EMA target encoder instead of observation-space MSE
2. **Spatial masking at VE/parcel level** (HIGH) — mask entire VE groups, not individual time bins
3. **Combined spatial+temporal masking** (HIGH) — neither BarISTA nor BIT does both; v12 could leapfrog
4. **Dilated CNN temporal encoder** (MEDIUM) — multi-scale temporal features without downsampling

### Key Numbers

| Metric | Value |
|--------|-------|
| Best config AUC (sentence/speech) | 0.862 / 0.869 |
| Parcel vs channel encoding gain | +8.4pp / +10.5pp |
| SSL vs random init gain | +8-25pp |
| Combined vs factored attention | +0.8pp / +2.2pp |
| Hold-out subject degradation | ~2pp |
| Params | ~1M |
| Training | <4h, 4×RTX 6000 Ada |
| Data | 29.2h, 10 sEEG subjects |

---

## 12. Charmander (Mahato et al. 2025)

**Citation:** Mahato et al., "A scalable self-supervised method for modeling human intracranial recordings during natural behavior," NeurIPS 2025 Workshop.
**Full summary:** `pastwork/summaries/mahato2025_charmander.md`
**Setup:** AJILE12 (12 ECoG patients, 64-106 surface + depth electrodes) + Brain Treebank (10 sEEG).

### A. Architecture

Perceiver-based (Poyo+ encoder). Temporal patches (P=5 samples) → learnable projection (d=128) + **per-channel per-participant learnable embedding** + RoPE → H=32 "virtual channels" × M=8 "virtual timesteps" latent tokens via cross-attention → 16 self-attention layers. 8M params. No coordinates. New participants: init new embeddings, 50-epoch warm-up.

**vs v12:** Converges on Perceiver bottleneck (32 latents ≈ 16 VEs). But Charmander uses purely learned embeddings (data-hungry, no atlas prior, per-patient per-channel) while v12 uses atlas-grounded positions with distance bias. Charmander's 8M→142M scaling shows NO downstream benefit — confirms v12's ~170K is right-sized.

### B. SSL Objective

50% random channel masking. Reconstruct masked channels' raw voltage from unmasked channels via cross-attention decoder. MSE loss. LAMB lr=3.125e-3, 300 epochs.

### C. Cross-Patient

Learnable per-channel per-participant embeddings. No coordinates, no atlas. New participants need 50-epoch embedding warm-up + progressive unfreezing. MP8 (8 patients) > MP3 > MP1. Data per patient matters at least as much as number of patients.

### E. Lessons

**Validates v12:**
- Perceiver bottleneck for iEEG works (independent convergence with v12)
- Model scaling doesn't help downstream — architecture > capacity
- Multi-patient pretraining helps

**Challenges v12:**
- Pure learned embeddings (no atlas) works if enough data
- 50% channel masking ratio works (validates spatial masking)

---

## 13. NDT3 (Joel Ye et al. 2025)

**Citation:** Ye et al., "A Generalist Intracortical Motor Decoder," bioRxiv, 2025.
**Full summary:** `pastwork/summaries/ye2025_ndt3.md`
**Setup:** 30+ monkeys/humans, Utah arrays + Neuropixels, 2000h spikes, motor tasks. 45M and 350M params.

### A. Architecture

Autoregressive causal Transformer. Neural data patched (32 ch × 20ms). Joint AR objective (MSE kinematics + CE spike counts). RoPE + modality embeddings. Linear readin/readout per modality. **NO per-subject layers.** Covariate dropout (M~U[0,1]) prevents teacher-forcing shortcut.

### C. Cross-Patient

**FUNDAMENTAL LIMITATION.** Cross-subject R² ~0.5 vs cross-session ~0.7. **Channel shuffle alone reduces cross-session to cross-subject level** — sensor order/identity is the binding constraint. Output stereotypy: AR fails to extrapolate to held-out reach angles. Explicitly identifies per-patient layers as needed future work.

### E. Lessons

**Validates v12:**
- **Sensor variability IS the bottleneck** for cross-subject transfer — exactly what v12's per-patient layers solve
- Channel shuffle experiment is the cleanest proof that spatial identity matters
- Per-patient layers needed (explicitly stated as future work)
- AR output stereotypy is a caution for v12's AR decoder

**Import:**
1. **Covariate dropout** (MEDIUM) — M~U[0,1] behavioral masking during AR training
2. **Channel shuffle diagnostic** (HIGH for paper) — quantify v12's spatial identity contribution

---

## 14. RPNT (Fang et al. 2026)

**Citation:** Fang et al., "RPNT: Robust Pre-trained Neural Transformer," arXiv:2601.17641, 2026.
**Full summary:** (from 2601.17641v2.pdf)
**Setup:** 4 macaques, 111 sessions, 43h spikes (LTRCH) + 1 macaque, 17 sessions Neuropixels (NPCS).

### Key Innovations for v12

1. **MRoPE** — Multidimensional rotary PE encoding site coordinates + session metadata + time as rotary embeddings on Q/K. Zero-shot generalization to unseen coordinates via rotation composition. Worth considering for v12's temporal self-attention.

2. **Uniform random masking ratio U(0,1)** — sample mask ratio from uniform per batch. Eliminates hyperparameter, consistently outperforms all fixed ratios. **Directly importable** for v12's temporal masking SSL.

3. **Cross-site contrastive loss** — InfoNCE encouraging same-site representations to cluster. +3.5pp on cross-site. Translates to: encourage same-patient VE consistency during SSL.

4. **No per-patient layers → limited cross-dataset transfer.** NPCS→LTRCH degrades substantially. Validates v12's per-patient diagonal.

---

## 15. Brain-OF (Guo et al. 2026)

**Citation:** Guo et al., "Brain-OF: An Omnifunctional Foundation Model for fMRI, EEG and MEG," arXiv:2602.23410, 2026.
**Setup:** 37 datasets, 32K participants, 5.9M samples. fMRI + EEG + MEG. 47.5M-1.7B params.

### Key Relevance

**ARNESS = v12's VE cross-attention at abstract level.** Perceiver-style cross-attention maps variable-length input to C=128 fixed learnable latent tokens (v12: 16 VEs). Independent validation of the Perceiver bottleneck for variable neural inputs. But Brain-OF's latents are purely learned; v12's are atlas-grounded.

**MTFM (dual time-frequency masking)** — adds frequency-domain reconstruction alongside temporal masking. Consistently outperforms temporal-only masking. Worth testing for v12's SSL.

**Brainnetome atlas used** for fMRI ROI extraction (246 parcels). Same atlas family as v12 (16 core ROIs). Validates atlas choice.

---

## Cross-Paper Synthesis

### Convergence Map

| Design Choice | Papers Supporting | Papers Against | v12 Confidence |
|---------------|------------------|----------------|---------------|
| Atlas/region-based common space | MIBRAIN, Neuro-MoBRE, H2DiLR, **BarISTA (+8-10pp)**, Chinese uECOG (proposes), Charmander (Perceiver) | PopT (works without atlas) | **VERY HIGH** |
| Per-patient layers critical | MIBRAIN, H2DiLR, BIT, Singh, seegnificant, Levin, **NDT3 (explicitly needed)** | PopT (zero), Neuro-MoBRE (zero), BarISTA (zero, ~2pp cost) | HIGH (cheap insurance) |
| SSL pretrain helps cross-patient | MIBRAIN, BIT, BrainBERT, PopT | — | HIGH |
| Coordinates help | PopT (PE critical), Chen 2025 | MIBRAIN (without), seegnificant (PE NS), H2DiLR (without), Neuro-MoBRE (without) | MEDIUM — test A_no_dist |
| Temporal masking > random masking | BIT | Brant (random, works for forecasting) | HIGH for speech |
| Discriminative > reconstructive SSL | PopT | BIT (reconstructive works for speech) | MEDIUM — try both |
| Small models for limited data | Neuro-MoBRE (16-512 collapses), v12 baseline | Brant (scales to 505M with 2528h) | HIGH for our regime |
| Factored attention (temporal → spatial) | seegnificant (+0.06 R²), Brant, v12 | **BarISTA (combined +1-2pp)** | MEDIUM — test both |
| Scaling initially hurts | MIBRAIN (≥6) | — | HIGH — plan for it |
| Data scales log-linearly | Evanson (no plateau at 100h) | — | HIGH |
| HGA > broadband for surface recordings | Evanson (2/3), Chinese uECOG (71.2%) | MIBRAIN (raw broadband works for sEEG) | HIGH for uECOG |

### Ideas to Import (Priority-Ranked)

| # | Idea | Source | Priority | v12 Ablation |
|---|------|--------|----------|-------------|
| 1 | VE masking SSL | MIBRAIN | HIGH | A_ve_mask |
| 2 | Content-aware reconstruction loss | BrainBERT | HIGH | SSL recipe |
| 3 | Channel-swap detection (auxiliary SSL) | PopT | MEDIUM | SSL aux loss |
| 4 | Temporal-proximity discrimination (aux SSL) | PopT | MEDIUM | SSL aux loss |
| 5 | Frequency-domain SSL target (DFT→IDFT) | Neuro-MoBRE, Neuro-BERT | MEDIUM | A_freq_ssl |
| 6 | Grad-CAM VE contributions | MIBRAIN | HIGH (paper) | Analysis |
| 7 | Neural-audio regularization | Feng 2025 | MEDIUM | Aux loss (we have audio) |
| 8 | Audio contamination check | MIBRAIN | MEDIUM | Verification |
| 9 | Identity/replace masking trick | BrainBERT | LOW | SSL recipe |
| 10 | Symmetric masking (2× efficiency) | LaBraM | LOW | SSL recipe |
| 11 | Co-upcycling initialization | Neuro-MoBRE | LOW | Multi-stage init |
| 12 | Functional collaboration viz | MIBRAIN | MEDIUM (paper) | Analysis |
| 13 | **JEPA-style latent reconstruction target** | BarISTA | **HIGH** | SSL recipe |
| 14 | **Uniform random mask ratio U(0,1)** | RPNT | **HIGH** | SSL recipe |
| 15 | Combined spatial+temporal masking | BarISTA + BIT | HIGH | A_combined_mask |
| 16 | Cross-patient contrastive loss on VEs | RPNT | MEDIUM | SSL aux loss |
| 17 | Covariate dropout for AR decoder | NDT3 | MEDIUM | Training recipe |
| 18 | Channel shuffle diagnostic | NDT3 | HIGH (paper) | Analysis |
| 19 | MTFM dual time-frequency masking | Brain-OF | MEDIUM | A_freq_mask |

### Cautionary Tales

| Lesson | Source | For v12 |
|--------|--------|---------|
| Scaling initially hurts (need ≥6) | MIBRAIN | Don't bail at N<6. Progressive training. |
| Over-parameterization → collapse | Neuro-MoBRE (T16-512) | Keep ~171K total. Don't scale up. |
| No SSL ablation → can't isolate contribution | MIBRAIN | Always include SSL-vs-scratch |
| Raw scale insufficient without mechanism | Brant (505M fails speech) | Architecture > scale for cross-patient |
| Supervised > SSL (same-subject) | Evanson, BIT Table 9 | SSL advantage is cross-patient only |
| Spatiotemporal MSE fragile at high mask | Neuro-BERT (>10% degrades) | Verify 50% temporal masking on HGA |
| Per-subject per-region encoders don't scale | MIBRAIN | Validates lightweight diagonal |
| Coordinate jitter doesn't help (PopT σ=5mm) | PopT | Test before committing to coord augmentation |
| Generic pretrained models fail cross-patient | MIBRAIN (Brant, BrainBERT baselines) | Need integration mechanism, not just pretraining |
| Log-linear scaling, no plateau at 100h | Evanson | Our ~7.6h uECoG is modest; pursue sEEG expansion |

### Per-Patient Parameter Comparison

| Paper | Per-patient params | Mechanism |
|-------|-------------------|-----------|
| v12 | **134** (128 diagonal + 6 Δ/ω) | Diagonal normalization + coordinate correction |
| PopT | **0** | 3D PE only |
| Neuro-MoBRE | **0** | Region embeddings + MoE routing |
| MIBRAIN | ~2-5K | Per-region conv banks + prototypes |
| Singh 2025 | ~Conv1D | Per-patient temporal encoder |
| H2DiLR | **~1.55M** | Entire VQ encoder + private codebook |
| Levin 2026 | ~262K | 512→512 affine + 512 bias |
| Boccato 2026 | ~66K | Freeze-adapt affine |
| Brant | **0** | None (fails for speech) |

v12's 134 per-patient params is the most lightweight approach that includes per-patient layers. The zero-per-patient approaches (PopT, Neuro-MoBRE) work but at lower absolute performance and on easier tasks.

### Architecture Family Map

```
                    Cross-Patient Mechanism
                    |
    Hard parcellation ←─────────→ Continuous coordinates
         |                              |
    MIBRAIN (21 FreeSurfer)     v12 (16 Brainnetome + MNI PE)
    Neuro-MoBRE (21 + MoE)     PopT (3D PE + self-attn)
    H2DiLR (per-pt encoder)     Chen 2025 SwinTW (MNI + ROI)
         |                              |
    Categorical region ID        Coordinate-aware spatial attention
         |                              |
    No coordinates needed        Coordinates may add value
    (cm-scale regions)           (sub-gyral, <2mm resolution)
```

### Di Wu Group Evolution (Westlake/Zhejiang)

```
Neuro-BERT (2024) → H2DiLR (ICLR 2025) → MIBRAIN (2025) → Neuro-MoBRE (Aug 2025) → Feng Brain-to-Sentence (2025)
     |                    |                    |                    |                         |
  FIP SSL            VQ codebook         Region prototypes     MoE routing              NAR acoustic
  Temporal mask      Shared/private      Hard parcellation     Soft routing              Articulatory decomp
  No cross-pt       Per-pt encoder       Per-region conv       No per-pt                 Per-pt only
  Scalp EEG         4 sEEG pts           11 sEEG pts          11 sEEG pts               4 sEEG pts
```

All from same group, same 11-patient cohort (MIBRAIN/Neuro-MoBRE), no coordinates, FreeSurfer parcellation. v12 generalizes their entire program: soft distance-biased attention subsumes hard region assignment, MNI coordinates provide finer resolution than gyrus-level labels, per-patient diagonal (134 params) scales better than per-region encoders or VQ encoders.

---

## Batch 3: Motor Decoding Foundation Models + Alignment (2026-04-07)

Papers from Dyer/Perich/Pandarinath groups (POYO family, FALCON, NoMAD) + scaling/alignment papers. Primarily spike-based motor decoding — different modality but convergent architectures.

### POYO (Azabou et al. 2023, NeurIPS) — Spike tokenization + Perceiver bottleneck

**One-line:** Per-spike tokenization with learned unit embeddings → Perceiver cross-attention (512 latents) → self-attention → output cross-attention for motor decoding.

| Axis | POYO | v12 |
|------|------|-----|
| **A. Architecture** | Individual spikes as tokens (unit_embed + RoPE timestamp). Cross-attn compresses variable M tokens → 512 latents (D=128, L=6). Delimiter tokens for absent units. ~7.4M params. Per-unit embedding (D per unit, ~12K/session for 100 units). Value rotation on RoPE. | Conv1d binning → per-patient diagonal (134/pt) → VE cross-attn (16 latents). ~171K params. Atlas-grounded positions. |
| **B. SSL** | **None.** Purely supervised (MSE on velocity). Acknowledged as limitation. | Temporal span masking + MSE reconstruction. |
| **C. Cross-Patient** | Per-unit embedding + session embedding. No coordinates. Unit identification (freeze backbone, learn embeddings, <1 min). Cross-animal R²=0.94 (fine-tuned) vs 0.90 (unit-ID only). | Atlas-grounded VE cross-attn with distance bias. Per-patient diagonal + Δ/ω. |
| **D. I/O** | Discrete spikes in, continuous 2D velocity out. | Continuous HGA in, 52 CVC/VCV tokens out. |
| **E. Lessons** | (1) Perceiver cross-attn validated at scale — architecturally convergent with VE cross-attn. (2) Value rotation importable. (3) Delimiter tokens for electrode presence/absence. (4) Gradual unfreezing (embeddings first, then backbone). (5) No coordinates works for chronic arrays but NOT for variable surgical placement. (6) Per-unit embeddings are ~100× heavier than v12's 134/pt. |

### FALCON (Karpowicz et al. 2024, NeurIPS) — Few-shot iBCI calibration benchmark

**One-line:** Benchmark establishing that 1-2 min calibration data + multi-session pretraining dominates all other approaches for iBCI stability.

| Axis | FALCON | v12 |
|------|--------|-----|
| **A. Architecture** | NDT2 Multi (transformer, multi-context), RNN with per-session affine (H2: 2-layer GRU 512 + Linear(N→N) per session ~37K/session). | Conv1d → diagonal (134/pt) → VE cross-attn → factored attn → AR decoder. |
| **B. SSL** | NDT2 uses neural reconstruction as secondary objective. | Temporal masking + MSE. |
| **C. Cross-Patient** | Same-subject cross-session (NOT cross-patient). Per-session affine layers. NoMAD/CycleGAN alignment marginal. NDT2 Multi (train on all sessions + few-shot FT) wins. | Cross-patient with VE atlas mapping. |
| **D. I/O** | Threshold crossings (20ms bins) → EMG/kinematics/characters/spectrograms. | HGA → 52 CVC/VCV phoneme sequences. |
| **E. Lessons** | (1) 1-2 min calibration sufficient — our 46-178 trials/pt is generous. (2) Deep networks catastrophically unstable without recalibration (RNN: -0.60 R²). (3) Multi-session pretrain + few-shot FT >> unsupervised alignment. (4) CORP test-time adaptation with LM pseudo-labels powerful for communication (WER 0.11) — adaptable to v12 with 52-token beam search. (5) Frame v12 results with FALCON-style evaluation taxonomy (ZS/FSU/FSS/TTA/OR). |

### Jiang et al. 2025 — Data heterogeneity limits scaling of neural transformers

**One-line:** Region-level heterogeneity kills scaling for spatial tasks; temporal prediction is robust; 5 ranked sessions > 40 random.

| Axis | Jiang 2025 | v12 |
|------|-----------|-----|
| **A. Architecture** | NDT (5-layer transformer, D=512, 12M shared). Per-session stitchers (~1.2M/session, full linear). Poisson NLL. | ~171K total, diagonal 134/pt. |
| **B. SSL** | Masked spike reconstruction (4 masking schemes: co-smoothing, forward-prediction, inter-region, intra-region). Mask ratio 0.3. | Temporal span masking, MSE, ratio 0.5. |
| **C. Cross-Patient** | Session-specific stitchers (encoding + decoding linear layers). No spatial encoding. Rankings target-specific (no session in top-5 for ALL held-out sessions). | Atlas-grounded VE cross-attn + per-patient diagonal. |
| **D. I/O** | Neuropixels spikes (20ms bins) → spike prediction / choice classification. | HGA → phoneme sequences. |
| **E. Lessons** | **(1) CRITICAL: Forward-prediction (temporal) is most robust to heterogeneity — validates temporal masking over spatial for v12's SSL.** (2) 5 ranked sessions achieve 86% of best 40-session performance (8× efficiency). For v12 SSL: don't blindly pool all 29 patients — rank and select. (3) Reverse-ranked sessions actively HURT. Per-patient loss weighting must down-weight harmful sources. (4) Rankings are target-specific: best source patients for S14 ≠ best for S26. Universal foundation model may be suboptimal vs per-target source selection. (5) Without spatial alignment mechanism, heterogeneous data doesn't scale — argument FOR v12's VE architecture. |

### NEDS (Zhang et al. 2025, ICML) — Multi-task masking for encoding + decoding at scale

**One-line:** Four masking schemes (neural, behavior, within-modal, cross-modal) create a model that simultaneously encodes and decodes, outperforming all baselines on both.

| Axis | NEDS | v12 |
|------|------|-----|
| **A. Architecture** | Encoder-only transformer (22L, D=256, 8 heads). 12M shared + ~150M session-specific (~1.86M/session: input/output matrices + session embed). Poisson + MSE + CE losses. | ~171K total, per-patient 134. |
| **B. SSL** | Multi-task masking: (1) mask all neural → predict from behavior, (2) mask all behavior → predict from neural, (3) within-modal random mask, (4) cross-modal random mask. Mask ratio 0.1. **Within-modality masking most critical for encoding** (ablation: -50% without it). | Temporal masking + MSE. |
| **C. Cross-Patient** | Per-session input/output matrices (full linear, N×D). Session embeddings essential. No spatial encoding — neuron identity purely from learned embeddings. Embeddings become 83% brain-region-predictive WITHOUT labels. | VE cross-attn + per-patient diagonal. Atlas-grounded. |
| **D. I/O** | Neuropixels spikes → spike rates + wheel speed + whisker ME + choice + block prior. | HGA → phoneme sequences. |
| **E. Lessons** | (1) Within-modality masking (temporal reconstruction) > cross-modal for encoding — validates v12's SSL focus. (2) 12M right-sized for 74 sessions; 3M for single-session; 12M overfits single — our 171K for 4-11 patients is correct regime. (3) Neuron embeddings become region-predictive without labels — diagnostic: check if v12 VE representations become functionally differentiated after SSL. (4) Session embeddings "essential" (not ablated) — validates per-patient identity mechanism. (5) Multi-task masking is powerful but requires multiple behavioral modalities; v12 has only phoneme labels (no continuous behaviors). |

### POSSM (Ryoo et al. 2025, NeurIPS) — Real-time cross-species neural decoding with SSMs

**One-line:** POYO cross-attention + SSM recurrent backbone enables real-time, cross-species transfer including monkey→human speech (PER 19.80%).

| Axis | POSSM | v12 |
|------|-------|-----|
| **A. Architecture** | Input cross-attn (variable spikes → 1 latent per 50ms chunk) → recurrent SSM (S4D/GRU/Mamba, 4L, hidden 512) → output cross-attn. 4.6-86M params. Per-unit embeddings + session embedding. | Conv1d → diagonal → VE cross-attn → factored self-attn → AR decoder. ~171K. |
| **B. SSL** | Two-phase: (1) reconstruction loss on cross-attn + embeddings, (2) CTC loss on full model. 148 sessions pretrained. | Temporal masking SSL → supervised FT. |
| **C. Cross-Patient** | Unit identification + gradual unfreezing. Cross-species: monkey→human handwriting (+2%), monkey→human speech (PER 19.80% with multi-input). | Atlas VE cross-attn + per-patient diagonal. |
| **D. I/O** | Discrete spikes (per-event) or binned counts → kinematics / characters / phonemes. For speech: 14 bins × 512 features (spike counts + spike-band power), stride 4. CTC output. | HGA → 52 CVC/VCV tokens. |
| **E. Lessons** | (1) Two-phase training (reconstruction → task loss) validated for speech — importable to v12's SSL→FT pipeline. (2) Multi-input modality (spike counts + spike-band power) improves speech (PER 30→20%) — consider multi-band HGA input for v12. (3) SSM backbone real-time capable (<6ms/chunk CPU) — v12 could use SSM instead of self-attn for temporal. (4) Cross-species transfer works — encouraging for cross-modality (sEEG→uECoG). (5) Model scaling 8M→86M helps modestly at this scale. |

### FunctionalMap (Javadzadeh et al. 2025, submitted ICLR 2026) — Learned functional embeddings for SEEG alignment

**One-line:** Contrastive learning on SEEG LFP produces 32-dim functional embeddings that outperform MNI coordinates for cross-subject masked-region reconstruction.

| Axis | FunctionalMap | v12 |
|------|--------------|-----|
| **A. Architecture** | Siamese CNN encoder (4-6 conv layers, 117K-694K) → 32-dim L2-normalized embedding per channel. Transformer (3+3 layers, D=128, 4 heads, ~1.37M) for masked reconstruction. Per-channel functional embed replaces coordinates. | VE cross-attn with Fourier PE on MNI. ~171K total. |
| **B. SSL** | Contrastive pretraining (Modified SupCon, τ=0.2) for embeddings. Masked-region reconstruction (MSE + correlation loss) for transformer. | Temporal masking + MSE. |
| **C. Cross-Patient** | **Zero per-patient params.** Functional embedding = universal channel identity. Region labels from expert clinical annotation (NOT atlas). Padding masks for variable channel counts. Functional > MNI (p<0.001). | Per-patient diagonal (134) + atlas VE positions. |
| **D. I/O** | Raw LFP (10s segments at 1kHz, then 25ms patches for transformer) → reconstructed LFP waveforms. | HGA → phoneme sequences. |
| **E. Lessons** | **(1) Functional similarity learned from neural dynamics outperforms coordinate-based alignment for reconstruction** — but for discriminative decoding, per-patient layers may still be needed. (2) Requires region labels (clinical expert annotation, not atlas) — impractical for cortical uECOG where boundaries are gradual. (3) **Complementary to v12:** functional embeddings could augment Fourier PE as hybrid channel identity (coordinates + learned functional signature). (4) Deep-brain nuclei (GPi, STN) have worse localization than cortical surface — coordinates may be more reliable for uECOG. (5) Contrastive pretraining of electrode embeddings applicable as v12 SSL auxiliary task. |

### NoMAD (Karpowicz et al. 2025, Nature Communications) — Manifold alignment via latent dynamics

**One-line:** LFADS + frozen backbone + KL divergence alignment network stabilizes BCI decoding for >200 days without behavioral labels.

| Axis | NoMAD | v12 |
|------|-------|-----|
| **A. Architecture** | LFADS (GRU-based seq VAE, 100 hidden units) + behavioral readout. Alignment: 2-layer Dense (ReLU, identity init) + low-D read-in + rates readout. ~74K alignment params per day. | Conv1d → diagonal → VE cross-attn → self-attn → AR decoder. 134/pt. |
| **B. SSL** | Day 0: supervised (Poisson NLL + behavioral MSE). Day K: unsupervised alignment (KL divergence on Generator state distributions). | Temporal masking + MSE. |
| **C. Cross-Patient** | **Within-subject only** (cross-day, same array). Single reference alignment (Day 0) — sequential alignment WORSE. Per-channel z-score normalization. | Cross-patient with atlas VE mapping. |
| **D. I/O** | 96-ch Utah spikes (50ms bins) → isometric force / hand kinematics. | HGA → phoneme sequences. |
| **E. Lessons** | (1) **Distribution matching loss importable:** KL divergence between Day 0 and Day K Generator states forces aligned manifolds. For v12: add aux loss minimizing KL between patients' VE representations and a reference distribution. (2) **Frozen backbone + trainable alignment is optimal** — validates v12's freeze-backbone-then-adapt. (3) Single reference > sequential — validates aligning all patients to atlas (not pairwise). (4) Initialize per-patient diagonal from channel statistics (per-channel mean/SD). (5) Authors explicitly note applicability to ECoG/field potentials. (6) Behavioral readout as regularizer during pretraining — add lightweight phoneme aux during SSL. |

### POYO+ (Azabou et al. 2025, ICLR) — Multi-session, multi-task across regions and cell types

**One-line:** Perceiver-IO with multi-modal decoder scales to 1335 sessions across 6 brain regions, 13 cell types, 12 tasks — diversity helps.

| Axis | POYO+ | v12 |
|------|-------|-----|
| **A. Architecture** | Perceiver (128 latents, D=128, 6L self-attn) + multi-modal decoder (task embed + session embed → shared cross-attn → task-specific linear). 12 simultaneous tasks. Per-neuron embed + session embed. | VE cross-attn (16 latents) + factored self-attn + AR decoder. |
| **B. SSL** | None — supervised multi-task. | Temporal masking + MSE. |
| **C. Cross-Patient** | Session embeddings only. No coordinates. Diversity helps: all regions > any single region (55.96% vs 54.12% best). Transfer to hippocampus works. | Atlas VE positions + per-patient diagonal. |
| **D. I/O** | Calcium imaging (regular timeseries) → 12 tasks (classification, segmentation, regression). | HGA → phoneme sequences. |
| **E. Lessons** | (1) Diversity helps even across very different sources — heterogeneous patient data should help IF architecture handles it correctly. (2) Multi-task training improves everything — auxiliary tasks could help v12. (3) Inhibitory neurons carry broadly useful info — unexpected data sources may help. (4) Session embedding is minimal but sufficient per-session identity. (5) Perceiver bottleneck consensus strengthened (now 128 latents, matching Charmander 32, v12 16). |

### Neuroformer (Antoniades et al. 2024, ICLR) — Multimodal AR with contrastive alignment

**One-line:** GPT-style AR spike prediction with CLIP-style contrastive alignment across neural/visual/behavioral modalities; pretrained 1% > unpretrained 10%.

| Axis | Neuroformer | v12 |
|------|-------------|-----|
| **A. Architecture** | GPT decoder (8L, D=512, 8 heads). Cascading cross-attention for multimodal fusion (Perceiver-style). 40-100M params. Per-neuron ID lookup. | ~171K. Perceiver cross-attn. |
| **B. SSL** | AR next-spike prediction + CLIP-style contrastive (neural↔visual↔behavioral). Contrastive consistently improves spike prediction across all datasets. τ=0.25, d_proj=1024. | Temporal masking + MSE. |
| **C. Cross-Patient** | Not addressed (single-dataset only). | Atlas VE cross-attn + per-patient layers. |
| **D. I/O** | Spike trains + video frames + behavior → predicted spikes + decoded speed/eye position. | HGA → phonemes. |
| **E. Lessons** | **(1) CLIP-style contrastive between neural and audio modalities importable:** align HGA representations with audio features (MFCCs, mel specs) from lavalier mic during SSL. Lightweight (one linear projection + softmax CE). (2) Pretrained 1% > unpretrained 10% — strong pretraining benefit. (3) AR on continuous signals is less natural than masked reconstruction — v12's temporal masking is better suited. |

### Ma et al. 2023 (eLife) — CycleGAN for BCI stability

**One-line:** Cycle-GAN adversarial alignment of cross-day neural recordings; full-dimensional > latent-space; 20 trials sufficient.

| Axis | CycleGAN BCI | v12 |
|------|-------------|-----|
| **A. Architecture** | Generator (1-hidden-layer feedforward, C neurons) + discriminator, paired for both directions. ~74K params. Static (no temporal modeling). | ~171K with temporal modeling. |
| **C. Cross-Patient** | Within-subject cross-day only. Cycle-consistency regularization. 20 trials sufficient for alignment. Full-dim alignment > latent-space (ADAN). | Cross-patient with atlas mapping. |
| **E. Lessons** | (1) 20 trials sufficient for unsupervised alignment — our 46-178 is plenty for per-patient layers. (2) Full-dimensional alignment before dimensionality reduction is better — supports v12's per-patient normalization in electrode space before VE projection. (3) Cycle-consistency as fallback if diagonal normalization insufficient. (4) MMD as alignment quality metric for VE space. |

### BrainLM (Caro et al. 2024, ICLR) — fMRI foundation model

**One-line:** MAE transformer on 6700h fMRI parcellated to AAL-424 atlas; atlas parcellation = universal cross-subject bridge.

| Axis | BrainLM | v12 |
|------|---------|-----|
| **A. Architecture** | MAE transformer (4L encoder, 2L decoder, 4 heads, 13-650M). AAL-424 atlas parcellation. Learnable 3D spatial + temporal embeddings. [CLS] token for subject summary. | VE cross-attn (16 Brainnetome). ~171K. |
| **B. SSL** | Masked patch reconstruction (MSE). Random masking (20-90%) or 2D spatiotemporal patch masking. | Temporal masking + MSE. |
| **C. Cross-Patient** | Atlas parcellation standardizes all subjects (same 424 parcels). Zero per-subject params. Robust scaling (median/IQR per parcel). Cross-dataset: UKB→HCP works. | 16 VEs + per-patient diagonal. |
| **E. Lessons** | **(1) Atlas parcellation as universal bridge validated at massive scale (61K subjects).** fMRI equivalent of v12's VE approach. Once in atlas space, cross-subject FM works. (2) Spatiotemporal patch masking (2D: parcels × time) — adaptable for v12: mask blocks of VE positions × time bins. (3) Scaling laws exist: log-linear with data and params. (4) [CLS] token for patient-level representation — could condition v12 decoder on patient identity. (5) Zero per-subject params works for fMRI (standard registration) but NOT for intracranial (per-patient layers still essential). |

### Safaie et al. 2023 (Nature) — Preserved neural dynamics across animals

**One-line:** Motor cortex latent dynamics are preserved across individuals performing the same behavior; CCA reveals shared manifold; cross-animal decoding R²≈0.86.

| Axis | Safaie 2023 | v12 |
|------|------------|-----|
| **A. Architecture** | PCA (m=10) + CCA alignment (linear, post-hoc). LSTM (2L, 300 hidden) for decoding. Not a learned model. | Learned end-to-end with VE cross-attn. |
| **C. Cross-Patient** | PCA reduces each animal to same dimensionality → CCA aligns pairwise. Linear alignment sufficient. Behavioral similarity required (r=0.89 monkeys, 0.72 mice — lower similarity = worse alignment). ~60 neurons minimum. | Atlas-grounded continuous mapping. |
| **E. Lessons** | **(1) Biological validation of cross-patient premise:** shared latent dynamics exist. THE foundational paper for why cross-patient decoding is possible. (2) Linear alignment sufficient — supports v12's diagonal normalization. (3) 10 PCs sufficient — 16 VEs should be generous. (4) Behavioral similarity required — verify our patients' speech behaviors are sufficiently similar. (5) "Species-wide neural landscape" = v12's "multi-view reconstruction" metaphor. (6) Motor reaching only (stereotyped); speech is faster, categorical, less stereotyped. (7) Post-hoc analysis, not a learned model — can't be used for real-time inference. |

---

## Cross-Paper Synthesis: Batch 3

### Dyer/Perich/Pandarinath Ecosystem (POYO → POYO+ → NEDS → POSSM → FALCON → NoMAD)

```
POYO (NeurIPS 2023) ──→ POYO+ (ICLR 2025) ──→ NEDS (ICML 2025)
  Spike tokenization       Multi-task decoder       Multi-task masking
  Perceiver bottleneck     12 simultaneous tasks     Encoding + decoding
  Supervised only          Session + task embeds     4 masking schemes
  7 monkeys, reaching      256 animals, visual       73 mice, all IBL tasks
       |
       └──→ POSSM (NeurIPS 2025)
              + SSM backbone
              Cross-species transfer
              Speech PER 19.80%
              Real-time (<6ms)

FALCON (NeurIPS 2024) ◄── NoMAD (NatComms 2025)
  Benchmark                 LFADS + KL alignment
  1-2 min calibration       Frozen backbone + align
  NDT2 Multi wins           R²=0.91, 208-day half-life
  CORP for communication    Within-subject only
```

**Common thread:** All use per-session learned embeddings (unit + session) with NO spatial coordinates. Works for chronic arrays with trackable neurons. **Fundamental mismatch with our setting:** intra-op uECOG has no neuron-level correspondence across patients and no chronic recordings. v12's atlas-grounded spatial mechanism fills exactly this gap.

### Architecture Convergence Table (updated)

| Paper | Bottleneck | Latent count | Spatial encoding | Per-session params | Total params |
|-------|-----------|-------------|-----------------|-------------------|-------------|
| **v12** | VE cross-attn | 16 | Atlas + MNI PE + distance bias | 134 diagonal | ~171K |
| POYO | Perceiver cross-attn | 512 | None (unit embeds) | ~12K/session | ~7.4M |
| POYO+ | Perceiver cross-attn | 128 | None (unit embeds) | ~1K/session | ~3-7M |
| NEDS | Linear projection | N/A (direct) | None (input matrix) | ~1.86M/session | ~150M |
| POSSM | Cross-attn → SSM | 1 per 50ms | None (unit embeds) | ~1K/session | ~4.6-86M |
| Charmander | Perceiver cross-attn | 32 | None (channel embeds) | ~5K/pt | ~8-142M |
| BarISTA | Parcel embedding | parcels | Destrieux atlas | 0 | ~1M |
| FunctionalMap | Padding masks | N/A | Functional embeddings (32D) | 0 | ~2M |
| BrainLM | Atlas parcellation | 424 | AAL-424 + 3D learned | 0 | 13-650M |
| NoMAD | LFADS generator | 100 | None | ~74K/day | ~200K |

### Import Priority Table (updated with Batch 3)

| Idea | Source | Priority | Effort | Expected Impact |
|------|--------|----------|--------|----------------|
| Content-aware reconstruction loss | BrainBERT | HIGH | Low | Prevents SSL collapse on sparse HGA |
| VE masking SSL | MIBRAIN | HIGH | Low | Complementary to temporal masking |
| JEPA latent reconstruction target | BarISTA | HIGH | Medium | Better than raw MSE |
| Uniform random mask ratio U(0,1) | RPNT | HIGH | Trivial | Free improvement over fixed ratio |
| Combined spatial+temporal masking | BarISTA + BIT | HIGH | Low | Neither paper does both |
| **Session/patient ranking for SSL data selection** | **Jiang 2025** | **HIGH** | **Medium** | **8× data efficiency; avoids harmful sources** |
| **Distribution matching loss (KL) on VE representations** | **NoMAD** | **HIGH** | **Low** | **Forces common space to be actually common** |
| **Two-phase training (reconstruction → task loss)** | **POSSM** | **HIGH** | **Low** | **Validated for speech CTC** |
| **CLIP-style neural-audio contrastive during SSL** | **Neuroformer** | **MEDIUM-HIGH** | **Medium** | **Paired audio exists; alignments neural→acoustic** |
| **Initialize diagonal from channel statistics** | **NoMAD** | **MEDIUM** | **Trivial** | **Better than random init** |
| Channel-swap detection auxiliary SSL | PopT | MEDIUM | Medium | Binary; may not help with temporal |
| Cross-patient contrastive loss on VEs | RPNT | MEDIUM | Medium | InfoNCE on overlapping pairs |
| Frequency-domain SSL target | Neuro-MoBRE | MEDIUM | Medium | Alternative to raw MSE |
| Neural-audio regularization | Feng | MEDIUM | High | Needs audio alignment |
| **Functional embeddings as channel identity** | **FunctionalMap** | **MEDIUM** | **High** | **Augments PE; needs contrastive pretraining stage** |
| **Delimiter tokens for electrode presence/absence** | **POYO** | **LOW-MEDIUM** | **Low** | **Explicit signal for missing electrodes in VE cross-attn** |
| **Value rotation in RoPE** | **POYO** | **LOW** | **Low** | **Marginal improvement; only relevant if using RoPE** |
| **Cycle-consistency as alignment fallback** | **Ma 2023** | **LOW** | **Medium** | **Only if diagonal normalization insufficient** |

### Key Negative Results from Batch 3

1. **Heterogeneity breaks scaling for spatial tasks** (Jiang 2025): Don't blindly pool all 29 patients for SSL. Forward-prediction scales; co-smoothing doesn't. Temporal masking SSL is the right choice.

2. **Unsupervised manifold alignment is marginal** (FALCON): NoMAD/CycleGAN improve over static decoders but far from oracle. Multi-session pretrain + few-shot FT is the winning recipe. Validates v12's SSL pretrain → supervised FT approach over unsupervised alignment.

3. **Cross-species transfer has limits** (POSSM): monkey→human speech works (PER 19.80%) but requires massive pretraining data (670M spikes). No analogous abundant source exists for uECOG HGA.

4. **NoMAD is within-subject only** (Karpowicz 2025): Manifold alignment assumes same underlying dynamics. Cross-patient variation is fundamentally harder — different subjects may have different manifold structures.

5. **No spatial encoding in any Dyer/Perich group paper**: Works for chronic arrays with trackable neurons. Fails for variable surgical placement with no neuron correspondence. THE gap v12 fills.

### The Scaling Debate: Resolved

| Paper | Finding | Implication |
|-------|---------|------------|
| Evanson 2025 | Log-linear, no plateau at 100h | More data helps |
| Brant family | 505M→1B, 40K+h, fails speech | Scale alone insufficient |
| NDT3 | 350M, 2000h, R²=0.5 cross-subject | Sensor variability = bottleneck |
| **Jiang 2025** | **Heterogeneity kills spatial scaling** | **Data selection > data quantity** |
| **NEDS** | **12M right for 74 sessions; 3M for single** | **Right-size model to data** |
| **POYO+** | **Diversity helps with Perceiver bottleneck** | **IF architecture handles heterogeneity** |

**Resolution:** Scaling helps IF AND ONLY IF the architecture handles heterogeneity. Atlas-grounded spatial mechanism (VE cross-attn) + per-patient normalization are prerequisites. Without them, more data can actively hurt. With them, log-linear scaling should hold. v12 is designed for exactly this.

### One-Sentence Batch Summary

"The Dyer/Perich/Pandarinath motor decoding ecosystem (POYO→POSSM→NEDS→FALCON→NoMAD, 2023-2025) independently converged on Perceiver cross-attention bottleneck + per-session embeddings + gradual unfreezing as the canonical architecture for multi-session neural decoding, while Jiang 2025 proved that data heterogeneity limits scaling without explicit alignment mechanisms — validating every design choice in v12's atlas-grounded VE cross-attention."
