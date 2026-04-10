# Reading List — Cross-Patient uECOG Speech Decoding

Read in order. Each paper teaches something the next one builds on. (33 papers)

---

## Tier 1 — Read before touching code

### 1. Spalding 2025 — Your baseline
`pastwork/summaries/spalding2025_cross_patient_uecog.md` · [bioRxiv](https://doi.org/10.1101/2025.08.21.671516)

Your data, your task, your number to beat. 8 patients, 9 phonemes, 128/256-ch uECOG, ±500ms window, HGA preprocessing pipeline. PCA+CCA alignment → bagged SVM → 0.31 balanced accuracy. Unaligned pooling hurts (0.19). TME surrogate control validates alignment isn't artificial.

**You need this to understand:** the evaluation protocol (LOPO + 20-fold CV), the three-condition comparison structure, the spatial resolution analysis (cross-patient only works below 3mm pitch), and the specific preprocessing pipeline you must preserve.

### 2. Duraivel 2023 — Your hardware and signal
`pastwork/summaries/duraivel2023_uecog_speech_decoding.md` · [Nature Comms](https://doi.org/10.1038/s41467-023-42555-1)

The foundational uECOG paper. Array specs (8×16 at 1.33mm, 12×22 at 1.72mm, 200μm contacts), HGA extraction pipeline (8 log-spaced Gaussian bands 70–150Hz → Hilbert → 200Hz), channel selection (HG-ESNR permutation test), and single-patient baselines (71% 4-way vowel, 50% 9-way phoneme). Adjacent electrodes at <2mm carry distinct information. Confusion errors track Chomsky-Halle phonological distance.

**You need this to understand:** why uECOG grid structure is exploitable (2D conv motivation), the preprocessing pipeline you cannot change, the per-patient channel count variation (63–149 significant channels), and why articulatory auxiliary loss is grounded in the data.

### 3. Duraivel 2025 — Your cross-task data source
`pastwork/summaries/duraivel2025_speech_planning_execution.md` · [bioRxiv](https://doi.org/10.1101/2024.10.07.617122)

Pseudo-word repetition task (CVC/VCV, 52 tokens, **same 9 phonemes** as Spalding). 52 epilepsy patients + 3 intra-op uECOG patients. The 3 uECOG patients overlap with Spalding (S8, S5, S1/S2). Reveals planning→execution neural dynamics: syllable codes appear 250–350ms before phoneme codes in prefrontal regions; anterior-to-posterior spatial gradient on uECOG arrays; motor cortex encodes phonotactic transitions (not just discrete phonemes).

**You need this to understand:** the cross-task pooling opportunity (same phonemes, overlapping patients, ~200 additional trials/pt), why 2D conv on uECOG grids is motivated (spatial planning→execution gradient), why CTC is the right decoder (motor cortex encodes continuous transitions), and the syllable-level coding that supports hierarchical CTC as a Phase 3 ablation.

### 4. Singh 2025 — Your training protocol
`pastwork/summaries/singh2025_cross_subject_seeg.md` · [Nature Comms](https://doi.org/10.1038/s41467-025-63825-0)

Defines everything about how training works. Per-patient Conv1D → shared BiLSTM 2×64 → per-patient readout. Three transfer modes tested — **Mode 3 "Recurrent Transfer" (freeze LSTM, train Conv1D + readout) is best.** Group model: PER 0.49 vs single 0.57 (25 sEEG patients). Coverage correlation predicts transfer success. Pre-training 500 epochs, fine-tuning 100 epochs.

**You need this to understand:** the two-stage LOPO protocol (Stage 1: train backbone on sources; Stage 2: freeze backbone, adapt read-in + readout on target), why H=64 is a defensible backbone size, and why no explicit alignment loss is needed (alignment emerges from joint training with per-patient layers).

### 5. Boccato 2026 — Your closest architecture
`pastwork/summaries/boccato2026_cross_subject_decoding.md` · [bioRxiv](https://doi.org/10.64898/2026.02.27.708564)

Per-patient affine → shared hierarchical GRU (d=2048) → CTC. Joint training on 2 Utah array patients. Transforms are **predominantly diagonal** (diagonal ratio ~0.9) — cross-patient variation is mostly per-channel gain. Hierarchical CTC with feedback (λ=0.3) outperforms plain CTC. Freeze-and-adapt: only ~66k params needed for a new patient. PER 16.1%.

**You need this to understand:** the affine transform analysis diagnostics (diagonal ratio, condition number, Frobenius distance — the exact metrics in Phase 2 post-convergence analysis), why hierarchical CTC is a Phase 3 ablation, and the important caveat that "predominantly diagonal" was found on Utah arrays with fixed placement (may not hold for uECOG).

---

## Tier 2 — Read before finalizing Phase 2 decisions

### 6. Levin 2026 — Transfer failure modes
`pastwork/summaries/levin2026_cross_brain_transfer.md` · [bioRxiv](https://doi.org/10.64898/2026.01.12.699110)

The most important negative result. Cross-brain transfer **fails without per-session input layers** — shared input makes things worse. Transfer helps only below ~200 sentences. For some patients, permuted same-user data outperforms cross-brain data, meaning neural representations genuinely differ across people. Source replay (30% source batches during fine-tuning) prevents forgetting.

**You need this to understand:** why the per-patient read-in layer is non-negotiable, why gains from cross-patient pooling are modest (not transformative), and the electrode-permutation control as a diagnostic.

### 7. Zhang 2026 (BIT) — Supervised vs SSL cross-subject
`pastwork/summaries/zhang2026_BIT_foundation_model.md`

Resolves an apparent contradiction: supervised cross-subject **pretraining** (pretrain backbone → fine-tune on target) **degrades** performance, but SSL cross-subject pretraining **helps**. Supervised pretraining creates conflicting gradients from misaligned patients. SSL learns shared signal structure without label conflicts.

**You need this to understand:** why our supervised joint training works despite BIT's warning — joint training with per-patient layers filters variation before the backbone sees it (backbone never sees raw misaligned data). This is the "critical nuance" that justifies the entire approach.

---

## Tier 3 — Read before Phase 3 ablations

### 8. Chen 2024 — Spatial processing on ECoG grids
`pastwork/summaries/chen2024_neural_speech_synthesis.md` · [Nature Machine Intelligence](https://doi.org/10.1038/s42256-024-00824-8)

3D ResNet vs LSTM on standard ECoG grids across N=48 patients: PCC 0.806 vs 0.745. Proves spatial convolution captures information temporal-only models miss. 18-parameter speech intermediate representation (pitch, formants, voicing, loudness). Causal ≈ non-causal. Right hemisphere ≈ left. Low-density grids suffice.

**You need this for:** Phase 3.1 (2D conv input layer ablation). Also relevant: the interpretable speech parameter representation as an alternative decoding target.

### 9. Chen 2025 (SwinTW) — Coordinate-based cross-patient ECoG speech
`pastwork/summaries/chen2025_swinTW_multisubject.md` · [J. Neural Eng.](https://doi.org/10.1088/1741-2552/ada741)

Follow-up to Chen 2024. Replaces grid-dependent architectures with SwinTW: individual electrode tokenization using MNI coordinates + ROI index. Eliminates per-patient layers entirely. A single 15-patient model matches 15 individuals (PCC 0.837 vs 0.831, NS). LOO on unseen patients: PCC 0.765. Also works sEEG-only (PCC 0.798, N=9). Same 18-parameter speech target.

**You need this for:** Phase 3.1 (coordinate PE variant) and Phase 3 spatial architecture decisions. The **untested hybrid** — coordinate PE + per-patient read-in — is a strong novel contribution candidate. Key caveat: validated on 1cm-spaced standard ECoG, not uECOG (<2mm). At uECOG resolution, coordinates may matter less than local spatial structure.

### 10. Mentzelopoulos 2024 (seegnificant) — Multi-subject sEEG with PE ablation
`pastwork/summaries/mentzelopoulos2024_seegnificant.md` · [NeurIPS 2024](https://gmentz.github.io/seegnificant)

First scalable multi-subject sEEG decoder. Per-electrode Conv tokenizer → factored temporal + spatial self-attention → MLP compress → per-subject regression heads. RBF PE on MNI coordinates. Multi-subject R²=0.39 vs single R²=0.30. **Critical ablation: per-subject heads are #1 (ΔR²=-0.18), spatial attention #2 (ΔR²=-0.10), PE barely helps (ΔR²=-0.02, p=0.73).** Factored > joint 2D (+0.06 R², 5.5× faster). 21 subjects, 3–28 electrodes each, reaction time regression.

**You need this to understand:** why per-patient capacity matters (A15/A16/A17 ablations), why spatial self-attention among real electrodes is a strong alternative to VE cross-attention (A14), and why coordinate PE should be treated as uncertain until empirically validated on our data. The sEEG vs uECOG topology difference (sparse 3D vs dense 2D grid) is a fundamental caveat.

### 11. Wu 2025 — Articulatory features as alignment targets
`pastwork/summaries/wu2025_articulatory_reconstruction.md`

Articulatory features extracted via TCA from EMA data, reconstructed from HD-ECoG with PCC 0.75–0.80 across 8 speakers. Articulatory dynamics are cross-speaker invariant because all humans share the same articulators.

**You need this for:** Phase 3.2 (articulatory auxiliary loss). The theoretical basis for predicting Chomsky-Halle phonological feature vectors from GRU hidden states as a regularizer toward cross-patient-invariant representations.

### 12. Wu, Di et al. 2025 (MIBRAIN) — Closest prior: atlas-grounded cross-patient sEEG decoding
`pastwork/summaries/MIBRAIN_2025.md` · [arXiv:2506.12055](https://arxiv.org/abs/2506.12055)

Independent convergence on our core idea. 11 epilepsy patients, sEEG, 23 Mandarin consonants. Maps variable electrode placements into 21 FreeSurfer gyral regions via hard parcellation, substitutes learnable prototype tokens for missing regions, applies masked region autoencoding (spatial SSL), then supervised fine-tuning with region attention encoder + ToMe merging. Multi-sub beats single-sub by 5-8%. Per-subject per-region Conv1D encoders (heavy per-patient params). NO coordinates used — hard anatomical region labels only. Scaling: adding 1-3 subjects initially HURTS; need ≥6 to see gains. Zero-shot to unseen subjects via majority voting (hacky, acknowledged as limitation). Baselines (Brant, BrainBERT) barely exceed chance.

**You need this to understand:** (1) Independent validation that atlas-grounded common spaces enable cross-patient intracranial decoding — confirms v12's VE cross-attention approach. (2) Their spatial region masking SSL is complementary to our temporal masking — we should consider VE masking as an additional pretext task. (3) Coordinates may not be needed (they succeed without them) — but their gyrus-level regions (cm-scale) are coarser than our sub-gyral Brainnetome ROIs, so coordinates may still add value at finer granularity. (4) Scaling initially hurts before helping — plan for this with N=10. (5) v12 advantages: soft distance-based assignment (vs hard parcellation), lightweight per-patient params (134 vs ~2-5K), cleaner new-patient handling (LP-FT vs majority voting), sequence decoding (AR vs single-class), per-electrode uncertainty weighting.

---

## Tier 4 — Contextual papers (motor decoding FMs, alignment, scaling)

### 22. Safaie 2023 (Nature) — Biological validation: preserved dynamics across animals
`pastwork/summaries/safaie2023_preserved_dynamics.md` · [Nature](https://doi.org/10.1038/s41586-023-06714-0)

THE foundational neuroscience paper for why cross-patient decoding is possible. PCA+CCA alignment reveals that motor cortex latent dynamics are preserved across monkeys (3) and mice (4) performing the same behavior. Cross-animal LSTM R²≈0.86 (aligned) vs 0.02 (unaligned). Linear alignment sufficient. ~60 neurons minimum. Behavioral similarity required (r=0.89 monkeys vs 0.72 mice → lower similarity = worse alignment).

**You need this to understand:** The biological premise for v12's cross-patient approach. v12's "multi-view reconstruction" metaphor maps directly to their "species-wide neural landscape." 10 PCs sufficient for motor cortex → 16 VEs generous for speech. Caveat: motor reaching is highly stereotyped; speech is faster and categorical.

### 23. Jiang 2025 — Data heterogeneity limits scaling of neural transformers
`pastwork/summaries/jiang2025_heterogeneity_scaling.md` · Preprint

**CRITICAL for SSL strategy.** Region-level heterogeneity kills scaling for spatial tasks (co-smoothing flat in BWM). Forward-prediction (temporal) is most robust to heterogeneity. Session ranking: 5 carefully selected > 40 random (8× data efficiency). Rankings are target-specific (best sources for S14 ≠ best for S26). NDT architecture, IBL Neuropixels, 84 sessions.

**You need this to understand:** (1) Temporal masking SSL is the right choice — temporal tasks scale despite heterogeneity. (2) Don't blindly pool all 29 patients for SSL — implement patient ranking. (3) Without spatial alignment mechanisms, heterogeneous data doesn't scale — argument FOR v12's VE architecture.

### 24. FALCON (Karpowicz 2024, NeurIPS) — Few-shot iBCI calibration benchmark
`pastwork/summaries/karpowicz2024_falcon.md` · [bioRxiv](https://doi.org/10.1101/2024.09.15.613126)

Benchmark for iBCI recalibration with 1-2 min calibration budget. NDT2 Multi (multi-session pretrain + few-shot FT) dominates movement tasks. CORP test-time adaptation with LM pseudo-labels wins communication (WER 0.11). Deep networks catastrophically unstable without per-session layers (RNN: -0.60 R² zero-shot). Unsupervised alignment (NoMAD, CycleGAN) marginal.

**You need this to understand:** (1) 1-2 min calibration sufficient — our 46-178 trials/pt is generous. (2) Multi-session pretrain + few-shot FT is the winning recipe (validates v12's SSL→FT pipeline). (3) CORP-style test-time adaptation with 52-token beam search could enable online v12 calibration. (4) Frame results with FALCON evaluation taxonomy (ZS/FSU/FSS/TTA/OR).

### 25. POYO (Azabou 2023, NeurIPS) — Perceiver bottleneck for spike populations
`pastwork/summaries/azabou2023_poyo.md` · [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/PoYo)

The paper that established Perceiver cross-attention as the canonical bottleneck for variable neural populations. Per-spike tokenization + 512 latents + RoPE. 178 sessions, 7 monkeys. Unit identification (<1% params, <1 min) for transfer. Gradual unfreezing. No coordinates, no SSL (supervised only). Value rotation in attention.

**You need this to understand:** Architectural convergence with v12's VE cross-attention. Gradual unfreezing protocol. Delimiter tokens for absent units. But: no spatial encoding, spike-specific tokenization, chronic arrays only.

### 26. FunctionalMap (Javadzadeh 2025) — Learned functional embeddings beat coordinates for SEEG
`pastwork/summaries/javadzadeh2025_functionalmap.md` · Submitted ICLR 2026

Contrastive learning on SEEG LFP produces 32-dim functional embeddings that outperform MNI coordinates for masked-region reconstruction (p<0.001). Zero per-patient params. 20 subjects, basal ganglia/thalamus. Requires expert region labels.

**You need this to understand:** (1) The strongest argument against pure coordinate-based alignment — functional similarity ≠ anatomical proximity. (2) Complementary to v12: functional embeddings could augment Fourier PE. (3) Requires region labels impractical for cortical uECOG. (4) Deep brain nuclei have worse localization than cortical surface.

### 27. NEDS (Zhang 2025, ICML) — Multi-task masking for encoding + decoding at scale
`pastwork/summaries/zhang2025_neds.md`

Four masking schemes (mask neural, mask behavior, within-modal random, cross-modal random) create the first model that does both encoding and decoding at SOTA. Within-modality masking most critical for encoding (-50% without it). 12M shared + ~1.86M/session. Neuron embeddings become 83% brain-region-predictive without labels. 74 sessions, 73 mice.

**You need this to understand:** Within-modality (temporal) masking > cross-modal for encoding — validates v12's SSL focus. Model right-sizing: 12M for 74 sessions, 3M for single-session — our 171K for 4-11 patients is the right regime.

### 28. POSSM (Ryoo 2025, NeurIPS) — Cross-species real-time decoding with SSMs
`pastwork/summaries/ryoo2025_possm.md`

POYO + SSM hybrid. Cross-species transfer: monkey→human handwriting (+2%), monkey→human speech (PER 19.80% with multi-input). Two-phase training (reconstruction→CTC). Real-time (<6ms CPU). 148 sessions pretrained.

**You need this to understand:** (1) Two-phase training validated for speech. (2) Multi-input modality (spike counts + spike-band power) helps speech — consider multi-band HGA. (3) SSM backbone alternative to self-attention for temporal processing.

### 29. NoMAD (Karpowicz 2025, Nature Comms) — Manifold alignment via latent dynamics
`pastwork/summaries/karpowicz2025_nomad.md` · [Nature Comms](https://doi.org/10.1038/s41467-025-59652-y)

LFADS + KL divergence alignment stabilizes BCI for >200 days without behavioral labels. Frozen backbone + lightweight alignment network (~74K/day). Single reference > sequential alignment. Within-subject only.

**You need this to understand:** (1) Distribution matching loss (KL divergence) importable for v12's VE space. (2) Single reference alignment validates mapping all patients to atlas. (3) Initialize per-patient diagonal from channel statistics. (4) Authors note applicability to ECoG/field potentials.

---

## Tier 5 — Background papers (read for breadth)

### 30. POYO+ (Azabou 2025, ICLR) — Multi-region, multi-task Perceiver at scale
`pastwork/summaries/azabou2025_poyo_plus.md`
1335 sessions, 256 animals, 12 simultaneous tasks across 6 visual regions and 13 cell types. Diversity helps (all regions > any single). Calcium imaging. Session + task embeddings.

### 31. Neuroformer (Antoniades 2024, ICLR) — Multimodal AR with CLIP contrastive
`pastwork/summaries/antoniades2024_neuroformer.md`
GPT-style AR spike prediction with CLIP-style contrastive across neural/visual/behavioral. Contrastive consistently improves representation quality. 40-100M params. Neural-audio contrastive is importable for v12 SSL.

### 32. Ma 2023 (eLife) — CycleGAN for cross-day BCI stability
`pastwork/summaries/ma2023_cyclegan_bci.md`
Adversarial alignment for cross-day neural drift. Full-dimensional > latent-space. 20 trials sufficient. Cycle-consistency as alignment regularizer. Within-subject only.

### 33. BrainLM (Caro 2024, ICLR) — fMRI foundation model with atlas parcellation
`pastwork/summaries/caro2024_brainlm.md`
MAE transformer on 6700h fMRI parcellated to AAL-424 atlas. Validates atlas parcellation as universal cross-subject bridge at massive scale. Scaling laws log-linear. Different modality (fMRI, ~1Hz, whole-brain).

### 13. Chau 2025 (Population Transformer) — Self-attention + 3D PE for variable electrodes
`pastwork/summaries/chau2025_population_transformer.md` · ICLR 2025

Self-attention over variable electrode ensembles using 3D sinusoidal PE on brain coordinates. 10 sEEG subjects, ~55.5h total, ~167 electrodes/subject. Frozen temporal encoder (BrainBERT) → 6-layer Transformer (d=512, 8 heads) + [CLS] token. ZERO per-patient params. Two discriminative SSL objectives (ensemble-wise temporal proximity + channel-swap detection) strongly outperform reconstructive. PE removal is the most damaging ablation (0.93→0.83). ~20M params, binary detection tasks only. Coordinate jitter (σ=5mm) didn't help.

**You need this to understand:** (1) Self-attention + 3D PE alone can work for cross-patient spatial alignment — provides evidence for v12's A_self_attn ablation. (2) PE is MORE critical than PopT's SSL losses — contradicts seegnificant's PE-null finding (but seegnificant tested PE on top of spatial self-attn). (3) Discriminative SSL >> reconstructive in their setting — caution for our MSE temporal masking plan (but BIT shows reconstructive works for speech). (4) No temporal sequence modeling — cannot do phoneme decoding. Binary detection is much easier than our task.

### 14. Neuro-MoBRE (Wu, Di et al. 2025) — MoE brain regional experts for multi-task decoding
`pastwork/summaries/wu2025_neuro_mobre.md` · arXiv:2508.04128

MIBRAIN evolution: same 11 patients. Decoder-only Transformer (4L, d=64) with MoE (21 regional expert FFNs, TopK=2 routing). Region-structured masking SSL with frequency-domain target (DFT→IDFT→MSE). Co-upcycling initialization (ties-merging per-subject pretrained models). Zero per-patient params. 28.26% initial (23-class, chance 4.3%), 43.41% tone. Transformer-16-512 collapses — small models correct for data-limited regime.

**You need this to understand:** (1) Region count = optimal expert count — validates 16 VEs = 16 ROIs. (2) Over-parameterization collapses at this data scale. (3) Frequency-domain SSL target is an alternative to raw MSE. (4) Zero per-patient params still works modestly but LOSO is near-chance for some subjects.

### 15. H2DiLR (Wu, Di et al. 2025) — VQ codebook disentanglement
`pastwork/summaries/wu2025_h2dilr.md` · ICLR 2025

Same group as MIBRAIN. 4 sEEG patients, 4-class Mandarin tone. Per-subject VQ encoders (~1.55M/pt!) → shared + private codebooks → frozen Transformer decoder. 43.67% tone (chance 25%). Proves shared neural representations exist across subjects (shared codes cluster by tone in UMAP). But massively over-parameterized per patient — only feasible with 1221 trials/pt (we have 46-178).

**You need this to understand:** VQ disentanglement is conceptually elegant but impractical for our data regime. The shared/private split diagnostic (ν factor) is useful analysis. v12's 134 per-patient params is the right scale.

### 16. BrainBERT (Wang et al. 2023) — Per-electrode intracranial SSL
`pastwork/summaries/wang2023_brainbert.md` · ICLR 2023

Per-electrode Transformer (d=768, 6L, ~28M) operating on STFT spectrograms from single sEEG electrodes. Masked spectrogram autoencoding with **content-aware L1 loss** (upweights reconstruction of non-zero bins — 68% of z-scored spectrogram is near-zero). ~43.7h pretraining. +0.23 AUC over random init. 5× data efficiency. Hold-one-out shows cross-subject transfer works.

**You need this to understand:** (1) Content-aware loss prevents collapse on sparse signals — directly importable for v12's SSL. (2) Per-electrode SSL on field potentials generalizes cross-subject. (3) Sidesteps spatial problem entirely (no multi-electrode model) — fails as baseline in MIBRAIN because it can't model inter-electrode interactions.

### 17. Brant (Zhang et al. 2023) — Intracranial foundation model (505M, fails cross-patient speech)
`pastwork/summaries/zhang2023_brant.md` · NeurIPS 2023

Factored Transformer (temporal 12L + spatial 5L, d=2048, 505M params) pretrained on 2528h sEEG with MAE. SOTA on forecasting/seizure detection. **Fails for cross-patient speech** (MIBRAIN baseline: near chance). Why: no spatial identity (channels are anonymous bag), no per-patient layers, no coordinates. Proves raw scale (505M, 2528h) is insufficient without spatial identity mechanism.

**You need this to understand:** The anti-pattern. Validates that v12's architecture (VE cross-attention + per-patient layers + coordinates) is necessary — scale alone doesn't solve cross-patient speech.

### 18. Evanson 2025 — Log-linear scaling with supervised pretraining
`pastwork/summaries/evanson2025_minutes_to_days.md` · Imaging Neuroscience

3 sEEG patients, CLIP-style contrastive pretraining on 83-108h ambient room audio per subject. Log-linear scaling with no plateau at 100h. Supervised pretrain >> SSL (beats PopT, BrainBERT). Zero-shot fails — fine-tuning essential. Gamma bipolar (70-120Hz) improved 2/3 subjects. Single-patient only — explicitly identifies cross-patient alignment as unsolved.

**You need this to understand:** (1) Data scaling is log-linear with no plateau — our 7.6h uECoG is modest. (2) Supervised > SSL for same-subject (consistent with BIT Table 9). (3) Cross-day neural drift is massive (r=0.95) — per-patient normalization essential.

### 19. BarISTA (Oganesian et al. 2025) — Strongest validation of atlas-level spatial encoding for iEEG
`pastwork/summaries/oganesian2025_barista.md` · NeurIPS 2025

Spatiotemporal Transformer for multi-regional sEEG (10 subjects, Brain Treebank). Tests three spatial encoding scales: channel-level (LPI coordinates), parcel-level (Destrieux atlas), lobe-level (Desikan-Killiany). **Parcel-level >> channel-level by +8-10pp AUC** — the paper's central finding and strongest external validation of v12's VE approach. JEPA-style spatial masking SSL with latent-space reconstruction (EMA target encoder). Combined space-time attention > factored (+1-2pp, contradicts seegnificant). ~1M params, zero per-patient params.

**You need this to understand:** (1) Atlas-level spatial grouping is THE mechanism enabling cross-patient iEEG — confirms VE design. (2) Combined vs factored attention is now contested — test both. (3) JEPA latent targets and spatial masking are importable SSL innovations.

### 20. NDT3 / Generalist Motor Decoder (Ye et al. 2025) — Cross-subject transfer failure from sensor variability
`pastwork/summaries/ye2025_ndt3.md` · bioRxiv

350M-param autoregressive Transformer pretrained on 2000h of intracortical spikes from 30+ subjects. Cross-subject R² ~0.5 vs cross-session ~0.7. **Channel shuffle alone reduces cross-session to cross-subject performance** — sensor variability is the binding constraint. Output stereotypy: AR fails to extrapolate. Explicitly identifies per-patient layers as needed future work.

**You need this to understand:** THE key negative result motivating per-patient layers. Sensor variability (electrode order/identity) limits cross-subject transfer even at 350M params and 2000h. v12's per-patient diagonal + VE cross-attention directly addresses this.

### 21. Charmander (Mahato et al. 2025) — Perceiver bottleneck for multi-patient iEEG
`pastwork/summaries/mahato2025_charmander.md` · NeurIPS 2025 Workshop

Perceiver-based SSL with 32 latent tokens (≈ v12's 16 VEs) and 50% channel masking on AJILE12 (12 ECoG patients). Learnable per-channel per-participant embeddings (no coordinates). Model scaling 8M→142M shows NO downstream benefit. Independent convergence on the Perceiver bottleneck architecture validates v12's VE cross-attention design.
