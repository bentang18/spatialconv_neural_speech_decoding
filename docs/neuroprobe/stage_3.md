# Neuroprobe Stage 3 — FM-embedding regression pretraining (skeleton)

*Drafted 2026-04-25. Provisional. Replaces the discrete linguistic-label supervised pretraining originally folded into Stage 2b — that approach was cut once the continuous-FM regression pretext was on the table.*

Strategy anchor: `docs/neuroprobe/plan.md`. Sequencing predecessor: `docs/neuroprobe/stage_2.md` (Stage-2a SSL backbone is the natural init for Stage 3 — Stage 3 adds continuous FM-regression heads on top of an already-SSL'd backbone).

## Objective

Pretrain v14's atlas-anchored backbone with continuous foundation-model embedding regression as the supervised pretext: at every timestep, predict frozen DINOv3 (vision), Whisper / Wav2Vec2 (audio), and GPT-2 (language) hidden states from neural input. **Regression-target framing — the FMs are *targets*, not inputs.**

Continuous FM embeddings are a strict superset of discrete linguistic labels: every discrete label (POS, phoneme, word ID, surprisal) is a low-rank summary of what's already in the corresponding FM hidden state, and the FM gives ~1024 dense dimensions per timestep instead of one categorical bit per word. Goldstein et al. 2025 (Nat Hum Behav) show empirically on ECoG that Whisper embeddings outperform symbolic feature models (phonemes, POS, GloVe) for predicting high-gamma activity — direct on-modality evidence that the cut from discrete labels was correct.

## Success criterion (sharp)

**Cross-subject mean AUROC ≥ 0.58** — the existing stretch target from `plan.md`. Stage 3 is the stage that delivers it.

If Stage 2 already cleared 0.58, Stage 3 must beat Stage 2 by ≥ +0.01 to justify added complexity (FM inference, embedding cache, joint loss schedule). If Stage 3 ≈ Stage 2, FM regression isn't earning its keep — fall back to Stage 2 backbone for submission.

**Realistic-ceiling band on the 33-h legal Tier-1 corpus.** Banville et al. 2025 fit per-log-hour decoding-AUROC slopes per device: EEG 0.045 / MEG 0.064 / 3T fMRI 0.048 / 7T fMRI 0.075. iEEG is not in their benchmark, but its SNR sits between 7T fMRI and high-density MEG, so a slope band of 0.06–0.08 per log-hour is the principled prior. Translating: at 33 h the achievable lift over the 0.539 Linear-baseline ceiling sits at roughly +0.05–0.08, putting the realistic Stage-3 ceiling in the **0.59–0.62** band. The 0.58 stretch threshold is well-inside this band. Doubling to ~70 h via D-cohort buys ~+0.02 if the slope holds. The Banville scaling-subjects null result (24→48 subjects, p=0.49) means scaling Tier-1 hours rather than Tier-2 patient count is the right axis — directly aligned with v14's atlas-anchored thesis.

## Why regression-target, not input-arm

The natural reading of "use FM embeddings to anchor the brain backbone" is to inject them as a parallel input arm. We considered and rejected this in favor of regression-target framing.

| Property | Regression-target (neural → FM) | Input-arm + JEPA (FM context) |
|---|---|---|
| Forces FM structure into backbone | Yes | No — FM info available but not required |
| Cross-subject transfer story | Strong (every subject's brain → same FM space) | Weaker (backbone learns subject-specific compensations to FM context) |
| Failure mode | Underperforms if FM features too brain-distant | Shortcut learning — FM tokens free-ride |
| Attribution | Sharp, per-modality held-out R² | Diffuse |
| Test-time contract | Drop heads, backbone unchanged | Drop FM arms — train/test marginal mismatch unless masking schedule fixes it |
| iEEG literature precedent | Dense (Caucheteux/King 2022, Pasad 2024, Antonello 2023, Goldstein 2022, Tang 2023) | Sparse |

The cross-subject transfer point decides it. Our thesis is *anchoring brain representations to a shared external coordinate system* (anatomy via parcels, semantics via FM). Regression-target enforces this explicitly. Input-arm doesn't.

**Lit anchors**:
- Raugel et al. 2025 (arXiv 2508.18226, King lab) — ridge-regression linear alignment between DINOv3 features and fMRI/MEG (NSD + THINGS-MEG); larger + human-centric DINOv3 variants align best; early-to-late layer order mirrors brain hierarchy. Establishes the vision-FM regression-target framing on non-invasive recordings.
- **Goldstein et al. 2025 (Nat Hum Behav) — direct on-modality (ECoG) precedent.** 4 patients × ~25 h naturalistic conversation. Whisper acoustic / speech / language layers map onto cortical hierarchy (primary auditory / sensorimotor / STG-IFG respectively), Whisper beats symbolic features for predicting high-gamma. This is the strongest existing demonstration that a multimodal speech-to-text FM captures the cortical hierarchy in iEEG.
- **Evanson "Minutes to Days" 2026 (arXiv 2512.15830, Meta FAIR + APHP) — direct iEEG scaling precedent.** 3 sEEG × ~100 h ambient + 120-min audiobook. CLIP-loss alignment between brain module and wav2vec2-xlsr-53 layer 19. Three load-bearing findings: (i) **log-linear scaling holds with pretraining hours, no plateau**; (ii) **zero-shot ambient-pretrain → audiobook task FAILS** (rank ≈ chance) — pretrain → finetune required; (iii) wav2vec2 contextualization beats melspectrogram for 2/3 subjects but not for the subject with electrodes in primary auditory — **contextual-FM lift is electrode-coverage-dependent**.
- Caucheteux & King 2022 (Comm Bio), Pasad et al. 2024, Antonello 2023 — supporting LM-/audio-FM → brain linear alignment lineage.
- Millet et al. 2022 (NeurIPS) — Wav2Vec 2.0 trained from scratch develops brain-like hierarchy independently; suggests audio FM choice may matter less than fact-of-being-a-speech-SSL.
- Banville et al. 2025 (arXiv 2501.15322, Meta) — image-decoding scaling laws across 4 devices. CLIP+MSE combined loss with λ=0.25 on MSE. Per-log-hour slopes: EEG 0.045 / MEG 0.064 / 3T 0.048 / 7T 0.075. **Scaling # subjects ≈ 0 lift** (24→48 subjects, p=0.49) under their per-subject-layer architecture — direct evidence v14's atlas-anchored alternative is the right axis.
- Gwilliams et al. PNAS 2025 (Hierarchical Dynamic Coding) — 21 MEG, 54 linguistic features, 6 levels: phonetic 184 ms / word-form 752 ms / lex-syntactic 384 ms / syntactic-state 720 ms / semantic 1600 ms representation duration. Higher-level features decodable earlier and last longer. **Per-FM-target window justification**: Whisper-acoustic regression should use ~200-ms context; Whisper-language regression should use ≥1-s context.
- Gadonneix et al. 2026 (arXiv 2604.03021, Meta + APHP) — 2 ALS pts × 8 Utah arrays in motor + IFG, 20.4k sentences (silent/mouthed production, BrainGate2). **Spike + binned-spike-power, not LFP/HG** — caveat for direct quantitative comparison to BT sEEG HG. Three findings transfer: (i) **microscale multiplexing**: 9.3/10 top electrodes encode phon + syll + word jointly, no anatomical segregation — argues against parcel-mean pooling for fine-grained features; (ii) **dynamic neural code = position-encoding-in-brain**: same speech unit evolves through orthogonal subspaces (diagonal temporal generalization), authors explicitly draw the Vaswani/RoPE analogy — direct ammunition for v14's RoPE temporal-only choice; (iii) **production-side hierarchical timescales differ from perception**: phoneme decoder valid 1.0 s, syllable 1.6 s, word 2.5 s; pre-onset emergence 2.2/2.3/3.1 s for phon/syll/word — production windows are longer than Gwilliams perception. Neuroprobe BT is perception (audiobook listening), so the Gwilliams times are the relevant prior; Gadonneix sharpens the production-side prior for PS-resume Stage 3.

Direct extension of these to iEEG cross-subject pretraining is the Stage-3 hypothesis.

## Regression targets (4 from 2 FMs)

Goldstein et al. 2025 establish that Whisper's internal hierarchy maps onto cortical hierarchy in ECoG: acoustic features → primary auditory; speech features → sensorimotor; language features → STG/IFG. We exploit this by extracting **three theory-grounded levels from Whisper** rather than treating Whisper as a flat encoder + sweeping layers blindly. Plus DINOv3 (vision) for visual-stimulus coverage.

| Target | FM + extraction point | Sampling | Target dim | Cortical correlate (Goldstein 2025) |
|---|---|---|---|---|
| Vision | DINOv3 ViT-L/16 (or ViT-G if compute permits) | ~24 Hz (per movie frame) | 1024 (L) / 1536 (G) | Visual cortex; scene structure that audio-only SSL never captures. |
| Whisper-acoustic | Whisper-large early encoder layer | 50 Hz | 1280 | Primary auditory cortex; spectro-temporal features. |
| Whisper-speech | Whisper-large late encoder layer | 50 Hz | 1280 | Sensorimotor cortex / STG; phonemic / syllable-level features. |
| Whisper-language | Whisper-large decoder hidden | per token (~2–4 Hz) | 1280 | STG / IFG; semantic / syntactic features. |

Per-target linear (or small MLP) head off the parcel-token backbone. Loss: MSE in z-scored FM space (per-dim z-score on legal-data targets so scale doesn't dominate). Joint loss schedule: weighted sum across all 4 targets. Default equal weights; reweight if one target dominates the loss surface.

**GPT-2 dropped as default** (was originally a 5th target, the pure-language comparison). Whisper-decoder is on-modality (ECoG, Goldstein 2025) language-level with direct precedent; GPT-2 is the pure-text version of essentially the same hidden-state regression — parallel work for marginal lift. Kept as an optional ablation only: if Whisper-decoder underperforms on the language-level target (e.g., audio-conditioning bias proves to hurt rather than help), GPT-2-medium hidden states on MFA-aligned transcripts are the natural fallback.

## Frozen design commitments (revisit on Stage-2 close)

- **Backbone init**: Stage-2a SSL pretrained checkpoint. Cold-start backbone → FM regression direct is a separate ablation only.
- **FM choice**: DINOv3 ViT-L (vision), Whisper-large with three extraction levels (acoustic / speech / language per Goldstein 2025), GPT-2-medium (pure-language comparison). All frozen.
- **Loss**: per-target combined **D-SigLIP + MSE** (λ=0.25 on MSE, per Banville 2025 recipe). D-SigLIP (deduplicated SigLIP, d'Ascoli 2025) over plain CLIP because (i) Stage-3 batches are large (33-h corpus chunked into 3-s windows ≈ 40 k samples/epoch; SigLIP's binary-classification framing scales without softmax bottleneck) and (ii) BT movie audio repeats lexical content across films, which corrupts CLIP/SigLIP's negative pool unless deduplicated. MSE term anchors absolute reconstruction; CLIP/SigLIP term anchors retrieval structure. FM-side z-score per-dim with **train-set statistics only** (Evanson 2026 protocol; without this, contrastive loss collapses to scale matching).
- **Window contract**: 3-s symmetric on both brain and FM sides (matches Défossez 2023 / Evanson 2026 brainmagick recipe). Brain side: 3 s @ 200 Hz HG envelope from Stage-0 loader. FM side: native FM rate → mean-pool over 3-s sliding windows. **Per-target window override**: Whisper-acoustic uses 200-ms windows, Whisper-language uses ≥1-s windows (Gwilliams HDC 2025 — feature-stability time-scales). Vision and Whisper-speech stay at 3 s.
- **Schedule**: joint SSL + 3-modality regression. Equal weights as default.
- **Pretrain → finetune ordering required, not zero-shot.** Evanson 2026 demonstrates ambient-pretrained features are OOD relative to the task corpus despite log-linear pretraining gains. Test contract: pretrain on legal Tier-1 corpus → finetune linear head on S2/trial-4. Skipping the finetune step is *not* on the table — zero-shot evaluation is for sanity-checking only.
- **Test contract**: drop all 4 regression heads, fit S2/trial-4 task heads on the regressed backbone. Identical to Stage 2 head-fit pattern.
- **Per-subject normalization**: inherit from Stage 2 SSL design constraint (per-subject parcel-feature standardization before SSL still applies).
- **Pretraining corpus**: same legal Tier-1 cocktail as Stage 2 (BT-only ~33 h, ~25 h whitelist). Corpus must span days, not single-day sampling: Evanson 2026 finds brain embeddings drift across days even when wav2vec2 features don't (UMAP clusters by recording date) — likely tracking drug-dose tapering during hospitalization. Single-day pretraining is suspect. Tier 3 expansion (specifically the Zada et al. Podcast dataset — 9 ECoG patients, 1330 electrodes, naturalistic listening with time-aligned audio + transcripts) is the natural Stage-3 corpus extension once Tier 2 clears: same comprehension-task structure as BT, audio + transcript are necessary inputs for FM target generation. Cross-sensor caveat (ECoG joins sEEG) tracked at plan-level.
- **Coverage caveat for FM lift**: Evanson 2026 finds wav2vec2 contextualization beats melspectrogram only for subjects with electrodes outside primary auditory cortex — Heschl's-gyrus electrodes already capture low-level features that contextual FM doesn't add to. Stage-3 expectation: contextual-FM lift skews toward subjects with STG / sensorimotor / IFG coverage, not primary-auditory-only coverage. Per-subject lift will not be uniform.
- **Augmentation on the head fit**: channel-dropout, time-warp ±5%, mixup. Inherited.

## Corpus eligibility filter

Stage 3 requires **paired stimulus** (audio + transcripts time-aligned to neural data; ideally video for the vision target) — a hard prereq Stage 2 doesn't have. Any iEEG corpus *without* aligned stimulus is Stage-2 SSL fuel, not Stage-3 corpus material. This drops most of `docs/references/data_acquisition_matrix.md`'s top-ranked SSL corpora (AJILE12, SWEC, UPenn RAM, HUP, Omni-iEEG) from Stage-3 contention regardless of patient count or hours.

**Stage-3-eligible corpora** (open + audio + transcript / paired stimulus):

| Corpus | Pts | Hours | Stimulus | What targets it supports |
|---|---|---|---|---|
| Brain Treebank (Tier 1) | 10 | ~33 h legal | Hollywood movies | All 4 (vision + 3 Whisper levels) |
| **Naturalistic film iEEG ds003688** | **51** | TBD | Audiovisual film | All 4 (vision + 3 Whisper levels) — premium fit |
| Podcast ECoG ds005574 | 9 | 4.5 h | Story audio + transcript + LLM features | 3 (Whisper levels only — audio-only) |
| NeuroListen | 5 | 10+ h | Natural-speech audio | 3 (Whisper levels only) |
| sEEG passive listening ds004703 | 10 | — | Audio | 3 (Whisper levels only); non-commercial license caveat |
| Auditory Naming ds006234 | 119+ | — | Single-word naming + audio | Whisper-acoustic / -speech only; language-level degenerate (single word) |

**Production-corpus subset** (own-speech audio, no external visual stimulus):
- Internal Cogan D-cohort (Tier 2: 87 D-pts, 180 h — the 4-speech-task subset of a larger 113-pt / 384.7 h / 14-task D-cohort; `memory/project_d_cohort_data_inventory_2026_06_03.md`) — Whisper-acoustic / -speech extractable from spoken-word audio. Limited Stage-3 value — single-CVC-token utterances make Whisper-decoder / language targets degenerate. Stays primarily Stage-2 SSL fuel.
- Verwoert 2022 ds003194 (10 sEEG, Dutch read-aloud + 48 kHz mic), Bouchard-Chang DANDI 000019 (4 HD ECoG, CV syllables), Du-IN (12 sEEG, Mandarin 61-word).

**Watch-list (paper-only, monitor for release)** — would transformatively scale Stage 3 if any opens up:
- **Evanson "Minutes to Days"** (Meta FAIR + APHP, arXiv 2512.15830) — 3 sEEG × {100, 108, 84} h ambient pretrain + {74, 43, 250} min audiobook task. Closest open chronic field-potential + audio dataset; first iEEG paper to demonstrate log-linear pretraining-hours scaling. **Already a strong validation of v14 Stage-3 design.** Pretrained model + ambient sounds released TBD.
- **Evanson "Emergence of Language in the Developing Brain"** (arXiv 2512.05718) — 46 sEEG French pts (ages 2-46), 7 427 electrodes, "Le Petit Prince" audiobook (~40 min/subj on average). Wav2vec2-xlsr-53 (53 k h) + Llama 3.1 used as encoders. Developmental finding: phonetic features in STG by age 2-5; word-level features only emerge 6+ yrs.
- **Goldstein / Zada / Flinker NYU 24/7 ECoG** (Nat Hum Behav 2025) — ~100 h / 4 pts dense-sampling chronic conversations. Currently private; the Goldstein 2025 dataset itself.
- **Neuro2Semantic** (Mesgarani / Flinker, arXiv 2506.00381) — perceived speech + transcripts, cohort size undisclosed.

## Stimulus prereq (Block-S — must close before Stage 3)

BT ships audio + MFA word-level alignments. Movie video frames are likely missing in the BT release; the films are publicly identified per BT's metadata, so source externally if needed. Research use of publicly available film stills for derived FM features is standard.

Block-S checklist:
1. Confirm BT release contents — frames? audio? transcripts? sample rates? clock-sync to neural data?
2. If frames missing, identify each film from BT's metadata + obtain.
3. Frame-level time-sync to neural data (BT publishes per-trial onsets; frames need re-anchoring to neural clock).
4. Build cache: `scripts/neuroprobe/cache_fm_targets.py` writes per-session per-modality FM embeddings to disk at native FM rate. Estimated: ~10–15 GB per modality across the legal corpus. Single-GPU half-day for inference per modality.
5. Time-alignment audit: for one session, plot FM-target time series vs neural envelope at known-event timestamps (sentence onset, scene-cut). Visually verify alignment within ±50 ms before launching pretraining.

## Open questions (defer to empirics)

- **FM layer choice — partly resolved by Goldstein 2025.** Three Whisper extraction levels are theory-grounded (acoustic / speech / language). Still need to fix exact layer indices: which Whisper-encoder layer is "early" vs "late", which decoder hidden depth. DINOv3 (layer 6 vs 18 vs 30) still needs a sweep — King 2025 shows early-to-late progression but doesn't pin specific layers for iEEG.
- **GPT-2 fallback trigger.** GPT-2 is dropped as default (see Regression targets) — fallback only if Whisper-decoder underperforms on language-level. Empirical trigger: Whisper-decoder regression R² on legal-data held-out fold below DINOv3 / Whisper-encoder R² by some margin.
- **Joint vs sequential schedule.** Train all 4 regressions simultaneously, or stage them (vision → audio-acoustic → audio-speech → audio-language)? Joint is sample-efficient; sequential gives cleaner attribution.
- **Target contribution.** Drop-one-target ablation to measure per-target marginal AUROC lift. Likely the most informative ablation in the program — tells us which of the 4 targets carry the load and which are redundant.
- **Cold-start backbone vs Stage-2a backbone init.** Conservative default: start from Stage-2a SSL backbone. Cold-start → FM regression directly is a useful sanity ablation: does FM regression alone do what SSL+FM regression does?
- **Frozen FM vs adapter on FM.** Frozen is the conservative default. If FM features are slightly brain-distant, a small adapter (linear or LoRA) on the FM might help. Defer.
- **GPT-2 hidden vs scalar surprisal — pretext-target overlap.** GPT-2 hidden is a continuous superset of the scalar surprisal eval label. Probably attestation-legal (proxy ≠ exact eval label) but worth flagging for the submission attestation.
- **Hybrid SSL + FM-regression weighting.** If FM regression dominates the loss, the backbone may forget masked-recon structure useful for time-locked decoding. Loss-weight sweep.
- **Head architecture (linear vs MLP).** Start linear (matches the encoding-model literature). Add small MLP only if linear ceiling underperforms.
- **Frozen backbone for fine-tune?** Default per Experiment #6: frozen backbone, per-task linear probe. End-to-end fine-tune as ablation if frozen ceiling underperforms.

## What Stage 3 explicitly does NOT do

- **Use FM embeddings as INPUT** (input-arm framing rejected — see Why regression-target).
- **Fine-tune the FMs** (DINOv3, Whisper, GPT-2 are frozen; only the regression head + backbone train).
- **Pretrain on the 12 off-limits eval sessions.**
- **Use the exact 15 eval labels as proxies** (continuous FM hidden states only).
- **Submit Within-Session or Cross-Session.**
- **Cross-sensor pretrain on uECoG.**
- **Mix modality embeddings before regression** (no shared linear head over `[DINOv3, Whisper, GPT-2]` concatenated — separate per-modality heads).
- **Cross-modal contrastive pretexts** (e.g. CLIP-style alignment between FM modalities directly) — out of scope; the brain backbone is the only learner here.
