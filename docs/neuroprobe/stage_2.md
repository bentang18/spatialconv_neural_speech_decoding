# Neuroprobe Stage 2 — Pretrain + S2/trial-4 fine-tune (skeleton)

*Drafted 2026-04-25. Sub-stage structure simplified to 2a / 2b on 2026-04-25 — discrete-linguistic-label supervised sub-stage cut in favor of continuous FM-embedding regression (see `docs/neuroprobe/stage_3.md`).*

*Updated 2026-04-25 (same day) after re-evaluating v14 against the King-lab + EEG/iEEG-FM evidence stack. Four pre-commits land here: (1) Stage-2 SSL = **D-SigLIP brain↔frozen-Whisper-encoder + cross-subject parcel-id contrastive aux + MSE λ=0.25** (Banville/Evanson recipe); (2) v14 backbone change before Stage-2: **`pe2d` → cortical-geodesic distance-bias attention** (sEEG-native, anatomy-respecting); (3) **Banville-style 4-point scaling curve** as a Stage-2 deliverable; (4) **decoupled SSL/eval windows + per-session affine normalization + `d_model` ∈ {32, 64, 128} sweep**.*

Strategy anchor: `docs/neuroprobe/plan.md`. Predecessor: `docs/neuroprobe/stage_1.md` (cold-start gate result determines Stage-2 emphasis — 2a SSL-only ceiling sufficient if cold-start clears 0.539 already; 2b multi-patient head prior becomes thesis-deciding if cold-start sits in 0.527–0.539). Successor: `docs/neuroprobe/stage_3.md` (continuous FM-embedding regression supervised pretraining).

## Objective

Pretrain v14's atlas-anchored backbone on legal data, fit linear head on S2/trial-4 → submission. The submission-bearing stage.

## Success criterion (sharp)

**Cross-subject mean AUROC ≥ 0.56 to submit. ≥ 0.58 stretch. < 0.539 abort.**

The pretrained-v14 must beat cold-start-v14 cross-subject — the inverse of the BrainBERT-trained < BrainBERT-untrained pattern. This is the load-bearing prediction of the alignment-is-the-bottleneck thesis: anchoring the representation to anatomy (not patient-specific channel indices) means pretraining accumulates cross-subject-transferable structure rather than per-subject overfit.

**Direct ammunition for the thesis** — Banville et al. 2025 (arXiv 2501.15322, Meta) train multi-subject image decoders across 4 devices (EEG/MEG/3T/7T fMRI) using Défossez 2022's BrainModule architecture (per-subject linear layer + sensor-position spatial attention). They find **scaling # subjects ≈ 0 lift** under their architecture: 24→48 subjects on Grootswagers2022 EEG buys 0.004 AUROC, p=0.49. "Inter-subject variability appears to harm overall model performance, especially in low-subject regimes." This is direct evidence that the per-subject-layer mechanism *cannot* extract cross-subject lift — leaving the door open for atlas-anchored alternatives. v14's BNA `P_emb` is precisely that alternative.

## Pre-Stage-2 backbone changes (Neuroprobe-only path)

Three v14 backbone changes before Stage-2 cold-start. All trace to a single root cause: the PS-pause defaults assumed a dense 2-D Utah-style array; BT sEEG depth shafts have no 2-D grid. Each Neuroprobe-side fix is paired with a "PS-resume keeps the old default" note since the 2-D grid is real on PS uECoG.

### (a) `pe2d` → BNA-connectivity-init attention bias

**`pe2d` (2-D virtual-grid positional encoding)** is modality-mismatched on BT sEEG. The principled replacement is *not* raw cortical-geodesic distance (a weak prior — adjacent parcels can be functionally distinct, e.g. IFG vs ventral premotor at ~5 mm geodesic), but BNA's own connectivity matrices, which are already on the same atlas as `P_emb`:

- **Across-parcel attention bias initialized from BNA functional connectivity** (resting-state fMRI, 246×246) — primary. Anatomic connectivity (DTI tractography, 246×246) and behavioral co-activation (BrainMap meta-analysis, 246×246) available as ablation alternates; default is functional, with a "summed functional + anatomic" variant kept on the ablation list.
- **Bias indexed via per-electrode soft parcel support** (consistent with `P_emb` indexing) — bias on edge (e_i, e_j) ≈ Σ_{P, Q} support(e_i, P) · BNA_conn(P, Q) · support(e_j, Q).
- **Optionally learnable** as a residual on top of the BNA init (init = BNA, learn small delta).
- **Within-parcel attention free** (no bias) — preserves Gadonneix-microscale multiplexing within local circuits (9.3/10 top electrodes encode phon+syll+word jointly within a 3.2-mm patch).
- **Drops `pe2d`**; **also drops cortical-geodesic distance** as primary bias (kept as fallback only for parcel pairs without BNA connectivity entries — shouldn't happen, BNA covers 246×246).

This is the cleaner v14-thesis-consistent move: same atlas drives both the parcel embeddings and the cross-parcel attention prior. The earlier geodesic proposal (2026-04-25 first pass) was a half-measure.

### (b) Drop `partialconv`

`partialconv` is 2-D masked convolution over a regular electrode grid — designed for PS uECoG dense Utah-style 8×16 layouts where neighbors-on-grid is real. BT sEEG depth shafts have no 2-D grid; using `partialconv` requires fabricating a virtual 2-D layout (same modality-mismatch failure mode as `pe2d`). **Drop on the Neuroprobe path entirely.** Architecture becomes per-electrode token sequence → attention with BNA-connectivity bias → readout. No spatial convolution. PS-resume keeps `partialconv` (the 2-D grid is real there).

### (c) `hierarchical_atlas` → flat per-parcel pool readout

`hierarchical_atlas` does two things: (i) per-electrode → per-parcel pool via BNA support; (ii) hierarchical aggregation Tier-1 → Tier-2 → global. (i) is on-thesis (parcel-pool kills per-electrode-index variance — the cross-subject killer). (ii) is multi-scale capacity bonus that was load-bearing on PS dense 2-D where multi-scale spatial aggregation across the array mattered.

For Neuroprobe per-task binary AUROC at 1-s windows, (i) does most of the work. **Default: flat per-parcel pool** — per-electrode tokens → BNA-soft-pool to per-parcel tokens → mean (or attention) pool over parcels → linear head. Hierarchy retained as **+tier ablation** only. Drops the `pool=(4,8)` hyperparameter from the default.

### Summary — what's kept, what's dropped on the Neuroprobe path

| Component | Status | Reason |
|---|---|---|
| Per-electrode tokens (B-1 path) | **kept** | Gadonneix multiplexing |
| BNA `P_emb` shared across subjects | **kept** | core thesis |
| No per-subject linear layer | **kept** | core thesis |
| RoPE temporal-only | **kept** | Gadonneix dynamic-code-as-position-encoding |
| BNA-connectivity-init attention bias (within-parcel free, cross-parcel biased) | **NEW (replaces `pe2d`)** | atlas-consistent prior; functional connectivity > geodesic |
| Flat per-parcel pool readout | **NEW (simplified from `hierarchical_atlas`)** | hierarchy load-bearing on PS dense 2-D, not on Neuroprobe sEEG |
| `partialconv` | **DROPPED on Neuroprobe path** | 2-D-grid assumption; no sEEG analog |
| `pe2d` | **DROPPED** | 2-D-grid assumption; modality-mismatched |
| `pool=(4,8)` | **DROPPED** | tied to hierarchical readout |
| `d_model` | **swept {32, 64, 128}** | capacity floor at scale |

**Lit anchors**: MV-BrainFM (Xu 2026) distance-aware attention bias is the closest prior; we differ by initializing from BNA connectivity (atlas-consistent) rather than learned distance. King-lab BrainModule's spatial-attention-over-2-D-sensor-positions is the EEG/MEG analog and is *not* portable to sEEG. REVE / DIVER-1 use continuous (x,y,z,t) Fourier PE without atlas — that's the field default for iEEG-FM but discards what makes v14 unique.

**Execution order**: these are v14 backbone changes, not Stage-2-internal ones. The Neuroprobe Stage-1 cold-start gate (≥0.539 vs Linear-Lap+spec) re-runs with all three pre-applied. Stage-2 enters cold-start with the new backbone already merged. PS-resume returns to the partialconv + hierarchical_atlas defaults where the 2-D grid justifies them.

## Sub-stages

| Sub-stage | Backbone training | Head fit | Question answered |
|---|---|---|---|
| **2a** | Pre-committed contrastive SSL (see *Pre-committed Stage-2 SSL design* below) on legal set | Linear, L2, S2/trial-4 | Contrastive-SSL ceiling on the atlas-anchored backbone with the new distance-bias attention. |
| **2b** *(cheap)* | Same backbone as 2a | Linear with multi-patient head prior init, L2 + Group-L1-by-parcel, S2/trial-4 | Cheap multi-patient regularization without backbone-supervised cost — does per-session head averaging recover most of what backbone-supervised pretraining would? |

**Submission decision rule**: best AUROC of {2a, 2b}. If neither clears 0.56, defer submission to Stage 3 (FM-embedding regression pretraining).

## The bet (explicit acknowledgment)

Cross-subject brain↔FM SSL on iEEG has *never been measured* by anyone. Évanson Minutes-to-Days runs per-subject; King-lab's cross-subject brain↔FM results are all EEG/MEG (Banville, d'Ascoli, Brain2Qwerty). The masked-recon iEEG SSL family (BrainBERT, PopT, presumed DIVER-1) has been measured cross-subject and **fails or stalls**:

- BrainBERT-trained 0.522 < BrainBERT-untrained 0.527 — *training hurt cross-subject transfer*. The "BrainBERT inversion."
- PopT 0.526 — same untrained-cluster ceiling.
- DIVER-1 (5.3 k h iEEG, largest pure-iEEG corpus) declined to submit cross-subject despite #1 within-session at 0.678. Strong implicit signal cross-subject didn't clear.

Stage 2 bets that combining **(atlas anchoring)** + **(King-lab brain↔FM SSL)** crosses the cross-subject gap that masked-recon couldn't. The bet rests on three legs:

1. **Negative evidence on the alternative**: masked-recon iEEG SSL has been tested cross-subject and failed. The BrainBERT inversion suggests the failure mode is mechanism-level (subject-id structure dominates the rep), not just arch-level. Atlas anchoring is supposed to remove that failure mode.
2. **Positive evidence on adjacent modalities**: King-lab brain↔FM works cross-subject on EEG/MEG with per-subject linear layers.
3. **Positive evidence on iEEG per-subject**: Évanson Minutes-to-Days log-linear scaling on 3 sEEG × ~100 h ambient + audiobook task. Same loss family, same modality, just per-subject.

The gap (1)+(2)+(3) leave open: atlas-anchored cross-subject brain↔FM SSL on iEEG — never demonstrated. Stage 2 fills it or fails.

**Risk**: if the bet loses, Stage 3 (FM regression) inherits the same architectural anchor and is unlikely to recover the cross-subject gap on its own. The architecture, not the SSL family, would be the bottleneck.

## Mandatory attribution comparison: atlas-anchored + masked-parcel-recon

Without a side-by-side, a Stage-2 positive result is ambiguous — did the lift come from atlas anchoring, from brain↔FM SSL, or only from their combination? Run a single attribution comparison at Tier-1 33 h:

- **Default cell** (Stage-2a): atlas-anchored backbone + D-SigLIP brain ↔ Whisper-mid-encoder.
- **Attribution cell**: same backbone (per-electrode + BNA-conn-bias + flat per-parcel pool + RoPE + per-subject + per-session affine norm + 3-s SSL windows) — only the SSL pretext changes to **masked-parcel-recon** (mask 30 % of (electrode × time) cells, predict pooled parcel representation). Closest atlas-anchored analog of BrainBERT.

**Read-out**:
- If both clear 0.56: atlas anchoring is doing the cross-subject work; SSL family is interchangeable. Submit the better of the two; story emphasizes architecture.
- If only default clears: brain↔FM SSL is the load-bearing piece on top of atlas. Story emphasizes the loss family.
- If neither clears 0.56 but both clear cold-start: SSL is helping marginally; architecture is the limit.
- If neither clears cold-start: the architecture itself is the bottleneck. Stage 3 won't save it; rethink before committing further compute.

**Cost**: one extra Stage-2 run at Tier-1 33 h. Cheap insurance for a result whose interpretation is otherwise undetermined.

## Pre-committed Stage-2 SSL design (single-loss, Evanson-aligned)

The thesis predicts pretrained-v14 > cold-start-v14 cross-subject. This holds **only if** SSL doesn't smuggle subject-specific structure back in. The constraints below are jointly load-bearing — without all of them, parcel features can carry per-subject scale/offset bias and the model learns the per-subject distribution of pooled parcel activity, replicating the BrainBERT-trained < BrainBERT-untrained inversion under our own banner.

**Single loss**: D-SigLIP brain ↔ frozen Whisper-large mid-encoder, no auxiliary terms. Matches the King-lab convergent stack (Évanson Minutes-to-Days CLIP-only on iEEG; d'Ascoli D-SigLIP at scale; Brain2Qwerty CLIP). The previously-proposed cross-subject parcel-id contrastive was demoted — without stimulus-aligned same-time pairs the loss is degenerate (collapses to per-parcel baseline statistics); with stimulus alignment it's largely redundant with brain↔FM (FM is the shared anchor that pulls both subjects' brain reps together). The previously-proposed MSE λ=0.25 aux was demoted — Évanson's iEEG result holds with CLIP-only. One loss, well-chosen.

### Loss specification

```
b = brain_backbone(neural[t : t+3s])       # ∈ R^d, d ∈ {32, 64, 128}
a = Whisper_mid(audio[t : t+3s])           # ∈ R^1280, mean-pooled over 3 s
z_b = proj_brain(b)                        # trainable linear (or 2-layer MLP), R^d → R^k
z_a = proj_audio(a)                        # trainable linear (or 2-layer MLP), R^1280 → R^k
L = D-SigLIP(z_b, z_a)                     # deduplicated SigLIP, d'Ascoli 2025
```

- **Whisper frozen**, brain backbone + both projection heads trainable.
- **Common space `R^k`**, k ∈ {256, 512} (pin in first ablation).
- **D-SigLIP > plain CLIP — load-bearing motivation: cross-subject same-content batching.** A Stage-2 batch draws `(brain_window, audio_window)` pairs across *all* legal BT subjects. Multiple subjects watched the same movies, so a batch routinely contains `(brain_subjA, audio_movieX_t=120s)`, `(brain_subjB, audio_movieX_t=120s)`, `(brain_subjC, audio_movieX_t=120s)` — identical audio targets, different brain windows. Plain CLIP's softmax denominator pushes `brain_subjA` *away from* every non-positive in-batch audio, including B's and C's audio (same content) → the loss actively penalizes exactly the cross-subject hyperalignment we want to learn. D-SigLIP (d'Ascoli 2025) replaces softmax with per-pair sigmoid (no "push-away-from-all-others" denominator) and explicitly deduplicates content-identical pairs from the negative pool. Évanson Minutes-to-Days uses plain CLIP because they train per-subject (no cross-subject batches → no same-content collision); the moment we cross-subject-batch, the loss family has to handle it. Plain SigLIP without dedup is intermediate but still puts content-identical samples as negatives — D-SigLIP's dedup is the load-bearing piece.
- **Whisper-large mid-encoder layer**: Goldstein 2025 ECoG hierarchy → mid encoder = "speech" features (phonemic / syllabic; STG / sensorimotor). Evanson uses wav2vec2-xlsr-53 layer 19; mid Whisper is the closest analog with the additional benefit of being multimodal-aware and multilingual. Pin exact layer on first ablation.
- **FM-side z-score per-dim** with train-set statistics only (Évanson protocol; without this, contrastive collapses to scale matching).
- **Post-SSL**: discard projection heads; brain backbone alone enters Stage-3 init or eval head-fit. (Brain projection optionally retained as Stage-3 regression init.)

### Constraints

1. **Per-subject + per-session affine normalization of parcel features before SSL.** Per-(subject, parcel) z-score kills cross-subject scale/offset bias. Per-session affine on top kills within-subject day-drift (Évanson Minutes-to-Days UMAP-by-recording-session finding — likely tracks drug-dose tapering during hospitalization). Applied at SSL pretrain *and* eval head-fit time. This is the cheap replacement for King-lab's per-subject linear layer at finer granularity, without re-introducing per-subject params that block zero-shot.

2. **3-s symmetric SSL windows / 1-s eval-window** (Défossez 2023 / Évanson 2026 brainmagick recipe). SSL pretrain doesn't need event-locking; benchmark-spec head-fit uses 1 s.

3. **No pretexts coupled to per-subject structure.** No "predict next electrode in this patient's montage" — anything tied to electrode index, montage layout, or per-patient channel ordering is contraindicated.

4. **Optional auxiliary (off by default): cross-subject parcel-id contrastive with stimulus-aligned same-time pairs.** Anchor (subject A, parcel P, time t), positive (subject B, parcel P, time t — *same movie, same offset*), negative (subject A, parcel Q ≠ P, time t). Requires BT subjects with overlapping movie coverage. Hyperalignment-style (Haxby 2011 / Guntupalli 2016 SRM analog at parcel granularity). Enabled only if scaling-curve under-delivers and we want to enforce parcel-pooling structure beyond what brain↔FM does.

**Contrast with King-lab convergent stack.** Défossez 2022 (brainmagick) → Banville 2025 → d'Ascoli 2025 (Nat Comms) → Brain2Qwerty (Lévy 2025) all converge to: spatial-attention layer over sensor 2-D positions + **per-subject linear layer (~64-dim)** + dilated conv stack + (optional) Transformer + CLIP / SigLIP / D-SigLIP loss. The per-subject linear layer is the field's standard cross-subject-heterogeneity fix; d'Ascoli explicitly notes it "prevents zero-shot generalization to new participants for which the subject layers need to be fine-tuned". v14 deletes this layer by design — parcel embeddings are shared across all subjects, no per-subject params. The bet is that anatomy-shared `P_emb` does what the subject-layer is *trying* to do, but in a way that allows zero-shot transfer.

**Microscale evidence for v14's RoPE choice (Gadonneix et al. 2026, arXiv 2604.03021).** Two ALS pts × 8 Utah arrays in motor + IFG, 20k sentences. Authors find a **dynamic neural code** — same speech unit (phoneme / syllable / word) evolves through orthogonal subspaces over its representation window (diagonal temporal generalization, every array). They explicitly frame this as a biological analog of transformer positional encoding (Vaswani 2017 / Su 2024 RoPE): "the small cortical patch effectively tags each phoneme or word with its relative position in the speech stream." Direct ammunition for v14's RoPE temporal-only choice — the brain implements *exactly* the rotation v14 imposes. Caveat: spike data, not HG-LFP; production paradigm. Microscale multiplexing finding (9.3/10 top electrodes encode phon+syll+word jointly) is also a caveat against parcel-mean pooling for fine features — v14 per-cell tokenization preserves within-patch multiplexing.

**Window-choice note from Gwilliams HDC (PNAS 2025).** Different linguistic features have different stable durations: phonetic 184 ms / word-form 752 ms / lex-syntactic 384 ms / syntactic-state 720 ms / semantic 1600 ms. SSL pretexts targeting different feature levels should use windows matched to the target feature's natural duration. For Stage-2a's masked-recon SSL, ~1-s event-locked windows cover phonetic-through-syntactic-state but cut off long-range semantic structure; that's fine since Neuroprobe eval tasks sit at the lower-to-mid levels of this hierarchy.

## Supervised pretraining → moved to Stage 3

Discrete linguistic-label supervised pretraining (phoneme/syllable identity, word2vec, GPT-2 hidden, speaker-onset) was originally folded into a Stage-2b supervised sub-stage. We dropped it once the continuous FM-embedding regression pretext (DINOv3 + Whisper/Wav2Vec + GPT-2 hidden) was on the table — every discrete linguistic label is a low-rank summary of what's already in the corresponding FM hidden state, and the FM gives ~1024 dense dimensions per timestep instead of one categorical bit per word.

See `docs/neuroprobe/stage_3.md` for the supervised-pretraining track.

## Multi-patient head prior (2b)

Decouples cross-subject regularization from full backbone-supervised cost.

1. SSL-pretrain backbone on legal set (no task labels).
2. Forward-pass each legal session through frozen SSL backbone → per-electrode features.
3. For each (legal session × eval task), fit a small linear head using that session's auto-derived eval-task labels. Each head is per-session-overfit.
4. Average / pool head weights across legal sessions per task → multi-patient prior head per task.
5. Use prior as **initialization** (or L2-regularization target) for the official S2/trial-4 head fine-tune.

**Cost**: a few hours of inference + linear regression. Each individual head is per-session overfit; the ensemble averages out per-subject overfit, retains cross-subject task structure. The S2/trial-4 head is anchored on a multi-patient prior without the backbone ever seeing task labels.

**Protocol read**: this is a *regularizer* on the official supervised fit (which still happens on S2/trial-4 only), not a separate training run on a separate dataset. The backbone never sees task labels at all — strictly conservative.

## Head design

- **Linear, L2-regularized** (matches BrainBERT/PopT pattern; the conservative default).
- **Group-L1 by parcel** as additional regularizer for cross-subject robustness — penalizes whole parcels, protects against the structural failure mode where the head weights features strong in S2's coverage but absent in test subjects.
- **Per-task heads** (protocol is per-task by definition).
- **Augmentation on the head fit**: channel-dropout, time-warp ±5%, mixup (same legal-and-underused logic as Stage 1). Inherited from Stage-0 / Stage-1.

## Pretraining corpus tiers (from `plan.md`, gated cumulatively)

- **Tier 1** — BrainTreebank-only, ~33 h sEEG (14 full + 5 partial whitelist). First-pass submission corpus.
- **Tier 2** — + D-cohort sEEG (~25–50 h), gated on Tier 1 > 0.539.
- **Tier 3** — + external public sEEG (Du-In, NeuroListen, Kunii/Tsuboyama, Utrecht/Martin), gated on Tier 2 lift.

uECoG (PS + lex) explicitly **not** in the pretrain corpus — cross-sensor (surface ↔ depth) transfer is a separate claim.

### Stage-2 deliverable: Banville-style 4-point scaling curve

Pretext choice matters less than slope-on-hours given Banville 2025's iEEG-band slope prior of **0.06–0.08 AUROC per log-h** (interpolated between 7T fMRI 0.075 and MEG 0.064). Submit a 4-point curve at the pre-committed SSL pretext:

| Anchor | Hours | Source |
|---|---|---|
| Quarter | ~8 h | Tier-1 random subset, 3 seeds |
| Half | ~16 h | Tier-1 random subset, 3 seeds |
| Full Tier-1 | ~33 h | Tier-1 whitelist |
| Tier-1 + D-cohort | ~66 h | Tier-2 (gated on Tier-1 > 0.539) |

Slope on log-h is the load-bearing measurement; if the iEEG slope falls inside the Banville prior band, Stage-3 ceiling estimates from `stage_3.md` carry through. **If slope < 0.04, the architecture isn't earning the SSL — fall back to cold-start submission and re-architect before Stage 3.**

### Stage-2 deliverable: `d_model` ∈ {32, 64, 128} sweep

`d_model = 32` was set in the 4-pt PS data-starved regime; Stage-2's 33-h corpus + Banville log-linear scaling imply capacity floor matters once data scales. Sweep at the pre-committed SSL pretext on Full Tier-1; the winner becomes the Stage-3 FM-regression backbone width. Cheap (single 3-cell ablation).

## Open questions (defer to empirics)

- **SSL objective choice** — *resolved 2026-04-26, single-loss*. See *Pre-committed Stage-2 SSL design*: single D-SigLIP brain ↔ frozen Whisper-large mid-encoder, with trained projection heads `R^d → R^k` and `R^1280 → R^k`. Aligns with Évanson Minutes-to-Days iEEG-on-modality precedent (CLIP-only) and the King-lab convergent stack. Cross-subject parcel-id contrastive kept as optional auxiliary only (degenerate without stimulus alignment; redundant with brain↔FM when aligned). MSE aux dropped (Évanson holds with contrastive-only on iEEG).
- **Whisper layer pin** (sub-question of the above): mid-encoder layer to match Goldstein 2025 ECoG cortical-hierarchy mapping (acoustic / speech / language). Pin exact layer on first sweep.
- **Common projection dimension `k`**: pin in first ablation; ∈ {256, 512}.
- **Within-parcel / intra-parcel spatial expressivity** *(elevated to early Stage-1 ablation, 2026-04-26)*. Current default carries cross-parcel structure via BNA-connectivity-init bias + parcel-id structure via `P_emb`, but two electrodes in the same parcel see the same bias term despite differing 3-D positions. Three candidate fixes to ablate before Stage-2 ships: (a) **within-parcel 3-D Euclidean pairwise bias** (`exp(-‖p_i − p_j‖² / σ²)` on MNI positions, applied only when both electrodes share a parcel — no cross-sulcus problem since within-parcel stays within a sulcus); (b) **shaft-aware additive bias** (use BT electrode name stems to identify same-shaft pairs; volume-conduction prior at ~3.3 mm intra-shaft spacing; sEEG-only); (c) **3-D MNI Fourier PE** added to per-electrode tokens (SwinTW-style — Chen 2025 demonstrates cross-subject without per-subject params using shared 3-D coords; here it adds within-parcel resolution on top of `P_emb`'s parcel-id). Run as Stage-1 ablation alongside the scaling curve; promote winner into Stage-2 default if any clears +0.005 cross-subject AUROC vs default. Plan-level: `plan.md` Experiment #11.
- **Frozen probe vs fine-tune (Experiment #6)**: per-task linear probe vs end-to-end fine-tune. Frozen is the honesty standard; fine-tune may raise the number.
- **BNA argmax vs probabilistic support**: inherit from Stage-0 D.3a vs D.3b winner.
- **Stage 2 (SSL + multi-patient head prior) vs Stage 3 (continuous FM regression)**: Stage 2 is the cheaper bet (no DINOv3 inference, simpler target). Stage 3 is the richer pretext but heavier. Decide submission ordering after Stage 2 numbers land — submit early from Stage 2 if it clears 0.56, then chase stretch with Stage 3.

## Parallel benchmark — EEG Foundation Challenge (NeurIPS 2025)

Aristimunha et al. 2025 (arXiv 2506.19141) launched a NeurIPS 2025 competition with the same scientific frame as Neuroprobe but at scalp-EEG scale: HBN-EEG, 3 000+ subjects, 128-ch, 6 cognitive tasks, code-submission. Two challenges: (i) zero-shot cross-task + cross-subject regression on Contrast Change Detection response time; (ii) psychopathology factor regression. Final ranking weights psychopathology 0.7 / cross-task 0.3.

Cite as parallel evidence that the field has converged on cross-subject as the load-bearing eval — same answer as Neuroprobe, different modality. Their HBN-EEG corpus (NEMAR, CC-BY-SA-4.0) is *not* a Stage-2 SSL fuel candidate (scalp ≠ iEEG, no surface↔depth transfer), but the competition's leaderboard and baseline-zoo (Braindecode + EEGDash, 30+ models) inform our framing of "what does it mean to win cross-subject in 2025-2026".

## What Stage 2 explicitly does NOT do

- Pretrain on the 12 off-limits eval sessions (`btbank1_{1,2}, btbank2_{0,4}, btbank3_{0,1}, btbank4_{0,1}, btbank7_{0,1}, btbank10_{0,1}`).
- Train the backbone with any Neuroprobe task labels (cut: discrete linguistic-label supervised pretraining; the multi-patient head prior is a head-side regularizer only — backbone never sees task labels). Brain↔Whisper contrastive is *not* task supervision: Whisper hidden states are not Neuroprobe labels.
- **Multi-target FM-embedding regression** — Stage 3 territory. Stage 2 carries one audio-FM target only (Whisper-encoder, contrastive-primary with MSE λ=0.25 aux). Stage 3 adds DINOv3 + Whisper-3-layer simultaneous regression.
- Submit Within-Session or Cross-Session.
- DK atlas anchor.
- Electrode selection beyond the BT Lite cap.
- Cross-sensor pretrain on uECoG (separate claim, separate program).
