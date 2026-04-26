# Session — Spatial Architecture Brainstorm (2026-04-19)

Brainstorm: how to handle cross-patient spatial structure in v14. Does the current
additive parcel embedding on cell tokens give up too much cross-patient sharing?
Is pooling to parcel tokens the right answer? Where is the middle?

## Starting position

Current v14-core:
- Grid scatter → Conv2d(1→8) → pool (4,8) = 32 cells
- Per-cell Conv1d(8→32, k=30, s=10) → cell tokens (32, 13, d=32)
- `pooled_support @ P_emb[15, d]` added as bias to cell features
- Combined attention backbone (3 layers, 2 heads × 16, FFN 128)
- D1 readout = global mean over all (cell × time) tokens + prev_phoneme
- **Zero per-patient calibration parameters**

Phase-2+ target (B-1 in contract): same pipeline with no (4,8) pool — per-electrode
tokens, shared interface via soft parcel embedding.

## Core tradeoff

- **Expressive (per-cell tokens, additive parcel bias)** — patient-specific spatial identity is preserved; parcel signal is a whisper the backbone can ignore.
- **Shared story (pool-to-parcels, 15 tokens)** — every patient has the same 15 tokens; within-parcel structure collapsed.

**Binding constraint:** sub-parcel cross-patient alignment is below SNR (reason #26 was killed). The shared story is parcel-level *only*; the expressive story is per-patient *only*. They must live on different axes.

## uECoG ≠ sEEG for within-parcel expressivity

Ben's correction, load-bearing for our design:
- sEEG: ~5mm contact spacing, 1–5 contacts per parcel per patient. Pool-to-parcel fine (MIBRAIN, BaRISTA).
- uECoG: 200 µm – 1.7 mm spacing. Within A4hf alone a patient has 20–40 electrodes. Sub-parcel somatotopy is resolvable (Duraivel 2023 on uECoG specifically; Bouchard 2013, Chartier 2018 on ventral SMC). Pooling destroys task signal.

Implication: MIBRAIN's single-stream parcel-pool pattern is right for sEEG but wrong for uECoG. Dual-stream (preserve cells + add parcel latents) is justified for our modality.

## Misjudgments owned (important for future-me)

1. **"Current design is an outlier" — wrong.** BaRISTA (parcel embedding +8–10 pp, zero per-patient params) validates the additive pattern. We are in BaRISTA's family, not an outlier.
2. **"POYO's 16 latents ≈ our 15 parcels" — rhetoric, not evidence.** POYO latents are emergent unit-positional anchors, not labeled anatomical parcels; POYO runs on 100 h + per-neuron embeddings.
3. **MIBRAIN is the closest prior (11 patients, ~1500 trials/patient, per-region encoders + prototypes + per-subject heads) — and it does single-stream parcel pooling, not dual-stream.** My dual-stream proposal is more ambitious than the closest published precedent. Defensible for uECoG (see above) but cannot cite MIBRAIN for it.
4. **Per-patient calibration entirely missing from early rounds.** Every HGA/field-potential speech decoder with good cross-subject transfer uses per-patient layers (BIT, Singh, Levin, MIBRAIN, seegnificant, Chen 2025). This is the biggest gap, independent of the parcel-token debate.
5. **NDT3's channel-shuffle result doesn't transfer.** Spike data sample random neuron subsets per channel — cross-subject correspondence is fundamentally broken. HGA is population-averaged over ~mm² and is much more cross-subject-comparable. NDT3's "sensor variability is THE bottleneck" needs modality-dependent discounting.
6. **BaRISTA's dilated CNN is for raw 2048 Hz → 250 ms windows.** We have 200 Hz HGA already. The Conv1d(k=30, s=10) at 150 ms kernel / 50 ms hop is already pre-matched to speech timescale and validated at PER 0.734. Not a reference for changing our temporal front-end.
7. **Register tokens at 47 k params is unsupported by in-modality evidence.** TMLR reproducibility (Bach 2025) shows modest gains at small ViT scale. We have labeled parcel latents that already serve the shared-aggregator role.
8. **FiLM, AdaIN, AdaIN-style patient-stat normalization — zero in-modality citations.** Dropped from defaults. Ablation candidates only.

## Literature map — what actually grounds our design

Numbers-heavy convergence from `pastwork/summaries/`:

| Primitive | In our design | Strong precedent (modality) |
|-----------|---------------|------------------------------|
| Per-patient calibration | Diagonal γ,β + softsign | Singh (sEEG HGA), BIT (Utah), Levin (Utah), MIBRAIN (sEEG raw), Chen 2025 (ECoG, per-patient speech encoder) |
| Atlas parcel embedding | `log(support)` as attention prior | BaRISTA (sEEG, +8–10 pp), MIBRAIN (region prototypes), BrainROI (fMRI soft-ROI matrix), BrainGFM (fMRI [A/P] tokens) |
| Labeled latent tokens via cross-attention | 15 parcel latents, Q = P_emb[15, d] | POYO (16 unit-latents, emergent), Charmander (32 virtual channels, learned), Set Transformer PMA, Perceiver IO |
| Combined space-time attention (not factored) | 3 layers, RoPE on time | BaRISTA (combined > factored +1–2 pp at our scale) |
| Readout from shared-identity tokens | Parcel-latent mean-pool | POYO (session-embedding queries), MIBRAIN (per-subject MLP heads), DETR (object queries), BIT (subject-specific read-out) |
| Temporal Conv1d front-end | k=30, stride=10 (kept) | Ben's 0.734 baseline; matches HGA timescale |
| Preserve within-parcel electrode-level structure | Cells + parcel latents dual-stream | Chen 2025 SwinTW on ECoG (per-electrode tokens + MNI coord PE, multi-subject = individual PCC 0.837 vs 0.831); Duraivel 2023 uECoG somatotopy |

Data-scale warnings:
- **Jiang 2025 (heterogeneity limits scaling):** 5 selected sessions beat 40 random. Patient selection > count at our scale. Temporal masking scales with heterogeneity; spatial masking does not.
- **NDT3:** pretraining benefit vanishes with 1.5 h downstream data. We have far less (1 min/patient epoched) — SSL should help.
- **MIBRAIN:** adding 1–3 subjects initially *hurts*; consistent improvement from 6+. Expect a curriculum, not monotone scaling.

## Converged ideal architecture

```
signal (B, N_e, 130)                              [200 Hz HGA, phoneme window]
   │
   │  [1]  Per-patient per-electrode diagonal + softsign
   │       γ, β initialized to identity, ~2·N_e params/patient
   ▼
   │  [2]  Grid scatter → Conv2d(1→8, k=3, pad=1) + GELU   (kept from baseline)
   ▼
   │  [3]  Masked-mean pool to (4, 8) = 32 cells           (Phase-1 compromise)
   ▼
   │  [4]  Per-cell Conv1d(8→32, k=30, stride=10)          (kept, validated 0.734)
   ▼
cell tokens (B, 32, 13, d=32)
   │
   │  [5]  Spawn 15 labeled parcel latents P_emb[15, d] per time-step
   │       Q = P_emb, K,V = cell tokens at that t
   │       attention bias = log(support_cell→parcel)
   ▼
parcel latents (B, 15, 13, d)
   │
   │  [6]  Concat → [cells | parcels] = 47 spatial × 13 time = 611 tokens
   │  [7]  3 × combined attn + FFN, 2 heads × 16, FFN 128, RoPE on time only
   │  [8]  Readout from 15 parcel latents: mean-time → mean-parcel → +prev_phoneme → Linear(d → 9)
```

### Three changes vs. current contract, priority order

1. **Per-patient diagonal + softsign calibration.** Closes the HGA speech-decoder consensus gap. Diagonal not full affine (201-electrode patients would be 40 k params). Initialized to identity, softsign per Levin. Counter-defense: if ablation shows diagonal underpowered, upgrade to full affine.
2. **Cross-attention parcel pool with log-support bias.** Replaces `pooled_support @ P_emb` additive bias. Strict generalization — collapses to current pool as Q,K → 0. Parcel identity moves from bias the backbone can ignore to tokens the readout must route through.
3. **Dual-stream backbone + parcel-latent readout.** Preserves uECoG within-parcel somatotopy (cells) while giving shared transfer-ready interface (parcel latents). 611 tokens vs 416 — 2.2× attention, still cheap.

### What's unchanged from current contract

- Grid scatter + Conv2d(1→8) — within-patient grid conv stays
- Pool (4,8) = 32 cells — Phase-1 compromise
- Per-cell Conv1d(8→32, k=30, s=10) — matches speech timescale
- 3-layer combined attention, 2 heads × 16, FFN 128, RoPE on time
- d_model = 32, dropout 0.1
- Phoneme-centered [-0.15, 0.5) s window, flat per-phoneme CE, teacher forcing, 9³ exhaustive AR decode

### What was dropped (and why)

- **FiLM, AdaIN, patient-stat normalization** — no in-modality citation. Ablation candidates only.
- **Register tokens** — modest gain at small ViT scale; labeled parcel latents already serve this role.
- **Single-stream parcel-only (Path B / MIBRAIN pattern)** — destroys uECoG within-parcel expressivity; appropriate for sEEG, wrong here.
- **Replacing Conv1d front-end with patch embedding or dilated CNN** — BaRISTA dilated CNN is for 2048 Hz raw voltage; our 200 Hz HGA is already well-matched to the current Conv1d.

## Forward compatibility to Phase 2+

Only the input side flexes; everything from [5] onward is identical byte-for-byte.

| Stage | Phase 1 | Phase 2+ (B-1) |
|-------|---------|----------------|
| [1] per-patient γ,β | per electrode | **unchanged** |
| [2] grid scatter + Conv2d | present | **removed** |
| [3] pool (4,8) = 32 cells | present | **removed** |
| [4] temporal Conv1d | 8→32 per cell | **1→32 per electrode** |
| [5] cross-attn parcel pool | Q=P_emb, K,V=32 cells | Q=P_emb, K,V=N_e electrodes |
| [5'] attention bias | `log(pooled_support)[32,15]` | `log(support)[N_e,15]` |
| [6–8] backbone + readout | **unchanged** | **unchanged** |

Cross-attention is shape-polymorphic in K,V — output shape set by queries (15 × 13). The parcel-latent interface at [5,8] is the cross-patient contract. Transfer (SSL pretrain → finetune, LOPO warm-start, cross-sensor) happens on those 15 tokens regardless of whether the input is 32 cells or 200 electrodes.

Compute scaling: Phase-1 611 tokens → Phase-2+ at 128 ch 1859 tokens (3 × quadratic = 10 × entries). At 256 ch 3523 tokens (33 ×). Compute-scaling problem, not architectural rework.

## Open ablations

- **Single-stream cells (current + per-patient calibration) vs. dual-stream (cells + parcel latents).** If dual-stream doesn't move PER or LOPO transfer, collapse.
- **Per-patient diagonal vs. full affine + softsign.** Levin baseline vs. data-scale concession.
- **Single spawn at input vs. re-spawn per backbone layer.** POYO uses iterative cross-attention. Start with single spawn.
- **Parcel-latent mean-pool readout vs. learnable output query (POYO-style).** Simplicity first.
- **Support-weighted mean (current) vs. attention pool.** Direct architectural A/B. Expect strict-generalization property to hold.

## Canonical-goal reminder (do not over-fit to Phase 1)

Every architectural choice must pay rent against the cross-patient / cross-sensor roadmap:
Phase 1 uECoG supervised → Phase 1.5 lexical-supervised + SSL → Phase 2 learned calibration → sEEG join → external corpora.

Ablation wins at 4 patients that rely on rigid-grid assumptions are local optima. The parcel-latent interface at [5,8] is explicitly forward-compatible; the dual-stream choice is explicitly justified by uECoG modality; the per-patient diagonal is explicitly chosen to scale to 128+ electrodes.

## Session sources (most load-bearing)

- BaRISTA (Oganesian 2025, NeurIPS) — `pastwork/summaries/oganesian2025_barista.md` — parcel embedding +8-10 pp, zero per-patient on binary detection
- MIBRAIN (Wu 2025, bioRxiv) — `pastwork/summaries/MIBRAIN_2025.md` — closest prior at 11 patients, region prototypes + per-subject heads
- Chen 2025 SwinTW — `pastwork/summaries/chen2025_swinTW_multisubject.md` — ECoG multi-subject = individual via coord PE
- Singh 2025 — `pastwork/summaries/singh2025_cross_subject_seeg.md` — Conv1D per-subject + shared LSTM + per-subject readout
- BIT (Zhang 2026, ICLR) — `pastwork/summaries/zhang2026_BIT_foundation_model.md` — subject-specific read-in/out, cross-species
- Levin 2026 — `pastwork/summaries/levin2026_cross_brain_transfer.md` — session-specific affine *enables* cross-brain transfer
- Charmander (Mahato 2025) — `pastwork/summaries/mahato2025_charmander.md` — Perceiver bottleneck, per-channel learnable embeddings
- POYO / POYO+ (Azabou 2023/2025) — latent-token cross-attention, session-embedding queries
- Jiang 2025 — `pastwork/summaries/jiang2025_heterogeneity_scaling.md` — heterogeneity limits scaling, selection > count
- PopT (Chau 2025, ICLR) — `pastwork/summaries/chau2025_population_transformer.md` — zero per-patient, 3D coord PE, binary detection
- Set Transformer (Lee 2019) — PMA as the math for labeled-query cross-attention pool
- Perceiver IO (Jaegle 2021) — latent bottleneck, output queries
- Registers in ViTs (Darcet 2024) — high-norm scratchpad problem at scale
- FiLM (Perez 2018) — multiplicative conditioning; γ is load-bearing
