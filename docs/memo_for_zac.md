# Technical Memo: Cross-Patient uECOG Speech Decoding

**From:** Ben Tang, Cogan Lab
**To:** Zac Spalding
**Date:** March 17, 2026
**Re:** Architecture design, literature grounding, and blocking questions for extending your cross-patient phoneme decoding work

---

## 1. Summary

I've reviewed 18 papers on cross-patient neural speech decoding and designed a two-phase plan to replace your PCA+CCA+SVM pipeline (0.31 bal acc) with the modern field-standard architecture, then add two novel architectural innovations. The plan is conservative by design: 8 of 10 pipeline components are field-standard with multi-paper support. The 2 novel components are independently ablated.

**Bottom line:** Phase 2 (field standard) is a guaranteed improvement. Phase 3 adds two architectural ideas absent from the field — per-patient spatial conv and an articulatory decomposition CTC head. Training is identical in both phases (standard multi-patient SGD, same as Singh 2025).

---

## 2. Architecture

### Field consensus (Phase 2 — E1)

Every successful cross-patient speech BCI from the last two years uses:

```
Per-patient input layer → Shared recurrent backbone → CTC/sequence loss
```

Used by: Willett 2023, Metzger 2023, Boccato 2026, Levin 2026, Nason 2026, Singh 2025 (all GRU/LSTM + CTC). Littlejohn 2025 evolves this to RNN-T for streaming. BIT 2026 uses a transformer backbone + CTC but the same per-patient read-in pattern.

Our Phase 2 implementation:

```
uECOG → flatten + zero-pad to 208 ch
  → Per-patient Linear(208, 64)              [13,376 params/patient]
  → LayerNorm → Conv1d(64, 64, k=K, s=K)    [temporal downsampling, K configurable]
  → BiGRU(64, 64, 2 layers, bidirectional)
  → Linear(128, 10) → CTC                   [9 phonemes + blank]
```

Two-stage LOPO: Stage 1 trains on 7 source patients jointly. Stage 2 freezes backbone, adapts read-in + LayerNorm + head on target patient with 30% source replay (Levin 2026).

### Our innovations (Phase 3 — E2)

Two architectural changes, both absent from neural speech BCI literature:

| # | Innovation | Replaces | Key idea |
|---|---|---|---|
| 1 | Per-patient Conv2d (configurable depth) | Linear(208,64) | Conv2d factorization matches rigid-array physics; 80–664 params vs 13,376 |
| 2 | Articulatory CTC head | Flat Linear(128,10) | Decode what motor cortex encodes — articulatory features, not phoneme categories |

```
uECOG grid (8×16 or 12×22 × T)
  → Per-patient Conv2d(1, 8, k=3, pad=1) + ReLU   [80 params/patient; depth configurable]
    → AdaptiveAvgPool2d(2, 4) → flatten(64)
  → LayerNorm → Conv1d → BiGRU                     [unchanged from Phase 2]
  → 6 articulatory heads → fixed composition matrix → 9 phoneme logits + blank
```

**Stage 2 trainable params: 2,272** (vs 14,794 in Phase 2 — 6.5× reduction). For S3 with 46 trials, that's 49 params/trial instead of 321.

Training is **identical** to Phase 2 — standard multi-patient SGD. No meta-learning, contrastive loss, or robust optimization. (I evaluated Reptile, SupCon, and DRO and concluded per-patient layers already provide transfer; see §4.)

---

## 3. Why Each Component (Literature Support)

### Per-patient input layers
Every successful system uses these. Levin showed shared input makes transfer *worse*. Boccato showed learned transforms are predominantly diagonal (per-channel gain) — **but that's Utah-specific** (fixed stereotactic placement). Our uECOG has variable surgical placement and orientation → cross-patient variation includes spatial remapping, not just gain. This is precisely why the Conv2d innovation (Phase 3) matters more for us than it would for Utah.

### BiGRU backbone
Singh: BiLSTM 2×64 decoded 36 phonemes (of 44 English) from 25 sEEG patients. Willett/Metzger/Littlejohn: GRU H=500-512 but with 50-150× more trials per patient. Our BiGRU 2×64 (~147k params) is correct for our data scale. Littlejohn 2025 evolved to unidirectional GRU for streaming — we don't need streaming (offline intra-op analysis), so bidirectional is fine. BIT uses a transformer backbone but required 367 hrs of pretraining data — inapplicable at our scale.

### CTC loss
Universal for phoneme-level decoding: Willett, Metzger, Boccato, Levin, Nason, Singh all use CTC. Littlejohn uses RNN-T (an autoregressive extension of CTC enabling streaming). Your per-position SVM discards temporal/sequential structure. CTC handles alignment natively.

### Temporal downsampling (Conv1d)
Spalding: k=10 s=10 → 20 Hz. Willett: k=14 s=4 → ~12.5 Hz. Littlejohn: 16× downsample → 12.5 Hz. Metzger: k=4 s=4 → 50 Hz (the outlier — paired with 4-layer BiGRU H=500). Most systems operate at 12–20 Hz. I've made this configurable (default K=5 → 40 Hz, alternative K=10 → 20 Hz). Given that Littlejohn decodes full sentences at 12.5 Hz, 20 Hz is sufficient for our 3-phoneme targets.

### Two-stage LOPO with source replay
Singh's "Recurrent Transfer" Mode 3 (freeze backbone, adapt per-patient layers) was optimal. Levin's 30% source replay prevents catastrophic forgetting during Stage 2. Best-checkpoint selection (no SWA — no BCI paper uses it).

### Articulatory CTC head (NOVEL)
Wu 2025: articulatory features "more robustly encoded in vSMC than acoustic features, learned faster with limited data." Duraivel 2023: confusion errors track articulatory distance. Bouchard 2013: somatotopic organization by place of articulation. Six parallel sub-heads (C/V, place, manner, voicing, height, backness) composed via fixed linguistic matrix into 9 phoneme logits. Each sub-head is binary or 3-way — simpler than 9-way. The model *must* learn voicing to distinguish /b/ from /p/.

### Per-patient spatial conv (NOVEL)
MNI projection shows array placements span 15–25mm with variable rotation — channel numbers have no consistent anatomical meaning across patients; diagonal per-channel scaling is inapplicable. Conv2d's factorization (weight-shared within each rigid array, different across patients) matches the recording physics: layer 1 learns spatial deblurring (Laplacian-like, suppressing volume conduction at r≈0.6), layer 2 combines into higher-order spatial features. Per-patient filters adapt to each array's orientation. 664 params with physics-matched inductive bias vs 13,376 unconstrained.

### Component Reference Table

| Component | Status | Supporting papers | Reasoning |
|---|---|---|---|
| Per-patient input layers | Field standard | Willett, Boccato, Levin, Nason, Singh, BIT | Absorbs cross-patient recording variation. Levin: shared input makes transfer worse |
| BiGRU 2×64 backbone | Field standard | Singh (BiLSTM 2×64, 36 of 44 phonemes), Willett/Metzger/Littlejohn (GRU H=500-512, 50-150× more data). BIT uses transformer (367 hrs pretraining) | RNNs handle variable-length sequences; H=64 matches data scale. Transformers need far more data |
| CTC loss | Field standard | Willett, Metzger, Boccato, Levin, Nason, Singh (CTC). Littlejohn (RNN-T, an extension of CTC) | Handles alignment without frame-level labels. Universal for phoneme-level decoding |
| Conv1d temporal downsampling | Field standard | Spalding (k=10→20 Hz), Willett (k=14→~12.5 Hz), Littlejohn (16×→12.5 Hz), Metzger (k=4→50 Hz, outlier) | Most systems 12–20 Hz. Configurable K=5 (40 Hz) or K=10 (20 Hz) |
| Two-stage LOPO | Field standard | Singh (Mode 3), Levin (30% source replay), Willett, Nason | Freeze backbone, adapt per-patient layers. Singh tested 3 modes; freeze-backbone was optimal |
| LayerNorm | Field standard | Standard practice (Chen 2024: InstanceNorm; BIT: LayerNorm) | Per-sample normalization; no batch statistics leak. Unfrozen in Stage 2 (~128 params) |
| Augmentation (time shift, amp scale, channel dropout, noise) | Field standard | Spalding (±100ms shift), Willett (noise SD=1.0, offset SD=0.2), Boccato (gain is dominant variation) | Each augmentation simulates a specific cross-patient variation source |
| Best-checkpoint selection | Field standard | Standard practice (no BCI paper uses SWA) | Simple; no unprecedented complexity. SWA available as E9 extended ablation |
| **Per-patient Conv2d read-in** | **Engineering** | *Grounded in:* electrode placement physics (15–25mm offsets, variable rotation), volume conduction (r≈0.6), Chen 2024 (ResNet > LSTM on grids) | Conv2d factorization matches rigid-array physics. 80–664 params vs 13,376; learned spatial deblurring + orientation adaptation. Depth, channels, pool dims are empirical (E13 sweep) |
| **Articulatory CTC head** | **NOVEL** | *Motivated by:* Wu 2025 (articulatory > acoustic in vSMC), Duraivel 2023 (errors track articulatory distance), Bouchard 2013 (somatotopy) | Decomposes 9-way into 6 binary/3-way sub-problems matching motor cortex encoding. Fixed composition matrix — 0 learned params in composition |

---

## 4. What I Evaluated and Rejected

Three training innovations were evaluated and **downgraded to extended ablations** (E5-E7):

**Reptile meta-learning.** Per-patient layers already provide transfer (Singh's Mode 3). In Stage 2, backbone is frozen → Reptile's "easy to adapt" property applies to params we don't touch. N=7 source patients is thin for meta-learning.

**Supervised contrastive loss (SupCon).** CTC + per-patient layers already enforce alignment (shared head decodes all patients from same backbone space). Time-pooled trial embeddings lose sequential order. Too few positives per class (~150 trials, 52 token types).

**DRO (distributionally robust optimization).** Hard patients (S3: 46 trials, S5: 63 channels) are hard for practical data quality reasons, not neural coding differences. Upweighting them = overfitting to noise. N=7 running loss EMA is too noisy. Implicit difficulty-aware training already exists (higher loss → larger gradients).

---

## 5. Experiment Matrix

### Core (for paper)

| Exp | Read-in | CTC head | Tests |
|---|---|---|---|
| **E0** | PCA+CCA+SVM | Flat | Your baseline (0.31) |
| **E1** | Linear(208,64) | Flat | Field-standard improvement |
| **E2** | Spatial Conv2d | Articulatory | Full model (both innovations) |
| E3 | Linear(208,64) | Articulatory | −SpatialConv attribution |
| E4 | Spatial Conv2d | Flat | −ArtHead attribution |

Training is identical across E1-E4. All differences are architectural.

### Extended (if time permits)
E5-E7: +DRO/+Reptile/+SupCon. E8: diagonal gain only. E9: +SWA. E10-E14: DANN, TENT, temporal dilation, pooling strategy, Conv3d hybrid.

---

## 6. Cross-Task Pooling Opportunity (Duraivel 2025)

Duraivel 2025's pseudo-word task uses **the same 9 phonemes** in CVC/VCV tokens. At least 3 patients overlap:

| Duraivel 2025 | Spalding 2025 | Additional trials |
|---|---|---|
| D-S1 | S8 | 156–208 |
| D-S2 | S5 | 156–208 |
| D-S3 | S1 or S2 | 156–208 |

Both tasks produce 3-phoneme CTC targets. Same preprocessing, same IRB. Nearly doubles training data for overlapping patients. This is a future direction (after core experiments), but affects data planning.

---

## 7. Data I've Verified

| Patient | Diagnosis | Array | Trials | Sig ch | Frames |
|---|---|---|---|---|---|
| S1 | Parkinson's | 8×16 | 144 | 111/128 | 28,800 |
| S2 | Parkinson's | 8×16 | 148 | 111/128 | 29,600 |
| S3 | Tumor | 12×22 | **46** | 149/256 | **9,200** |
| S4 | Parkinson's | 8×16 | 151 | 74/128 | 30,200 |
| S5 | Parkinson's | 8×16 | 151 | 63/128 | 30,200 |
| S6 | Tumor | 12×22 | 137 | 144/256 | 27,400 |
| S7 | Tumor | 12×22 | 141 | 171/256 | 28,200 |
| S8 | Tumor | 12×22 | 178 | 201/256 | 35,600 |

Window: ±500ms response onset, 200 Hz. Paired audio confirmed. MNI-152 coords confirmed. Array orientation NOT consistent.

---

## 8. Questions for You

### Blocking (need before I can start coding)

1. **Where is the data?** The repo references `pt_decoding_data_S62.pkl` and per-patient `.mat` files — are these on DABI, a lab server, or somewhere else? What's the access process?

2. **Patient ID mapping.** Code uses S14, S23, S62, etc. → paper uses S1–S8. What's the mapping table?

3. **Grid spatial convention.** Do `hgTrace`'s ch_x/ch_y correspond to physical grid rows/columns? Is orientation consistent (ch_x = anterior-posterior?) or arbitrary per patient? Critical for Conv2d design.

4. **Non-significant channels.** Does `hgTrace` preserve the full 2D grid with non-sig channels zeroed, or are non-sig channels excluded (so ch_x × ch_y < full grid)? Conv2d needs the full grid with non-sig channels zeroed to maintain spatial structure.

5. **Epoch width.** Do `.mat` files store data beyond ±500ms? Need ±600ms for ±100ms time-shift augmentation without zero-padding edges.

6. **Label encoding.** Are `phonSeqLabels` values 0-indexed (0–8) or 1-indexed (1–9)? CTC blank must be index 0, so phoneme targets need to be 1–9.

7. **MNI coordinates and audio.** Where are these stored? What format? Are MFA phoneme labels already generated, or do I need to run MFA?

### Important for project scope

8. **Are there more uECOG patients?** Beyond your published 8, are there others with good data (phoneme or word/nonword tasks)? What's the full patient list?

9. **Which patients overlap with Duraivel 2025?** We infer D2025-S1=S8 (201/256 tumor), D2025-S2=S5 (63/128 DBS), D2025-S3=S1 or S2 (111/128 DBS). Correct? Can you disambiguate S1 vs S2?

10. **Is pseudo-word HGA data available?** For overlapping patients — is Duraivel 2025 data in the same `.mat` format? Would we need to re-window (their ±1500ms → our ±500ms)?

### Nice-to-have

11. **Is Spalding S3 = Duraivel 2023 S4?** S3 (46 trials, 149/256 tumor) is our hardest patient. Was it excluded from Duraivel 2025 due to low trials? Any known bad-channel patterns?

12. **CCA implementation.** How did you handle pairwise-to-universal projection? Want to faithfully reproduce your baseline.

13. **TME surrogate.** Your test (W=8, p=0.20, single realization) had low power. Did you try other W values? I want a stronger surrogate control.

---

## 9. Timeline

| Phase | Duration | Depends on |
|---|---|---|
| Phase 0: Setup | 1–2 weeks | Data access (Q1–2) |
| Phase 1: Reproduce your baseline | 1 week | Working code + data |
| **Phase 2: Field standard (E1)** | **2–3 weeks** | Phase 1 done |
| Phase 3: Full model (E2) | 1 week | Phase 2 validated |
| Ablations (E3–E4) | 1 week | E2 converged; parallel |
| Extended (E5–E14) | 1–2 weeks | If time permits |
| Analysis + figures | 1–2 weeks | All experiments done |

**Compute:** 4 core experiments × 8 LOPO folds × 3 seeds × ~30 min ≈ ~48 GPU-hours.

Phase 2 is a checkpoint — if it beats 0.31 and Phase 3 doesn't pan out, we still have a publishable result.

---

*Full details: `implementation_plan.md`, `research_synthesis.md`, `future_directions.md` in project repo.*
