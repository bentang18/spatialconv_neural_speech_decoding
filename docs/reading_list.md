# Reading List — Cross-Patient uECOG Speech Decoding

Read in order. Each paper teaches something the next one builds on. (10 papers)

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

### 10. Wu 2025 — Articulatory features as alignment targets
`pastwork/summaries/wu2025_articulatory_reconstruction.md`

Articulatory features extracted via TCA from EMA data, reconstructed from HD-ECoG with PCC 0.75–0.80 across 8 speakers. Articulatory dynamics are cross-speaker invariant because all humans share the same articulators.

**You need this for:** Phase 3.2 (articulatory auxiliary loss). The theoretical basis for predicting Chomsky-Halle phonological feature vectors from GRU hidden states as a regularizer toward cross-patient-invariant representations.
