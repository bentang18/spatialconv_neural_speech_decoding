# Paper Index — Cross-Patient Speech Decoding Research (19 papers)

Organized by **contribution category → papers**. Papers may appear in multiple categories.
All summaries in `summaries/<paper_name>.md`. Archived papers (16) in `archive/summaries/`.

---

## 1. Cross-Patient / Cross-Subject Alignment

| Paper | Alignment Method | Modality | Key Result |
|-------|-----------------|----------|------------|
| spalding2025 | PCA + CCA | uECOG | 0.31 bal. acc. (9-way phoneme), 8 patients |
| boccato2026 | Affine transforms (end-to-end) | Utah arrays | PER 16.1% (T12), adapts with ~66k params |
| levin2026 | Per-session affine + softsign | Utah arrays | Transfer helps below ~200 sentences |
| singh2025 | Conv1D per-patient + shared LSTM | sEEG | Group PER 0.49 vs single 0.57, 25 patients |
| MIBRAIN_2025 | Brain-region prototypes + MAE | sEEG | Above-chance zero-shot, scales with N≥6 |
| zhang2026_BIT | Per-subject read-in + shared transformer | Utah arrays | WER 5.10%; SSL advantage is cross-subject only (Table 9) |
| boccato2026_brainwhisperer | Pretrained Whisper + per-session LoRA | Utah arrays | WER 8.7% E2E (SOTA); cross-dataset improves without fine-tuning |
| wu2025 | TCA articulatory features (cross-speaker) | HD-ECoG | PCC 0.80/0.75/0.76, generalizable articulatory targets |
| chen2025_swinTW | MNI coordinate PE (no per-pt decoder layers) | ECoG sub + sEEG | PCC 0.837 multi-pt; 0.765 LOO unseen; N=52 |

## 2. Cross-Session Stability

| Paper | Method | Key Result |
|-------|--------|------------|
| willett2023 | Day-specific input layers | WER 9.1% (50-word) |
| nason2026 | Day-specific dense input layers | WER 19.6% (125k-word), 2 years |

## 3. Foundation Models / SSL

| Paper | Approach | Scale | Key Result |
|-------|----------|-------|------------|
| zhang2026_BIT | Masked modeling on spikes | 367 hrs, cross-species | WER 5.10%; SSL same as supervised for same-subject (Table 9) |
| boccato2026_brainwhisperer | Pretrained ASR (Whisper 680k hrs audio) | Cross-dataset MEA | WER 8.7% E2E; Whisper phoneme layers transfer to neural data |
| wav2vec_ecog_ssl_2024 | wav2vec contrastive on ECoG | ~1 hr/patient | 60% WER reduction; cross-pt 14.6% WER gain |

## 4. Coordinate-Based / Spatial Architectures

| Paper | Method | Modality | Key Result |
|-------|--------|----------|------------|
| chen2024 | 3D ResNet on ECoG grids | ECoG sub, N=48 | PCC 0.806; ResNet >> LSTM (0.745) |
| chen2025_swinTW | MNI coordinate tokenization | ECoG sub + sEEG | PCC 0.837 multi-pt = 0.831 individual; 0.765 LOO; N=52 |

## 5. uECOG-Specific

| Paper | Contribution |
|-------|-------------|
| duraivel2023 | uECOG speech decoding fundamentals: 71% 4-way vowel, 50% 9-way phoneme; HGA pipeline |
| duraivel2025 | Pseudo-word repetition task (CVC/VCV, same 9 phonemes); 52 epilepsy + 3 uECOG patients; planning→execution neural dynamics; spatial gradient on uECOG |
| spalding2025 | Only cross-patient uECOG study; PCA+CCA; 8 patients, phoneme repetition |
| qian2025 | 256-ch uECOG, 394 Mandarin syllables, 71.2% accuracy, real-time 49.7 CPM |

## 6. High-Performance Speech BCIs

| Paper | Setup | Key Result |
|-------|-------|------------|
| willett2023 | Utah arrays, chronic | WER 9.1% (50-word), 23.8% (125k-word) |
| metzger2023 | ECoG 253-ch, chronic | WER 25.5% (1024-word), 78 WPM, avatar |
| littlejohn2025 | ECoG 253-ch, chronic | PER 10.8% (50-phrase speech), streaming 47.5 WPM |
| nason2026 | Utah 64-ch, chronic | WER 19.6% (125k-word), dysarthria, 2 years |
| qian2025 | HD-ECoG 256-ch, intra-op | 71.2% syllable acc (394 syllables), 49.7 CPM |
| chen2024 | ECoG subdural 8×8, N=48, intra-op | PCC 0.806 (ResNet), 18-param speech synthesis |
| chen2025_swinTW | ECoG sub + sEEG, N=52 | PCC 0.825 (SwinTW); multi-pt 0.837; LOO 0.765 |

## 7. Low-Data / Rapid Adaptation

| Paper | Data Amount | Result |
|-------|------------|--------|
| dual_pathway_ecog_2025 | ~20 min ECoG/patient | WER 18.9%, R²=0.824 |
| nason2026 | ~36 sentences (~6 min) | Reaches near-full performance |
| levin2026 | <200 sentences | Cross-brain helps in low-data regime |
| spalding2025 | ~8 min/patient | Cross-patient CCA works at this scale |

## 8. Per-Patient Input Layers (Architecture Pattern)

| Paper | Layer Type | Position | Params |
|-------|-----------|----------|--------|
| boccato2026 | Affine (W*x + b) | Before backbone | ~66k (256ch) to ~262k (512 features) |
| levin2026 | Affine + softsign | Before backbone | ~262k per speech session |
| willett2023 | **Affine + softsign** (256→256, same-dim) | Before backbone | **65.8k/day** (not dimensionality-reducing) |
| nason2026 | Day-specific dense | Before backbone | ~66k |
| singh2025 | Conv1D per-patient | Before backbone | Variable (hyperparams not specified) |
| zhang2026_BIT | Linear read-in/out | Before backbone | Variable |
| boccato2026_brainwhisperer | Month full-rank W + day low-rank A·B | Before encoder | Month: C×C, Day: C×R×R×C (R<<C) |

## 9. Loss Functions

| Loss | Papers |
|------|--------|
| CTC | willett2023, metzger2023, littlejohn2025, boccato2026, boccato2026_brainwhisperer, levin2026, nason2026, zhang2026_BIT, spalding2025 |
| Cross-entropy | duraivel2023, duraivel2025 (SVD-LDA), spalding2025 (SVM), singh2025, MIBRAIN_2025, qian2025 |
| InfoNCE / contrastive | wav2vec_ecog_ssl_2024 |
| MSE (regression) | dual_pathway_ecog_2025, wu2025, chen2024 |
| Spectral (MSS + STOI+) | chen2024 |
| Masked reconstruction | zhang2026_BIT, MIBRAIN_2025 |

## 10. HGA Feature Extraction Pipeline

Cogan lab standard (duraivel2023, duraivel2025, spalding2025):
1. Decimate 2 kHz → CAR → impedance exclusion (>1 MOhm)
2. Bandpass 70-150 Hz (8 log-spaced Gaussian filters)
3. Hilbert envelope → average across bands → downsample 200 Hz
4. Baseline normalize (500 ms pre-stimulus)
5. Channel selection: HG-ESNR permutation test (FDR p<0.05)

---

## Quick-Find

| Question | Papers |
|----------|--------|
| uECOG work? | duraivel2023, duraivel2025, spalding2025, qian2025 |
| Cross-patient methods? | boccato2026, levin2026, singh2025, wu2025 |
| Best speech BCI? | willett2023, littlejohn2025 |
| SSL + limited data? | wav2vec_ecog_ssl_2024, zhang2026_BIT, boccato2026_brainwhisperer |
| Pretrained ASR for neural? | boccato2026_brainwhisperer (Whisper → MEA) |
| Zero-shot transfer? | MIBRAIN_2025 |
| ~20 min data? | dual_pathway_ecog_2025, spalding2025 |
| Spatial encoding? | chen2024, chen2025_swinTW |
| Cross-task alignment? | **No paper exists — your gap.** duraivel2025 provides the pseudo-word task data; same 9 phonemes + overlapping uECOG patients with Spalding |
| Articulatory alignment? | wu2025 (TCA), duraivel2023, duraivel2025 (phonotactic transitions in motor cortex) |
| Speech parameter synthesis? | chen2024, chen2025_swinTW, dual_pathway_ecog_2025 |
| Large cohort (N≥20)? | chen2024 (N=48), chen2025_swinTW (N=52), singh2025 (N=25) |
| Source replay during fine-tuning? | levin2026 (30% source / 70% target, uniform across source days) |
| Training protocol gaps? | singh2025 (no LR/optimizer reported), levin2026 (hyperparams in external spreadsheet) |
