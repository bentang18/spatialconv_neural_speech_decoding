# Summary Verification Results (March 2026)

> **Status: ALL ERRORS CORRECTED.** 19 errors found across 8 of 27 summaries; all fixed on 2026-03-04. 3 new papers (wu2025, benticha2025, qian2025) added after verification and not yet verified against PDFs.

## Batch 1 (spalding2025 through littlejohn2025)

| Paper | Verdict | Errors |
|-------|---------|--------|
| spalding2025_cross_patient_uecog | PASS | 0 |
| levin2026_cross_brain_transfer | PASS | 0 |
| boccato2026_cross_subject_decoding | PASS | 0 |
| zhang2026_BIT_foundation_model | PASS | 0 |
| duraivel2023_uecog_speech_decoding | PASS | 0 |
| willett2023_speech_neuroprosthesis | PASS | 0 |
| metzger2023_avatar_neuroprosthesis | PASS | 0 |
| jude2025_locked_in_speech_bci | PASS | 0 |
| littlejohn2025_brain_to_voice | FAIL | 5 |

### Detailed Errors — littlejohn2025 (MAJOR)
Error rates for 50-phrase-AAC and 1024-word-General are **systematically swapped** in the summary.

Correct values from PDF:

| Metric | 50-phrase-AAC Speech | 1024-General Speech | 50-phrase-AAC Text | 1024-General Text |
|--------|---------------------|--------------------|--------------------|-------------------|
| PER | 10.8% | 45.3% | 7.58% | 23.9% |
| WER | 12.3% | 58.8% | 10.3% | 31.9% |
| CER | 11.2% | 44.7% | 7.23% | 22.8% |

Also: auditory feedback P-value should be 0.172 (not 0.09), and "phrases" should be "sentences".

## Batch 2 (nason2026 through karpowicz2025)

| Paper | Verdict | Errors |
|-------|---------|--------|
| nason2026_dysarthria_bci | PASS | 0 |
| pandarinath2018_LFADS | FAIL | 2 |
| silva2024_speech_neuroprosthesis_review | PASS | 0 |
| mathis2024_decoding_brain_review | PASS | 0 |
| kneeland2026_ENIGMA_eeg_to_image | PASS | 0 |
| kunz2025_inner_speech | FAIL | 3 |
| singh2025_cross_subject_seeg | FAIL | 2 |
| wav2vec_ecog_ssl_2024 | FAIL | 2 |
| karpowicz2025_NoMAD | FAIL | 2 |

### Detailed Errors

**pandarinath2018_LFADS:**
1. Summary says `beta1=0.999` — PDF says `beta1=0.9, beta2=0.999`. Wrong assignment.
2. Summary says LR decayed "when validation error increases" — PDF says when *training error* does not decrease.

**kunz2025_inner_speech:**
1. Summary says "all with ALS or related conditions" — T16 actually has tetraplegia from **pontine stroke**, not ALS.
2. T16 counting task p-value: summary says `1.57e-6`, PDF says `1.5684e-08` (off by 100x).
3. T15-55b attempted vocalized: summary says 72.1%, PDF shows ~85.7%.

**singh2025_cross_subject_seeg:**
1. Summary says "all 44 English phonemes" — PDF says **36 of 44** phonemes.
2. Group model significance: summary says `p < 0.0001`, PDF says `p < 0.001`.

**wav2vec_ecog_ssl_2024:**
1. Summary describes decoder as "GRU-based LSTMs" — it's GRUs (LSTMs were replaced).
2. Participant c improvement range: summary says 7-24%, PDF shows 7-**35%**.

**karpowicz2025_NoMAD (MAJOR):**
1. Baseline R² values (Aligned FA 0.749, ADAN 0.916, Static 0.842) are **Day 0 within-day baselines**, NOT cross-session results. Actual cross-session: Aligned FA ~0.59, ADAN ~0.65, Static ~0.14. This dramatically understates NoMAD's advantage.
2. "Static decoder: 223 failures" — this number doesn't appear in the PDF. Actual numbers: 51 (Aligned FA iso), 78 (Static reaching), 53 (Aligned FA reaching).

## Batch 3 (SPINT_2025 through brain_foundation_models_survey_2025)

| Paper | Verdict | Errors |
|-------|---------|--------|
| SPINT_2025 | PASS | 0 |
| TNVAE_2025_consciousness | PASS | 0 |
| dual_pathway_ecog_2025 | PASS | 0 |
| MIBRAIN_2025 | FAIL | 2 |
| NDT3_2025 | PASS | 0 |
| NeurIPT_2025 | PASS | 0 |
| RPNT_2026 | PASS | 0 |
| POSSM_2025 | PASS | 0 |
| brain_foundation_models_survey_2025 | FAIL | 1 |

### Detailed Errors

**MIBRAIN_2025:**
1. Summary says "23 consonant initials (plus 2 simple finals 'i' and 'u')" — the "(plus 2 simple finals)" is not in the PDF. It's just 23 consonant initial classes.
2. Multi-sub exception for participant 4: summary implies multi-sub specifically failed, but PDF says **both** MIBRAIN variants failed for participant 4.

**brain_foundation_models_survey_2025:**
1. Summary says "Table II, 24 models" — table actually has **25 models** (summary lists all 25 names but states count as 24).

## Overall Summary (27/27 verified)

- **PASS: 19** (spalding2025, levin2026, boccato2026, zhang2026_BIT, duraivel2023, willett2023, metzger2023, jude2025, nason2026, silva2024, mathis2024, kneeland2026, SPINT_2025, TNVAE_2025, dual_pathway_2025, NDT3_2025, NeurIPT_2025, RPNT_2026, POSSM_2025)
- **FAIL: 8** (littlejohn2025, pandarinath2018, kunz2025, singh2025, wav2vec_ecog_2024, karpowicz2025, MIBRAIN_2025, brain_fm_survey_2025)
- **Total errors found: 19** across 8 papers

### Critical errors to fix:
1. **littlejohn2025** — Error rates for 50-phrase vs 1024-word sets systematically swapped
2. **karpowicz2025_NoMAD** — Day 0 baselines presented as cross-session results
3. **kunz2025** — T16 condition wrong (pontine stroke, not ALS), p-value off by 100x
4. **singh2025** — "44 phonemes" should be "36 of 44"
