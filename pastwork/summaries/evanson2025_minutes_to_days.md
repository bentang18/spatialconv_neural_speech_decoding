# Evanson et al. 2025 -- From Minutes to Days: Scaling Intracranial Speech Decoding with Supervised Pretraining

## Citation

Evanson, L., et al. (2025). From Minutes to Days: Scaling Intracranial Speech Decoding with Supervised Pretraining. *Imaging Neuroscience*. (Rothschild Hospital, Paris)

## Setup

- **Patient count:** 3 sEEG subjects (epilepsy, Rothschild Hospital Paris)
- **Recording modality:** sEEG, 141-230 channels per subject
- **Brain region:** Distributed sEEG coverage (clinical placement)
- **Pretrain data:** 83-108h per subject from week-long clinical monitoring (daytime 6:00-23:00). Ambient room audio captured from camera microphone, paired with neural recordings.
- **Task data:** Audiobook listening, 43-250 min per subject
- **Key innovation:** Supervised contrastive pretraining using ambient audio as paired signal -- NOT self-supervised

## Architecture

Defossez 2023 "brainmagick" ConvNet:
- Linear input projection -> convolutional blocks with skip connections -> Bahdanau attention pooling -> d-dimensional output embedding
- CLIP-style contrastive loss: brain embedding matched against wav2vec2-large-xlsr-53 layer 19 embeddings (frozen audio encoder)
- 3s non-overlapping windows
- Pretrain on ambient audio pairs, fine-tune on audiobook listening task

## SSL/Pretraining

**Supervised contrastive pretraining (NOT SSL).** Brain recordings are paired with concurrent ambient audio from the hospital room camera mic. The model learns to match brain activity windows to corresponding audio embeddings via contrastive loss (CLIP-style). This requires paired audio -- it is supervised by the ambient sound signal.

Key pretraining findings:
- Pretrain+finetune significantly outperforms train-from-scratch baseline (all 3 subjects, p<0.017)
- Log-linear scaling with pretrain hours -- no plateau observed at 100h
- Zero-shot fails completely (retrieval rank ~0.50, chance)
- Supervised pretrain >> SSL pretrain (beats PopT, BrainBERT)
- Night data provides no benefit (ambient audio is silence/noise)
- Pretraining benefit persists across fine-tuning data amounts

## Cross-Patient

None. Single-patient models only. No cross-patient alignment attempted. Authors explicitly identify cross-patient alignment as an unsolved bottleneck and key future direction.

Additional cross-patient-relevant finding: **Cross-day drift is massive** -- recording date is decodable from neural features with r=0.95. This demonstrates that per-session/per-day normalization is essential even within a single patient, validating per-patient normalization layers in cross-patient models.

## I/O Features

| Component | Detail |
|-----------|--------|
| **Input signal** | Broadband sEEG or gamma bipolar (70-120Hz) |
| **Preprocessing** | Bipolar montage, downsampled, z-scored |
| **Input window** | 3s non-overlapping |
| **Channels** | 141-230 (full clinical sEEG) |
| **Output** | d-dimensional embedding (matched against wav2vec2 audio embedding) |
| **Loss** | CLIP-style contrastive (brain vs audio) |
| **Audio encoder** | wav2vec2-large-xlsr-53 layer 19 (frozen) |
| **Eval metric** | Top-10 retrieval accuracy (brain -> audio matching) |

Gamma bipolar (70-120Hz) improved performance for 2/3 subjects over broadband, supporting HGA as preferred feature.

## Key Results

| Metric | Value | Condition |
|--------|-------|-----------|
| Pretrain benefit | Significant for all 3 subjects | p<0.017, vs train-from-scratch |
| Scaling law | Log-linear with pretrain hours | No plateau at 100h |
| Zero-shot | Fails (rank ~0.50) | Cannot skip fine-tuning |
| Supervised pretrain vs SSL | Supervised >> SSL | Beats PopT, BrainBERT |
| Night data | No benefit | Ambient audio is noise |
| Gamma bipolar vs broadband | HGA improved 2/3 subjects | 70-120Hz bipolar |
| Cross-day drift | r=0.95 | Date decodable from neural features |
| Pretrain data per subject | 83-108h | Daytime clinical monitoring |
| Task data per subject | 43-250 min | Audiobook listening |

## v12 Comparison

| Dimension | Evanson 2025 | v12 |
|-----------|-------------|-----|
| **Patients** | 3 (single-patient models) | 4 core + 7 extended + sEEG |
| **Modality** | sEEG (141-230 ch) | uECoG (128-256 ch) + sEEG |
| **Pretrain type** | Supervised contrastive (paired audio) | SSL temporal span masking (BIT-style) |
| **Pretrain data** | 83-108h/subject | ~24h total (7.6h uECoG + 16.7h sEEG) |
| **Task data** | 43-250 min/subject | ~1 min utterance/patient |
| **Cross-patient** | None | Core design goal (VE cross-attention) |
| **Per-patient params** | Full model per patient | 134 (diagonal norm + delta/omega) |
| **Signal** | Broadband or gamma bipolar 70-120Hz | HGA 70-150Hz |
| **Scaling** | Log-linear, no plateau at 100h | Unknown (to be tested) |

Key implications for v12:
1. **Validates data scaling direction.** Log-linear scaling with no plateau at 100h is encouraging -- our ~24h pool should provide meaningful pretrain signal, though 7.6h uECoG alone may be insufficient.
2. **HGA > broadband** confirmed (2/3 subjects), consistent with v12's HGA-only design.
3. **Per-patient normalization essential** -- cross-day drift r=0.95 within a single patient validates v12's per-patient diagonal normalization.
4. **Supervised pretraining unavailable for us.** We lack paired audio for continuous recordings (only epoched data has MFA labels). Our SSL approach (temporal span masking) is the feasible alternative, though Evanson shows supervised >> SSL.
5. **Zero-shot failure** confirms v12's fine-tuning requirement -- per-patient layers cannot be skipped.
6. **Cross-patient gap.** Evanson explicitly identifies cross-patient as unsolved. v12's VE cross-attention + atlas mapping addresses exactly this gap.

## Regime Table

| Dimension | Value |
|-----------|-------|
| Patients | 3 (single-patient only) |
| Channels | 141-230 (sEEG) |
| Signal | Broadband or gamma bipolar 70-120Hz |
| Pretrain hours | 83-108h/subject (ambient audio paired) |
| Task hours | 0.7-4.2h/subject (audiobook listening) |
| Architecture | Defossez 2023 ConvNet + Bahdanau attention |
| Pretrain loss | CLIP contrastive (brain vs wav2vec2 embeddings) |
| Task loss | CLIP contrastive (brain vs wav2vec2 embeddings) |
| Eval metric | Top-10 retrieval accuracy |
| Scaling | Log-linear, no plateau at 100h |
| Cross-patient | None |
| SSL comparison | Supervised >> SSL (PopT, BrainBERT) |
| Key finding | Paired ambient audio pretraining works; zero-shot fails |
