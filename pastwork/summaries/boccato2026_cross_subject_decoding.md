# Boccato et al. 2026 - Cross-Subject Decoding for Speech BCIs

## Citation
Boccato, T., Olak, M., & Ferrante, M. (2026). Cross-subject decoding of human neural data for speech Brain Computer Interfaces. *bioRxiv* preprint, doi: 10.64898/2026.02.27.708564. Tether Evo.

## Setup
- **Recording modality:** Intracortical Utah microelectrode arrays (4x 64-channel arrays = 256 channels per subject); features are threshold crossing spike counts + spike-band power in 20 ms bins
- **Patient count:** 2 for training (T12 from Willett et al. 2023, T15 from Card et al. 2024), 4 for generalization evaluation (T12, T15, T16, T17 from Kunz et al. 2025)
- **Brain region:** Ventral premotor cortex (area 6v), Broca's area (area 44) for T12; left ventral precentral gyrus for T15; ventral motor and premotor speech areas for Kunz participants
- **Task:** Attempted speech production (instructed-delay sentence vocalization). Kunz dataset includes attempted and imagined (inner) speech of isolated words and sentences
- **Data amount:** Willett: ~9,000 trials over 24 days/25 sessions spanning 4 months; Card: ~10,948 trials over 45 sessions spanning >8 months/84 sessions; Kunz: 2,420 total trials across 4 participants (836 T12, 1040 T15, 224 T16, 320 T17)

## Problem Statement
All prior high-performance speech BCIs (Willett, Card, Moses) are trained on single participants, requiring hours of supervised calibration data per new user. This is a major bottleneck for clinical deployment. The paper asks: **can models trained on one participant generalize to others?** They argue this is feasible because (i) cortical speech representations have a conserved topography across individuals, (ii) phoneme-related tuning is reproducible across participants, and (iii) within-subject signal drift over time can be as large as cross-subject variability. If cross-subject models work, they could serve as pretrained foundations that reduce calibration for new users -- analogous to pretraining in ASR.

## Method

### Neural preprocessing
- **Willett (T12):** Raw signals bandpass filtered 250-5000 Hz. Threshold crossings detected at -1.5 RMS. Spike counts binned in non-overlapping 20 ms windows. Spike-band power = mean squared signal per 20 ms bin per channel. Both features concatenated per channel per time bin. Only premotor cortex electrodes used (Broca's area excluded as uninformative).
- **Card (T15):** Spike counts + spike-band power preprocessed similarly to Willett.
- **Kunz (T12-T17):** Features standardized and zero-padded to 512-dimensional representation for consistent input dimensionality across subjects.
- Official train/test/competition splits used for Willett and Card. Kunz split 80/20 into train/validation.

### Alignment approach (affine transforms -- detailed)
The core alignment mechanism is a **learned, subject-and-day-specific affine projection** applied before the shared encoder. For each subject *s* and recording day *d*, a separate linear transform is learned:

```
x_bar_t^(d,s) = W_{d,s} * x_t + b_{d,s}
```

where:
- `x_t` is the C-dimensional neural feature vector at time *t* (C = number of channels)
- `W_{d,s}` is a C x C trainable weight matrix (one per subject-day combination)
- `b_{d,s}` is a C-dimensional trainable bias vector
- The output `x_bar_t` lives in a shared latent space

**Key properties of the learned transforms (from extensive appendix analysis):**
- Transforms are predominantly **diagonal** (diagonal ratio ~0.75-1.0; mostly ~0.9+ but Willett sessions can drop to ~0.75), meaning most of the correction is per-channel rescaling with only modest cross-channel mixing
- Frobenius distance to identity is moderate, indicating transforms deviate meaningfully from identity but are not drastic
- Condition numbers are moderate for Card (1-20 from figures, some up to ~20) but extreme for early Willett sessions (>1000), reflecting larger day-to-day drift
- Log absolute determinant varies, with negative values indicating volume contraction
- Orthogonality gap is nonzero, meaning transforms include shearing/scaling beyond pure rotation
- Transform metrics correlate moderately with PER (|r| ~ 0.4-0.6), suggesting days requiring stronger alignment are genuinely harder to decode

**For new-subject adaptation:** The entire pretrained model is frozen except for a new `W` and `b` for the new subject. Only these O(C^2 + C) parameters are optimized on the new subject's data. This is relatively lightweight (e.g., 256 channels -> ~65,792 parameters). **Note:** for Kunz data padded to 512 dimensions, adaptation params are ~262k, not ~66k. T12 uses only 128 of its 256 electrodes (premotor only, Broca's excluded).

**Transform swapping experiments:** Applying one day's transform to another day's data yields reasonable (though degraded) PER, suggesting transforms capture generalizable session-invariant structure and are not purely overfitting to individual days. This hints that fewer transforms (perhaps one per subject rather than per day) could suffice.

### Decoder architecture
A **hierarchical three-block GRU decoder** with feedback connections (~100M+ total params, d=2048 — far larger than our planned ~130K shared backbone):

1. **Early GRU Block:** Two bidirectional GRU layers process the transformed neural data. Hidden dimension d=2048. An auxiliary classifier projects hidden states to phoneme logits (`l_1 = W_early * z_1 + b_early`). Logits are softmaxed to probabilities `p_1`, then projected back to hidden dimension as feedback: `p_bar_1 = W_proj,1 * p_1 + b_proj,1`. Feedback is summed with hidden states: `h_1 = z_1 + p_bar_1`.

2. **Middle GRU Block:** Two bidirectional GRU layers process `h_1`. Same feedback mechanism: produces logits `l_2`, softmax to `p_2`, projects back, sums with hidden states to form `h_2 = z_2 + p_bar_2`.

3. **Final GRU Block:** A single GRU layer processes `h_2`. Final phoneme logits `l_3 = W_final * z_3 + b_final`. No feedback (terminal layer).

**Hierarchical CTC Loss:** The total loss combines CTC losses from all three layers:
```
L_total = L_CTC(l_3, y) + lambda * [L_CTC(l_2, y) + L_CTC(l_1, y)]
```
where lambda=0.3 (optimal). This design partially mitigates CTC's conditional independence assumption by feeding intermediate phoneme predictions as explicit context into deeper layers.

**Phoneme-to-word decoding:** WFST-based beam search combining pronunciation lexicon + 5-gram language model. Optional rescoring with larger LM (e.g., OPT).

### Training procedure
- Joint training on concatenated Willett + Card datasets
- 120k steps, batch size 64, Adam optimizer
- Learning rate: linear warmup 0 to 5e-3 over first 1k steps, then cosine decay to 1e-4 at step 120k
- Weight decay: 1e-5
- Mixed-precision training with gradient accumulation
- Data augmentation: Gaussian noise + small per-channel offsets
- Hyperparameters identical to original Card baseline except d=2048 (to accommodate larger neural subspace)

## Key Results

### Joint training matches/outperforms single-subject baselines (Table 1)
| Model | Test Set | PER (%) | WER (%) / Competition WER (%) |
|---|---|---|---|
| Willett single-subject baseline | Willett | 19.7 | 17.4 / 11.06 |
| Card single-subject baseline | Card | 10.2 | 7.34 / 6.70 |
| Joint plain CTC | Willett | 17.6 | 14.54 / 10.9 |
| Joint plain CTC | Card | 9.6 | 7.57 / 6.39 |
| **Joint hierarchical CTC** | **Willett** | **16.1** | **14.54 / 10.3** |
| **Joint hierarchical CTC** | **Card** | **9.1** | **6.67 / 5.9** |

- Willett PER improved from 19.7 to 16.1 (18% relative improvement)
- Card PER improved from 10.2 to 9.1 (11% relative improvement)
- Card WER of 5.9% on competition set outperforms single-subject baseline (6.70%)

### Cross-subject generalization to Kunz dataset (Table 2)
| Participant | Linear-only adapt PER | Fine-tune whole model PER | From scratch PER |
|---|---|---|---|
| T12 | 30.2 | 21.3 | 11.8 |
| T15 | 28.8 | 26.3 | 26.1 |
| T16 | 41.1 | 26.1 | 40.9 |
| T17 | 58.9 | 53.3 | 30.6 |

- Training only the linear transform already yields substantial reduction from chance (~100% PER)
- Fine-tuning whole model for 5k steps further reduces PER by 20-40% relative to linear-only
- For T16 (entirely new subject), fine-tuning (26.1%) dramatically outperforms from-scratch (40.9%)
- Note: Kunz dataset is inner speech (not overt), so this tests cross-task generalization too

### Cross-dataset fraction sensitivity
- Adding even 10% of a second dataset's data consistently improves performance on both datasets
- Effect is most pronounced at small fractions (10-25%), demonstrating strong structural priors from cross-subject pretraining

### Hierarchical CTC weight sensitivity
- Optimal lambda = 0.3 for both datasets
- Performance degrades at lambda >= 0.5, suggesting intermediate supervision should regularize but not dominate

## Cross-Patient Relevance

### Affine alignment vs. CCA for the Cogan lab uECOG setting

**How affine alignment differs from CCA:**
- **CCA** finds shared linear subspaces by maximizing correlation between paired datasets, requiring some form of paired/aligned data (e.g., same stimuli or phoneme labels). It produces orthogonal projection matrices that map both subjects into a shared lower-dimensional space.
- **Affine transforms** here are learned end-to-end via backpropagation through the decoder loss. They operate in the *original* channel space (C x C matrix + bias), not a reduced subspace. They are optimized directly for the downstream decoding objective, not for inter-subject correlation.

**Advantages of the affine approach for uECOG:**
1. **Task-driven alignment:** The transforms are optimized for decoding performance, not a proxy objective like correlation. This may capture task-relevant structure that CCA misses.
2. **Minimal assumptions:** No need for paired data or explicit correspondence between subjects. Just needs labeled speech data from each subject.
3. **Very few parameters:** O(C^2 + C) per subject-day. For uECOG with ~256 channels, that is ~66k parameters -- trainable with very little data.
4. **Handles day-to-day drift and subject differences simultaneously:** The same mechanism addresses temporal nonstationarity (drift) and cross-subject variability.
5. **Preserves channel structure:** The diagonal-dominant nature of the learned transforms suggests most information is per-channel, with modest mixing. This aligns well with spatially organized electrode grids.

**Potential disadvantages / considerations:**
1. **Requires a pretrained decoder:** The affine approach assumes you already have a good shared decoder. CCA can find shared spaces without a downstream model.
2. **Square transform assumption:** W is C x C, assuming same dimensionality in and out. If uECOG subjects have very different channel counts or electrode configurations, this requires padding/standardizing to a common dimensionality first (as they did for Kunz).
3. **No explicit geometric constraints:** CCA enforces orthogonality and maximizes correlation, providing interpretable geometric relationships. The learned affine transforms are unconstrained, which gives more flexibility but less interpretability.
4. **Scalability to many subjects:** Each new subject needs its own W and b, optimized by backprop through the frozen decoder. For intra-operative uECOG with very short recording windows, the question is whether there is enough data to learn even these ~66k parameters.
5. **Electrode placement variability:** uECOG in the Cogan lab likely has more heterogeneous electrode placement than Utah arrays targeting the same region. The affine transform may not be expressive enough if the underlying neural populations sampled are fundamentally different across patients.

**Practical implication for Cogan lab:** The affine approach could serve as an alternative or complement to CCA. One could try: (a) train a shared decoder on pooled data with per-patient affine transforms, or (b) use CCA for initial alignment (unsupervised), then refine with task-driven affine fine-tuning. The extremely low parameter count makes this attractive even for intra-operative settings with limited data.

## Limitations
1. **Only 2 training subjects:** Willett (T12) and Card (T15) are both ALS patients with Utah arrays in similar cortical locations. Unclear if the approach scales to more diverse populations or electrode configurations.
2. **CTC conditional independence:** Despite the hierarchical feedback, the model still inherits CTC's frame-level independence assumption. Most residual errors come from phoneme-to-word reconstruction, not neural-to-phoneme decoding.
3. **Inner speech generalization is weak:** PER on Kunz T17 is 53-59%, suggesting the pretrained model struggles with qualitatively different neural regimes (inner vs. overt speech).
4. **Language model does heavy lifting:** WER improvements partly reflect WFST and LM rescoring rather than purely neural decoding gains.
5. **No comparison to other alignment methods:** The paper does not benchmark against CCA, Procrustes, optimal transport, or other established alignment techniques.
6. **Same electrode type across all subjects:** All participants use Utah arrays. Generalization to ECoG, uECoG, or other electrode types is untested.
7. **Bidirectional GRU:** The decoder is non-causal (bidirectional), which is fine for offline evaluation but not directly usable for real-time BCI without modification.
8. **Hyperparameter sensitivity not exhaustive:** lambda=0.3 is optimal but d=2048 and other choices were not systematically ablated.

## Reusable Ideas

1. **Per-session affine transforms as a general drift-correction mechanism:** The idea of learning a simple W*x + b per recording session, trained end-to-end, is broadly applicable to any neural decoding setting with session-to-session variability. Could be applied to intra-operative uECOG recordings where electrode impedance changes within a single session.

2. **Hierarchical CTC with feedback connections:** Feeding intermediate phoneme predictions back into deeper GRU layers partially addresses CTC's conditional independence limitation without the instability of autoregressive training. This is architecture-agnostic and could be added to any CTC-based decoder.

3. **Freeze-and-adapt paradigm for new subjects:** Training only the input alignment layer while freezing the rest of the model is an extremely efficient adaptation strategy. For the Cogan lab, this means: pretrain a shared decoder on pooled data, then for each new intra-operative patient, only learn the affine transform from their limited data.

4. **Transform analysis toolkit:** The appendix provides a useful set of metrics for analyzing learned alignment transforms (Frobenius distance to identity, condition number, diagonal ratio, orthogonality gap, spectral entropy, etc.). These could be applied to analyze CCA projections or any other alignment method.

5. **Cross-dataset fraction sensitivity analysis:** The finding that even 10% of a second dataset improves performance suggests that for the Cogan lab, even small amounts of data from additional patients could meaningfully improve a shared model.

6. **Day-transform swapping as a diagnostic:** Testing whether one session's alignment transform works on another session's data is a useful diagnostic for assessing alignment quality and session similarity. Could apply this to evaluate CCA alignment stability across sessions.

7. **Padding to common dimensionality:** For subjects with different channel counts, standardizing and zero-padding to a common representation is a simple approach that enables weight sharing. Relevant for uECOG where electrode coverage varies across patients.
