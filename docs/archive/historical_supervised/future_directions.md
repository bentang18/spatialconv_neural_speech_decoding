# Future Directions

Ideas explored during the v4/v5/v6 redesign that warrant investigation after core experiments (E1–E5) are complete. See `implementation_plan.md` for the core pipeline.

---

## Cross-Task Pooling (Duraivel 2025)

Pseudo-word repetition task (CVC/VCV, 52 tokens, **same 9 phonemes**) with overlapping patients:

- D2025-S1 (201/256) = Spalding S8; D2025-S2 (63/128) = Spalding S5; D2025-S3 (111/128) = Spalding S1/S2
- 156–208 additional trials/patient — nearly doubles training data
- Both tasks produce length-3 CTC targets: Spalding non-words (e.g., [a,b,e]) and Duraivel pseudo-words (CVC/VCV)
- Same preprocessing pipeline, same IRB

**Questions for Zac:** (1) Confirm patient overlap; (2) Are pseudo-word HGA features pre-extracted? (3) Additional uECOG patients beyond Spalding's 8?

---

## Tier 2 Ideas

### Downgraded training innovations (from v4/v5)

Originally core innovations, downgraded after analysis (2026-03-16). Per-patient input layers already provide the cross-patient transfer mechanism — the shared backbone is implicitly transfer-ready from joint training with multiple per-patient read-ins. Listed as E6/E7 in extended ablations.

**Reptile meta-learning.** Replace standard SGD with Reptile outer loop (Nichol 2018): each inner loop simulates Stage 2 few-shot adaptation on a source patient, outer loop moves initialization toward "easy to adapt" parameters. **Why downgraded:** In Stage 2, the backbone is frozen — Reptile's "easy to adapt" property applies to parameters we don't touch. Standard joint training with per-patient layers already produces a backbone that accommodates diverse per-patient projections. Singh achieves good cross-patient transfer (N=25) without meta-learning. With only N=7 source patients, the meta-learning task distribution is thin.

**Cross-patient SupCon.** Periodic supervised contrastive loss on time-pooled backbone embeddings across patients, alongside CTC. **Why downgraded:** (1) CTC + per-patient layers already enforce cross-patient alignment — the shared CTC head must decode all patients from the same backbone space. (2) Time-pooled trial embeddings are a poor contrastive target: trials with /aba/ and /baa/ share phonemes but differ in order, yet their time-averaged embeddings are similar. (3) With 52 token types and ~150 trials/patient, contrastive batches have very few positive pairs per class.

**DRO (Distributionally Robust Optimization).** Replace uniform loss averaging with softmax-weighted loss (upweight high-loss patients), optimizing for worst-case patient. **Why downgraded:** (1) Hard patients (S3: 46 trials, S5: 63 sig channels) are hard for practical data quality reasons, not neural coding differences — upweighting them means overfitting to noise, not learning transferable patterns. (2) N=7 source patients is too few for robust worst-case estimation — running EMA of 7 loss values is extremely noisy. (3) DRO hurts ~5-6 of 8 LOPO folds where the target is an easy patient (average effect likely washes out). (4) Gradient accumulation across patients already provides implicit difficulty-aware weighting (higher loss → larger gradients). (5) Adds 3 hyperparameters (τ, α, clamping) that cannot be tuned under our no-tuning policy. Listed as E5 in extended ablations.

### After E2 converges: low-cost additions

**TENT (test-time adaptation).** Adapt LayerNorm γ/β on unlabeled target data via entropy minimization before Stage 2. Zero labels required. Listed as E11 in extended ablations. ~40 lines. **Risk:** Uncertain impact since LayerNorm stats are per-sample.

**DANN (adversarial domain adaptation).** Gradient reversal patient discriminator on backbone hidden states. Pushes patient-discriminative info out of backbone representations. Chen 2024 has this in their codebase (unused, weight=0.0005). Listed as E10. ~40 lines. **Risk:** With 8–10 patients, discriminator is easy to fool.

**Autoencoder read-in init.** Pre-train target spatial conv via spatial reconstruction (mask electrodes, predict from neighbors) on unlabeled data. Most beneficial for S3 (46 trials). **Risk:** Marginal benefit with only 80 params.

### Speech foundation model alignment (HuBERT/Wav2Vec2.0)

**The idea.** Project backbone representations into the latent space of a pre-trained speech foundation model (speech FM) — HuBERT or Wav2Vec2.0, trained on 960h LibriSpeech. The speech FM has learned rich phoneme-level structure; aligning our neural representations to this space imports that structure without additional neural data.

**State of the field (as of 2026-03):**

| System | Flavor | Brain region | Speech FM role | Data/patient |
|---|---|---|---|---|
| Metzger 2023 / Littlejohn 2025 (Chang lab) | Discrete: predict 100 HuBERT K-means units via CTC/RNN-T | SMC + STG (253ch ECoG) | Define discrete codebook + frozen speech LM prior | 11k–23k trials |
| Li et al. 2025 Dual Pathway (eLife) | Continuous: regress to 768-dim Wav2Vec2.0 features | **STG only** (auditory) | Frozen encoder defines target space; frozen HiFi-GAN decodes | 20 min |
| Zhang et al. 2026 BIT v3 (ICLR) | Contrastive: InfoNCE aligns neural encoder with audio LLM (Aero1-Audio 1.5B) | Intracortical arrays (spikes) | Audio LLM decodes aligned representations end-to-end | 367h pretraining |
| Wairagkar/Card 2025 (Stavisky lab) | Direct: predict acoustic features, no speech FM | vPCG (motor, 256 MEA) | None — bypasses speech FM entirely | Single patient |

**Key fact: no one has tested the motor cortex → speech FM mapping.** Dual Pathway validated the near-linear correspondence for auditory cortex (STG) only. Chang lab's systems use broad SMC+STG coverage but never isolate the motor contribution. Stavisky lab bypasses speech FMs for motor cortex entirely. Our uECOG arrays over left sensorimotor cortex would be the first direct test.

**Why motor cortex → speech FM is harder than auditory cortex → speech FM.** The causal chain has one extra nonlinear step:

- Auditory cortex: acoustic input → auditory activity ≈ HuBERT features (near-linear, empirically validated)
- Motor cortex: motor commands → articulatory configuration → acoustic output → HuBERT features (nonlinear vocal tract transfer function in the middle)

However: (1) HuBERT mid-layers (layer 6) encode phoneme identity, which is defined by articulatory features — so the mapping from articulatory commands to HuBERT-layer-6 may be more direct than to raw acoustics; (2) our uECOG captures somatosensory cortex (postcentral gyrus) alongside motor cortex, providing proprioceptive feedback signals closer to the acoustic output; (3) Wu 2025 shows articulatory features are "more robustly encoded than acoustic features" in vSMC, suggesting the motor→articulatory→acoustic chain is learnable.

**Why an auxiliary training loss is NOT the most elegant first step.** Our pipeline already captures phoneme similarity structure via the articulatory decomposition head (6 linguistically-grounded sub-problems). With 9 well-separated phonemes in isolated repetition (no coarticulation), within-phoneme HuBERT variation is mostly speaker/trial noise, not task-relevant signal. An aux loss during training entangles HuBERT's contribution with CTC, preventing clean attribution. The scientific question — "does the E2 backbone learn speech-FM-compatible representations on its own?" — is better answered by a post-hoc probe.

**Design: two-phase, diagnostic-first.**

**Phase 1 — Linear probe diagnostic (after E2, ~20 lines, zero pipeline disruption):**
1. Pre-extract HuBERT-Base features (layers 1–12) for all paired audio using `transformers`. Output: 768-dim at 50 Hz per trial.
2. Segment by MFA phoneme boundaries. Average HuBERT features within each phoneme segment → one 768-dim vector per segment. PCA to 64 dims (captures >99% variance for 9 phonemes).
3. For each LOPO fold's trained E2 model: freeze backbone. Average backbone output (128-dim) within matching MFA segments.
4. Train ridge regression (per-layer): backbone segment features → PCA-reduced HuBERT segment features. 10-fold CV within each LOPO fold.
5. Report: R² per HuBERT layer, per patient. Compare to permutation baseline (shuffled segment pairing). ~450 segments/patient (3 segments × ~150 trials), well-conditioned for 128→64 regression.
6. **This is the publishable finding**: first measurement of motor cortex uECOG → speech FM representational correspondence. Even R² ≈ 0 is informative (confirms the auditory-specific nature of the Dual Pathway mapping).

**Phase 2 — Conditional auxiliary loss (only if Phase 1 R² is significant, ~30 lines):**
1. Add projection head: `Linear(128, 64)` on backbone output (8,192 params, discarded at inference).
2. Loss: segment-level MSE between projected backbone features and PCA-reduced HuBERT layer-K features (K chosen from Phase 1 peak R² layer). Weight λ=0.1 relative to CTC.
3. Temporal alignment: average both signals within MFA phoneme boundaries — no frame-level alignment needed.
4. Compare E2+HuBERT vs E2 in full LOPO evaluation. Listed as E13 in extended ablations.

**Why segment-level MSE over alternatives:**

| Approach | Problem for our regime |
|---|---|
| Frame-level MSE | Motor planning delay (100–500ms) requires precise temporal alignment; rate mismatch (40 vs 50 Hz) |
| Trial-level contrastive (CLIP) | ~150 trials/patient, 9 classes → most negatives are same-phoneme → contrastive collapse |
| Discrete HuBERT units (100-class CTC) | ~20 trials per unit with our data budget — catastrophically sparse |
| Full Dual Pathway regression (768-dim) | 768-dim target with ~450 segments → overdetermined without PCA; motor cortex mapping unvalidated |
| **Segment-level MSE to PCA-reduced HuBERT** | **Avoids temporal alignment, well-conditioned (128→64 with ~450 points), interpretable (R²)** |

**Not-yet-viable directions that this diagnostic would unlock:**
- If R² is high: speech synthesis from motor cortex uECOG (regress to full HuBERT space → HiFi-GAN vocoder)
- If R² varies by layer: identify which level of the speech processing hierarchy motor cortex best aligns with
- If R² correlates with posterior electrode position: somatosensory/auditory feedback contribution vs pure motor
- Cross-patient alignment via shared HuBERT space (project all patients into HuBERT → distance metrics in speech FM space)

**Prerequisites:** (1) Paired audio access (confirmed for all 8 patients). (2) MFA phoneme boundaries (confirmed: MFA + Audacity labels exist). (3) E2 trained and converged.

### After ablation analysis: other data efficiency ideas

**Frame-level MFA supervision.** Soft frame-level cross-entropy from MFA phoneme timing alongside CTC. Standard in ASR (Watanabe 2017), absent from neural speech BCI. **Risk:** Depends on MFA quality on intra-op audio.

**Knowledge distillation.** Distill per-patient teachers into cross-patient student via soft labels: `loss = α·CTC(hard) + (1-α)·KL(soft)`. Soft labels encode patient-specific phoneme confusion structure. **Risk:** Per-patient models on ~100 trials may be too noisy. Only viable if per-patient accuracy substantially exceeds chance.

### Structured augmentation for cross-patient robustness

Domain randomization (Tobin et al. 2017): augment so aggressively that real-world variation becomes a subset of training variation.

| Augmentation | Simulates | Feasibility |
|---|---|---|
| Aggressive per-channel gain (σ=0.5–1.0) | Large impedance variation | Very high |
| Heavy channel dropout (p=0.2–0.5) | Missing/bad electrodes | Very high |
| Random temporal dilation (±20%) | Response time variation | Medium |
| Random spatial shift on grid (±1–2 px) | Surgical placement variation | Medium |
| Cross-patient Mixup in backbone space | Interpolating distributions | Medium |

Start with aggressive gain + heavy channel dropout (trivial, simulate the two largest sources of cross-patient variation).

### Temporal alignment

**Soft DTW pre-normalization.** Different patients have different response latencies (CVC: 0.56±0.35s). BiGRU + CTC handle this implicitly, but explicit DTW (Cuturi & Blondel 2017) could help. **Test only if** CTC alignment analysis shows systematic timing mismatches across patients.

### Architecture alternatives (if core design plateaus)

**Prototypical inference.** Already a diagnostic (§2.7). Promote to inference method if prototypical accuracy ≈ CTC accuracy — eliminates all per-patient learned parameters at inference.

**Hypernetworks.** Meta-network generates spatial conv weights from patient descriptor (signal stats, array type). Needs N>30 patients.

**In-context learning.** Present support trials as context; decode without weight updates. Needs transformer backbone + N>100 patients.

### Evaluation extensions

**Label-shuffling diagnostic.** Train on shuffled phoneme labels. If shuffled backbone still improves target PER → benefit is regularization, not alignment. Directly answers a question nobody in the field asks. **Feasibility: very high** (~10 lines — just shuffle labels).

**Conformal prediction.** Distribution-free coverage guarantees on CTC predictions. Clinical relevance. Defer to paper revision.

**Active learning for intra-op calibration.** Adaptively select phonemes based on model uncertainty. Changes experimental protocol — not applicable to existing data.
