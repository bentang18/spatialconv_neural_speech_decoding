Here's the complete data flow for the E2 (full model) pipeline, with stress tests at each stage.

---

## Stage 0: Data Loading

```
Raw .mat per patient per phoneme
  → hgTrace: (trials, ch_x, ch_y, time)
     S1: (144, 8, 16, 200)    S3: (46, 12, 22, 200)
     S5: (151, 8, 16, 200)    S8: (178, 12, 22, 200)
  → phonSeqLabels: (trials, 3) e.g., [0, 1, 4] for /abe/

Assemble per-patient: stack across phonemes → (all_trials, H, W, 200)
```

**Stress tests:**

**S0-A: Do non-sig channels actually appear as zeros in `hgTrace`?** The file name says "sigChannel_goodTrials" — but does `hgTrace` contain the full grid with non-sig channels zeroed, or only sig channels stacked? If the latter, the spatial grid structure is destroyed and we can't use Conv2d. **This is a blocking question for Zac.** The plan assumes `hgTrace` preserves the 2D grid with zeros at non-sig positions.

**S0-B: How sparse is the grid?** S5 has 63/128 significant channels on an 8×16 grid — **51% of the grid is zeros before any augmentation**. S4 has 74/128 = 42% zeros. The Conv2d is processing images where nearly half the pixels are dead. Combined with channel dropout (up to 20%), S5 could have ~60% zeros. The 3×3 conv kernel will see a mix of signal and zeros at boundaries. This isn't necessarily fatal — the model can learn to ignore zeros — but it's a much harder learning problem than processing dense images.

**S0-C: What's in the stored epoch beyond ±500ms?** Time shifting augmentation shifts by ±100ms (±20 frames). If `.mat` files contain exactly ±500ms = 200 frames, shifting left by 20 means the last 20 frames are zero-padded. If the stored epoch is wider (say ±600ms), we can shift without padding. **Blocking question for Zac.**

---

## Stage 1: Pre-Read-In Augmentation

```
Input: (B, H, W, T=200) per patient

1. Time shift:       ±20 frames (±100ms), zero-pad edges
2. Amp scaling:      per-channel scale = exp(N(0, 0.15²)), constant within trial
3. Channel dropout:  zero entire electrodes, p ~ U[0, 0.2]
4. Gaussian noise:   x += N(0, (0.02·std(x))²) per frame

Output: (B, H, W, T=200) — same shape, augmented
```

**Stress tests:**

**S1-A: Amplitude scaling — what does "per-channel" mean on a 2D grid?** Each (h, w) position gets its own random scale factor? Or each row/column? The plan says "per-channel" which in the HGA context means per-electrode — each grid position gets an independent scale. At σ=0.15, scales range ~[0.74, 1.35] (95% interval). This simulates impedance variation — the dominant cross-patient factor per Boccato.

**S1-B: Channel dropout stacks with missing channels.** For S5 (51% already zero), dropping 20% of the remaining 63 channels → ~50 active channels out of 128 = 39% grid active. The effective signal is very sparse. Is this too aggressive? The plan says p ~ U[0, 0.2] — so on average p=0.1 (10% additional dropout). Worst case (p=0.2 on S5): 63 × 0.8 = ~50 channels → 39% active. This seems harsh but is designed to simulate the cross-patient channel count variation (63–201).

**S1-C: Gaussian noise std computation.** `(0.02·std(x))²` — what is `std(x)` computed over? Per-trial across all channels and time? Per-frame across channels? This determines whether high-activation frames get more noise or if noise is uniform. The plan doesn't specify. Most natural: `std(x)` is the global std of the trial → noise is proportional to overall signal strength.

---

## Stage 2: Per-Patient Spatial Conv Read-In

```
Input: (B=16, H, W, T=200) — one patient's augmented batch

Step 1: Reshape        (B, H, W, T) → (B*T, 1, H, W)
        S1: (3200, 1, 8, 16)    S8: (3200, 1, 12, 22)

Step 2: Conv2d(1,8,k=3,pad=1) + ReLU
        (3200, 1, H, W) → (3200, 8, H, W)
        80 params per patient

Step 3: AdaptiveAvgPool2d(2,4)
        (3200, 8, H, W) → (3200, 8, 2, 4)
        S1: pools over 4×4 regions    S8: pools over 6×5.5 regions

Step 4: Flatten        (3200, 8, 2, 4) → (3200, 64)

Step 5: Reshape        (3200, 64) → (16, 200, 64)

Step 6: Permute        (16, 200, 64) → (16, 64, 200)   # channels-first for Conv1d

Output: (B=16, D=64, T=200)
```

**Stress tests:**

**S2-A: Can 80 params learn meaningful spatial filters?** 8 filters × (3×3 kernel + 1 bias) = 80 params. Each filter is a single 3×3 spatial template — it detects one local pattern (e.g., "adjacent electrodes co-activate"). With only 80 params, the model has minimal capacity for overfitting, which is good. But can 8 spatial templates capture the relevant articulatory information? The hypothesis is that a few spatial detectors (tongue area, lip area, larynx area) suffice. With 9 phonemes and 6 articulatory features, 8 filters should be enough. **But:** all 8 filters share the same 3×3 receptive field. They can detect different patterns, but all at the same spatial scale. If some articulatory features are encoded at larger spatial scales (e.g., whole-array gradients), a single 3×3 conv can't capture them.

**S2-B: The B*T reshape is correct but subtle.** Each timepoint is treated as an independent image. The Conv2d learns spatial features that are **time-invariant** — the same spatial filter detects "tongue activation" whether it's at t=50ms or t=300ms. This is the right inductive bias: spatial anatomy doesn't change within a trial. Temporal dynamics are handled later by Conv1d + BiGRU.

**S2-C: AdaptiveAvgPool2d asymmetry across patients.** For 8×16 → 2×4: each output cell averages exactly 4×4 = 16 electrodes. For 12×22 → 2×4: PyTorch splits as:
- Height 12 → 2: bins of [6, 6]
- Width 22 → 4: bins of [6, 6, 5, 5] (or [5, 6, 5, 6] depending on implementation)

So S8's output cells average over 30-36 electrodes each, vs S1's 16. **S8's features are much more spatially blurred than S1's.** This means the 64-dim feature vector encodes spatial information at different effective resolutions per patient. Since the backbone treats all patients' 64-dim vectors the same, this resolution asymmetry is absorbed by the per-patient Conv2d + pool combination. But it means S8's per-patient read-in must compensate for heavier pooling by learning sharper conv filters.

**S2-D: Output dimension is always 64, regardless of patient.** The `assert C * 2 * 4 == D_shared` enforces this. Both LinearReadIn and SpatialConvReadIn produce (B, 64, T). The backbone never knows which read-in was used. This is the correct interface contract.

---

## Stage 3: Shared Backbone — LayerNorm + Feature Dropout

```
Input: (B=16, D=64, T=200)

Step 7:  Permute           (B, D, T) → (B, T, D) = (16, 200, 64)
Step 8:  LayerNorm(64)     normalize across D=64 for each (b, t)
         128 params (64 γ + 64 β)
Step 9:  Permute back      (B, T, D) → (B, D, T)

Step 10: Permute           (B, D, T) → (B, T, D) = (16, 200, 64)
Step 11: Feature dropout   sample p ~ U[0, 0.3], zero random dims
         mask shape: (64,) — SAME dims dropped for ALL batch items AND all timepoints
Step 12: Permute back      (B, T, D) → (B, D, T)

Output: (B=16, D=64, T=200)
```

**Stress tests:**

**S3-A: Four permute operations in two steps.** The code permutes to (B,T,D) for LayerNorm, back to (B,D,T), then permutes to (B,T,D) again for feature dropout, then back to (B,D,T). This is redundant — could be fused into one permute, apply both operations, permute back. Not a correctness issue, but confusing code.

**S3-B: Feature dropout is aggressive and batch-global.** The mask `(torch.rand(64) > p).float()` is a single (64,) vector applied to ALL batch items AND all 200 timepoints. At p=0.3, ~19 of 64 features are zeroed for the entire forward pass. This means the BiGRU can NEVER see those 19 features for any timepoint in any batch item. The inverted dropout scaling `/ (1 - p + 1e-8)` compensates for expected value, but the variance is high.

Is this the right behavior? The plan says it "simulates unreliable read-in features during Stage 2 adaptation." That's a reasonable motivation — when a new read-in is initialized randomly, some of its output features will be garbage. But dropping the same features for ALL timepoints is more like "entire feature channel failure" than "noisy features." **Consider:** per-frame or per-sample feature dropout might be more realistic. But this is a minor design choice, not a bug.

**S3-C: LayerNorm is unfrozen in Stage 2.** It's one of the 2,272 trainable params (128 of them). During Stage 2, LayerNorm adapts its γ/β to normalize the new read-in's output distribution. This is important — different read-ins produce features with different means and variances, and LayerNorm bridges the gap. **Good design decision.**

---

## Stage 4: Temporal Downsampling (Conv1d)

```
Input: (B=16, D=64, T=200)

Step 13: Conv1d(64, 64, kernel_size=5, stride=5)
         (B, 64, 200) → (B, 64, 40)
         No padding → output length = (200 - 5)/5 + 1 = 40
         20,544 params

Step 14: GELU activation

Output: (B=16, D=64, T=40)  — now at 40 Hz
```

**Stress tests:**

**S4-A: No padding means edge frames get truncated.** The first output frame covers input frames 0-4 (0-25ms). The last covers frames 195-199 (975-1000ms). All 200 input frames contribute to exactly one output frame — no overlap, no waste. Clean.

**S4-B: 5-frame kernel = 25ms temporal window.** What happens in 25ms of speech? A consonant burst is ~5-15ms. Formant transitions are ~25-50ms. So each Conv1d output captures roughly one phonemic event or transition. This is reasonable granularity for CTC.

**S4-C: Conv1d is FROZEN in Stage 2.** This means the temporal feature extraction is locked after Stage 1. New patients with different temporal dynamics (faster/slower response latencies) can't adapt the temporal features — they must work with the temporal representations learned from source patients. **Potential weakness:** if the target patient has systematically different temporal dynamics, frozen Conv1d can't compensate. But time-shifting augmentation in Stage 1 should train the Conv1d to handle timing variation.

**S4-D: The spatial information is already collapsed.** At this point, the 64-dim vector is a mix of 8 spatial filter activations at 8 pooled positions. The Conv1d processes this across time, mixing spatial features together. Any spatial structure that survived the pool is now irretrievably mixed with temporal processing. This is the (2+1)D design — spatial processing is complete.

---

## Stage 5: BiGRU

```
Input: (B=16, T=40, D=64)  — after permute from (B, D, T)

Step 15: GRU Layer 1
         input_size=64, hidden_size=64, bidirectional=True
         Forward:  processes t=0→39, hidden state carries left context
         Backward: processes t=39→0, hidden state carries right context
         Output: (B, 40, 128)  — concat forward[64] + backward[64]
         49,920 params

Step 16: Dropout(0.2) between layers

Step 17: GRU Layer 2
         input_size=128, hidden_size=64, bidirectional=True
         Output: (B, 40, 128)
         74,496 params

Output: (B=16, T=40, 2H=128)
```

**Stress tests:**

**S5-A: Bidirectional context for CTC.** Each output frame h_t at position t contains:
- Forward GRU: information from frames 0 through t (all past)
- Backward GRU: information from frames t through 39 (all future)

So h_t sees the **entire sequence** — full bidirectional context. This is why BiGRU works so well for CTC: the model can look at the whole trial to decide what phoneme each frame should emit.

**S5-B: Hidden state capacity.** H=64 per direction → the GRU compresses the entire past (or future) into a 64-dim vector. Is 64 enough? Singh uses BiLSTM(64,64) for 36 phonemes with 25 patients. We have 9 phonemes with 8 patients — smaller problem. 64 should be sufficient. Going larger (128) would double BiGRU params from 124k to ~500k — risky in our data regime.

**S5-C: BiGRU is FROZEN in Stage 2.** The entire temporal modeling capacity is fixed after Stage 1. Stage 2 can only adapt what features are extracted (via read-in) and how they're classified (via head). The temporal dynamics — how the model interprets phoneme sequences over time — are locked. This is the Singh-style transfer assumption: temporal dynamics of speech are shared across patients, only the spatial encoding differs.

**S5-D: Dropout(0.2) between layers.** Applied to the 128-dim output of layer 1 before feeding to layer 2. Standard regularization. During inference (eval mode), dropout is disabled, so the full 128-dim representation is used.

---

## Stage 6: Articulatory CTC Head

```
Input: h = (B=16, T=40, 2H=128)

Step 18: Six parallel linear projections
         cv_head(h):      (B, 40, 128) → (B, 40, 2)    C/V
         place_head(h):   (B, 40, 128) → (B, 40, 3)    bilabial/labiodental/velar
         manner_head(h):  (B, 40, 128) → (B, 40, 2)    stop/fricative
         voicing_head(h): (B, 40, 128) → (B, 40, 2)    voiced/voiceless
         height_head(h):  (B, 40, 128) → (B, 40, 3)    low/mid/high
         back_head(h):    (B, 40, 128) → (B, 40, 3)    front/central/back
         1,935 params total

Step 19: Concatenate → features: (B, 40, 15)

Step 20: Composition: phoneme_logits = features @ A.T → (B, 40, 9)
         A is (9, 15) fixed binary matrix
         Each phoneme logit = sum of its active articulatory feature scores

Step 21: blank_logit = blank_head(h) → (B, 40, 1)
         129 params

Step 22: Concatenate [blank, phonemes] → (B, 40, 10)

Step 23: log_softmax(dim=-1) → (B, 40, 10)

Output: log_probs = (B=16, T=40, C=10)
```

**Stress tests:**

**S6-A: The composition matrix creates structural constraints.** Let me trace what happens for /b/ vs /p/:
```
/b/ logit = C_score + bilabial + stop + voiced
/p/ logit = C_score + bilabial + stop + voiceless
/b/ - /p/ = voiced - voiceless
```
The model MUST learn to produce different voiced vs voiceless scores to distinguish /b/ from /p/. Everything else cancels. This is the desired inductive bias — it forces the backbone to encode voicing distinctly.

**But what about soft competition within feature groups?** The place_head produces 3 scores (bilabial, labiodental, velar). These are NOT softmax-normalized — they're raw linear projections. So all three can be high simultaneously. When composing /b/ (bilabial) vs /g/ (velar):
```
/b/ logit = C + bilabial + stop + voiced
/g/ logit = C + velar + stop + voiced
/b/ - /g/ = bilabial - velar
```
If both bilabial and velar are high (because the place_head hasn't learned to make them exclusive), the difference is small → confusion between /b/ and /g/. **The model learns mutual exclusion implicitly through CTC loss, not through architecture.** This is fine — adding softmax within sub-heads would create non-differentiable hard decisions.

**S6-B: Inapplicable features are zeroed by A.** For vowel /a/, the consonant features (place, manner, voicing) are all zeroed in row 6 of A. But the place_head, manner_head, and voicing_head still COMPUTE scores for /a/ frames — they just don't affect /a/'s logit. The backbone must route vowel-relevant information to height_head and back_head, while routing consonant-relevant info to place/manner/voicing heads. **This routing happens in the shared 128-dim BiGRU output — there's no architectural separation.** The 128-dim vector must encode both consonant and vowel features simultaneously, relying on the linear heads to extract the relevant ones.

**S6-C: Number of active features differs across phonemes.** Consonants activate 4 features each (C + place + manner + voicing). Vowels activate 3 features each (V + height + backness). This means consonant logits are sums of 4 terms, vowel logits of 3 terms. **The variance of consonant logits is higher** (sum of more independent terms). After log_softmax, this could make consonant predictions more "confident" — but the model can compensate by learning smaller feature magnitudes for consonants. Not a fundamental problem, but worth monitoring in the confusion matrix.

**S6-D: The blank logit is independent.** `blank_head` is a separate Linear(128, 1) — it doesn't go through the composition matrix. The model freely learns when to emit blank vs phoneme. CTC needs the blank to be dominant most of the time (40 frames for 3 phonemes → ~87% should be blank in a well-aligned model). **The blank logit competes directly with all 9 phoneme logits in the softmax.** If the articulatory feature scores are all large, phoneme logits are all large, and blank gets suppressed → the model over-emits. Monitor blank ratio.

**S6-E: Gradient routing through A.** When CTC backpropagates dL/d(phoneme_logit_i), the composition matrix routes this gradient to the specific articulatory features of phoneme i:
```
dL/d(features) = dL/d(phoneme_logits) @ A
```
So the gradient for "improve /b/ classification" flows to the C, bilabial, stop, and voiced features — the CORRECT articulatory dimensions. Features not used by /b/ (labiodental, fricative, vowel features) receive zero gradient from /b/ errors. **This is elegant gradient routing with zero learned parameters.**

---

## Stage 7: CTC Loss

```
Input:
  log_probs: (B, T=40, C=10) → MUST permute to (T=40, B, C=10) for PyTorch CTCLoss
  targets:   (B, 3) — phoneme indices, must be 1-9 (NOT 0-indexed, since blank=0)
  input_lengths:  (B,) all 40
  target_lengths: (B,) all 3

Loss = CTCLoss(blank=0, reduction='mean')
```

**Stress tests:**

**S7-A: CTC input format.** PyTorch `CTCLoss` expects **(T, B, C)**, not (B, T, C). The articulatory head returns (B, T, C). **The training loop must transpose: `log_probs.permute(1, 0, 2)`.** This isn't shown explicitly in the plan's training code — it just says `ctc_loss(logits, y)`. Need to ensure the wrapper handles the transpose.

**S7-B: Target label encoding.** CTC blank = index 0. Phoneme targets must be indices 1-9. If `phonSeqLabels` uses 0-indexed phonemes (0-8), they need to be shifted to 1-9. If they're already 1-indexed, no shift needed. **Need to verify from data.**

**S7-C: CTC with length-3 targets on 40 frames.** The CTC forward-backward algorithm considers all valid alignments of [b, a, e] into 40 frames. Minimum valid alignment:
- If all 3 phonemes are distinct: b, a, e → minimum 3 frames (no blanks needed between distinct labels)
- If phonemes repeat at positions 1 and 3 (e.g., [b, a, b]): b, blank, a, blank, b → minimum 5 frames

With T=40 >> 5, CTC has abundant alignment space. Each phoneme gets ~13 frames on average, with CTC learning the optimal alignment.

**S7-D: CTC can produce wrong-length outputs.** Greedy decode of [blank, blank, b, b, blank, a, a, a, blank, e, blank, blank...] → [b, a, e] (length 3) ✓. But [blank, b, blank, a, blank, e, blank, b, blank...] → [b, a, e, b] (length 4) ✗. The model might emit extra phonemes or too few. **CTC doesn't constrain output length.** The plan monitors "CTC length accuracy" (% producing exactly 3 phonemes) as a health check.

**S7-E: CTC loss magnitude.** For 10 classes and 40 frames, the initial loss (random init) should be approximately `-log(1/10) × 40 ≈ 92`. After normalization per target length, it's ~30.7 per trial. With B=16 and 7 patients, the initial accumulated loss is substantial. Grad clipping at 5.0 is important for stability.

---

## Stage 8: Training Loop

```
For each LOPO fold (target = 1 of 8 patients):

  STAGE 1: Train on 7 source patients
    For epoch in 1..200:
      optimizer.zero_grad()
      For each source patient p:
        x, y = sample_batch(data[p], B=16)
        x = augment(x)                        # time shift, amp, ch drop, noise
        features = spatial_convs[p](x)         # per-patient Conv2d
        h = backbone(features)                 # shared Conv1d + BiGRU
        logits = ctc_head(h)                   # articulatory or flat
        loss = ctc_loss(logits, y) / 7         # normalize by #patients
        loss.backward()                        # accumulate gradients
      clip_grad_norm_(5.0)
      optimizer.step()                          # one step per epoch
      scheduler.step()

      if epoch % 40 == 0: save_snapshot()
      if epoch % 10 == 0: check_early_stopping()

    SWA: average saved snapshots

  STAGE 2: 5-fold CV on target patient
    For each fold (train/val/test split):
      Load SWA model
      Init new spatial_conv for target (Kaiming)
      Freeze: Conv1d + BiGRU
      Unfreeze: spatial_conv + LayerNorm + ctc_head (2,272 params)

      For epoch in 1..100:
        # Target trials (70%)
        x_tgt, y_tgt = sample(target_train, B≈5)
        loss_tgt = ctc_loss(model(spatial_conv_tgt(augment_light(x_tgt))), y_tgt)

        # Source replay (30%)
        p_src = random source patient
        x_src, y_src = sample(data[p_src], B≈2)
        loss_src = ctc_loss(model(frozen_spatial_convs[p_src](x_src)), y_src)

        loss = 0.7 * loss_tgt + 0.3 * loss_src
        loss.backward()  # gradient to spatial_conv_tgt + LN + head (not source read-in)
        optimizer.step()

      Evaluate on test fold → PER, balanced accuracy
```

**Stress tests:**

**S8-A: Effective batch size.** Stage 1: 16 trials × 7 patients = 112 trials per optimizer step. But gradients are computed sequentially per patient (one forward-backward per patient, accumulated). Memory peak = one patient's batch (16 × 200 × grid_size) + model + gradients. This is low — the model is tiny (~147k params). **No memory concern.**

**S8-B: Early stopping evaluates on TRAINING data.** The plan says "evaluated on training CTC loss across all source patients." This is NOT validation loss — it's training loss. Early stopping on training loss prevents the model from diverging (NaN, exploding loss) but doesn't detect overfitting. **Is there a held-out validation set?** The plan doesn't create one for Stage 1. With 7 source patients and small data, holding out a validation set from each patient further reduces already-thin training data. The plan instead relies on SWA + augmentation to prevent overfitting. **Reasonable trade-off, but monitor for overfitting in practice.**

**S8-C: Source replay gradient routing in Stage 2.** Source trials pass through frozen source read-ins and the frozen backbone. Gradient from source replay flows to the CTC head and LayerNorm only. **But:** the target read-in doesn't receive gradient from source trials (source input doesn't touch target read-in). And the backbone doesn't receive gradient at all (frozen). So source replay ONLY keeps the head and LayerNorm calibrated on source phoneme representations. **This is correct — it prevents catastrophic forgetting of the shared head.**

**S8-D: Stage 2 for S3 (46 trials, 5-fold).** Per fold: ~37 train, ~9 test (hold out ~7 for val). Training on 30 trials with 2,272 params. That's 30 × 200 = 6,000 frames for 2,272 params ≈ 2.6 frames per param. With augmentation (~2-3× effective multiplier), this becomes ~7-8 frames per param. Tight but feasible — the params are mostly in the articulatory head (2,064) and the read-in is tiny (80). **S3 is the stress test for Stage 2 overfitting.** Early stopping (patience 10) on the 7-trial val set is noisy.

**S8-E: The 30/70 source/target split in Stage 2 batches.** With B≈7 total (5 target + 2 source), the gradient per step is very noisy. 100 epochs × ~6 batches per epoch (assuming ~30 training trials / 5 per batch) = ~600 gradient steps. At ~7 trials per step, each training trial is seen ~600 × 5/30 = 100 times. Heavy repetition — augmentation is critical to prevent overfitting.

---

## Stage 9: Inference (CTC Decode)

```
Input: log_probs = (1, T=40, C=10) — single trial

Step 1: Greedy decode
        best_class = argmax(log_probs, dim=-1) → (40,) indices in [0..9]

Step 2: Collapse consecutive duplicates
        [0,0,0,1,1,0,0,5,5,5,0,0,8,8,0,...] → [0,1,0,5,0,8,0]

Step 3: Remove blanks (index 0)
        [1,5,8] → phonemes [/b/, /a/, /u/]

Output: decoded phoneme sequence, variable length
```

**Stress tests:**

**S9-A: Greedy vs beam search.** The plan uses greedy decoding. For 9 phonemes with no language model, beam search offers minimal benefit — the phoneme space is small and there's no sequential constraint (all CVC/VCV combinations are possible). Greedy is appropriate.

**S9-B: CTC can produce length ≠ 3.** If the model emits too many or too few phonemes, PER increases. Length-0 (all blanks) gives PER = 3/3 = 1.0. Length-6 gives edit_distance(6-phoneme, 3-phoneme) / 3 which can exceed 1.0. **Monitor CTC length accuracy.** If <80% of trials produce exactly 3 phonemes, something is wrong.

---

## Overall Architecture Summary

```
FROZEN IN STAGE 2          ADAPTED IN STAGE 2
┌──────────────────┐       ┌──────────────────┐
│                  │       │                  │
│  Conv1d(64→64)   │       │  SpatialConvReadIn│
│  20,544 params   │       │  80 params        │
│                  │       │                  │
│  BiGRU(64,64,×2) │       │  LayerNorm(64)    │
│  124,416 params  │       │  128 params       │
│                  │       │                  │
│                  │       │  ArticulatoryCTC   │
│                  │       │  2,064 params     │
│                  │       │                  │
│  Total: 144,960  │       │  Total: 2,272     │
└──────────────────┘       └──────────────────┘
```

**Key architectural properties:**
- **98.5% of shared params are frozen** in Stage 2 — temporal processing is locked
- **The read-in is the patient-specific adapter** — 80 params bridge patient-specific spatial layout to the shared representation
- **No skip connections** — information flows strictly forward through the pipeline
- **Two bottlenecks:** the 64-dim feature vector (after read-in) and the 128-dim BiGRU output (before head)
- **Factored (2+1)D design:** spatial processing (Conv2d) is fully complete before temporal processing (Conv1d + BiGRU) begins

**Open questions that need answers from Zac before implementation:**
1. Does `hgTrace` preserve the 2D grid with non-sig channels zeroed?
2. What's the stored epoch width beyond ±500ms?
3. Patient ID mapping (code IDs → paper IDs)