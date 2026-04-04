# Training Log — S14 Per-Patient + LOPO Pilot

> **HISTORICAL**: This log covers CTC→CE tuning and LOPO pilot (2026-03). The current best is per-phoneme MFA + flat head (PER 0.734, 2026-04-04). See `experiment_log.md` findings 86-101 for current results.

## Per-Patient Summary (S14, 153 trials, 8×16 grid, 5-fold CV)

| Run | Key change | Seed 42 | Seed 137 | Seed 256 | Mean PER |
|-----|-----------|---------|----------|----------|----------|
| 1 | H=64 baseline | 1.000 | — | — | 1.000 |
| 2 | H=32, bias=2.0 | 1.000 | 1.000 | 0.976 | 0.992 |
| 3 | bias=0.5 | 0.863 | 0.850 | 0.811 | 0.841 |
| **4** | **bias=0.0** | **0.767** | **0.751** | **0.815** | **0.778** |
| 5 | aug tune | 0.784 | 0.811 | 0.823 | 0.806 |
| 6 | weight_decay=1e-3 | 0.782 | 0.811 | 0.823 | 0.805 |
| 7 | lr=5e-4 | 0.767 | 0.859 | 0.858 | 0.828 |
| 8 | gru_dropout=0.1 | 0.803 | 0.749 | 0.812 | 0.788 |
| **Flat** | **flat CTC head** | **0.763** | **0.760** | **0.792** | **0.772** |
| **CE** | **CE loss, H=32, s=5** | **0.693** | **0.691** | **0.715** | **0.700** |
| CE-H128 | CE H=128 (killed) | — | — | — | overfits worse |
| CE-s10 | CE stride=10 (20Hz) | 0.699 | 0.711 | 0.737 | 0.716 |
| CE-pool48 | CE s10 + pool(4,8) d=256 | 0.709 | — | — | ~0.71 (killed) |
| CE-c16 | CE s10 + C=16 d=128 | — | — | — | not run |
| CE-balanced | CE s10 + pool(4,8) + H=64 | — | — | — | next run |

## Per-Patient Run Details

### Run 1: H=64 baseline
- **Model**: ~147k params, stride=10, bias=2.0, compound dropout ~40%
- **Result**: PER=1.000 — catastrophic overfitting

### Run 2: H=32 + reduced dropout + stride=5 + warmup
- **Model**: ~43k params, stride=5 (60 GRU frames), bias=2.0, compound ~23%
- **Result**: PER=0.992 — **blank collapse**. Model learns (train→1.4) but decode stays all-blank.

### Run 3: blank_bias=0.5
- **Result**: PER=0.841. High fold variance. Some folds escape blank, others collapse.

### Run 4: blank_bias=0.0 ★ BEST ARTICULATORY
- **Result**: PER=0.778. Consistent (no collapses). Blank≈95%.

### Run 5: Augmentation tuning
- **Changes**: time_shift 30→15, amp_scale 0.3→0.15, noise 0.05→0.10
- **Result**: PER=0.806. Worse — original augmentation is better.

### Run 6: Weight decay 1e-3
- **Result**: PER=0.805. No improvement. Overfitting is data-limited, not regularization-limited.
- **Note**: Seeds 137/256 may have used Run 5 aug (sed issue) — only seed 42 is clean.

### Run 7: LR 5e-4
- **Result**: PER=0.828. Worse. Lower LR + same patience = insufficient learning before early stop.

### Run 8: GRU dropout 0.1
- **Result**: PER=0.788. Marginal. Lower dropout doesn't help — bottleneck is data.

### Flat Head Ablation ★ KEY FINDING
- **Result**: PER=0.772 (mean 3 seeds). **Identical to articulatory (0.778)**.
- **Conclusion**: Articulatory decomposition provides **zero per-patient benefit**. Value is cross-patient only.

### CE Loss Ablation ★ KEY FINDING
- **Config**: Same architecture (Conv2d + BiGRU-32), but per-position CE instead of CTC
- **Head**: Linear(64, 27) — 3 positions × 9 phonemes, global temporal mean pool, CE per position
- **Result**: PER=0.700 (mean 3 seeds), bal_acc=0.266
- **vs CTC Run 4**: PER 0.778→0.700 (10% relative improvement), bal_acc 0.15→0.27 (77% improvement)
- **Training dynamics**: No blank collapse, train loss drops to ~0.5 (vs CTC floor ~1.4), early stops ~100-160 epochs
- **Conclusion**: CTC hurts in small-data per-patient regime. The alignment-freedom is wasted capacity — we already know phoneme positions from epoch structure. CE provides stronger per-position gradients.
- **Gap to Duraivel 2023**: His BiLSTM-800 seq2seq got ~50.8% accuracy on same patient with ~300× more params. Our 26.6% bal_acc suggests RNN capacity and/or input layer compression are bottlenecks.

### CE H=128 Ablation ★ KEY FINDING (negative)
- **Config**: CE loss + BiGRU-128 (4× larger, ~330K backbone params)
- **Result**: KILLED — train loss dropped to 0.045 (vs 0.5 for H=32) while val stayed at 2.5. 55× train/val gap vs 4× for H=32. Massive overfitting.
- **Conclusion**: **GRU capacity is NOT the bottleneck.** The model memorizes training data faster with more capacity but doesn't learn better features. The bottleneck is upstream in the input layer — Conv2d→pool(2,4)→64 dims compresses too aggressively.

### CE stride=10 (20Hz) Ablation (in progress)
- **Config**: Same as CE baseline but temporal_stride=10 → 30 GRU frames instead of 60
- **Result so far**: seed 42 PER=0.699, seed 137 PER=0.711 — essentially identical to stride=5 (0.700)
- **Conclusion**: 20Hz is sufficient; 40Hz added compute without benefit. Matches Spalding/Duraivel field standard.

### Input Layer Sweep (partial — interrupted)
- **Run A** (CE s10, pool 2×4, d=64): PER=0.716 (mean 3 seeds). Stride change neutral vs s=5 (0.700).
- **Run B** (CE s10, pool 4×8, d=256): seed 42 PER=0.709. **Pool(4,8) alone doesn't help** — severe overfitting (train 0.31 vs val 2.96). Conv1d(256,32,k=10) has 82K params projecting 256→32 dims; too many params for 120 trials.
- **Run C** (CE s10, C=16, d=128): not run yet.

### Balanced Config ★ NEXT EXPERIMENT
- **Config**: `per_patient_ce_balanced.yaml` — pool(4,8) + H=64, d_shared=256
- **Rationale**: Pool(4,8) alone failed because H=32 creates a massive projection bottleneck (Conv1d 256→32). Increasing H=64 gives Conv1d(256,64) = 164K + BiGRU(64,64,2L) = 99K = **~263K total** (6× current, 50× smaller than Duraivel).
- **Key insight**: Input richness and backbone capacity must be scaled together. Duraivel's 50.8% accuracy came from rich input (111 raw channels) + large backbone (BiLSTM-800, 4.5M params). Our prior experiments showed that increasing only one (H=128 alone or pool(4,8) alone) makes overfitting worse.
- **Prediction**: If the balanced config still doesn't improve, the bottleneck is data quantity (~120 trials/fold), not architecture.

## Key Patterns

1. **CE > CTC per-patient** — PER 0.700 vs 0.778 (CE wins by removing wasted alignment learning)
2. **Input layer is the bottleneck, not GRU** — H=128 overfits MORE (train/val gap 55× vs 4×). The 128→64 spatial compression via pool(2,4) loses articulatory resolution
3. **Stride=10 (20Hz) ≈ stride=5 (40Hz)** — 0.70 vs 0.70. Field standard is correct, 40Hz was unnecessary
4. **Blank bias was the dominant CTC lever** — 2.0→0.0 took PER from 1.000 to 0.778
5. **Flat head ≈ articulatory head per-patient** — both ~0.77. Articulatory value is cross-patient only
6. **CTC hyperparameter sweeps exhausted** — aug, WD, LR, dropout all within noise of Run 4
7. **Cross-lab comparison**: We're 100-300× smaller than every system achieving >30% accuracy. Even Singh (closest) uses BiLSTM-64. Every BCI lab with good results uses ≥500 hidden units + per-patient input layers with 66K-262K params

## LOPO Pilot (complete)

**Config**: `lopo_pilot.yaml` — 8 Spalding patients, 1 seed (42), H=64, CTC, blank_bias=1.0
- Stage 1: 2000 steps (early-stops ~800), 7 source patients
- Stage 2: 100 steps per fold, 5-fold CV (4 for S33), 30% source replay
- ~10 min per target, ~80 min total

| Patient | LOPO PER | Per-Patient CTC PER | Per-Patient CE PER |
|---------|----------|--------------------|--------------------|
| S14 | 0.828 | 0.778 | 0.700 |
| S22 | 0.851 | — | — |
| S23 | 0.849 | — | — |
| S26 | 0.846 | — | — |
| S33 | 0.853 | — | — |
| S39 | 0.856 | — | — |
| S58 | 0.854 | — | — |
| S62 | 0.832 | — | — |
| **Population** | **0.846 ± 0.010** | — | — |

**LOPO dynamics**:
- Stage 1 trains 7 sources, consistently early-stops ~step 800 (val diverges after step 500)
- Stage 2 folds complete in seconds (~3-5s each, early-stop 13-46 steps)
- Very low patient variance (0.828-0.856) — near-chance uniformly

**Conclusions**:
- LOPO (0.846) is **worse** than per-patient CTC (0.778 on S14) — cross-patient transfer is not helping
- Per-patient CE (0.700) is best, suggesting the model architecture + loss matters more than data pooling
- The near-uniform LOPO PER suggests the model isn't extracting patient-specific signal during Stage 2
- **Next steps**: Try CE for LOPO, larger input layer (pool 4×8), or more Stage 2 adaptation steps

## Pending
- [ ] Input layer sweep results (pool 4×8, C=16)
- [ ] CE for LOPO
- [ ] Run per-patient CE on all 8 Spalding patients (not just S14)

## Speech-Embedding Regression Pivot (real S14, 2026-03-18)

### Goal
- Test whether a dense auxiliary target improves the CE baseline by regularizing the backbone toward speech-relevant representations.
- Original hypothesis: paired audio could provide denser supervision beyond the 3-position CE labels.
- Actual execution path:
  - Phase 0: validate neural/audio alignment
  - Phase 1: framewise acoustic auxiliary target with matched CE-only controls
  - Negative result triggered a pivot away from acoustic regression targets

### What was implemented
- Added audio feature extraction, alignment validation, regression trainer, masked/segment losses, and diagnostics.
- New outputs:
  - `results/alignment_checks/sub-S14_alignment.png`
  - `results/audio_features/hubert/sub-S14_hubert.npz`
  - `results/audio_features/mel/sub-S14_mel.npz`
- Regression path constraints:
  - CE branch kept identical to the matched CE baseline
  - `lambda=0` always used as the control
  - PCA fit per fold on train only
  - Time-shift and temporal-stretch disabled for regression runs to preserve frame alignment

### Phase 0: Alignment validation ★ PASSED
- **Patient**: S14
- **Check**: MNE epoch response-onset samples vs phoneme timing CSV response-onset timestamps
- **Result**:
  - median diff = `0.000000 s`
  - max absolute diff = `0.000233 s`
- **Conclusion**: paired audio and neural epochs are aligned well enough for downstream experiments. The later regression failure is **not** due to a gross timing bug.

### Regression target experiments (all on real S14, 5-fold CV, seed 42)

#### 1. HuBERT framewise regression ★ NEGATIVE
- **Config**:
  - target = HuBERT layer-6 embeddings
  - framewise masked MSE on speech frames
  - matched CE-only control under identical config
  - shifted-control = neural data shifted +500 ms
- **Results**:
  - `joint` (`lambda=0.3`): `PER=0.780`, `bal_acc=0.173`, `r2_speech=-0.014`, `r2_silence=-0.157`
  - `ce_only` (`lambda=0.0`): `PER=0.786`, `bal_acc=0.178`
  - `shifted_500ms`: `PER=0.773`, `bal_acc=0.198`, `r2_speech=0.013`, `r2_silence=-0.147`
- **Conclusion**:
  - No meaningful aligned regression signal
  - Shifted control is not worse than aligned
  - HuBERT target is **not** a useful auxiliary objective in this regime

#### 2. Mel framewise regression ★ NEGATIVE FOR ALIGNMENT
- **Reason to try**: determine whether HuBERT was the problem or whether acoustic targets more generally were the problem
- **Results**:
  - `joint` (`lambda=0.3`): `PER=0.773`, `bal_acc=0.188`, `r2_speech=0.069`
  - `ce_only`: `PER=0.786`, `bal_acc=0.167`
  - `shifted_500ms`: `PER=0.784`, `bal_acc=0.176`, `r2_speech=0.065`
- **Conclusion**:
  - Mel features are slightly more predictable than HuBERT
  - But the shifted control is nearly identical
  - So the positive `R²` is **not alignment-sensitive**
  - This suggests the model is learning broad trial structure, not a useful motor-to-acoustic mapping

#### 3. HuBERT segment-level regression with speech-only PCA ★ NEGATIVE
- **Reason to try**: relax the dense framewise constraint and use coarser MFA-derived phoneme segments
- **Config**:
  - segment-level pooled targets from phoneme boundaries
  - PCA fit on speech frames only
- **Result**:
  - `joint` (`lambda=0.3`): `PER=0.775`, `bal_acc=0.179`, `r2_segment=-0.002`
- **Conclusion**:
  - Coarsening the target did **not** rescue the auxiliary signal
  - Failure is not simply “framewise objective too strict”

### Overall conclusion from regression pivot
- **Important negative result**: acoustic auxiliary targets are not useful here, even when:
  - alignment is verified
  - controls are matched
  - target family changes (HuBERT vs mel)
  - target granularity changes (frame vs segment)
- **Best interpretation**:
  - The limiting issue is not an implementation bug or a single bad target choice
  - The motor-cortex HGA in this small intra-op regime does not support a sufficiently alignment-sensitive mapping to acoustic feature targets to improve phoneme decoding
- **Decision**: stop pursuing acoustic regression targets and pivot to label-derived phonological/articulatory auxiliary targets instead

## Phonological Auxiliary Loss Pivot (real S14, 2026-03-19)

### Goal
- Replace the failed acoustic auxiliary path with a target defined entirely from the phoneme labels.
- Use the existing `ARTICULATORY_MATRIX` (15 binary features) as a biologically grounded auxiliary target.
- Keep the CE branch unchanged so comparisons remain clean.

### What was implemented
- Added a pooled per-position phonological auxiliary loss:
  - target = 15 binary articulatory/phonological features derived from each phoneme label
  - head = linear projection from backbone states to `3 × 15` logits
  - loss = per-position BCE on mean-pooled hidden states
- Control remains the same per-position CE setup with `lambda=0`.

### Sweep setup
- **Patient**: S14
- **Window**: `[-0.5, 1.0]`
- **Model**: CE baseline config (`temporal_stride=10`, `hidden=32`, `spatial_conv`, 5-fold CV, seed 42)
- **Metric of interest**: phoneme PER / balanced accuracy
- **Aux metrics**:
  - `feature_acc`: mean binary feature accuracy
  - `feature_exact`: exact match of all 15 features at a position

### Completed lambda sweep results

| Lambda | PER | Bal Acc | Feature Acc | Feature Exact | Interpretation |
|--------|-----|---------|-------------|---------------|----------------|
| `0.00` | `0.699` | `0.266` | — | — | Matched CE baseline |
| `0.10` | `0.685` | `0.274` | `0.626` | `0.009` | Small positive |
| `0.20` | `0.712` | `0.250` | `0.605` | `0.009` | Worse than baseline |
| `0.30` | `0.681` | `0.278` | `0.617` | `0.022` | **Best completed run** |
| `0.50` | `0.697` | `0.254` | `0.622` | `0.004` | Back toward baseline |

### Best completed result ★ CURRENT BEST AUXILIARY
- **Lambda = 0.30**
- **vs matched CE baseline**:
  - `PER`: `0.699 → 0.681` (`-0.018` absolute, about `2.6%` relative reduction)
  - `bal_acc`: `0.266 → 0.278` (`+0.012` absolute, about `4.5%` relative increase)
- **Conclusion**:
  - This is the first auxiliary-target experiment that produced a real positive result on S14
  - The gain is modest, but it is larger than the acoustic auxiliary gains and is directionally consistent with the motor-cortex hypothesis

### Shape of the lambda curve
- `0.10`: helps
- `0.20`: hurts
- `0.30`: helps more than `0.10`
- `0.50`: loses most of the benefit
- **Interpretation**:
  - The auxiliary target is useful, but only in a narrow weighting range
  - Too much auxiliary pressure pulls the model away from the phoneme objective

### Incomplete fine sweep
- Started a tighter sweep around the best region:
  - `lambda = 0.15, 0.25, 0.35`
- This run was **interrupted by the user** before completion, so no final numbers should be reported from it.
- At interruption time, the evidence already supported:
  - useful region around `0.1–0.3`
  - `0.3` outperforming both the CE baseline and `0.5`

## Current Takeaways

1. **Acoustic auxiliary targets failed cleanly.**
   - Alignment is good.
   - HuBERT failed.
   - Mel gave nonzero `R²` but not alignment-sensitive.
   - Segment-level HuBERT failed.
   - This is a real negative result, not a tooling bug.

2. **Label-derived phonological targets are meaningfully better than acoustic targets.**
   - They are cheap, stable, and aligned by construction.
   - They produced the first real auxiliary-loss improvement on S14.

3. **The best completed auxiliary setting is currently `lambda=0.30`.**
   - `PER=0.681`, `bal_acc=0.278`
   - Better than the matched CE baseline (`0.699`, `0.266`)

4. **The improvement is modest, not transformative.**
   - Enough to justify continuing this direction
   - Not enough to claim a major representation breakthrough yet

5. **Next step should be generalization, not more acoustic experiments.**
   - Finish the fine lambda sweep around `0.3`
   - Then test the winning setting on another patient (`S22` or `S23`)
   - If it generalizes, then fold it into the broader per-patient / LOPO plan

## Pending
- [ ] Input layer sweep results (pool 4×8, C=16)
- [ ] CE for LOPO
- [ ] Run per-patient CE on all 8 Spalding patients (not just S14)
- [ ] Finish fine phonological auxiliary lambda sweep (`0.15`, `0.25`, `0.35`)
- [ ] Validate phonological auxiliary gain on a second patient (`S22` or `S23`)
- [ ] Decide whether to combine phonological aux with the best future input-layer config
