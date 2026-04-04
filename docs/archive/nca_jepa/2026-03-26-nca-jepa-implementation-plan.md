# NCA-JEPA Implementation Plan — MVP-Staged, Unified Architecture

**Date**: 2026-03-26
**Parent spec**: `2026-03-26-nca-jepa-pretraining-design.md` (v14)
**Status**: Draft — post-architecture-review revision

---

## Revisions from the Design Spec

The design spec (v14) has correct experimental logic but front-loads the entire implementation before any gate fires. This plan fixes three problems:

### 1. Implementation scope is not staged to match gated execution

The spec's Section 14 (Implementation Handoff) requires building all 6 generators, all 3 architectures, full anti-leakage defenses, transfer-proxy with guardrails, and all diagnostics before running Step 1 (Method E), which only needs a frozen conv-GRU and a CE head. If Step 1 shows D ≈ E (architecture has no value), weeks of generator and ViT engineering are wasted.

**Fix**: Build only what the next gate needs. Six phases, each gated on the prior phase's result.

### 2. Three separate architecture families → one configurable architecture

The spec defines Conv-GRU, Conv-Token GRU, and ViT as separate families with separate code paths. But the primary experimental axis is **spatial processing** (collapse vs preserve), not the temporal model. The Conv-Token GRU and Conv-GRU share the same temporal model (BiGRU) — they differ only in whether spatial tokens survive to the temporal stage.

**Fix**: One `UnifiedPretrainModel` with `spatial_mode` as the key configuration axis:
- `collapse` = Family A behavior (pool early, single temporal track)
- `preserve` = Conv-Token GRU behavior (keep all positions, per-position temporal)
- `attend` = lightweight ViT-like behavior (preserve + cross-position attention, deferred)

This shares ~80% of code across modes and makes the comparison cleaner.

### 3. The ViT (930K) is replaced with a lightweight attention option

The full ViT (4-layer encoder + 2-layer factorized predictor = 930K params) is 23× the Conv-Token GRU and designed for data scales we don't have. The Conv-Token GRU already tests spatial-token preservation. Cross-position attention can be tested with a single `TransformerEncoderLayer` added to the Conv-Token GRU (~20K extra params) rather than building an entirely separate architecture.

**Fix**: `spatial_mode=attend` adds 1-2 cross-position attention layers on top of `preserve` mode. Not a separate family.

### 4. Missing baseline: no-temporal-model sanity check

Before investing in temporal modeling (GRU, attention, masked prediction), test whether the spatial pattern alone is discriminative. If per-frame spatial features → temporal mean pool → classify achieves PER < 0.85, the spatial signal is strong and temporal modeling is a bonus. This takes minutes to run and directly informs how much temporal sophistication is warranted.

### 5. Spatial receptive field limits what `preserve` mode can learn from pretraining

**Critical finding from capacity analysis**: The per-position GRU in `preserve` mode gets spatial context ONLY through the 3×3 Conv2d input — a 1-hop neighborhood (~4mm). For spatially-coupled dynamics (waves, RD, FHN, NCA), information propagates across the grid at speeds far exceeding 1 cell/frame:

```
Cortical wave speed: 100-800 mm/s
At 1.33mm pitch, 20Hz: 3.7 - 30 cells/frame
Conv2d receptive field: 1 cell (3×3 kernel)
```

This means `preserve` mode **cannot model multi-cell spatial propagation within a single frame**. It can only exploit spatial coupling that unfolds slowly enough for the GRU to accumulate neighborhood information over multiple frames. For generators Level 2+ (waves, RD, FHN, NCA), `preserve` mode is structurally handicapped during pretraining even though it may outperform `collapse` at downstream classification (where spatial resolution matters but spatial dynamics don't).

**Implications**:
- Generator ladder results will be **architecture-dependent**: `collapse` can exploit complex generators (sees full spatial field); `preserve` will plateau at Level 1-2
- Conv2d receptive field is an early ablation, not Tier 2: 2-layer Conv2d (5×5 effective RF, 664 params) or dilated Conv2d should be tested in Phase 3
- `attend` mode (cross-position attention) is more important than initially proposed — it's required to fully exploit the generator ladder in `preserve`-style architectures
- If `preserve` underperforms `collapse` at pretraining but outperforms it at downstream, the spatial information flow is the pretraining bottleneck — a finding the spec doesn't predict

**Fix**: Promote `attend` mode from Phase 5 to Phase 3 (alongside `preserve`). Add Conv2d receptive field ablation (1-layer vs 2-layer) to Phase 3.

---

## Unified Architecture

### Design

```
Input: (B, H, W, T) @ 20Hz (30 frames / 1.5s trial)
                    │
        ┌───────────┼───────────┐
        │ Per-patient Conv2d    │   80 params/patient (shared across all modes)
        │ (1, C=8, k=3, pad=1) │
        │ + ReLU                │
        └───────────┬───────────┘
                    │
     ┌──────────────┼──────────────┐
     │ spatial_mode config knob    │
     ├──────────┬───────┬──────────┤
     │ collapse │preserve│ attend  │
     │          │        │         │
     │ Pool(4,8)│ keep   │ keep    │
     │ flatten  │ all    │ all     │
     │ →256-dim │ pos    │ pos     │
     │          │        │ + cross-│
     │          │        │ position│
     │          │        │ attn    │
     └────┬─────┴───┬───┴────┬────┘
          │         │        │
     ┌────┴─────────┴────────┴────┐
     │ Temporal model              │
     │ collapse: BiGRU(d,d/2,2L)  │
     │   on single d-dim track    │
     │ preserve/attend:            │
     │   shared BiGRU(d,d/2,2L)   │
     │   per spatial position     │
     └────────────┬───────────────┘
                  │
     ┌────────────┴───────────────┐
     │ Readout (downstream only)   │
     │ collapse: temporal mean     │
     │ preserve/attend:            │
     │   spatial {mean|topk}       │
     │   → temporal mean           │
     │   → CE head Linear(2H, 27) │
     └────────────────────────────┘
```

### Pretraining objective (masked span prediction)

```
Input frames → mask 40-60% of frames (replace with learned [MASK] token)
           → spatial encoder
           → temporal model (BiGRU/attention)
           → linear decoder projects back to input space
           → MSE loss on masked positions only
```

For `collapse` mode: decoder is `Linear(2H, d_flat)` → reshape to (pool_h, pool_w, C) → upsample. Predicts the pooled representation, not the raw grid (bottleneck-pretext mismatch, acknowledged and deliberate).

For `preserve` mode: decoder is `Linear(2H, 1)` per position per frame. Predicts per-electrode values directly.

### Parameter budgets

| Mode | Spatial encoder | Temporal model | Decoder | Total pretrain | Stage 3 trainable |
|------|----------------|---------------|---------|----------------|-------------------|
| `collapse` | Pool+flatten: 0 | BiGRU(256→128, 2L): ~600K; or BiGRU(64→32, 2L): ~37K | Linear(64,256): 16K | ~71K (d=64) | ~2.3K |
| `preserve` | Linear(8,d)+PE: ~800 | Shared BiGRU(64,32,2L): ~37K | Linear(64,1): 65 | ~39K (d=64) | ~2.2K |
| `attend` | Linear(8,d)+PE+Attn: ~20K | Shared BiGRU(64,32,2L): ~37K | Linear(64,1): 65 | ~58K (d=64) | ~2.5K |

All modes use the same per-patient Conv2d (80 params) and the same BiGRU configuration. The `attend` mode adds ~20K params for 1 cross-position attention layer — NOT a separate 930K ViT.

### Interface

```python
# Config-driven, extends existing assembler.py pattern
pretrain_config = {
    "spatial_mode": "preserve",       # collapse | preserve | attend
    "d": 64,                          # latent dimension
    "gru_hidden": 32,                 # BiGRU hidden (per direction)
    "gru_layers": 2,
    "readout": "mean",                # mean | topk
    "topk_k": 16,                     # only if readout=topk
    "mask_ratio": [0.4, 0.6],         # uniform sample per batch
    "mask_spans": [3, 6],             # number of contiguous spans
    "local_geometry_pe": True,        # only for preserve/attend modes
    "cross_position_layers": 0,       # 0 = preserve, 1-2 = attend
}
```

---

## MVP Phases

### Phase 0: Evaluation Infrastructure + Baselines (2-3 days)

**Build scope**:
1. Grouped-by-token CV splitter (deterministic from patient ID, save to JSON)
2. CE mean-pool readout using existing `SharedBackbone` + new simple `Linear(2H, 27)` head
3. Patient-level PER aggregation (mean across folds per patient)
4. Content-collapse metrics (output entropy, unigram-KL, stereotypy index)

**Reuse from existing code**:
- `SpatialConvReadIn` (spatial_conv.py) — unchanged
- `SharedBackbone` (backbone.py) — unchanged, use stride=10 for 20Hz
- `load_patient_data`, `load_grid_mapping` — unchanged
- Per-patient training loop (trainer.py) — adapt to grouped-by-token CV + CE loss

**Run**:
- **Method E** (random-init frozen): Freeze conv-GRU, train only CE head. Minutes.
- **Method D** (supervised from scratch): Train conv-GRU end-to-end with CE. ~30 min/patient.
- **Spatial-only baseline** (NEW): Per-frame Conv2d features → temporal mean pool → CE head. No BiGRU. Tests whether the spatial pattern alone is discriminative. Minutes.

**Gate 0**:
- If D ≈ E → architecture has no value at all with this evaluation protocol. Debug before proceeding.
- If spatial-only ≈ D → temporal model adds nothing. Focus on spatial representation, skip temporal SSL.
- If D > spatial-only > E → temporal model helps, proceed to SSL.

**Output**: Baseline PER table (E, D, spatial-only) per patient. Content-collapse report.

### Phase 1: Minimal SSL on Real Data (1 week)

**Build scope** (only what Method B needs):
1. Masked span prediction module:
   - Learnable `[MASK]` token (d-dim)
   - Span masking strategy: select 3-6 non-overlapping spans totaling 40-60% of frames
   - Linear decoder: `Linear(2H, d_input)` for reconstruction
   - MSE loss on masked positions
2. Stage 2 training loop (simplified):
   - Load real trial data (existing loaders, no anti-leakage yet)
   - Train encoder+predictor with masked span prediction
   - Checkpoint every N steps
3. Stage 3 fine-tuning:
   - Freeze backbone, train CE head + Conv2d on grouped-by-token splits
   - Reuse Phase 0 evaluation pipeline

**NOT building yet**: Anti-leakage (off-center crops, time-reversal), transfer-proxy, phase-rebalanced loss, any synthetic data, conv-token GRU mode.

**Run**:
- **Method B** (neural-only pretrain): Masked span prediction on all source patients' unlabeled trials → freeze → CE head per target patient. S_total steps.
- Compare B vs D vs E.

**Gate 1**:
- If B ≈ E → pretraining objective is broken. Debug masking ratio, learning rate, decoder capacity before proceeding.
- If B > E but B ≈ D → SSL matches supervised but doesn't improve. Interesting but not sufficient for the paper's claims. Proceed cautiously.
- If B > D → SSL helps. Proceed to synthetic.

**Diagnostics**: Pretext MSE curve, interpolation baseline comparison, event-frame vs background MSE split.

### Phase 2: Minimal Synthetic Transfer (1 week)

**Build scope** (only what Methods C and one-generator-A need):
1. **ONE generator**: Smooth Gaussian AR (Level 0, simplest, ~50 lines of code)
   - AR(1) process: `x_{t+1} = alpha * x_t + (1-alpha) * smooth(noise)`, sigma=2-4 cells
   - No calibration needed (no regime-dependent parameters)
2. Synthetic data pipeline:
   - Z-score normalize
   - IID noise injection (sigma ~ U[0.3, 0.8])
   - Dead electrode simulation (real templates)
   - Grid size sampling (8×16, 12×22)
3. Stage 1 training loop:
   - Masked span prediction on synthetic data (same objective as Phase 1)
   - S_total/2 steps synthetic → S_total/2 steps neural → freeze → CE head
4. **ONE structured generator**: Switching LDS or damped wave (Level 1 or 2)
   - To test if structure adds value beyond smoothness

**Run**:
- **Method C** (smooth AR → neural): S_total/2 synthetic + S_total/2 neural
- **Method A-minimal** (one structured generator → neural): Same schedule, switching LDS
- Compare A-minimal vs C vs B vs D vs E

**Gate 2**:
- If C ≈ B → synthetic data doesn't transfer at all. The entire synthetic pretraining hypothesis is dead. Publish null result with B vs D.
- If C > B → synthetic exposure helps. Structure question: if A-minimal > C, structured dynamics matter.
- If A-minimal ≈ C > B → smooth is enough. Narrower claim but still publishable. Don't build 4 more generators.
- If A-minimal > C > B → structure helps. Expand generator ladder.

**Matching statistics** (lightweight version): Compare temporal PSD and spatial covariance of smooth AR and switching LDS vs real data. Full 6-stat battery deferred.

### Phase 3: Spatial Architecture Comparison (1-2 weeks, can overlap Phase 2)

**Build scope**:
1. Refactor existing `SpatialConvReadIn` to support `spatial_mode`:
   - `collapse`: existing behavior (Pool → flatten)
   - `preserve`: remove pool, add `Linear(C, d)` per position + local geometry PE `Linear(3, d)`
   - `attend`: `preserve` + 1-2 cross-position `TransformerEncoderLayer(d, nhead=2)` per frame (~20K extra params)
2. Refactor `SharedBackbone` to support per-position GRU:
   - `collapse`: existing behavior (single BiGRU over d-dim track)
   - `preserve`/`attend`: shared BiGRU applied independently per spatial position
3. Spatial pooling at readout:
   - Mean-pool across positions
   - Top-k pool (k=16, select by L2 norm per frame, then temporal mean)
4. Unified config: `spatial_mode` field controls all the above
5. Conv2d receptive field ablation: 1-layer 3×3 (default, 80 params) vs 2-layer 3×3 (5×5 effective RF, 664 params)

**Why `attend` is promoted to Phase 3** (from original Phase 5 deferral): The capacity analysis (Revision §5) revealed that `preserve` mode's per-position GRU is structurally limited in what it can learn during pretraining — it cannot model spatial dynamics that propagate >1 cell/frame (waves, RD, NCA). Cross-position attention in `attend` mode provides the spatial information flow needed to learn from the full generator ladder. Without testing `attend` alongside `preserve`, we can't distinguish "spatial preservation doesn't help" from "spatial preservation helps but the model can't learn the pretraining task."

**Run**:
- Method D (supervised) with `collapse` vs `preserve` vs `attend` → downstream spatial preservation gate
- Method B (neural SSL) with `collapse` vs `preserve` vs `attend` → pretraining × architecture interaction
- If Phase 2 complete: Method C or A-minimal with all 3 modes → synthetic × architecture interaction
- Conv2d RF ablation: 1-layer vs 2-layer with `preserve` mode on Method B

**Gate 3**:
- If `preserve` D >> `collapse` D → spatial collapse hurts even supervised. Strong result.
- If `preserve` B >> `collapse` B but `preserve` D ≈ `collapse` D → spatial preservation specifically amplifies pretraining. Strongest architecture result.
- If `attend` B >> `preserve` B but `attend` D ≈ `preserve` D → cross-position attention specifically helps pretraining (the model needs spatial info flow to learn from masked prediction). This confirms the spatial RF limitation.
- If `preserve` ≈ `collapse` everywhere → spatial preservation doesn't matter at this resolution. Skip `attend`.
- If 2-layer Conv2d >> 1-layer with `preserve` → receptive field is the bottleneck, not attention. Cheaper fix.

**Predicted outcome** (from capacity analysis): `attend` will outperform `preserve` at pretraining (because it can exploit spatially-coupled generators), but `preserve` may match `attend` at downstream classification (because mean-pool readout doesn't need spatial dynamics). If this happens, the headline architecture depends on the pretraining→downstream pipeline: `attend` for pretraining, `preserve`-level readout for classification.

### Phase 4: Full Experimental Design (2 weeks)

**Build scope** (only if Phases 1-3 show signal):
1. Full generator ladder (remaining 4 generators):
   - Damped wave, Gray-Scott, FHN, NCA random MLP
   - Coarse source-only calibration per generator
   - Full matching statistics (6-stat battery)
2. Anti-leakage defenses for Stage 2:
   - Full-epoch off-center random crops (load [-1.0, 1.5s], crop 30-40 frames)
   - Time-reversal augmentation (10%)
3. Transfer-proxy gate:
   - Single fixed dev patient
   - 5K-step probe, earliest-within-1SE rule
   - Source-only sensitivity analysis
4. Remaining comparison methods:
   - Method F (random-scaffold control)
   - Method K (destroyed-dynamics control)
   - Method J (B-extended, automatic if A > B)
5. Diagnostics:
   - Per-phase prediction loss tracking
   - Patient-ID and phoneme-ID linear probes
   - JEPA promotion trigger monitoring

**Run**:
- Methods A (full mixed pool), F, K with best architecture from Phase 3
- Full generator ladder ablation (Tier 1 E)
- Interpretation matrix (Section 6, Step 4 of spec)

**Gate 4**:
- A > K > C > B → structured dynamics help beyond matched statistics. Strongest claim.
- A ≈ K > C > B → spatial statistics help, not dynamics. Narrower claim.
- A ≈ C > B → any smooth pretraining helps. Weakest publishable.
- A ≈ B ≈ D → SSL doesn't help beyond supervised. Null result on pretraining (publish as negative).

### Phase 5: Expansion + Polish (1-2 weeks)

**Conditional on Phase 4 signal**:
1. Scale up `attend` mode if it won Phase 3:
   - 2-4 attention layers (from 1-2), larger d
   - This is the path toward the original ViT concept, but incrementally justified
2. JEPA follow-up (if reconstruction is shortcut-prone):
   - Replace observation-space MSE with latent target prediction
   - Stop-gradient on target encoder, EMA update
3. Ablation matrix:
   - Masking ratio (30/50/70%), temporal resolution (20/40Hz), patch size (1×1 vs 2×2)
   - Freeze level (head+embed+LN vs full fine-tune)
   - Source weighting (uniform vs similarity-weighted)
4. Full evaluation:
   - Surrogate null (100× label permutation)
   - Wilcoxon signed-rank with Holm-Bonferroni
   - Bootstrap 95% CIs
   - Per-patient plots

---

## What's Explicitly Deferred

| Item | Spec section | Reason for deferral |
|------|-------------|---------------------|
| Full 930K ViT | §5 Family B | Replaced by lightweight `attend` mode at 1/16 the params |
| Factorized space-time predictor | §5 Family B | Not needed — BiGRU serves as both encoder and predictor for all modes |
| Per-patient conv stem for ViT | §5, Resolved #54 | No ViT, no need for ViT adapter ablation |
| Hotspot trajectory generator | §4.1 | Add to generator pool in Phase 4 only if ladder shows signal |
| Micro-event structure (40Hz) | §4.1 | Requires 40Hz, which showed no benefit per-patient. Tier 2 ablation |
| Phase-rebalanced loss | §4.2 | Phase-aware variant, not default. Add in Phase 4 if needed |
| Lexical patients in Stage 2 | §8 Tier 2 | Cross-task pooling adds confounds. Phase 5+ |
| MNI-based PE | §5, Open Q | Local geometry PE is default. MNI is Tier 2 |
| Transductive evaluation | §7 | Secondary mode. Implement after inductive is working |
| SVD-based per-patient layer init | TRIBE v2 (d'Ascoli 2026) | Low-rank factorization of source read-ins → init target read-in. Needs ≥10 source patients. Phase 5+ |
| Patient-identity dropout | TRIBE v2 (d'Ascoli 2026) | During Stage 1, p=0.1 route through shared read-in (bypass per-patient Conv2d). Forces patient-agnostic backbone. Phase 4+ |

---

## File Structure (New Code)

```
src/speech_decoding/
├── pretraining/
│   ├── __init__.py
│   ├── unified_model.py          # UnifiedPretrainModel (spatial_mode config)
│   ├── masking.py                # Span masking strategy + [MASK] token
│   ├── decoder.py                # Linear decoder for masked prediction
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── base.py               # Generator interface
│   │   ├── smooth_ar.py          # Phase 2: Level 0
│   │   ├── switching_lds.py      # Phase 2: Level 1
│   │   ├── wave.py               # Phase 4: Level 2
│   │   ├── gray_scott.py         # Phase 4: Level 3
│   │   ├── fhn.py                # Phase 4: Level 4
│   │   └── nca.py                # Phase 4: Level 5
│   ├── stage1_trainer.py         # Synthetic pretraining loop
│   ├── stage2_trainer.py         # Neural adaptation loop
│   ├── stage3_evaluator.py       # CE fine-tune + content-collapse diagnostics
│   ├── transfer_proxy.py         # Phase 4: dev-patient checkpoint selection
│   └── matching_stats.py         # Phase 4: PSD, spatial cov, amplitude, etc.
├── evaluation/
│   ├── grouped_cv.py             # Phase 0: grouped-by-token CV splitter
│   ├── content_collapse.py       # Phase 0: entropy, unigram-KL, stereotypy
│   └── metrics.py                # (existing) PER, balanced accuracy
configs/
├── pretrain_base.yaml            # Shared pretraining defaults
├── pretrain_collapse.yaml        # spatial_mode=collapse
├── pretrain_preserve.yaml        # spatial_mode=preserve
└── pretrain_attend.yaml          # Phase 5: spatial_mode=attend
scripts/
├── train_pretrain.py             # CLI for pretraining pipeline
└── run_phase0_baselines.py       # Quick script for Methods E, D, spatial-only
tests/
├── test_unified_model.py         # Forward pass for all spatial_modes
├── test_masking.py               # Span masking coverage, ratio verification
├── test_generators.py            # Generator output shapes, statistics
└── test_grouped_cv.py            # CV split coverage constraints
```

### What's reused from existing code (no changes needed)

| Existing file | Reused for |
|---|---|
| `data/bids_dataset.py` | `load_patient_data()` — data loading for all stages |
| `data/grid.py` | `load_grid_mapping()` — grid shapes, dead masks |
| `data/audio_features.py` | `load_phoneme_timing()` — for Phase 4 anti-leakage |
| `data/augmentation.py` | Pre-read-in augmentation (time shift, amplitude, noise) |
| `models/spatial_conv.py` | Per-patient Conv2d — reused inside `unified_model.py` |
| `evaluation/metrics.py` | PER, balanced accuracy |
| `training/ctc_utils.py` | CTC decode for transfer stress test |

### What's adapted from existing code (minor changes)

| Existing file | Adaptation |
|---|---|
| `models/backbone.py` | Extract BiGRU into reusable component; add per-position mode |
| `models/assembler.py` | Add `assemble_pretrain_model()` alongside existing `assemble_model()` |
| `training/trainer.py` | Adapt to grouped-by-token CV + CE loss for Stage 3 |

---

## Key Architectural Decisions (Revised)

### 1. No separate ViT family

**Rationale**: The Conv-Token GRU already tests the spatial-preservation hypothesis at 40K params. The ViT's additional contribution is cross-position temporal attention, which can be tested by adding a single attention layer to the Conv-Token GRU (`attend` mode, ~58K params). Building a 930K factorized space-time transformer from scratch is unjustified before seeing whether spatial preservation matters at all.

**If `attend` >> `preserve`**: Cross-position interaction is valuable. At that point, scaling up to a fuller transformer becomes a justified investment. But this is Phase 5, not Phase 1.

### 2. BiGRU is the temporal model for all modes

**Rationale**: Field precedent (Singh, Willett, Nason all use GRU/LSTM). Changing the temporal model confounds the spatial-processing comparison. For T=30 frames, BiGRU is adequate. TCN is noted as a Tier 2 ablation for the Conv-Token architecture where parallelism matters (264 independent GRU sequences), but is not part of the first round.

### 3. The per-position BiGRU is shared (weight-tied)

**Rationale**: The Conv-Token GRU processes each spatial position independently through the same BiGRU. This is analogous to weight sharing in Conv2d — the same temporal filter applied everywhere. Cross-position information flows only through the 3×3 Conv2d kernel and through readout pooling. This is the cleanest test of spatial-token preservation.

**Compute note**: 264 positions × 30 frames at batch_size=8 = 2112 effective sequences for the shared GRU. This is within PyTorch's efficient batched GRU regime.

### 4. Masked span prediction operates on the temporal model's INPUT, not output

For `collapse` mode: mask 40-60% of the 30 d-dim frame vectors, replace with learned [MASK], run through BiGRU, decode from GRU output back to d-dim frame vectors.

For `preserve` mode: same, but applied independently per spatial position. Each position's 30-frame temporal sequence is masked identically (all positions share the same temporal mask). The decoder reconstructs per-electrode values.

**Why same mask across positions**: Different masks per position would allow the model to "cheat" by inferring masked frames at position i from observed frames at neighboring position j (spatial correlation shortcut). Shared temporal mask forces genuine temporal prediction.

### 5. spatial_mode=collapse uses the existing architecture verbatim

The `collapse` mode is literally the current `SpatialConvReadIn` + `SharedBackbone` + CE head. This ensures backward compatibility with all existing training results and makes Method D/E baselines directly comparable to the PER 0.700 result (modulo grouped-by-token CV vs stratified CV).

---

## Verification Tasks (External Facts)

These are **not design blockers** — proceed with defaults and flag results:

1. **8×32 / 8×34 array pitch**: Use 1.33mm surrogate. Flag as `surrogate-pitch` in results. Ask Zac.
2. **Dev patient**: Default S26 (median trials, 8×16, 74 sig channels). Confirm with Zac.
3. **Motor-focused crop boundaries**: Default [-0.15, +0.6s]. Patient-specific from MFA timing is Phase 4.

---

## Timeline Summary

| Phase | Duration | Build | Run | Gate |
|-------|----------|-------|-----|------|
| 0 | 2-3 days | Grouped-CV, CE head, collapse diagnostics | E, D, spatial-only | Architecture viability |
| 1 | ~1 week | Masked span prediction, Stage 2/3 loops | B | SSL viability |
| 2 | ~1 week | Smooth AR + switching LDS generators | C, A-minimal | Synthetic transfer viability |
| 3 | 1-2 weeks | `preserve` + `attend` modes, per-position GRU, PE, Conv2d RF | D/B/C × {collapse, preserve, attend} | Spatial processing + capacity |
| 4 | ~2 weeks | Full generators, anti-leakage, transfer-proxy, F/K | A, F, K, full ladder | Dynamics vs statistics |
| 5 | 1-2 weeks | Scale `attend`, JEPA, ablations, full evaluation | Expansion | Final claims |

**Critical path**: Phase 0 → 1 → 2 (serial, each gated). Phase 3 can overlap with Phase 2.
**Total to first publishable result**: ~5-6 weeks (if each gate passes).
**Total to null result**: 1-2 weeks (if Gate 0 or Gate 1 fails).

---

## Capacity Analysis: Can These Models Learn Synthetic Dynamics?

### Parameter-to-data ratio is adequate

| System | Input/sample | Params | Ratio |
|--------|-------------|--------|-------|
| Our `preserve` (d=64) | 128 × 30 = 3,840 | ~39K | 0.10 |
| Our `collapse` (d=64) | 32 × 30 = 960 | ~71K | 0.01 |
| VideoMAE (ViT-S) | 224² × 3 × 16 = 2.4M | 22M | 0.11 |

The data-to-parameter ratio is comparable to successful masked prediction models. With unlimited synthetic data, the parameter count is not the bottleneck — the model won't overfit.

### Per-generator capacity requirements

| Level | Generator | Rule complexity | Hidden state needed | 37K GRU feasible? |
|-------|-----------|----------------|--------------------|--------------------|
| 0 | Smooth AR | 1 param (α) | Trivial | Yes — massively overparameterized |
| 1 | Switching LDS | 3-6 regimes × 9-param kernel | Regime ID + linear dynamics | Yes — 64-dim hidden can represent ~8 modes |
| 2 | Damped wave | Speed + damping, local PDE | Neighborhood history | Yes — if spatial RF covers propagation speed |
| 3 | Gray-Scott | 2 params (F, k), nonlinear | Reaction-diffusion manifold | Marginal — needs to approximate nonlinear dynamics |
| 4 | FHN | 4 params, partially observed (w hidden) | State estimation of unobserved w | Marginal — 64-dim must encode hidden variable |
| 5 | NCA random MLP | 321 params per rule, DIFFERENT each sequence | In-context rule inference | Insufficient for exact prediction — 64-dim compresses 321-dim rule space at 5:1 |

**Key insight**: The model will learn Level 0-2 well, Level 3-4 approximately, and Level 5 poorly. But **imperfect learning of complex dynamics may still produce useful features**. The model's limited capacity forces it to learn GENERAL spatial-temporal features (correlation structure, temporal smoothness, local coupling patterns) rather than memorizing specific rules. These general features are exactly what should transfer to real neural data.

### The spatial receptive field is the binding constraint, not parameters

For `preserve` mode, the 3×3 Conv2d provides 1-hop spatial context. Information that propagates >1 cell/frame is invisible to the per-position GRU:

```
Generator spatial propagation speeds (at 20Hz, 1.33mm pitch):
  Smooth AR:     0 cells/frame (no spatial coupling)     → Conv2d RF: sufficient
  Switching LDS: 0-1 cells/frame (local kernel)          → Conv2d RF: sufficient
  Damped wave:   1-10 cells/frame (wave speed dependent)  → Conv2d RF: INSUFFICIENT for fast waves
  Gray-Scott:    0.5-3 cells/frame (diffusion + reaction) → Conv2d RF: marginal
  FHN:           1-5 cells/frame (excitable propagation)  → Conv2d RF: INSUFFICIENT
  NCA:           1 cell/frame (3×3 neighborhood rule)     → Conv2d RF: sufficient (by construction)
```

This creates a **generator × architecture interaction**: `preserve` mode will benefit most from generators with local dynamics (AR, switching LDS, NCA), while `collapse` mode (which sees the full spatial field, coarsened) will benefit from generators with long-range dynamics (waves, FHN).

The `attend` mode resolves this by providing global spatial information flow at each frame, enabling the model to exploit all generator levels.

### Practical implication for the generator ladder

The generator ladder results should be interpreted **per architecture mode**:
- In `collapse` mode: the ladder tests the full inductive bias hierarchy
- In `preserve` mode: the ladder primarily tests local dynamics (Levels 0-1, 5) and partially tests coupled dynamics (Levels 2-4, limited by RF)
- In `attend` mode: the ladder tests the full hierarchy with spatial-token preservation

If `preserve` + Level 5 (NCA, local rules) >> `preserve` + Level 2 (waves, non-local) but `collapse` + Level 2 >> `collapse` + Level 5, this confirms that spatial RF is the bottleneck, not model capacity. This interaction is an interesting finding in itself.

---

## Relationship to Design Spec

This plan **implements** the design spec's experimental logic but **restructures** the build order. All comparison methods (A through K), generator families, evaluation metrics, and anti-leakage defenses from the spec are preserved — they're just built incrementally as gates pass rather than all upfront.

The one substantive change is the architecture: three separate families → one configurable model. The experimental comparisons are identical (collapse vs preserve isolates spatial processing; preserve vs attend isolates cross-position attention). The parameter budgets are comparable. The scientific claims are unchanged.

The spec's 82 resolved decisions remain in force. This plan adds:
- **RD-83**: MVP staging — build only what the next gate needs
- **RD-84**: Unified architecture with `spatial_mode` configuration axis
- **RD-85**: ViT replaced by lightweight `attend` mode (~58K, not 930K)
- **RD-86**: No-temporal-model baseline added to Phase 0
- **RD-87**: Same temporal mask across all spatial positions (anti-spatial-correlation shortcut)
- **RD-88**: `attend` mode promoted to Phase 3 (was Phase 5) — spatial RF limits `preserve` mode's ability to learn from spatially-coupled generators
- **RD-89**: Conv2d receptive field ablation (1-layer vs 2-layer) added to Phase 3
- **RD-90**: Generator ladder results interpreted per architecture mode — `preserve` structurally limited to local-dynamics generators (Levels 0-1, 5)
- **RD-91**: SVD-based per-patient layer initialization deferred to Phase 5+ — factorize source read-in weights via SVD, use low-rank approximation to initialize target patient read-in. Requires ≥10 source patients to be meaningful. Inspired by TRIBE v2 (d'Ascoli et al. 2026) subject block factorization for OOD subjects.
- **RD-92**: Patient-identity dropout deferred to Phase 4+ — during Stage 1 multi-patient training, with p=0.1 bypass per-patient Conv2d and route through a shared average read-in. Forces backbone to learn patient-agnostic representations. Inspired by TRIBE v2 "unseen subject" dropout mechanism. Risk: may hurt performance if patient-specific spatial patterns are essential for learning good temporal representations.
