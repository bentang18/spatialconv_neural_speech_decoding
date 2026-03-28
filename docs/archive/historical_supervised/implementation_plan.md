# Implementation Plan v9 — Cross-Patient uECOG Speech Decoding

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** Data verified (2026-03-14). Architecture redesigned (2026-03-15). Reptile/SupCon/DRO downgraded to exploratory (2026-03-16). Spatial conv revised: 1-layer Conv2d grounded in electrode placement physics (2026-03-17). Full pipeline stress test: blank init fix, Stage 1 validation set, augmentation corrections, compute estimate (2026-03-17).

**Goal:** End-to-end cross-patient phoneme decoder from intra-op uECOG, beating Spalding's 0.31 balanced accuracy. Task: non-word repetition (52 CVC/VCV tokens, 3 phonemes each, e.g. /abe/, from 9 phonemes). CTC targets are 3-phoneme sequences (e.g., [a, b, e]). Phase 2 modernizes the pipeline to field consensus. Phase 3 introduces two architectural innovations absent from the neural speech BCI field.

**Tech Stack:** Python 3.10+, PyTorch 2.x, wandb. All experiments run with 3 seeds (42, 137, 256); report mean ± std across seeds.

---

## Two-Stage Strategy

**Phase 2 — Field standard.** Replace Spalding's PCA+CCA+SVM with the architecture every modern speech BCI converges on: per-patient learned read-in → shared GRU backbone → CTC loss (Singh, Boccato, Levin, Willett, Nason). No novelty — just catching up to the field. This is a complete, validated pipeline and a publishable improvement over Spalding.

**Phase 3 — Our innovations.** Replace Phase 2 components with two architectural techniques absent from the field:

| # | Innovation | Replaces (Phase 2 →) | Ablation | Core idea |
|---|---|---|---|---|
| 1 | Per-patient spatial conv | Linear(208,64) | E3 | Conv2d factorization matches uECOG physics; 80–664 params vs 13,376 |
| 2 | Articulatory CTC head | Flat Linear(128,10) | E4 | Decode what motor cortex encodes — articulatory gestures, not phoneme categories |

E1 (Phase 2 output) has neither. E2 (Phase 3 full model) has both. E3–E4 each remove one for attribution. Training procedure is identical to Phase 2 (standard multi-patient SGD, no meta-learning or contrastive loss).

**Downgraded to exploratory (see `future_directions.md`):** Reptile, SupCon, and DRO. Per-patient input layers already provide cross-patient transfer (Singh-style). Reptile's "adaptability" is marginal when backbone is frozen in Stage 2. SupCon is redundant with CTC + per-patient layers. DRO upweights patients that are hard for practical reasons (few trials, poor signal quality), not neural coding differences — with N=7 source patients, the worst-case estimate is too noisy to be useful.

---

## Phase 0: Setup & Data Verification

### 0.1 Codebase

Clone `github.com/coganlab/cross_patient_speech_decoding`. Identify reusable components (data loading, evaluation) and what we replace (PCA+CCA → learned read-in, SVM → GRU+CTC).

**Data access (confirmed 2026-03-17):** Data on BOX (Ben downloading) + Duke compute cluster (debug + run modes). BOX contains **15 patients/sessions**: 8 phoneme task (Spalding's published patients) + ~7 word/nonword task. Raw data AND Zac-preprocessed version both available. Zac sharing preprocessing repo — check his pipeline first, re-preprocess from raw if we want changes (wider epoch ±600ms for augmentation, keep all channels instead of sig-only for Conv2d).

**Workflow:** Develop and debug locally first. Deploy to Duke cluster for full experiment runs (~88 GPU-hours).

**HGA features are pre-extracted** as `.mat` files: `{subj}_HG_p{phoneme}_sigChannel_goodTrials.mat` containing `hgTrace` (trials × ch_x × ch_y × time), `hgMap` (trials × time × channels), `phonSeqLabels` (trials × 3). Training pickle: `pt_decoding_data_S62.pkl`. Patient IDs in code (S14, S23…) differ from paper IDs (S1–S8) — mapping will be apparent from filenames.

**Questions — resolve on data inspection (Zac meeting 2026-03-17):**
1. ~~Confirm hgTrace ch_x, ch_y = physical grid dims~~ → Check raw data. Orientation varies per patient (confirmed).
2. ~~Epoch width~~ → Customizable in preprocessing. Re-preprocess from raw with ±600ms if needed.
3. **Grid shape discrepancy:** 12×22 = 264 positions but arrays are labeled "256ch." Check raw data for dead positions.
4. Verify `hgTrace` has full grid with non-sig channels zeroed (not excluded). If sig-only, re-preprocess.
5. Verify `phonSeqLabels` indexing (0-based or 1-based). CTC blank = index 0.
6. Inventory the ~7 word/nonword patients: which task, how many trials, do they share the 9 phonemes?

### 0.2 Data — CONFIRMED

| Patient | Diagnosis | Array | Trials | Sig channels | Frames |
|---|---|---|---|---|---|
| S1 | Parkinson's | 8×16, 128ch, 1.33mm | 144 | 111/128 | 28,800 |
| S2 | Parkinson's | 8×16, 128ch, 1.33mm | 148 | 111/128 | 29,600 |
| S3 | Tumor | 12×22, 256ch, 1.72mm | **46** | 149/256 | **9,200** |
| S4 | Parkinson's | 8×16, 128ch, 1.33mm | 151 | 74/128 | 30,200 |
| S5 | Parkinson's | 8×16, 128ch, 1.33mm | 151 | 63/128 | 30,200 |
| S6 | Tumor | 12×22, 256ch, 1.72mm | 137 | 144/256 | 27,400 |
| S7 | Tumor | 12×22, 256ch, 1.72mm | 141 | 171/256 | 28,200 |
| S8 | Tumor | 12×22, 256ch, 1.72mm | 178 | 201/256 | 35,600 |

**Totals:** ~8.23 min utterance, ~68.51 min experiment. ±500ms window at 200 Hz = 200 frames/trial. MNI-152 coordinates and paired audio (lavalier mic, MFA labels) confirmed. **Array placements span 15–25mm in MNI space** — arrays are shifted AND rotated relative to each other, with some barely overlapping. A dense central overlap zone exists where most arrays share coverage, but peripheral coverage is patient-specific. This means: (a) electrode channel numbers are meaningless across patients (channel 47 records from different cortex in different patients), (b) per-channel diagonal scaling is insufficient, (c) the per-patient layer must handle both orientation and positional variation.

### 0.3 Code structure

```
src/
├── config/
│   ├── default.yaml              # Full model (E2)
│   ├── field_standard.yaml       # Field standard (E1)
│   └── ablations/{no_spatial_conv, no_art_decomp}.yaml
├── data/
│   ├── dataset.py                # uECOG dataset with grid-shaped loading
│   ├── augmentation.py           # Time shift, amp scale, ch dropout, noise (pre-read-in)
│   └── inspect_data.py           # Phase 0: data inspection (grid shapes, label encoding, patient inventory)
├── models/
│   ├── spatial_conv.py           # Per-patient Conv2d read-in (Phase 3)
│   ├── linear_readin.py          # Per-patient Linear read-in (Phase 2)
│   ├── backbone.py               # Conv1d temporal + BiGRU (shared)
│   ├── articulatory_head.py      # Articulatory CTC head (Phase 3)
│   ├── flat_head.py              # Flat CTC head (Phase 2)
│   └── model.py                  # Full model assembly (config-driven)
├── training/
│   ├── standard.py               # Standard multi-patient SGD (Phase 2 + 3)
│   ├── lopo.py                   # LOPO cross-validation orchestration (shared)
│   └── evaluator.py              # PER, balanced accuracy, CTC stats (shared)
├── analysis/
│   ├── rsa.py                    # Representational similarity analysis (Phase 3)
│   ├── prototypical.py           # Prototypical accuracy diagnostic (Phase 3)
│   └── ablation_table.py         # Compile results across experiments
└── scripts/
    ├── train.py                  # Main entry (config-driven)
    └── analyze.py                # Post-training diagnostics
```

### 0.4 Deliverables

- [ ] Data inspection script run — resolve §0.1 questions 3–6 (grid shape, hgTrace format, label indexing, patient inventory)
- [ ] Patient inventory: confirm 8 phoneme + ~7 word/nonword, document task/trial count/grid size per patient
- [ ] Decide if re-preprocessing needed (wider epoch ±600ms, full grid with non-sig zeroed)
- [ ] D_padded confirmed (currently 208 = max sig channels across 8 patients; may change with 15)
- [ ] Documented codebase structure (reusable vs replace)
- [x] Data verification table (8 phoneme patients) — **DONE**
- [x] Frame budget → backbone size locked (H=64) — **DONE**
- [x] Audio and MNI coords confirmed — **DONE**
- [x] Array orientation variability confirmed — **DONE**

---

## Phase 1: ~~Reproduce Spalding Baseline~~ SKIPPED

**Decision (2026-03-17):** Skip baseline reproduction. Use Spalding's published 0.31 balanced accuracy as E0 reference. For paired statistical tests (E2 vs E0), Zac's original scripts can be run to generate per-patient E0 numbers. This saves ~1 week — the scientific contribution starts at Phase 2.

**E0 reference values (from Spalding 2025):**

| Condition | Value |
|---|---|
| Patient-specific SVM | ~0.24 |
| Unaligned cross-patient SVM | ~0.19 |
| **CCA-aligned cross-patient SVM** | **~0.31** |

---

## Phase 2: Field-Standard Pipeline (E1)

Modernize Spalding's PCA+CCA+SVM to the architecture every modern speech BCI converges on. This phase builds a **complete, validated pipeline** — a standalone deliverable and the baseline for Phase 3.

### 2.1 Architecture

```
uECOG grid (8×16 or 12×22 × T, 200 Hz)
  ├─ Per-patient Linear(D_padded, 64) read-in
  ├─ LayerNorm(64) → Conv1d(64,64,k=K,s=K)+GELU [K=5→40Hz or K=10→20Hz, configurable]
  ├─ BiGRU(64, 64, 2 layers, dropout=0.2)
  └─ Linear(128, 10) → CTC (9 phonemes + blank)
```

**Temporal downsampling rate:** Configurable via `temporal_stride` (default K=5 → 40Hz). Spalding validated K=10 → 20Hz on our exact data/task. Willett uses ~12.5Hz for full sentences. 20Hz is sufficient for 3-phoneme CTC (20 frames for 3 phonemes + blanks). Start with K=5; switch to K=10 if training is slow or CTC converges without issues at lower resolution.

#### Per-Patient Linear Read-In

Each patient's significant channels are zero-padded to D_padded (max sig channels across all patients; 208 for the 8 phoneme patients — confirm after inspecting all 15 patients' data) and projected to D_shared=64. Per-patient because array placements span 15–25mm in MNI space with variable orientation — electrode channel numbers have no consistent anatomical meaning across patients, so the projection must learn patient-specific remapping. The full matrix W can learn arbitrary remapping, but at ~13k params from ~150 trials, overfitting is a serious concern — motivating Phase 3's spatial conv (80 params with spatial inductive bias matching the rigid-array physics).

```python
class LinearReadIn(nn.Module):
    """Per-patient: one instance per patient."""
    def __init__(self, D_padded=208, D_shared=64):
        super().__init__()
        self.linear = nn.Linear(D_padded, D_shared)  # 208*64+64 = 13,376 params

    def forward(self, x):
        # x: (B, D_padded, T) — flattened, zero-padded channels
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, D_shared, T)
```

#### Shared Backbone

Conv1d for temporal downsampling (200→40 Hz default, 200→20 Hz with K=10), BiGRU for temporal dynamics. Shared across all patients — the common representational space. Temporal stride is configurable — Spalding validated 20 Hz on our exact task; 40 Hz provides finer resolution but doubles BiGRU sequence length.

```python
class SharedBackbone(nn.Module):
    def __init__(self, D=64, H=64, feat_drop_max=0.3, temporal_stride=5):
        super().__init__()
        self.layernorm = nn.LayerNorm(D)
        self.feat_drop_max = feat_drop_max
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(D, D, kernel_size=temporal_stride, stride=temporal_stride),  # K=5→40Hz, K=10→20Hz
            nn.GELU()
        )
        self.gru = nn.GRU(D, H, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.2)

    def _feature_dropout(self, x):
        """Spatial dropout on shared features: p ~ U[0, feat_drop_max].
        Drops entire feature channels (same mask for all B×T positions).
        This is intentional — forces backbone robustness to losing entire
        feature dimensions, not just individual activations."""
        if not self.training:
            return x
        p = torch.rand(1).item() * self.feat_drop_max
        mask = (torch.rand(x.shape[-1], device=x.device) > p).float()
        return x * mask / (1 - p + 1e-8)

    def _time_mask(self, x):
        """Zero 2–4 contiguous frames with cosine taper after temporal conv."""
        if not self.training:
            return x
        T = x.shape[1]
        mask_len = torch.randint(2, 5, (1,)).item()  # min 2: mask_len=1 has no effect with cosine taper
        start = torch.randint(0, max(T - mask_len, 1), (1,)).item()
        taper = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, mask_len, device=x.device)))
        x = x.clone()
        x[:, start:start+mask_len, :] *= (1 - taper).unsqueeze(-1)
        return x

    def forward(self, x):
        # x: (B, D, T) from read-in
        x = self.layernorm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self._feature_dropout(x.permute(0, 2, 1)).permute(0, 2, 1)  # §2.3 aug #5
        x = self.temporal_conv(x)       # (B, D, T//K)
        x = x.permute(0, 2, 1)          # (B, T//K, D)
        x = self._time_mask(x)          # §2.3 aug #6
        h, _ = self.gru(x)              # (B, T//K, 2H)
        return h                         # (B, T//K, 128) — T//K=40 at K=5, 20 at K=10
```

#### Flat CTC Head

Standard projection from backbone hidden states to phoneme log-probabilities. Same interface as ArticulatoryCTCHead (§3.2) — both return log_softmax'd output.

```python
class FlatCTCHead(nn.Module):
    def __init__(self, input_dim=128, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)  # 128*10+10 = 1,290 params

    def forward(self, h):
        # h: (B, T, 2H)
        return F.log_softmax(self.linear(h), dim=-1)  # (B, T, 10)
```

#### Parameter Budget

| Component | Params |
|---|---|
| Per-patient Linear(208,64) | 13,376 |
| LayerNorm(64) | 128 |
| Conv1d(64, 64, k=5) | 20,544 |
| BiGRU(64, 64, 2 layers, bidir) | 124,416 |
| Flat CTC head Linear(128, 10) | 1,290 |
| **Total shared** | **~146k** |
| **Per-patient** | **13,376** |

*BiGRU layer 2 receives 2H=128 input (bidirectional output of layer 1).*

### 2.2 Training

#### Stage 1 — Multi-Patient Training

Standard SGD on all source patients jointly (Singh-style group model). Per-patient read-ins absorb cross-patient variation; the shared backbone learns patient-invariant representations via CTC.

**Stage 1 validation:** Hold out 20% of each source patient's trials (stratified by phoneme) as a validation set. Early stopping evaluates on this held-out data, not training data. This costs ~210 trials from training (~840 remain) but provides genuine overfitting detection. Without this, early stopping monitors training loss, which always decreases and cannot detect overfitting to source patient idiosyncrasies.

```python
def standard_stage1(model, read_ins, train_data, val_data, config):
    """Multi-patient training with gradient accumulation across patients.
    train_data/val_data: per-patient dicts (80/20 stratified split of source trials)."""
    optimizer = AdamW(
        [{'params': read_ins[p].parameters(), 'lr': config.lr * 3}
         for p in source_patients] +
        [{'params': model.parameters(), 'lr': config.lr}],
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    best_val_loss, patience_counter = float('inf'), 0
    best_state = None

    for epoch in range(config.epochs):  # 200
        optimizer.zero_grad()
        for p in source_patients:
            x, y = sample_batch(train_data[p], B=16)
            x = augment(x)                     # time shift, amp scale, ch dropout, noise
            h = model.backbone(read_ins[p](x))
            logits = model.ctc_head(h)
            loss = ctc_loss(logits, y) / len(source_patients)  # accumulate normalized
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()                       # one step per epoch (eff. batch = 7×16 = 112)
        scheduler.step()

        # Early stopping on HELD-OUT validation data (evaluated every 10 epochs)
        if (epoch + 1) % 10 == 0:
            val_loss = mean([evaluate_ctc(model, read_ins[p], val_data[p])
                            for p in source_patients])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 5:          # 50 epochs without improvement
                break

    model.load_state_dict(best_state)          # restore best checkpoint
```

| Parameter | Value | Rationale |
|---|---|---|
| Epochs | 200 (max) | Early stopping may terminate sooner |
| Backbone LR | 1e-3 | Standard for GRU |
| Read-in LR | 3e-3 | Partially compensates for fewer gradient contributions per step (1 gradient per read-in vs 7 accumulated for backbone). Note: backbone still updates ~2.3× faster in effective magnitude due to 7× gradient accumulation |
| Batch size | 16 per patient | Gradient accumulated across 7 patients → effective batch 112 |
| Weight decay | 1e-4 | Light regularization |
| Grad clip | 5.0 | CTC stability |
| Scheduler | CosineAnnealingLR(T_max=200) | Single cosine decay to 0; simple, no restarts |
| Early stopping | Patience 5 (×10 epochs = 50 epochs) | Evaluated on held-out 20% of source trials; prevents overfitting to source patient idiosyncrasies |
| Stage 1 val split | 80/20 stratified per source patient | ~840 training / ~210 validation trials across 7 source patients |

#### Stage 2 — Target Adaptation

For each fold of the target patient's trials:

1. Load Stage 1 best checkpoint (backbone + flat head + LayerNorm)
2. Initialize new LinearReadIn for target (Kaiming init)
3. **Freeze:** Conv1d + BiGRU
4. **Unfreeze:** LinearReadIn + LayerNorm + flat CTC head
5. Train on target with CTC loss + source replay
6. Evaluate on target's test fold

| Setting | Value |
|---|---|
| Folds | 5-fold **stratified** CV on target patient (StratifiedKFold on phoneme-at-each-position labels). Fold seed fixed per training seed for reproducibility |
| Epochs | 100 (max) |
| LR | 1e-3, constant |
| Weight decay | 1e-3 (Phase 2: 14,794 params → appropriate). **Phase 3 note:** with only 2,272 trainable params (80 in Conv2d), 1e-3 may suppress spatial filter learning — reduce to 1e-4 if Conv2d filters collapse toward zero (monitor filter norm) |
| Augmentation | Time shift ±50ms, amplitude scaling σ=0.05, noise 1% RMS (lighter than Stage 1) |
| Source replay | 30% source / 70% target (Levin 2026); uniform across all source patients using their frozen Stage 1 read-ins |
| Early stopping | Patience 10 epochs on target fold's validation loss (20% of training fold held out) |
| Trainable params | 13,376 + 128 + 1,290 = **~14,794** |

**Source replay mechanics:** Each Stage 2 batch contains ~5 target trials (70%) + ~2 source trials (30%). Source trials are drawn uniformly across all 7 source patients, processed through their frozen Stage 1 read-ins and the frozen backbone. Source replay gradient flows only to the CTC head and LayerNorm (not the target read-in), preventing the head from forgetting source phoneme representations during adaptation.

**Inference:** Stage 2 adapts read-in, LayerNorm, and head against the best Stage 1 checkpoint. Final inference uses a single forward pass with greedy CTC decode — no ensemble.

### 2.3 Augmentation

| Augmentation | Placement | Implementation | Simulates |
|---|---|---|---|
| Time shifting | Before read-in | Shift window ±100ms (±20 frames), zero-pad edges | Response onset variation (0.56 ± 0.35s RT) and cross-patient neural latency differences |
| Per-channel amplitude scaling | Before read-in | `scale = exp(N(0, 0.15²))`, constant within trial | Electrode impedance variation (dominant for Utah per Boccato; uECOG also has spatial remapping) |
| Channel dropout | Before read-in | Zero entire electrodes, p ~ U[0, 0.1] (Phase 2); p ~ U[0, 0.2] (Phase 3, after Conv2d validated) | Bad/missing electrodes; sig channel variation (63–201 across patients). Conservative in Phase 2 because Conv2d handles grid sparsity better than Linear |
| Gaussian noise | Before read-in | `x += N(0, (0.02·std(x))²)` per frame | Recording noise, biological noise |
| Feature dropout (spatial) | After LayerNorm | Zero entire feature channels (same mask for all B×T positions), p ~ U[0, 0.3] in D_shared space. Intentional spatial dropout — forces backbone robustness to losing whole feature dimensions | Unreliable read-in features during Stage 2 adaptation |
| Smooth time masking | After temporal conv | Zero 2–4 contiguous frames, cosine taper (min 2: mask_len=1 has no effect with cosine ramp) | Transient artifacts; forces BiGRU temporal context usage |

Time shifting is critical for cross-patient generalization: without it, CTC learns a narrow alignment prior tied to source patients' timing profiles. Spalding's pipeline includes ±100ms time shifting. Epoch width is customizable in preprocessing (confirmed 2026-03-17) — re-preprocess from raw with ±600ms to enable clean ±100ms shifts without zero-padding.

### 2.4 Evaluation

| Metric | Description |
|---|---|
| **PER** (primary) | edit_distance(pred, target) / 3; chance ≈ 0.89 |
| **Balanced accuracy** (secondary) | Per position (P1, P2, P3), averaged |
| Per-patient PER | Identifies which patients benefit from modernized pipeline |
| CTC length accuracy | % trials producing exactly 3 phonemes |

**CTC target format:** Each trial's target is a 3-phoneme non-word sequence (e.g., [a, b, e] for /abe/). The 52 unique CVC/VCV tokens mostly have different phonemes at each position, so CTC rarely faces repeated consecutive labels. Some tokens repeat a phoneme at positions 1 and 3 (e.g., /bab/ → [b, a, b]), but these are separated by a different phoneme — CTC handles this easily. Monitor CTC length accuracy (% trials producing exactly 3 phonemes) as a basic health check.

**Hyperparameter policy:** All hyperparameters (LR, batch size, weight decay, augmentation strength, etc.) are fixed a priori from literature values (Singh, Boccato, Levin, Nichol). No hyperparameter tuning on evaluation data. With N=8 patients in LOPO, nested cross-validation is prohibitively expensive and any tuning on LOPO folds constitutes data leakage.

**Statistical tests:** Wilcoxon signed-rank on per-patient PER (N=8 paired observations). Each patient's PER is the **mean across 3 seeds** (not individual seed results, which would inflate N and violate independence).
- **Primary comparisons (α=0.05):** E2 vs E0 (total gain), E2 vs E1 (innovation gain). These two test the main hypotheses and are pre-specified.
- **Exploratory comparisons:** E2 vs E{3–7} (per-innovation attribution). Report effect sizes (Cohen's d) alongside p-values. These are not corrected for multiple comparisons — they inform scientific understanding, not confirmatory claims.
- At N=8, Wilcoxon achieves p < 0.05 only if ≥7/8 patients improve. Report per-patient results regardless of significance.

**Target:** Significantly beat Spalding's 0.31 bal acc. Singh's group model improved PER from 0.57 (single) to 0.49 (group) — a ~14% relative improvement. We expect a similar or larger jump given the end-to-end training advantage over Spalding's staged PCA+CCA.

### 2.5 Sanity Checks

1. **Shape:** Forward pass through LinearReadIn → backbone → flat CTC head → scalar CTC loss → gradient to all params
2. **Overfit:** Single patient, no LOPO/augmentation → CTC loss → 0, PER → 0 within 100 epochs
3. **CTC output:** Verify predictions contain multiple phoneme classes (not collapsed to one)
4. **Dry run:** One LOPO fold, Stage 1 (50 epochs) + Stage 2 (50 epochs) → reasonable PER
5. **CTC length accuracy:** On the overfit test, verify CTC outputs length-3 sequences for >80% of trials. If model collapses to length <3, check blank emission patterns. Targets are 3-phoneme non-words (e.g., [a, b, e]) — repeated consecutive labels are rare, so CTC alignment should be straightforward.
6. **Blank ratio:** Monitor fraction of frames assigned to CTC blank. Healthy: 40–70%. If >90% → model under-emitting; if <20% → model over-emitting (likely producing length >3 sequences)

### 2.6 Deliverables

- [ ] Field-standard pipeline validated (beats Spalding, p < 0.05)
- [ ] Per-patient results table (E0 vs E1)
- [ ] Training curves and CTC statistics
- [ ] Data loading, LOPO, evaluation infrastructure confirmed working

---

## Phase 3: Two Architectural Innovations (E2–E4)

Replace Phase 2 components with two architectural techniques absent from the neural speech BCI field. Training procedure is identical to Phase 2 (standard multi-patient SGD with uniform loss weighting). All infrastructure (LOPO, augmentation, metrics) carries over from Phase 2.

### 3.0 Full Model Architecture

```
uECOG grid (8×16 or 12×22 × T, 200 Hz)
  ├─ Per-patient Conv2d(1,8,k=3,pad=1)+ReLU → AdaptiveAvgPool2d(2,4) → flatten(64)
  │   [default 1-layer 80 params; configurable depth, channels, pool — see §3.2]
  ├─ LayerNorm(64) → Conv1d(64,64,k=K,s=K)+GELU [configurable]     ← unchanged
  ├─ BiGRU(64, 64, 2 layers, dropout=0.2)                           ← unchanged
  └─ Articulatory CTC: 6 feature heads → fixed composition → 9 phoneme logits + blank
Training: Standard multi-patient SGD (same as Phase 2), best-checkpoint selection
```

### 3.1 Development Path: Progressive Addition

Innovations are added **incrementally**, verifying each helps before adding the next. Both innovations are architectural and independent — ordering is flexible.

| Step | Change from previous | Config |
|---|---|---|
| **E1: Field Standard** | — | Linear + SGD + flat CTC |
| **E1a: +SpatialConv** | Replace Linear with Conv2d | SpatialConv + SGD + flat CTC |
| **E2: +ArtHead (Full Model)** | Replace flat CTC with articulatory | SpatialConv + SGD + articulatory CTC |

**Development rule:** Only proceed to the next step if the current step does not degrade PER.

**Quick validation:** E1a can be validated with a single LOPO fold + 1 seed before full 3-seed runs. Full 3-seed evaluation is reserved for the final experiment matrix.

### 3.1.1 Final Experiment Matrix (for paper)

Once E2 is validated via progressive addition, run the full ablation matrix for attribution:

| Experiment | Read-in | Training | CTC head |
|---|---|---|---|
| **E0: Spalding** | PCA+CCA | SVM | Flat (SVM) |
| **E1: Field Standard** | Linear(208,64) | Standard SGD | Flat(128→10) |
| **E2: Full Model** | Spatial Conv2d | Standard SGD | Articulatory |
| E3: −SpatialConv | Linear(208,64) | Standard SGD | Articulatory |
| E4: −ArtDecomp | Spatial Conv2d | Standard SGD | Flat(128→10) |

**Attribution:**
- E1 − E0 = gain from modernizing the pipeline (Phase 2)
- E2 − E1 = gain from two architectural innovations (Phase 3)
- E2 − E0 = total gain
- E2 − E3 = spatial conv contribution
- E2 − E4 = articulatory head contribution
- E3 vs E4 = which innovation contributes more
- Progressive addition (E1→E1a→E2) shows each innovation's contribution when added incrementally

**Results columns:** Mean PER, worst-patient PER, mean balanced accuracy, per-patient PER, Wilcoxon p vs E0, Wilcoxon p vs E1.

### 3.2 Architecture Innovations

#### Innovation 1: Per-Patient Spatial Conv

Replaces Phase 2's LinearReadIn. The design is grounded in three physical facts about uECOG recording:

**1. Arrays have uniform intra-array properties but variable inter-array orientation.** Each rigid array has fixed pitch and orientation — a 3×3 spatial filter means the same thing at every position on a given patient's grid. But electrode (1,1) varies anatomically across patients (confirmed: arrays are visibly rotated and shifted in MNI space). This exactly matches Conv2d's factorization: weight-shared within image (same filter everywhere on one patient's grid), different parameters across images (each patient learns its own oriented filters).

**2. Adjacent electrodes are correlated at r≈0.6 due to volume conduction**, blurring the true spatial signal. A learned 3×3 filter can act as a surface Laplacian (center-minus-surround) or oriented gradient, suppressing common-mode volume conduction and sharpening somatotopic activation patterns. CAR (upstream preprocessing) is a coarse spatial filter — learned per-patient filters can do better, adapting to each array's impedance pattern and orientation.

**3. Array placements span 15–25mm across patients** (MNI projection shows some arrays barely overlap). At this offset scale, electrode channel numbers are meaningless across patients — channel 47 in one patient records from different cortex than channel 47 in another. Diagonal per-channel scaling (Boccato-style) is inapplicable. Conv2d + pooling provides a principled alternative: learned spatial filters detect local features (edges, gradients) regardless of position, and spatial pooling provides coarse positional encoding.

**Architecture:** Configurable number of Conv2d layers (default 1). A single 3×3 layer already provides all three physics-grounded operations (deblurring, gradient detection, orientation adaptation). A second layer expands the receptive field from 3×3 to 5×5 and enables cross-channel mixing before pooling — but the subsequent AdaptiveAvgPool2d(2,4) averages over 4×4 (8×16) or 6×5.5 (12×22) electrodes, which may wash out the second layer's contribution. **Whether the second layer helps is an empirical question** — start with 1 layer, quick-validate 2-layer on 1 LOPO fold before committing.

```python
class SpatialConvReadIn(nn.Module):
    """Per-patient: one instance per patient. Configurable depth."""
    def __init__(self, grid_h, grid_w, C=8, num_layers=1, pool_h=2, pool_w=4):
        super().__init__()
        layers = [nn.Conv2d(1, C, kernel_size=3, padding=1), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Conv2d(C, C, kernel_size=3, padding=1), nn.ReLU()]
        self.convs = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        self.out_dim = C * pool_h * pool_w  # default: 8*2*4 = 64

    def forward(self, x):
        # x: (B, grid_h, grid_w, T) — raw grid-shaped HGA
        B, H, W, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B * T, 1, H, W)
        x = self.convs(x)                                    # (B*T, C, H, W)
        x = self.pool(x)                                     # (B*T, C, pool_h, pool_w)
        x = x.reshape(B, T, -1)                              # (B, T, out_dim)
        return x.permute(0, 2, 1)                             # (B, out_dim, T)
```

#### Spatial Conv Hyperparameters

First-principles analysis determines the design family (per-patient Conv2d + pool) and constrains the search space. The exact settings within that space are empirical. Quick-validate on 1 LOPO fold, 1 seed before committing to full 3-seed runs.

| Hyperparameter | Default | Alternatives | What it controls | Derivable from first principles? |
|---|---|---|---|---|
| `num_layers` | **1** | 2, 3 | Receptive field (3×3, 5×5, 7×7) and capacity | No. 1-layer gives all basic spatial operations (Laplacian, gradients). 2+ layers add cross-channel mixing and larger RF, but aggressive pooling may negate the RF benefit |
| `C` (channels) | **8** | 4, 16 | Number of learned spatial filters | Partially. Need ≥4 for basic operations (Laplacian, 2 gradients, smoothing). 8 allows specialization. >8 increases params with diminishing returns |
| `pool_h, pool_w` | **(2, 4)** | (3, 4), (4, 4), (4, 8) | Spatial resolution vs cross-patient robustness | Yes — constrained by placement offsets. Coarser pooling (2,4) is more robust to 15–25mm offsets; finer pooling (4,4) preserves more spatial detail but pool cells map to different cortex across patients |
| `kernel_size` | **3** | 5 | Single-layer receptive field | Yes — 3×3 matches spatial correlation length (~3 electrodes at r≈0.6). 5×5 is viable but fewer alternatives per layer |

**Parameter count by configuration:**

| Config | Layers | C | Pool | Params | Output dim | RF |
|---|---|---|---|---|---|---|
| Minimal | 1 | 4 | (4,4) | 40 | 64 | 3×3 |
| **Default** | **1** | **8** | **(2,4)** | **80** | **64** | **3×3** |
| Moderate | 2 | 8 | (2,4) | 664 | 64 | 5×5 |
| Large | 3 | 8 | (2,4) | 1,248 | 64 | 7×7 |
| Fine-pool | 1 | 4 | (4,4) | 40 | 64 | 3×3 |

All configurations are 20–330× fewer params than Linear(208,64) = 13,376. Even the largest (1,248) is well within the parameter budget for S3 (46 trials → ~370 effective samples).

**What IS derivable from first principles** (and therefore fixed):
- Per-patient Conv2d (not shared) — arrays differ in orientation
- Conv2d (not Linear) — weight sharing matches rigid-array uniformity; channels don't correspond across patients
- AdaptiveAvgPool (not flatten) — handles two grid sizes; provides coarse positional encoding
- Coarse pooling preferred — 15–25mm placement offsets

**What is NOT derivable** (and therefore tuned):
- Number of layers, channels, exact pool dimensions

| | Phase 2: Linear read-in | Phase 3: Spatial conv (default) |
|---|---|---|
| Per-patient params | 13,376 | **80** (167× fewer) |
| Spatial structure | Ignored (flattened vector) | Exploited (2D conv on grid) |
| Grid size handling | Zero-pad to max channels | AdaptiveAvgPool (native) |
| Inductive bias | None | Weight sharing matches rigid-array physics |
| Volume conduction | Must decorrelate from data | Learned Laplacian/gradient filters |
| Orientation adaptation | Arbitrary per-electrode weights | Per-patient oriented spatial filters |
| Stage 2 trainable | 14,794 | **2,272** (6.5× fewer) |

Both 8×16 and 12×22 grids pool to (2,4) transparently. Non-significant channels are zeroed in the grid before conv. Pool resolution and layer count are swept in E13.

**Framing note:** The spatial conv is a principled engineering choice matching architecture to recording physics, not a radical methodological innovation. The articulatory CTC head (Innovation 2) is the primary novel contribution.

#### Innovation 2: Articulatory CTC Head

Replaces Phase 2's flat Linear(128,10). Six parallel heads classify articulatory features. A fixed composition matrix A maps feature scores to phoneme logits. The model learns what motor cortex encodes (articulatory gestures), not what linguistics defines (phoneme categories). Each sub-head is binary or 3-way — simpler than the full 9-way problem.

| Phoneme | C/V | Place | Manner | Voicing | Height | Backness |
|---|---|---|---|---|---|---|
| /b/ | C | bilabial | stop | voiced | — | — |
| /p/ | C | bilabial | stop | voiceless | — | — |
| /v/ | C | labiodental | fricative | voiced | — | — |
| /g/ | C | velar | stop | voiced | — | — |
| /k/ | C | velar | stop | voiceless | — | — |
| /a/ | V | — | — | — | low | central |
| /æ/ | V | — | — | — | mid | front |
| /i/ | V | — | — | — | high | front |
| /u/ | V | — | — | — | high | back |

```python
class ArticulatoryCTCHead(nn.Module):
    def __init__(self, input_dim=128):  # 2H = 128
        super().__init__()
        self.cv_head      = nn.Linear(input_dim, 2)  # consonant, vowel
        self.place_head   = nn.Linear(input_dim, 3)  # bilabial, labiodental, velar
        self.manner_head  = nn.Linear(input_dim, 2)  # stop, fricative
        self.voicing_head = nn.Linear(input_dim, 2)  # voiced, voiceless
        self.height_head  = nn.Linear(input_dim, 3)  # low, mid, high
        self.back_head    = nn.Linear(input_dim, 3)  # front, central, back
        self.blank_head   = nn.Linear(input_dim, 1)  # CTC blank

        # Blank logit init: phoneme logits are SUMS of 3-4 sub-head outputs,
        # giving them ~√3-√4 × larger std than blank at Kaiming init.
        # Without correction, blank starts suppressed → CTC emits too many
        # phonemes initially → slow convergence or NaN from forward-backward.
        # +2.0 bias makes blank competitive with composed phoneme logits at init.
        nn.init.constant_(self.blank_head.bias, 2.0)

        # Fixed composition: A[i,j] = 1 iff phoneme i has articulatory feature j
        # Cols: [C, V, bil, lab, vel, stp, fri, vcd, vcl, low, mid, hi, frt, cen, bck]
        self.register_buffer('A', torch.tensor([
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # /b/
            [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # /p/
            [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # /v/
            [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # /g/
            [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # /k/
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # /a/
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # /æ/
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # /i/
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],  # /u/
        ], dtype=torch.float))

    def forward(self, h):
        # h: (B, T, 2H)
        features = torch.cat([
            self.cv_head(h), self.place_head(h), self.manner_head(h),
            self.voicing_head(h), self.height_head(h), self.back_head(h),
        ], dim=-1)                        # (B, T, 15)
        phoneme_logits = features @ self.A.T  # (B, T, 9) — fixed composition
        blank_logit = self.blank_head(h)      # (B, T, 1)
        return F.log_softmax(torch.cat([blank_logit, phoneme_logits], dim=-1), dim=-1)
```

**Key properties:**
- Phoneme logits are *composed* from articulatory scores via fixed A — 0 learned params in composition
- /b/ vs /p/ difference = voicing score alone (model MUST learn voicing to distinguish them)
- Inapplicable features are zeroed by A (e.g., height doesn't affect consonants), forcing the backbone to route consonant vs vowel information to appropriate heads
- Blank logit initialized with +2.0 bias to compensate for scale mismatch with composed phoneme logits (sums of 3-4 terms)
- **Known asymmetry:** consonant logits sum 4 features, vowel logits sum 3 → ~15% consonant magnitude bias at init. Self-correcting during training; monitor per-class accuracy to verify
- Total: 128×15 + 15 + 128×1 + 1 = **2,064 params** (vs 1,290 for flat head)

#### Updated Parameter Budget

| Component | Phase 2 (E1) | Phase 3 (E2) |
|---|---|---|
| Per-patient read-in | 13,376 (Linear) | **80** (1-layer Conv2d default; 664 if 2-layer) |
| LayerNorm(64) | 128 | 128 |
| Conv1d(64, 64, k=5) | 20,544 | 20,544 |
| BiGRU(64, 64, 2 layers, bidir) | 124,416 | 124,416 |
| CTC head | 1,290 (flat) | 2,064 (articulatory) |
| **Total shared** | **~146k** | **~147k** |
| **Per-patient** | **13,376** | **80** (default; 664 if 2-layer) |
| **Stage 2 trainable** | **14,794** | **2,272** (default; 2,856 if 2-layer) |

### 3.3 Training

Phase 3 uses **the same training procedure as Phase 2** (§2.2). Standard multi-patient SGD with uniform loss weighting, CosineAnnealingLR, early stopping with best-checkpoint selection — the only changes are architectural (spatial conv replaces linear read-in, articulatory head replaces flat head).

**Why no training innovations?** Per-patient input layers are the primary cross-patient transfer mechanism. The shared backbone is trained jointly with 7 different per-patient read-ins — it's already implicitly "transfer-ready" (Singh achieves good transfer at N=25 with this approach, no meta-learning or contrastive loss). Reptile, SupCon, and DRO were evaluated and downgraded to exploratory — see `future_directions.md` for rationale.

#### Stage 2

Same structure as Phase 2's Stage 2, with two differences:

1. **Far fewer trainable params** — 2,272 vs 14,794 (6.5× reduction) at default config, reducing overfitting risk for small patients (S3: 46 trials)
2. **Articulatory head** — decomposes 9-way classification into articulatory feature sub-problems

| Setting | Phase 2 (E1) | Phase 3 (E2) |
|---|---|---|
| Read-in init | Kaiming random | Kaiming random |
| Trainable params | 14,794 | **2,272** (default) |
| Freeze | Conv1d + BiGRU | Conv1d + BiGRU |
| Unfreeze | Linear + LN + flat head | SpatialConv + LN + articulatory head |
| Early stopping | Same: patience 10 on target val loss |
| Other | Same: 5-fold, 100 epochs max, lr=1e-3, 30% source replay |

### 3.4 Sanity Checks (Innovation-Specific)

Run after Phase 2 infrastructure is validated:

1. **Spatial conv shape:** Forward pass through SpatialConvReadIn with both grid sizes (8×16, 12×22) → backbone → articulatory head → CTC loss → gradient to all params
2. **Composition matrix:** For each phoneme, verify `A @ feature_vector` produces correct logit pattern

### 3.5 Ablations (E3–E4)

Each removes one innovation from E2. Same LOPO, splits, seeds. Config-driven — one flag per ablation. Since both innovations are architectural with no training changes, ablations differ only in model construction.

| Experiment | Change from E2 | Tests | Expected signal |
|---|---|---|---|
| **E3: −SpatialConv** | Linear(208,64) replaces Conv2d | Does spatial factorization help beyond parameter reduction? | Stage 2 adapts 14,794 params → overfitting risk for S3. E3a (bottleneck Linear at matched params) isolates spatial bias from param count |
| **E4: −ArtDecomp** | Flat Linear(128,10) replaces articulatory head | Does articulatory decomposition help? | Improvements should concentrate on confusable pairs (/b/↔/p/, /g/↔/k/) |

### 3.6 Extended Ablations (if time permits)

| Experiment | Change from E2 | Tests |
|---|---|---|
| E5: +DRO | DRO-weighted loss (upweight hard patients) replaces uniform (see `future_directions.md`) | Does worst-case optimization help with ~8 heterogeneous patients? |
| E6: +Reptile | Replace standard SGD with Reptile meta-learning (see `future_directions.md`) | Does meta-learning help when per-patient layers already handle transfer? |
| E7: +SupCon | Add periodic cross-patient supervised contrastive loss (see `future_directions.md`) | Does explicit alignment help beyond CTC + per-patient layers? |
| E3a: Bottleneck Linear | Linear(D_padded→8→64) at ~2,200 params replaces Conv2d | Is the benefit from spatial bias or just parameter reduction? (If E2 ≈ E3a, it's param count; if E2 > E3a, spatial bias matters) |
| E8: Diagonal gain | Per-channel scalar gain (~200 params) replaces spatial conv | Dead by design (channels don't correspond across patients with 15–25mm placement offsets) — included to confirm |
| E9: +SWA | Weight-average 5 cosine cycle snapshots (Izmailov 2018) instead of best checkpoint | Does SWA improve generalization? (No BCI precedent) |
| E10: +DANN | Add gradient reversal patient discriminator | Adversarial DA value? |
| E11: +TENT | Entropy minimization on unlabeled target before Stage 2 | Free test-time adaptation? |
| E12: +Temporal dilation | Random ±20% temporal stretch (`F.interpolate`) during augmentation | Cross-patient neural rate variation? |
| E13: Spatial conv hyperparams | Sweep num_layers (1,2), C (4,8), pool (2×4, 3×4, 4×4) on 1 LOPO fold, 1 seed | Which spatial conv config is best? Separates first-principles decisions (Conv2d, per-patient, coarse pool) from empirical tuning (depth, width, pool resolution) |
| E14: Conv3d hybrid | 2 layers Conv3d before spatial collapse (replaces immediate flatten) | Does joint spatiotemporal processing in early layers help? |

**E13 details — Spatial conv hyperparameter sweep:**

Sweep the three empirical hyperparameters of `SpatialConvReadIn`. Each config is a 1 LOPO fold, 1 seed run. All configs use the same backbone, CTC head, and training procedure — only the read-in architecture changes. `D_shared` must be configurable (backbone Conv1d input dim must match output dim of read-in).

| Variant | Layers | C | Pool | Output dim | Params | Tests |
|---|---|---|---|---|---|---|
| E13a | **1** | **8** | **(2,4)** | 64 | 80 | **Default** — all physics-grounded, minimal params |
| E13b | 2 | 8 | (2,4) | 64 | 664 | Does a 2nd layer help given aggressive pooling? |
| E13c | 1 | 4 | (4,4) | 64 | 40 | Fewer filters + finer spatial resolution |
| E13d | 1 | 8 | (4,4) | 128 | 80 | More spatial detail (D_shared=128, backbone must adapt) |
| E13e | 1 | 8 | (3,4) | 96 | 80 | Compromise pool resolution |
| E13f | GAP | 1 | 32 | GAP | 32 | 296 | Spatial collapse to 1 value per filter — is positional info needed? |

**Key questions E13 answers:**
- **E13a vs E13b:** Does the 2nd conv layer help, or does pooling wash out its contribution?
- **E13a vs E13c:** More filters (8) with coarse pool vs fewer filters (4) with fine pool — which tradeoff is better?
- **E13a vs E13f:** Does positional info from the 2×4 pool matter, or does the backbone only need aggregate filter activations?
- **E13c vs E13d:** At finer pool resolution, do we need more channels to maintain output dim?

If E13a ≈ E13b, the 2nd layer doesn't help — stay at 1 layer. If E13f ≈ E13a, positional pooling is unnecessary — spatial filtering alone is what matters.

**E14 details — Conv3d hybrid:**

Preserve spatial structure through 2 shared Conv3d layers before collapsing to 1D for BiGRU. Tests whether local spatiotemporal features improve over immediate spatial flattening.

```
Per-patient Conv2d → AdaptiveAvgPool2d(4,8) → (8, 4, 8, T=200)
  → AvgPool temporal k=5 s=5 → (8, 4, 8, 40)
  → Conv3d(8, 16, k=3, pad=1) + GELU                              ~3.5k shared params
  → Conv3d(16, 32, k=(3,3,3), s=(2,4,1), pad=(1,1,1)) + GELU     ~13.9k shared params
  → (32, 2, 2, 40) → flatten spatial → (128, 40)
  → BiGRU(128, 64, 2 layers) → CTC head
```

Adds ~17k shared params for 2 Conv3d layers. Spatial goes 4×8 → 2×2 (still meaningful local patterns in layer 1). After 2 layers, spatial is 2×2 — effectively global — so we collapse and hand off to BiGRU. Run on 1 LOPO fold, 1 seed.

### 3.7 Diagnostics

#### RSA (Representational Similarity Analysis)

Tests whether the backbone learns shared phoneme geometry across patients.

```python
def compute_rsa(model, spatial_convs, data, patients):
    rdms = {}
    for p in patients:
        centroids = {}
        for phoneme in phonemes:
            h = model.backbone(spatial_convs[p](data[p][phoneme]))
            centroids[phoneme] = h.mean(dim=(0, 1))
        rdm = torch.zeros(9, 9)
        for i, j in combinations(range(9), 2):
            rdm[i, j] = rdm[j, i] = 1 - F.cosine_similarity(
                centroids[phonemes[i]], centroids[phonemes[j]], dim=0)
        rdms[p] = rdm

    rsa_scores = [spearmanr(rdms[p1].flatten(), rdms[p2].flatten()).correlation
                  for p1, p2 in combinations(patients, 2)]
    return mean(rsa_scores)
```

| RSA score | Interpretation |
|---|---|
| > 0.7 | Shared phoneme geometry — cross-patient transfer working |
| 0.3–0.7 | Partially shared — per-patient adaptation critical |
| < 0.3 | Different coding strategies — benefit is regularization, not transfer |

**Secondary RSA:** Compare backbone RDM to articulatory feature RDM (Hamming distance between feature vectors). Correlation → backbone learned biologically meaningful representations.

#### Prototypical Accuracy

Classify test trials by nearest class prototype (mean embedding from training fold). No per-patient parameters — pure backbone metric.

```python
def prototypical_accuracy(model, spatial_conv, train_data, test_data):
    prototypes = {}
    for phoneme in phonemes:
        h = model.backbone(spatial_conv(train_data[phoneme]))
        prototypes[phoneme] = h.mean(dim=(0, 1))

    correct = 0
    for trial, label in test_data:
        h = model.backbone(spatial_conv(trial)).mean(dim=(0, 1))
        pred = min(phonemes, key=lambda p: torch.dist(h, prototypes[p]))
        correct += (pred == label)
    return correct / len(test_data)
```

If prototypical ≈ CTC accuracy → backbone produces a clean metric space. If prototypical << CTC → temporal dynamics matter.

#### Additional Diagnostics

1. Confusion matrix colored by articulatory distance
2. Per-feature-head accuracy (which articulatory features are hardest?)
3. **Consonant vs vowel accuracy** — monitor for asymmetry from composition (consonants sum 4 features, vowels sum 3). If consonant accuracy systematically exceeds vowel accuracy, consider normalizing A rows
4. Spatial conv filter visualization (do learned 3×3 filters show spatial structure?)
5. **Spatial conv filter norm** (Phase 3 Stage 2) — if filter weights collapse toward zero, Stage 2 weight decay (1e-3) is too aggressive; reduce to 1e-4
6. Stage 1 training dynamics (per-patient loss curves, **training vs held-out validation loss** to confirm no overfitting)

### 3.8 Deliverables

- [ ] E1a quick validation (1 LOPO fold, 1 seed): Conv2d does not degrade PER vs Linear
- [ ] E2 full model validated via progressive addition (E1 → E1a → E2)
- [ ] Full experiment matrix (E1–E4) × 8 folds × 3 seeds completed
- [ ] Per-patient results table with attribution (E2−E3, E2−E4)
- [ ] Diagnostics: RSA, confusion matrix, per-feature-head accuracy, filter visualization
- [ ] Wilcoxon tests: E2 vs E0, E2 vs E1

---

## Risk Register

| Risk | Prob | Impact | Mitigation | Detection |
|---|---|---|---|---|
| Phase 2 doesn't beat Spalding | Low | High | Debug data loading, splits, hyperparams | Per-patient results |
| Negative transfer (pooled hurts some patients) | Med | High | Compare per-patient PER vs 0.24 (patient-specific) | Results table |
| CTC length collapse | Low | Med | Targets are 3-phoneme non-words (e.g., [a,b,e]) — repeated consecutive labels are rare. Monitor blank ratio + length accuracy | CTC length accuracy <50%, blank ratio >90% |
| CTC mode collapse (class) | Med | Med | Articulatory heads harder to collapse | Per-class prediction rates |
| Spatial conv overfits S3 | Low | Med | 80–664 params (config-dependent); early stopping; 30% source replay in Stage 2 | S3 Stage 2 loss curve |
| Non-significant results (N=8) | High | Med | Pre-specify primary comparisons (E2 vs E0, E2 vs E1). Report effect sizes + per-patient results regardless | Wilcoxon p-values |

---

## Timeline

| Phase | Duration | Dependencies |
|---|---|---|
| Phase 0: Setup + data inspection | 1–2 weeks | Data download from BOX |
| ~~Phase 1: Reproduce baseline~~ | ~~1 week~~ | **SKIPPED** — use Spalding's 0.31 as E0 |
| **Phase 2 (field standard)** | **2–3 weeks** | Phase 0 done |
| ↳ Milestone: E1 validated, beats Spalding | | |
| Phase 3 E2 (full model) | 1 week | Phase 2 validated |
| Phase 3 E3–E4 (ablations) | 1 week | E2 converged; run in parallel |
| E5–E14 (extended) | 1–2 weeks | If time permits |
| Analysis + figures | 1–2 weeks | All experiments done |

**Critical path:** Data download → Phase 0 (inspect data, resolve questions, write infrastructure) → **Phase 2 (checkpoint)** → E2 → ablations → analysis.

**Overlap:** Model/training code during Phase 0. Phase 3 architecture code during Phase 2. Ablations parallel on Duke cluster GPUs. Develop locally first, deploy to cluster for full runs.

**Compute:** Each experiment requires 8 LOPO folds × 3 seeds. Per fold per seed: ~30 min Stage 1 + 5 Stage 2 CV folds × ~5 min = ~55 min. Four experiments (E1–E4) each have distinct (read-in, CTC head) combinations → no Stage 1 sharing. Total: 4 experiments × 8 folds × 3 seeds × ~55 min ≈ **~88 GPU-hours**.

---

*Future directions (cross-task pooling, Tier 2 ideas) are in [`future_directions.md`](future_directions.md).*
