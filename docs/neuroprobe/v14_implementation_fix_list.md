# v14 Implementation Fix List

> **SUPERSEDED 2026-06-03 by `docs/neuroprobe/b36_implementation_plan.md`.** This ledger was written against the pre-B36 design (B31 inert SSL, B29 joint phase, Perceiver soft routing). B36 (2026-06-01) retires that design — hard per-parcel pool, paradigm-B masked JEPA, staged P1/P2. Use the B36 plan as the active implementation doc; this file is kept for the closed-item audit trail only.

**Generated**: 2026-05-26 from a 4-agent cross-reference audit (canonical memos × `docs/neuroprobe/v14_blockers.md` × `docs/neuroprobe/training_recipe.md` × `src/speech_decoding/`).
**Purpose**: actionable cross-product of design-vs-code gaps. Treat as the **B18 closure ledger** — when every item here is closed (or explicitly accepted as deferred), B18 is closed.

## Trust order when items conflict

`memory/project_v14_*.md` (later memo wins per lock chain) → `training_recipe.md` → `v14_blockers.md` → `docs/neuroprobe/plan.md` → code.

## Status legend

- **BLOCKER** — Phase-1 dispatch trains the wrong architecture without this fix
- **SCAFFOLD** — pre-dispatch infrastructure; no spec ambiguity but missing
- **DOC** — doc-only edit; no code risk
- **DECISION** — memo defers to runtime; default must be pinned before code lands
- **TEST** — test pins a stale default and must update with its target

## Index

- §A. Code blockers (28 items)
  - A.1 Front-end (B20, 4 items)
  - A.2 Latent stack & collapse prevention (B21+B22, 5 items)
  - A.3 Mask discipline (B03, 7 items)
  - A.4 Loss objective (B19+B21+B22, 6 items)
  - A.5 Preproc (1 item)
  - A.6 Phase-3 distillation (B05+B06, 4 items)
  - A.7 EMA schedules (B01+B11, 1 item)
- §B. Scaffolding (9 items)
- §C. Doc fixes (9 items)
- §D. Pending decisions (4 items)
- §E. Tests that pin stale defaults (5 items)
- §F. Dependency graph
- §G. Recommended execution order

---

## §A. Code blockers for Phase-1 dispatch

### A.1 Front-end — B20 v4-invisible-frontend lock (2026-05-24)

#### FE-01: hop_length 128 → 256
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/extractors/view.py:385` (`MultiStftView.hop_length`)
- **Change**: `hop_length=128` → `hop_length=256`. At 2048 Hz sample rate this changes frame rate from 16 Hz to 8 Hz, matching Whisper teacher-pool target.
- **Knock-on**:
  - `experiments/dispatch_v14.py:59` `DEFAULT_N_TIME_BINS = 17` recompute for 1 s clip at 8 Hz
  - All downstream T_p math in encoder
- **Tests affected**: `extractors/test_view.py` (frame-rate assertions); regenerate any baked smoke fixtures.
- **Citation**: `project_v14_v4_invisible_frontend_lock_2026_05_24.md`; recipe:135; blockers:33
- **Dep**: none (pure default change)

#### FE-02: Conv2d (3,2) patch stem
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/models/v14_encoder.py` — new module `_PatchStem`
- **Change**: `Conv2d(in_channels=1, out_channels=d=256, kernel_size=(3, 2), stride=(3, 2))` non-overlapping. Replaces the per-bin Linear stem.
- **Output shape**: `(C electrodes, F_p=10, T_p=T/2, d=256)` per electrode
- **Init**: trunc-normal std=0.02
- **Tests affected**: `models/test_v14_encoder.py` (shape assertions)
- **Citation**: `project_v14_v4_invisible_frontend_lock_2026_05_24.md`; recipe:42-45
- **Dep**: FE-01 (hop must be 256 so T_p math is correct)

#### FE-03: Per-patch freq embed F_p=10 vectors
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/models/v14_encoder.py:417-419`
- **Change**: replace `freq_embed = nn.Parameter((F=30, d))` with `freq_embed = nn.Parameter((F_p=10, d))`. Additive broadcast onto patches (same vector for all electrodes/clips at a given F-patch index).
- **Init**: trunc-normal std=0.02
- **Citation**: `project_v14_v4_invisible_frontend_lock_2026_05_24.md`; recipe:44
- **Dep**: FE-02 (patches must exist first)

#### FE-04: JOINT token-block attention (N=6, distinct from factorized latent stack)
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/models/v14_encoder.py` — new token-block stack `_TokenBlocks`
- **Change**: N=6 layers of JOINT (t·f) self-attention over patches per electrode, with RoPE on time axis only (no time PE at input), pre-norm, GeLU, MLP 4×, heads=8, hard −∞ cross-electrode mask.
- **Important**: latent stack ❻ STAYS factorized (TimeSformer-divided time-then-parcel). The distinction is:
  - ❷ token block = JOINT (B20 v4)
  - ❻ latent stack = FACTORIZED (B20 retains this)
- Current code (`models/v14_encoder.py:264-312` `_LatentSelfAttnBlock`) is factorized — correct for latent stack but **the token-block ❷ stem itself is absent in code**.
- **Tests affected**: `models/test_v14_encoder.py`
- **Citation**: `project_v14_v4_invisible_frontend_lock_2026_05_24.md`; recipe:47
- **Dep**: FE-02, FE-03

### A.2 Latent stack & collapse prevention — B21+B22 lock (2026-05-25)

#### LAT-01: Identity-anchored latent init
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/models/v14_encoder.py:420-422`
- **Change**: replace single `parcel_embedding = nn.Parameter((K=80, M=4, d=256))` trunc-normal with sum:
  - `LearnableParcelEmbed: nn.Parameter((80, d=256))` trunc-normal std=0.02
  - `LearnableSubSlotEmbed: nn.Parameter((4, d=256))` trunc-normal std=0.02
  - `ε ~ N(0, 0.02²)` per-forward construction noise
  - `z[p·M + s] = LearnableParcelEmbed[p] + LearnableSubSlotEmbed[s] + ε`
- **Param delta**: ~+22k vs single tensor
- **Citation**: `project_v14_collapse_prevention_lock_2026_05_25.md`; recipe:15-26
- **Dep**: none (independent of front-end fixes)

#### LAT-02: LN_mid (post cross-attn-1, M3 head)
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/models/v14_encoder.py`
- **Change**: dedicated `LayerNorm(d=256)` applied to M3 stream (post cross-attn-1 / pre self-attn-0). Used by B22's L_mid_slot loss head, NOT inserted in forward path of M4 stream.
- **Param delta**: ~+512
- **Citation**: `project_v14_b22_collapse_prevention_dense_features_2026_05_25.md`; recipe:60

#### LAT-03: LN_frame (M4 head, L_post_frame)
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/models/v14_encoder.py`
- **Change**: dedicated `LayerNorm(d=256)` at M4 divergence for L_post_frame head. Mirror on EMA teacher (LN_frame_T).
- **Citation**: `project_v14_collapse_prevention_lock_2026_05_25.md`; recipe:77

#### LAT-04: LN_utt (M4 head, L_post_utterance)
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/models/v14_encoder.py`
- **Change**: dedicated `LayerNorm(d=256)` at M4 divergence for L_post_utterance head (pre-PMA). Mirror on EMA teacher (LN_utt_T).
- **Citation**: `project_v14_collapse_prevention_lock_2026_05_25.md`; recipe:78

#### LAT-05: Expose M3 tap (post cross-attn-1 / pre self-attn-0)
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/models/v14_encoder.py`
- **Change**: encoder forward returns intermediate `(M2, M3, M4)` tuple (or dict). M3 feeds L_mid_slot; M4 feeds L_post_frame / L_post_utterance / L_DKoleo.
- **Citation**: `project_v14_b22_collapse_prevention_dense_features_2026_05_25.md`; recipe:60-63
- **Dep**: LAT-02 (LN_mid must exist)

### A.3 Mask discipline — B03 lock (2026-05-25 PM)

#### MASK-01: Per-corpus mains notch
- **Severity**: BLOCKER (SWEC currently fed 60 Hz silently)
- **File**: `src/speech_decoding/experiments/dispatch_v14.py:123` (`notch_filter=60.0` hardcoded)
- **Change**:
  - Lift `notch_filter` to per-corpus dispatch field
  - SWEC manifest → 50.0; BT / D-cohort / AJILE12 → 60.0
- **Tests affected** (see §E): `extractors/test_view.py:36,165,210,362`, `experiments/test_v14_dispatch_wired.py:29,75,163`
- **Citation**: MEMORY.md "Mains-notch is per-corpus"; M28 in `v14_blockers.md`
- **Dep**: none

#### MASK-02: `parcels_supervised[subject]` extractor
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/extractors/parcel.py` (extend) or new `extractors/parcels_supervised.py`
- **Change**: per-subject DK parcel set with ≥1 electrode, computed once at extractor build time (NOT per-clip post-mask). Type: `dict[subject_id, set[parcel_id]]`.
- **SWEC fallback**: `parcels_supervised[swec_subj] = ∅` → loss sites supervise all 320 slots (anatomy-blind).
- **Citation**: `project_v14_b03_mask_lock_2026_05_25.md` (B03f); recipe:102, 221, 347

#### MASK-03: Shaft-mask via key_padding_mask DROP
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/models/v14_encoder.py:188-191` (cross-attn key_padding_mask path)
- **Change**:
  - `cross_attn.key_padding_mask = pad_mask | shaft_mask` at ❺a and ❺b
  - `shaft_mask` default `torch.zeros_like(pad_mask)` outside P2
  - Shaft K=3 mixed-extent blocks, ~40% effective rate (Brain-JEPA pattern)
- **Param delta**: 0 (no [MASK] token; pure drop)
- **Speedup**: ~40% cross-attn cost reduction on P2 student
- **Citation**: `project_v14_b03_mask_lock_2026_05_25.md` (B03 core); recipe:346

#### MASK-04: Predictor2Block (paradigm B)
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/models/v14_encoder.py` — new module
- **Change**: `Predictor2Block(d=d, hidden=128, heads=4, depth=2)`, ~0.2M params. Operates on per-electrode patch mask. Trained P1 + P2, **discarded at P2→P3 boundary**.
- **Citation**: `project_v14_b03_mask_lock_2026_05_25.md` (B03c); B08 in `v14_blockers.md`
- **Dep**: FE-02 (patches must exist)

#### MASK-05: Predictor warm-start P1→P2
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/experiments/v14_phase2.py` (when written)
- **Change**: at P2 step 0, `predictor.load_state_dict(p1_checkpoint['predictor'])`. P1 checkpoint must include `predictor.state_dict()`.
- **Citation**: `project_v14_b03_mask_lock_2026_05_25.md` (B03c PM revert 2026-05-25)
- **Dep**: MASK-04, SCAFFOLD-01 (Experiment class)

#### MASK-06: Asymmetric EMA teacher (full electrodes + patches in P2)
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/ssl/ema.py`; `src/speech_decoding/models/v14_encoder.py` forward path
- **Change**:
  - `teacher_forward(electrode_mask=torch.zeros_like, patch_mask=torch.zeros_like)`
  - Student keeps both shaft and patch masks
  - This asymmetry IS the JEPA supervision signal (do NOT make it symmetric)
- **Unit test**: `teacher_forward_no_shaftmask.shape[-2] == C` every step
- **Monitor**: pre-register MON-MASK-002 (orphan/visible MSE ratio ∈ [0.7, 1.5]; >50 consecutive out-of-band steps escalates)
- **Citation**: `project_v14_b03_mask_lock_2026_05_25.md` (B03d)

#### MASK-07: Latent SA key_padding_mask via `parcels_supervised`
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/models/v14_encoder.py` (latent self-attn block)
- **Change**: `latent_self_attn.key_padding_mask = ~supervised_slot_mask` where `supervised_slot_mask` is the broadcast of `parcels_supervised[subject]` across M=4 sub-slots per parcel.
- **Verify**: `supervised_slot_mask.sum() == M × len(parcels_supervised[subject])` per BT subject.
- **Citation**: `project_v14_b03_mask_lock_2026_05_25.md` (B03b)
- **Dep**: MASK-02

#### MASK-08: bf16 attention-mask sentinel constant + unit test
- **Severity**: BLOCKER (test only; constant already in code)
- **File**: `src/speech_decoding/models/v14_encoder.py:82` (constant `NEG_INF_MASK_VALUE = -1e4` already exists) + new test
- **Change**: ensure all `masked_fill`/`masked_fill_` sites use `MASK_NEG = -1e4`. Add `tests/test_bf16_mask_floor.py`: all-invalid bias → `softmax(scores) < 1e-6` across bf16/fp16/fp32.
- **Sites**: cross-attn ❺a/❺b key_padding_mask (B03), latent SA key_padding_mask (B03b), PMA softmax (B03f), freq-SA invalid-bin (EX09).
- **Citation**: B12 in `v14_blockers.md`

### A.4 Loss objective — B19 + B21 + B22 (5-term P1/P2)

#### LOSS-01: 5-term `L_total` wired in v14_phase{1,2}.py
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/experiments/v14_phase1.py`, `v14_phase2.py` (NEW)
- **Change**: `L_total = L_pre_frame@M2 + L_mid_slot@LN_mid(M3) + L_post_frame@LN_frame(M4) + 1.0·L_post_utterance@LN_utt(M4)-PMA + 0.1·L_DKoleo@M4`
- **Coefficients**: fixed `(1, 1, 1, 1.0, 0.1)`. Joint from step 1, no curriculum, no schedule.
- **Reactive arms (off-default)**:
  - `+ 0.05·L_DKoleo@M3` if M3 trigger fires
  - `+ 0.1·L_Gram` if M4 trigger fires
- **Citation**: `project_v14_loss_design_lock_2026_05_24.md` (B19) + `project_v14_b22_collapse_prevention_dense_features_2026_05_25.md` + `project_v14_b03_mask_lock_2026_05_25.md` (B03f)
- **Dep**: LAT-02 .. LAT-05, MASK-02

#### LOSS-02: `L_mid_slot @ LN_mid(M3)`
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/experiments/v14_phase{1,2}.py`
- **Change**:
  - Target: EMA teacher M3, all-L=6 layer-averaged + per-layer instance-norm
  - Student / teacher both pass through `LN_mid` / `LN_mid_T`
  - Gate: `parcels_supervised[subject]` (B03f) — per-subject set, NOT per-clip
  - Axis: `(parcel ∈ supervised, time-patch, d)`
  - Loss: MSE; divisor `|parcels_supervised| × T_p × d`
  - Weight: 1.0; no predictor at M3 (cross-attn-1 IS the predictor)
- **Citation**: `project_v14_b22_collapse_prevention_dense_features_2026_05_25.md`; recipe:212-216
- **Dep**: LAT-02, LAT-05, MASK-02

#### LOSS-03: `L_post_frame @ LN_frame(M4)`
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/experiments/v14_phase{1,2}.py`
- **Change**:
  - Target: EMA teacher M4, all-L=6 layer-averaged + per-layer instance-norm
  - Both sides pass through `LN_frame` / `LN_frame_T`
  - Gate: `parcels_supervised[subject]`
  - Weight: 1.0
- **Citation**: `project_v14_collapse_prevention_lock_2026_05_25.md` (B21)
- **Dep**: LAT-03, MASK-02

#### LOSS-04: `L_post_utterance @ LN_utt(M4)-PMA`
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/experiments/v14_phase{1,2}.py`
- **Change**:
  - PMA k=1, learned seed query, cross-attends over 320 latents on parcel axis
  - Softmax mask = `parcels_supervised[subject]` (B03f; NOT per-clip valid_parcels)
  - Path: PMA(M4_student) → (T_p, d) → mean(T_p) → (d,) → MSE vs teacher PMA path
  - Both sides pre-norm via `LN_utt` / `LN_utt_T`
  - Weight: 1.0; no predictor
  - **PMA query training**: P1 + P2 (via L_post_utterance) + P3 (Smooth-L1) — frozen at P4 only (B07)
- **Citation**: `project_v14_collapse_prevention_lock_2026_05_25.md` (B21); recipe:78, 82, 110
- **Dep**: LAT-04, MASK-02

#### LOSS-05: `L_DKoleo @ M4` (slot-level uniformity)
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/ssl/koleo.py` (primitive at line 22-51 exists ✅) + wiring in `experiments/v14_phase{1,2}.py`
- **Change**:
  - Operand: **all 320 slots** (per B21 default; **NOT** restricted to `parcels_supervised`)
  - L2-norm per slot
  - Metric: `-(1/320) · Σ_i log(min_{j≠i} ||z_i - z_j||_2)`
  - Weight: 0.1 always-on in P1 + P2
- **Reactive cousin** (off-default): `L_DKoleo @ M3` weight 0.05 — arms only if M3 cosine-sim > 0.7 or parcel-ID F1 < 0.4
- **Citation**: `project_v14_collapse_prevention_lock_2026_05_25.md` (B21 B) + `project_v14_b22_collapse_prevention_dense_features_2026_05_25.md` (B22 reactive)
- **Dep**: LAT-05

#### LOSS-06: Per-layer instance-norm + K=6 averaging on EMA teacher target
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/ssl/ema.py`
- **Change**: `target = mean_over_K=6( instance_norm(layer_k) for k in range(6) )`. Data2vec-2.0 recipe. K=6 in P1 (token blocks) and P2 (latent-stack layers); symmetric.
- **Verify**: grep `top_k|last_layer_only` returns 0; teacher target uses all 6 layers.
- **Citation**: B11 in `v14_blockers.md`

### A.5 Preprocessing

(See MASK-01 above for per-corpus notch — listed under §A.3 because it's a mask-discipline blocker.)

### A.6 Phase-3 distillation — B05 + B06 (2026-05-25 PM)

#### P3-01: Whisper-side 2-layer MLP adapter (~393k params)
- **Severity**: BLOCKER for P3 only
- **File**: new module in `src/speech_decoding/models/` (NOT `models/mlp.py` — that file's `TinyMLPModel` is smoke-only with hidden=16)
- **Change**: `Linear(1280, 256) → GeLU → Linear(256, 256)`. LLaVA-1.5 shape. Operates Whisper-side: takes (B, 40, 1280) → (B, 40, 256).
- **Citation**: `project_v14_b03_mask_lock_2026_05_25.md` (B05 + B06 lock 5/25 PM); recipe:380-385

#### P3-02: Pin teacher rate to 8 Hz default
- **Severity**: BLOCKER for P3 only
- **File**: `src/speech_decoding/experiments/v14_phase3.py` (when written)
- **Change**: `teacher_rate_hz = 8` constant. Whisper teacher layer `teacher_layer = 8` (L8).
- **Note**: This contradicts `training_recipe.md:391` which still says "default candidate 10 Hz" — see DOC-FIX-07.
- **Citation**: MEMORY.md "Phase-3 readout (B05+B06 lock 5/25 PM): teacher rate 8 Hz"; B06 in blockers:137-143

#### P3-03: Teacher-side triangular pool 50→8 Hz
- **Severity**: BLOCKER for P3 only
- **File**: `src/speech_decoding/extractors/` (Whisper teacher feature extractor)
- **Change**:
  - Frozen Whisper-L8 hidden state in dataloader: `(250, 1280)` → `(40, 1280)`
  - Triangular FWHM = 250 ms; each bucket ~12.5 Whisper frames; linearly-decaying weights
  - Normalization: sum-to-1 per bucket
  - Edge: zero-pad start/end
  - **Teacher-side only** — student stays 8 Hz native (identity passthrough)
- **Verify**: `weight_matrix.sum(dim=-1).allclose(torch.ones(40))`
- **Citation**: B05 in `v14_blockers.md:128-136`; recipe:391-392 (note: recipe §5 needs P3-04 doc update too)

#### P3-04: Smooth-L1 β=1.0 default
- **Severity**: BLOCKER for P3 only
- **File**: `src/speech_decoding/ssl/distill.py:39-40`
- **Change**: pin `beta = 1.0` default (currently requires explicit argument; M04 blocker).
- **Citation**: B01 v3 lock; recipe:398

### A.7 EMA decay — phase-aware schedule

#### EMA-01: Phase-aware decay schedule
- **Severity**: BLOCKER
- **File**: `src/speech_decoding/ssl/ema.py:35-41`
- **Change**: replace single hardcoded `linear_ema_schedule(start=0.99, end=0.9999, total_steps=10_000)` with phase-aware:
  - **P1**: linear 0.99 → 0.9999 over 400k steps
  - **P2**: linear 0.999 → 0.9999 over 40k steps
  - **P3**: no EMA (teacher is external frozen Whisper)
- **Citation**: B01 in `v14_blockers.md:68`; recipe:258, 332

---

## §B. Scaffolding (pre-dispatch infrastructure)

### SCAFFOLD-01: V14Experiment class
- **Severity**: SCAFFOLD
- **File**: `src/speech_decoding/experiments/v14_experiment.py` (NEW); extend `experiments/experiment.py`
- **Change**: `V14Experiment(NeuralTrainExperiment)` with `phase: Literal[1, 2, "3a", "3b", 4]` parametrization. One class branches per phase.
- **Citation**: NT01 in `v14_blockers.md:382`

### SCAFFOLD-02: V14Data class (phase + corpus_mix + sampler_spec)
- **Severity**: SCAFFOLD
- **File**: `src/speech_decoding/experiments/data.py` (extend)
- **Change**: parametrize on `phase`, `corpus_mix` (per-phase corpus set), `sampler_spec` (α=0.5 hierarchical for P1, uniform-per-subject for P2, paired iEEG+audio for P3).
- **Citation**: NT02 in `v14_blockers.md:383`

### SCAFFOLD-03: SWECStudy scaffold
- **Severity**: SCAFFOLD
- **File**: `src/speech_decoding/studies/swec/study.py` (extend; scaffold exists with `0.5–120 Hz, 50 patients, DP03 gated`)
- **Change**: NeuralFetch Study contract; valid-bin range k0–k21; mains notch 50 Hz (CH); per-subject anatomy fallback `parcels_supervised = ∅`.
- **Citation**: DP03 in `v14_blockers.md:377`

### SCAFFOLD-04: DCohortStudy scaffold
- **Severity**: SCAFFOLD
- **File**: `src/speech_decoding/studies/cogan_dcohort/study.py` (extend; scaffold exists, `DP03 gated`. Native rate is **mixed, mostly 2048 Hz** — NOT a flat 2000 Hz; loader reads per-run rate + resamples to 2048. See `memory/project_d_cohort_data_inventory_2026_06_03.md`)
- **Change**: NeuralFetch Study contract; full 30-bin valid range; mains notch 60 Hz; 85 D-cohort subjects per audit 2026-05-23.
- **Citation**: DP03; `project_d_cohort_phase2_cohort_audit_2026_05_23.md`

### SCAFFOLD-05: AJILE12Study scaffold
- **Severity**: SCAFFOLD
- **File**: `src/speech_decoding/studies/ajile12/study.py` (NEW)
- **Change**: NeuralFetch Study contract; valid-bin range k0–k20 (500 Hz Nyquist); mains notch 60 Hz; ECoG-only (89.7% per Peterson 2022) → P1-only corpus.
- **Citation**: DP03; `project_v14_cross_subject_pretraining_data_strategy_2026_05_22.md`

### SCAFFOLD-06: Variable-T RoPE + variable-C collate
- **Severity**: SCAFFOLD
- **File**: `src/speech_decoding/extractors/` collate functions
- **Change**: phases differ in T_max and C_max (P1: 4 corpora, P2: 2 sEEG, P3: 1 paired). Collate must handle.
- **Citation**: DP01 in `v14_blockers.md:376`

### SCAFFOLD-07: pytest `must_pass_before_dispatch` marker + CI gate
- **Severity**: SCAFFOLD
- **File**: `pyproject.toml` (pytest config); `scripts/dcc/dispatch` shell hook
- **Change**:
  - Register `must_pass_before_dispatch` marker in pytest config
  - `scripts/dcc/dispatch` runs `pytest -m must_pass_before_dispatch` before sync; aborts on failure
- **Citation**: TST01 + TST10 in `v14_blockers.md:341, 350`

### SCAFFOLD-08: P1↔P2 checkpoint strict=True test
- **Severity**: SCAFFOLD
- **File**: `src/speech_decoding/experiments/test_phase_boundary_ckpt.py` (NEW)
- **Change**: load P1 checkpoint into P2 model with `strict=True`. Fail if any keys missing.
- **Reason**: `strict=False` silently random-inits missing keys; at phase boundary this masks stale-cascade bugs (e.g., predictor warm-start broken).
- **Citation**: TST03 + RT10 in `v14_blockers.md:343, 359`

### SCAFFOLD-09: bf16 NaN detector test
- **Severity**: SCAFFOLD
- **File**: `src/speech_decoding/ssl/test_recon_bf16_nan.py` (NEW)
- **Change**: P1 loss NaN detector under bf16 Multi-STFT + (no-)log + robust-z. Pin `log_eps ≥ 1e-6`.
- **Citation**: TST05 + RT01 in `v14_blockers.md:345`

---

## §C. Doc fixes (no code risk)

### DOC-FIX-01: plan.md params ~13M → ~15.1M
- **File**: `docs/neuroprobe/plan.md:24`
- **Change**: "~13M params" → "~15.1M params (post-B19/B21/B22 SSL stack + identity-anchored init + dense-feature supervision)"

### DOC-FIX-02: plan.md N=4 → N=6
- **File**: `docs/neuroprobe/plan.md:24`
- **Change**: "N=4 token blocks" → "N=6 token blocks (amended 2026-05-23)"

### DOC-FIX-03: plan.md hop 128 / 14.7 Hz → hop 256 / 8 Hz
- **File**: `docs/neuroprobe/plan.md:33`
- **Change**: "hop = 128 samples @ 2048 Hz → 14.7 Hz frame rate" → "hop = 256 samples @ 2048 Hz → 8 Hz frame rate (B20 v4 lock 2026-05-24)"

### DOC-FIX-04: plan.md P3 Goldstein-locked → empirical preflight
- **File**: `docs/neuroprobe/plan.md:49`
- **Change**: "Goldstein 2025 (L8, 10 Hz) defaults" → "Goldstein search-range anchors (empirical sEEG preflight picks k\*, r\*; **B06 PM lock 2026-05-25**: r\* = 8 Hz, k\* = L8 with `R-rate-{5,10,16}Hz` falsifiers)"
- **Citation**: `feedback_no_default_ecog_to_seeg_transfer_2026_05_24.md`; B06 in blockers

### DOC-FIX-05: plan.md 3-term → 5-term loss
- **File**: `docs/neuroprobe/plan.md:59`
- **Change**: "three unified bootstrap mask-prediction losses" → "five unified bootstrap mask-prediction losses (L_pre_frame + L_mid_slot + L_post_frame + L_post_utterance + 0.1·L_DKoleo per B19/B21/B22 locks)"

### DOC-FIX-06: v14_blockers.md line 499 frame-rate + value-axis
- **File**: `docs/neuroprobe/v14_blockers.md:499`
- **Change**: "Multi-STFT front-end (3 windows + 14.7 Hz hop + log ⅓-octave 30-bin filterbank)" → "Multi-STFT front-end (3 windows + 256-sample hop = 8 Hz frame rate per B20 v4 + ⅓-octave 30-bin filterbank, raw |STFT| value axis per 5/25 swap)"

### DOC-FIX-07: training_recipe.md §5 P3 rate 10 Hz → 8 Hz
- **File**: `docs/neuroprobe/training_recipe.md:383, 391-392`
- **Change**: replace "Default rate r* = 10 Hz (requires 8→10 Hz upsample)" / "(50, 1280) @ 10 Hz" with:
  - "Default rate r* = **8 Hz** (B06 PM lock 5/25; matches v14 student native 8 Hz; identity passthrough on student side; teacher-side triangular pool 50→8 Hz factor 6.25 FWHM 250 ms produces (40, 1280))"
- **Reconcile**: §3 / §4 elsewhere already cite 8 Hz; §5 is the only stale section.
- **Knock-on**: `R-rate-{5,10,16}Hz` sister cells remain as falsifiers; 10 Hz becomes a sister, not default.

### DOC-FIX-08: training_recipe.md §2 per-corpus notch explicit
- **File**: `docs/neuroprobe/training_recipe.md:125-126`
- **Change**: add explicit per-corpus call-out: "Mains notch: **60 Hz** for BT / D-cohort / AJILE12 (US sites); **50 Hz** for SWEC (CH site). Currently hardcoded to 60.0 in `dispatch_v14.py:123` — must lift to per-corpus dispatch field before SWEC pretrain (see implementation fix MASK-01)."

### DOC-FIX-09: Close B18 by adopting this fix list as the ledger
- **File**: `docs/neuroprobe/v14_blockers.md` (B18 entry around line 207-228)
- **Change**: mark B18 ✅ CLOSED 2026-05-26; reference `docs/neuroprobe/v14_implementation_fix_list.md`; treat this fix-list closure as the dispatch gate.

---

## §D. Pending decisions

### DECISION-01: Whisper layer k for BT sEEG
- **Status**: preflight sweep `{L4, L6, L8, L10, L12, L16, L20} × {5, 8, 10, 20} Hz` on one BT-Lite subject. Default search-range anchor: L8 (Antonello / Shimizu acoustic-phonetic). NOT Goldstein.
- **When**: before P3 dispatch
- **Citation**: B06; `feedback_no_default_ecog_to_seeg_transfer_2026_05_24.md`; `project_whisper_ceiling_prerun_test_2026_05_24.md`

### DECISION-02: ε anatomy-prior strength
- **Status**: default ε = 1e-2 (DK-first-pass lock 2026-05-13). Sister sweep `{1e-4, 1e-3, 1e-2, 1e-1}` runs after main P1.
- **When**: parallel with P1 dispatch (not blocking)
- **Citation**: `project_v14_dk_first_pass_2026_05_13.md`

### DECISION-03: HB03 / HB06 grad-checkpointing strategy for 32 GB Ada-5000 ceiling
- **Status**: M0 (MFU micro-profile) collapses 350→1,400 H100-h range to ±30%. Not run.
- **Default candidate**: stream from `/work/` with mid-job warm-cache OR grad-checkpoint every other latent-stack block (TorchTitan §4, MosaicML Composer).
- **When**: before P1 dispatch — highest-leverage de-risk
- **Citation**: HB03 / HB06 in `v14_blockers.md:195`; `project_v14_hb02_compute_estimate_2026_05_23.md`

### DECISION-04: M0 — Lite-cell rerun post-BTWordEvents fix
- **Status**: Lite cell 46970613 was invalidated by `project_btwordevents_split_class_imbalance_bug_2026_05_15.md`. Rerun needed.
- **When**: before any new sweep
- **Citation**: MEMORY.md Dispatch state; HB03

---

## §E. Tests that pin stale defaults

(Each will need update when its target fix lands.)

### TEST-01: `extractors/test_view.py` lines 36, 165, 210, 362 — `notch=60.0` hardcoded
- **Update**: parameterize per-corpus
- **Targets**: MASK-01

### TEST-02: `experiments/test_v14_dispatch_wired.py` lines 29, 75, 163 — `notch=60.0` hardcoded
- **Update**: parameterize per-corpus
- **Targets**: MASK-01

### TEST-03: `models/test_v14_encoder.py` — pins v3-era shapes (no Conv2d, F=30 not F_p=10)
- **Update**: rewrite for Conv2d (3,2) stem + F_p=10 patches + JOINT token-block
- **Targets**: FE-02, FE-03, FE-04

### TEST-04: `experiments/test_phase3_preflight.py` — ridge sweep retired by B06 5/25 PM
- **Update**: either retire the test or repurpose for the brain-fit ridge preflight (sister-cell falsification, not gate)
- **Targets**: M19 in `v14_blockers.md`; B06 PM revert

### TEST-05: New `test_bf16_mask_floor.py` (currently missing)
- **Add**: all-invalid bias → softmax < 1e-6 across bf16/fp16/fp32
- **Targets**: MASK-08

---

## §F. Dependency graph

```
Tier 0 — no dependencies, can land first:
  DOC-FIX-01..09        (pure doc edits)
  MASK-01               (per-corpus notch — independent of arch)
  LAT-01                (latent init — independent of front-end)
  EMA-01                (phase-aware decay)
  LOSS-06               (per-layer inst-norm + K=6 averaging in ssl/ema.py)
  MASK-08               (bf16 sentinel test)
  P3-04                 (Smooth-L1 β default)
  SCAFFOLD-07           (pytest marker)

Tier 1 — front-end, gates downstream model fixes:
  FE-01 → FE-02 → FE-03 → FE-04

Tier 2 — depends on Tier 1 (patches exist):
  LAT-02, LAT-03, LAT-04 (LayerNorms)
  LAT-05                 (M3 tap, depends on LAT-02)
  MASK-02                (parcels_supervised extractor — independent of FE)
  MASK-04                (Predictor2Block — depends on FE-02)

Tier 3 — depends on Tier 2:
  MASK-03                (shaft-mask DROP at cross-attn)
  MASK-05                (predictor warm-start — depends on MASK-04 + SCAFFOLD-01)
  MASK-06                (asymmetric teacher)
  MASK-07                (latent SA gate — depends on MASK-02 + LAT-01)
  LOSS-01..05            (5-term loss — depends on LAT-02..05 + MASK-02 + MASK-07)

Tier 4 — scaffolding (parallel to Tier 1-3):
  SCAFFOLD-01..06        (Experiment, Data, Studies, collate)
  SCAFFOLD-08, 09        (boundary tests, bf16 NaN)

Tier 5 — Phase-3 specific (gated by P1+P2 success):
  P3-01, P3-02, P3-03
  (P3-04 is Tier 0)
```

---

## §G. Recommended execution order

### Wave 1 — pure doc updates (~30 min, no code risk)
DOC-FIX-01 through DOC-FIX-09. Lands the spec-vs-doc drift. Includes closing B18.

### Wave 2 — independent code fixes (parallelizable)
- MASK-01 (per-corpus notch) — single dispatch field
- LAT-01 (identity-anchored init) — single module
- EMA-01 (phase-aware decay)
- LOSS-06 (per-layer inst-norm in EMA)
- P3-04 (Smooth-L1 β default)
- MASK-08 (bf16 sentinel test)
- SCAFFOLD-07 (pytest marker)

### Wave 3 — front-end rewrite (gates everything else)
FE-01 → FE-02 → FE-03 → FE-04. Rewrite token-block stem; update tests TEST-03.

### Wave 4 — latent stack collapse-prevention
LAT-02 → LAT-03 → LAT-04 → LAT-05. Add three LayerNorms and M3 tap.

### Wave 5 — mask discipline
MASK-02 (parcels_supervised extractor) → MASK-03 → MASK-04 → MASK-05 → MASK-06 → MASK-07.

### Wave 6 — loss objective
LOSS-01 (5-term sum) → LOSS-02 → LOSS-03 → LOSS-04 → LOSS-05.

### Wave 7 — scaffolding for dispatch
SCAFFOLD-01 → SCAFFOLD-02 → SCAFFOLD-03 → SCAFFOLD-04 → SCAFFOLD-05 → SCAFFOLD-06 → SCAFFOLD-08 → SCAFFOLD-09.

### Wave 8 — Phase-3 readout
P3-01 → P3-02 → P3-03. (P3-04 in Wave 2.)

### Wave 9 — pre-dispatch de-risk (NOT code fixes; runtime decisions)
- DECISION-03 (HB03/HB06 MFU micro-profile, ~1 day Cogan-lab)
- DECISION-04 (M0 Lite-cell rerun)
- DECISION-01 (Whisper preflight — only blocks P3, not P1)
- DECISION-02 (ε sweep — parallel, not blocking)

---

## Verification checklist before P1 dispatch

A green dispatch means EVERY box below is checked:

- [ ] All Wave 1 doc edits landed (DOC-FIX-01..09)
- [ ] B18 marked closed in `v14_blockers.md`
- [ ] FE-01 hop=256: `grep "hop_length=128" src/speech_decoding/extractors/view.py` returns 0
- [ ] FE-02 Conv2d patcher: grep `Conv2d.*kernel.*=.*\(3,.*2\)` finds the patch stem
- [ ] FE-03 F_p=10: `freq_embed.shape[0] == 10`
- [ ] FE-04 JOINT token-block: token-block forward has single joint t·f attention call per layer
- [ ] LAT-01 identity-anchored init: grep `LearnableParcelEmbed` + `LearnableSubSlotEmbed` both present
- [ ] LAT-02..04 three LayerNorms: grep `LN_mid|LN_frame|LN_utt` ≥3 hits
- [ ] LAT-05 M3 tap: encoder forward returns `(M2, M3, M4)` or equivalent dict
- [ ] MASK-01 per-corpus notch: SWEC manifest has `notch_filter=50.0`; tests parameterized
- [ ] MASK-02 parcels_supervised: extractor exists; type is `dict[subject_id, set[parcel_id]]`
- [ ] MASK-03 shaft-mask: `cross_attn.key_padding_mask = pad_mask | shaft_mask` present
- [ ] MASK-04 Predictor2Block: module exists with hidden=128 heads=4 depth=2
- [ ] MASK-05 predictor warm-start: `predictor.load_state_dict` called at P2 step 0
- [ ] MASK-06 asymmetric teacher: teacher forward gets `electrode_mask=zeros, patch_mask=zeros`
- [ ] MASK-07 latent SA gate: `latent_self_attn.key_padding_mask = ~supervised_slot_mask`
- [ ] MASK-08 bf16 mask test: test_bf16_mask_floor.py passes
- [ ] LOSS-01..05 5-term loss: `grep "L_pre_frame + L_mid_slot + L_post_frame + .* L_post_utterance + .* L_DKoleo"` matches
- [ ] LOSS-06 per-layer inst-norm: `instance_norm` called per layer in EMA target
- [ ] EMA-01 phase-aware decay: separate P1/P2 schedules
- [ ] SCAFFOLD-01..06: Experiment, Data, Studies, collate scaffolded
- [ ] SCAFFOLD-07 pytest marker: `scripts/dcc/dispatch` aborts if `pytest -m must_pass_before_dispatch` fails
- [ ] SCAFFOLD-08 boundary ckpt test: P1→P2 strict=True passes
- [ ] SCAFFOLD-09 bf16 NaN: test passes
- [ ] DECISION-03 MFU profile run + grad-ckpt strategy chosen
- [ ] DECISION-04 M0 Lite-cell rerun green

---

## Provenance

This list is the cross-product of four parallel audits performed 2026-05-26:

- **Blockers doc audit**: `docs/neuroprobe/v14_blockers.md` (244 enumerated gaps, 16 BIG closed, 6 code-vs-spec gaps tracked)
- **Training recipe audit**: `docs/neuroprobe/training_recipe.md` (679 lines, internally consistent, all 5/25 PM amendments propagated)
- **Code audit**: `src/speech_decoding/` (~6,665 lines across 33 modules) — found 16 unimplemented locks (10 beyond the MEMORY.md "6-row gap")
- **Canonical memo extraction**: ~19 v14 memo files under `memory/` — produced the single source-of-truth spec used as the reference

Canonical memo trust order when conflict: later-dated memo wins per lock chain:
B20 (5/24) → B19 (5/24) → B21 (5/25) → B22 (5/25) → B03 (5/25 PM).

When this fix list closes, B18 (closure-gate meta-audit) closes by adopting this ledger as the verification record.

---

# Post-cascade Δ (5/27 → 5/28)

**Generated**: 2026-05-28 — captures only the deltas since the 5/26 baseline above. The 5/26 items (FE-01..04, LAT-01..05, LOSS-01..06, MASK-01..08, SCAFFOLD-01..09, P3-01..04) still apply except where superseded in §J below.

## Cascade chain

B25 (5/27 AM, Smooth-L1) → **B26** (5/27 PM, pure L1 + EMA fixed + teacher full-input) → **B27** (5/27 PM-late, DROP context loss) → **REF-aug** (5/27 PM) → **B28** (5/27 PM-late, DKoleo demote + 1 cross-attn + anatomy warmup + Perceiver/Graphormer citation cleanup) → **B29 Items 1–15** (5/27 PM-late → 5/28, joint phase + AJILE12 back + α=0.3 + subtype/ref_embed + drop gating + new monitors + M=1 default + MoE-FFN deferred + R-d-bump-384 P0) → **MoE-FFN audit** (5/28) → **HB02 re-cost** (5/28, envelope-safe) → **MON-SLOT-REDUNDANCY rescale** (5/28) → **Agent 2 subtype_embed precedent** (5/28, M3AE not MultiMAE; LaBraM/Brant-2 have no sensor-type embed; DIVER-1 §4.1 = net-neutral expected).

## §H. Code changes by file

Files marked **NEW** don't exist yet. Files marked **EDIT** exist; only the touched defaults / fields are listed.

### `src/speech_decoding/experiments/v14_joint.py` **NEW**

Replaces the never-built `experiments/v14_p1.py` + `experiments/v14_p2.py` pair. **Single phase**, joint from step 1 (B29 Item 1). 4-term loss (B27 + B28):

```
L = L_pre_frame_masked@M2  +  L_mid_slot@LN_mid(M3)
   + L_post_frame@LN_frame(M4)  +  1.0·L_post_utterance@LN_utt(M4)-PMA
```

- **Loss form**: pure **L1** across all 4 terms (B26 / B27). NOT Smooth-L1, NOT MSE. P3 distillation stays Smooth-L1.
- **NO context loss** (B27 reverts the B25 / B26 addition).
- **DKoleo not in default** (B28 Item 1). Dispatch field `dkoleo_mode: tp.Literal["off","intra_clip_slots","batch_cls_unit","vicreg_slot_variance"] = "off"`. The 3 sister modes activate on MON-SLOT-REDUNDANCY escalation.
- **No `parcels_supervised` gating** (B29 Item 12). L_mid_slot / L_post_frame / L_post_utterance-PMA aggregate over all 80 slots regardless of subject's parcel coverage. **Do NOT build `extractors/parcels_supervised.py`** — `R-parcels-supervised-gating` P0 sister is the only thing that needs it.
- Phase-1 / Phase-2 split sister: `R-keep-phase-split` P0 — keep the old 2-phase machinery behind a flag.

### `src/speech_decoding/ssl/ema.py` **EDIT**

- **EMA τ=0.999 fixed, no ramp** (B26 per V-JEPA 2 §2.4). Drop the V-JEPA-1 ramp schedule entirely.
- **Teacher full-input contract** (B26): `teacher.forward(electrode_mask=zeros_like(...), patch_mask=zeros_like(...))` every step, regardless of student's masking. Per-layer instance-norm + K-layer (=6) averaging at teacher target (LOSS-06 from 5/26 still applies).
- Sisters: `R-ema-tau` P1, `R-ema-ramp-v-jepa1` P1.

### `src/speech_decoding/models/v14_encoder.py` **EDIT**

1. **Cross-attn count: 2 @ {0, 3} → 1 @ {0}** (B28 Item 2). Dispatch field `n_cross_attns: int = 1` (already wired via `cross_attn_positions` kwarg). Sister `R-perceiver-original-2-cross-attns` P0 flips to 2. ~−0.79M params.
2. **M=1 default** (B29 Item 13):
   - Drop `LearnableSubSlotEmbed: (M=4, d)` entirely.
   - `z_init[p] = LearnableParcelEmbed[p] + ε` (was `z_init[p·M+s] = LearnableParcelEmbed[p] + LearnableSubSlotEmbed[s] + ε`).
   - `K_total = K * M` config: drop K_total derivation; use `K=80, M=1` so K_total=80 (was 320).
   - `sub_slot_of_latent` index map → identity (`parcel_of_latent[i] = i`).
   - LN_mid scope 320 → 80.
   - Dispatch field `M: int = 1`. Sister `R-m4-slots` P0 flips back.
3. **`subtype_embed: nn.Embedding(2, d=256)`** (B29 Item 11): additive at A1 patch-embed (broadcast over electrodes per clip). Reused as additive in cross-attn K/V (same broadcast). Dispatch field `subtype_embed_enabled: bool = True`. Sisters: `R-no-subtype-embed` P0; **`R-subtype-embed-input-only` P1 NEW** (add at A1 but NOT in cross-attn K/V — M3AE-faithful); **`R-subtype-embed-3way` P2-if-budget NEW** (3-way `{sEEG-depth, ECoG-grid, ECoG-strip}` matching DIVER-1).
4. **`ref_embed: nn.Embedding(3, d=256)`** (B29 Item 11): additive at A1 + cross-attn K/V (same as subtype_embed). Dispatch field `ref_embed_enabled: bool = True`. Sister `R-no-ref-embed` P1.
5. **Anatomy bias `λ_anat` schedule** (B28 Item 3 → SUPERSEDED by B29 per-clip gate):
   - Keep `lambda_anat` forward kwarg (already wired) — plumbing reused.
   - Replace `if bias_enabled: logits += bias` with `logits += λ_anat * bias` (already wired — `lambda_anat * torch.log(support + eps)`).
   - **Drive `λ_anat` from per-clip metadata**: 1.0 for anatomy-rich clips (BT / D / AJILE12), 0.0 for SWEC (when shaft-pseudo-parcel is off).
   - DROP the B28 step warmup schedule from default. Sister `R-with-anatomy-step-warmup` P1 reinstates the step schedule. Sister `R-anatomy-bias-step` P0 (instant on at P2 step 0, B19 behavior). Sister `R-anatomy-bias-on-from-p1` P0.
   - Sister `R-learned-bias-scaling` P1 — per-parcel scalar `α_p` (~80 params) multiplying the `log(support+ε)` bias.
   - Sister `R-eps-{1e-1, 1e-3}` P2 — ε ablation.
6. **Dense FFN preserved** (B29 Item 14). Do NOT touch the SA-block MLPs. Sister `R-moe-ffn-soft-4` P2-if-budget would need a new `models/soft_moe.py` (Puigcerver 2024 arXiv:2308.00951; ~2-day implementation) IF dispatched.
7. **d=256 stays default; d=384 is the must-run sister** (B29 Item 15). Verify `d_model` is a single dispatch field (not duplicated across modules) so `R-d-bump-384` P0 Cell-0 sister can override with one field. Sister `R-d-bump-512` P2-if-budget.

### `src/speech_decoding/experiments/dispatch_v14.py` **EDIT**

Lift mains-notch to per-corpus (5/26 MASK-01 still). Plus B29 fields:

```python
notch_filter_hz: dict[str, float]  # {"bt": 60.0, "d": 60.0, "ajile12": 60.0, "swec": 50.0}
ref_operator_alpha: float = 0.3     # B29 corpus sampler weight
corpus_mix: dict[str, float]        # rebuilt from per-corpus vb_eh totals; share table must sum to 1.0
M: int = 1                          # B29 Item 13
d_model: int = 256                  # ensure single source of truth
n_cross_attns: int = 1              # B28 Item 2
ffn_variant: tp.Literal["dense","soft_moe_4"] = "dense"
dkoleo_mode: tp.Literal["off","intra_clip_slots","batch_cls_unit","vicreg_slot_variance"] = "off"
anatomy_bias_mode: tp.Literal["per_clip_gate_b29","warmup_b28","step_b19","on_from_p1"] = "per_clip_gate_b29"
subtype_embed_enabled: bool = True
subtype_embed_vocab: tp.Literal["binary","three_way"] = "binary"
subtype_embed_reuse_kv: bool = True
ref_embed_enabled: bool = True
phase_mode: tp.Literal["joint_b29","split_p1_p2"] = "joint_b29"
include_ajile12: bool = True
```

**Doc-quality flag** (Agent 1, HB02 re-cost 5/28): B29 share-table headline (SWEC 35 / AJILE12 22 / D 18 / BT 12) sums to 87% not 100%. When you wire `corpus_mix`, rebuild against actual `vb_eh[corpus]` totals and assert `sum == 1.0` ± 1e-4 in the dispatch test. ~5 min CPU.

### `src/speech_decoding/extractors/view.py` **EDIT**

- Per-clip uniform-random reference draw (REF-01): `ref_operator ∈ {shaftCAR, bipolar, Laplacian}` sampled per clip PRE Multi-STFT. Raw operator skipped per per-corpus ambiguity. SWEC degenerates to global-CAR-only.
- Propagate `ref_idx ∈ {0,1,2}` per clip through to the encoder forward so the `ref_embed` lookup matches.
- 3-cell operator dataloader fan-out (one per ref) cached on `/work/` per HB02 cache envelope (~+18–30 TB across corpora).

### `src/speech_decoding/extractors/` **NEW**

- `subtype_meta.py` — emits per-clip `subject_subtype ∈ {0=sEEG-depth, 1=ECoG}` (or 3-way `{depth, grid, strip}` under the sister). Lookup table by `subject_id`.
- Per-clip `λ_anat` metadata propagation (could fold into existing metadata extractor; new module not required).

### `src/speech_decoding/experiments/monitors/` **NEW**

- `slot_redundancy.py` — MON-SLOT-REDUNDANCY (B28). Every 10k steps on held-out 256-clip probe batch. **Threshold defaults rescaled for M=1** (B29 Item 13, pre-preflight 5/28):
  - `per_clip_cos.pct95 > 0.5` sustained ≥ 50k (was 0.7) → escalate `R-dkoleo-intra-clip-slots`
  - `batch_cos.pct95 > 0.7` sustained ≥ 50k (unchanged — K-invariant) → escalate `R-dkoleo-batch-cls-unit`
  - `per_clip_cos.diag-zeroed.mean > 0.35` sustained ≥ 50k (was 0.5) → escalate `R-vicreg-slot-variance`
  - **Probe batch 256 → 1024** to compensate for 4× pct95 estimator noise from 16× fewer pairs.
  - Pre-preflight numbers; **1-GPU-h BT-Lite calibration preflight required** before Phase-1 dispatch — protocol spec'd in B28 memo §MON-SLOT-REDUNDANCY threshold rescale.
- `sensor_type_canary.py` — MON-SENSOR-TYPE-CANARY (B29 Item 9). Subject-subtype linear probe on the encoder output: kill if F1 ↑ > 0.05 over a baseline (catches over-reliance on `subtype_embed`).
- `ref_type_canary.py` — MON-REF-TYPE-CANARY (B29 Item 9). Ref-operator linear probe analog.
- `head_balance.py` — MON-HEAD-BALANCE-005 demoted from kill criterion to health canary (B29 Item 9). Same bounds [0.3, 3.0] but "investigate, don't gate."

### `src/speech_decoding/atlas/support.py` **EDIT**

- Multiply `log(support[e, p] + ε)` bias by per-clip `λ_anat` (driven by metadata, not step schedule).
- No change to ε default (1e-2); sisters drive ε sweep.
- For `R-swec-pseudo-parcel-per-shaft` P1 sister: build per-shaft pseudo-parcel mapping for SWEC clips so they get bias too. Default stays SWEC `λ_anat = 0`.

## §I. Tests

### Updates

- `models/test_v14_encoder.py`: param-count assertions — remove M=4 SubSlotEmbed (−1024 params), update for 1 cross-attn (was 2), add subtype_embed (512 params), add ref_embed (768 params). New total **~14.235M** (was ~15M at B19).
- `extractors/test_view.py`: assert 3 ref operators in per-clip draw + propagation of `ref_idx`.
- `experiments/test_v14_dispatch_wired.py`: assert all new dispatch fields default to spec values; assert `corpus_mix` sums to 1.0.

### NEW

- `experiments/test_v14_joint.py`: 4-term L1 loss formula matches spec; no context-loss term; DKoleo off in default forward; EMA teacher receives zeros mask.
- `models/test_v14_subtype_embed.py`: subtype_embed broadcast shapes; K/V reuse identity check; sister `subtype_embed_reuse_kv=False` matches M3AE-faithful pattern (no reuse).
- `experiments/monitors/test_slot_redundancy.py`: synthetic input where pct95 < / > thresholds → correct escalation tag.
- `experiments/test_lambda_anat_per_clip.py`: anatomy-rich clip gets `λ_anat=1.0`, SWEC clip gets `0.0`; sister `anatomy_bias_mode=step_b19` restores instant-on at P2 step 0.

## §J. Superseded from §A (5/26)

| 5/26 item                                          | Superseded by    | Resolution                                                                 |
| -------------------------------------------------- | ---------------- | -------------------------------------------------------------------------- |
| MASK-02 `parcels_supervised` extractor             | B29 Item 12      | **Do NOT build.** Only needed for `R-parcels-supervised-gating` P0 sister. |
| LOSS-01..05 5-term loss (incl. `0.1·L_DKoleo`)     | B27 + B28 Item 1 | 4-term L1; DKoleo off in default.                                          |
| LOSS-04 EMA ramp/decay schedule                    | B26              | Fixed τ=0.999 no ramp.                                                     |
| LAT-01 identity-anchored init                      | B29 Item 13      | M=1: drop SubSlotEmbed component.                                          |
| MASK-07 latent SA gate via `~supervised_slot_mask` | B29 Item 12      | Latent SA runs on all 80 slots (no gating).                                |

Still load-bearing from §A: FE-01..04 (front-end), LAT-02..05 (LayerNorms + M3 tap), MASK-01 (per-corpus notch), MASK-03 (shaft-mask DROP at cross-attn key_padding), MASK-04 / 05 / 06 (predictor + warm-start + asymmetric teacher), MASK-08 (bf16 sentinel), LOSS-06 (per-layer inst-norm in EMA), SCAFFOLD-01..09, P3-01..04.

## §K. Recommended dispatch order

**Tier 0 — parallelizable doc + small code** (Wave 1–2 from 5/26 still applies)

- 5/26 Wave 1: DOC-FIX-01..09 + remaining doc updates
- 5/26 Wave 2: MASK-01 (per-corpus notch), LOSS-06 (per-layer inst-norm), MASK-08 (bf16 sentinel), SCAFFOLD-07 (pytest marker)
- B26: EMA τ=0.999 fixed (one-line change in `ssl/ema.py`)
- B26: teacher full-input contract (drop mask args from teacher forward)
- HB02 doc-quality: fix `corpus_mix` to sum to 1.0

**Tier 1 — front-end** (still gates everything else)

- FE-01 → FE-02 → FE-03 → FE-04 (unchanged from 5/26)

**Tier 2 — encoder structural** (depends on Tier 1)

- B28 Item 2: cross-attn 2 → 1 (in `v14_encoder.py`) — **already landed via `cross_attn_positions` kwarg**
- B29 Item 13: M=1 default (drop SubSlotEmbed)
- LAT-02..05 (LayerNorms + M3 tap, unchanged from 5/26)

**Tier 3 — encoder conditioning** (depends on Tier 2)

- B29 Item 11: subtype_embed + ref_embed at A1 + K/V reuse
- B29 per-clip `λ_anat` gate (replaces B28 step warmup)
- REF-01 / REF-02 in `extractors/view.py`

**Tier 4 — loss + experiment scaffold**

- `experiments/v14_joint.py` NEW (4-term L1, no context loss, no gating, DKoleo off)
- MASK-03 / 04 / 05 / 06 from 5/26 (predictor + warm-start + asymmetric teacher) **adapted to single-phase**: predictor still warm-starts but boundary is now "epoch X → epoch Y" within the joint phase, not P1→P2.

**Tier 5 — monitors**

- MON-SLOT-REDUNDANCY + MON-SENSOR-TYPE-CANARY + MON-REF-TYPE-CANARY + demoted MON-HEAD-BALANCE-005

**Tier 6 — pre-dispatch derisks** (parallel)

- **1-GPU-h MFU micro-profile** (collapses HB02 H100-h range 510–1,900 → ±15%)
- **1-GPU-h MON-SLOT-REDUNDANCY preflight calibration** (locks empirical thresholds at μ + 3.1·σ)
- **Cell-0 BT-Lite sister** (5–10 H100-h) — gates full P1 rollout via 4 kill criteria (HG-patch 6–9 reconstruction loss > 5%, MON-MASK-002 out of [0.7, 1.5], Monitor F1 dev > 0.1, MON-MASK-004 subject-ID F1 ↑ > 0.05)
- **R-d-bump-384 Cell-0 sister** (~11–22 H100-h) — must-run; promotion gate ≥ 0.02 AUROC over d=256

**Tier 7 — Full P1 all-corpora dispatch** IF Cell-0 + R-d-bump-384 pass

## §L. Verification checklist Δ (append to 5/26 list)

- [ ] `ssl/ema.py`: no `tau_schedule` callable; `tau = 0.999` constant
- [ ] `ssl/ema.py`: `teacher_forward(electrode_mask=zeros, patch_mask=zeros)` regardless of student mask
- [ ] `experiments/v14_joint.py` exists; loss = `L1(M2_masked) + L1(M3_LN_mid) + L1(M4_LN_frame) + 1.0·L1(M4_LN_utt_PMA)`
- [ ] `experiments/v14_joint.py`: no `L_context_*` term
- [ ] `experiments/v14_joint.py`: no `parcels_supervised` gating; aggregates over all 80 slots
- [x] `models/v14_encoder.py`: `n_cross_attns == 1` (default; sister `R-perceiver-original-2-cross-attns` flips to 2 via `cross_attn_positions=[0, 3]`)
- [ ] `models/v14_encoder.py`: no `LearnableSubSlotEmbed`; `LearnableParcelEmbed.shape[0] == 80`
- [ ] `models/v14_encoder.py`: `subtype_embed: nn.Embedding(2, 256)` + reused in cross-attn K/V
- [ ] `models/v14_encoder.py`: `ref_embed: nn.Embedding(3, 256)` + reused in cross-attn K/V
- [x] `models/v14_encoder.py`: cross-attn forward `logits += λ_anat * bias` where `λ_anat` is per-clip from metadata (kwarg landed; metadata wiring still open)
- [ ] `dispatch_v14.py`: `notch_filter_hz["swec"] == 50.0`
- [ ] `dispatch_v14.py`: `sum(corpus_mix.values()) == 1.0` (asserted in test)
- [ ] `extractors/view.py`: per-clip uniform-random `ref_operator` draw + `ref_idx` propagation
- [ ] `monitors/slot_redundancy.py`: probe batch = 1024; pct95 threshold = 0.5
- [ ] **Preflight calibration completed** before P1: empirical (μ + 3.1·σ) thresholds locked
- [ ] **MFU micro-profile completed**: H100-h estimate ±15% locked

## §M. Open derisks carried forward

1. **MFU micro-profile** (1 GPU-h, DCC scavenger). Closes the 4× H100-h range. Same call HB02 v4 flagged 2 weeks ago; still open.
2. **MON-SLOT-REDUNDANCY preflight calibration** (1 GPU-h, DCC scavenger). Locks empirical thresholds at M=1. Protocol in B28 memo.
3. **Cell-0 BT-Lite sister rerun**: previous attempt invalidated by BTWordEvents class-imbalance bug.
4. **Cogan-lab Item 4 paper-framing**: B28 citation cleanup needs to propagate into the paper draft (M3AE not MultiMAE; LaBraM / Brant-2 have no sensor-type embed; DIVER-1 §4.1 expects net-neutral — see Agent 2 audit).

## §N. Agent 2 follow-ups (NOT yet cascaded into B29 memo)

- B29 memo "MultiMAE-faithful additive-at-input" → cite **M3AE 2022 §3.1** (Geng et al., arXiv 2205.14204). MultiMAE has no modality token.
- B29 memo "Field-standard binary distinction per LaBraM / Brant-2" → drop both. Neither has a sensor-type embedding (LaBraM = per-channel-name lookup; Brant-2 = no sensor embed, uses augmentation only).
- B29 memo DIVER-1 binary characterization → DIVER-1 is **3-way concat, not binary add**. Mention v14's binary additive collapse as a v14-specific choice, not a field standard.
- B29 paper-framing addition: DIVER-1 §4.1 ablation row "w.o Channel sub-modality emb." shows it's **net-neutral on iEEG**. Calibrate paper claim accordingly; don't scope subtype_embed as a novelty contribution.
- New sisters to add to roster: `R-subtype-embed-input-only` P1 (M3AE-faithful; no K/V reuse) + `R-subtype-embed-3way` P2-if-budget (DIVER-1's vocab).

## Δ Provenance

Cascade memos (5/27 → 5/28):

- [[project_v14_loss_design_amendment_b26_2026_05_27]] (B26 pure L1 + EMA fixed + teacher full-input)
- [[project_v14_loss_design_amendment_b27_2026_05_27]] (B27 context-loss revert)
- [[project_v14_ref_aug_input_distribution_lock_2026_05_27]] (REF-aug 3-cell per-clip draw)
- [[project_v14_loss_design_amendment_b28_2026_05_27]] (B28 7-item amendment)
- [[project_v14_b29_joint_default_2026_05_27]] (B29 Items 1–15)
- [[project_v14_moe_ffn_audit_2026_05_28]] (Item 14 MoE-FFN deferred to v15)
- [[feedback_v14_routing_atlas_is_dk_not_bna_2026_05_27]] (DK not BNA STOP-and-verify rule)

5/27–5/28 cascade gate: B28 (later memo) wins on cross-attn count, anatomy-bias warmup framing, Perceiver-IO / Graphormer citation framing. B29 (later still) wins on phase mode, M-default, subtype_embed / ref_embed, λ_anat per-clip gate, monitor demotions.
