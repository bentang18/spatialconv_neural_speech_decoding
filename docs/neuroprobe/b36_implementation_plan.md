# v14 B36 Implementation Plan

**Generated** 2026-06-03 from the B36 lock-vs-implementation audit (28-agent workflow + hand-verification against `HEAD=38ed421`).
**Definition of done**: the WS-I capstone — a must-pass P1→P2→P3→P4 fast-dev-run on a synthetic BT fixture — is green in CI.
**Trust order on conflict**: `memory/project_v14_b36_*` (newest lock) → this doc → `training_recipe.md` → `v14_blockers.md` → `plan.md` → code. Supersedes `docs/neuroprobe/v14_implementation_fix_list.md` (pre-B36, stale).

## State of the world (verified, not assumed)

- **B36 is doc-only.** `HEAD=38ed421` is B35; no B36 commit exists. Every B36 surface is unwired.
- **`src/` implements the *retired* design**: soft `λ_anat·log(support+ε)` routing as the pool (`v14_encoder.py:73,1028`); inert B31 2-term self-distill SSL (full input both sides, no mask, no predictor — `v14_joint_module.py:368-374`); B29 single joint phase (`dispatch_v14.py:148 DEFAULT_PHASE_MODE="joint_b29"`; `--phase 2/3` raise at `:1107`); B03 shaft-mask on the default path; M4 double-LN'd via a student-owned `ln_frame` head (`:424`); P4 trains the whole encoder unfrozen (`module.py:74`).
- **The 901-pass suite pins the retired design as correct.** CI-green is no signal for B36. Every task below ships its own test.
- **Correctly built, matches lock (do not touch):** B35 P4 readout module (`V14PmaReadout`, `v14_encoder.py:1529`); Whisper teacher = all-layer mean over 32 layers + triangular 50→8 Hz pool + per-channel z-score + SmoothL1 1280-d (`whisper_teacher_pool.py`, `distill.py`, numerically verified bit-parity 1.2e-7); pure-L1 + EMA τ-fixed + teacher-full-input primitives (`recon.py`, `ema.py`); B20 Conv2d(3,2) front-end + per-patch freq embed.
- **Today exactly one thing runs e2e**: a single-phase, scratch-init, BT-only Phase-4 supervised dry-run (`test_v14_wiring.py:165`) — and it trains the encoder unfrozen, so it isn't even the B35 frozen probe. Nothing chains phases; no checkpoint handoff exists.

## Decisions locked (2026-06-03)

| # | Decision |
|---|---|
| D1 | **M2/P1 mask = 0.5 fixed.** M4/P2 mask = sweep {0.2, **0.3** default, 0.4}. Defer the M2 `R-mask-ratio-{15,45}` sisters. |
| D2 | Predictor depth **= 3** (sweep center; {2,3,4} a config knob). I/O shape pinned by test now. |
| D3 | α-temperature sampling **= 0.3**. |
| D4 | AJILE12 native-rate (1000 vs 500) deferred — **BT-only run is the focus**; resolve before WS-H2. |
| D5 | **DK (K=80) = first-pass default.** Destrieux (`R-destrieux-pool`) = **must-run sister**. |
| D6 | Whisper layer-merge sisters **must-run**: `R-whisper-single-layer-L8`, `R-whisper-layer-weighted-sum`, `R-whisper-perlayer-instancenorm` (data2vec/EAT). |
| D7 | Mains notch: **adopt MNE `notch_filter` (fundamental + harmonics)** — suitable (notches line harmonics across the filterbank). Drop the `iircomb(Q=30)` spec requirement; WS-C verifies harmonic coverage spans to the top bin (~813 Hz) per corpus. |
| D8 | RoPE downward generalization (train T_p=20 / read T_p=4) **accepted** as a non-issue (RoPE is relative). Capstone asserts it. |
| D9 | sub_2 shaft bug: **fix via single source of truth** — clean `ch_names` in `loader.py` with `clean_bt_electrode_label` (matches `anatomy.py` *and* upstream Neuroprobe `braintreebank_subject.py:62`). **Clean, not drop** — `*`/`#` are cosmetic per the dataset authors (upstream strips them identically; benchmark numbers were computed on stripped labels). |
| D10 | STFT value axis: keep **magnitude `\|X\|`** default (Neuroprobe `stft_abs` parity); **log** stays the `F-log-amplitude` sister; **power `\|X\|²`** pooling is a minor sister axis. |

**Log rationale (settled, recorded so we don't relitigate):** robust-z (median/MAD per electrode×freq×session) removes a pure multiplicative per-channel gain *exactly* in linear space (`median(kP)=k·median(P)`, `MAD(kP)=k·MAD(P)` → k cancels), and MAD is already tail-robust. So log is **not** required for gain removal or tail control; its only residual job is mild Gaussianization. Default stays raw `|X|` on iMINDBench transformer evidence (Multi-STFT Logistic 0.663 / PopT-v2 0.660 beat waveform by ~12 AUC, no log) + dead D-SigLIP rationale. `F-log-amplitude` measures whether log buys anything at v14 scope.

---

## Workstreams (dependency-ordered, test-first)

Each task: **what** · `file` · **TEST** (quantifiable assertion). Tests are written *before* the change and must fail against `HEAD`, pass after.

### WS-A — Pool: soft routing → hard block-diagonal per-parcel pool  *(deps: none)*
B36's central architectural claim. One-hot DK source (`V14DKHardSupportExtractor`, K=80) is already wired as `support` (`dispatch_v14.py:520`) — the fix consumes existing input.

- **A1** Replace the soft additive bias with a block-diagonal one-hot DK assignment mask: query k attends ONLY to parcel-k electrodes, off-parcel = `NEG_INF_MASK_VALUE`, strict per-time-patch (`key_rope=None`, output `(B, K·M, T_p, d)` stay). · `models/v14_encoder.py` · **TEST**: for a fixture with known parcel assignment, post-softmax attention weight on every off-parcel (electrode, k) pair == 0.0 exactly (not ≤ tolerance); on-parcel rows sum to 1.0.
- **A2** Delete `compute_gated_log_support_bias` + `λ_anat` kwarg threading (encoder fwd, `V14ParcelPerceiverModel`/`WithHead`, joint-module `batch_data`). · `models/v14_encoder.py`, `atlas/support.py` · **TEST**: `grep` for `compute_gated_log_support_bias`/`lambda_anat` in non-test `src/` returns empty; encoder forward signature has no `lambda_anat`.
- **A3** Drop `LambdaAnatExtractor` + `DEFAULT_CORPUS_LAMBDA_ANAT` construction/registration from dispatch (keep `SubjectSubtypeExtractor` sister). · `experiments/dispatch_v14.py` · **TEST**: dispatch builds a Data spec with no `LambdaAnatExtractor`; existing dispatch tests still pass.
- **A4** Tighten the invalid-parcel/channel grad tests to exact-zero off-parcel (was ≤10× soft-leak tolerance). · `models/test_v14_*` · **TEST**: gradient of a masked-out electrode's pool contribution == 0.0.

### WS-B — SSL objective: paradigm-B masked JEPA  *(deps: WS-A)*
The biggest build. `src/` has no mask sampler, no real predictor, and an inert loss. Canonical V-JEPA-2 target-norm CONFIRMED from arXiv:2506.09985 §2 (L1, sg directly on EMA encoder output, normalized only by the encoder's own terminal LN, no separate head).

- **B1** Add terminal `frontend_ln` at M2 (mirror on EMA teacher); return post-`frontend_ln` M2 tap. · `models/v14_encoder.py` · **TEST**: M2 tap mean≈0/var≈1 per feature on a fixture; teacher M2 tap uses the EMA-copied `frontend_ln`.
- **B2** Build `JepaPredictor`: default **3 blocks @ d=128, 4 heads, MLP 4×, terminal Linear→d_model, NO per-head LN** (~0.6M); context = visible tokens, queries = learnable mask tokens tagged by (id/parcel + position) embed; output at masked positions only. **Depth is a config knob (D2, center 3).** · `models/v14_encoder.py` (replace `Predictor2Block`) · **TEST**: instantiate at depth∈{2,3,4}; param count at depth 3 within ±2% of target; output shape == (n_masked, d_model); depth-2 ≠ B36 default (regression guard).
- **B3** Build M2 mask sampler: seeded per-(electrode, freq-patch, time-patch) Bernoulli at **0.5** over the (C, F_p, T_p) grid → visible/masked index sets. · `ssl/mask.py` (new) · **TEST**: empirical masked fraction → 0.5 ± 0.02 over 1000 draws; same seed → identical mask; visible ∩ masked == ∅.
- **B4** Build M4 mask sampler: (parcel × contiguous-time-block) at **0.3** of COVERED cells (sweep {0.2,0.3,0.4}); return BOTH the (K, T_p) mask AND the derived upstream (electrode, time-patch) drop set via the WS-A DK one-hot. · `ssl/mask.py` · **TEST**: masked cells are contiguous in time per parcel; masked fraction of covered cells → 0.3 ± 0.02; every dropped (electrode,t) maps to a masked parcel under the one-hot.
- **B5** Build visible-only student forward + upstream (electrode, time-patch) drop into the token-block input (front-end never encodes masked-parcel cells → leakage-free). · `models/v14_encoder.py`, `experiments/v14_joint_module.py` · **TEST**: with an M4 mask, the front-end input tensor has the dropped (electrode,t) zeroed/absent; changing a masked-region input value does NOT change any visible token (leakage assertion).
- **B6** Replace the inert loss with masked-only L1: predictor output vs `sg(EMA-teacher full-input post-terminal-LN)` at MASKED positions only — M2 vs `frontend_ln(M2)`, M4 vs post-`encoder_ln(M4)`. **Delete the student-owned `ln_frame` head + teacher mirror** (canonical V-JEPA). · `ssl/total_loss.py`, `v14_joint_module.py` · **TEST**: loss == 0 when masked set empty; loss is L1 (not MSE) — check gradient magnitude constant in error; no `ln_frame` module on the path; target tensor is `detach()`ed.
- **B7** Wire `assert_teacher_full_input` with the ACTUAL student-side masks (currently `None,None` → vacuous). · `v14_joint_module.py` · **TEST**: guard raises if a non-None mask reaches the teacher pass.
- **B8** Build 3-way `latent_valid` (visible = covered & ~masked / target = covered & masked / teacher = covered); thread visible→encoder SA mask, target→loss+predictor queries, teacher→EMA full-input pass. · `models/v14_encoder.py` · **TEST**: the three sets partition `covered`; visible ⊎ target == covered; teacher == covered.
- **B9** Quarantine dropped-term scaffolding from the default path: M3 tap + `L_mid_slot`, `L_post_utterance` (keep `pma_then_mean` for P3 only), `context_loss`, DKoleo, `layer_avg_with_instance_norm` (dead, mislabeled live in `ema.py` docstring). · `ssl/`, `v14_joint_module.py` · **TEST**: default loss composition has exactly 1 active term per phase (P1: M2; P2: M4); dead helpers raise or are removed from imports.

### WS-C — Front-end: Multi-STFT + robust-z + phase-conditional clip  *(deps: none)*
`MultiStftView` (F=30, hop 256→8 Hz, raw `|X|`) + `Nv14RobustZTransform` exist + tested but are orphaned; dispatch default is single-STFT `LogStftView` + StandardScaler-on-voltage.

- **C1** Thread a phase-conditional `clip_len` into `build_v14_experiment` → segmenter duration (**5.0s** P1/2/3 → T_p=20; **1.0s** P4 → T_p=4 @ 8 Hz). · `experiments/dispatch_v14.py` · **TEST**: segmenter emits T_bin=40→T_p=20 at 5s, T_bin=8→T_p=4 at 1s.
- **C2** Flip dispatch default `electrode_tokens_extractor` to `MultiStftView` (hop=256, nperseg 1024/512/256, `apply_log=False`, F=30); set `DEFAULT_N_FREQ_BINS=30` → encoder F_p=10. · `dispatch_v14.py` · **TEST**: default view is `MultiStftView`; output (C, 30, T_bin); encoder F_p==10.
- **C3** Wire `Nv14RobustZTransform` as a pre-segmenter SESSION-level fit (median/MAD once per (electrode, freq-bin, session) over the full recording, honoring per-corpus `valid_bin_mask`); drop StandardScaler from the default view. · `dispatch_v14.py`, `extractors/normalize.py` · **TEST**: a pure ×k gain on one channel leaves the normalized output unchanged (gain-invariance, ρ=1.0); stats fit on train only.
- **C4** Add HPF 0.5 Hz to the active front-end (`filter=(0.5, None)`; currently `None`). · `dispatch_v14.py` · **TEST**: DC/slow-drift component attenuated > 40 dB at 0.1 Hz; passband flat ≥ 1 Hz.
- **C5** Thread per-corpus `valid_bin_mask` (SWEC k0–21 → F-patch 0–6) into robust-z stats, post-z zero-fill, encoder freq-SA key mask, and SSL masked-cell target exclusion. · `dispatch_v14.py`, `v14_encoder.py`, `ssl/` · **TEST**: SWEC fixture → bins k22–29 are zero-filled and excluded from both the freq-SA keys and the L1 target.

**WS-C LANDED 2026-06-03 (audited 4/4 adversarial PASS — correctness/spec/test-quality/elegance).** Leaf consumers + tests done: C1 (clip_len→duration, T_p 20/4), C2 (`MultiStftView` default, F=30→F_p=10), C3-primitive (`SessionRobustZNormalizer` fit-on-train/apply-frozen + StandardScaler dropped), C4 (HPF 0.5 Hz), C5 (`freq_patch_valid_mask` + freq-patch exclusion in token-block SA keys, cross-attn pool keys, P1 L1-target). Two intentional carve-outs, both off the BT capstone critical path:
  - **C3 live in-chain wiring deferred to WS-H/H4 (`V14Data`).** The session-fit precompute (full-recording per-(electrode,freq,session) median/MAD → cache → apply pre-segmenter) is a MapInfra data-pipeline concern; only the primitive + StandardScaler-drop landed. **Consequence: the default dispatch currently emits UN-normalized Multi-STFT features** — fine for the BT `fast_dev_run` smoke (per-token encoder LayerNorm handles scale), MUST be closed before any real training run.
  - **C5 producer-wiring deferred to WS-H/H4.** No caller computes/passes `freq_patch_valid` yet (SWEC `_load_raw` is a `NotImplementedError` stub); the encoder/SSL leaves are tested via synthetic SWEC-shaped fixtures. BT (all 30 bins valid) → strict no-op.
  - **C4 deviation (accepted):** plan's "40 dB at 0.1 Hz" is unreachable via the `(0.5,None)` MNE tuple (0.1 Hz sits in the auto transition band → ~23 dB); full DC removal + >50 dB at 0.02 Hz. The C4 test pins the real attenuation profile.
  - **WS-H reconcile (receipts):** `multi_stft_valid_bin_mask(0.5,120)` returns **k0–22** (23 bins; k22 lo_edge 114 Hz < 120 passes "any overlap"), NOT the memory/plan shorthand **k0–21**. F-patch result (0–6) is identical either way (patch 7 needs bin 23). When WS-H wires the SWEC producer, reconcile the valid-bin criterion vs the frozen B1 contract WITH BEN (calibration gate) — do not change the criterion silently.

### WS-D — MNE-LOF bad-channel drop + per-run channel-quality output  *(deps: none; Ben #1)*
+ **the sub_2 shaft fix (D9).** `MneLofBadChannelMask` only *returns* a mask today (zero consumers); the shaft-CAR drop seam is `CARIeegExtractor._preprocess_raw` pre-CAR.

- **D1** Clean `ch_names` in the BT loader via `clean_bt_electrode_label` (single source of truth, matches `anatomy.py` + upstream Neuroprobe). · `studies/braintreebank/loader.py:44` · **TEST**: (a) a regression test reproduces the bug on RAW sub_2 labels → 34 singleton shafts; (b) after the fix, `parse_shaft` on loader `ch_names` == `parse_shaft` on `anatomy`-cleaned labels for all 9 subjects; (c) post-shaft-CAR, no channel is all-zero on a sub_2 fixture (the 34 zeroed contacts are gone).
- **D2** Wire a session-level LOF screen pre-shaftCAR: run `lof_bad_channel_mask` on post-HPF/notch voltage, push bad indices into `raw.info['bads']` before `reference.py` drops them. · `studies/*/...`, `extractors/quality.py`, `reference.py` · **TEST**: a synthetic flat/noise channel is flagged and excluded from the shaft-CAR mean; clean channels untouched.
- **D3** Emit per-run channel-quality output: n_bad, bad-channel ids, dropped fraction → run record + a logged scalar (per Ben "so we know clearly the quality of the channels"). · `experiments/...` · **TEST**: run record contains `channel_quality` with the expected keys; scalar logged once per fit.

**WS-D STATUS 2026-06-03 (audited; re-audited after spec-fidelity fix):** **D1 LANDED** — `bt_load_raw` cleans `*`/`#` at the loader boundary via `clean_bt_electrode_label` (single source of truth; idempotent; matches `anatomy.py` + upstream Neuroprobe `braintreebank_subject.py:62` byte-for-byte). Regression locked in `studies/braintreebank/test_loader.py`: (a) raw sub_2 → 34 singleton shafts, cleaning collapses to one; (b) loader `ch_names` == an INDEPENDENT hand-spelled expected (trailing+mid-label+plain marks) + grouping changes + no surviving mark; (c) differential shaft-CAR — raw path zeros all 34, cleaned path wipes none; (d) **all real vendored subjects** — loader strips every mark vs an independent in-test strip, exercising the mid-label cohort sub_7/9/10 (skips when the untracked `.cache/` fixtures are absent; asserts {2,7,9,10} coverage so it cannot pass vacuously). **D2 primitive LANDED** (`quality.lof_bad_channel_mask` + `MneLofBadChannelMask`). **D2-live + D3 DEFERRED to WS-H.** The drop seam `CARIeegExtractor._preprocess_raw` is session-cached on the BT path (`IeegExtractor._get_data` keys on the full-session `Ieeg` row; word windows are sliced downstream), so granularity is NOT the blocker — an earlier note claiming a per-trial re-screen was wrong. The real reasons: (1) off the capstone critical path — the synthetic-BT capstone needs neither; (2) D2-live changes which channels are dropped from CAR = a calibration/label-derivation change → Ben-approval-gated (CLAUDE.md infra gate); (3) `quality.py`'s docstring already assigns the wiring to the Study/Transform layer; (4) same leaf-now / live-to-WS-H precedent as C3/C5. The BT session seam (`Wang2024Treebank._load_raw`) is in fact live and dependency-free, so wiring D2-live for BT is a one-seam post-capstone change pending Ben's calibration sign-off (not blocked by the SWEC/D-cohort stubs). D3's `channel_quality` has no bad-channel source until D2-live, so it ships with it.

### WS-E — Phase orchestration: staged P1/P2, freeze/LR groups, checkpoint handoff  *(deps: WS-B; keystone)*
`_train_and_test` is a single fit→test; nothing chains phases; zero `load_state_dict` in non-test experiments; both optimizers take one flat LR.

- **E1** Replace the B29 joint default with staged **P1** (front-end-only optimizer, L_pre_frame/M2 only, all corpora; pool/encoder/M4-predictor get NO gradient). · `experiments/v14_joint*.py`, `dispatch_v14.py` · **TEST**: after a P1 backward, pool/encoder/M4-predictor grads are None-or-zero; front-end grads nonzero.
- **E2** Build staged **P2** (pool + inter-parcel + M4-predictor trainable; front-end param-group at **LR/10**; anatomy corpora only; L_post_frame/M4 only). · same · **TEST**: optimizer has 2 param groups with LR ratio 10:1; front-end group == base/10.
- **E3** Build the multi-phase driver chaining P1→P2→P3→P4 (run → snapshot ckpt → next phase loads it); add a `pretrained_ckpt` arg to `build()`. · `experiments/...` · **TEST**: driver runs 4 phases in sequence; each phase's build receives the prior ckpt path.
- **E4** Build checkpoint handoff for all pairs (P2→P3 encoder; P3→P4 encoder+PMA strict-load, `StudentWhisperProjector` keys dropped). · `experiments/...` · **TEST**: no missing/unexpected keys at each boundary (strict on shared keys; projector keys explicitly dropped).
- **E5** Fix the false "blockers fire from `_train_and_test`" docstrings/tests: `V14JointExperiment` overrides neither `run` nor `_train_and_test`; `_PHASE1_BLOCKERS` is dead. · `experiments/v14_joint.py` · **TEST**: docstrings match actual call graph; no dead `_PHASE1_BLOCKERS` reference.

### WS-F — P3 Whisper-distillation phase  *(deps: WS-E)*
All P3 primitives are correct + tested (project-up: `StudentWhisperProjector` 256→1280 LLaVA-shape ~1.97M / linear ~0.33M; `triangular_pool_50_to_8_hz`; `fit_channel_stats`/`TargetStandardizer` two-pass fp32 train-only; SmoothL1 β=1.0) but have zero training callers; `--phase 3` raises.

- **F1** Build `V14Phase3DistillModule`: frozen-P2 encoder → PMA k=1 → `StudentWhisperProjector(mlp)` → SmoothL1(β=1.0) vs cached Whisper `mean_all` → `triangular_pool_50_to_8_hz` → `TargetStandardizer` (train-only stats). · `experiments/v14_phase3.py` (new) · **TEST**: finite SmoothL1 in 1280-d on a 2-clip fixture; target standardizer stats fit on train pool only.
- **F2** Implement 3a-warmup (freeze encoder, train PMA+projector) → 3b-unfreeze (encoder A@LR/10, B@/3, PMA+head@full) LR groups; wire `--phase 3` (remove the NotImplementedError gate). · `v14_phase3.py`, `dispatch_v14.py:1107` · **TEST**: 3a → `encoder.requires_grad` all False, only PMA+projector train; 3b → 3 param groups with the {/10, /3, full} ratio.
- **F3** Add numerical Whisper-adapter regression tests (rows sum to 1 within 1.2e-7; 50→8 Hz grid alignment; `mean_all` bit-parity to the ceiling probe; fp32-accum upcast; per-channel z-score zero-var guard). · `extractors/test_whisper_teacher_pool.py` · **TEST**: as listed (these PASS today; pin them so a regression fails CI).
- **F4** **[Ben-owned prose, IRONCLAD — I flag, Ben writes]** Reword the `mean_all` lock prose (it overclaims "beat every single-layer pick"; truth = 4/8 cross-movie, 2/9 cross-session; defensible rationale is per-task-AGNOSTIC); fix the stale project-down / "Whisper-L8" encoder comment; resolve the β-required-vs-default contradiction in `distill.py`. · `bt_alignment/contract.py`, `distill.py` docstrings · **TEST**: I supply the diff for Ben's approval; no auto-write.

### WS-G — P4 readout: encoder freeze + P3-PMA load + `v14_phase4.py`  *(deps: WS-E)*
B35 module is correct; the *trainer* violates B35 (whole encoder trainable, PMA random-init, no ckpt load, `v14_phase4.py` absent, `test_v14_encoder.py:573` pins the wrong unfrozen state).

- **G1** Create `experiments/v14_phase4.py`: P4 BrainModule that loads the P3 ckpt (encoder+PMA strict), calls `encoder.requires_grad_(False)`, and builds the optimizer over the readout classifier params ONLY. · new file · **TEST**: encoder AND PMA `requires_grad==False`; optimizer param set == the Linear only.
- **G2** Add exact param-count regression pins. · `test_v14_phase4.py` · **TEST**: frozen PMA == **263,424**; classifier == **514** (binary) / **2,570** (10-way) — both stable (B35, pool-independent). Total: assert `frozen == encoder(measured) + 263,424` and `total < 30M`; **measure-and-pin the exact total once the B36 hard-pool encoder is assembled** (do NOT pin the pre-B36 ~12.12M figure — the hard pool changes the encoder count). A trainable-PMA or B34-attentive regression fails this.
- **G3** Build the CSubject leave-2-subjects-out collator with the no-electrode-overlap assertion + emit a Neuroprobe probe + metric. · `v14_phase4.py` · **TEST**: `set(probe_train_electrodes) & set(probe_test_electrodes) == ∅` (BP20); test step emits a finite AUROC/accuracy scalar. **(D8)** Also assert P4 reads T_p=4 from an SSL encoder trained at T_p=20 without shape error (RoPE downward-generalization guard).

### WS-H — Multi-corpus: SWEC/AJILE12/D-cohort loaders + α-sampler  *(deps: WS-C; NOT on the capstone critical path)*
Three of four corpora are `NotImplementedError` stubs; the α-temperature WRS sampler was deleted; `corpus_mix` is validated then dropped. **BT-only capstone needs none of this** (all 30 BT bins valid) — land after green e2e.

- **H1** SWEC DP03 loader (part-file timeline read; 5 s non-overlap tile grid minus seizure-guard `[onset−30 min, offset+90 min]`; native CMR; **50 Hz** notch at native fs THEN resample 512/1024→2048). · `studies/swec/study.py` · **TEST**: a tile never overlaps the guard window; output sfreq==2048; notch fundamental 50 Hz.
- **H2** AJILE12 loader (no seizure guard) + **resolve SAMPLE_RATE_HZ 1000-vs-500 (D4)** against Peterson 2022 / DANDI before locking the resample. · `studies/ajile12/study.py` · **TEST**: resample to 2048 from the verified native rate.
- **H3** D-cohort loader + 2000→2048 resample + DP05 per-subject DK index map. · `studies/cogan_dcohort/study.py` · **TEST**: output sfreq==2048; DK index map covers all channels.
- **H4** Build `V14Data` + **α=0.3** temperature WeightedRandomSampler over per-(corpus,session) valid-bin-electrode-hours; P1 all corpora, P2 anatomy-only; route per-corpus notch (D7 MNE, 60/50 Hz) to per-corpus loaders. · `experiments/data.py` · **TEST**: empirical corpus sampling frequencies match the α=0.3 temperature law within tolerance.
- **H5** Drop `BTShaftMaskExtractor` from the default segmenter (keep as explicit sister); give split/staged a real runtime branch. · `dispatch_v14.py` · **TEST**: default segmenter has no shaft-mask extractor; sister flag re-adds it.

### WS-I — RankMe-from-step-0 + capstone + doc/lock consistency  *(deps: WS-F, WS-G)*

- **I1** Fire RankMe on the TRAIN loop from step 0 (drop the val/test gate) on the bare post-`encoder_ln` M4 parcel target (after `ln_frame` is removed in B6); `latent_valid`-masked, sub-sampled. · `monitors/teacher_rank.py`, `v14_joint_module.py:762` · **TEST**: RankMe logged at global_step 0; reads the post-`encoder_ln` tap, not `ln_frame`.
- **I2** **Build the must-pass capstone** (spec below). · `experiments/test_v14_capstone.py` (new) · **TEST**: the capstone itself.
- **I3** Reconcile docs + drift guard with B36: re-flag the closed-against-retired `v14_blockers.md` rows; add the B36 rows; banner `training_recipe.md` §1/§3/§4 as SUPERSEDED; give `check_lock_drift.py` its B36 pass (add retired literals as forbidden, relax the now-canonical ones). · `docs/`, `check_lock_drift.py` · **TEST**: drift guard flags soft-routing/inert-SSL literals as forbidden and passes the B36 design.

---

## Critical path (20 tasks → capstone)

```
A1 → A2 → B1 → B2 → B3 → B4 → B5 → B6 → B8 → C1 → C2 → E1 → E3 → E2 → E4 → F1 → F2 → G1 → I1 → I2
```

Dependency-free, can start in parallel with A1: **WS-D** (LOF + sub_2 fix), **C3/C4** (robust-z, HPF), **F3** (pin the already-correct Whisper numerics). WS-H is deferred until after the first green e2e.

## Capstone test spec (WS-I2 — definition of done)

`experiments/test_v14_capstone.py :: test_p1_p2_p3_p4_end_to_end_on_bt` — **must-pass, no skip** (runs on a synthetic Nano-shaped BT fixture, no `ROOT_DIR_BRAINTREEBANK`).

**Fixture**: 2 clips, ~8 electrodes over a few DK parcels (K=80 slots, a few covered), Multi-STFT F=30/F_p=10, **5 s** clips for P1/P2 (T=40→T_p=20), **P3 at T=80→T_p=40**, and **1 s** for P4 (T=8→T_p=4), d=256, encoder depth 2 (speed), M=1. A 2-clip synthetic Whisper cache (`mean_all`, 250 frames @ 50 Hz → 40 @ 8 Hz) with `fit_channel_stats` over those 2 clips for the P3 `TargetStandardizer`. Lightning `fast_dev_run=1` per phase; each phase loads the prior phase's ckpt via the WS-E driver.

> **P3 `T_p` reconcile (flagged by the WS-I audit, 2026-06-03).** This table originally said "P1/P2/P3 at T_p=20." That is unrunnable: the live Whisper teacher pool (F2) hard-pins **40** frames (8 Hz × 5 s), and `V14Phase3DistillModule` asserts student `T_p == teacher T_p`, so P3 cannot run at `T_p=20`. The capstone therefore feeds P3 at `T_bin=80 → T_p=40` — the only config aligned with the pinned teacher. This is sound for the test (the shared encoder is size-agnostic — RoPE is `persistent=False`, the conv stem is size-agnostic — so the strict `T_p=20→40` cross-phase load is clean, verified). But the **production** P3 input geometry that yields `T_p=40` (10 s clips vs a non-time-halving stem for P3 vs re-pinning the teacher pool to 20 frames) is a **Ben-gated design decision**, not settled here. C1/G3 (`clip_len` threading) and the F2 teacher-pool frame count must be reconciled before P3 dispatch.

| Phase | Assertions |
|---|---|
| **P1** (front-end masked JEPA, M2 0.5, BT-only) | (a) finite non-NaN loss; (b) masked-only L1 (== 0 when masked set empty); (c) **M2-gradient-scope**: pool / inter-parcel encoder / M4-predictor grads None-or-zero, front-end (stem + token-blocks + `frontend_ln`) grads nonzero; (d) EMA teacher params moved by the fixed-τ rule. → save `ckpt_P1`. |
| **P2** (pool+encoder+M4-predictor, M4 0.3 parcel×time, anatomy-only) | load `ckpt_P1` (assert front-end state == P1's); (a) finite; (b) **leakage-free** (masked-parcel electrodes dropped upstream); (c) front-end LR == base/10, pool/encoder/predictor == base; (d) loss vs `sg(post-encoder_ln M4)`, NO `ln_frame` present; (e) RankMe logged from step 0 on the post-`encoder_ln` target. → save `ckpt_P2`. |
| **P3** (Whisper distill, project-up) | load `ckpt_P2` encoder (strict on shared keys); 3a (encoder all `requires_grad==False`, only PMA+projector train) 1 step; 3b (LR groups encoder-A@base/10, encoder-B@base/3, PMA+head@base) 1 step; finite SmoothL1 in 1280-d vs the standardized triangular-pooled `mean_all`. → save `ckpt_P3` (encoder+PMA; projector ignorable). |
| **P4** (frozen readout, B35) | load `ckpt_P3` (encoder+PMA strict; projector keys dropped); **P4 invariant**: encoder AND PMA `requires_grad==False`; only-trainable == per-task Linear (514 binary / 2,570 10-way); frozen PMA == 263,424; total < 30M (exact total measure-and-pinned, not the pre-B36 figure); 1 train + 1 test step on a binary probe; (a) finite CE, backward updates ONLY the Linear; (b) finite Neuroprobe probe metric. RoPE T_p=20→T_p=4 read succeeds. |
| **Global** | no missing/unexpected keys at any boundary; all four phases complete without raising. |

## Deferred / sister roster (not on the critical path)

- **Mask**: `R-mask-ratio-{15,45}` (M2; deferred per D1).
- **Pool/anatomy**: `R-destrieux-pool` (must-run, D5), `R-no-bottleneck` (BaRISTA-shape control, mandatory), `R-finer-parcellation`.
- **SSL**: `R-paradigm-a-mlp`, `R-joint-ssl` (B29 P0 falsifier), `R-predictor-depth-{2,4}`, `R-cross-time-predictor`, `R-add-m3-loss`.
- **Whisper** (all must-run, D6): `R-whisper-single-layer-L8`, `R-whisper-layer-weighted-sum`, `R-whisper-perlayer-instancenorm`, `R-no-whisper` (mandatory clean-novelty control).
- **P3**: `R-project-down`, `R-head-linear`, `R-no-target-standardize`, `R-frozen-throughout`.
- **P4**: `R-p4-flatten`, `R-p4-time-attn-pool`, `R-p4-attentive`, `R-p4-meanpool-no-pma`.
- **Front-end**: `F-log-amplitude` (D10), STFT power-pooling sister (D10), `R-swec-wrs-sampler`.
- **Eval**: LOSO (frozen-probe + selective-FT) head-to-head vs BaRISTA 0.841.

## References
- B36 lock: `memory/project_v14_b36_perparcel_pool_structured_jepa_2026_06_01.md`
- P4/P3/SSL: `project_v14_b35_*`, `project_v14_b33_*`, `project_v14_b31_*`, `project_v14_whisper_teacher_all_layer_mean_2026_05_30`
- Drift table: `docs/neuroprobe/v14_blockers.md` · Compiled contract: `docs/neuroprobe/training_recipe.md`
- V-JEPA-2 target-norm: arXiv:2506.09985 §2/§2.4 (verified)
