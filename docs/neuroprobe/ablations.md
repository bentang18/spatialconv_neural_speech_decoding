
# Neuroprobe Ablations — Running Menu

Single source of truth for every ablation cell across Stage-0 (linear preprocessing hill-climb), Stage-1 (architectural defenses), Stage-2 (loss-component + schedule), and Stage-3 (foundation-model swap).

**Source-of-truth rule.** This doc is the menu. Per-cell results, freeze write-ups, and dispatch artifacts live in their canonical homes: Stage-0 cells → `docs/neuroprobe/stage_0.md`; Stage-1/2/3 cells → `docs/strategy/stage_<N>.md` + `docs/neuroprobe/stage_<N>.md` + relevant memos. Cite, don't duplicate. Reports under `reports/neuroprobe_stage*/`.

**Status legend.** `frozen` = decision landed, default inherited downstream. `in-flight` = dispatched on DCC, results pending. `planned` = spec'd, awaiting prereq. `blocked` = waiting on external. `deferred` = post-Stage-0 / off-critical-path.

**Effect-size threshold.** "Load-bearing" = ΔAUROC ≥ 0.02 multiclass CrossSession (one CI half-width of Neuroprobe linear baseline). Below threshold → freeze upstream parity.

---

## 1. Stage-0 — V block (data-contract QC gates)

Not ablations strictly — data-QC gates that decide which signal views are eligible Stage-1 candidates. Canonical: `docs/neuroprobe/stage_0.md §1`. **Completed 2026-05-01** (report `reports/neuroprobe_stage0_v_data_qc_2026_05_01_v3/`).

| Cell | Subject | Headline finding |
|---|---|---|
| **V0** | Lite vs Full selection audit | Lite is leaderboard-parity set, biased toward decodable electrodes; record explicitly. |
| **V0.x** | Stimulus-overlap audit (CrossSession leakage) | Per (subj
ect, test-trial), unique-word overlap **median 0.414, max 0.450** across 12 BT Lite pairs. **No (subject, task) pair exceeds 50% kill threshold** at upper bound. Per-task DCC pass pending. |
| **V1** | Raw signal health | raw monopolar high common-mode burden (median top eigen-fraction 0.400, max 0.711, median |corr| 0.412) |
| **V2** | Reference transform QC (R0/R1/R2/R3) | Robust global CAR reduces 0.214/0.169 — diagnostic. Shaft-local median + shaft-bipolar reduce 0.167-0.179/0.124 — **the real Stage-1 references** (consistent with L.2 R3/R4 winners) |
| **V3** | Frequency-view QC | local-reference HG/HFA = biologically privileged supervised view; raw 2048 Hz = auxiliary/ablation |
| **V4** | Artifact source battery | nuisance probes decode subject-ID at balanced acc 0.699 (chance 0.167) + session-ID 0.518 (chance 0.083) — corroborates L.5.P1/P2 necessity |
| **V5** | Event-locked sanity | decides whether anchor-robustness ablation (= L.4) is mandatory |
| **V6** | Stage-1 input-view decision table | synthesizes V0-V5 into role table per view (default / required-ablation / diagnostic) |
| **V7-V8** | Surface geometry + parcel coverage plots | conditional on A0-A4 clearing |

### Stage-0 geometry / public-label gates

These are blockers and support controls, not model ablations, but they feed the Stage-1 AC6 and Stage-2 JEPA coverage contract. Canonical: `docs/neuroprobe/stage_0.md §2-§3`.

| Gate | Subject | Output / decision |
|---|---|---|
| Public hard-label coverage | public BT `depth-wm.csv:DesikanKilliany` labels across full filtered + Lite electrodes | valid only as `support_kind="hard_public_bt_label"` DK/Destrieux control; not BNA/fsaverage support |
| Shaft parser | canonical parse from electrode label to `(subject, shaft_id, contact_index)` | required before local reference provenance or depth features |
| Contact-order orientation | check deep-to-superficial vs superficial-to-deep consistency | signed depth disallowed unless orientation is verified |
| Transferable shaft/depth features | no depth / ordinal index / normalized position / centered normalized / orientation-invariant features | no numeric default until orientation and nuisance risk are audited |
| Local-reference virtual-channel metadata | source contacts, offsets, virtual labels, unresolved coordinate/support status | prevents rereferenced channels from masquerading as physical contacts |
| Shaft/depth nuisance gate | subject/session decoding with and without shaft/depth features | depth promotes only if transfer improves without subject-ID shortcut |

## 2. Stage-0 — L / A / D ablation blocks

Three parallel Stage-0 blocks below: **L** (linear preprocessing hill-climb, greedy L.1 → L.2 → L.3 → L.4 + L.5 probe gates + L.6 deferred + L.7 audio-FM upper bound), **A** (atlas/surface-mapping gates, anatomy-dependent), **D** (atlas/pooling cells, gated on A0-A4). L validated by L.4 interaction sanity (max |residual| = 0.0008, 2026-05-10). Canonical: `docs/neuroprobe/stage_0.md §4-§6`.

### L.1 — Normalization scope

| Cell | Recipe | Status |
|---|---|---|
| L.1.N0 | per-window z (BrainBERT/PopT) | rejected (−0.054 vs N1) |
| **L.1.N1** | **train-set fixed (StandardScaler)** | **FROZEN 2026-05-08** |
| L.1.N2 | per-session fixed (transductive) | +0.004 over N1 (overlap), v15-territory |
| L.1.N3 | train-set scale-only | tied N1 (p=0.46) |
| L.1.N4 | none / raw Lap+spec | rejected |
| L.1.N5 | per-session robust median/MAD | tied N1 ± SEM |
| L.1.N6 | train-set robust MAD | rejected |
| L.1.N7 | per-session robust scale | rejected |
| L.1.N8 | per-channel train-set z | rejected (−0.058 vs N1, **v14 token-norm must pool across channels within a parcel, not per-electrode**) |

Headline N0 − N1 = −0.054: BrainBERT/PopT recipe costs ~5.4 pp — preprocessing IS load-bearing for the linear-vs-FM gap. Detail: `reports/neuroprobe_stage0_l1_normalization_2026_05_05/`.

### L.2 — Reference × input-view (Tier-A → Tier-B → Tier-C)

**Reference axis** (R0 raw / R1 global CAR / R2 shaft CAR / R3 bipolar / R4 shaftLap) × **view axis** (I0 voltage / I1 low-LFP / I2 stft_abs / I2L log-STFT / I3 HG env / I3W wide HG 70-250 / I4 multi-band log-power / I5 wavelet / I6 theta phase).

**Tier-A (9-cell grid: 3 ref × 3 view) — FROZEN 2026-05-09 at R4×I2**

| Cell | Recipe | Mean | Δ vs baseline |
|---|---|---|---|
| R3×I2 | bipolar × stft_abs | 0.6157 | +0.0025 (CI overlap, NOT load-bearing) |
| **R4×I2** | **shaftLap × stft_abs (D.0 parity)** | **0.6132** | **0** |
| R0×I2 | raw × stft_abs | 0.5923 | −0.0209 |
| R3×I3 | bipolar × HG env | 0.5893 | −0.0239 |
| R4×I3 | shaftLap × HG env | 0.5868 | −0.0264 |
| R0×I3 | raw × HG env | 0.5743 | −0.0389 |
| R0×I0 | raw × voltage | 0.5534 | −0.0598 |
| R3×I0 | bipolar × voltage | 0.5516 | −0.0616 |
| R4×I0 | shaftLap × voltage | 0.5498 | −0.0634 |

Decision rule 2 → freeze upstream parity (R4×I2). Detail: `reports/neuroprobe_stage0_l2_neuralset_2026_05_06/`.

**Tier-B (12 additional cells, gated on Tier-A) — FREEZE HOLDS 2026-05-10 at 24-cell exhaustive**

Tier-B fires if Tier-A's R3 vs R4 < 0.005 ahead, or HG envelope wins, or any Tier-A loser comes from a feature class not covered.

| Cell | Recipe | Mean | Note |
|---|---|---|---|
| R4×I2L | shaftLap × log-STFT | 0.6150 | tied baseline (+0.0018, below threshold) |
| R2×I2 | shaft-CAR × stft_abs | 0.6138 | tied baseline |
| R1×I2 | global-CAR × stft_abs | 0.6111 | top-5 I2-family |
| R4×I3W | shaftLap × wide HG 70-250 Hz | 0.5924 | +0.006 over I3 standard, modest |
| R4×I4 | shaftLap × multi-band log-power | 0.5793 | **spectral richness ≠ better; multi-band/wavelet add dim without signal** |
| R4×I5 | shaftLap × wavelet (Morlet, 6 scales) | 0.5766 | underperforms stft_abs |
| R4×I1 | shaftLap × low-LFP <30 Hz | 0.5243 | **sub-delta/theta/alpha not speech-relevant; don't waste FM channels here** |
| R4×I6 | shaftLap × theta-band phase | **0.5004** | **chance — phase-only features dead for v14 unless paired with magnitude** |

Detail: `reports/neuroprobe_stage0_l2_exhaustive_2026_05_09/`.

**Tier-C (CrossSubject parity, gated on L.2 + L.1 freeze) — IN-FLIGHT**

| Cell | Recipe | Status |
|---|---|---|
| C.1 | N1 + R4×I2 at CrossSubject | in-flight (11 BT-Lite sessions, sub2 excluded) |
| C.2 | N1 + R3×I2 (bipolar) at CrossSubject | **KNOWN ISSUE → wrapper fix landed 2026-05-10** (commit `e62cc12`): upstream `combine_regions()` IndexError when bipolar virtual channel counts differ across subjects. DCC re-run in flight. |

**L.2 seed43/seed44 reruns** (24 jobs total): seed-variance check on L.2 winner R4×I2. Status 2026-05-10: 14/24 done; 10 OOM jobs re-dispatched at 64G. Cells whose ΔAUROC < 1× seed-variance SEM are not load-bearing regardless of CI.

**Headline**: view marginal Δ = 0.055 swamps reference marginal Δ = 0.012 by 4.5×. **View matters; reference does not (for linear — may matter once non-linear backbone is in place).** Spectral pretokenizer mandatory. Linear-on-raw floor: 0.55 ± 0.005 across 3 I0 cells.

### L.3 — Filtering + bad-channel sweep

| Cell | Recipe | Status |
|---|---|---|
| L.3.F0 | none | dispatched 2026-05-10 (commit `86409a8`) |
| L.3.F1 | 60 Hz notch + harmonics | in-flight |
| L.3.F2 | F1 + 0.5 Hz HPF | in-flight |
| L.3.F3 | F1 + 1 Hz HPF | in-flight |
| L.3.E0 | BT Lite mask only | planned |
| L.3.E1 | E0 + flatline + amplitude-outlier + clipping | planned |

Gated on L.1 N1 + L.2 R4×I2 winners.

### L.4 — Window anchor robustness

| Cell | Window | Status |
|---|---|---|
| L.4.W0 | [0, 1] s (D.0 default) | in-flight (24 jobs, 15/24 done 2026-05-10) |
| L.4.W1 | [−0.375, +0.625] s | in-flight |
| L.4.W2 | [−0.125, +0.875] s | planned |
| L.4.W3 | [+0.125, +1.125] s | planned |
| L.4.W4 | [0, 2] s | planned |
| L.4.W5 | [0, 0.5] s | planned |

Output is a robustness curve, not a single winner. Decides whether anchor-robustness ablation is mandatory or optional for Stage-1. Tag-along: L.4 norm × view × ref interaction (8 cells × 12 sessions) — **VERDICT 2026-05-10: max |residual| = 0.0008, greedy hill-climb safe**.

### L.5 — Diagnostic probes (kill-criteria gates)

Run on each sweep winner. Kill criteria use raw thresholds (no MC correction) — they are pre-registered conservative floors, not exploratory tests.

| Cell | Probe | Threshold | Status |
|---|---|---|---|
| L.5.P1 | subject-id from features | KILL if held-out AUROC > 0.95 | planned (V0-V6 found 0.699 on raw monopolar → corroborates need) |
| L.5.P2 | session-id from features | KILL if held-out AUROC > 0.95 | planned (V0-V6: 0.518 on raw) |
| L.5.P3 | reference-id (R0 vs winner) | positive sanity: AUROC ≈ 1 | planned |
| L.5.P4 | pre-stim window [−1, 0] s | flag if > chance + 0.05 | planned |
| L.5.P5 | shifted window [+5, +6] s | flag if > chance + 0.05 | planned |
| L.5.P6 | channel-shuffled-per-subject | KILL if cross-subject task acc doesn't drop substantially | planned |
| L.5.P7 | movie-time / block-order | flag if > chance + 0.10 | planned |
| L.5.P8 | 60 Hz residual power post-notch | flag if median residual > floor + 6 dB | planned (post-L.3) |
| L.5.P9 | acoustic/FM-leakage retrieval: brain → (env + f0 + Whisper-L8) | **v14-load-bearing**: contrastive must beat retrieval@10 by ≥ 5 pts + ≥ 0.05 R² on L8 | **blocked** (shares Whisper cache with L.7) |
| L.5.P10 | per-band identity leakage (multi-band L log-power) | diagnostic only; identifies leak-band | conditional (only if multi_band becomes v14 tokenizer candidate) |
| L.5.P11 | feature-permutation null | KILL if AUROC > empirical-chance + 0.02 (3 seeds) | planned |
| L.5.P12 | split-membership null | KILL (WithinSession only); collapses to P1/P2 elsewhere | planned |
| L.5.P13 | post-aggregation identity (DK-mean-pooled features) | **v14-load-bearing**: tests parcel-anchoring premise; soft flag at AUROC > 0.85, soft positive at < 0.70 | planned |

**L.5.P1+P2 — dispatched 2026-05-10** on L.2 winner view (commit `8e11d2f`).

### L.6 — Deferred Tier-2 (post-Stage-0 close)

| Cell | Tests |
|---|---|
| L.6.ES | electrode-set robustness: Lite-120 vs random-120×3 vs anatomy-120 vs Full uncapped |
| L.6.NR | feature-level nuisance regression: per-trial channel-mean / top-k PCs / subject-mean / ComBat / CORAL |
| L.6.WL | window length × sub-windowing |
| L.6.CB | class-balance schemes |
| L.6.FA | feature-aggregation schemes |

### L.7 — Audio-FM upper bound (Conwell veRSA control)

| Cell | Pipeline | Status |
|---|---|---|
| L.7.A0 | stim audio → frozen Whisper-large-v3 L8 → mean-pool → LogReg → label | **BLOCKED** (no BT movie audio on DCC, only mel/RMS/pitch features) |
| L.7.A1 | same as A0 but L16 (~50% depth) | blocked (layer-depth control) |
| L.7.A2 | same as A0 but HuBERT-large L6 | blocked (FM-identity control) |

**v14-load-bearing** for paper framing — establishes a "no-brain" upper bound v14 must clear by ≥ 0.05. Blocker memo: `memory/project_l7_audio_fm_blocked_audio_source_2026_05_10.md`. Two paths: source movies externally OR ask upstream for cached Whisper features. **Shares cache with L.5.P9** — extract once, both consume.

### A — Atlas/surface-mapping gates (blocker before D-block + Stage-2 JEPA mask)

Anatomy-dependent path. Run only after BT surface mapping resolves (per-electrode fsaverage hemisphere + vertex index + surface RAS table, or explicit Destrieux fallback approval). Christopher Wang followup pinged 2026-05-08. Canonical: `docs/neuroprobe/stage_0.md §5`.

A0-A4 verify the **internal correctness** of the surface-route fsaverage-bake pipeline. A5-A6 (added 2026-05-11) verify **cross-route consistency** between the surface-route BNA assignment and the (post-Chris-MNI) volumetric MNI → BNA route, both against BT's shipped DK labels. Canonical: `memory/project_v14_mni_bna_parity_gate_2026_05_11.md`. **A5-A6 are P0 blockers before Stage-1 dispatch on BT** — wrong `support[i,p]` collapses the atlas-anchored thesis silently.

| Cell | Gate |
|---|---|
| **A0** | derive BT Tier-1 BNA parcel list from Lite electrode surface positions. Gate: parcel ids ∈ 1..246; cardinality + LH/RH split recorded; cohort coverage ≥ 99%. |
| **A1** | verify fsaverage mesh identity + electrode snap distances. Gate: 163842 vertices/hemisphere; mean snap < 0.5 mm; max < 2 mm. |
| **A2** | snapped Destrieux labels vs BT region labels. Gate: ≥ 95% exact match, no hemisphere-clustered failures. |
| **A3** | verify BNA fsaverage bake at Lite electrode vertices. Gate: overall argmax match ≥ 90%; every BT Tier-1 parcel Dice ≥ 0.85; no Tier-1 parcel < 80%. |
| **A4** | BT Lite parcel co-coverage graph. **Blocker for any Stage-1/2 claim v14 can learn cross-parcel completion**: direct A-C structure learnable only when parcels co-occur; indirect A-B-C plausible only through connected overlap paths. Stage-2 JEPA contract must distinguish covered / intentionally-masked / uncovered parcels before training. |
| **A5 (P0)** | **MNI ↔ BNA parity gate — surface route**. Three sub-gates, runnable now without Chris MNI: **(a)** surface-route BNA(electrode) lobe == BT DK(electrode) lobe, ≥ 95% pooled / ≥ 85% per subject; **(b)** surface-route BNA(electrode) ∈ BNA_subset_of(BT DK(electrode)) via fixed DK ↔ BNA Tier-1 crosswalk table, ≥ 80% pooled; **(c)** per-subject drift: no subject < 70% on (a) or (b). Failing subjects flagged for BNA-anchored exclusion. Output: `reports/neuroprobe_parity_check_<date>/`. **Prerequisites**: build `data/atlas/dk_bna_tier1_crosswalk.csv` (independent of Chris). |
| **A6 (P0, post-Chris-MNI)** | **MNI ↔ BNA parity gate — volumetric route + cross-route**. Adds **(d)** volumetric BNA(MNI(electrode)) ↔ BT DK lobe parity, ≥ 95%; **(e)** volumetric BNA ↔ BT DK gyrus crosswalk, ≥ 80%; **(f)** surface ↔ volumetric cross-route agreement, ≥ 85% pooled / ≥ 75% at gyral-boundary electrodes; **(g)** per-subject drift on (d)/(e)/(f). Gates the §6b continuous-MNI-Fourier-PE P1 cell. Failure paths: A passes / C fails → MNI registration broken (try ANTs nonlinear re-derivation); A fails / C passes → migrate v14 default to volumetric route; global failure → fall back to channel-level for Neuroprobe submission. |
| C | build BT BNA support cache (parcel-soft-support per electrode). |
| V7-V8 | plot surface geometry + parcel coverage. |

### D — Atlas/pooling cells (after surface-mapping)

Extends upstream linear baseline; gated on A0-A4. Canonical: `docs/neuroprobe/stage_0.md §"Block D After D.0"`.

**Always-run (subset; D.0 cells run now without surface mapping)**:

| Cell | Eval | Prep | Atlas/pooling | Purpose |
|---|---|---|---|---|
| **D.0a** | CrossSubject binary | Lap + STFT | upstream public | reproduce upstream cross-subject binary baseline |
| **D.0b** | CrossSession multiclass | Lap + STFT | upstream public | reproduce upstream cross-session multiclass baseline (= N1 = L.1.N1 byte-equivalent) |
| **D.public** | CrossSession multiclass | Lap + STFT | DK hard-mean (upstream `combine_regions()`) | **upstream cross-subject baseline ARCHITECTURE itself — not a control** |
| **D.1a** | CrossSession multiclass | Lap + STFT | BNA Tier-1 hard mean | BNA hard vs DK baseline |
| **D.1b** | CrossSession multiclass | Lap + STFT | BNA Tier-1 soft support | **v14 novelty**: soft vs hard support |
| D.2 | CrossSession multiclass | CAR + HG | DK mean | prep-only control |
| D.3a | CrossSession multiclass | CAR + HG | BNA hard mean | HG + BNA hard |
| D.3b | CrossSession multiclass | CAR + HG | BNA soft support | HG + BNA soft |
| D.5 | CrossSession multiclass | CAR + HG | old PS LH-only parcels | anti-control |
| D.8 | CrossSession multiclass | Lap + STFT | zero-fill missing parcels | tests always-include parcel commitment |
| D.10 | CrossSession multiclass | engineered composite | BNA soft support | Better-Linear candidate |
| D.11 | CrossSession multiclass | raw 2048 Hz | BNA soft support | raw-view ceiling + artifact-risk baseline (no longer default-input coronation test) |
| D.12 | CrossSession multiclass | Laplacian raw | BNA soft support | raw re-reference check |
| D.13 | CrossSession multiclass | Laplacian/local + HG/HFA | BNA soft support | biologically privileged local population-firing view |

**Conditional**:

| Cell | Fires if |
|---|---|
| D.4 | Tier-1 looks too narrow |
| D.6 / D.7 | D.10 strong enough to warrant attribution |
| D.9 | WM rejection approved as free label/filter |
| **D.14** | pooled multi-source CrossSubject multiclass — train all allowed source subjects/sessions, test held-out. **Scientific generalization default.** Implement locally if upstream lacks it. |
| **D.15** | upstream all-source CrossSubject robustness if Christopher's newer `include_all_train_subjects=True` lands. Record pairwise vs pooled. |
| **D.16** | electrode-set robustness: Lite-120 parity / random-120 / anatomy-120 / Full uncapped. **Lite-120 for leaderboard parity only.** |

---

## 3. Stage-1 entry — Architectural ablation roster

Pre-committed before Stage-1 dispatch. Each cell reports pooled multi-source CrossSubject + S2/trial-4 parity + per-task breakdown. Canonical: `docs/neuroprobe/stage_0.md §"Stage-1 Entry Pre-Commitments"`.

| Cell | Tests | Hypothesis prior | Status |
|---|---|---|---|
| **AC1 FM-swap** | v14 with frozen Whisper-L8 swapped to HuBERT-L9 / WavLM-L9 / EnCodec / w2v-BERT-2-mid at fixed v14 arch | Conwell-2024 "diet > arch" test for our FM choice; if FM identity dominates, v14's contribution = FM-selection not arch | planned |
| **AC2 frozen-features linear probe** | Whisper-L8 → linear → labels, no brain features | must be beaten by brain↔FM contrastive; if linear-on-FM matches v14, brain contribution decorative; **identical pipeline to L.7.A0 — L.7 IS the AC2 baseline** | blocked-on-L.7 |
| **AC3 anatomy-blind random Perceiver** | Same Perceiver IO architecture + budget, but `parcel_latents` random-init (no `P_emb[p]` BNA prior) AND no `log(support[i,p])` cross-attn bias | Bhattacharjee 2024 SRM PCA-control analog; tests whether anatomy-as-routing does work vs random latents | planned |
| **AC4 P_emb drift** | Unfreeze `P_emb[p]` (BNA-init, learnable); keep `support[i,p]` fixed and `log(support)` cross-attn active | Cogan's functional-vs-anatomical-alignment question; triangulates with AC3 (full v14 vs routing-only vs neither); free interpretability story either way. Spec: `memory/archive/project_v14_p_emb_drift_ablation_2026_05_09.md` | planned |
| **AC5 post-hoc SRM baseline** | k=5 SRM on raw HFB → linear → labels (Bhattacharjee 2025 Nat Comp Sci precedent, +37% on 8-pt ECoG) | stronger baseline than anatomy-blind random Perceiver (matched-k orthogonal projection + cross-subject alignment). v14 must beat. | planned |

**AC6 shaft/depth feature matrix (sEEG-specific, gated on Stage-0 §3 shaft/depth geometry contract)**: 3-cell matrix before promoting depth as default — `hard_public_bt_label` only / shaft+depth only / hard label + shaft+depth. Tests whether within-shaft depth/position features add cross-subject signal beyond anatomy alone, and whether raw `shaft_id` slips through as a subject-identity shortcut (paired with subject-ID nuisance probe). Canonical: `docs/neuroprobe/stage_0.md §3`.

**Optional cell (only after the 6 above land)**: inter-parcel orthogonality regularizer on `P_emb[p]` — SRM-borrowed `λ_ortho · ||P_emb @ P_embᵀ − I||_F²`. Empirical question whether orthogonality matters given anatomy already separates parcels via `log(support[i,p])` bias.

### Stage-1 cold-start architecture cells

Canonical: `docs/neuroprobe/stage_1.md §"Frozen design commitments"` and `docs/neuroprobe/stage_2.md §"(a) pe2d → Perceiver IO"`. These run under the Stage-1 split contract: CrossSession multiclass submit lane, pooled CrossSubject multiclass scientific lane, S2/trial-4 parity lane, and threshold/regression robustness.

| Family | Cell(s) | Status / decision rule |
|---|---|---|
| Temporal tokenizer | default single-layer 1D patch conv; **A** BaRISTA 5-layer dilated 1D CNN; **B** HG envelope @ 200 Hz + linear patch; **C** Daubechies-4 wavelet + linear patch; **D** optional single strided 1D conv k4/s4 | mandatory Stage-1 tokenizer ablation set |
| Anatomy enforcement | hard-mask; **Z** `log(support + eps)` softmax bias; **L** learned linear initialized at softmax(log support); no-constraint baseline | mandatory 4-cell sweep. No-constraint ≥ Z falsifies the anatomy-routing claim. |
| Perceiver variant | default Perceiver IO bottleneck; **H** within-parcel electrode self-attn then Perceiver; **concat** joint self-attn over electrodes + parcel register tokens; **X** additive electrode embedding; **Y** X+Z; **unstructured** Charmander-style unstructured latents | isolates parcel bottleneck, support-bias form, and anatomical tagging |
| Latent multiplicity | `M ∈ {2, 4, 8}` | Stage-1/2 capacity and within-parcel expressivity sweep |
| Readout | task-attention pool default; mean-pool diagnostic; layer L ∈ {final, mid, mid-1} | mid > final by ≥0.005 promotes intermediate-layer readout |
| Attention stability | QK RMSNorm default; `sink-on` vs `sink-off` learned per-head null/sink option | promote sink only if AUROC or nuisance probes improve without sparse-subject regression |
| BNA connectivity prior | no BNA-conn-init default; BNA-conn-init additive latent self-attn bias | ablation only; cleaner no-baked-prior story wins if no lift |
| Width × depth | `d ∈ {64, 128}` × depth `{4, 6, 8, 12}` | default d=64/depth=6; run if cold-start sits below 0.611 or capacity is suspect |
| Hierarchy | flat per-parcel pool default; hierarchical atlas readout as +tier ablation | retained to prove PS-era hierarchy is not load-bearing on BT sEEG |
| Support source | argmax-hard vs probabilistic support | inherited from Stage-0 D.3a/D.3b once BNA surface mapping clears |

### Stage-1 optimization, view, and control cells

| Family | Cell(s) | Status / decision rule |
|---|---|---|
| Optimizer | AdamW+cosine verification baseline; Schedule-Free AdamW; Muon hybrid for matrix params only | Schedule-Free promotes if ≥0.005 or tie; Muon promotes on AUROC or wall-clock/step efficiency without nuisance lift |
| Precision / kernel | bf16 default; FlashAttention-3 default where hardware supports it; 4 ViT register tokens alongside parcel latents | engineering pins, not architecture-selection cells |
| Input view | raw voltage; local/Lap raw; local/Lap STFT/log-power; local/Lap HG/HFA; CAR+HG | v14 must tag non-default views; raw stays default only if it ties/wins without artifact-probe cost |
| Normalization | train-set/global; session/recording; window-local anti-control; robust train/session | L.1 freezes Stage-1 default at train-set fixed; window-local cannot promote unless it wins without losing amplitude-sensitive tasks |
| Re-reference | none/raw; per-probe/shaft; whole-brain CAR; Laplacian optional | per-probe wins by ≥0.005 → adopt; ≥0.02 → architectural finding |
| Temporal scale | 1 s eval-shaped crop; 3 s Stage-2 compatibility; 5 s anti-control | 5 s must improve 1 s probes and not raise nuisance decodability |
| Anchor robustness | in-band starts about `[-0.375, +0.125]` around word onset; out-of-band shifted controls | Stage-0 L.4 decides mandatory vs optional, but Stage-1 serious cells must report it when required |
| Anti-controls | shifted-window `[-1.5, -0.5]`; within-session label shuffle; subject-ID nuisance; stimulus-overlap flag | mandatory for any freeze or default promotion |
| Artifact probes | subject/session/reference/input-view/coverage/timing/line-common-mode probes from frozen backbone/readout | any task lift with higher nuisance decodability fails promotion |

---

## 4. Stage-2 — Loss-component triangulation (the 2×2 / actually 4-cell)

The "no intrinsic recon / no joint / no brain↔FM" set. Triggered by "why don't we just do pure brain↔FM — why intrinsic recon at all?" All four run on top of joint-from-step-1, BT-only diet, full v14 architecture. Canonical: `memory/project_v14_paper_corrections_post_newpapers6_batch2_2026_05_09.md §"Stage-2 SSL loss-component ablation triangulation"`.

| Cell | Loss | What it tests |
|---|---|---|
| **Default** | `L_recon (JEPA) + L_DSigLIP + 0.1·L_KoLeo` | reference recipe |
| **No-JEPA** | `L_DSigLIP + 0.1·L_KoLeo` | whether intrinsic masked-prediction contributes when paired data fully covers the diet — King-lab / Evanson-style pure cross-modal on corrected (zero-per-subject) architecture |
| **No-DSigLIP** | `L_recon + 0.1·L_KoLeo` | whether cross-modal alignment contributes — Charmander-equivalent intrinsic-only on v14 arch |
| **No-KoLeo** | `L_recon + L_DSigLIP` | whether KoLeo readout-uniformity is load-bearing — existing DINOv2-style cell |

Stage-2 loss terms from `stage_2.md` that are not removed by the 4-cell triangulation:

| Term / cell | Role | Promotion / trigger |
|---|---|---|
| `L_view_invariance` | align high-level parcel/global content across valid measurement views while preserving content-bearing amplitude/power | default Stage-2 invariance pressure; destructive augmentations are rejected |
| `L_nuisance_suppression` | gradient reversal / HSIC / MMD / CORAL-style penalty against subject/session/reference/coverage shortcuts | off by default; promote only if nuisance probes stay high after augmentation/sampling fixes |
| Coverage-aware JEPA mask | loss only on `masked_covered` parcels; no fake targets for `uncovered` parcels | mandatory for any completion/imputation claim |
| Optional parcel-id contrastive | same-movie/same-time cross-subject parcel positive; other parcels negative | off by default; use only where parcel is covered in both subjects |

**Hypothesis priors**:
- Default ≥ No-JEPA on Neuroprobe non-speech tasks (Onset/Pitch/Volume — parcels not aligned to Whisper-L8 need intrinsic signal).
- Default ≥ No-DSigLIP on stimulus-aligned tasks (Word/Sentence/Speaker — D-SigLIP gives explicit semantic anchor).
- KoLeo's main role is collapse-prevention — sister-run check pattern.
- No-JEPA ≈ Default on BT-only (paired-only diet) but No-JEPA ≪ Default on Tier-0 mixed paired/unpaired (`1{paired}=0` zeros out the only signal).

**Triangulation outcomes**:
- Default ≥ both No-cells → joint recipe justified.
- No-JEPA ≥ Default on BT-only, Default ≥ No-JEPA on Tier-0 → JEPA's value is unpaired-data utilization.
- No-DSigLIP ≥ Default → cross-modal anchor overrated; falls back to PopT-style intrinsic-only.

**Bake all four into Experiment 1 from day one** (analogous to sister-run leakage defense). Cheap (~4 short pretraining runs at Experiment-1 budget); reviewer-defensible. Conwell-2024 "diet > arch" lesson applies: loss FORM is more diagnostic than architectural sweeps.

### Stage-2 core attribution tracks

Canonical: `docs/neuroprobe/stage_2.md §"Mandatory attribution comparison"`. These are not optional polish; a positive Stage-2 result is not interpretable without them.

| Track | Cell(s) | Question |
|---|---|---|
| Track 1 — architecture | `v14` Perceiver IO bottleneck vs `v14-concat` joint self-attn with parcel-CLS register tokens | does the parcel bottleneck itself do work, or are parcel tokens just readout positions? |
| Track 2A — input-space recon | Charmander-style 50% channel mask + raw-voltage MSE | strongest published iEEG-scale input-reconstruction family; stress test for artifact learning |
| Track 2B — predict-next contrastive | MVPFormer-style next-segment InfoNCE | covers temporal contrastive SSL distinct from reconstruction |
| Track 2C — discriminative | PopT-style channel-wise + ensemble-wise discrimination | direct lineage control |
| Track 2D — latent prediction / JEPA | BaRISTA-style masked latent prediction with target encoder / EMA | closest intrinsic-SSL family to the default |
| Track 2E — autoregressive generative | Brant-style next-token CE on quantized vocabulary | closes the "loss is discriminative, not generative" reviewer attack |
| BrainStratify DPQ control | DPQ/VQ code prediction with train-only channel/parcel selection isolated from DPQ/Fine lift | stress test for within-parcel sub-slot expressivity and subject-specific selection |
| Track 2a — Tier-A recon recipe | JEPA-style latent prediction vs data2vec 2.0 contextualized targets vs Charmander-MSE | confirms the Tier-A latent-target default before broader Stage-2 scaling |
| Track 2b — cross-modal marginal lift | full joint recipe vs `L_DSigLIP` masked off | cross-modal must add ≥0.03 for the multi-FM paper claim to survive |

**Read-out**: v14 must beat the strongest intrinsic family by ≥0.03 for the multi-FM claim. If `v14-concat` ties v14, reframe away from parcel-bottleneck enforcement. If an intrinsic family clears the gate and cross-modal does not, submit the intrinsic winner and demote multi-FM.

---

## 5. Stage-2 — Schedule ablations

Canonical: `memory/project_v14_stage2_schedule_audit_2026_05_09.md` + `memory/project_stage2_ssl_initial_diet_bt_only_joint_step1_2026_05_09.md`.

| Cell | Schedule | Status |
|---|---|---|
| **Default — joint-from-step-1 + λ_DSigLIP warmup** | `L = L_recon + 1{paired}·L_DSigLIP + 0.1·L_KoLeo` from step 1; λ_DSigLIP linear warmup 0 → 1.0 over first 5-10% of steps | **PROMOTED 2026-05-09** (DINOv2 + REPA + DINOv3 precedent; KoLeo on micro-batches of 16) |
| Sequential 2a → 2b sister run | recon-only 2a then add D-SigLIP 2b | sister-run baked from day one (off-ramp; matches Podcast leakage-defense playbook) |
| Late-add intrinsic | cross-modal-only first 80%, then add intrinsic | new cell, cited to SigLIP-2 honestly (opposite of original miscitation) |
| Curriculum-warmup (recon-first 50-80% then add cross-modal) | — | **REMOVED** as candidate — no precedent (was attributed to SigLIP-2 but SigLIP-2 is the opposite pattern) |

**Reconciliation note.** `stage_2.md` still preserves older 2026-05-01 wording around SigLIP-2-style recon-first staged training, SALT, and VideoPrism-direction as a three-way establishment comparison. The newer 2026-05-09 schedule memos promote joint-from-step-1 + `λ_DSigLIP` warmup. Until `stage_2.md` is rewritten, keep the older cells below in the menu so no establishment comparison is lost.

### Stage-2 establishment cells

Canonical: `docs/neuroprobe/stage_2.md §"Sub-stages"` and `docs/neuroprobe/stage_2.md §"SSL recipe ablation cells"`.

| Cell | Description | Adoption / role |
|---|---|---|
| EMA JEPA / SigLIP-style staged default | data2vec 2.0 + V-JEPA 2.1 latent prediction with paired-aware D-SigLIP addition | baseline establishment recipe preserved from `stage_2.md` |
| SALT static-teacher | frozen controlled-capacity teacher latents replace EMA teacher inside `L_recon` | required head-to-head; promotes on ≥0.005 lift OR loss→downstream R² ≥ 0.5 without sparse-subject regression |
| VideoPrism cross-modal-first | D-SigLIP brain↔Whisper-L8 teacher first, then masked prediction to frozen teacher local+global latents | required head-to-head; promotes on ≥0.005 lift without sparse-subject regression |
| Sequential 2a→2b | intrinsic SSL first, cross-modal second | ablation/off-ramp if joint or staged default underperforms |
| REPA intermediate-layer alignment | cosine alignment from brain layer `depth/3` to Whisper-L8 during Phase 1 | Tier-1, highest expected performance lift; composes with SALT |
| REPA-E end-to-end | REPA as only stabilizer, replacing `L_recon` | Tier-1 test of whether all Stage-2a budget should be alignment |
| Multi-teacher MSE distillation | Whisper-L8 at depth/3 + EMA/self motion analog near depth-1 via MSE | Tier-1 third frozen-teacher pattern beyond SALT/VideoPrism/REPA |
| WhisperBCI per-subject embedder | v14 zero-per-subject vs day-specific low-rank + subject-specific input embedders | head-to-head for "zero-param cross-subject + Whisper-intermediate" novelty hook |

### Stage-2 SSL recipe cells

| Family | Cell(s) | Role |
|---|---|---|
| Dense contextual loss | dense-ctx-loss; distance-weighted `λ_i = λ / sqrt(d_min)` with warmup | V-JEPA 2.1 update to default; fixed-λ is the anti-control |
| Deep supervision | deep-SSL / DSS at intermediate layers + final | paired with intermediate-layer readout; if DSS wins, final-layer readout may be enough |
| Masking | dual-mask; random MIM ratio; BEVT block masks `{Tube-0.5, BEVT-0.5, BEVT-0.65, BEVT-0.75}` | BEVT-0.65 is sub-default candidate; cheap Tier-1 sweep |
| Frequency curriculum | FastDINOv2-style bandpass schedule `1-8 Hz → 1-80 Hz → 1-500 Hz` | Tier-1 efficiency/stability cell |
| Anti-collapse | **Default = EMA + StopGrad + KoLeo @ weight 0.1** (data2vec-2.0 / V-JEPA-2.1 / DINOv3 lineage; brain-FM precedent 5/5 — Brain-JEPA, EchoJEPA, EEG-DINO, REVE, plus the general SSL stack); SIGReg ablation cell (P2 defensive, see §6b) | early cell because collapse invalidates downstream reads; canonical decision in `memory/project_v14_post_eeg_dino_synthesis_2026_05_10.md` |
| Projection / distillation | shared projection head; similarity-distribution distillation + temperature `{0.5, 0.7, 1.0}` | tests iBOT/PE-Core full-distribution lessons |
| Register / sink stability | 4 ViT register tokens default; G global register tokens `{4, 8}`; attention-sink checks | avoids attention-sink artifacts and supports PopT-style analysis |
| Frozen-teacher targets | token shuffling; global+local distillation; local-only vs local+global | local-only can drop global if phoneme/motion tasks improve by ≥0.005 |
| Mask semantics | phase-specific masking + no unmasked-token alignment | default unless direct ablation overturns it |
| Decoder capacity | controlled small decoder; small decoder cap | prevents decoder-only reconstruction wins |
| Optimizer / precision / kernel | Schedule-Free AdamW; Muon matrix-param hybrid; bf16; FlashAttention-3 | Schedule-Free and bf16 are defaults/pins; Muon remains Tier-1; FA-3 is engineering pin |
| EMA mechanics | Busbridge κ scaling; p-EMA; BEMA; constant EMA | Tier-2 if EMA teacher survives; eliminated if SALT wins |
| Context curriculum | high-res / long-context cool-down phase; RoPE-box jittering | tests variable-window robustness and late long-context benefit |
| Long-context attention | NSA sliding-window attention | only after short-window backbone passes task/nuisance gates and context expands |
| Multi-window tokenization | modality/window-type learnable embedding + dual conv tokenizers | prevents mixed 1 s / 3 s / continuous chunks from sharing a biased tokenizer contract |
| Evaluation layer | layerwise frozen probes; alignment-after-core short phase; dense-probing-as-eval | mandatory for serious checkpoints before declaring success/failure |
| Analysis cells | straightening probe; embedding-cluster recovery | retroactive interpretability, not selection |
| Long-run regularization | Gram anchoring | Tier-2 or triggered by P_emb drift / long runs |
| Specialist consolidation | OPD-inspired distribution distillation across complementary specialists | only after specialists exist; unified student must match/better best specialist without nuisance lift |
| Submit-time distillation | V-JEPA 2.1 two-stage distill protocol if winning model exceeds ≤30M cap | keeps model under gate without inventing a new distillation recipe |
| Baseline | linear-FIR multi-FM baseline | mandatory; v14 must beat by ≥0.05 |

### Stage-2 corpus, scaling, and head cells

| Family | Cell(s) | Role |
|---|---|---|
| Source-family scaling | BT-only; BT + same-modality sEEG; BT + ECoG/iEEG; full Tier-0 under matched BT exposure | required to interpret multi-source gains |
| Banville 4-point curve | ~8h, ~16h, ~33h Tier-1, ~66h Tier-1 + D-cohort | slope < 0.04 AUROC/log-h means SSL is not earning its keep |
| Downstream transfer curve | pretrain budget × Neuroprobe finetune-size 2-D grid | fits scaling-law / alignment coefficient; prevents single-budget overread |
| Capacity | `d_model ∈ {32, 64, 128}` | Full Tier-1 sweep; winner becomes Stage-3 width |
| μP / μTransfer | tune LR + weight-decay once on the 13M proxy under Maximal Update Parametrization (Tensor Programs V, Yang/Hu, NeurIPS 2021); μTransfer copies HPs to 25M/40M | prerequisite for the 13M/25M/40M sizing sweep — keeps HP optima width-invariant so capacity is isolated from a mistuned LR (DIVER meta-review #3 confound, [[reference_diver1_ieeg_fm_2026_05_18]]); ~⅔ HP-search compute saved. Width transfer reliable, depth not — hold depth fixed across the sweep; verify with a coord-check |
| Distribution controls | AGD + distribution-balanced sampling; capacity-threshold check; effective seen-window export | prevents blaming objectives for exposure/capacity failures |
| Cross-subject batching | implicit random same-content vs active same-audio cross-subject batching | activate if 8h/16h slope is below prior band |
| Projection dimension | common D-SigLIP projection `k ∈ {256, 512}` | pin in first ablation |
| Normalization | per-session affine; per-(subject, parcel) z-score | conditional drift ablations only |
| Within-parcel expressivity | 3-D distance bias; shaft-aware bias | early Stage-1/2 ablation; promote on ≥0.005 cross-subject lift. **MNI Fourier PE moved to §6b "Continuous MNI Fourier PE head-to-head" P1 cell — gated on Chris MNI** |
| Probe mode | frozen vs fine-tuned heads | frozen is honesty standard; fine-tune is a curve, not sole evidence |
| Head regularization | linear L2; Group-L1 by parcel | default head design for CrossSession and pooled CrossSubject |

---

## 6. Stage-2 — Refinement cells (added 2026-05-10 from 5/10 papers sweep)

Canonical: archived `memory/archive/project_v14_ablation_cells_post_5_10_papers_2026_05_10.md` (content folded into this doc).

| Cell | Origin | Hypothesis | Slot | Priority |
|---|---|---|---|---|
| **Gram anchoring on parcel latents** | DINOv3 §4 (Meta Aug 2025) | regularize parcel-latent Gram matrix to prevent representation collapse; activates if P_emb-drift (AC4) shows instability | Stage-2 refinement | P3 (if P_emb-drift unstable) |
| **L_DSigLIP layer-sweep** (L4 / L8 / L16 / all-mix) | TRIBE v2 §5.2 (King lab Mar 2026, learned-mix all layers) | tests whether single-Whisper-L8 commitment costs anything vs learned multi-layer mix; v14 commits to L8 on independent grounds (Goldstein-2025-Nature + Vaidya-2022 + Antonello-2023 + Hong-2024 triangulation at ~25% depth in large models) — needs explicit defense | Stage-2 ablation | **P1 defensive** |
| **J1 — JEPA Level A mask-rate sweep** (added 2026-05-15) | Ben 2026-05-15 — `project_stage2_ssl_initial_diet_bt_only_joint_step1_2026_05_09` 5/15 addendum | Sweep `mask_rate ∈ {0.25, 0.30, 0.40, 0.50}` on JEPA Level A random electrode mask. **Default 0.30** (lower than Charmander's 0.50 because iEEG coverage is sparse — BT subjects have ~150 contacts and uneven per-parcel coverage means random 50% mask empties parcels entirely + BT-Lite is smaller-data than Charmander's AJILE12 pretrain corpus, less robust to aggressive masking). Settles mask-rate empirically: if 0.30 loses to 0.40 by ≥+0.005 → promote 0.40; if 0.30 beats all → ratify default. Curriculum option (25 → 50 ramp over first N epochs) and per-batch sampling `p ∼ U(0.25, 0.50)` preserved as schedule variants if static sweep is inconclusive. | Stage-2 hyperparameter sweep | **P1** |

---

## 6b. Stage-2 — Post-EEG-DINO architectural cells (added 2026-05-10)

Cells converged after reading EEG-DINO (MICCAI 2026) + consolidating 5/10-papers sweep on spatial PE / JEPA target / collapse prevention. Canonical synthesis: `memory/project_v14_post_eeg_dino_synthesis_2026_05_10.md`.

| Cell | Tests | Origin | Slot | Priority |
|---|---|---|---|---|
| **Continuous MNI Fourier PE head-to-head** | REVE/Laya-style continuous joint Fourier of true MNI on electrode-token side vs v14 default parcel-id-only spatial PE. Closes the question BaRISTA's confound prevented from being tested cleanly (factorized-affine-Talairach-PE ≠ joint-Fourier-true-MNI-PE) | REVE NeuralBench SOTA + BaRISTA confound resolution + Laya/Brain-JEPA/MV-BrainFM continuous-PE convergence | Electrode-token spatial PE | **P1, gated on Chris MNI confirmation AND §2 A6 parity gate passing, paper-load-bearing** |
| **JEPA target level: B vs C vs D** | Where JEPA masking lives in v14 stack. **B** = within-parcel temporal mask (per-parcel time-series prediction); **C** = cross-parcel mask at fixed time (the v14 spec'd default, mask ~50% of parcels, predict masked-parcel latents from visible-parcel latents); **D** = spatiotemporal block mask (V-JEPA video analog) | Brain-JEPA spatiotemporal masks; v14 Level C contract = mask-parcels-predict-from-visible-parcels via Perceiver IO cross-attn | Stage-2 SSL target choice | **P1 (mandatory before locking JEPA target)** |
| **Brain-JEPA Cross-ROI / Cross-Time / Double-Cross mask design** | 6× pretraining-efficiency multiplier from mask design alone (Brain-JEPA Fig 6: spatiotemporal multi-block at 50 epochs ≥ vanilla random multi-block at 300 epochs). Three target regions: Cross-ROI (α, same time different parcels), Cross-Time (β, same parcels different times), Double-Cross (γ, both) | Brain-JEPA NeurIPS 2024 §3.2-3.3 + Fig 6 | Stage-2 SSL mask design | **P1 (cheap efficiency cell)** |
| **Anatomically-realistic mask sampling vs uniform** | Sample JEPA training mask from empirical clinical-coverage distribution (lateral temporal / frontal / sensorimotor heavy; occipital / medial sparse) vs uniform random parcel mask. Train-inference mask-distribution match | This conversation (2026-05-10) | Stage-2 SSL mask sampling | P2 |
| **Per-parcel coverage diagnostic (post-pretraining)** | Real cross-subject transitivity vs pattern completion within coverage manifold. Measure per-parcel inference accuracy stratified by training-set coverage; if v14 predicts well only on high-coverage parcels, transitivity claim is pattern completion only; if low-coverage parcels also predict well (via A↔B↔C bridging through shared parcel-B latent), transitivity holds | This conversation (2026-05-10) | Free post-hoc, all runs | **P1 (free diagnostic, single empirical answer to A↔B↔C transitivity question)** |
| **Brain-JEPA-style functional gradient PE as post-hoc diagnostic** | After Stage-2a, compute diffusion-map functional gradient PE from learned attention map (analog to Brain-JEPA's Brain Gradient Positioning derived from rs-fMRI feature similarity), freeze, inject as additional latent-side PE. Tests whether v14 latent self-attention recovers the functional gradient on its own or whether baking it in adds. **Retroactive-validation thesis check** | Brain-JEPA NeurIPS 2024 + v14 retroactive-interpretability template | Stage-2a post-hoc, all runs | **P1 (free)** |
| **SIGReg vs EMA + StopGrad** | Whether LeJEPA's Sketched-Isotropic-Gaussian Regularizer replaces EMA target encoder cleanly on iEEG at v14 scale. Brain-FM precedent strongly favors EMA (Brain-JEPA, EchoJEPA, EEG-DINO, REVE, DINOv3, V-JEPA 2.1, data2vec 2.0 — 7/7 scaling papers); only Laya + LeWM use SIGReg, and Laya-B did not scale cleanly over Laya-S | LeWM (Maes Mar 2026) + Laya (UCLA 2026) | Stage-2 collapse prevention | P2 defensive |

**EMA + StopGrad = v14 default** (data2vec-2.0 + V-JEPA-2.1 lineage). EMA momentum linear 0.996 → 1.0 (Brain-JEPA pattern). EchoJEPA's mechanistic argument: EMA target encoder gradually denoises speckle/BOLD-noise — load-bearing for JEPA-beating-MAE on noisy biosignals.

---

## 6c. Stage-2 / 1 — NEWPAPERS7 cells (added 2026-05-12)

Cells from the 4-paper NEWPAPERS7 batch (Shapovalenko-Auster wav2vec/CLIP probe, MultiDiffNet, ASPEN, Evanson/King developmental sEEG). Canonical: `memory/reference_aspen_lee_2026.md`, `memory/reference_multidiffnet_zhang_2025.md`, `memory/reference_shapovalenko_auster_wav2vec_clip_2025.md`, `memory/reference_dascoli_brain2qwerty_gwilliams_evanson_dev_2025.md` (Evanson dev paper section).

| Cell | Tests | Origin | Slot | Priority |
|---|---|---|---|---|
| **AC-Multi-Layer-Whisper concat** | Concatenated dual-layer L_DSigLIP target: shallow Whisper-L8 (acoustic-phonetic per Goldstein-2025, ~25% depth in large-v3) AND deep Whisper-L20 or L24 (lexical proxy, ~62-75% depth) used jointly as contrastive targets. Distinct from §6 L_DSigLIP-layer-sweep (which swaps individual layers L4/L8/L16/all-mix); this cell **concatenates** shallow+deep in the Evanson 2025 wav2vec2 10%+80% pattern. Tests whether v14's lexical-feature decoding benefits from explicit deep-layer anchoring, ratifying Stage-3 multi-FM-extension as a real ablation. Evanson finding: deep-vs-shallow hierarchy gain ΔL R is **age-graded** (large in 6-11 and 12+, small in 2-5 yr) → adult BT cohort should benefit from deep-layer addition | Evanson/King 2025 (n=46 pediatric sEEG, wav2vec2 L10% + L80% concat) + Goldstein-2025 Whisper-L8 acoustic-phonetic anchor | Stage-2 SSL contrastive target | **P1** |
| **AC-Dual-Stream temporal + spectral** | Add parallel temporal voltage stream alongside v14's spectral (stft_abs/log-STFT) input. Fuse at latent level via multiplicative gate (Hadamard product after learned projections) à la ASPEN. Motivated by ASPEN's SPEN-fails-on-MI asymmetry: spectral-magnitude alone discards precise temporal dynamics critical for time-derivative tasks. **Gated on observation**: triggered only if v14 underperforms on Neuroprobe time-derivative tasks (onset, delta_volume, delta_pitch). L.2 already showed temporal cells (I0) at 0.55 vs spectral 0.61 on linear readout — temporal is weaker for static feature extraction, but may add at Perceiver-IO scale where the encoder can do non-linear feature combination | ASPEN (Lee 2026) multiplicative fusion + SPEN-MI failure mode (76.27%→53.50% drop when spectral-only) | Stage-1 input pretokenizer | **P2 (conditional)** |

**Note on dual-stream as anatomy analog**: ASPEN's multiplicative fusion = "cross-modal AND gate" (feature propagates only if both spectral AND temporal projections agree). v14's Graphormer `log(support[i,p])` bias already encodes this *in the anatomy dimension* (feature propagates only if anatomical support AND learned activation agree). Dual-stream would add the *signal-view dimension* of the same principle. Worth noting in paper-framing as conceptual lineage.

---

## 6d. Stage-1 — Post-Charmander architectural ablations (added 2026-05-15)

Cells from the 2026-05-15 SOTA-EEG-FM synthesis session (NeuralBench REVE / Liu CBraMod / ZUNA / Charmander deep-read / MTDP). Locks four architectural deltas (D1-D4) + one scope deferral (L1) per `memory/project_v14_first_pass_simplification_2026_05_15.md`. The single architectural difference between Charmander and v14 is anatomy-anchored vs abstract latents — V1 directly tests whether the parcel-anchored bottleneck does real work or whether anatomy-as-PE in a vanilla raster transformer suffices.

**2026-05-15 promotion + temporal-first re-flip (same day)**: V9 (per-channel temporal IN ADDITION to within-shaft spatial) **promoted to v14 first-pass DEFAULT** based on (a) phoneme decoding is fundamentally temporal — plosive bursts/formants/HG bursts are 50–200ms within-channel events that per-channel temporal extracts directly, (b) universality across EEG/iEEG FMs (BIOT/LaBraM/BrainBERT/BrainWave converge on per-channel temporal pre-processing), (c) STFT bins are independent and don't encode cross-time evolution by themselves, (d) cost negligible (+150K params, +150K ops, well under 10% of 1.6M total), (e) orthogonal to within-shaft spatial — different attention surfaces, non-overlapping jobs.

**Order re-flipped same day to TEMPORAL-FIRST → SPATIAL-NEXT** (default order). Initial promotion went spatial-first as a chronological accident (extending the original D2 spatial block as "first"); same-day re-flip per signal-processing principle (each electrode is an independent measurement → extract per-channel features in atomic-measurement domain first, integrate spatially second), EEG/iEEG FM literature convention (BIOT/LaBraM/BrainBERT/BrainWave all temporal-first), information-flow argument (spatial-first contaminates temporal signal with shaft-neighbor mixing; temporal-first contamination of spatial signal is less harmful because spatial integration is aggregation over already-extracted features), and shaftCAR-denoising-already-upstream-of-both-blocks (which removes the spatial-first-to-denoise rationale).

V8 → "drop spatial axis" (per-channel temporal only); V9 → "drop temporal axis" (within-shaft spatial only = original D2 single-block default); V10 → reverse-order sequencing test (= the chronologically-first promotion default before re-flip); V11 → "drop both blocks" (pure Perceiver baseline). V8/V9/V11 trio provides full causal attribution for the input-side self-attn stack.

| Cell | Tests | Origin | Slot | Priority |
|---|---|---|---|---|
| **V1 — Vanilla raster + parcel-id-PE + shaft-bias vs Perceiver IO + parcel-tagged latents** | Whether the parcel-anchored latent bottleneck does work beyond what parcel-id PE on inputs alone provides. Architecture: flatten (C×T_bin) tokens with parcel-id PE added (D4) + shaft-T5-bias self-attn + PMA readout. Vanilla cost O((C×T_bin)²) ≈ 9.4M ops/layer × L=6 (Flash Attention manageable); Perceiver O(K·M × C·T_bin + (K·M)²) ≈ 1M + 100K. Outcome shapes story: vanilla wins → "PE recipe + JEPA + Whisper-L8" thesis (Charmander+anatomy-PE); Perceiver wins → "anatomy-anchored bottleneck does inductive-bias work PE alone cannot." | Charmander 2025 (single-arch-delta-from-v14 framing); the only difference is latent identity | Stage-1 first ablation cell after Lite baseline lands | **P0** |
| **V2 — Shaft-T5-bias on / off** | Whether the input-side self-attn block with T5/Toeplitz shaft-bias improves over no input self-attn. Bias formula: `α · 1{same_shaft} + γ_h · emb[bucket(\|Δdepth\|)]`, per-head, **HARD `-inf` mask on cross-shaft pairs** (block-sparse partition; each shaft = its own attention world at this layer; cross-shaft anatomy is the parcel-routing layer's job). Single layer, same-time-bin only → ~41K ops. Run in whichever encoder won V1. | Shaft/depth contract admissible features: same-shaft binary + \|Δdepth\| | Stage-1 after V1 | **P1** |
| **V3 — Readout: PMA / single-query attention-pool vs CLS-token vs avg-pool** | Whether single-query class-attention readout (Set Transformer PMA k=1 / CaiT) recovers per-task selectivity over parcels. Avg-pool predicted to lose (collapses K*M=320 parcel-tagged latents to mean = D.public DK-hard-mean linear baseline at the readout step). CLS-token less natural for Perceiver IO (needs to be added as 321st latent). | Set Transformer (Lee 2019), CaiT (Touvron 2021) | Stage-1 alongside V1-V2 | **P1** |
| **V4 — Encoder depth sweep including Charmander L=16** | Charmander uses L=16 transformer self-attn layers in its 8M-param Perceiver. v14 default depth=6. Test L ∈ {4, 6, 8, 12, 16}. | Charmander Table A3 | Stage-1 sweep extension | **P1** |
| **V5 — Charmander-style heavy dropout 0.20/0.40/0.20** | FFN/Linear/Attention dropout rates. Charmander tunes high (0.40 on linear); v14 currently default. iEEG SNR is low; regularization may matter. | Charmander Table A3 | Stage-1 sweep | **P3** |
| **V6 — Lamb vs AdamW optimizer** | Charmander uses Lamb at lr=3.125e-4 (constant 150 ep + cosine 150 ep). Single comparison cell. | Charmander §B.5 | Stage-1 sweep | **P3** |
| **V7 — Concat vs add for parcel-id PE on input tokens** | D4 default = add (ML-standard, model can learn orthogonal subspaces). Charmander uses concat for `Concat(W_p · patch, c_j)` but didn't justify (inherited from Poyo+). Test concat as defensive ablation if interference suspected. | Ben 2026-05-15 simplification (D4) | Stage-1 P3 | **P3** |
| **V8 — Drop ❸b within-shaft spatial (per-channel temporal only)** | Whether spatial axis earns its keep on top of per-channel temporal. Removes ❸b spatial block from default. If default beats V8 by ≥+0.005 → spatial earns its keep; if V8 ties or beats default → demote spatial axis to ablation, revert default to per-channel temporal only. | Defensive ablation post-V9 promotion (Ben 2026-05-15) | Stage-1 P1 | **P1** |
| **V9 — PROMOTED TO DEFAULT 2026-05-15** (was: per-channel temporal IN ADDITION to within-shaft spatial) | Now the v14 first-pass architecture: **❸a per-channel temporal → ❸b within-shaft spatial** → parcel-id PE → cross-attn. Order re-flipped same day from spatial-first to temporal-first per signal-processing principle (extract per-channel features in atomic-measurement domain first, integrate spatially second), EEG/iEEG FM convention (BIOT/LaBraM/BrainBERT/BrainWave all temporal-first), and information-flow (spatial-first contaminates temporal signal with shaft-neighbor mixing; temporal-first contamination of spatial signal is less harmful). Tests whether per-channel temporal earns its keep on top of within-shaft spatial. Promotion rationale: phoneme decoding is fundamentally temporal, universality across EEG/iEEG FMs, STFT bins don't encode cross-time evolution by themselves, cost negligible (+150K params <10% of total). See `memory/project_v14_first_pass_simplification_2026_05_15.md` D2 revision. | Ben 2026-05-15 | **DEFAULT** | **n/a (default)** |
| **V9' — Drop ❸a per-channel temporal (within-shaft spatial only = original D2 single-block default)** | Whether per-channel temporal earns its keep on top of within-shaft spatial. Mirror of V8: V8 drops spatial, V9' drops temporal. If default beats V9' by ≥+0.005 → temporal earns its keep; if V9' ties or beats default → demote temporal axis, revert default to within-shaft spatial only (= original D2). | Defensive ablation post-V9 promotion + temporal-first re-flip (Ben 2026-05-15) | Stage-1 P1 | **P1** |
| **V10 — Reverse order: within-shaft spatial THEN per-channel temporal (= chronologically-first promotion default before re-flip)** | Tests sequencing sensitivity for the two-axis default. Reverse order = ❸b spatial first → ❸a temporal next. If V10 beats default by ≥+0.005 → re-flip the re-flip; if V10 ties default → sequencing is degenerate (pick whichever is cheaper to implement). | Defensive sequencing test against default order (= measure whether the re-flip matters) | Stage-1 P2 | **P2** |
| **V11 — Drop both ❸a and ❸b (pure Perceiver IO baseline, Charmander-style)** | Whether ANY input self-attn earns its keep on top of cross-attn at ❺ + latent self-attn at ❻. Removes the entire input-side self-attn stack; input tokens go directly into Perceiver cross-attn (Charmander/Poyo+ baseline). If V11 ties default → both blocks are over-engineering; if default wins ≥+0.01 → input self-attn earns its keep collectively. Pairs with V8 + V9' (which isolate the spatial and temporal blocks individually) for full causal attribution. | Charmander/Poyo+ pure-Perceiver baseline (Ben 2026-05-15) | Stage-1 P2 | **P2** |
| **V12 — ❸b mask intersection: same_shaft AND same_parcel (instead of same_shaft alone)** | Quantifies anatomy-smear cost of within-shaft spatial mixing across parcel boundaries. Default ❸b mask = `same_shaft AND same_time_bin`; V12 intersects with parcel-coherence: mask = `same_shaft AND same_time_bin AND same_parcel`. Hard-restricts ❸b attention to within-shaft AND within-parcel pairs, eliminating cross-parcel smear directly. **One-line mask change**, bias formula unchanged, no extra params. Promotion rule: if V12 beats default by ≥+0.005 on CrossSession → promote (anatomy-smear was hurting; 5-stack mitigations of audit-locality + non-cortical exclusion + post-block parcel-id PE injection + cross-attn re-segregation + T5 |Δdepth| bias not adequate); if V12 ties → 5-stack mitigations adequate (architecture is already handling smear); if default wins → cross-parcel smear is net-useful for spatial integration. | Anatomy-smearing concern surfaced in 2026-05-15 follow-up; mitigation-stack is real but residual smear is empirically open | Stage-1 P1 | **P1** |
| **V13 — Replace ❸a per-channel temporal SA with depthwise Conv1d-over-time** | Tests whether ❸a's content-dependent global self-attn (T5 |Δt_bin| bias, hard cross-channel mask) earns its keep over a simpler depthwise-separable Conv1d (kernel K=5, ≈100ms receptive field, translation-invariant, content-independent). At T_bin≈50 + BT-Lite data scale, conv's strong local inductive bias may generalize better; SA with strongly-decreasing T5 bias structurally degenerates toward conv anyway (Cordonnier 2020 *On the relationship between self-attention and convolutional layers*). Compute parity: Conv1d K=5 ≈ 4.1M FLOPs/electrode vs SA ≈ 3.6M at d=128. **One-line swap**: `self.temporal = PerChannelSelfAttn(...)` → `self.temporal = DepthwiseConv1d(d=128, kernel=5, padding=2)`. **Sequencing**: GATED on V11 outcome — if V11 (drop both ❸a+❸b) ties default, input self-attn is dead and V13 is moot; if V11 loses, run V13. **Promotion rule**: V13 ≥ default + 0.005 → promote conv (Occam + better small-data generalization); V13 ties → swap for conv (simpler op for free); default beats V13 by ≥ 0.005 → SA's content-dependent global attention earns its keep. Conformer-style hybrid (Conv + SA branches summed) explicitly out-of-scope for this cell — paper-2 territory; v14 thesis is parcel-routing not temporal-front-end engineering. | Ben + Claude 2026-05-18 discussion: SA-vs-Conv1d at small-T + small-data; Conformer (Gulati 2020) precedent in speech; EEGNet conv-only at small scale | Stage-1 P1 (gated on V11) | **P1** |

**Implementation order (D2-sequencing fix, Ben 2026-05-15; updated for temporal-first re-flip)**: ❸a per-channel temporal self-attn on **signal-only tokens** (no parcel-id PE yet) → ❸b within-shaft spatial self-attn (still signal-only) → parcel-id PE injected fresh (D4 add) → cross-attn to parcel-tagged latents. If parcel-id PE were added BEFORE input self-attn blocks, within-shaft mixing would smear parcel labels (electrodes on same depth-shaft can map to different parcels at GM/WM/GM transitions); per-channel temporal would also operate on parcel-tagged signal which contaminates temporal extraction. Sequencing keeps the three-level anatomy/dynamics story (per-channel temporal + within-shaft spatial + cross-shaft parcel-routing) cleanly separated.

**Distillation (L1)**: explicitly **deferred to v15 / future paper**. Closest published precedent (MTDP) requires teacher inference per batch; SOTA EEG-FM teachers (REVE / ZUNA) blocked on same MNI gate as v14's continuous-coord PE; unblocked teachers (CBraMod / DINOv3+Chronos via MTDP rendering trick) give diet inheritance but not free spatial PE; scope creep on a 4-piece architectural thesis; reviewer-hostility-to-stacking risk per `memory/feedback_publication_bar_2026_iclr_lessons.md`. Run v14 baseline first, see the gap, decide later.

---

## 6e. Stage-1/2 — Zero-per-subject thesis defense (added 2026-05-16, revised 2026-05-16 for spike-vs-field-potential framing + JEPA/DSigLIP asymmetry + 3-arm S1; precedent list extended 2026-05-16 with Knight + NuCLR)

v14's headline claim is **zero learnable per-subject parameters in the deployment forward path** — parcel-id-tagged latents + `log(support+ε)` cross-attn routing as the only subject-conditioning mechanism at inference. **Seven** independent 2024-2026 results define the empirical bar:

- **BIT (Zhang/He/Linderman/Paninski, ICLR 2026)** — Utah-array cross-species FM with per-subject linear read-in + read-out kept at inference (367h, human + monkey).
- **Levin et al. (BrainGate2, Jan 2026)** — Utah-array cross-brain transfer (5 BG2 participants, 48.9h): per-session affine + softsign input layer; shared-layer ablation collapses below scratch.
- **sEEGnificant (Mentzelopoulos NeurIPS 2024)** — sEEG response-time regression (21 subjects, 100+ electrode-hours): per-subject regression head at OUTPUT side; **-ΔR² = 0.18** without. Only in-modality precedent. Canonical: `memory/reference_seegnificant_mentzelopoulos_2024_11.md`.
- **NEDS (Zhang et al., ICML 2025)** — IBL Neuropixels (83 mice, 27k neurons, 74 sessions): per-neuron + per-session learnable input matrices + per-session decoders; ~92% of total params are per-session.
- **OmniMouse (Willeke et al., ICLR 2026)** — Stanford mouse V1 calcium imaging (3.1M neurons / 73 mice / 150B+ tokens): per-neuron + per-session + per-animal identity embeddings; explicitly lists "per-neuron embedding scaling cost" as a paper Limitation.
- **Knight et al. (NeurIPS 2025 FM4B&B Workshop)** — POYO-based unified encoder for mixed Allen Brain Observatory (2P calcium) + Allen Neuropixels (spikes), 100 sessions each across 5 mouse visual regions. Per-session + per-neuron embeddings (unit + session embeddings unfrozen first during finetuning per Appendix C). Cross-region transfer claim: *"exposure to a region through EPhys is sufficient to generalize to that region in the OPhys modality"* — direct structural analogue to v14's cross-subject-via-anatomy claim, but at neuron-grain not parcel-grain.
- **NuCLR (Arora et al., NeurIPS 2025)** — Spatiotemporal transformer + contrastive SSL for **per-neuron identity** representations across Allen VC + IBL + Steinmetz + Bugeon. Same lab as NEDS / POYO+ (Dyer). Explicit followup to NEDS Fig 4's emergent brain-region-from-neuron-embedding finding. Inductive-zero-shot Macro-F1: 0.72 cell-type / 0.53 region (vs 0.42/0.38 NEMO, 0.41/0.25 LOLCAT). Ablation: removing spatial attention layers drops Allen VC 0.72→0.55 (-0.17) and IBL 0.53→0.36 (-0.17) — population context is load-bearing. **Population definition argument**: each Neuropixels probe insertion treated as its own population; cross-insertion interactions deliberately blocked because *"allowing interactions across insertions led the model to cluster neurons based on probe identity rather than biologically meaningful properties."* Direct corroborating signal for v14's parcel-id-tagged (not subject-id-tagged) latent design — same biology-over-recording-identity inductive bias, different mechanism.

### Structural argument: spike vs. field-potential modality gap

Six of the seven precedents (BIT, Levin, NEDS, OmniMouse, Knight, NuCLR) record at the **spike level** — Utah arrays, Neuropixels, 2-photon calcium imaging — where each electrode/neuron samples ~100 μm of tissue and captures genuinely heterogeneous functional identity. Anatomy alone (region, layer, cell-type) does NOT predict tuning at single-cell resolution; per-neuron embeddings are recovering a physical reality that anatomy cannot reach.

v14 records at the **field potential level** — sEEG, mm-scale volume conduction. The brain itself spatially low-pass-filters the signal before it reaches the electrode. Two contacts in the same parcel record near-redundant smoothed versions of the same population activity. The sub-parcel functional heterogeneity that spike recording captures has been **physically averaged away** before our sensors see it. **v14's anatomy-routing is well-justified precisely because the recording modality has already done most of the subject conditioning via spatial smoothing.** v14 is not gambling that "we can do without subject conditioning" — it is observing that the recording modality eliminated the need for sub-parcel per-electrode embeddings.

The only in-modality precedent is **sEEGnificant**, and their per-subject capacity is at the **output side** (response-time regression head), not the input. Encoder-side anatomy routing and output-side per-subject readout are separate questions — see S1-C below.

### JEPA-vs-DSigLIP asymmetry inside v14's SSL stack

Inside v14's two-loss SSL contract, only one path could plausibly need a per-subject head:

- **L_recon (JEPA Level A)**: student and EMA-teacher both use the **same shared encoder**; target lives in v14's own shared latent space, defined by anatomy-routing. Both sides go through the same subject-conditioned function. **No subject bridge to learn.** A per-subject head here solves a non-problem.
- **L_DSigLIP (Whisper-L8 alignment)**: target is in a **frozen external** FM space that knows nothing about v14's subjects. Cross-subject systematic offsets in latent scale, parcel coverage of auditory cortex, hemispheric speech lateralization could create per-subject bridge variance between v14 latent and Whisper space. A per-subject linear projection on the DSigLIP path alone absorbs that bridge variance, lets the shared encoder focus on what's universally shared.

A pretraining-only auxiliary head on the L_DSigLIP path, **dropped at deployment**, is the maximally surgical form of subject conditioning: zero per-subject params at inference, while still solving the brain↔external-FM bridge problem during pretraining. Standard ML auxiliary-head pattern (SimCLR projection heads, MoCo heads, BYOL predictors all dropped at deployment by convention).

### Three-arm ablation

| Cell | Tests | Origin | Slot | Priority |
|---|---|---|---|---|
| **S1-A — v14 default (zero per-subject anywhere)** | Baseline arm. Strongest version of v14's claim: anatomy-routing alone, no subject conditioning in either SSL path or deployment readout. Primary metric: **CrossSubject AUROC**; report CrossSession too. | v14 thesis (`memory/project_v14_unique_contribution_2026_04_26.md`) | Stage-1 first CrossSubject Lite cell after v14 baseline lands | **P0** |
| **S1-B — Per-subject linear projection on L_DSigLIP path only, dropped at deployment** | Tests whether the brain↔Whisper-L8 bridge needs subject-specific conditioning beyond what anatomy-routing provides. Per-subject `nn.Linear(d, d)` between mean-pooled v14 latent and the InfoNCE projection head; **NOT applied to L_recon path**; dropped at downstream inference (forward path stays subject-agnostic). Pre-registered: if S1-B ≥ S1-A + 0.02 → ship S1-B with claim "zero per-subject params in deployment forward path" (BIT-style auxiliary, strictly cleaner because dropped); if S1-B ≈ S1-A → anatomy-routing already handled DSigLIP bridge; ship S1-A. | BIT per-subject linear read-in/read-out (kept at inference) + JEPA-vs-DSigLIP asymmetry framing | Stage-2 (requires SSL stack live) | **P0** |
| **S1-C — Per-subject linear readout at deployment (sEEGnificant/BIT-style, kept at inference)** | Tests whether sEEGnificant's -ΔR² = 0.18 finding (the only in-modality precedent for per-subject capacity) replicates on v14's CrossSubject task after anatomy-routing has already absorbed the encoder-side variance. Small affine post-PMA, one head per subject, params-matched. **Kept at inference** — explicitly violates zero-per-subject-at-deployment. Pre-registered: if S1-C >> S1-B + 0.02 → zero-per-subject-at-deployment is empirically false; ship with the readout and reframe as "anatomy-aware routing + minimal per-subject calibration" (still strictly smaller per-subject footprint than all 5 precedents); if S1-C ≈ S1-B → deployment-time per-subject capacity is redundant with the pretraining-only auxiliary; ship S1-B. | sEEGnificant -ΔR²=0.18 (in-modality, output side) | Stage-1 / Stage-2 (run alongside S1-A) | **P0** |

**Why P0 across all three**: strongest single threat to v14's headline claim, sEEGnificant -ΔR² = 0.18 is the closest in-modality empirical bar, architectural changes are one-line each (`nn.Linear` gated on `subject_id`, either on DSigLIP path or post-PMA). Cheap to run; reviewer-defense load-bearing.

**Composability**: orthogonal to §6d V1-V12 (those test architectural axes inside the zero-per-subject defaults); S1-A/B/C test those defaults themselves against learnable per-subject capacity. Run S1-A as the Stage-1 baseline; S1-C alongside S1-A; S1-B requires Stage-2 SSL stack live.

**Paper framing — three outcomes**:
- **S1-B wins** (DSigLIP-only auxiliary helps, dropped at deployment): "v14 records field potentials, which the brain has spatially smoothed before they reach the sensor; anatomy-aware routing absorbs the variance that spike-recording FMs (BIT/NEDS/OmniMouse) spend per-neuron embeddings on. A pretraining-only auxiliary head on the brain↔Whisper bridge closes the remaining alignment gap. Final v14 has zero learnable per-subject parameters in the deployment forward path."
- **S1-A wins** (no subject capacity needed at all): "v14's anatomy-aware routing fully recovers the cross-subject calibration that spike-recording FMs achieve via per-neuron embeddings (BIT, NEDS, OmniMouse) and that the only matched-modality precedent (sEEGnificant) achieves via per-subject regression heads (-ΔR²=0.18). Zero learnable per-subject parameters anywhere."
- **S1-C wins** (deployment-time per-subject capacity needed): "We ship v14 with a matched-params per-subject readout, still strictly smaller than the per-subject footprints of BIT, Levin, sEEGnificant, NEDS, OmniMouse, Knight et al., and NuCLR. Anatomy-aware routing reduces per-subject capacity by [X]× while preserving cross-subject performance."

All three framings are reviewer-defensible; asserting any of them without the 3-arm cell is not. Canonical synthesis: `memory/project_v14_spike_vs_field_potential_per_subject_defense_2026_05_16.md`.

---

## 6g. Stage-1 — v14 full architecture sister cells (added 2026-05-18; revised end-of-session 2026-05-18)

**Architecture locked 2026-05-18 end-of-session** in [[project_v14_arch_revision_2026_05_18]]. Full stack:

```
INPUT log-STFT (B, C, F=38, T≈52) after Nv14
  ↓
❶  A2 conv stem (3-layer Conv1d per (electrode, freq_bin), band-preserving, shared weights)
❶b freq PE (additive, NEW 2026-05-18, FAC-defended)
  ↓
TOKEN STACK × N=4:  Pre-LN→❸a temporal SA→residual ; Pre-LN→❸c freq SA→residual ; Pre-LN→MLP→residual
  ↓
GEOMETRY STACK × N_g=1:  Pre-LN→❸g spatial SA (modality-configurable)→residual ; Pre-LN→MLP→residual
  ↓
❹  parcel-id PE (additive)
❺  Perceiver cross-attn (320 parcel-tagged latents, log(support+ε) bias)
❻  Latent SA × L=6
❼  Single-query PMA readout
```

~2.8M params total. Modality-configurable ❸g (RoPE-family position encoding): sEEG-BT default = within-shaft 1D RoPE on depth; ECoG = 2D RoPE on (row, col); MNI = 3D RoPE OR Fourier PE.

**Canonical source**: `memory/project_v14_arch_revision_2026_05_18.md`. Derivation memos: `memory/project_v14_freq_pe_addition_2026_05_18.md` (audio FM survey + FDY/FAC + freq-axis-property mapping), `memory/project_v14_input_tokenizer_a2_conv_stem_2026_05_18.md` (A2 stem spec).

**Killed alternatives (do NOT bring back without revisiting)**:
- F_patch > 1 AST-style 2D patches (band-mixing pathology on linear-Hz)
- 2D conv stem with 3×3 kernels / CoAtNet pattern (same pathology)
- Hierarchical Perceiver IO at input (P3 future-scale; zero in-modality precedent)
- Joint per-electrode (freq, time) SA as default (TimeSformer factorized beats joint at lower cost)

---

### Tokenizer cells (F0–F10) — test ❶ and ❶b choices

| Cell | Variant | Tests | Priority |
|------|---------|-------|----------|
| **DEFAULT** | A2 shared-weight + ❶b freq PE | — | DEFAULT |
| **F0** | Linear F→d per time bin (BrainBERT/BIOT/Whisper baseline) | Does band-preservation matter empirically? Most important comparison; if F0 ties A2 within noise, revert to F0. | **P0** |
| **F1** | A1: single Conv1d per (c, f), kernel=stride=T_patch=10, no nonlinearity | Does multi-layer conv depth earn its keep over single linear patch? | **P1** |
| **F3** | 2-layer A2 (Whisper depth) | Conv-depth sensitivity (shallower) | P2 |
| **F4** | 5-layer A2 (Wav2Vec depth) | Conv-depth sensitivity (deeper) | P2 |
| **F6** | A2 per-freq groupwise (`groups=F=38`); each freq has own conv weights | Tests freq-equivariance assumption (FAC says PE > per-freq weights; per-freq specialization is SED design that hasn't transferred to audio FMs) | **P1** |
| **F7** | A2 shared-weight, **drop ❶b freq PE** | Tests FAC claim in iEEG: is freq positional embedding load-bearing for downstream attention? Must-run alongside default to prove ❶b earns its 5k params. | **P0** |
| **F8** | FDY adaptive per-freq conv (Nam 2022 attention-weighted basis of K=4 kernels per freq) | Heavyweight per-freq specialization; only run if F6 wins decisively | P2 |
| **F9** | A2 + SubSpectral Normalization (Chang 2021, per-sub-band BatchNorm replaces GroupNorm) | Normalization-only specialization; cheaper than per-freq weights | P2 |
| **F10** | AST 16×16 patches over log-STFT (band-mixing baseline) | Quantifies cost of band-mixing pathology; if F10 ≈ A2 then band-preservation is decorative | P2 |

---

### Token-level structure cells (F11–F19) — test factorization + depth choices

| Cell | Variant | Tests | Priority |
|------|---------|-------|----------|
| **F11** | Joint per-electrode (❸a + ❸c → single joint SA over F×T_chunk=494 tokens per electrode) | TimeSformer claim on iEEG: does factorized (default) beat joint? Joint is 38× more compute. | P1 |
| **F12** | Hierarchical per-electrode PMA pool over (F, T_chunk) before ❺; M_per_elec ≈ 4-8 queries per electrode summarize to ~1k tokens entering ❺ | Tests whether per-(freq, time) detail at ❺ is load-bearing or summarizable; ~49× compute savings at ❺ if pooling works | P1 |
| **F13** | ❸a / ❸c order flip (❸c freq first, then ❸a temporal) | TimeSformer says order is minor; verify | P2 |
| **F14** | ❸c replaced with band-aware GroupNorm (cheaper alternative to attention-based freq mixing) | Tests whether explicit attention is needed for freq mixing or normalization-only works | P2 |
| **F15** | Token depth N sweep {1, 2, 4, 6, 8} | Default N=4 is reasoned not measured; sweep finds optimum | **P0** |
| **F16** | Token-vs-latent depth trade: fix total budget, sweep N vs L ∈ {(N=2,L=10), (N=4,L=6), (N=6,L=4), (N=8,L=2)} | Where to spend depth budget for best perf-per-param? | P1 |
| **F17** | Drop MLP from each token block (only attention + LN, no FFN) | Tests whether MLP earns its 130k/block params at token level | P2 |
| **F18** | Add band-id embedding alongside bin-level freq PE (additional 6-vocab embedding for delta/theta/alpha/beta/low-γ/high-γ) | Tests whether explicit band-level prior helps over bin-level PE alone | P1 |
| **F19** | Multi-scale temporal stem: parallel branches with different RFs (e.g. RF=5 / 9 / 17 / 33) addressing band-specific time constants directly | Per-band time-constant argument structurally; F19 = stem-level fix, ❸a-as-multi-scale = attention-level fix | P2 |

---

### Geometry slot cells (F20–F27) — test ❸g instantiation + N_g

| Cell | Variant | Tests | Priority |
|------|---------|-------|----------|
| **F21** | Flat cross-electrode ❸g at fixed (freq, time_chunk) — no within-shaft restriction (PopT-style) | Whether SHAFT-restriction matters or just any cross-electrode SA helps | P1 |
| **F22** | ❸g gated to attend only when within-shaft contacts are in DIFFERENT parcels | Tests "geometry only matters where parcel routing isn't sufficient"; if F22 ≈ default, simplify | P2 |
| **F23** | sEEG pia-distance alignment: data-side preprocessing per-shaft to make "contact 1 = closest to pia" consistent across subjects (uses FreeSurfer recons we already have). Architecture stays 1D RoPE; this fix maximizes RoPE's directional capacity for the ~25% of cohort with inconsistent shaft orientation. | **Data-side fix, runs in parallel with architecture work.** P1. | **P1** |
| **F24** | sEEG learned per-shaft direction embedding (2D vector per shaft, ~256 params for 128 shafts; modulates signed bias) | Tests "can model learn its own orientation from data" | P2 |
| **F25** | MNI 3D RoPE OR 3D Fourier PE in ❸g (when Chris ships MNI; both modern ML standards for continuous 3D position) | Most expressive geometry; trivially directional | **Blocked on Chris MNI delivery** |
| **F26** | Interleaved (❸a + ❸c + ❸g + MLP) × N=4 vs default separated (❸a + ❸c + MLP × N=4 then ❸g + MLP × N_g=1) | Tests whether separated loses anything to interleaved; default is separated for compute + clarity | P1 |
| **F27** | N_g sweep {0, 1, 2, 4} | N_g=0 tests drop-❸g entirely (subsumes prior F20); higher N_g tests deeper spatial integration. P0 for ECoG when added. | **P0** |
| **F28** | Position-encoding alternatives sweep for ❸g: 1D RoPE (DEFAULT, modern ML standard) vs T5 \|Δdepth\| symmetric vs ALiBi symmetric vs learnable absolute depth PE | Sanity-check that RoPE wins (or at least ties) the alternatives at sEEG-BT scale. Default is RoPE on first-principles ML standard; this cell empirically confirms. | P2 |

---

### Readout cells (F29–F30) — test ❼ readout pattern

| Cell | Variant | Tests | Priority |
|------|---------|-------|----------|
| **F29** | CLS token prepended to 320 parcel-tagged latents; participates in ❻ latent SA × L=6 bidirectionally; read CLS state at end → linear head → logits (replaces ❼ PMA) | BERT/ViT/AST/AudioMAE standard readout. Whether bidirectional SA participation aggregates better than single cross-attn at end. | P1 |
| **F30** | GAP (global average pool) over 320 latents → linear head → logits (replaces ❼ PMA, no learnable readout query) | ViT-22B / DINOv2 standard readout (both moved from CLS to GAP at scale). Tests whether explicit readout aggregation is needed at all. | P2 |

**DEFAULT keeps PMA** for Perceiver IO framing + SSL-pretrain→finetune separation (latents stay task-agnostic during SSL; PMA query is task-specific finetune head; multi-task extension via additional queries is natural).

**For Stage-2 SSL `L_DSigLIP`**: the contrastive summary vector should MATCH the readout choice — same representation for SSL summary and classification summary. PMA query output (default), CLS state (F29), or GAP (F30).

---

### Execution order (P0 must-run before Stage-1 lock)

1. **F0 vs DEFAULT**: band-preservation hypothesis test
2. **F7 vs DEFAULT**: freq PE load-bearing test
3. **F15 (token depth sweep)**: find optimal N
4. **F27 (N_g sweep including N_g=0)**: validate geometry slot decision

Then P1 cells based on what P0 reveals.

---

### Composability

Cells orthogonal to:
- §6d V8-V13 (test attention-axis hard-mask variants on prior simpler architecture; some superseded by current 3-axis factorization)
- §6e S1-A/B/C (zero-per-subject thesis defense, latent vs head-side)
- §6f Stage-2 SSL recipe cells

When a cell flips a default (e.g. F0 ties A2), update [[project_v14_arch_revision_2026_05_18]] accordingly.

---

## 6h. Stage-1/2 — Post-competitor-review cells (added 2026-05-20)

Cells from the 2026-05-20 full read of three iEEG competitors — MVPFormer (arXiv 2506.20354, ICLR 2026 poster), Neuro-MoBRE (2508.04128), BrainWave (2402.10251v7). Canonical synthesis: `memory/project_v14_competitor_review_2026_05_20.md`. **Only MVPFormer touches BrainTreebank**, and only on within-subject binary — like DIVER-1, not a submission-gate ceiling. The takeaways are four cells. Explicit non-takeaways (co-upcycling / TIES-merge, the wide-CLS-in-sequence mechanism, 1.2B scaling, the db4 wavelet encoder, BrainWave's scale-alignment layer, and the attention quadratic-vs-subquadratic cost question — irrelevant at BT data scale) are recorded in the memo, not here.

| Cell | Tests | Origin | Slot | Priority |
|---|---|---|---|---|
| **X1 — Few-shot prototype eval, zero training** | On a frozen pretrained backbone, build one class prototype = mean readout vector over `K` support trials per class, classify test trials by nearest prototype. Zero gradient steps. Sweep `K ∈ {3, 8}`; **lead with CrossSubject** (the regime v14 differentiates on), report CrossSession too. Complements the linear probe by removing the probe's own fitted capacity from the measurement — isolates the backbone representation. Readout vector follows the §6g readout choice (PMA query / CLS / GAP). | BrainWave (2402.10251v7) few-shot prototype protocol | Eval mode, run on all P1+ pretrained checkpoints; also noted in `plan.md` frozen-probe experiment | **P1** |
| **X2 — Per-task PMA query sets** | Replace ❼'s single learned PMA query with one learned query per Neuroprobe task (small query set). One forward pass yields a dedicated pooled summary per task instead of a shared summary + per-task linear head. Adopts Neuro-MoBRE's **principle** (task-disentangled pooled readout) not its **mechanism** (J wide CLS tokens spliced into the sequence — PMA queries are the cleaner Perceiver-IO form; see F29). Tests whether per-task readout queries beat shared-query + per-task head. | Neuro-MoBRE (2508.04128) task-disentangled wide CLS principle | Stage-2 readout (composes with §6g F29/F30) | P2 |
| **X3 — Learned-correction routing bias** | Make the parcel-routed cross-attn bias correctable: `log(support+ε) + γ·Δ`, where `Δ` is a learned per-(electrode, parcel) correction **initialized at zero** and `γ` small/learned. At init it is *exactly* the frozen default; training can deform it. Sharpens §3's "Anatomy enforcement: L" cell to the cleaner zero-init additive-correction form (same param-count question; zero-init guarantees the cell departs from the frozen default rather than from a re-parameterized init). Adopts Neuro-MoBRE's **principle** — an anatomy-seeded learnable spatial router is viable (their channel-wise router converges to region structure on its own) — **not** its MoE mechanism (per-region FFN experts + top-2 routing). **Pre-registered**: frozen `log(support)` stays the default; X3 promotes only on ≥0.005 CrossSubject lift without raising subject-ID nuisance decodability. "Learned strictly dominates frozen" holds only in *expressivity*, not generalization — the frozen prior is a real regularizer at 9-subject scale and carries the source-localization `p(sensor|source)` paper claim. | Neuro-MoBRE learned-router principle (mechanism NOT adopted) | Stage-1 anatomy enforcement (§3) | **P1** |
| **X4 — SSL direction: causal vs bidirectional** | A/B the SSL pretext *direction* on v14 iEEG: bidirectional masked-latent prediction (default, data2vec / V-JEPA family) vs causal next-segment latent prediction (MVPFormer-style), **same target space, same encoder**. Sharpens §4 Track 2B (specified as MVPFormer-style next-segment InfoNCE, but as a separate SSL family rather than a direction knob). Laya's matched-MAE result settled target-space (latent > raw-MSE) but said nothing about direction — this is a genuinely untested knob for v14. | MVPFormer causal CPC vs v14 bidirectional masked-latent | Stage-2 SSL pretext (§4 Track 2B) | P2 |

---

## 7. Stage-3 — Foundation-model swap + cold-start

| Cell | Origin | Hypothesis | Status |
|---|---|---|---|
| **MTDP-style stimulus-agnostic cold-start** | MTDP §4.2 (Oxford Mar 2026, DINOv3 + Chronos teachers, EEG-as-(3,C,T)) | cold-start v14 from stimulus-agnostic teachers before brain↔FM contrastive; MTDP is closest v14 competitor in spirit but stimulus-AGNOSTIC. **DEMOTED 2026-05-10 from "Stage-3 candidate" to "fallback if Phase-1 JEPA saturates."** Empirical chain "MTDP > latent-JEPA" is NOT in the literature — MTDP only beat raw/token-MAE baselines (LaBraM/CBraMod/BIOT), never head-to-head against V-JEPA-style latent prediction. Architecturally a monkeypatch (3-channel tensor for DINOv3 is mechanically arbitrary; principled spectrogram + CWT renderings tested by authors and gave no improvement). If v14 needs an unpaired-data scaling lever, prefer principled options inside v14's own structure (Level C JEPA at scale, multi-target V-JEPA 2.1, cross-session JEPA) before reaching for foreign-FM distillation | **DEMOTED — Stage-3 fallback only** |
| D-SigLIP multi-FM extension | Tang 2023 NeurIPS multimodal precedent (BridgeTower image-caption → fMRI generalization) | v14's audio + vision + (language) D-SigLIP anchors are direct iEEG extension; Stage-3 SSL motivation | planned |
| DINOv3 vision FM addition | `project_neuroprobe_stage2b_drop_stage3_reframe_2026_04_26.md` | adds vision modality for movie-watching tasks | planned |
| PHI-S Hadamard standardization | heterogeneous-teacher feature standardization | combines Whisper + DINOv3 + GPT/Llama feature distributions without naive concat bias | planned |
| MTDP gated multi-teacher fusion | masked latent denoising with learned teacher gates | alternative to separate per-FM REPA losses | candidate |
| BrainGFM atlas-mix pretraining | BNA Tier-1 only vs BNA Tier-1 + Tier-2 mix | multi-parcellation as augmentation | candidate |
| BioX-Bridge cross-FM bridges | lightweight prototype-network bridges between frozen biosignal/FMs | parameter-efficient multi-FM extension under ≤30M cap | candidate |
| Continuous-corpus alignment gate | precision event-time alignment / clock-drift audit for ds003688, Podcast, NeuroListen | Stage-3 pre-bake before continuous paired data enters | required when continuous paired data lands |

---

## 8. Out of Neuroprobe scope (pointers)

- **PS-program Stage-1 architectural ablations** (T1.2 aug / T2.2 per_electrode capacity / T3.1 atlas-anchored hierarchical / T3.4 free q_cell / T3.5 noemb / T3.6 cross-attn) — PS-paused 2026-04-24. Canonical: `docs/strategy/stage_1.md`. Stage 1 closed 2026-04-20 at `per_cell + partialconv + pe2d + hierarchical_atlas @ d=32, depth=3, pool=(4,8)`.
- **PS-program Stage-2 ablations** (H2.2 SSL objective selection / position ablation / per-electrode d=64 re-test / `P_emb` re-test at N≥12) — paused. Canonical: `docs/strategy/stage_2.md`.
- **PS-program Stage-3** (cross-sensor sEEG D-cohort transfer) — TBD. Canonical: `docs/strategy/stage_3_rh_expansion.md`.

## 9. Archived / discontinued

| Cell | Why dropped |
|---|---|
| **Stage-2b as DEFAULT schedule** | dropped 2026-04-26 for head-side leak risk; superseded 2026-04-27 night by joint-from-step-1 default. Sequential 2a→2b retained as sister-run ablation only (§5). Ref: `project_neuroprobe_stage2b_drop_stage3_reframe_2026_04_26.md` |
| **Curriculum warmup (recon-first 50-80%)** | no precedent; originally miscited to SigLIP-2 which is the opposite pattern (cross-modal first → +intrinsic at 80%) |
| **v14 kernel ablation** (pre-reset, 2026-04-17) | superseded by Stage-0 L-sweeps + Stage-1 AC roster; backed up externally |

---

## Statistical method (binding for all cells)

Every freeze decision must cite Stage-0's stat appendix: bootstrap N=2000 percentile CIs, paired Wilcoxon + rank-biserial, BH within sweep, ≥ 3 seeds (42/43/44) on chosen + nearest competitor, train/test pair-overlap assert, upstream-commit + Whisper-commit + uv.lock SHA pinned. Source: `docs/neuroprobe/stage_0.md §"Statistical Methods"`.

**Trend-level reporting (added 2026-05-12 from MultiDiffNet)**: alongside the bootstrap-CI primary stats, report Cohen's d + 95% CI + win-rate matrix + Bayesian baycomp posterior `(P(left), P(rope), P(right))` with ROPE threshold ρ=0.01. Hierarchical evidence assessment (strong / moderate / weak / minimal) based on `|d|` magnitude × relative improvement × cross-seed consistency. Designed for n=3-seed regimes where Wilcoxon/permutation p-values cannot pass correction even with systematic improvements. **This complements, does not replace, classical tests** — it's a reviewer-defensible reporting layer for the high-variance low-trial regime. Source: MultiDiffNet (Zhang/Shapovalenko 2025) §D. Canonical: `memory/reference_multidiffnet_zhang_2025.md`.

## Source memos (cited, not duplicated)

- `memory/project_l1_normalization_freeze_2026_05_08.md` — L.1 freeze
- `memory/project_l2_reference_view_freeze_2026_05_09.md` — L.2 freeze
- `memory/project_l7_audio_fm_blocked_audio_source_2026_05_10.md` — L.7 blocker
- `memory/project_v14_paper_corrections_post_newpapers6_batch2_2026_05_09.md` — 4-cell loss triangulation, 5 AC cells, paper-framing
- `memory/project_v14_stage2_schedule_audit_2026_05_09.md` — Stage-2 schedule audit (50+ methods)
- `memory/project_stage2_ssl_initial_diet_bt_only_joint_step1_2026_05_09.md` — Stage-2 SSL Experiment 1 default
- `memory/project_three_paper_biosignal_ssl_convergence_2026_05_10.md` — 3-paper convergence framing (NeuralBench + Neuroprobe + Laya)
- `memory/project_v14_post_eeg_dino_synthesis_2026_05_10.md` — **canonical post-EEG-DINO synthesis**: Level C JEPA target contract, 3-channel spatial PE composition, EMA + StopGrad default (5/5 brain-FM precedent), BaRISTA-confound resolution, MTDP demotion, per-parcel coverage diagnostic, anatomically-realistic mask sampling. Source for §6b cells.
- 8 paper reference memos under `memory/reference_{tribe_v2,neuralbench,laya_ucla,dinov3,mtdp_oxford,labram,reve,mentality}_*.md`. EEG-DINO (MICCAI 2026) added as precedent point for "DINO-family beats raw-MAE on biosignals" and "EMA on biosignals at scale" — not architecturally borrowable (channel-id-only PE only works for fixed 10-20 layout).
- NEWPAPERS7 batch (2026-05-12): `memory/reference_aspen_lee_2026.md` (L.2 spectral-vs-temporal external ratification + AC-Dual-Stream origin), `memory/reference_multidiffnet_zhang_2025.md` (trend-level reporting framework origin), `memory/reference_shapovalenko_auster_wav2vec_clip_2025.md` (layer-probing cautionary tale). Evanson/King developmental sEEG covered under `memory/reference_dascoli_brain2qwerty_gwilliams_evanson_dev_2025.md` (4-paper King-lab cluster).
