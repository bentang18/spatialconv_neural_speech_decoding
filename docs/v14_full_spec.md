# v14 — Architecture Spec

iEEG FM · anatomy-tagged factorized Perceiver IO + 3-phase staged SSL · BrainTreebank 9-subject cohort. State: v4 (post-v3 amendment 2026-05-19) · d=256 · ~13M params.

**Thesis.** iEEG FM with anatomy-tagged factorized Perceiver IO + soft parcel-routing cross-attn (`log(support+ε)` over BT-DK one-hot), pretrained with factorized-t×f Stage A + electrode-mask Stage B + multi-teacher P3 distill (Whisper-L8 + DINOv3), beats PopT cross-subject ≥0.05 AUROC at ≤30M params.

| | |
|---|---|
| Cohort | 9 BT subjects {1,2,3,4,6,7,8,9,10}; S5 dropped (lesion). Leaderboard subset {1,2,3,4,7,10}. |
| Parcels | BT FreeSurfer Desikan-Killiany (Pipeline C, native-vol aparc+aseg, 0.5 mm snap). K=80 vocab, 98.1% coverage. |
| Size | d=256, 8 heads, ~13M params (≤30M cap). |
| Sampling | BT native 2048 Hz. |

## 1. Preprocessing

Raw iEEG voltage → `(C electrodes, F=30 freq-bins, T time-bins)`.

| # | Op | Params |
|---|---|---|
| 1 | HPF | Butterworth filtfilt, cutoff 0.5 Hz |
| 2 | Line-noise comb | `iircomb(w0=60, Q=30, notch, fs)` — rejects all 17 harmonics in one pass |
| 3 | Channel QC | MNE LOF, threshold 1.5, n_neighbors 20, per session |
| 4 | Re-reference | shaftCAR (within-shaft CAR over good channels) |
| 5 | Drop bad channels | flagged channels removed before slicing |
| 6 | Window | onset-anchored slice `[0,1] s` (A.0) |
| 7 | STFT | magnitude STFT, Neuroprobe swept-optimal window/hop; STFT cell = patch |
| 8 | Filterbank | triangular ⅓-octave, 30 bins, mel-style edges, uniform log spacing |
| 9 | Log-power | `log(energy + 1e-6)`; phase discarded |
| 10 | Normalize | Nv14 robust-z per `(electrode, freq, session)`: `(x−median)/(1.4826·MAD)`, full-session time pool, transductive at inference |

## 2. Filterbank — 30-bin log ⅓-octave

Centers `f_k = 2^(k/3)` Hz, k=0…29 (1–813 Hz). Triangular filter for bin k spans `[f_{k-1}, f_{k+1}]`, 50% overlap. Bin-29 upper edge = 1024 Hz = Nyquist.

| Bins | Center (Hz) | Band | # |
|---|---|---|---|
| k0–k5 | 1.0–3.17 | Delta | 6 |
| k6–k8 | 4.0–6.35 | Theta | 3 |
| k9–k11 | 8.0–12.7 | Alpha | 3 |
| k12–k14 | 16.0–25.4 | Beta | 3 |
| k15–k18 | 32.0–64.0 | Low gamma | 4 |
| k19–k21 | 80.6–128.0 | High gamma | 3 |
| k22–k23 | 161.3–203.2 | Very-HG/ripple | 2 |
| k24–k27 | 256.0–512.0 | Ripple/MUA | 4 |
| k28–k29 | 645.1–812.7 | MUA-envelope | 2 |

Uniform ⅓-octave spacing (neutral attribution axis). SWEC band-limited 0.5–150 Hz → trains only k0–k21; k22–k29 get gradient from broadband corpora only.

## 3. Architecture — 7 stages

Factorized throughout: token blocks `t×f`, latent stack `t×parcel`. Latents keep a time axis `(320,T,d)`.

```
(C,F=30,T)
  ❶ A1 linear embed + categorical freq embed     → (C,F,T,d)
  ❷ Token block ×N=4  (per-electrode, t×f)        → (C,F,T,d)
  ❺ Parcel-routed cross-attn ×2 @ {0,3}           → (320,T,d)
  ❻ Latent stack ×L=6  (time×parcel)              → (320,T,d)
  ❼ PMA readout                                   → (k,d) → out
```

| Stage | Spec |
|---|---|
| ❶ Input embed | `Linear(1→d)` per (t,f) cell (= patch embed, patch 1×1). 30 flat learnable d-vectors added per freq-bin (non-separable, so freq clustering recovers bands). No time PE. |
| ❷ Token block ×4 | TimeSformer-divided: temporal SA (RoPE θ=10000) → freq SA → MLP, each residual. Hard −∞ cross-electrode mask. Pre-norm LN, GeLU, MLP 4×, 8 heads (32 dim/head). Sweep N∈{2,4,6}. |
| ❺ Cross-attn ×2 | 320 free latents `nn.Parameter(320,d)` trunc-normal std 0.02 (= K=80 parcels × M=4 slots). Pools (electrode,freq)→parcels strict 1:1 per time-step (`T_latent=T`). QK bias `log(support[e,p]+ε)`, ε=1e-2, support = BT-DK one-hot (Graphormer-style). 2 layers at positions {0,3} of the L=6 stack. Latents keep time axis. |
| ❻ Latent stack ×6 | Per layer: time SA (RoPE) → parcel SA → MLP, each residual. Native rate (~13 Hz STFT). Sweep L; factorization order time-first. |
| ❼ PMA readout | k learnable seed queries cross-attend latent set (Set Transformer). Downstream k=1 (or n_tasks) → `Linear(d→n_classes)`. Phase-3 k=50 @ 10 Hz → Smooth-L1 vs fused teacher. |

**Spatial info — 2 mechanisms:** parcel routing @ ❺ (discrete, `log(support+ε)` bias); geometry slot @ ❸g (continuous, modality-configurable; empty for sparse sEEG default). No additive position features. MNI Fourier PE dropped 2026-05-19. Within-parcel depth = P1 sister (`F-depth-bias`), not first pass.

## 4. Parameter budget

d=256, 8 heads. MHA ≈ 4d² ≈ 262k; MLP-4× ≈ 524k/block.

| Component | Per unit | Count | Subtotal |
|---|---|---|---|
| ❶ Input stem | — | — | ~0.01M |
| ❷ Token block | ~1.05M | 4 | ~4.2M |
| ❺ Cross-attn | ~0.79M | 2 | ~1.6M |
| ❺ latents 320×256 | — | — | ~0.08M |
| ❻ Latent stack | ~1.05M | 6 | ~6.3M |
| ❼ PMA readout | ~0.8M | 1 | ~0.8M |
| **Total** | | | **~13M** |

## 5. Training recipe — 3-phase staged

Max data scale on unpaired SSL backbones (P1+P2); small adapter on paired data with frozen backbones (P3).

**Phase 1 — Stage A SSL.** Trains ❶❷ (A1+freq embed+4 token blocks, ~1–1.5M). Objective `L_recon` Level B (token-mask JEPA). Mask 80% inverse-block on (t,f) grid per electrode. Target EMA-teacher latent, layer-averaged (data2vec-2.0). Loss MSE; UFO frame+utterance λ=1. EMA decay 0.99→0.9999 linear warmup. Teacher discarded.

**Phase 2 — Stage B SSL.** Stage A frozen. Trains ❺❻ (parcel routing + cross-attn + L=6 stack, ~0.8–1M). Objective `L_recon` Level A (electrode-mask JEPA). Mask 30% random whole-electrode before cross-attn (J1 sweep {0.25,0.30,0.40,0.50}). Target EMA-teacher Stage-B latents, layer-avg, StopGrad. Loss MSE.

**Phase 3 — MTDP cross-modal distillation.** A+B frozen by default. ~200 h paired iEEG-audio. Target K=50 bucketed-pooled @ 10 Hz (syllable rate).
- *3a gating* — Whisper-L8 + DINOv3 frozen; train 2-layer gating MLP (~10k params, softmax over teacher weights); masked-latent-denoising MSE.
- *3b-warmup* — adapter-only (~0.6–0.8M), ~3–5% of 3b budget; closes cold-start gap.
- *3b-unfreeze* — ~95% of 3b budget; Stage A slow-LR (LR/10), Stage B + adapter normal LR; teachers + gating frozen.
- Adapter: 1-layer PMA, K=50 queries d=256. Whisper-L8 mean-pool-by-5 (250→50, d1280→256); DINOv3 120 frames→50→256. Teacher instance-norm per-token before pooling.

**Pretrain corpus (P1+P2):** SWEC 6,672 h (50 subj, 0.5–150 Hz → k0–k21 / Stage A only) · AJILE12 1,280 h · D-cohort 180 h · BrainTreebank 43.5 h · PS+lex ~7 h. **Total ~8,180 h.**

**Phase-3 optim:** AdamW (β=0.9/0.98, wd=0.05), peak LR 3e-4 (LR/10 on Stage A), 2k warmup + cosine → 1e-5 over 50k steps, batch 256.

## 6. Objectives & losses

| Phase | Loss | Mask | Target | Collapse prevention |
|---|---|---|---|---|
| 1 — Stage A | MSE (Level B) | 80% inverse-block (t,f) | EMA-teacher latent, layer-avg | EMA + StopGrad |
| 2 — Stage B | MSE (Level A) | 30% random electrode | EMA-teacher Stage-B latents | EMA + StopGrad |
| 3a | masked-latent MSE | — | fused teacher rep | — |
| 3b | Smooth-L1 | — | 50 fused-teacher buckets @ 10 Hz | teacher instance-norm |

- `L_recon_B = MSE(predictor(student, mask), StopGrad(EMA-teacher latent, layer-avg))`
- `L_recon_A = MSE(student latents, StopGrad(EMA-teacher Stage-B latents, full electrode set))`
- `L_distill = Smooth-L1(adapter[50 buckets], fused-teacher[50 buckets])` — β=0.5–2.0; K=50 @ 10 Hz = syllable rate (Goldstein 2025 alignment timescale); fused teacher = gating-weighted Whisper-L8 + DINOv3.

**Rejected:** per-frame 50 Hz regression (below alignment timescale) · InfoNCE/SigLIP/contrastive (beaten 4/4 vs regression) · hybrid loss (no <1k h SOTA uses it) · EMA on P3 adapter (EMA is self-distill only) · BLIP-2 Q-Former (100×+ more paired data) · `L_KoLeo` (pruned 2026-05-12).

## 7. Evaluation & submission gate

| Path | Setup |
|---|---|
| A1 (headline) | A+B frozen + adapter + linear-probe per task; competes with DIVER-1 frozen probe (0.678 within-session) |
| A2 (gap-closing) | Stage A unfrozen via 3b-unfreeze + linear-probe |
| B | frozen + 2-layer MLP probe |
| C | light task fine-tune (Charmander: decoder warmup → 4-layer unfreeze) |

**Gate** (dual-prong, SOTA-at-submission): ≥ **0.667** CrossSession multi-class accuracy **AND** ≥ **0.628** CrossSubject binary AUROC, ≥ 4 tasks, ≤ 30M params. Always report A1 and A2; the Δ is itself an ablation.

## 8. Hyperparameters

| Group | Param | Value |
|---|---|---|
| Model | d_model | 256 (sweep {128,192,256}) |
| | heads | 8 (32 dim/head) |
| | token blocks N | 4 (sweep {2,4,6}) |
| | latent-stack depth L | 6 |
| | latents | 320 (K=80 × M=4) |
| | cross-attn layers | 2 @ {0,3} |
| Blocks | norm / act / MLP | pre-norm LN / GeLU / 4× |
| | RoPE θ | 10000 |
| | latent init | free `nn.Parameter(320,d)`, trunc-normal std 0.02 |
| Routing | support / bias / ε | BT-DK one-hot / `log(support+ε)` / 1e-2 (sweep {1e-4…1e-1}) |
| SSL mask | P1 / P2 / EMA | 80% inverse-block (t,f) / 30% random electrode / 0.99→0.9999 |
| P3 optim | optimizer / LR / sched / batch | AdamW β=0.9/0.98 wd=0.05 / 3e-4 (LR/10 Stage A) / 2k warmup+cosine→1e-5 over 50k / 256 |
| | adapter | 1-layer PMA, K=50 queries, d=256 |

## 9. Ablation / sister-cell suite

**P0:** F-joint-tF (joint vs factorized t×f) · F-linear-STFT-38 · F-150Hz-cap · single-/3-cross-attn @ {0,2,4} · M-sweep {4,8,16} · F10b (AST 2×4 patch stem) · BiL-NoSpatial/BiL-MNI/BiL-Freq-Plain (Bitter-Lesson priors) · J1 mask rate / J2 mask level · N-/L-/d-sweep / N2 factorization-order · R-joint/R-staged · R-no-phase-3.

**P1:** F-depth-bias · F-additive-time-PE · factored-freq-embed/F-no-freq-embed · R-frame/R-cosine/R-no-gating/R-no-warmup · F-Morlet/B-edges/F-½-octave/F-band-prior-spacing.

---
Compiled from canonical project memory: post-v3 arch amendment (2026-05-19, v4) · 3-phase staged SSL recipe (2026-05-18 + 5/19) · preproc recipe (2026-05-12/17) · DK-first routing (2026-05-13) · v14 pruning (2026-05-12). Generated 2026-05-20.
