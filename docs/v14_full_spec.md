# v14 — Architecture Spec

iEEG FM · anatomy-tagged factorized Perceiver IO + 3-phase staged SSL · BrainTreebank. v4 (2026-05-19) · d=256 · ~13M params.

**Thesis.** Anatomy-tagged factorized Perceiver IO + soft parcel-routing cross-attn (`log(support+ε)`, BT-DK one-hot), pretrained factorized-t×f Stage A + electrode-mask Stage B + multi-teacher P3 distill → beats PopT cross-subject ≥0.05 AUROC at ≤30M params.

Cohort: 9 BT subj {1,2,3,4,6,7,8,9,10}, S5 dropped. Parcels: FS Desikan-Killiany, K=80, 98.1% coverage. 2048 Hz native.

## 1. Preprocessing → `(C, F=30, T)`

HPF 0.5 Hz → comb notch `w0=mains_hz,Q=30` (per-corpus: 60 Hz BT/D/AJILE12, 50 Hz SWEC) → MNE-LOF QC (thr 1.5) → **ref draw** (5/27 PM ref-aug lock: per-clip uniform-random over `{shaftCAR, bipolar, Laplacian}`; raw skipped; SWEC degenerates to global-CAR-only) → drop bad → slice `[0,1]s` → magnitude STFT → triangular ⅓-octave 30-bin filterbank → Nv14 robust-z per `(electrode,freq,session)` `(x−median)/(1.4826·MAD)`. (5/25 swap: post-filterbank `log(energy+1e-6)` step dropped; raw filterbank magnitude is the default. F-log-amplitude sister re-enables `log` via `apply_log=True`. Bin spacing is unchanged — "log" in §2 refers to log-SPACED bin centers, not value-axis compression.) Ref-aug dispatch is sister-first: BT-Lite paired run (ref-aug ON vs OFF) gates full P1 all-corpora rollout via 4 kill criteria; details in `docs/neuroprobe/training_recipe.md §3` + memo `[[project_v14_ref_aug_input_distribution_lock_2026_05_27]]`.

## 2. Filterbank — 30-bin log ⅓-octave

`f_k = 2^(k/3)` Hz, k=0…29 (1–813 Hz), triangular 50% overlap, Nyquist 1024 Hz.

| k0–5 | k6–8 | k9–11 | k12–14 | k15–18 | k19–21 | k22–23 | k24–27 | k28–29 |
|---|---|---|---|---|---|---|---|---|
| Delta | Theta | Alpha | Beta | LowГ | HighГ | v-HG | Ripple | MUA |

SWEC 0.5–150 Hz → trains k0–k21 only.

## 3. Architecture — 7 stages

Factorized throughout. Latents keep time axis `(320,T,d)`.

```
(C,F,T) → ❶ embed → ❷ token block ×4 → ❺ cross-attn ×2 → ❻ latent stack ×6 → ❼ PMA → out
```

| Stage | Spec | Shape |
|---|---|---|
| ❶ | Conv2d (3,2) patches @ A1 + per-patch freq embed (10 vec) + `ref_embed[ref_idx]` (3,d) additive (5/27 PM ref-aug); no time PE | (C,F_p=10,T_p,d) |
| ❷ ×4 | per-electrode: temporal SA (RoPE θ=1e4) → freq SA → MLP; hard −∞ cross-electrode mask; 8 heads, MLP 4× | (C,F,T,d) |
| ❺ ×2 @{0,3} | 320 free latents (K=80×M=4, std 0.02); pool (elec,freq)→parcel 1:1/timestep; QK bias `log(support[e,p]+ε)`, ε=1e-2 | (320,T,d) |
| ❻ ×6 | time SA (RoPE) → parcel SA → MLP | (320,T,d) |
| ❼ | k PMA seed queries; downstream k=1, P3 k=50 @10 Hz | (k,d)→out |

Spatial: discrete parcel routing @❺; continuous geometry slot @❸g (empty for sEEG default). No additive PE.

## 4. Parameter budget (d=256, 8 heads)

| ❶ | ❷×4 | ❺×2 +latents | ❻×6 | ❼ | **Total** |
|---|---|---|---|---|---|
| 0.01M | 4.2M | 1.7M | 6.3M | 0.8M | **~13M** |

## 5. Recipe — 3-phase staged

| Phase | Trains | Frozen | Objective | Mask | Target |
|---|---|---|---|---|---|
| P1 Stage A | ❶❷ | — | `L_recon` Lvl B, MSE, UFO λ=1 | 80% inverse-block (t,f) | EMA-teacher latent, layer-avg |
| P2 Stage B | ❺❻ | Stage A | `L_recon` Lvl A, MSE | 30% random electrode (J1 sweep) | EMA-teacher Stage-B latents |
| P3 3a | gating MLP (~10k) | A,B,teachers | masked-latent MSE | — | fused teacher |
| P3 3b-warm | adapter (~0.7M) | +gating | Smooth-L1 | — | 50 buckets @10 Hz |
| P3 3b-unfrz | A (LR/10), B, adapter | gating,teachers | Smooth-L1 | — | 50 buckets @10 Hz |

Adapter: 1-layer PMA, K=50 queries. Teachers: Whisper-L8 + DINOv3, instance-norm per token.
EMA 0.99→0.9999. P3 optim: AdamW β=.9/.98 wd=.05, LR 3e-4, 2k warmup+cosine→1e-5/50k, batch 256.

Pretrain corpus P1+P2: SWEC 6,672 h · AJILE12 1,280 h · D-cohort 180 h · BT 43.5 h · PS+lex 7 h ≈ **8,180 h**.

## 6. Losses

- `L_recon_B = MSE(predictor(student,mask), StopGrad(EMA-teacher latent, layer-avg))`
- `L_recon_A = MSE(student latents, StopGrad(EMA-teacher Stage-B latents))`
- `L_distill = Smooth-L1(adapter[50], fused-teacher[50])` — K=50 @10 Hz syllable rate.

Rejected: per-frame 50 Hz · InfoNCE/contrastive · hybrid loss · adapter EMA · BLIP-2 Q-Former · `L_KoLeo`.

## 7. Eval & gate

Paths: A1 frozen+probe (headline) · A2 unfrozen+probe · B 2-layer MLP · C light fine-tune.
**Gate:** ≥ 0.667 CrossSession multi-class **AND** ≥ 0.628 CrossSubject binary AUROC, ≥4 tasks, ≤30M params.

## 8. Hyperparameters

d=256 · 8 heads · N=4 (sweep 2/4/6) · L=6 · 320 latents · 2 cross-attn @{0,3} · pre-norm LN · GeLU · MLP 4× · RoPE θ=1e4 · latent std 0.02 · ε=1e-2 (sweep 1e-4…1e-1) · P1 mask 80% · P2 mask 30%.

## 9. Ablations

**P0:** F-joint-tF · F-linear-STFT-38 · F-150Hz-cap · 1-/3-cross-attn · M-sweep · F10b · BiL-NoSpatial/MNI/Freq-Plain · J1/J2 · N-/L-/d-sweep · R-joint/staged · R-no-phase-3.
**P1:** F-depth-bias · F-additive-time-PE · freq-embed structure · R-frame/cosine/no-gating/no-warmup · filterbank shape.

---
Memory: post-v3 arch (2026-05-19) · 3-phase recipe (2026-05-18) · preproc (2026-05-12) · DK routing (2026-05-13) · prune (2026-05-12).
