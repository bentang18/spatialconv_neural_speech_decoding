# Neural Decoding from Stereotactic EEG: Accounting for Electrode Variability Across Subjects

**Authors:** Georgios Mentzelopoulos*, Evangelos Chatzipantazis, Ashwin G. Ramayya, Michelle J. Hedlund, Vivek P. Buch, Kostas Daniilidis, Konrad P. Kording, and Flavia Vitale*
**Venue/Year:** NeurIPS 2024 (38th Conference on Neural Information Processing Systems)
**Affiliations:** University of Pennsylvania, Stanford University, Archimedes / Athena RC
**Project page:** https://gmentz.github.io/seegnificant
**Code:** To be released (stated in paper)
**Contact:** gment@upenn.edu, vitalef@pennmedicine.upenn.edu

## TL;DR

Introduces **seegnificant**, a multi-subject sEEG framework for behavioral decoding (reaction time regression) that handles variable electrode number/placement across 21 subjects. Per-electrode 1D Conv tokenization → alternating temporal self-attention + spatial self-attention (×L blocks) → MLP compress → per-subject regression heads. RBF positional encoding on MNI coordinates stamps spatial identity before spatial attention. Multi-subject training (R²=0.39) beats single-subject (R²=0.30) by ΔR²=+0.09. LOO transfer+finetune achieves R²=0.38. Critical ablation finding: **per-subject heads are the most important component (ΔR²=-0.18), spatial attention is #2 (ΔR²=-0.10), but PE barely helps (ΔR²=-0.02, p=0.73, NOT significant).**

## Problem & Motivation

sEEG electrodes are placed by clinical need, not experimental design. Each subject has 3–28 electrodes at unique brain locations. No spatial correspondence exists between subjects — you can't align electrode indices across people. Prior sEEG decoding work (Angrick 2021, Petrosyan 2022, Wu 2022/2024, Meng 2021) was limited to single-subject models (≤12 subjects). The authors aim to build the first scalable multi-subject sEEG decoder.

## Method

### Architecture

```
Raw sEEG (E electrodes × T_trial timepoints, 400 Hz broadband)
  │
  ├─ Per-electrode 1D Conv tokenizer: K=2 kernels across time
  │   → z_e ∈ R^{T_trial×K} per electrode → BN → avg pool → z_e ∈ R^{T×K}
  │   T = 105 temporal tokens, K = 2 features per token
  │   Conv weights SHARED across all electrodes and subjects
  │
  ├─ Temporal self-attention (per electrode independently):
  │   z_e ∈ R^{T×K} → standard transformer block (pre-norm, FFN)
  │   Parallelized across electrodes. Output: z_int ∈ R^{E×T×K}
  │
  ├─ Spatial positional encoding (RBF on MNI coordinates):
  │   For each electrode: MNI (x,y,z) → RBF(μ, σ²) → concat → Linear → R^K
  │   μ ∈ {-90,-70,...,50,70}, σ² ∈ {1,2,4,...,64}
  │   PE added to latents: z_int += PE
  │
  ├─ Spatial self-attention (per timepoint independently):
  │   z_t ∈ R^{E×K} → standard transformer block
  │   Parallelized across timepoints. Output: z_l ∈ R^{E×T×K}
  │
  │  [Temporal + spatial blocks alternate ×L layers]
  │
  ├─ MLP compress: unroll E×T×K → R^d (d ≪ E·T·K)
  │   Compresses variable-size spatiotemporal latent to fixed-dim
  │
  └─ Per-subject regression head: MLP(R^d → R^1)
      2,081 params per subject. Predicts trial-wise reaction time.
```

### Key Design Choices

1. **Factored attention (not joint 2D):** Separate temporal and spatial self-attention, alternating. Reduces O((E·T)²) to O(T²) + O(E²) per block. 5.5× faster than joint 2D variant, +0.06 R² better (Table 1).

2. **Per-electrode tokenization:** 1D Conv operates on each electrode independently — no spatial mixing before attention. This handles variable E naturally (attention is set-based). Avoids 2D conv which would blend randomly-placed sEEG electrodes.

3. **RBF positional encoding:** Gaussian kernels centered at regular intervals along each MNI axis. Projects to K dims and adds to latents before spatial attention. Encodes brain location, not electrode index.

4. **Per-subject heads:** Shallow MLP per subject maps shared features → prediction. Accounts for individual differences in behavioral response profiles. The model's most critical component.

### Signal Processing

- Bipolar re-referencing (adjacent contacts on same shank) → virtual electrodes at midpoint locations
- Line noise exclusion: spectral power ratio 58–62 Hz / 18–22 Hz, reject if SNR > 1
- Electrode selection: HGA SNR (Paraskevopoulou 2021) — bootstrap randomization test, BH-FDR correction, α=0.05
- Selected broadband voltage traces z-scored within electrodes, downsampled to 400 Hz
- Window: [0, 1500] ms after stimulus color change
- 2 subjects rejected (0 or 1 significant electrodes) → 21 final

### Training

- AdamW, lr=10⁻³, β₁=0.5, β₂=0.999
- 1000 epochs, step LR decay (0.5 every 200 epochs single-subj, 0.9 every 100 epochs multi-subj)
- Huber loss (MSE for finetuning)
- Batch size: 64 (single), 1024 (multi-subject)
- 70/15/15 train/val/test split, 5 random splits averaged
- Compute: AMD EPYC 7502P + Nvidia A40 (45 GB). Single-subj: ~5 min, multi-subj: ~1 hr

## Data

- **N = 21 subjects** (23 recorded, 2 rejected), 29 recording sessions (some subjects multi-session)
- **3,600+ behavioral trials**, 100+ electrode-hours
- **3–28 electrodes per subject** (after selection)
- **Task:** Color-change detection — visual stimulus changes color after variable foreperiod (500 or 1500 ms), subject presses button. Goal: decode trial-wise reaction time from sEEG
- **Electrodes:** Ad-tech stereo EEG (1.1mm diameter, 4 contacts, 5mm spacing). One subject also had subdural grid (4mm contacts, 10mm spacing)
- **Recording:** Natus system, 512 or 1024 Hz
- **Electrode locations:** MNI coordinates via Talairach transform. Span cortical, subcortical, and deep structures
- **Population:** Epilepsy patients (pharmacoresistant), ages 16–57, 13F/8M

## Key Results

### Single-Subject vs Multi-Subject

| Condition | R² (per-subject, mean±sem) |
|---|---|
| Single-subject (21 separate models) | 0.30 ± 0.05 |
| Multi-subject (1 shared model, per-subj heads) | **0.39 ± 0.05** |
| Multi-subject + per-subject finetuning | **0.41 ± 0.05** |
| Multi-subject (pooled test R²) | 0.54 ± 0.01 |

Multi-subject training: **ΔR²=+0.09** over single-subject (per-subject), **+0.11** with finetuning.

### Transfer to New Subjects (LOO)

- **LOO pretrained model R² = 0.48 ± 0.006** (pooled across all trials, all subjects)
- **LOO + finetune R² = 0.38 ± 0.05** (per-subject average)
- Transfer+finetune vs single-subject: **ΔR²=+0.08** (per-subject)
- Transfer+finetune vs multi-subject: ΔR²=-0.01 (nearly matches full multi-subject)
- Finetuning protocol: first 400 epochs train only regression head (2,081 params), then 600 epochs train all params

### Baselines Comparison (Figure 5)

Seegnificant outperforms all baselines in single-subject (ΔR²≥0.03), multi-subject (ΔR²≥0.12), and transfer (ΔR²≥0.11) settings. Baselines: Wiener Filter, Ridge, Lasso, XGB, MLP, CNN+MLP.

### Ablation Study (Section 4.6.1)

| Ablation | Per-Subject R² | ΔR² from full |
|---|---|---|
| **Full model** | **0.39 ± 0.05** | — |
| -Temporal attention (AT) | 0.38 | **-0.01** (trivial) |
| -Positional encoding (PE) | 0.37 | **-0.02** (p=0.73, **NOT significant**) |
| -Spatial attention (AS) | 0.29 | **-0.10** (substantial) |
| -Subject heads (RH) | 0.21 | **-0.18** (devastating) |

**Importance ranking:** Subject heads >> Spatial attention >> PE ≈ Temporal attention.

### PE Comparison (Table 3, Appendix A.4.3)

| Positional Encoding | Per-Subject R² |
|---|---|
| Vaswani sinusoidal (sequence position) | 0.16 ± 0.04 |
| MNI-Fourier | 0.39 ± 0.05 |
| MNI-RBF (theirs) | 0.39 ± 0.05 |

Sequence-order PE catastrophically fails (R²=0.16) — electrode identity ≠ spatial position. Both MNI-based approaches perform identically. But neither significantly outperforms no-PE at all.

### Factored vs Joint 2D Attention (Table 1)

| Model | Per-Subject R² | Train time (mins) |
|---|---|---|
| Joint 2D attention variant | 0.33 ± 0.05 | 141.6 ± 3.23 |
| **Factored (theirs)** | **0.39 ± 0.05** | **25.7 ± 0.03** |

Factored attention is both better (+0.06 R²) and 5.5× faster.

## Model Complexity

- **Total params:** 797,095
- **Shared backbone:** 753,394 (Conv tokenizer + temporal/spatial attention blocks + MLP compress)
- **Per-subject heads:** 43,701 total (2,081 per subject × 21 subjects)
- **Per-subject fraction:** 5.5% of total, but most critical component
- **Inference:** <10 ms on both CPU and GPU (real-time capable)

## Relevance to Cross-Patient uECOG Speech Decoding

### Directly applicable

1. **Factored attention validates our design.** v12 uses spatial cross-attention (electrode→VE) then temporal self-attention — a factored approach. Their result (+0.06 R², 5.5× faster vs joint 2D) confirms factored > joint for intracranial data with moderate electrode counts.

2. **Per-subject heads are critical (ΔR²=-0.18).** Our v12 currently budgets only 70 params/patient (Conv2d read-in bias) — 30× less than their 2,081/subject. This motivated A15 (per-patient input projection, ~4K/pt), A16 (per-patient output head, ~600/pt), and A17 (both, ~4.6K/pt) ablations. The diagnostic: if A15 alone helps, variability is in measurement; if A16 is needed, the "phoneme features are universal" assumption fails.

3. **Spatial attention is the #2 component (ΔR²=-0.10).** Self-attention across electrodes is the main mechanism handling heterogeneous electrode placement. A14 ablation tests seegnificant-style spatial self-attention (no VEs) vs our VE cross-attention.

4. **PE barely helps — calibrate expectations.** ΔR²=-0.02, p=0.73. Coordinate-based spatial encoding is NOT the magic ingredient. The spatial attention mechanism itself does the heavy lifting. However, their task (distributed reaction time networks) differs from ours (focal somatotopic phoneme classification) — position may matter more when the spatial scale of the relevant cortex is small (vSMC ~20×40mm) and functionally organized (tongue M1 ≠ lip M1).

5. **Broadband input, not HGA.** They use z-scored broadband voltage (not HGA). Their electrode selection uses HGA SNR, but the model sees broadband. We use HGA features — different signal characteristics but same general approach (select responsive electrodes, tokenize, attend).

### What doesn't transfer

1. **sEEG vs uECOG topology is fundamentally different.** sEEG = sparse 3D scatter (3–28 electrodes across whole brain, no local spatial structure). uECOG = dense 2D grid (63–201 electrodes, 1.3–1.7mm pitch, rich local structure). Their per-electrode tokenization is natural for sEEG (no grid to exploit); for uECOG, discarding the grid for independent tokens is a hypothesis, not a given. Conv2d on the grid may be strictly better for intra-patient spatial processing.

2. **Reaction time regression ≠ phoneme classification.** Reaction time is a single scalar per trial involving distributed brain networks. Phoneme classification is a 9-way categorical task involving focal somatotopic cortex. The relative importance of spatial attention, PE, and per-subject heads may differ substantially.

3. **Scale mismatch in electrode count.** 3–28 electrodes vs 63–201. O(E²) spatial attention at E=200 is 100× more expensive than at E=20. Their architecture is designed for sparse coverage; ours must handle dense arrays.

4. **No reconstruction loss, no SSL.** Their model works well with supervised regression alone. But they have 3,600+ trials × 21 subjects. We have ~1,600 trials × 11 patients with only ~46–178 per patient. Reconstruction loss may matter more in our low-data regime.

5. **No virtual electrodes.** They don't need VEs because spatial self-attention among real electrodes handles cross-subject alignment. Whether VEs add value over direct spatial attention (with PE) is exactly what A14 tests.

### Critical comparison

| Dimension | seegnificant | Our v12 |
|---|---|---|
| Modality | sEEG (depth, sparse 3D) | uECOG (surface, dense 2D grid) |
| Electrodes/subject | 3–28 | 63–201 |
| Subjects | 21 (29 sessions) | 11 |
| Trials | 3,600+ | ~1,600 |
| Task | Reaction time (regression) | 9-phoneme (classification) |
| Spatial scale | Whole-brain distributed | Focal vSMC (~20×40mm) |
| PE approach | RBF on MNI → add to latents | Fourier on ACPC/MNI → token embedding |
| Spatial mechanism | Self-attention among real electrodes | Cross-attention to virtual electrodes |
| Per-subject params | 2,081 (MLP head) | 70 (Conv2d bias) |
| Total params | 797K (753K shared) | ~112K |
| Grid exploitation | None (no grid) | Conv2d (exploits 2D grid) |
| Reconstruction loss | None | Yes (spatial interpolation) |

## Limitations

1. **Single task (reaction time)** — unclear if findings generalize to speech/phoneme decoding
2. **Broadband input at 400 Hz** — computationally expensive tokenization (T=105 tokens for 1.5s). Our HGA at 200 Hz is more compact
3. **No code released yet** (stated as forthcoming)
4. **PE ablation not significant** despite being a claimed contribution — authors acknowledge this honestly
5. **Electrode selection uses HGA SNR** — requires task-related activity, so electrode selection can't be fully unsupervised
6. **23→21 subjects after electrode selection** — 2 subjects lost entirely (0–1 significant electrodes), highlighting that sEEG coverage is not guaranteed

## Reusable Ideas

### 1. Factored temporal + spatial attention (validated)
Alternating T-then-S attention blocks. Better than joint 2D and faster. Already in v12 design (spatial cross-attention then temporal self-attention). Their result provides direct validation.

### 2. Per-subject capacity as systematic ablation target
Their 2,081 params/subject vs our 70 highlights a design space to explore. A15/A16/A17 ablation ladder tests input, output, and combined per-patient capacity.

### 3. Graduated finetuning for transfer
First 400 epochs: train only per-subject head (2,081 params). Then 600 epochs: unfreeze all. This progressive unfreezing prevents catastrophic forgetting of shared representations. Compare to Singh's freeze-backbone approach and Levin's source replay.

### 4. MLP compress as alternative to pooling
Instead of mean/max pooling over the electrode dimension, they use a learned MLP to compress the flattened E×T×K latent to a fixed-dim vector. This preserves more information than pooling but requires variable-length input handling (padding + masking for different E).

## Key References

- Azabou, M. et al. (2023) A unified, scalable framework for neural population decoding. arXiv:2310.16046.
- Paraskevopoulou, S.E. et al. (2021) Within-subject reaction time variability. *NeuroImage* 237.
- Ramayya, A.G. et al. (2022) Simple human response times are governed by dual anticipatory processes. bioRxiv.
- Le, T. and Shlizerman, E. (2022) SNDT: Modeling neural population activity with a spatiotemporal transformer. arXiv:2206.04727.
- Ye, J. and Pandarinath, C. (2021) Representation learning for neural data transformers. *Neurons, Behavior, Data Analysis, and Theory* 5(3).
