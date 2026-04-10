# Current Direction

Updated: 2026-04-08 (SSL simplified: JEPA L1 with temporal masking replaces 3-headed discriminative+JEPA hybrid. V-JEPA 2/2.1 recipe.)

## Active Priority: Intracranial Field Potential Foundation Model (v12)

**Design doc**: `docs/neural_field_perceiver_v12.tex`

**Architecture** (atlas-grounded multi-view reconstruction):
```
electrode HGA + MNI coord
→ Conv1d(1→d, k=10, s=10)           shared temporal binning
→ s^(i) ⊙ x + b^(i)                per-patient diagonal (128 params/pt)
→ + Fourier PE(MNI + Δ/ω)           directional spatial identity
→ VE cross-attention (L=16 atlas, distance bias)  maps N_i → 16 common space
→ [VE self-attn (16×16) → Temporal self-attn] × 2
→ Lightweight AR: pos attn → [z̄_k; emb(y_{k-1})] → Linear(2d→9) × 3 + beam search (52 tokens)
```

~175K total params (B=2), 182 per-patient (128 diagonal + 6 geometric + 48 VE offsets). Per-patient VE positions (atlas + leashed offset ≤15mm) adapt to individual somatotopy. Electrode self-attention is an ablation (A_elec_attn), not default — detailed spatial layout varies too much (R²=0.05-0.13) for shared electrode-level spatial processing to transfer. The transferable signal is temporal dynamics at VE resolution. Dual spatial encoding: distance bias (proximity) + Fourier PE (direction). Lightweight AR decoder (4.1K). 16 Brainnetome atlas positions (N=40; somatotopic motor/sensory subdivisions > HCP-MMP for speech). Full-trial input tmin=-0.2, tmax=1.0s (premotor + production). Works identically for sEEG, uECoG, and macro-ECoG.

**Paper direction**: First intracranial field potential (HGA) foundation model. Multi-view reconstruction framing: each patient is a camera view of a shared neural manifold; VEs are the template mesh; per-patient layers are camera calibration. BIT did this for Utah array spikes; nobody for HGA/field potentials.

## Key Design Choices (2026-04-06)

- **VEs over self-attention** — N=10 patients can't learn population-average anatomy from data. Atlas (Brainnetome, N=40; somatotopic subdivisions for speech) provides the spatial prior. Self-attention is an ablation (A_self_attn) testing whether the model can learn without the atlas.
- **Distance-biased cross-attention** — creates ~25mm soft receptive fields around atlas positions, naturally accommodating 15-25mm inter-patient anatomical variation. Learned per-head scale α_h.
- **Diagonal normalization over full affine** — impedance/gain = scale+offset, not rotation. 128/pt vs 4160/pt. Willett's full affine is for Utah array day-to-day drift; we have single-session surface electrodes.
- **Δ/ω hard-clamped** — 15mm translation, 0.15 rad rotation. Covers worst-case registration error.
- **Deformable atlas (ablation A_deform)** — per-patient per-VE position adjustments (+48/pt). Like 3D Morphable Model deformation. Tests nonlinear coordinate correction beyond global Δ/ω.
- **Modality-agnostic** — same VE atlas for sEEG and uECoG. Full weight transfer in SSL.
- **HGA ≠ spikes** — HGA is postsynaptic (inputs), spikes are action potentials (outputs). Different biophysics. Cannot pool across Utah arrays and surface/depth field potentials.

## Immediate Next Steps

1. **Request external chronic ECoG data (CRITICAL)** — Talk to Greg about PI-to-PI data requests:
   - **Flinker/Chen lab (NYU)**: 48 patients, ECoG speech production. Code public (Chen 2024, Nature Machine Intelligence). Data upon request with sharing agreement.
   - **Chang lab (UCSF)**: ~15-25 patients, high-density ECoG. Papers state "data available from corresponding author."
   - **Bouchard/Chang Figshare**: 3 patients, CV syllables, **public today** — download immediately for pipeline validation.
   - This is THE highest-leverage action. With external data: 2-stage pipeline (external SSL → per-patient supervised FT). Without: fallback 3-stage on CoganLab-only data.
2. **Nonlinear MNI normalization (HARD BLOCKER)** — talairach.xfm is a 12-param linear affine; residual nonlinear error is ~5-15mm. Ask Zac: do T1 MRIs exist? If yes, ANTs SyN could halve error. Do before Phase 0.
3. **HGA extraction pipeline** — 456 min raw EDF across 29 uECoG patients. Need CAR → filterbank → Hilbert → 200Hz. Blocker for fallback SSL pipeline.
4. **v12 implementation** — Implement VE cross-attention architecture. Run Phase 0 sanity check on 4 core patients.
5. **Brainnetome atlas VEs** — DONE (2026-04-06). 16 core ROIs selected (reachability-verified).

## Data Readiness (audited 2026-04-05)

**Electrode coordinates**: ACPC space, 11/11 patients. `build_electrode_coordinates()` handles both 128-ch (chanMap) and 256-ch (direct) paths.

**Artifact exclusion**: `exclude_artifacts=True` in `bids_dataset.py`. S39=20, S57=15, S58=37 channels excluded.

**Raw continuous data**: 456 min across 29 unique patients (13 PS + 17 Lexical). EDF at 2kHz. Need HGA extraction.

**sEEG speech data**: ~1000 min (25 patients) on Box. Same temporal masking SSL, same architecture.

**Patient selection**: Core: S14, S26, S33, S62. Excluded: S32 (no HG), S57 (52/256 sig). Extended: S16, S22, S23, S39, S58.

## Scalable Data Sources (all field potentials, all poolable)

| Source | Modality | Hours | Status |
|---|---|---|---|
| CoganLab uECoG continuous | uECoG | ~7.6h | Have EDF, need HGA extraction (~52% task cycles, ~48% overhead) |
| CoganLab sEEG speech | sEEG | ~16.7h | On Box, need to inventory |
| CoganLab sEEG other tasks | sEEG | ? | Ask Zac |
| Epilepsy monitoring (DABI, OpenNeuro) | sEEG/ECoG | 100s-1000s h | Public, need harmonization |
| NOT poolable: Utah arrays | Spikes | — | Different biophysics |

## Per-Patient Baseline (v12 must beat this)

**PER 0.734 ± 0.007** (S14, grouped-by-token CV, 3-seed). Per-phoneme MFA flat head + full recipe.
**Population: 0.825 mean** across 11 patients.

## Experimental Design (overhauled 2026-04-07)

Phased, ~43 experiments, ~108 runs. Each phase gates the next. See design doc Section 6 for full details.

- **Phase 0**: Sanity + temporal window + sEEG coverage inventory — sweep tmin∈{-0.5,-0.3,-0.2,-0.1,0.0} and tmax∈{0.6,0.8,1.0} on per-patient v12 S14 (~8 runs), then cross-patient sanity check. Also: compute VE reachability for CoganLab sEEG patients (≥3 patients must cover motor VEs to justify Stage 1 SSL).
- **Phase 1**: Scaling — N-scaling curve (1→2→4→7→10 patients) + sEEG cross-modality transfer
- **Phase 2**: Mechanism — controlled ablations + mandatory negative controls:
  - *Spatial*: E2.1 (learned latents vs atlas — KEY), A_self_attn, A_no_dist, A3, **A_deform** (per-patient VE position offsets, +48/pt), **A_elec_attn** (add electrode self-attn before VE — tests if local spatial preprocessing helps), **A_elec_attn_no_pe** (electrode self-attn with distance bias only, no Fourier PE — isolates smoothing from position-specific patterns)
  - *Negative controls*: A_shuffle_coord (permute MNI coords), A_random_atlas (random VE positions), A_no_auditory (remove auditory VEs), A_late_window (tmin=0.2s, production only)
  - *Per-patient*: A15, E2.6 (zero per-patient)
  - *Decoder*: A_no_ar
- **Phase 3**: Adaptation — training protocol (LP vs full FT vs gradual unfreezing) + calibration efficiency curve ({5,10,20,50,100,all} target trials)
- **Phase 4**: Diagnostics — per-patient residual analysis (diagonal magnitudes, Δ/ω, variance decomposition, cross-patient CKA) + VE interpretability (activation profiles, region probes, phoneme selectivity, attention maps, trajectory analysis)
- **Phase 5**: Hyperparameters — A_dim (d∈{32,64,128} — novel), A10 (B∈{1,2,3}), A_thresh (VE threshold {15,25}mm)
- **Phase 6**: SSL (gated on HGA extraction) — A18 (SSL vs scratch), A_mse (L2 vs JEPA L1), A_dense (V-JEPA 2.1 context loss), A_deep (hierarchical supervision), A_disc (+ temporal discrimination), A_spatial_mask (+ VE masking), A_rank, A_domain + SSL scaling curve ({1,2,4,8,16,50,100}h)
- **Phase 7**: Refinements — full-trial vs per-phoneme, cross-task data, joint attention, KL matching, TTO

## Practical Rules

- Active design = `neural_field_perceiver_v12.tex`.
- VE cross-attn + distance bias + Fourier PE is the default. Self-attention/Conv2d are ablations.
- Cross-task data (Duraivel 2025) included by default.
- Core patients: S14, S26, S33, S62. Extended: S16, S22, S23, S39, S58. Excluded: S32, S57.
- Always `exclude_artifacts=True`. Always grouped-by-token CV.
- All training on DCC. See `docs/dcc_setup.md`.
- Default regularization: E_Δω + E_smooth + standard dropout. CLIP/neural-audio contrastive is future work, NOT default. Contrastive/KL alignment losses are Phase 7, NOT default.
- uECoG SSL uses full continuous data (456 min) with content-aware weighting + segment-level sampling to prioritize task-engaged periods (~52% task cycles, ~48% overhead from awake craniotomy setup/breaks). Fallback: drop uECoG SSL entirely (sEEG SSL → supervised).
- If a doc references: electrode self-attention as default, fixed atlas VE positions as default (A_deform as ablation), Conv2d as default, full affine W·x+b, L=12 VEs, AR transformer decoder as default, B=1 as default, 4-stage pipeline, ~171K params, ~191K params, 134/pt, "only temporal layers transfer", the old 37-ablation flat table, CLIP as default SSL, a 9-term loss equation, Brainnetome N>200, ~35 experiments/~85 runs, ~39 experiments/~100 runs, ~41 experiments/~104 runs, ~42 experiments/~106 runs, ~44 experiments/~110 runs, "89% baseline", A_no_elec_attn, A_deform (now A_no_deform), "discriminative + JEPA hybrid" SSL, A_jepa_only, A_disc_only, swap detection, L_swap_detect, 3-headed SSL, or L2/MSE as JEPA loss — it's stale.
