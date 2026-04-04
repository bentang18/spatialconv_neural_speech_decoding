# NCA-JEPA Pretraining Design — Resolved Decisions

Companion to `2026-03-26-nca-jepa-pretraining-design.md`. 82 entries, numbered by addition order.

---

1. **"Synthetic grid-dynamics pretraining"** not "NCA pretraining." NCA is one generator family in a controlled ladder.
2. **Preserve spatial tokens through predictor** (ViT family): Mean-pool AFTER temporal modeling, not before. Conv-GRU family collapses spatial structure at the encoder — this is a deliberate architecture comparison axis.
3. **Local geometry PE**: Linear(3 → d) on (row_mm, col_mm, dead_fraction), not normalized (x,y) in [0,1] which discards physical scale and collapses distinct geometries. This encodes within-array spatial relationships only, not cross-patient anatomical position.
4. **Masked temporal spans as default pretext** (v10): Mask contiguous frames, predict from bidirectional context. Resolves the causal inconsistency — BiGRU is inherently bidirectional. Causal multi-step prediction is an ablation variant.
5. **Step-matched comparison**: A/B/C all get S_total optimizer steps. Residual data-entropy confound acknowledged — "step-matched" not "information-matched."
6. **Shared per-patch decoder**: Linear(d, patch_dim), position-agnostic — applied identically to all patch tokens regardless of grid size.
7. **Factorized space-time predictor** (ViT family): TimeSformer-style divided attention with separate Q/K/V per block. 197,760 params/layer, 395,520 total.
8. **Bidirectional attention at all stages** (v10): Masked-span objective requires bidirectional context. No pretraining→fine-tuning mask mismatch. Causal variants tested only in Ablation A.
9. **Flips + 180° default, 90° deferred**: 90° rotation swaps H/W on rectangular grids, requires geometry remapping.
10. **20Hz default, 40Hz as early ablation** (v12 update): 20Hz is evidence-backed — CE ablation showed 20Hz ≈ 40Hz on S14. 40Hz alive for pretraining/micro-event variants only.
11. **1x1 patches as pilot default for Family B** (v10): Family B's point is spatial-token preservation; 2x2 partially defeats that. 2x2 is the compute control, not the default.
12. **Inductive as primary evaluation**: Target patient excluded from Stage 2.
13. **Lexical as ablation, not default**: Different temporal statistics.
14. **Stage 3 default**: Freeze all except head + per-patient embed + predictor LayerNorms.
15. **FHN is a toy model**: "Excitable-medium toy dynamics" — qualitative geometric resemblance, no biophysical claim.
16. **Synthetic JEPA / latent target prediction is the first promoted follow-up** (v14): Observation-space reconstruction remains the default, but if shortcut-prone behavior appears under the locked triggers, JEPA is the immediate next branch before broadening the generator matrix.
17. **CE primary at Stage 3 — full-window headline, motor-focused secondary** (v13): 27-way head (3 pos × 9 phonemes) matches the 0.700 PER configuration. Full-window is headline primary; motor-focused is preregistered mechanistic secondary.
18. **Grouped-by-token CV**: All repetitions of same CVC/VCV token in same fold. Adaptive fold count per patient.
19. **Three architecture variants, pilot-first** (v13): Conv-GRU (~71K) as conservative control; conv-token GRU (~40K) as bridge isolating spatial preservation; ViT (~927K) as high-capacity spatial-token model. Conv-GRU runs first; conv-token GRU in Step 3b; ViT after signal confirmed.
20. **Generator ladder, not flat list**: Comparative hierarchy of inductive biases enables ordered comparison (not strictly causal attribution — confound acknowledged).
21. **Transfer-proxy model selection gate with guardrails** (v13): Probe every 5K steps, locked dev split, earliest-within-1SE rule, source-only sensitivity analysis.
22. **Real-data matching statistics** (v10): Temporal PSD (Welch), spatial covariance (exponential decay), amplitude distribution, event sparsity, trial-phase structure, cue-to-response delay. Source patients only.
23. **Anti-alias cutoff tied to target rate**: 0.8× Nyquist (8Hz for 20Hz, 16Hz for 40Hz). Not an independent smoothing parameter.
24. **Full-window headline, motor-focused secondary** (v13): Full-window [-0.5, +1.0s] is headline primary (validated at 0.700 PER). Motor-focused [-0.15, +0.6s] is preregistered mechanistic secondary. Divergence is a finding.
25. **Real dead-electrode templates**: Use actual mask patterns from known array layouts plus optional random dropout, not purely random.
26. **Shared decoder in Stage 2**: Per-patient decoders let reconstruction become patient-specific. Decoder is shared across patients.
27. **Top-k spatial pooling as first-round co-default** (v13): Mean-pool erases focal activation on dense grids. Top-k (parameter-free) runs alongside mean-pool in first round.
28. **Negative transfer mitigation**: Uniform source sampling default; similarity-weighted and loss-balanced are ablations motivated by LOPO failure.
29. **Single fixed dev patient for transfer-proxy** (v10): 1 patient held out, keeps headline N=7-11. Dev excluded from headline claims.
30. **Coordinate-augmentation consistency** (v7): Flips/rotations applied to grid images must also transform PE coordinate inputs.
31. **Local geometry ≠ anatomical position** (v7): Pitch-scaled coordinates encode within-array relationships. MNI-based encoding is a future ablation.
32. **Grouped-by-token CV not comparable to logged baselines** (v7): The 0.700 PER used first-phoneme stratified CV. Grouped-by-token is more rigorous but produces different PER.
33. **Event-frame pretext diagnostic** (v7): Report pretext MSE separately on event vs background frames. If event MSE plateaus, add event-weighted loss variant.
34. **Conv-token GRU is first-round architecture** (v13): Shared BiGRU per spatial position, d=64, ~40K params. No cross-position temporal interaction. Isolates spatial-token preservation from temporal-model-class change.
35. **Parameter counts require prototype verification** (v7): Hand-computed counts are error-prone. Build and count from prototype modules.
36. **Temporal hyperparameters in milliseconds** (v7): All frame-based specs in ms with derived frame counts at each target rate.
37. **Forced generators, not free-running** (v8): `x_{t+1} = f_theta(x_t) + u_t` where `u_t` is phase-dependent external drive.
38. **Cue-to-response delay is the dominant timing variability** (v8): RT ~300-800ms (log-normal), not 0-25ms onset jitter.
39. **Generators model execution dynamics, not whole trial biology** (v8→v14): Switching-LDS / wave / RD / FHN / NCA generators produce the autonomous component inside the execution phase only.
40. **Trial-calendar leakage requires active defense** (v13): Diagnostics alone insufficient (acoustic regression lesson). Full-epoch off-center crops + time-reversal augmentation attack the shortcut directly.
41. **Full-window headline, motor-focused secondary** (v13): Encoder/predictor sees full trial. Full-window readout [-0.5, +1.0s] is headline. Motor crop [-0.15, +0.6s] is preregistered secondary.
42. **Synthetic data does not simulate auditory drive** (v8): Synthetic sequences are planning → execution → decay. No auditory evoked response modeling.
43. **Synthetic sub-event templates are generic, not phoneme-specific** (v9): External drive uses random focal spatial patterns, NOT tied to real phoneme identity.
44. **Stage 2 sees full trials; transfer-proxy uses full-window readout** (v13): Stage 2 prediction uses full-trial windows. Transfer-proxy evaluates full-window readout (headline endpoint).
45. **Dead-fraction PE, not binary dead-position** (v9): With 2x2 patches, dead is continuous. PE input: (row_mm, col_mm, dead_fraction): Linear(3→d).
46. **Motor-focused readout crop is a new hypothesis** (v9): The 0.700 PER used full-window. Motor-focused crop is expected to help but untested.
47. **Masked temporal spans as default pretext** (v10): Bidirectional, compatible with both families without modification.
48. **Random-scaffold control (Method F) in first round** (v10): 3-event scaffold directly encodes task structure; random phase count is the critical control.
49. **Phase-rebalanced prediction loss uses excess error** (v12): Weight by excess error over phase-template baseline. Phase-rebalancing is the phase-aware variant, not default.
50. **Family A PE is N/A** (v10): After spatial collapse, no spatial tokens for PE. Spatial relationships implicit in Conv2d kernel positions.
51. **Pilot-first execution order** (v10): Conv-GRU first (simpler, faster, ~13x smaller), expand only after signal confirmed.
52. **Family A temporal projection is pointwise** (v11): Conv1d(256, d, k=1, s=1) is per-frame dimensionality reduction. Preprocessing handles temporal downsampling.
53. **Family A bottleneck-pretext mismatch is deliberate** (v11): Observation-space prediction through 64-dim bottleneck rewards coarse reconstruction. Deliberate conservative-control property.
54. **ViT patient adapter: Linear(1,128) is a known weakness** (v11): Per-patient conv stem (Conv2d, ~1.2K params) is first-round paired ablation.
55. **Pilot gate includes ViT spot-check with paired adapter + Method B** (v13): Step 4b runs ViT Method A × 2 adapters + Method B × best adapter. Resolves both adapter and pretraining confounds.
56. **Active anti-leakage: full-epoch off-center random crops** (v13): Load [-1.0, 1.5s] epochs, crop 30-40 frames with random offset. Removes entire phases from some training samples.
57. **CV split feasibility: per-position class coverage required** (v11): Each fold's training set must contain all 9 phonemes × 3 positions. Fallback: repeated random grouped holdout.
58. **Matching statistics: evoked + residual** (v11): Spatial covariance and PSD computed on both trial-averaged and single-trial residual data.
59. **Stage 2 is response-locked trial adaptation** (v13): Not pure dynamics SSL. Every sample is a speech trial; model never sees inter-trial silence. Weak supervision via response-onset timestamps.
60. **Method J (B-extended) automatic first-round follow-up** (v12): If A > B confirmed by bootstrap CI, run Method J (B with 2× steps) automatically.
61. **Three-bin temporal pooling as secondary probe** (v12): Fixed three-bin pooling always run alongside global mean-pool. Zero extra parameters. Tests temporal discriminability.
62. **Fixed split reuse across all methods** (v12): Token-group fold assignments generated once per patient, saved to JSON, reused by all methods.
63. **Patient-level aggregation defined** (v12): Atomic unit = mean PER across folds per patient per method. Wilcoxon on paired patient-level PERs.
64. **Label-permutation surrogate null** (v12): 100× per patient for headline A-vs-D comparison.
65. **Source sampling decoupled from dev patient** (v12): Similarity-weighted uses leave-one-out within source patients.
66. **Generator coarse calibration** (v12): Source-only one-time pass. Not optimization — prevents degenerate instantiation.
67. **20Hz is evidence-backed default** (v12): CE ablation showed 20Hz ≈ 40Hz. Burden of evidence on 40Hz.
68. **Full-epoch off-center random crops in Stage 2** (v13): Load [-1.0, 1.5s] epochs (50 frames at 20Hz), crop to 30-40 frames with random start offset. Crops starting at different phases genuinely disrupt the calendar.
69. **Time-reversal augmentation** (v13): 10% probability of reversing temporal order. Destroys causal phase ordering; if it hurts, model relies on phase structure.
70. **Masking ratio 40-60% default** (v13): Raised from 10-17%. Multiple non-overlapping spans (3-6). VideoMAE insight: aggressive masking critical for SSL on smooth signals.
71. **Conv-token GRU fully specified** (v13): Shared BiGRU(64,32,2L) per spatial position, d=64, ~40K shared params, local geometry PE. No cross-position temporal interaction.
72. **Top-k spatial pooling first-round co-default** (v13): k=16, parameter-free. Selects most active electrodes. Runs alongside mean-pool.
73. **ViT spot-check includes Method B** (v13): Disambiguates architecture from pretraining effect. If ViT-A >> ViT-B, pretraining helps; if ViT-A ≈ ViT-B, architecture matters.
74. **Method K: destroyed-dynamics control** (v13): Phase-randomized and time-shuffled variants preserve per-frame statistics, destroy temporal structure. Required for Claim 3.
75. **Transfer-proxy guardrails** (v13): 5K step probe frequency, locked dev split, earliest-within-1SE rule, source-only sensitivity analysis.
76. **Full-window is headline primary** (v13): Co-primary was rhetorically permissive. Full-window validated at 0.700; motor-focused is preregistered mechanistic secondary.
77. **JEPA escalation rule** (v13): Promote if event-frame gains don't survive execution-only decoding, patient-ID rises without phoneme-ID, or pretext MSE plateaus without transfer-proxy improvement.
78. **Content-collapse diagnostics** (v13): Output entropy, unigram-KL, stereotypy index. Catches concentrated distributions and stereotyped sequences that loss curves miss.
79. **Destroyed-dynamics as first-round control** (v13): Method K runs alongside A, C, F. If A ≈ K, benefit is statistics exposure, not dynamics. If A > K, temporal dynamics are load-bearing.
80. **Switching linear state-space generator is first-round default content** (v14): Added to the generator ladder between smooth AR and wave dynamics, and included in Method A's mixed structured synthetic pool from the first round rather than deferred.
81. **Synthetic nuisance scale is calibrated against raw residual HGA, not clamped renders** (v14): Visualization-only clamping such as zeroing [0, 2 SD] is for hotspot inspection only. Synthetic IID and correlated noise ranges are set from raw residual statistics so nuisance amplitude is a real SNR choice, not a rendering artifact.
82. **Implementation handoff now includes JEPA promotion gate** (v14): After the first-round raw-reconstruction pipeline, the next follow-up run is synthetic JEPA / latent target prediction if shortcut-prone behavior appears, before further widening the generator matrix or chasing smaller ablations.
