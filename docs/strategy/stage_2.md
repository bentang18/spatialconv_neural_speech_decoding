# Strategy — Stage 2 (Phase 1.5)

Strategy layer of the triad, scoped to **Stage 2: in-sensor scaling on uECoG**.

- Objectives: `../objectives.md`
- Tactics: `../tactics.md`
- Stage index: `../strategy.md`
- Previous stage: `stage_1.md`

## Stage 2 recap (from objectives)

**Data:** up to 23 LH uECoG patients supervised (7 PS + up to 16 lexical, gated on Zac's quality assessment + FreeSurfer localizations landing); SSL on continuous uECoG corpus.

**Hypotheses:**
- **H2.1 (primary):** more patients → better LOPO warm-start on held-out patients.
- **H2.2 (primary):** SSL pretrain on the continuous corpus → better LOPO warm-start than supervised-only pretrain.
- **H2.3:** PS-pretrained encoder transfers to the lexical corpus (different task, broader phoneme inventory).

**Stage scope (narrowed 2026-04-21):** uECoG only. Cogan sEEG D-cohort (~60–80 speech-task subjects available on Box under `ECoG_Recon_Full/D*`) is deferred to Stage 3. Get lexical uECoG working end-to-end before spanning sensor types.

---

## Default architecture

**Frozen from Stage 1 close-out** (2026-04-20):

```
per_cell + partialconv + pe2d + hierarchical_atlas @ d=32, depth=3, pool=(4, 8)
```

Full pipeline in `stage_1.md §Default architecture`. Architectural ablation is paused until Stage-2 data lands. Two architectural questions become decidable at Stage-2 scale and are queued:

1. **Does `P_emb` earn its LOPO keep at scale?** Stage-1 T3.5 (noemb LOPO) showed atlas-mechanism inert at 4-core (−0.013 aggregate, inside ±0.020 noise). Re-test at N≥12 patients + SSL init.
2. **Does per-electrode (d=64) beat per-cell under scale?** Stage-1 T2.2 showed per_electrode+fourier edges per_cell+partialconv at d=64 (0.786 vs 0.795 pooled). Re-test at Stage-2 cohort.

Neither question drives the default until Stage-2 scoreboard evidence lands.

---

## Field context — cross-subject FM architectures (2025–2026)

How current EEG/iEEG/fMRI FMs handle electrode + anatomy heterogeneity. Frames the Stage-2 `noemb` retest: whether v14's atlas-token bottleneck is the right contrarian bet on iEEG.

| FM | Modality / scale | Token unit | Spatial encoding | Atlas? |
|---|---|---|---|---|
| REVE (El Ouahidi, Oct 2025) | scalp EEG, 60k h / 25k subj | per-channel patch | fixed 4D Fourier on (x,y,z,t) + learned linear adapter | none |
| DIVER-1 (Han, Dec 2025) | iEEG (5.3k h) + EEG (54k h) | per-channel patch (1 s or 0.1 s @ 500 Hz, 3-layer CNN) | (a) fixed 3D MNI sinusoidal PE (PopT-style) + (b) STCPE — conditional PE from sliding MOIRAI attention, channel-permutation-equivariant; modality + subtype embeddings | none (modality tags only) |
| DeeperBrain (Wang, Mar 2026) | scalp EEG, 17k h | (ch × 1 s) patch | volume-conduction kernel exp(−D_ij/τ), τ learnable; row-normalized convex mixing of electrode coords | none |
| **BarISTA (Oganesian, NeurIPS 2025)** | **sEEG, 10 subj / 29.2 h** | **per-channel patch (250 ms)** | **additive learnable parcel-lookup PE (one embedding per Destrieux parcel); JEPA SSL with parcel-masking** | **yes (Destrieux 148-hard)** |
| MV-BrainFM (Xu, Mar 2026) | fMRI, 20k subj / 17 datasets | **per-ROI** | learned Fourier on MNI ROI centroid + atlas-specific linear projection W^a + per-head learnable Gaussian distance bias | yes (cross-atlas consistency loss; `P_emb`-style learnable supernodes via Node Alignment) |
| **v14 (this work)** | uECoG / sEEG | **per-cell** (≈ per-electrode); hierarchical readout pools to **per-parcel** | per-cell embedding + atlas `P_emb[15,d]` mixed in via Brainnetome PM support weights; pe2d on cell grid; hierarchical parcel attention (`pool=(4,8)`, depth 3) at readout | yes (soft BNA Tier-1 15 on fsaverage strict snap-to-pial) |

Three implications:

1. **Per-electrode token + coord-PE is the field default on iEEG/EEG; atlas-aware tokens are the rare alternative.** REVE, DIVER-1, DeeperBrain, PopT (and the EEG cluster) all keep tokens per-electrode and bake anatomy into coord-PE or attention bias. Only **BarISTA** (iEEG) and **MV-BrainFM** (fMRI, born parcel-tokenized) add an explicit atlas-keyed embedding. v14 sits in the BarISTA camp — atlas-aware via `P_emb` mixed into per-cell tokens, not pure coord-PE. The Stage-2 `noemb` arm at N≥12 + SSL init is the decisive test: if `P_emb` is still inert (Stage-1 4-core: −0.013 aggregate, inside ±0.020 noise), v14 should fall back to per-electrode + coord-PE (REVE/DIVER-1/PopT-style); if it earns its keep, v14 + BarISTA validate atlas-aware iEEG against the field default.
2. **Coord-only PE is fragile to anatomy distribution shift.** DIVER-1 reports absolute 3D PE hurts performance under adult→pediatric shift. uECoG cohorts have analogous variability (head size, array placement, sulcal geometry) across patients; this is the regularization argument for atlas-keyed embeddings. **Neuroprobe cross-subject leaderboard** (binary AUROC, train on S2/trial-4 only): Linear Lap+spec **0.539** > Linear spec 0.528 > BrainBERT-untrained 0.527 > **PopT 0.526** > BrainBERT-trained 0.522 > raw 0.510. Linear baseline tops every trained iEEG FM that uses the per-electrode + coord-PE recipe; BarISTA was not evaluated.
3. **Distance-aware spatial priors are the rising trend** — beyond raw coord-PE. DeeperBrain mixes electrode coordinates via a learnable distance kernel before the PE projection (volume-conduction-aware spatial encoding). MV-BrainFM adds a per-head learnable Gaussian distance bias to attention logits. Both encode "closer electrodes interact more strongly," which v14's hierarchical parcel attention encodes discretely — parcels are the distance buckets.

**Adjacent precedent worth tracking** (not in table):

- **Population Transformer / PopT** (Chau et al., ICLR 2025, arXiv 2406.03044) — population-level head that stacks on a frozen per-channel temporal encoder (default BrainBERT). Per-channel tokens summed with sinusoidal PE on 3D anatomical coordinates (Left/Posterior/Inferior axes for iEEG, XYZ for EEG, plus an ensemble-identifier scalar); CLS token aggregates across electrodes. Neuroprobe **cross-subject 0.526**, cross-session 0.566, within-session 0.545. Architectural ancestor for v14's per_cell variant if it drops the parcel embedding.
- **FunctionalMap** (Javadzadeh, ICLR 2026 preprint) — counter-thesis: 32-D Siamese functional embedding per electrode beats MNI coords on masked recon (subcortical sEEG only). If function > anatomy holds on cortex, atlas-first is wrong; treat as Stage-3+ ablation, not Stage-2 contender.
- **Neuro-MoBRE** (Wu et al., Aug 2025) — regional MoE experts as alternative to soft parcel embedding.
- **Cross-subject EEG cluster** (LUNA, HEAR, DIVER-0, NeurIPT, UNI-NTFM, BrainPro, CSBrain) — all converge on REVE's per-electrode + coord-PE recipe with minor heterogeneity tricks. Same conclusion as REVE; no separate row needed.
- **Recent intracranial cross-subject decoders** (not FMs): `2603.12628` unified prod+perception (Mar 2026), `2026.02.27.708564` "closest published analogue" (Feb 2026), `2026.01.12.699110` cross-brain transfer (Jan 2026), `2411.10458` sEEG electrode variability (NeurIPS 2024). Watch list for Stage-3 sensor-transfer.

References live in `memory/reference_*` (REVE, DIVER-1, DeeperBrain, MV-BrainFM, BarISTA, FunctionalMap, Neuro-MoBRE in memory; PopT and the EEG cluster catalogued in `pastwork/NEWPAPERS_catalog.md` only).

---

## Frozen Stage-2 contract

Changes vs Stage 1 are marked `[CHANGED]`. Everything else is inherited.

| Item | Contract |
|---|---|
| Spatial base | fsaverage strict snap-to-pial (inherited). |
| Support cache | `data/atlas/support_cache_v2c_snap/<pt>_support_tier1.csv` (inherited). |
| Tier-1 parcel set | 15 LH Brainnetome parcels (inherited). **Revisit:** re-derive argmax_wins ≥ 10 rule at Stage-2 LH cohort (may add temporal-lobe parcels for lex IFG/STG coverage). |
| Loader — per-phoneme | Phoneme-level `.fif` (`derivatives/epoch(phonemeLevel)(CAR)/...`), `[-0.15, 0.5)` s, 130 samples @ 200 Hz (inherited). |
| Loader — continuous **`[NEW]`** | Raw `.fif`, non-epoched, ~1 s window with 50% overlap, per-channel z-score on pre-auditory baselines (recipe A, verified 2026-04-18). For SSL pretrain only. |
| Hemisphere | LH only (inherited). |
| Label alphabet **`[CHANGED]`** | **28-ARPABET** (lexical phoneme superset). PS contributes 9/28 classes; lex covers full 28. S78 covers 27/28 (missing AE, UH). Readout head is a single 28-way Linear. |
| Loss **`[CHANGED]`** | Flat per-phoneme CE over 28 classes, with **inverse-sqrt class-weighting** to balance PS (over-represented at 9 tight classes) vs lex (sparser, 28 classes). Revisit to focal CE (γ=2) if class-balance probe flags PS-favoring. |
| Eval **`[CHANGED]`** | Slot-averaged PER + per-phoneme PER over 28 classes. Exhaustive AR decode uses the class subset present in the trial's task (9³ for PS, 28³ for lex). Monitor PS-acc and lex-acc separately — treat convergence to the majority class as failure. |
| CV | grouped-by-token, same-patient-per-batch (inherited). |
| Artifact channels | hard-exclude only (inherited). |
| Normalization | upstream `productionZscore_highgamma` for supervised; continuous SSL uses recipe A. |
| Channel bridge | per-patient map as in Stage 1 for PS; lex patients use Map 3 or Map 6/7/8 depending on array. Rebuild `channel_maps/<pt>_channelMap.mat` lookup for each lex patient as recons land. |

---

## Patient scope (Stage 2)

Stage-2 is a growing cohort. Scope is gated on FreeSurfer recons landing from Zac's pipeline.

### uECoG — PS cohort (inherited)

| Role | Patients | Status |
|---|---|---|
| Stage-1 core | S14, S26, S33, S62 | LH, full Phase-1 scope. |
| Stage-1 extended | S16, S23, S39 | LH, in 7-LH pooled validation. |
| Deferred to Stage 3 | S22, S58 | RH. Join with sEEG. |
| Excluded | S32, S57 | out of program. |

**PS Stage-2 active:** S14, S16, S23, S26, S33, S39, S62 (7 LH).

### uECoG — lexical cohort (new)

16 lex patients on the spreadsheet (S41, S45, S47, S51, S52, S53, S55, S56, S63, S67, S71, S73, S74, S75, S76, S78, S81). Status per Box audit 2026-04-21:

| Role | Patients | HG sig channels | Localization status |
|---|---|---|---|
| **Ready now** | S76, S78, S81 | 117/256, 147/256, 157/256 | FreeSurfer recon on Box (ECoG_Recon). Projectable today. |
| **High-value, recon pending** | S73, S75, S74 | 210/256, 227/256, 156/256 | Zac's "best lex" list. Recon pipeline backlog. |
| **Med-value, recon pending** | S56, S67, S41, S53, S47, S45 | 186/256, 159/256, 146/256, 114/256, 108/128, 101/128 | DICOMs on Box for S41/S51/S53/S55; nothing for S45/S47/S56/S63/S67. |
| **Questionable** | S52, S55, S71 | 37/256, 100/256, 47/256 | S52 missing raw in BIDS (Zac checking); S71 incompatible events.tsv (Zac checking); S71 HG too weak for supervised. |
| **Drop** | S51, S63 | 1/128, 62/256 | HG too weak for supervised use. Keep for continuous SSL corpus only. |

**Stage-2 supervised ceiling (best case):** 7 PS + 10 lex ready (S41, S53, S56, S67, S73, S74, S75, S76, S78, S81) = **17 LH**.

**Stage-2 continuous SSL corpus:** all 27 disjoint uECoG patients (11 PS + 16 lex) = 6.79 h raw regardless of localization (SSL does not need coordinates).

### Growth cadence

| Cohort stop | Patients | Trigger |
|---|---|---|
| **7-LH (inherited)** | 7 PS | Stage-1 baseline; re-run under T3.1 default. |
| **10-LH** | + S76, S78, S81 | Projectable today, no Zac dependency. |
| **17-LH (target)** | + S41, S53, S56, S67, S73, S74, S75 | Zac's localization pipeline landing 7 recons. |
| **19-LH (stretch)** | + S45, S47 | Pipeline runs beyond S73; S45/S47 have DICOMs only. |

Each cohort stop produces a scoreboard row (pooled joint + LOPO warm-start on the added patient) before proceeding to the next.

---

## Scoreboard

Two metrics on every arm:
1. **Joint pooled PER** — one model on all patients in the cohort; each patient's held-out fold on the shared model.
2. **LOPO warm-start PER** — pretrain on `N−1`, finetune on held-out's fold-train.

**Stage-2 scoreboard is the cohort-growth curve.** The shape of pooled-vs-LOPO as N grows is H2.1 + H2.2 evidence; absolute numbers are less informative than the deltas.

### Baselines

| arm | cohort | pooled PER | LOPO mean | source |
|---|---|---|---|---|
| Stage-1 default (old, pe2d_frozen + D1) | 7-LH PS | 0.833 ± 0.060 | not run on 7-LH | job 45793090 |
| Stage-1 default (T3.1) | 4-core PS | 0.765 ± 0.042 | 0.788 ± 0.014 | job 45798311 |
| Stage-1 default (T3.1) | 7-LH PS | **pending** | **pending** | re-run under T3.1 before Stage-2 ablation, save ckpt for SSL warm-start |

### Stage-2 scoreboard template

Filled as cohort grows. Each row = one cohort stop under the frozen T3.1 architecture.

| cohort | N | pooled PER | LOPO mean | ΔpooledN−N₀ | ΔLOPON−N₀ | H2.1 verdict |
|---|---|---|---|---|---|---|
| 7-LH PS (T3.1) | 7 | pending | pending | 0 | 0 | reference |
| 10-LH (+S76/S78/S81) | 10 | | | | | |
| 17-LH target | 17 | | | | | |

**H2.1 gate:** LOPO mean improves monotonically as N grows. A non-monotonic dip flags corpus heterogeneity — triage by the held-out patient's pooled-joint performance first (data issue, not model issue).

### SSL ablation (H2.2)

Triggered once 10-LH supervised arm is on the board. Uses the continuous corpus for pretraining.

| arm | pretrain corpus | pretrain objective | finetune | pooled PER | LOPO mean | gate |
|---|---|---|---|---|---|---|
| scratch | — | — | 17-LH supervised | — | — | reference |
| SSL → supervised | 27-pt continuous uECoG | **TBD — see below** | 17-LH supervised | | | H2.2: ΔLOPO > 0 pp vs scratch |

**SSL objective selection (separate Stage-2 ablation):** candidate stack from memory-file literature audit, ordered by expected transfer strength:
1. **Multi-domain reconstruction** (DIVER-1, Han 2026) — time + FFT + STFT masked reconstruction. Temporal SSL validated by Jiang (2025) at heterogeneous scale.
2. **Frequency-domain RMAE** (Neuro-MoBRE, Wu 2025) — region-masked reconstruction in DFT space.
3. **Next-segment prediction** (DeeperBrain, Wang 2026) — auxiliary PLV + sample-entropy prediction.

Spatial-only masking objectives (BarISTA-style) are *not* candidates — Jiang's co-smoothing result shows they flatten on heterogeneous data. See `memory/reference_jiang_heterogeneity_2025_05.md`.

Pick one SSL objective first; stack a second only if the first lands.

### Cross-task transfer (H2.3)

After H2.1 arm lands. PS-pretrained encoder → finetune on lex cohort only:

| arm | pretrain cohort | finetune cohort | PS-acc | lex-acc | gate |
|---|---|---|---|---|---|
| scratch (lex) | — | 10-lex ready | reference | reference | reference |
| PS→lex | 7 PS | 10-lex ready | should hold | should match-or-beat scratch | H2.3: ΔLOPO ≥ 0 |

Anchor patient = **S73** (Zac: best lex, PS-aligned CCA p1 = 0.351 in Zac's baselines).

### Gate thresholds (inherited from Stage 1 with Stage-2 additions)

- Pooled advance: PER < 0.800 on PS-subset of the 28-class head; < 0.850 on lex-subset (broader class space).
- LOPO promote: mean across held-out cohort improves by ≥ seed-noise (±0.020) vs scratch.
- **[NEW] Class-balance probe:** PS-acc / lex-acc ratio within 1.25× of prior (computed from train-set class frequencies). Ratio > 1.25× → model is collapsing to PS-only subset; rebalance loss.

---

## Rejected paths (Stage 2)

- **9-class PS-only head** — superseded by 28-ARPABET joint head. Zac's guidance: output space = full lex phoneme inventory, PS contributes a subset.
- **Cogan sEEG D-cohort as Stage-2 scope** — deferred to Stage 3. Sensor transfer (uECoG → sEEG) is its own hypothesis (H3.1); do not mix with in-sensor scaling (H2.1).
- **Learned per-patient calibration in Stage 2** — belongs to Phase 2 / Stage 3. Stage-2 keeps fixed-atlas calibration (Stage-1 contract).
- **External-lab ECoG (Flinker, Chang, AJILE12)** — Stage 4. PI access is the bottleneck, not a Stage-2 gate.
- **Per-session / per-patient input-layer params** (Boccato / Levin style, ~66k–262k per session) — explicit anti-goal for Stage 2. Program hypothesis commits to zero per-patient params; the whole point is that `P_emb` + fsaverage anchoring carries cross-patient calibration.

---

## Discipline

Three rules for Stage 2:

1. **Cohort scoreboard is the only architecture gate until 17-LH lands.** No architectural ablation (P_emb re-test, per_electrode d=64, SSL objective selection) runs until the T3.1 default has a full growth curve from 7 to 17 patients. The data is what speaks, not more variants at a single cohort size.
2. **Every cohort stop reports both protocols.** A pooled win at N=17 without the matching LOPO curve is not Stage-2 evidence.
3. **Lex supervised arm must not collapse to PS.** The 28-ARPABET head creates a class-imbalance trap (PS overrepresented by ~2× raw trial count after lex supervised expansion, and PS has tight 9-class support). Monitor per-cohort accuracy; treat PS-acc ≫ lex-acc as a training bug, not progress.

Stage 2 → Stage 3 trigger: H2.2 confirmed (SSL pretrain → LOPO improvement ≥ 0.020 pp over supervised-only at 17-LH), OR H2.1 confirmed and H2.3 evidence is strong enough to justify sensor-transfer hypothesis testing.
