# Cross-Patient Speech Decoding: A Foundation Model Approach

**Ben Tang | Cogan Lab, Duke University | April 2026**

---

## The Problem

We have intra-operative uECoG recordings from ~10 patients during a non-word repetition task (52 CVC/VCV tokens, 9 phonemes). Each patient has a thin-film electrode array on left sensorimotor cortex --- 128 or 256 channels at 1.3-1.7mm pitch. The goal: decode the 3-phoneme sequence the patient is saying from neural activity.

The fundamental challenge is that **every patient's array sits in a different place**. Arrays are offset 15-25mm from each other in standardized brain coordinates, with variable rotation. Channel 47 in Patient A records from a completely different patch of cortex than channel 47 in Patient B.

This means we can't simply pool data across patients --- the spatial correspondence doesn't exist at the electrode level. Current approaches either train separate models per patient (missing shared structure) or use linear alignment methods like PCA+CCA (Spalding 2025, 31% balanced accuracy) --- a reasonable first pass, but one that doesn't exploit the spatial correspondence between arrays.

**What we want:** A single model trained on all patients that outperforms individual per-patient models, and that can be quickly calibrated to a new patient with minimal data.

## The Insight

Think of each patient's electrode array as a **camera pointed at the brain from a slightly different angle**. Each camera sees a partial, noisy view of the same underlying scene --- the neural dynamics of speech production. The scene is shared; the viewpoints differ.

In computer vision, multi-view reconstruction solves exactly this: given images from different cameras, recover the 3D scene. The recipe is:

1. **Calibrate each camera** (intrinsics/extrinsics)
2. **Define sample points** in 3D space where you want to reconstruct
3. **Aggregate information** from whichever cameras can see each point

We do the same thing with brains:

1. **Per-patient calibration** --- a small set of learned parameters (182 per patient) that absorb differences in signal gain/impedance and electrode registration error. This is the "camera calibration."

2. **Virtual electrodes (VEs)** --- 16 fixed positions in standardized brain space, chosen from a neuroanatomical atlas (Brainnetome, derived from 40 healthy subjects). These positions correspond to known speech-relevant regions: tongue motor cortex, face motor cortex, Broca's area, auditory cortex, etc. These are the "3D sample points."

3. **Cross-attention** --- each virtual electrode "looks at" the real electrodes nearby and extracts a weighted summary. Electrodes close to a VE contribute more (distance bias), and the model learns *what* to extract from them (shared attention weights). This is the "multi-view aggregation."

The result: every patient's variable-count, variable-position electrode array gets mapped into the **same 16-dimensional functional space**. A tongue motor cortex VE always represents tongue motor cortex activity, regardless of which patient's electrodes contributed to it.

## Why This Should Work

Three lines of evidence:

**1. Speech motor cortex has conserved organization across people.** The gross layout --- lips, jaw, tongue, larynx arranged along the cortical surface --- is the same in every human (Breshears 2015, n=33; Bouchard 2013; Roux 2020). The *ordinal* arrangement is invariant; the *absolute positions* vary by 5-15mm. Our architecture handles this: the atlas provides the average positions, and per-patient offsets (leashed to 15mm) adapt to individual anatomy.

**2. Multiple independent groups have converged on this design.** At least 5 groups have independently arrived at Perceiver-style cross-attention bottlenecks for multi-patient intracranial data: MIBRAIN (Wu 2025, sEEG, cross-patient consonant decoding), BarISTA (Oganesian 2025, NeurIPS, showing atlas-level encoding beats channel-level by 8-10 percentage points), Charmander (Mahato 2025, NeurIPS workshop), POYO (Azabou 2023, NeurIPS, motor decoding across 7 monkeys), and Population Transformer (Chau 2025, ICLR). None has applied this to uECoG speech, and none combines atlas-grounded positions with per-patient adaptation --- but the convergence validates the core mechanism.

**3. Scale without spatial mechanism fails.** The Brant family of models (Zhang 2023, NeurIPS) trained on up to 40,000 hours of intracranial data with up to 1 billion parameters --- and fails at cross-patient speech decoding. NDT3 (Ye 2025) used 2,000 hours with 350M parameters and found cross-subject transfer fundamentally limited by sensor variability. These are the clearest negative results in the field: raw scale is insufficient. You need an explicit mechanism to handle electrode placement variation.

## The Architecture (Brief)

```
Raw HGA signal (per electrode, 200 Hz)
  --> Temporal binning (Conv1d, 50ms windows)
  --> Per-patient gain/offset normalization (128 params/patient)
  --> Add spatial identity (Fourier encoding of MNI coordinates)
  --> VE cross-attention (16 atlas positions, distance-biased)
  --> 2 blocks of [spatial self-attention + temporal self-attention]
  --> Autoregressive decoder: predict phoneme 1, then 2, then 3
```

**~175,000 shared parameters. 182 per-patient parameters.**

For context: BIT (the Utah array foundation model from Zhang 2026) uses ~7 million parameters and 6 million per patient. Willett's clinical system uses ~66,000 per session. We use 182.

The per-patient parameters do three things:
- **Signal normalization** (128 params): absorbs impedance/gain differences. Initialized from each patient's channel statistics --- starts as z-scoring, learns residual corrections.
- **Coordinate correction** (6 params): a small rigid translation + rotation of the electrode positions, absorbing systematic registration error (~4-8mm from surgical reconstruction).
- **VE position offsets** (48 params): shifts each of the 16 atlas positions by up to 15mm to match the patient's individual somatotopy.

Everything else --- the attention weights, the temporal processing, the decoder --- is shared across all patients. The shared weights learn *what speech activity looks like*; the per-patient weights learn *where to find it in this patient*.

## Training Plan

**Phase 1 (immediate): Supervised cross-patient training on our data.**

Train on N-1 patients, evaluate on the held-out patient. Establish whether the architecture produces a useful cross-patient model at all. Compare against the per-patient baseline (PER 0.734 on our best patient, S14).

This phase requires no new data --- just implementation and compute on DCC.

**Phase 2 (contingent on data access): Foundation model with external pretraining.**

This is where the project becomes a foundation model paper. The idea:

1. **Pretrain** on a large corpus of chronic ECoG speech recordings using self-supervised learning (predict masked neural activity). No phoneme labels needed --- just raw neural data during speech.
2. **Fine-tune** on our labeled CoganLab data (per-patient, 46-178 trials each).

The architecture is **modality-agnostic** --- it treats any intracranial recording as "electrodes with positions and time series." This means we can pretrain on clinical ECoG (10mm electrode spacing) and fine-tune on our research uECoG (1.3mm spacing). The VE bottleneck abstracts away electrode density.

**This cross-density transfer is the unique story no other lab can tell alone.**

## What We Need: External Data

The single highest-leverage action is obtaining external chronic ECoG speech data for pretraining. Two sources:

**Priority 1: Flinker/Chen lab (NYU).** 48 patients, chronic epilepsy monitoring ECoG, speech production tasks. Code is already public (Chen 2024, Nature Machine Intelligence). Data available upon request with a standard sharing agreement. This is a straightforward PI-to-PI data request.

**Priority 2: Chang lab (UCSF).** ~15-25 patients, high-density 256-channel ECoG, speech production. Papers state "data available from corresponding author." This is better framed as a **collaboration**: three institutions (Duke, NYU, UCSF) contributing three electrode densities (1.3mm, 10mm, 4mm) to build the first cross-density intracranial foundation model. Each lab needs the others to tell the cross-density story.

Combined, this would give us ~50-100 hours of diverse speech-related neural data for pretraining --- 100x more than our CoganLab data alone.

**Without external data**, we fall back to pretraining on our own smaller corpus (~24 hours of CoganLab sEEG + uECoG). The architecture still works, but the paper becomes a methods paper instead of a foundation model paper.

## The Paper

**Title direction:** *Neural Field Perceiver: A Foundation Model for Intracranial Field Potentials*

**Core claim:** The first foundation model for intracranial high-gamma activity (HGA). BIT (Zhang 2026) built a foundation model for Utah array spikes; we do the same for field potentials. These are fundamentally different signals --- HGA reflects postsynaptic potentials (inputs to cortex), spikes reflect action potentials (outputs). Different biophysics, different clinical modalities, no existing foundation model.

**Key results we expect to show:**
1. Cross-patient v12 beats per-patient baselines (the architecture works)
2. Performance scales with number of patients (the pooling works)
3. Atlas-grounded VEs beat learned latents at small N (the atlas prior matters)
4. SSL pretraining shifts the calibration curve left (fewer trials needed per new patient)
5. Cross-density transfer works (clinical ECoG pretraining helps research uECoG)

**Worst case:** If nothing transfers, that's still a publishable negative result showing why intracranial field potential foundation models are hard. The systematic evaluation (8 experiment phases, ~44 controlled ablations) would be the most thorough investigation of cross-patient intracranial transfer to date.

## Current Status

- Architecture fully designed (22-page design document)
- Per-patient baseline established (PER 0.734, S14)
- 55 prior cross-patient experiments characterizing the problem
- All electrode coordinates verified and mapped to MNI space (11 patients)
- 16 atlas positions selected and validated for patient reachability
- DCC compute ready (8x RTX 5000 Ada)

**Immediate blockers:**
1. **External data request** --- Greg needs to initiate PI-to-PI contact with Flinker (NYU) and/or Chang (UCSF)
2. **Nonlinear MNI normalization** --- current coordinate transform is linear-only (~5-15mm residual error). If T1 MRIs exist, ANTs nonlinear warp could halve this. Need to check with Zac.
3. **Implementation** --- architecture is designed but not yet coded. Estimate ~2 weeks to first training run on DCC.

## Summary

Each patient's electrode array is a partial view of shared speech motor dynamics. We use anatomical atlas positions to define a common coordinate system, map each patient's electrodes into that space via cross-attention, and learn shared temporal representations that transfer across patients. 175K shared parameters, 182 per patient. The architecture is validated by convergent independent work from 5+ groups. What's missing is scale --- and that requires data from other labs.
