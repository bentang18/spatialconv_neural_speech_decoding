---
title: "Neural Field Beta"
subtitle: "Minimum Viable Sequential Test of Coordinate-Aware Cross-Patient Transfer"
author: "Ben Tang | Greg Cogan Lab, Duke University"
date: "April 2026"
toc: true
numbersections: true
geometry: margin=1in
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
header-includes:
  - \usepackage[dvipsnames]{xcolor}
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{array}
  - \usepackage{microtype}
---

# Status

This document is the **beta** version of the neural-field direction. Its purpose is not to realize the full Neural Field Perceiver at once. Its purpose is to test the smallest defensible version of the hypothesis:

> **Minimum viable hypothesis.** Coarse anatomical coordinates improve cross-patient LOPO speech decoding when added to the current best-performing local spatial model, under the existing grouped-by-token evaluation protocol.

The design is deliberately conservative. It preserves the current strongest LOPO recipe wherever possible and changes one thing at a time.

# Why A Beta Document Exists

The full `v10` design is much cleaner than `v9`, but it still bundles several unresolved bets:

- It introduces a new encoder family, new latent geometry, virtual electrodes, attention-based readout, geometric correction, and two reconstruction losses in one proposal.
- It still relies on raw volumetric MNI geometry as the main spatial scaffold, even though the broader project notes argue that MNI is trustworthy mainly at coarse cross-patient scale, while local within-patient structure should still come from the grid.
- Its cross-patient reconstruction loss remains a weak and ambiguous supervisory signal: same-token pairing plus full-trial averaging can regularize, but it is not a clean first test of whether coordinates help.

The beta design removes those confounds. If the coordinate hypothesis is real, the beta path should show it with far less architectural risk.

# Design Principles

1. Preserve the current LOPO winner unless a change is required to test the hypothesis.
2. Use coordinates only at the scale where they are scientifically defensible.
3. Keep local within-patient topology in the model.
4. Add no self-supervised or cross-patient reconstruction loss until plain supervised coordinate conditioning shows signal.
5. Gate every additional mechanism on a prior positive result.
6. Judge success on **all patients**, not on S14 alone.

# Non-Goals

This beta architecture explicitly does **not** try to test all of the following at once:

- Perceiver-style latent arrays
- Virtual electrodes
- Cross-patient reprojection
- Trial-matched latent swapping
- Patient embeddings in the encoder
- Point-cloud attention as the first implementation
- Full neural-field identifiability claims

If the beta path fails, that is evidence against the immediate value of the coordinate hypothesis in this data regime. It is not evidence that a larger transformer would have rescued it.

# Anchor Model

The anchor is the **current best LOPO pipeline**, unchanged except where the beta stages explicitly intervene:

- Existing grouped-by-token LOPO protocol
- Existing multi-scale temporal frontend
- Existing best local spatial read-in / pooling path
- Existing articulatory bottleneck classification head
- Existing LP-FT target adaptation regime
- Existing evaluation recipe and reporting contract

This matters. The beta comparison must answer:

> Does adding coarse anatomical information help **the model that already works best**, rather than a brand-new model family?

# Core Architectural Decision

The beta architecture uses **dual spatial encoding**:

- **Local spatial structure** comes from the patient's native grid topology, as in the current pipeline.
- **Cross-patient spatial structure** comes from coarse anatomical coordinates attached to each electrode.

This matches the project's theoretical framing:

- MNI is useful at coarse somatotopic scale.
- Grid adjacency is useful at fine within-array scale.
- Neither should be forced to do the other's job.

# Beta Architecture

## 1. Inputs

For patient $i$, each trial provides:

- HGA grid $X^{(i)} \in \mathbb{R}^{H_i \times W_i \times T}$
- Electrode-validity mask $M^{(i)} \in \{0,1\}^{H_i \times W_i}$
- Static coordinate maps $C^{(i)} \in \mathbb{R}^{H_i \times W_i \times c_{\text{coord}}}$

The coordinate channels are attached only at valid electrode locations. Dead grid positions remain zeroed.

## 2. Coordinate Representation

The beta path uses **coarse coordinates only**.

Stage-dependent coordinate variants:

- `Beta-1`: standardized raw $(x, y, z)$ MNI channels
- `Beta-2+`: coarse Fourier features with $F = 3$, not $F = 4$

Rationale:

- Raw XYZ is the simplest possible coordinate test.
- $F=3$ allows smooth coarse nonlinear spatial structure.
- The finest `v10` Fourier scale is intentionally excluded from the beta path because it pushes below plausible cross-patient correspondence precision.

## 3. Fusion Strategy

The model remains grid-native. Coordinates are **added as static side channels**, not as a replacement for the grid model.

For each trial:

1. Broadcast coordinate channels across time.
2. Concatenate activity and coordinates:

   $$
   U^{(i)} = \mathrm{concat}\bigl(X^{(i)}, C^{(i)}_{\text{broadcast}}\bigr)
   $$

3. Apply a shallow shared channel mixer:

   $$
   \tilde{U}^{(i)} = \mathrm{Conv2d}_{1 \times 1}(U^{(i)})
   $$

4. Feed the mixed tensor into the **existing** local spatial read-in path.

Interpretation:

- The current model still learns local spatial filters on the patient's array.
- Coordinates tell the read-in what cortical zone each local pattern likely belongs to.
- The architecture never asks MNI to substitute for local grid structure.

## 4. Backbone and Decoder

Backbone and phoneme decoder remain unchanged from the current LOPO winner.

This is a hard rule of the beta design:

- no new temporal backbone
- no new latent bottleneck
- no new query readout
- no new inference recipe

If coordinates help, they should help **before** a backbone rewrite.

# Sequential Experiment Ladder

## Beta-0: Population Anchor

**Question:** What is the true all-patient anchor under the current best LOPO recipe?

Model:

- Current best LOPO architecture
- No coordinates
- No new losses

Purpose:

- Establish the population baseline
- Quantify patient-wise variability
- Prevent S14-specific overinterpretation

Gate:

- Required before any architectural conclusion

## Beta-1: Minimal Coordinate Injection

**Question:** Do coarse anatomical coordinates help at all?

Model:

- Anchor model
- Add standardized raw XYZ channels
- Shared `1x1` fusion layer
- No $\Delta$, no $\omega$, no reconstruction loss

This is the minimum viable test.

Why this first:

- It isolates the value of coordinates without importing geometric correction, latent inversion, or auxiliary supervision.
- It preserves the local spatial inductive bias already validated by the current model.
- Failure here is highly informative: if raw coarse coordinates do not help the current best model, the case for a much larger coordinate-aware stack weakens substantially.

Primary comparison:

- `Beta-1` vs `Beta-0`

Success gate:

- Mean PER improves by at least `3pp`
- Effect sign is positive for a majority of patients
- Paired patient-wise test passes the pre-registered threshold

Stop rule:

- If `Beta-1` fails, stop the coordinate path and prioritize data pooling / evaluation fixes instead

## Beta-2: Lightweight Geometric Correction

**Question:** Is residual rigid misregistration the missing piece after plain coordinates?

Model:

- `Beta-1`
- Add per-patient rigid correction on coordinates only:

  $$
  \tilde{\mathbf{p}}_j^{(i)} = R(\omega^{(i)})\,\mathbf{p}_j^{(i)} + \Delta^{(i)}
  $$

- Use strong regularization toward identity:

  $$
  \mathcal{L}_{\text{geom}} = \lambda_{\Delta}\|\Delta^{(i)}\|_2^2 + \lambda_{\omega}\|\omega^{(i)}\|_2^2
  $$

Rules:

- Coordinates only are transformed
- No patient embedding
- No decoder-side patient code

Why this second:

- It directly tests whether the coordinate problem is mostly “use anatomy” or “correct anatomy.”
- It is still low-dimensional and interpretable.

Primary comparison:

- `Beta-2` vs `Beta-1`

Success gate:

- Consistent positive patient-wise effect
- Learned transforms remain small and anatomically plausible

Stop rule:

- If transforms grow large or unstable, treat them as a failed identifiability channel and remove them

## Beta-3: Coarse Nonlinear Coordinate Encoding

**Question:** Are raw XYZ channels too linear to expose the coordinate signal?

Model:

- Best of `Beta-1` / `Beta-2`
- Replace raw XYZ with coarse Fourier features $F=3$
- Keep the rest unchanged

Why this is third:

- It tests nonlinear spatial encoding only after raw coordinates have shown some value
- It avoids blaming Fourier engineering for failure of the underlying hypothesis

Primary comparison:

- `Beta-3` vs best prior beta stage

Success gate:

- Small but consistent incremental benefit
- No evidence that performance gain comes only from one or two patients

## Beta-4: Within-Patient Masked Reconstruction Only

**Question:** Does a light reconstruction objective improve spatial robustness once supervised coordinate conditioning already works?

Model:

- Best of `Beta-1` through `Beta-3`
- Add a **within-patient only** masked-electrode reconstruction head
- Reconstruct held-out electrode activity from the same patient's remaining electrodes

Loss:

$$
\mathcal{L} = \mathcal{L}_{\text{class}} + \lambda_{\text{recon}} \mathcal{L}_{\text{masked-recon}}
$$

Properties:

- No cross-patient trial pairing
- No token matching
- No patient latent swapping
- No full neural-field decoder

Why this is the first auxiliary loss:

- It is operationally simple
- It tests spatial interpolation robustness without creating a new ambiguous cross-patient objective
- It uses trustworthy within-patient structure before touching noisy cross-patient reconstruction

Primary comparison:

- `Beta-4` vs best prior beta stage

Stop rule:

- If reconstruction helps the auxiliary metric but not PER, remove it

## Beta-5: Escalation Threshold

Only after the above sequence should the project consider:

- virtual electrodes
- point-set / Perceiver attention
- cross-patient reprojection
- patient-aware decoder terms
- trial-swapped latent supervision

The escalation condition is:

> Coordinates show clear population-level value in the beta path, but the remaining gap to the anchor still appears spatial rather than purely data-limited.

# Why Cross-Patient Reprojection Is Excluded From Beta

Cross-patient reprojection is not the minimum viable test because it adds several ambiguities at once:

- same-token matching can reward token templates
- trial-level averaging discards temporal structure
- cross-patient residuals are large and hard to interpret
- failure is ambiguous: alignment problem, loss-noise problem, or architecture problem

That loss may still be worth exploring later. It is not the right first experiment.

# Training Protocol

## Fixed Across Beta-0 Through Beta-4

- Same grouped-by-token splits
- Same LOPO patient reporting
- Same target adaptation regime
- Same seeds
- Same augmentation policy unless an ablation explicitly changes it
- Same evaluation recipe

This is necessary because the current evidence shows recipe variance is already large enough to obscure small architecture effects.

## Optimization Policy

- Reuse the current optimizer and scheduler unless a stage introduces a new parameter type
- New geometric parameters use lower LR than backbone weights
- Coordinate channels themselves are deterministic inputs, not learned embeddings, until `Beta-3`

# Evaluation Contract

Primary metric:

- Mean patient-wise PER under LOPO LP-FT

Required reporting:

- Per-patient PER table
- Mean and median PER
- Seed variance
- Paired comparisons against the immediately previous beta stage

Secondary diagnostics:

- Whether gains concentrate in low-coverage or high-coverage patients
- Whether gains are larger for displaced arrays
- Whether rigid correction magnitude correlates with known registration uncertainty

# Decision Logic

## If Beta-1 Fails

Interpretation:

- Coarse coordinates do not add enough value to the current best local model in this regime

Action:

- Stop architecture escalation
- Prioritize cross-task pooling and population evaluation

## If Beta-1 Works but Beta-2/Beta-3 Fail

Interpretation:

- Coordinates help, but only in a simple coarse form

Action:

- Keep the simple coordinate-conditioned model
- Do not escalate to the full Neural Field Perceiver

## If Beta-1 Through Beta-4 All Help

Interpretation:

- The coordinate hypothesis is real and robust enough to justify larger architectural investment

Action:

- Re-open the Perceiver / latent-field direction with much stronger prior evidence

# Recommended Initial Build

If only one beta experiment is implemented first, it should be:

## `Beta-1`

- Current best LOPO model
- Add standardized XYZ coordinate channels
- Shared `1x1` coordinate-activity fusion
- No geometric correction
- No reconstruction loss
- Full all-patient evaluation

This is the cleanest first test of the hypothesis and the one least likely to produce uninterpretable failure.

# Summary

The beta program asks one question at a time:

1. Do coordinates help at all?
2. If yes, does rigid correction help?
3. If yes, does coarse nonlinear encoding help?
4. If yes, does a simple within-patient reconstruction loss help?
5. Only then: is a larger field model justified?

That sequence is scientifically stronger than starting with the full Neural Field Perceiver. It preserves the current best model, respects the actual reliability scale of MNI coordinates, and maximizes what can be learned from either a positive or a negative result.
