# Cross-Patient Neural Speech Decoding from Intracranial Recordings

A sensor-unit transformer, pretrained by masked reconstruction, that decodes
speech-related neural activity across patients and electrode geometries.

Cogan Lab, Duke University · **Ben Tang** · with Greg Cogan (PI) and Zac Spalding

## Overview

Intracranial speech decoders are usually trained per-patient: every subject has a
unique electrode layout, so models learn channel-specific weights that transfer to
nobody else. This project asks what a *single* model — no per-subject parameters,
no per-subject fine-tuning — can learn from many subjects at once.

The current architecture treats **each electrode contact as a token**. A contact
carries its own signal plus its physical position (which shaft it belongs to, how
deep along that shaft) and an anatomical tag from a DKT atlas lookup. Subjects
differ in how many tokens they contribute and where those tokens sit, but never in
the shape of the model. Pretraining is masked reconstruction over unlabeled
recordings; evaluation is a linear readout on frozen features, so what is measured
is the representation rather than the decoder.

An earlier version of this repo pooled electrodes into cortical parcels and treated
the parcel as the shared cross-subject unit. That design was tested and dropped:
pooling to parcels discarded more than the anatomical alignment bought back.
Anatomy survives as an additive tag on a sensor token, not as the token itself.

## Architecture

- **Front end.** Three `|STFT|` bands per contact, all sharing one hop so every band
  lands on a common 32 Hz frame clock: slow 2–14 Hz (7 bins), mid 16–56 Hz (6 bins),
  high-gamma 64–160 Hz (7 bins). Bands are magnitude-only and per-(electrode, bin,
  session) robust-z normalized.
- **Encoder.** 12 layers, `d_model` 256, 4 heads. Rotary position embeddings over two
  axes — contact index along a shaft, and time.
- **Pretraining.** Masked autoencoding with a 6-layer, `d_model` 128 predictor and an
  EMA target tower. Masking is two-tier: temporal blocks within a band plus a spatial
  term across contacts. Reconstruction targets are the real (denoised, low-dimensional)
  band features, not pixels — so the usual JEPA argument for a learned latent target
  does not apply here.
- **Readout.** Features are frozen and a ridge regression is fit per task. Nothing in
  the evaluation path is trained end-to-end.

## Results

Evaluated on [Neuroprobe](https://github.com/insight-neuro/neuroprobe), a public cross-subject
intracranial benchmark. The cross-subject (CS) split trains and tests on disjoint
subjects; the score below is macro-averaged over 15 tasks and 10 evaluation cells.

| | CS macro |
|---|---|
| Ours — ridge on pretrained features | **.602** |
| Ours — ridge on the untrained front end | .587 |
| Published leaderboard best — CNN on Laplacian spectrograms | .578 |

**Read the comparison carefully.** The leaderboard does not hold the decoder
constant across entries: on one fixed feature set the benchmark harness reports .539
for logistic regression, .566 for an MLP, and .578 for a CNN, so most of the
published leader's margin over its own linear baseline is decoder, not
representation. Matched linear-to-linear, the honest comparison is our .587 against
their .539 — and that .587 uses *no learned parameters at all*, which says the front
end is doing much of the work. Pretraining adds .015 on top of it. Running the
benchmark's own CNN and MLP decoders on our features did not beat our ridge.

These numbers are from a manuscript in preparation and have not been peer reviewed.

## Repository layout

Active code lives under `src/speech_decoding/`:

- `studies/` — dataset adapters (BrainTreebank, AJILE12, SWEC, and the lab's own cohorts)
- `extractors/` — signal front ends, normalization, electrode coordinates, atlas support
- `models/v14_converged_v3/` — the current architecture: front end, masking, towers,
  objective, and its data pipeline
- `ssl/` — self-supervised pretraining primitives (EMA teacher, masked reconstruction)
- `experiments/` — training dispatch, evaluation, and analysis probes
- `atlas/` — cortical atlas and electrode → parcel lookup
- `bt_alignment/` — audio/neural alignment verification for BrainTreebank
- `whisper_ceiling/`, `wav2vec2_ceiling/`, `vjepa_ceiling/` — speech- and video-model
  reference probes

`scripts/` holds cluster launchers and audit utilities. Tests are colocated next to
the modules they cover; the suite is roughly 3,600 tests.

## Setup

Python ≥ 3.12, managed with [`uv`](https://github.com/astral-sh/uv).
`pyproject.toml` + `uv.lock` are authoritative.

```bash
uv sync                              # create .venv/ from uv.lock
.venv/bin/python -m pytest -q        # run the test suite
```

Training runs on NCSA DeltaAI and Duke's DCC cluster, never locally — this
repository holds code, not data. Recordings, electrode coordinates, and cache bakes
live outside the repo and are not redistributable.

## Status

Active research; manuscript in preparation. Design notes, experiment logs, and the
draft are kept outside this public tree.
