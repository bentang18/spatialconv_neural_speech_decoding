# Cross-Patient Neural Speech Decoding from Intracranial Recordings

Atlas-anchored, parcel-token transformer for decoding speech-related neural
activity across patients and recording setups.

Cogan Lab, Duke University · **Ben Tang** · with Greg Cogan (PI) and Zac Spalding

## Overview

Intracranial speech decoders are usually trained per-patient: every subject has a
unique electrode layout, so models learn channel-specific weights that do not
transfer to anyone else. This project tests a different hypothesis — that
**anatomical parcels, not electrodes, are the shared unit of representation across
subjects.** Each electrode is mapped onto a cortical atlas (Desikan–Killiany on
fsaverage) and pooled into parcel tokens, so a single transformer with *zero
per-subject parameters* can be trained and evaluated across many subjects and
electrode geometries.

The architecture extends PopT (a zero-per-subject population transformer) with a
multi-foundation-model self-supervised pretraining stage and an anatomy-grounded
parcel readout. Current work validates the approach on a public cross-subject
intracranial benchmark before returning to the lab's intra-operative
micro-ECoG speech cohort.

Background: extends Spalding 2025 (PCA+CCA, 8 patients, 9 phonemes).

## Repository layout

Active code lives under `src/speech_decoding/`:

- `studies/` — dataset adapters (NeuralFetch-style `Study` classes, e.g. BrainTreebank)
- `extractors/` — signal front-ends and parcel-metadata extraction
- `atlas/` — cortical atlas + parcel-token infrastructure (fsaverage projection,
  electrode → parcel support)
- `ssl/` — self-supervised pretraining primitives (EMA teacher, masked
  reconstruction, distillation)
- `models/` — encoder and readout
- `experiments/` — training and evaluation scaffolding (NeuralTrain / Exca)
- `whisper_ceiling/`, `vjepa_ceiling/` — analysis probes

`scripts/` holds operational and audit utilities. Tests are colocated next to the
modules they cover under `src/`.

## Setup

Python ≥ 3.12, managed with [`uv`](https://github.com/astral-sh/uv).
`pyproject.toml` + `uv.lock` are authoritative.

```bash
uv sync                              # create .venv/ from uv.lock
.venv/bin/python -m pytest -q        # run the test suite
```

All model training runs on Duke's DCC cluster, never locally — this repository
holds code, not data. Datasets (intracranial recordings, electrode coordinates,
atlas bakes) are stored separately and are not part of the repo.

## Status

Active research; manuscript in preparation. Design notes, experiment logs, and the
draft are kept outside this public tree.
