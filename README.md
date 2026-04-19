# Cross-Patient Neural Speech Decoding

Atlas-grounded parcel-token architecture for decoding speech from intra-operative intracranial recordings across patients and sensor types. Extending Spalding 2025 (PCA+CCA, 8 patients, 9 phonemes, 0.31 balanced accuracy) with a scalable transformer architecture grounded in the Brainnetome parcellation on fsaverage.

Cogan Lab, Duke University · Ben Tang · with Greg Cogan (PI) and Zac Spalding

## Program hypothesis

**Parcels, not electrodes, are the shared representation across patients and sensors.** Atlas-grounded parcel tokens should transfer cross-patient (within a sensor), cross-sensor (within an anatomy), and cross-lab (within a modality). No iEEG foundation model has tested cross-sensor representational transfer rigorously; that is the destination.

## Docs (Sun Tzu triad)

- [**Objectives**](docs/objectives.md) — program hypothesis, stage roadmap, evaluation philosophy, advance gates
- [**Strategy**](docs/strategy.md) — per-stage default architecture, frozen contract, live scoreboard (current: [`stage_1.md`](docs/strategy/stage_1.md))
- [**Tactics**](docs/tactics.md) — concrete task list, in-flight jobs, blockers

Full docs index: [`docs/README.md`](docs/README.md). Project conventions + code structure: [`CLAUDE.md`](CLAUDE.md).

## Current focus

**Stage 1 (Phase 1)** — single-sensor supervised correctness pass on intra-op uECoG, 7 LH patients (~7 min supervised). Architectural ablation wave in flight on DCC; Stage 2 (in-sensor scaling + SSL) blocked on 13 missing lexical FreeSurfer recons pending from Zac.

## Environment

Python ≥ 3.11 managed via `uv`. `pyproject.toml` + `uv.lock` are authoritative.

```bash
uv sync                                      # bootstrap .venv/
.venv/bin/python -m pytest tests/v14 -q      # run test suite (338 tests)
```

All training runs on Duke's DCC cluster, never local. See [`docs/references/dcc_setup.md`](docs/references/dcc_setup.md).

## Layout

- `src/speech_decoding/v14/` — active per-phoneme Stage-1 implementation
- `scripts/v14_core/` — DCC sbatch wrappers + ablation log aggregator
- `scripts/ablation/` — default DCC tooling (submit / status / logs / collect / peek / query / sync)
- `docs/` — triad + references + experiment log
- `tests/v14/` — test suite
