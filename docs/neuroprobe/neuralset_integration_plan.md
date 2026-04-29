# NeuralSet integration plan — v14

**Date**: 2026-04-26 (rev 4 — pre-Stage 0 adoption locked in).
**Decision**: build v14 calibration code as NeuralSet `BaseExtractor` subclasses; adopt Whisper/DINOv2/GPT-2 + exca caching + SLURM dispatch wholesale.
**Sequencing**: full adoption + repo reorganization happen **before any Stage 0 code** — see `project_pre_stage0_reorg_neuralset_adoption_2026_04_26.md` for the rationale. Reorg-first means Stage 0 author-time = clean target shape; NeuralSet adoption is additive on top.

---

## Findings against v14 needs

Source paths refer to the scratch venv install at `/tmp/neuralset_scratch/.venv/lib/python3.12/site-packages/neuralset/` (will be `repo/speech/.venv/...` once adopted).

### Load-bearing wins

| v14 need | NeuralSet support | Source |
|---|---|---|
| Whisper L8 (~25% depth) for D-SigLIP | `Whisper(layers=0.25, layer_aggregation=None, cache_n_layers=None)` (sweep {L8, L16, L30}; `cache_n_layers` precomputes equidistant layers once for ablation reuse) | `extractors/audio.py:517-543` + `extractors/base.py:451-589` |
| Per-channel metadata pattern (parcel id, support, xyz) | `BaseStatic` subclass depending on `IeegExtractor`; pattern proven by `ChannelPositions` | `extractors/test_neuro.py:898-926` |
| Variable-channel batching across subjects | `IeegExtractor(channel_order="original")` → bounded by max-per-subject, not sum-unique | `extractors/neuro.py:275, 468-503` |
| Time alignment | `Segmenter(start, duration, trigger_query, extractors)` slices around triggers, dispatches all extractors on the same window | `dataloader.py:462-577` |
| exca caching + SLURM dispatch | `_get_data` decorated `@infra.apply` (hash-keyed); `MapInfra(cluster="slurm")` per-extractor | `extractors/neuro.py:403-409`; `dataloader.py:178-230` |
| Custom Study + custom event reader | `Study(base.Step)` subclass + `MneRaw._read()` returning `mne.io.RawArray` | `events/study.py:253-310`; `events/etypes.py:884-981` |

### Real friction (2 items)

1. **Python ≥3.12 required.** Project is 3.11. Bump `pyproject.toml`, `uv.lock`, DCC conda env. Verified packages available for 3.12: torch 2.11, transformers 5.6, mne 1.12, nilearn 0.13, exca 0.5.22.

2. **BrainTreebank data is h5, not MNE-readable directly.** Solution: custom `Ieeg` subclass with `_read()` calling `BrainTreebankSubject.get_electrode_data()` and wrapping in `mne.io.RawArray`. ~30 lines. PS data is `mne.Epochs`; same pattern, deferred until PS-resume.

Default-collate uniformity and event-level metadata are non-issues — Perceiver IO's DETR-style task-attention readout outputs fixed `(T_tasks, d)` regardless of variable electrode count (parcel-latent stack is fixed at ~50 × M sub-slots per the always-include-all-parcel-latents design), and the `BaseStatic` extractor pattern is established in NeuralSet's own code.

---

## Adapter footprint (post-reorg locations)

```
src/speech_decoding/
├── extractors/
│   └── parcel.py                       # V14ParcelMetadataExtractor (BaseStatic; shared across cohorts)
└── studies/braintreebank/
    ├── study.py                        # BraintreebankIeeg (Ieeg subclass) + BraintreebankStudy
    ├── loader.py                       # bt_load_raw() — raw 2048 Hz voltage, no re-ref  ← called by BraintreebankIeeg._read()
    ├── labels.py                       # 15-task label derivation
    └── manifest.py                     # Tier-1 whitelist + BT-Tier-1 parcel list
```

What does **not** get written: HF model wrappers, embedding cache, time-alignment glue, SLURM dispatch glue, parcel projector (parcel-CLS handles it), custom collate (parcel-CLS gives fixed shape), HG-envelope extractor (preprocessing happens at h5-read time inside `BraintreebankIeeg._read()`). NeuralSet handles the orchestration.

`CoganIeeg` + `CoganStudy` (PS .fif epochs) get written when Stage-2 SSL hits PS data. They land at `studies/cogan_ps/study.py` alongside the existing vanilla `studies/cogan_ps/dataset.py`.

---

## Adapter design — concrete code

### 1. Custom Study + custom Ieeg event for BrainTreebank

```python
# src/speech_decoding/studies/braintreebank/study.py
import mne
import pandas as pd
import neuralset as ns
from neuralset.events.etypes import Ieeg

class BraintreebankIeeg(Ieeg):
    """h5-backed Ieeg event. _read() returns raw 2048 Hz voltage, no re-reference.
    Re-ref (CAR / Laplacian) and HG-envelope-at-200Hz are Stage-1 ablation cells,
    not loader behavior. Matches Neuroprobe `__getitem__` native output."""
    subject_num: int
    trial: int

    def _read(self) -> mne.io.RawArray:
        from neuroprobe.braintreebank_subject import BrainTreebankSubject
        from speech_decoding.studies.braintreebank.loader import bt_load_raw
        bt = BrainTreebankSubject(self.subject_num, self.trial)
        data, ch_names, sfreq = bt_load_raw(bt)   # raw voltage at native 2048 Hz, no re-ref
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")
        return mne.io.RawArray(data, info, verbose=False)

class BraintreebankStudy(ns.Study):
    """BrainTreebank — 10 subjects, 26 movie-watching trials. Tier-1 whitelist only."""
    aliases = ("braintreebank", "BT")

    def iter_timelines(self) -> list[dict]:
        from speech_decoding.studies.braintreebank.manifest import TIER1_WHITELIST
        return [{"subject_num": s, "trial": t} for (s, t) in TIER1_WHITELIST]

    def _load_timeline_events(self, timeline: dict) -> pd.DataFrame:
        # one BraintreebankIeeg event per (subject, trial) covering the whole continuous trial,
        # plus N Word events at trigger times from the trial's transcript.
        ...
```

### 2. Per-electrode metadata extractor — the v14 contribution

```python
# src/speech_decoding/extractors/parcel.py
import torch
import typing as tp
from neuralset.extractors.base import BaseStatic
from speech_decoding.atlas.support import load_support_cache

class V14ParcelMetadataExtractor(BaseStatic):
    """Output (n_channels, 5) per Ieeg event:
      [:, 0]   = Tier-1 parcel id (int, 0..14)
      [:, 1]   = support weight (float, [0,1])
      [:, 2:5] = fsaverage_xyz (mm)
    Loads from atlas.support cache, keyed by event.subject.
    `event_types="Ieeg"` covers BraintreebankIeeg + CoganIeeg via subclass polymorphism."""
    event_types: tp.Literal["Ieeg"] = "Ieeg"
    support_cache_dir: str

    def get_static(self, event) -> torch.Tensor: ...
```

### 3. Stock NeuralSet extractors — used directly

```python
neural = ns.extractors.IeegExtractor(
    picks=("seeg",), apply_hilbert=False, filter=None,
    frequency=2048.0, channel_order="original", scaler=None,
)
stim = ns.extractors.audio.Whisper(
    model_name="openai/whisper-large-v3",
    layers=0.5, layer_aggregation=None, cache_n_layers=None,
    token_aggregation=None, frequency=50.0,
)
stim.infra = exca.MapInfra(cluster="slurm", timeout_min=240, cpus_per_task=4)
```

### 4. Stage-2 SSL pairing

```python
segmenter = ns.Segmenter(
    start=-1.5, duration=3.0, trigger_query="type=='Word'",
    extractors={
        "neural":     neural,
        "metadata":   V14ParcelMetadataExtractor(support_cache_dir="..."),
        "stim_audio": stim,
    },
    drop_incomplete=True,
)
dataset = segmenter.apply(events)
dataset.prepare()                              # exca-cached precompute

for batch in dataset.build_dataloader(batch_size=32, num_workers=4):
    brain_emb = v14_model(batch.data["neural"], batch.data["metadata"])
    audio_emb = v14_audio_proj(batch.data["stim_audio"])
    loss = d_siglip_loss(brain_emb, audio_emb)
```

---

## Smoke test — 1-2 days, no DCC

**Goal**: confirm `BraintreebankIeeg + IeegExtractor + Pulse + Segmenter` returns a coherent first batch with stub stimulus (no HF model download).

**Script location**: `scripts/scratch/neuralset_smoke_bt.py`, **fully self-contained** — `BraintreebankIeeg` subclass + helpers defined inline, **no `from speech_decoding...` imports for BT-specific glue**. Survives the reorg unchanged; deleted in Phase 3 after the real `studies/braintreebank/study.py` lands. (Earlier rev parked a temp file under `src/speech_decoding/v14/_smoke_braintreebank.py`; abandoned because it would need to move during the reorg.)

**Data source for `BraintreebankIeeg._read()`**:
- **Default — synthetic stub** (recommended): a tiny class shaped like `BrainTreebankSubject` returning `np.random.randn(n_ch, n_samples).astype("float32")` with a static `n_ch=128`. Validates the API contract — Segmenter slicing, channel-order pinning, dtype/shape — without depending on BT data being on the laptop. The smoke's purpose is API plumbing, not loader correctness.
- Alternate 1: run on DCC where BT lives at `/work/ht203/data/braintreebank/`. Adds DCC round-trip; only worth it if synthetic stub raises a question that requires real h5.
- Alternate 2: download `sub_2/trial_4` h5 (~few GB) locally. Heaviest; only if alternates 1/3 fail.
- Local laptop has **`/tmp/bt_metadata/`** (electrode_labels + localization) but **no h5** — verified 2026-04-28.

**Steps**:
1. Construct one `BraintreebankIeeg` event covering a fake trial (synthetic stub).
2. Build a triggers DataFrame with 4 stub Word events at arbitrary times within the synthetic trial.
3. Run `Segmenter(start=0.0, duration=1.0, ...)` with `IeegExtractor(frequency=2048.0)` + `Pulse` (stim placeholder). Window matches Neuroprobe eval (`START_NEURAL_DATA_BEFORE_WORD_ONSET=0`, `END=1`).
4. Assert `batch.data["neural"].shape == (4, n_ch, 2048)` (raw voltage at native 2048 Hz).

Day-2 follow-up adds `V14ParcelMetadataExtractor` against a real BT support cache (data already local under `data/atlas/support_cache_v2c_snap/`) and confirms shapes line up.

---

## Migration sequence — pre-Stage 0

| Phase | Scope | Time | Gate |
|---|---|---|---|
| **0. Smoke test** | `BraintreebankIeeg + IeegExtractor + Pulse + Segmenter` end-to-end with synthetic h5-stub. Self-contained under `scripts/scratch/`. | 1-2 days | Confirms API contract before reorg PR opens. Very high prior on passing. |
| **1. Python 3.12 bump** | `pyproject.toml` + `uv.lock` + sibling DCC conda env `speech_py312/` (keep `speech/` as fallback ≥1 week). Re-run existing test suite. | 1 day | Independent of Phase 0; whichever finishes first is fine. |
| **2. Reorg PR** | Full restructure: `src/speech_decoding/v14/` → `atlas/ + models/ + training/ + studies/cogan_ps/`. 4 commits. Plan: `docs/neuroprobe/repo_reorg_plan.md`. | ~1 week | Tests pass; `scripts/v14_core/train_v14_core.py --help` clean. |
| **3. NeuralSet adoption PR** | Add `studies/braintreebank/{study, loader, labels, manifest}.py` + `extractors/parcel.py`; depend on `neuralset>=0.1.0`. Resolve `CACHE_FOLDER` before first DCC `prepare()`. | ~3 days | Smoke-test path now lives in the canonical location. |
| **4. Stage 0 begins** | Stage 0 Block A starts on the new shape. | — | Reorg + adoption complete. |
| (later) **5. Stage-2 SSL build-out** | D-SigLIP head, projection heads, Whisper paired loader (Whisper extractor + SLURM dispatch are free here) | 1-2 weeks | — |
| (later) **6. Stage-3 multi-FM + PS adapter** | DINOv3 + GPT-2 stim extractors (one-line each); `CoganIeeg` + `CoganStudy` when SSL extends to PS | 1-2 weeks | — |

**Off-ramp**: NeuralSet adapter is *additive* — it adds files in `extractors/` + `studies/braintreebank/`. The vanilla loader path (`studies/cogan_ps/dataset.py`) doesn't depend on NeuralSet. If exca fights the v14 atlas-coordinate cross-link in surprising ways at Stage 2, fall back to a hand-written Stage-2 SSL driver consuming the vanilla loader. The reorg shape is right whether or not we keep NeuralSet.

---

## Open questions to resolve in Phase 3

1. **exca cache location on DCC.** `/work/ht203` auto-purges after 75 days; the cache is the load-bearing artifact for SSL pretraining. Set `CACHE_FOLDER=/hpc/group/coganlab/ht203/exca_cache` (or similar persistent volume). Confirm exca respects env vars vs. requiring config-file entry.
2. **SLURM partition config.** Does `MapInfra(cluster="slurm")` accept arbitrary sbatch kwargs (e.g. `--partition=coganlab-gpu`, `--account=coganlab`)? If not, will need a thin wrapper or upstream PR. Check `exca.MapInfra` source.
3. **Coexistence with `scripts/ablation/` workflow.** Existing submit/status/logs/collect tooling assumes one job per ablation cell; NeuralSet's `prepare_extractors` parallelizes inside a job. Recommend extract-as-its-own-stage (one big precompute, then ablations consume from cache) — single hash-keyed cache shared across all downstream training runs.
