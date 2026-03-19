# Speech Embedding Regression for Phoneme Decoding

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add frame-level HuBERT embedding regression as a joint training objective alongside per-position CE, testing whether denser supervision from speech features improves phoneme decoding accuracy.

**Architecture:** Offline HuBERT extraction from paired audio → frame-level MSE regression target alongside existing CE classification loss. Per-patient read-in → shared backbone (reused) → dual heads (CE + regression). Masked MSE excludes pre-speech silence frames where motor planning activity would be penalized for not matching acoustic silence.

**Tech Stack:** PyTorch, transformers (HuBERT), soundfile, scipy (resampling + PCA), existing speech_decoding infrastructure.

---

## Hypotheses

- **Q1 (Predictability):** Can neural data predict HuBERT speech embeddings? Measured by frame-wise R² on speech frames.
- **Q2 (Utility):** Does joint CE + MSE training improve phoneme decoding over CE alone? Measured by PER and balanced accuracy on S14.

These are evaluated independently. If Q1 fails (R² < 0.05 on speech frames), stop.

## Data Contract

### Canonical Frame Grid (20 Hz)

All tensors share one frame grid after processing:

| Source | Raw Rate | Epoch Window | Raw Frames | After Processing | Final Frames |
|--------|----------|-------------|------------|-----------------|-------------|
| Neural (HGA) | 200 Hz | [-1.0, +1.5s] | 500 | Conv1d stride=10 | **50** |
| HuBERT | 50 Hz | [-1.0, +1.5s] | 125 | `scipy.signal.resample(x, 50)` | **50** |
| Speech mask | — | [-1.0, +1.5s] | — | from phoneme timing | **50** |

- Frame duration: 50 ms (20 Hz)
- Frame 0 = epoch start = response_onset - 1.0s
- Frame 20 = response_onset (t=0)
- Frame 50 = epoch end = response_onset + 1.5s

### Tensor Shapes Per Patient

| Tensor | Shape | dtype | Source |
|--------|-------|-------|--------|
| Neural grid data | `(N_trials, H, W, 500)` | float32 | Existing `BIDSDataset` |
| HuBERT targets (saved) | `(N_trials, 50, 768)` | float32 | Offline extraction (.npz) |
| HuBERT targets (per-fold) | `(N_train, 50, D_emb)` | float32 | PCA-reduced inside trainer (train-only fit) |
| Speech mask | `(N_trials, 50)` | float32 | From phoneme CSV |
| CTC labels | `list[list[int]]` len=N_trials | int | Existing `BIDSDataset` |

Where `D_emb` = 64 (PCA from 768-dim HuBERT layer 6).

### Audio-Neural Alignment

The phoneme events CSV (e.g., `derivatives/phoneme/sub-S14/event/..._production_events.csv`) provides:
- `response_onset`: absolute time (seconds) in the neural recording for each trial
- `onset`, `offset`: absolute time for each phoneme within the trial

The `.fif` epochs use `response_onset` as the epoch center (t=0). To extract audio:
1. Read continuous production audio WAV
2. Determine audio time offset: use audio events TSV `sample` field to calibrate
3. For each trial: extract audio segment `[response_onset - 1.0, response_onset + 1.5]` in audio time
4. Run HuBERT on that 2.5s segment

**Phase 0 validates this alignment before any model work.**

### Speech Mask Construction

From phoneme CSV, for each trial with `response_onset = t0`:
- Phoneme intervals: `[(onset_1 - t0, offset_1 - t0), (onset_2 - t0, offset_2 - t0), (onset_3 - t0, offset_3 - t0)]`
- These are relative times within the epoch window (epoch starts at `-1.0s`)
- Convert to frame indices: `frame = int((relative_time + 1.0) / 0.05)` (50ms per frame at 20Hz)
- Set mask=1 for frames overlapping any phoneme interval, mask=0 otherwise
- **Pre-onset frames (0-19)** are typically silence → mask=0
- **Post-speech frames** after last phoneme offset → mask=0

### Phase 1 Baselines/Controls

| Baseline | Implementation | Purpose |
|----------|---------------|---------|
| Mean embedding | Predict training-set per-dimension mean for every frame | R² > 0 is already relative to this |
| Time-shifted control | Shift neural data +500ms (+10 frames), retrain | If R² drops < 50%, signal is temporal autocorrelation, not content |
| Speech vs silence | Report R² for mask=1 and mask=0 frames separately | R²(speech) must be > R²(silence) |
| Per-dimension breakdown | R² for top-5 PCA components individually | Which features are decodable |

**Stop condition:** R²(speech frames) < 0.05 on S14 validation fold → neural signal does not predict speech embeddings meaningfully.

### CE Branch (Identical to Baseline)

The existing `per_position_ce_loss` from `ctc_utils.py`: mean-pools the **entire** temporal sequence `(B, T, 2H)` → `(B, 2H)`, then `Linear(2H, 3*9)` projects to 3 position-specific 9-way classifiers. Summed CE, divided by 3.

The regression pipeline uses **this exact same CE code path** — same loss function, same temporal pooling.

### Matched CE-Only Control (Critical)

Disabling time-shift and temporal-stretch for regression also changes the training regime. Any PER comparison between CE+MSE and a prior CE baseline that used those augmentations is confounded — we'd be measuring the effect of removing augmentations, not the effect of adding MSE.

**Solution:** Every regression experiment includes a matched CE-only control (λ=0) run under the **identical** config: same stride, same augmentation settings, same number of epochs, same random seed. The comparison is always CE-only-λ0 vs CE+MSE-λ>0 from the same config, never against historical baselines from different configs.

The `train_regression.py` script runs λ=0 as the first entry in every sweep automatically. The ablation table below reflects this.

### Joint Loss

```
Loss = CE_loss + λ × masked_MSE_loss
```

Where:
- `CE_loss`: `per_position_ce_loss(ce_logits, labels, n_positions=3)`
- `masked_MSE_loss`: `(prediction - target)² × mask`, averaged over non-zero mask frames
- `λ ∈ {0.0, 0.1, 0.3, 1.0}` — λ=0.0 is the matched CE-only control

### CRITICAL: Augmentation and Frame Alignment

**Time-shift and temporal-stretch augmentations MUST be disabled** for regression training. These augmentations shift or warp the neural time axis but NOT the HuBERT target embeddings and speech masks, which are pre-extracted at fixed frame positions. With augmentation enabled, backbone frame k no longer corresponds to HuBERT frame k, making the MSE loss train against misaligned targets.

The regression config sets `time_shift_frames: 0` and `temporal_stretch: false`. Other augmentations (amplitude scale, channel dropout, Gaussian noise, feature dropout, time mask) are safe because they don't change temporal alignment.

Both the CE-only control (λ=0) and CE+MSE experiments use this same restricted augmentation set. This ensures any PER difference is attributable to the MSE objective, not augmentation changes.

### PCA Per-Fold (No Leakage)

PCA is fit **per CV fold on training data only**. The regression trainer receives raw 768-dim HuBERT embeddings and fits PCA inside each fold using only the training split. This avoids leaking validation variance structure into targets, which matters because Q1 and Q2 are small-data claims where even unsupervised leakage can inflate R².

The offline extraction script saves raw 768-dim embeddings (not PCA-reduced). PCA reduction happens inside the trainer at fold-split time.

### Ablations

| Ablation | λ | What it tests |
|----------|---|--------------|
| CE only, matched config (control) | 0.0 | Same stride/augmentation as regression runs |
| CE + MSE (HuBERT, speech-masked) | 0.1, 0.3, 1.0 | Main experiment |
| CE + MSE (mel spec, speech-masked) | 0.3 | Is HuBERT better than simpler audio target? |
| CE + MSE (HuBERT, full-window unmasked) | 0.3 | Does masking help or does pre-onset activity add signal? |

### Decision Gates

| Phase | Gate | Action if Fail |
|-------|------|---------------|
| 0 | Audio-neural alignment validated (see Phase 0 spec below) | Debug alignment; try different audio file (raw vs production) |
| 1 | R²(speech) > 0.05 on S14 val, AND R² drops > 50% with +500ms shift | Stop — neural signal doesn't predict speech embeddings |
| 3 | PER(CE+MSE) < PER(CE-only-λ0) on S14 for any λ | Regression doesn't help phonemes under matched conditions |
| 4 | Population PER improvement on ≥ 5/8 patients vs matched CE-only control | Not robust → per-patient effect only |

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/speech_decoding/data/audio_features.py` | Load audio, extract HuBERT embeddings, build speech mask, save/load targets |
| `src/speech_decoding/models/regression_head.py` | `RegressionHead(nn.Linear)` — trivial linear projection |
| `src/speech_decoding/training/regression_trainer.py` | Joint CE + masked MSE training loop with dual heads |
| `configs/per_patient_regression.yaml` | Config for regression experiments |
| `scripts/extract_audio_features.py` | CLI: extract + save HuBERT targets for all patients |
| `scripts/validate_alignment.py` | Phase 0: visual alignment check |
| `scripts/train_regression.py` | CLI: per-patient regression training |
| `tests/test_audio_features.py` | Tests for audio loading, alignment, HuBERT extraction, mask |
| `tests/test_regression.py` | Tests for regression head, masked MSE, joint trainer |

### Modified Files

| File | Change |
|------|--------|
| `pyproject.toml` | Add `transformers`, `soundfile` to dependencies |
| `src/speech_decoding/models/assembler.py` | **No changes** — regression head is owned by the trainer, not the assembler |
| `src/speech_decoding/evaluation/metrics.py` | Add `framewise_r2()` and `embedding_diagnostics()` |

---

## Tasks

### Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml:10-20`

- [ ] **Step 1: Add transformers and soundfile to dependencies**

```toml
dependencies = [
    "torch>=2.0",
    "mne>=1.6",
    "mne-bids>=0.14",
    "numpy>=1.24",
    "scipy>=1.11",
    "scikit-learn>=1.3",
    "pyyaml>=6.0",
    "h5py>=3.9",
    "tqdm>=4.65",
    "transformers>=4.36",
    "soundfile>=0.12",
]
```

- [ ] **Step 2: Install**

Run: `cd /Users/bentang/Documents/Code/speech && uv pip install -e ".[dev]"`
Expected: Successfully installed transformers, soundfile

- [ ] **Step 3: Verify imports**

Run: `python -c "import transformers; import soundfile; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add transformers and soundfile dependencies for audio feature extraction"
```

---

### Task 2: Phase 0 — Audio Alignment Validation

**Files:**
- Create: `scripts/validate_alignment.py`

This is a diagnostic script, not a library module. It produces plots/prints for manual inspection.

- [ ] **Step 1: Write alignment validation script**

```python
"""Phase 0: Validate audio-neural alignment for S14.

Loads production audio + phoneme events CSV + neural epoch events.
Extracts audio segments aligned to neural epochs and checks that
speech is present at expected times.

Usage: python scripts/validate_alignment.py
"""
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


def load_phoneme_events(bids_root: Path, subject: str) -> dict:
    """Load per-phoneme timing from events CSV."""
    import csv

    csv_path = (
        bids_root / "derivatives" / "phoneme" / f"sub-{subject}" / "event"
        / f"sub-{subject}_task-phoneme_acq-01_run-01_desc-production_events.csv"
    )
    trials: dict[int, dict] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial_num = int(row["trial"])
            if trial_num not in trials:
                trials[trial_num] = {
                    "response_onset": float(row["response_onset"]),
                    "response_offset": float(row["response_offset"]),
                    "syllable": row["syllable"],
                    "phonemes": [],
                }
            trials[trial_num]["phonemes"].append({
                "phoneme": row["phoneme"],
                "onset": float(row["onset"]),
                "offset": float(row["offset"]),
            })
    return trials


def load_audio_events(bids_root: Path, subject: str) -> list[dict]:
    """Load trial-level audio events from TSV."""
    import csv

    tsv_path = (
        bids_root / "derivatives" / "audio" / f"sub-{subject}" / "events"
        / f"sub-{subject}_task-phoneme_acq-01_run-01_desc-production_events.tsv"
    )
    events = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            events.append({
                "onset": float(row["onset"]),
                "duration": float(row["duration"]),
                "trial_type": row["trial_type"],
                "sample": int(row["sample"]),
            })
    return events


def main():
    with open("configs/paths.yaml") as f:
        paths = yaml.safe_load(f)
    bids_root = Path(paths["ps_bids_root"])
    subject = "S14"

    # 1. Load audio
    audio_dir = bids_root / "derivatives" / "audio" / f"sub-{subject}" / "microphone"
    production_wav = audio_dir / f"sub-{subject}_task-phoneme_acq-01_run-01_desc-production_microphone.wav"
    audio, sr = sf.read(str(production_wav))
    print(f"Audio: {len(audio)} samples, {sr} Hz, {len(audio)/sr:.1f}s duration")

    # 2. Load events
    phoneme_events = load_phoneme_events(bids_root, subject)
    audio_events = load_audio_events(bids_root, subject)
    print(f"Phoneme events: {len(phoneme_events)} trials")
    print(f"Audio events: {len(audio_events)} trials")

    # 3. Calibrate time offset between neural recording and audio file
    # Audio events TSV has 'onset' (absolute time) and 'sample' (sample index).
    # Check if sample refers to audio or neural time base.
    ae = audio_events[0]
    time_from_sample_audio = ae["sample"] / sr
    time_from_sample_neural_2k = ae["sample"] / 2000
    print(f"\nCalibration (first trial '{ae['trial_type']}'):")
    print(f"  TSV onset: {ae['onset']:.3f}s")
    print(f"  sample/{sr}Hz = {time_from_sample_audio:.3f}s")
    print(f"  sample/2000Hz = {time_from_sample_neural_2k:.3f}s")

    # Determine audio_offset: the absolute time corresponding to audio sample 0
    # If sample is in audio space: audio_offset = onset - sample/sr
    # If sample is in neural space: we need a different approach
    audio_offset_audio = ae["onset"] - ae["sample"] / sr
    audio_offset_neural = ae["onset"] - ae["sample"] / 2000

    # Check consistency across multiple trials
    offsets_audio = []
    offsets_neural = []
    for ae2 in audio_events[:10]:
        offsets_audio.append(ae2["onset"] - ae2["sample"] / sr)
        offsets_neural.append(ae2["onset"] - ae2["sample"] / 2000)

    std_audio = np.std(offsets_audio)
    std_neural = np.std(offsets_neural)
    print(f"\n  Offset consistency (audio space): mean={np.mean(offsets_audio):.4f}, std={std_audio:.6f}")
    print(f"  Offset consistency (neural space): mean={np.mean(offsets_neural):.4f}, std={std_neural:.6f}")

    if std_audio < std_neural:
        audio_offset = np.mean(offsets_audio)
        print(f"  → Using audio sample space. Audio offset = {audio_offset:.4f}s")
    else:
        # Sample is in neural space; need to map to audio
        audio_offset = np.mean(offsets_neural)
        print(f"  → Using neural sample space. Audio offset = {audio_offset:.4f}s")
        print("  WARNING: sample field is in neural time, not audio time.")

    # 4. Phoneme-boundary validation on ALL trials
    # For each trial, check that audio energy within MFA phoneme boundaries
    # is significantly higher than energy in the pre-onset silence window.
    # This is stricter than just post/pre RMS — it validates that specific
    # phoneme intervals contain speech.
    print("\n=== Phoneme-Boundary Validation (all trials) ===")
    trial_nums = sorted(phoneme_events.keys())

    gate_pass = 0
    gate_fail = 0
    ratios = []
    phoneme_energy_ratios = []

    for tnum in trial_nums:
        trial = phoneme_events[tnum]
        t0 = trial["response_onset"]
        syll = trial["syllable"]

        # Extract audio segment [-1.0, +1.5] around response onset
        seg_start = t0 - 1.0 - audio_offset
        start_sample = max(0, int(seg_start * sr))
        end_sample = min(len(audio), int((t0 + 1.5 - audio_offset) * sr))
        segment = audio[start_sample:end_sample]
        if len(segment) < sr:  # less than 1s, skip
            gate_fail += 1
            continue

        # Pre-onset silence: [-1.0, -0.2] (avoid any early articulation near onset)
        pre_silence = segment[:int(0.8 * sr)]
        rms_silence = np.sqrt(np.mean(pre_silence**2)) + 1e-10

        # Energy within each MFA phoneme boundary
        phon_rms = []
        for p in trial["phonemes"]:
            p_start = p["onset"] - t0 + 1.0  # relative to segment start
            p_end = p["offset"] - t0 + 1.0
            s0 = max(0, int(p_start * sr))
            s1 = min(len(segment), int(p_end * sr))
            if s1 > s0:
                phon_rms.append(np.sqrt(np.mean(segment[s0:s1]**2)))

        if phon_rms:
            mean_phon_rms = np.mean(phon_rms)
            ratio = mean_phon_rms / rms_silence
            phoneme_energy_ratios.append(ratio)
            if ratio > 3.0:
                gate_pass += 1
            else:
                gate_fail += 1
        else:
            gate_fail += 1

    # Print detailed results for first 5 trials
    print("\nDetailed (first 5 trials):")
    for tnum in trial_nums[:5]:
        trial = phoneme_events[tnum]
        t0 = trial["response_onset"]
        seg_start = t0 - 1.0 - audio_offset
        start_sample = max(0, int(seg_start * sr))
        end_sample = min(len(audio), int((t0 + 1.5 - audio_offset) * sr))
        segment = audio[start_sample:end_sample]
        pre_silence = segment[:int(0.8 * sr)]
        rms_silence = np.sqrt(np.mean(pre_silence**2)) + 1e-10

        phon_str = ""
        for p in trial["phonemes"]:
            p_start = p["onset"] - t0 + 1.0
            p_end = p["offset"] - t0 + 1.0
            s0 = max(0, int(p_start * sr))
            s1 = min(len(segment), int(p_end * sr))
            if s1 > s0:
                rms = np.sqrt(np.mean(segment[s0:s1]**2))
                phon_str += f"  {p['phoneme']}[{p_start:.2f}-{p_end:.2f}]={rms/rms_silence:.1f}x"

        print(f"  Trial {tnum} '{trial['syllable']}':{phon_str}")

    # 5. Summary with strict gate
    print(f"\n=== GATE CHECK ===")
    n_total = gate_pass + gate_fail
    pass_rate = gate_pass / n_total if n_total > 0 else 0
    median_ratio = np.median(phoneme_energy_ratios) if phoneme_energy_ratios else 0
    print(f"Trials with phoneme energy > 3x silence: {gate_pass}/{n_total} ({pass_rate:.0%})")
    print(f"Median phoneme/silence energy ratio: {median_ratio:.1f}x")

    if pass_rate >= 0.8 and median_ratio >= 5.0:
        print("✓ PASS: Phoneme boundaries are well-aligned with audio energy.")
    elif pass_rate >= 0.5:
        print("⚠ MARGINAL: Some alignment, but many trials have weak phoneme energy.")
        print("  → Manually inspect spectrograms before proceeding.")
    else:
        print("✗ FAIL: Audio-neural alignment is broken.")
        print("  → Try desc-raw or desc-denoised audio. Check time base.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run alignment validation**

Run: `cd /Users/bentang/Documents/Code/speech && source .venv/bin/activate && python scripts/validate_alignment.py`

Expected: ≥80% of trials have phoneme-boundary energy >3x silence, AND median phoneme/silence ratio ≥5x. Inspect the 5-trial detail printout to verify phoneme intervals are plausible.

**DECISION GATE:** PASS requires both quantitative checks AND manual inspection of the first 5 trials. MARGINAL requires spectrogram spot-checks before proceeding. FAIL: try `desc-raw` (22050 Hz) or `desc-denoised` audio. If all fail, the time base is incompatible.

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_alignment.py
git commit -m "feat: Phase 0 audio-neural alignment validation script"
```

---

### Task 3: Audio Loading and Phoneme Timing Module

**Files:**
- Create: `src/speech_decoding/data/audio_features.py`
- Create: `tests/test_audio_features.py`

- [ ] **Step 1: Write failing tests for phoneme timing and speech mask**

```python
"""Tests for audio feature extraction and speech mask construction."""
import numpy as np
import pytest

from speech_decoding.data.audio_features import (
    build_speech_mask,
    load_phoneme_timing,
    PhonemeTimingInfo,
)


class TestPhonemeTimingLoading:
    """Test loading phoneme events CSV."""

    def test_load_returns_list_of_timing_info(self, tmp_path):
        """CSV with 2 trials (6 rows) → 2 PhonemeTimingInfo objects."""
        csv_content = (
            ",subject,trial,auditory,response_onset,response_offset,"
            "phoneme_idx,onset,offset,phoneme,syllable,structure,category,phonetype\n"
            "0,S14,1,382.4,383.1,384.0,1,383.1,383.8,b,bak,CVC,labial,consonant\n"
            "1,S14,1,382.4,383.1,384.0,2,383.8,383.9,a,bak,CVC,low_vowel,vowel\n"
            "2,S14,1,382.4,383.1,384.0,3,383.9,384.0,k,bak,CVC,dorsal,consonant\n"
            "3,S14,2,386.3,388.4,389.5,1,388.4,388.9,u,uvi,VCV,high_vowel,vowel\n"
            "4,S14,2,386.3,388.4,389.5,2,388.9,389.4,v,uvi,VCV,labial,consonant\n"
            "5,S14,2,386.3,388.4,389.5,3,389.4,389.5,i,uvi,VCV,high_vowel,vowel\n"
        )
        csv_path = tmp_path / "events.csv"
        csv_path.write_text(csv_content)

        timing = load_phoneme_timing(csv_path)
        assert len(timing) == 2
        assert timing[0].syllable == "bak"
        assert timing[0].response_onset == pytest.approx(383.1)
        assert len(timing[0].phoneme_intervals) == 3
        assert timing[0].phoneme_intervals[0] == pytest.approx((383.1, 383.8))

    def test_phoneme_intervals_are_relative_to_epoch_start(self, tmp_path):
        """Relative intervals should be computed from response_onset - tmin."""
        csv_content = (
            ",subject,trial,auditory,response_onset,response_offset,"
            "phoneme_idx,onset,offset,phoneme,syllable,structure,category,phonetype\n"
            "0,S14,1,0.0,10.0,11.0,1,10.0,10.3,b,bak,CVC,labial,consonant\n"
            "1,S14,1,0.0,10.0,11.0,2,10.3,10.5,a,bak,CVC,low_vowel,vowel\n"
            "2,S14,1,0.0,10.0,11.0,3,10.5,11.0,k,bak,CVC,dorsal,consonant\n"
        )
        csv_path = tmp_path / "events.csv"
        csv_path.write_text(csv_content)

        timing = load_phoneme_timing(csv_path)
        # With tmin=-1.0: epoch starts at response_onset - 1.0 = 9.0
        # Phoneme 1 relative start: 10.0 - 9.0 = 1.0s
        intervals = timing[0].get_relative_intervals(tmin=-1.0)
        assert intervals[0] == pytest.approx((1.0, 1.3))
        assert intervals[1] == pytest.approx((1.3, 1.5))


class TestSpeechMask:
    """Test speech mask construction at 20Hz frame grid."""

    def test_mask_shape(self):
        """Mask should have n_frames entries."""
        intervals = [(1.0, 1.3), (1.3, 1.5), (1.5, 2.0)]
        mask = build_speech_mask(intervals, n_frames=50, frame_dur=0.05)
        assert mask.shape == (50,)
        assert mask.dtype == np.float32

    def test_mask_is_one_during_speech(self):
        """Frames overlapping phoneme intervals should be 1."""
        # Frame 20 = 1.0s, frame 30 = 1.5s (50ms per frame)
        intervals = [(1.0, 1.5)]
        mask = build_speech_mask(intervals, n_frames=50, frame_dur=0.05)
        assert mask[20] == 1.0  # 1.0s
        assert mask[29] == 1.0  # 1.45s
        assert mask[30] == 0.0  # 1.5s (interval is [1.0, 1.5), right-exclusive at frame boundary)

    def test_mask_is_zero_during_silence(self):
        """Pre-onset frames should be 0."""
        intervals = [(1.0, 1.5)]
        mask = build_speech_mask(intervals, n_frames=50, frame_dur=0.05)
        # Frames 0-19 = [0.0, 1.0) = silence
        assert mask[:20].sum() == 0.0

    def test_mask_with_no_phonemes(self):
        """Empty intervals → all-zero mask."""
        mask = build_speech_mask([], n_frames=50, frame_dur=0.05)
        assert mask.sum() == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_audio_features.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_speech_mask'`

- [ ] **Step 3: Implement PhonemeTimingInfo and mask construction**

```python
"""Audio feature extraction: load audio, extract HuBERT embeddings, build speech masks.

This module handles the offline extraction pipeline:
1. Load phoneme timing from events CSV
2. Load and segment audio aligned to neural epochs
3. Extract HuBERT embeddings from audio segments
4. Build binary speech masks from phoneme intervals
5. PCA-reduce embeddings and resample to backbone frame rate
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhonemeTimingInfo:
    """Timing information for one trial's phonemes."""

    trial_num: int
    response_onset: float  # absolute time (seconds)
    response_offset: float
    syllable: str
    phoneme_intervals: list[tuple[float, float]]  # absolute (onset, offset) pairs
    phoneme_labels: list[str] = field(default_factory=list)

    def get_relative_intervals(
        self, tmin: float = -1.0
    ) -> list[tuple[float, float]]:
        """Convert absolute intervals to epoch-relative times.

        Epoch starts at response_onset + tmin (tmin is negative).
        """
        epoch_start = self.response_onset + tmin
        return [
            (onset - epoch_start, offset - epoch_start)
            for onset, offset in self.phoneme_intervals
        ]


def load_phoneme_timing(csv_path: str | Path) -> list[PhonemeTimingInfo]:
    """Load per-phoneme timing from BIDS events CSV.

    Args:
        csv_path: Path to production events CSV.

    Returns:
        List of PhonemeTimingInfo, one per trial, ordered by trial number.
    """
    csv_path = Path(csv_path)
    trials: dict[int, PhonemeTimingInfo] = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tnum = int(row["trial"])
            if tnum not in trials:
                trials[tnum] = PhonemeTimingInfo(
                    trial_num=tnum,
                    response_onset=float(row["response_onset"]),
                    response_offset=float(row["response_offset"]),
                    syllable=row["syllable"],
                    phoneme_intervals=[],
                    phoneme_labels=[],
                )
            trials[tnum].phoneme_intervals.append(
                (float(row["onset"]), float(row["offset"]))
            )
            trials[tnum].phoneme_labels.append(row["phoneme"])

    return [trials[k] for k in sorted(trials.keys())]


def build_speech_mask(
    relative_intervals: list[tuple[float, float]],
    n_frames: int = 50,
    frame_dur: float = 0.05,
) -> np.ndarray:
    """Build binary speech mask at target frame rate.

    Args:
        relative_intervals: Phoneme (onset, offset) times relative to epoch start.
        n_frames: Number of output frames.
        frame_dur: Duration of each frame in seconds.

    Returns:
        Binary mask array of shape (n_frames,), dtype float32.
    """
    mask = np.zeros(n_frames, dtype=np.float32)
    for onset, offset in relative_intervals:
        start_frame = int(onset / frame_dur)
        end_frame = int(offset / frame_dur)
        start_frame = max(0, min(start_frame, n_frames))
        end_frame = max(0, min(end_frame, n_frames))
        mask[start_frame:end_frame] = 1.0
    return mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_audio_features.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/speech_decoding/data/audio_features.py tests/test_audio_features.py
git commit -m "feat: phoneme timing loader and speech mask construction"
```

---

### Task 4: HuBERT Feature Extraction

**Files:**
- Modify: `src/speech_decoding/data/audio_features.py`
- Modify: `tests/test_audio_features.py`

- [ ] **Step 1: Write failing tests for audio segmentation and HuBERT extraction**

```python
class TestAudioSegmentation:
    """Test extracting audio segments aligned to neural epochs."""

    def test_extract_segment_shape(self, tmp_path):
        """2.5s segment at 16kHz = 40000 samples."""
        # Create a fake 60s audio file
        sr = 16000
        audio = np.random.randn(60 * sr).astype(np.float32)
        wav_path = tmp_path / "audio.wav"
        import soundfile as sf
        sf.write(str(wav_path), audio, sr)

        from speech_decoding.data.audio_features import load_audio, extract_audio_segment

        audio_data, audio_sr = load_audio(wav_path)
        assert audio_sr == sr

        # Extract segment at t=30.0 with window [-1.0, +1.5]
        segment = extract_audio_segment(
            audio_data, audio_sr, onset_time=30.0,
            audio_offset=0.0, tmin=-1.0, tmax=1.5,
        )
        expected_samples = int(2.5 * sr)
        assert segment.shape == (expected_samples,)

    def test_extract_segment_content_aligned(self, tmp_path):
        """Segment should contain audio from the correct time region."""
        sr = 16000
        audio = np.zeros(60 * sr, dtype=np.float32)
        # Put a pulse at t=30.0s
        pulse_sample = 30 * sr
        audio[pulse_sample:pulse_sample + 100] = 1.0

        wav_path = tmp_path / "audio.wav"
        import soundfile as sf
        sf.write(str(wav_path), audio, sr)

        from speech_decoding.data.audio_features import load_audio, extract_audio_segment
        audio_data, _ = load_audio(wav_path)
        segment = extract_audio_segment(
            audio_data, sr, onset_time=30.0, audio_offset=0.0,
            tmin=-1.0, tmax=1.5,
        )
        # Pulse is at onset_time=30.0, which in the segment is at offset 1.0s
        pulse_in_seg = int(1.0 * sr)
        assert segment[pulse_in_seg:pulse_in_seg + 100].sum() > 0
        assert segment[:pulse_in_seg - 100].sum() == 0.0


class TestHuBERTExtraction:
    """Test HuBERT embedding extraction (uses real model, mark slow if needed)."""

    def test_extract_embeddings_shape(self):
        """HuBERT on 2.5s audio → (125, 768) at 50Hz."""
        from speech_decoding.data.audio_features import extract_hubert_embeddings

        # 2.5s of silence at 16kHz
        audio_segment = np.zeros(40000, dtype=np.float32)
        embeddings = extract_hubert_embeddings(audio_segment, sr=16000)

        # HuBERT outputs at 50Hz: 2.5s → ~125 frames (±1 due to padding)
        assert embeddings.shape[0] in range(123, 127)
        assert embeddings.shape[1] == 768

    def test_extract_embeddings_deterministic(self):
        """Same input should give same output."""
        from speech_decoding.data.audio_features import extract_hubert_embeddings

        audio = np.random.RandomState(42).randn(40000).astype(np.float32) * 0.01
        emb1 = extract_hubert_embeddings(audio, sr=16000)
        emb2 = extract_hubert_embeddings(audio, sr=16000)
        np.testing.assert_array_almost_equal(emb1, emb2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_audio_features.py::TestAudioSegmentation -v`
Expected: FAIL — `cannot import name 'load_audio'`

- [ ] **Step 3: Implement audio loading, segmentation, and HuBERT extraction**

Add to `audio_features.py`:

```python
import soundfile as sf
import torch
from scipy.signal import resample


def load_audio(wav_path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio file, return (samples, sample_rate)."""
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono
    return audio.astype(np.float32), sr


def extract_audio_segment(
    audio: np.ndarray,
    sr: int,
    onset_time: float,
    audio_offset: float = 0.0,
    tmin: float = -1.0,
    tmax: float = 1.5,
) -> np.ndarray:
    """Extract audio segment aligned to a neural epoch.

    Args:
        audio: Full continuous audio array.
        sr: Audio sample rate.
        onset_time: Response onset in absolute time (from phoneme CSV).
        audio_offset: Time of audio sample 0 in absolute time.
        tmin: Start of window relative to onset.
        tmax: End of window relative to onset.

    Returns:
        Audio segment of length int((tmax - tmin) * sr).
    """
    seg_start_time = onset_time + tmin - audio_offset
    seg_end_time = onset_time + tmax - audio_offset
    expected_len = int((tmax - tmin) * sr)

    start_sample = int(seg_start_time * sr)
    end_sample = start_sample + expected_len

    # Pad if out of bounds
    if start_sample < 0:
        pad_left = -start_sample
        start_sample = 0
    else:
        pad_left = 0

    if end_sample > len(audio):
        pad_right = end_sample - len(audio)
        end_sample = len(audio)
    else:
        pad_right = 0

    segment = audio[start_sample:end_sample]
    if pad_left > 0 or pad_right > 0:
        segment = np.pad(segment, (pad_left, pad_right), mode="constant")

    return segment[:expected_len]


# Lazy-loaded HuBERT model (loaded once, reused)
_hubert_model = None
_hubert_processor = None


def _get_hubert():
    """Load HuBERT model and processor (cached)."""
    global _hubert_model, _hubert_processor
    if _hubert_model is None:
        from transformers import HubertModel, Wav2Vec2Processor

        logger.info("Loading HuBERT-base model...")
        _hubert_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/hubert-base-ls960"
        )
        _hubert_model = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960"
        )
        _hubert_model.eval()
    return _hubert_model, _hubert_processor


def extract_hubert_embeddings(
    audio_segment: np.ndarray,
    sr: int = 16000,
    layer: int = 6,
) -> np.ndarray:
    """Extract HuBERT embeddings from an audio segment.

    Args:
        audio_segment: 1D audio at target sample rate.
        sr: Sample rate (must be 16000 for HuBERT).
        layer: Which hidden layer to extract (6 = phonetically discriminative).

    Returns:
        Embeddings array of shape (n_frames, 768) at 50Hz.
    """
    model, processor = _get_hubert()

    inputs = processor(
        audio_segment, sampling_rate=sr, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract specified layer
    hidden = outputs.hidden_states[layer]  # (1, n_frames, 768)
    return hidden.squeeze(0).cpu().numpy()


def resample_to_backbone_frames(
    embeddings: np.ndarray,
    n_target_frames: int,
) -> np.ndarray:
    """Resample HuBERT embeddings to match backbone output frame count.

    Args:
        embeddings: (n_hubert_frames, D) array.
        n_target_frames: Target number of frames (e.g., 50 for backbone stride=10).

    Returns:
        Resampled array of shape (n_target_frames, D).
    """
    return resample(embeddings, n_target_frames, axis=0).astype(np.float32)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/bentang/Documents/Code/speech && python -m pytest tests/test_audio_features.py -v`
Expected: All tests PASS (HuBERT tests will download model on first run, ~360MB)

- [ ] **Step 5: Commit**

```bash
git add src/speech_decoding/data/audio_features.py tests/test_audio_features.py
git commit -m "feat: audio loading, segmentation, and HuBERT embedding extraction"
```

---

### Task 5: Offline Feature Extraction Script

**Files:**
- Create: `scripts/extract_audio_features.py`
- Modify: `src/speech_decoding/data/audio_features.py` (add `extract_patient_features` and `extract_all_patients`)

- [ ] **Step 1: Write the patient-level extraction function**

Add to `audio_features.py`:

```python
from sklearn.decomposition import PCA


def extract_patient_features(
    bids_root: str | Path,
    subject: str,
    tmin: float = -1.0,
    tmax: float = 1.5,
    n_backbone_frames: int = 50,
    hubert_layer: int = 6,
) -> dict:
    """Extract HuBERT embeddings and speech masks for one patient.

    Returns dict with:
        - 'embeddings': (n_trials, n_backbone_frames, 768) raw HuBERT
        - 'speech_mask': (n_trials, n_backbone_frames) binary mask
        - 'timing': list of PhonemeTimingInfo
        - 'sr': audio sample rate
    """
    bids_root = Path(bids_root)

    # Load phoneme timing
    csv_path = (
        bids_root / "derivatives" / "phoneme" / f"sub-{subject}" / "event"
        / f"sub-{subject}_task-phoneme_acq-01_run-01_desc-production_events.csv"
    )
    timing = load_phoneme_timing(csv_path)
    logger.info("%s: %d trials from phoneme CSV", subject, len(timing))

    # Load audio
    audio_dir = bids_root / "derivatives" / "audio" / f"sub-{subject}" / "microphone"
    wav_path = audio_dir / f"sub-{subject}_task-phoneme_acq-01_run-01_desc-production_microphone.wav"
    audio, sr = load_audio(wav_path)
    logger.info("%s: audio %d samples @ %d Hz (%.1fs)", subject, len(audio), sr, len(audio) / sr)

    # Calibrate audio offset from audio events TSV
    audio_offset = calibrate_audio_offset(bids_root, subject, sr)
    logger.info("%s: audio_offset = %.4fs", subject, audio_offset)

    frame_dur = (tmax - tmin) / n_backbone_frames

    all_embeddings = []
    all_masks = []

    for trial in timing:
        # Extract audio segment
        segment = extract_audio_segment(
            audio, sr, trial.response_onset, audio_offset, tmin, tmax
        )

        # Resample to 16kHz if needed (HuBERT requires 16kHz)
        if sr != 16000:
            n_target = int(len(segment) * 16000 / sr)
            segment = resample(segment, n_target)

        # Extract HuBERT embeddings
        emb = extract_hubert_embeddings(segment, sr=16000, layer=hubert_layer)

        # Resample to backbone frame count
        emb = resample_to_backbone_frames(emb, n_backbone_frames)
        all_embeddings.append(emb)

        # Build speech mask
        rel_intervals = trial.get_relative_intervals(tmin=tmin)
        mask = build_speech_mask(rel_intervals, n_backbone_frames, frame_dur)
        all_masks.append(mask)

    return {
        "embeddings": np.stack(all_embeddings),  # (n_trials, n_frames, 768)
        "speech_mask": np.stack(all_masks),  # (n_trials, n_frames)
        "timing": timing,
        "sr": sr,
    }


def calibrate_audio_offset(
    bids_root: Path, subject: str, audio_sr: int,
) -> float:
    """Determine the absolute time corresponding to audio sample 0.

    Uses audio events TSV to find consistent offset between
    absolute onset times and sample indices.
    """
    tsv_path = (
        bids_root / "derivatives" / "audio" / f"sub-{subject}" / "events"
        / f"sub-{subject}_task-phoneme_acq-01_run-01_desc-production_events.tsv"
    )
    offsets = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            onset = float(row["onset"])
            sample = int(row["sample"])
            # Try both interpretations
            offsets.append(onset - sample / audio_sr)

    # Check consistency
    offset_std = np.std(offsets)
    if offset_std > 0.1:
        # Try neural sample rate (2000 Hz)
        offsets_neural = []
        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                onset = float(row["onset"])
                sample = int(row["sample"])
                offsets_neural.append(onset - sample / 2000)
        if np.std(offsets_neural) < offset_std:
            logger.warning(
                "%s: sample field is in neural time (2kHz), not audio time. "
                "Using neural-based offset.", subject
            )
            return float(np.mean(offsets_neural))

    return float(np.mean(offsets))


def fit_pca_and_reduce(
    embeddings_by_patient: dict[str, np.ndarray],
    n_components: int = 64,
) -> tuple[PCA, dict[str, np.ndarray]]:
    """Fit PCA on pooled embeddings and reduce all patients.

    Args:
        embeddings_by_patient: {patient_id: (n_trials, n_frames, 768)}
        n_components: Target dimensionality.

    Returns:
        (fitted_pca, {patient_id: (n_trials, n_frames, n_components)})
    """
    # Pool all frames from all patients
    all_frames = []
    for emb in embeddings_by_patient.values():
        n_trials, n_frames, d = emb.shape
        all_frames.append(emb.reshape(-1, d))
    pooled = np.concatenate(all_frames)

    logger.info("Fitting PCA on %d frames, %d → %d dims", len(pooled), pooled.shape[1], n_components)
    pca = PCA(n_components=n_components)
    pca.fit(pooled)
    logger.info("PCA explained variance: %.3f", pca.explained_variance_ratio_.sum())

    reduced = {}
    for pid, emb in embeddings_by_patient.items():
        n_trials, n_frames, d = emb.shape
        flat = emb.reshape(-1, d)
        reduced_flat = pca.transform(flat).astype(np.float32)
        reduced[pid] = reduced_flat.reshape(n_trials, n_frames, n_components)

    return pca, reduced
```

- [ ] **Step 2: Write the extraction CLI script**

```python
"""Extract HuBERT embeddings for all PS patients.

Saves per-patient .npz files with embeddings, masks, and labels.
Fits joint PCA across all patients.

Usage: python scripts/extract_audio_features.py [--patients S14] [--d_emb 64]
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from speech_decoding.data.audio_features import extract_patient_features

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
logger = logging.getLogger(__name__)

PS_PATIENTS = ["S14", "S16", "S22", "S23", "S26", "S32", "S33", "S36",
               "S39", "S57", "S58", "S62"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", nargs="+", default=PS_PATIENTS)
    parser.add_argument("--d_emb", type=int, default=64)
    parser.add_argument("--hubert_layer", type=int, default=6)
    parser.add_argument("--output_dir", default="data/audio_features")
    args = parser.parse_args()

    with open("configs/paths.yaml") as f:
        paths = yaml.safe_load(f)
    bids_root = paths["ps_bids_root"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Extract raw HuBERT embeddings per patient
    raw_embeddings = {}
    all_masks = {}
    for pid in args.patients:
        logger.info("Extracting features for %s...", pid)
        try:
            result = extract_patient_features(
                bids_root, pid, hubert_layer=args.hubert_layer
            )
            raw_embeddings[pid] = result["embeddings"]
            all_masks[pid] = result["speech_mask"]
            logger.info(
                "%s: %d trials, embeddings %s, mask speech_frac=%.2f",
                pid, len(result["timing"]),
                result["embeddings"].shape,
                result["speech_mask"].mean(),
            )
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", pid, e)
            continue

    # Save raw 768-dim embeddings (PCA is done per-fold inside the trainer
    # to avoid leaking validation data into the dimensionality reduction)
    for pid in raw_embeddings:
        out_path = output_dir / f"{pid}_hubert.npz"
        np.savez(
            str(out_path),
            embeddings=raw_embeddings[pid],   # (n_trials, n_frames, 768)
            speech_mask=all_masks[pid],        # (n_trials, n_frames)
        )
        logger.info("Saved %s: %s", out_path, raw_embeddings[pid].shape)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run extraction for S14 only**

Run: `cd /Users/bentang/Documents/Code/speech && source .venv/bin/activate && python scripts/extract_audio_features.py --patients S14`
Expected: Saves `data/audio_features/S14_hubert.npz` and `hubert_pca.pkl`. Log shows trial count, embedding shape, speech mask fraction.

- [ ] **Step 4: Commit**

```bash
git add src/speech_decoding/data/audio_features.py scripts/extract_audio_features.py
git commit -m "feat: offline HuBERT feature extraction with PCA and speech masks"
```

---

### Task 6: Regression Head and Assembler Update

**Files:**
- Create: `src/speech_decoding/models/regression_head.py`
- Modify: `src/speech_decoding/models/assembler.py`
- Modify: `tests/test_audio_features.py` → or create `tests/test_regression.py`

- [ ] **Step 1: Write failing test for regression head**

Create `tests/test_regression.py`:

```python
"""Tests for regression components: head, masked MSE, joint trainer."""
import numpy as np
import pytest
import torch

from speech_decoding.models.regression_head import RegressionHead


class TestRegressionHead:
    """Test the regression output head."""

    def test_output_shape(self):
        """(B, T, 2H) → (B, T, D_emb)."""
        head = RegressionHead(input_dim=64, d_emb=64)
        x = torch.randn(4, 50, 64)
        out = head(x)
        assert out.shape == (4, 50, 64)

    def test_different_dimensions(self):
        """Works with various input/output dims."""
        head = RegressionHead(input_dim=128, d_emb=32)
        x = torch.randn(2, 50, 128)
        out = head(x)
        assert out.shape == (2, 50, 32)

    def test_gradients_flow(self):
        """Backward pass produces gradients."""
        head = RegressionHead(input_dim=64, d_emb=64)
        x = torch.randn(4, 50, 64, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_regression.py::TestRegressionHead -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement regression head**

```python
"""Regression head: projects backbone output to speech embedding space.

Trivially simple — a single Linear layer. The complexity is in the loss
and training loop, not the head architecture.
"""
from __future__ import annotations

import torch.nn as nn


class RegressionHead(nn.Module):
    """Linear projection from backbone hidden states to embedding space.

    Input:  (B, T, input_dim)  — backbone output (2H for bidirectional GRU)
    Output: (B, T, d_emb)      — predicted speech embeddings per frame
    """

    def __init__(self, input_dim: int = 128, d_emb: int = 64):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_emb)

    def forward(self, x):
        return self.linear(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_regression.py::TestRegressionHead -v`
Expected: 3 PASS

- [ ] **Step 5: Do NOT modify assembler.py**

The regression head is an auxiliary head owned entirely by `regression_trainer.py`. It is NOT part of the main model assembly pipeline. The assembler builds (backbone, CE_head, read_ins) as before; the trainer adds the regression head on top. This avoids a dual-path design where two different code paths can construct the same component.

- [ ] **Step 6: Run existing tests to verify no regressions**

Run: `python -m pytest tests/test_models.py tests/test_integration.py -v -m "not slow"`
Expected: All existing tests PASS (assembler.py is unchanged)

- [ ] **Step 7: Commit**

```bash
git add src/speech_decoding/models/regression_head.py tests/test_regression.py
git commit -m "feat: regression head for speech embedding prediction"
```

---

### Task 7: Masked MSE Loss

**Files:**
- Modify: `tests/test_regression.py`
- Create loss function in `src/speech_decoding/training/regression_loss.py`

- [ ] **Step 1: Write failing tests for masked MSE**

Add to `tests/test_regression.py`:

```python
from speech_decoding.training.regression_loss import masked_mse_loss


class TestMaskedMSELoss:
    """Test speech-masked MSE loss."""

    def test_full_mask_equals_mse(self):
        """When mask is all 1s, should equal standard MSE."""
        pred = torch.randn(4, 50, 64)
        target = torch.randn(4, 50, 64)
        mask = torch.ones(4, 50)

        loss = masked_mse_loss(pred, target, mask)
        expected = torch.nn.functional.mse_loss(pred, target)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_zero_mask_returns_zero(self):
        """When mask is all 0s, loss should be 0."""
        pred = torch.randn(4, 50, 64)
        target = torch.randn(4, 50, 64)
        mask = torch.zeros(4, 50)

        loss = masked_mse_loss(pred, target, mask)
        assert loss.item() == 0.0

    def test_partial_mask_ignores_masked_frames(self):
        """Only unmasked frames should contribute to loss."""
        pred = torch.zeros(1, 10, 4)
        target = torch.ones(1, 10, 4)
        mask = torch.zeros(1, 10)
        mask[0, 5:8] = 1.0  # Only frames 5-7 contribute

        loss = masked_mse_loss(pred, target, mask)
        # MSE on frames 5-7: (0-1)^2 = 1.0 for all dims
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)

    def test_gradient_only_through_unmasked(self):
        """Gradients should be zero for masked frames."""
        pred = torch.randn(1, 10, 4, requires_grad=True)
        target = torch.randn(1, 10, 4)
        mask = torch.zeros(1, 10)
        mask[0, 5:8] = 1.0

        loss = masked_mse_loss(pred, target, mask)
        loss.backward()

        # Frames 0-4 and 8-9 should have zero gradient
        assert pred.grad[0, :5].abs().sum() == 0.0
        assert pred.grad[0, 8:].abs().sum() == 0.0
        # Frames 5-7 should have nonzero gradient
        assert pred.grad[0, 5:8].abs().sum() > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_regression.py::TestMaskedMSELoss -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement masked MSE loss**

```python
"""Masked MSE loss for speech embedding regression.

Only computes loss on frames where speech_mask=1 (speech frames).
Pre-onset silence frames are excluded to avoid penalizing motor planning
activity for not matching acoustic silence embeddings.
"""
from __future__ import annotations

import torch


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss masked to speech frames only.

    Args:
        prediction: (B, T, D) predicted embeddings.
        target: (B, T, D) target embeddings.
        mask: (B, T) binary mask, 1=speech, 0=silence.

    Returns:
        Scalar loss, averaged over unmasked frames and dimensions.
        Returns 0 if mask is all zeros.
    """
    # Expand mask to match embedding dims: (B, T) → (B, T, 1)
    mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)

    # Squared error, masked
    sq_err = (prediction - target) ** 2  # (B, T, D)
    masked_sq_err = sq_err * mask_expanded  # zero out silence frames

    # Average over unmasked frames
    n_unmasked = mask.sum()
    if n_unmasked == 0:
        return torch.tensor(0.0, device=prediction.device, requires_grad=True)

    # Sum over all dims, divide by (n_unmasked_frames * D)
    d = prediction.shape[-1]
    return masked_sq_err.sum() / (n_unmasked * d)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_regression.py::TestMaskedMSELoss -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/speech_decoding/training/regression_loss.py tests/test_regression.py
git commit -m "feat: masked MSE loss for speech-frame-only regression"
```

---

### Task 8: Regression Trainer (Joint CE + MSE)

**Files:**
- Create: `src/speech_decoding/training/regression_trainer.py`
- Modify: `tests/test_regression.py`

- [ ] **Step 1: Write failing test for regression trainer**

Add to `tests/test_regression.py`:

```python
from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.training.regression_trainer import train_per_patient_regression


class TestRegressionTrainer:
    """Test joint CE + MSE training loop."""

    def _make_synthetic_data(self):
        """Create synthetic neural + embedding data for testing."""
        np.random.seed(42)
        n_trials = 20
        grid_h, grid_w, T = 8, 16, 100
        n_backbone_frames = T // 10  # stride=10 → 10 frames
        d_emb = 16

        grid_data = np.random.randn(n_trials, grid_h, grid_w, T).astype(np.float32)
        ctc_labels = [[1 + i % 9, 1 + (i + 1) % 9, 1 + (i + 2) % 9] for i in range(n_trials)]

        # Synthetic embeddings: different per-phoneme
        embeddings = np.random.randn(n_trials, n_backbone_frames, d_emb).astype(np.float32)
        speech_mask = np.zeros((n_trials, n_backbone_frames), dtype=np.float32)
        speech_mask[:, 3:8] = 1.0  # frames 3-7 are "speech"

        ds = BIDSDataset(grid_data, ctc_labels, "S_test", (grid_h, grid_w))
        return ds, embeddings, speech_mask

    def test_trainer_returns_metrics(self):
        """Trainer should return dict with per, bal_acc, r2 keys."""
        ds, embeddings, speech_mask = self._make_synthetic_data()
        config = {
            "model": {
                "readin_type": "spatial_conv",
                "head_type": "flat",  # CE head type (regression head added internally)
                "d_shared": 64,
                "hidden_size": 32,
                "gru_layers": 2,
                "gru_dropout": 0.2,
                "temporal_stride": 10,
                "num_classes": 10,
                "blank_bias": 0.0,
                "d_emb": 16,
                "spatial_conv": {
                    "channels": 8, "num_layers": 1,
                    "kernel_size": 3, "pool_h": 2, "pool_w": 4,
                },
            },
            "training": {
                "loss_type": "ce",
                "ce_segments": 3,
                "regression_lambda": 0.3,
                "stage1": {
                    "epochs": 10, "lr": 1e-3, "readin_lr_mult": 3.0,
                    "weight_decay": 1e-4, "batch_size": 8, "grad_clip": 5.0,
                    "patience": 5, "eval_every": 5, "warmup_epochs": 0,
                },
                "augmentation": {
                    "time_shift_frames": 0, "amp_scale_std": 0.0,
                    "channel_dropout_max": 0.0, "noise_frac": 0.0,
                    "feat_dropout_max": 0.0, "time_mask_min": 2, "time_mask_max": 4,
                },
            },
            "evaluation": {"seeds": [42], "cv_folds": 2, "primary_metric": "per"},
        }

        result = train_per_patient_regression(
            ds, embeddings, speech_mask, config, seed=42, device="cpu"
        )

        assert "per_mean" in result
        assert "r2_speech_mean" in result
        assert np.isfinite(result["per_mean"])
        assert np.isfinite(result["r2_speech_mean"])

    def test_trainer_ce_only_when_lambda_zero(self):
        """With regression_lambda=0, should behave like pure CE."""
        ds, embeddings, speech_mask = self._make_synthetic_data()
        config = {
            "model": {
                "readin_type": "spatial_conv",
                "head_type": "flat",
                "d_shared": 64,
                "hidden_size": 32,
                "gru_layers": 2,
                "gru_dropout": 0.2,
                "temporal_stride": 10,
                "num_classes": 10,
                "blank_bias": 0.0,
                "d_emb": 16,
                "spatial_conv": {
                    "channels": 8, "num_layers": 1,
                    "kernel_size": 3, "pool_h": 2, "pool_w": 4,
                },
            },
            "training": {
                "loss_type": "ce",
                "ce_segments": 3,
                "regression_lambda": 0.0,
                "stage1": {
                    "epochs": 5, "lr": 1e-3, "readin_lr_mult": 3.0,
                    "weight_decay": 1e-4, "batch_size": 8, "grad_clip": 5.0,
                    "patience": 5, "eval_every": 5, "warmup_epochs": 0,
                },
                "augmentation": {
                    "time_shift_frames": 0, "amp_scale_std": 0.0,
                    "channel_dropout_max": 0.0, "noise_frac": 0.0,
                    "feat_dropout_max": 0.0, "time_mask_min": 2, "time_mask_max": 4,
                },
            },
            "evaluation": {"seeds": [42], "cv_folds": 2, "primary_metric": "per"},
        }

        result = train_per_patient_regression(
            ds, embeddings, speech_mask, config, seed=42, device="cpu"
        )
        assert "per_mean" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_regression.py::TestRegressionTrainer -v`
Expected: FAIL — `cannot import name 'train_per_patient_regression'`

- [ ] **Step 3: Implement regression trainer**

Create `src/speech_decoding/training/regression_trainer.py`:

```python
"""Per-patient trainer with joint CE + masked MSE regression loss.

Extends the standard per-patient CE trainer by adding a regression head
that predicts speech embeddings at each backbone output frame. The joint
loss is: CE_phoneme + lambda * masked_MSE_embedding.

The CE branch is identical to the baseline trainer (global temporal pool →
per-position classification). The regression head is a simple Linear(2H, D_emb).
"""
from __future__ import annotations

import logging
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from speech_decoding.data.augmentation import augment_from_config
from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.evaluation.metrics import evaluate_predictions
from speech_decoding.models.assembler import assemble_model
from speech_decoding.models.regression_head import RegressionHead
from speech_decoding.training.ctc_utils import per_position_ce_loss, per_position_ce_decode
from speech_decoding.training.regression_loss import masked_mse_loss

logger = logging.getLogger(__name__)


def _compute_r2(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
) -> dict[str, float]:
    """Compute frame-wise R² on speech and silence frames separately.

    Args:
        pred: (N, T, D) predicted embeddings.
        target: (N, T, D) target embeddings.
        mask: (N, T) binary speech mask.

    Returns:
        Dict with r2_speech, r2_silence, r2_all.
    """
    results = {}
    for name, m in [("speech", mask > 0.5), ("silence", mask < 0.5), ("all", np.ones_like(mask, dtype=bool))]:
        idx = m.reshape(-1)
        if idx.sum() == 0:
            results[f"r2_{name}"] = float("nan")
            continue
        p = pred.reshape(-1, pred.shape[-1])[idx]
        t = target.reshape(-1, target.shape[-1])[idx]
        ss_res = ((p - t) ** 2).sum()
        ss_tot = ((t - t.mean(axis=0)) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        results[f"r2_{name}"] = float(r2)
    return results


def train_per_patient_regression(
    dataset: BIDSDataset,
    embeddings: np.ndarray,
    speech_mask: np.ndarray,
    config: dict,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Train with joint CE + masked MSE on a single patient.

    Args:
        dataset: BIDSDataset with grid data and CTC labels.
        embeddings: (n_trials, n_frames, D_emb) target speech embeddings.
        speech_mask: (n_trials, n_frames) binary speech mask.
        config: Full YAML config dict.
        seed: Random seed.
        device: Device string.

    Returns:
        Dict with per-fold and mean metrics (PER, bal_acc, R²).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tc = config["training"]["stage1"]
    ec = config["evaluation"]
    n_folds = ec["cv_folds"]
    reg_lambda = config["training"].get("regression_lambda", 0.3)

    strat_labels = [y[0] for y in dataset.ctc_labels]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(
        np.zeros(len(dataset)), strat_labels
    )):
        logger.info("Fold %d/%d", fold_idx + 1, n_folds)
        result = _train_regression_fold(
            dataset, embeddings, speech_mask, config,
            train_idx, val_idx, reg_lambda, seed, device,
        )
        fold_results.append(result)
        logger.info(
            "Fold %d: PER=%.3f, bal_acc=%.3f, R²(speech)=%.3f",
            fold_idx + 1, result["per"], result["bal_acc_mean"],
            result.get("r2_speech", float("nan")),
        )

    # Aggregate
    mean_metrics = {}
    for key in fold_results[0]:
        vals = [r[key] for r in fold_results]
        if all(isinstance(v, (int, float)) for v in vals):
            mean_metrics[f"{key}_mean"] = float(np.nanmean(vals))
            mean_metrics[f"{key}_std"] = float(np.nanstd(vals))
    mean_metrics["fold_results"] = fold_results
    mean_metrics["patient_id"] = dataset.patient_id
    mean_metrics["seed"] = seed

    return mean_metrics


def _train_regression_fold(
    dataset: BIDSDataset,
    embeddings: np.ndarray,
    speech_mask: np.ndarray,
    config: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    reg_lambda: float,
    seed: int,
    device: str,
) -> dict:
    """Train one CV fold with joint CE + masked MSE."""
    tc = config["training"]["stage1"]
    ac = config["training"].get("augmentation", {})
    mc = config["model"]
    n_segments = config["training"].get("ce_segments", 3)
    d_emb = mc.get("d_emb", 64)

    # Build model components
    patients = {dataset.patient_id: dataset.grid_shape}
    backbone, _, readins = assemble_model(config, patients)
    readin = readins[dataset.patient_id]

    # CE head (same as baseline — owned by trainer, not assembler)
    input_dim = mc["hidden_size"] * 2
    n_phonemes = mc["num_classes"] - 1
    ce_head = nn.Linear(input_dim, n_segments * n_phonemes)

    # Per-fold PCA: fit on training embeddings only (no leakage)
    from sklearn.decomposition import PCA
    train_emb_raw = embeddings[train_idx]  # (n_train, T, 768)
    val_emb_raw = embeddings[val_idx]
    n_train_frames = train_emb_raw.shape[0] * train_emb_raw.shape[1]
    pca = PCA(n_components=d_emb)
    pca.fit(train_emb_raw.reshape(n_train_frames, -1))
    # Transform both train and val using train-fit PCA
    train_emb_pca = pca.transform(
        train_emb_raw.reshape(-1, train_emb_raw.shape[-1])
    ).reshape(train_emb_raw.shape[0], train_emb_raw.shape[1], d_emb).astype(np.float32)
    val_emb_pca = pca.transform(
        val_emb_raw.reshape(-1, val_emb_raw.shape[-1])
    ).reshape(val_emb_raw.shape[0], val_emb_raw.shape[1], d_emb).astype(np.float32)

    # Regression head (auxiliary, owned by trainer)
    reg_head = RegressionHead(input_dim=input_dim, d_emb=d_emb)

    backbone = backbone.to(device)
    ce_head = ce_head.to(device)
    reg_head = reg_head.to(device)
    readin = readin.to(device)

    # Optimizer — include both heads
    optimizer = AdamW(
        [
            {"params": readin.parameters(), "lr": tc["lr"] * tc["readin_lr_mult"]},
            {"params": backbone.parameters(), "lr": tc["lr"]},
            {"params": ce_head.parameters(), "lr": tc["lr"]},
            {"params": reg_head.parameters(), "lr": tc["lr"]},
        ],
        weight_decay=tc["weight_decay"],
    )
    warmup_epochs = tc.get("warmup_epochs", 0)
    total_epochs = tc.get("epochs", 0)

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Split data (embeddings are already PCA-reduced per-fold above)
    train_x = torch.from_numpy(dataset.grid_data[train_idx])
    train_y = [dataset.ctc_labels[i] for i in train_idx]
    train_emb = torch.from_numpy(train_emb_pca)
    train_mask = torch.from_numpy(speech_mask[train_idx])

    val_x = torch.from_numpy(dataset.grid_data[val_idx])
    val_y = [dataset.ctc_labels[i] for i in val_idx]
    val_emb = torch.from_numpy(val_emb_pca)
    val_mask = torch.from_numpy(speech_mask[val_idx])

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    B = tc["batch_size"]
    n_train = len(train_x)
    all_params = (
        list(backbone.parameters()) + list(ce_head.parameters())
        + list(reg_head.parameters()) + list(readin.parameters())
    )

    for epoch in range(total_epochs):
        backbone.train()
        ce_head.train()
        reg_head.train()
        readin.train()

        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, B):
            idx = perm[start:start + B]
            x_batch = train_x[idx]
            y_batch = [train_y[i] for i in idx.tolist()]
            emb_batch = train_emb[idx].to(device)
            mask_batch = train_mask[idx].to(device)

            x_batch = augment_from_config(x_batch, ac, training=True)
            x_batch = x_batch.to(device)

            optimizer.zero_grad()
            shared = readin(x_batch)
            h = backbone(shared)  # (B, T_down, 2H)

            # CE loss
            ce_out = ce_head(h)  # (B, T_down, n_seg * n_phon)
            ce_loss = per_position_ce_loss(ce_out, y_batch, n_segments)

            # Regression loss (only if lambda > 0)
            if reg_lambda > 0:
                reg_out = reg_head(h)  # (B, T_down, D_emb)
                mse_loss = masked_mse_loss(reg_out, emb_batch, mask_batch)
                loss = ce_loss + reg_lambda * mse_loss
            else:
                loss = ce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, tc["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        if (epoch + 1) % tc["eval_every"] == 0:
            backbone.eval()
            ce_head.eval()
            reg_head.eval()
            readin.eval()

            with torch.no_grad():
                val_x_dev = val_x.to(device)
                shared = readin(val_x_dev)
                h = backbone(shared)

                ce_out = ce_head(h)
                val_ce = per_position_ce_loss(ce_out, val_y, n_segments).item()

                if reg_lambda > 0:
                    reg_out = reg_head(h)
                    val_mse = masked_mse_loss(
                        reg_out, val_emb.to(device), val_mask.to(device)
                    ).item()
                    val_loss = val_ce + reg_lambda * val_mse
                else:
                    val_loss = val_ce

            logger.info(
                "  epoch %d: train=%.4f val=%.4f (ce=%.4f)",
                epoch + 1, epoch_loss / max(n_batches, 1), val_loss, val_ce,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "backbone": deepcopy(backbone.state_dict()),
                    "ce_head": deepcopy(ce_head.state_dict()),
                    "reg_head": deepcopy(reg_head.state_dict()),
                    "readin": deepcopy(readin.state_dict()),
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= tc["patience"]:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # Restore best
    if best_state is not None:
        backbone.load_state_dict(best_state["backbone"])
        ce_head.load_state_dict(best_state["ce_head"])
        reg_head.load_state_dict(best_state["reg_head"])
        readin.load_state_dict(best_state["readin"])

    # Final evaluation
    backbone.eval()
    ce_head.eval()
    reg_head.eval()
    readin.eval()

    with torch.no_grad():
        val_x_dev = val_x.to(device)
        shared = readin(val_x_dev)
        h = backbone(shared)

        # Phoneme metrics (CE branch)
        ce_out = ce_head(h)
        predictions = per_position_ce_decode(ce_out, n_segments)

        # Regression metrics
        reg_out = reg_head(h)
        r2 = _compute_r2(
            reg_out.cpu().numpy(),
            val_emb.numpy(),
            val_mask.numpy(),
        )

    metrics = evaluate_predictions(predictions, val_y, n_positions=3)
    metrics.update(r2)
    metrics["blank_ratio"] = 0.0
    return metrics
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_regression.py::TestRegressionTrainer -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/speech_decoding/training/regression_trainer.py tests/test_regression.py
git commit -m "feat: joint CE + masked MSE regression trainer"
```

---

### Task 9: Evaluation Diagnostics

**Files:**
- Modify: `src/speech_decoding/evaluation/metrics.py`
- Modify: `tests/test_regression.py`

- [ ] **Step 1: Write failing test for framewise R² diagnostics**

Add to `tests/test_regression.py`:

```python
from speech_decoding.evaluation.metrics import framewise_r2_diagnostics


class TestDiagnostics:
    """Test Phase 1 evaluation diagnostics."""

    def test_perfect_prediction_r2_is_one(self):
        """R² = 1 when prediction equals target."""
        target = np.random.randn(10, 50, 16).astype(np.float32)
        mask = np.ones((10, 50), dtype=np.float32)
        result = framewise_r2_diagnostics(target, target, mask)
        assert result["r2_speech"] == pytest.approx(1.0, abs=1e-5)

    def test_mean_prediction_r2_is_zero(self):
        """R² ≈ 0 when prediction is the mean."""
        target = np.random.randn(10, 50, 16).astype(np.float32)
        mean_pred = np.broadcast_to(
            target.reshape(-1, 16).mean(axis=0),
            target.shape,
        ).copy()
        mask = np.ones((10, 50), dtype=np.float32)
        result = framewise_r2_diagnostics(mean_pred, target, mask)
        assert abs(result["r2_speech"]) < 0.05

    def test_per_dim_breakdown(self):
        """Should report R² for each PCA dimension."""
        target = np.random.randn(10, 50, 16).astype(np.float32)
        pred = target + np.random.randn(*target.shape).astype(np.float32) * 0.1
        mask = np.ones((10, 50), dtype=np.float32)
        result = framewise_r2_diagnostics(pred, target, mask)
        assert "r2_per_dim" in result
        assert len(result["r2_per_dim"]) == 16
        assert all(r > 0.5 for r in result["r2_per_dim"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_regression.py::TestDiagnostics -v`
Expected: FAIL — `cannot import name 'framewise_r2_diagnostics'`

- [ ] **Step 3: Implement diagnostics**

Add to `src/speech_decoding/evaluation/metrics.py`:

```python
def framewise_r2_diagnostics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Comprehensive R² diagnostics for regression evaluation.

    Args:
        prediction: (N, T, D) predicted embeddings.
        target: (N, T, D) target embeddings.
        mask: (N, T) binary speech mask.

    Returns:
        Dict with r2_speech, r2_silence, r2_all, r2_per_dim, cosine_sim_speech.
    """
    results = {}

    for name, m in [("speech", mask > 0.5), ("silence", mask < 0.5), ("all", np.ones_like(mask, dtype=bool))]:
        idx = m.reshape(-1)
        if idx.sum() == 0:
            results[f"r2_{name}"] = float("nan")
            continue
        p = prediction.reshape(-1, prediction.shape[-1])[idx]
        t = target.reshape(-1, target.shape[-1])[idx]
        ss_res = ((p - t) ** 2).sum()
        ss_tot = ((t - t.mean(axis=0)) ** 2).sum()
        results[f"r2_{name}"] = float(1 - ss_res / (ss_tot + 1e-8))

    # Per-dimension R²
    speech_idx = (mask > 0.5).reshape(-1)
    if speech_idx.sum() > 0:
        p = prediction.reshape(-1, prediction.shape[-1])[speech_idx]
        t = target.reshape(-1, target.shape[-1])[speech_idx]
        per_dim = []
        for d in range(p.shape[1]):
            ss_res = ((p[:, d] - t[:, d]) ** 2).sum()
            ss_tot = ((t[:, d] - t[:, d].mean()) ** 2).sum()
            per_dim.append(float(1 - ss_res / (ss_tot + 1e-8)))
        results["r2_per_dim"] = per_dim
    else:
        results["r2_per_dim"] = [float("nan")] * prediction.shape[-1]

    # Cosine similarity on speech frames
    if speech_idx.sum() > 0:
        p = prediction.reshape(-1, prediction.shape[-1])[speech_idx]
        t = target.reshape(-1, target.shape[-1])[speech_idx]
        cos = (p * t).sum(axis=1) / (
            np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1) + 1e-8
        )
        results["cosine_sim_speech"] = float(cos.mean())
    else:
        results["cosine_sim_speech"] = float("nan")

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_regression.py::TestDiagnostics -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/speech_decoding/evaluation/metrics.py tests/test_regression.py
git commit -m "feat: framewise R² diagnostics for regression evaluation"
```

---

### Task 10: Config and Training Script

**Files:**
- Create: `configs/per_patient_regression.yaml`
- Create: `scripts/train_regression.py`

- [ ] **Step 1: Create regression config**

```yaml
# Per-patient regression: CE + masked MSE on HuBERT embeddings
# Baseline comparison: per_patient_ce.yaml (PER 0.700, bal_acc 0.266)
experiment: per_patient_regression

model:
  readin_type: spatial_conv
  head_type: flat            # CE head; regression head added by trainer
  d_shared: 64
  hidden_size: 32
  gru_layers: 2
  gru_dropout: 0.3
  temporal_stride: 10        # 200Hz → 20Hz, 50 frames per trial
  num_classes: 10
  blank_bias: 0.0
  d_emb: 64                  # PCA-reduced HuBERT embedding dim

  spatial_conv:
    channels: 8
    num_layers: 1
    kernel_size: 3
    pool_h: 2
    pool_w: 4

training:
  loss_type: ce
  ce_segments: 3
  regression_lambda: 0.3     # sweep: {0.1, 0.3, 1.0}

  stage1:
    epochs: 300
    lr: 1.0e-3
    warmup_epochs: 20
    readin_lr_mult: 3.0
    weight_decay: 1.0e-4
    batch_size: 16
    grad_clip: 5.0
    patience: 7
    eval_every: 10

  augmentation:
    time_shift_frames: 0           # MUST be 0: shift breaks neural-HuBERT frame alignment
    amp_scale_std: 0.3
    channel_dropout_max: 0.2
    noise_frac: 0.05
    feat_dropout_max: 0.3
    time_mask_min: 3
    time_mask_max: 10
    spatial_cutout: false
    temporal_stretch: false         # MUST be false: warp breaks neural-HuBERT frame alignment
    temporal_stretch_max_rate: 0.0

evaluation:
  seeds: [42]
  cv_folds: 5
  primary_metric: per

audio_features:
  dir: data/audio_features    # where .npz files are stored
  d_emb: 64
  hubert_layer: 6
```

- [ ] **Step 2: Create training script**

```python
"""Per-patient regression training: CE + masked MSE on HuBERT embeddings.

Usage:
  python scripts/train_regression.py --config configs/per_patient_regression.yaml
  python scripts/train_regression.py --config configs/per_patient_regression.yaml --patients S14 --lambda_sweep 0.1 0.3 1.0
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.training.regression_trainer import train_per_patient_regression

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
logger = logging.getLogger(__name__)

PS_PATIENTS_SPALDING = ["S14", "S22", "S23", "S26", "S33", "S39", "S58", "S62"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/per_patient_regression.yaml")
    parser.add_argument("--patients", nargs="+", default=["S14"])
    parser.add_argument("--lambda_sweep", nargs="+", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open("configs/paths.yaml") as f:
        paths = yaml.safe_load(f)

    bids_root = paths["ps_bids_root"]
    feat_dir = Path(config.get("audio_features", {}).get("dir", "data/audio_features"))
    lambdas = args.lambda_sweep or [config["training"]["regression_lambda"]]
    # Always include λ=0.0 as the matched CE-only control
    if 0.0 not in lambdas:
        lambdas = [0.0] + list(lambdas)

    for pid in args.patients:
        logger.info("=== Patient %s ===", pid)

        # Load neural data
        ds = load_patient_data(
            pid, bids_root, task="PhonemeSequence", n_phons=3,
            tmin=-1.0, tmax=1.5,
        )

        # Load pre-extracted features
        feat_path = feat_dir / f"{pid}_hubert.npz"
        if not feat_path.exists():
            logger.error("Features not found: %s. Run extract_audio_features.py first.", feat_path)
            continue
        feat = np.load(str(feat_path))
        embeddings = feat["embeddings"]
        speech_mask = feat["speech_mask"]

        # Verify trial count matches
        if len(ds) != len(embeddings):
            logger.error(
                "%s: trial count mismatch: neural=%d, audio=%d. "
                "This may indicate trials excluded during epoch extraction. "
                "Run extract_audio_features.py with --validate to cross-check "
                "syllable labels between phoneme CSV and .fif epoch events. Skipping.",
                pid, len(ds), len(embeddings),
            )
            continue

        for lam in lambdas:
            config["training"]["regression_lambda"] = lam
            for seed in config["evaluation"]["seeds"]:
                logger.info("--- λ=%.2f, seed=%d ---", lam, seed)
                result = train_per_patient_regression(
                    ds, embeddings, speech_mask, config,
                    seed=seed, device=args.device,
                )
                logger.info(
                    "%s λ=%.2f seed=%d: PER=%.3f bal_acc=%.3f R²(speech)=%.3f",
                    pid, lam, seed,
                    result["per_mean"], result["bal_acc_mean_mean"],
                    result.get("r2_speech_mean", float("nan")),
                )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add configs/per_patient_regression.yaml scripts/train_regression.py
git commit -m "feat: regression config and training script"
```

---

### Task 11: Integration Test with Synthetic Data

**Files:**
- Modify: `tests/test_regression.py`

- [ ] **Step 1: Write integration test**

Add to `tests/test_regression.py`:

```python
class TestRegressionIntegration:
    """End-to-end integration: synthetic data → joint training → metrics."""

    def test_overfit_with_structured_embeddings(self):
        """With class-specific embeddings, regression should help CE."""
        np.random.seed(42)
        n_trials = 30
        grid_h, grid_w, T = 8, 16, 100
        n_backbone_frames = T // 10  # 10
        d_emb = 16

        grid_data = np.random.randn(n_trials, grid_h, grid_w, T).astype(np.float32) * 0.1

        # Create 3 phoneme classes with distinct neural + embedding signatures
        labels = []
        embeddings = np.zeros((n_trials, n_backbone_frames, d_emb), dtype=np.float32)
        speech_mask = np.zeros((n_trials, n_backbone_frames), dtype=np.float32)

        for i in range(n_trials):
            p1 = 1 + (i * 3) % 9
            p2 = 1 + (i * 3 + 1) % 9
            p3 = 1 + (i * 3 + 2) % 9
            labels.append([p1, p2, p3])

            # Neural signal: class-specific spatial activation
            for j, p in enumerate([p1, p2, p3]):
                r, c = p % grid_h, p % grid_w
                t_start = j * T // 4 + T // 8
                grid_data[i, r, c, t_start:t_start + T // 6] += 3.0

            # Embedding target: class-specific pattern in speech frames
            speech_mask[i, 3:8] = 1.0
            for j, p in enumerate([p1, p2, p3]):
                frame_start = 3 + j
                if frame_start < n_backbone_frames:
                    embeddings[i, frame_start, p % d_emb] = 2.0

        ds = BIDSDataset(grid_data, labels, "S_test", (grid_h, grid_w))

        config = {
            "model": {
                "readin_type": "spatial_conv",
                "head_type": "flat",
                "d_shared": 64,
                "hidden_size": 32,
                "gru_layers": 2,
                "gru_dropout": 0.2,
                "temporal_stride": 10,
                "num_classes": 10,
                "blank_bias": 0.0,
                "d_emb": d_emb,
                "spatial_conv": {
                    "channels": 8, "num_layers": 1,
                    "kernel_size": 3, "pool_h": 2, "pool_w": 4,
                },
            },
            "training": {
                "loss_type": "ce",
                "ce_segments": 3,
                "regression_lambda": 0.3,
                "stage1": {
                    "epochs": 40, "lr": 1e-3, "readin_lr_mult": 3.0,
                    "weight_decay": 1e-4, "batch_size": 8, "grad_clip": 5.0,
                    "patience": 10, "eval_every": 20, "warmup_epochs": 0,
                },
                "augmentation": {
                    "time_shift_frames": 0, "amp_scale_std": 0.0,
                    "channel_dropout_max": 0.0, "noise_frac": 0.0,
                    "feat_dropout_max": 0.0, "time_mask_min": 2, "time_mask_max": 4,
                },
            },
            "evaluation": {"seeds": [42], "cv_folds": 2, "primary_metric": "per"},
        }

        result = train_per_patient_regression(
            ds, embeddings, speech_mask, config, seed=42, device="cpu"
        )

        # Should produce finite metrics
        assert np.isfinite(result["per_mean"])
        assert np.isfinite(result["r2_speech_mean"])
        # PER should be below random chance (0.89)
        assert result["per_mean"] < 1.0
```

- [ ] **Step 2: Run all regression tests**

Run: `python -m pytest tests/test_regression.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run existing test suite for no regressions**

Run: `python -m pytest tests/ -v -m "not slow"`
Expected: All 115+ fast tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_regression.py
git commit -m "test: integration test for joint CE + MSE regression training"
```

---

### Task 12: Phase 1 Diagnostic — Real S14

**Files:**
- Create: `scripts/diagnose_regression.py`

This script runs after Tasks 1-11 are complete and feature extraction has been done for S14.

- [ ] **Step 1: Write Phase 1 diagnostic script**

```python
"""Phase 1 diagnostic: Can S14 neural data predict HuBERT embeddings?

Reports:
  - Frame-wise R² (speech vs silence vs all)
  - Per-dimension R² breakdown
  - Time-shifted control: +500ms shift
  - Cosine similarity

Usage: python scripts/diagnose_regression.py
"""
import logging
from pathlib import Path

import numpy as np
import yaml

from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.evaluation.metrics import framewise_r2_diagnostics
from speech_decoding.training.regression_trainer import train_per_patient_regression

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    with open("configs/paths.yaml") as f:
        paths = yaml.safe_load(f)
    with open("configs/per_patient_regression.yaml") as f:
        config = yaml.safe_load(f)

    bids_root = paths["ps_bids_root"]
    feat_dir = Path(config.get("audio_features", {}).get("dir", "data/audio_features"))

    # Load data
    ds = load_patient_data("S14", bids_root, task="PhonemeSequence", n_phons=3, tmin=-1.0, tmax=1.5)
    feat = np.load(str(feat_dir / "S14_hubert.npz"))
    embeddings = feat["embeddings"]
    speech_mask = feat["speech_mask"]

    print(f"S14: {len(ds)} trials, embeddings {embeddings.shape}, "
          f"speech_frac={speech_mask.mean():.2f}")

    # Quick config for diagnostic (fewer epochs, 2 folds)
    config["training"]["stage1"]["epochs"] = 100
    config["evaluation"]["cv_folds"] = 2
    config["evaluation"]["seeds"] = [42]

    # Main experiment: CE + MSE
    print("\n=== Main: CE + λ*MSE (λ=0.3) ===")
    config["training"]["regression_lambda"] = 0.3
    result = train_per_patient_regression(
        ds, embeddings, speech_mask, config, seed=42, device="cpu"
    )
    print(f"PER: {result['per_mean']:.3f}")
    print(f"R²(speech): {result.get('r2_speech_mean', 'N/A')}")
    print(f"R²(silence): {result.get('r2_silence_mean', 'N/A')}")

    # Control: CE only (λ=0)
    print("\n=== Control: CE only (λ=0) ===")
    config["training"]["regression_lambda"] = 0.0
    result_ce = train_per_patient_regression(
        ds, embeddings, speech_mask, config, seed=42, device="cpu"
    )
    print(f"PER: {result_ce['per_mean']:.3f}")

    # Control: Time-shifted neural data (+500ms = +10 frames at 20Hz)
    print("\n=== Control: Time-shifted +500ms ===")
    config["training"]["regression_lambda"] = 0.3
    shifted_grid = np.roll(ds.grid_data, shift=100, axis=-1)  # +100 samples at 200Hz = +500ms
    from speech_decoding.data.bids_dataset import BIDSDataset
    ds_shifted = BIDSDataset(shifted_grid, ds.ctc_labels, ds.patient_id, ds.grid_shape)
    result_shift = train_per_patient_regression(
        ds_shifted, embeddings, speech_mask, config, seed=42, device="cpu"
    )
    print(f"PER: {result_shift['per_mean']:.3f}")
    print(f"R²(speech): {result_shift.get('r2_speech_mean', 'N/A')}")

    # Summary
    print("\n=== PHASE 1 SUMMARY ===")
    r2 = result.get("r2_speech_mean", 0)
    r2_shift = result_shift.get("r2_speech_mean", 0)
    print(f"R²(speech, aligned): {r2:.3f}")
    print(f"R²(speech, shifted): {r2_shift:.3f}")
    print(f"R² drop from shift: {1 - r2_shift/(r2+1e-8):.0%}")
    if r2 > 0.05:
        print("✓ PASS: Neural data predicts speech embeddings (R² > 0.05)")
    else:
        print("✗ FAIL: R² too low. Speech embeddings not predictable from neural data.")
        print("  → STOP. Do not proceed to Phase 2.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run diagnostic**

Run: `cd /Users/bentang/Documents/Code/speech && source .venv/bin/activate && python scripts/diagnose_regression.py`

**DECISION GATE:**
- R²(speech) > 0.05 → Proceed to Phase 2 (λ sweep)
- R²(speech) < 0.05 → Stop. Neural data does not predict speech embeddings.
- R²(speech, shifted) close to R²(speech, aligned) → Signal is temporal autocorrelation, not neural content.

- [ ] **Step 3: Commit**

```bash
git add scripts/diagnose_regression.py
git commit -m "feat: Phase 1 diagnostic script for regression validation"
```

---

## Execution Order

```
Task 1  → Add dependencies
Task 2  → Phase 0: alignment validation (DECISION GATE)
Task 3  → Phoneme timing + speech mask
Task 4  → HuBERT extraction
Task 5  → Offline extraction script (run on S14)
Task 6  → Regression head + assembler
Task 7  → Masked MSE loss
Task 8  → Regression trainer
Task 9  → Evaluation diagnostics
Task 10 → Config + training script
Task 11 → Integration tests
Task 12 → Phase 1 diagnostic on real S14 (DECISION GATE)
```

After Task 12, if gates pass:
- Run λ sweep: `{0.1, 0.3, 1.0}` on S14
- Run all 8 Spalding patients with best λ
- Compare population PER to CE baseline (0.700)
- If population improves → proceed to LOPO design
