# Neuroprobe Benchmark — Data Reference

*Compiled 2026-04-24 from `insight-neuro/neuroprobe` (commit at clone time) + Zahorodnii et al. arXiv 2509.21671.*

Consult when working on the cross-subject hillclimb (see `docs/neuroprobe/plan.md`). Everything below is derived from the repo's `neuroprobe/config.py`, `neuroprobe/datasets.py`, `neuroprobe/train_test_splits.py`, `examples/eval_population.py`, `examples/eval_utils.py`, `tests/test_submission_format.py`, and `SUBMIT.md`.

## Where to find things

- Paper: `pastwork/NEWPAPERS4/2509.21671v1.pdf`
- Repo: `https://github.com/insight-neuro/neuroprobe` (mirror of `azaho/neuroprobe`)
- Website + submission portal: `https://neuroprobe.dev`
- Local clone (exploration): `/tmp/neuroprobe_explore/neuroprobe`
- Pip package: `pip install neuroprobe` (version 0.1.7 at the clone)

## Dataset

- Source: BrainTreebank sEEG (Wang et al. 2024). 10 subjects, ages 4–19, watching 26 Hollywood movies. ~40–43 h raw.
- Sampling rate: **2048 Hz**, raw voltage, h5 format: `sub_{subject_id}_trial{trial_id:03}.h5`, one dataset `data/electrode_<i>`.
- Electrode localization: `localization/elec_coords_full.csv` (cortical-projected XYZ); per-subject `localization/sub_{id}/depth-wm.csv` carries the Desikan-Killiany region label per electrode (`DesikanKilliany` column).
- Movie transcripts: `transcripts/<movie_id>/features.csv`.
- Per-trial trigger timings: `subject_timings/sub_{id}_trial{t:03}_timings.csv` maps movie time → h5 sample index.

## Canonical subject / trial layout

`NEUROPROBE_FULL_SUBJECT_TRIALS` (27 sessions total in BrainTreebank):
```
S1 : 0,1,2
S2 : 0,1,2,3,4,5,6
S3 : 0,1,2
S4 : 0,1,2
S5 : 0
S6 : 0,1,4
S7 : 0,1
S8 : 0
S9 : 0
S10: 0,1
```

**Eval subset** (`NEUROPROBE_LITE_SUBJECT_TRIALS`, 12 sessions):
```
(1,1), (1,2), (2,0), (2,4), (3,0), (3,1),
(4,0), (4,1), (7,0), (7,1), (10,0), (10,1)
```

**Nano subset** (`NEUROPROBE_NANO_SUBJECT_TRIALS`, 6 sessions, same but S2/trial-4 and smaller electrode+sample caps):
```
(1,1), (2,4), (3,1), (4,0), (7,1), (10,1)
```

Nano mode is restricted: it works with Within-Session and Cross-Subject only, NOT Cross-Session (asserted in `eval_population.py`).

## Electrode subsets (fixed — not chosen by submitter)

`NEUROPROBE_LITE_ELECTRODES[subject_identifier]` → hand-picked list (~120 per subject) chosen by the authors for highest linear decoding performance on the tasks. `NEUROPROBE_NANO_ELECTRODES` → ~30 per subject. The full tables are inlined in `neuroprobe/config.py`; don't reproduce here, consult source.

Subject identifier format: `btbank{subject_id}`.

Corrupted + missing-coordinate electrodes per subject are filtered at `BrainTreebankSubject._filter_electrode_labels`. Trigger channels (names starting with `DC` or `TRIG`) are dropped unconditionally.

## Tasks (15 binary AUROC probes)

Keys → display names (`NEUROPROBE_TASKS_MAPPING`):

```
onset              → Sentence Onset
speech             → Speech
volume             → Volume
delta_volume       → Delta Volume
pitch              → Voice Pitch
word_index         → Word Position
word_gap           → Inter-word Gap
gpt2_surprisal     → GPT-2 Surprisal
word_head_pos      → Head Word Position
word_part_speech   → Part of Speech
word_length        → Word Length
global_flow        → Global Optical Flow
local_flow         → Local Optical Flow
frame_brightness   → Frame Brightness
face_num           → Number of Faces
```

### How labels are derived (`datasets.py::BrainTreebankSubjectTrialBenchmarkDataset.__init__`)

Under `binary_tasks=True` (default):

- **Continuous features** (`volume`, `delta_volume`, `pitch`, `frame_brightness`, `global_flow`, `local_flow`, `gpt2_surprisal`, `word_length`): label 1 = top quartile of the trial's distribution; label 0 = bottom quartile. Middle 50% discarded.
- **`onset`**: positive = words at sentence-initial positions (`is_onset == 1`); negative = nonverbal windows.
- **`speech`**: positive = all words; negative = nonverbal windows.
- **`face_num`**: positive = frame at word-onset shows > 0 faces; negative = 0 faces.
- **`word_index`**: positive = word position 0 in sentence; negative = word position 1.
- **`word_head_pos`**: positive = head word (`bin_head == 0`); negative = `bin_head == 1`.
- **`word_part_speech`**: positive = VERB; negative = NOUN. (Other POS classes dropped.)
- **`word_gap`**: positive = gap in top quartile; negative = bottom quartile; computed within-sentence only.

Classes are balanced: `n_samples_each = min(len(pos), len(neg))`. Then capped at `max_samples // n_classes`.

### Sample caps

```
NEUROPROBE_LITE_MAX_SAMPLES = 3500  (balanced → 1750 per class)
NEUROPROBE_NANO_MAX_SAMPLES = 1000  (balanced → 500 per class)
```

This is **per session per task**, after the per-class cap.

### Eval window

Leaderboard convention: exactly `[0, 1]` s from word onset = 2048 samples at native 2048 Hz. Controlled by the `--only_1second` flag in `eval_population.py`; without it the script sweeps 0.25s windows over `[-0.5, 1.5]` s. The leaderboard parser only ingests the `one_second_after_onset` bin.

Nonverbal windows are defined at cache-build time (`process_subject_trials.py::obtain_nonverbal_df`): gaps between words of ≥ 1 s pad + 1 s window, stepped at 50% overlap (`NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP = 0.5`, `NEURAL_DATA_NONVERBAL_WINDOW_PADDING_TIME = 2`).

## Splits (`train_test_splits.py`)

### Within-Session

2-fold CV on a single `(subject, trial)`. `KFold(shuffle=False)` — temporal blocks. First half of test-fold becomes val, second half becomes test.

Leaderboard runs: **12 sessions × 15 tasks × 2 folds = 360 evaluations**, reported as 180 (one `folds` list of length 2 per session-task).

### Cross-Session

Train on the **other** trial of the same subject (the one from `NEUROPROBE_LITE_SUBJECT_TRIALS`), test on target. One fold. Same val/test half-split.

Leaderboard runs: **12 × 15 = 180 evaluations**.

### Cross-Subject (the one we target)

Default — `include_all_train_subjects=False`:
- **Training: S2 / trial-4 only** (`DS_DM_TRAIN_SUBJECT_ID = 2, DS_DM_TRAIN_TRIAL_ID = 4`).
- Test: each of the 10 remaining `(subject, trial)` pairs in `NEUROPROBE_LITE_SUBJECT_TRIALS` (S2 is never tested on).
- One fold per test session.
- Val = first half of test session samples; test = second half.

Non-default `include_all_train_subjects=True`:
- Returns a **list of 10 folds per test session**, one per other source subject — pairwise 1-to-1, NOT N−1 combined. The repo does not expose a combined N−1 → 1 split.

Legacy aliases: `generate_splits_DS_DM == generate_splits_cross_subject`, `SS_DM == cross_session`, `SS_SM == within_session`.

Leaderboard runs: **10 × 15 = 150 evaluations**.

### Cross-subject baseline uses DK region averaging

`eval_population.py` invokes `combine_regions()` for Cross-Subject runs (`examples/eval_utils.py:get_region_labels` → `subject.get_all_electrode_metadata()['DesikanKilliany']`). Train and test electrode matrices are mean-pooled within each Desikan-Killiany region, then the intersection of regions is taken. This is how **every submitted cross-subject baseline aligns subjects** — region-averaging discards intra-region variance but makes the cross-subject feature dim fixed.

This is the most important hidden detail of the benchmark. Our thesis is that a v14 BNA-parcel-embedding with attention is exactly this move done better.

## Coordinate types (`BrainTreebankSubject.get_electrode_coordinates`)

```
coordinates_type="cortical"  # "STANDARDIZED BRAIN ATLAS CORTICAL PROJECTION" — exact space unclear
coordinates_type="mni"       # NotImplementedError — "will be added in the future ASAP"
coordinates_type="lpi"       # raw L, P, I coordinates (50–200 mm range)
```

Cortical coords come from `localization/elec_coords_full.csv` (columns `Electrode, Subject, X, Y, Z`). The quickstart uses `coordinates_type="cortical"` by default.

**Open question for our program**: what anatomical space is `cortical`? Likely fsaverage-like but unverified in code comments. First Stage-0 task.

## Preprocessing options (reference baselines, `eval_population.py`)

Chainable via dash in `--preprocess.type`:
- `none` — raw voltage
- `stft_absangle` / `stft_realimag` / `stft_abs` — FFT magnitude (+ phase / + imag) after STFT; params `nperseg=512`, `poverlap=0.75`, `window=hann`, `max_frequency=150`, `min_frequency=0`
- `laplacian` — Laplacian re-reference using adjacent-numbered same-stem electrodes (e.g. `T1b2` averages `T1b1, T1b3`). Keeps non-Laplacian channels untouched by default.
- `remove_line_noise` — notch at 60 Hz + harmonics up to 300 Hz
- `downsample_200` — resample_poly to 200 Hz
- `projection` — PCA or random projection down to `dim=192`

The winning baseline: `--preprocess.type laplacian-stft_abs --classifier_type linear`.

After preprocessing, features are flattened, StandardScaler-normalized, and passed to `LogisticRegression(max_iter=10000, tol=1e-3)` or an MLP (`hidden_dims=[1024,1024]`, Adam `lr=1e-5`, batch 200, Dropout 0.2).

## Submission format (`SUBMIT.md` + `tests/test_submission_format.py`)

Directory name: `leaderboard/MODELNAME_FIRSTNAME_LASTNAME_DD_MM_YYYY/`. Required contents:

- `metadata.json` — fields `model_name, description, author, organization, organization_url, timestamp` (all required; `logo_url` optional).
- `PUBLICATION.bib` — any non-empty bibtex; `N/A` accepted as placeholder but reviewers won't like it.
- `ATTESTATION.txt` — must contain the two required phrases verbatim, with ≥ 2 `SIGN` statements:
  ```
  I attest that the training and test splits of Neuroprobe were respected and taken from the `neuroprobe/train_test_splits.py` function.
  SIGN **Full Name**

  I attest that the submitted model was not pretrained on any data that intersects with any data of Neuroprobe.
  SIGN **Full Name**
  ```
- At least one split directory (`Within-Session/`, `Cross-Session/`, `Cross-Subject/`) — partial submissions are legal. DIVER-1 submitted only Within-Session, and got ranked #1 there; RNN-gru submitted only Cross-Session.
- Inside each split directory: exactly 15 `population_<task>.json` files, one per task key from `NEUROPROBE_TASKS_MAPPING`. Extra filenames = fail. Missing = fail.

### `population_<task>.json` shape

```json
{
  "model_name": "...",
  "description": "...",
  "author": "...",
  "organization": "...",
  "organization_url": "...",
  "timestamp": 0,
  "evaluation_results": {
    "btbank1_1": {
      "population": {
        "one_second_after_onset": {
          "time_bin_start": 0.0,
          "time_bin_end": 1.0,
          "folds": [
            {"test_roc_auc": 0.82}
          ]
        }
      }
    },
    ...
  }
}
```

Only `test_roc_auc` is mandatory per fold. `train_accuracy, train_roc_auc, test_accuracy` are optional (validator commented them out). Metric must be in `[0, 1]`.

For Within-Session the `folds` list has length 2; for Cross-Session and Cross-Subject it has length 1.

### CI validation

`.github/workflows/validate_submissions.yml` runs **only** `tests/test_submission_format.py`. Pure format + attestation check. No code execution, no reproducibility verification, no numerical ceiling check. The gate is social — attestation plus a reviewable `PUBLICATION.bib`.

## Pretraining rules (`SUBMIT.md`)

**Off-limits** (the 12 eval sessions):
```
btbank1_1, btbank1_2,
btbank2_0, btbank2_4,
btbank3_0, btbank3_1,
btbank4_0, btbank4_1,
btbank7_0, btbank7_1,
btbank10_0, btbank10_1
```

**Allowed full sessions**:
```
btbank1_0,
btbank2_1, btbank2_2, btbank2_3, btbank2_5, btbank2_6,
btbank3_2,
btbank4_2,
btbank5_0,
btbank6_0, btbank6_1, btbank6_4,
btbank8_0,
btbank9_0
```

**Allowed partial sessions** (time-slices of eval trials for S7/S10 that don't overlap eval windows, ~20 min each, hosted on Google Drive):
```
btbank7_100, btbank7_101, btbank7_102,
btbank10_100, btbank10_101
```

14 full + 5 partial ≈ **~25 h of legal in-distribution BrainTreebank pretraining**. External data (non-BrainTreebank) is unconstrained.

## Existing leaderboard (as of 2026-04-24)

Mean AUROC per split. DIVER-1 and RNN-gru submitted partial entries only.

```
Model                       Within-Session  Cross-Session  Cross-Subject
DIVER-1 (tiny, frozen)      0.678           —              —
Linear (Lap+spec)           0.660           0.648          0.539
Linear (spectrogram)        —               0.626          0.528
BrainBERT untrained frozen  0.585           —              0.527
PopulationTransformer       0.545           0.566          0.526
BrainBERT frozen            0.586           0.581          0.522
Linear (raw voltage)        —               0.576          0.510
RNN (gru)                   —               ~              —
Chance                      0.500           0.500          0.500
```

Precise per-task numbers are in `leaderboard/<model>/<split>/population_<task>.json`.

## Data-loading entry points

```python
from neuroprobe import BrainTreebankSubject, BrainTreebankSubjectTrialBenchmarkDataset
from neuroprobe import (generate_splits_within_session,
                        generate_splits_cross_session,
                        generate_splits_cross_subject)

subject = BrainTreebankSubject(subject_id=1, cache=True, dtype=torch.float32,
                               coordinates_type="cortical")
splits  = generate_splits_cross_subject({1: subject, 2: train_subj}, 1, 1,
                                         eval_name="onset", output_dict=True)
# splits is a list of one fold dict: {"train_dataset", "val_dataset", "test_dataset"}
```

Sample `__getitem__` output with `output_dict=True`:
```python
{
  "data": torch.Tensor,          # (n_electrodes, 2048)
  "label": int,                  # 0 or 1 (binary_tasks=True)
  "electrode_labels": list[str], # length n_electrodes
  "electrode_coordinates": torch.Tensor,  # (n_electrodes, 3)
  "metadata": {"dataset_identifier": "braintreebank",
               "subject_id": int, "trial_id": int, "sampling_rate": 2048}
}
```

`output_indices=True` replaces `"data"` with `(index_from, index_to)` tuple into the session h5 — useful for custom preprocessing pipelines.

## Useful constants reference

```
NEUROPROBE_GLOBAL_RANDOM_SEED     = 42
DS_DM_TRAIN_SUBJECT_ID            = 2
DS_DM_TRAIN_TRIAL_ID              = 4
NEUROPROBE_LITE_MAX_SAMPLES       = 3500
NEUROPROBE_LITE_N_FOLDS           = 2   (Within-Session only)
NEUROPROBE_NANO_MAX_SAMPLES       = 1000
NEUROPROBE_NANO_N_FOLDS           = 2
START_NEURAL_DATA_BEFORE_WORD_ONSET = 0   (overridden to 0 by leaderboard's --only_1second)
END_NEURAL_DATA_AFTER_WORD_ONSET    = 1
NEURAL_DATA_NONVERBAL_WINDOW_PADDING_TIME = 2
NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP      = 0.5
```

## Reference scripts in the repo

- `examples/quickstart.ipynb` — 30-line end-to-end example
- `examples/eval_population.py` + `examples/eval_utils.py` — reference population-level eval pipeline (the submitted linear and MLP baselines)
- `examples/eval_single_electrode.py` — per-electrode eval variant
- `examples/run_eval_population_mlp.sh` — reference SLURM array (12 tasks × 15 eval names)
- `analyses/process_subjects_trials/process_subject_trials.py` — builds `words_df` and `nonverbal_df` caches from the raw BrainTreebank trigger timings
- `analyses/neuroprobe_generate_leaderboard_results.py` — aggregates per-session result JSONs into the leaderboard format
- `braintreebank_download_extract.py --lite` — downloads only the 12 Lite sessions (~90 GB)

## Author group (for strategic context)

Andrii Zahorodnii (MIT CBMM/McGovern) + Christopher Wang (MIT, BrainBERT) + Bennett Stankovits (MIT) + Charikleia Moraitaki (MIT) + Geeling Chau (Caltech, PopT) + Andrei Barbu, Boris Katz (MIT CSAIL) + Ila Fiete (MIT McGovern). This is the iEEG-FM community; a credible cross-subject result is a natural cold-open.
