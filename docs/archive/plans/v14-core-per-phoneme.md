# v14-core per-phoneme + AR teacher-forcing variant

**Plan written:** 2026-04-17. **Status:** DRAFT — awaiting sign-off. **Scope:** an alternate decoder/loss variant for v14-core to address the slot-wise-CE position-conditional limitation surfaced in the 2026-04-16 S14 array run (test PER ≈ chance after best-val restoration).

## Motivation

Current `v14-core` uses `P(phoneme_k | full_trial_window, slot_idx=k)` — slot-wise CE with `slot_emb[k]` making each position a separate classifier. At 97 trials / 56 unique tokens, this encourages token-memorization rather than phoneme compositionality: `/B/` at slot 0 (in `bak`) and `/B/` at slot 2 (in `kab`) are trained as independent classes.

Ben's pre-v14 baseline (`per-phoneme MFA + flat head`, PER 0.734) wins exactly because it inverts the conditioning: `P(phoneme | localized window around that phoneme)`. Same phoneme → same training signal regardless of position.

The proposed variant keeps v14's architectural strengths (Brainnetome-grounded tokens, combined spatiotemporal backbone) while switching to position-invariant labeling, plus an AR prev-phoneme embedding for language-model-like prior.

## Design

### Data contract

**Source of phoneme onsets.** The phoneme-level sister `.fif` at `epoch(phonemeLevel)(CAR)/sub-<pt>/epoch(band)(power)/sub-<pt>_task-PhonemeSequence_desc-productionZscore_highgamma.fif`. Zac does MFA alignment upstream; the file is delivered phoneme-locked. This is exactly what Ben's 0.734 baseline loaded — see `src/speech_decoding/archive/legacy/data/bids_dataset.py:207–296` (`load_per_position_data`). No MFA sidecar, no new artifact.

Key facts the legacy loader exploits (and we'll inherit):
- `epochs.events` has `n_trials × 3` rows, ordered consecutively `[t0p0, t0p1, t0p2, t1p0, t1p1, t1p2, ...]`.
- `event_id` is the 9 phoneme keys `{'a':1, 'ae':2, 'b':3, 'g':4, 'i':5, 'k':6, 'p':7, 'u':8, 'v':9}` (the `#18` mapping).
- `trial_id = epoch_idx // 3`, `phoneme_pos = epoch_idx % 3` — derivable from row index alone; no upstream metadata needed.
- The per-phoneme window `[-0.15, 0.5]s` is a plain `epochs.crop(...)` on the already-aligned file.

The earlier worry that the phoneme-level `.fif` might be "pos-0 only" was wrong. The `#34` audit finding — "pos-0 phoneme epoch equals trial-level at raw-sample resolution" — means the file's pos-0 rows coincide with the trial-level `.fif`'s trial-onset epochs, not that the file contains only pos-0. It contains all 3 phonemes per trial, in the canonical order.

**Sample shape.**
```
signal[N_e, T_pp]                # float32, z-scored HGA, 200 Hz, MFA-aligned
patient_id                       # str
label                            # long scalar, 0..8 (ARPA index, #16)
prev_phoneme                     # long scalar, 0..8 or BOS sentinel -1
trial_id                         # int — CV group key (keeps grouped-by-token)
phoneme_pos                      # int 0/1/2 — kept for eval-side slot-averaged PER
electrode_grid_layout[N_e, 2]    # int (row, col)
electrode_grid_shape             # (H_p, W_p)
electrode_active_mask[N_e]       # bool
support[N_e, 15]                 # float32
```

**Window width `T_pp`.** Match Ben's baseline: `tmin=-0.15, tmax=0.5` (0.65 s) per-phoneme. At 200 Hz → 130 samples. Apples-to-apples with the 0.734 baseline. The temporal tokenizer's `kernel=30, stride=10` gives `(130 - 30)/10 + 1 = 11` temporal tokens per sample (vs 28 at T=300 in slot mode) — every shape test that hardcodes 28 needs updating.

**Batch structure.** One batch = samples from the same patient and the same `phoneme_pos`? Or mixed positions? **Default:** mixed positions within a patient — that's the whole point; the model should learn phoneme-invariant-of-position. Grouped-by-trial batching (all 3 phonemes of a trial in same batch) is an optional refinement for CV only, not training.

### Model

Same `NeuralFieldPerceiver` backbone, **modified decoder** — not a slot-decoder at all:

- Drop `slot_emb[L, d]`.
- Keep `base_query`, `bos_emb`, `prev_emb: Embedding(9, d)`, shared `head: Linear(d, 9)`.
- Single query per sample: `q = base_query + prev_emb_or_bos(prev_phoneme)` → `(B, 1, d)`.
- Same one-block self-attn + cross-attn + FFN + head. With `L=1` the self-attn sublayer degenerates to a learned linear projection of `q` (single key/query, softmax over length 1 = 1) — kept for structural parity with the slot decoder, not for expressive power. The real work is cross-attn + FFN.
- Loss: plain CE over 9 classes (no slot average — each sample is one phoneme).

No new architecture primitives; just a simpler decoder head. Call it `ARPhonemeDecoder` to distinguish from `ARDecoder`.

### Training

- Teacher forcing: `prev_phoneme` is always ground truth during training (BOS for `phoneme_pos=0`).
- Same optimizer / LR / cosine-warmup / grad-clip as v14-core.
- Same val-min best-state restoration (the fix landed in `train.py` on 2026-04-17).
- Effective batch: keep grad-accum at 4. `trials_per_batch=8` becomes `phonemes_per_batch=24` naturally (3× more gradient signal per epoch, which is the other reason to try this).

### Eval

Two modes, both reported:

1. **Per-phoneme PER (direct):** flat accuracy over all phoneme-level val/test samples, with `prev_phoneme` = ground truth at eval time. This is what the loss directly optimizes; sanity check, not the baseline comparison number.
2. **Slot-averaged PER via exhaustive AR decode (comparable to the 0.734 baseline):** per trial, compute 3 backbone memories (one per phoneme window). Enumerate all `9³=729` phoneme-sequence hypotheses; for each hypothesis `(p_0, p_1, p_2)`, run three decoder forwards with `prev=BOS, p_0, p_1`, gather the log-prob of `p_i` at each slot, sum, and take argmax across hypotheses. Same protocol as the slot decoder's `exhaustive_decode`, now over per-phoneme memories.

Compute sanity: 3 backbone forwards per trial (unavoidable, same as greedy). Decoder: 2187 decoder-block forwards per trial, each is one-layer attn over `(N_e × T_pp, d) ≈ (270, 64)` memory — batches to one GPU call in microseconds. Backbone dominates; there's no reason to prefer greedy over exhaustive.

### CV + grouping

`cv.py`'s grouped-by-token splitter must stay intact: the CV key is `token` at the trial level (same trial → same fold). Implementation: build folds on **trial IDs**, then expand each trial into its 3 phoneme samples inside the fold. No trial crosses a fold boundary; all 3 of a trial's phonemes land in the same fold together.

## Files

New:
- `src/speech_decoding/v14/phoneme_dataset.py` — `V14PhonemeDataset`, `V14PhonemeSample`, `collate_v14_phoneme_batch` (mirrors `dataset.py` structure).
- `src/speech_decoding/v14/phoneme_decoder.py` — `ARPhonemeDecoder` (single-query variant of `ARDecoder`).
- `src/speech_decoding/v14/phoneme_model.py` — `NeuralFieldPerceiverPerPhoneme` (swaps decoder, keeps tokenizer + grid_mixer + parcel_embedding + backbone).
- `src/speech_decoding/v14/phoneme_run_fold.py` — parallels `run_fold.py` but iterates over phoneme samples; trains via same `train_one_fold` with a different collate + loss hook.
- `tests/v14/test_phoneme_dataset.py`, `tests/v14/test_phoneme_decoder.py`, `tests/v14/test_phoneme_run_fold.py`.

Modified:
- `src/speech_decoding/v14/train.py` — split `compute_slot_ce` into `compute_slot_ce` (unchanged) and `compute_flat_ce` (new, one logit per sample). `train_one_fold` learns a `loss_fn` kwarg (defaults to current slot-CE for back-compat).
- `src/speech_decoding/v14/config.py` — add `PerPhonemeConfig` (T_pp, window tmin/tmax, MFA onset file path) and a top-level `V14PerPhonemeConfig` that composes the existing subconfigs + `PerPhonemeConfig`.
- `scripts/v14_core/train_v14_core.py` — learn a `--mode {slot,per-phoneme}` flag; dispatches to the right dataset + model + run_fold.
- `scripts/v14_core/v14_core_dcc.sh` — second sbatch variant or a flag; arrays unchanged (same 5 folds × 3 seeds × 2 depths).

Artifacts needed: none new. The phoneme-level `.fif` per patient is already on Box / DCC (same BIDS tree as the trial-level `.fif` we already load).

## Phases (TDD, one commit per task)

### Phase P1 — Phoneme-level `.fif` audit

- [ ] For S14 (sole validation patient), load `epoch(phonemeLevel)(CAR)/sub-S14/epoch(band)(power)/sub-S14_task-PhonemeSequence_desc-productionZscore_highgamma.fif` and assert: `len(events) == 3 × n_trials_from_trial_level_fif`, `event_id == {'a':1, 'ae':2, 'b':3, 'g':4, 'i':5, 'k':6, 'p':7, 'u':8, 'v':9}`, raw `tmin ≤ -0.15` and `tmax ≥ 0.5` so the `[-0.15, 0.5]s` crop is in-bounds.
- [ ] Cross-check that the trial tokens recovered from the phoneme-level file (group-by-3, PS-encode via `ARPA2PS`) match the trial-level `.fif`'s `event_id` tokens for the same trial indices.
- [ ] One-page QC report at `docs/qc/phoneme_level_fif_audit_s14.md`. Blocker if any assertion fails.

### Phase P2 — Dataset

- [ ] `V14PhonemeDataset` loads the phoneme-level `.fif`, crops to `[-0.15, 0.5]s` (T_pp=130), and emits `V14PhonemeSample` per epoch (`trial_id = idx // 3`, `phoneme_pos = idx % 3`, `prev_phoneme = BOS if pos == 0 else label[pos-1]` via ground-truth group-of-3 lookup).
- [ ] Uses the same `#12` channel-map bridge and A1 per-electrode support cache as `V14TrialDataset`. Artifact-channel filtering from `info["bads"]` per `#11`.
- [ ] `collate_v14_phoneme_batch` stacks into a grouped-by-patient batch with `prev_phoneme`, `phoneme_pos`, `trial_id` alongside the per-electrode tensors.
- [ ] Tests: 3 samples per trial; `prev_phoneme` chain matches labels with BOS at pos 0; signal shape `(N_e, 130)`; `trial_id` increases every 3 epochs; tokens recovered from consecutive triplets match the trial-level `.fif`.

### Phase P3 — Decoder + model

- [ ] `ARPhonemeDecoder` with single query `base + prev_emb_or_bos`. Shape test: `(B, 9)` logits.
- [ ] `NeuralFieldPerceiverPerPhoneme` wires tokenizer + grid_mixer + parcel_embedding + backbone + `ARPhonemeDecoder`.
- [ ] Overfit test: 10 train_step on a fixed batch drops CE by ≥0.1.

### Phase P4 — Train loop plumbing

- [ ] `compute_flat_ce(logits, labels)` + `train_one_fold` `loss_fn` kwarg.
- [ ] Best-val restoration path unchanged (already fixed).
- [ ] Eval: per-phoneme PER (teacher-forced) + exhaustive AR slot-averaged PER.

### Phase P5 — CV + runner

- [ ] `run_one_fold` variant expands folds (trial IDs) into phoneme-sample subsets.
- [ ] CLI + DCC sbatch can select mode. One-patient smoke (S14, 1 fold, 1 seed, depth=1) on DCC.

### Phase P6 — Full S14 run

- [ ] Submit the 30-job S14 array in per-phoneme mode.
- [ ] Compare per-phoneme-PER and AR-chain slot-averaged-PER against the slot-wise CE run and Ben's 0.734 baseline.

## Success criteria

1. **Per-phoneme PER on S14 val < 0.5** (vs chance 8/9 ≈ 0.889). Proof-of-life that the label switch works.
2. **AR-chain slot-averaged test PER on S14 ≤ 0.85**. Beating the prior v14-core run (≈0.89 at chance) is the bare minimum.
3. **Stretch: match Ben's 0.734 baseline** on S14 over 3 seeds. That would mean the v14 backbone + Brainnetome tokens are carrying their weight.

If (1) fails, the problem is not slot-wise CE — it's somewhere upstream (tokenizer / grid mixer / backbone / support cache). That's a useful signal on its own.

## Open questions for review

1. **Batch structure.** Mixed phoneme_pos per batch, or separate? Default: mixed (the whole compositionality point).
2. **`T` assumptions in tests.** The shared backbone now sees `T=11` tokens instead of `T=28`; any tests that hardcode 28 need updating.

Resolved before sign-off: window = `[-0.15, 0.5]s`; onset source = phoneme-level `.fif`; eval = exhaustive AR.

## Not in scope

- Cross-patient shared-weights training (v14-full).
- Learned per-patient calibration (Phase 2).
- Any atlas-side change (still `v2c + snap`).
- Any change to tokenizer / grid mixer / parcel embedding / backbone architecture.
