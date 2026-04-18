# v14-core baseline-aligned rework

**Plan written:** 2026-04-17. **Status:** DRAFT — awaiting sign-off. **Scope:** full Phase-1 v14-core rewrite that re-aligns the spatial/temporal stack with Ben's 0.734 baseline architecture, adds the Brainnetome parcel embedding as the cross-patient hook, and switches to per-phoneme labeling + minimal AR teacher-forcing. Supersedes `docs/plans/v14-core-per-phoneme.md` (which captured only the decoder/labeling change).

## Motivation

The 2026-04-16 S14 array run at v14-core (B-1, d=64, slot-wise CE over 3 positions, per-electrode tokens, no pool) landed at ~chance test PER after best-val restoration. Two root causes, both addressable by re-aligning with baseline:

1. **Slot-wise CE is position-conditional.** `/B/` at slot 0 (`bak`) and slot 2 (`kab`) are trained as independent classes via `slot_emb[i]`. At 97 trials, the model memorizes tokens rather than phonemes. Ben's 0.734 baseline avoids this by construction: per-phoneme MFA windowing + flat head gives `P(phoneme | localized window)`, position-invariant.

2. **v14 B-1 was massively overparameterized for 97 trials.** d=64 backbone × 3 attention blocks = ~150k params just in attention, total model ~500k–1M. Baseline was ~40k. At 97 trials × 3 positions = 291 samples, the 10-samples-per-param rule caps useful capacity around 30k. v14 was 15–25× over.

This plan re-aligns the spatial/temporal stack with baseline's proven shape, adds parcel embedding as the one genuine v14 addition, and keeps the attention backbone + AR decoder but at a width the data can support.

## Full architecture

```
(B, N_e, 130)                                 per-phoneme window [-0.15, 0.5]s at 200 Hz
  ▼  grid-scatter                             (B, 1, H_p, W_p, 130)
  ▼  Conv2d(1→8, k=3, pad=1)                  72 params       — baseline-exact spatial filter bank
  ▼  masked-mean pool to (4, 8)               (B, 8, 4, 8, 130)  — 32 cells
  ▼  per-cell Conv1d(8→32, k=30, stride=10)   7.7k params     — temporal compress (130→11) + lift (8→32)
                                              (B, 32, 4, 8, 11)
  ▼  + pooled_support @ P_emb[15, 32]         480 params      — cell-identity parcel embedding
                                              (B, 32, 4, 8, 11)
  ▼  flatten → (B, 352, 32)
  ▼  attention backbone × 3 blocks at d=32    ~36k params     — combined spatiotemporal attn, RoPE on t
                                              (B, 352, 32)
  ▼  D1 decoder: mean-pool + prev_emb + Linear(32→9)   ~617 params
  exhaustive 9³ AR chain at eval (per trial: 3 backbone forwards + 2187 decoder forwards)
```

**Total params: ~45k.** Aligned with baseline's ~40k.

## Data contract

### Source

Load `{bids_root}/derivatives/epoch(phonemeLevel)(CAR)/sub-{pt}/epoch(band)(power)/sub-{pt}_task-PhonemeSequence_desc-productionZscore_highgamma.fif`. This is MFA-aligned upstream by Zac and is exactly what Ben's 0.734 baseline loaded (`src/speech_decoding/archive/legacy/data/bids_dataset.py:207–296`).

Key properties the loader exploits:
- `epochs.events` has `n_trials × 3` rows, ordered `[t0p0, t0p1, t0p2, t1p0, ...]`.
- `event_id` is the 9-phoneme mapping `{'a':1, 'ae':2, 'b':3, 'g':4, 'i':5, 'k':6, 'p':7, 'u':8, 'v':9}` (per `#18`).
- `trial_id = idx // 3`, `phoneme_pos = idx % 3` — both derivable from row position alone.
- `epochs.crop(tmin=-0.15, tmax=0.495, include_tmax=True)` yields 130 samples at 200 Hz (half-open upstream convention; same pattern as the v14-core trial-level loader).

### Sample shape

```
signal[N_e, T_raw=130]             # float32, z-scored HGA, 200 Hz, MFA-locked
patient_id                         # str
label                              # long scalar 0..8 (alphabetical ARPA index, #16)
prev_phoneme                       # long scalar 0..8 or BOS sentinel -1
trial_id                           # int — CV group key (preserves grouped-by-token)
phoneme_pos                        # int 0/1/2 — kept for slot-averaged PER eval
electrode_grid_layout[N_e, 2]      # long (row, col) on patient device grid
electrode_grid_shape               # tuple (H_p, W_p) — per-patient
electrode_active_mask[N_e]         # bool (non-artifact AND not pad)
support[N_e, 15]                   # float32, raw BNA Tier-1 probability (#5)
```

Channel inclusion: all non-artifact channels per `#11`. Artifact mask from `info["bads"]`. Support from the existing A1 cache `data/atlas/support_cache_v2c_snap/<pt>_support_tier1.csv`.

### Collator

Grouped-by-patient batch (`#31`):
```
signal[B, N_e, 130]               # stacked
labels[B]                          # long 0..8
prev_tokens[B]                     # long, BOS at pos 0, else label[b, pos-1]
phoneme_pos[B]                     # long 0..2
trial_id[B]                        # long
electrode_grid_layout[B, N_e, 2]   # stacked; shared across batch
electrode_grid_shape               # tuple, shared
electrode_active_mask[B, N_e]      # stacked; shared across batch (same per patient)
support[B, N_e, 15]                # stacked; shared across batch
patient_id                         # str
```

## Per-stage design decisions

### Stage 1 — grid-scatter

`(B, N_e, T_raw) → (B, 1, H_p, W_p, T_raw)`. Electrode `e` at `(row_e, col_e)` writes its signal into `grid[:, 0, row_e, col_e, :]`. Empty grid slots (rare; uECoG is rigid rectangular) stay zero. No cropping, no discarding.

### Stage 2 — Conv2d (learned)

`Conv2d(1→8, k=3, stride=1, padding=1)` per time-step, pre-norm LayerNorm + GELU + residual. **72 params.** Single-channel input = raw voltage at each grid cell. 8 output filters = baseline's validated spatial capacity.

### Stage 3 — masked-mean pool (fixed)

Precompute once per patient:
```python
cell_of: (N_e,)                    long, which pooled cell each electrode belongs to
                                   — derived from electrode_grid_layout + target pool shape (4, 8)
                                   — matches AdaptiveAvgPool2d's deterministic integer assignment
active_count: (n_cells,)           float, Σ_e active[e] for e in each cell
                                   — patient-constant since info["bads"] doesn't change trial-to-trial
```

Per batch, apply **masked-mean** via `scatter_mean`-style op:
```
pooled_feat[c, d, t] = (Σ_{e in c} active[e] · feat[e, d, t]) / active_count[c]
pooled_supp[c, p]    = (Σ_{e in c} active[e] · supp[e, p])    / active_count[c]
```

Same operator, two inputs. Cells with zero active electrodes (edge case, likely none on uECoG) get zeros + `cell_active_mask[c] = False`.

**Pool shape `(4, 8)` = 32 cells.** Baseline-exact.

### Stage 4 — per-cell Conv1d (learned)

`Conv1d(8→32, k=30, stride=10)` applied independently per cell, shared weights across cells. **7,680 params.** Does two jobs:
- Temporal compression: 130 samples → 11 tokens.
- Width lift: 8 channels → 32 (information-bearing, not rank-8 padded).

Input: `(B, 32 cells, 8, 130)`. Output: `(B, 32 cells, 32, 11)`. Flatten cells as batch in implementation.

### Stage 5 — parcel embedding (learned)

`P_emb: Embedding-like(15, 32)`, Xavier-uniform init. **480 params.**

```
cell_embed = pooled_supp @ P_emb         # (n_cells, 32)
out = conv1d_out + cell_embed[None, :, None, None, :]   # broadcast across B and T_tokens
```

Added **after Conv1d at d=32** (not before at d=8). Parcel embedding is time-invariant per cell; injecting it at the backbone interface keeps cell-identity clean from Conv1d's temporal smoothing and lets P_emb live at its natural width.

### Stage 6 — attention backbone (learned)

Flatten `(B, 32 cells, 32 dim, 11 tokens) → (B, 352, 32)`. Combined spatiotemporal attention, **3 blocks at `d_model=32`**:
- Multi-head attention: **2 heads × head_dim=16** (or 4×8; pick 2×16 for cleaner head_dim).
- FFN: width 128 (= 4d), GELU, dropout 0.1, pre-norm.
- RoPE on temporal axis only (rotation depends on `t_a − t_b`, not cell index).
- Attention mask: `cell_active_mask` broadcast across temporal token axis.

**~36k params** total across 3 blocks.

### Stage 7 — D1 decoder (learned, minimal AR)

```
memory: (B, 352, 32)                               — backbone output
mem_pooled = memory.mean(dim=1)                    — (B, 32), global mean pool like baseline
prev_vec   = prev_emb(prev_phoneme) if prev != -1 else bos_emb    — (B, 32)
q          = mem_pooled + prev_vec                 — (B, 32)
logits     = head(q)                               — (B, 9)
```

Params:
- `prev_emb: (9, 32)` = 288
- `bos_emb: (32,)` = 32
- `head: Linear(32, 9)` = 297
- **Total: ~617.**

Loss: plain 9-class CE per phoneme sample. No focal, no label smoothing, no mixup (per `#9`).

### Stage 8 — exhaustive AR decode at eval

Per trial:
- Run backbone 3× (one per phoneme window).
- Enumerate 9³=729 hypothesis sequences.
- For each hypothesis `(p_0, p_1, p_2)`, run decoder 3× with `prev ∈ {BOS, p_0, p_1}`, gather `log P(p_i | memory_i, prev_i)`, sum across slots.
- Argmax across hypotheses; report slot-averaged PER.

Backbone dominates compute; decoder cost ~2,187 passes/trial at d=32 is microseconds.

## Training contract

Unchanged from v14-core First-Run Protocol except where noted:
- Optimizer: AdamW `lr=1e-3`, `wd=1e-4`.
- Schedule: cosine with 20-epoch linear warmup.
- Grad clip: 1.0.
- Grad accum: 4 steps → effective batch 32.
- Batch structure: same patient per batch (`#31`), mixed phoneme_pos within batch.
- Val every 5 epochs; early stop after 10 non-improving checks.
- Best-val state_dict restoration at end of training (landed 2026-04-17 in `train.py`).
- Teacher forcing: `prev_phoneme` is always ground truth during training.

**Effective gradient signal per epoch:** 291 phoneme samples × 1 loss term each vs previous `97 trials × 3 slots` in slot-CE mode. Same total loss contributions, but each sample gets its own optimizer step accumulation — slightly better gradient signal density.

## CV

Grouped-by-trial (equivalent to grouped-by-token — all 3 phonemes of a trial land in the same fold). Implementation: build folds on `trial_id`, expand each fold into phoneme samples. **5 folds × 3 seeds × 2 depth ablations = 30 runs per patient**, same shape as the existing DCC array driver.

## B-1 contract amendments (explicit)

Six frozen clauses change; worth flagging in `docs/implementation_tasks.md`:

| Clause | B-1 | This plan |
|---|---|---|
| `#2, #6` (Stage 1 temporal tokenizer) | per-electrode Conv1d upstream | per-cell Conv1d post-pool |
| `#15` (width) | `d_model=64` default, `128` first ablation | `d_model=32` default, `64` ablation |
| `#27` (Conv2d) | 64→64 learned spatial mixing, no pool | 1→8 + masked-mean pool to (4, 8), width lift in Conv1d |
| `#28` (AR decoder) | 3-query AR block, self+cross-attn, FFN | single-sample AR, D1 minimum (mean-pool + prev_emb + Linear) |
| `#29` (epoch) | trial-level, `[-0.5, 1.0)s`, T=300 | phoneme-level, `[-0.15, 0.5)s`, T_raw=130 |
| `#13` (loader) | per-electrode tokens with per-trial labels | per-phoneme samples with `prev_phoneme` + `trial_id` |

Unchanged: `#1, #3, #4, #5, #11, #16, #17, #18, #30, #31, #34, #36` + fsaverage v2c + snap atlas stack.

## Files

### New

- `src/speech_decoding/v14/phoneme_dataset.py` — `V14PhonemeDataset`, `V14PhonemeSample`, `collate_v14_phoneme_batch`.
- `src/speech_decoding/v14/pool.py` — masked-mean pool primitive + `cell_of` / `active_count` precompute utilities.
- `src/speech_decoding/v14/phoneme_decoder.py` — D1 decoder (mean-pool + prev_emb + Linear).
- `src/speech_decoding/v14/phoneme_model.py` — `NeuralFieldPerceiverPerPhoneme` (full stack assembly at d=32).
- `src/speech_decoding/v14/phoneme_run_fold.py` — fold/seed driver analogous to `run_fold.py`, expands trial folds to phoneme samples.
- `tests/v14/test_phoneme_dataset.py`
- `tests/v14/test_masked_mean_pool.py`
- `tests/v14/test_phoneme_decoder.py`
- `tests/v14/test_phoneme_model_shapes.py`
- `tests/v14/test_phoneme_run_fold.py`

### Modified

- `src/speech_decoding/v14/config.py` — new `PerPhonemeConfig`, `PoolConfig`; backbone default `d_model=32`; decoder config for D1.
- `src/speech_decoding/v14/train.py` — add `compute_flat_ce(logits, labels)`; add `loss_fn` kwarg to `train_one_fold` (default kept as slot-CE for back-compat of existing run mode).
- `src/speech_decoding/v14/eval.py` — add `exhaustive_ar_per(memories_per_pos, labels, decoder)`.
- `scripts/v14_core/train_v14_core.py` — `--mode {slot, per-phoneme}` flag, dispatches to the right dataset/model/runner.
- `docs/implementation_tasks.md` — record the six B-1 contract amendments listed above.

### Not touched

- Atlas stack (`data/atlas/fsaverage_bake_v2c/`, `data/atlas/support_cache_v2c_snap/`).
- `support_cache.py`, `channel_map.py`, `coordinates.py`, `fsaverage_projection.py`.
- `cv.py` (grouped-by-token splitter; we pass trial IDs, expand phoneme samples inside the fold).
- DCC sbatch wrappers (same array shape: 5 folds × 3 seeds × 2 depths per patient).

## Phases (TDD, one commit per task)

### P1 — phoneme-level `.fif` audit (full Phase-1 cohort)

Run up front across all 7 LH Phase-1 patients: **S14, S16, S23, S26, S33, S39, S62** (core + extended per `#30`). S22/S58 remain deferred to Phase 2 with the sEEG join.

Per patient:
- [ ] Load `epoch(phonemeLevel)(CAR)/.../sub-{pt}_*.fif`.
- [ ] Assert `len(events) == 3 × n_trials` (where `n_trials` matches the trial-level `.fif` from `#34` audit).
- [ ] Assert `event_id == {'a':1, 'ae':2, 'b':3, 'g':4, 'i':5, 'k':6, 'p':7, 'u':8, 'v':9}`.
- [ ] Assert raw `tmin ≤ -0.15` and `tmax ≥ 0.495` (so the `[-0.15, 0.5)` crop is in-bounds).
- [ ] Cross-check: group-by-3 from phoneme-level events → PS-encoded token → matches trial-level file's token at same trial index. S26 specifically: confirm phoneme-level `.fif` carries the trial 71 "vaek" that the trial-level events TSV is missing (per `#34` closure notes).
- [ ] QC report `docs/qc/phoneme_level_fif_audit_{pt}.md` per patient + cohort summary at `docs/qc/phoneme_level_fif_audit_cohort.md`.

Any patient that fails is excluded from that P1 wave; remaining patients proceed. Total-cohort blocker only if S14 fails (everything downstream is gated on S14 smoke).

Independent of Zac's 2026-04-17 trial-level regen — this plan consumes the phoneme-level `.fif` tree, which sits in a separate derivative folder.

### P2 — masked-mean pool primitive

- [ ] `pool.py` exposes `precompute_pool_assignment(H_p, W_p, pool_shape=(4, 8))` → `cell_of[H_p*W_p]`.
- [ ] `masked_mean_pool(x, cell_of, active, active_count)` supports `(B, C, H_p, W_p, T)` and `(B, N_e, 15)` tensor shapes via a common scatter-mean path.
- [ ] Tests: shape parity with `AdaptiveAvgPool2d` on all-active inputs; masked output excludes artifact electrodes from numerator and denominator; deterministic `cell_of` across runs.

### P3 — dataset + collator

- [ ] `V14PhonemeDataset` loads phoneme-level `.fif`, crops `[-0.15, 0.5)s`, wires the `#12` channel-map bridge and support cache (shared with `V14TrialDataset`).
- [ ] Emits `V14PhonemeSample` with `label, prev_phoneme, trial_id, phoneme_pos, signal, support, electrode_active_mask, electrode_grid_layout, electrode_grid_shape`.
- [ ] `collate_v14_phoneme_batch` stacks into the grouped-by-patient batch dict above.
- [ ] Tests: 3 samples per trial; `prev_phoneme` chain with BOS at pos 0; signal shape `(N_e, 130)`; tokens recovered from consecutive triplets match trial-level file; `trial_id` increases every 3 epochs.

### P4 — model

- [ ] `NeuralFieldPerceiverPerPhoneme` wires Conv2d(1→8) → masked-mean pool → per-cell Conv1d(8→32) → + parcel embedding at d=32 → backbone at d=32 → D1 decoder.
- [ ] Overfit test: 10 train_step on one fake batch drops CE by ≥ 0.1.
- [ ] Shape test: full forward on fake S14 shapes produces `(B, 9)` logits.

### P5 — training + eval plumbing

- [ ] `compute_flat_ce(logits, labels)`. `train_one_fold` takes `loss_fn` kwarg.
- [ ] `exhaustive_ar_per(memories_per_pos, labels, decoder)` — per-trial exhaustive 9³ decode, returns slot-averaged PER.
- [ ] Best-val restoration path inherits the 2026-04-17 fix; test covers both CE modes.
- [ ] Integration test: one-patient one-fold one-seed E2E run on CPU fake data, asserts per-phoneme PER < chance.

### P6 — CV + runner

- [ ] `phoneme_run_fold.run_one_fold` expands trial folds to phoneme-sample subsets via `trial_id`.
- [ ] CLI dispatch via `--mode per-phoneme`.

### P7 — DCC S14 smoke

- [ ] Single `(S14, fold=0, seed=0, depth=1)` job on DCC, 10 epochs.
- [ ] Verify val per-phoneme PER < 0.5 (proof-of-life; chance is 8/9 ≈ 0.889).

### P8 — full S14 array

- [ ] 30-job array (5 folds × 3 seeds × 2 depths).
- [ ] Compare per-phoneme-PER and exhaustive-AR slot-averaged-PER against (a) 0.734 baseline, (b) 2026-04-16 slot-CE run at ~chance.

### P9 — extend to remaining Phase-1 patients

P1 audits are already done for this set (P1 runs the full cohort up front). For each patient that passed audit:
- [ ] Launch 30-job array (5 folds × 3 seeds × 2 depths).
- [ ] Per-patient PER + cohort summary.

Target cohort: core {S26, S33, S62} first, then extended {S16, S23, S39}. 180 jobs if all 6 pass audit.

## Success criteria

1. **Per-phoneme val PER on S14 < 0.5** (proof-of-life; chance is 8/9 ≈ 0.889).
2. **Exhaustive-AR slot-averaged test PER on S14 ≤ 0.85** (beats the 2026-04-16 ~chance run).
3. **Stretch: match or beat Ben's 0.734 baseline on S14** over 3 seeds.
4. **Core cohort stretch: beat 0.734 baseline on ≥ 2 of {S14, S26, S33, S62}**.
5. **Extended cohort stretch: positive (< chance) per-phoneme PER on ≥ 2 of {S16, S23, S39}**.

If (1) fails, the problem is not in the labeling/loss. Look upstream (data, pool, parcel embedding, support cache).

## Ablation ladder (deferred; run only if default doesn't beat 0.734)

Ordered cheapest → most-capacity:

1. **Decoder depth.** D1 → D2 (add cross-attn, ~4k params) → D3 (+ FFN, ~12k) → D4 (full single-query block, ~18k).
2. **Width.** d=32 → d=64 (B-1 original) → d=16 (tighter).
3. **Temporal front-end.** Conv1d kept → no-Conv1d backbone at d=8 (A-no-temporal; tells us if Conv1d's temporal inductive bias is load-bearing).
4. **Conv1d kernel.** k=30 → k=10 (weaker temporal prior, cheaper: 2.5k vs 7.7k params).
5. **Conv2d depth.** 1 layer → 2 layers (B-1's A3).
6. **Pool shape.** (4, 8) → (3, 4) tighter → (5, 8) looser.
7. **Parcel embedding placement.** post-Conv1d at d=32 (default) → pre-Conv1d at d=8.

## Not in scope

- Cross-patient shared-weights training (v14-full).
- Learned per-patient calibration (Phase 2).
- Any atlas-side change (v2c + snap remains; kernel-ablation memo locks this).
- SSL on the full uECoG corpus (Phase 1.5).
- sEEG join or external-dataset scaling.

## Related docs

- `docs/v14_core_contract_amendment_2026-04-16.md` — B-1 amendment this plan further amends (#2, #6, #15, #27, #28, #29).
- `docs/plans/v14-core.md` — original v14-core plan (now superseded for the spatial/temporal stack; the Phase A atlas pre-flight and CV splitter are still load-bearing).
- `docs/plans/v14-core-per-phoneme.md` — earlier plan this one supersedes; captures only the decoder/labeling change.
- `docs/implementation_tasks.md` — blocker list; record the six amendments from the table above.
